# Architecture

How the engine works, for someone reading the code for the first time, with the module and
function that implements each piece. Formats are in [outputs.md](outputs.md), knobs in
[configuration.md](configuration.md).

## 1. Geometry

The particle state and the force share one nested geometry. Every unit below is a cube of
the one beneath it, so every range is a contiguous run of indices.

| Unit | What it is | Size in the production presets |
|---|---|---|
| bucket | `bucket_cells` particle spacings per side (default 2); a particle stores its position relative to its bucket | 1 Mpc/h |
| brick | the largest side in fine cells dividing `n_tile`, `n_fine` and the realized buffer, and no larger than the buffer (`layout.choose_brick`); owns one contiguous run of slots and one velocity scale | 32 fine cells = 8 buckets = 8 Mpc/h |
| tile | `n_tile^3` fine cells, the unit of the short-range force, solved with a buffer of `b_fine` cells on every side | 8 bricks per side at `n_tile = 256`, 16 at 512 |
| x-slab | one brick thick in x, the whole box in y and z; the unit the state is stored, streamed and checkpointed in | `n_fine / n_brick` slabs (256 at 4096^3) |
| tile plane | the x-slabs under one layer of tiles (`n_tile / n_brick` slabs) | |
| y-block | a run of whole tile rows of an x-slab (`decomp.y_blocks`); an (x-slab, y-block) pair is a *unit*, one contiguous run of bricks and slots | 1 unless a slab is too large for a card |

Bricks are numbered x, then y, then z, and buckets brick-major (`layout.bucket_order_key`),
so a brick's buckets, an x-slab's bricks and a unit's bricks are each contiguous. Because the
brick divides the buffer, the bricks covering a tile plus its buffer are exactly that box
(`layout.brick_span`: `pad = ceil(b / n_brick)` buffer bricks per side, one in production).

## 2. Memory model

### What a particle costs: the T9 codec (`codec.py`)

A particle is 9 bytes of payload:

- **Position:** three `uint8` offsets inside the particle's bucket at quantum
  `bucket_size / 256` = `box_size / n_levels`, `n_levels = n_buckets_side * 256`. `T9Layout`
  refuses an `n_levels` that is not a power of two, so `quantum * n_levels == box_size`
  exactly and the periodic wrap is modular. The bucket is not stored; it is implied by the
  slot (below).
- **Velocity:** three `int16` codes of the D-time velocity `v = dx/dD` at scale
  `max|v| / 32767` over the brick (`encode_velocities`), so the brick's largest component
  lands exactly on +-32767. The scale is one float64 per brick (`SlotState.vel_scale`).
- **Ids:** an opt-in `int32` column (+4 B/p), refused above `n_part = 1290`
  (`refuse_ids_above_int32`). Production state carries none.

Every float-to-int conversion goes through `rint_i` (round-half-even to int32). A decode
rebuilds the global lattice index `bucket * 256 + off` and multiplies once by the quantum
(`decode_positions`), so the host mirrors in `state.py` are bitwise the codec.

### Slots, spare room and the arena (`state.py`)

`SlotState` stores `off` and `w` in slot order, with:

- `brick_start` (int64, per brick): each brick owns a contiguous run of slots;
- `occupancy` (uint32, per bucket), the only index: a brick's live rows are its first
  `sum(occupancy[brick])` slots, and bucket boundaries inside it are the prefix sum of its
  occupancy slice (`bucket_slot_starts`), derived, never stored. The rest of the run is
  spare, so free slots need no sentinel;
- spare of `ceil(count * brick_slack)` per brick (at least one if non-empty), the total
  grown by `alloc_margin` (`state._alloc_geometry`);
- a shared **arena** of `ceil(arena_frac * N)` rows after the runs, which takes brick
  overflow between repacks. Each arena row records its bucket in `arena_bucket` (int64,
  -1 = free); a resident still belongs to its brick and every decode returns it.

There is no per-particle key, permutation or float array. (`layout.BrickPackedLayout` is an
index over caller-held arrays with per-particle bookkeeping; the engine does not use it.)
Resident bytes per particle as the planner (`python -m inexor.plan`) accounts them:

```
9 (1 + brick_slack)(1 + alloc_margin)   payload rows, spare and margin
+ 4 / bucket_cells^3                    occupancy
+ arena_frac * (9 + 8)                  arena rows and arena_bucket
+ 16 n_bricks / N                       brick_start and vel_scale
```

This is 11.56 B/p at the defaults (`brick_slack = 0.10`, `alloc_margin = 0.10`,
`arena_frac = 0.01`, `bucket_cells = 2`).

**Nothing O(N) in float.** Floats exist per brick or per tile: `decode_brick(s)` decodes a
brick list; the coarse paint decodes one chunk of bricks at a time, the tile loop one tile
plus buffer, the migrate one x-slab (or unit), the export one brick chunk. `occupancy_total`
sums the index in int64 without widening it.

### Wrap, never clamp

No saturating operation touches integer state; an overflow raises. A compiled GPU program
cannot raise, so overflow there is excluded by construction or checked on the host from
values the program returns. The refusals:

| Condition | Where |
|---|---|
| `n_levels` not a power of two, `bucket_cells` not dividing `n_part` | `T9Layout` |
| a velocity code outside int16 (encode, kick, rescale) | `assert_int16_range`, `state._rescale_w` |
| a bucket count above the uint32 index | `layout._to_index`, `SlotState.repack` |
| brick overflow with the arena full; repack past the allocation | `SlotState._to_arena`, `.repack` |
| int32 paint headroom; coarse accumulated peak >= 2^31 | `painting.check_tsc_paint_headroom`, `forces.check_tile_paint_headroom`, `engine._delta_from_accumulated` |
| a particle lost in migration; staging over `max_staged_slabs` | `state.drift_and_migrate` |
| tile ownership not a partition; a row outside its padded tile | `engine.step` |
| a TSC stencil leaving the staged coarse sub-block | `forces.check_stencil_guard` |
| compiled paths without `jax_enable_x64` (int64 lattice index) | `eject_jax.require_x64`, `EngineConfig.validate` |

The library never sets `jax_enable_x64`; drivers do.

## 3. The step

### BullFrog with LCDM growth (`integrate.py`)

BullFrog (Rampf, List & Hahn 2025, JCAP 02, 020; arXiv:2409.19049) is drift-kick-drift in
growth-factor time with a kick whose weights are set from second-order LPT. `bullfrog_table`
computes per step `(alpha, beta, dD, D_mid)` in `_bullfrog_weights`:

```
D_mid = D0 + dD/2
F_mid = (E0 + E0' dD/2) / D_mid - D_mid
alpha = (E1' - F_mid) / (E0' - F_mid),   beta = 1 - alpha
```

`E` is the second-order growth and `E'` its slope `dE/dD`, from the LCDM ODE by default
(`cosmology.growth2_and_slope`; `growth2="eds"` uses `-(3/7) D^2`). The kick is
`v <- alpha v + (beta / D_mid) g` with `g` the dimensionless force of Section 4; all
cosmological prefactors live in the weights. `bullfrog_float_coeffs` packs the columns
`(dD/2, alpha, beta/D_mid)` the engine consumes. Steps are uniform in `ln a` by default
(`integrate.a_grid`). The FastPM and exact KDK tables in the module are not used by the
engine.

### Fused drifts and snapshots (`engine.fused_drifts`, `engine.drift_plan`)

Step `k`'s trailing half-drift and step `k+1`'s leading half-drift use the same velocity, so
they fuse into one drift of `h_k + h_{k+1}` (`h = dD/2`). State is carried
drift-synchronized: positions at step midpoints, velocities at step boundaries. `run` makes
one leading half-drift `h_0`, then each step drifts by the fused amount, except the last,
which drifts `h_{K-1}` alone so the final positions land on the last boundary with the
velocities. Each step computes the force once and quantizes and migrates once. Because
`mod(mod(x+a)+b) != mod(x+a+b)` in float, the matched float reference is
`engine.float_run_bullfrog_sync`.

A snapshot splits the fused drift: a step ending at a snapshot drifts only its own trailing
half, the synchronized state is written (`snapshot_dir/step<NNNN>/gen0`, checkpoint format,
marked `snapshot`), and the next step's leading half then runs as its own drift and repack.
Snapshot steps therefore enter the checkpoint fingerprint. `engine.synchronized` says whether
a checkpoint's positions and velocities share its epoch (the final step or a snapshot); a
regular checkpoint's positions are half a step past it.

### One step (`engine.step`, `engine.run`)

```
 x at D_mid(k) (T9 offsets), v at D_k (int16, per-brick scales)
   | coarse paint   int TSC per brick chunk -> int64 mesh -> delta (coarse_dtype)
   | coarse solve   S(k) ik/k^2 x match factor, factorized FFT -> 3 long-force meshes
   | per tile       decode tile+buffer bricks
   |                short: int CIC paint on P^3 -> (1-S) ik/k^2 -> CIC gather
   |                long:  TSC gather from the core's coarse sub-block
   |                kick owned rows: v = alpha v + (beta/D_mid)(g_short + g_long)
   |                requantize each owned brick at its max|v| / 32767
   | drift+migrate  x += c_k v on the integer lattice, eject / insert per x-slab
   | repack         every repack_every steps (default 1); then checkpoint if due
   v
 x at D_mid(k+1), v at D_{k+1}
```

**Kick** (`engine.tile_task`, `apply_result`). Each tile decodes its member bricks, computes
both arms at its *owned* rows (those stored in its core bricks,
`forces.owned_mask_from_bricks`), kicks them, and re-encodes each owned brick at
`max|v_new| / 32767` over that brick. A brick lies wholly in one tile's core, so its scale
is set in one place, and tiles write disjoint rows. Buffer rows are read for the short-range
density only; their velocities are never used.

**Drift and migrate** (`state.drift_and_migrate`). Positions are bucket-relative, so a drift
is a re-homing. `_eject_slab` drifts one x-slab of bricks in the integer lattice domain,
`i_new = mod(floor(x/q + c v/q + u), n_levels)`, and splits its rows into keepers and
emigrants, still coded at their source brick's scale. The dither `u` in [0, 1)
(`state.dither`) hashes the pre-drift lattice index, the axis and the drift's key (lead
drift 0, step k's drift k + 1, the drift after a snapshot at step k (k + 1) + 2^32), so the
rounding is unbiased per drift and depends on no row order, layout or rank. Rounding to
nearest instead (`drift_rounding="nearest"`, `rint(x/q + c v/q)`, the v0.1.0 rule) holds
back every particle that drifts less than q/2 in a step, which biases P(k) low for early
starts and many steps and makes small halos lighter as the step count grows. `_insert_slab` writes a slab once
every slab that can reach it has been ejected: per brick it takes the new scale over
keepers plus immigrants, re-expresses every row at it in one rounding, stable-sorts by
bucket (`_write_brick`), and spills past the allocation into the lowest free arena rows.
The insert's input order is fixed (the slab's keepers, then each source slab's emigrants in
ascending slab order), so the row order does not depend on which worker, card or rank did
the work. The reach `ceil(|c| max(vel_scale) 32767 / brick_extent)` (`brick_reach`) sets
the staging to `2 reach + 1` slabs; a particle census raises on any loss. `eject_jax.py` and
`insert_jax.py` are compiled twins of the row work (the default `eject_kernel="jax"`),
gated elementwise against the numpy paths.

**Repack** (`SlotState.repack`). Frozen capacity overflows as structure forms, so each
repack resizes every run to `count * (1 + brick_slack)`, folds the arena back into its
bricks and empties it, in place: an ascending pass compacts runs left, a descending pass
writes each brick's block at its new start, merging residents. The result is bytewise the
out-of-place `_repack_reference`.

**Shapes.** Compiled programs need static shapes, and the densest tile's row count moves
every step. Row capacities are rounded up a geometric ladder (`forces.capacity_shape`,
`cap_rungs` rungs per factor of two, padding masked) and only ever grow within a run, so a
run compiles few programs. The shapes are recorded in each checkpoint and restored on resume.

## 4. The force (`forces.py`, `painting.py`)

The solve is geometric: `div g = -delta`, `g = -grad phi`, kernel `ik/k^2`, the continuous
Green's function and spectral gradient on both levels. The split is PM-PM with
`S(k) = exp(-k^2 r_s^2)`, `r_s = alpha * coarse_cell` (`EngineConfig.r_s`): the long force
`S ik/k^2` on the global coarse mesh, the short force `(1 - S) ik/k^2` on fine tiles, each
tile plus buffer transformed as its own periodic box. `split_factor` forms the short factor
literally as `1.0 - S` from the same array, so the two arms are complements by construction
and every residual is a discretization error. Each knob sets one error term (`s_of_k`):

```
alpha = r_s / d_coarse     coarse representation   exp(-pi^2 alpha^2)
beta  = b / r_s            buffer truncation       erfc(beta / 2)
```

with `b` the physical buffer, so `b_fine = alpha * beta * n_fine / n_coarse` fine cells.
Production (`plan.RATIFIED`) uses `alpha = 1`, a float32 coarse mesh and float64 fine meshes;
the production presets use a 32-cell buffer (`beta = 8` at `n_fine = 4 n_coarse`).

### Coarse arm

- **Paint** (`engine.coarse_delta_streamed`): bricks in chunks of `chunk_bricks`, each a
  spatial cuboid painted with the integer TSC paint into a local sub-block
  (`painting.paint_tsc_int_subblock`) and added into an int64 mesh by wrapped index.
  Chunks are padded with masked rows to shapes on a geometric ladder
  (`forces.capacity_shape`), so few programs compile. `_delta_from_accumulated` refuses a
  peak at or above 2^31, decodes `counts * 2^-frac_bits / mean - 1` in float64 and narrows
  to `coarse_dtype`.
- **Solve** (`forces.coarse_force_meshes`, `_coarse_solve_factorized`): kernel parts built
  once per run (`coarse_kernel_parts`); each component is `(pref * ik_j) * mf`, one at a
  time. The FFT is `ooc_fft`'s factorization: an `rfft2` per axis-0 plane, then an axis-0
  `fft` per y-pencil. The plane is the unit, so slab thickness never moves a bit. The kernel
  multiply rides the inverse's axis-0 pass (`coarse_fold_kernel`), split by y-pencils across
  the cards; each card keeps its rows of `pref` and `mf` resident for the run
  (`coarse_kernel_on_cards`), built one card's block at a time (`coarse_kernel_block`).
- **Match factor** (`forces.cic_match_factor`): the long arm paints and gathers TSC on the
  coarse cell, the short arm CIC on the fine cell. The long kernel is multiplied by

  ```
  prod_i sinc^(2 p_t)(k_i d_fine / 2) / sinc^(2 p_s)(k_i d_coarse / 2)
  ```

  which removes the coarse window and applies the fine one, squared for paint plus gather,
  so both levels see the same window. `p_s = coarse_match_order` (default 3, TSC) and
  `p_t = 2` (CIC): `sinc^4 / sinc^6` per axis. It is the principal-zone window (no alias
  sum); `S(k)` suppresses it towards the coarse Nyquist. `p_s = 2` leaves a residual
  `sinc^2` per axis and is kept for reference comparisons. The fine CIC window is not
  deconvolved.
- **Gather** (`forces.gather_coarse_subblock`): each tile stages its core's coarse cells
  plus `COARSE_HALO = 2` and reads TSC weights from the global coordinate, bitwise a gather
  from the whole mesh.

### Fine arm

- **Tiles** (`forces.padded_size`, `tile_geom`): `n_tile^3` fine cells plus buffer, padded to
  an FFT-friendly `P >= n_tile + 2 b_fine` (the buffer grows to fill `P`). Members are the
  bricks covering tile plus buffer (`SlotState.tile_bricks`); owners are the core bricks.
- **Solve** (`make_tile_force_fn`): integer CIC paint (`tile_paint_int`) on `P^3`,
  normalized by the *global* mean `n_total / n_fine^3` (the short kernel vanishes at k = 0,
  so no mean is subtracted, and a tile's own mean would rescale its short force), then
  `(1 - S) ik/k^2` and a CIC gather (`tile_gather_vector`). One jitted program serves every
  tile: row capacity is quantized to a shared shape, padding masked.

### Integer mass assignment

Both arms paint fixed-point weights (`frac_bits`, default 12) into int32 meshes; the coarse
mesh accumulates the chunks in int64. Every corner but the one nearest the particle adds its
weight rounded to `frac_bits`; the nearest adds the remainder (`painting._add_exact_mass`), so
each particle paints exactly `2^frac_bits` and the mean density is the particle count
exactly. Integer addition is associative, so the density is bit-identical under any
particle, atomic, chunk, card or rank order. The layout reorders particles every step, so a
float paint would not be reproducible even on one machine; the integer paint is also what
makes the chunked, pooled, device and multi-rank coarse paints bitwise one monolithic paint.
A configuration is refused if an assumed `1e4` particles per cell could overflow either
int32 paint (`painting.check_tsc_paint_headroom`, `forces.check_tile_paint_headroom`), and the
coarse paint's accumulated peak is checked every step. The float twins (`paint_f32`,
`paint_tsc_f64`, `tile_paint_f64`) remain for reference arms.

## 5. Initial conditions (`icgen.py`, `ic.py`, `lpt.py`, `cosmology.py`, `ooc_fft.py`)

- **Linear power** (`cosmology.linear_power`, `ic_k_table`): EH98 (Eisenstein & Hu 1998)
  normalized to `sigma8`, or a tabulated z = 0 `(k, P)` (`backend="table"`, a checked
  `LinearPkTable` whose `sigma8` must match the cosmology's, which the IC manifest embeds
  and the checkpoints carry). The transfer function in the potential is EH98's, or with a
  table derived from it, `T = sqrt(P / k^n_s)` scaled to 1 at k = 1e-4 h/Mpc.
- **Noise** (`ic.white_plane`): one unit-normal `(N, N)` plane per axis-0 index, keyed
  `jax.random.fold_in(key, i)` and drawn on the CPU backend, so any slab decomposition gives
  the same field. The stream is named `ic.IC_STREAM` and needs `jax_threefry_partitionable`.
- **Field** (`ic.py`): colour by `sqrt(P N^3 / L^3)`, divide by
  `M(k) = (2/3)(c/H0)^2 k^2 T(k) D_md / Omega_m` (`D_md` normalized to `a` in matter
  domination) for the potential, apply local `phi + f_NL (phi^2 - <phi^2>)` in real space,
  multiply back.
- **2LPT** (`lpt.py`). With `phi1` the linear potential, `lap phi1 = delta`, and `phi2` the
  second-order one, `lap phi2 = delta2 = sum_{i<j} [phi1_ii phi1_jj - phi1_ij^2]`
  (`lpt2_source_from_spec`; quadratic in `phi1`), the displacement kernel `ik/k^2` gives

  ```
  Psi1 = (ik/k^2) delta  = -grad phi1,     div Psi1 = -delta
  Psi2 = (ik/k^2) delta2 = -grad phi2,     div Psi2 = -delta2
  x    = q + D1 Psi1 - D2 Psi2            (= q - D1 grad phi1 + D2 grad phi2)
  v_D  = dx/dD1 = Psi1 - (D2 f2)/(D1 f1) Psi2
  ```

  the standard 2LPT form (Scoccimarro 1998) with `D2 < 0` (`-(3/7) D1^2` in EdS, so
  `v_D = Psi1 + (6/7) D1 Psi2`). `D2`, `f2` come from the same LCDM second-order ODE as the
  BullFrog weights (`cosmology.growth_factor_2`, `growth_rate_2`).
- **CPU generator** (`icgen.generate_t9_slabs`): every FFT uses `ooc_fft`'s one-plane unit
  with one spectrum at a time in host memory; real-space intermediates are staged to `.npy`
  under `stage/`. `_emit_t9_slabs` walks Lagrangian x-slabs through a sliding window of
  `window` brick slabs into one `t9_slab_XXXX.npz` per destination slab with per-brick
  scales and crc32 sums, refusing if the maximum displacement reaches the window; the
  manifest is written last. `load_slot_state` reassembles a `SlotState`. This generator is
  bitwise `SlotState.build` on `lpt.lpt_ics` output.
- **GPU generator** (`icgen.generate_t9_slabs_device`): the same file format. Transforms and
  k-space kernels run on the cards inside the axis-0 pass, the spectra stay in host memory,
  and only the velocities are staged to disk (`U_x` held on the cards, `U_y` and `U_z` on
  the host). The 2LPT source is formed as `1/2 (delta^2 - sum phi_ii^2) - sum_{i<j} phi_ij^2`,
  algebraically the one above. By default the noise is drawn on the cards
  (`ooc_fft.noise_plane_program`: the same plane-keyed construction, transformed where it is
  drawn). The normal sampler's bits differ across backends, so on a GPU this is a different
  stream at the same seed, tagged `ic.IC_STREAM_DEVICE`; on the CPU backend it is bitwise
  the CPU stream. `noise="host"` draws the CPU stream instead. From the same noise the two
  generators agree to one code unit in positions and velocities, and to roundoff in the
  per-brick scales (`tests/test_icgen_device.py`). Emission runs on the cards
  (`device/emit.py`), bitwise the host emission on the CPU backend, in y-blocks of brick rows
  when a slab is too large for a card (same bytes at any count).

## 6. CPU lane: the worker pool (`executor.py`)

`TilePool` runs `tile_workers` persistent spawned processes. The `SlotState` arrays in
`SHM_FIELDS` (plus `ids` when present) and the three coarse force meshes move into shared
memory and the parent's fields are rebound to the views, so in-place updates reach workers
without copies. Segments use `memfd_create` where available (outside the `/dev/shm` size
cap) and POSIX `shm_open` otherwise; demand is checked before any segment is created.
Workers are pinned to disjoint cores before jax loads. The pool runs the same `tile_task`
as the serial loop, the coarse paint chunks, and the migrate
(`state.drift_and_migrate_pooled`: workers eject and insert whole slabs, and
order-dependent arena mutations are replayed in serial order by `_replay_arena_pass`). It
refuses a non-CPU parent backend. The CPU lane runs on one node only.

## 7. GPU lane (`src/inexor/device/`)

The state stays in host memory as x-slabs, and each phase streams the slabs it needs through
the cards and writes the updated slabs back. There is one backend knob per phase
(`coarse_backend`, `tile_backend`, `migrate_backend`); the production lane sets all three to
`"device"`. One process drives its node's cards, one host thread per card (`device_cards`).
Slabs reach the cards as ordinary copies from pageable host memory.

- **Coarse paint** (`paint.coarse_delta_cards`): decode, containment and integer TSC paint
  per chunk in one program (default chunk: a quarter x-slab of bricks); each card
  accumulates the chunks starting in its x-planes into an int64 mesh with ghost planes,
  folds the ghosts into their owners (`fold_ghosts`) and decodes its planes.
- **Coarse solve**: the factorized FFT on the cards (`ooc_fft.forward_from_card_planes`,
  `inverse_to_card_shards`), the spectrum in host memory, the kernel multiply on the cards'
  y-pencils, leaving the three force meshes as x-plane shards (`coarse.CardShards`): each
  card the coarse planes under its tile planes plus `COARSE_HALO` each side.
- **Tile loop** (`tile.py`, `window.py`): decode (`decode.py`), both arms, kick and per-brick
  quantize (`kick.py`, a segmented max) as one compiled program at fixed per-step shapes.
  `tile_loop_windowed` walks the card's tile planes in x order and, per (tile plane,
  y-block), stages a *window*: the plane's `n_tile / n_brick + 2 pad` x-slabs, restricted to
  the brick-y range its tiles reach (`off` and `w` of the bricks' slot runs plus their arena
  residents). It runs the window's tiles, checks the guards and that the block owned exactly
  the particles of its core bricks, and copies back only the core rows. Write-backs are
  disjoint, so the cards' threads run concurrently.
- **Destination census**: on a repack step the windowed loop also runs each core slab's rows
  through the migrate's eject kernel and counts the post-migrate membership per brick,
  before the migrate runs.
- **Migrate and repack** (`migrate.py`, `repack.py`, `fused.py`): each unit visits a card
  once; the compiled eject/insert kernels run there, and order-dependent arena mutations are
  replayed on the host (`state._replay_arena_pass`). On a repack step (every step by
  default) `fused.migrate_repack_device` builds each unit's repacked block directly from the
  insert's output, sized by the census, so a unit crosses the bus once each way; it refuses
  if a brick's inserted membership differs from its census count. With several cards each
  owns a contiguous run of slabs and hands emigrant-only copies of its boundary ejects to its
  neighbours.

What a card holds, phase by phase (the allocator returns memory between phases):

| Phase | On each card |
|---|---|
| whole run | its y-pencil rows of the coarse kernel parts (`pref`, `mf`) |
| coarse paint | an int64 accumulator over its coarse x-planes plus ghost planes; one paint chunk |
| coarse solve | the planes or pencils in flight (the spectrum is on the host); its force shards from here through the tile loop |
| tile loop | its force shards, the tile kernels and `P^3` meshes, one window (one tile plane's slabs, one y-block of them) |
| migrate + repack | the units within one drift's reach of the unit being inserted, whose emigrants it takes |

**y-blocks.** `device_y_blocks` cuts every x-slab-sized card working set (tile window,
census, device migrate, fused repack) into runs of whole tile rows (`decomp.y_blocks`). The
default (`decomp.auto_y_blocks`) is the fewest blocks that keep a unit at or below one
x-slab of the 4096^3 run (`UNIT_ROWS_MAX`); any count gives the same bytes. The planner
(`python -m inexor.plan`) prices each phase's card and host memory from these terms.

## 8. Multiple nodes (`decomp.py`, `comm.py`, `device/ghost.py`)

One MPI rank per node (`comm.MPIComm`; `comm.LoopbackComm` runs ranks as threads for the
tests). The decomposition is one-dimensional in x by whole tile planes (`decomp.Decomp`):
each rank owns a contiguous range of tile planes (`ooc_fft.partition_units`; when the count
does not divide, the first ranks take one plane more) and with them their brick x-slabs and
coarse x-planes; its cards split that range again. The spectrum is split by y-pencils, per
rank and then per card. `Decomp.build` refuses fewer tile planes than cards in total, and a
rank with fewer than `2 reach + 1` or `2 pad` brick slabs, since particles and ghost slabs
move between immediate neighbours only. Across ranks only the production device lane runs
(`engine.rank_lane_refusal`).

Every rank calls `engine.step` with its node-local state. Per step, the exchanges are:

- **Coarse paint ghost planes**: accumulator planes owned by the neighbour go to it and are
  added there (`paint.fold_ghosts`; exact, integer).
- **FFT transposes**: plane-to-pencil before the axis-0 forward transform and
  pencil-to-plane after the inverse (`ooc_fft.forward_card_planes_to_pencils`,
  `inverse_pencils_to_card_shards`), each an `Alltoallv`.
- **Ghost slabs** (`ghost.exchange_ghosts`): the `pad` boundary x-slabs of each neighbour
  that this rank's first and last tile planes read as buffer, sent as verbatim copies of
  the owner's slot ranges (`off`, occupancy, arena residents; no velocities, which buffer
  rows never use). Tile member counts are taken over owned plus ghost slabs and the tile
  capacity is maximized over ranks.
- **Hand-off**: the migrate's boundary ejects send emigrant-only copies, plus the pre-pass
  velocity scales of the `reach` boundary slabs, to the neighbour ranks; the destination
  census crosses the same way (`fused.census_across_ranks`). A particle whose destination
  lies more than `reach` slabs past its rank's boundary is a collective refusal.
- **Shapes**: every compiled shape is maximized over ranks (`comm.allreduce_shapes`), so
  each rank compiles the programs one rank would.

Exchanges are MPI on host arrays. `exchange_neighbours` moves named arrays to the left and
right ranks of the periodic x ring; `Alltoallv` is a pairwise exchange (at shift `k`, send to
`rank + k`, receive from `rank - k`) in `chunk_bytes` pieces (`DEFAULT_CHUNK_BYTES`, 64 MiB);
`allreduce` is an allgather folded in rank order, so a float reduction is the same on every
rank. `MPIComm` polls every wait against a watchdog, and a rank that fails aborts the others
at their next exchange. Each rank keeps a ledger of its exchange bytes and seconds
(`Comm.take_ledger`), recorded per step under `stats["ranks"]`.

The GPU IC generator splits the same way across ranks (destination brick slabs and their
particle x-planes per rank, y-pencils per rank; plane/pencil exchanges in batches of
`batch_planes`), with only `noise="device"` and card emission. Products split too: the card
(`summary.pk_summary_card_cards`) bins each rank's pencils and combines the partials
exactly, and the export writes one part per rank (`export.write_particle_parts`).

## 9. Checkpoints and resume (`engine.py`, `icgen.write_t9_slabs`)

A checkpoint (`engine._write_checkpoint`) is a T9 slab directory in the IC format: each
brick's live rows and arena residents written tight in bucket order, one `.npz` per x-slab
with crc32 sums, written by a background thread, and a `manifest.json` written last.
Across ranks every rank writes its own slabs and rank 0 the manifest. Checkpoints alternate
between two generations (`gen0`, `gen1`); the old manifest is removed before a rewrite, so a
torn generation cannot load and the other survives. `load_checkpoint` takes the complete
generation with the highest step, each rank loading its own slabs; a resumed run first
writes the generation it did not load.

The provenance carries a **fingerprint** (`checkpoint_fingerprint`): a hash of the config
fields that move numbers (`_FINGERPRINTED`: geometry, `alpha`, paints, `frac_bits`, mesh
dtypes, the buffer-shape knobs `cap_rungs`, `pad_ladder`, `paint_subblock`, and
`coarse_match_order`, and `drift_rounding` when it is stochastic), the BullFrog coefficients as bytes (so cosmology, a-grid and step
count are covered) and the snapshot steps. A resume under a different fingerprint is
refused. Not fingerprinted: the execution policy of the pool and the migrate (worker, card,
rank and y-block counts, `migrate_backend`, the fused pass) and the layout (`brick_slack`,
`chunk_bricks`, `repack_every`), whose alternatives are bitwise the same trajectory; and
`coarse_backend`, `tile_backend` and `device_tile_jit`, which change only roundoff
(Section 10). The coefficients are computed with the platform's math library, so a run
resumes on the platform that wrote it. `run(resume=, stop_at=)` composes segments bitwise into the uninterrupted run
(always passing the full coefficient table, since drifts fuse across steps); the compiled
shapes are restored from the checkpoint so the resumed run compiles the same programs.
Checkpoints also record the epoch, cosmology and any tabulated linear spectrum, so products
can be made from the directory alone.

## 10. Determinism

**Byte-identical** (gated in the tests, on the CPU backend for the multi-rank and multi-card
cases):

- The checkpoint *state*, the slab files, at any number of ranks, cards per rank and
  y-blocks, and across checkpoint-and-resume, including resuming at another count
  (`tests/test_ranks_run.py`). The manifest is also identical across rank and card counts;
  across y-block counts it differs in one entry, `provenance.device_shapes.window`, the
  window buffer's shape.
- The IC slab files of the GPU generator at any rank, card and emission y-block count (the
  manifest adds `n_ranks`, and records the y-block count when above one).
- The pooled tile loop, coarse paint and migrate against the serial CPU path; the device
  coarse paint, decode, migrate, repack and fused pass against their host references; the
  `ooc_fft` device transforms at any card count; the device IC emission against the host
  emission on the CPU backend; the device P(k) card at any rank and card count; a multi-part
  export concatenated against the single-file export.

The ingredients: integer paints; a migrate whose row order is fixed by bucket and source
slab, not by who did the work; arena spills into the lowest free rows; checkpoints that
write each brick tight in bucket order; plane-keyed noise and FFTs whose unit is the plane;
exchanges that are copies; reductions folded in rank order; and compiled shapes maximized
over ranks and restored on resume.

**Not byte-identical**: the compiled device tile program against host `tile_task` (the
compiled coarse gather differs by a few eps of the long force, and velocity codes by at most
one; the eager `device_tile_jit=False` path is bitwise on CPU); FFTs across backends (cuFFT
against the CPU FFTs), hence a GPU step against a CPU step; the factorized coarse solve
against the monolithic one; the GPU IC generator against the CPU one (one code unit from the
same noise; a different noise stream on a GPU).

## 11. Products

Products are made only from synchronized states (the final step or a snapshot); the
drivers and `python -m inexor.export` refuse other checkpoints unless `--allow-partial`.

**P(k) card** (`summary.pk_summary_card`, and `pk_summary_card_cards` on the cards and
across ranks). The coarse density from the step's streamed integer paint is transformed with
`ooc_fft` and binned with hermitian weights (`binned_power`), with no particle array
materialized. Each ky-plane's bin sums are formed alone and the planes combined per bin with
`math.fsum`, so the card is bitwise the same however the planes are blocked or split across
ranks (`binned_power_partials`, `combine_partials`). The TSC window is divided out
analytically and shot noise `V/N` subtracted; the default bins are 64 linear bins to half
the coarse Nyquist. The oracle `D(a_out)^2 P_lin(k)` is averaged over each bin's realized
modes in the same pass, never evaluated at a bin centre, and
`z = (P / P_oracle - 1) / sqrt(2 / n_modes)` is reported per bin. The card decides no
verdict; `band_verdict` computes `max|z|` over a band the caller names.

**Particle export** (`export.write_particles`, `write_particle_parts`). Streams brick
chunks through `SlotState.decode_bricks` (the force's decode, arena residents included), or
the same decode on the cards (`device/export.py`, bitwise), into `x.npy`, `v.npy` (and
`ids.npy`) with an `export.json` header carrying crc32 sums and provenance, written last so
its presence marks a complete export. Rows are brick-major with no Lagrangian identity.
Velocities are D-time `dx/dD`, or peculiar km/s when `a` and `cosmo` are given (factor
`100 a f D E`, `peculiar_velocity_factor`).
