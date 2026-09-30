# Architecture

How the engine works, for someone reading the code for the first time, with the module and
function that implements each piece. Formats are in [outputs.md](outputs.md), knobs in
[configuration.md](configuration.md).

## 1. Memory model

### What a particle costs: the T9 codec (`codec.py`)

A particle is 9 bytes of payload:

- **Position:** three `uint8` offsets inside the particle's *bucket*, a cube of
  `bucket_cells` particle spacings (default 2), at quantum `bucket_size / 256` =
  `box_size / n_levels`, `n_levels = n_buckets_side * 256`. `T9Layout` refuses an `n_levels`
  that is not a power of two, so `quantum * n_levels == box_size` exactly and the periodic
  wrap is modular. The bucket is not stored; it is implied by the slot (below).
- **Velocity:** three `int16` codes of the D-time velocity `v = dx/dD` at scale
  `max|v| / 32767` (`encode_velocities`), so the extremes land exactly on +-32767. The scale
  is one float64 per brick (`SlotState.vel_scale`).
- **Ids:** an opt-in `int32` column (+4 B/p), refused above `n_part = 1290`
  (`refuse_ids_above_int32`). Production state carries none.

Every float-to-int conversion goes through `rint_i` (round-half-even to int32). A decode
rebuilds the global lattice index `bucket * 256 + off` and multiplies once by the quantum
(`decode_positions`), so the host mirrors in `state.py` are bitwise the codec.

### Slots, buckets and bricks (`state.py`, `layout.py`)

Two grids share one slot array: the **bucket** (the codec's cell) and the **brick** (the
force's tiling and streaming unit). `layout.choose_brick` picks the largest brick side (in
fine cells) dividing `n_tile`, `n_fine` and the realized buffer, so a tile's bricks cover
exactly the tile plus its buffer. Buckets are numbered brick-major
(`layout.bucket_order_key`), so a brick's buckets are contiguous. `SlotState` stores `off`
and `w` in slot order, with:

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

This is 11.56 B/p at the defaults (0.10, 0.10, 0.01, `bucket_cells=2`).

**Nothing O(N) in float.** Floats exist per brick or per tile: `decode_brick(s)` decodes a
brick list; the coarse paint decodes `chunk_bricks` at a time, the tile loop one tile plus
buffer, the migrate one x-slab, the export one brick chunk. `occupancy_total` sums the
index in int64 without widening it.

### Wrap, never clamp

No saturating operation touches integer state; an overflow raises. The refusals:

| Condition | Where |
|---|---|
| `n_levels` not a power of two, `bucket_cells` not dividing `n_part` | `T9Layout` |
| a velocity code outside int16 (encode, kick, rescale) | `assert_int16_range`, `state._rescale_w` |
| a bucket count above the uint32 index | `layout._to_index`, `SlotState.repack` |
| brick overflow with the arena full; repack past the allocation | `SlotState._to_arena`, `.repack` |
| int32 paint headroom; coarse accumulated peak >= 2^31 | `painting.check_tsc_paint_headroom`, `engine._delta_from_accumulated` |
| a particle lost in migration; staging over `max_staged_slabs` | `state.drift_and_migrate` |
| tile ownership not a partition; a row outside its padded tile | `engine.step` |
| a TSC stencil leaving the staged coarse sub-block | `forces.check_stencil_guard` |
| compiled paths without `jax_enable_x64` (int64 lattice index) | `eject_jax.require_x64`, `EngineConfig.validate` |

The library never sets `jax_enable_x64`; drivers do.

## 2. The step

### BullFrog with LCDM growth (`integrate.py`)

BullFrog (Rampf, List & Hahn 2024, arXiv:2409.19049) is DKD in D-time with an affine kick.
`bullfrog_table` computes per step `(alpha, beta, dD, D_mid)` in `_bullfrog_weights`:

```
D_mid = D0 + dD/2
F_mid = (E0 + E0' dD/2) / D_mid - D_mid
alpha = (E1' - F_mid) / (E0' - F_mid),   beta = 1 - alpha
```

`E` is the second-order growth and `E'` its slope `dE/dD`, from the LCDM ODE by default
(`cosmology.growth2_and_slope`; `growth2="eds"` uses `-(3/7) D^2`). The kick is
`v <- alpha v + (beta / D_mid) g` with `g` the dimensionless force of Section 3; all
cosmological prefactors live in the weights. `bullfrog_float_coeffs` packs the columns
`(dD/2, alpha, beta/D_mid)` the engine consumes. The FastPM and exact KDK tables in the
module are not used by the engine.

### Fused drifts (`engine.fused_drifts`)

Step `k`'s trailing half-drift and step `k+1`'s leading half-drift use the same velocity, so
they fuse into one drift of `h_k + h_{k+1}` (`h = dD/2`). State is carried
drift-synchronized: positions at step midpoints, velocities at step boundaries. `run` makes
one leading half-drift `h_0`, then each step drifts by the fused amount, except the last,
which drifts `h_{K-1}` alone so the final positions land on the last boundary with the
velocities. Each step quantizes and migrates once. Because `mod(mod(x+a)+b) != mod(x+a+b)`
in float, the matched float reference is `engine.float_run_bullfrog_sync`.

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
is set in one place, and tiles write disjoint rows.

**Drift and migrate** (`state.drift_and_migrate`). Positions are bucket-relative, so a drift
is a re-homing. `_eject_slab` drifts one x-slab of bricks in the integer lattice domain,
`i_new = mod(rint(x/q + c v/q), n_levels)`, and splits its rows into keepers and
emigrants, still coded at their source brick's scale. `_insert_slab` writes a slab once
every slab that can reach it has been ejected: per brick it takes the new scale over
keepers plus immigrants, re-expresses every row at it in one rounding, stable-sorts by
bucket (`_write_brick`), and spills past the allocation into the lowest free arena rows.
The reach `ceil(|c| max(vel_scale) 32767 / brick_extent)` (`brick_reach`) sets the staging
to `2 reach + 1` slabs; a particle census raises on any loss. `eject_jax.py` and
`insert_jax.py` are compiled twins of the row work (the default `eject_kernel="jax"`),
gated elementwise against the numpy paths.

**Repack** (`SlotState.repack`). Frozen capacity overflows as structure collapses, so each
repack resizes every run to `count * (1 + brick_slack)`, folds the arena back into its
bricks and empties it, in place: an ascending pass compacts runs left, a descending pass
writes each brick's block at its new start, merging residents. The result is bytewise the
out-of-place `_repack_reference`.

Checkpoints (`engine._write_checkpoint`) are T9 slab directories with a fingerprint of the
config fields and coefficients that move numbers; `run(resume=, stop_at=)` composes
segments bitwise into the uninterrupted run.

## 3. The force (`forces.py`, `painting.py`)

The solve is geometric: `div g = -delta`, `g = -grad phi`, kernel `ik/k^2`. The split is
PM-PM with `S(k) = exp(-k^2 r_s^2)`, `r_s = alpha * coarse_cell` (`EngineConfig.r_s`): the
long force `S ik/k^2` on the global coarse mesh, the short force `(1 - S) ik/k^2` on fine
tiles, each tile plus buffer transformed as its own periodic box. `split_factor` forms the
short factor literally as `1.0 - S` from the same array, so `long + short == ik/k^2`
bit-exactly and every residual is a discretization error. Each knob sets one error term
(`s_of_k`):

```
alpha = r_s / d_coarse     coarse representation   exp(-pi^2 alpha^2)
beta  = b / r_s            buffer truncation       erfc(beta / 2)
```

with `b` the physical buffer, so `b_fine = alpha * beta * n_fine / n_coarse` fine cells.
Production (`plan.RATIFIED`) uses `alpha = 1`, a float32 coarse mesh, float64 fine meshes.

### Coarse arm

- **Paint** (`engine.coarse_delta_streamed`): bricks in chunks of `chunk_bricks`, each a
  spatial cuboid painted with the integer TSC paint into a local sub-block
  (`painting.paint_tsc_int_subblock`) and added into an int64 host mesh by wrapped index.
  Chunks are padded with masked rows to shapes on a geometric ladder
  (`forces.capacity_shape`), so few programs compile. `_delta_from_accumulated` decodes
  `counts * 2^-frac_bits / mean - 1` in float64 and narrows to `coarse_dtype`.
- **Solve** (`forces.coarse_force_meshes`, `_coarse_solve_factorized`): kernel parts built
  once per run (`coarse_kernel_parts`); each component is `(pref * ik_j) * mf`, one at a
  time. The FFT is `ooc_fft`'s factorization: an `rfft2` per axis-0 plane, then an axis-0
  `fft` per y-pencil plane. The plane is the unit, so slab thickness never moves a bit. The
  kernel multiply rides the inverse's axis-0 pass (`coarse_fold_kernel`), split by y-pencils
  across the cards; each card keeps its rows of `pref` and `mf` resident for the run
  (`coarse_kernel_on_cards`), built one card's block at a time (`coarse_kernel_block`).
- **Match factor** (`forces.cic_match_factor`): the long arm paints and gathers TSC on the
  coarse cell, the short arm CIC on the fine cell. The factor
  `prod_i sinc^(2 p_t)(k_i d_fine/2) / sinc^(2 p_s)(k_i d_coarse/2)` removes the coarse window
  and applies the fine one, squared for paint plus gather. `p_s = coarse_match_order`
  (default 3, TSC) and `p_t = 2` (CIC); `p_s = 2` leaves a residual `sinc^2` per axis and is
  kept for reference comparisons.
- **Gather** (`forces.gather_coarse_subblock`): each tile stages its core's coarse cells
  plus `COARSE_HALO = 2` and reads TSC weights from the global coordinate, bitwise a gather
  from the whole mesh.

### Fine arm

- **Tiles** (`forces.padded_size`, `tile_geom`): `n_tile^3` fine cells plus buffer, padded to
  an FFT-friendly `P >= n_tile + 2 b_fine` (the buffer grows to fill `P`). Members are the
  bricks covering tile plus buffer (`SlotState.tile_bricks`); owners are the core bricks.
- **Solve** (`make_tile_force_fn`): integer CIC paint (`tile_paint_int`) on `P^3`,
  normalized by the *global* mean `n_total / n_fine^3` (a tile's own mean would rescale its
  short force), then `(1 - S) ik/k^2` and a CIC gather (`tile_gather_vector`). One jitted
  program serves every tile: row capacity is quantized to a shared shape, padding masked.

### Why integer paints

Both arms paint fixed-point weights (`frac_bits`, default 12) into int32 meshes. Integer
addition is associative, so the density is bit-identical under any particle, atomic or
chunk order. The layout reorders particles every step, so a float paint would not be
reproducible even on one machine; the integer paint is also what makes the chunked, pooled
and device coarse paints bitwise one monolithic paint. The float twins (`paint_f32`,
`paint_tsc_f64`, `tile_paint_f64`) remain for reference arms.

## 4. Initial conditions (`icgen.py`, `ic.py`, `lpt.py`, `cosmology.py`, `ooc_fft.py`)

- **Linear power** (`cosmology.linear_power`, `ic_k_table`): EH98 (Eisenstein & Hu 1998)
  normalized to `sigma8`, or a tabulated z = 0 `(k, P)` (`backend="table"`). The transfer
  function in the potential is always EH98.
- **Noise** (`ic.white_plane`): one unit-normal `(N, N)` plane per axis-0 index, keyed
  `jax.random.fold_in(key, i)` and drawn on the CPU backend, so any slab decomposition gives
  the same field. The stream is named `ic.IC_STREAM` and needs `jax_threefry_partitionable`.
- **Field** (`ic.py`): colour by `sqrt(P N^3 / L^3)`, divide by
  `M(k) = (2/3)(c/H0)^2 k^2 T(k) D_md / Omega_m` for the potential, apply local
  `phi + f_NL (phi^2 - <phi^2>)`, multiply back.
- **2LPT** (`lpt.py`): `x = q + D1 Psi1 - D2 Psi2`, `v_D = Psi1 - (D2 f2)/(D1 f1) Psi2`,
  with `Psi = +grad lap^-1 delta` and source `sum_{i<j} [phi_ii phi_jj - phi_ij^2]`
  (`lpt2_source_from_spec`). `D2`, `f2` come from the same LCDM second-order ODE as the
  BullFrog weights (`cosmology.growth_factor_2`, `growth_rate_2`).
- **Streaming** (`icgen.generate_t9_slabs`): every FFT uses `ooc_fft`'s one-plane unit with
  the spectrum in host memory; real-space intermediates are staged to `.npy` under `stage/`.
  `_emit_t9_slabs` walks Lagrangian x-slabs through a sliding window of `window` brick slabs
  into one `t9_slab_XXXX.npz` per destination slab with per-brick scales, refusing if the
  maximum displacement reaches the window; the manifest is written last.
  `load_slot_state` reassembles a `SlotState`. The host generator is bitwise
  `SlotState.build` on `lpt.lpt_ics` output.
- **Device generator** (`icgen.generate_t9_slabs_device`): same format, with transforms and
  kernels on the GPUs, noise on the cards (`ic.IC_STREAM_DEVICE`, a different stream) or on
  the host (`noise="host"`), and emission on the cards (`device/emit.py`, bitwise the host
  emission on the CPU backend). It is not bitwise the host generator.

## 5. Parallelism and placement

### CPU lane: the worker pool (`executor.py`)

`TilePool` runs `tile_workers` persistent spawned processes. The `SlotState` arrays in
`SHM_FIELDS` (plus `ids` when present) and the three coarse force meshes move into shared
memory and the parent's fields are rebound to the views, so in-place updates reach workers
without copies. Segments use `memfd_create` where available (outside the `/dev/shm` size
cap) and POSIX `shm_open` otherwise; demand is checked before any segment is created.
Workers are pinned to disjoint cores before jax loads. The pool runs the same `tile_task`
as the serial loop, the coarse paint chunks, and the migrate
(`state.drift_and_migrate_pooled`: workers eject and insert whole slabs, and
order-dependent arena mutations are replayed in serial order by `_replay_arena_pass`). It
refuses a non-CPU parent backend.

### GPU lane (`src/inexor/device/`)

The state stays in host memory and each phase streams it through the cards, with one
backend knob per phase (`coarse_backend`, `tile_backend`, `migrate_backend`):

- **Coarse paint** (`paint.coarse_delta_cards`): decode, containment and integer TSC paint
  per chunk in one program; each card accumulates its x-planes into an int64 mesh with ghost
  planes, folds the ghosts and decodes its planes.
- **Coarse solve**: the factorized FFT on the cards (`ooc_fft.forward_from_card_planes`,
  `inverse_to_card_shards`), leaving the force meshes as x-plane shards
  (`coarse.CardShards`) that the tile program gathers from.
- **Tile loop** (`tile.py`, `window.py`): decode (`decode.py`), both arms, kick and per-brick
  quantize (`kick.py`, a segmented max) as one compiled program at fixed per-step shapes.
  `tile_loop_windowed` holds one tile plane's x-slabs on the card and copies back only the
  core rows it owns.
- **Migrate and repack** (`migrate.py`, `repack.py`): each slab visits the card once, the
  compiled eject/insert kernels run there, and arena mutations are replayed on the host. On
  repack steps `fused.py` does both in one visit per slab, sized by a destination census
  the windowed tile loop takes.
- **Cards** (`device_cards`): paint and solve split by x-planes, the tile loop by tile planes
  (one thread per card), migrate and repack by x-slabs.

### What is bitwise between lanes

Bitwise by construction, and gated in the tests: the pooled tile loop, coarse paint and
migrate against the serial CPU path; the device coarse paint, decode, migrate, repack and
fused pass against their host references; the `ooc_fft` device transforms at any card
count; the device IC emission against the host emission on the CPU backend; checkpointed
segments against the uninterrupted run.

Not bitwise: the compiled device tile program against host `tile_task` (the compiled coarse
gather differs by a few eps of the long force; the eager `device_tile_jit=False` path is
bitwise on CPU); FFTs across backends (cuFFT against the CPU FFTs), hence a GPU step against
a CPU step; the factorized coarse solve against the monolithic one; the device IC generator
against the host generator.

## 6. Products

**P(k) card** (`summary.pk_summary_card`). The coarse density from the step's streamed
integer paint is transformed with `ooc_fft` and binned slab by slab with hermitian weights
(`binned_power`), with no particle array materialized. The TSC window is divided out
analytically and shot noise `V/N` subtracted. The oracle `D(a_out)^2 P_lin(k)` is averaged
over each bin's realized modes in the same pass, never evaluated at a bin centre, and
`z = (P / P_oracle - 1) / sqrt(2 / n_modes)` is reported per bin. The card decides no
verdict; `band_verdict` computes `max|z|` over a band the caller names.

**Particle export** (`export.write_particles`). Streams brick chunks through
`SlotState.decode_bricks` (the force's decode, arena residents included) into `x.npy`,
`v.npy` (and `ids.npy`) with an `export.json` header carrying crc32 sums and provenance,
written last so its presence marks a complete export. Rows are brick-major with no
Lagrangian identity. Velocities are D-time `dx/dD`, or peculiar km/s when `a` and `cosmo`
are given (factor `100 a f D E`, `peculiar_velocity_factor`).
