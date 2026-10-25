# Configuration

Reference for every knob that sets up a run: presets, resolution overrides, schedule and
physics, layout and memory, the `EngineConfig` fields, the memory planner, and environment
variables. Lengths are comoving Mpc/h throughout. Flags are those of the two drivers,
`scripts/run/realization.py` (CPU) and `scripts/run/device_run.py` (GPU); see
[running.md](running.md) for how the drivers are invoked.

## Presets

`inexor.plan.PRESETS` holds the named geometries. Both drivers take a preset name
(`realization.py --config`, `device_run.py --preset`), and so does the planner (`--preset`).
The fine cell is 0.25 Mpc/h and the particle spacing 0.5 Mpc/h on every preset except
`smoke`; the presets differ in volume.

| preset | n_part (per side) | box | n_fine | n_coarse | tile | buf | padded tile | brick | bricks per side | tiles per side |
|---|---|---|---|---|---|---|---|---|---|---|
| `smoke`        | 32   | 32   | 64    | 16   | 16  | 8  | 32  | 8  | 8   | 4  |
| `cdev8`        | 128  | 64   | 256   | 64   | 64  | 32 | 128 | 32 | 8   | 4  |
| `cdev8-tile32` | 128  | 64   | 256   | 64   | 32  | 8  | 48  | 8  | 32  | 8  |
| `cdev`         | 256  | 128  | 512   | 128  | 256 | 32 | 320 | 32 | 16  | 2  |
| `cgh64`        | 512  | 256  | 1024  | 256  | 256 | 32 | 320 | 32 | 32  | 4  |
| `c-1024`       | 1024 | 512  | 2048  | 512  | 256 | 32 | 320 | 32 | 64  | 8  |
| `c-gh`         | 2048 | 1024 | 4096  | 1024 | 256 | 32 | 320 | 32 | 128 | 16 |
| `c-hero`       | 4096 | 2048 | 8192  | 2048 | 512 | 32 | 576 | 32 | 256 | 16 |
| `c-8192`       | 8192 | 4096 | 16384 | 4096 | 512 | 32 | 576 | 32 | 512 | 32 |

- `tile` is the fine tile side and `buf` the buffer around each tile, both in fine cells.
  The total particle count is `n_part**3`.
- `padded tile`, `brick` and `bricks per side` are derived, not set: the padded tile is the
  smallest FFT-friendly size at least `tile + 2*buf` (`forces.padded_size`), and the brick is
  the largest side dividing the tile, the fine mesh and the realized buffer
  (`layout.choose_brick`). The bricks per side is the number of slab files a T9 directory
  holds (see [outputs.md](outputs.md#t9-slab-directory)).
- `cdev8-tile32` is `cdev8`'s volume cut into 8 tile planes over 32 brick slabs, small enough
  for multi-rank tests on a laptop.
- For `smoke`, `cdev8`, `cdev` and `cgh64`, `realization.py` checks the preset against a
  second geometry table in `scripts/run/_instruments.py` and refuses if they disagree.

### Production dtypes and split scale

`inexor.plan.RATIFIED` holds the knobs both drivers and the planner apply on top of a preset:

| key | value | meaning |
|---|---|---|
| `coarse_dtype` | `"float32"` | coarse (long-range) mesh dtype |
| `fine_dtype` | `"float64"` | fine tile mesh dtype |
| `alpha` | `1.0` | `r_s / coarse_cell`: the split scale is one coarse cell |

`EngineConfig` itself defaults `coarse_dtype` to `"float64"`; that default serves reference
comparisons. Build the production config with:

```python
from inexor.plan import engine_config
ec = engine_config("c-gh", tile_workers=16, brick_slack=0.20)   # preset name or a geometry dict
```

`engine_config(preset, **overrides)` accepts a preset name or a dict with keys
`n_part, box, n_fine, n_coarse, tile, buf`. Overrides are for per-invocation knobs (workers,
slack, checkpointing, backends). The drivers' `--fine-dtype float32` is the one override of a
`RATIFIED` value they expose, for precision comparisons; it is in the checkpoint
fingerprint, so a card or export of such a run must pass it too.

## Force split and resolution knobs

The force is split into a long arm `S(k) ik/k^2` on the global coarse mesh and a short arm
`(1 - S(k)) ik/k^2` on fine tiles, with `S(k) = exp(-k^2 r_s^2)` (`src/inexor/forces.py`).
Two dimensionless knobs each control one analytic error term:

| knob | definition | error term |
|---|---|---|
| alpha | `r_s / coarse_cell` | coarse representation, `exp(-pi^2 alpha^2)` |
| beta | `buffer_length / r_s`, buffer length = `buf * fine_cell` | buffer truncation, `erfc(beta/2)` |

So `buf = (n_fine / n_coarse) * alpha * beta` fine cells. On the presets (4:1 fine:coarse,
alpha = 1, buf = 32, r_s = 1 Mpc/h) beta = 8: the coarse term is `exp(-pi^2)` = 5.2e-5 and
the truncation term `erfc(4)` = 1.5e-8. The realized buffer is
`(padded tile - tile) / 2`, which can exceed `buf` (it never falls short), so the realized
beta is at least the requested one. A tile plus buffer larger than the fine mesh is refused.

### Resolution overrides

`realization.py` (function `_geom`) takes a preset and applies these overrides, then prints
the geometry and both split error terms:

| flag | effect |
|---|---|
| `--n-fine N` | new fine mesh. `buf` is re-derived to hold beta fixed: `buf = round((N // n_coarse) * alpha * beta)` with the preset's beta. `tile + 2*buf > N` is refused. An explicit `--buf` wins. |
| `--buf B` | buffer in fine cells; changes beta and the truncation term. |
| `--n-coarse N` | new coarse mesh. alpha is re-derived as `r_s / (box / N)` to hold `r_s` at the preset's value in Mpc/h, and passed to the engine. A coarser mesh (alpha below 1.0) is refused. |
| `--n-part N` | particles per side; box and meshes held, so only the spacing changes. Must be a power of two. A different `n_part` at the same seed is an unrelated realization. |

`device_run.py` `preflight`, `run`, `card` and `export` take `--n-fine` only, re-derived the same way
(they call `realization._geom`). `--n-part` changes the ICs themselves, `--n-fine` and
`--buf` set the bricks per side the ICs are laid out on (`n_fine / brick`), and every override
changes the checkpoint fingerprint, so give the ICs and every later phase of a run the same
overrides. `device_run.py ics` has no
`--n-fine`; make such ICs with `realization.py ics --n-fine N` (with `--generator device` on
GPUs).

Other geometry constraints, checked at construction: the brick must divide the tile, the
fine mesh and the buffer (`EngineConfig.validate`; `choose_brick` guarantees it),
`bucket_cells` must divide `n_part` and `n_part / bucket_cells` must be a power of two
(`codec.T9Layout`), and the bricks per side must divide both `n_part` and the bucket grid
(the IC generators).

## Schedule and physics

| setting | default | where | notes |
|---|---|---|---|
| `a_init` | 0.1 (z = 9) | `--a-init` | Must lie in (0, 1). Baked into the ICs and recorded in their manifest. The CPU driver takes it explicitly and refuses ICs made at another value. The GPU driver's `run`, `card` and `export` start the schedule at the epoch recorded with the ICs (or with the checkpoint they read); an explicit `--a-init` that disagrees is refused. |
| `a_final` | 1.0 | fixed | Every schedule ends at a = 1, so cards at different step counts share their epoch and k bins. |
| K steps | 40 | `--k-steps` | BullFrog steps from `a_init` to `a_final`. |
| spacing | `log` | fixed in the drivers | `integrate.a_grid` also supports `linear`. |
| `growth2` | `lcdm` | `--growth2 lcdm\|eds` | Second-order growth for the 2LPT ICs and the BullFrog weights: the LCDM ODE solution, or the EdS form `-(3/7) D^2`. Recorded in the IC manifest; both drivers refuse ICs made with the other. |
| `coarse_match_order` | 3 | `realization.py --coarse-match-order 2\|3` | Assignment order the coarse match factor divides out. The coarse arm paints TSC, so 3 is correct; 2 (CIC) is kept for reference comparisons. |
| seed | 0 | `--seed` | `jax.random.PRNGKey(seed)` for the IC noise. |
| `f_NL` | 0 | `device_run.py ics --f-nl` | Local primordial non-Gaussianity of the ICs. |

`a_init`, K, spacing and `growth2` all enter the BullFrog coefficients, which the checkpoint
fingerprint hashes, so runs differing in any of them cannot resume from each other.

### Cosmology

`inexor.config.Cosmology` (frozen dataclass, flat LCDM). The drivers always use the defaults
(`Cosmology()`); there is no command-line override except in `python -m inexor.export`, whose
`--omega-m` and `--h` affect only the velocity conversion.

| field | default | derived |
|---|---|---|
| `Omega_m` | 0.31 | `Omega_Lambda = 1 - Omega_m` |
| `Omega_b` | 0.049 | `Omega_cdm = Omega_m - Omega_b` |
| `h` | 0.677 | `H0 = 100 h` km/s/Mpc |
| `n_s` | 0.965 | |
| `sigma8` | 0.81 | |
| `T_cmb_K` | 2.7255 | |

### Linear power spectrum

The ICs are coloured with EH98 (Eisenstein & Hu 1998, with baryon oscillations, normalized to
`sigma8`) unless `ics` is given `--pk-table FILE` (either driver), a tabulated z = 0 linear
P(k) in the `inexor-linear-pk-1` JSON format. `scripts/run/camb_linear_pk.py` makes one from
CAMB at the engine's cosmology (flat LCDM, massless neutrinos, A_s solved for `sigma8`), in a
throwaway env so CAMB never enters the project's:

```bash
pixi exec --spec python=3.12 --spec camb --spec numpy -- \
    python scripts/run/camb_linear_pk.py -o data/linear_pk_camb.json
```

Its options: `--n-points` (4000 log-spaced k nodes) and `--cosmology FILE` (a JSON file with
a `cosmology` block; default the engine's).

The table (`cosmology.LinearPkTable`) is refused unless its cosmology equals the run's, it is
at z = 0, its sigma8 is within 1e-3 (relative) of the cosmology's, its k and P are finite and
positive with k increasing, and it covers k = 1e-4 to 1e2 h/Mpc. With a table, the transfer
function in the f_NL potential is derived from it, T = sqrt(P / k^n_s) scaled to 1 at
k = 1e-4 h/Mpc, so the potential matches the spectrum. The whole table is embedded in the IC
manifest and carried into every checkpoint, where the P(k) card takes its linear oracle from
it; each embedded copy is re-checked (including its sha256) when it is read. Cards and export
headers record the table's `source` and `sha256`. EH98 runs embed nothing.

## Layout and memory knobs

The particle state is T9: 9 bytes per row (three uint8 in-bucket offsets and three int16
velocity codes) stored in brick-major slot order, plus a uint32 count per bucket. Rows beyond
the live particles are the capacity these knobs buy. Capacity is chosen when a state is
loaded (from ICs or a checkpoint) and is not stored on disk, so it can change between
segments of a run.

| knob | `realization.py` (default) | `device_run.py` (default) | library default | what it does and costs |
|---|---|---|---|---|
| `brick_slack` | `--slack` (0.20) | `--slack` (0.10) | 0.10 | Spare rows per brick, `ceil(count * slack)` (at least 1 per non-empty brick). 9 B per spare row. The repack redistributes capacity each step. |
| `alloc_margin` | `--alloc-margin` (0.10) | `--alloc-margin` (0.10) | 0.10 | Extra fraction on the total allocation after slack. 9 B per row. |
| `arena_frac` | `--arena-frac` (0.20) | `--arena-frac` (0.01) | 0.01 | Overflow arena rows as a fraction of N: 9 B (off, w) + 8 B (bucket key) per arena row. `load_checkpoint` uses the value passed, else the checkpoint's own arena size; both drivers always pass theirs. |
| `bucket_cells` | `--bucket-cells` (2), `ics` only | `ics --bucket-cells` (2) | 2 | Bucket side in particle cells. The position quantum is `bucket_cells * spacing / 256` (1/256 Mpc/h on the presets). 1 halves the quantum and multiplies the bucket index by 8. Recorded in the IC manifest and inherited by every later phase. |

A particle that overflows its brick's capacity is parked in the arena. If the arena is full
the step raises (`ValueError: ... overflow their brick's capacity and the arena of ... slots
has only N free`); particles are never clamped or dropped. The fix is a larger `--slack` or
`--arena-frac`. The pooled CPU lane also checks shared memory before building (see
`INEXOR_SHM_BACKEND`), raising `MemoryError` rather than faulting later.

Each `realization.py run` card records `arena_peak_rows` and `n_arena`, the arena's peak
occupancy against its size, so the arena can be sized from a short run.

### Workers and pooled lanes (CPU)

`tile_workers` (`realization.py --tile-workers`, default 16; library default 1) above 1
starts an `executor.TilePool` of spawned CPU worker processes, each pinned to a disjoint core
set (`worker_affinity`). The state is placed in shared memory so it exists once. The pool
runs:

- the tile loop (short arm and kick), bitwise the serial loop;
- the coarse paint chunks (accumulation stays in the parent, integer, bitwise);
- the drift + migrate, when `migrate_pooled` is `None` (auto) or `True`
  (`--migrate-pooled`; `--serial-migrate` sets `False`);
- the P(k) card's paint, via a paint-only pool (`--card-pool`, default on when
  `--tile-workers > 1`; `--serial-card` turns it off).

Every fine-arm mesh term is per worker, so memory grows with the worker count. The GPU lane
runs with `tile_workers=1`; the pool refuses a non-CPU parent.

### Eject kernel

`eject_kernel` (`realization.py --eject-kernel numpy|jax`, default `jax`) picks the CPU
migrate's eject. The two are bitwise identical; `jax` holds more host memory per slab row,
`numpy` is slower. The planner's per-row coefficients are `EngineConfig.EJECT_BYTES_PER_ROW`.
In the pooled migrate each worker holds one slab, so this multiplies by the worker count;
`migrate_eject_inflight` (library only) caps concurrent ejects separately from the worker
count.

### Y-blocks (GPU)

`device_y_blocks` (`device_run.py run` and `preflight --y-blocks N|auto`, default auto) cuts every x-slab-sized
card working set (tile window, destination census, device migrate, fused repack) into N
y-blocks of whole tile rows. The bytes are the same at any count, and the count is not in the
checkpoint fingerprint. Auto is the fewest blocks that keep a unit within a 4096^3 x-slab's
rows (`decomp.auto_y_blocks`): 1 up to 4096^3, 4 at 8192^3. `device_run.py ics --y-blocks`
applies the same rule to the IC emission's destination slabs. More blocks lower each card's
peak and add kernel launches; the planner reports the count it priced and the smallest that
fits.

## `EngineConfig` (src/inexor/engine.py)

Plain attributes, no jax. `validate()` (called by `engine.run` and by the drivers) refuses
inconsistent combinations, including a knob that could not apply. A float64 mesh dtype is
refused unless the caller has enabled `jax_enable_x64`; the library never enables it (the
drivers do).

### Operating fields

| field | default | meaning |
|---|---|---|
| `box_size` | required | box side, Mpc/h |
| `n_part` | required | particles per side |
| `n_fine` | required | fine mesh cells per side |
| `n_coarse` | required | coarse mesh cells per side |
| `n_tile` | required | fine tile side, fine cells |
| `b_fine` | required | requested buffer, fine cells (the realized buffer may be larger, see above) |
| `alpha` | 1.0 | split scale `r_s = alpha * coarse_cell` |
| `coarse_dtype` | `"float64"` | coarse mesh dtype, `float32` or `float64` (production: `float32`) |
| `fine_dtype` | `"float64"` | fine tile mesh dtype |
| `coarse_match_order` | 3 | 2 or 3; see Schedule and physics |
| `drift_rounding` | `"stochastic"` | how a drift lands on the position lattice: `"stochastic"` (unbiased, keyed on lattice index and drift) or `"nearest"` (the v0.1.0 rule; reproduces v0.1.0 runs) |
| `frac_bits` | 12 | fixed-point fraction bits of the integer paint |
| `chunk_bricks` | 64 | bricks per coarse-paint chunk (host) |
| `brick_slack` | 0.10 | spare capacity per brick (used by the repack and by a load that is not given one) |
| `repack_every` | 1 | repack cadence in steps; the repack is required, not optional |
| `cap_rungs` | `forces.CAP_RUNGS_PER_OCTAVE` | rungs per octave of the per-tile buffer shape ladder (more rungs: less padding, more compilations) |
| `tile_workers` | 1 | 1 = serial; > 1 = worker pool |
| `migrate_pooled` | `None` | `None` auto, `True` require a pool, `False` serial |
| `migrate_window` | `None` | migrate scratch slots in flight; `None` sized to the workers |
| `eject_kernel` | `"jax"` | `"jax"` or `"numpy"` |
| `migrate_eject_inflight` | `None` | cap on concurrent ejects in the pooled migrate |
| `checkpoint_dir` | `None` | checkpoint directory; checkpointing is off without one |
| `checkpoint_every` | 1 | cadence in steps (`realization.py` default 5; `device_run.py` default 0, off); 0 disables |

### Device-lane fields

These move phases to accelerators (`src/inexor/device/`). Each device backend requires
`jax_enable_x64`, and `tile_backend="device"` requires `tile_workers=1`. `device_run.py`
sets all three backends to `"device"` and leaves the tri-state fields on auto.

| field | default | meaning |
|---|---|---|
| `coarse_backend` | `"host"` | `"host"` or `"device"` coarse paint and solve |
| `tile_backend` | `"host"` | `"host"` or `"device"` tile loop |
| `migrate_backend` | `"host"` | `"host"` or `"device"` migrate and repack |
| `migrate_device_budget_bytes` | `None` | per-slab device envelope the device migrate refuses above |
| `device_cards` | 1 | cards the device step splits across; above 1 it needs the coarse and tile backends on the device, the compiled tile and the window |
| `device_tile_window` | `None` | run the compiled device tile loop on an x-slab window; `None` auto (wherever the compiled device tile runs) |
| `migrate_repack_fused` | `None` | fuse device migrate and repack on repack steps; `None` auto (wherever the device migrate and the window both run) |
| `device_paint_chunk_bricks` | `None` | bricks per device coarse-paint chunk; `None` a quarter x-slab |
| `device_y_blocks` | `None` | y-blocks per x-slab (see Y-blocks); at most the tile rows per side; `None` auto |

### Comparison and reference fields

These exist for A/B comparisons and reference arms. Their non-default values are not
operating points (except `coarse_dtype`, whose production value is the non-default
`float32`).

| field | default | non-default is |
|---|---|---|
| `paint_short`, `paint_long` | `"int"` | a float paint; the streamed coarse paint requires `"int"` |
| `pad_ladder` | `True` | `False`: unpadded coarse chunk shape |
| `paint_subblock` | `True` | `False`: full-mesh paint per chunk |
| `coarse_fold_kernel` | `True` | `False`: separate kernel-multiply pass |
| `coarse_kernel_on_cards` | `None` | `False`: coarse kernel arrays on the host, sliced per pencil block; `None` auto (on the cards wherever the folded solve writes card shards) |
| `worker_affinity` | `True` | `False`: unpinned workers |
| `device_tile_jit` | `True` | `False`: eager per-tile device reference |
| `coarse_match_order` | 3 | 2: CIC-order match |

### Checkpoint fingerprint

A resume, and the drivers' card and export, must match `engine.checkpoint_fingerprint(cfg,
coeffs, snapshot_steps)` (`python -m inexor.export` does not check it): a SHA-256 over
these config fields plus the BullFrog coefficients as float64 bytes (which cover cosmology,
a-grid and K):

`box_size, n_part, n_fine, n_coarse, n_tile, b_fine, alpha, paint_short, paint_long,
frac_bits, coarse_dtype, fine_dtype, cap_rungs, pad_ladder, paint_subblock`, plus
`coarse_match_order` when it is not 2, plus `drift_rounding` when it is stochastic, plus the
run's snapshot steps when it has any. A v0.1.0 checkpoint therefore resumes only under
`drift_rounding="nearest"`.

Execution policy (workers, pooling, eject kernel, backends, y-blocks), layout
(`brick_slack`, `chunk_bricks`, `repack_every`) and checkpoint settings are excluded and may
change across a resume. The coefficients come from numerical integration, so their bytes can
differ between platforms' math libraries, and a checkpoint can be refused on another platform
even with an identical configuration.

## Memory planner: `python -m inexor.plan`

Prices a configuration's host (and optionally device, shared-memory and IC disk) bytes from the
code's own byte accounting, and reports whether it fits the budgets you give. Budgets have no
defaults, so no verdict is printed for a budget you do not pass.

```bash
pixi run --frozen python -m inexor.plan --preset cdev
pixi run --frozen python -m inexor.plan --preset c-gh --host-gb 116 --shm-gb 128 --workers 16 \
    --slack 0.20 --arena-frac 0.20
pixi run --frozen python -m inexor.plan --preset c-hero --backend device --n-gpus 4 \
    --host-gb 1026 --device-gb 199
pixi run --frozen python -m inexor.plan --n-part 2048 --box 1024 --n-fine 4096 \
    --n-coarse 1024 --tile 256 --buf 32 --host-gb 237
```

| flag | default | meaning |
|---|---|---|
| `--preset` | none | a preset; explicit geometry flags override its fields |
| `--n-part --box --n-fine --n-coarse --tile --buf` | none (`--buf` 32) | explicit geometry, required without `--preset`; with a preset, `--n-fine` scales its buffer with the fine mesh (as the drivers do) unless `--buf` is given |
| `--backend cpu\|device` | `cpu` | `device`: the host holds the state and the GPUs run the step; replaces the mesh tables with host and per-GPU tables and adds the on-card IC tables |
| `--n-gpus` | 4 | device: cards per node the coarse mesh is sharded across |
| `--n-nodes` | 1 | device: nodes, split in x by whole tile planes; prices the busiest node and adds the exchange table |
| `--reach` | 1 | with `--n-nodes`: the migrate's brick reach per drift |
| `--net-gbs` | none | with `--n-nodes`: host-to-host GB/s, to price the exchange in seconds per step |
| `--y-blocks` | auto | device: y-blocks per x-slab, for the per-GPU tables and the IC emission |
| `--separate-passes` | off | device: price the migrate and repack as two passes instead of the fused pass |
| `--coarse-kernel cards\|host` | `cards` | device: where the coarse kernel arrays live (`coarse_kernel_on_cards`) |
| `--paint-chunk-bricks` | none | device: bricks per coarse-paint chunk (default a quarter x-slab) |
| `--workers` | none | cpu: `tile_workers`. Unset or above 1 prices the pooled lane (shared-memory table, state loaded straight into shared memory); a value above 1 also adds each worker's measured startup footprint |
| `--coarse-dtype`, `--fine-dtype` | `float32`, `float64` | mesh dtypes |
| `--bucket-cells` | 2 | bucket side |
| `--slack`, `--alloc-margin`, `--arena-frac` | 0.10, 0.10, 0.01 | layout knobs. These are the GPU driver's defaults; pass 0.20, 0.10, 0.20 to price a `realization.py` run |
| `--eject-kernel` | `jax` | cpu: eject kernel |
| `--eject-inflight` | none | cpu: cap on concurrent ejects |
| `--cap` | none | measured max padded rows per tile; adds the `tile_buffers` term |
| `--slabs` | 128 | slab files the ICs were written as, for the load-stage tables; not derived from the geometry, so pass the preset's bricks per side |
| `--host-gb`, `--device-gb`, `--shm-gb` | none | budgets in GB (1e9 B); `--device-gb` on the CPU backend compares the resident mesh plus the largest transient |

`device_run.py preflight` runs the planner with `--backend device` at the run's preset,
cards, layout, y-blocks and fine dtype (with its own `--n-nodes`, `--host-gb` default 1026
and `--device-gb` default 199), and refuses if the host or load peak exceeds `--host-margin`
(0.95) times the memory of the node's CPU NUMA nodes.

### What the tables mean

CPU backend, in order:

- **STATE**: the particle state, resident for the whole run: the 9 B/row T9 payload, the
  slack and margin rows, the bucket index, the arena rows and their bucket keys, and the
  per-brick start offsets and velocity scales; with the total in bytes per particle.
- **MESH, resident** and **MESH, transient**: the coarse and fine mesh terms
  (`EngineConfig.mesh_bytes`), each charged to the phase that allocates it
  (`engine.MESH_PHASE`). Fine-arm terms are per worker.
- **PER-STEP HOST TERMS**: terms that scale with particles (`EngineConfig.step_bytes`): the
  repack scratch, the migrate's slabs in flight and, with `--cap`, the per-tile buffers.
- **BY PHASE**: transients summed within each phase, and the in-step phases summed together,
  because freed allocator arenas are not returned between phases, so a step's high-water
  accumulates.
- **SHARED MEMORY**: the pooled lane's `/dev/shm` segments, a separate budget from host RAM
  (`--shm-gb`; a tmpfs, by default half the node's RAM).
- **LOADING THE STATE**: the peak at each stage of `icgen.load_slot_state`; the stages do not
  coexist, so the table's total is a maximum.
- **IC STAGE**: the host IC generator's out-of-core peak.
- **BINDING TERMS**: state + resident mesh + the worst phase (+ worker startup) as a lower
  bound on the run's peak, the load stage beside it (marked `<- BINDING` when it is larger),
  the largest single term, and a `FITS` / `DOES NOT FIT` verdict with the ratio to
  `--host-gb`.

Device backend: the state table, then **HOST, resident** and **HOST, by phase** (the host
side of the device step: state, the host-side coarse spectrum and kernel pieces, the
per-slab windows the device migrate and repack stage, and the host memory the compiled
programs keep), **HOST, peak by phase**, then the per-card tables: **PER GPU resident**,
**held through the phases in brackets**, **transient**, **AFTER THE TILE LOOP** and **BY
PHASE**. On a card, memory in use falls back to the whole-run terms between phases, so the
per-GPU column takes the largest phase where the host column sums them. Then the load stage,
the IC generation on the cards (host, per GPU and disk), and **BINDING TERMS**: the host
lower bound and the load stage against `--host-gb`, the per-GPU in-step and after-the-loop
peaks against `--device-gb`, the y-block count priced, and the smallest `--y-blocks` that
fits both card peaks. With `--n-nodes`, `[multi-node]` lines price each node's exchanges and
an EXCHANGE table gives the bytes each node sends per step.

### Measured and derived terms

Most terms are derived from the code: array shapes times dtype widths (state, meshes, load
stages, the out-of-core IC plan, the device emission's array inventories, the host windows of
the device migrate and repack), the slab window a tile plane needs (from `layout.brick_span`),
and where each term lives on the device lane (`plan.DEVICE_PLACEMENT`, a statement of the
design). Where an allocation cannot be read from the code, the planner uses a measured
coefficient, scaled by arithmetic to the configuration:

- CPU lane: a pool worker's startup footprint (`WORKER_STARTUP_BYTES`); the eject, insert and
  serial-migrate bytes per slab row (`EngineConfig.EJECT_BYTES_PER_ROW`,
  `INSERT_BYTES_PER_ROW`, `SERIAL_MIGRATE_BYTES_PER_ROW`); the repack's bytes per row; the
  float64 working sets of the coarse and tile kernel builds.
- GPU lane: the device paint chunk, device migrate, device repack and fused-pass peaks per row
  (the fused pass by y-block fraction, interpolated between measured fractions); the host
  memory that compiled programs keep from the first step and from a capacity rung on, and the
  tile loop's compile transient, by tile size and cards per node; the process's host baseline
  (CUDA context and runtime) by cards per node; the cross-card migrate hand-off rate. The
  planner labels these lines `MEASURED`.

The planner is arithmetic over the configuration, not a measurement, and its totals are lower
bounds: it cannot see XLA's intra-jit scratch, and `tile_buffers` is left out unless `--cap`
is given. The compiled-program terms for an unmeasured (tile size, cards per node) pair take
the largest per-card value measured at that tile size.

## Environment variables

Read by the package and the drivers:

| variable | read by | effect |
|---|---|---|
| `INEXOR_SHM_BACKEND` | `executor.shm_backend` | `memfd` or `posix` forces the pooled lane's shared-memory backend. Default: `memfd` where the kernel has it (not capped by `/dev/shm`; checked against `MemAvailable`), else `posix` (checked against the `/dev/shm` tmpfs cap). |
| `INEXOR_LOAD_TRACE` | `icgen.load_slot_state`, `executor.TilePool` | Any non-empty value prints `[load]` progress lines (with `MemAvailable` and `Shmem` from `/proc/meminfo`) while a state loads, and a `[pool]` line with the workers' RSS once the pool is up. |
| `JAX_PLATFORMS` | jax; set by `TilePool` | `realization.py run`, `card` and `export` (and `ics` unless `--generator device`) refuse a non-CPU backend: in the `gpu` env, run them with `JAX_PLATFORMS=cpu`, which is also needed for anything in that env on a node without a GPU. Pool workers always run with `JAX_PLATFORMS=cpu` and `OMP_NUM_THREADS=1`. |
| `XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB` | XLA | The XLA GPU client's limit on host memory for host-resident buffers. Streaming host-resident state through the GPUs needs it set to most of the node's host memory (the job scripts use 900 on a four-GB200 node and 160 on a GH200 node). |
| `XLA_PYTHON_CLIENT_ALLOCATOR`, `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_CLIENT_MEM_FRACTION` | XLA | GPU allocator settings. The job scripts use `cuda_async`, `false` and `0.95`. |
| `XLA_FLAGS` | XLA | `--xla_gpu_deterministic_ops=true` is needed on a GPU for the bit-equality tests (`pixi run test-det`); `--xla_force_host_platform_device_count=4` gives the multi-card tests four CPU devices. |
| `JAX_COMPILATION_CACHE_DIR` | jax | Persistent compilation cache. The job scripts set it; the planner's compiled-program host terms were measured with it. |
| `SCRATCH` | `device_run.py preflight` | Default of `--scratch`, the filesystem whose free space the preflight reports. |
| `SLURM_JOB_ID` | `device_run.py` | Recorded as the run card's `job`. |

What is recorded where:

- `realization.py ics` and `device_run.py ics` record `XLA_PYTHON_CLIENT_ALLOCATOR` (as
  `allocator`, `bfc` when unset), `XLA_CLIENT_MEM_FRACTION`, `XLA_PYTHON_CLIENT_PREALLOCATE`,
  `XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB` and `XLA_FLAGS` verbatim in the IC manifest's
  `provenance` (`"(unset)"` when absent).
- Every `device_run.py` card records all `XLA_*`, `JAX_*` and `INEXOR_*` variables under
  `env`; `preflight` also records the allocator settings under `allocator`.

The Slurm job scripts read their own variables: every one needs `INEXOR_RUNS` (the run root)
and `INEXOR_SRC` (the checkout; default the submission directory), and each takes its inputs
(IC, checkpoint and product directories, preset, step counts, layout) from variables listed
in its header and in [running.md](running.md#slurm-job-scripts). They refuse to start without
the required ones. They also set `PYTHONUNBUFFERED=1`, so the drivers' per-phase lines survive
a kill, and most set `INEXOR_LOAD_TRACE=1`.
