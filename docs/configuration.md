# Configuration

Reference for every knob that sets up a run: presets, resolution overrides, schedule and
physics, layout and memory, the `EngineConfig` fields, the memory planner, and environment
variables. Lengths are comoving Mpc/h throughout.

## Presets

`inexor.plan.PRESETS` holds the named geometries. The fine cell is 0.25 Mpc/h on every
preset except `smoke`; presets differ in volume.

| preset | n_part (per side) | box | n_fine | n_coarse | tile | buf | particle spacing | fine cell | coarse cell |
|---|---|---|---|---|---|---|---|---|---|
| `smoke`  | 32   | 32   | 64   | 16   | 16  | 8  | 1.0 | 0.5  | 2.0 |
| `cdev8`  | 128  | 64   | 256  | 64   | 64  | 32 | 0.5 | 0.25 | 1.0 |
| `cdev`   | 256  | 128  | 512  | 128  | 256 | 32 | 0.5 | 0.25 | 1.0 |
| `cgh64`  | 512  | 256  | 1024 | 256  | 256 | 32 | 0.5 | 0.25 | 1.0 |
| `c-1024` | 1024 | 512  | 2048 | 512  | 256 | 32 | 0.5 | 0.25 | 1.0 |
| `c-gh`   | 2048 | 1024 | 4096 | 1024 | 256 | 32 | 0.5 | 0.25 | 1.0 |
| `c-hero` | 4096 | 2048 | 8192 | 2048 | 512 | 32 | 0.5 | 0.25 | 1.0 |

`tile` is the fine tile side and `buf` the buffer around each tile, both in fine cells. The
total particle count is `n_part**3`.

### Default dtypes and split scale

`inexor.plan.RATIFIED` holds the knobs the production driver and the planner both apply on top
of a preset:

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
`n_part, box, n_fine, n_coarse, tile, buf`. Overrides are meant for per-invocation knobs
(workers, slack, checkpointing), not for undoing the defaults above.

## Force split and resolution knobs

The force is split into a long arm `S(k) ik/k^2` on the global coarse mesh and a short arm
`(1 - S(k)) ik/k^2` on fine tiles, with `S(k) = exp(-k^2 r_s^2)` (`src/inexor/forces.py`).
Two dimensionless knobs each control one analytic error term:

| knob | definition | error term |
|---|---|---|
| alpha | `r_s / coarse_cell` | coarse representation, `exp(-pi^2 alpha^2)` |
| beta | `buffer_length / r_s`, buffer length = `buf * fine_cell` | buffer truncation, `erfc(beta/2)` |

With a 4:1 coarse:fine ratio, `buf = 4 * alpha * beta` fine cells. At the default alpha = 1
the coarse term is `exp(-pi^2)` = 5.2e-5; at buf = 32 fine cells and r_s = 1 Mpc/h, beta = 8
and the truncation term is `erfc(4)` = 1.5e-8. The padded tile side is rounded up to an
FFT-friendly size (`forces.padded_size`), so the realized buffer can exceed `buf`; a padded
tile larger than the fine mesh is refused.

The driver `scripts/run/realization.py` (function `_geom`) takes a preset and applies these
overrides, printing the geometry and both error terms:

| flag | effect |
|---|---|
| `--n-fine N` | new fine mesh; `buf` is recomputed in fine cells to hold beta fixed. `tile + 2*buf > n_fine` is refused. An explicit `--buf` wins. |
| `--buf B` | buffer in fine cells; changes beta and the truncation term. |
| `--n-coarse N` | new coarse mesh; alpha is recomputed to hold `r_s` fixed in Mpc/h. A coarser mesh (alpha below 1.0) is refused. |
| `--n-part N` | particles per side; box and meshes held. Must be a power of two. A different `n_part` at the same seed is an unrelated realization. |

Other geometry constraints, checked at construction: the brick size must divide the buffer
(`EngineConfig.validate`), `bucket_cells` must divide `n_part` and `n_part / bucket_cells`
must be a power of two (`codec.T9Layout`), and the bricks per side must divide both `n_part`
and the bucket grid (`icgen.generate_t9_slabs`).

## Schedule and physics

| setting | default | where | notes |
|---|---|---|---|
| `a_init` | 0.1 (z = 9) | `--a-init` | Must lie in (0, 1). Baked into the ICs; `run` and `card` refuse ICs made at another `a_init`. |
| `a_final` | 1.0 | fixed | Every schedule ends at a = 1, so cards at different step counts share their epoch and k bins. |
| K steps | 40 | `--k-steps` | BullFrog steps from `a_init` to `a_final`. |
| spacing | `log` | fixed in the driver | `integrate.a_grid` also supports `linear`. |
| `growth2` | `lcdm` | `--growth2 lcdm\|eds` | Second-order growth for the 2LPT ICs and the BullFrog weights: the LCDM ODE solution, or the EdS form `-(3/7) D^2`. Recorded in the IC manifest; `run` and `card` refuse ICs made with the other. |
| `coarse_match_order` | 3 | `--coarse-match-order 2\|3` | Assignment order the coarse match factor divides out. The coarse arm paints TSC, so 3 is correct; 2 (CIC) is kept for reference comparisons. |
| seed | 0 | `--seed` | `jax.random.PRNGKey(seed)` for the IC noise. |

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

The linear power spectrum is EH98 (`cosmology.ic_k_table(backend="eh98")`).

## Layout and memory knobs

The particle state is T9: 9 bytes per row (three uint8 in-bucket offsets and three int16
velocity codes) stored in brick-major slot order. Rows beyond the live particles are the
capacity these knobs buy.

| knob | driver flag (default) | library default | what it does and costs |
|---|---|---|---|
| `brick_slack` | `--slack` (0.20) | 0.10 | Spare rows per brick, `ceil(count * slack)` (at least 1 per non-empty brick). Costs 9 B per spare row. The repack redistributes capacity each step. |
| `alloc_margin` | `--alloc-margin` (0.10) | 0.10 | Extra fraction on the total allocation after slack. 9 B per row. |
| `arena_frac` | `--arena-frac` (0.20) | 0.01 | Overflow arena rows as a fraction of N. Costs 9 B (off/w) + 8 B (`arena_bucket`) per arena row. Not stored in checkpoints: on resume `load_checkpoint` uses the value passed, else the checkpoint's own. |
| `bucket_cells` | `--bucket-cells` (2) | 2 | Bucket side in particle cells. The position quantum is `bucket_cells * spacing / 256` (1/256 Mpc/h at the presets' 0.5 Mpc/h spacing). 1 halves the quantum and multiplies the bucket index (4 B per bucket) by 8. Set at `ics`, recorded in the manifest, inherited by `run` and `card`. |

A particle that overflows its brick's capacity is parked in the arena. If the arena is full
the step raises (`ValueError: ... overflow their brick's capacity and the arena ... has only
N free`); particles are never clamped or dropped. The fix is a larger `brick_slack` or
`arena_frac`. The pooled lane also checks shared memory before building (see
`INEXOR_SHM_BACKEND`), raising `MemoryError` rather than faulting later.

### Workers and pooled lanes

`tile_workers` (driver `--tile-workers`, default 16; library default 1) above 1 starts an
`executor.TilePool` of spawned CPU worker processes, each pinned to a disjoint core set
(`worker_affinity`). The state is placed in shared memory so it exists once. The pool runs:

- the tile loop (short arm and kick), bitwise the serial loop;
- the coarse paint chunks (accumulation stays in the parent, integer, bitwise);
- the drift + migrate, when `migrate_pooled` is `None` (auto) or `True`
  (`--migrate-pooled`; `--serial-migrate` sets `False`);
- the P(k) card's paint, via a paint-only pool (`--card-pool`, default on when
  `--tile-workers > 1`; `--serial-card` turns it off).

Every fine-arm mesh term is per worker, so memory grows with the worker count.

### Eject kernel

`eject_kernel` (`--eject-kernel numpy|jax`, default `jax`) picks the migrate's eject. The two
are bitwise identical; `jax` holds more host memory per slab row, `numpy` is slower. The
planner's per-row coefficients are `EngineConfig.EJECT_BYTES_PER_ROW`. In the pooled migrate
each worker holds one slab, so this multiplies by the worker count; `migrate_eject_inflight`
caps concurrent ejects separately from the worker count.

## `EngineConfig` (src/inexor/engine.py)

Plain attributes, no jax. `validate()` (called by `engine.run` and by the driver) refuses
inconsistent combinations. A float64 mesh dtype is refused unless the caller has enabled
`jax_enable_x64`; the library never enables it (the drivers do).

### Operating fields

| field | default | meaning |
|---|---|---|
| `box_size` | required | box side, Mpc/h |
| `n_part` | required | particles per side |
| `n_fine` | required | fine mesh cells per side |
| `n_coarse` | required | coarse mesh cells per side |
| `n_tile` | required | fine tile side, fine cells |
| `b_fine` | required | buffer, fine cells (realized value may be larger, see above) |
| `alpha` | 1.0 | split scale `r_s = alpha * coarse_cell` |
| `coarse_dtype` | `"float64"` | coarse mesh dtype, `float32` or `float64` (production: `float32`) |
| `fine_dtype` | `"float64"` | fine tile mesh dtype |
| `coarse_match_order` | 3 | 2 or 3; see Schedule and physics |
| `frac_bits` | 12 | fixed-point fraction bits of the integer paint |
| `chunk_bricks` | 64 | bricks per coarse-paint chunk |
| `brick_slack` | 0.10 | spare capacity per brick (used by repack and resume) |
| `repack_every` | 1 | repack cadence in steps; the repack is required, not optional |
| `cap_rungs` | `forces.CAP_RUNGS_PER_OCTAVE` | rungs per octave of the per-tile buffer shape ladder (more rungs: less padding, more compilations) |
| `tile_workers` | 1 | 1 = serial; > 1 = worker pool |
| `migrate_pooled` | `None` | `None` auto, `True` require a pool, `False` serial |
| `migrate_window` | `None` | migrate scratch slots in flight; `None` sized to the workers |
| `eject_kernel` | `"jax"` | `"jax"` or `"numpy"` |
| `migrate_eject_inflight` | `None` | cap on concurrent ejects in the pooled migrate |
| `checkpoint_dir` | `None` | checkpoint directory; checkpointing is off without one |
| `checkpoint_every` | 1 | cadence in steps (driver default 5); 0 disables |

### Device-lane fields

These move phases to accelerators (`src/inexor/device/`). Each device backend requires
`jax_enable_x64`, and `tile_backend="device"` requires `tile_workers=1`.

| field | default | meaning |
|---|---|---|
| `coarse_backend` | `"host"` | `"host"` or `"device"` coarse paint and solve |
| `tile_backend` | `"host"` | `"host"` or `"device"` tile loop |
| `migrate_backend` | `"host"` | `"host"` or `"device"` migrate and repack |
| `migrate_device_budget_bytes` | `None` | per-slab device envelope the device migrate refuses above |
| `device_cards` | 1 | cards the device step splits across (needs all device backends and the window) |
| `device_tile_window` | `None` | run the compiled device tile loop on an x-slab window; `None` auto |
| `migrate_repack_fused` | `None` | fuse device migrate and repack on repack steps; `None` auto |
| `device_paint_chunk_bricks` | `None` | bricks per device coarse-paint chunk; `None` a quarter x-slab |

### Comparison and reference fields

These exist for A/B comparisons and reference arms. Their non-default values are not
operating points.

| field | default | non-default is |
|---|---|---|
| `paint_short`, `paint_long` | `"int"` | a float paint; the streamed coarse paint requires `"int"` |
| `pad_ladder` | `True` | `False`: unpadded coarse chunk shape |
| `paint_subblock` | `True` | `False`: full-mesh paint per chunk |
| `coarse_fold_kernel` | `True` | `False`: separate kernel-multiply pass |
| `coarse_kernel_on_cards` | `None` | `False`: coarse kernel arrays on the host, sliced per pencil block; `None` auto (on the cards wherever the folded solve writes card shards) |
| `worker_affinity` | `True` | `False`: unpinned workers |
| `device_tile_jit` | `True` | `False`: eager per-tile device reference |
| `coarse_match_order` | 3 | 2: legacy CIC-order match |
| `coarse_dtype` | `"float64"` | the library default is the reference arm; production uses `float32` |

### Checkpoint fingerprint

A resume must match `engine.checkpoint_fingerprint(cfg, coeffs)`: a SHA-256 over these fields
plus the BullFrog coefficients as bytes (which cover cosmology, a-grid and K):

`box_size, n_part, n_fine, n_coarse, n_tile, b_fine, alpha, paint_short, paint_long,
frac_bits, coarse_dtype, fine_dtype, cap_rungs, pad_ladder, paint_subblock`, plus
`coarse_match_order` when it is not 2.

Execution policy (workers, pooling, eject kernel, backends), layout (`brick_slack`,
`chunk_bricks`, `repack_every`) and checkpoint settings are excluded and may change across a
resume.

## Memory planner: `python -m inexor.plan`

Prices a configuration's host (and optionally device, shared-memory and disk) bytes from the
code's own byte accounting and reports whether it fits the budgets you give. It prints:
the resident state (bytes per particle), mesh terms (resident and transient), per-step terms
that scale with particles, a by-phase table, the pooled lane's `/dev/shm` sub-budget, the
state-loading stages, the IC stage peak, and a BINDING TERMS verdict.

```
pixi run --frozen python -m inexor.plan --preset cdev
pixi run --frozen python -m inexor.plan --preset c-gh --host-gb 116 --shm-gb 128 --workers 16
pixi run --frozen python -m inexor.plan --n-part 2048 --box 1024 --n-fine 4096 \
    --n-coarse 1024 --tile 256 --buf 32 --host-gb 237 --disk-gb 2000
```

| flag | default | meaning |
|---|---|---|
| `--preset` | none | a preset; explicit geometry flags override its fields |
| `--n-part --box --n-fine --n-coarse --tile --buf` | none (`--buf` 32) | explicit geometry, required without `--preset` |
| `--backend cpu\|device` | `cpu` | `device`: the host holds the state, GPUs run the step; adds per-GPU and IC-on-card tables |
| `--n-gpus` | 4 | cards the coarse mesh is sharded across (`--backend device`) |
| `--separate-passes` | off | device: price migrate and repack as two passes instead of the fused pass |
| `--coarse-kernel cards\|host` | `cards` | device: where the coarse kernel arrays live (`coarse_kernel_on_cards`) |
| `--paint-chunk-bricks` | none | device: bricks per coarse-paint chunk |
| `--workers` | none | `tile_workers`; unset or > 1 prices the pooled lane (shared memory table, state loaded once) |
| `--coarse-dtype`, `--fine-dtype` | `float32`, `float64` | mesh dtypes |
| `--bucket-cells` | 2 | bucket side |
| `--slack`, `--alloc-margin`, `--arena-frac` | 0.10, 0.10, 0.01 | layout knobs; pass the driver's values (0.20, 0.10, 0.20) to price a driver run |
| `--eject-kernel` | `jax` | eject kernel |
| `--eject-inflight` | none | cap on concurrent ejects |
| `--cap` | none | measured max padded rows per tile; adds the `tile_buffers` term |
| `--slabs` | 128 | T9 slab files the ICs were written as (load-stage pricing) |
| `--host-gb`, `--device-gb`, `--shm-gb`, `--disk-gb` | none | budgets in GB (1e9 B); each verdict is printed only when its budget is given |

Reading the verdict: each budget line reads `FITS` or `DOES NOT FIT` with the ratio of the
estimate to the budget. The host estimate is state + resident mesh + the worst phase + worker
startup; the load stage is reported beside it and marked `<- BINDING` when it is larger.
Within a step, phases are summed rather than maxed, because freed allocator arenas are not
returned between phases. The shared-memory table is a separate budget from host RAM.

The planner is arithmetic over the config, not a measurement: it is a lower bound. It cannot
see XLA's intra-jit scratch, and `tile_buffers` is omitted unless `--cap` is given. Budgets
have no defaults, so no verdict is invented.

## Environment variables

| variable | read by | effect |
|---|---|---|
| `INEXOR_SHM_BACKEND` | `executor.shm_backend` | `memfd` or `posix` forces the shared-memory backend. Default: `memfd` where available (not capped by `/dev/shm`, checked against `MemAvailable`), else `posix` (checked against the `/dev/shm` tmpfs cap). |
| `INEXOR_LOAD_TRACE` | `icgen.load_slot_state`, `executor.TilePool` | Any non-empty value prints `[load]` progress lines (with `MemAvailable`/`Shmem` from `/proc/meminfo`) during state loading, and a `[pool]` line with worker RSS once the pool is up. |
| `JAX_PLATFORMS` | jax; set by `TilePool` | Pool workers always run with `JAX_PLATFORMS=cpu` (and `OMP_NUM_THREADS=1`). The driver's `run`, `export` and `card` refuse a non-CPU default backend, and so does `ics` unless `--generator device`; on a GPU node run them with `JAX_PLATFORMS=cpu`. |
| `XLA_PYTHON_CLIENT_ALLOCATOR`, `XLA_CLIENT_MEM_FRACTION`, `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB`, `XLA_FLAGS` | `realization.py ics` | Recorded verbatim in the IC manifest's `provenance` (`"(unset)"` when absent; the allocator as `bfc` when unset). |
| `XLA_FLAGS=--xla_gpu_deterministic_ops=true` | tests | Needed on GPU for tests that assert bitwise float equality (`pixi run test-det`); a no-op on CPU. |
| `PYTHONUNBUFFERED` | the batch scripts | Set so the `--phase-instrument peak` streaming lines survive a kill. |

The batch scripts under `scripts/run/*.sbatch` also read their own variables (for example
`INEXOR_SRC` for the checkout and `INEXOR_RUNS` for the run root); see each script's header.
