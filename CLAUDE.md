# CLAUDE.md -- inexor

inexor is a particle-mesh N-body engine in JAX built to minimize memory per particle, so a
large realization fits on a single node. Particle state is a compressed fixed-point codec (T9)
in brick-sorted slots; the force is a two-level split (a coarse PM mesh plus tiled fine
short-range meshes); the integrator is BullFrog with the LCDM growth; initial conditions are
generated streamed through an out-of-core FFT. On GPUs the state stays in host memory and every
per-step phase runs on the cards, streamed slab by slab.

User documentation is in `docs/`; start with `docs/getting_started.md`.

## Layout

- `src/inexor/` -- the package. `engine.py` (config, step, run, checkpoints), `state.py`
  (slot state, migrate, repack), `codec.py` (T9), `forces.py` + `painting.py` (two-level
  force), `integrate.py` (BullFrog), `icgen.py` + `ooc_fft.py` (streamed ICs), `summary.py`
  (P(k) card), `export.py` (particle export), `plan.py` (memory planner), `device/` (the GPU
  lane).
- `scripts/run/` -- drivers (`realization.py` CPU lane: ics / run / export / card;
  `device_run.py` GPU lane; `device_ics.py`) and Slurm job scripts.
- `scripts/compare/` -- the DISCO-DJ cross-check and the EE2/CAMB boost comparison.
- `tests/` -- pytest.
- `runs/` and `data/` are gitignored outputs.

## Conventions

- Environments: pixi, `default` (CPU JAX) and `gpu` (CUDA 12, linux-64 and linux-aarch64).
  Use `pixi run --frozen ...` for ad-hoc commands so the lock file is not rewritten; any
  dependency change is committed together with the regenerated `pixi.lock`.
- Tasks: `pixi run test` (full suite, `-n auto`), `test-fast`, `test-det` (bit-equality tests
  on GPU with XLA deterministic ops), `lint` (ruff). Do not run `pixi run format`
  repo-wide: the tree is not in ruff-format style.
- The library never toggles `jax_enable_x64`; callers enable it. Host-side coefficient tables
  are numpy float64.
- Wrap, never clamp: no saturating operation may touch integer state; overflow raises.
- `.py` files are ASCII-only (`->`, `x`, spelled-out Greek). ruff, line length 100.
- Comments and docstrings state the current contract concisely; no development history.

## Tests and lanes

- Most tests run on the CPU backend. Multi-card tests need four devices and skip otherwise;
  on CPU, run them with `XLA_FLAGS=--xla_force_host_platform_device_count=4`.
- `detflag` tests assert bit equality; on a GPU they need `XLA_FLAGS=--xla_gpu_deterministic_ops=true`
  and are skipped visibly without it (`pixi run test-det` runs them). On CPU they run, except
  the across-ranks gates on macOS (below).
- Some gates are CPU-backend only and skip on a GPU; the executor's worker pool refuses a
  non-CPU parent, so any test that builds a `TilePool` must run in the CPU lane.
- The across-ranks gates (`tests/test_ranks_*.py`, loopback ranks) run with `pixi run
  test-ranks`: 8 emulated devices and `--xla_cpu_multi_thread_eigen=false`. On macOS the
  threaded XLA-CPU pool makes a jitted FFT's bytes vary run to run (at the 48^3 tile mesh of
  `cdev8-tile32`), so `test` skips them there.
- Tests that need real MPI processes (`tests/test_comm_mpi.py`, `tests/test_driver_mpi.py`)
  skip without mpi4py; `scripts/run/mpi_lane.sh` runs them in a throwaway `pixi exec` env.

## Cluster notes

- On a node without a GPU, set `JAX_PLATFORMS=cpu` for anything run in the `gpu` env: the CUDA
  plugin raises at initialization otherwise. Installing the `gpu` env on such a node needs
  `CONDA_OVERRIDE_CUDA=12`.
- Streaming host-resident state through the GPUs needs `XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB`
  set in the job (the job scripts set it).
- The job scripts in `scripts/run/` and `scripts/compare/` take their run root, checkout and
  inputs from environment variables and refuse to start without the required ones; pass the
  account with `sbatch -A`.
