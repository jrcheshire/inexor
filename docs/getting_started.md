# Getting started

inexor is a particle-mesh N-body code in JAX built around memory per particle, so that a large
cosmological realization runs on a single node. Particle state is a compressed fixed-point
code in brick-sorted slots, the gravitational force is split between a coarse PM mesh and
tiled fine short-range meshes, and the initial conditions (2LPT) are generated streamed
through an out-of-core FFT. The same engine runs on a multi-core CPU or, with the state kept
in host memory and streamed through the cards, on the GPUs of one node or across nodes.

## Install

inexor uses [pixi](https://pixi.sh). From the repository root:

```bash
pixi install --locked            # default env: CPU JAX (osx-arm64, linux-64, linux-aarch64)
pixi install -e gpu --locked     # gpu env: CUDA 12 JAX, mpi4py, MPICH (linux-64, linux-aarch64)
```

- The `gpu` env declares a CUDA 12 system requirement. To install it on a node without a
  GPU (a login node, for example), set `CONDA_OVERRIDE_CUDA=12`:
  `CONDA_OVERRIDE_CUDA=12 pixi install -e gpu --locked`.
- Run ad-hoc commands with `pixi run --frozen ...` so the lock file is never rewritten.
- Without pixi, `pip install .` works too (Python >= 3.11, jax >= 0.10, numpy >= 2, scipy);
  `pixi.lock` pins the versions the tests and the paper's runs used.

**Hardware.** The CPU lane runs anywhere JAX does. The GPU lane has run on NVIDIA Grace Hopper
(GH200) and Grace Blackwell (GB200) nodes, where host memory is attached to the GPU over
NVLink-C2C. It uses ordinary host-device copies, so it does not depend on that link; on an x86
node with PCIe-attached H100s the gpu env runs and host-to-GPU streaming measured about 7x slower
than on a GH200, but the engine itself has not been run there.
  Add `-e gpu` to use the GPU env.
- On a node without a GPU, anything run in the `gpu` env needs `JAX_PLATFORMS=cpu`.

## A first run on a laptop CPU

This runs one complete realization at the `cdev8` preset: 128^3 particles in a 64 Mpc/h box,
40 steps from a = 0.1 to a = 1. The driver is `scripts/run/realization.py`, and each phase runs
as its own process:

```bash
W=runs/cdev8
pixi run --frozen python scripts/run/realization.py ics    --config cdev8 --workdir $W --tile-workers 2
pixi run --frozen python scripts/run/realization.py run    --config cdev8 --workdir $W --tile-workers 2
pixi run --frozen python scripts/run/realization.py card   --config cdev8 --workdir $W --tile-workers 2
pixi run --frozen python scripts/run/realization.py export --config cdev8 --workdir $W --tile-workers 2
```

`--tile-workers 2` sizes the worker pool for a laptop (the default is 16). It sets the number
of worker processes, not the number of cores: each worker's XLA CPU runtime still uses the
cores it finds. On an Apple M4 Max shared with other jobs, `ics`, `card` and `export` took a
few seconds each and `run` about 4.5 minutes (6.6 s per step, using about six cores); `run`
peaked at about 9 GB summed over the driver and its two workers, the other phases below 1 GB.

`runs/` is ignored by git. The card needs no extra flags at this size. By default it bins P(k)
into 64 bins from 0 to half the coarse mesh's Nyquist frequency and drops any bin with 100 or
fewer modes (`--min-weight`). At a = 1 in a 64 Mpc/h box every bin that passes may lie above
the nonlinear scale; the card prints a note when that happens.

What each phase writes, all under `--workdir`:

| Phase | Writes |
|---|---|
| `ics` | `t9_slab_NNNN.npz` (one per x-slab of bricks), `manifest.json`, `realization_ics.json`. The `stage/` intermediates are removed unless you pass `--keep-stage` |
| `run` | `ckpt/gen0/` and `ckpt/gen1/` (two alternating checkpoint generations, written every `--checkpoint-every` = 5 steps), `realization_run_00_40.json` (`realization_run_<from>_<to>.json` for each segment) |
| `card` | `realization_pk.json` |
| `export` | `export/x.npy`, `export/v.npy`, `export/export.json` (the header), `realization_export.json` |

The `realization_*.json` files record the wall time, peak host memory and configuration of
each phase. To card the initial conditions at a = 0.1 as well, pass `--ic-dir $W` to `card`.
That writes `realization_pk_ics.json` next to the card of the final state:

```bash
pixi run --frozen python scripts/run/realization.py card --config cdev8 --workdir $W --tile-workers 2 --ic-dir $W
```

### Loading the outputs

```python
import json
from inexor.export import load_particles

head, x, v, ids = load_particles("runs/cdev8/export")   # memmaps; ids is None here
# x: (N, 3) float32, comoving Mpc/h in [0, box_size)
# v: (N, 3) float32, peculiar km/s at head["a"]
print(head["n_particles"], head["box_size"], head["a"], head["units"])

card = json.load(open("runs/cdev8/realization_pk.json"))["summary"]
k, p, p_lin, z = card["k_mean"], card["p"], card["p_oracle"], card["z_profile"]
print(card["a_out"], card["k_nonlinear"], card["n_bins"])
```

`load_particles(..., mmap=False)` loads the arrays into memory and verifies their crc32 sums;
an export written one part per rank (the GPU driver's) must be read this way or with
`inexor.export.iter_particle_parts`. The rows are in the engine's brick-major order and do not
carry particle identity. The card holds the measured P(k), linear theory averaged over each
bin's modes (`p_oracle`), the per-bin z score, mode counts and bin edges. It gives no pass/fail
verdict. The formats are in [outputs.md](outputs.md).

## Sizing a bigger run

`python -m inexor.plan` does the memory arithmetic for a configuration against the budgets
you give it. It runs nothing. The CPU driver's layout defaults are `--slack 0.20
--arena-frac 0.20`, while the planner's (and the GPU driver's) are 0.10 / 0.01, so pass the
CPU driver's values and your worker count to price a CPU run:

```bash
pixi run --frozen python -m inexor.plan --preset c-gh --workers 16 --slack 0.20 --arena-frac 0.20 \
    --host-gb <host RAM in GB> --shm-gb <size of /dev/shm in GB>
```

The output lists the state, mesh and per-phase terms, then a `BINDING TERMS` block. Its
verdict line reads, for example, `against --host-gb 512.0: FITS (0.62x the budget)`, or
`DOES NOT FIT`. It
compares a **lower bound** on the run's peak, not a prediction, so leave headroom. Two more
things to check:

- if the load stage peaks higher than the run, it is marked `<- BINDING`;
- the pooled CPU lane keeps the state in shared memory, which has its own verdict
  (`--shm-gb`).

With no budget given, the planner prints no verdict. For the GPU lane, add
`--backend device --n-gpus <cards per node> --device-gb <memory per card>`, and
`--n-nodes N` for a run across nodes (it prices the busiest node). Presets: `smoke`, `cdev8`,
`cdev8-tile32`, `cdev`, `cgh64`, `c-1024`, `c-gh`, `c-hero`, `c-8192` (see
[configuration.md](configuration.md#presets)). You can also give an explicit geometry with
`--n-part --box --n-fine --n-coarse --tile --buf`. A preset with `--n-fine` scales the tile
buffer with the fine mesh, as the drivers do, unless `--buf` is given.

## Tests

```bash
pixi run test        # full suite (pytest -n auto)
pixi run test-fast   # skips tests marked slow
pixi run test-ranks  # the across-ranks gates (tests/test_ranks_*.py) on 8 emulated devices
pixi run test-det    # the bit-equality (detflag) tests, with XLA deterministic ops
pixi run lint        # ruff
```

- Most tests run on the CPU backend in the default env. Multi-card tests need four JAX
  devices and skip without them; on CPU, provide four with
  `XLA_FLAGS=--xla_force_host_platform_device_count=4 pixi run test`.
- `test-ranks` runs the across-ranks gates with a single-threaded XLA-CPU pool
  (`--xla_cpu_multi_thread_eigen=false`). On macOS `test` skips them, because the threaded
  pool makes a jitted FFT's bytes vary from run to run there.
- On a GPU node, run `pixi run -e gpu test` and `pixi run -e gpu test-det` (or
  `pixi run -e gpu test-all` for both). There `test` skips the `detflag` tests with a visible
  reason, since they need `--xla_gpu_deterministic_ops=true` in their own process, and some
  CPU-only tests also skip.
- The tests that need real MPI processes (`tests/test_comm_mpi.py`,
  `tests/test_driver_mpi.py`, and the job-script tests in `tests/test_job_scripts.py`) skip
  without mpi4py. `scripts/run/mpi_lane.sh` runs them in a throwaway `pixi exec` env with the
  lock's python, jax, jaxlib and numpy plus mpi4py and MPICH, leaving the lock and the default
  env untouched:

  ```bash
  scripts/run/mpi_lane.sh                                   # the MPI test files
  scripts/run/mpi_lane.sh tests/test_comm_mpi.py -k raise   # a selection
  ```

## Next

- [running.md](running.md): the drivers, segments and resume, the GPU lane, multi-node runs,
  the Slurm job scripts
- [configuration.md](configuration.md): presets and engine configuration
- [outputs.md](outputs.md): file formats of ICs, checkpoints, exports and cards
- [architecture.md](architecture.md): how the engine is organized
