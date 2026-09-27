# Getting started

inexor is a particle-mesh N-body engine in JAX, built so that a large cosmological
realization fits on a single node. Particle state is stored as a compressed fixed-point
code in brick-sorted slots, the gravitational force is split between a coarse PM mesh and
tiled fine short-range meshes, and the initial conditions (2LPT) are generated streamed
through an out-of-core FFT. The same engine runs on a multi-core CPU or, with the state kept
in host memory, on one or more GPUs.

## Install

inexor uses [pixi](https://pixi.sh). From the repository root:

```bash
pixi install --locked            # default env: CPU JAX (osx-arm64, linux-64, linux-aarch64)
pixi install -e gpu --locked     # gpu env: CUDA 12 JAX (linux-64, linux-aarch64 only)
```

- The `gpu` env declares a CUDA 12 system requirement. To install it on a node without a
  GPU (a login node, for example), set `CONDA_OVERRIDE_CUDA=12`:
  `CONDA_OVERRIDE_CUDA=12 pixi install -e gpu --locked`.
- Run ad-hoc commands with `pixi run --frozen ...` so the lock file is never rewritten.
  Add `-e gpu` to use the GPU env.
- On a node without a GPU, anything run in the `gpu` env needs `JAX_PLATFORMS=cpu`.

## A first run on a laptop CPU

This runs one complete realization at the `cdev8` preset: 128^3 particles in a
64 Mpc/h box, 40 steps from a = 0.1 to a = 1. The driver is `scripts/run/realization.py`,
and each phase runs as its own process. `--tile-workers 2` sizes the worker pool for a
laptop; the default is 16.

On an Apple M4 Max the four phases took about 30 s, 4 min, 2 s and 1 s, with peak memory
0.7, 4.6, 0.8 and 0.6 GB. `--tile-workers` sets the size of the worker pool, not the number
of cores: JAX's own CPU thread pool still uses the cores it finds (the `run` phase averaged
about six).

```bash
W=runs/cdev8
pixi run --frozen python scripts/run/realization.py ics    --config cdev8 --workdir $W --tile-workers 2
pixi run --frozen python scripts/run/realization.py run    --config cdev8 --workdir $W --tile-workers 2
pixi run --frozen python scripts/run/realization.py card   --config cdev8 --workdir $W --tile-workers 2
pixi run --frozen python scripts/run/realization.py export --config cdev8 --workdir $W --tile-workers 2
```

`runs/` is ignored by git. The card needs no extra flags at this size. By default it bins
P(k) into 64 bins from 0 to half the coarse mesh's Nyquist frequency, and it drops any bin
with 100 or fewer modes (`--min-weight`). At cdev8 enough bins pass this cut, so the card is
not empty. At a = 1, every bin that passes may lie above the nonlinear scale. The card
prints a note when that happens, which is expected in a 64 Mpc/h box.

What each phase writes, all under `--workdir`:

| Phase | Writes |
|---|---|
| `ics` | `t9_slab_NNNN.npz` (one per x-slab of bricks), `manifest.json`, `realization_ics.json`. The `stage/` intermediates are removed unless you pass `--keep-stage` |
| `run` | `ckpt/gen0/` and `ckpt/gen1/` (two alternating checkpoint generations, every `--checkpoint-every` = 5 steps), `realization_run_00_40.json` (named `realization_run_<from>_<to>.json` for each segment) |
| `card` | `realization_pk.json` |
| `export` | `export/x.npy`, `export/v.npy`, `export/export.json` (the header), `realization_export.json` |

The `realization_*.json` files record the wall time, peak host memory, the configuration and
the commit for each phase. To card the initial conditions at a = 0.1 as well, pass
`--ic-dir $W` to `card`. That writes `realization_pk_ics.json` next to the card of the final
state:

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
print(head["n_particles"], head["box_size"], head["units"])

card = json.load(open("runs/cdev8/realization_pk.json"))["summary"]
k, p, p_lin, z = card["k_mean"], card["p"], card["p_oracle"], card["z_profile"]
print(card["a_out"], card["k_nonlinear"], card["n_bins"])
```

`load_particles(..., mmap=False)` loads the arrays into memory and verifies their crc32
sums. The rows are in the engine's brick-major order and do not carry particle identity. The
card holds the measured P(k), linear theory averaged over each bin's modes (`p_oracle`), the
per-bin z score, mode counts and bin edges. It does not give a pass/fail verdict.

## Sizing a bigger run

`python -m inexor.plan` does the memory arithmetic for a configuration against the budgets
you give it. It runs nothing. The CPU driver's defaults are `--slack 0.20 --arena-frac 0.20`,
while the planner's are 0.10 / 0.01, so pass the driver's values and your worker count to
price a CPU run:

```bash
pixi run --frozen python -m inexor.plan --preset c-gh --workers 16 --slack 0.20 --arena-frac 0.20 \
    --host-gb <host RAM in GB> --shm-gb <size of /dev/shm in GB>
```

The output lists the state, mesh and per-phase terms, then a `BINDING TERMS` block. Its
verdict line reads `against --host-gb N: FITS (0.xx the budget)` or `DOES NOT FIT`. It compares
a **lower bound** on the run's peak, not a prediction, so leave headroom. Two more things to
check:

- if the load stage peaks higher than the run, it is marked `<- BINDING`;
- the pooled CPU lane keeps the state in `/dev/shm`, which has its own verdict (`--shm-gb`).

With no budget given, the planner prints no verdict. For the GPU lane, add
`--backend device --n-gpus <cards> --device-gb <memory per card>`. Presets: `smoke`, `cdev8`,
`cdev`, `cgh64`, `c-1024`, `c-gh`, `c-hero`. You can also give an explicit geometry with
`--n-part --box --n-fine --n-coarse --tile --buf`.

## Tests

```bash
pixi run test        # full suite, pytest -n auto
pixi run test-fast   # skips tests marked slow
pixi run test-det    # the bit-equality (detflag) tests, with XLA deterministic ops
```

Most tests run on the CPU backend in the default env. Multi-device tests need four JAX devices
and skip without them. On CPU, provide four with
`XLA_FLAGS=--xla_force_host_platform_device_count=4 pixi run test`. On a GPU node, run `pixi run -e gpu test` and `pixi run -e gpu test-det` (or `pixi run -e gpu
test-all` for both). There, `test` skips the `detflag` tests with a visible reason because
they need `--xla_gpu_deterministic_ops=true` in their own process, and some CPU-only tests
also skip.

## Next

- [running.md](running.md): the drivers, segments and resume, the GPU lane, Slurm job scripts
- [configuration.md](configuration.md): presets and engine configuration
- [outputs.md](outputs.md): file formats of ICs, checkpoints, exports and cards
- [architecture.md](architecture.md): how the engine is organized
