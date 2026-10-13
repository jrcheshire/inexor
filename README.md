# inexor

inexor is a particle-mesh N-body code in JAX built around memory per particle, so that a
large cosmological realization runs on a single node. A particle is stored as a 9-byte
fixed-point code in brick-sorted slots, about 11.6 bytes per particle including allocation
spare at the production layout. On GPUs that state stays in host memory and every per-step
phase streams it through the cards slab by slab, so the particle count is bounded by the
node's host memory rather than by GPU memory.

- The force is split between a coarse PM mesh and tiled fine short-range meshes, both painted
  with integer arithmetic so the result does not depend on particle order.
- Time stepping is BullFrog with the LCDM growth factors.
- Initial conditions (2LPT, optionally with local f_NL) are generated streamed through an
  out-of-core FFT and written as slabs, so no full-size float array is ever held.
- The same code runs on a multi-core CPU, on the GPUs of one node, or across nodes under MPI;
  checkpoints are the same bytes at any rank count.

## Install

Environments are managed with [pixi](https://pixi.sh):

```bash
pixi install --locked            # CPU JAX (osx-arm64, linux-64, linux-aarch64)
pixi install -e gpu --locked     # CUDA 12 JAX (linux-64, linux-aarch64)
```

`pip install .` also works. The GPU lane has been run on GH200 and GB200 nodes; see
[docs/getting_started.md](docs/getting_started.md#install) for versions and hardware.

## A first run

A complete 128^3 realization in a 64 Mpc/h box on a laptop CPU, one process per phase (a few
minutes, almost all of it in `run`):

```bash
W=runs/cdev8
for phase in ics run card export; do
  pixi run --frozen python scripts/run/realization.py $phase --config cdev8 --workdir $W --tile-workers 2
done
```

This writes the initial conditions, two alternating checkpoints, a P(k) card
(`$W/realization_pk.json`) and the final particles (`$W/export/x.npy`, `$W/export/v.npy`).
[docs/getting_started.md](docs/getting_started.md) explains each phase and how to load the
outputs.

## Documentation

- [Getting started](docs/getting_started.md): install, a first run, loading outputs, sizing a
  run, tests
- [Running](docs/running.md): the CPU and GPU drivers, segments and resume, multi-node runs,
  the Slurm job scripts used for the paper's runs on TACC Vista, `python -m inexor.export`
- [Configuration](docs/configuration.md): presets, resolution and layout knobs, the memory
  planner, environment variables
- [Outputs](docs/outputs.md): IC slab, checkpoint, P(k) card and particle export formats
- [Architecture](docs/architecture.md): how the engine works
- [scripts/validation](scripts/validation/README.md): the measurements behind the paper's
  validation (force accuracy, high-k power spectrum, initial conditions, velocities)
- [scripts/compare](scripts/compare/README.md): the comparisons with DISCO-DJ and with
  EuclidEmulator2 / HMcode

## Tests

```bash
pixi run test          # full suite on the CPU backend
pixi run test-ranks    # the across-ranks gates on 8 emulated devices
pixi run lint
```

See [docs/getting_started.md](docs/getting_started.md#tests) for the GPU and MPI lanes.

## Citing

If you use inexor, please cite it using the metadata in [CITATION.cff](CITATION.cff). The
paper describing the code is in preparation.

## License

BSD-3-Clause; see [LICENSE](LICENSE).
