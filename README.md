# inexor

A memory-lean particle-mesh N-body engine in JAX, built so that a large cosmological
realization fits on a single node.

- Particle state is a compressed fixed-point code (T9) in brick-sorted slots with a shared
  spare arena: 9 bytes per particle of payload, about 11.6 in all with 10% spare
  capacity per brick.
- The force is split between a coarse PM mesh and tiled fine short-range meshes, both painted
  with integer arithmetic so the result does not depend on particle order.
- Time stepping is BullFrog with the LCDM growth factors.
- Initial conditions (2LPT) are generated streamed through an out-of-core FFT and written as
  slabs, so no full-size float array is ever held.
- On GPUs the state stays in host memory and every per-step phase runs on the cards, streamed
  slab by slab.

## Install

```bash
pixi install --locked            # CPU JAX
pixi install -e gpu --locked     # CUDA 12 JAX (linux-64, linux-aarch64)
```

## A first run

A complete 128^3 realization in a 64 Mpc/h box on a laptop CPU, one process per phase:

```bash
W=runs/cdev8
for phase in ics run card export; do
  pixi run --frozen python scripts/run/realization.py $phase --config cdev8 --workdir $W --tile-workers 2
done
```

This writes the initial conditions, checkpoints, a P(k) card (`realization_pk.json`) and the
final particles (`export/x.npy`, `export/v.npy`). See
[docs/getting_started.md](docs/getting_started.md) for what each phase does and how to load
the outputs.

## Documentation

- [Getting started](docs/getting_started.md): install, a first run, sizing, tests
- [Running](docs/running.md): the drivers, segments and resume, the GPU lane, Slurm job scripts
- [Configuration](docs/configuration.md): presets, resolution and layout knobs, the memory
  planner, environment variables
- [Outputs](docs/outputs.md): slab, checkpoint, P(k) card and particle export formats
- [Architecture](docs/architecture.md): how the engine works
- [scripts/compare](scripts/compare/README.md): the DISCO-DJ and EuclidEmulator2 comparison tools

## Tests

```bash
pixi run test        # full suite, CPU
pixi run lint
```

## License

BSD-3-Clause.
