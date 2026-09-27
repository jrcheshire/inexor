"""inexor: a memory-lean particle-mesh N-body engine in JAX.

Particle state is compressed (codec, layout, state), the force is a two-level PM split (coarse
mesh plus tiled short-range), time stepping is BullFrog with LCDM growth (integrate, engine),
initial conditions stream through an out-of-core FFT (ic, ooc_fft), and `device` streams
host-resident state through GPUs. Start with `config`, `engine` and `state`.

The library never touches jax.config: callers opt into x64.
"""

from importlib.metadata import version as _metadata_version

from .config import PLANCK, BoxConfig, Cosmology

__all__ = [
    "PLANCK",
    "BoxConfig",
    "Cosmology",
]

# Declared statically in pyproject.toml (not hatch-dynamic): pixi must read package metadata
# to solve foreign-platform envs, which a dynamic version would prevent.
__version__ = _metadata_version("inexor")
__author__ = "James Cheshire"
