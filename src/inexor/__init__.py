"""inexor: a memory-floor particle-mesh mock engine in JAX.

v2 maximizes (volume x halo-grade resolution) per single GPU or node, trading
compute freely for memory. These simulations were never time-expensive -- 2LPT
and BullFrog take very few steps -- but they are notoriously memory-expensive,
routinely taking large fractions of a cluster. Subverting that is the point.

The ratified architecture (D-v2-14..18, 2026-08-08):
- **State** is T9 at 10.15 B/p all-in: int8 positions relative to a 1.0 Mpc/h
  bucket at quantum `fine_cell/64`, int16 velocities, on a brick-sorted layout
  with per-bucket capacity. Host-resident state larger than HBM streams.
- **Force** is a two-level PM split -- a global coarse mesh plus a short-range
  kernel solved tile by tile on a padded sub-box -- and is never materialized
  globally, because its only consumer is an elementwise kick and ownership is a
  partition.
- **Accuracy** is |dP/P| <= 3e-2 in-band against a monolithic reference
  (D-v2-9), read on the UNCORRECTED tiling split; the low-k split error is a
  correctable transfer calibrated once off-box and applied per mock (D-v2-11).

Status: building M-v2-1 (codec + layout). `docs/plan-plan-v2.md` is the entry
point; `docs/decisions.md` is the ADR log; the probes in `scripts/v2_*.py` are
the ratified measurement oracles and are not package code.

v1 -- exactly reversible, compressed-state *differentiable* N-body -- was HALTED
2026-07-14 when its premise measured false, and its machinery was retired from
the package on 2026-08-08. Read `docs/retrospective.md` before re-proposing
anything v1-flavoured. Differentiability is deferred, not precluded (D-v2-3).

House rule: this library NEVER touches jax.config -- callers opt into x64.
"""

from importlib.metadata import version as _metadata_version

from .config import PLANCK, BoxConfig, Cosmology, QuantConfig, TimeConfig

__all__ = [
    "PLANCK",
    "BoxConfig",
    "Cosmology",
    "QuantConfig",
    "TimeConfig",
]

# Single-sourced from pyproject.toml's [project] version. It is declared there
# STATICALLY (not hatch-dynamic) because pixi must read this package's metadata
# to solve foreign-platform envs -- notably the linux-aarch64 gpu env for Vista
# -- and a dynamic version forces it to execute hatchling with an interpreter it
# cannot have for a platform it cannot run. Do not reintroduce dynamic version.
__version__ = _metadata_version("inexor")
__author__ = "James Cheshire"
