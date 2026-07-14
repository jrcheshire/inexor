"""inexor: exactly reversible, compressed-state differentiable N-body in JAX.

A particle-mesh N-body code whose phase space lives on a fixed-point integer
lattice (int16 default, int8 opt-in), making time evolution bit-exactly
reversible (JANUS pattern) and the reverse-mode adjoint an exact replay with
memory independent of the number of time steps -- at 12 (or 6) bytes per
particle of persistent state (CUBE pattern).

Status: M1 (forward PM). The forward path (BullFrog w-frame ladder +
exact-KDK/FastPM integer integrators, deterministic int paint, ZA/2LPT/f_NL
ICs) is package code; the custom_vjp exact-replay adjoint is M2. M0's five
kill-or-confirm probes all passed (docs/decisions.md D-010..D-012).

House rule: this library NEVER touches jax.config -- callers opt into x64.
"""

from importlib.metadata import version as _metadata_version

from .adjoint import adjoint_grad_fnl, adjoint_grad_ic, evolve_grad
from .config import PLANCK, BoxConfig, Cosmology, QuantConfig, TimeConfig
from .integrate import evolve, evolve_float, replay_roundtrip, simulate

__all__ = [
    "PLANCK",
    "BoxConfig",
    "Cosmology",
    "QuantConfig",
    "TimeConfig",
    "adjoint_grad_fnl",
    "adjoint_grad_ic",
    "evolve",
    "evolve_float",
    "evolve_grad",
    "replay_roundtrip",
    "simulate",
]

# Single-sourced from pyproject.toml's [project] version. It is declared there
# STATICALLY (not hatch-dynamic) because pixi must read this package's metadata
# to solve foreign-platform envs -- notably the linux-aarch64 gpu env for Vista
# -- and a dynamic version forces it to execute hatchling with an interpreter it
# cannot have for a platform it cannot run. Do not reintroduce dynamic version.
__version__ = _metadata_version("inexor")
__author__ = "James Cheshire"
