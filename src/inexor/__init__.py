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

from .adjoint import evolve_grad
from .config import PLANCK, BoxConfig, Cosmology, QuantConfig, TimeConfig
from .integrate import evolve, evolve_float, replay_roundtrip, simulate

__all__ = [
    "PLANCK",
    "BoxConfig",
    "Cosmology",
    "QuantConfig",
    "TimeConfig",
    "evolve",
    "evolve_float",
    "evolve_grad",
    "replay_roundtrip",
    "simulate",
]

__version__ = "0.0.1.dev0"
__author__ = "James Cheshire"
