"""inexor: exactly reversible, compressed-state differentiable N-body in JAX.

A particle-mesh N-body code whose phase space lives on a fixed-point integer
lattice (int16 default, int8 opt-in), making time evolution bit-exactly
reversible (JANUS pattern) and the reverse-mode adjoint an exact replay with
memory independent of the number of time steps -- at 12 (or 6) bytes per
particle of persistent state (CUBE pattern).

Status: pre-M0 (design documents only; see docs/architecture.md and
docs/roadmap.md). No implementation yet.
"""

__version__ = "0.0.1.dev0"
__author__ = "James Cheshire"
