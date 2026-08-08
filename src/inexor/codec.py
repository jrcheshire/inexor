"""Integer lattice primitives and the straight-through autodiff boundary.

**What used to be here and is gone (2026-08-08, v1 retirement).** This module
was v1's "NOVEL CORE 1": a global uint16 position lattice spanning the box
(`encode_x`/`dequant_x`, D-004), an int16 velocity in the w-frame, and the
w-frame scale ladder (`Ladder`/`build_ladder`/`s_w0_policy`) that made
BullFrog's contracting affine kick integer-bijective so the trajectory was
bit-exactly reversible. That machinery served the v1 thesis, which measured
false on 2026-07-14 (`docs/retrospective.md`).

v2's state tier is different in kind: T9 (D-v2-14) stores positions as int8
**relative to a 1.0 Mpc/h bucket** rather than globally, at quantum
`fine_cell/64`, on a brick-sorted layout -- there is no ladder because there is
no reversibility requirement, and the velocity is a plain max-range int16. The
T9 codec lands here at M-v2-1.

What survives, and why:
- `rint_i` / `iadd` / `isub` -- the float->int and modular-arithmetic
  primitives. `painting.py` imports `rint_i`.
- `ste_round` / `ste_wrap_u` / `ste_wrap_s` -- the straight-through estimator
  boundary. Nothing calls these today. They are kept deliberately: D-v2-3 defers
  differentiability without precluding it, and this is the tested boundary a
  quantized gradient walks back through when seed VD opens. Twelve lines is a
  cheap door to leave open.

Invariants enforced here:
- wrap-never-clamp (D-007): no saturating op touches integer state; overflow
  is monitored loudly, never corrected.
- float->int ALWAYS routes through int32 (`rint_i`): float->int16 overflow is
  backend-defined, int32->int16/uint16 narrowing is guaranteed modular.
"""

import jax
import jax.numpy as jnp

# ============================================================================
# Integer lattice primitives
# ============================================================================


def rint_i(z):
    """Round-half-even to int32. Route float->int through int32 ALWAYS:
    float->int16 overflow is backend-defined; int32->int16/uint16 narrowing is
    guaranteed modular (architecture.md Sec. 4)."""
    return jnp.rint(z).astype(jnp.int32)


def iadd(state, inc32):
    """Modular lattice add: widen to int32, add, narrow back (exactly modular)."""
    return (state.astype(jnp.int32) + inc32).astype(state.dtype)


def isub(state, inc32):
    """Modular lattice subtract; exact inverse of iadd with the same inc32."""
    return (state.astype(jnp.int32) - inc32).astype(state.dtype)


# ============================================================================
# Straight-through estimators (the autodiff boundary, architecture.md Sec. 8)
# ============================================================================


def ste_round(z):
    """Straight-through rint: primal == rint(z), Jacobian == identity."""
    return z + jax.lax.stop_gradient(jnp.rint(z) - z)


def ste_wrap_u(z, mod):
    """Straight-through unsigned modular wrap: primal == z mod M, Jacobian == I."""
    return z + jax.lax.stop_gradient(jnp.mod(z, mod) - z)


def ste_wrap_s(z, mod):
    """Straight-through signed wrap into [-M/2, M/2): primal wraps, Jacobian == I."""
    half = mod / 2.0
    return z + jax.lax.stop_gradient(jnp.mod(z + half, mod) - half - z)
