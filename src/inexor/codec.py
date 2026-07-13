"""NOVEL CORE 1: lattice codecs, STE boundary, and the w-frame scale ladder.

Migrated verbatim from scripts/_m0_common.py (the M0-verdicted bits;
architecture.md Secs. 3-4). One structural change vs the probe module:
build_ladder takes the BullFrog weights as PLAIN ARRAYS (pure function), so
codec.py never imports the integrator coefficient layer -- integrate.py owns
the composition (ladder_for_schedule) per the architecture module layout.

Invariants enforced here:
- wrap-never-clamp (D-007): no saturating op touches integer state; overflow
  is monitored loudly via W_ABS_WARN / w_headroom_bits, never corrected.
- float->int ALWAYS routes through int32 (rint_i): float->int16 overflow is
  backend-defined, int32->int16/uint16 narrowing is guaranteed modular.
- |alpha_k| >= alpha_floor (D-012): near-zero alpha collapses the ladder
  (kappa ~ 1/P diverges); build_ladder refuses unfit schedules at setup time.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

INT16_MAX = 32767
U16_MOD = 2**16

# D-007 monitor threshold: |w| above 0.9 * INT16_MAX means imminent wrap ->
# wrong physics (never wrong gradients). Drivers warn loudly, never clamp.
W_ABS_WARN = int(0.9 * INT16_MAX)


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


def imask_add(x32, inc32, bits):
    """B-bit unsigned modular add on int32 storage (R4's quantization-width knob).

    bits=16 is bit-identical to the uint16 iadd path (asserted in tests).
    """
    return (x32 + inc32) & (2**bits - 1)


def imask_sub(x32, inc32, bits):
    return (x32 - inc32) & (2**bits - 1)


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


# ============================================================================
# Encode / decode
# ============================================================================


def encode_x(x_phys, s_x):
    """Physical positions -> uint16 box lattice (nearest; modular)."""
    return (rint_i(x_phys / s_x) & (U16_MOD - 1)).astype(jnp.uint16)


def encode_w(v_phys, s_w0):
    """D-time velocities -> int16 w-frame at P_0 = 1 (nearest; modular narrow)."""
    return rint_i(v_phys / s_w0).astype(jnp.int16)


def dequant_x(x_int, s_x, fdtype=jnp.float32):
    return x_int.astype(fdtype) * jnp.asarray(s_x, dtype=fdtype)


def dequant_w(w_int, s_eff, fdtype=jnp.float32):
    """Decode w -> velocity with the CURRENT ladder scale s_eff = s_w0 * P_k."""
    return w_int.astype(fdtype) * jnp.asarray(s_eff, dtype=fdtype)


# ============================================================================
# The w-frame scale ladder (architecture.md Sec. 4; host float64)
# ============================================================================


@dataclass(frozen=True)
class Ladder:
    """Per-step w-frame ladder constants (architecture.md Sec. 4), float64.

    kick:  w' = w + rint(kappa_k * g)            (purely additive -> JANUS-legal)
    drift: x += rint(c1_k * w)  pre-kick;  x += rint(c2_k * w') post-kick
    decode: v = s_w0 * P_k * w  (P has K+1 entries; P[0] = 1)
    """

    a_steps: np.ndarray  # (K+1,)
    D_steps: np.ndarray  # (K+1,) growth at the step edges
    alphas: np.ndarray  # (K,)
    betas: np.ndarray  # (K,)
    dD: np.ndarray  # (K,)
    D_mid: np.ndarray  # (K,)
    P: np.ndarray  # (K+1,) ladder scale; P[0] = 1
    c1: np.ndarray  # (K,) pre-kick half-drift, w -> x-lattice units
    c2: np.ndarray  # (K,) post-kick half-drift
    kappa: np.ndarray  # (K,) kick, force -> w-lattice units
    s_w0: float
    s_x: float

    @property
    def n_steps(self):
        return len(self.alphas)

    @property
    def bits_consumed(self):
        """Ladder dynamic-range cost in bits: log2(1 / min_k |P_k|)."""
        return float(np.log2(1.0 / np.abs(self.P).min()))


def build_ladder(a_steps, D_steps, alphas, betas, dD, D_mid, s_w0, s_x, alpha_floor=0.05):
    """Build the w-frame ladder from per-step BullFrog weights. Pure arrays in.

    Refuses unfit schedules: any |alpha_k| < alpha_floor makes kappa ~ 1/P_{k+1}
    diverge (the ladder collapses) -- D-012. Measured (M0): linear-in-a
    schedules from a_i=0.04 cross alpha=0 near K=11; log schedules from a_i=0.1
    keep alpha_1 >= 0.48 for K=5-15 at ~5-bit cost.

    The weights come from integrate.bullfrog_table (or any compatible source);
    integrate.ladder_for_schedule is the standard composition.
    """
    a_steps = np.asarray(a_steps, dtype=np.float64)
    D_steps = np.asarray(D_steps, dtype=np.float64)
    alphas = np.asarray(alphas, dtype=np.float64)
    betas = np.asarray(betas, dtype=np.float64)
    dD = np.asarray(dD, dtype=np.float64)
    D_mid = np.asarray(D_mid, dtype=np.float64)
    bad = np.abs(alphas) < alpha_floor
    if bad.any():
        ks = np.nonzero(bad)[0]
        raise ValueError(
            f"unfit schedule: |alpha| < {alpha_floor} at step(s) {ks.tolist()} "
            f"(alpha = {alphas[ks].round(5).tolist()}); near-zero alpha collapses "
            "the w-frame ladder (kappa ~ 1/P diverges). Use log spacing / later a_init, "
            "or the FastPM integrator (alpha == 1, no ladder)."
        )
    P = np.concatenate([[1.0], np.cumprod(alphas)])
    kappa = betas / (D_mid * P[1:] * s_w0)
    c1 = 0.5 * dD * s_w0 * P[:-1] / s_x
    c2 = 0.5 * dD * s_w0 * P[1:] / s_x
    return Ladder(a_steps, D_steps, alphas, betas, dD, D_mid, P, c1, c2, kappa, s_w0, s_x)


def s_w0_policy(v_max0, P_final, c_growth=2.5, margin=0.9):
    """Initial w scale so late-time |w| stays inside int16.

    s_w0 = c_growth * max|v_0| / (margin * 32767 * |P_K|). c_growth is the
    velocity growth factor over the run; R2 measured 1.94 (128^3 exact LCDM,
    K=5-15 log schedules) -- default 2.5 ratified at the M0 gate review
    (D-011).
    """
    return c_growth * v_max0 / (margin * 32767.0 * abs(P_final))


def w_headroom_bits(w_int):
    """Bits of headroom between max|w| and the int16 edge (D-007 monitor).

    Returns log2(INT16_MAX / max|w|); <= log2(1/0.9) ~ 0.152 means the
    W_ABS_WARN threshold has been crossed. Pure diagnostic -- never gates,
    never clamps.
    """
    max_abs = jnp.max(jnp.abs(w_int.astype(jnp.int32)))
    return jnp.log2(INT16_MAX / jnp.maximum(max_abs, 1).astype(jnp.float32))
