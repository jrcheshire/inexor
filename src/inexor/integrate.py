"""The forward path: coefficient tables, reversible integer step kernels,
float reference steppers, drivers, and the public evolve/simulate API
(architecture.md Secs. 4 + 8 forward half; M2 wraps custom_vjp around evolve
without changing its signature).

Three integrators (roadmap M1):
- "bullfrog": DKD with an affine kick, made purely additive in the w-frame
  by the scale ladder (codec.build_ladder). K <~ 8-12 regime; the flagship.
- "fastpm":   growth-corrected KDK (Feng et al. 2016); kick natively
  additive (alpha == 1, NO ladder) -- the many-step / int8 integrator.
- "exact":    KDK with literal background integrals (mbody "exact"); the
  fallback. ~2% growth deficit at low step count (mbody docstring) --
  expected, tested at the converged limit only.

Integer state and units: x uint16 box lattice (s_x), w int16. For BullFrog
w = v_D / (s_w0 P_k) (ladder frame). For the KDK pair w = p / s_w0 at a
FIXED scale, with p the a-time momentum (mbody H0=1 units); the public API
speaks D-time v = dx/dD throughout and converts at the boundary via
G_f = a^3 E D' (v_to_p / p_to_v).

Every step kernel takes the force callable EXPLICITLY; pass the SAME object
to fwd and rev (forces.make_force_fn caches, so equal args give the same
object -- the Sec. 5 fusion mitigation).
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from . import codec
from .codec import (
    dequant_w,
    dequant_x,
    encode_w,
    encode_x,
    iadd,
    isub,
    rint_i,
    ste_round,
    ste_wrap_s,
    ste_wrap_u,
)
from .cosmology import E_of_a, growth_factor_a, growth_rate_a
from .forces import make_force_fn
from .ic import linear_density
from .lpt import lpt_ics

# ============================================================================
# Coefficient tables (host numpy/scipy float64 island; constants of the run)
# ============================================================================


def a_grid(a_init, a_final, n_steps, spacing="log"):
    """Scale-factor step edges (n_steps+1 points), 'log' or 'linear' spacing."""
    if spacing == "log":
        return np.geomspace(a_init, a_final, n_steps + 1)
    if spacing == "linear":
        return np.linspace(a_init, a_final, n_steps + 1)
    raise ValueError(f"spacing must be 'log' or 'linear', got {spacing!r}")


def _bullfrog_weights(D0, D1):
    """BullFrog (alpha, beta, dD, D_mid) for a step D0 -> D1. Verbatim mbody port.

    EdS second-order growth E = -(3/7)D^2, E' = -(6/7)D; paper Eqs 2.3-2.4
    (Rampf, List & Hahn 2024). Pure function of the two growth values.
    """
    dD = D1 - D0
    D_mid = D0 + 0.5 * dD
    E0 = -(3.0 / 7.0) * D0 * D0
    E0p = -(6.0 / 7.0) * D0
    E1p = -(6.0 / 7.0) * D1
    F_mid = (E0 + E0p * 0.5 * dD) / D_mid - D_mid
    alpha = (E1p - F_mid) / (E0p - F_mid)
    return alpha, 1.0 - alpha, dD, D_mid


class BullFrogTable(NamedTuple):
    """Per-step BullFrog weights over a schedule (float64 numpy arrays)."""

    a_steps: np.ndarray  # (K+1,)
    D_steps: np.ndarray  # (K+1,)
    alphas: np.ndarray  # (K,)
    betas: np.ndarray  # (K,)
    dD: np.ndarray  # (K,)
    D_mid: np.ndarray  # (K,)


def bullfrog_table(a_steps, cosmo, D_of_a=None):
    """BullFrog weights for a schedule; D_of_a overrides growth (EdS pin test)."""
    a_steps = np.asarray(a_steps, dtype=np.float64)
    if D_of_a is None:
        D_steps = np.array([growth_factor_a(a, cosmo) for a in a_steps])
    else:
        D_steps = np.array([D_of_a(a) for a in a_steps])
    K = len(a_steps) - 1
    alphas, betas, dD, D_mid = (np.empty(K) for _ in range(4))
    for k in range(K):
        alphas[k], betas[k], dD[k], D_mid[k] = _bullfrog_weights(D_steps[k], D_steps[k + 1])
    return BullFrogTable(a_steps, D_steps, alphas, betas, dD, D_mid)


def ladder_for_schedule(a_steps, cosmo, s_w0, s_x, alpha_floor=0.05, D_of_a=None):
    """bullfrog_table composed with codec.build_ladder (the standard path)."""
    t = bullfrog_table(a_steps, cosmo, D_of_a)
    return codec.build_ladder(
        t.a_steps, t.D_steps, t.alphas, t.betas, t.dD, t.D_mid, s_w0, s_x, alpha_floor
    )


def kick_factor(a0, a1, cosmo):
    """Exact leapfrog kick coefficient: integral of (3/2) Om / (a^2 E) da."""
    val, _ = quad(lambda a: 1.5 * cosmo.Omega_m / (a**2 * E_of_a(a, cosmo)), a0, a1)
    return val


def drift_factor(a0, a1, cosmo):
    """Exact leapfrog drift coefficient: integral of 1 / (a^3 E) da."""
    val, _ = quad(lambda a: 1.0 / (a**3 * E_of_a(a, cosmo)), a0, a1)
    return val


def _G_f(a, cosmo):
    """FastPM auxiliary G_f(a) = a^3 E(a) D'(a), D' = D f / a."""
    D = growth_factor_a(a, cosmo)
    f = growth_rate_a(a, cosmo)
    return a**3 * E_of_a(a, cosmo) * (D * f / a)


def _g_f(a, cosmo, rel=1e-5):
    h = rel * a
    return (_G_f(a + h, cosmo) - _G_f(a - h, cosmo)) / (2.0 * h)


def fastpm_drift_factor(a0, a1, a_r, cosmo):
    """FastPM drift coefficient (Feng et al. 2016, Eq. 24). Verbatim mbody port."""
    D0 = growth_factor_a(a0, cosmo)
    D1 = growth_factor_a(a1, cosmo)
    fr = growth_rate_a(a_r, cosmo)
    Dr = growth_factor_a(a_r, cosmo)
    Dpr = Dr * fr / a_r
    return (D1 - D0) / (a_r**3 * E_of_a(a_r, cosmo) * Dpr)


def fastpm_kick_factor(a0, a1, a_r, cosmo):
    """FastPM kick coefficient (Feng et al. 2016, Eq. 25). Verbatim mbody port."""
    num = _G_f(a1, cosmo) - _G_f(a0, cosmo)
    den = a_r**2 * E_of_a(a_r, cosmo) * _g_f(a_r, cosmo)
    return 1.5 * cosmo.Omega_m * num / den


def kdk_table(a_steps, cosmo, integrator="exact"):
    """Per-step KDK (k1, drift, k2) coefficients, (K, 3) float64 (mbody
    _step_coeffs port). k1 kicks over [a0, a_mid], drift over [a0, a1], k2
    over [a_mid, a1]. integrator: "exact" or "fastpm" (FastPM reference
    scales: a0 / a_mid / a1, mbody convention)."""
    a_steps = np.asarray(a_steps, dtype=np.float64)
    co = np.empty((len(a_steps) - 1, 3))
    for i in range(len(a_steps) - 1):
        a0, a1 = float(a_steps[i]), float(a_steps[i + 1])
        a_c = 0.5 * (a0 + a1)
        if integrator == "exact":
            co[i] = (
                kick_factor(a0, a_c, cosmo),
                drift_factor(a0, a1, cosmo),
                kick_factor(a_c, a1, cosmo),
            )
        elif integrator == "fastpm":
            co[i] = (
                fastpm_kick_factor(a0, a_c, a0, cosmo),
                fastpm_drift_factor(a0, a1, a_c, cosmo),
                fastpm_kick_factor(a_c, a1, a1, cosmo),
            )
        else:
            raise ValueError(f"integrator must be 'exact' or 'fastpm', got {integrator!r}")
    return co


def v_to_p(v_d, a, cosmo):
    """D-time velocity -> a-time momentum (H0=1 units): p = G_f(a) v_D."""
    return v_d * np.float32(_G_f(a, cosmo))


def p_to_v(p, a, cosmo):
    """a-time momentum -> D-time velocity: v_D = p / G_f(a)."""
    return p * np.float32(1.0 / _G_f(a, cosmo))


def assert_kdk_range(w0_max, growth_ratio, margin=0.9):
    """Setup-time float64 range assertion for the fixed-scale KDK w-frame:
    the encoded |w| grown by the linear momentum factor must stay inside
    int16 (D-005-class refusal; the nonlinear excess is the policy's
    c_growth headroom)."""
    predicted = float(w0_max) * float(growth_ratio)
    if predicted >= margin * 32767.0:
        raise ValueError(
            f"unfit KDK quantization: encoded max|w| = {w0_max:.0f} grown by the "
            f"linear momentum ratio {growth_ratio:.2f} predicts |w| ~ {predicted:.0f} "
            f">= {margin} * 32767; enlarge s_w0 (use codec.s_w0_policy with "
            "P_final = 1/growth_ratio) or shorten the run."
        )


# ============================================================================
# Per-step device constants
# ============================================================================


class StepConsts(NamedTuple):
    """BullFrog per-step constants as fdtype device scalars (or stacked (K,))."""

    c1: jnp.ndarray
    c2: jnp.ndarray
    kappa: jnp.ndarray


def step_consts(ladder, fdtype=jnp.float32):
    """Stack the ladder's per-step constants as (K,) device arrays of fdtype."""
    return StepConsts(
        c1=jnp.asarray(ladder.c1, dtype=fdtype),
        c2=jnp.asarray(ladder.c2, dtype=fdtype),
        kappa=jnp.asarray(ladder.kappa, dtype=fdtype),
    )


class KDKConsts(NamedTuple):
    """KDK per-step constants in lattice units (fastpm/exact integrators)."""

    kappa1: jnp.ndarray
    c: jnp.ndarray
    kappa2: jnp.ndarray


def kdk_consts(table, s_w0, s_x, fdtype=jnp.float32):
    """Lattice-unit KDK constants from a (K, 3) kdk_table: w += rint(kappa_i g),
    x += rint(c w), with w = p / s_w0 at a FIXED scale (alpha == 1, no ladder)."""
    t = np.asarray(table, dtype=np.float64)
    return KDKConsts(
        kappa1=jnp.asarray(t[:, 0] / s_w0, dtype=fdtype),
        c=jnp.asarray(t[:, 1] * s_w0 / s_x, dtype=fdtype),
        kappa2=jnp.asarray(t[:, 2] / s_w0, dtype=fdtype),
    )


# ============================================================================
# Reversible integer step kernels (verbatim M0-verdicted BullFrog + new KDK)
# ============================================================================


def step_fwd(x, w, c, force_fn, s_x, fdtype=jnp.float32):
    """One reversible integer BullFrog step. x uint16[n,3], w int16[n,3].

    force_fn maps physical positions (n,3) fdtype -> geometric force (n,3) fdtype.
    Pass the SAME force_fn object to step_rev (Sec. 5 fusion mitigation).
    """
    x1 = iadd(x, rint_i(c.c1 * w.astype(fdtype)))  # half-drift 1
    g = force_fn(dequant_x(x1, s_x, fdtype))
    w1 = iadd(w, rint_i(c.kappa * g))  # additive kick (w-frame)
    x2 = iadd(x1, rint_i(c.c2 * w1.astype(fdtype)))  # half-drift 2
    return x2, w1


def step_rev(x2, w1, c, force_fn, s_x, fdtype=jnp.float32):
    """Exact inverse of step_fwd: same bits in -> same bits out."""
    x1 = isub(x2, rint_i(c.c2 * w1.astype(fdtype)))
    g = force_fn(dequant_x(x1, s_x, fdtype))
    w = isub(w1, rint_i(c.kappa * g))
    x = isub(x1, rint_i(c.c1 * w.astype(fdtype)))
    return x, w


def step_float(xf, wf, c, force_fn, s_x, x_bits=16, fdtype=jnp.float32):
    """The STE float twin of step_fwd (architecture.md Sec. 8): LATTICE UNITS.

    Lattice-unit f32 state is exact (values < 2^17 fit the f32 mantissa), so the
    primal bit-matches the integer trajectory while the Jacobian treats rounding
    and wrap as identity.
    """
    xmod = jnp.asarray(2.0**x_bits, dtype=fdtype)
    wmod = jnp.asarray(2.0**16, dtype=fdtype)
    s_xf = jnp.asarray(s_x, dtype=fdtype)
    x1 = ste_wrap_u(xf + ste_round(c.c1 * wf), xmod)
    g = force_fn(x1 * s_xf)
    w1 = ste_wrap_s(wf + ste_round(c.kappa * g), wmod)
    x2 = ste_wrap_u(x1 + ste_round(c.c2 * w1), xmod)
    return x2, w1


def step_kdk_fwd(x, w, c, force_fn, s_x, fdtype=jnp.float32):
    """One reversible integer KDK step (fastpm/exact tables; alpha == 1).

    JANUS shear form directly -- two force solves per step make the step a
    self-contained invertible map (mbody _make_steppers precedent; no
    half-kick fusion on the reversible path)."""
    w1 = iadd(w, rint_i(c.kappa1 * force_fn(dequant_x(x, s_x, fdtype))))
    x1 = iadd(x, rint_i(c.c * w1.astype(fdtype)))
    w2 = iadd(w1, rint_i(c.kappa2 * force_fn(dequant_x(x1, s_x, fdtype))))
    return x1, w2


def step_kdk_rev(x1, w2, c, force_fn, s_x, fdtype=jnp.float32):
    """Exact inverse of step_kdk_fwd."""
    w1 = isub(w2, rint_i(c.kappa2 * force_fn(dequant_x(x1, s_x, fdtype))))
    x = isub(x1, rint_i(c.c * w1.astype(fdtype)))
    w = isub(w1, rint_i(c.kappa1 * force_fn(dequant_x(x, s_x, fdtype))))
    return x, w


def step_kdk_float(xf, wf, c, force_fn, s_x, x_bits=16, fdtype=jnp.float32):
    """STE float twin of step_kdk_fwd (lattice units; M2 linearization point)."""
    xmod = jnp.asarray(2.0**x_bits, dtype=fdtype)
    wmod = jnp.asarray(2.0**16, dtype=fdtype)
    s_xf = jnp.asarray(s_x, dtype=fdtype)
    w1 = ste_wrap_s(wf + ste_round(c.kappa1 * force_fn(xf * s_xf)), wmod)
    x1 = ste_wrap_u(xf + ste_round(c.c * w1), xmod)
    w2 = ste_wrap_s(w1 + ste_round(c.kappa2 * force_fn(x1 * s_xf)), wmod)
    return x1, w2


# ============================================================================
# Never-quantized float reference steppers (physical units; parity arms)
# ============================================================================


def float_step_bullfrog(x, v, coeff, force_fn, box_size):
    """One float BullFrog DKD step on (x, v_D) in physical units (mbody
    _bullfrog_forward port). coeff = (dD_half, alpha, beta_over_Dmid)."""
    dD_half, alpha, bcoef = coeff
    x = jnp.mod(x + dD_half * v, box_size)
    g = force_fn(x)
    v = alpha * v + bcoef * g
    x = jnp.mod(x + dD_half * v, box_size)
    return x, v


def float_step_kdk(x, p, coeff, force_fn, box_size):
    """One float KDK step on (x, p) in physical units (mbody one_step port).
    coeff = (k1, dr, k2) from kdk_table."""
    k1, dr, k2 = coeff
    p = p + k1 * force_fn(x)
    x = jnp.mod(x + dr * p, box_size)
    p = p + k2 * force_fn(x)
    return x, p


def bullfrog_float_coeffs(table):
    """(K, 3) float array (dD/2, alpha, beta/D_mid) for float_step_bullfrog."""
    t = table
    return np.stack([0.5 * t.dD, t.alphas, t.betas / t.D_mid], axis=1)


# ============================================================================
# Drivers (both R1-certified)
# ============================================================================


def run_scan(x, w, consts, step, force_fn, s_x, reverse=False, fdtype=jnp.float32):
    """lax.scan driver: consts is a NamedTuple of stacked (K,) arrays riding as
    xs. The test/small-problem driver (per-step jit is production, Sec. 9)."""

    def body(state, c):
        return step(*state, c, force_fn, s_x, fdtype=fdtype), None

    (x, w), _ = jax.lax.scan(body, (x, w), consts, reverse=reverse)
    return x, w


def run_perstep(
    x, w, consts, step, force_fn, s_x, reverse=False, fdtype=jnp.float32, monitor=False, donate=True
):
    """Per-step-jit driver: python loop over ONE compiled executable with buffer
    donation (production driver; guarantees fwd/rev share compiled programs and
    in-place carry semantics -- Sec. 9 ceiling (1)). monitor=True collects
    per-step max|w| (host sync per step) for diagnostics.overflow_report."""
    donate_args = (0, 1) if donate else ()
    step_jit = jax.jit(
        partial(step, force_fn=force_fn, s_x=s_x, fdtype=fdtype), donate_argnums=donate_args
    )
    K = len(consts[0])
    order = range(K - 1, -1, -1) if reverse else range(K)
    max_w = []
    for k in order:
        c = type(consts)(*(arr[k] for arr in consts))
        x, w = step_jit(x, w, c)
        if monitor:
            max_w.append(int(jnp.max(jnp.abs(w.astype(jnp.int32)))))
    return (x, w, max_w) if monitor else (x, w)


# ============================================================================
# Public forward API (M2 wraps custom_vjp around evolve unchanged)
# ============================================================================


def _bullfrog_setup(box, time, quant, cosmo, v_max0, fdtype):
    a_steps = a_grid(time.a_init, time.a_final, time.n_steps, time.spacing)
    unit = ladder_for_schedule(a_steps, cosmo, 1.0, 1.0, quant.alpha_floor)
    s_w0 = codec.s_w0_policy(v_max0, unit.P[-1], quant.c_growth, quant.margin)
    ladder = ladder_for_schedule(a_steps, cosmo, s_w0, box.s_x, quant.alpha_floor)
    return ladder, step_consts(ladder, fdtype)


def _kdk_setup(box, time, quant, cosmo, p_max0, fdtype):
    a_steps = a_grid(time.a_init, time.a_final, time.n_steps, time.spacing)
    table = kdk_table(a_steps, cosmo, time.integrator)
    g_ratio = _G_f(time.a_final, cosmo) / _G_f(time.a_init, cosmo)
    # reuse the ratified policy with P_final = 1/growth_ratio (same formula)
    s_w0 = codec.s_w0_policy(p_max0, 1.0 / g_ratio, quant.c_growth, quant.margin)
    assert_kdk_range(p_max0 / s_w0, g_ratio, quant.margin)
    return table, kdk_consts(table, s_w0, box.s_x, fdtype), s_w0, g_ratio


def evolve(
    box,
    time,
    quant,
    cosmo,
    x0_f,
    v0_f,
    *,
    driver="scan",
    paint="int",
    frac_bits=None,
    return_int_state=False,
    monitor=False,
    fdtype=jnp.float32,
):
    """Quantized forward evolution: floats in, floats out (integer inside).

    x0_f: physical positions (n, 3) in [0, L); v0_f: D-time velocities dx/dD
    (lpt.lpt_ics convention). Returns (x_f, v_f) at a_final, v again D-time.
    The encode is the STE boundary (architecture Sec. 8); s_w0 follows the
    ratified policy (D-011). driver: "scan" | "perstep". paint: "int"
    (deterministic primal, default) | "f32".
    """
    if frac_bits is None:
        frac_bits = quant.frac_bits
    force_fn = make_force_fn(box, fdtype=fdtype, paint=paint, frac_bits=frac_bits)
    integ = time.integrator
    if integ == "bullfrog":
        v_max0 = float(jnp.max(jnp.abs(v0_f)))
        ladder, consts = _bullfrog_setup(box, time, quant, cosmo, v_max0, fdtype)
        x_i, w_i = encode_x(x0_f, box.s_x), encode_w(v0_f, ladder.s_w0)
        step = step_fwd
        s_out = ladder.s_w0 * ladder.P[-1]  # decode scale for v_D at a_final
    else:  # fastpm / exact: fixed-scale w = p / s_w0
        p0 = v_to_p(v0_f, time.a_init, cosmo)
        p_max0 = float(jnp.max(jnp.abs(p0)))
        _, consts, s_w0, _ = _kdk_setup(box, time, quant, cosmo, p_max0, fdtype)
        x_i, w_i = encode_x(x0_f, box.s_x), encode_w(p0, s_w0)
        step = step_kdk_fwd
        s_out = s_w0
    if driver == "scan":
        x, w = run_scan(x_i, w_i, consts, step, force_fn, box.s_x, fdtype=fdtype)
        max_w = None
    elif driver == "perstep":
        x, w, max_w = run_perstep(
            x_i, w_i, consts, step, force_fn, box.s_x, fdtype=fdtype, monitor=True
        )
    else:
        raise ValueError(f"driver must be 'scan' or 'perstep', got {driver!r}")
    x_f = dequant_x(x, box.s_x, fdtype)
    v_f = dequant_w(w, s_out, fdtype)
    if integ != "bullfrog":
        v_f = p_to_v(v_f, time.a_final, cosmo)
    out = (x_f, v_f)
    if return_int_state:
        out = out + ((x, w),)
    if monitor:
        out = out + (max_w,)
    return out


def evolve_float(box, time, cosmo, x0_f, v0_f, *, paint="f32", fdtype=jnp.float32):
    """Never-quantized float reference evolution (parity Tier-A arm; f64 when
    the CALLER enables x64). Same conventions as evolve; physical units."""
    force_fn = make_force_fn(box, fdtype=fdtype, paint=paint)
    a_steps = a_grid(time.a_init, time.a_final, time.n_steps, time.spacing)
    if time.integrator == "bullfrog":
        coeffs = bullfrog_float_coeffs(bullfrog_table(a_steps, cosmo))
        x, v = x0_f.astype(fdtype), v0_f.astype(fdtype)
        for c in coeffs:
            x, v = float_step_bullfrog(
                x, v, tuple(np.asarray(c, np.float64)), force_fn, box.box_size
            )
        return x, v
    table = kdk_table(a_steps, cosmo, time.integrator)
    x = x0_f.astype(fdtype)
    p = v_to_p(v0_f.astype(fdtype), time.a_init, cosmo)
    for c in table:
        x, p = float_step_kdk(x, p, tuple(c), force_fn, box.box_size)
    return x, p_to_v(p, time.a_final, cosmo)


def replay_roundtrip(
    box,
    time,
    quant,
    cosmo,
    x0_f,
    v0_f,
    *,
    driver="scan",
    paint="int",
    s_w0_div=1.0,
    fdtype=jnp.float32,
):
    """Tier-0 primitive: encode once, K forward + K reverse, EXACT integer
    equality with the initial state. s_w0_div shrinks s_w0 to force mid-run w
    wraps (the R1 wrap-adversarial arm; D-007). Returns (ok, n_diff)."""
    force_fn = make_force_fn(box, fdtype=fdtype, paint=paint, frac_bits=quant.frac_bits)
    if time.integrator == "bullfrog":
        v_max0 = float(jnp.max(jnp.abs(v0_f)))
        a_steps = a_grid(time.a_init, time.a_final, time.n_steps, time.spacing)
        unit = ladder_for_schedule(a_steps, cosmo, 1.0, 1.0, quant.alpha_floor)
        s_w0 = codec.s_w0_policy(v_max0, unit.P[-1], quant.c_growth, quant.margin) / s_w0_div
        ladder = ladder_for_schedule(a_steps, cosmo, s_w0, box.s_x, quant.alpha_floor)
        consts = step_consts(ladder, fdtype)
        fwd, rev = step_fwd, step_rev
        x_i, w_i = encode_x(x0_f, box.s_x), encode_w(v0_f, s_w0)
    else:
        p0 = v_to_p(v0_f, time.a_init, cosmo)
        p_max0 = float(jnp.max(jnp.abs(p0)))
        a_steps = a_grid(time.a_init, time.a_final, time.n_steps, time.spacing)
        table = kdk_table(a_steps, cosmo, time.integrator)
        g_ratio = _G_f(time.a_final, cosmo) / _G_f(time.a_init, cosmo)
        s_w0 = codec.s_w0_policy(p_max0, 1.0 / g_ratio, quant.c_growth, quant.margin) / s_w0_div
        consts = kdk_consts(table, s_w0, box.s_x, fdtype)
        fwd, rev = step_kdk_fwd, step_kdk_rev
        x_i, w_i = encode_x(x0_f, box.s_x), encode_w(p0, s_w0)
    # fresh copies into the drivers: the perstep driver DONATES its inputs and
    # x_i/w_i must survive for the equality check (the R1 GPU lesson, ce130fa)
    if driver == "scan":
        x, w = run_scan(x_i.copy(), w_i.copy(), consts, fwd, force_fn, box.s_x, fdtype=fdtype)
        x, w = run_scan(x, w, consts, rev, force_fn, box.s_x, reverse=True, fdtype=fdtype)
    else:
        x, w = run_perstep(x_i.copy(), w_i.copy(), consts, fwd, force_fn, box.s_x, fdtype=fdtype)
        x, w = run_perstep(x, w, consts, rev, force_fn, box.s_x, reverse=True, fdtype=fdtype)
    n_diff = int(jnp.count_nonzero(x != x_i)) + int(jnp.count_nonzero(w != w_i))
    return n_diff == 0, n_diff


def simulate(
    box,
    time,
    quant,
    cosmo,
    seed=0,
    f_NL=0.0,
    lpt_order=2,
    amplitude=1.0,
    quantized=True,
    **evolve_kw,
):
    """IC -> LPT -> evolve, the mbody leapfrog() analog. Returns (x_f, v_f[, ...]).

    quantized=False routes through evolve_float (never-quantized reference).
    """
    key = jax.random.PRNGKey(seed)
    if f_NL == 0.0:
        from .ic import gaussian_delta

        delta0 = amplitude * gaussian_delta(key, box.n_mesh, box.box_size, cosmo)
    else:
        delta0 = amplitude * linear_density(key, box.n_mesh, box.box_size, cosmo, f_NL=f_NL)
    x0, v0 = lpt_ics(delta0, box.box_size, time.a_init, cosmo, order=lpt_order)
    if quantized:
        return evolve(box, time, quant, cosmo, x0, v0, **evolve_kw)
    return evolve_float(box, time, cosmo, x0, v0, **evolve_kw)
