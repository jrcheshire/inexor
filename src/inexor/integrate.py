"""Integrator coefficient tables and never-quantized float reference steppers.

The host float64 coefficient layer -- BullFrog weights, exact/FastPM KDK
factors, the D-time <-> a-time momentum conversion -- plus the float steppers
that consume them. This is the half of the v1 integrator that survived the halt:
it is pure cosmology arithmetic and knows nothing about state representation.

**What used to be here and is gone (2026-08-08, v1 retirement).** The reversible
integer step kernels (`step_fwd`/`step_rev`/`step_kdk_*`), their STE float twins,
the `run_scan`/`run_perstep` drivers, and the `evolve`/`evolve_float`/
`replay_roundtrip`/`simulate` public API were the v1 thesis: a bit-exactly
reversible integer trajectory whose adjoint is an exact replay. That premise
measured false on 2026-07-14 (`docs/retrospective.md`); v2 is a memory-floor
forward mock engine and does not use them. The v2 engine's own driver lands at
M-v2-3 (D-v2-18), so there is deliberately no end-to-end driver in the package
between the codec/force milestones and that one.

Three integrator families, all still live as coefficient sources:
- "bullfrog": DKD with an affine kick (Rampf, List & Hahn 2024). The flagship.
  Its weights take the LCDM second-order growth by default; the EdS
  -(3/7) D^2 form converges to a different, EdS-coupled solution (paper Sec. 4.4)
  and is kept only as `growth2="eds"`.
- "fastpm":   growth-corrected KDK (Feng et al. 2016).
- "exact":    KDK with literal background integrals (mbody "exact"); the
  fallback. ~2% growth deficit at low step count -- expected, tested at the
  converged limit only.

Units: the public surface speaks D-time velocity v = dx/dD throughout. The KDK
families work in a-time momentum p (H0=1 units) internally; convert at the
boundary via G_f = a^3 E D' (`v_to_p` / `p_to_v`).

Every float stepper takes the force callable EXPLICITLY -- there is no internal
`make_force_fn` call, which is what lets the v2 probes drive these steppers with
a two-level force.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from .cosmology import (
    GROWTH2_MODELS,
    E_of_a,
    growth2_and_slope,
    growth_factor_a,
    growth_rate_a,
)

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


def _bullfrog_weights(D0, D1, e=None):
    """BullFrog (alpha, beta, dD, D_mid) for a step D0 -> D1 (Rampf, List & Hahn
    2024, Eqs. 2.3-2.4).

    `e = (E0, E0', E1')` is the second-order growth at D0 and its slope dE/dD at
    D0 and D1, from `cosmology.growth2_and_slope`. The weights are invariant under
    D -> cD, E -> c^2 E, so any consistent normalization works. `e=None` is the
    EdS special case E = -(3/7) D^2, E' = -(6/7) D, the closed form Eq. 2.3 reduces
    to at Omega_m = 1; in LCDM it is not consistent (paper Sec. 4.4).
    """
    dD = D1 - D0
    D_mid = D0 + 0.5 * dD
    if e is None:
        E0, E0p, E1p = -(3.0 / 7.0) * D0 * D0, -(6.0 / 7.0) * D0, -(6.0 / 7.0) * D1
    else:
        E0, E0p, E1p = e
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


def bullfrog_table(a_steps, cosmo, D_of_a=None, growth2="lcdm"):
    """BullFrog weights for a schedule.

    growth2="lcdm" (default) takes the second-order growth from the LCDM ODE;
    "eds" uses -(3/7) D^2, only to reproduce runs made with it. D_of_a overrides
    the linear growth (the EdS pin test) and so requires growth2="eds": an
    overridden D has no matching LCDM second-order growth.
    """
    if growth2 not in GROWTH2_MODELS:
        raise ValueError(f"growth2 must be one of {GROWTH2_MODELS}, got {growth2!r}")
    if D_of_a is not None and growth2 != "eds":
        raise ValueError("D_of_a overrides the linear growth; pass growth2='eds' with it")
    a_steps = np.asarray(a_steps, dtype=np.float64)
    if D_of_a is None:
        D_steps = np.array([growth_factor_a(a, cosmo) for a in a_steps])
    else:
        D_steps = np.array([D_of_a(a) for a in a_steps])
    K = len(a_steps) - 1
    if growth2 == "lcdm":
        E, Ep = np.array([growth2_and_slope(a, cosmo) for a in a_steps]).T
    alphas, betas, dD, D_mid = (np.empty(K) for _ in range(4))
    for k in range(K):
        e = None if growth2 == "eds" else (E[k], Ep[k], Ep[k + 1])
        alphas[k], betas[k], dD[k], D_mid[k] = _bullfrog_weights(D_steps[k], D_steps[k + 1], e)
    return BullFrogTable(a_steps, D_steps, alphas, betas, dD, D_mid)


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
