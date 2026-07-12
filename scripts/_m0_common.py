"""Shared support for the inexor M0 kill-or-confirm probes (R1-R5).

Probe-support code, not package code (see docs/roadmap.md: M0 exit artifacts are
standalone scripts). The codec and step primitives ARE written to the exact spec
of docs/architecture.md Secs. 3-8, because R1 certifies the same bits that R4
gradient-tests, and both migrate into src/inexor/{codec,integrate}.py at M1.

Conventions (house rules; see CLAUDE.md):
- This module NEVER touches jax.config. Probe scripts are the callers and may
  enable x64 (R2 does) BEFORE creating any arrays. To keep that safe, this
  module creates no arrays at import time -- functions only.
- Host-side coefficient math is a numpy/scipy float64 island (mbody pattern),
  off the AD graph.
- Wrap, never clamp (decisions.md D-007): no saturating op touches integer
  state anywhere in this file.
- Port sources: mbody/{integrate,cosmology,lpt,painting,forces}.py (MLX ->
  JAX; coefficient layer ported verbatim).
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad, simpson

# ============================================================================
# Cosmology: background + growth + EH98 linear P(k)  (host float64 island)
# Ported from mbody/cosmology.py (:145 transfer_eh98, :365 growth_factor).
# ============================================================================

# Constant the EH98 reference C code uses in place of Euler's e; kept verbatim
# so the port reproduces the original formula bit-for-bit (mbody convention).
_E_EH98 = 2.718282


@dataclass(frozen=True)
class Cosmology:
    """Flat-LCDM parameters (Planck-2018-flavoured defaults; mbody values)."""

    Omega_m: float = 0.31
    Omega_b: float = 0.049
    h: float = 0.677
    n_s: float = 0.965
    sigma8: float = 0.81
    T_cmb_K: float = 2.7255

    @property
    def Omega_Lambda(self):
        return 1.0 - self.Omega_m


PLANCK = Cosmology()


def E_of_a(a, cosmo):
    """Dimensionless Hubble rate E(a) = H/H0, flat LCDM, radiation neglected."""
    return np.sqrt(cosmo.Omega_m / a**3 + cosmo.Omega_Lambda)


def _growth_integrand(a, Om, OL):
    e = np.sqrt(Om / a**3 + OL)
    return 1.0 / (a * e) ** 3


def _growth_unnorm(a, cosmo):
    # D(a) propto (5 Om / 2) E(a) * integral_0^a da' / (a' E(a'))^3 (exact flat LCDM).
    integral, _ = quad(_growth_integrand, 0.0, a, args=(cosmo.Omega_m, cosmo.Omega_Lambda))
    return 2.5 * cosmo.Omega_m * E_of_a(a, cosmo) * integral


@lru_cache(maxsize=None)
def _D0(cosmo):
    return _growth_unnorm(1.0, cosmo)


def growth_factor_a(a, cosmo):
    """Linear growth factor D(a), normalized to D(a=1) = 1. Scalar in, scalar out."""
    return _growth_unnorm(float(a), cosmo) / _D0(cosmo)


def growth_rate_a(a, cosmo):
    """Linear growth rate f = dlnD/dlna at scale factor a (exact flat LCDM)."""
    Om, OL = cosmo.Omega_m, cosmo.Omega_Lambda
    a = float(a)
    G, _ = quad(_growth_integrand, 0.0, a, args=(Om, OL))
    e = E_of_a(a, cosmo)
    return -1.5 * Om / (a**3 * e**2) + 1.0 / (a**2 * e**3 * G)


class _EH98:
    """Scalar EH98 parameters for one cosmology (mbody port of TFset_parameters)."""

    def __init__(self, cosmo):
        Om, Ob, h = cosmo.Omega_m, cosmo.Omega_b, cosmo.h
        f_baryon = Ob / Om
        omhh = Om * h * h
        obhh = omhh * f_baryon
        theta = cosmo.T_cmb_K / 2.7

        z_eq = 2.50e4 * omhh / theta**4  # really 1 + z_eq
        k_eq = 0.0746 * omhh / theta**2  # Mpc^-1

        b1 = 0.313 * omhh**-0.419 * (1 + 0.607 * omhh**0.674)
        b2 = 0.238 * omhh**0.223
        z_drag = 1291 * omhh**0.251 / (1 + 0.659 * omhh**0.828) * (1 + b1 * obhh**b2)

        R_drag = 31.5 * obhh / theta**4 * (1000.0 / (1 + z_drag))
        R_eq = 31.5 * obhh / theta**4 * (1000.0 / z_eq)

        sound_horizon = (
            2.0
            / (3.0 * k_eq)
            * np.sqrt(6.0 / R_eq)
            * np.log((np.sqrt(1 + R_drag) + np.sqrt(R_drag + R_eq)) / (1 + np.sqrt(R_eq)))
        )

        k_silk = 1.6 * obhh**0.52 * omhh**0.73 * (1 + (10.4 * omhh) ** -0.95)

        a1 = (46.9 * omhh) ** 0.670 * (1 + (32.1 * omhh) ** -0.532)
        a2 = (12.0 * omhh) ** 0.424 * (1 + (45.0 * omhh) ** -0.582)
        alpha_c = a1 ** (-f_baryon) * a2 ** (-(f_baryon**3))

        bb1 = 0.944 / (1 + (458.0 * omhh) ** -0.708)
        bb2 = (0.395 * omhh) ** -0.0266
        beta_c = 1.0 / (1 + bb1 * ((1 - f_baryon) ** bb2 - 1))

        y = z_eq / (1 + z_drag)
        sqy = np.sqrt(1 + y)
        alpha_b_G = y * (-6.0 * sqy + (2.0 + 3.0 * y) * np.log((sqy + 1) / (sqy - 1)))
        alpha_b = 2.07 * k_eq * sound_horizon * (1 + R_drag) ** -0.75 * alpha_b_G

        beta_node = 8.41 * omhh**0.435
        beta_b = 0.5 + f_baryon + (3.0 - 2.0 * f_baryon) * np.sqrt((17.2 * omhh) ** 2 + 1)

        self.h = h
        self.omhh = omhh
        self.obhh = obhh
        self.k_eq = k_eq
        self.sound_horizon = sound_horizon
        self.k_silk = k_silk
        self.alpha_c = alpha_c
        self.beta_c = beta_c
        self.alpha_b = alpha_b
        self.beta_b = beta_b
        self.beta_node = beta_node


def transfer_eh98(k_hmpc, cosmo):
    """Full EH98 transfer function T(k), k in h/Mpc; T -> 1 as k -> 0.

    Verbatim mbody port (Eq. 16 of Eisenstein & Hu 1998; TFfit_onek).
    """
    k_arr = np.atleast_1d(np.asarray(k_hmpc, dtype=np.float64))
    p = _EH98(cosmo)
    k = k_arr * p.h  # Mpc^-1
    out = np.ones_like(k)
    m = k > 0
    kk = k[m]

    q = kk / 13.41 / p.k_eq
    xx = kk * p.sound_horizon

    ln_beta = np.log(_E_EH98 + 1.8 * p.beta_c * q)
    ln_nobeta = np.log(_E_EH98 + 1.8 * q)
    C_alpha = 14.2 / p.alpha_c + 386.0 / (1 + 69.9 * q**1.08)
    C_noalpha = 14.2 + 386.0 / (1 + 69.9 * q**1.08)

    f = 1.0 / (1.0 + (xx / 5.4) ** 4)
    T_c = f * ln_beta / (ln_beta + C_noalpha * q**2) + (1 - f) * ln_beta / (
        ln_beta + C_alpha * q**2
    )

    s_tilde = p.sound_horizon * (1 + (p.beta_node / xx) ** 3) ** (-1.0 / 3.0)
    xx_tilde = kk * s_tilde
    T_b_T0 = ln_nobeta / (ln_nobeta + C_noalpha * q**2)
    T_b = (
        np.sin(xx_tilde)
        / xx_tilde
        * (
            T_b_T0 / (1 + (xx / 5.2) ** 2)
            + p.alpha_b / (1 + (p.beta_b / xx) ** 3) * np.exp(-((kk / p.k_silk) ** 1.4))
        )
    )

    f_baryon = p.obhh / p.omhh
    out[m] = f_baryon * T_b + (1 - f_baryon) * T_c
    return out


def _tophat_window(x):
    return 3.0 * (np.sin(x) - x * np.cos(x)) / np.where(x == 0.0, 1.0, x) ** 3


@lru_cache(maxsize=None)
def _eh98_amplitude(cosmo):
    # Amplitude A such that sigma(8 Mpc/h, z=0) = sigma8 for P = A k^n_s T^2.
    lnk = np.linspace(np.log(1e-4), np.log(1e2), 4000)
    k = np.exp(lnk)
    Pk = k**cosmo.n_s * transfer_eh98(k, cosmo) ** 2
    W = _tophat_window(k * 8.0)
    sig2 = simpson(k**3 * Pk * W**2 / (2.0 * np.pi**2), x=lnk)
    return cosmo.sigma8**2 / sig2


def linear_power(k_hmpc, cosmo, z=0.0):
    """Linear matter P(k, z) in (Mpc/h)^3, EH98 backend, sigma8-normalized.

    Probes use eh98 (analytic, no external dep) per the M0 plan; CAMB parity
    is an M1 concern.
    """
    k_arr = np.atleast_1d(np.asarray(k_hmpc, dtype=np.float64))
    P0 = _eh98_amplitude(cosmo) * k_arr**cosmo.n_s * transfer_eh98(k_arr, cosmo) ** 2
    if z != 0.0:
        a = 1.0 / (1.0 + z)
        P0 = P0 * growth_factor_a(a, cosmo) ** 2
    return P0


# ============================================================================
# BullFrog / FastPM coefficients + the w-frame scale ladder (host float64)
# Ported from mbody/integrate.py (:68-:234); ladder per architecture.md Sec. 4.
# ============================================================================


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


def a_grid(a_init, a_final, n_steps, spacing="log"):
    """Scale-factor step edges (n_steps+1 points), 'log' or 'linear' spacing."""
    if spacing == "log":
        return np.geomspace(a_init, a_final, n_steps + 1)
    if spacing == "linear":
        return np.linspace(a_init, a_final, n_steps + 1)
    raise ValueError(f"spacing must be 'log' or 'linear', got {spacing!r}")


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


def ladder_constants(a_steps, cosmo, s_w0, s_x, alpha_floor=0.05, D_of_a=None):
    """Build the w-frame ladder for a BullFrog schedule. Refuses unfit schedules.

    alpha_floor: refuse any |alpha_k| below this -- near-zero alpha makes
    kappa ~ 1/P_{k+1} diverge (the ladder collapses). Measured this session
    (EdS proxy): linear-in-a schedules from a_i=0.04 cross alpha=0 near K=11;
    log schedules from a_i=0.1 keep alpha_1 >= 0.48 for K=5-15 at ~5-bit cost.
    R2 confirms with exact LCDM growth.

    D_of_a: optional growth override (e.g. lambda a: a for the EdS pin test);
    default is the exact flat-LCDM growth_factor_a.
    """
    a_steps = np.asarray(a_steps, dtype=np.float64)
    if D_of_a is None:
        D_steps = np.array([growth_factor_a(a, cosmo) for a in a_steps])
    else:
        D_steps = np.array([D_of_a(a) for a in a_steps])
    K = len(a_steps) - 1
    alphas = np.empty(K)
    betas = np.empty(K)
    dD = np.empty(K)
    D_mid = np.empty(K)
    for k in range(K):
        alphas[k], betas[k], dD[k], D_mid[k] = _bullfrog_weights(D_steps[k], D_steps[k + 1])
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


def s_w0_policy(v_max0, P_final, c_growth=4.0, margin=0.9):
    """Initial w scale so late-time |w| stays inside int16 (plan Sec. R2 formula).

    s_w0 = c_growth * max|v_0| / (margin * 32767 * |P_K|). c_growth is the
    velocity growth factor over the run (measured by R2; ~2-4 expected --
    the default 4.0 is a pre-R2 placeholder, revisit after R2 reports).
    """
    return c_growth * v_max0 / (margin * 32767.0 * abs(P_final))


# ============================================================================
# Codec primitives (architecture.md Secs. 3-4; the future codec.py core)
# ============================================================================

INT16_MAX = 32767
U16_MOD = 2**16


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

    bits=16 is asserted bit-identical to the uint16 iadd path in the self-checks.
    """
    return (x32 + inc32) & (2**bits - 1)


def imask_sub(x32, inc32, bits):
    return (x32 - inc32) & (2**bits - 1)


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
# The reversible integer BullFrog step (architecture.md Sec. 4 pseudocode)
# ============================================================================


class StepConsts(NamedTuple):
    """Per-step constants as fdtype device scalars (or stacked (K,) for scan xs)."""

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
    """The STE float twin (architecture.md Sec. 8): state in LATTICE UNITS.

    Lattice-unit f32 state is exact (values < 2^17 fit the f32 mantissa), so the
    primal bit-matches the integer trajectory while the Jacobian treats rounding
    and wrap as identity. x_bits parametrizes R4's position-width sweep.
    """
    xmod = jnp.asarray(2.0**x_bits, dtype=fdtype)
    wmod = jnp.asarray(2.0**16, dtype=fdtype)
    s_xf = jnp.asarray(s_x, dtype=fdtype)
    x1 = ste_wrap_u(xf + ste_round(c.c1 * wf), xmod)
    g = force_fn(x1 * s_xf)
    w1 = ste_wrap_s(wf + ste_round(c.kappa * g), wmod)
    x2 = ste_wrap_u(x1 + ste_round(c.c2 * w1), xmod)
    return x2, w1


# ============================================================================
# CIC paint / read + geometric Poisson force (jnp ports of mbody patterns)
# ============================================================================

_CORNERS = [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]


def _cic_pieces(positions, n_mesh, box_size):
    """Base cell (stop_gradient, mbody painting.py:33 pattern) + fractional offset."""
    d = box_size / n_mesh
    xp = positions / d
    base_f = jnp.floor(xp)
    frac = xp - base_f  # gradients flow through frac only; the cell map is p.w.-constant
    base = jax.lax.stop_gradient(base_f).astype(jnp.int32)
    return base, frac


def _corner_flat_weight(base, frac, corner, n_mesh):
    dx, dy, dz = corner
    wlo = 1.0 - frac
    wx = frac[:, 0] if dx else wlo[:, 0]
    wy = frac[:, 1] if dy else wlo[:, 1]
    wz = frac[:, 2] if dz else wlo[:, 2]
    ix = (base[:, 0] + dx) % n_mesh
    iy = (base[:, 1] + dy) % n_mesh
    iz = (base[:, 2] + dz) % n_mesh
    flat = (ix * n_mesh + iy) * n_mesh + iz
    return flat, wx * wy * wz


def paint_f32(positions, n_mesh, box_size, fdtype=jnp.float32):
    """Differentiable CIC paint (counts). The VJP-path twin; NOT deterministic
    under f32 atomics on GPU -- primal force paths use paint_int."""
    mesh = jnp.zeros((n_mesh**3,), dtype=fdtype)
    base, frac = _cic_pieces(positions, n_mesh, box_size)
    for corner in _CORNERS:
        flat, w = _corner_flat_weight(base, frac, corner, n_mesh)
        mesh = mesh.at[flat].add(w.astype(fdtype), mode="promise_in_bounds")
    return mesh.reshape(n_mesh, n_mesh, n_mesh)


def paint_int(positions, n_mesh, box_size, frac_bits=12):
    """Deterministic integer-accumulation CIC paint (architecture.md Sec. 5).

    Corner weights quantized to frac_bits fixed point, scatter-added into an
    int32 mesh: integer addition is associative, so the result is bit-identical
    regardless of atomic order. Primal-only (no VJP rule needed -- D-006).
    Returns the RAW int32 mesh; decode counts as mesh * 2.0**-frac_bits.
    """
    scale = jnp.float32(2.0**frac_bits)
    mesh = jnp.zeros((n_mesh**3,), dtype=jnp.int32)
    base, frac = _cic_pieces(positions, n_mesh, box_size)
    for corner in _CORNERS:
        flat, w = _corner_flat_weight(base, frac, corner, n_mesh)
        mesh = mesh.at[flat].add(rint_i(w.astype(jnp.float32) * scale), mode="promise_in_bounds")
    return mesh.reshape(n_mesh, n_mesh, n_mesh)


def cic_read_vector(gx, gy, gz, positions, n_mesh, box_size):
    """Read 3 mesh fields with ONE shared CIC stencil (mbody painting.py:105;
    ~40% reverse-mode memory saving measured there)."""
    base, frac = _cic_pieces(positions, n_mesh, box_size)
    fx, fy, fz = gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)
    n = positions.shape[0]
    ax = jnp.zeros((n,), dtype=gx.dtype)
    ay = jnp.zeros((n,), dtype=gx.dtype)
    az = jnp.zeros((n,), dtype=gx.dtype)
    for corner in _CORNERS:
        flat, w = _corner_flat_weight(base, frac, corner, n_mesh)
        ax = ax + w * fx[flat]
        ay = ay + w * fy[flat]
        az = az + w * fz[flat]
    return jnp.stack([ax, ay, az], axis=1)


def k_components(n_mesh, box_size, fdtype=np.float32):
    """i k_j (complex, low-rank) and 1/k^2 (full) on the rfftn half-grid, k=0 safe.

    Host-numpy build (mbody lpt.py:45 port); returns jnp arrays of the dtype
    matching fdtype (complex64 for f32, complex128 for f64).
    """
    N, L = n_mesh, box_size
    d = L / N
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=d)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=d)
    M = N // 2 + 1
    KX = kx.reshape(N, 1, 1)
    KY = kx.reshape(1, N, 1)
    KZ = kz.reshape(1, 1, M)
    k2 = KX**2 + KY**2 + KZ**2
    k2[0, 0, 0] = 1.0  # avoid 0/0; the ik_j are zero at k=0 anyway
    cdtype = np.complex128 if fdtype == np.float64 else np.complex64
    return (
        jnp.asarray((1j * KX).astype(cdtype)),
        jnp.asarray((1j * KY).astype(cdtype)),
        jnp.asarray((1j * KZ).astype(cdtype)),
        jnp.asarray((1.0 / k2).astype(fdtype)),
    )


def make_force_fn(n_mesh, box_size, n_particles, fdtype=jnp.float32, paint="f32", frac_bits=12):
    """Geometric PM force: div g = -delta (all cosmology prefactors live in the
    integrator coefficients -- mbody forces.py convention).

    paint='f32' -> differentiable float paint (CPU probes / VJP path);
    paint='int' -> deterministic integer paint (primal force path, R1/R3).
    Returns force(positions (n,3) physical) -> (n,3) fdtype. Create ONE force fn
    and share the object between step_fwd and step_rev.
    """
    npdt = np.float64 if fdtype == jnp.float64 else np.float32
    ikx, iky, ikz, inv_k2 = k_components(n_mesh, box_size, npdt)
    mean = n_particles / n_mesh**3
    N = n_mesh

    def force(pos):
        if paint == "int":
            mesh = paint_int(pos, N, box_size, frac_bits).astype(fdtype) * jnp.asarray(
                2.0**-frac_bits, dtype=fdtype
            )
        else:
            mesh = paint_f32(pos, N, box_size, fdtype)
        delta = mesh / mean - 1.0
        dk = jnp.fft.rfftn(delta)
        # Sequential per-component solves; g_j = ik_j delta_k / k^2 (div g = -delta).
        gx = jnp.fft.irfftn(dk * ikx * inv_k2, s=(N, N, N))
        gy = jnp.fft.irfftn(dk * iky * inv_k2, s=(N, N, N))
        gz = jnp.fft.irfftn(dk * ikz * inv_k2, s=(N, N, N))
        return cic_read_vector(gx, gy, gz, pos, N, box_size).astype(fdtype)

    return force


# ============================================================================
# Initial conditions: Gaussian linear density -> Zel'dovich (D-time velocities)
# ============================================================================


def linear_delta0(key, n_mesh, box_size, cosmo, fdtype=jnp.float32, amplitude=1.0):
    """Seeded z=0 linear density on the mesh: white noise coloured by eh98 P(k).

    Convention: delta_k = rfftn(white) * sqrt(P(|k|) * N^3 / L^3), so the
    measured P(k) of the returned field matches linear_power (self-check #3).
    """
    N, L = n_mesh, box_size
    white = jax.random.normal(key, (N, N, N), dtype=fdtype)
    dk = jnp.fft.rfftn(white)
    # |k| grid + sqrt(P) colour, host f64 then cast (precision island).
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)
    kk = np.sqrt(
        kx.reshape(N, 1, 1) ** 2 + kx.reshape(1, N, 1) ** 2 + kz.reshape(1, 1, -1) ** 2
    )
    colour = np.sqrt(linear_power(kk.ravel(), cosmo).reshape(kk.shape) * N**3 / L**3)
    colour[0, 0, 0] = 0.0  # zero the mean mode
    npdt = np.float64 if fdtype == jnp.float64 else np.float32
    dk = dk * jnp.asarray(colour.astype(npdt))
    return amplitude * jnp.fft.irfftn(dk, s=(N, N, N))


def za_psi(delta0, n_mesh, box_size, fdtype=jnp.float32):
    """Zel'dovich displacement Psi1 = (ik/k^2) delta0, shape (N^3, 3), z=0 norm."""
    N = n_mesh
    npdt = np.float64 if fdtype == jnp.float64 else np.float32
    ikx, iky, ikz, inv_k2 = k_components(N, box_size, npdt)
    dk = jnp.fft.rfftn(delta0)
    px = jnp.fft.irfftn(dk * ikx * inv_k2, s=(N, N, N))
    py = jnp.fft.irfftn(dk * iky * inv_k2, s=(N, N, N))
    pz = jnp.fft.irfftn(dk * ikz * inv_k2, s=(N, N, N))
    return jnp.stack([px.reshape(-1), py.reshape(-1), pz.reshape(-1)], axis=1).astype(fdtype)


def lagrangian_grid(n_mesh, box_size, fdtype=jnp.float32):
    """Unperturbed particle positions (one per cell), shape (N^3, 3)."""
    N = n_mesh
    d = box_size / N
    coords = jnp.arange(N, dtype=fdtype) * d
    qx, qy, qz = jnp.meshgrid(coords, coords, coords, indexing="ij")
    return jnp.stack([qx.reshape(-1), qy.reshape(-1), qz.reshape(-1)], axis=1)


def za_ics(delta0, n_mesh, box_size, a_init, cosmo, fdtype=jnp.float32, D_of_a=None):
    """ZA state at a_init: x = wrap(q + D_i Psi1), v = dx/dD = Psi1 (D-time).

    The D-time velocity of a ZA mode is Psi1, CONSTANT in D -- the w-frame's
    design premise (architecture.md Sec. 3). Returns (x_phys, v) each (N^3, 3).
    """
    L = box_size
    D_i = D_of_a(a_init) if D_of_a is not None else growth_factor_a(a_init, cosmo)
    psi = za_psi(delta0, n_mesh, L, fdtype)
    q = lagrangian_grid(n_mesh, L, fdtype)
    x = jnp.mod(q + jnp.asarray(D_i, dtype=fdtype) * psi, L)
    return x, psi


# ============================================================================
# Diagnostics: P(k) estimator + minimum-image displacement
# ============================================================================


def pk_estimator(delta, box_size, n_bins=32, k_min=None, k_max=None):
    """Binned auto P(k) of a real mesh field: |delta_k|^2 L^3 / N^6, hermitian
    double-count weights (2 except the kz=0 and kz=Nyquist planes).

    Returns (k_centers, P, n_modes) as numpy arrays (host-side; diagnostics).
    """
    delta = np.asarray(delta, dtype=np.float64)
    N = delta.shape[0]
    L = box_size
    dk = np.fft.rfftn(delta)
    p3 = np.abs(dk) ** 2 * (L**3 / N**6)
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)
    kk = np.sqrt(kx.reshape(N, 1, 1) ** 2 + kx.reshape(1, N, 1) ** 2 + kz.reshape(1, 1, -1) ** 2)
    wts = np.full(p3.shape, 2.0)
    wts[:, :, 0] = 1.0
    if N % 2 == 0:
        wts[:, :, -1] = 1.0
    k_f = 2.0 * np.pi / L
    lo = k_f if k_min is None else k_min
    hi = kk.max() if k_max is None else k_max
    edges = np.linspace(lo, hi, n_bins + 1)
    which = np.digitize(kk.ravel(), edges) - 1
    valid = (which >= 0) & (which < n_bins) & (kk.ravel() > 0)
    w = wts.ravel()[valid]
    b = which[valid]
    psum = np.bincount(b, weights=w * p3.ravel()[valid], minlength=n_bins)
    ksum = np.bincount(b, weights=w * kk.ravel()[valid], minlength=n_bins)
    nmod = np.bincount(b, weights=w, minlength=n_bins)
    good = nmod > 0
    return (ksum[good] / nmod[good], psum[good] / nmod[good], nmod[good])


def min_image_rms(x_a, x_b, box_size):
    """RMS per-component displacement between two position sets, minimum-image."""
    d = np.asarray(x_a, dtype=np.float64) - np.asarray(x_b, dtype=np.float64)
    d = d - box_size * np.round(d / box_size)
    return float(np.sqrt(np.mean(d**2)))


# ============================================================================
# Self-checks: `pixi run python scripts/_m0_common.py` must pass before any
# probe result is trusted (M0 plan, Build order step 1).
# ============================================================================


def _check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(ok)


def _self_checks():
    print("_m0_common self-checks (CPU)")
    results = []

    # 1. BullFrog EdS closed-form pin (mbody test_integrate.py:472 oracle).
    err = 0.0
    for n in [1.0, 2.0, 3.5, 7.0, 20.0]:
        D0, dD = n * 0.05, 0.05
        alpha, beta, _, _ = _bullfrog_weights(D0, D0 + dD)
        m = D0 / dD
        alpha_ref = (4 * m * (4 * m + 1) - 5) / (4 * m * (4 * m + 7) + 7)
        beta_ref = (24 * m + 12) / (4 * m * (4 * m + 7) + 7)
        err = max(err, abs(alpha - alpha_ref), abs(beta - beta_ref))
    results.append(_check("BullFrog EdS closed form < 1e-12", err < 1e-12, f"max err {err:.2e}"))

    # 2. Ladder assertion fires on the near-zero-alpha schedule (this session's
    # finding: linear a_i=0.04 crosses alpha=0 near K=11) and accepts log a_i=0.1.
    try:
        ladder_constants(a_grid(0.04, 1.0, 11, "linear"), PLANCK, 1.0, 1.0, D_of_a=lambda a: a)
        fired = False
    except ValueError:
        fired = True
    lad = ladder_constants(a_grid(0.1, 1.0, 10, "log"), PLANCK, 1.0, 1.0)
    results.append(
        _check(
            "ladder guard: rejects linear-0.04 K=11, accepts log-0.1 K=10",
            fired and lad.n_steps == 10,
            f"log-0.1 K=10 bits={lad.bits_consumed:.2f}",
        )
    )

    # 3. Codec: imask_add(B=16) bit-equals the uint16 iadd path; iadd/isub invert.
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    vals = jax.random.randint(k1, (4096,), 0, 2**16, dtype=jnp.int32)
    incs = jax.random.randint(k2, (4096,), -(2**17), 2**17, dtype=jnp.int32)
    u16 = iadd(vals.astype(jnp.uint16), incs)
    m16 = imask_add(vals, incs, 16)
    ok_mask = bool(jnp.array_equal(u16.astype(jnp.int32), m16))
    ok_inv = bool(jnp.array_equal(isub(u16, incs), vals.astype(jnp.uint16)))
    results.append(_check("codec: imask(16) == uint16 path; isub inverts iadd", ok_mask and ok_inv))

    # 4. force == ZA identity: the force solve applied to a linear-density mesh
    # must reproduce (ik/k^2) delta -- the SAME kernel via a different code path.
    N, L = 32, 200.0
    cosmo = PLANCK
    delta0 = linear_delta0(jax.random.PRNGKey(1), N, L, cosmo)
    psi = za_psi(delta0, N, L)  # (N^3, 3) displacement of the SAME delta
    ikx, iky, ikz, inv_k2 = k_components(N, L)
    dk = jnp.fft.rfftn(delta0)
    gx = jnp.fft.irfftn(dk * ikx * inv_k2, s=(N, N, N))
    fmax = float(jnp.max(jnp.abs(gx.reshape(-1) - psi[:, 0])))
    scale = float(jnp.max(jnp.abs(psi[:, 0])))
    results.append(
        _check("force kernel == ZA kernel (machine precision)", fmax < 1e-6 * scale,
               f"max abs diff {fmax:.2e} vs scale {scale:.2e}")
    )

    # 5. P(k) estimator recovers the input eh98 spectrum (binned, ~sample variance).
    N, L = 64, 500.0
    delta0 = linear_delta0(jax.random.PRNGKey(2), N, L, cosmo)
    kc, pk, nm = pk_estimator(np.asarray(delta0), L, n_bins=12, k_max=0.6 * np.pi * N / L)
    p_ref = linear_power(kc, cosmo)
    sel = nm > 50
    rel = np.abs(pk[sel] / p_ref[sel] - 1.0)
    exp = 3.0 * np.sqrt(2.0 / nm[sel])  # ~3 sigma of the chi^2 sample variance
    results.append(
        _check("P(k) estimator recovers eh98 within sample variance",
               bool((rel < np.maximum(exp, 0.1)).all()), f"max rel dev {rel.max():.3f}")
    )

    # 6. paint_int vs paint_f32 agree to the fixed-point tolerance.
    pos = jax.random.uniform(jax.random.PRNGKey(3), (20000, 3), minval=0.0, maxval=L)
    F = 12
    mi = paint_int(pos, 32, L, frac_bits=F).astype(jnp.float32) * 2.0**-F
    mf = paint_f32(pos, 32, L)
    dmax = float(jnp.max(jnp.abs(mi - mf)))
    # Bound: each corner deposit errs by <= 2^-(F+1); a cell collects ~lambda*8
    # deposits (lambda ~ 5 here) -> allow 32 deposits' worth.
    results.append(
        _check(f"paint_int(F={F}) vs paint_f32 within 32*2^-{F + 1}", dmax < 32 * 2.0 ** -(F + 1),
               f"max cell diff {dmax:.2e}")
    )

    # 7. CPU micro-replay A: synthetic deterministic force, 100 steps, exact bits.
    key = jax.random.PRNGKey(4)
    kx_, kw_, kc_ = jax.random.split(key, 3)
    n = 4096
    x0 = jax.random.randint(kx_, (n, 3), 0, 2**16, dtype=jnp.int32).astype(jnp.uint16)
    w0 = jax.random.randint(kw_, (n, 3), -(2**14), 2**14, dtype=jnp.int32).astype(jnp.int16)
    cs = StepConsts(
        c1=jnp.abs(jax.random.normal(kc_, (100,), dtype=jnp.float32)) * 0.03,
        c2=jnp.abs(jax.random.normal(kc_, (100,), dtype=jnp.float32)) * 0.03,
        kappa=jnp.ones((100,), dtype=jnp.float32) * 0.7,
    )

    def toy_force(xp):  # deterministic elementwise pseudo-force
        return jnp.sin(xp * 0.37) * 55.0 + jnp.cos(xp[:, ::-1] * 0.11) * 21.0

    x, w = x0, w0
    for k in range(100):
        c = StepConsts(cs.c1[k], cs.c2[k], cs.kappa[k])
        x, w = step_fwd(x, w, c, toy_force, s_x=1.0)
    for k in reversed(range(100)):
        c = StepConsts(cs.c1[k], cs.c2[k], cs.kappa[k])
        x, w = step_rev(x, w, c, toy_force, s_x=1.0)
    ok_a = bool(jnp.array_equal(x, x0) and jnp.array_equal(w, w0))
    results.append(_check("micro-replay A: 100 synthetic steps, exact bits", ok_a))

    # 8. CPU micro-replay B: real 16^3 BullFrog PM, K=8, exact bits + the float
    # twin's primal bit-matches the integer trajectory.
    N, L, K = 16, 100.0, 8
    delta0 = linear_delta0(jax.random.PRNGKey(5), N, L, cosmo)
    xph, v = za_ics(delta0, N, L, 0.1, cosmo)
    s_x = L / U16_MOD
    lad_tmp = ladder_constants(a_grid(0.1, 1.0, K, "log"), cosmo, 1.0, s_x)
    s_w0 = s_w0_policy(float(jnp.max(jnp.abs(v))), lad_tmp.P[-1])
    lad = ladder_constants(a_grid(0.1, 1.0, K, "log"), cosmo, s_w0, s_x)
    force = make_force_fn(N, L, N**3)
    x0i, w0i = encode_x(xph, s_x), encode_w(v, s_w0)
    cs = step_consts(lad)
    xi, wi = x0i, w0i
    xf = x0i.astype(jnp.float32)
    wf = w0i.astype(jnp.float32)
    twin_mismatch = 0
    for k in range(K):
        c = StepConsts(cs.c1[k], cs.c2[k], cs.kappa[k])
        xi, wi = step_fwd(xi, wi, c, force, s_x)
        xf, wf = step_float(xf, wf, c, force, s_x)
        twin_mismatch += int(jnp.sum(xf != xi.astype(jnp.float32)))
        twin_mismatch += int(jnp.sum(wf != wi.astype(jnp.float32)))
    for k in reversed(range(K)):
        c = StepConsts(cs.c1[k], cs.c2[k], cs.kappa[k])
        xi, wi = step_rev(xi, wi, c, force, s_x)
    ok_rep = bool(jnp.array_equal(xi, x0i) and jnp.array_equal(wi, w0i))
    results.append(_check("micro-replay B: 16^3 BullFrog K=8 PM, exact bits", ok_rep))
    results.append(
        _check("STE float twin primal bit-matches integer trajectory", twin_mismatch == 0,
               f"{twin_mismatch} mismatched components")
    )

    n_pass = sum(results)
    print(f"{n_pass}/{len(results)} checks passed")
    return n_pass == len(results)


if __name__ == "__main__":
    import sys

    sys.exit(0 if _self_checks() else 1)
