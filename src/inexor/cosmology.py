"""Background + linear-theory cosmology: the host float64 precision island.

STRICTLY jax-free (numpy/scipy only; mbody "precision island" pattern) so any
caller can import it before or without configuring JAX. Everything here is a
constant of the run, computed in float64 and consumed by the integrator
coefficient tables (integrate.py) and the IC generators (ic.py / lpt.py).

Migrated verbatim from scripts/_m0_common.py (M0-verdicted bits) with mbody
ports for the 2LPT growth pair and sigma_R (mbody/cosmology.py:377-493).
linear_power gains a backend dispatch: "eh98" (analytic, self-contained) or
"table" (tabulated (k, P) at z=0, e.g. a CAMB dump from the parity env --
the M1 CAMB-parity hook, plan 2026-07-13).
"""

import math
from functools import lru_cache

import numpy as np
from scipy.integrate import quad, simpson

from .config import PLANCK, Cosmology  # noqa: F401  (re-exported for callers)

# Constant the EH98 reference C code uses in place of Euler's e; kept verbatim
# so the port reproduces the original formula bit-for-bit (mbody convention).
_E_EH98 = 2.718282


# ============================================================================
# Background + linear growth (exact flat LCDM, radiation neglected)
# ============================================================================


def E_of_a(a, cosmo):
    """Dimensionless Hubble rate E(a) = H/H0, flat LCDM, radiation neglected."""
    return np.sqrt(cosmo.Omega_m / a**3 + cosmo.Omega_Lambda)


def _growth_integrand(a, Om, OL):
    e = np.sqrt(Om / a**3 + OL)
    return 1.0 / (a * e) ** 3


def _growth_unnorm(a, cosmo):
    # D(a) propto (5 Om / 2) E(a) * integral_0^a da' / (a' E(a'))^3 (exact flat LCDM).
    # Tends to a as a -> 0 (matter domination), which is the MD normalization.
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


def growth_factor_md(a, cosmo):
    """Growth factor normalized to D = a in matter domination (mbody port).

    This is the convention used in the local-f_NL relation between the
    primordial potential and the linear density, delta(k) = M(k, z) phi(k).
    It is the unnormalized growth integral, which tends to a as a -> 0; it
    differs from growth_factor_a by the constant 1 / lim_{a->0} [D(a)/a].
    """
    return _growth_unnorm(float(a), cosmo)


def growth_factor_2(a, cosmo):
    """Second-order growth factor D2(a) for 2LPT, the EdS approximation.

    D2 = -(3/7) D1^2 (Bouchet et al. 1995; the standard 2LPT-IC choice).
    Normalized consistently with growth_factor_a (D1(a=1) = 1), so the
    Lagrangian displacement is Psi = D1 Psi1 + D2 Psi2. The dropped LCDM
    correction Omega_m(a)^(-1/143) is < 0.9% at z=0 and ~2e-5 at IC redshifts
    (mbody docstring, carried verbatim).
    """
    D1 = growth_factor_a(a, cosmo)
    return -(3.0 / 7.0) * D1**2


def growth_rate_2(a, cosmo):
    """Second-order growth rate f2 = dlnD2/dlna = 2 f1 (EdS approximation)."""
    return 2.0 * growth_rate_a(a, cosmo)


# ============================================================================
# EH98 transfer + sigma8-normalized linear P(k)  (verbatim _m0_common port)
# ============================================================================


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


def linear_power(k_hmpc, cosmo, z=0.0, backend="eh98", table=None):
    """Linear matter P(k, z) in (Mpc/h)^3, k in h/Mpc.

    backend="eh98": analytic EH98, sigma8-normalized (self-contained default).
    backend="table": tabulated z=0 spectrum, table = (k_table, P_table) arrays
    (e.g. a CAMB dump from the parity env); log-log interpolated, refuses k
    outside the table range (loud, never extrapolates). Both backends scale to
    z with growth_factor_a**2.
    """
    k_arr = np.atleast_1d(np.asarray(k_hmpc, dtype=np.float64))
    if backend == "eh98":
        P0 = _eh98_amplitude(cosmo) * k_arr**cosmo.n_s * transfer_eh98(k_arr, cosmo) ** 2
    elif backend == "table":
        if table is None:
            raise ValueError("backend='table' requires table=(k_table, P_table)")
        k_t = np.asarray(table[0], dtype=np.float64)
        P_t = np.asarray(table[1], dtype=np.float64)
        # 1-ulp slack: exact-endpoint queries (e.g. sigma_R on the table's own
        # k range) must not trip the guard
        if k_arr.min() < k_t.min() * (1 - 1e-12) or k_arr.max() > k_t.max() * (1 + 1e-12):
            raise ValueError(
                f"requested k in [{k_arr.min():.3e}, {k_arr.max():.3e}] outside the "
                f"table range [{k_t.min():.3e}, {k_t.max():.3e}]; refusing to extrapolate"
            )
        P0 = np.exp(np.interp(np.log(k_arr), np.log(k_t), np.log(P_t)))
    else:
        raise ValueError(f"backend must be 'eh98' or 'table', got {backend!r}")
    if z != 0.0:
        a = 1.0 / (1.0 + z)
        P0 = P0 * growth_factor_a(a, cosmo) ** 2
    return P0


def sigma_R(R, cosmo, z=0.0, backend="eh98", table=None):
    """Rms linear density fluctuation in spheres of radius R (Mpc/h) at z.

    By construction sigma_R(8, z=0) recovers cosmo.sigma8 for the eh98 backend
    (a normalization cross-check, especially useful against a CAMB table).
    """
    lnk = np.linspace(math.log(1e-4), math.log(1e2), 4000)
    k = np.exp(lnk)
    Pk0 = linear_power(k, cosmo, z=0.0, backend=backend, table=table)
    W = _tophat_window(k * R)
    s2 = simpson(k**3 * Pk0 * W**2 / (2.0 * math.pi**2), x=lnk)
    return math.sqrt(s2) * growth_factor_a(1.0 / (1.0 + z), cosmo)
