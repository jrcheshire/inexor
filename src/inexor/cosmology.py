"""Background and linear-theory cosmology: the host float64 precision island.

numpy/scipy only (no JAX), so it can be imported before JAX is configured. Everything here is a
constant of the run consumed by the integrator tables and IC generators. The second-order growth
D2 solves the LCDM ODE, which the BullFrog weights need (Rampf, List & Hahn 2024,
arXiv:2409.19049, Sec. 4.4); EdS -(3/7) D^2 is an explicit option. `linear_power` backends: "eh98"
(analytic, Eisenstein & Hu 1998) or "table" (tabulated z=0 (k, P), e.g. from CAMB; a
`LinearPkTable` is the checked form the IC generators and their manifests carry).
"""

import dataclasses
import hashlib
import json
import math
from functools import lru_cache

import numpy as np
from scipy.integrate import quad, simpson

from .config import PLANCK, Cosmology  # noqa: F401  (re-exported for callers)

# The EH98 reference C code uses this in place of Euler's e; kept to reproduce it exactly.
_E_EH98 = 2.718282


# Background + linear growth (exact flat LCDM, radiation neglected)


def E_of_a(a, cosmo):
    """Dimensionless Hubble rate E(a) = H/H0, flat LCDM, radiation neglected."""
    return np.sqrt(cosmo.Omega_m / a**3 + cosmo.Omega_Lambda)


def _growth_integrand(a, Om, OL):
    e = np.sqrt(Om / a**3 + OL)
    return 1.0 / (a * e) ** 3


def _growth_unnorm(a, cosmo):
    # D(a) = (5 Om / 2) E(a) integral_0^a da' / (a' E(a'))^3; tends to a as a -> 0.
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
    """Growth factor normalized to D -> a in matter domination.

    The convention of the local-f_NL relation delta(k) = M(k, z) phi(k); differs from
    growth_factor_a by the constant D0 = _growth_unnorm(1).
    """
    return _growth_unnorm(float(a), cosmo)


GROWTH2_MODELS = ("lcdm", "eds")

_A_ODE_START = 1e-5
_A_ODE_END = 2.0


@lru_cache(maxsize=None)
def _growth2_solution(cosmo):
    """Dense solution of the linear and second-order growth ODEs in x = ln a.

    With Om(a) = Omega_m / (a^3 E^2), both orders obey
        y_xx + (2 - 3/2 Om(a)) y_x - 3/2 Om(a) y = src,
    src = 0 for D and -3/2 Om(a) D^2 for the second-order growth E (Rampf, List &
    Hahn 2024, eqs. 3.5). Started on the growing mode D = a - (2L/11) a^4,
    E = -(3/7) D^2 - (3L/1001) D^5, L = Omega_Lambda / Omega_m, so both are
    UNNORMALISED (D -> a as a -> 0, the `_growth_unnorm` convention). State is
    (D, D_x, E, E_x).
    """
    from scipy.integrate import solve_ivp

    Om0, lam = cosmo.Omega_m, cosmo.Omega_Lambda / cosmo.Omega_m

    def rhs(x, y):
        a = np.exp(x)
        om = Om0 / (a**3 * E_of_a(a, cosmo) ** 2)
        D, Dx, E, Ex = y
        fr = 2.0 - 1.5 * om
        return [Dx, 1.5 * om * D - fr * Dx, Ex, 1.5 * om * (E - D * D) - fr * Ex]

    a0 = _A_ODE_START
    D = a0 - (2.0 * lam / 11.0) * a0**4
    Dx = a0 - (8.0 * lam / 11.0) * a0**4
    E = -(3.0 / 7.0) * D**2 - (3.0 * lam / 1001.0) * D**5
    Ex = (-(6.0 / 7.0) * D - (15.0 * lam / 1001.0) * D**4) * Dx
    sol = solve_ivp(rhs, (np.log(a0), np.log(_A_ODE_END)), [D, Dx, E, Ex], method="DOP853",
                    rtol=1e-13, atol=1e-30, dense_output=True, max_step=0.02)
    if not sol.success:
        raise RuntimeError(f"second-order growth ODE failed: {sol.message}")
    return sol.sol


def _growth2_state(a, cosmo):
    a = float(a)
    if not (_A_ODE_START <= a <= _A_ODE_END):
        raise ValueError(f"a = {a} is outside the growth ODE's range "
                         f"[{_A_ODE_START}, {_A_ODE_END}]")
    return _growth2_solution(cosmo)(np.log(a))


def _check_model(model):
    if model not in GROWTH2_MODELS:
        raise ValueError(f"model must be one of {GROWTH2_MODELS}, got {model!r}")


def growth_factor_2(a, cosmo, model="lcdm"):
    """Second-order growth factor D2(a) for 2LPT, normalized with growth_factor_a.

    Normalized consistently with growth_factor_a (D1(a=1) = 1); D2 < 0 and carries 1/D0^2.
    In `lpt`'s convention the 2LPT position is x = q + D1 Psi1 - D2 Psi2.

    model="lcdm" (default) is the solution of the LCDM second-order growth ODE
    (`_growth2_solution`). model="eds" is the approximation D2 = -(3/7) D1^2
    (Bouchet et al. 1995), off by ~2e-5 at z = 9 and ~0.8% at z = 0; kept only to
    reproduce runs made with it.
    """
    _check_model(model)
    if model == "eds":
        return -(3.0 / 7.0) * growth_factor_a(a, cosmo) ** 2
    return float(_growth2_state(a, cosmo)[2]) / _D0(cosmo) ** 2


def growth_rate_2(a, cosmo, model="lcdm"):
    """Second-order growth rate f2 = dlnD2/dlna. model="eds" gives 2 f1."""
    _check_model(model)
    if model == "eds":
        return 2.0 * growth_rate_a(a, cosmo)
    s = _growth2_state(a, cosmo)
    return float(s[3] / s[2])


def growth2_and_slope(a, cosmo):
    """(D2, dD2/dD1) at `a` from the LCDM ODE, normalized with growth_factor_a.

    The pair the BullFrog weights need (`integrate._bullfrog_weights`); the slope
    is taken against the NORMALIZED D1, so it carries 1/D0.
    """
    D, Dx, E, Ex = _growth2_state(a, cosmo)
    D0 = _D0(cosmo)
    return float(E) / D0**2, float(Ex / Dx) / D0


# EH98 transfer + sigma8-normalized linear P(k)


class _EH98:
    """Scalar EH98 parameters for one cosmology (TFset_parameters of the reference code)."""

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
    """Full EH98 transfer function T(k), k in h/Mpc; T -> 1 as k -> 0 (Eisenstein & Hu 1998, Eq. 16)."""
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
    backend="table": tabulated z=0 spectrum, table = (k_table, P_table) arrays or a
    `LinearPkTable` (e.g. a CAMB dump); log-log interpolated, refuses k
    outside the table range (loud, never extrapolates). Both backends scale to
    z with growth_factor_a**2.
    """
    k_arr = np.atleast_1d(np.asarray(k_hmpc, dtype=np.float64))
    if backend == "eh98":
        P0 = _eh98_amplitude(cosmo) * k_arr**cosmo.n_s * transfer_eh98(k_arr, cosmo) ** 2
    elif backend == "table":
        if table is None:
            raise ValueError("backend='table' requires table=(k_table, P_table)")
        if isinstance(table, LinearPkTable):
            table = (table.k, table.P)
        k_t = np.asarray(table[0], dtype=np.float64)
        P_t = np.asarray(table[1], dtype=np.float64)
        # relative slack so exact-endpoint queries do not trip the guard
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


# 1D |k| table for IC generation


def _refuse_outside(k_arr, k_lo, k_hi, what):
    # As linear_power's table guard; an empty query (a k-cut can empty a slab) is a no-op.
    if k_arr.size == 0:
        return
    if k_arr.min() < k_lo * (1 - 1e-12) or k_arr.max() > k_hi * (1 + 1e-12):
        raise ValueError(
            f"requested k in [{k_arr.min():.3e}, {k_arr.max():.3e}] outside the "
            f"{what} table range [{k_lo:.3e}, {k_hi:.3e}]; refusing to extrapolate"
        )


class ICKTable:
    """1D log-spaced |k| table of linear P(k, z=0) and transfer T(k).

    Lets IC colouring evaluate P and T per slab without a 3D |k| grid. P is interpolated
    log-log (positive); T linearly in ln k (it may cross zero). Both refuse k outside the table.
    Deliberately not cached or hashable: built once per run by `ic_k_table` and passed
    explicitly, so nothing can alias across cosmologies.
    """

    __slots__ = ("k", "P", "T", "_lnk", "_lnP")

    def __init__(self, k, P, T):
        self.k = np.asarray(k, dtype=np.float64)
        self.P = np.asarray(P, dtype=np.float64)
        self.T = np.asarray(T, dtype=np.float64)
        if not (self.k.ndim == 1 and self.k.shape == self.P.shape == self.T.shape):
            raise ValueError("k, P, T must be 1D arrays of equal length")
        if not np.all(np.diff(self.k) > 0):
            raise ValueError("k must be strictly increasing")
        if np.any(self.P <= 0):
            raise ValueError("P must be positive (log-log interpolation)")
        self._lnk = np.log(self.k)
        self._lnP = np.log(self.P)

    def P_of_k(self, k_hmpc):
        """Linear P(k, z=0), log-log interpolated; refuses outside the range."""
        k_arr = np.atleast_1d(np.asarray(k_hmpc, dtype=np.float64))
        _refuse_outside(k_arr, self.k[0], self.k[-1], "P(k)")
        return np.exp(np.interp(np.log(k_arr), self._lnk, self._lnP)).reshape(
            np.shape(k_hmpc) if np.ndim(k_hmpc) else ()
        )

    def T_of_k(self, k_hmpc):
        """Transfer T(k), linear-in-ln(k) interpolated; refuses outside the range."""
        k_arr = np.atleast_1d(np.asarray(k_hmpc, dtype=np.float64))
        _refuse_outside(k_arr, self.k[0], self.k[-1], "T(k)")
        return np.interp(np.log(k_arr), self._lnk, self.T).reshape(
            np.shape(k_hmpc) if np.ndim(k_hmpc) else ()
        )


# Universal table node range, h/Mpc (same span as the sigma8 integral). Fixed rather than
# per-grid so the interpolation error depends on physical k alone and cancels between runs at
# different resolutions or box sizes that share modes.
K_TABLE_MIN = 1e-4
K_TABLE_MAX = 1e2


def ic_k_table(cosmo, n_mesh, box_size, n_points=32768, backend="eh98", table=None):
    """Build the ICKTable on n_points universal log nodes over [K_TABLE_MIN, K_TABLE_MAX].

    (n_mesh, box_size) are used only to refuse at build time a grid whose |k| range
    [2 pi/L, sqrt(3) pi n/L] the universal range does not cover. backend="eh98": P and T
    from EH98. backend="table": P resampled from a (k, P) dump that must cover the range, and
    T derived from it, T = sqrt(P / k^n_s) scaled to 1 at K_TABLE_MIN, so the potential of an
    f_NL transform matches the spectrum it colours (T(1e-4 h/Mpc) is 1 to ~1e-4 in LCDM). The
    default n_points keeps the interpolation error (quadratic in node spacing) at the few x
    1e-7 level.
    """
    n_mesh = int(n_mesh)
    if n_mesh < 2:
        raise ValueError(f"n_mesh must be >= 2, got {n_mesh}")
    k_f = 2.0 * np.pi / box_size
    k_grid_hi = math.sqrt(3.0) * np.pi * n_mesh / box_size
    if k_f < K_TABLE_MIN or k_grid_hi > K_TABLE_MAX:
        raise ValueError(
            f"grid |k| range [{k_f:.3e}, {k_grid_hi:.3e}] exceeds the universal table "
            f"range [{K_TABLE_MIN:.0e}, {K_TABLE_MAX:.0e}]; refusing at build time"
        )
    k = np.exp(np.linspace(np.log(K_TABLE_MIN), np.log(K_TABLE_MAX), int(n_points)))
    P = linear_power(k, cosmo, z=0.0, backend=backend, table=table)
    if backend == "table":
        T = np.sqrt(P / k**cosmo.n_s)
        T = T / T[0]
    else:
        T = transfer_eh98(k, cosmo)
    return ICKTable(k, P, T)


# Tabulated linear spectrum, carried with the data it seeds

LINEAR_PK_FORMAT = "inexor-linear-pk-1"
# a table whose sigma8 differs from the cosmology's by more than this (relative) is refused
LINEAR_PK_SIGMA8_RTOL = 1e-3


class LinearPkTable:
    """A tabulated z = 0 linear P(k) (k in h/Mpc, P in (Mpc/h)^3) for one cosmology.

    Made by `load_linear_pk` (a file from `scripts/run/camb_linear_pk.py`) or `from_record`
    (the copy an IC or checkpoint manifest embeds). Both refuse a table that would silently
    seed another run: another format, z != 0, any cosmological parameter other than the
    run's, sigma8 off the cosmology's by more than LINEAR_PK_SIGMA8_RTOL, or k not covering
    [K_TABLE_MIN, K_TABLE_MAX]. `sha256` hashes the float64 (k, P) bytes, so the file and
    every embedded copy share it. Pass it as `table` with backend="table".
    """

    __slots__ = ("k", "P", "source", "meta")

    def __init__(self, k, P, source, meta=None):
        self.k = np.ascontiguousarray(k, dtype=np.float64)
        self.P = np.ascontiguousarray(P, dtype=np.float64)
        self.source = str(source)
        self.meta = dict(meta or {})

    @property
    def sha256(self):
        return hashlib.sha256(self.k.tobytes() + self.P.tobytes()).hexdigest()

    def stamp(self):
        """{source, sha256}: what a card or an export header records."""
        return dict(source=self.source, sha256=self.sha256)

    def record(self):
        """The whole table as a JSON-able dict (the file format, and what manifests embed)."""
        return dict(format=LINEAR_PK_FORMAT, source=self.source, sha256=self.sha256,
                    **self.meta, k=self.k.tolist(), P=self.P.tolist())

    @classmethod
    def from_record(cls, rec, cosmo):
        """A checked table from `record()`'s dict; refuses as the class docstring says."""
        fmt = rec.get("format")
        if fmt != LINEAR_PK_FORMAT:
            raise ValueError(f"linear P(k) table format {fmt!r}, expected {LINEAR_PK_FORMAT!r}")
        if float(rec.get("z", math.nan)) != 0.0:
            raise ValueError(f"the linear P(k) table is at z = {rec.get('z')!r}; it must be "
                             "the z = 0 spectrum (the generators scale it with D(a)^2)")
        want = dataclasses.asdict(cosmo)
        have = rec.get("cosmology") or {}
        bad = sorted(n for n in set(want) | set(have)
                     if n not in have or n not in want
                     or not math.isclose(float(have[n]), float(want[n]), rel_tol=1e-12))
        if bad:
            raise ValueError(
                "the linear P(k) table's cosmology differs from the run's in "
                + ", ".join(f"{n} ({have.get(n)!r} vs {want.get(n)!r})" for n in bad))
        k = np.asarray(rec.get("k"), dtype=np.float64)
        P = np.asarray(rec.get("P"), dtype=np.float64)
        if not (k.ndim == 1 and k.shape == P.shape and k.size >= 2):
            raise ValueError("the linear P(k) table needs 1D k and P of equal length >= 2")
        if not (np.all(np.isfinite(k)) and np.all(np.isfinite(P)) and np.all(k > 0)
                and np.all(P > 0) and np.all(np.diff(k) > 0)):
            raise ValueError("the linear P(k) table needs finite, positive P and positive, "
                             "strictly increasing k")
        if k[0] > K_TABLE_MIN * (1 + 1e-12) or k[-1] < K_TABLE_MAX * (1 - 1e-12):
            raise ValueError(
                f"the linear P(k) table spans k = [{k[0]:.3e}, {k[-1]:.3e}] h/Mpc and must "
                f"cover [{K_TABLE_MIN:.0e}, {K_TABLE_MAX:.0e}]")
        meta = {n: v for n, v in rec.items()
                if n not in ("format", "source", "sha256", "k", "P")}
        tab = cls(k, P, rec.get("source", "unknown"), meta)
        if "sha256" in rec and rec["sha256"] != tab.sha256:
            raise ValueError("the linear P(k) table's sha256 does not match its (k, P): "
                             "the embedded copy was altered")
        s8 = sigma_R(8.0, cosmo, backend="table", table=tab)
        if abs(s8 / cosmo.sigma8 - 1.0) > LINEAR_PK_SIGMA8_RTOL:
            raise ValueError(
                f"the linear P(k) table has sigma8 = {s8:.6f} and the cosmology {cosmo.sigma8}"
                f" (relative {s8 / cosmo.sigma8 - 1.0:+.2e}, refused beyond "
                f"{LINEAR_PK_SIGMA8_RTOL:g}); make the table at the run's sigma8")
        return tab


def load_linear_pk(path, cosmo):
    """The checked `LinearPkTable` in the JSON file at `path` (see `LinearPkTable`)."""
    with open(path) as fh:
        return LinearPkTable.from_record(json.load(fh), cosmo)
