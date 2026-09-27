"""cosmology.py: growth identities, EH98 sanity, sigma8 normalization, backends.

Tolerances follow mbody's test_cosmology conventions; the module is a
jax-free float64 island, so everything here is plain numpy.
"""

import numpy as np
import pytest

from inexor.config import PLANCK, Cosmology
from inexor.cosmology import (
    E_of_a,
    growth_factor_2,
    growth_factor_a,
    growth_factor_md,
    growth_rate_2,
    growth_rate_a,
    ic_k_table,
    linear_power,
    sigma_R,
    transfer_eh98,
)


def test_module_is_jax_free():
    """The precision island must never grow a jax dependency (house rule)."""
    import inexor.cosmology as mod

    assert not any(name in ("jax", "jnp") for name in vars(mod)), "cosmology must stay jax-free"


def test_growth_normalization_and_md_limit():
    assert growth_factor_a(1.0, PLANCK) == pytest.approx(1.0, abs=1e-12)
    # Matter domination: D_md(a) -> a as a -> 0.
    for a in (1e-3, 1e-2):
        assert growth_factor_md(a, PLANCK) == pytest.approx(a, rel=5e-3)


def test_growth_rate_limits():
    # f -> 1 in matter domination; f(1) ~ Omega_m^0.55 (classic approximation).
    assert growth_rate_a(1e-3, PLANCK) == pytest.approx(1.0, abs=2e-3)
    assert growth_rate_a(1.0, PLANCK) == pytest.approx(PLANCK.Omega_m**0.55, rel=1e-2)


def test_second_order_growth_eds_identities():
    """The legacy EdS option; the LCDM default is pinned in test_growth2.py."""
    a = 0.5
    D1 = growth_factor_a(a, PLANCK)
    assert growth_factor_2(a, PLANCK, "eds") == pytest.approx(-(3.0 / 7.0) * D1**2, rel=1e-12)
    assert growth_rate_2(a, PLANCK, "eds") == pytest.approx(2.0 * growth_rate_a(a, PLANCK),
                                                             rel=1e-12)


def test_E_of_a_endpoints():
    assert E_of_a(1.0, PLANCK) == pytest.approx(1.0, abs=1e-12)
    a = 1e-2
    assert E_of_a(a, PLANCK) == pytest.approx(np.sqrt(PLANCK.Omega_m / a**3), rel=1e-3)


def test_transfer_low_k_limit():
    T = transfer_eh98(np.array([1e-5, 1e-4]), PLANCK)
    assert np.all(np.abs(T - 1.0) < 2e-2)
    # monotone suppression at high k
    assert transfer_eh98(10.0, PLANCK) < 1e-2


def test_sigma8_round_trip():
    assert sigma_R(8.0, PLANCK, z=0.0) == pytest.approx(PLANCK.sigma8, rel=1e-3)


def test_linear_power_z_scaling():
    k = np.array([0.05, 0.1])
    P0 = linear_power(k, PLANCK, z=0.0)
    P1 = linear_power(k, PLANCK, z=1.0)
    D = growth_factor_a(0.5, PLANCK)
    assert np.allclose(P1, P0 * D**2, rtol=1e-12)


def test_table_backend_round_trips_eh98():
    k_t = np.geomspace(1e-4, 1e2, 2048)
    P_t = linear_power(k_t, PLANCK)
    k = np.geomspace(1e-3, 10.0, 64)
    P_direct = linear_power(k, PLANCK)
    P_table = linear_power(k, PLANCK, backend="table", table=(k_t, P_t))
    # log-log interpolation of a smooth spectrum on a dense grid
    assert np.allclose(P_table, P_direct, rtol=2e-4)
    # sigma_R agrees through the table backend too
    assert sigma_R(8.0, PLANCK, backend="table", table=(k_t, P_t)) == pytest.approx(
        PLANCK.sigma8, rel=1e-3
    )


def test_table_backend_refuses_extrapolation():
    k_t = np.geomspace(1e-3, 1.0, 128)
    P_t = linear_power(k_t, PLANCK)
    with pytest.raises(ValueError, match="refusing to extrapolate"):
        linear_power(np.array([10.0]), PLANCK, backend="table", table=(k_t, P_t))
    with pytest.raises(ValueError, match="requires table"):
        linear_power(np.array([0.1]), PLANCK, backend="table")
    with pytest.raises(ValueError, match="backend"):
        linear_power(np.array([0.1]), PLANCK, backend="camb")


def test_distinct_cosmologies_do_not_share_caches():
    other = Cosmology(Omega_m=0.35, sigma8=0.75)
    assert sigma_R(8.0, other) == pytest.approx(0.75, rel=1e-3)
    assert sigma_R(8.0, PLANCK) == pytest.approx(0.81, rel=1e-3)


# ---------------------------------------------------------------------------
# ICKTable (M-v2-5, D-v2-15 clause 2)
# ---------------------------------------------------------------------------


def _realized_kmag(n, L):
    """The exact |k| multiset of the (n, n, n//2+1) rfft half-grid."""
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=L / n)
    kk = np.sqrt(
        kx.reshape(n, 1, 1) ** 2 + kx.reshape(1, n, 1) ** 2 + kz.reshape(1, 1, -1) ** 2
    )
    return kk.ravel()


def test_ic_k_table_meets_the_bar_on_the_full_cdev_multiset():
    """Charter bar (JC 2026-08-10): max rel error of table P(k) vs analytic EH98
    over EVERY realized |k| on the production rfft grid < 1e-4. Asserted here on
    the exact C-dev multiset (n=256, L=128 -- 8.5e6 values, the one scale where
    materializing it is the point); the C-gh/C-hero grids are the probe's job.
    """
    n, L = 256, 128.0
    tab = ic_k_table(PLANCK, n, L)
    kk = _realized_kmag(n, L)
    kk = kk[kk > 0]  # DC is overwritten by every caller, never interpolated
    P_tab = tab.P_of_k(kk)
    P_ref = linear_power(kk, PLANCK)
    e_max = np.max(np.abs(P_tab / P_ref - 1.0))
    assert e_max < 1e-4, f"table error {e_max:.3e} over the 1e-4 bar"


def test_ic_k_table_bar_can_fail():
    """Anti-vacuity: a 32-point table must MISS the bar, or the bar tests nothing."""
    n, L = 64, 128.0
    tab = ic_k_table(PLANCK, n, L, n_points=32)
    kk = _realized_kmag(n, L)
    kk = kk[kk > 0]
    e_max = np.max(np.abs(tab.P_of_k(kk) / linear_power(kk, PLANCK) - 1.0))
    assert e_max > 1e-4, f"32-point table read {e_max:.3e}; the bar cannot fail"


def test_ic_k_table_covers_the_grid_endpoints():
    """The DC substitute (smallest nonzero |k| = k_f) and sqrt(3)*k_Nyq are both
    interior: the refusal must NOT fire anywhere on the realized grid."""
    n, L = 64, 128.0
    tab = ic_k_table(PLANCK, n, L)
    kk = _realized_kmag(n, L)
    kk = kk[kk > 0]
    tab.P_of_k(kk)  # would raise on any excursion
    tab.T_of_k(kk)
    k_f = 2.0 * np.pi / L
    assert kk.min() == pytest.approx(k_f, rel=1e-12)
    assert tab.k[0] < k_f and tab.k[-1] > kk.max()


def test_ic_k_table_refuses_extrapolation():
    tab = ic_k_table(PLANCK, 64, 128.0)
    for bad in (tab.k[0] * 0.5, tab.k[-1] * 2.0):
        with pytest.raises(ValueError, match="refusing to extrapolate"):
            tab.P_of_k(np.array([bad]))
        with pytest.raises(ValueError, match="refusing to extrapolate"):
            tab.T_of_k(np.array([bad]))


def test_ic_k_table_transfer_matches_eh98():
    """T_of_k: exact at the nodes (np.interp identity), interp-error-class between."""
    tab = ic_k_table(PLANCK, 256, 128.0)
    sub = tab.k[:: len(tab.k) // 199]
    assert np.array_equal(tab.T_of_k(sub), transfer_eh98(sub, PLANCK))
    mid = np.sqrt(tab.k[100:-100:37] * tab.k[101:-99:37])  # geometric midpoints
    assert np.allclose(tab.T_of_k(mid), transfer_eh98(mid, PLANCK), rtol=1e-4, atol=1e-8)


def test_ic_k_tables_do_not_alias_across_cosmologies():
    """No cache exists to share, and the tables must actually differ."""
    other = Cosmology(Omega_m=0.35, sigma8=0.75)
    t1 = ic_k_table(PLANCK, 64, 128.0)
    t2 = ic_k_table(other, 64, 128.0)
    assert not np.array_equal(t1.P, t2.P)
    assert not np.array_equal(t1.T, t2.T)
    assert np.array_equal(t1.k, t2.k)  # same grid geometry, different physics


def test_ic_k_table_shape_and_scalar_handling():
    tab = ic_k_table(PLANCK, 64, 128.0)
    k2d = np.full((3, 4), 0.5)
    assert tab.P_of_k(k2d).shape == (3, 4)
    assert np.ndim(tab.P_of_k(0.5)) == 0
    assert np.ndim(tab.T_of_k(0.5)) == 0


def test_ic_k_table_accepts_empty_queries():
    """A slab-streamed caller's k-cut can empty a slab; an empty query is a
    no-op, not a crash (found by the leg-VI Jensen diagnostic, 2026-08-10)."""
    tab = ic_k_table(PLANCK, 64, 128.0)
    assert tab.P_of_k(np.array([])).size == 0
    assert tab.T_of_k(np.array([])).size == 0
