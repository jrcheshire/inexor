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


def test_second_order_growth_identities():
    a = 0.5
    D1 = growth_factor_a(a, PLANCK)
    assert growth_factor_2(a, PLANCK) == pytest.approx(-(3.0 / 7.0) * D1**2, rel=1e-12)
    assert growth_rate_2(a, PLANCK) == pytest.approx(2.0 * growth_rate_a(a, PLANCK), rel=1e-12)


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


def test_ccl_cross_check():
    """Optional-import CCL cross-check (mbody test_external_ccl pattern)."""
    ccl = pytest.importorskip("pyccl")
    cosmo_ccl = ccl.Cosmology(
        Omega_c=PLANCK.Omega_cdm,
        Omega_b=PLANCK.Omega_b,
        h=PLANCK.h,
        sigma8=PLANCK.sigma8,
        n_s=PLANCK.n_s,
        transfer_function="eisenstein_hu",
    )
    for a in (0.25, 0.5, 1.0):
        assert growth_factor_a(a, PLANCK) == pytest.approx(
            ccl.growth_factor(cosmo_ccl, a), rel=2e-3
        )
        assert growth_rate_a(a, PLANCK) == pytest.approx(ccl.growth_rate(cosmo_ccl, a), rel=2e-3)
    k = np.geomspace(1e-3, 1.0, 32)
    P_ccl = ccl.linear_matter_power(cosmo_ccl, k * PLANCK.h, 1.0) * PLANCK.h**3
    assert np.allclose(linear_power(k, PLANCK), P_ccl, rtol=5e-2)
