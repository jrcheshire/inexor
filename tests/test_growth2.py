"""The LCDM second-order growth and the BullFrog weights built on it.

BullFrog is consistent only with the true LCDM growth pair (Rampf, List & Hahn
2024, Sec. 4.4); with E = -(3/7) D^2 it converges to an EdS-coupled solution
whose late-time kicks are 12% weak. These pin the ODE solution against the
limits it must reproduce, and the weights against the continuum equation of
motion they must converge to. Bars sit above measured values (2026-09-25):
the linear-growth comparisons are limited by `growth_factor_a`'s quadrature
(~7e-9), everything internal to the ODE by roundoff.
"""

import numpy as np
import pytest

from inexor import cosmology as C
from inexor.config import PLANCK, Cosmology
from inexor.integrate import _bullfrog_weights, a_grid, bullfrog_table

EDS = Cosmology(Omega_m=1.0)


def test_ode_linear_growth_is_growth_factor_a():
    """Measured 7.1e-9, the quadrature's floor."""
    for a in np.geomspace(1e-3, 1.5, 25):
        assert C._growth2_state(a, PLANCK)[0] / C._growth_unnorm(a, PLANCK) == pytest.approx(
            1.0, abs=1e-7)


def test_eds_cosmology_gives_the_closed_form():
    """At Omega_m = 1 the ODE's own E is -(3/7) D^2 (measured 1.3e-15)."""
    for a in np.geomspace(1e-3, 1.5, 25):
        D, _, E, _ = C._growth2_state(a, EDS)
        assert E / (-(3.0 / 7.0) * D * D) == pytest.approx(1.0, abs=1e-12)


def test_eds_cosmology_tables_agree():
    """Measured 8.1e-9: the ODE's E against -(3/7) D^2 from the quadrature D."""
    ag = a_grid(0.1, 1.0, 40)
    lcdm, eds = bullfrog_table(ag, EDS), bullfrog_table(ag, EDS, growth2="eds")
    assert np.max(np.abs(lcdm.alphas - eds.alphas)) < 1e-7


def test_small_a_growing_mode_series():
    """E = -(3/7) D^2 - (3 L / 1001) D^5 + O(L^2 D^8), unnormalised (paper eq. 3.5)."""
    lam = PLANCK.Omega_Lambda / PLANCK.Omega_m
    for a in np.geomspace(1e-3, 1e-2, 5):
        D, _, E, _ = C._growth2_state(a, PLANCK)
        assert E / (-(3.0 / 7.0) * D**2 - (3.0 * lam / 1001.0) * D**5) == pytest.approx(
            1.0, abs=1e-12)


def test_lcdm_correction_has_the_known_size():
    """D2 / D2_EdS tracks Omega_m(a)^(-1/143) (Bouchet et al. 1995) to ~2e-4."""
    for a in (0.1, 0.5, 1.0):
        om = PLANCK.Omega_m / (a**3 * C.E_of_a(a, PLANCK) ** 2)
        ratio = C.growth_factor_2(a, PLANCK) / C.growth_factor_2(a, PLANCK, "eds")
        assert ratio == pytest.approx(om ** (-1.0 / 143.0), abs=3e-4)
    assert C.growth_factor_2(1.0, PLANCK) / C.growth_factor_2(1.0, PLANCK, "eds") > 1.008


def test_growth_rate_2_and_slope_are_derivatives():
    """Against central differences (measured 6e-9 and 4e-9)."""
    h = 1e-4
    for a in (0.1, 0.3, 1.0):
        lo, hi = a * (1 - h), a * (1 + h)
        fd = (np.log(abs(C.growth_factor_2(hi, PLANCK))) - np.log(abs(C.growth_factor_2(lo, PLANCK)))
              ) / (np.log(hi) - np.log(lo))
        assert fd / C.growth_rate_2(a, PLANCK) == pytest.approx(1.0, abs=1e-6)
        slope = C.growth2_and_slope(a, PLANCK)[1]
        fd = (C.growth2_and_slope(hi, PLANCK)[0] - C.growth2_and_slope(lo, PLANCK)[0]) / (
            C.growth_factor_a(hi, PLANCK) - C.growth_factor_a(lo, PLANCK))
        assert fd / slope == pytest.approx(1.0, abs=1e-6)


def test_weights_are_normalisation_invariant():
    """(cD, c^2 E) gives the same weights as (D, E): why normalised D is safe."""
    ag = a_grid(0.1, 1.0, 40)
    t = bullfrog_table(ag, PLANCK)
    E, Ep = np.array([C.growth2_and_slope(a, PLANCK) for a in ag]).T
    c = 3.7
    for k in range(40):
        alpha = _bullfrog_weights(c * t.D_steps[k], c * t.D_steps[k + 1],
                                  (c * c * E[k], c * Ep[k], c * Ep[k + 1]))[0]
        assert alpha == pytest.approx(t.alphas[k], abs=1e-13)


@pytest.mark.parametrize("K", [1200, 4800])
def test_kick_weight_converges_to_the_lcdm_equation_of_motion(K):
    """(1 - alpha) D_mid / dD -> (3/2) Omega_m(a) / f(a)^2 at first order in the step.

    The D-time equation of motion's coupling. Measured error x K = 1.72-1.73 at
    both K; the EdS weights plateau at 12% instead and must fail this.
    """
    ag = a_grid(0.1, 1.0, K)
    am = np.sqrt(ag[:-1] * ag[1:])
    lim = np.array([1.5 * PLANCK.Omega_m / (x**3 * C.E_of_a(x, PLANCK) ** 2)
                    / C.growth_rate_a(x, PLANCK) ** 2 for x in am])

    def err(growth2):
        t = bullfrog_table(ag, PLANCK, growth2=growth2)
        return np.max(np.abs((1 - t.alphas) * t.D_mid / t.dD / lim - 1))

    assert err("lcdm") < 2.0 / K
    assert err("eds") > 0.1


def test_refusals():
    with pytest.raises(ValueError, match="growth2='eds'"):
        bullfrog_table(a_grid(0.1, 1.0, 4), PLANCK, D_of_a=lambda a: a)
    with pytest.raises(ValueError):
        bullfrog_table(a_grid(0.1, 1.0, 4), PLANCK, growth2="EdS")
    with pytest.raises(ValueError):
        C.growth_factor_2(0.5, PLANCK, model="bouchet")
    with pytest.raises(ValueError, match="outside"):
        C.growth_factor_2(3.0, PLANCK)
