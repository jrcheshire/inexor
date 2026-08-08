"""integrate.py: coefficient oracles, linear-mode growth pins, float steppers.

Trimmed 2026-08-08 with the v1 retirement. What went with it: the ladder
composition guard, the integer micro-replays (BullFrog and KDK, 100 synthetic
steps each), the PM exact-bit replay and its STE twin bit-match, the tier-0
reversibility tests and the wrap-adversarial arm, and every test of
`evolve`/`evolve_float`/`replay_roundtrip`/`simulate`. All of them asserted
properties of the reversible integer trajectory, which v2 does not have.

What remains is the coefficient layer, which is pure cosmology arithmetic and
carries forward unchanged: the EdS closed-form oracle (the roadmap M1 gate),
the FastPM small-step limit, the v/p conversion, and the linear-mode growth
pins that fix each integrator's defining property.

Growth-pin methodology (mbody `_grow_linear_mode`): a linear mode's geometric
force equals its displacement -- both track D -- so the COEFFICIENT TABLES
alone determine the growth. It is a scalar recurrence with no PM or CIC
resolution effects; those belong to the deficit matrix, not to unit tests.
"""

import jax.numpy as jnp
import pytest

from inexor.config import PLANCK
from inexor.cosmology import growth_factor_a
from inexor.integrate import (
    _bullfrog_weights,
    a_grid,
    bullfrog_float_coeffs,
    bullfrog_table,
    drift_factor,
    fastpm_drift_factor,
    fastpm_kick_factor,
    float_step_bullfrog,
    kdk_table,
    kick_factor,
    p_to_v,
    v_to_p,
)

# ----------------------------------------------------------------- oracles


def test_bullfrog_eds_closed_form():
    """M0 self-check 1 / the roadmap M1 GATE: < 1e-12 vs the published EdS
    closed form (mbody test_integrate.py:472 oracle)."""
    err = 0.0
    for n in [1.0, 2.0, 3.5, 7.0, 20.0]:
        D0, dD = n * 0.05, 0.05
        alpha, beta, _, _ = _bullfrog_weights(D0, D0 + dD)
        m = D0 / dD
        alpha_ref = (4 * m * (4 * m + 1) - 5) / (4 * m * (4 * m + 7) + 7)
        beta_ref = (24 * m + 12) / (4 * m * (4 * m + 7) + 7)
        err = max(err, abs(alpha - alpha_ref), abs(beta - beta_ref))
    assert err < 1e-12


def test_fastpm_reduces_to_exact_small_step():
    """mbody pattern: over a tiny interval the growth-corrected FastPM kernels
    converge to the exact background integrals."""
    a0, a1 = 0.5, 0.5005
    a_c = 0.5 * (a0 + a1)
    assert fastpm_kick_factor(a0, a1, a_c, PLANCK) == pytest.approx(
        kick_factor(a0, a1, PLANCK), rel=1e-5
    )
    assert fastpm_drift_factor(a0, a1, a_c, PLANCK) == pytest.approx(
        drift_factor(a0, a1, PLANCK), rel=1e-5
    )


def test_v_p_conversion_round_trip():
    v = jnp.asarray([[1.0, -2.0, 3.0]])
    a = 0.3
    assert jnp.allclose(p_to_v(v_to_p(v, a, PLANCK), a, PLANCK), v, rtol=1e-6)


# ------------------------------------------------- linear-mode growth pins


def _grow_linear_mode_kdk(integrator, K, a_i=0.1, a_f=1.0):
    from inexor.cosmology import E_of_a, growth_rate_a

    a = a_grid(a_i, a_f, K, "log")
    Di = growth_factor_a(a_i, PLANCK)
    x = Di
    p = a_i**2 * E_of_a(a_i, PLANCK) * Di * growth_rate_a(a_i, PLANCK)
    for k1, dr, k2 in kdk_table(a, PLANCK, integrator):
        p = p + k1 * x
        x = x + dr * p
        p = p + k2 * x
    return x


def _grow_linear_mode_bullfrog(K, a_i=0.1, a_f=1.0):
    t = bullfrog_table(a_grid(a_i, a_f, K, "log"), PLANCK)
    x, v = t.D_steps[0], 1.0  # x = D Psi with Psi = 1; v_D = Psi = 1
    for dD_half, alpha, bcoef in bullfrog_float_coeffs(t):
        x = x + dD_half * v
        v = alpha * v + bcoef * x
        x = x + dD_half * v
    return x


def test_fastpm_linear_mode_exact_growth():
    """FastPM's defining property: a linear mode grows as D(a) exactly at ANY
    step count (pins the previously-unexercised fastpm table)."""
    D_f = growth_factor_a(1.0, PLANCK)
    for K in (2, 4, 8):
        assert abs(_grow_linear_mode_kdk("fastpm", K) / D_f - 1.0) < 1e-4


def test_bullfrog_linear_mode_exact_growth():
    """BullFrog's Zel'dovich consistency: exact linear growth per step."""
    D_f = growth_factor_a(1.0, PLANCK)
    for K in (2, 4, 8):
        assert abs(_grow_linear_mode_bullfrog(K) / D_f - 1.0) < 1e-6


def test_exact_kdk_deficit_and_convergence():
    """The exact-KDK fallback has a real low-K growth deficit, converging
    toward D as K rises. Measured (this toy, a 0.1 -> 1.0): log spacing
    0.61% at K=2 -> 4.8e-6 at K=32; mbody's documented ~2%+ K=2 number is
    its LINEAR-spacing default (7.7% here) -- spacing matters more than K.
    """
    D_f = growth_factor_a(1.0, PLANCK)
    err = {K: abs(_grow_linear_mode_kdk("exact", K) / D_f - 1.0) for K in (2, 8, 32)}
    assert err[2] > 3e-3
    assert err[2] > err[8] > err[32]
    assert err[32] < 1e-4


# --------------------------------------------------------- float steppers


@pytest.fixture
def _x64():
    """Enable x64 for one test, then restore (test_bispectrum.py pattern; x64 is
    process-global in jax 0.10.2 and the library never sets it -- callers do)."""
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def test_float_step_bullfrog_matches_the_hand_recurrence(_x64):
    """The float stepper is what the v2 probes drive with a two-level force, so
    it needs coverage that does not route through a deleted driver.

    Linear mode: the geometric force equals the displacement, so an identity
    force_fn makes float_step_bullfrog's DKD reproduce the scalar recurrence the
    growth pin above uses. box_size is large enough that the periodic mod is
    inert -- what is under test is the coefficient application, not the wrap.
    """
    K, L_INERT = 8, 1.0e9
    t = bullfrog_table(a_grid(0.1, 1.0, K, "log"), PLANCK)
    x = jnp.asarray([[t.D_steps[0]]], dtype=jnp.float64)
    v = jnp.asarray([[1.0]], dtype=jnp.float64)
    for c in bullfrog_float_coeffs(t):
        x, v = float_step_bullfrog(x, v, tuple(c), lambda xp: xp, L_INERT)
    assert x.dtype == jnp.float64, "x64 fixture did not take; the tolerance below is f64-grade"
    assert float(x[0, 0]) == pytest.approx(_grow_linear_mode_bullfrog(K), rel=1e-12)
