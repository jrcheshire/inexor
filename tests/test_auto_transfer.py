"""The AUTO transfer A(k) = sqrt(P_t/P_m) and the statistic built on it.

WHY THIS EXISTS. The G3 gate divided its bispectrum ratio by the CROSS transfer
T(k) = P_tm/P_mm to make it window-invariant. Identically T = r * A, so that
division is also a division by the correlation r, and it diverges wherever the
arms decorrelate even when no power has been lost. The Stage 5 pilot measured
the consequence: `rho` tracked 1/r^2 - 1 across sixteen cells within 10-20% and
reached 3.6e4 where r crossed zero, so it reported the decorrelation already in
the eligibility map instead of the tiling error it was built for
(runs/v2/g3_stage5_record.md sec. 8).

WHAT THE SUITE HAS TO ESTABLISH, and the order matters:

  1. the degenerate limit reads zero                       (nothing else counts
     if it does not)
  2. T = r * A identically                                 (the conflation is
     real, not rhetorical)
  3. window-invariance SURVIVES                            (the property rho
     was built for is not lost)
  4. the divergence is GONE on a null rho fails            (the actual fix)
  5. it is not INERT                                       (a statistic that
     reads zero on everything would pass 1-4)

Point 5 is the one that makes points 1-4 mean anything: window-invariance is
easy to buy by making a statistic blind, and this gate has already been burned
twice by controls that could not exhibit the behaviour under test (the gaussian
window with no support at the equilateral, then the flat window whose ratio is
exactly seed-independent).

TOLERANCES ARE MEASURED, NOT ASSUMED. Each bound below quotes the value
actually measured on these fixtures; the bound is set an order of magnitude
above it, except where the quantity is analytically exact and the bound is
float64 round-off.
"""

import os
import sys

import jax
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

import v2_g3_floors as fl  # noqa: E402

N_MESH = 64
L_BOX = 128.0
KF = 2.0 * np.pi / L_BOX
LONG_MULTS = (1, 2, 3, 4)
K_SHORT_MULT = 12

# float64 round-off on quantities that are analytically exact here
EXACT = 1e-13


@pytest.fixture(autouse=True)
def _x64():
    """Enable x64 for this module only, then restore (test_bispectrum.py pattern)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _field(seed, index=-1.0, quad=0.35):
    """Gaussian field with P(k) ~ k^index plus a quadratic term.

    The quadratic term is what gives the fixture a genuine, nonzero bispectrum.
    Without it B == 0 up to noise and every ratio below is 0/0 -- a fixture that
    cannot exhibit the behaviour under test.
    """
    rng = np.random.default_rng(seed)
    wk = np.fft.rfftn(rng.standard_normal((N_MESH, N_MESH, N_MESH)))
    k1 = 2.0 * np.pi * np.fft.fftfreq(N_MESH, d=L_BOX / N_MESH)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N_MESH, d=L_BOX / N_MESH)
    kmag = np.sqrt(k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + kz[None, None, :] ** 2)
    amp = np.zeros_like(kmag)
    amp[kmag > 0] = kmag[kmag > 0] ** (index / 2.0)
    g = np.fft.irfftn(wk * amp, s=(N_MESH,) * 3, axes=(0, 1, 2))
    g /= g.std()
    return g + quad * (g**2 - (g**2).mean())


def _tris():
    return fl._triangles(KF, LONG_MULTS, K_SHORT_MULT)


def _centers(tris):
    return sorted({float(k) for tri in tris for k in tri})


# ---------------------------------------------------------------------------
# 1. the degenerate limit, first
# ---------------------------------------------------------------------------


def test_identical_arms_read_zero():
    """A == 1 and rho_auto == 0 when the two arms are the same field.

    Measured: all four quantities exactly 0.0. The bound is 1e-13 rather than
    exact equality only because the bispectrum runs through JAX, whose CPU
    reductions are not guaranteed bit-reproducible on macOS-arm64.
    """
    tris, _ = _tris()
    d = _field(11)
    t, r, a = fl.shell_transfer(d, d, L_BOX, _centers(tris), KF)
    s = fl.stats(d, d, L_BOX, tris)

    assert np.abs(a - 1.0).max() < EXACT
    assert np.abs(t - 1.0).max() < EXACT
    assert np.abs(r - 1.0).max() < EXACT
    assert np.abs(s["rho_auto"]).max() < EXACT
    assert np.abs(s["A_prod"] - 1.0).max() < EXACT


# ---------------------------------------------------------------------------
# 2. the identity that makes the conflation a fact rather than an argument
# ---------------------------------------------------------------------------


def test_cross_transfer_is_correlation_times_auto_transfer():
    """T(k) = r(k) * A(k) exactly. Measured max deviation 2.2e-16.

    This is the whole diagnosis in one line: anything that divides by T divides
    by r, so it cannot help but diverge as the arms decorrelate.
    """
    tris, _ = _tris()
    d_m = _field(11)
    d_t = 1.05 * d_m + 0.3 * _field(12)  # both an amplitude change and a phase change
    t, r, a = fl.shell_transfer(d_t, d_m, L_BOX, _centers(tris), KF)

    assert np.abs(t - r * a).max() < EXACT
    # the fixture must actually exercise both factors, or the identity is vacuous
    assert np.abs(r - 1.0).max() > 0.05
    assert np.abs(a - 1.0).max() > 0.05


# ---------------------------------------------------------------------------
# 3. window-invariance survives
# ---------------------------------------------------------------------------


def test_scale_flat_window_is_divided_out_exactly():
    """delta_t = c * delta_m: A == c, rho_auto == 0, and R_Q == 1/c - 1.

    Analytic throughout, so the bounds are round-off. The last assertion is the
    contrast that motivated a window-divided statistic in the first place: R_Q
    does NOT cancel a deterministic window, it responds as 1/T.

    A == c (not c^2) is also the guard on the sqrt: P_t = c^2 P_m, so an
    implementation that forgot the square root would return 1.1025 here.
    """
    tris, _ = _tris()
    c = 1.05
    d_m = _field(11)
    t, _, a = fl.shell_transfer(c * d_m, d_m, L_BOX, _centers(tris), KF)
    s = fl.stats(c * d_m, d_m, L_BOX, tris)

    assert np.abs(a - c).max() < EXACT
    assert np.abs(a - c**2).max() > 0.05  # the sqrt guard, stated explicitly
    assert np.abs(t - c).max() < EXACT
    assert np.abs(s["rho_auto"]).max() < EXACT
    assert np.abs(s["R_B"] - (c**3 - 1.0)).max() < EXACT
    # R_Q responds to a pure window and must not be silently "fixed"
    assert np.abs(s["R_Q"] - (1.0 / c - 1.0)).max() < EXACT
    assert np.abs(s["R_Q"]).max() > 0.04


def test_scale_dependent_window_matches_the_cross_transfer():
    """Under a DETERMINISTIC window A and T agree, so rho_auto inherits rho's
    behaviour exactly where rho was correct.

    Measured: max|A - T| = 2.0e-6, and rho_auto = 4.386e-3 against rho =
    4.385e-3. That residual is not estimator error -- it is the within-shell
    variation of the window, the floor v2_g3_floors' module docstring already
    documents for W. Both statistics inherit it identically.
    """
    tris, _ = _tris()
    d_m = _field(11)
    d_t, _ = fl.apply_window(d_m, L_BOX, window="gauss")
    t, _, a = fl.shell_transfer(d_t, d_m, L_BOX, _centers(tris), KF)
    s = fl.stats(d_t, d_m, L_BOX, tris)

    assert np.abs(a - t).max() < 1e-4
    assert np.abs(s["rho_auto"]).max() < 1e-2
    assert np.abs(s["rho_auto"] - s["rho"]).max() < 1e-4


# ---------------------------------------------------------------------------
# 4. THE FIX: a null that rho fails and rho_auto passes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shift", [5, 13, 21])
def test_translation_null_diverges_for_rho_and_not_for_rho_auto(shift):
    """An integer-cell translation is a NULL, and rho reports up to 4.3e4 on it.

    A rigid shift by a whole number of cells is a relabelling of grid points. It
    preserves |delta_k| mode by mode, so P_t = P_m exactly; and the bispectrum
    is translation-invariant on closed triangles (the phases pick up
    exp(i(k1+k2+k3).s) = 1), so B_t = B_m exactly. The correct answer is
    therefore identically zero for any statistic claiming to measure a tiling
    error -- and a translated monolithic field is already one of the instrument
    nulls this gate uses (v2_g3_floors section A).

    r(k), however, falls to 0.07 / -0.35 / -0.25 at these shifts, so T -> 0 and
    the cross-divided rho blows up: measured 3.45e3, 1.02e4 and 4.34e4. This is
    the defect, reproduced on a case where the truth is known exactly.

    Measured for rho_auto: 8.9e-16, 1.1e-15, 6.7e-16.
    """
    tris, _ = _tris()
    d_m = _field(11)
    d_t = np.roll(d_m, (shift, shift // 2, -shift), axis=(0, 1, 2))
    _, r, a = fl.shell_transfer(d_t, d_m, L_BOX, _centers(tris), KF)
    s = fl.stats(d_t, d_m, L_BOX, tris)

    # the null is a null: power and bispectrum both exactly preserved
    assert np.abs(a - 1.0).max() < EXACT
    assert np.abs(s["R_B"]).max() < EXACT
    # the new statistic reads it as one
    assert np.abs(s["rho_auto"]).max() < EXACT
    # the fixture genuinely decorrelates, or it would not discriminate
    assert np.abs(r).min() < 0.5
    # and the old statistic does not: this is what the change is for
    assert np.abs(s["rho"]).max() > 1e3


def test_window_and_decorrelation_together():
    """The realistic case: a real tiled arm carries both at once.

    Measured: A recovers the 1.05 window to 2.2e-16 THROUGH the decorrelation,
    rho_auto = 6.7e-16, rho = 1.02e4.
    """
    tris, _ = _tris()
    c = 1.05
    d_m = _field(11)
    d_t = np.roll(c * d_m, (13, 6, -13), axis=(0, 1, 2))
    _, _, a = fl.shell_transfer(d_t, d_m, L_BOX, _centers(tris), KF)
    s = fl.stats(d_t, d_m, L_BOX, tris)

    assert np.abs(a - c).max() < EXACT
    assert np.abs(s["rho_auto"]).max() < EXACT
    assert np.abs(s["rho"]).max() > 1e3


# ---------------------------------------------------------------------------
# 5. and it is NOT inert -- the assertion that makes the rest mean something
# ---------------------------------------------------------------------------


def test_responds_to_an_injected_long_short_coupling():
    """A genuine squeezed coupling must still register, monotonically.

    delta_t = delta_m (1 + g delta_L/rms(delta_L)) is a real long-short
    coupling, not a window: it changes the squeezed bispectrum through a phase
    correlation that no amplitude division can remove.

    Measured max|rho_auto| over the squeezed triangles along the ladder:
    0.000, 1.076, 2.126, 4.094, 7.257 for g = 0, 0.05, 0.1, 0.2, 0.4.
    """
    tris, names = _tris()
    sq = [i for i, nm in enumerate(names) if nm != "equi"]
    d_m = _field(11)

    got = []
    for g in (0.0, 0.05, 0.1, 0.2, 0.4):
        s = fl.stats(fl.inject_coupling(d_m, L_BOX, 2 * KF, KF, g), d_m, L_BOX, tris)
        got.append(float(np.abs(s["rho_auto"][sq]).max()))

    assert got[0] < EXACT                      # g = 0 is a null and reads as one
    assert all(b > a for a, b in zip(got, got[1:]))   # strictly monotone in g
    assert got[1] > 0.5                        # visible at the smallest nonzero g
    assert got[-1] > 5.0                       # and large where the coupling is large
