"""diagnostics.py: P(k) binning on a hand-computed grid, Parseval, CIC window limits, r(k)
identities, minimum-image displacement."""

import numpy as np
import pytest

from inexor.diagnostics import (
    cic_window,
    cross_r,
    min_image_rms,
    pk_estimator,
)


def test_single_mode_power_hand_computed():
    # delta = A cos(k1 x): P concentrates in the fundamental bin with
    # |delta_k|^2 = (A N^3 / 2)^2 at +-k1 (one appears on the half-grid).
    N, L, A = 16, 100.0, 0.01
    x = np.arange(N) * (L / N)
    delta = A * np.cos(2.0 * np.pi * x / L)[:, None, None] * np.ones((N, N, N))
    kc, pk, nm = pk_estimator(delta, L)
    kf = 2.0 * np.pi / L
    # first bin straddles the fundamental
    assert kc[0] == pytest.approx(kf, rel=0.5)
    expected_mode_power = (A / 2.0) ** 2 * L**3  # (V/N^6) |A N^3/2|^2
    # bin mean = mode power / n_modes_in_bin (only 2 hot modes of nm[0])
    assert pk[0] * nm[0] == pytest.approx(2.0 * expected_mode_power, rel=1e-6)
    assert np.all(pk[1:] < 1e-12 * pk[0])


def test_parseval_total_power():
    rng = np.random.default_rng(0)
    N, L = 16, 100.0
    delta = rng.normal(size=(N, N, N))
    kc, pk, nm = pk_estimator(delta, L, kmax=np.sqrt(3.0) * np.pi * N / L * 1.001)
    # sum over bins of P*n recovers (V/N^6) * sum |delta_k|^2 (half-grid raw sum)
    dk = np.fft.rfftn(delta)
    total_half = (np.abs(dk) ** 2).sum() * (L**3 / N**6)
    k0_power = np.abs(dk[0, 0, 0]) ** 2 * (L**3 / N**6)
    assert (pk * nm).sum() == pytest.approx(total_half - k0_power, rel=1e-9)


def test_cic_window_limits():
    w = cic_window(32, 200.0)
    assert w[0, 0, 0] == pytest.approx(1.0)
    # corner: all three axes at Nyquist -> (sinc(1/2)^2)^3 = (2/pi)^6
    assert w.min() == pytest.approx((2.0 / np.pi) ** 6, rel=1e-3)


def test_cross_r_identities():
    rng = np.random.default_rng(1)
    N, L = 16, 100.0
    a = rng.normal(size=(N, N, N))
    _, r_same, _ = cross_r(a, 3.0 * a, L)  # amplitude-independent
    assert np.allclose(r_same, 1.0, atol=1e-12)
    b = rng.normal(size=(N, N, N))
    _, r_indep, _ = cross_r(a, b, L)
    assert np.max(np.abs(r_indep)) < 0.5  # uncorrelated fields: r ~ 0(1/sqrt(nm))


def test_min_image_rms():
    L = 100.0
    xa = np.array([[1.0, 1.0, 1.0]])
    xb = np.array([[99.0, 1.0, 1.0]])  # min-image distance 2, not 98
    assert min_image_rms(xa, xb, L) == pytest.approx(np.sqrt(4.0 / 3.0))


