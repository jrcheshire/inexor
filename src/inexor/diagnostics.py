"""Host-side diagnostics: P(k) / r(k) estimators, reversibility check,
overflow monitor (architecture.md Module layout; mbody fields/diagnostics
binning conventions).

BINNING CONVENTION (deliberate change vs the frozen _m0_common estimator):
fundamental-width spherical shells anchored at kmin = 0.5 k_f (the first bin
straddles the fundamental, k = 0 excluded), UNWEIGHTED histogram of the raw
rfftn half-grid modes, bin centers at edge midpoints -- exactly mbody
fields.power_spectrum, so in-package numbers are directly comparable to the
mbody parity reference. The parity harness additionally measures every code
with one neutral estimator (scripts/_m1_common.py); this module is for
in-package use. Interlacing is deliberately deferred (deconvolve_cic suffices
at the low-k scales M1 gates on; recorded future option).
"""

import numpy as np


def _k_grid(n_mesh, box_size):
    """(k_1d, kz_1d, k_mag) on the rfftn half-grid, float64 h/Mpc (mbody port)."""
    N, L = n_mesh, box_size
    d = L / N
    k_1d = 2.0 * np.pi * np.fft.fftfreq(N, d=d)
    kz_1d = 2.0 * np.pi * np.fft.rfftfreq(N, d=d)
    k_mag = np.sqrt(
        k_1d[:, None, None] ** 2 + k_1d[None, :, None] ** 2 + kz_1d[None, None, :] ** 2
    )
    return k_1d, kz_1d, k_mag


def cic_window(n_mesh, box_size):
    """CIC mass-assignment window W(k) on the rfftn half-grid (float64).

    W(k) = prod_i sinc^2(k_i / (2 k_nyq)); a particle-painted P(k) is
    suppressed by W^2 (negligible at low k, ~50% near Nyquist). Divide a
    measured particle power by W^2 to deconvolve; grid fields carry no window.
    """
    k_1d, kz_1d, _ = _k_grid(n_mesh, box_size)
    knyq = np.pi * n_mesh / box_size
    wx = np.sinc(k_1d / (2.0 * knyq)) ** 2
    wz = np.sinc(kz_1d / (2.0 * knyq)) ** 2
    return wx[:, None, None] * wx[None, :, None] * wz[None, None, :]


def _bin_edges(n_mesh, box_size, dk=None, kmin=None, kmax=None):
    kf = 2.0 * np.pi / box_size
    if dk is None:
        dk = kf
    if kmin is None:
        kmin = 0.5 * kf  # first bin straddles the fundamental; excludes k=0
    if kmax is None:
        kmax = np.pi * n_mesh / box_size
    return np.arange(kmin, kmax + dk, dk)


def pk_estimator(delta, box_size, dk=None, kmin=None, kmax=None, deconvolve_cic=False):
    """Binned auto P(k) of a real mesh field, P = (L^3/N^6) |delta_k|^2.

    mbody fields.power_spectrum binning (see module docstring). Returns
    (k_centers, P, n_modes) float64 numpy arrays.
    """
    delta = np.asarray(delta, dtype=np.float64)
    N, L = delta.shape[0], box_size
    pm = (np.abs(np.fft.rfftn(delta)) ** 2 * (L**3 / N**6)).ravel()
    _, _, k_mag = _k_grid(N, L)
    if deconvolve_cic:
        pm = pm / (cic_window(N, L) ** 2).ravel()
    edges = _bin_edges(N, L, dk, kmin, kmax)
    km = k_mag.ravel()
    sum_p, _ = np.histogram(km, bins=edges, weights=pm)
    counts, _ = np.histogram(km, bins=edges)
    centers = 0.5 * (edges[1:] + edges[:-1])
    good = counts > 0
    return centers[good], sum_p[good] / counts[good], counts[good]


def cross_r(delta_a, delta_b, box_size, dk=None, kmin=None, kmax=None):
    """Cross-correlation coefficient r(k) = P_ab / sqrt(P_aa P_bb), same bins
    as pk_estimator. r == 1 for fields differing only by a k-independent
    amplitude -- the primary parity metric ("do the evolved phases track?").
    Returns (k_centers, r, n_modes) float64 (mbody diagnostics port).
    """
    a = np.asarray(delta_a, dtype=np.float64)
    b = np.asarray(delta_b, dtype=np.float64)
    N, L = a.shape[0], box_size
    ak = np.fft.rfftn(a)
    bk = np.fft.rfftn(b)
    paa = (np.abs(ak) ** 2).ravel()
    pbb = (np.abs(bk) ** 2).ravel()
    pab = np.real(ak * np.conj(bk)).ravel()
    _, _, k_mag = _k_grid(N, L)
    km = k_mag.ravel()
    edges = _bin_edges(N, L, dk, kmin, kmax)
    saa, _ = np.histogram(km, bins=edges, weights=paa)
    sbb, _ = np.histogram(km, bins=edges, weights=pbb)
    sab, _ = np.histogram(km, bins=edges, weights=pab)
    counts, _ = np.histogram(km, bins=edges)
    centers = 0.5 * (edges[1:] + edges[:-1])
    good = counts > 0
    denom = np.sqrt(saa[good] * sbb[good])
    return centers[good], sab[good] / np.where(denom > 0, denom, 1.0), counts[good]


def min_image_rms(x_a, x_b, box_size):
    """RMS per-component displacement between two position sets, minimum-image."""
    d = np.asarray(x_a, dtype=np.float64) - np.asarray(x_b, dtype=np.float64)
    d = d - box_size * np.round(d / box_size)
    return float(np.sqrt(np.mean(d**2)))


def reversibility_check(state_a, state_b):
    """Exact integer equality of two (x, w) states -- the tier-0 primitive.

    Returns (ok, n_diff). EXACT equality, never a tolerance (house rule).
    """
    xa, wa = state_a
    xb, wb = state_b
    xa, wa = np.asarray(xa), np.asarray(wa)
    xb, wb = np.asarray(xb), np.asarray(wb)
    n_diff = int(np.count_nonzero(xa != xb) + np.count_nonzero(wa != wb))
    return n_diff == 0, n_diff


def overflow_report(max_w_per_step, warn_abs=None):
    """D-007 monitor: per-step max|w| trace -> headroom summary + loud warning.

    max_w_per_step: sequence of per-step max|w| (ints). Prints a WARNING line
    for every step above warn_abs (default 0.9 * 32767) -- monitor, NEVER
    clamp. Returns dict(max_w=..., n_warn=..., headroom_bits=...).
    """
    if warn_abs is None:
        warn_abs = int(0.9 * 32767)
    mw = np.asarray(max_w_per_step, dtype=np.int64)
    hot = np.nonzero(mw > warn_abs)[0]
    for k in hot:
        print(f"WARNING: |w| = {mw[k]} > {warn_abs} at step {k} -- int16 wrap imminent "
              "(wrong physics, never wrong gradients; D-007)")
    head = float(np.log2(32767.0 / max(int(mw.max()), 1)))
    return dict(max_w=int(mw.max()), n_warn=int(len(hot)), headroom_bits=head)
