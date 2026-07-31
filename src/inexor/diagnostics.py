"""Host-side diagnostics: P(k) / r(k) estimators, the Scoccimarro bispectrum,
reversibility check, overflow monitor (architecture.md Module layout; mbody
fields/diagnostics binning conventions).

BINNING CONVENTION (deliberate change vs the frozen _m0_common estimator):
fundamental-width spherical shells anchored at kmin = 0.5 k_f (the first bin
straddles the fundamental, k = 0 excluded), UNWEIGHTED histogram of the raw
rfftn half-grid modes, bin centers at edge midpoints -- exactly mbody
fields.power_spectrum, so in-package numbers are directly comparable to the
mbody parity reference. The parity harness additionally measures every code
with one neutral estimator (scripts/_m1_common.py); this module is for
in-package use. Interlacing is deliberately deferred (deconvolve_cic suffices
at the low-k scales M1 gates on; recorded future option).

BACKEND SPLIT (bispectrum path only). The k-space coefficient tables -- _k_grid,
_bin_edges, cic_window, _shell_mask -- stay numpy float64: they ARE the binning
convention, and keeping them in one place is what stops the bispectrum forking
it (CLAUDE.md, "host-side coefficient tables are numpy float64"). Only the
field-side work (FFTs, band fields, triple-product reductions) is JAX, so the
estimator is shared with ichnaea rather than ported twice. pk_estimator,
cross_r and everything below them remain pure numpy.

x64 IS THE CALLER'S JOB. Library code never toggles jax_enable_x64 (jht/sfbfs
convention), and an f64 bispectrum silently computed in f32 does not fail -- it
just relocates every tolerance in tests/test_bispectrum.py to a floor about five
orders of magnitude higher. bispectrum() therefore hard-fails on a non-f64 band
field rather than proceeding.
"""

import numpy as np


def _k_grid(n_mesh, box_size):
    """(k_1d, kz_1d, k_mag) on the rfftn half-grid, float64 h/Mpc (mbody port)."""
    N, L = n_mesh, box_size
    d = L / N
    k_1d = 2.0 * np.pi * np.fft.fftfreq(N, d=d)
    kz_1d = 2.0 * np.pi * np.fft.rfftfreq(N, d=d)
    k_mag = np.sqrt(k_1d[:, None, None] ** 2 + k_1d[None, :, None] ** 2 + kz_1d[None, None, :] ** 2)
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


def _shell_mask(k_mag, k_lo, k_hi):
    """Boolean mask for the |k| shell [k_lo, k_hi) on the rfftn half-grid.

    HALF-OPEN, so adjacent shells of width dk tile without double-counting a
    mode sitting exactly on a bin edge (mbody fields._shell_mask:138). This is
    the convention _shell_index and the bispectrum band fields both realize;
    see _shell_index for why pk_estimator no longer uses np.histogram.
    """
    return (k_mag >= k_lo) & (k_mag < k_hi)


def _shell_index(k_mag, edges):
    """Per-mode bin index for `edges`, -1 outside. Uniformly half-open.

    np.histogram closes its LAST bin ([lo, hi] rather than [lo, hi)), so a mode
    landing exactly on the top edge was counted by pk_estimator but excluded by
    a _shell_mask-built band field. That is a one-bin disagreement at Nyquist
    between two estimators in this same module, which is exactly the kind of
    convention fork the module docstring exists to prevent -- so the binning is
    now digitize-based for every bin (the scripts/_m1_common.py convention).

    Kept as index arithmetic rather than a stack of boolean masks because a
    mask per bin costs n_bins * N^2 * (N/2+1) bytes, which is the wrong price
    for a convention fix.
    """
    idx = np.digitize(k_mag.ravel(), edges) - 1
    idx[idx >= len(edges) - 1] = -1
    return idx


def _binned_sums(idx, n_bins, weights=None):
    """np.bincount over _shell_index output, ignoring the -1 (outside) modes."""
    keep = idx >= 0
    w = None if weights is None else np.asarray(weights).ravel()[keep]
    return np.bincount(idx[keep], weights=w, minlength=n_bins)[:n_bins]


def pk_estimator(delta, box_size, dk=None, kmin=None, kmax=None, deconvolve_cic=False):
    """Binned auto P(k) of a real mesh field, P = (L^3/N^6) |delta_k|^2.

    mbody fields.power_spectrum binning (see module docstring). Returns
    (k_centers, P, n_modes) float64 numpy arrays. Binning is half-open in every
    bin including the last (see _shell_index).
    """
    delta = np.asarray(delta, dtype=np.float64)
    N, L = delta.shape[0], box_size
    pm = (np.abs(np.fft.rfftn(delta)) ** 2 * (L**3 / N**6)).ravel()
    _, _, k_mag = _k_grid(N, L)
    if deconvolve_cic:
        pm = pm / (cic_window(N, L) ** 2).ravel()
    edges = _bin_edges(N, L, dk, kmin, kmax)
    nb = len(edges) - 1
    idx = _shell_index(k_mag, edges)
    sum_p = _binned_sums(idx, nb, pm)
    counts = _binned_sums(idx, nb)
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
    edges = _bin_edges(N, L, dk, kmin, kmax)
    nb = len(edges) - 1
    idx = _shell_index(k_mag, edges)
    saa = _binned_sums(idx, nb, paa)
    sbb = _binned_sums(idx, nb, pbb)
    sab = _binned_sums(idx, nb, pab)
    counts = _binned_sums(idx, nb)
    centers = 0.5 * (edges[1:] + edges[:-1])
    good = counts > 0
    denom = np.sqrt(saa[good] * sbb[good])
    return centers[good], sab[good] / np.where(denom > 0, denom, 1.0), counts[good]


# ===========================================================================
# Scoccimarro bispectrum (mbody fields.py:147,174 port; JAX field path)
# ===========================================================================


def _shell_spec(triangles, box_size, dk=None):
    """Host-side prep: unique (center, width) shells + per-triangle shell indices.

    Returns (centers, widths, tri_idx) with tri_idx an (n_tri, 3) int array.

    Shells are keyed by the (center, width) PAIR, not by the center alone, so a
    per-shell dk is expressible: the gate wants dk_long = k_f to keep resolution
    at the squeezed threshold and dk_short = 4 k_f for n_tri, and the same
    center may legitimately carry different widths in different legs. Indexing
    by position also removes mbody's float-keyed band-field dicts, which forced
    callers to pass bit-identical float objects (its finding N3).

    dk is None (-> k_fundamental on every leg), a scalar, or a 3-sequence read
    as (dk_1, dk_2, dk_3) per triangle leg.
    """
    tris = np.atleast_2d(np.asarray(triangles, dtype=np.float64))
    if tris.shape[1] != 3:
        raise ValueError(f"triangles must be (n, 3), got {tris.shape}")
    kf = 2.0 * np.pi / box_size
    if dk is None:
        w = np.full(3, kf, dtype=np.float64)
    else:
        w = np.asarray(dk, dtype=np.float64)
        w = np.full(3, float(w)) if w.ndim == 0 else w
    if w.shape != (3,):
        raise ValueError(f"dk must be None, a scalar, or a 3-sequence; got shape {w.shape}")

    shells, tri_idx = [], np.empty(tris.shape, dtype=np.int32)
    lookup = {}
    for t in range(tris.shape[0]):
        for j in range(3):
            key = (float(tris[t, j]), float(w[j]))
            if key not in lookup:
                lookup[key] = len(shells)
                shells.append(key)
            tri_idx[t, j] = lookup[key]
    centers = np.array([s[0] for s in shells], dtype=np.float64)
    widths = np.array([s[1] for s in shells], dtype=np.float64)
    return centers, widths, tri_idx


def _theta_stack(n_mesh, box_size, centers, widths):
    """(n_shell, N, N, N//2+1) float64 shell indicators. Numpy coefficient island.

    Costs n_shell * N^2 * (N/2+1) * 8 bytes and is freed once the band fields
    exist; the band fields themselves are the resident cost (see bispectrum).
    """
    _, _, k_mag = _k_grid(n_mesh, box_size)
    return np.stack(
        [_shell_mask(k_mag, c - 0.5 * w, c + 0.5 * w).astype(np.float64)
         for c, w in zip(centers, widths)]
    )


def _band_fields(delta, theta, want_i=True):
    """Real-space band-filtered fields, the JAX half of the estimator.

        I_n(x) = irfftn(delta_k * Theta_n),   J_n(x) = irfftn(Theta_n)

    I carries the data; J is the same filter with the field replaced by ones,
    and its triple product counts closeable mode-triplets. J is INDEPENDENT of
    delta, so a caller comparing two arms (tiled vs monolithic) should compute
    it once and pass it to both -- that halves the resident cost, which is
    2 * n_shell * N^3 * 8 bytes (N=256, n_shell=7 -> 1.9 GB; N=512 -> 15 GB).
    """
    import jax.numpy as jnp

    n = theta.shape[1]
    axes = (1, 2, 3)
    if not want_i:
        return jnp.fft.irfftn(theta.astype(jnp.complex128), s=(n, n, n), axes=axes)
    delta_k = jnp.fft.rfftn(delta)[None, ...]
    return jnp.fft.irfftn(delta_k * theta, s=(n, n, n), axes=axes)


def _triple_sums(f_a, f_b, f_c, tri_idx):
    """sum_x A_a B_b C_c for each triangle, via lax.scan.

    Three separate stacks so the estimator does CROSS-bispectra: leg j of every
    triangle is drawn from stack j. Pass the same stack three times for the auto
    case. This is what the W discriminator needs -- W puts a residual field on
    the long leg and the reference field on the two short legs.

    A scan rather than a vmap on purpose: vmap would materialize one real
    (N, N, N) temporary PER TRIANGLE, so the peak would scale with the triangle
    count. The scan holds exactly one.
    """
    import jax
    import jax.numpy as jnp

    def body(carry, t):
        return carry, jnp.sum(f_a[t[0]] * f_b[t[1]] * f_c[t[2]])

    _, out = jax.lax.scan(body, None, tri_idx)
    return out


def bispectrum_core(delta, theta, tri_idx, alpha, n_mesh, j_fields=None):
    """The jit-able core. Returns (B, n_tri) as JAX f64 arrays.

    Split out from bispectrum() so the whole thing is jit-able from the start:
    everything here is traceable, while the shell bookkeeping in _shell_spec /
    _theta_stack is host-side numpy that must NOT be traced. Callers wanting
    jit should wrap this, holding theta/tri_idx/alpha/n_mesh fixed.
    """
    import jax.numpy as jnp

    legs = delta if isinstance(delta, (tuple, list)) else (delta, delta, delta)
    if len(legs) != 3:
        raise ValueError(f"delta must be one field or exactly 3 (one per leg); got {len(legs)}")
    # Distinct arrays only: the auto case must not pay 3x the band-field memory,
    # which is the resident cost of the whole estimator.
    uniq, stacks = [], []
    for f in legs:
        for k, seen in enumerate(uniq):
            if f is seen:
                stacks.append(stacks[k])
                break
        else:
            uniq.append(f)
            stacks.append(_band_fields(f, theta))
    if stacks[0].dtype != jnp.float64:
        raise TypeError(
            f"band fields are {stacks[0].dtype}, not float64 -- enable x64 in the CALLER "
            "(jax.config.update('jax_enable_x64', True)). Library code does not toggle "
            "it, and an f32 bispectrum does not fail, it just moves every tolerance."
        )
    if j_fields is None:
        j_fields = _band_fields(None, theta, want_i=False)
    s = _triple_sums(stacks[0], stacks[1], stacks[2], tri_idx)
    norm = _triple_sums(j_fields, j_fields, j_fields, tri_idx)
    n_tri = float(n_mesh) ** 6 * norm
    # Non-closing bin triples have no mode-triplets at all. mbody documents
    # n_tri = 0 for these but still divides, which in f32 round-off returns a
    # large finite garbage B rather than anything a caller would notice.
    #
    # The threshold is on n_tri, NOT on the raw J-product: a non-closing triple
    # does not give an exact zero, it gives f64 FFT round-off (measured 5.7e-14
    # in n_tri units at N=16), so `norm > 0` admits garbage. n_tri is a COUNT of
    # mode-triplets, so it is >= 1 whenever the configuration closes at all and
    # anything below 0.5 is round-off with no scale ambiguity to argue about.
    closes = n_tri >= 0.5
    b = jnp.where(closes, alpha * s / jnp.where(closes, norm, 1.0), jnp.nan)
    return b, jnp.where(closes, n_tri, 0.0)


def bispectrum(delta, box_size, triangles, dk=None, j_fields=None):
    """Binned bispectrum B(k1, k2, k3) via the Scoccimarro FFT estimator.

    For each triangle,

        B = (V^2 / N^9) * sum_x I1 I2 I3 / sum_x J1 J2 J3,   V = L^3,

    with I, J the band-filtered fields of _band_fields. The V^2/N^9 prefactor is
    exact in the same DFT convention that fixes pk_estimator's V/N^6 (the
    triangle count cancels between the data and the J normalization), so a
    correct field returns B with no free constant. mbody fields.py:174 port.

    delta : real (N, N, N) field, or a 3-tuple of them for a CROSS-bispectrum
        (leg j is drawn from field j, in the triangle's own leg order). The
        auto case shares one band-field stack rather than building three.
    triangles : sequence of (k1, k2, k3) shell centers in h/Mpc. Fully general;
        a triple whose bins cannot close returns n_tri = 0 and B = NaN.
    dk : None (k_fundamental), a scalar, or a 3-sequence for a PER-LEG width.
    j_fields : optional precomputed J stack (see _band_fields) to share across
        two arms of a comparison.

    Returns (B, n_tri) as float64 numpy arrays, one entry per triangle. n_tri is
    the mode-triplet count -- a sampling diagnostic, small counts are noisy.

    NO CIC deconvolution and NO shot-noise subtraction, matching mbody. For a
    tiled-vs-monolithic RATIO both arms paint identically and the window divides
    out, so this is the right default there; a B quoted in absolute terms off a
    particle-painted field is NOT window-corrected and must not be compared to a
    continuum template. The oracle test uses a grid field, where no window
    enters at all.
    """
    centers, widths, tri_idx = _shell_spec(triangles, box_size, dk)
    if isinstance(delta, (tuple, list)):
        delta = tuple(np.asarray(f, dtype=np.float64) for f in delta)
        n = delta[0].shape[0]
    else:
        delta = np.asarray(delta, dtype=np.float64)
        n = delta.shape[0]
    theta = _theta_stack(n, box_size, centers, widths)
    alpha = box_size**6 / float(n) ** 9
    b, n_tri = bispectrum_core(delta, theta, tri_idx, alpha, n, j_fields=j_fields)
    return np.asarray(b, dtype=np.float64), np.asarray(n_tri, dtype=np.float64)


def band_power(delta, box_size, centers, dk=None):
    """P(k) on the SAME shells the bispectrum uses -- the R_Q denominator.

    The gate statistic is the reduced bispectrum ratio
    Q = B / (P1 P2 + P2 P3 + P3 P1), so its P must come from shells that are
    bit-identically the estimator's. pk_estimator bins on a uniform edge grid
    from kmin, which in general contains neither the requested centers nor a
    per-shell dk, so reading Q's denominator off it would silently mix two
    binnings.

    Shell-averaged, unweighted over the raw rfftn half-grid, exactly as
    pk_estimator averages -- so band_power at a bin pk_estimator also resolves
    returns the same number and the same mode count (asserted in the tests away
    from Nyquist). NOTE this is the shell average of P, not the triplet-weighted
    average implicit in B's normalization; that is the standard convention for Q
    and is stated here because the two differ at the few-percent level in bins
    with steep P(k).

    centers : sequence of shell centers (h/Mpc). dk as in bispectrum, except a
    3-sequence is not meaningful here -- pass a scalar or None.

    Returns (P, n_modes) float64 numpy arrays, one entry per center.
    """
    delta = np.asarray(delta, dtype=np.float64)
    n, ell = delta.shape[0], box_size
    kf = 2.0 * np.pi / ell
    w = kf if dk is None else float(np.asarray(dk, dtype=np.float64).ravel()[0])
    pm = np.abs(np.fft.rfftn(delta)) ** 2 * (ell**3 / n**6)
    _, _, k_mag = _k_grid(n, ell)
    p_out = np.empty(len(centers), dtype=np.float64)
    n_out = np.empty(len(centers), dtype=np.float64)
    for i, c in enumerate(centers):
        m = _shell_mask(k_mag, float(c) - 0.5 * w, float(c) + 0.5 * w)
        cnt = int(m.sum())
        n_out[i] = cnt
        p_out[i] = pm[m].sum() / cnt if cnt else np.nan
    return p_out, n_out


def local_bispectrum_binned(delta_shape_n, box_size, cosmo, triangles, f_NL, z=0.0, dk=None):
    """Bin-AVERAGED tree-level local-f_NL bispectrum -- the exact oracle for
    bispectrum() above. mbody ic.py:151 port.

    bispectrum() returns the mean of B_tree over every mode-triplet in each
    (b1, b2, b3) bin, so comparing it to the continuum B_tree at the bin CENTRE
    (ic.local_bispectrum_template) carries a binning systematic: the shell mode
    density ~ k^2 pushes the effective k above centre, biasing the steep
    squeezed long side low. This evaluates the same bin average exactly, so a
    correct estimator matches it with unit calibration.

    With B_tree = 2 f_NL sum_perm g(k_a) f(k_b) f(k_c), g = M, f = P_lin / M,
    the bin average is

        2 f_NL sum_x (F1 F2 G3 + G1 F2 F3 + F1 G2 F3) / sum_x (J1 J2 J3),
        G_n = irfftn(M Theta_n), F_n = irfftn((P/M) Theta_n), J_n = irfftn(Theta_n),

    the same shell-product identity the estimator uses, so the N^6 factors
    cancel against the triangle count on both sides.

    delta_shape_n : the mesh size N the estimator will be run at. The bin
    average depends on the mesh (it is an average over that mesh's modes), so
    this is not an optional convenience argument.
    """
    import jax.numpy as jnp

    from inexor.cosmology import linear_power
    from inexor.ic import poisson_factor

    n = int(delta_shape_n)
    centers, widths, tri_idx = _shell_spec(triangles, box_size, dk)
    _, _, k_mag = _k_grid(n, box_size)
    m_k = poisson_factor(n, box_size, cosmo, z=z)
    p_lin = linear_power(k_mag.ravel(), cosmo, z=z).reshape(k_mag.shape)
    # k = 0 is excluded from every shell (kmin > 0), but poisson_factor parks a
    # placeholder 1 there, so zero the weight explicitly rather than relying on
    # the mask to hide a value that is not physical.
    f_wt = np.where(k_mag > 0, p_lin / np.where(k_mag > 0, m_k, 1.0), 0.0)

    theta = _theta_stack(n, box_size, centers, widths)
    axes, shp = (1, 2, 3), (n, n, n)
    g = jnp.fft.irfftn((m_k[None, ...] * theta).astype(jnp.complex128), s=shp, axes=axes)
    f = jnp.fft.irfftn((f_wt[None, ...] * theta).astype(jnp.complex128), s=shp, axes=axes)
    j = jnp.fft.irfftn(theta.astype(jnp.complex128), s=shp, axes=axes)

    def perm_sum(carry, t):
        a, b, c = t[0], t[1], t[2]
        num = jnp.sum(f[a] * f[b] * g[c] + g[a] * f[b] * f[c] + f[a] * g[b] * f[c])
        return carry, num / jnp.sum(j[a] * j[b] * j[c])

    import jax

    _, ratio = jax.lax.scan(perm_sum, None, tri_idx)
    return 2.0 * f_NL * np.asarray(ratio, dtype=np.float64)


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
        print(
            f"WARNING: |w| = {mw[k]} > {warn_abs} at step {k} -- int16 wrap imminent "
            "(wrong physics, never wrong gradients; D-007)"
        )
    head = float(np.log2(32767.0 / max(int(mw.max()), 1)))
    return dict(max_w=int(mw.max()), n_warn=int(len(hot)), headroom_bits=head)
