"""Host-side diagnostics: P(k), r(k), the Scoccimarro bispectrum, sub-volume response.

Binning: fundamental-width spherical shells from kmin = 0.5 k_f (k = 0 excluded), half-open
[lo, hi) in every bin, unweighted over the raw rfftn half-grid modes, centers at edge
midpoints. The k-space tables (_k_grid, _bin_edges, cic_window, _shell_mask) are numpy
float64 so every estimator shares one binning. The bispectrum's field-side work (FFTs, band
fields, triple products) is JAX; pk_estimator and cross_r are pure numpy.

x64 is the caller's job (the library never toggles jax_enable_x64); `bispectrum` raises on
non-f64 band fields, since an f32 result does not fail, it only loses ~5 orders of precision.
"""

import numpy as np


def _k_grid(n_mesh, box_size):
    """(k_1d, kz_1d, k_mag) on the rfftn half-grid, float64 h/Mpc."""
    N, L = n_mesh, box_size
    d = L / N
    k_1d = 2.0 * np.pi * np.fft.fftfreq(N, d=d)
    kz_1d = 2.0 * np.pi * np.fft.rfftfreq(N, d=d)
    k_mag = np.sqrt(k_1d[:, None, None] ** 2 + k_1d[None, :, None] ** 2 + kz_1d[None, None, :] ** 2)
    return k_1d, kz_1d, k_mag


def cic_window(n_mesh, box_size):
    """CIC mass-assignment window W(k) = prod_i sinc^2(k_i / (2 k_nyq)), float64 half-grid.

    Divide particle-painted power by W^2 to deconvolve; grid fields carry no window.
    """
    k_1d, kz_1d, _ = _k_grid(n_mesh, box_size)
    knyq = np.pi * n_mesh / box_size
    wx = np.sinc(k_1d / (2.0 * knyq)) ** 2
    wz = np.sinc(kz_1d / (2.0 * knyq)) ** 2
    return wx[:, None, None] * wx[None, :, None] * wz[None, None, :]


def tsc_window(n_mesh, box_size):
    """TSC mass-assignment window W(k) = prod_i sinc^3(k_i / (2 k_nyq)), float64 half-grid.

    The window of the engine's coarse (TSC) paint; use it, not `cic_window`, for a coarse
    `delta`. Divide particle power by W^2 to deconvolve. No interlacing, so aliasing remains:
    reliable well below Nyquist only.
    """
    k_1d, kz_1d, _ = _k_grid(n_mesh, box_size)
    knyq = np.pi * n_mesh / box_size
    wx = np.sinc(k_1d / (2.0 * knyq)) ** 3
    wz = np.sinc(kz_1d / (2.0 * knyq)) ** 3
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
    """Boolean mask for the |k| shell [k_lo, k_hi); half-open, so adjacent shells tile."""
    return (k_mag >= k_lo) & (k_mag < k_hi)


def _shell_index(k_mag, edges):
    """Per-mode bin index for `edges`, -1 outside; half-open in every bin.

    digitize rather than np.histogram, which closes its last bin and would disagree with
    `_shell_mask` at the top edge.
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

    Returns (k_centers, P, n_modes) float64 numpy arrays; empty bins are dropped.
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
    """Cross-correlation coefficient r(k) = P_ab / sqrt(P_aa P_bb), pk_estimator's bins.

    r == 1 for fields differing only by a k-independent amplitude. Returns
    (k_centers, r, n_modes) float64.
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
# Scoccimarro bispectrum (JAX field path)
# ===========================================================================


def _shell_spec(triangles, box_size, dk=None):
    """Host-side prep: unique (center, width) shells + per-triangle shell indices.

    Returns (centers, widths, tri_idx), tri_idx an (n_tri, 3) int array. Shells are keyed by
    the (center, width) pair, so one center may carry different widths on different legs.
    dk is None (k_fundamental on every leg), a scalar, or a per-leg 3-sequence.
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
    """(n_shell, N, N, N//2+1) float64 shell indicators (numpy)."""
    _, _, k_mag = _k_grid(n_mesh, box_size)
    return np.stack(
        [_shell_mask(k_mag, c - 0.5 * w, c + 0.5 * w).astype(np.float64)
         for c, w in zip(centers, widths)]
    )


def _band_fields(delta, theta, want_i=True):
    """Real-space band fields I_n = irfftn(delta_k Theta_n), or J_n = irfftn(Theta_n).

    J (want_i=False) is independent of delta and its triple product counts mode-triplets;
    compute it once and share it across compared arms. Each stack is n_shell * N^3 * 8 bytes.
    """
    import jax.numpy as jnp

    n = theta.shape[1]
    axes = (1, 2, 3)
    if not want_i:
        return jnp.fft.irfftn(theta.astype(jnp.complex128), s=(n, n, n), axes=axes)
    delta_k = jnp.fft.rfftn(delta)[None, ...]
    return jnp.fft.irfftn(delta_k * theta, s=(n, n, n), axes=axes)


def _triple_sums(f_a, f_b, f_c, tri_idx):
    """sum_x A_a B_b C_c per triangle; leg j is drawn from stack j (cross-bispectra).

    lax.scan rather than vmap so only one (N, N, N) temporary is live.
    """
    import jax
    import jax.numpy as jnp

    def body(carry, t):
        return carry, jnp.sum(f_a[t[0]] * f_b[t[1]] * f_c[t[2]])

    _, out = jax.lax.scan(body, None, tri_idx)
    return out


def bispectrum_core(delta, theta, tri_idx, alpha, n_mesh, j_fields=None):
    """Traceable core of `bispectrum`; returns (B, n_tri) as JAX f64 arrays.

    The host-side shell prep (_shell_spec, _theta_stack) stays outside; to jit, wrap this
    with theta/tri_idx/alpha/n_mesh held fixed.
    """
    import jax.numpy as jnp

    legs = delta if isinstance(delta, (tuple, list)) else (delta, delta, delta)
    if len(legs) != 3:
        raise ValueError(f"delta must be one field or exactly 3 (one per leg); got {len(legs)}")
    # one band-field stack per distinct array, so the auto case builds one
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
    # Non-closing triples give FFT round-off, not an exact zero; n_tri is a count, so < 0.5
    # means none (B = NaN).
    closes = n_tri >= 0.5
    b = jnp.where(closes, alpha * s / jnp.where(closes, norm, 1.0), jnp.nan)
    return b, jnp.where(closes, n_tri, 0.0)


def bispectrum(delta, box_size, triangles, dk=None, j_fields=None):
    """Binned bispectrum B(k1, k2, k3) via the Scoccimarro FFT estimator.

    B = (V^2 / N^9) sum_x I1 I2 I3 / sum_x J1 J2 J3 per triangle (V = L^3, I, J from
    _band_fields); no free constant in pk_estimator's DFT convention.

    delta : real (N, N, N) field, or a 3-tuple for a cross-bispectrum (leg j from field j).
    triangles : (k1, k2, k3) shell centers in h/Mpc; non-closing bins give n_tri = 0, B = NaN.
    dk : None (k_fundamental), a scalar, or a per-leg 3-sequence.
    j_fields : optional precomputed J stack to share across compared arms.

    Returns (B, n_tri) float64 numpy arrays, one per triangle; n_tri is the mode-triplet count.
    No window deconvolution and no shot-noise subtraction: fine for ratios of identically
    painted fields, not for absolute comparison of a particle-painted B to a template.
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
    """P(k) on exactly the bispectrum's shells, for the reduced bispectrum's denominator.

    Unweighted shell average as in pk_estimator (not the triplet-weighted average in B's
    normalization; they differ by a few percent where P(k) is steep). centers in h/Mpc; dk
    None or a scalar. Returns (P, n_modes) float64, one per center.
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
    """Bin-averaged tree-level local-f_NL bispectrum: the exact expectation of `bispectrum`.

    Unlike the continuum template at bin centers, this averages B_tree over the same
    mode-triplets the estimator sums, via the same shell-product identity:

        2 f_NL sum_x (F1 F2 G3 + G1 F2 F3 + F1 G2 F3) / sum_x (J1 J2 J3),
        G_n = irfftn(M Theta_n), F_n = irfftn((P/M) Theta_n), J_n = irfftn(Theta_n).

    delta_shape_n is the mesh size N the estimator runs at (the average depends on it).
    """
    import jax.numpy as jnp

    from inexor.cosmology import linear_power
    from inexor.ic import poisson_factor

    n = int(delta_shape_n)
    centers, widths, tri_idx = _shell_spec(triangles, box_size, dk)
    _, _, k_mag = _k_grid(n, box_size)
    m_k = poisson_factor(n, box_size, cosmo, z=z)
    p_lin = linear_power(k_mag.ravel(), cosmo, z=z).reshape(k_mag.shape)
    # poisson_factor holds a placeholder at k = 0; zero the weight there explicitly
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


def _subvolume_blocks(delta, n_sub, offset_cells):
    """Split a periodic mesh into n_sub^3 equal cubes, origin shifted by offset_cells.

    Returns (n_sub^3, s, s, s), s = N // n_sub, blocks raveled C-order; a partition of the
    periodic box.
    """
    n = delta.shape[0]
    s = n // n_sub
    off = int(offset_cells) % n
    d = np.roll(delta, (-off, -off, -off), axis=(0, 1, 2))
    d = d.reshape(n_sub, s, n_sub, s, n_sub, s)
    return np.ascontiguousarray(d.transpose(0, 2, 4, 1, 3, 5)).reshape(n_sub**3, s, s, s)


def _straddle_fraction(n_mesh, n_sub, offset_cells, tiles_per_side):
    """Fraction of sub-volumes that cross at least one tile wall.

    A sub-volume inside one tile is blind to tile seams. With N and tiles_per_side powers of
    two, n_sub cannot be coprime to the tiling, so only the lattice offset breaks alignment.
    """
    s = n_mesh // n_sub
    tile_cells = n_mesh // tiles_per_side
    off = int(offset_cells) % n_mesh
    nested = 0
    for b in range(n_sub):
        lo = off + b * s
        # interior wall positions inside (lo, lo + s), in the unrolled frame
        walls = [w for w in range(0, 2 * n_mesh, tile_cells) if lo < w < lo + s]
        nested += 0 if walls else 1
    return 1.0 - (nested / float(n_sub)) ** 3


def subvolume_response(delta, box_size, n_sub, k_centers, dk=None, offset_frac=0.5,
                       tiles_per_side=None):
    """Position-dependent P(k) (integrated bispectrum, Chiang et al. 2014).

    Over n_sub^3 equal sub-cubes, regresses P_sub(k) / <P_sub(k)> - 1 = slope(k) * delta_bar.

    `slope` is dlnP/ddelta_bar. Built on sub-volume band power, it is insensitive to
    small-scale phases, unlike the reduced-bispectrum ratio.

    n_sub must divide the mesh; a center below the sub-volume fundamental 2 pi n_sub / L
    raises. offset_frac shifts the lattice by that fraction of a sub-volume; with
    tiles_per_side given, `straddle_frac` (sub-volumes crossing a tile wall) is computed and
    0.0 raises.

    Returns a dict: delta_bar, p_sub, p_mean, slope, slope_err, n_modes, k_centers, n_sub,
    n_blocks, sub_box, k_f_sub, offset_cells, straddle_frac.
    """
    delta = np.asarray(delta, dtype=np.float64)
    n = delta.shape[0]
    if n % n_sub:
        raise ValueError(f"n_sub={n_sub} must divide the mesh N={n} (equal cubes, no trimming)")
    s = n // n_sub
    sub_box = box_size / n_sub
    kf_sub = 2.0 * np.pi / sub_box
    centers = np.atleast_1d(np.asarray(k_centers, dtype=np.float64))
    w = kf_sub if dk is None else float(np.asarray(dk, dtype=np.float64).ravel()[0])
    low = centers[centers - 0.5 * w < 0.5 * kf_sub]
    if low.size:
        raise ValueError(
            f"k_centers {low.tolist()} reach below the sub-volume fundamental "
            f"{kf_sub:.4f} h/Mpc (n_sub={n_sub}, sub-box {sub_box:.2f} Mpc/h). That band has "
            "no modes inside a sub-volume; use a smaller n_sub or a larger center."
        )

    offset_cells = int(round(offset_frac * s))
    straddle = None
    if tiles_per_side is not None:
        straddle = _straddle_fraction(n, n_sub, offset_cells, int(tiles_per_side))
        if straddle == 0.0:
            raise ValueError(
                f"every sub-volume lies inside one tile (n_sub={n_sub}, "
                f"tiles_per_side={tiles_per_side}, offset {offset_cells} cells): the lattice "
                "is blind to the seams by construction and cannot discriminate the arms."
            )

    blocks = _subvolume_blocks(delta, n_sub, offset_cells)
    delta_bar = blocks.mean(axis=(1, 2, 3))

    _, _, k_mag = _k_grid(s, sub_box)
    masks = [_shell_mask(k_mag, float(c) - 0.5 * w, float(c) + 0.5 * w) for c in centers]
    n_modes = np.array([float(m.sum()) for m in masks])
    if (n_modes == 0).any():
        raise ValueError(f"empty sub-volume band(s) at centers {centers[n_modes == 0].tolist()}")

    dk_blocks = np.fft.rfftn(blocks, axes=(1, 2, 3))
    pm = np.abs(dk_blocks) ** 2 * (sub_box**3 / s**6)
    del dk_blocks
    p_sub = np.stack([pm[:, m].sum(axis=1) / cnt for m, cnt in zip(masks, n_modes)], axis=1)
    del pm

    p_mean = p_sub.mean(axis=0)
    # OLS with delta_bar centered explicitly
    x = delta_bar - delta_bar.mean()
    sxx = float((x**2).sum())
    y = p_sub / p_mean[None, :] - 1.0
    y = y - y.mean(axis=0)[None, :]
    slope = (x[:, None] * y).sum(axis=0) / sxx
    resid = y - x[:, None] * slope[None, :]
    dof = max(len(x) - 2, 1)
    slope_err = np.sqrt((resid**2).sum(axis=0) / dof / sxx)

    return dict(
        delta_bar=delta_bar, p_sub=p_sub, p_mean=p_mean, slope=slope, slope_err=slope_err,
        n_modes=n_modes, k_centers=centers, n_sub=int(n_sub), n_blocks=int(n_sub**3),
        sub_box=float(sub_box), k_f_sub=float(kf_sub), offset_cells=int(offset_cells),
        straddle_frac=straddle,
    )


def min_image_rms(x_a, x_b, box_size):
    """RMS per-component displacement between two position sets, minimum-image."""
    d = np.asarray(x_a, dtype=np.float64) - np.asarray(x_b, dtype=np.float64)
    d = d - box_size * np.round(d / box_size)
    return float(np.sqrt(np.mean(d**2)))
