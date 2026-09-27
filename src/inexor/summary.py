"""The per-run accuracy card: large-scale P(k) against linear theory, as a z profile.

Pipeline: `coarse_delta_streamed` -> `ooc_fft` -> slab-streamed binned P(k) -> a linear oracle
averaged over each bin's realized modes (never evaluated at a bin centre: at large mode counts
the curvature of P across a bin alone gives |z| >> 1). No particle or O(N) position array is
materialized; only the coarse mesh and its spectrum are held.

The card emits no verdict: past the nonlinear scale the evolved field is expected to leave
linear theory. `band_verdict` computes `max|z|` over a band the caller names.
"""

import numpy as np

from .cosmology import growth_factor_a, ic_k_table

CARD = "inexor-pk-summary-1"

# The k range `nonlinear_scale` scans; recorded on the card so a None is interpretable.
NL_SCAN_K = (1e-3, 10.0)


def _mode_grid(n_mesh, box_size):
    """(kx_1d, kz_1d, hermitian weights along kz) for the rfft half-grid.

    The weight is the full-grid multiplicity of a half-grid element: 2, except 1 on the
    self-conjugate planes kz=0 and (even N) kz=Nyquist. It weights both the power average
    and the mode count that sets sigma.
    """
    kx = 2.0 * np.pi * np.fft.fftfreq(n_mesh, d=box_size / n_mesh)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n_mesh, d=box_size / n_mesh)
    wgt = np.full(len(kz), 2.0)
    wgt[0] = 1.0
    if n_mesh % 2 == 0:
        wgt[-1] = 1.0
    return kx, kz, wgt


def binned_power(
    spec,
    n_mesh,
    box_size,
    p_of_k=None,
    edges=None,
    slab=32,
    window=None,
    shot_noise=0.0,
    min_weight=100.0,
    progress=None,
):
    """Slab-streamed, hermitian-weighted binned P(k) of an rfft spectrum.

    `spec` is the (N, N, N//2+1) array `ooc_fft` returns, read slab by slab, never copied.
    `p_of_k`, if given, is averaged in the same pass over the same modes and weights (the
    bin-averaged oracle). `window(kx_slab, kx, kz) -> W` divides power by W**2 per mode
    before binning; `shot_noise` is subtracted after, i.e. `P_raw / W**2 - 1/nbar`.
    `min_weight` drops bins with fewer modes (100 -> ~7% error on sigma itself).
    `progress(stage, done, total)` is called once per slab.

    Returns a dict of parallel per-bin arrays: `k_mean` (weighted), `p`, `n_modes`,
    `window_correction`, `shot_fraction`, `k_edges`, and `p_oracle`, `z` when `p_of_k` given.
    """
    n = int(n_mesh)
    kx, kz, wgt_z = _mode_grid(n, box_size)
    if edges is None:
        k_nyq = np.pi * n / box_size
        edges = np.linspace(0.0, 0.5 * k_nyq, 65)
    edges = np.asarray(edges, dtype=np.float64)
    nb = len(edges) - 1

    acc = {k: np.zeros(nb) for k in ("p", "raw", "oracle", "k", "w", "wcorr")}
    for lo in range(0, n, int(slab)):
        hi = min(lo + int(slab), n)
        if progress is not None:
            progress("bin power", lo, n)
        kk = np.sqrt(
            kx[lo:hi].reshape(-1, 1, 1) ** 2
            + kx.reshape(1, n, 1) ** 2
            + kz.reshape(1, 1, -1) ** 2
        )
        s = spec[lo:hi]
        p = s.real.astype(np.float64) ** 2 + s.imag.astype(np.float64) ** 2
        p *= box_size**3 / float(n) ** 6
        raw = p
        if window is not None:
            w2 = window(kx[lo:hi], kx, kz) ** 2
            p = p / w2
        else:
            w2 = None
        w3 = np.broadcast_to(wgt_z, kk.shape)

        flat = kk.ravel()
        idx = np.digitize(flat, edges) - 1
        sel = (idx >= 0) & (idx < nb) & (flat > 0.0)
        if not sel.any():
            continue
        i, wk = idx[sel], w3.ravel()[sel]
        np.add.at(acc["p"], i, (p.ravel()[sel] - shot_noise) * wk)
        np.add.at(acc["raw"], i, raw.ravel()[sel] * wk)
        np.add.at(acc["k"], i, flat[sel] * wk)
        np.add.at(acc["w"], i, wk)
        np.add.at(acc["wcorr"], i, (1.0 if w2 is None else w2.ravel()[sel]) * wk)
        if p_of_k is not None:
            np.add.at(acc["oracle"], i, p_of_k(flat[sel]) * wk)
    if progress is not None:
        progress("bin power", n, n)

    good = acc["w"] > float(min_weight)
    w = acc["w"][good]
    out = dict(
        k_mean=acc["k"][good] / w,
        p=acc["p"][good] / w,
        n_modes=w,
        window_correction=acc["wcorr"][good] / w,
        shot_fraction=(shot_noise / (acc["raw"][good] / w)) if shot_noise else np.zeros(int(good.sum())),
        k_edges=edges,
    )
    if p_of_k is not None:
        out["p_oracle"] = acc["oracle"][good] / w
        # Gaussian Var(P_hat)/P^2 = 2/N_modes, normalized by the oracle so a low bin does not
        # get a smaller sigma.
        out["z"] = (out["p"] / out["p_oracle"] - 1.0) / np.sqrt(2.0 / w)
    return out


def tsc_window_slab(kx_slab, kx, kz, k_nyq):
    """The TSC window over one axis-0 slab of modes.

    Same function as `diagnostics.tsc_window` (pinned by tests), built per slab from the
    separable 1D factors to avoid a second full (N, N, N//2+1) f64 array.
    """
    wx = np.sinc(np.asarray(kx_slab) / (2.0 * k_nyq)) ** 3
    wy = np.sinc(np.asarray(kx) / (2.0 * k_nyq)) ** 3
    wz = np.sinc(np.asarray(kz) / (2.0 * k_nyq)) ** 3
    return wx.reshape(-1, 1, 1) * wy.reshape(1, -1, 1) * wz.reshape(1, 1, -1)


def nonlinear_scale(p_of_k, k_lo=NL_SCAN_K[0], k_hi=NL_SCAN_K[1], n=4096):
    """The k where the linear dimensionless variance `k^3 P / (2 pi^2)` reaches 1.

    Reported, never gated on. Returns None if the crossing is outside the scanned range
    (no extrapolation).
    """
    k = np.geomspace(k_lo, k_hi, int(n))
    d2 = k**3 * np.asarray(p_of_k(k)) / (2.0 * np.pi**2)
    above = np.nonzero(d2 >= 1.0)[0]
    if not len(above) or above[0] == 0:
        return None
    i = int(above[0])
    lo, hi = np.log(d2[i - 1]), np.log(d2[i])
    f = (0.0 - lo) / (hi - lo)
    return float(np.exp(np.log(k[i - 1]) + f * (np.log(k[i]) - np.log(k[i - 1]))))


def pk_summary_card(
    st,
    cfg,
    cosmo,
    a_out,
    delta=None,
    slab=32,
    edges=None,
    deconvolve_window=True,
    subtract_shot_noise=True,
    min_weight=100.0,
    provenance=None,
    progress=None,
    pool=None,
):
    """The card: measured P(k), the bin-averaged linear oracle, the z profile.

    Oracle is `D(a_out)**2 P_lin(k, z=0)` with D(1) = 1. `delta` reuses a coarse field the
    caller already has; otherwise `coarse_delta_streamed` paints one (needs the integer coarse
    accumulator, since an f64 paint is order-dependent). The window correction is analytic TSC
    (default band stops at half Nyquist); shot noise is V/N.

    Raises if no bin reaches `min_weight` (the default 64 bins are sized for a 1024^3 coarse
    mesh): a zero-bin card would carry no measurement. `pool` (a `paint_only` `TilePool`)
    parallelizes the paint chunks only and is bitwise-neutral by integer associativity.
    `progress` goes to the paint, transform and binning stages.

    Returns the card dict; writes nothing and decides no verdict.
    """
    from .engine import coarse_delta_streamed

    n = int(cfg.n_coarse)
    box = float(cfg.box_size)
    if delta is None:
        delta = coarse_delta_streamed(st, cfg, pool=pool, progress=progress)
    if delta.shape != (n, n, n):
        raise ValueError(f"delta has shape {delta.shape}, want {(n, n, n)} from cfg.n_coarse")

    from . import ooc_fft

    spec = ooc_fft.forward_from_slabs(lambda lo, hi: delta[lo:hi], n, slab=int(slab),
                                      progress=progress)

    tab = ic_k_table(cosmo, n, box)
    d2 = growth_factor_a(a_out, cosmo) ** 2

    def p_lin(k):
        return d2 * tab.P_of_k(k)

    k_nyq = np.pi * n / box

    def _window(kx_slab, kx, kz):
        return tsc_window_slab(kx_slab, kx, kz, k_nyq)

    n_part_total = int(st.n_particles)
    shot = (box**3 / n_part_total) if subtract_shot_noise else 0.0

    res = binned_power(
        spec, n, box,
        p_of_k=p_lin,
        edges=edges,
        slab=int(slab),
        window=_window if deconvolve_window else None,
        shot_noise=shot,
        min_weight=min_weight,
        progress=progress,
    )
    del spec
    if not len(res["k_mean"]):
        n_edges = 64 if edges is None else len(np.asarray(edges)) - 1
        raise ValueError(
            f"no bin reached min_weight={min_weight} modes at n_coarse={n}: "
            f"{n_edges} bins over [0, k_Nyquist/2] is sized for a full-scale coarse mesh. "
            "Widen the bins (`edges`) or lower `min_weight` -- an empty card is not a card."
        )

    k_nl = nonlinear_scale(p_lin)
    card = dict(
        card=CARD,
        n_coarse=n,
        box_size=box,
        n_particles=n_part_total,
        a_out=float(a_out),
        growth_factor=float(growth_factor_a(a_out, cosmo)),
        k_nyquist=float(k_nyq),
        k_nonlinear=k_nl,
        k_nonlinear_scan=[float(NL_SCAN_K[0]), float(NL_SCAN_K[1])],
        shot_noise=float(shot),
        deconvolved="tsc" if deconvolve_window else None,
        oracle="bin-averaged linear, D(a)^2 P_lin; NEVER a bin centre",
        n_bins=int(len(res["k_mean"])),
        # edges, since the weighted `k_mean` does not reconstruct the binning
        k_edges=[float(v) for v in res["k_edges"]],
        k_mean=[float(v) for v in res["k_mean"]],
        p=[float(v) for v in res["p"]],
        p_oracle=[float(v) for v in res["p_oracle"]],
        n_modes=[float(v) for v in res["n_modes"]],
        z_profile=[float(v) for v in res["z"]],
        window_correction=[float(v) for v in res["window_correction"]],
        shot_fraction=[float(v) for v in res["shot_fraction"]],
        provenance=provenance or {},
    )
    return card


def band_verdict(card, k_max, k_min=0.0, bar=5.0):
    """`max|z|` over a band the CALLER names, with the name recorded beside it.

    No default band: past `card["k_nonlinear"]` the z measures gravity, not the code, so
    the limits are returned with the number.
    """
    k = np.asarray(card["k_mean"])
    z = np.asarray(card["z_profile"])
    sel = (k >= float(k_min)) & (k <= float(k_max))
    if not sel.any():
        raise ValueError(f"no bins in [{k_min}, {k_max}]; the card spans {k.min()}-{k.max()}")
    m = float(np.abs(z[sel]).max())
    knl = card.get("k_nonlinear")
    return dict(
        k_min=float(k_min),
        k_max=float(k_max),
        n_bins=int(sel.sum()),
        max_abs_z=m,
        bar=float(bar),
        ok=bool(m < float(bar)),
        band_reaches_nonlinear=bool(knl is not None and float(k_max) > knl),
        k_nonlinear=knl,
    )
