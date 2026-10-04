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


_SUMS = ("p", "raw", "oracle", "k", "w", "wcorr")


def binned_power_partials(
    spec,
    n_mesh,
    box_size,
    y0=0,
    p_of_k=None,
    edges=None,
    slab=32,
    window=None,
    shot_noise=0.0,
    progress=None,
):
    """Per-ky-plane partial sums of the binned power over the y-pencils `spec` holds.

    `spec[:, j, :]` is ky-plane `y0 + j` of the (N, N, N//2+1) rfft spectrum (the whole
    spectrum with `y0=0`, or one rank's pencils), read `slab` planes at a time. Each plane's
    partial is summed in its own element order, so it depends on that plane alone: block size
    and the split of planes across ranks cannot move a bit. Arguments as `binned_power`.

    Returns dict(n_mesh, y0, n_planes, k_edges, sums): `sums[name]` is (n_planes, n_bins) f64
    for each of "p", "raw", "oracle", "k", "w", "wcorr". Combine with `combine_partials`.
    """
    n = int(n_mesh)
    kx, kz, wgt_z = _mode_grid(n, box_size)
    if edges is None:
        k_nyq = np.pi * n / box_size
        edges = np.linspace(0.0, 0.5 * k_nyq, 65)
    edges = np.asarray(edges, dtype=np.float64)
    nb = len(edges) - 1
    n_planes = int(spec.shape[1])

    sums = {k: np.zeros((n_planes, nb)) for k in _SUMS}
    for lo in range(0, n_planes, int(slab)):
        hi = min(lo + int(slab), n_planes)
        if progress is not None:
            progress("bin power", lo, n_planes)
        ky = kx[y0 + lo:y0 + hi]
        kk = np.sqrt(
            kx.reshape(-1, 1, 1) ** 2
            + ky.reshape(1, -1, 1) ** 2
            + kz.reshape(1, 1, -1) ** 2
        )
        s = spec[:, lo:hi]
        p = s.real.astype(np.float64) ** 2 + s.imag.astype(np.float64) ** 2
        p *= box_size**3 / float(n) ** 6
        raw = p
        if window is not None:
            w2 = window(kx, ky, kz) ** 2
            p = p / w2
        else:
            w2 = None
        w3 = np.broadcast_to(wgt_z, kk.shape)

        flat = kk.ravel()
        idx = np.digitize(flat, edges) - 1
        sel = (idx >= 0) & (idx < nb) & (flat > 0.0)
        if not sel.any():
            continue
        # (plane, bin) as one index; bincount adds in element order, which within one plane
        # is (kx, kz) whatever the block
        plane = np.broadcast_to(np.arange(hi - lo).reshape(1, -1, 1), kk.shape).ravel()
        at = plane[sel] * nb + idx[sel]
        wk = w3.ravel()[sel]

        def add(name, v):
            sums[name][lo:hi] += np.bincount(at, weights=v, minlength=(hi - lo) * nb
                                             ).reshape(hi - lo, nb)

        add("p", (p.ravel()[sel] - shot_noise) * wk)
        add("raw", raw.ravel()[sel] * wk)
        add("k", flat[sel] * wk)
        add("w", wk)
        add("wcorr", (1.0 if w2 is None else w2.ravel()[sel]) * wk)
        if p_of_k is not None:
            add("oracle", p_of_k(flat[sel]) * wk)
    if progress is not None:
        progress("bin power", n_planes, n_planes)
    return dict(n_mesh=n, y0=int(y0), n_planes=n_planes, k_edges=edges, sums=sums,
                oracle=p_of_k is not None, shot_noise=float(shot_noise))


def combine_partials(parts, min_weight=100.0):
    """`binned_power`'s result from `binned_power_partials` over planes covering [0, N) once.

    Per bin, the plane partials are combined with `math.fsum` (correctly rounded), so the
    result does not depend on the order or grouping of `parts`.
    """
    import math

    parts = list(parts)
    if not parts:
        raise ValueError("no partials to combine")
    head = parts[0]
    n = head["n_mesh"]
    for q in parts[1:]:
        if (q["n_mesh"] != n or not np.array_equal(q["k_edges"], head["k_edges"])
                or q["oracle"] != head["oracle"] or q["shot_noise"] != head["shot_noise"]):
            raise ValueError("partials disagree on the mesh, the bins, the oracle or the shot "
                             "noise")
    cover = np.zeros(n, dtype=np.int64)
    for q in parts:
        cover[q["y0"]:q["y0"] + q["n_planes"]] += 1
    if not np.all(cover == 1):
        raise ValueError(f"partials cover ky-planes {np.count_nonzero(cover == 0)} times zero "
                         f"and {np.count_nonzero(cover > 1)} times more than once; each of "
                         f"the {n} planes must be summed exactly once")
    nb = len(head["k_edges"]) - 1
    acc = {}
    for name in _SUMS:
        stacked = np.concatenate([q["sums"][name] for q in parts], axis=0)
        acc[name] = np.array([math.fsum(stacked[:, b]) for b in range(nb)])
    shot_noise = head["shot_noise"]

    good = acc["w"] > float(min_weight)
    w = acc["w"][good]
    out = dict(
        k_mean=acc["k"][good] / w,
        p=acc["p"][good] / w,
        n_modes=w,
        window_correction=acc["wcorr"][good] / w,
        shot_fraction=(shot_noise / (acc["raw"][good] / w)) if shot_noise else np.zeros(int(good.sum())),
        k_edges=head["k_edges"],
    )
    if head["oracle"]:
        out["p_oracle"] = acc["oracle"][good] / w
        # Gaussian Var(P_hat)/P^2 = 2/N_modes, normalized by the oracle so a low bin does not
        # get a smaller sigma.
        out["z"] = (out["p"] / out["p_oracle"] - 1.0) / np.sqrt(2.0 / w)
    return out


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
    """Hermitian-weighted binned P(k) of an rfft spectrum, streamed by y-pencil blocks.

    `spec` is the (N, N, N//2+1) array `ooc_fft` returns, read `slab` ky-planes at a time,
    never copied; the result is bitwise independent of `slab` (`binned_power_partials` then
    `combine_partials`). `p_of_k`, if given, is averaged in the same pass over the same modes
    and weights (the bin-averaged oracle). `window(kx, ky_block, kz) -> W` divides power by
    W**2 per mode before binning; `shot_noise` is subtracted after, i.e. `P_raw / W**2 - 1/nbar`.
    `min_weight` drops bins with fewer modes (100 -> ~7% error on sigma itself).
    `progress(stage, done, total)` is called once per block.

    Returns a dict of parallel per-bin arrays: `k_mean` (weighted), `p`, `n_modes`,
    `window_correction`, `shot_fraction`, `k_edges`, and `p_oracle`, `z` when `p_of_k` given.
    """
    part = binned_power_partials(spec, n_mesh, box_size, p_of_k=p_of_k, edges=edges,
                                 slab=slab, window=window, shot_noise=shot_noise,
                                 progress=progress)
    return combine_partials([part], min_weight=min_weight)


def tsc_window_slab(kx, ky, kz, k_nyq):
    """The TSC window over a block of modes (any axis may be a sub-range).

    Same function as `diagnostics.tsc_window` (pinned by tests), built per block from the
    separable 1D factors to avoid a second full (N, N, N//2+1) f64 array.
    """
    wx = np.sinc(np.asarray(kx) / (2.0 * k_nyq)) ** 3
    wy = np.sinc(np.asarray(ky) / (2.0 * k_nyq)) ** 3
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


def _card_model(cfg, cosmo, a_out):
    """(p_lin, window, k_nyq) for a card on `cfg`'s coarse mesh at `a_out`."""
    n = int(cfg.n_coarse)
    box = float(cfg.box_size)
    tab = ic_k_table(cosmo, n, box)
    d2 = growth_factor_a(a_out, cosmo) ** 2

    def p_lin(k):
        return d2 * tab.P_of_k(k)

    k_nyq = np.pi * n / box

    def window(kx, ky, kz):
        return tsc_window_slab(kx, ky, kz, k_nyq)

    return p_lin, window, k_nyq


def _card_from_result(res, cfg, cosmo, a_out, p_lin, n_part_total, shot, deconvolve_window,
                      min_weight, edges, transform, provenance):
    """The card dict from `combine_partials`' result; refuses a card with no bin."""
    n = int(cfg.n_coarse)
    if not len(res["k_mean"]):
        n_edges = 64 if edges is None else len(np.asarray(edges)) - 1
        raise ValueError(
            f"no bin reached min_weight={min_weight} modes at n_coarse={n}: "
            f"{n_edges} bins over [0, k_Nyquist/2] is sized for a full-scale coarse mesh. "
            "Widen the bins (`edges`) or lower `min_weight` -- an empty card is not a card."
        )
    return dict(
        card=CARD,
        n_coarse=n,
        box_size=float(cfg.box_size),
        n_particles=int(n_part_total),
        a_out=float(a_out),
        growth_factor=float(growth_factor_a(a_out, cosmo)),
        k_nyquist=float(np.pi * n / float(cfg.box_size)),
        k_nonlinear=nonlinear_scale(p_lin),
        k_nonlinear_scan=[float(NL_SCAN_K[0]), float(NL_SCAN_K[1])],
        shot_noise=float(shot),
        deconvolved="tsc" if deconvolve_window else None,
        oracle="bin-averaged linear, D(a)^2 P_lin; NEVER a bin centre",
        transform=transform,
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
    (default band stops at half Nyquist); shot noise is V/N. The transform runs on the host
    (`transform: "host"`); `pk_summary_card_cards` is the multi-rank card on the GPUs.

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
    p_lin, window, _k_nyq = _card_model(cfg, cosmo, a_out)
    n_part_total = int(st.n_particles)
    shot = (box**3 / n_part_total) if subtract_shot_noise else 0.0

    res = binned_power(
        spec, n, box,
        p_of_k=p_lin,
        edges=edges,
        slab=int(slab),
        window=window if deconvolve_window else None,
        shot_noise=shot,
        min_weight=min_weight,
        progress=progress,
    )
    del spec
    return _card_from_result(res, cfg, cosmo, a_out, p_lin, n_part_total, shot,
                             deconvolve_window, min_weight, edges, "host", provenance)


def pk_summary_card_cards(
    st,
    cfg,
    cosmo,
    a_out,
    devices=None,
    decomp=None,
    comm=None,
    slab=32,
    edges=None,
    deconvolve_window=True,
    subtract_shot_noise=True,
    min_weight=100.0,
    provenance=None,
    progress=None,
    phase=None,
):
    """`pk_summary_card` with the paint and the transform on the cards, across ranks.

    Every rank calls this with its node-local state (or one rank with the whole state):
    `device.paint.coarse_delta_cards` paints the rank's coarse planes on `devices`,
    `ooc_fft.forward_card_planes_to_pencils` returns its y-pencils of the spectrum, each rank
    bins its own pencils (`binned_power_partials`) and the partials are allgathered and
    combined exactly. The card is bitwise the same at any rank and card count, and every rank
    returns it. Against `pk_summary_card` the density is bitwise; the spectrum differs at the
    FFT's rounding (`transform: "cards"`). Shot noise uses the particle count over all ranks.

    `phase(name)`, if given, is called after "card_paint", "card_transform" and "card_bin".
    Other arguments as `pk_summary_card`.
    """
    from .device.paint import coarse_delta_cards
    from .ooc_fft import forward_card_planes_to_pencils

    def mark(name):
        if phase is not None:
            phase(name)

    n = int(cfg.n_coarse)
    box = float(cfg.box_size)
    if decomp is None:
        x_parts, y_parts = [(0, n)], [(0, n)]
    else:
        cpt = int(decomp.coarse_per_tile)
        x_parts = [(lo * cpt, hi * cpt) for lo, hi in decomp.rank_planes]
        y_parts = [tuple(p) for p in decomp.rank_pencils]
    rank = 0 if comm is None else int(comm.rank)
    shards = coarse_delta_cards(st, cfg, devices=devices, decomp=decomp, comm=comm)
    mark("card_paint")
    spec = forward_card_planes_to_pencils(shards, n, x_parts, y_parts, comm)
    del shards
    mark("card_transform")

    p_lin, window, _k_nyq = _card_model(cfg, cosmo, a_out)
    n_part_total = (int(st.n_particles) if comm is None
                    else int(comm.allreduce(int(st.n_particles))))
    shot = (box**3 / n_part_total) if subtract_shot_noise else 0.0
    part = binned_power_partials(
        spec, n, box,
        y0=y_parts[rank][0],
        p_of_k=p_lin,
        edges=edges,
        slab=int(slab),
        window=window if deconvolve_window else None,
        shot_noise=shot,
        progress=progress,
    )
    del spec
    parts = [part] if comm is None else comm.allgather(part)
    res = combine_partials(parts, min_weight=min_weight)
    mark("card_bin")
    return _card_from_result(res, cfg, cosmo, a_out, p_lin, n_part_total, shot,
                             deconvolve_window, min_weight, edges, "cards", provenance)


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
