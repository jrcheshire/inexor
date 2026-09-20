"""The per-run accuracy statement: large-scale P(k) with a z profile.

M-v2-6 Stage 4, the summary a finished realization ships with. Until this
existed a run produced a state on disk and no evidence that the state was
right, which is the whole of what an accuracy statement is for.

The measurement is `coarse_delta_streamed` -> `ooc_fft` -> a slab-streamed
binned P(k) -> a **bin-averaged** linear oracle, and it never holds a full-size
float array: the coarse mesh is 4.3 GB at C-gh and its spectrum 8.6 GB, against
a state of 164.6 GB, and neither the particles nor an O(N) position array is
materialized at any point.

**The oracle is averaged over the realized modes of each bin, never evaluated
at a bin centre.** That is not a refinement. At 2048^3 mode counts the
deterministic Jensen term from P's curvature across a bin reaches z = +15 near
k ~ 0.2, and it failed a leg of M-v2-5 at max|z| 8.71 before the bin-averaged
form replaced it (Vista 902091 vs 902182). A bin-centre oracle reads as a code
defect and is an instrument defect.

**What this card does NOT do is emit a verdict.** It stores the z profile, the
mode counts and the corrections, and stops. A scalar over a band that reaches
past the nonlinear scale measures gravity rather than the code: the evolved
field is SUPPOSED to depart from linear theory there, and a `max|z|` that mixes
the two cannot be read. `k_nonlinear` is on the card so a caller can state its
own band, and `band_verdict` computes one over a band the caller names and
records the name alongside the number.
"""

import numpy as np

from .cosmology import growth_factor_a, ic_k_table

CARD = "inexor-pk-summary-1"

# The k range `nonlinear_scale` scans, on the card beside its answer: a None
# there means "no crossing in here", which a reader cannot act on without
# knowing what "here" was.
NL_SCAN_K = (1e-3, 10.0)


def _mode_grid(n_mesh, box_size):
    """(kx_1d, kz_1d, hermitian weights along kz) for the rfft half-grid.

    The weight is the multiplicity a half-grid element stands for on the full
    grid: 2 for every kz plane except kz=0 and, on an even grid, kz=Nyquist,
    which are self-conjugate and hold each mode beside its own conjugate. It
    weights the power average and the mode COUNT alike, and the count is what
    sets sigma, so getting it wrong inflates every z by a uniform sqrt(2) --
    visible only against a field whose answer is known, which is what the null
    test in `tests/test_summary.py` is.
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
):
    """Slab-streamed, hermitian-weighted binned P(k) of an rfft spectrum.

    `spec` is the (N, N, N//2+1) array `ooc_fft` returns; it is read slab by
    slab and never copied. `p_of_k`, if given, is averaged over the SAME modes
    and the SAME weights to produce the bin-averaged oracle -- one accumulation
    pass, so the oracle cannot drift onto a different mode set than the
    measurement it is compared against.

    `window` is a callable `(kx_slab, kx, kz) -> W` giving the mass-assignment
    window over the slab's modes; measured power is divided by `W**2` per MODE,
    before binning, because the correction varies across a bin. `shot_noise` is
    subtracted after that, which is the order the painted field builds them in:
    a discrete sample painted with W has <|delta|^2> = W^2 (P + 1/nbar), so the
    estimator is `P_raw / W**2 - 1/nbar`.

    `min_weight` drops bins too sparse for the Gaussian sigma to mean anything;
    100 modes puts the fractional error on sigma itself at ~7%.

    Returns a dict of parallel arrays: `k_mean` (weighted), `p`, `n_modes`,
    `p_oracle` when `p_of_k` was given, plus the per-bin `window_correction`
    and `shot_fraction` so neither correction is hidden inside `p`.
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

    good = acc["w"] > float(min_weight)
    w = acc["w"][good]
    out = dict(
        k_mean=acc["k"][good] / w,
        p=acc["p"][good] / w,
        n_modes=w,
        # what the corrections did, per bin, so `p` is never a black box
        window_correction=acc["wcorr"][good] / w,
        shot_fraction=(shot_noise / (acc["raw"][good] / w)) if shot_noise else np.zeros(int(good.sum())),
        k_edges=edges,
    )
    if p_of_k is not None:
        out["p_oracle"] = acc["oracle"][good] / w
        # Gaussian: Var(P_hat)/P^2 = 2/N_modes. The oracle is the denominator
        # rather than the measurement, so a bin that came out low is not handed
        # a smaller sigma for having done so.
        out["z"] = (out["p"] / out["p_oracle"] - 1.0) / np.sqrt(2.0 / w)
    return out


def tsc_window_slab(kx_slab, kx, kz, k_nyq):
    """The TSC window over one axis-0 slab of modes, built rather than sliced.

    `diagnostics.tsc_window` returns the whole half-grid, which is another
    (N, N, N//2+1) f64 array beside the spectrum -- 17 GB at C-gh to hold a
    separable product of three 1D factors. This is the same function on a slab;
    `tests/test_summary.py` pins the two against each other, which is what stops
    them drifting.
    """
    wx = np.sinc(np.asarray(kx_slab) / (2.0 * k_nyq)) ** 3
    wy = np.sinc(np.asarray(kx) / (2.0 * k_nyq)) ** 3
    wz = np.sinc(np.asarray(kz) / (2.0 * k_nyq)) ** 3
    return wx.reshape(-1, 1, 1) * wy.reshape(1, -1, 1) * wz.reshape(1, 1, -1)


def nonlinear_scale(p_of_k, k_lo=NL_SCAN_K[0], k_hi=NL_SCAN_K[1], n=4096):
    """The k where the linear dimensionless variance `k^3 P / (2 pi^2)` reaches 1.

    Reported, never gated on: it is where the evolved field is EXPECTED to leave
    linear theory, so it is the scale past which a z against a linear oracle
    stops being about the code. Returns None if the crossing is outside the
    scanned range rather than extrapolating to a number that looks measured.
    """
    k = np.geomspace(k_lo, k_hi, int(n))
    d2 = k**3 * np.asarray(p_of_k(k)) / (2.0 * np.pi**2)
    above = np.nonzero(d2 >= 1.0)[0]
    if not len(above) or above[0] == 0:
        return None
    i = int(above[0])
    # log-linear interpolation across the crossing; the grid is log-spaced
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
):
    """The card: measured P(k), the bin-averaged linear oracle, the z profile.

    `a_out` is the output scale factor; the oracle is `D(a_out)**2` times the
    z=0 linear spectrum, `growth_factor_a` being normalized to D(1) = 1.

    `delta` lets a caller pass a coarse field it already has (the engine can
    hand over the last step's) rather than paying for the paint twice. Without
    one this calls `coarse_delta_streamed`, which needs the integer coarse
    accumulator -- an f64 paint is order-dependent and cannot be streamed.

    The window correction is TSC because the coarse paint is TSC, and it is the
    analytic window rather than interlacing, so it is trustworthy well below
    Nyquist and the default band stops at half Nyquist. Shot noise is `V/N`.

    `min_weight` drops bins too sparse for a Gaussian sigma to mean anything,
    and an empty result is REFUSED rather than returned: the default 64 bins
    are sized for a 1024^3 coarse mesh, and at a small `n_coarse` every bin
    falls under the floor. A card of zero bins passes every structural check a
    caller is likely to make while carrying no measurement at all.

    Returns the card dict. Nothing here writes a file or decides a verdict.
    """
    from .engine import coarse_delta_streamed

    n = int(cfg.n_coarse)
    box = float(cfg.box_size)
    if delta is None:
        delta = coarse_delta_streamed(st, cfg)
    if delta.shape != (n, n, n):
        raise ValueError(f"delta has shape {delta.shape}, want {(n, n, n)} from cfg.n_coarse")

    from . import ooc_fft

    spec = ooc_fft.forward_from_slabs(lambda lo, hi: delta[lo:hi], n, slab=int(slab))

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
        k_mean=[float(v) for v in res["k_mean"]],
        p=[float(v) for v in res["p"]],
        p_oracle=[float(v) for v in res["p_oracle"]],
        n_modes=[float(v) for v in res["n_modes"]],
        # the max-over-band lesson: the SHAPE is the product, not a scalar
        z_profile=[float(v) for v in res["z"]],
        window_correction=[float(v) for v in res["window_correction"]],
        shot_fraction=[float(v) for v in res["shot_fraction"]],
        provenance=provenance or {},
    )
    return card


def band_verdict(card, k_max, k_min=0.0, bar=5.0):
    """`max|z|` over a band the CALLER names, with the name recorded beside it.

    There is no default band on purpose. Any band reaching past
    `card["k_nonlinear"]` measures gravity rather than the code, so the number
    is only meaningful next to the limits that produced it -- which is why they
    come back in the result instead of being left at the call site.
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
        # loud rather than silent: a band past the nonlinear scale is expected
        # to fail and the failure is not about the code
        band_reaches_nonlinear=bool(knl is not None and float(k_max) > knl),
        k_nonlinear=knl,
    )
