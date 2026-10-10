"""Check P(k) cards against CAMB HMcode2020 / halofit and EuclidEmulator2.

    pixi exec --spec python=3.12 --spec camb --spec matplotlib --spec numpy --spec scipy \
        -- python scripts/compare/pk_boost_reference.py RUN/realization_pk.json \
        --cosmology RUN/export.json -o figures/hero_pk_boost.png

Run through `pixi exec`, not the project env: camb is a one-off reference, not an
engine dependency, and adding it would move `pixi.lock`. EuclidEmulator2 is used if
`euclidemu2` imports (see `ee2_ratio_figure.py` for its install), else skipped.

Left panel: the nonlinear boost B(k) = P/P_lin, each card over its own linear oracle and
each reference over CAMB's P_lin, for scale. Right panel: P_inexor / P_reference, every
reference averaged over each card bin's own lattice modes (`card_bin_modes`), the average
the card takes of its own P; a reference at a bin's mean k instead would leave the BAO as
a +-1-2% zigzag. P_reference is CAMB's P_nl for HMcode/halofit and CAMB P_lin x B_EE2 for
EE2, all at the A_s that matches the cosmology's sigma8. A card made from EH98 ICs shows
EH98's linear difference from CAMB at low k. One card: the ratio against every
reference. Several cards (one redshift; any boxes): each against EE2 (HMcode without it).

Scope: HMcode2020 is quoted at a few percent for LCDM here, so this is a sanity
check at the several-percent level (is a ~5x boost at k = 1 the right size), not a
validation. At the top of the band the card is on the coarse mesh, where PM force
resolution and mass assignment suppress power while nonlinearity raises it; the
comparison is clean for roughly k < 0.5 h/Mpc. The cosmology must come from the card
or `--cosmology`; there is no default.
"""

import argparse
import json

import numpy as np
from scipy.signal import fftconvolve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SHORT = {"ee2": "EE2", "mead2020": "HMcode", "takahashi": "halofit"}
# quoted accuracies drawn as the ratio panel's band: EE2 1% for 0.01 <= k <= 10 h/Mpc,
# z <= 3 (Knabenhans et al. 2021, arXiv:2010.11288); HMcode2020 2.5% (the band this
# script has always drawn for it)
ACCURACY = {"ee2": 0.01, "mead2020": 0.025}


def camb_boost(cos, z, kmax, version, npoints=400):
    """B(k) = P_nl/P_lin from CAMB at `cos`, normalized to its sigma8, on `npoints` log-spaced
    k in [1e-3, kmax]. Returns (kh, P_lin, P_nl, sigma8, A_s)."""
    import camb

    h = cos["h"]
    pars = camb.set_params(
        H0=100.0 * h,
        ombh2=cos["Omega_b"] * h * h,
        omch2=(cos["Omega_m"] - cos["Omega_b"]) * h * h,
        ns=cos["n_s"],
        TCMB=cos.get("T_cmb_K", 2.7255),
        # the engine's Cosmology has massless neutrinos; CAMB defaults to 0.06 eV
        mnu=0.0, num_massive_neutrinos=0, omk=0.0,
        halofit_version=version,
    )
    pars.set_matter_power(redshifts=[z], kmax=max(kmax * 2.0, 10.0))

    # sigma8 is an output of the solve: rescale As to hit it (sigma8 ~ sqrt(As))
    pars.NonLinear = camb.model.NonLinear_none
    s8 = camb.get_results(pars).get_sigma8_0()
    pars.InitPower.As = pars.InitPower.As * (cos["sigma8"] / s8) ** 2
    lin = camb.get_results(pars)
    kh, _, plin = lin.get_matter_power_spectrum(minkh=1e-3, maxkh=kmax, npoints=npoints)

    pars.NonLinear = camb.model.NonLinear_both
    nl = camb.get_results(pars)
    kh2, _, pnl = nl.get_matter_power_spectrum(minkh=1e-3, maxkh=kmax, npoints=npoints)
    assert np.allclose(kh, kh2)
    return kh, plin[0], pnl[0], float(nl.get_sigma8_0()), float(pars.InitPower.As)


def ee2_boost(cos, z, k, A_s):
    """EuclidEmulator2's boost B(k) at `cos` and redshift `z`, evaluated at `k`.

    EE2 emulates B(k) directly, so nothing is divided here. `A_s` comes from a CAMB
    solve: the engine is sigma8-normalized and EE2 is parameterized by A_s. Refuses a
    cosmology outside EE2's training range (an emulator off-range returns silently).
    """
    import euclidemu2

    e = euclidemu2.PyEuclidEmulator()
    par = dict(Omega_b=cos["Omega_b"], Omega_m=cos["Omega_m"], m_ncdm=0.0,
               n_s=cos["n_s"], h=cos["h"], w0_fld=-1.0, wa_fld=0.0, A_s=A_s)
    for name, (lo, hi) in e.bounds.items():
        if not (lo <= par[name] <= hi):
            raise SystemExit(
                f"EE2 is not valid at this cosmology: {name}={par[name]:g} is "
                f"outside its training range [{lo:g}, {hi:g}]. An emulator "
                f"evaluated off its range returns a number and no warning."
            )
    kv, bz = e.get_boost(par, [float(z)], custom_kvec=np.asarray(k, float))
    return np.asarray(bz[0], float)


def card_bin_modes(summary):
    """Each of the card's bins as lattice shells: a list, parallel to `k_mean`, of
    (k, count) arrays, k = kf sqrt(|n|^2) and count the full-grid modes of that shell in
    the bin.

    Mirrors `inexor.summary.binned_power_partials`: |k| = sqrt((kx^2 + ky^2) + kz^2) with
    k_i = 2 pi * fftfreq, digitized against the card's edges (the binner's default, 64 bins
    to half the coarse Nyquist, for cards that predate `k_edges`). A shell on an edge is
    split vector by vector with that same float expression, so rounding sends each mode
    where the card sent it. Refuses unless every bin's count equals the card's `n_modes`.
    """
    n = int(summary["n_coarse"])
    box = float(summary["box_size"])
    if "k_edges" in summary:
        edges = np.asarray(summary["k_edges"], float)
    else:
        edges = np.linspace(0.0, 0.5 * np.pi * n / box, 65)
    val = 1.0 / (n * (box / n))  # np.fft.fftfreq's spacing

    def kc(i):
        return 2.0 * np.pi * (np.asarray(i, float) * val)

    kf = float(kc(1))
    m_max = int((edges[-1] / kf) ** 2) + 2
    r = int(np.sqrt(m_max)) + 1
    if r >= n // 2:
        raise SystemExit(f"bins reach the mesh's Nyquist plane (n={n}); the infinite-lattice "
                         "shell counts do not apply")
    r1 = np.zeros(m_max + 1)
    for i in range(-r, r + 1):
        if i * i <= m_max:
            r1[i * i] += 1
    r3 = np.rint(fftconvolve(fftconvolve(r1, r1)[: m_max + 1], r1)[: m_max + 1])
    m = np.arange(m_max + 1)
    have = (r3 > 0) & (m > 0)
    m, cnt = m[have], r3[have]
    k = kf * np.sqrt(m)
    nb = len(edges) - 1
    shells = [([], []) for _ in range(nb)]

    on_edge = np.zeros(m.size, bool)
    for e in edges:
        on_edge |= np.abs(k - e) <= 1e-9 * max(e, kf)
    for mm, kk, cc in zip(m[on_edge], k[on_edge], cnt[on_edge]):
        a, b = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
        c2 = mm - a * a - b * b
        ok = c2 >= 0
        a, b, c2 = a[ok], b[ok], c2[ok]
        c = np.rint(np.sqrt(c2)).astype(np.int64)
        sq = c * c == c2
        a, b, c = a[sq], b[sq], c[sq]
        mult = np.where(c > 0, 2, 1)  # +-c
        kv = np.sqrt((kc(a) ** 2 + kc(b) ** 2) + kc(c) ** 2)
        idx = np.digitize(kv, edges) - 1
        if mult.sum() != cc:
            raise SystemExit(f"shell |n|^2={mm}: enumerated {mult.sum()} modes, counted {cc}")
        for bi in np.unique(idx):
            if 0 <= bi < nb:
                shells[bi][0].append(kk)
                shells[bi][1].append(float(mult[idx == bi].sum()))
    idx = np.digitize(k, edges) - 1
    for bi in range(nb):
        sel = (idx == bi) & ~on_edge
        shells[bi][0].extend(k[sel])
        shells[bi][1].extend(cnt[sel])

    out = []
    for km, nm in zip(summary["k_mean"], summary["n_modes"]):
        bi = int(np.digitize(km, edges)) - 1
        ks, cs = np.asarray(shells[bi][0]), np.asarray(shells[bi][1])
        if cs.sum() != nm or not np.isclose((ks * cs).sum() / cs.sum(), km, rtol=1e-9, atol=0):
            raise SystemExit(f"bin {bi} (k_mean {km:.4f}): rebuilt {cs.sum():.0f} modes at mean "
                             f"k {(ks * cs).sum() / max(cs.sum(), 1):.6f}, card has {nm:.0f}")
        out.append((ks, cs))
    return out


def bin_average(bins, p_of_k):
    """Mode-weighted mean of `p_of_k` over each bin of `card_bin_modes`; `p_of_k` is
    called once, on every shell's k."""
    k_all = np.unique(np.concatenate([ks for ks, _ in bins]))
    p_all = np.asarray(p_of_k(k_all), float)
    return np.array([(np.interp(ks, k_all, p_all) * cs).sum() / cs.sum() for ks, cs in bins])


def loglog(k, p):
    """P(k) log-log interpolated from a tabulated (k, p)."""
    lk, lp = np.log(k), np.log(p)
    return lambda q: np.exp(np.interp(np.log(q), lk, lp))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cards", nargs="+")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="legend labels, one per card (default: built from each card)")
    ap.add_argument("--linestyles", nargs="+", default=None,
                    help="matplotlib line styles, one per card (default solid)")
    ap.add_argument("--legend-loc", default="lower left",
                    help="legend placement in the ratio panel")
    ap.add_argument("--cosmology", default=None,
                    help="a JSON file carrying a `cosmology` block (the export "
                         "header has one). Required unless the card does -- there "
                         "is no default, because a reference built at the wrong "
                         "parameters looks exactly like a discrepancy")
    ap.add_argument("-o", "--out", default="figures/hero_pk_boost.png")
    ap.add_argument("--no-ee2", action="store_true",
                    help="skip EuclidEmulator2 even if it imports")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    cards = []
    for path in args.cards:
        with open(path) as fh:
            cards.append(json.load(fh))
    for name, opt in (("--labels", args.labels), ("--linestyles", args.linestyles)):
        if opt and len(opt) != len(cards):
            raise SystemExit(f"{name} needs one entry per card")
    lstyles = args.linestyles or ["-"] * len(cards)
    a_out = float(cards[0]["a_out"])
    if any(abs(float(c["a_out"]) - a_out) > 1e-9 for c in cards):
        raise SystemExit("cards are at different epochs; the references are built at one "
                         "redshift")
    z = 1.0 / a_out - 1.0
    cos = cards[0].get("cosmology") or cards[0]["summary"].get("cosmology")
    if cos is None and args.cosmology:
        with open(args.cosmology) as fh:
            cos = json.load(fh).get("cosmology")
    if cos is None:
        raise SystemExit(
            "no cosmology: this card carries none (cards written before this "
            "check existed do not), so pass --cosmology pointing at a JSON with "
            "a `cosmology` block -- the export header has one. There is no "
            "default on purpose."
        )
    print(f"  cosmology {cos}")

    # simulations take the categorical order from C0; the references use gray, green and
    # red (HMcode, halofit, EE2) so the two never share a color
    sim_colors = ["C0", "C1", "C4", "C5", "C6", "C8", "C9"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    series = []
    for i, c in enumerate(cards):
        s = c["summary"]
        k = np.asarray(s["k_mean"], float)
        p = np.asarray(s["p"], float)
        n = round(s["n_particles"] ** (1.0 / 3.0))
        lab = (args.labels[i] if args.labels else
               f"inexor {n}$^3$, {s['box_size']:g} $h^{{-1}}$Mpc")
        series.append((k, p, card_bin_modes(s), lab))
        ax.plot(k, p / np.asarray(s["p_oracle"], float), ls=lstyles[i], lw=1.7, marker="o",
                ms=2.4, color=sim_colors[i], label=lab)

    k_top = max(max(c["summary"]["k_mean"]) for c in cards)
    styles = {"mead2020": ("0.35", "--", "HMcode2020"),
              "takahashi": ("C2", ":", "halofit (Takahashi)")}
    refs = {}  # name -> P(k) callable
    for version, (col, ls, lab) in styles.items():
        kh, plin, pnl, s8, A_s = camb_boost(cos, z, k_top * 1.3, version, npoints=8000)
        refs[version] = loglog(kh, pnl)
        ax.plot(kh, pnl / plin, ls=ls, lw=1.6, color=col, label=f"{lab}, $\\sigma_8={s8:.3f}$")
        print(f"  {lab:22s} sigma8={s8:.4f}")
    print(f"  A_s for sigma8={cos['sigma8']:g}: {A_s:.4e}")
    p_lin = loglog(kh, plin)

    if not args.no_ee2:
        try:
            ee2_boost(cos, z, np.array([0.1]), A_s)
        except ImportError:
            print("  EuclidEmulator2        not installed; skipped")
        else:
            refs["ee2"] = lambda q: p_lin(q) * ee2_boost(cos, z, q, A_s)
            styles["ee2"] = ("C3", "-.", "EuclidEmulator2")
            kk = np.geomspace(0.01, k_top * 1.05, 400)
            ax.plot(kk, ee2_boost(cos, z, kk, A_s), ls="-.", lw=1.6, color="C3",
                    label="EuclidEmulator2")

    k_lo = min(k.min() for k, *_ in series)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(k_lo * 0.9, k_top * 1.05)
    ax.set_xlabel(r"$k\ \ [h\,{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$B(k)=P_{\rm nl}/P_{\rm lin}$")
    ax.legend(frameon=False, fontsize=9)

    names = [nm for nm in ("ee2", "mead2020", "takahashi") if nm in refs]
    ratios = {}  # (card index, reference) -> P_inexor / P_reference per bin
    for i, (k, p, bins, lab) in enumerate(series):
        for nm in names if len(series) == 1 else names[:1]:
            ratios[i, nm] = p / bin_average(bins, refs[nm])
    if len(series) == 1:
        k = series[0][0]
        for nm in names:
            col, ls, lab = styles[nm]
            bx.plot(k, ratios[0, nm], ls=ls, lw=1.7, color=col, marker="o", ms=2.8, label=lab)
        bx.set_ylabel(r"$P_{\rm inexor}\,/\,P_{\rm reference}$")
    else:
        for i, (k, _, _, lab) in enumerate(series):
            bx.plot(k, ratios[i, names[0]], ls=lstyles[i], lw=1.7, color=sim_colors[i],
                    marker="o", ms=2.4, label=lab)
        bx.set_ylabel(rf"$P\,/\,P_{{\rm {SHORT[names[0]]}}}$")  # cards may be other codes
    bx.axhline(1.0, lw=1.0, color="0.45", ls="-")
    # the band is the quoted accuracy of the reference the panel divides by (HMcode's when
    # one card is read against all three)
    band = names[0] if len(series) > 1 else "mead2020"
    bx.fill_between([k_lo * 0.9, k_top * 1.05], 1.0 - ACCURACY[band], 1.0 + ACCURACY[band],
                    color="0.6", alpha=0.18, lw=0, label=f"{styles[band][2]} quoted accuracy")
    bx.set_xscale("log")
    bx.set_xlim(k_lo * 0.9, k_top * 1.05)
    bx.set_xlabel(r"$k\ \ [h\,{\rm Mpc}^{-1}]$")
    bx.legend(frameon=False, fontsize=9, loc=args.legend_loc)

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"-> {args.out}")

    for i, (k, _, _, lab) in enumerate(series):
        cols = [nm for nm in names if (i, nm) in ratios]
        print(f"\n  {lab}: P_inexor / P_reference")
        print("  " + f"{'k':>8} " + " ".join(f"{styles[nm][2][:9]:>10}" for nm in cols))
        for b in range(len(k)):
            if b % 4 and k[b] > 0.15 and b != len(k) - 1:
                continue
            print(f"  {k[b]:8.4f} " + " ".join(f"{ratios[i, nm][b]:10.4f}" for nm in cols))


if __name__ == "__main__":
    main()
