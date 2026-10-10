"""One realization's P(k) against EuclidEmulator2's, with its sample variance.

    pixi exec --spec python=3.12 --spec camb --spec matplotlib --spec numpy --spec scipy \
        --spec gsl --spec pip -- bash -c "pip install -q euclidemu2; \
        python scripts/compare/ee2_ratio_figure.py RUN_A/realization_pk.json RUN_B/realization_pk.json \
        --cosmology RUN_A/export.json \
        --labels '0.25 Mpc/h fine cell' '0.125 Mpc/h fine cell' \
        -o figures/inexor_over_ee2.png"

euclidemu2 is pip-only, and its wheel needs `gsl` in the env to load.

Plots P_inexor / P_EE2 for up to three cards, with P_EE2 = P_lin,CAMB x B_EE2 averaged over
each card bin's own lattice modes, the average the card takes of its own P. Both sides are
bin averages: EE2 at a bin's mean k instead would leave the BAO wiggles (each bin spans ~40%
of a period) as a +-1-2% zigzag. The bins are rebuilt from the card's mesh and edges, and the
script refuses a card whose mode counts it does not reproduce. P_lin is CAMB's (EE2's own is a
CLASS solve) at the A_s that matches the cosmology's sigma8; a card whose ICs were EH98 shows
EH98's linear-spectrum difference from CAMB where its P is still linear.
EE2 is evaluated at each card's own k and redshift, so cards from different boxes may be
overlaid; all must share the `--cosmology`. Two kinds of band around unity:
  - +-1 sigma Gaussian sample variance of ONE realization, sqrt(2 / n_modes) per bin,
    one band per distinct k grid (from its first card's mode counts), in that card's
    color. A floor: non-Gaussian covariance raises it at high k.
  - EE2's quoted accuracy, 1% for 0.01 <= k <= 10 h/Mpc at z <= 3
    (Knabenhans et al. 2021, arXiv:2010.11288, abstract).
EE2 (trained on paired-and-fixed sims) carries almost no realization scatter; cards on
the same ICs share theirs, so their difference is read without it. Cards on different
ICs differ by both bands' scatter.
"""

import argparse
import json

import numpy as np
from scipy.signal import fftconvolve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pk_boost_reference import camb_boost, ee2_boost  # noqa: E402

EE2_ACCURACY = 0.01
COLORS = ["#2a78d6", "#eb6834", "#1b9e77"]
MARKERS = ["o", "s", "^"]


def card_bin_modes(summary):
    """Each of the card's bins as lattice shells: a list, parallel to `k_mean`, of
    (k, count) arrays, k = kf sqrt(|n|^2) and count the full-grid modes of that shell in
    the bin.

    Mirrors `inexor.summary.binned_power_partials`: |k| = sqrt((kx^2 + ky^2) + kz^2) with
    k_i = 2 pi * fftfreq, digitized against the card's edges. A shell on an edge is split
    vector by vector with that same float expression, so rounding sends each mode where the
    card sent it. Refuses unless every bin's count equals the card's `n_modes`.
    """
    n = int(summary["n_coarse"])
    box = float(summary["box_size"])
    edges = np.asarray(summary["k_edges"], float)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cards", nargs="+")
    ap.add_argument("--cosmology", required=True,
                    help="a JSON file carrying a `cosmology` block (the export header has one)")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="legend labels, one per card (default: built from each card)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    if len(args.cards) > len(COLORS):
        raise SystemExit(f"at most {len(COLORS)} cards")
    if args.labels and len(args.labels) != len(args.cards):
        raise SystemExit("--labels needs one label per card")
    with open(args.cosmology) as fh:
        cos = json.load(fh)["cosmology"]

    cards = []
    for path in args.cards:
        with open(path) as fh:
            cards.append(json.load(fh))
    k_top = max(c["summary"]["k_edges"][-1] for c in cards)
    # EE2 takes A_s; the engine is sigma8-normalized, so tie them with a CAMB solve, whose
    # P_lin (fine in k, to resolve the BAO before bin-averaging) is the EE2 side's
    lin = {}
    for card in cards:
        z = 1.0 / float(card["a_out"]) - 1.0
        if z not in lin:
            kh, plin, _, _, A_s = camb_boost(cos, z, k_top * 1.2, "mead2020", npoints=8000)
            lin[z] = (kh, plin, A_s)

    series = []
    grids = []  # [(k, sigma, color, [series index, ...])], one per distinct k grid
    for i, (path, card) in enumerate(zip(args.cards, cards)):
        s = card["summary"]
        k = np.asarray(s["k_mean"], float)
        z = 1.0 / float(card["a_out"]) - 1.0
        kh, plin, A_s = lin[z]
        bins = card_bin_modes(s)
        k_all = np.unique(np.concatenate([ks for ks, _ in bins]))
        p_all = (np.exp(np.interp(np.log(k_all), np.log(kh), np.log(plin)))
                 * ee2_boost(cos, z, k_all, A_s))
        p_ee2 = np.array([(np.interp(ks, k_all, p_all) * cs).sum() / cs.sum()
                          for ks, cs in bins])
        n = round(s["n_particles"] ** (1.0 / 3.0))
        src = (s.get("linear_pk") or {}).get("source", "eh98").upper()
        label = (args.labels[i] if args.labels else
                 f"inexor {n}$^3$, {s['box_size']:g} $h^{{-1}}$Mpc, "
                 f"{card.get('k_steps', '?')} steps, {src} ICs")
        series.append((path, label, k, np.asarray(s["p"], float) / p_ee2))
        for grid in grids:
            if np.array_equal(grid[0], k):
                grid[3].append(i)
                break
        else:
            sigma = np.sqrt(2.0 / np.asarray(s["n_modes"], float))
            grids.append((k, sigma, COLORS[i], [i]))

    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    for j, (k, sigma, color, _) in enumerate(grids):
        ax.fill_between(k, 1.0 - sigma, 1.0 + sigma, color=color, alpha=0.15, lw=0,
                        label=r"sample variance, $\pm1\sigma$ (Gaussian)" if j == 0 else None)
    ax.axhspan(1.0 - EE2_ACCURACY, 1.0 + EE2_ACCURACY, color="0.35", alpha=0.18, lw=0,
               label="EE2 quoted accuracy (1%)")
    ax.axhline(1.0, color="0.45", lw=1.0)
    for i, (_, label, k, ratio) in enumerate(series):
        ax.plot(k, ratio, color=COLORS[i], marker=MARKERS[i], ms=3, lw=1.4, label=label)
    ax.set_xscale("log")
    k_lo = min(k.min() for k, *_ in grids)
    k_hi = max(k.max() for k, *_ in grids)
    ax.set_xlim(k_lo * 0.95, k_hi * 1.05)
    ax.set_xlabel(r"$k\ \ [h\,{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$P_{\rm inexor}\,/\,P_{\rm EE2}$")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"-> {args.out}")

    for k, sigma, _, members in grids:
        heads = " ".join(f"{'card ' + str(i):>8}" for i in members)
        print(f"\n  {'k':>7} {'sigma':>7} {heads}")
        for b in range(k.size):
            if b % 4 and k[b] > 0.15 and b != k.size - 1:
                continue
            print(f"  {k[b]:7.3f} {sigma[b]:7.4f} "
                  + " ".join(f"{series[i][3][b]:8.4f}" for i in members))
    for i, (path, *_) in enumerate(series):
        print(f"  card {i}: {path}")


if __name__ == "__main__":
    main()
