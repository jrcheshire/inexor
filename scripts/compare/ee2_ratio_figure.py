"""One realization's P(k) against EuclidEmulator2's, with its sample variance.

    pixi exec --spec python=3.12 --spec camb --spec matplotlib --spec numpy --spec scipy \
        --spec gsl --spec pip -- bash -c "pip install -q euclidemu2; \
        python scripts/compare/ee2_ratio_figure.py RUN_A/realization_pk.json RUN_B/realization_pk.json \
        --cosmology RUN_A/export.json \
        --labels '0.25 Mpc/h fine cell' '0.125 Mpc/h fine cell' \
        -o figures/inexor_over_ee2.png"

euclidemu2 is pip-only, and its wheel needs `gsl` in the env to load.

Plots P_inexor / P_EE2 for up to three cards, with P_EE2 = P_lin,CAMB x B_EE2 averaged over
each card bin's own lattice modes (`pk_boost_reference.card_bin_modes`), the average the card
takes of its own P. Both sides are bin averages: EE2 at a bin's mean k instead would leave the
BAO wiggles (each bin spans ~40% of a period) as a +-1-2% zigzag. The bins are rebuilt from
the card's mesh and edges, and a card whose mode counts are not reproduced is refused.
P_lin is CAMB's (EE2's own is a CLASS solve) at the A_s that matches the cosmology's sigma8;
a card whose ICs were EH98 shows EH98's linear-spectrum difference from CAMB where its P is
still linear. EE2 is evaluated at each card's own k and redshift, so cards from different boxes may be
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pk_boost_reference import (  # noqa: E402
    bin_average, camb_boost, card_bin_modes, ee2_boost, loglog)

EE2_ACCURACY = 0.01
COLORS = ["#2a78d6", "#eb6834", "#1b9e77"]
MARKERS = ["o", "s", "^"]


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
        p_lin = loglog(kh, plin)
        p_ee2 = bin_average(card_bin_modes(s), lambda q: p_lin(q) * ee2_boost(cos, z, q, A_s))
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
