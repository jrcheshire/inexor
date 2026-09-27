"""One realization's nonlinear boost against EuclidEmulator2, with its sample variance.

    pixi exec --spec python=3.12 --spec camb --spec matplotlib --spec numpy --spec scipy \
        --spec gsl --spec pip -- bash -c "pip install -q euclidemu2; \
        python scripts/compare/ee2_ratio_figure.py RUN_A/pk_card.json RUN_B/pk_card.json \
        --cosmology RUN_A/export.json \
        --labels '0.25 Mpc/h fine cell' '0.125 Mpc/h fine cell' \
        -o figures/inexor_over_ee2.png"

euclidemu2 is pip-only, and its wheel needs `gsl` in the env to load.

Plots B_inexor / B_EE2 for up to three cards, B = P / P_lin with each side against its
own linear theory (see `pk_boost_reference.py` for why the boost, not P, is compared).
All cards must share k bins and redshift. Two bands around unity:
  - +-1 sigma Gaussian sample variance of ONE realization, sqrt(2 / n_modes) per bin,
    from the first card's mode counts. A floor: non-Gaussian covariance raises it at
    high k.
  - EE2's quoted accuracy, 1% for 0.01 <= k <= 10 h/Mpc at z <= 3
    (Knabenhans et al. 2021, arXiv:2010.11288, abstract).
EE2 (trained on paired-and-fixed sims) carries almost no realization scatter; cards on
the same ICs share theirs, so their difference is read without it.
"""

import argparse
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pk_boost_reference import camb_boost, ee2_boost  # noqa: E402

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

    series = []
    for i, path in enumerate(args.cards):
        with open(path) as fh:
            card = json.load(fh)
        s = card["summary"]
        k = np.asarray(s["k_mean"], float)
        boost = np.asarray(s["p"], float) / np.asarray(s["p_oracle"], float)
        z = 1.0 / float(card["a_out"]) - 1.0
        if i == 0:
            k0, z0 = k, z
            n_modes = np.asarray(s["n_modes"], float)
            # EE2 takes A_s; the engine is sigma8-normalized, so tie them with a CAMB solve
            *_, A_s = camb_boost(cos, z, float(k.max()) * 1.2, "mead2020")
            b_ee2 = ee2_boost(cos, z, k, A_s)
        elif not (np.array_equal(k, k0) and z == z0):
            raise SystemExit(f"{path} has different k bins or redshift from {args.cards[0]}")
        n = round(s["n_particles"] ** (1.0 / 3.0))
        label = (args.labels[i] if args.labels else
                 f"inexor {n}$^3$, {s['box_size']:g} $h^{{-1}}$Mpc, {card['k_steps']} steps")
        series.append((path, label, boost / b_ee2))
    k = k0
    sigma = np.sqrt(2.0 / n_modes)

    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    ax.fill_between(k, 1.0 - sigma, 1.0 + sigma, color="#2a78d6", alpha=0.15, lw=0,
                    label=r"sample variance, $\pm1\sigma$ (Gaussian)")
    ax.axhspan(1.0 - EE2_ACCURACY, 1.0 + EE2_ACCURACY, color="0.35", alpha=0.18, lw=0,
               label="EE2 quoted accuracy (1%)")
    ax.axhline(1.0, color="0.45", lw=1.0)
    for i, (_, label, ratio) in enumerate(series):
        ax.plot(k, ratio, color=COLORS[i], marker=MARKERS[i], ms=3, lw=1.4, label=label)
    ax.set_xscale("log")
    ax.set_xlim(k.min() * 0.95, k.max() * 1.05)
    ax.set_xlabel(r"$k\ \ [h\,{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$B_{\rm inexor}\,/\,B_{\rm EE2}$")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"-> {args.out}")

    heads = " ".join(f"{'card ' + str(i):>8}" for i in range(len(series)))
    print(f"\n  {'k':>7} {'sigma':>7} {heads}")
    for i in range(k.size):
        if i % 4 and k[i] > 0.15 and i != k.size - 1:
            continue
        print(f"  {k[i]:7.3f} {sigma[i]:7.4f} " + " ".join(f"{r[i]:8.4f}" for _, _, r in series))
    for i, (path, _, _) in enumerate(series):
        print(f"  card {i}: {path}")


if __name__ == "__main__":
    main()
