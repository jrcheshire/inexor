"""One realization's nonlinear boost against EuclidEmulator2, with its sample variance.

    pixi exec --spec camb --spec euclidemu2 --spec matplotlib --spec numpy -- \
        python scripts/v2_ee2_ratio_figure.py runs/v2/lcw_1023904_k120.json \
        --cosmology runs/v2/d7g_1011376_export.json -o figures/inexor_512_over_ee2.png

Plots B_inexor / B_EE2 with B = P / P_lin, each side against its own linear
theory (see `v2_pk_boost_reference.py` for why the boost, not P, is compared).
Two bands around unity:
  - +-1 sigma Gaussian sample variance of ONE realization, sqrt(2 / n_modes) per
    bin from the card. A floor: non-Gaussian covariance raises it at high k.
  - EE2's quoted accuracy, 1% for 0.01 <= k <= 10 h/Mpc at z <= 3
    (Knabenhans et al. 2021, arXiv:2010.11288, abstract).
EE2 is trained on paired-and-fixed simulations, so its boost carries almost no
realization scatter; a single Gaussian realization carries all of it.
"""

import argparse
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from v2_pk_boost_reference import camb_boost, ee2_boost  # noqa: E402

EE2_ACCURACY = 0.01


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("card")
    ap.add_argument("--cosmology", required=True,
                    help="a JSON file carrying a `cosmology` block (the export header has one)")
    ap.add_argument("--label", default=None, help="legend label for the measured curve")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    with open(args.card) as fh:
        card = json.load(fh)
    with open(args.cosmology) as fh:
        cos = json.load(fh)["cosmology"]
    s = card["summary"]
    k = np.asarray(s["k_mean"], float)
    n_modes = np.asarray(s["n_modes"], float)
    boost = np.asarray(s["p"], float) / np.asarray(s["p_oracle"], float)
    z = 1.0 / float(card["a_out"]) - 1.0

    # EE2 takes A_s; the engine is sigma8-normalized, so tie them with a CAMB solve
    *_, A_s = camb_boost(cos, z, float(k.max()) * 1.2, "mead2020")
    ratio = boost / ee2_boost(cos, z, k, A_s)
    sigma = np.sqrt(2.0 / n_modes)

    n = round(s["n_particles"] ** (1.0 / 3.0))
    label = args.label or (f"inexor {n}$^3$, {s['box_size']:g} $h^{{-1}}$Mpc, "
                           f"{card['k_steps']} steps")

    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    ax.fill_between(k, 1.0 - sigma, 1.0 + sigma, color="#2a78d6", alpha=0.15, lw=0,
                    label=r"sample variance, $\pm1\sigma$ (Gaussian)")
    ax.axhspan(1.0 - EE2_ACCURACY, 1.0 + EE2_ACCURACY, color="0.35", alpha=0.18, lw=0,
               label="EE2 quoted accuracy (1%)")
    ax.axhline(1.0, color="0.45", lw=1.0)
    ax.plot(k, ratio, color="#2a78d6", marker="o", ms=3, lw=1.4, label=label)
    ax.set_xscale("log")
    ax.set_xlim(k.min() * 0.95, k.max() * 1.05)
    ax.set_xlabel(r"$k\ \ [h\,{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$B_{\rm inexor}\,/\,B_{\rm EE2}$")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"-> {args.out}")

    pull = (ratio - 1.0) / sigma
    print(f"\n  {'k':>7} {'ratio':>7} {'sigma':>7} {'pull':>6}")
    for i in range(k.size):
        if i % 4 and k[i] > 0.15:
            continue
        print(f"  {k[i]:7.3f} {ratio[i]:7.4f} {sigma[i]:7.4f} {pull[i]:6.2f}")


if __name__ == "__main__":
    main()
