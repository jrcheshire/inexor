"""Plot a P(k) card written by `v2_m6_realization.py card`.

    pixi run python scripts/v2_plot_hero_pk.py runs/v2/d7f_1013309_hero_pk.json \
        [--ics runs/v2/<ic card>.json] -o figures/hero_pk.png

Left panel is the spectrum against the bin-averaged linear oracle the card
carries; right is the ratio over the band where linear theory applies, with the
card's own Gaussian bar, `sqrt(2/n_modes)`, as a shaded envelope -- the same
denominator `z_profile` divides by, so a point outside the band is a |z| > 1.

`--ics` overlays a second card of the same realization at an earlier epoch,
divided by ITS own oracle. Linear evolution preserves mode amplitudes, so the
two ratio curves coincide wherever the evolution is linear, and the difference
between them is what the forty steps did.
"""

import argparse
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# the band the right panel covers. Deliberately well BELOW k_nonlinear:
# nonlinear corrections to P(k) reach several percent long before the linear
# Delta^2 reaches 1, and letting them into the panel costs the y axis the
# resolution the percent-level comparison needs (at 0.30 the curve runs to 1.42
# and the Gaussian band becomes invisible).
K_LINEAR_MAX = 0.15


def _series(path):
    with open(path) as fh:
        card = json.load(fh)
    s = card["summary"]
    k = np.asarray(s["k_mean"], float)
    ratio = np.asarray(s["p"], float) / np.asarray(s["p_oracle"], float)
    sigma = np.sqrt(2.0 / np.asarray(s["n_modes"], float))
    return card, s, k, ratio, sigma


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("card")
    ap.add_argument("--ics", default=None, help="a second card at an earlier epoch")
    ap.add_argument("-o", "--out", default="figures/hero_pk.png")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    card, s, k, ratio, sigma = _series(args.card)
    p = np.asarray(s["p"], float)
    p_lin = np.asarray(s["p_oracle"], float)
    k_nl = s.get("k_nonlinear")
    a_out = card["a_out"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.3))

    ax.loglog(k, p_lin, ls="--", lw=1.4, color="0.45",
              label=r"linear, $D(a)^2P_{\rm lin}$")
    ax.loglog(k, p, ls="-", lw=1.8, color="C0", marker="o", ms=2.6,
              label=f"measured, $a={a_out:g}$")
    if k_nl:
        ax.axvline(k_nl, ls=":", lw=1.2, color="0.25",
                   label=rf"$k_{{\rm nl}}={k_nl:.3f}$")
    ax.set_xlabel(r"$k\ \ [h\,{\rm Mpc}^{-1}]$")
    ax.set_ylabel(r"$P(k)\ \ [h^{-3}{\rm Mpc}^{3}]$")
    ax.legend(frameon=False, fontsize=9)

    sel = k <= K_LINEAR_MAX
    bx.axhline(1.0, lw=1.0, color="0.45", ls="--")
    bx.fill_between(k[sel], 1 - sigma[sel], 1 + sigma[sel], color="C0", alpha=0.18,
                    lw=0, label=r"$\pm\sqrt{2/N_{\rm modes}}$")
    bx.plot(k[sel], ratio[sel], ls="-", lw=1.8, color="C0", marker="o", ms=3.4,
            label=f"$a={a_out:g}$")

    if args.ics:
        icard, _, ki, ri, si = _series(args.ics)
        m = ki <= K_LINEAR_MAX
        bx.fill_between(ki[m], 1 - si[m], 1 + si[m], color="C3", alpha=0.15, lw=0)
        bx.plot(ki[m], ri[m], ls="-.", lw=1.8, color="C3", marker="s", ms=3.4,
                label=f"$a={icard['a_out']:g}$ (ICs)")

    if k_nl and k_nl <= K_LINEAR_MAX:
        bx.axvline(k_nl, ls=":", lw=1.2, color="0.25")
    bx.set_xscale("log")
    # the default log locator puts minor labels on top of each other across
    # this narrow a decade span
    bx.set_xticks([0.02, 0.03, 0.05, 0.07, 0.10, 0.15])
    bx.set_xticklabels(["0.02", "0.03", "0.05", "0.07", "0.10", "0.15"])
    bx.minorticks_off()
    bx.set_xlabel(r"$k\ \ [h\,{\rm Mpc}^{-1}]$")
    bx.set_ylabel(r"$P(k)\,/\,D(a)^2P_{\rm lin}(k)$")
    bx.set_xlim(k.min() * 0.85, K_LINEAR_MAX)
    bx.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"-> {args.out}")

    below = k < (k_nl if k_nl else np.inf)
    z = np.asarray(s["z_profile"], float)
    print(f"   {s['n_particles']:,} particles, n_coarse={s['n_coarse']}, "
          f"box={s['box_size']:g}, {s['n_bins']} bins")
    print(f"   |z| below k_nl: median {np.median(np.abs(z[below])):.2f} "
          f"max {np.max(np.abs(z[below])):.2f}")
    for i in np.where(k <= 0.14)[0]:
        print(f"   k={k[i]:7.4f}  P/P_lin={ratio[i]:6.4f}  z={z[i]:+7.2f}  "
              f"sigma={sigma[i] * 100:5.2f}%")


if __name__ == "__main__":
    main()
