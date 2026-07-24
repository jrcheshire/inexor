"""v2 G6b: does the split-error transfer TRANSPORT across box size?

The compute question behind G6 (JC, 2026-07-24): a low-k correction is only
cheap if the transfer T-bar(k) can be CALIBRATED ONCE on a small box and applied
to bigger production boxes -- because the only memory-expensive object anywhere
in the scheme is the MONOLITHIC reference, and it is needed ONLY during
calibration. If the transfer converges at a small box, calibration is cheap even
for a cosmology grid; if it drifts with box size, calibration inherits the
production box's cost and the ballooning worry is real.

The config table (D-v2-8) is built for exactly this test: cdev8 / cdev / cgh64
hold the fine CELL (0.25), the coarse mesh, and the gate band FIXED and vary only
VOLUME (1x / 8x / 64x). So comparing their ensemble-mean transfers is a pure
box-transport measurement at fixed resolution. No matched-phase subtlety across
boxes: the transfer is an ENSEMBLE-MEAN property, so each config's T-bar(k) is
self-contained (its own tiled-vs-mono ratio) and the comparison is curve-vs-curve
over the overlapping k range. The small box cannot probe below its own
fundamental (cdev8 k_f = 0.098 vs cdev 0.049), so the low-k POWER-LAW INDEX is
reported too: a shared index is what licenses extrapolating T-bar below the small
box's reach.

RESOLUTION (changing the fine cell) is a DIFFERENT axis and NOT tested here --
that asks whether you must re-calibrate when you change the production cell, a
rare event; this asks whether you can calibrate cheaply at the production cell.

Reads the per-config G6 aggregates (g6_split_stability_<cfg>.json). Reports the
overlap agreement against the COMBINED calibration error (sigma/sqrt(N) of each
mean); sets no pass threshold -- that is JC's gate call.

Usage:
  pixi run python scripts/v2_g6b_calib_transport.py --small cdev8 --big cdev
"""

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
LOW_K = 0.5  # h/Mpc -- the band D-v2-10 quotes the coherent suppression on


def load_agg(d, cfg, prefix):
    path = os.path.join(d, f"{prefix}_{cfg}.json")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run v2_g6_split_stability.py --config {cfg} first")
    with open(path) as fh:
        return json.load(fh)


def pivot_curve(agg):
    """The first plain arm's ensemble mean R(k) and its calibration error.

    P-independence (G6, both machines) means the arm choice does not matter for
    the low-k transfer; the first arm is the pivot.
    """
    a = agg["arms"][0]
    k = np.asarray(a["k"], float)
    mean = np.asarray(a["mean"], float)
    sd = np.asarray(a["sd"], float)
    n = int(a["n_seeds"])
    return dict(k=k, mean=mean, sem=sd / np.sqrt(n), n=n, arm=a["arm"], P=a["padded_P"])


def lowk_index(k, R, kmax=LOW_K, kmin=None):
    """Fit R(k) = -A k^n over the low-k band; the index that governs extrapolation.

    R is the (negative) coherent suppression; fit log|R| vs log k. kmin defaults
    to the curve's own fundamental so each config is fit over its real reach.
    """
    kmin = kmin if kmin is not None else k.min()
    m = (k >= kmin) & (k <= kmax) & (R < 0)
    if m.sum() < 3:
        return None
    p = np.polyfit(np.log(k[m]), np.log(-R[m]), 1)
    return dict(index=float(p[0]), amp=float(np.exp(p[1])), n_bins=int(m.sum()))


def compare(small, big):
    ks, kb = small["k"], big["k"]
    lo, hi = max(ks.min(), kb.min()), min(ks.max(), kb.max())
    band = (kb >= lo) & (kb <= hi)
    kc = kb[band]
    # interpolate the small-box mean (and its error) onto the big-box grid over
    # the overlap; linear in k on R (R is smooth and monotone at low k).
    Rs = np.interp(kc, ks, small["mean"])
    Es = np.interp(kc, ks, small["sem"])
    Rb = big["mean"][band]
    Eb = big["sem"][band]
    diff = Rs - Rb
    comb = np.sqrt(Es**2 + Eb**2)  # combined calibration error on the difference
    denom = np.where(np.abs(Rb) > 0, np.abs(Rb), np.nan)
    frac = np.abs(diff) / denom
    low = kc <= LOW_K
    return dict(
        k=kc,
        R_small=Rs,
        R_big=Rb,
        sem_small=Es,
        sem_big=Eb,
        diff=diff,
        comb_err=comb,
        overlap=(float(lo), float(hi)),
        band=dict(
            median_frac=float(np.nanmedian(frac)),
            max_frac=float(np.nanmax(frac)),
            # the difference measured in units of the combined calibration error:
            # ~1 means the two means agree within how well each is pinned.
            median_diff_in_sigma=float(np.nanmedian(np.abs(diff) / comb)),
            max_diff_in_sigma=float(np.nanmax(np.abs(diff) / comb)),
        ),
        low_k=dict(
            k_max=LOW_K,
            median_frac=float(np.nanmedian(frac[low])),
            max_frac=float(np.nanmax(frac[low])),
            median_diff_in_sigma=float(np.nanmedian((np.abs(diff) / comb)[low])),
        ),
    )


def make_figure(small, big, cmp_, idx_s, idx_b, out_png, labels):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1]))
    ax = axes[0]
    for cur, lab, c in ((big, labels[1], "C0"), (small, labels[0], "C1")):
        ax.plot(cur["k"], cur["mean"], color=c, lw=1.6, label=f"{lab} (N={cur['n']})")
        ax.fill_between(cur["k"], cur["mean"] - cur["sem"], cur["mean"] + cur["sem"],
                        color=c, alpha=0.3, lw=0)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axvline(LOW_K, color="k", ls=":", lw=0.8)
    ax.axvspan(*cmp_["overlap"], color="0.9", zorder=0)
    ax.set_xscale("log")
    ax.set_ylabel(r"$\bar R(k) = \langle \Delta P/P\rangle$  (tiled $-$ mono)")
    ax.legend(frameon=False, fontsize=9)
    if idx_s and idx_b:
        ax.text(0.03, 0.05,
                f"low-$k$ index: {labels[1]} {idx_b['index']:.2f}, {labels[0]} {idx_s['index']:.2f}",
                transform=ax.transAxes, fontsize=8)

    ax = axes[1]
    ax.plot(cmp_["k"], cmp_["diff"], color="C3", lw=1.4, label="small $-$ big")
    ax.fill_between(cmp_["k"], -cmp_["comb_err"], cmp_["comb_err"], color="0.7", alpha=0.5, lw=0,
                    label=r"$\pm$ combined calib. error")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axvline(LOW_K, color="k", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$  [$h\,\mathrm{Mpc}^{-1}$]")
    ax.set_ylabel(r"$\bar R_{\rm small} - \bar R_{\rm big}$")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"\nwrote {out_png}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--small", default="cdev8")
    ap.add_argument("--big", default="cdev")
    ap.add_argument("--dir", default=os.path.join(REPO, "runs", "v2"))
    ap.add_argument("--prefix", default="g6_split_stability")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    aggs = dict(
        small=load_agg(args.dir, args.small, args.prefix),
        big=load_agg(args.dir, args.big, args.prefix),
    )
    small = pivot_curve(aggs["small"])
    big = pivot_curve(aggs["big"])
    cmp_ = compare(small, big)
    idx_s = lowk_index(small["k"], small["mean"])
    idx_b = lowk_index(big["k"], big["mean"])

    print(f"=== G6b calibration transport: {args.small} -> {args.big} ===")
    print(f"small box: {args.small}  arm {small['arm']}  P={small['P']}  N={small['n']}  "
          f"k_f={small['k'].min():.3f}")
    print(f"big box  : {args.big}  arm {big['arm']}  P={big['P']}  N={big['n']}  "
          f"k_f={big['k'].min():.3f}")
    print(f"overlap  : k in [{cmp_['overlap'][0]:.3f}, {cmp_['overlap'][1]:.3f}]\n")

    lo, bd = cmp_["low_k"], cmp_["band"]
    print("--- agreement of the two ensemble-mean transfers over the overlap ---")
    print(f"  in-band : {bd['median_frac']:.1%} median, {bd['max_frac']:.1%} max  |  "
          f"{bd['median_diff_in_sigma']:.1f} sigma_calib median")
    print(f"  k<={LOW_K}: {lo['median_frac']:.1%} median, {lo['max_frac']:.1%} max  |  "
          f"{lo['median_diff_in_sigma']:.1f} sigma_calib median")
    if idx_s and idx_b:
        print("\n--- low-k power-law index (governs extrapolation below the small k_f) ---")
        print(f"  {args.big:6s}: R ~ k^{idx_b['index']:.2f}   {args.small:6s}: R ~ k^{idx_s['index']:.2f}")
        print(f"  {'MATCH' if abs(idx_s['index'] - idx_b['index']) < 0.15 else 'DIFFER'} "
              f"(delta index {abs(idx_s['index'] - idx_b['index']):.2f})")

    print("\nVerdict is JC's: transport is established if the means agree to within the "
          "combined calibration error (~few sigma_calib) over k <= 0.5 AND the low-k "
          "indices match. No threshold is hard-coded here.")

    out_prefix = args.out_prefix or f"g6b_transport_{args.small}_{args.big}"
    out = dict(
        small=args.small, big=args.big,
        n_small=small["n"], n_big=big["n"],
        overlap=cmp_["overlap"],
        band=cmp_["band"], low_k=cmp_["low_k"],
        index_small=idx_s, index_big=idx_b,
        note="Box transport at FIXED cell/coarse-mesh. Resolution axis untested. "
        "Verdict (transport established?) is JC's gate call.",
    )
    out_json = os.path.join(args.dir, f"{out_prefix}.json")
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {out_json}")
    make_figure(small, big, cmp_, idx_s, idx_b,
                os.path.join(args.dir, f"{out_prefix}.png"), labels=(args.small, args.big))


if __name__ == "__main__":
    main()
