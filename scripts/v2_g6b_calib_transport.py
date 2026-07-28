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

Reads the per-config G6 aggregates (g6_split_stability_<cfg>.json). Sets no pass
threshold -- that is JC's gate call.

READ THE ABSOLUTE RESIDUAL FIRST (2026-07-25). |R_small - R_big| is what applying
the small box's T-bar actually leaves behind, in the same units as D-v2-9's
absolute bar, and the reference it should be read against is the big box's OWN
leave-one-out residual -- the best any calibration achieves at that box. The
fractional and sigma_calib statistics are cross-checks: the transfer's amplitude
varies by an order of magnitude across the band, so dividing by it overstates
disagreement exactly at the low-k end this test is about, and sigma_calib is so
small (the means are cheap to pin) that physically irrelevant offsets show up as
many sigma. The cdev8 -> cdev run read as "23% median, 3.5 sigma_calib" while the
absolute residual was 1.02e-3 against an own-box leave-one-out of 9.9e-4 -- a 3%
difference in the units that decide anything.

READ THE GATED BAND AND THE COMMON-WINDOW INDEX (2026-07-27, from G6c). Two ways
this card misread the 64x rung, both fixed here, both additive so older cards stay
comparable:
  - `band` is the whole k OVERLAP, out to the big box's Nyquist (k=12.6 for
    cgh64), while D-v2-9's bar is written on k <= k_gate = 2.51. Over the full
    overlap the cdev8 -> cgh64 correction reads as a REGRESSION (4.97e-3
    transported vs 4.32e-3 uncorrected); on the gate band it is a 7x improvement
    (1.56e-2 -> 2.23e-3). The excess is high-k ringing a low-k T-bar has no
    business correcting. `gate_band` is now reported and is what to gate on.
  - the low-k index was fit from each box's OWN k_f, so a bigger box fit a wider
    window; the apparent index drift (cdev8 1.44 / cdev 1.54 / cgh64 1.63) is
    mostly that. On the common window they are 1.44 / 1.53 / 1.55, and the 64x
    delta falls 0.20 -> 0.12, inside the 0.15 MATCH criterion.
Also: the own-box leave-one-out these are measured against is itself an N-seed
estimate (cgh64 ran 6 seeds), so a transport ratio at or below ~1 means the
comparison is floor-limited -- a BOUND, not a demonstration that calibrating on a
foreign box beats calibrating locally.

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

    That default makes two configs' indices NOT directly comparable: a bigger box
    reaches lower k, so it fits over a wider window, and any departure from a pure
    power law then shows up as an index difference with no box dependence behind
    it. Pass kmin=<the overlap floor> for the apples-to-apples comparison; the
    own-window fit is what licenses extrapolating each curve below its own reach.
    """
    kmin = kmin if kmin is not None else k.min()
    m = (k >= kmin) & (k <= kmax) & (R < 0)
    if m.sum() < 3:
        return None
    p = np.polyfit(np.log(k[m]), np.log(-R[m]), 1)
    return dict(index=float(p[0]), amp=float(np.exp(p[1])), n_bins=int(m.sum()))


def compare(small, big, k_gate=None):
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

    def stats(m):
        # ABSOLUTE first, deliberately. |diff| IS the residual left behind when the
        # SMALL box's T-bar is applied to the big box, in the same units as
        # D-v2-9's bar; read it against `uncorrected` (the split error the transfer
        # exists to remove) and against the big box's own leave-one-out residual.
        # The fractional statistic divides by a curve whose amplitude varies by an
        # order of magnitude across the band, so it exaggerates disagreement
        # exactly where the transfer is smallest -- which is the low-k end this
        # whole test is about. It is a cross-check, not the headline.
        return dict(
            median_abs_resid=float(np.nanmedian(np.abs(diff)[m])),
            max_abs_resid=float(np.nanmax(np.abs(diff)[m])),
            median_uncorrected=float(np.nanmedian(np.abs(Rb)[m])),
            max_uncorrected=float(np.nanmax(np.abs(Rb)[m])),
            median_frac=float(np.nanmedian(frac[m])),
            max_frac=float(np.nanmax(frac[m])),
            # the difference measured in units of the combined calibration error:
            # ~1 means the two means agree within how well each is pinned.
            median_diff_in_sigma=float(np.nanmedian((np.abs(diff) / comb)[m])),
            max_diff_in_sigma=float(np.nanmax((np.abs(diff) / comb)[m])),
        )

    out = dict(
        k=kc,
        R_small=Rs,
        R_big=Rb,
        sem_small=Es,
        sem_big=Eb,
        diff=diff,
        comb_err=comb,
        overlap=(float(lo), float(hi)),
        # `band` is the WHOLE k overlap, out to the big box's Nyquist -- kept
        # under this name (and first) so every previously recorded card stays
        # comparable. It is NOT the band D-v2-9's bar is written on; see
        # `gate_band` below, which is.
        band=stats(np.ones_like(kc, dtype=bool)),
        low_k=dict(k_max=LOW_K, **stats(low)),
    )
    if k_gate is not None:
        # D-v2-9's bar (absolute |dP/P| <= 3e-2) is written on k <= k_gate. The
        # overlap runs ~5x past that edge, where the transfer is the high-k
        # ringing plateau rather than the coherent low-k suppression a T-bar
        # exists to remove -- so the full-overlap statistic can read as a
        # regression while the gated one improves. Report both; gate on this one.
        out["gate_band"] = dict(k_max=float(k_gate), **stats(kc <= k_gate))
    return out


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
    k_gate = aggs["big"].get("k_gate")
    cmp_ = compare(small, big, k_gate=k_gate)
    # own-window (each box from its own k_f) and common-window (both from the
    # overlap floor) fits. Only the second is a like-for-like shape comparison.
    idx_s = lowk_index(small["k"], small["mean"])
    idx_b = lowk_index(big["k"], big["mean"])
    k_common = cmp_["overlap"][0]
    idx_s_com = lowk_index(small["k"], small["mean"], kmin=k_common)
    idx_b_com = lowk_index(big["k"], big["mean"], kmin=k_common)

    print(f"=== G6b calibration transport: {args.small} -> {args.big} ===")
    print(f"small box: {args.small}  arm {small['arm']}  P={small['P']}  N={small['n']}  "
          f"k_f={small['k'].min():.3f}")
    print(f"big box  : {args.big}  arm {big['arm']}  P={big['P']}  N={big['n']}  "
          f"k_f={big['k'].min():.3f}")
    print(f"overlap  : k in [{cmp_['overlap'][0]:.3f}, {cmp_['overlap'][1]:.3f}]\n")

    lo, bd = cmp_["low_k"], cmp_["band"]
    loo = None
    try:
        loo = float(aggs["big"]["arms"][0]["low_k"]["loo_residual_median"])
    except (KeyError, IndexError, TypeError):
        pass

    print("--- READ FIRST: residual left by applying the SMALL box's T-bar, "
          "ABSOLUTE |dP/P| ---")
    print("  (the units D-v2-9's bar is written in; compare against the big box's "
          "OWN\n   leave-one-out residual, which is the best any calibration can do "
          "at this box)")
    gb = cmp_.get("gate_band")
    if gb is not None:
        print(f"  GATE BAND k<={gb['k_max']:.2f} (D-v2-9's bar, 3.0e-02): uncorrected "
              f"{gb['median_uncorrected']:.3e} -> transported\n"
              f"           {gb['median_abs_resid']:.3e} median  "
              f"(max {gb['max_abs_resid']:.3e})")
    print(f"  k<={LOW_K}: uncorrected {lo['median_uncorrected']:.3e} -> transported "
          f"{lo['median_abs_resid']:.3e} median  (max {lo['max_abs_resid']:.3e})")
    if loo is not None:
        print(f"           {args.big} own-box leave-one-out {loo:.3e} median "
              f"-> transport costs x{lo['median_abs_resid'] / loo:.2f}")
        print(f"           NB that reference is an N={big['n']} leave-one-out, so it "
              f"carries its own\n           sampling floor; a ratio at or below ~1 is a "
              f"BOUND, not a measured margin.")
    print(f"  full overlap (to k={cmp_['overlap'][1]:.1f}, PAST the gate band -- "
          f"cross-check only):\n"
          f"           uncorrected {bd['median_uncorrected']:.3e} -> transported "
          f"{bd['median_abs_resid']:.3e} median  (max {bd['max_abs_resid']:.3e})")

    print("\n--- cross-check: the same agreement in fractional and sigma_calib units ---")
    print("  (fractional divides by a curve that varies by an order of magnitude "
          "across\n   the band, so it overstates disagreement where the transfer is "
          "small)")
    if gb is not None:
        print(f"  gate k<={gb['k_max']:.2f}: {gb['median_frac']:.1%} median, "
              f"{gb['max_frac']:.1%} max  |  {gb['median_diff_in_sigma']:.1f} "
              "sigma_calib median")
    print(f"  k<={LOW_K}: {lo['median_frac']:.1%} median, {lo['max_frac']:.1%} max  |  "
          f"{lo['median_diff_in_sigma']:.1f} sigma_calib median")
    print(f"  full overlap : {bd['median_frac']:.1%} median, {bd['max_frac']:.1%} max  |  "
          f"{bd['median_diff_in_sigma']:.1f} sigma_calib median")
    if idx_s_com and idx_b_com:
        print("\n--- low-k power-law index (governs extrapolation below the small k_f) ---")
        print(f"  COMMON window k in [{k_common:.3f}, {LOW_K}] -- the like-for-like "
              "comparison:")
        print(f"    {args.big:6s}: R ~ k^{idx_b_com['index']:.2f} ({idx_b_com['n_bins']} bins)"
              f"   {args.small:6s}: R ~ k^{idx_s_com['index']:.2f} ({idx_s_com['n_bins']} bins)")
        d_com = abs(idx_s_com["index"] - idx_b_com["index"])
        print(f"    {'MATCH' if d_com < 0.15 else 'DIFFER'} (delta index {d_com:.2f})")
    if idx_s and idx_b:
        d_own = abs(idx_s["index"] - idx_b["index"])
        print("  own-window (each box from its own k_f; sets each curve's own "
              "extrapolation,\n  but the windows differ so the delta is NOT a pure "
              "box effect):")
        print(f"    {args.big:6s}: R ~ k^{idx_b['index']:.2f} ({idx_b['n_bins']} bins)"
              f"   {args.small:6s}: R ~ k^{idx_s['index']:.2f} ({idx_s['n_bins']} bins)"
              f"   delta {d_own:.2f}")

    print("\nVerdict is JC's: transport is established if the ABSOLUTE residual left by "
          "the\nsmall box's T-bar is comparable to the big box's own leave-one-out "
          "residual over\nk <= 0.5 AND the low-k indices match on the COMMON window. "
          "No threshold is\nhard-coded here.")

    out_prefix = args.out_prefix or f"g6b_transport_{args.small}_{args.big}"
    out = dict(
        small=args.small, big=args.big,
        n_small=small["n"], n_big=big["n"],
        overlap=cmp_["overlap"],
        # `band` (= full overlap) and the own-window indices keep their original
        # names and meanings so cards recorded before 2026-07-27 stay comparable.
        band=cmp_["band"], low_k=cmp_["low_k"],
        gate_band=cmp_.get("gate_band"),
        index_small=idx_s, index_big=idx_b,
        index_small_common=idx_s_com, index_big_common=idx_b_com,
        index_common_kmin=float(k_common),
        note="Box transport at FIXED cell/coarse-mesh. Resolution axis untested. "
        "Read gate_band (D-v2-9's bar) and the common-window indices; `band` is the "
        "full overlap and runs past the gate edge. The own-box leave-one-out this is "
        "compared against carries its own N-seed sampling floor. "
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
