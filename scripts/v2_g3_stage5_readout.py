"""G3 Stage 5 readout -- reproduces every number in runs/v2/g3_stage5_record.md.

Prints only: no JSON, no figure, so the record and this script cannot drift into
two different answers. `--level` swaps the correlation threshold the eligibility
map is drawn at, which matters because the geometry ranking is threshold-
dependent (the T advantage is 1.297 at r=0.5 and 1.034 at r=0.9) and a gate
written on one does not select the geometry the other would.

THE ENSEMBLE COMPARISON IS PAIRED, and that is not a style choice. Arms share
initial conditions, so ranking them WITHIN a seed cancels cosmic variance
(measured 800-2000x for the bispectrum ratio at cdev, 100-500x for P(k) in G6).
The absolute k(r=0.5) scatters 12-22% seed to seed while the paired ratio
scatters 12.5% and never crosses 1 -- an unpaired reading dissolves the ranking.
Never average the arms separately and then divide.

Usage:
    pixi run python scripts/v2_g3_stage5_readout.py
    pixi run python scripts/v2_g3_stage5_readout.py --level 0.9
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "runs", "v2")


def load(cfg):
    """Pilot card plus every seed card present, newest glob order irrelevant."""
    cards = []
    p0 = os.path.join(OUT_DIR, f"g3_stage5_{cfg}.json")
    if os.path.exists(p0):
        with open(p0) as f:
            cards.append(json.load(f))
    for p in sorted(glob.glob(os.path.join(OUT_DIR, f"g3_stage5_{cfg}_seed*.json"))):
        with open(p) as f:
            cards.append(json.load(f))
    if not cards:
        sys.exit(f"no g3_stage5_{cfg}*.json in {OUT_DIR} -- pull them from the run host")
    return cards


def log_stats(vals):
    """Geometric mean with 1sd and sem bands, for RATIOS.

    A ratio aggregated linearly is not the same statistic in both directions;
    the log is what makes "does it bracket 1.0" a symmetric question.
    """
    v = np.asarray([x for x in vals if np.isfinite(x) and x > 0], dtype=np.float64)
    if v.size == 0:
        return None
    lg = np.log(v)
    m, sd = lg.mean(), lg.std(ddof=1) if v.size > 1 else 0.0
    sem = sd / np.sqrt(v.size) if v.size > 1 else 0.0
    return dict(n=int(v.size), gm=float(np.exp(m)),
                sd_lo=float(np.exp(m - sd)), sd_hi=float(np.exp(m + sd)),
                sem_lo=float(np.exp(m - sem)), sem_hi=float(np.exp(m + sem)),
                frac_sd=float(sd), n_gt1=int((v > 1).sum()))


def section_eligibility(cards, level):
    print(f"\n=== 1. ELIGIBILITY: r at k_short, threshold {level} ===")
    ks = cards[0]["gate"]["k_short_mults"]
    kf = 2.0 * np.pi / cards[0]["geometry"]["L"]
    print("    arm               b[Mpc/h]  cost   " + "".join(f"{k * kf:>10.3f}" for k in ks))
    for arm in [r["arm"] for r in cards[0]["rows"]]:
        cells, meta = [], None
        for k in ks:
            vals = []
            for c in cards:
                row = next((r for r in c["rows"] if r["arm"] == arm), None)
                if row is None:
                    continue
                meta = meta or row
                vals.append(row["cells"][str(k)]["r_k_short"])
            mean = float(np.mean(vals)) if vals else np.nan
            cells.append(f"{mean:>9.3f}" + ("*" if mean >= level else " "))
        print(f"    {arm:16s} {meta['b_mpc']:7.1f} {meta['vol_ratio']:6.2f}x " + "".join(cells))
    print(f"    ({len(cards)} card(s); * = gradeable, mean over seeds)")


def section_verdict(cards, level):
    print(f"\n=== 2. MAX |R_Q| over gate-eligible squeezed triangles (bar 0.15, r >= {level}) ===")
    ks = cards[0]["gate"]["k_short_mults"]
    for arm in [r["arm"] for r in cards[0]["rows"]]:
        out = []
        for k in ks:
            vals, rho_ok = [], True
            for c in cards:
                row = next((r for r in c["rows"] if r["arm"] == arm), None)
                if row is None:
                    continue
                cell = row["cells"][str(k)]
                if cell["r_k_short"] >= level and "max_abs_R_Q" in cell:
                    vals.append(cell["max_abs_R_Q"])
                    rho_ok = rho_ok and cell.get("rho_agrees", False)
            if vals:
                sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                out.append(f"{np.mean(vals):>7.4f}+-{sd:<7.4f}{'' if rho_ok else '!rho'}")
            else:
                out.append(f"{'--':>7s}          ")
        print(f"    {arm:16s} " + " ".join(out))
    print("    -- = no gradeable cell. NEVER a pass: it means the statistic cannot")
    print("       speak there, not that the tiling agreed.")


def section_paired(cards, level):
    """The buffer-vs-tile null, ranked WITHIN each seed."""
    print("\n=== 3. PAIRED: same absolute buffer, different tile size ===")
    if len(cards) < 2:
        print("    needs >= 2 seed cards; skipped")
        return
    by_b = {}
    for row in cards[0]["rows"]:
        by_b.setdefault(round(row["b_mpc"], 3), []).append(row["arm"])
    for b_mpc, arms in sorted(by_b.items()):
        if len(arms) < 2:
            continue
        a, bb = arms[0], arms[1]
        ratios = []
        for c in cards:
            ra = next((r for r in c["rows"] if r["arm"] == a), None)
            rb = next((r for r in c["rows"] if r["arm"] == bb), None)
            if ra and rb:
                ratios.append(ra["k_usable"][str(level)] / rb["k_usable"][str(level)])
        st = log_stats(ratios)
        if st:
            print(f"    b = {b_mpc:5.1f} Mpc/h   k(r={level}) {a} / {bb}: gm {st['gm']:.3f} "
                  f"1sd [{st['sd_lo']:.3f}, {st['sd_hi']:.3f}] sem "
                  f"[{st['sem_lo']:.3f}, {st['sem_hi']:.3f}]  {st['n_gt1']}/{st['n']} > 1")
    print("    A ratio whose sem band brackets 1.0 is a NULL, and a null here is the")
    print("    finding: absolute buffer sets the physics, tile size sets the price.")


def section_response(cards):
    print("\n=== 4. Position-dependent P(k): tiled/mono response ratio - 1 ===")
    print("    REPORTED, NOT GATED -- no bar for this statistic has been ratified.")
    for arm in [r["arm"] for r in cards[0]["rows"]]:
        for pr0 in cards[0]["rows"][0]["response"]:
            n_sub = pr0["n_sub"]
            acc, ks, strad = [], None, None
            for c in cards:
                row = next((r for r in c["rows"] if r["arm"] == arm), None)
                pr = next((p for p in (row or {}).get("response", []) if p["n_sub"] == n_sub),
                          None)
                if pr:
                    acc.append(pr["ratio"])
                    ks, strad = pr["k_centers"], pr["straddle_frac"]
            if acc:
                m = np.mean(np.asarray(acc), axis=0)
                print(f"    {arm:16s} n_sub={n_sub} straddle={strad:.3f}  "
                      + "  ".join(f"k={k:.2f}: {v:+.4f}" for k, v in zip(ks, m)))


def section_brackets(cards):
    print("\n=== 5. BRACKETS (on r, never on R_Q) ===")
    c = cards[0]
    if "brackets" not in c:
        print("    absent from the pilot card; skipped")
        return
    sh = c["brackets"]["shell_r"]
    piv, kil, spn = sh["pivot"], sh["kill_control"], sh["span_check"]
    print("      k        pivot     kill(b=0)   2LPT")
    for i, k in enumerate(piv["centers"]):
        print(f"    {k:7.3f}  {piv['r'][i]:9.4f}  {kil['r'][i]:9.4f}  {spn['r'][i]:9.4f}")
    ok_kill = all(k <= p + 1e-12 for k, p in zip(kil["r"], piv["r"]))
    ok_span = all(s >= p - 1e-12 for s, p in zip(spn["r"], piv["r"]))
    print(f"    kill control less correlated than the pivot at every shell: {ok_kill}")
    print(f"    2LPT more correlated than the pivot at every shell:         {ok_span}")
    print("    R_Q cannot express this: it saturates once the arms decorrelate and is")
    print("    NOT monotone in brokenness, so the criterion is built on r.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="cdev")
    ap.add_argument("--level", type=float, default=0.5, choices=(0.9, 0.5, 0.2))
    args = ap.parse_args()

    cards = load(args.config)
    print(f"=== G3 Stage 5 readout: {args.config}, {len(cards)} card(s), "
          f"seeds {[c['seed'] for c in cards]} ===")
    section_eligibility(cards, args.level)
    section_verdict(cards, args.level)
    section_paired(cards, args.level)
    section_response(cards)
    section_brackets(cards)
    print("\n=== CAVEATS ===")
    print("  - Wall times are MACHINE-dependent (7.4x on Vista vs 4.0x on deneb for the")
    print("    same box and configs). Compare wall only within a host.")
    print("  - deneb is correctness ground (D-v2-10): no cost or memory number from")
    print("    these cards is a Pareto point. Those come from the config-table homes.")
    print("  - This is the MECHANISM ensemble at cdev, not the A3 geometry verdict.")
    print("    The verdict point is cgh64, and the two have different jobs.")


if __name__ == "__main__":
    main()
