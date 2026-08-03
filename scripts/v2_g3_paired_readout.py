"""G3 -- the PAIRED readout of the cdev decorrelation ensemble and the box ladder.

THE DECISION THIS EXISTS FOR. `v2_g3_decorrelation_map.py` measures k_usable for
cost-controlled (T, b) pairs, and at cdev8 it said T=128 buys ~1.8x the usable k
of T=64 at the same price. Two things stood between that and a Stage 5 geometry:

  1. THE CONFOUND IS EXACT. At fixed cost, cost = (P/T)^3 fixes b/T, so
     P/n_fine = T(1 + 2b/T)/n_fine is EXACTLY proportional to T. Tile size and
     box fraction are ONE variable inside a single box; only the box ladder
     (cdev8 -> cdev -> cgh64, same fine cell, 8x volume per rung) separates them.
  2. SINGLE-SEED RANKINGS ARE NOT RANKINGS. Job 297 at cdev showed a ~20%
     apparent non-monotonicity in T at fixed ABSOLUTE buffer, which no mechanism
     explains and which is the size of the 29% effect being ranked.

WHY THE COMPARISON MUST BE PAIRED. Every seed's arms share initial conditions, so
a WITHIN-seed ratio cancels cosmic variance and an across-seed ratio of
independent means throws that cancellation away. The absolute k(r=0.5) scatters
12-22% seed to seed; the paired ratio is what carries the ranking. This script
therefore ranks configurations inside each seed first and only then aggregates,
in log (the quantity is a ratio), never averaging the arms separately.

INPUTS (produced by `v2_g3_decorrelation_map.py`, all gitignored):
    runs/v2/g3_decorrelation_cdev8.json          job 296,        seed 0, cdev8
    runs/v2/g3_decorrelation_cdev.json           job 297,        seed 0, cdev
    runs/v2/g3_decorrelation_cdev_seed{1..8}.json  Vista 883777_[1-8], cdev
    runs/v2/g3_decorrelation_cgh64.json          Vista 883778,   seed 0, cgh64

Usage:
    pixi run python scripts/v2_g3_paired_readout.py
    pixi run python scripts/v2_g3_paired_readout.py --level k_r0.9
"""

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(HERE), "runs", "v2")

SEEDS = tuple(range(1, 9))
TILES = (32, 64, 128)
# The two cost-matched triples the ensemble ran. vol_ratio == (P/T)^3 == the cost
# in monolithic evolves, so a triple is an iso-cost line through the (T, b) grid.
COSTS = (3.375, 8.0)
LEVELS = ("k_r0.9", "k_r0.5", "k_r0.2")


def load(name):
    """Return (card, rows keyed by (vol_ratio, T))."""
    path = os.path.join(RUNS, name)
    if not os.path.exists(path):
        sys.exit(f"missing card: {path}\n(pull it from the run host; runs/ is gitignored)")
    card = json.load(open(path))
    return card, {(round(r["vol_ratio"], 3), r["n_tile"]): r for r in card["rows"]}


def log_stats(vals):
    """Geometric mean and 1sd / sem bands. Ratios aggregate in log, not linearly."""
    lg = [math.log(v) for v in vals]
    n = len(lg)
    m = sum(lg) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in lg) / (n - 1))
    sem = sd / math.sqrt(n)
    return {
        "gm": math.exp(m),
        "lo_sd": math.exp(m - sd), "hi_sd": math.exp(m + sd),
        "lo_sem": math.exp(m - sem), "hi_sem": math.exp(m + sem),
        "frac_sd": math.exp(sd) - 1.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="k_r0.5", choices=LEVELS,
                    help="r threshold defining k_usable (default the gate's r=0.5)")
    args = ap.parse_args()
    lev = args.level

    seeds = {s: load(f"g3_decorrelation_cdev_seed{s}.json")[1] for s in SEEDS}
    _, cdev8 = load("g3_decorrelation_cdev8.json")
    _, cdev0 = load("g3_decorrelation_cdev.json")
    cgh_card, cgh64 = load("g3_decorrelation_cgh64.json")

    def ratio(rows, cost):
        return rows[(cost, 128)][lev] / rows[(cost, 64)][lev]

    print(f"=== 1. per-seed k_usable at {lev}, cdev, ranked WITHIN each seed ===")
    for cost in COSTS:
        print(f"\n  cost {cost:.2f}x            T=32     T=64    T=128   128/64  ordering")
        for s in SEEDS:
            v = [seeds[s][(cost, T)][lev] for T in TILES]
            order = ">".join(f"T{t}" for _, t in sorted(zip(v, TILES), reverse=True))
            print(f"    seed {s}        {v[0]:8.4f} {v[1]:8.4f} {v[2]:8.4f} {v[2]/v[1]:8.3f}  {order}")
        v0 = [cdev0[(cost, T)][lev] for T in TILES]
        print(f"    seed 0 (297) {v0[0]:8.4f} {v0[1]:8.4f} {v0[2]:8.4f} {v0[2]/v0[1]:8.3f}  "
              f"(not in the ensemble)")

    print(f"\n=== 2. THE PAIRED RATIO T=128 / T=64 at {lev}, seeds 1-8 ===")
    for cost in COSTS:
        rats = [ratio(seeds[s], cost) for s in SEEDS]
        st = log_stats(rats)
        n_up = sum(1 for r in rats if r > 1.0)
        brackets = min(rats) < 1.0 < max(rats)
        print(f"\n  cost {cost:.2f}x   " + " ".join(f"{r:.3f}" for r in sorted(rats)))
        print(f"    geometric mean {st['gm']:.3f}   1sd [{st['lo_sd']:.3f}, {st['hi_sd']:.3f}]"
              f"   sem [{st['lo_sem']:.3f}, {st['hi_sem']:.3f}]")
        print(f"    T=128 ahead in {n_up}/8 seeds;  per-seed ratios bracket 1.0: "
              f"{'YES -- no ranking' if brackets else 'NO -- the ranking holds'}")

    print("\n=== 3. the same ratio at every r level (cost 3.375x) ===")
    print("    a high-r gate and a low-r gate do not buy the same thing from T")
    for level in LEVELS:
        rats = [seeds[s][(3.375, 128)][level] / seeds[s][(3.375, 64)][level] for s in SEEDS]
        st = log_stats(rats)
        print(f"  {level:8s} gm {st['gm']:6.3f}  1sd [{st['lo_sd']:.3f}, {st['hi_sd']:.3f}]"
              f"  n>1 {sum(1 for r in rats if r > 1):d}/8")

    print(f"\n=== 4. seed scatter of the ABSOLUTE {lev} (this is what the pairing removes) ===")
    print("   cost    T   b_mpc     mean       sd   sd/mean      min      max    seed0")
    for cost in COSTS:
        for T in TILES:
            v = [seeds[s][(cost, T)][lev] for s in SEEDS]
            m = sum(v) / len(v)
            sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
            print(f"  {cost:5.2f} {T:4d} {seeds[1][(cost, T)]['b_mpc']:7.1f}  {m:8.4f} {sd:8.4f}"
                  f"  {sd/m:7.1%}  {min(v):8.4f} {max(v):8.4f} {cdev0[(cost, T)][lev]:8.4f}")

    print("\n=== 5. FIXED ABSOLUTE BUFFER: T=64 vs T=128 at b = 8 Mpc/h ===")
    print("   job 297 read this pair as non-monotone in T; paired over 8 seeds it is a NULL.")
    print("   Not cost-matched, deliberately -- that is the finding.")
    rats = [seeds[s][(8.0, 64)][lev] / seeds[s][(3.375, 128)][lev] for s in SEEDS]
    st = log_stats(rats)
    for lab, key in (("T=64 ", (8.0, 64)), ("T=128", (3.375, 128))):
        v = [seeds[s][key][lev] for s in SEEDS]
        w = sum(seeds[s][key]["wall"] for s in SEEDS) / len(SEEDS)
        r0 = seeds[1][key]
        print(f"    {lab} b={r0['b_mpc']:.0f} Mpc/h  P={r0['p_side']:3d}  {r0['n_tiles']:5d} tiles"
              f"  vol_ratio {r0['vol_ratio']:5.3f}  mean {lev} {sum(v)/len(v):.4f}"
              f"  mean wall {w:6.0f} s")
    print(f"    paired T64/T128 {' '.join(f'{r:.3f}' for r in sorted(rats))}")
    print(f"    gm {st['gm']:.3f}  sem [{st['lo_sem']:.3f}, {st['hi_sem']:.3f}]"
          f"  n>1 {sum(1 for r in rats if r > 1):d}/8")

    print("\n=== 6. THE BOX LADDER at cost 3.375x -- what the confound needed ===")
    print("   box          T=32     T=64    T=128   128/64   excess   seeds")
    ens = log_stats([ratio(seeds[s], 3.375) for s in SEEDS])
    for name, rows, n in (("cdev8", cdev8, 1), ("cdev", cdev0, 1), ("cgh64", cgh64, 1)):
        v = [rows[(3.375, T)][lev] for T in TILES]
        print(f"  {name:9s} {v[0]:8.4f} {v[1]:8.4f} {v[2]:8.4f} {v[2]/v[1]:8.3f} {v[2]/v[1]-1:8.3f}"
              f"   {n:5d}")
    print(f"  {'cdev ens':9s} {'':8s} {'':8s} {'':8s} {ens['gm']:8.3f} {ens['gm']-1:8.3f}"
          f"   {len(SEEDS):5d}   1sd [{ens['lo_sd']:.3f}, {ens['hi_sd']:.3f}]")

    cgh_ratio = ratio(cgh64, 3.375)
    inside = ens["lo_sd"] < cgh_ratio < ens["hi_sd"]
    sigma_same = (cgh_ratio - 1.0) / (cgh_ratio * ens["frac_sd"])
    sigma_shrunk = sigma_same * math.sqrt(8.0)
    print(f"\n   cgh64 is ONE seed. Its {cgh_ratio:.3f} lies inside the cdev ensemble 1sd "
          f"[{ens['lo_sd']:.3f}, {ens['hi_sd']:.3f}]: {inside}")
    print("   so the cdev -> cgh64 step is NOT resolved by this realization. Above 1.0 it is")
    print(f"   {sigma_same:.1f} sigma if cgh64 scatters like cdev ({ens['frac_sd']:.1%}),"
          f" {sigma_shrunk:.1f} sigma if scatter")
    print("   falls as sqrt(volume). The direction is established; the size at cgh64 is not.")

    print("\n=== 7. cgh64 rung in full (Vista 883778, seed 0, the 3.38x triple) ===")
    print("     T  b_fine  b_mpc     P   p_frac  n_tiles  k(r=0.9)  k(r=0.5)  k(r=0.2)   wall_s")
    for T in TILES:
        r = cgh64[(3.375, T)]
        flag = "  DEGENERATE" if r["degenerate"] else ""
        print(f"  {T:4d} {r['b_fine']:7d} {r['b_mpc']:6.1f} {r['p_side']:5d} {r['p_frac']:8.4f}"
              f" {r['n_tiles']:8d} {r['k_r0.9']:9.4f} {r['k_r0.5']:9.4f} {r['k_r0.2']:9.4f}"
              f" {r['wall']:8.0f}{flag}")
    g = cgh_card["geometry"]
    print(f"   geometry: n_part {g['n_part']:.0f}  n_fine {g['n_fine']:.0f}  L {g['L']:.0f} Mpc/h"
          f"  fine cell {g['fine_cell']:.2f} Mpc/h")

    print("\n=== caveats ===")
    print("  - WALL TIMES ARE NOT COMPARABLE ACROSS BOXES HERE: cdev8 and the seed-0 cdev card")
    print("    ran on deneb, the ensemble and cgh64 on Vista gg. Compare wall only within a host.")
    print("  - cdev8 and cgh64 carry one seed each; only the cdev rung has an error bar.")
    print("  - k_usable is an interpolated crossing of r(k), so it inherits the shell binning.")


if __name__ == "__main__":
    main()
