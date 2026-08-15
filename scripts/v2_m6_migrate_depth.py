"""W0b: WHY did `migrate` grow ~70x for 8x particles? The depth/volume one-axis probe.

Job 464 measured `migrate` as the LARGEST cgh64 phase (197.4 s/step, 32.2%)
against 2.80 s/step at cdev -- ~70x for 8x the particles -- with no measured
decomposition. The known one-axis instruments do not answer it: job 456/457
(`v2_m6_insert_scaling.py`) pinned reach at 1 with a near-zero drift, so it
validated the scan fix in a regime where almost nothing migrates, and job 461's
"linear in N at pinned depth" holds in that same regime. The production regime
has reach [3,3,2] and real migrant volume.

**The confound this design exists to kill:** sweeping drift moves TWO variables
-- the staged depth (reach, a DISCRETE step) and the migrant volume (rows that
actually change brick, CONTINUOUS in the drift). A two-point reach-1-vs-reach-3
comparison could not tell them apart. So each config runs a five-point drift
ladder, in fractions f of the reach-1 threshold c1 = extent / (s_max * 2^15):

    f = 0.3, 0.6, 0.95   -> reach 1, migrant volume rising  (the VOLUME slope)
    f = 1.9              -> reach 2                         (first DEPTH step)
    f = 2.85             -> reach 3, the production regime  (second DEPTH step)

The three reach-1 points fit cost vs migrant rows at fixed depth; the reach-2/3
points are then read AGAINST that fit, so their excess is attributable to depth
rather than to the extra migrants they also carry. A rung whose realized
`brick_reach` is not its target is VOID and excluded from any fit, not noisy.

Configs mirror the production ladder in the variables the mechanism can depend
on -- N, nb, rows/brick (4096 at both), brick extent (8 Mpc/h at both):
    cdev-like : n_part=256, nb=16, box=128
    cgh64-like: n_part=512, nb=32, box=256
State is synthetic (uniform positions, Gaussian velocities), as in job 456: the
transferable output is the LAW (volume slope, depth increments, N scaling at
matched depth), not the absolute seconds.

Every rung REBUILDS the state from the same seed -- `drift_and_migrate` mutates
its input, so repeats on one state would measure successively different states.

Readouts per rung: eject_s / insert_s / total_s, migrant rows (counted from the
eject returns; `drift_and_migrate` does not report it), realized reach,
peak_staged_slabs, arena_used. Then three comparisons, printed and carded:
  1. the reach-1 volume fit per config (slope s/Mrow, intercept),
  2. depth excess at reach 2 and 3 vs that fit's prediction,
  3. cgh64/cdev at matched f: is migrate linear in N at REAL volume and depth?

Usage:
  pixi run python scripts/v2_m6_migrate_depth.py --config smoke          # laptop
  pixi run python scripts/v2_m6_migrate_depth.py --config cdev cgh64    # antares
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from inexor import state  # noqa: E402
from inexor.codec import T9Layout  # noqa: E402

SEED = 7
BUCKET_CELLS = 2
FRACTIONS = (0.3, 0.6, 0.95, 1.9, 2.85)
TARGET_REACH = (1, 1, 1, 2, 3)

CONFIGS = {
    "smoke": dict(n_part=64, nb=8, box=32.0, repeats=2),
    "cdev": dict(n_part=256, nb=16, box=128.0, repeats=3),
    "cgh64": dict(n_part=512, nb=32, box=256.0, repeats=2),
}


def _build(n_part, nb, box, seed=SEED):
    rng = np.random.default_rng(seed)
    n = int(n_part) ** 3
    x = rng.uniform(0.0, box, size=(n, 3))
    v = rng.normal(scale=1.0, size=(n, 3))
    t9 = T9Layout(box, int(n_part), BUCKET_CELLS)
    # arena 0.20 = the timing-leg convention: the drift ladder migrates up to
    # ~78% of particles (smoke, f=2.85) against the engine's few percent, and a
    # D-007 arena refusal must not be able to end a TIMING rung
    return state.SlotState.build(x, v, t9, int(nb), arena_frac=0.20)


def _timed_migrate(st, c_drift):
    """One `drift_and_migrate` with eject/insert timed and MIGRANTS counted.

    Same class-level wrap as `v2_m6_insert_scaling._timed_migrate`; extended
    here (rather than imported) because the emigrant count only exists inside
    the eject return values, which that wrapper discards.
    """
    cls = type(st)
    orig_e, orig_i = cls._eject_slab, cls._insert_slab
    acc = {"eject_s": 0.0, "insert_s": 0.0, "n_emig": 0}

    def eject(self, *a, **k):
        t0 = time.perf_counter()
        out = orig_e(self, *a, **k)
        acc["eject_s"] += time.perf_counter() - t0
        emig = out[1]
        if emig and emig.get("dest") is not None:
            acc["n_emig"] += int(len(emig["dest"]))
        return out

    def insert(self, *a, **k):
        t0 = time.perf_counter()
        out = orig_i(self, *a, **k)
        acc["insert_s"] += time.perf_counter() - t0
        return out

    cls._eject_slab, cls._insert_slab = eject, insert
    try:
        t0 = time.perf_counter()
        stats = state.drift_and_migrate(st, c_drift)
        acc["total_s"] = time.perf_counter() - t0
    finally:
        cls._eject_slab, cls._insert_slab = orig_e, orig_i
    for k in ("brick_reach", "brick_reach_realized", "peak_staged_slabs",
              "arena_used", "n_arena_overflow"):
        acc[k] = int(stats[k])
    return acc


def run_config(name, cfg):
    print(f"== {name}: n_part={cfg['n_part']} nb={cfg['nb']} box={cfg['box']}")
    extent = cfg["box"] / cfg["nb"]
    rungs = []
    for f, target in zip(FRACTIONS, TARGET_REACH):
        reps = []
        for _ in range(cfg["repeats"]):
            st = _build(cfg["n_part"], cfg["nb"], cfg["box"])
            c1 = extent / (float(np.max(st.vel_scale)) * state.INT16_MAX)
            reps.append(_timed_migrate(st, f * c1))
        r = dict(
            f=float(f), target_reach=int(target),
            c_drift=float(f * c1),
            eject_s=float(np.median([x["eject_s"] for x in reps])),
            insert_s=float(np.median([x["insert_s"] for x in reps])),
            total_s=float(np.median([x["total_s"] for x in reps])),
            total_s_spread=float(np.ptp([x["total_s"] for x in reps])),
            n_emig=int(reps[-1]["n_emig"]),
            brick_reach=reps[-1]["brick_reach"],
            brick_reach_realized=reps[-1]["brick_reach_realized"],
            peak_staged_slabs=reps[-1]["peak_staged_slabs"],
            arena_used=reps[-1]["arena_used"],
            n_arena_overflow=reps[-1]["n_arena_overflow"],
        )
        r["void"] = r["brick_reach"] != target
        rungs.append(r)
        flag = "  VOID (reach != target)" if r["void"] else ""
        print(f"  f={f:4.2f} reach {r['brick_reach']} (want {target}, realized "
              f"{r['brick_reach_realized']})  emig {r['n_emig']:>10,}  "
              f"eject {r['eject_s']:7.2f} s  insert {r['insert_s']:7.2f} s  "
              f"total {r['total_s']:7.2f} s (spread {r['total_s_spread']:.2f}){flag}",
              flush=True)

    out = dict(config=name, **{k: cfg[k] for k in ("n_part", "nb", "box", "repeats")},
               n_particles=cfg["n_part"] ** 3, rungs=rungs)

    # 1. the volume law at fixed depth: total_s vs migrant rows over the three
    # reach-1 rungs (a line through three points -- reported, and the residual
    # of the middle point is printed so a curve cannot masquerade as a line)
    r1 = [r for r in rungs[:3] if not r["void"]]
    if len(r1) == 3:
        x = np.array([r["n_emig"] for r in r1], dtype=np.float64)
        y = np.array([r["total_s"] for r in r1])
        slope, icpt = np.polyfit(x, y, 1)
        mid_resid = float(y[1] - (slope * x[1] + icpt))
        out["volume_fit"] = dict(slope_s_per_row=float(slope), intercept_s=float(icpt),
                                 mid_residual_s=mid_resid)
        print(f"  volume law (reach 1): {slope * 1e6:.2f} s/Mrow, intercept "
              f"{icpt:.2f} s, mid-point residual {mid_resid:+.2f} s")
        # 2. depth excess: reach-2/3 rungs against the volume fit's prediction
        for r in rungs[3:]:
            if r["void"]:
                continue
            pred = slope * r["n_emig"] + icpt
            r["volume_pred_s"] = float(pred)
            r["depth_excess_s"] = float(r["total_s"] - pred)
            print(f"  depth excess at reach {r['brick_reach']}: total "
                  f"{r['total_s']:.2f} s vs volume-law {pred:.2f} s -> "
                  f"{r['depth_excess_s']:+.2f} s "
                  f"({r['total_s'] / pred:.2f}x the volume prediction)")
    else:
        print("  volume law NOT fit (a reach-1 rung is void)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=["smoke"],
                    choices=sorted(CONFIGS))
    ap.add_argument("--out", default=os.path.join("runs", "v2", "m6_migrate_depth.json"))
    a = ap.parse_args()

    # card written after EVERY config, so a death in a late rung cannot lose a
    # finished config's rungs (the raw-series-survive-a-wrong-reduction rule)
    results = []
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for name in a.config:
        results.append(run_config(name, CONFIGS[name]))
        with open(a.out, "w") as fh:
            json.dump(dict(results=results, partial=(len(results) < len(a.config))),
                      fh, indent=1)

    # 3. cross-config at matched f: N scaling at real volume and depth
    if len(results) == 2:
        lo, hi = results
        n_ratio = hi["n_particles"] / lo["n_particles"]
        print(f"== {hi['config']}/{lo['config']} at matched f (N ratio {n_ratio:.0f}x):")
        for rl, rh in zip(lo["rungs"], hi["rungs"]):
            if rl["void"] or rh["void"] or rl["total_s"] <= 0:
                continue
            print(f"  f={rl['f']:4.2f}: total {rh['total_s'] / rl['total_s']:6.2f}x  "
                  f"eject {rh['eject_s'] / max(rl['eject_s'], 1e-9):6.2f}x  "
                  f"insert {rh['insert_s'] / max(rl['insert_s'], 1e-9):6.2f}x  "
                  f"(linear would be {n_ratio:.0f}x)")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(dict(results=results, commit=commit, machine=platform.machine(),
                       system=platform.system(), argv=sys.argv[1:]), fh, indent=1)
    print(f"card -> {a.out}")
    voids = sum(r["void"] for res in results for r in res["rungs"])
    return 1 if voids else 0


if __name__ == "__main__":
    sys.exit(main())
