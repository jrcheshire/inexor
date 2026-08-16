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
    # --- the nb axis at FIXED N (M-v2-6 owed item 10, added 2026-08-15) ---
    #
    # `migrate` is 45% of a W=16 step on gg and 23.3 h of a projected 51.7 h
    # C-gh realization -- 4.7x the whole 5 h bar on its own (section 5j). The
    # leading hypothesis is section 6's "per-brick Python loop": both
    # `_eject_slab` and `_insert_slab` walk bricks in a Python `for`, so a step
    # runs ~2 * nb^3 iterations -- 65,536 at cgh64 and 4.2M at C-gh, since nb
    # goes 32 -> 128.
    #
    # THE POINT: per-brick overhead (~nb^3) and per-row work (~N) give the SAME
    # 64x projection from cgh64 to C-gh, so no arithmetic separates them and
    # the config ladder cannot either (particles, bricks and coarse cells move
    # together on it). Only nb at fixed N does. Box is fixed too, so the
    # physical volume and the particle count are both held and ONLY brick
    # granularity moves: rows/brick goes 32,768 / 4,096 / 512.
    #
    # Reach is held across the scan by construction rather than by luck: `f` is
    # a fraction of the reach-1 threshold `extent / (s_max * 2^15)` and
    # `brick_reach` is `ceil(f)`, so a matched `f` is a matched DEPTH at every
    # nb. Migrant VOLUME should also be ~invariant at matched f (a displacement
    # of f brick-extents crosses a boundary with a probability set by f, not by
    # the extent) -- `n_emig` is reported per rung so that is CHECKED, never
    # assumed. Two fractions only: one reach-1 point and the production reach-3
    # point, because the volume law itself is already fit by the cgh64 leg.
    "nb16": dict(n_part=512, nb=16, box=256.0, repeats=2, fractions=(0.95, 2.85),
                 nb_scan=True),
    "nb32": dict(n_part=512, nb=32, box=256.0, repeats=2, fractions=(0.95, 2.85),
                 nb_scan=True),
    "nb64": dict(n_part=512, nb=64, box=256.0, repeats=2, fractions=(0.95, 2.85),
                 nb_scan=True),
    # laptop-scale twins of the scan (262,144 particles), so the reader and the
    # exponent fit are exercised end to end before any cluster time is spent --
    # the reporting path is untested code until it has run once
    "nbs4": dict(n_part=64, nb=4, box=32.0, repeats=2, fractions=(0.95, 2.85),
                 nb_scan=True),
    "nbs8": dict(n_part=64, nb=8, box=32.0, repeats=2, fractions=(0.95, 2.85),
                 nb_scan=True),
    "nbs16": dict(n_part=64, nb=16, box=32.0, repeats=2, fractions=(0.95, 2.85),
                  nb_scan=True),
}


def _build(n_part, nb, box, seed=SEED, brick_slack=0.10, arena_frac=0.20):
    rng = np.random.default_rng(seed)
    n = int(n_part) ** 3
    x = rng.uniform(0.0, box, size=(n, 3))
    v = rng.normal(scale=1.0, size=(n, 3))
    t9 = T9Layout(box, int(n_part), BUCKET_CELLS)
    # arena 0.20 = the timing-leg convention: the drift ladder migrates up to
    # ~78% of particles (smoke, f=2.85) against the engine's few percent, and a
    # D-007 arena refusal must not be able to end a TIMING rung.
    # `brick_slack` is the ARENA-OCCUPANCY axis (job 465 follow-up): at the
    # default 0.10 a uniform state never overflows a brick and the arena stays
    # EMPTY (job 465 measured arena_used = 0 on every rung), which is the one
    # structural difference from the engine's clustered state. slack 0.0 makes
    # immigrants overflow into the arena, so migrate runs in the engine-like
    # regime where `_eject_slab`'s per-brick index invalidation interleaves
    # with O(n_arena) rebuilds.
    return state.SlotState.build(x, v, t9, int(nb), brick_slack=brick_slack,
                                 arena_frac=arena_frac)


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


def run_config(name, cfg, brick_slack=0.10, arena_frac=0.20, calls=1):
    print(f"== {name}: n_part={cfg['n_part']} nb={cfg['nb']} box={cfg['box']} "
          f"slack={brick_slack} arena={arena_frac} calls={calls}")
    extent = cfg["box"] / cfg["nb"]
    # a config may run a SUBSET of the ladder (the nb-scan configs run one
    # reach-1 and one reach-3 point); targets stay tied to their fraction, so
    # the reach assertion below cannot silently drift off its rung
    fracs = tuple(cfg.get("fractions", FRACTIONS))
    targets = tuple(TARGET_REACH[FRACTIONS.index(f)] for f in fracs)
    rungs = []
    for f, target in zip(fracs, targets):
        reps = []
        for _ in range(cfg["repeats"]):
            st = _build(cfg["n_part"], cfg["nb"], cfg["box"],
                        brick_slack=brick_slack, arena_frac=arena_frac)
            c1 = extent / (float(np.max(st.vel_scale)) * state.INT16_MAX)
            # `calls > 1` CHAINS migrates on one state: call 1 populates the
            # arena from the drift, call 2+ measures migrate with the arena
            # already resident -- the engine's steady condition. Each call is
            # its own timing; only the LAST lands in `reps` (the steady one),
            # the chain is carried on the rung as `chain_total_s`/`chain_arena`.
            chain_t, chain_a = [], []
            try:
                for _ in range(int(calls) - 1):
                    warm = _timed_migrate(st, f * c1)
                    chain_t.append(warm["total_s"])
                    chain_a.append(warm["arena_used"])
                r_last = _timed_migrate(st, f * c1)
            except (RuntimeError, ValueError) as exc:
                # the D-007 arena/capacity refusal: a REFUSED rung is a marked
                # row, not a lost card -- slack-0 arms push migrant volumes the
                # arena cannot always absorb, and that is a finding, not noise
                print(f"  f={f:4.2f} REFUSED: {str(exc).splitlines()[0][:100]}")
                reps = []
                rungs.append(dict(f=float(f), target_reach=int(target), void=True,
                                  refused=str(exc).splitlines()[0][:200]))
                break
            if chain_t:
                r_last["chain_total_s"] = chain_t
                r_last["chain_arena"] = chain_a
            reps.append(r_last)
        if not reps:
            continue
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
               n_particles=cfg["n_part"] ** 3, brick_slack=brick_slack,
               arena_frac=arena_frac, calls=calls,
               n_bricks=int(cfg["nb"]) ** 3,
               rows_per_brick=cfg["n_part"] ** 3 / float(cfg["nb"]) ** 3,
               rungs=rungs)

    # 1. the volume law at fixed depth: total_s vs migrant rows over the three
    # reach-1 rungs (a line through three points -- reported, and the residual
    # of the middle point is printed so a curve cannot masquerade as a line).
    # A config running a subset of the ladder has no three reach-1 points and
    # skips this; it is not a failure, and the nb scan reads a different law.
    r1 = [r for r in rungs if not r["void"] and r["target_reach"] == 1]
    if len(r1) < 3:
        print(f"  volume law not fit ({len(r1)} reach-1 rungs; needs 3)")
    elif len(r1) == 3:
        x = np.array([r["n_emig"] for r in r1], dtype=np.float64)
        y = np.array([r["total_s"] for r in r1])
        slope, icpt = np.polyfit(x, y, 1)
        mid_resid = float(y[1] - (slope * x[1] + icpt))
        out["volume_fit"] = dict(slope_s_per_row=float(slope), intercept_s=float(icpt),
                                 mid_residual_s=mid_resid)
        print(f"  volume law (reach 1): {slope * 1e6:.2f} s/Mrow, intercept "
              f"{icpt:.2f} s, mid-point residual {mid_resid:+.2f} s")
        # 2. depth excess: reach-2/3 rungs against the volume fit's prediction
        for r in rungs:
            if r["void"] or r["target_reach"] == 1:
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
    ap.add_argument("--brick-slack", type=float, default=0.10)
    ap.add_argument("--arena-frac", type=float, default=0.20)
    ap.add_argument("--calls", type=int, default=1)
    # THE REPRODUCTION GATE. This probe's state is SYNTHETIC (uniform positions,
    # Gaussian velocities) while the engine's is evolved and clustered, and
    # arena pressure is a clustering property -- so the probe describing the
    # engine is a hypothesis, not a given. `--expect-s` pre-registers the
    # engine's own measured `migrate` s/step; the deepest non-void rung of the
    # LAST config is read against it and the run returns 3 if it misses. A miss
    # is not a bad number, it means the hunt moves engine-side and no downstream
    # leg should burn cluster time on this instrument.
    ap.add_argument("--expect-s", type=float, default=None)
    ap.add_argument("--expect-within", type=float, default=2.0)
    ap.add_argument("--profile", type=int, default=0,
                    help="cProfile one migrate at the deepest rung and print "
                         "the top N by cumulative time. RANKING only -- the "
                         "profiler's own overhead makes the seconds unusable.")
    ap.add_argument("--out", default=os.path.join("runs", "v2", "m6_migrate_depth.json"))
    a = ap.parse_args()

    # card written after EVERY config, so a death in a late rung cannot lose a
    # finished config's rungs (the raw-series-survive-a-wrong-reduction rule)
    results = []
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for name in a.config:
        results.append(run_config(name, CONFIGS[name], brick_slack=a.brick_slack,
                                  arena_frac=a.arena_frac, calls=a.calls))
        with open(a.out, "w") as fh:
            json.dump(dict(results=results, partial=(len(results) < len(a.config))),
                      fh, indent=1)

    # 3. cross-config at matched f: N scaling at real volume and depth.
    # Skipped for the nb scan, whose configs hold N FIXED -- this block's
    # "linear would be Nx" line is a statement about an axis that is not moving
    # there, and printing it beside the nb result invites reading one as the
    # other.
    if len(results) == 2 and not any(CONFIGS[r["config"]].get("nb_scan") for r in results):
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

    # 4. THE nb AXIS at fixed N: is `migrate` per-BRICK or per-ROW?
    #
    # Fit total_s ~ nb^alpha across the scan at each matched f. The two
    # hypotheses are pre-registered and far apart, so this does not need a
    # tuned threshold, only a measured exponent:
    #     alpha ~ 3  -> per-brick. The Python loop over nb^3 bricks IS the cost,
    #                   and migrate is linear in N only because nb^3 is. The fix
    #                   is vectorizing across the bricks of a slab.
    #     alpha ~ 0  -> per-row. The cost is honest work on N rows, the loop is
    #                   innocent, and the only lever is parallelism -- which
    #                   means migrate joins the pool and inherits the bandwidth
    #                   ceiling section 5j measured.
    # Anything between says both terms are live and the exponent gives their
    # mix at this config. The scan holds N, box, f (hence reach) and arena_frac;
    # `n_emig` is printed because its invariance is the ASSUMPTION that makes a
    # matched f a matched migrant volume, and it is checked rather than trusted.
    nb_out = []
    scan = [r for r in results if CONFIGS[r["config"]].get("nb_scan")]
    if len(scan) >= 2:
        scan.sort(key=lambda r: r["nb"])
        by_f = {}
        for res in scan:
            for r in res["rungs"]:
                if not r["void"]:
                    by_f.setdefault(round(float(r["f"]), 4), []).append((res, r))
        nb_list = "/".join(str(res["nb"]) for res in scan)
        rpb_list = "/".join("%.0f" % res["rows_per_brick"] for res in scan)
        print(f"== nb scan at fixed N = {scan[0]['n_particles']:,} "
              f"(nb {nb_list}, rows/brick {rpb_list}):")
        for f_val, pairs in sorted(by_f.items()):
            if len(pairs) < 2:
                continue
            nb = np.array([res["nb"] for res, _ in pairs], dtype=np.float64)
            tot = np.array([r["total_s"] for _, r in pairs], dtype=np.float64)
            emig = np.array([r["n_emig"] for _, r in pairs], dtype=np.float64)
            row = dict(f=f_val, nb=[int(v) for v in nb],
                       total_s=[float(v) for v in tot],
                       eject_s=[float(r["eject_s"]) for _, r in pairs],
                       insert_s=[float(r["insert_s"]) for _, r in pairs],
                       n_emig=[int(v) for v in emig],
                       reach=[int(r["brick_reach"]) for _, r in pairs])
            row["alpha_total"] = float(np.polyfit(np.log(nb), np.log(tot), 1)[0])
            # alpha IS a mixing fraction, and that is the actionable form. With
            # cost = A*nb^3 (per-brick) + B*N (per-row) at fixed N,
            #     d log t / d log nb = 3 * A*nb^3 / (A*nb^3 + B*N)
            # so alpha / 3 is the per-brick SHARE of migrate -- i.e. the
            # fraction that vectorizing the slab loop could remove. It is a
            # local quantity, evaluated at the geometric mean of the scanned
            # nb, and it is only meaningful while the volume check below holds.
            row["per_brick_share"] = float(np.clip(row["alpha_total"] / 3.0, 0.0, 1.0))
            for key in ("eject_s", "insert_s"):
                y = np.array(row[key], dtype=np.float64)
                row["alpha_" + key.split("_")[0]] = (
                    float(np.polyfit(np.log(nb), np.log(y), 1)[0])
                    if np.all(y > 0) else float("nan"))
            # the assumption under the matched-f design, reported as a spread
            row["emig_spread"] = float(emig.max() / max(emig.min(), 1.0))
            nb_out.append(row)
            print(f"  f={f_val:4.2f} reach {row['reach']}: total "
                  + " / ".join(f"{v:.2f}" for v in tot)
                  + f" s  ->  alpha_total {row['alpha_total']:+.2f}"
                  f"  (eject {row['alpha_eject']:+.2f}, "
                  f"insert {row['alpha_insert']:+.2f})"
                  f"  -> per-brick share "
                  f"{row['per_brick_share'] * 100:.0f}%")
            print("           migrants " + " / ".join(f"{int(v):,}" for v in emig)
                  + f"  (spread {row['emig_spread']:.2f}x"
                  + ("; matched-f volume invariance HOLDS)"
                     if row["emig_spread"] < 1.25 else
                     "; NOT invariant -- alpha is contaminated by volume)"))
        if nb_out:
            print("  pre-registered: alpha ~ 3 = per-brick (vectorize the slab "
                  "loop); alpha ~ 0 = per-row (parallelism is the only lever)")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(dict(results=results, nb_scan=nb_out, commit=commit,
                       machine=platform.machine(), system=platform.system(),
                       argv=sys.argv[1:]), fh, indent=1)
    print(f"card -> {a.out}")

    if a.expect_s is not None:
        live = [r for r in results[-1]["rungs"] if not r["void"]]
        if not live:
            print(f"REPRODUCTION GATE: NO READABLE RUNG in {results[-1]['config']}")
            return 3
        deep = max(live, key=lambda r: r["f"])
        ratio = deep["total_s"] / float(a.expect_s)
        ok = (1.0 / a.expect_within) <= ratio <= a.expect_within
        print(f"REPRODUCTION GATE ({results[-1]['config']}, f={deep['f']:.2f}, "
              f"reach {deep['brick_reach']}): probe {deep['total_s']:.2f} s vs "
              f"engine {a.expect_s:.2f} s = {ratio:.2f}x, bar {a.expect_within:.1f}x "
              f"-> {'PASS' if ok else 'FAIL'}")
        if not ok:
            print("  The synthetic state does not reproduce the engine's migrate. "
                  "The nb scan would be measuring this probe, not the engine -- "
                  "move the hunt engine-side rather than running the long legs.")
            return 3

    if a.profile:
        import cProfile
        import pstats
        res = results[-1]
        cfg = CONFIGS[res["config"]]
        live = [r for r in res["rungs"] if not r["void"]]
        if live:
            deep = max(live, key=lambda r: r["f"])
            print(f"== profile: {res['config']} at f={deep['f']:.2f} "
                  f"(ranking only; profiler overhead makes the seconds unusable)")
            st = _build(cfg["n_part"], cfg["nb"], cfg["box"],
                        brick_slack=a.brick_slack, arena_frac=a.arena_frac)
            pr = cProfile.Profile()
            pr.enable()
            state.drift_and_migrate(st, deep["c_drift"])
            pr.disable()
            pstats.Stats(pr).sort_stats("cumulative").print_stats(int(a.profile))

    voids = sum(r["void"] for res in results for r in res["rungs"])
    return 1 if voids else 0


if __name__ == "__main__":
    sys.exit(main())
