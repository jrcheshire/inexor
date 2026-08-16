"""W2 executor identity at cluster scale: pooled vs serial, bitwise, one process.

The laptop's identity role ends at the smoke-config unit tests; this probe
runs the whole-run comparison at any config on a cluster node. Serial arm
first, snapshot every state array, then the pooled arm from identically
rebuilt state (`pt._build` caches the ICs, so both arms start from the same
container); compare elementwise, ids included, plus the cap ladder and
`arena_base`. A nonzero diff means the parallel premise itself is wrong, not
something to tune.

The `if __name__ == "__main__"` guard is LOAD-BEARING: the pool executor
spawns workers and spawn re-imports __main__ -- an unguarded driver re-runs
itself per worker and recurses into pool creation.

Usage:
  pixi run python scripts/v2_m6_w2_identity.py --config cdev --k 4 --workers 4
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

FIELDS = ("off", "w", "occupancy", "brick_start", "vel_scale", "arena_bucket")


def _arm(a, workers):
    import v2_m3_engine_gate as m3
    import v2_m6_phase_time as pt

    engine, ec, st, cosmo, a_grid, bfc, bft = pt._build(
        a.config, a.slack, a.arena_frac, tile=a.tile, buf=a.buf, tile_workers=workers
    )
    co = bfc(bft(a_grid(m3.A_INIT, m3.A_FINAL, a.k, m3.SPACING), cosmo))
    t0 = time.perf_counter()
    stats = engine.run(st, ec, co)
    return st, stats, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="cdev8")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--slack", type=float, default=0.20)
    ap.add_argument("--arena-frac", type=float, default=0.20)
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--buf", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    st1, stats1, wall1 = _arm(a, 1)
    snap = {f: np.array(getattr(st1, f), copy=True) for f in FIELDS}
    ids1 = None if st1.ids is None else np.array(st1.ids, copy=True)
    ab1, caps1 = int(st1.arena_base), [int(s["cap"]) for s in stats1]
    del st1  # halve the resident state before the pooled arm builds its own

    st2, stats2, wall2 = _arm(a, a.workers)
    diffs = {f: int((snap[f] != np.asarray(getattr(st2, f))).sum()) for f in FIELDS}
    if ids1 is not None and st2.ids is not None:
        diffs["ids"] = int((ids1 != np.asarray(st2.ids)).sum())
    caps2 = [int(s["cap"]) for s in stats2]
    total = sum(diffs.values())
    cap_ok = caps1 == caps2
    ab_ok = ab1 == int(st2.arena_base)
    # the knob must prove it applied, on both lanes of the pooled arm
    pool_last = stats2[-1].get("pool") or {}
    coarse_pooled = int(stats2[-1].get("coarse_pooled_workers", 0))
    applied = pool_last.get("workers") == a.workers and coarse_pooled == a.workers

    ok = total == 0 and cap_ok and ab_ok and applied
    for f, n in diffs.items():
        print(f"  {f:12s} n_diff={n}")
    print(f"  cap ladder equal: {cap_ok} ({caps1} vs {caps2})")
    print(f"  arena_base equal: {ab_ok}")
    print(f"  pooled arms applied (tile={pool_last.get('workers')}, "
          f"coarse={coarse_pooled}, want {a.workers}): {applied}")
    print(f"  walls: serial {wall1:.1f} s, pooled W={a.workers} {wall2:.1f} s")
    print(f"VERDICT: {'IDENTICAL' if ok else 'FAIL'} (total n_diff {total})")

    try:
        commit = subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    out = a.out or os.path.join(REPO, "runs", "v2", f"m6_w2_identity_{a.config}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(dict(
            config=a.config, k=a.k, workers=a.workers, tile=a.tile, buf=a.buf,
            slack=a.slack, arena_frac=a.arena_frac, diffs=diffs,
            cap_ladder_serial=caps1, cap_ladder_pooled=caps2,
            arena_base_equal=ab_ok, arms_applied=applied,
            serial_wall_s=wall1, pooled_wall_s=wall2,
            pool_last_step=pool_last, verdict="IDENTICAL" if ok else "FAIL",
            commit=commit, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            argv=sys.argv[1:],
        ), fh, indent=1)
    print(f"card -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
