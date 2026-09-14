"""What is the compiled migrate's host share made of: per brick or per row?

Vista 995264 put 9.84 s of a 15.17 s compiled cgh64 migrate step outside the
kernel calls. At the production 4096 rows per brick, bricks and rows scale
together, so that pair cannot say whether the host cost is per brick (Python
loops over bricks) or per row (array work). This holds the particle count FIXED
and varies the brick count, so a per-brick term moves and a per-row term does not.

Per rung: build the state, one warm compiled migrate (compiles land there), then
one migrate under cProfile with `eject_rows` / `insert_rows` wrapped by a timer.
Host = step minus those calls. The drift moves the fastest row half a brick at
every rung, so the reach is 1 throughout (reported, and a rung whose reach is not
1 is flagged). The top host functions by own time are on the card.

CPU XLA on whatever machine runs it: the kernel calls are not a GPU's, but the
host remainder is the same code the GPU path runs.
"""

from __future__ import annotations

import argparse
import cProfile
import importlib.util
import json
import os
import pstats
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))


def _probe():
    path = os.path.join(REPO, "scripts", "v2_d3_device_migrate.py")
    spec = importlib.util.spec_from_file_location("d3probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rung(P, n_part, nb, frac, top):
    from inexor import eject_jax, insert_jax, state

    t0 = time.perf_counter()
    st = P._build_state(n_part, nb, seed=13, brick_slack=0.10, arena_frac=0.01, with_ids=False)
    build_s = time.perf_counter() - t0
    c = P._c_drift(st, frac)
    t1 = time.perf_counter()
    r_warm = state.drift_and_migrate(st, c, kernel="jax", insert_kernel="jax")
    warm_s = time.perf_counter() - t1

    acc = dict(eject_s=0.0, insert_s=0.0, eject_n=0, insert_n=0)
    real_e, real_i = eject_jax.eject_rows, insert_jax.insert_rows

    def te(*a, **k):
        t = time.perf_counter()
        out = real_e(*a, **k)
        acc["eject_s"] += time.perf_counter() - t
        acc["eject_n"] += 1
        return out

    def ti(*a, **k):
        t = time.perf_counter()
        out = real_i(*a, **k)
        acc["insert_s"] += time.perf_counter() - t
        acc["insert_n"] += 1
        return out

    eject_jax.eject_rows, insert_jax.insert_rows = te, ti
    prof = cProfile.Profile()
    try:
        t2 = time.perf_counter()
        prof.enable()
        r = state.drift_and_migrate(st, c, kernel="jax", insert_kernel="jax")
        prof.disable()
        step_s = time.perf_counter() - t2
    finally:
        eject_jax.eject_rows, insert_jax.insert_rows = real_e, real_i

    host_s = step_s - acc["eject_s"] - acc["insert_s"]
    ps = pstats.Stats(prof)
    rows = []
    for (fname, line, func), (cc, nc, tt, ct, _callers) in ps.stats.items():
        rows.append(dict(func=f"{os.path.basename(fname)}:{line}:{func}", ncalls=nc,
                         tottime=tt, cumtime=ct))
    rows.sort(key=lambda d: -d["tottime"])
    n_bricks = nb**3
    rec = dict(n_part=n_part, nb=nb, n_bricks=n_bricks, rows_per_brick=n_part**3 // n_bricks,
               build_s=build_s, warm_s=warm_s, step_s=step_s, eject_calls_s=acc["eject_s"],
               insert_calls_s=acc["insert_s"], calls=(acc["eject_n"], acc["insert_n"]),
               host_s=host_s, host_us_per_brick=1e6 * host_s / n_bricks,
               host_ns_per_row=1e9 * host_s / n_part**3, reach=r["brick_reach"],
               reach_warm=r_warm["brick_reach"], overflow=r["n_arena_overflow"],
               profiler_note="cProfile inflates Python-call-heavy code; step_s is profiled",
               top_tottime=rows[:top])
    flag = "" if r["brick_reach"] == 1 else "  REACH NOT 1: rung not comparable"
    print(f"\n[nb={nb}] {n_bricks:,} bricks x {n_part**3 // n_bricks:,} rows: step "
          f"{step_s:.2f}s (profiled) = eject calls {acc['eject_s']:.2f}s + insert calls "
          f"{acc['insert_s']:.2f}s + HOST {host_s:.2f}s = {rec['host_us_per_brick']:.1f} us/brick "
          f"= {rec['host_ns_per_row']:.1f} ns/row; reach {r['brick_reach']}, overflow "
          f"{r['n_arena_overflow']}{flag}", flush=True)
    print(f"  {'own s':>7} {'cum s':>7} {'calls':>10}  function", flush=True)
    for d in rows[:top]:
        print(f"  {d['tottime']:7.2f} {d['cumtime']:7.2f} {d['ncalls']:10d}  {d['func']}",
              flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-part", type=int, default=256)
    ap.add_argument("--nbs", default="8,16,32", help="bricks per side, comma-separated")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args(argv)

    import jax

    jax.config.update("jax_enable_x64", True)
    P = _probe()
    out = os.path.join(REPO, "runs", "v2", f"d3_host_profile{args.out_suffix}.json")
    card = dict(argv=sys.argv, started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                node=os.uname().nodename, platform=str(jax.devices()[0].platform),
                commit=subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
                rungs=[])
    for nb in (int(x) for x in args.nbs.split(",")):
        card["rungs"].append(rung(P, args.n_part, nb, args.frac, args.top))
        with open(out, "w") as f:
            json.dump(card, f, indent=1)
    card["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(out, "w") as f:
        json.dump(card, f, indent=1)
    print(f"\ncard: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
