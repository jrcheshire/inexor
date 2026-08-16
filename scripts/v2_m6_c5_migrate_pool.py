"""C5: does `migrate` PARALLELISE, or does it inflate like `tile_long`?

Section 5l closed the per-brick lane -- the Python slab loop is 2.0% of migrate
at the 4,096 rows/brick the config table fixes at every rung -- so migrate is
per-ROW work and parallelism is the only lever left. Section 5j is the reason it
matters: migrate is 32.78 s of a 72.64 s W=16 step on gg and **23.3 h of a
projected 51.7 h C-gh realization**, and it is host numpy, so neither the tile
pool nor the GPU lane touches it today.

THE CANARY PRINCIPLE, and it is why this is a script and not engine surgery:
C2 measured the pool ceiling on a real state before W2 touched `engine.py`, and
that is what kept the build from being wasted. This does the same for migrate.

## What is measured

`_eject_slab` only, and deliberately so. It is the half whose parallelism needs
no premise: its own docstring says it "reads the state; writes NOTHING back",
which is a phase separation the module maintains on purpose. So leg A can be
driven from W workers over shm with no correctness question at all, and the
number it returns is a clean ceiling for the read side.

**The Amdahl bound this is read against, fixed before the run.** At cgh64/nb32,
job 478 measured eject 12.99 s of migrate's 33.26 s. So even if eject
parallelises PERFECTLY, migrate goes 33.3 -> 20.3 s, a **1.64x ceiling on the
phase**. Anything claiming more than that from this leg is misreading it.
`_insert_slab` carries the other 19.7 s and needs the disjoint-write premise
proved as C2 proved it for tiles; that is a separate job and is NOT this one.

## The pre-registration

**I expect migrate to scale BETTER than `tile_long`, and this can embarrass me.**
`tile_long` inflates 4.13x in worker-seconds at W=32 (5j) because FFTs are
bandwidth-hungry per unit of work. migrate's traffic is small in aggregate: even
generously counting the f64 decode transients (`decode_brick` materialises
(rows,3) float64 positions AND velocities, ~48 B/row, so ~6.4 GB across N at
cgh64, plus index and mask arrays), eject moves order tens of GB in ~13 s -- a
few GB/s against gg's ~500 GB/s/socket, order 1% of the fabric.

    KILL CRITERION: efficiency < 50% at W=8 kills the pool lane for migrate.
    Then 51.7 h/realization is near a floor and the answer is architectural.

If it fails to scale ANYWAY, DRAM bandwidth is not the cause and the candidates
are allocator/page-fault serialisation or per-brick call overhead going
superlinear under concurrency -- a different fix, and worth knowing that early.

## Transport is an ARM, not a detail

Eject RETURNS large arrays (all of a slab's keepers and emigrants). In a process
pool those cross a boundary, and that cost is real for any implementation:

    discard   worker drops the return    -> the COMPUTE ceiling (optimistic)
    counts    worker returns row counts  -> compute + task overhead
    pickle    worker returns the arrays  -> what a naive implementation pays

C2 hit the same fork and named the shm-slab return path as the priced
alternative to pickling. Reporting all three keeps "the compute scales but the
transport eats it" from being invisible, which is exactly how a ceiling gets
quoted as a speedup.

Usage:
  pixi run python scripts/v2_m6_c5_migrate_pool.py --config smoke --workers 2 4
  pixi run python scripts/v2_m6_c5_migrate_pool.py --config cgh64 --workers 8 16 32
"""

import argparse
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from multiprocessing import shared_memory

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from inexor import state  # noqa: E402
from inexor.executor import SHM_FIELDS  # noqa: E402
from v2_m6_migrate_depth import CONFIGS, FRACTIONS, _build  # noqa: E402

_G = {}


def _rss_mb():
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1e3
    except OSError:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    return -1.0


def _worker_init(shm_names, shapes, dtypes, small, scales, core_sets, rank_counter):
    """Attach the shm views and build a read-only SlotState facade.

    No jax anywhere in this path -- migrate is pure numpy, so a worker starts in
    milliseconds rather than paying an import and a compile as C2's tile workers
    did. Affinity is set BEFORE anything allocates, matching the executor: 5i
    measured pinning worth 20-49% on gg's two NUMA domains.
    """
    t0 = time.perf_counter()
    if core_sets is not None and hasattr(os, "sched_setaffinity"):
        with rank_counter.get_lock():
            rank = rank_counter.value
            rank_counter.value += 1
        os.sched_setaffinity(0, core_sets[rank % len(core_sets)])
    views, segs = {}, []
    for name in SHM_FIELDS:
        seg = shared_memory.SharedMemory(name=shm_names[name])
        segs.append(seg)
        views[name] = np.ndarray(shapes[name], dtype=dtypes[name], buffer=seg.buf)
    st = state.SlotState(
        t9=small["t9"], bricks_per_side=small["bricks_per_side"],
        brick_start=views["brick_start"], occupancy=views["occupancy"],
        off=views["off"], w=views["w"], vel_scale=views["vel_scale"],
        arena_base=small["arena_base"], arena_bucket=views["arena_bucket"],
        n_particles=small["n_particles"], ids=None,
    )
    _G.update(st=st, segs=segs, scales=scales, init_s=time.perf_counter() - t0)


def _worker_eject(arg):
    """One slab's ejection. Returns per the transport arm."""
    bx, c_drift, transport = arg
    t0 = time.perf_counter()
    keep, emig = _G["st"]._eject_slab(int(bx), float(c_drift), _G["scales"])
    busy = time.perf_counter() - t0
    out = dict(bx=int(bx), busy_s=busy, rss_mb=_rss_mb(), pid=os.getpid())
    if transport == "counts":
        out["n_keep"] = int(len(keep["dest"])) if keep.get("dest") is not None else 0
        out["n_emig"] = int(len(emig["dest"])) if emig.get("dest") is not None else 0
    elif transport == "pickle":
        out["keep"], out["emig"] = keep, emig
    return out


def _serial_eject(st, c_drift, scales, nb):
    """The in-process reference. Same calls, same order, same state."""
    t0 = time.perf_counter()
    busy = []
    for bx in range(nb):
        t1 = time.perf_counter()
        st._eject_slab(bx, c_drift, scales)
        busy.append(time.perf_counter() - t1)
    return time.perf_counter() - t0, busy


def run(cfg_name, workers, transport, brick_slack, arena_frac, f_frac, affinity):
    cfg = CONFIGS[cfg_name]
    nb = int(cfg["nb"])
    print(f"== C5 {cfg_name}: n_part={cfg['n_part']} nb={nb} slack={brick_slack} "
          f"f={f_frac} transport={transport} affinity={affinity}", flush=True)
    st = _build(cfg["n_part"], nb, cfg["box"], brick_slack=brick_slack,
                arena_frac=arena_frac)
    extent = cfg["box"] / nb
    scales = np.array(st.vel_scale, dtype=np.float64, copy=True)
    c1 = extent / (float(np.max(st.vel_scale)) * state.INT16_MAX)
    c_drift = float(f_frac) * c1
    reach = int(state.brick_reach(st, c_drift, scales))

    # SERIAL FIRST, on the untouched state. Eject writes nothing back, so the
    # state is unchanged and every pooled arm below runs against the identical
    # input -- which is what makes these ratios comparable at all.
    ser_wall, ser_busy = _serial_eject(st, c_drift, scales, nb)
    # A SCALING NUMBER NEEDS WORK TO SCALE. At `smoke` the whole slab set ejects
    # in ~0.03 s against ~0.26 s of pool creation and dispatch, so every arm
    # measures the HARNESS and prints a speedup below 1 that looks like a
    # finding. Flagged rather than suppressed -- the smoke config exists to
    # exercise the plumbing on the laptop, and it should say that is all it did.
    harness_dominated = ser_wall < 1.0
    print(f"  serial: {ser_wall:.3f} s for {nb} slabs "
          f"({nb / ser_wall:.2f} slabs/s), busy sum {sum(ser_busy):.3f} s",
          flush=True)
    if harness_dominated:
        print("  HARNESS-DOMINATED: serial eject is under 1 s, so pool creation "
              "and dispatch set the wall. This arm validates PLUMBING ONLY -- "
              "its speedups are not a scaling measurement and must not be "
              "quoted as one.", flush=True)

    # adopt the state arrays into shm ONCE; every W arm reuses the segments
    names, shapes, dtypes, segs = {}, {}, {}, []
    for fld in SHM_FIELDS:
        arr = np.asarray(getattr(st, fld))
        seg = shared_memory.SharedMemory(create=True, size=max(arr.nbytes, 1))
        view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=seg.buf)
        view[...] = arr
        setattr(st, fld, view)
        segs.append(seg)
        names[fld], shapes[fld], dtypes[fld] = seg.name, arr.shape, arr.dtype
    st._invalidate_arena_index()
    small = dict(t9=st.t9, bricks_per_side=st.bricks_per_side,
                 n_particles=st.n_particles, arena_base=st.arena_base)

    arms = []
    try:
        for W in workers:
            ctx = mp.get_context("spawn")
            rank_counter = ctx.Value("i", 0)
            core_sets = None
            if affinity and hasattr(os, "sched_getaffinity"):
                cores = sorted(os.sched_getaffinity(0))
                per = max(1, len(cores) // W)
                core_sets = [cores[i * per:(i + 1) * per] or cores[-per:]
                             for i in range(W)]
            saved = os.environ.get("OMP_NUM_THREADS")
            os.environ["OMP_NUM_THREADS"] = "1"
            t_start = time.perf_counter()
            try:
                pool = ctx.Pool(W, initializer=_worker_init,
                                initargs=(names, shapes, dtypes, small, scales,
                                          core_sets, rank_counter))
            finally:
                if saved is None:
                    os.environ.pop("OMP_NUM_THREADS", None)
                else:
                    os.environ["OMP_NUM_THREADS"] = saved
            startup = time.perf_counter() - t_start
            tasks = [(bx, c_drift, transport) for bx in range(nb)]
            t0 = time.perf_counter()
            res = list(pool.imap_unordered(_worker_eject, tasks))
            wall = time.perf_counter() - t0
            pool.close()
            pool.join()
            busy = float(sum(r["busy_s"] for r in res))
            rss = max(r["rss_mb"] for r in res)
            arm = dict(
                workers=W, wall_s=wall, busy_total_s=busy,
                concurrency=busy / wall if wall > 0 else 0.0,
                idle_s=W * wall - busy,
                speedup=ser_wall / wall if wall > 0 else 0.0,
                efficiency=(ser_wall / wall) / W if wall > 0 else 0.0,
                busy_inflation=busy / sum(ser_busy) if sum(ser_busy) > 0 else 0.0,
                slabs_per_s=nb / wall if wall > 0 else 0.0,
                rss_mb_max=rss, startup_s=startup,
                n_distinct_pids=len({r["pid"] for r in res}),
            )
            arms.append(arm)
            print(f"  W={W:<3} wall {wall:7.3f} s  speedup {arm['speedup']:5.2f}x  "
                  f"eff {arm['efficiency'] * 100:5.1f}%  conc {arm['concurrency']:5.2f}  "
                  f"busy-inflation {arm['busy_inflation']:5.2f}x  "
                  f"idle {arm['idle_s']:7.2f} s  RSS {rss:.0f} MB  "
                  f"start {startup:.2f} s", flush=True)
    finally:
        for seg in segs:
            seg.close()
            seg.unlink()

    return dict(config=cfg_name, n_part=cfg["n_part"], nb=nb,
                n_particles=cfg["n_part"] ** 3, f=float(f_frac),
                brick_reach=reach, c_drift=c_drift, transport=transport,
                affinity=bool(affinity), brick_slack=brick_slack,
                arena_frac=arena_frac, serial_wall_s=ser_wall,
                harness_dominated=bool(harness_dominated),
                serial_busy_s=float(sum(ser_busy)), arms=arms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="smoke", choices=sorted(CONFIGS))
    ap.add_argument("--workers", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--transport", nargs="+", default=["discard"],
                    choices=["discard", "counts", "pickle"])
    ap.add_argument("--brick-slack", type=float, default=0.0)
    ap.add_argument("--arena-frac", type=float, default=0.20)
    ap.add_argument("--f", type=float, default=FRACTIONS[-1],
                    help="fraction of the reach-1 threshold; default = the "
                         "production rung (reach 3)")
    ap.add_argument("--no-affinity", action="store_true")
    ap.add_argument("--out", default=os.path.join("runs", "v2", "m6_c5_migrate_pool.json"))
    a = ap.parse_args()

    results = []
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for transport in a.transport:
        results.append(run(a.config, a.workers, transport, a.brick_slack,
                           a.arena_frac, a.f, not a.no_affinity))
        with open(a.out, "w") as fh:
            json.dump(dict(results=results, partial=True), fh, indent=1)

    # THE VERDICT, against the pre-registered bar and the Amdahl bound. Read on
    # the compute arm if one was run: `discard` is the ceiling the other arms
    # are measured against, and quoting a transport-loaded number as "migrate
    # does not parallelise" would blame the wrong thing.
    print("== verdict")
    for res in results:
        if res.get("harness_dominated"):
            print(f"  {res['transport']:8s}: HARNESS-DOMINATED, no scaling verdict "
                  f"(serial {res['serial_wall_s']:.3f} s)")
            continue
        w8 = [x for x in res["arms"] if x["workers"] == 8]
        line = f"  {res['transport']:8s}: "
        if w8:
            eff = w8[0]["efficiency"]
            line += (f"efficiency at W=8 = {eff * 100:.1f}% -> "
                     f"{'PASS' if eff >= 0.50 else 'FAIL (pool lane killed)'}")
        else:
            best = max(res["arms"], key=lambda x: x["speedup"])
            line += (f"no W=8 arm; best {best['speedup']:.2f}x at W={best['workers']} "
                     f"(eff {best['efficiency'] * 100:.1f}%)")
        print(line)
        best = max(res["arms"], key=lambda x: x["speedup"])
        # eject is 12.99 of migrate's 33.26 s at cgh64/nb32 (job 478); the phase
        # ceiling follows from Amdahl WITHIN migrate and is 1.64x even at
        # infinite W, so it is printed beside every speedup on purpose
        ej, tot = 12.99, 33.26
        phase = tot / (tot - ej + ej / best["speedup"]) if best["speedup"] > 0 else 1.0
        print(f"            best {best['speedup']:.2f}x at W={best['workers']} "
              f"-> migrate {tot:.2f} -> {tot - ej + ej / best['speedup']:.2f} s "
              f"= {phase:.2f}x on the PHASE (ceiling 1.64x; insert is untouched)")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    with open(a.out, "w") as fh:
        json.dump(dict(results=results, commit=commit, machine=platform.machine(),
                       system=platform.system(), argv=sys.argv[1:]), fh, indent=1)
    print(f"card -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
