"""C2 of the wall plan: the process-pool tile loop -- bitwise gate, scaling, RSS.

THE load-bearing canary. W2's executor design is persistent worker processes
over `multiprocessing.shared_memory`; this script IS that skeleton, run
against a real engine state, before any engine surgery:

  1. Build a real state at a named config (same `_build` as the ratified
     phase-time instrument), apply the lead drift, run the ONE-step preamble
     exactly as `engine.step` does (coarse solve, membership, `cap`,
     `make_tile_force_fn`).
  2. Factor the engine.py:695-786 loop body into `tile_task(...) ->
     TileResult` + `apply_result(st, res)` -- the same seam the real executor
     will use -- and drive it two ways from IDENTICAL state snapshots:
     serially in-process, and over W spawned workers reading the state through
     shm, with the parent applying results in ARRIVAL order.
  3. Gate: `st.w` and `st.vel_scale` after the two arms must be IDENTICAL --
     n_diff == 0, elementwise. Arrival-order application makes this the strong
     form of the disjointness claim. A nonzero does not mean "tune it"; it
     means the parallel premise itself is wrong (a cross-tile dependence the
     seam analysis missed).

Why a frozen state snapshot is bitwise-equivalent to the serial loop's live
reads: the kick consumes velocities only at OWNED rows (engine.py:758), a
brick is written only by the tile that owns it, and positions are untouched by
the kick -- so the buffer-brick velocities that DO differ under a frozen
snapshot are decoded and discarded in both arms.

Readouts (the plan's C2 kill criteria):
  bitwise      n_diff(w) + n_diff(vel_scale)     nonzero kills the premise
  efficiency   serial_wall / (W x pool_wall)     < 50% at W=8 -> redesign
  RSS/worker   max over workers (VmHWM on Linux) > 4 GB at P=320 -> width fails
  startup      pool creation incl. per-worker one_tile compile

Transport is pickled results (transport A); the shm-slab return path is the
priced alternative if pickling shows up in the idle numbers. Worker env: the
parent sets OMP_NUM_THREADS=1 (+ optional --xla-flags) before spawning;
which flag actually caps the XLA-CPU spin pool is C3's question, not assumed
here. macOS runs validate the BITWISE gate and the dispatch shape; scaling and
RSS verdicts come from the Linux legs (deneb/antares/gg).

Usage:
  pixi run python scripts/v2_m6_c2_pool.py --config cdev8 --tile 64 --workers 2 4
  pixi run python scripts/v2_m6_c2_pool.py --config cdev --tile 256 --workers 4 8
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import shared_memory  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

SHM_FIELDS = ("off", "w", "occupancy", "brick_start", "vel_scale", "arena_bucket")


# --------------------------------------------------------------- shared body
def tile_task(st, one_tile, C, g_coarse, t, bricks):
    """The engine.py:695-786 body for ONE tile, writes returned not applied.

    `st` is read-only here. Mirrors the engine line for line; the docstrings
    for each move live there, not here.
    """
    import jax.numpy as jnp

    from inexor.forces import (
        COARSE_HALO,
        coarse_subblock_origin_extent,
        gather_coarse_subblock,
        owned_mask_from_bricks,
        stage_coarse_subblock,
        tile_origin_extent,
    )

    t_dec = time.perf_counter()
    slots, x, v = st.decode_bricks(bricks)
    brick_of_row = np.repeat(
        np.asarray(bricks, dtype=np.int64), [st.brick_member_count(b) for b in bricks]
    )
    m = len(slots)
    if m == 0:
        return dict(t=t, empty=True, n_owned=0, n_out=0)
    if m > C["cap"]:
        raise RuntimeError(f"tile {t}: {m} members > cap {C['cap']}")
    idx = np.resize(np.arange(m), C["cap"])
    live = np.zeros(C["cap"], dtype=bool)
    live[:m] = True
    origin, _ = tile_origin_extent(t, C["n_tile"], C["b_real"], C["cell"])
    u = jnp.mod(jnp.asarray(x[idx]) - jnp.asarray(origin), C["box"])
    own_rows = owned_mask_from_bricks(
        brick_of_row, t, C["n_tile"], C["n_brick"], C["n_fine"] // C["n_brick"]
    )
    own = np.zeros(C["cap"], dtype=bool)
    own[:m] = own_rows
    own &= live
    t_short = time.perf_counter()
    g_short, owned, n_out = one_tile(u, jnp.asarray(live), jnp.asarray(own))
    g_short = np.asarray(g_short)[:m]
    owned = np.asarray(owned)[:m]
    t_long = time.perf_counter()
    if not owned.any():
        return dict(t=t, empty=True, n_owned=0, n_out=int(n_out))
    o_cells, extent = coarse_subblock_origin_extent(
        t, C["n_tile"], C["n_coarse"], C["n_fine"], halo=COARSE_HALO
    )
    sub = [stage_coarse_subblock(g, o_cells, extent) for g in g_coarse]
    n_own = int(owned.sum())
    xo = np.zeros((C["cap"], 3), dtype=np.float64)
    xo[:n_own] = x[owned]
    lv = np.zeros(C["cap"], dtype=bool)
    lv[:n_own] = True
    g_long = np.asarray(
        gather_coarse_subblock(
            *sub, jnp.asarray(xo), o_cells, C["coarse_cell"], C["n_coarse"],
            assign="tsc", live=lv,
        )
    )[:n_own]
    t_quant = time.perf_counter()
    g_tot = g_short[owned] + g_long
    v_new = C["alpha_k"] * v[owned] + C["bcoef"] * g_tot
    slots_o, bricks_o = slots[owned], brick_of_row[owned]
    cut = np.flatnonzero(np.diff(bricks_o)) + 1
    run_lo = np.concatenate(([0], cut))
    run_hi = np.concatenate((cut, [len(bricks_o)]))
    if len(np.unique(bricks_o)) != len(run_lo):
        raise AssertionError(f"tile {t}: owned rows are not grouped by brick")
    from inexor.codec import INT16_MAX, assert_int16_range

    w_codes = np.empty((n_own, 3), dtype=np.int16)
    run_bricks = np.empty(len(run_lo), dtype=np.int64)
    run_scales = np.empty(len(run_lo), dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(run_lo, run_hi)):
        vb = v_new[lo:hi]
        s_b = float(np.max(np.abs(vb))) / INT16_MAX
        s_b = s_b if s_b > 0.0 else 1.0
        w_b = np.rint(vb / s_b)
        assert_int16_range(w_b)
        w_codes[lo:hi] = w_b.astype(np.int16)
        run_bricks[i] = int(bricks_o[lo])
        run_scales[i] = s_b
    t_end = time.perf_counter()
    return dict(
        t=t, empty=False, slots_o=slots_o, w_codes=w_codes, run_bricks=run_bricks,
        run_scales=run_scales, n_owned=n_own, n_out=int(n_out),
        busy=dict(decode=t_short - t_dec, short=t_long - t_short,
                  long=t_quant - t_long, quant=t_end - t_quant),
    )


def apply_result(st, res):
    """Parent-side write application. Disjoint across tiles by the partition."""
    if res["empty"]:
        return
    st.write_velocities(res["slots_o"], res["w_codes"])
    st.vel_scale[res["run_bricks"]] = res["run_scales"]


# ------------------------------------------------------------------- workers
_G = {}


def _rss_mb():
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1e3  # kB -> MB
    except OSError:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # macOS bytes
    return -1.0


def _worker_init(shm_names, shapes, dtypes, small, C, fn_args, core_sets, rank_counter):
    """Attach shm state views, rebuild the read-only SlotState, compile one_tile.

    AFFINITY FIRST, before jax exists in this process: XLA-CPU sizes its
    spin-waiting pool by the VISIBLE cores and ignores every thread env var
    (umbrella: thread-count-is-part-of-the-pin; affinity is the only knob).
    Job 466 measured the un-pinned consequence on antares: walls GROWING with
    W (9.05 -> 11.50 s, W=2 -> 16) as 16 workers each spun a 28-core pool.
    """
    if core_sets is not None and hasattr(os, "sched_setaffinity"):
        with rank_counter.get_lock():
            rank = rank_counter.value
            rank_counter.value += 1
        os.sched_setaffinity(0, set(core_sets[rank % len(core_sets)]))
    t0 = time.perf_counter()
    from inexor import state as state_mod
    from inexor.forces import make_tile_force_fn

    import jax

    jax.config.update("jax_enable_x64", True)

    views, segs = {}, []
    for name in SHM_FIELDS:
        seg = shared_memory.SharedMemory(name=shm_names[name])
        segs.append(seg)
        views[name] = np.ndarray(shapes[name], dtype=dtypes[name], buffer=seg.buf)
    g_coarse = []
    for i in range(3):
        seg = shared_memory.SharedMemory(name=shm_names[f"g{i}"])
        segs.append(seg)
        g_coarse.append(np.ndarray(shapes[f"g{i}"], dtype=dtypes[f"g{i}"], buffer=seg.buf))
    st = state_mod.SlotState(
        t9=small["t9"], bricks_per_side=small["bricks_per_side"],
        brick_start=views["brick_start"], occupancy=views["occupancy"],
        off=views["off"], w=views["w"], vel_scale=views["vel_scale"],
        arena_base=small["arena_base"], arena_bucket=views["arena_bucket"],
        n_particles=small["n_particles"], ids=None,
    )
    one_tile, _ = make_tile_force_fn(**fn_args)
    # compile NOW, in init, so the timed pass is steady-state in every worker
    import jax.numpy as jnp

    cap = C["cap"]
    u0 = jnp.zeros((cap, 3), dtype=jnp.float64)
    z = jnp.zeros(cap, dtype=bool)
    jax.block_until_ready(one_tile(u0, z, z))
    _G.update(st=st, one_tile=one_tile, C=C, g_coarse=g_coarse, segs=segs,
              init_s=time.perf_counter() - t0)


def _worker_task(arg):
    t, bricks = arg
    res = tile_task(_G["st"], _G["one_tile"], _G["C"], _G["g_coarse"], tuple(t), bricks)
    res["worker"] = os.getpid()
    res["rss_mb"] = _rss_mb()
    res["init_s"] = _G["init_s"]
    return res


# -------------------------------------------------------------------- parent
def _clone(st):
    arrays = {f: np.array(getattr(st, f), copy=True) for f in SHM_FIELDS}
    return dataclasses.replace(st, ids=None, _arena_by_brick=None, **arrays)


def _preamble(engine, ec, st, coeff, jnp_mod):
    """engine.py:640-686, verbatim calls: coarse solve, membership, cap, force fn."""
    from inexor.forces import capacity_shape, coarse_force_meshes, make_tile_force_fn, tile_capacity

    delta = engine.coarse_delta_streamed(st, ec)
    g_coarse = coarse_force_meshes(
        jnp_mod.asarray(delta), ec.n_coarse, ec.box_size, "long", r_s=ec.r_s,
        match=(ec.coarse_cell, ec.fine_cell), fdtype=ec.np_coarse_dtype,
    )
    g_coarse = [np.asarray(g) for g in g_coarse]
    b_real = ec._b_realized
    members = {
        t: st.tile_bricks(t, ec.n_tile, b_real, ec.n_brick, ec.n_fine) for t in ec.tiles
    }
    counts = [sum(st.brick_member_count(b) for b in members[t]) for t in ec.tiles]
    cap = capacity_shape(tile_capacity(counts), rungs=ec.cap_rungs, floor_shape=0)
    fn_args = dict(
        n_fine=ec.n_fine, box_size=ec.box_size, n_particles_total=ec.n_total,
        n_tile=ec.n_tile, b_fine=ec.b_fine, r_s=ec.r_s, paint=ec.paint_short,
        frac_bits=ec.frac_bits, fdtype=ec.np_fine_dtype,
    )
    one_tile, geom = make_tile_force_fn(**fn_args)
    C = dict(cap=int(cap), n_tile=ec.n_tile, n_brick=ec.n_brick, n_fine=ec.n_fine,
             n_coarse=ec.n_coarse, box=ec.box_size, coarse_cell=ec.coarse_cell,
             cell=geom["cell"], b_real=int(b_real),
             alpha_k=float(coeff[0]), bcoef=float(coeff[1]))
    return g_coarse, members, C, one_tile, fn_args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cdev8")
    ap.add_argument("--tile", type=int, default=64)
    ap.add_argument("--buf", type=int, default=32)
    ap.add_argument("--workers", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--slack", type=float, default=0.20)
    ap.add_argument("--arena-frac", type=float, default=0.20)
    ap.add_argument("--xla-flags", default=None, help="C3 sweep passthrough for workers")
    ap.add_argument("--affinity", action="store_true",
                    help="pin each worker to a disjoint core set (Linux; the C3 knob)")
    ap.add_argument("--out", default=os.path.join("runs", "v2", "m6_c2_pool.json"))
    a = ap.parse_args()

    # worker thread caps are inherited at spawn; set them BEFORE any pool
    os.environ["OMP_NUM_THREADS"] = "1"
    if a.xla_flags:
        os.environ["XLA_FLAGS"] = a.xla_flags

    import v2_m3_engine_gate as m3
    import v2_m6_phase_time as pt

    engine, ec, st, cosmo, a_grid, bfc, bft = pt._build(
        a.config, a.slack, a.arena_frac, tile=a.tile, buf=a.buf
    )
    import jax
    import jax.numpy as jnp

    from inexor.state import drift_and_migrate

    co = bfc(bft(a_grid(m3.A_INIT, m3.A_FINAL, 1, m3.SPACING), cosmo))
    lead, fused = engine.fused_drifts(co)
    drift_and_migrate(st, lead)  # onto the first midpoint, as engine.run does
    coeff = (co[0][1], co[0][2])

    g_coarse, members, C, one_tile, fn_args = _preamble(engine, ec, st, coeff, jnp)
    tiles = [t for t in ec.tiles if len(members[t])]
    tasks = [(t, np.asarray(members[t], dtype=np.int64)) for t in tiles]
    print(f"C2: {a.config} tile={a.tile} -> {len(tiles)} tiles, cap={C['cap']}, "
          f"P={C['n_tile'] + 2 * C['b_real']}")

    # ---- serial reference (the gate arm), warmed then timed
    u0 = jnp.zeros((C["cap"], 3), dtype=jnp.float64)
    z = jnp.zeros(C["cap"], dtype=bool)
    jax.block_until_ready(one_tile(u0, z, z))
    st_ser = _clone(st)
    t0 = time.perf_counter()
    n_owned = 0
    for t, bricks in tasks:
        res = tile_task(st_ser, one_tile, C, g_coarse, t, bricks)
        apply_result(st_ser, res)
        n_owned += res["n_owned"]
    serial_wall = time.perf_counter() - t0
    assert n_owned == st.n_particles, f"partition broken in serial arm: {n_owned}"
    print(f"  serial: {serial_wall:.2f} s over {len(tiles)} tiles "
          f"({serial_wall / len(tiles) * 1e3:.0f} ms/tile)")

    # ---- shm segments for the parallel arms
    shm_names, shapes, dtypes, segs = {}, {}, {}, []

    def _share(key, arr):
        seg = shared_memory.SharedMemory(create=True, size=max(arr.nbytes, 1))
        view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=seg.buf)
        view[...] = arr
        segs.append(seg)
        shm_names[key], shapes[key], dtypes[key] = seg.name, arr.shape, str(arr.dtype)
        return view

    for f in SHM_FIELDS:
        _share(f, np.asarray(getattr(st, f)))
    for i, g in enumerate(g_coarse):
        _share(f"g{i}", g)
    small = dict(t9=st.t9, bricks_per_side=st.bricks_per_side,
                 arena_base=st.arena_base, n_particles=st.n_particles)

    ctx = mp.get_context("spawn")
    arms = {}
    mismatch_total = 0
    try:
        for W in a.workers:
            core_sets = None
            if a.affinity and hasattr(os, "sched_getaffinity"):
                cores = sorted(os.sched_getaffinity(0))
                per = max(1, len(cores) // W)
                core_sets = [cores[i * per:(i + 1) * per] or cores[-per:] for i in range(W)]
            rank_counter = ctx.Value("i", 0)
            t0 = time.perf_counter()
            with ctx.Pool(W, initializer=_worker_init,
                          initargs=(shm_names, shapes, dtypes, small, C, fn_args,
                                    core_sets, rank_counter)) as pool:
                # first pass warms nothing extra (compile ran in init) but
                # faults shm pages into each worker; timed passes follow
                st_par = _clone(st)
                p0 = time.perf_counter()
                startup = p0 - t0
                walls, busy = [], []
                for rep in range(a.repeats + 1):
                    st_rep = st_par if rep == 0 else _clone(st)
                    r0 = time.perf_counter()
                    n_owned = 0
                    b_sum = 0.0
                    rss = {}
                    inits = {}
                    for res in pool.imap_unordered(_worker_task, tasks):
                        apply_result(st_rep, res)  # ARRIVAL order, deliberately
                        n_owned += res["n_owned"]
                        if not res["empty"]:
                            b_sum += sum(res["busy"].values())
                        rss[res["worker"]] = max(rss.get(res["worker"], 0.0), res["rss_mb"])
                        inits[res["worker"]] = res["init_s"]
                    wall = time.perf_counter() - r0
                    assert n_owned == st.n_particles, f"partition broken at W={W}"
                    if rep == 0:
                        n_w = int((st_rep.w != st_ser.w).sum())
                        n_s = int((st_rep.vel_scale != st_ser.vel_scale).sum())
                        mismatch_total += n_w + n_s
                        print(f"  W={W}: bitwise n_diff(w)={n_w} n_diff(vel_scale)={n_s}")
                    else:
                        walls.append(wall)
                        busy.append(b_sum)
                wall = float(np.min(walls))
                eff = serial_wall / (W * wall)
                # NB `startup` is pool CREATION only -- Pool() returns before the
                # initializers finish, so the honest per-worker startup is
                # init_s (shm attach + kernel build + one_tile compile),
                # reported from inside each worker.
                arms[str(W)] = dict(
                    pool_create_s=startup, walls_s=walls, wall_s=wall,
                    speedup=serial_wall / wall, efficiency=eff,
                    busy_s=float(np.mean(busy)), idle_s=float(W * wall - np.mean(busy)),
                    rss_mb=rss, init_s=inits,
                )
                print(f"  W={W}: wall {wall:.2f} s  speedup {serial_wall / wall:.2f}x  "
                      f"eff {eff * 100:.0f}%  worker init {max(inits.values()):.1f} s  "
                      f"max RSS {max(rss.values()):.0f} MB")
    finally:
        for seg in segs:
            seg.close()
            seg.unlink()

    verdict = []
    if mismatch_total:
        verdict.append("BITWISE FAIL: the parallel premise is dead, stop here")
    wmax = str(max(int(w) for w in arms))
    if arms and arms[wmax]["efficiency"] < 0.5 and int(wmax) >= 8:
        verdict.append(f"efficiency {arms[wmax]['efficiency']:.0%} at W={wmax} < 50%: redesign")
    if not verdict:
        verdict.append("PASS so far (Linux legs own the scaling/RSS verdicts)")
    print("VERDICT: " + "; ".join(verdict))

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(dict(config=a.config, tile=a.tile, buf=a.buf, n_tiles=len(tiles),
                       affinity=bool(a.affinity),
                       cap=C["cap"], serial_wall_s=serial_wall, arms=arms,
                       bitwise_mismatches=mismatch_total, verdict=verdict,
                       commit=commit, machine=platform.machine(),
                       system=platform.system(), argv=sys.argv[1:]), fh, indent=1)
    print(f"card -> {a.out}")
    return 1 if mismatch_total else 0


if __name__ == "__main__":
    sys.exit(main())
