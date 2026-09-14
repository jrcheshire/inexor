"""E1 on a GB200: the engine with the coarse paint + solve and the tile loop on the card, against the host engine at cgh64, and the arena fold-in at brick slack 0.

Every arm is a fresh subprocess, so each device high-water mark belongs to one
arm. Every arm runs `engine.run` (tile_workers=1, the ratified dtypes,
`--steps` steps + the lead drift, a repack after every step) on a cgh64-family
state, hashes every state field, keeps the per-step stats, and times each phase
boundary.

  engine-host     host backends throughout.
  engine-eager    coarse_backend, tile_backend (device_tile_jit=False) and
                  migrate_backend all "device". PRE-REGISTERED: every state hash
                  and every host stat equal to engine-host.
  engine-device   the same with the COMPILED tile loop: the phase table and the
                  device peak. Not comparable field by field with the host run:
                  the compiled tile moves codes by up to one (record sec. 17), and
                  after that the layouts part.
  slack0-host     the host pair of the next arm, built and run at brick slack 0
                  (arena_frac `--slack0-arena`), so bricks overflow every step.
  slack0-eager    engine-eager at brick slack 0: the arena fold-in and resident
                  re-homing on the GPU at cgh64. PRE-REGISTERED: equal to
                  slack0-host, with a nonzero spill count as the receipt.

Each arm saves `w` and `vel_scale` at its FIRST `tile_loop_end` (step 1, after
the lead drift) under `--snap-dir`, outside the phase timing. The orchestrator
compares eager against host there (bitwise expected; it places a final-hash
mismatch in the tile step or after it) and compiled against eager (no code may
move by more than one, the step-level form of the sec. 17 floor; scale
differences reported).

Device arms record their platform and refuse `cpu` unless `--allow-cpu` (the
laptop smoke). Output streams line by line and the card is rewritten after
every arm, so a killed job keeps what finished.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from v2_d3_device_migrate import (  # noqa: E402
    _maxrss_gb, _nb_of, _peak, _require_device, _say, _x64,
)
from v2_r2_device_engine import _coeffs, _hashes, _json_ready, _Phases, _preset_of  # noqa: E402

EAGER = dict(coarse_backend="device", tile_backend="device", device_tile_jit=False,
             migrate_backend="device")
ARM_KW = {
    "engine-host": {},
    "engine-eager": dict(EAGER),
    "engine-device": dict(coarse_backend="device", tile_backend="device",
                          migrate_backend="device"),
    "slack0-host": dict(brick_slack=0.0),
    "slack0-eager": dict(EAGER, brick_slack=0.0),
}
SLACK0 = ("slack0-host", "slack0-eager")
#: (device arm, host arm) pairs gated bitwise
PAIRS = (("engine-eager", "engine-host"), ("slack0-eager", "slack0-host"))
#: stats keys that NAME the lane rather than describe the step
RECEIPTS = ("migrate_backend", "migrate_device", "coarse_backend", "tile_backend",
            "device_shapes")
#: stats keys that describe the paint's CHUNKING: the device lane paints at its
#: own default chunk (a quarter x-slab) where the host paints `chunk_bricks`, so
#: these describe different chunks. The state cannot depend on the chunk length
#: (integer addition; `test_the_device_paint_chunk_length_moves_no_state_bit`).
CHUNK_SHAPED = ("coarse_pad", "coarse_pad_true", "coarse_subblock_chunks")


def _state(preset, nb, brick_slack, arena_frac, seed=13):
    """`v2_r2_device_engine._engine_state` with the slack and arena as arguments."""
    from inexor import state
    from inexor.codec import T9Layout
    from inexor.plan import PRESETS

    g = PRESETS[preset]
    n, L = int(g["n_part"]), float(g["box"])
    sp = L / n
    ax = (np.arange(n) + 0.5) * sp
    x = np.empty((n**3, 3), dtype=np.float64)
    x[:, 0] = np.repeat(ax, n * n)
    x[:, 1] = np.tile(np.repeat(ax, n), n)
    x[:, 2] = np.tile(ax, n * n)
    rng = np.random.default_rng(seed)
    for c in range(3):
        x[:, c] += rng.normal(scale=0.25 * sp, size=n**3)
        np.mod(x[:, c], L, out=x[:, c])
    v = np.random.default_rng(seed + 1).normal(scale=0.5, size=x.shape)
    t9 = T9Layout(box_size=L, n_part=n, bucket_cells=2)
    return state.SlotState.build(x, v, t9, nb, brick_slack=brick_slack,
                                 arena_frac=arena_frac, with_ids=False)


# ------------------------------------------------------------------ the arm


def arm_engine(args):
    _x64()
    from inexor import engine
    from inexor.device import migrate, paint, repack, tile
    from inexor.plan import engine_config

    platform = _require_device(args.allow_cpu)
    n = args.n_part
    nb = args.nb or _nb_of(n)
    preset = _preset_of(n)
    kw = dict(ARM_KW[args.arm], tile_workers=1)
    cfg = engine_config(preset, **kw)
    slack = cfg.brick_slack
    arena = args.slack0_arena if args.arm in SLACK0 else 0.01
    t0 = time.perf_counter()
    st = _state(preset, nb, slack, arena)
    build_s = time.perf_counter() - t0
    assert nb == st.bricks_per_side == cfg.n_fine // cfg.n_brick, (nb, cfg.n_brick)
    co = _coeffs(args.steps)
    ph = _Phases()
    os.makedirs(args.snap_dir, exist_ok=True)
    snap = dict(w=os.path.join(args.snap_dir, f"{args.arm}{args.out_suffix}_w.npy"),
                vel_scale=os.path.join(args.snap_dir,
                                       f"{args.arm}{args.out_suffix}_vel_scale.npy"))
    wrote = {}

    def hook(name):
        ph(name)
        if name == "tile_loop_end" and not wrote:
            t = time.perf_counter()
            np.save(snap["w"], st.w)
            np.save(snap["vel_scale"], st.vel_scale)
            wrote["s"] = time.perf_counter() - t
            ph.t = time.perf_counter()  # the write is not the next phase's

    calls0 = (paint.CALLS, tile.CALLS, migrate.CALLS, repack.CALLS)
    _say(f"[{args.arm}] {preset}: n_part={n} nb={nb} slack={slack} arena_frac={arena} built in "
         f"{build_s:.1f}s on {platform}; K={args.steps}; coarse={cfg.coarse_backend} "
         f"tile={cfg.tile_backend} jit={cfg.device_tile_jit} migrate={cfg.migrate_backend}")
    t1 = time.perf_counter()
    out = engine.run(st, cfg, co, phase=hook)
    run_s = time.perf_counter() - t1
    calls = tuple(b - a for a, b in zip(calls0, (paint.CALLS, tile.CALLS, migrate.CALLS,
                                                 repack.CALLS)))
    K = args.steps
    dev = cfg.coarse_backend == "device"
    want = (K if dev else 0, K if cfg.tile_backend == "device" and cfg.device_tile_jit else 0,
            K + 1 if cfg.migrate_backend == "device" else 0,
            K if cfg.migrate_backend == "device" else 0)
    rc = 0
    if calls != want:
        _say(f"FATAL: device calls (paint, tile loop, migrate, repack) {calls}, expected {want}")
        rc = 3
    spills = int(sum(o["n_arena_overflow"] for o in out))
    if args.arm in SLACK0 and spills == 0:
        _say(f"FATAL: [{args.arm}] VACUOUS -- nothing spilled at brick slack 0")
        rc = 3
    if not wrote:
        _say(f"FATAL: [{args.arm}] no tile_loop_end boundary fired; no snapshot")
        rc = 3
    steps = [dict(cap=int(o["cap"]), overflow=int(o["n_arena_overflow"]),
                  arena_used=int(o["arena_used"]), reach=int(o["brick_reach"]),
                  coarse_device_chunks=o.get("coarse_device_chunks"),
                  device_shapes=o["device_shapes"],
                  vel_scale_kick_max=o["vel_scale_kick_max"]) for o in out]
    stats = []
    for o in out:
        d = {k: v for k, v in o.items() if k not in ("pool", "busy", "loop_wall")}
        stats.append(d)
    _say(f"[{args.arm}] run {run_s:.1f}s; spills {spills}; phases: " + ", ".join(
        f"{k} {v:.1f}" for k, v in sorted(ph.total.items(), key=lambda kv: -kv[1])))
    rec = dict(arm=args.arm, platform=platform, preset=preset, n_part=n, nb=nb, steps=K,
               brick_slack=slack, arena_frac=arena, build_s=build_s, run_s=run_s,
               phases_total=ph.total, phase_trace=ph.trace, hashes=_hashes(st),
               stats=_json_ready(stats), per_step=_json_ready(steps), spills=spills,
               device_calls=calls, snapshot=snap, snapshot_write_s=wrote.get("s"),
               device_peak=_peak(), host_maxrss_gb=_maxrss_gb(), rc=rc)
    _say(f"[{args.arm}] device peak {(rec['device_peak'] or 0) / 2**30:.2f} GiB, host maxrss "
         f"{rec['host_maxrss_gb']:.1f} GB")
    return rec, rc


# ------------------------------------------------------------ the orchestrator


def _stats_diffs(host, dev):
    """(step, key) for every host stat the device card lacks or disagrees on."""
    bad = []
    if len(host) != len(dev):
        return [("steps", len(host), len(dev))]
    for k, (h, d) in enumerate(zip(host, dev)):
        for key in h:
            if key in RECEIPTS or key in CHUNK_SHAPED:
                continue
            if key not in d:
                bad.append((k, key, "missing"))
                continue
            hv, dv = h[key], d[key]
            if key == "repack" and hv is not None:
                skip = ("repack_device", "scratch_bytes")
                hv = {x: y for x, y in hv.items() if x not in skip}
                dv = {x: y for x, y in dv.items() if x not in skip}
            if hv != dv:
                bad.append((k, key))
    return bad


def _codes_compare(a_path, b_path, chunk=1 << 22):
    a = np.load(a_path, mmap_mode="r")
    b = np.load(b_path, mmap_mode="r")
    if a.shape != b.shape:
        return dict(same_shape=False, a=list(a.shape), b=list(b.shape))
    max_abs, n_diff = 0, 0
    for i in range(0, a.shape[0], chunk):
        d = np.abs(a[i:i + chunk].astype(np.int32) - b[i:i + chunk].astype(np.int32))
        if d.size:
            max_abs = max(max_abs, int(d.max()))
            n_diff += int(np.count_nonzero(d))
    return dict(same_shape=True, max_abs=max_abs, n_diff=n_diff, n=int(a.size))


def _scales_compare(a_path, b_path):
    a, b = np.load(a_path), np.load(b_path)
    if a.shape != b.shape:
        return dict(same_shape=False)
    diff = a != b
    rel = np.abs(a[diff] - b[diff]) / np.abs(b[diff]) if diff.any() else np.zeros(0)
    return dict(same_shape=True, n_diff=int(diff.sum()), n=int(a.size),
                max_rel=float(rel.max()) if rel.size else 0.0)


def _run_worker(argv, tag):
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", *argv]
    _say(f"\n--- arm {tag}: {' '.join(argv)}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=env)
    rec = None
    for line in p.stdout:
        if line.startswith("WORKER_JSON "):
            rec = json.loads(line[len("WORKER_JSON "):])
        else:
            print(line, end="", flush=True)
    rc = p.wait()
    if rec is None:
        _say(f"--- arm {tag}: rc={rc}, NO RECORD CAME BACK (not a reading)")
        return dict(arm=tag, rc=rc, record=None), max(rc, 1)
    _say(f"--- arm {tag}: rc={rc}")
    return dict(arm=tag, rc=rc, record=rec), rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--arm", choices=sorted(ARM_KW), help=argparse.SUPPRESS)
    ap.add_argument("--n-part", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--nb", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--arms",
                    default="engine-host,engine-device,slack0-host,engine-eager,slack0-eager",
                    help="comma-separated, run in this order")
    ap.add_argument("--engine-n", type=int, default=512, help="particles per side (cgh64)")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--slack0-arena", type=float, default=0.05,
                    help="arena_frac of the brick-slack-0 arms")
    ap.add_argument("--snap-dir", default=os.path.join(REPO, "runs", "v2", "e1_snap"))
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny everything, CPU allowed: exercises the apparatus only")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args(argv)

    if args.worker:
        rec, rc = arm_engine(args)
        print("WORKER_JSON " + json.dumps(_json_ready(rec)), flush=True)
        return rc

    if args.smoke:
        args.engine_n = 32
    common = ["--steps", str(args.steps), "--slack0-arena", str(args.slack0_arena),
              "--snap-dir", args.snap_dir, "--out-suffix", args.out_suffix] + \
        (["--allow-cpu"] if args.allow_cpu or args.smoke else [])
    out = os.path.join(REPO, "runs", "v2", f"e1_device_step{args.out_suffix}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    card = dict(argv=sys.argv, started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename,
                commit=subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
                arms=[])

    def write():
        with open(out, "w") as f:
            json.dump(_json_ready(card), f, indent=1)

    worst = 0
    recs = {}
    for a in [a for a in args.arms.split(",") if a]:
        res, rc = _run_worker(["--arm", a, "--n-part", str(args.engine_n), *common], a)
        card["arms"].append(res)
        recs[a] = res.get("record")
        worst = max(worst, rc)
        write()

    card["gates"] = {}
    for dev_arm, host_arm in PAIRS:
        if dev_arm not in recs and host_arm not in recs:
            continue
        d, h = recs.get(dev_arm), recs.get(host_arm)
        if d is None or h is None:
            card["gates"][dev_arm] = "MISSING an arm"
            worst = max(worst, 1)
            continue
        hashes_equal = d["hashes"] == h["hashes"]
        diffs = _stats_diffs(h["stats"], d["stats"])
        codes = _codes_compare(d["snapshot"]["w"], h["snapshot"]["w"])
        scales = _scales_compare(d["snapshot"]["vel_scale"], h["snapshot"]["vel_scale"])
        at_loop = codes.get("n_diff") == 0 and scales.get("n_diff") == 0
        card["gates"][dev_arm] = dict(against=host_arm, hashes_equal=hashes_equal,
                                      stats_diffs=diffs, tile_loop_end_codes=codes,
                                      tile_loop_end_scales=scales,
                                      spills=(d["spills"], h["spills"]),
                                      run_s=(d["run_s"], h["run_s"]))
        _say(f"\n=== {dev_arm} vs {host_arm}: state hashes equal = {hashes_equal}; host stats "
             f"differing = {len(diffs)} {diffs[:5]}; at step 1's tile_loop_end codes differ "
             f"{codes.get('n_diff')} (max {codes.get('max_abs')}), scales differ "
             f"{scales.get('n_diff')}; spills {d['spills']} vs {h['spills']}; run "
             f"{d['run_s']:.1f} vs {h['run_s']:.1f}s ===")
        if not (hashes_equal and not diffs and at_loop):
            worst = max(worst, 3)

    if recs.get("engine-device") and recs.get("engine-eager"):
        d, e = recs["engine-device"], recs["engine-eager"]
        codes = _codes_compare(d["snapshot"]["w"], e["snapshot"]["w"])
        scales = _scales_compare(d["snapshot"]["vel_scale"], e["snapshot"]["vel_scale"])
        card["gates"]["engine-device"] = dict(against="engine-eager", tile_loop_end_codes=codes,
                                              tile_loop_end_scales=scales)
        _say(f"\n=== compiled vs eager at step 1's tile_loop_end: codes differ "
             f"{codes.get('n_diff')} of {codes.get('n')} (max {codes.get('max_abs')}; floor 1); "
             f"scales differ {scales.get('n_diff')} of {scales.get('n')} (max rel "
             f"{scales.get('max_rel')}) ===")
        if not codes.get("same_shape") or codes["max_abs"] > 1:
            worst = max(worst, 3)

    if recs.get("engine-device") and recs.get("engine-host"):
        d, h = recs["engine-device"], recs["engine-host"]
        card["phase_tables"] = dict(device=d["phases_total"], host=h["phases_total"])
        _say("\n=== phases, s over the run (compiled device lane | host) ===")
        for k in sorted(set(d["phases_total"]) | set(h["phases_total"]),
                        key=lambda k: -max(d["phases_total"].get(k, 0), h["phases_total"].get(k, 0))):
            _say(f"  {k:<16} {d['phases_total'].get(k, 0):8.1f} | {h['phases_total'].get(k, 0):8.1f}")

    card["rc"] = worst
    write()
    _say(f"\ncard: {out}\nrc={worst}")
    return worst


if __name__ == "__main__":
    sys.exit(main())
