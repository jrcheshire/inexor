"""E2-E4 + R3 on a gb node: the device step on four cards -- the force meshes solved onto the cards, the tile window, the paint/solve/tile split, and the migrate and repack split -- each against its one-card reference at cgh64.

Every arm is a fresh subprocess, so each device high-water mark belongs to one
arm. State: the cgh64-family perturbed lattice of `v2_e1_device_step._state`.

  solve-cards    E4. One cgh64 density (painted on one card); the factorized
                 solve into four card shards (`CardShards`) against the same
                 solve into host meshes then `shard_coarse_meshes`. BITWISE (in
                 the arm). Warm wall of each, host tracemalloc peak of each.
  window-off     compiled lane on one card, whole state on the card (E1's path),
                 K=`--steps` + lead drift, repack every step.
  window-on      the same with `device_tile_window=True` (E2). PRE-REGISTERED:
                 hashes and stats equal window-off.
  r3-w1 / r3-w4  R3 alone: two device migrate + repack passes on one card and on
                 four, same state. PRE-REGISTERED: hashes and stats equal. The
                 second migrate is timed (synced, per card).
  cards          compiled lane, window on, `device_cards=4`: paint, solve, tile
                 loop, migrate and repack split (E3 + R3). PRE-REGISTERED: hashes
                 and stats (card receipts aside) equal window-on.
  slack0-eager   the eager lane on one card at brick slack 0 (bitwise the host
                 engine on a GB200, record sec. 34).
  slack0-cards   `cards` at brick slack 0. PRE-REGISTERED: hashes and stats equal
                 slack0-eager, and step 1's codes and scales at `tile_loop_end`
                 equal (the compiled lane with arena residents on the GPU).
  window-sep / cards-sep
                 M4: `window-on` / `cards` with `migrate_repack_fused=False`. Since M4
                 `window-on` and `cards` run the fused migrate + repack (auto); these
                 are the separate passes. PRE-REGISTERED: hashes and stats (the
                 knob's receipt aside) equal their fused arm.

Device arms record their platform and refuse `cpu` unless `--allow-cpu` (the
laptop smoke, which needs four forced host devices for the card arms). Output
streams line by line and the card is rewritten after every arm.
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
    _c_drift, _maxrss_gb, _nb_of, _peak, _require_device, _say, _x64,
)
from v2_e1_device_step import (  # noqa: E402
    _codes_compare, _scales_compare, _state, _stats_diffs,
)
from v2_r2_device_engine import _coeffs, _hashes, _json_ready, _Phases, _preset_of  # noqa: E402

CARDS = 4
LANE = dict(coarse_backend="device", tile_backend="device", migrate_backend="device")
ENGINE_ARMS = {
    "window-off": dict(LANE, device_tile_window=False),
    "window-on": dict(LANE, device_tile_window=True),
    "cards": dict(LANE, device_tile_window=True, device_cards=CARDS),
    "slack0-eager": dict(coarse_backend="device", tile_backend="device",
                         device_tile_jit=False, migrate_backend="device", brick_slack=0.0),
    "slack0-cards": dict(LANE, device_tile_window=True, device_cards=CARDS, brick_slack=0.0),
    "window-sep": dict(LANE, device_tile_window=True, migrate_repack_fused=False),
    "cards-sep": dict(LANE, device_tile_window=True, device_cards=CARDS,
                      migrate_repack_fused=False),
}
SNAP_ARMS = ("slack0-eager", "slack0-cards")
#: keys that describe the card split, the lane or the compile history, not the step
SPLIT_KEYS = ("coarse_card_chunks", "coarse_cards", "coarse_card_ranges",
              "coarse_ghost_planes_nonzero", "device_cards", "tile_cards",
              "coarse_jit_traces", "coarse_jit_shapes", "coarse_device_jit",
              "migrate_repack_fused", "census_slabs")
#: (arm, reference) pairs gated on hashes + stats
ENGINE_GATES = (("window-on", "window-off"), ("cards", "window-on"),
                ("slack0-cards", "slack0-eager"), ("window-on", "window-sep"),
                ("cards", "cards-sep"))


def _devices(n):
    import jax

    devs = jax.devices()
    if len(devs) < n:
        raise SystemExit(f"FATAL: {n} cards asked, {len(devs)} jax devices present")
    return list(devs[:n])


# ------------------------------------------------------------------ the arms


def arm_engine(args):
    _x64()
    from inexor import engine
    from inexor.device import migrate, paint, repack, tile, window
    from inexor.plan import engine_config

    platform = _require_device(args.allow_cpu)
    n = args.n_part
    nb = _nb_of(n)
    preset = _preset_of(n)
    cfg = engine_config(preset, tile_workers=1, **ENGINE_ARMS[args.arm])
    arena = args.slack0_arena if cfg.brick_slack == 0.0 else 0.01
    t0 = time.perf_counter()
    st = _state(preset, nb, cfg.brick_slack, arena)
    build_s = time.perf_counter() - t0
    co = _coeffs(args.steps)
    ph = _Phases()
    snap, wrote = None, {}
    if args.arm in SNAP_ARMS:
        os.makedirs(args.snap_dir, exist_ok=True)
        snap = {k: os.path.join(args.snap_dir, f"{args.arm}{args.out_suffix}_{k}.npy")
                for k in ("w", "vel_scale")}

    def hook(name):
        ph(name)
        if snap and name == "tile_loop_end" and not wrote:
            t = time.perf_counter()
            np.save(snap["w"], st.w)
            np.save(snap["vel_scale"], st.vel_scale)
            wrote["s"] = time.perf_counter() - t
            ph.t = time.perf_counter()

    calls0 = (paint.CALLS, tile.CALLS, window.CALLS, migrate.CALLS, repack.CALLS)
    _say(f"[{args.arm}] {preset}: n_part={n} nb={nb} slack={cfg.brick_slack} built in "
         f"{build_s:.1f}s on {platform}; K={args.steps}; tile jit={cfg.device_tile_jit} "
         f"window={cfg.tile_window} cards={cfg.device_cards}")
    t1 = time.perf_counter()
    out = engine.run(st, cfg, co, phase=hook)
    run_s = time.perf_counter() - t1
    calls = [b - a for a, b in zip(calls0, (paint.CALLS, tile.CALLS, window.CALLS,
                                             migrate.CALLS, repack.CALLS))]
    rc = 0
    if cfg.tile_window and calls[2] == 0:
        _say(f"FATAL: [{args.arm}] the window never ran")
        rc = 3
    split = []
    for o in out:
        md = o.get("migrate_device") or {}
        rd = (o.get("repack") or {}).get("repack_device") or {}
        split.append(dict(fused=o.get("migrate_repack_fused"),
                          tile_cards=o.get("tile_cards"), coarse_card_chunks=o.get(
            "coarse_card_chunks"), migrate_cards=md.get("cards"), migrate_fallback=md.get(
            "fallback"), cross_card_bytes=md.get("cross_card_bytes"), repack_cards=rd.get(
            "cards"), cross_card_early_uploads=rd.get("cross_card_early_uploads")))
        if cfg.device_cards > 1 and (o.get("device_cards") != cfg.device_cards
                                     or (md.get("cards") != cfg.device_cards
                                         and not md.get("fallback"))):
            _say(f"FATAL: [{args.arm}] the step did not split across {cfg.device_cards} cards")
            rc = 3
    spills = int(sum(o["n_arena_overflow"] for o in out))
    if cfg.brick_slack == 0.0 and spills == 0:
        _say(f"FATAL: [{args.arm}] VACUOUS -- nothing spilled at brick slack 0")
        rc = 3
    stats = [{k: v for k, v in o.items() if k not in ("pool", "busy", "loop_wall")}
             for o in out]
    _say(f"[{args.arm}] run {run_s:.1f}s; spills {spills}; split {split[-1]}; phases: "
         + ", ".join(f"{k} {v:.1f}" for k, v in sorted(ph.total.items(), key=lambda kv: -kv[1])))
    rec = dict(arm=args.arm, kind="engine", platform=platform, preset=preset, n_part=n, nb=nb,
               steps=args.steps, config=ENGINE_ARMS[args.arm], build_s=build_s, run_s=run_s,
               phases_total=ph.total, phase_trace=ph.trace, hashes=_hashes(st),
               stats=_json_ready(stats), split=_json_ready(split), spills=spills,
               device_calls=calls, snapshot=snap, snapshot_write_s=wrote.get("s"),
               device_peak=_peak(), host_maxrss_gb=_maxrss_gb(), rc=rc)
    _say(f"[{args.arm}] device peak {(rec['device_peak'] or 0) / 2**30:.2f} GiB, host maxrss "
         f"{rec['host_maxrss_gb']:.1f} GB")
    return rec, rc


def arm_solve(args):
    import tracemalloc

    _x64()
    import jax
    import jax.numpy as jnp

    from inexor.device.coarse import CardShards, shard_coarse_meshes
    from inexor.device.paint import coarse_delta_cards, gather_card_delta
    from inexor.forces import COARSE_HALO, coarse_force_meshes, coarse_kernel_parts
    from inexor.ooc_fft import partition_units
    from inexor.plan import engine_config

    platform = _require_device(args.allow_cpu)
    n_part = args.n_part
    preset = _preset_of(n_part)
    cfg = engine_config(preset, coarse_backend="device")
    t0 = time.perf_counter()
    st = _state(preset, _nb_of(n_part), 0.10, 0.01)
    build_s = time.perf_counter() - t0
    delta = gather_card_delta(coarse_delta_cards(st, cfg))
    st = None
    n = cfg.n_coarse
    parts = coarse_kernel_parts(n, cfg.box_size, "long", r_s=cfg.r_s,
                                match=cfg.coarse_match, fdtype=cfg.np_coarse_dtype)
    kw = dict(r_s=cfg.r_s, match=cfg.coarse_match, fdtype=cfg.np_coarse_dtype,
              parts=parts)
    devs = _devices(CARDS)
    per_tile = cfg.n_tile // (cfg.n_fine // n)
    ranges = [(a * per_tile - COARSE_HALO, (b - a) * per_tile + 2 * COARSE_HALO, devs[k])
              for k, (a, b) in enumerate(partition_units(cfg.tiles_side, CARDS, 1))]

    def host():
        return coarse_force_meshes(jnp.asarray(delta), n, cfg.box_size, "long", **kw)

    def cards():
        return coarse_force_meshes(jnp.asarray(delta), n, cfg.box_size, "long",
                                   out=CardShards(ranges, n), **kw)

    host()
    cards()  # warm both: compiles are not the transform
    walls, peaks, results = {}, {}, {}
    for name, fn in (("host", host), ("cards", cards)):
        t = time.perf_counter()
        results[name] = fn()
        if name == "cards":
            jax.block_until_ready([m for s in results[name] for m in s["meshes"]])
        walls[name] = time.perf_counter() - t
        tracemalloc.start()
        fn()
        peaks[name] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    g = results["host"]
    diffs = 0
    for k, (x0, nx, _d) in enumerate(ranges):
        ref = shard_coarse_meshes(g, x0, nx)["meshes"]
        got = results["cards"][k]["meshes"]
        for axis in range(3):
            diffs += int(np.count_nonzero(np.asarray(got[axis]) != np.asarray(ref[axis])))
    rc = 0 if diffs == 0 else 3
    _say(f"[solve-cards] {preset} n_coarse={n}: BITWISE = {diffs == 0} ({diffs} elements "
         f"differ); warm wall host {walls['host']:.2f}s vs cards {walls['cards']:.2f}s; host "
         f"tracemalloc peak host {peaks['host'] / 2**20:.1f} MiB vs cards "
         f"{peaks['cards'] / 2**20:.1f} MiB")
    rec = dict(arm="solve-cards", kind="solve", platform=platform, preset=preset,
               n_coarse=n, cards=CARDS, ranges=[(x0, nx) for x0, nx, _ in ranges],
               build_s=build_s, bitwise=diffs == 0, n_diff=diffs, wall_s=walls,
               host_tracemalloc_peak=peaks, device_peak=_peak(),
               host_maxrss_gb=_maxrss_gb(), rc=rc)
    return rec, rc


def arm_r3(args):
    _x64()
    from inexor.device.migrate import drift_and_migrate_device
    from inexor.device.repack import repack_device
    from inexor.plan import engine_config

    platform = _require_device(args.allow_cpu)
    w = CARDS if args.arm == "r3-w4" else 1
    n_part = args.n_part
    preset = _preset_of(n_part)
    cfg = engine_config(preset)
    t0 = time.perf_counter()
    st = _state(preset, _nb_of(n_part), 0.10, 0.01)
    build_s = time.perf_counter() - t0
    devs = None if w == 1 else _devices(w)
    c = _c_drift(st, args.real_frac)
    passes = []
    for p in range(2):
        tm = {} if p == 1 else None
        t = time.perf_counter()
        m = drift_and_migrate_device(st, c, timings=tm, devices=devs)
        m_s = time.perf_counter() - t
        t = time.perf_counter()
        r = repack_device(st, brick_slack=cfg.brick_slack, devices=devs)
        r_s = time.perf_counter() - t
        md, rd = m.pop("migrate_device"), r.pop("repack_device")
        r.pop("scratch_bytes", None)
        passes.append(dict(migrate_s=m_s, repack_s=r_s, migrate_stats=m, repack_stats=r,
                           migrate_receipt=md, repack_receipt=rd, timings=tm))
        _say(f"[{args.arm}] pass {p}: migrate {m_s:.2f}s ({'timed' if tm is not None else 'untimed'}),"
             f" repack {r_s:.2f}s; spills {m['n_arena_overflow']}; migrate receipt cards "
             f"{md.get('cards', 1)} cross-card bytes {md.get('cross_card_bytes')} fallback "
             f"{md.get('fallback')}; repack early uploads {rd.get('cross_card_early_uploads')}")
    if passes[1]["timings"]:
        for k, v in sorted(passes[1]["timings"].items(), key=lambda kv: -kv[1])[:12]:
            _say(f"    {v:8.3f} s  {k}")
    rec = dict(arm=args.arm, kind="r3", platform=platform, preset=preset, cards=w,
               build_s=build_s, c_drift=c, passes=_json_ready(passes), hashes=_hashes(st),
               device_peak=_peak(), host_maxrss_gb=_maxrss_gb(), rc=0)
    return rec, 0


ARMS = {**{a: arm_engine for a in ENGINE_ARMS}, "solve-cards": arm_solve,
        "r3-w1": arm_r3, "r3-w4": arm_r3}
DEFAULT_ORDER = ("solve-cards,window-off,window-on,r3-w1,r3-w4,cards,slack0-eager,"
                 "slack0-cards")


# ------------------------------------------------------------ the orchestrator


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


def _strip_r3(passes):
    return [dict(migrate_stats=p["migrate_stats"], repack_stats=p["repack_stats"]) for p in passes]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--arm", choices=sorted(ARMS), help=argparse.SUPPRESS)
    ap.add_argument("--n-part", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--arms", default=DEFAULT_ORDER, help="comma-separated, run in this order")
    ap.add_argument("--engine-n", type=int, default=512, help="particles per side (cgh64)")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--real-frac", type=float, default=0.5,
                    help="r3 arms: drift of the fastest row in bricks")
    ap.add_argument("--slack0-arena", type=float, default=0.05)
    ap.add_argument("--snap-dir", default=os.path.join(REPO, "runs", "v2", "e2e4_snap"))
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny everything, CPU allowed: exercises the apparatus only")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args(argv)

    if args.worker:
        rec, rc = ARMS[args.arm](args)
        print("WORKER_JSON " + json.dumps(_json_ready(rec)), flush=True)
        return rc

    if args.smoke:
        args.engine_n = 32
    common = ["--steps", str(args.steps), "--real-frac", str(args.real_frac),
              "--slack0-arena", str(args.slack0_arena), "--snap-dir", args.snap_dir,
              "--out-suffix", args.out_suffix] + \
        (["--allow-cpu"] if args.allow_cpu or args.smoke else [])
    out = os.path.join(REPO, "runs", "v2", f"e2e4_device_cards{args.out_suffix}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    card = dict(argv=sys.argv, started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename,
                commit=subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
                arms=[], gates={})

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

    for arm, ref in ENGINE_GATES:
        if arm not in recs or ref not in recs:
            continue
        gate_key = arm if arm not in card["gates"] else f"{arm} vs {ref}"
        d, h = recs[arm], recs[ref]
        if d is None or h is None:
            card["gates"][gate_key] = "MISSING an arm"
            worst = max(worst, 1)
            continue
        hashes_equal = d["hashes"] == h["hashes"]
        diffs = [x for x in _stats_diffs(
            [{k: v for k, v in o.items() if k not in SPLIT_KEYS} for o in h["stats"]],
            [{k: v for k, v in o.items() if k not in SPLIT_KEYS} for o in d["stats"]])]
        gate = dict(against=ref, hashes_equal=hashes_equal, stats_diffs=diffs,
                    run_s=(d["run_s"], h["run_s"]), spills=(d["spills"], h["spills"]))
        ok = hashes_equal and not diffs
        if d.get("snapshot") and h.get("snapshot"):
            gate["tile_loop_end_codes"] = _codes_compare(d["snapshot"]["w"], h["snapshot"]["w"])
            gate["tile_loop_end_scales"] = _scales_compare(d["snapshot"]["vel_scale"],
                                                           h["snapshot"]["vel_scale"])
            ok = ok and gate["tile_loop_end_codes"].get("n_diff") == 0 and \
                gate["tile_loop_end_scales"].get("n_diff") == 0
        # The compiled-vs-eager pair IS the job's question (compiled lane with arena
        # residents on the GPU), and on CPU XLA the compiled tile is not bitwise the
        # eager one by construction (record sec. 17). In the smoke it is a reading,
        # not the apparatus: a mismatch there must not stop the job that asks it.
        reading_only = args.smoke and arm == "slack0-cards"
        gate["reading_only"] = reading_only
        card["gates"][gate_key] = gate
        _say(f"\n=== {arm} vs {ref}: hashes equal = {hashes_equal}; stats differing "
             f"{len(diffs)} {diffs[:5]}; tile_loop_end {gate.get('tile_loop_end_codes')}; "
             f"run {d['run_s']:.1f} vs {h['run_s']:.1f}s"
             f"{' (smoke: reported, not gated)' if reading_only else ''} ===")
        if not ok and not reading_only:
            worst = max(worst, 3)

    if recs.get("r3-w4") and recs.get("r3-w1"):
        d, h = recs["r3-w4"], recs["r3-w1"]
        hashes_equal = d["hashes"] == h["hashes"]
        stats_equal = _strip_r3(d["passes"]) == _strip_r3(h["passes"])
        card["gates"]["r3-w4"] = dict(against="r3-w1", hashes_equal=hashes_equal,
                                      stats_equal=stats_equal,
                                      migrate_s=[(pd["migrate_s"], ph["migrate_s"])
                                                 for pd, ph in zip(d["passes"], h["passes"])],
                                      repack_s=[(pd["repack_s"], ph["repack_s"])
                                                for pd, ph in zip(d["passes"], h["passes"])])
        _say(f"\n=== r3-w4 vs r3-w1: hashes equal = {hashes_equal}; stats equal = "
             f"{stats_equal}; migrate s (4 cards, 1 card) {card['gates']['r3-w4']['migrate_s']}; "
             f"repack s {card['gates']['r3-w4']['repack_s']} ===")
        if not (hashes_equal and stats_equal):
            worst = max(worst, 3)

    for pair in (("cards", "window-on"), ("window-on", "window-off"),
                 ("window-on", "window-sep"), ("cards", "cards-sep")):
        if recs.get(pair[0]) and recs.get(pair[1]):
            a, b = recs[pair[0]]["phases_total"], recs[pair[1]]["phases_total"]
            _say(f"\n=== phases, s over the run ({pair[0]} | {pair[1]}) ===")
            for k in sorted(set(a) | set(b), key=lambda k: -max(a.get(k, 0), b.get(k, 0))):
                _say(f"  {k:<16} {a.get(k, 0):8.1f} | {b.get(k, 0):8.1f}")

    card["rc"] = worst
    write()
    _say(f"\ncard: {out}\nrc={worst}")
    return worst


if __name__ == "__main__":
    sys.exit(main())
