"""D3b R1b + R2 on a GB200: the engine with migrate_backend='device' against the host engine, the migrate's budget estimate against its measured peak, and a 4096^3-shape slab's repack.

Every arm is a fresh subprocess, so each device high-water mark belongs to one
arm and nothing earlier in the job can set it.

  engine-device   `engine.run` with `migrate_backend="device"` (tile_workers=1,
                  the ratified dtypes, `--steps` steps + the lead drift, a repack
                  after every step) on a cgh64-family state. Every state field is
                  hashed (sha256 of its bytes) and the per-step stats are kept
                  with the backend receipts stripped; per-boundary wall from the
                  engine's `phase` hook; device peak. The device migrate's
                  `peak_estimate_bytes` per step is on the record.
  engine-host     the same schedule on the same state with the host engine
                  (serial migrate + host repack); the same hashes and stats. The
                  orchestrator compares the two arms: every hash equal and the
                  stripped stats equal is the R2 gate ON A GB200.
  migrate-budget  the R1 owed reading: `drift_and_migrate_device` alone, twice,
                  on the cgh64-family state; the receipt's `peak_estimate_bytes`
                  (largest per-slab estimate) beside `memory_stats` peak, in ONE
                  process, so the estimate is read against what it estimates.
  repack-slab     a production-shape slab (`--shape-nb` bricks per side, 4096
                  rows per brick, 512 buckets per brick: 268,435,456 rows at
                  nb=256) built with every particle in x-slab 0, with residents
                  planted so the fold-in runs. The device repack (synced phase
                  split, device peak, receipt) and the host repack on an
                  identical copy, compared bitwise. NB the tile-force geometry
                  of the state does not matter here; only the slab does.

The engine arms run the tile loop on THIS backend either way (jax's default
device), so their device peak is the run's, not the migrate's -- that is what
`migrate-budget` is for.

Device arms record their platform and refuse `cpu` unless `--allow-cpu` (the
laptop smoke). Output streams line by line and the card is rewritten after every
arm, so a killed job keeps what finished.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
    FIELDS, _build_state, _c_drift, _diff_fields, _maxrss_gb, _nb_of, _peak,
    _require_device, _say, _x64,
)

STRIP = ("migrate_backend", "migrate_device")


def _preset_of(n_part):
    from inexor.plan import PRESETS

    for name, g in PRESETS.items():
        if g["n_part"] == n_part:
            return name
    raise SystemExit(f"no preset with n_part={n_part}")


def _engine_state(preset, nb, seed=13):
    """A perturbed lattice in the PRESET's box (the R1 builder's n/2 convention
    is the cgh64/cdev box but not smoke's), gaussian velocities, no ids."""
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
    return state.SlotState.build(x, v, t9, nb, brick_slack=0.10, arena_frac=0.01,
                                 with_ids=False)


def _coeffs(k):
    from inexor.config import Cosmology
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

    return bullfrog_float_coeffs(bullfrog_table(a_grid(0.1, 1.0, k, "log"), Cosmology()))


def _hashes(st):
    out = {}
    for name in FIELDS:
        a = getattr(st, name)
        if a is None:
            out[name] = None
            continue
        out[name] = hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()
    out["arena_base"] = int(st.arena_base)
    out["n_live"] = int(st.n_live)
    return out


def _strip(stats):
    out = []
    for o in stats:
        d = {k: v for k, v in o.items() if k not in STRIP}
        r = d.get("repack")
        if r is not None:
            d["repack"] = {k: v for k, v in r.items() if k not in ("repack_device", "scratch_bytes")}
        # walls and rss are not identity
        for k in ("pool", "busy", "loop_wall"):
            d.pop(k, None)
        out.append(d)
    return out


def _json_ready(x):
    if isinstance(x, dict):
        return {str(k): _json_ready(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_ready(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


class _Phases:
    """`engine.run(phase=)` hook: wall since the previous boundary, per name."""

    def __init__(self):
        self.t = time.perf_counter()
        self.total = {}
        self.trace = []

    def __call__(self, name):
        now = time.perf_counter()
        dt = now - self.t
        self.t = now
        self.total[name] = self.total.get(name, 0.0) + dt
        self.trace.append((name, dt))


# ------------------------------------------------------------------ the arms


def arm_engine(args):
    _x64()
    from inexor import engine
    from inexor.device import migrate, repack
    from inexor.plan import engine_config

    platform = _require_device(args.allow_cpu)
    device = args.arm == "engine-device"
    n, nb = args.n_part, args.nb or _nb_of(args.n_part)
    preset = _preset_of(n)
    cfg = engine_config(preset, tile_workers=1,
                        migrate_backend="device" if device else "host")
    t0 = time.perf_counter()
    st = _engine_state(preset, nb)
    build_s = time.perf_counter() - t0
    assert nb == st.bricks_per_side == cfg.n_fine // cfg.n_brick, (nb, cfg.n_brick)
    co = _coeffs(args.steps)
    ph = _Phases()
    m0, r0 = migrate.CALLS, repack.CALLS
    _say(f"[{args.arm}] {preset}: n_part={n} nb={nb} built in {build_s:.1f}s on {platform}; "
         f"K={args.steps}, migrate_backend={cfg.migrate_backend}")
    t1 = time.perf_counter()
    out = engine.run(st, cfg, co, phase=ph)
    run_s = time.perf_counter() - t1
    calls = (migrate.CALLS - m0, repack.CALLS - r0)
    rc = 0
    if device and calls != (args.steps + 1, args.steps):
        _say(f"FATAL: device passes ran {calls}, expected {(args.steps + 1, args.steps)}")
        rc = 3
    if not device and calls != (0, 0):
        _say(f"FATAL: the host arm touched the device passes {calls}")
        rc = 3
    if any(o["migrate_backend"] != cfg.migrate_backend for o in out):
        rc = 3
    est = [((o.get("migrate_device") or {}).get("peak_estimate_bytes")) for o in out]
    steps = []
    for k, o in enumerate(out):
        steps.append(dict(cap=int(o["cap"]), overflow=int(o["n_arena_overflow"]),
                          arena_used=int(o["arena_used"]),
                          reach=int(o["brick_reach"]), migrate_estimate=est[k],
                          repack=_json_ready(o.get("repack"))))
    _say(f"[{args.arm}] run {run_s:.1f}s; phases: " + ", ".join(
        f"{k} {v:.1f}" for k, v in sorted(ph.total.items(), key=lambda kv: -kv[1])))
    rec = dict(arm=args.arm, platform=platform, preset=preset, n_part=n, nb=nb,
               steps=args.steps, build_s=build_s, run_s=run_s, phases_total=ph.total,
               phase_trace=ph.trace, hashes=_hashes(st), stats=_json_ready(_strip(out)),
               per_step=steps, device_calls=calls, device_peak=_peak(),
               host_maxrss_gb=_maxrss_gb(), rc=rc)
    _say(f"[{args.arm}] device peak {(rec['device_peak'] or 0) / 2**30:.2f} GiB, host maxrss "
         f"{rec['host_maxrss_gb']:.1f} GB")
    return rec, rc


def arm_migrate_budget(args):
    _x64()
    from inexor import state
    from inexor.device import migrate

    platform = _require_device(args.allow_cpu)
    n, nb = args.n_part, args.nb or _nb_of(args.n_part)
    t0 = time.perf_counter()
    st = _build_state(n, nb, seed=13, brick_slack=0.10, arena_frac=0.01, with_ids=False)
    ref = copy.deepcopy(st)
    c = _c_drift(st, args.real_frac)
    rec = dict(arm=args.arm, platform=platform, n_part=n, nb=nb, rows_per_slab=n**3 // nb,
               build_s=time.perf_counter() - t0, steps=[])
    rc = 0
    for k in range(2):
        r_ref = state.drift_and_migrate(ref, c)
        t1 = time.perf_counter()
        r = migrate.drift_and_migrate_device(st, c)
        t2 = time.perf_counter()
        receipt = r.pop("migrate_device")
        diff = _diff_fields(ref, st)
        peak = _peak()
        ratio = None if not peak else receipt["peak_estimate_bytes"] / peak
        rec["steps"].append(dict(device_s=t2 - t1, estimate_bytes=receipt["peak_estimate_bytes"],
                                 measured_peak_bytes=peak, estimate_over_measured=ratio,
                                 field_diffs=diff, stats_equal=r == r_ref))
        _say(f"[{args.arm}] step {k}: device {t2 - t1:.2f}s; estimate "
             f"{receipt['peak_estimate_bytes'] / 2**30:.3f} GiB vs measured peak "
             f"{(peak or 0) / 2**30:.3f} GiB (estimate/measured "
             f"{ratio if ratio is None else round(ratio, 3)}); BITWISE numpy = "
             f"{not diff and r == r_ref}")
        if diff or r != r_ref:
            rc = 3
    rec.update(host_maxrss_gb=_maxrss_gb())
    return rec, rc


def _slab_state(shape_nb, seed=17, planted=4096):
    """A production-shape state with every particle in x-slab 0 and residents
    planted in low buckets (the engine's own spill never puts one there)."""
    from inexor import state
    from inexor.codec import T9Layout

    nb = int(shape_nb)
    n_part = nb * 16                 # 4096 rows per brick at bucket_cells 2
    L = n_part / 2.0
    n_rows = n_part**3 // nb         # one slab's worth of particles
    rng = np.random.default_rng(seed)
    x = np.empty((n_rows, 3), dtype=np.float64)
    x[:, 0] = rng.uniform(0.0, L / nb, size=n_rows)
    x[:, 1] = rng.uniform(0.0, L, size=n_rows)
    x[:, 2] = rng.uniform(0.0, L, size=n_rows)
    v = rng.normal(scale=0.5, size=x.shape)
    t9 = T9Layout(box_size=L, n_part=n_part, bucket_cells=2)
    st = state.SlotState.build(x, v, t9, nb, brick_slack=0.05, arena_frac=0.0005,
                               with_ids=False)
    x = v = None
    p3 = int(st.buckets_per_brick)
    run = st.occupancy[: nb * nb * p3].reshape(nb * nb, p3).astype(np.int64)
    cands = np.flatnonzero(run[:, 1:].sum(axis=1) > 0)[:planted]
    for b in cands:
        st._to_arena(np.array([int(b) * p3], dtype=np.int64),
                     np.array([[7, 8, 9]], dtype=np.uint8), np.array([[-3, 2, 1]], dtype=np.int16))
        st.n_particles += 1
    return st, int(len(cands))


def arm_repack_slab(args):
    _x64()
    from inexor.device import repack

    platform = _require_device(args.allow_cpu)
    t0 = time.perf_counter()
    st, planted = _slab_state(args.shape_nb)
    build_s = time.perf_counter() - t0
    nb = int(st.bricks_per_side)
    rows = int(st.n_particles)
    _say(f"[{args.arm}] nb={nb}, {rows:,} rows in slab 0 ({rows // (nb * nb):,} per brick), "
         f"{planted} residents planted, built in {build_s:.1f}s on {platform}")
    ref = copy.deepcopy(st)
    t1 = time.perf_counter()
    r_ref = ref.repack(brick_slack=0.10)
    t2 = time.perf_counter()
    _say(f"[{args.arm}] host repack {t2 - t1:.2f}s (fast {r_ref['bricks_fast']}, merged "
         f"{r_ref['bricks_merged']})")
    ref_h = _hashes(ref)
    ref = None
    # warm (compiles), then the timed pass on a fresh identical copy would need a
    # second build; instead: one untimed pass IS the reading (compiles included,
    # reported), then a synced timed pass on the already-repacked state (a repack
    # of a repacked state moves every row again, same work, no residents)
    p0 = len(repack._repack_programs())
    t3 = time.perf_counter()
    r = repack.repack_device(st, brick_slack=0.10)
    t4 = time.perf_counter()
    receipt = r.pop("repack_device")
    same = _hashes(st) == ref_h and all(r[k] == r_ref[k] for k in
                                       ("slots_used", "bricks_fast", "bricks_merged"))
    peak1 = _peak()
    _say(f"[{args.arm}] device repack {t4 - t3:.2f}s incl. {len(repack._repack_programs()) - p0} "
         f"compiles; BITWISE host = {same}; receipt {receipt}; device peak "
         f"{(peak1 or 0) / 2**30:.2f} GiB")
    phases = {}
    t5 = time.perf_counter()
    r2 = repack.repack_device(st, brick_slack=0.10, timings=phases)
    t6 = time.perf_counter()
    _say(f"[{args.arm}] timed second repack {t6 - t5:.2f}s (synced; phases sum "
         f"{sum(phases.values()):.2f}s)")
    for k, v in sorted(phases.items(), key=lambda kv: -kv[1]):
        _say(f"    {v:8.3f} s  {k}")
    rec = dict(arm=args.arm, platform=platform, shape_nb=nb, rows=rows, planted=planted,
               build_s=build_s, host_repack_s=t2 - t1, host_stats=r_ref,
               device_repack_s=t4 - t3, device_stats=r, receipt=receipt, bitwise=bool(same),
               device_peak_after_first=peak1, second_repack_s=t6 - t5, phases=phases,
               second_receipt=r2.pop("repack_device"), device_peak=_peak(),
               host_maxrss_gb=_maxrss_gb())
    return rec, 0 if same else 3


ARMS = {"engine-device": arm_engine, "engine-host": arm_engine,
        "migrate-budget": arm_migrate_budget, "repack-slab": arm_repack_slab}


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--arm", choices=sorted(ARMS), help=argparse.SUPPRESS)
    ap.add_argument("--n-part", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--nb", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--arms", default="engine-device,engine-host,migrate-budget,repack-slab",
                    help="comma-separated, run in this order")
    ap.add_argument("--engine-n", type=int, default=512, help="particles per side (cgh64)")
    ap.add_argument("--real-frac", type=float, default=0.5,
                    help="migrate-budget: drift of the fastest row in bricks")
    ap.add_argument("--shape-nb", type=int, default=256, help="bricks per side (c-hero 256)")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny everything, CPU allowed: exercises the apparatus only")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args(argv)

    if args.worker:
        rec, rc = ARMS[args.arm](args)
        print("WORKER_JSON " + json.dumps(_json_ready(rec)), flush=True)
        return rc

    nb = []  # the engine arms take the preset's brick grid (`_nb_of`)
    if args.smoke:
        args.engine_n, args.shape_nb = 32, 4
    common = ["--steps", str(args.steps), "--real-frac", str(args.real_frac)] + \
        (["--allow-cpu"] if args.allow_cpu or args.smoke else [])
    out = os.path.join(REPO, "runs", "v2", f"r2_device_engine{args.out_suffix}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    card = dict(argv=sys.argv, started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename,
                commit=subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
                arms=[])

    def write():
        with open(out, "w") as f:
            json.dump(card, f, indent=1)

    arms = [a for a in args.arms.split(",") if a]
    worst = 0
    recs = {}
    for a in arms:
        if a in ("engine-device", "engine-host"):
            argv_a = ["--arm", a, "--n-part", str(args.engine_n), *nb, *common]
        elif a == "migrate-budget":
            argv_a = ["--arm", a, "--n-part", str(args.engine_n), *nb, *common]
        else:
            argv_a = ["--arm", a, "--shape-nb", str(args.shape_nb), *common]
        res, rc = _run_worker(argv_a, a)
        card["arms"].append(res)
        recs[a] = res.get("record")
        worst = max(worst, rc)
        write()

    # THE R2 GATE: the two engine arms agree on every hash and every stripped stat
    if "engine-device" in recs and "engine-host" in recs:
        d, h = recs["engine-device"], recs["engine-host"]
        if d is None or h is None:
            card["engine_gate"] = "MISSING an arm"
            worst = max(worst, 1)
        else:
            hashes_equal = d["hashes"] == h["hashes"]
            stats_equal = d["stats"] == h["stats"]
            card["engine_gate"] = dict(hashes_equal=hashes_equal, stats_equal=stats_equal,
                                       device_run_s=d["run_s"], host_run_s=h["run_s"],
                                       device_phases=d["phases_total"],
                                       host_phases=h["phases_total"])
            _say(f"\n=== R2 GATE: state hashes equal = {hashes_equal}; stripped stats equal = "
                 f"{stats_equal}; run {d['run_s']:.1f}s device vs {h['run_s']:.1f}s host; "
                 f"migrate {d['phases_total'].get('migrate', 0):.1f} vs "
                 f"{h['phases_total'].get('migrate', 0):.1f}s; repack "
                 f"{d['phases_total'].get('repack', 0):.1f} vs "
                 f"{h['phases_total'].get('repack', 0):.1f}s ===")
            if not (hashes_equal and stats_equal):
                worst = max(worst, 3)
    card["rc"] = worst
    write()
    _say(f"\ncard: {out}\nrc={worst}")
    return worst


if __name__ == "__main__":
    sys.exit(main())
