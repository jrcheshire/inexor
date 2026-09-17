"""D7: the device step on four GB200s at a preset's full shape, K steps from ICs on disk, instrumented to leave a readable record whatever happens.

    preflight  refuse early: devices, the allocator's receipt, host memory against
               the planner's load peak, the IC manifest, scratch space
    run        load the ICs, run K steps of the 40-step schedule on the device lane
               (window and fused pass on auto, checkpoints off), last step timed
    summarize  after the fact, no jax: the run card against the job's sampler CSVs,
               per phase (card memory max per GPU, host MemAvailable min, utilization)

WHAT SURVIVES A FAILURE, and why each piece exists:
- Every engine phase boundary prints one line when it is crossed (wall clock, step,
  seconds, host VmRSS / per-phase VmHWM / MemAvailable, each card's allocator bytes),
  and the card JSON is rewritten atomically at every boundary. A killed run leaves
  every boundary it reached.
- A heartbeat thread prints every `--beat` seconds: the last boundary and the time
  since it, host memory, Lustre bytes read/written, each card's allocator bytes. A
  long compile, a slow phase and a hang look different in it.
- `STEP_JSON` per step: the engine's stats with every receipt.
- Any exception writes the traceback, every card's full `memory_stats()` and the last
  heartbeat into the card before re-raising.
- `faulthandler` on SIGUSR1 dumps every thread's stack (the sbatch sends it before the
  wall), and on a fatal signal.
- The job's own samplers (nvidia-smi and /proc/meminfo, 5 s, epoch-stamped) run
  outside this process, so an OOM kill or a hang cannot silence them; `summarize`
  lines them up against the boundaries recorded here.

`D7_FAIL_AT=<phase>` raises at that boundary, to exercise the failure path.
"""

from __future__ import annotations

import argparse
import datetime
import faulthandler
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

GB = 1e9


def _stamp(t=None):
    return datetime.datetime.fromtimestamp(time.time() if t is None else t).strftime("%H:%M:%S")


def _git_commit():
    return subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def _write_json(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(tmp, path)


def _proc_kv(path, scale=1):
    out = {}
    try:
        with open(path) as fh:
            for line in fh:
                k, _, v = line.partition(":")
                parts = v.split()
                if parts and parts[0].isdigit():
                    out[k.strip()] = int(parts[0]) * scale
    except OSError:
        pass
    return out


def host_memory():
    """(VmRSS, VmHWM, MemAvailable) in bytes; VmHWM falls back to ru_maxrss off Linux."""
    status = _proc_kv("/proc/self/status", 1024)
    meminfo = _proc_kv("/proc/meminfo", 1024)
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    maxrss = r if sys.platform == "darwin" else r * 1024
    return (status.get("VmRSS"), status.get("VmHWM", maxrss), meminfo.get("MemAvailable"))


def reset_hwm():
    """Per-phase host peak: reset VmHWM (Linux clear_refs). False where unavailable."""
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
        return True
    except OSError:
        return False


def card_memory(full=False):
    """Each device's allocator stats: `bytes_in_use` and `peak_bytes_in_use` where the
    allocator reports them (None otherwise), or the whole dict with `full`."""
    import jax

    out = []
    for d in jax.devices():
        try:
            s = d.memory_stats() or {}
        except Exception as e:  # an instrument must never be the failure
            s = {"error": repr(e)}
        out.append(dict(s) if full else dict(in_use=s.get("bytes_in_use"),
                                             peak=s.get("peak_bytes_in_use")))
    return out


def _gb(x):
    return "   -  " if x is None else f"{x / GB:6.1f}"


class Monitor:
    """`engine.run(phase=)` hook plus heartbeat, both streaming, and the card."""

    def __init__(self, card_path, card, beat_s=60.0, fail_at=None):
        self.card_path = card_path
        self.card = card
        self.lock = threading.Lock()
        self.t_start = time.time()
        self.t_last = self.t_start
        self.last = "start"
        self.step = 0
        self.fail_at = fail_at
        self.hwm_resets = reset_hwm()
        self.run_peak = 0
        self.last_beat = None
        card.update(boundaries=[], beats=[], steps=[], hwm_per_phase=self.hwm_resets)
        self._io0 = _proc_kv("/proc/self/io")
        self._stop = threading.Event()
        self._beat_s = float(beat_s)
        self._thread = threading.Thread(target=self._beat_loop, daemon=True, name="d7-beat")

    def start(self):
        self.save()
        self._thread.start()

    def stop(self):
        self._stop.set()

    def save(self):
        with self.lock:
            _write_json(self.card_path, self.card)

    def __call__(self, name):
        now = time.time()
        rss, hwm, avail = host_memory()
        self.run_peak = max(self.run_peak, hwm or 0)
        cards = card_memory()
        if name == "coarse_paint":
            self.step += 1
        rec = dict(t=now, name=name, step=self.step, dt=now - self.t_last, rss=rss,
                   hwm=hwm, avail=avail, cards=cards)
        print(f"[phase] {_stamp(now)} step {self.step:2d} {name:<16} {rec['dt']:9.1f} s | host "
              f"rss {_gb(rss)} peak {_gb(hwm)} avail {_gb(avail)} GB | cards in_use "
              f"{[round((c['in_use'] or 0) / GB, 1) for c in cards]} peak "
              f"{[None if c['peak'] is None else round(c['peak'] / GB, 1) for c in cards]} GB",
              flush=True)
        with self.lock:
            self.card["boundaries"].append(rec)
            self.last, self.t_last = name, now
        self.save()
        if self.hwm_resets:
            reset_hwm()
        if self.fail_at and name == self.fail_at:
            raise RuntimeError(f"D7_FAIL_AT={name}: the failure path, exercised on purpose")

    def _beat_loop(self):
        while not self._stop.wait(self._beat_s):
            try:
                self.beat()
            except Exception as e:  # the heartbeat must not die quietly either
                print(f"[beat] {_stamp()} instrument error {e!r}", flush=True)

    def beat(self):
        now = time.time()
        rss, hwm, avail = host_memory()
        io = _proc_kv("/proc/self/io")
        read = (io.get("read_bytes", 0) - self._io0.get("read_bytes", 0)) if io else None
        wrote = (io.get("write_bytes", 0) - self._io0.get("write_bytes", 0)) if io else None
        cards = card_memory()
        with self.lock:
            last, since = self.last, now - self.t_last
        rec = dict(t=now, after=last, since=since, rss=rss, hwm=hwm, avail=avail,
                   read=read, wrote=wrote, cards=cards)
        print(f"[beat]  {_stamp(now)} +{(now - self.t_start) / 60:6.1f} min, {since:7.0f} s "
              f"since '{last}' | host rss {_gb(rss)} avail {_gb(avail)} GB | io read "
              f"{_gb(read)} wrote {_gb(wrote)} GB | cards in_use "
              f"{[round((c['in_use'] or 0) / GB, 1) for c in cards]} GB", flush=True)
        with self.lock:
            self.card["beats"].append(rec)
            self.last_beat = rec
        self.save()

    def fail(self, exc):
        with self.lock:
            self.card["failure"] = dict(
                t=time.time(), after=self.last, error=repr(exc),
                traceback=traceback.format_exc(), cards=card_memory(full=True),
                host=host_memory(), last_beat=self.last_beat)
        self.save()
        print(f"[FAIL] {_stamp()} after '{self.last}': {exc!r}; card {self.card_path}",
              flush=True)


def _signals(card_path):
    faulthandler.enable(all_threads=True)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    print(f"  faulthandler: fatal signals + SIGUSR1 (pid {os.getpid()}); card {card_path}",
          flush=True)


def _planner(preset, cards, slack, arena):
    cmd = [sys.executable, "-m", "inexor.plan", "--preset", preset, "--backend", "device",
           "--n-gpus", str(cards), "--host-gb", "1026", "--device-gb", "199",
           "--arena-frac", str(arena), "--slack", str(slack)]
    out = subprocess.run(cmd, capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"))).stdout

    def num(pattern):
        m = re.search(pattern, out)
        return None if m is None else float(m.group(1)) * GB

    return dict(text=out,
                host_gb=num(r"host, a lower bound on the run's peak:\s+([\d.]+) GB"),
                load_gb=num(r"the LOAD stage peaks at:\s+([\d.]+) GB"),
                card_in_step_gb=num(r"per GPU, resident \+ worst phase:\s+([\d.]+) GB"),
                card_after_loop_gb=num(r"per GPU, resident \+ after-the-loop:\s+([\d.]+) GB"))


def _base_card(kind, args):
    return dict(card=f"inexor-d7-{kind}-1", preset=args.preset, workdir=args.workdir,
                commit=_git_commit(), host=os.uname().nodename,
                job=os.environ.get("SLURM_JOB_ID"), started=time.time(),
                env={k: v for k, v in os.environ.items()
                     if k.startswith(("XLA_", "JAX_", "INEXOR_"))})


# ---------------------------------------------------------------------- preflight


def cmd_preflight(args):
    import jax
    import numpy as np

    from inexor import icgen
    from inexor.plan import PRESETS

    jax.config.update("jax_enable_x64", True)
    card = _base_card("preflight", args)
    refusals = []
    devs = jax.devices()
    card["devices"] = [str(d) for d in devs]
    if len(devs) < args.cards:
        refusals.append(f"{len(devs)} jax devices, {args.cards} asked")
    if not args.allow_cpu and any(d.platform == "cpu" for d in devs):
        refusals.append("a jax device is the CPU backend")

    # the allocator's receipt: what it IS, and which stats it actually reports
    for d in devs[:args.cards]:
        jax.device_put(np.zeros(1 << 20, np.uint8), d)
    stats = card_memory(full=True)
    card["allocator"] = dict(
        name=os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "(default BFC)"),
        fraction=os.environ.get("XLA_CLIENT_MEM_FRACTION", "(default)"),
        preallocate=os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "(default)"),
        stats_keys=sorted(stats[0]) if stats else [],
        reports_in_use=bool(stats and stats[0].get("bytes_in_use") is not None),
        reports_peak=bool(stats and stats[0].get("peak_bytes_in_use") is not None))
    print(f"  allocator: {card['allocator']}", flush=True)

    g = PRESETS[args.preset]
    man = icgen.read_manifest(args.workdir)
    card["manifest"] = {k: man.get(k) for k in ("n_part", "n_particles", "bricks_per_side",
                                                 "box_size", "provenance")}
    card["manifest"]["files"] = len(man["files"])
    if int(man["n_part"]) != int(g["n_part"]):
        refusals.append(f"manifest n_part {man['n_part']} != preset {g['n_part']}")
    if int(man["n_particles"]) != int(g["n_part"]) ** 3:
        refusals.append(f"manifest holds {man['n_particles']} particles")
    if len(man["files"]) != int(man["bricks_per_side"]):
        refusals.append(f"{len(man['files'])} slab files for {man['bricks_per_side']} slabs")

    plan = _planner(args.preset, args.cards, args.slack, args.arena_frac)
    card["planner"] = plan
    _rss, _hwm, avail = host_memory()
    card["host_available"] = avail
    if avail is not None and plan["load_gb"] and avail < args.host_margin * plan["load_gb"]:
        refusals.append(f"MemAvailable {avail / GB:.0f} GB < {args.host_margin} x the planner's "
                        f"load peak {plan['load_gb'] / GB:.0f} GB")
    du = shutil.disk_usage(args.scratch)
    card["scratch"] = dict(path=args.scratch, free=du.free, total=du.total)
    print(f"  planner: host {_gb(plan['host_gb'])} load {_gb(plan['load_gb'])} per card in-step "
          f"{_gb(plan['card_in_step_gb'])} after-loop {_gb(plan['card_after_loop_gb'])} GB; "
          f"MemAvailable {_gb(avail)} GB; scratch free {du.free / GB:.0f} GB", flush=True)
    card["refusals"] = refusals
    _write_json(args.card, card)
    for r in refusals:
        print(f"  REFUSED: {r}", flush=True)
    print(f"  card {args.card}", flush=True)
    return 2 if refusals else 0


# ---------------------------------------------------------------------- run


def cmd_run(args):
    import jax

    from inexor import engine, icgen
    from inexor.plan import engine_config
    from v2_m6_realization import _coeffs, _cosmo

    jax.config.update("jax_enable_x64", True)
    ics = os.path.realpath(args.workdir)
    for what, path in (("the card", os.path.dirname(os.path.abspath(args.card))),
                       ("a checkpoint", args.checkpoint_dir)):
        if path and os.path.realpath(path).startswith(ics):
            raise SystemExit(f"FATAL: {what} would be written under the IC directory")
    if args.checkpoint_every and not args.checkpoint_dir:
        raise SystemExit("FATAL: --checkpoint-every needs --checkpoint-dir")
    _signals(args.card)
    card = _base_card("run", args)
    mon = Monitor(args.card, card, beat_s=args.beat, fail_at=os.environ.get("D7_FAIL_AT"))
    card["plan"] = dict(stop_at=args.stop_at, timed_last=args.timed_last, cards=args.cards,
                        slack=args.slack, alloc_margin=args.alloc_margin,
                        arena_frac=args.arena_frac, checkpoint_dir=args.checkpoint_dir,
                        checkpoint_every=args.checkpoint_every)
    try:
        devs = jax.devices()
        if len(devs) < args.cards:
            raise RuntimeError(f"{len(devs)} jax devices, {args.cards} asked")
        ec = engine_config(args.preset, coarse_backend="device", tile_backend="device",
                           migrate_backend="device", device_cards=args.cards, tile_workers=1,
                           brick_slack=args.slack,
                           checkpoint_dir=args.checkpoint_dir if args.checkpoint_every else None,
                           checkpoint_every=args.checkpoint_every)
        ec.validate()
        card["config"] = dict(tile_window=ec.tile_window, fused_pass=ec.fused_pass,
                              device_cards=ec.device_cards, checkpoint_dir=ec.checkpoint_dir,
                              coarse_dtype=ec.coarse_dtype, fine_dtype=ec.fine_dtype)
        print(f"== D7 run {args.preset}: {args.cards} cards, K={args.stop_at}, "
              f"window={ec.tile_window} fused={ec.fused_pass}, from {args.workdir}", flush=True)
        mon.start()

        st = icgen.load_slot_state(args.workdir, brick_slack=args.slack,
                                   alloc_margin=args.alloc_margin, arena_frac=args.arena_frac)
        card["state"] = dict(n_particles=st.n_particles, n_bricks=st.n_bricks,
                             rows=int(st.off.shape[0]), n_arena=int(st.n_arena))
        mon("load")
        co, a_steps = _coeffs(_cosmo())
        timed = (args.stop_at - 1,) if args.timed_last else ()

        def collect(stats):
            s = {k: v for k, v in stats.items() if k not in ("pool", "busy", "loop_wall")}
            print("STEP_JSON " + json.dumps(s, default=str), flush=True)
            with mon.lock:
                card["steps"].append(s)

        out = engine.run(st, ec, co, phase=mon, stop_at=args.stop_at, collect=collect,
                         timed_steps=timed)
        card["finished"] = time.time()
        # a checkpoint is written after the last boundary and has none of its own
        card["after_last_boundary_s"] = card["finished"] - mon.t_last
        # the receipt lands on the step's stats after `collect` has seen them
        card["checkpoints"] = [o.get("checkpoint") for o in out]
        card["run_host_peak"] = mon.run_peak
        print(f"  {card['after_last_boundary_s']:.1f} s after the last boundary; checkpoints "
              f"{card['checkpoints']}", flush=True)
        mon.save()
        _print_steps(card)
        return 0
    except BaseException as e:
        mon.fail(e)
        raise
    finally:
        mon.stop()


def _print_steps(card):
    by = {}
    for b in card["boundaries"]:
        by.setdefault(b["step"], []).append(b)
    names = []
    for bs in by.values():
        for b in bs:
            if b["name"] not in names:
                names.append(b["name"])
    print("\n== seconds per phase, by step (step 0 = load, kernel build, lead drift)")
    print("  " + f"{'phase':<16}" + "".join(f"{f'step {k}':>11}" for k in sorted(by)))
    for n in names:
        row = [sum(b["dt"] for b in by[k] if b["name"] == n) for k in sorted(by)]
        print("  " + f"{n:<16}" + "".join(f"{v:11.1f}" for v in row))
    print("  " + f"{'TOTAL':<16}" + "".join(f"{sum(b['dt'] for b in by[k]):11.1f}"
                                          for k in sorted(by)))
    for i, st in enumerate(card["steps"]):
        t = st.get("timings")
        if not t:
            continue
        print(f"\n== synced breakdown, step {i + 1} (a breakdown, not the step's cost)")
        for section, d in t.items():
            flat = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        flat[kk] = flat.get(kk, 0.0) + vv
                else:
                    flat[k] = flat.get(k, 0.0) + v
            print(f"  [{section}] {sum(flat.values()):.1f} s summed over cards")
            for k, v in sorted(flat.items(), key=lambda kv: -kv[1])[:12]:
                print(f"    {v:9.2f} s  {k}")


# ---------------------------------------------------------------------- summarize


def _read_samples(path, n_fields):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path) as fh:
        for line in fh:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < n_fields:
                continue
            try:
                rows.append([float(p) for p in parts[:n_fields]])
            except ValueError:
                continue
    return rows


def cmd_summarize(args):
    with open(args.card) as fh:
        card = json.load(fh)
    gpu = _read_samples(args.gpu_csv, 4)      # epoch, index, memory.used MiB, util %
    mem = _read_samples(args.mem_csv, 2)      # epoch, MemAvailable kB
    t0 = card["started"]
    rows, prev = [], t0
    for b in card["boundaries"]:
        lo, hi = prev, b["t"]
        g = [r for r in gpu if lo <= r[0] <= hi]
        per = {}
        for _t, idx, mib, util in g:
            e = per.setdefault(int(idx), dict(mib=0.0, util=[]))
            e["mib"] = max(e["mib"], mib)
            e["util"].append(util)
        m = [r[1] * 1024 for r in mem if lo <= r[0] <= hi]
        rows.append(dict(step=b["step"], name=b["name"], dt=b["dt"], host_phase_peak=b["hwm"],
                         host_avail_min=min(m) if m else None,
                         card_max_gib={k: v["mib"] / 1024 for k, v in sorted(per.items())},
                         card_util_mean={k: sum(v["util"]) / len(v["util"])
                                         for k, v in sorted(per.items())},
                         samples=len(g)))
        prev = hi
    tail = None
    if "failure" in card or "finished" not in card:
        last = card["boundaries"][-1]["t"] if card["boundaries"] else t0
        g = [r for r in gpu if r[0] > last]
        m = [r[1] * 1024 for r in mem if r[0] > last]
        tail = dict(after=card["boundaries"][-1]["name"] if card["boundaries"] else "start",
                    seconds=(max(r[0] for r in g) - last) if g else None,
                    card_max_gib={int(i): max(r[2] for r in g if int(r[1]) == int(i)) / 1024
                                  for i in {r[1] for r in g}},
                    host_avail_min=min(m) if m else None)
    out = dict(card=args.card, phases=rows, unfinished_tail=tail,
               failure=(card.get("failure") or {}).get("error"))
    _write_json(args.out, out)
    done = "FINISHED" if "finished" in card else "DID NOT FINISH"
    print(f"== {card.get('preset')} {card.get('job')}: {done}"
          f"{'; failure ' + out['failure'] if out['failure'] else ''}")
    print(f"  {'step':>4} {'phase':<16} {'s':>9} {'host peak':>10} {'avail min':>10}  card max GiB")
    for r in rows:
        print(f"  {r['step']:>4} {r['name']:<16} {r['dt']:9.1f} {_gb(r['host_phase_peak']):>10} "
              f"{_gb(r['host_avail_min']):>10}  "
              f"{[round(v, 1) for v in r['card_max_gib'].values()]}")
    if tail:
        print(f"  UNFINISHED after '{tail['after']}': {tail['seconds']} s more sampled; card max "
              f"GiB {tail['card_max_gib']}; host avail min {_gb(tail['host_avail_min'])} GB")
    print(f"  summary {args.out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "run"):
        p = sub.add_parser(name)
        p.add_argument("--preset", required=True)
        p.add_argument("--workdir", required=True, help="the IC directory (read only)")
        p.add_argument("--card", required=True)
        p.add_argument("--cards", type=int, default=4)
        p.add_argument("--slack", type=float, default=0.10)
        p.add_argument("--alloc-margin", type=float, default=0.10)
        p.add_argument("--arena-frac", type=float, default=0.01)
    pf = sub.choices["preflight"]
    pf.add_argument("--scratch", default=os.environ.get("SCRATCH", "/tmp"))
    pf.add_argument("--host-margin", type=float, default=1.02,
                    help="refuse unless MemAvailable >= this x the planner's load peak")
    pf.add_argument("--allow-cpu", action="store_true")
    pr = sub.choices["run"]
    pr.add_argument("--stop-at", type=int, required=True, help="steps to run from the ICs")
    pr.add_argument("--timed-last", action="store_true",
                    help="synced per-phase breakdown of the device passes on the last step")
    pr.add_argument("--beat", type=float, default=60.0, help="heartbeat seconds")
    pr.add_argument("--checkpoint-dir", default=None)
    pr.add_argument("--checkpoint-every", type=int, default=0,
                    help="0 = no checkpoints; else every this many steps (--stop-at a multiple)")
    ps = sub.add_parser("summarize")
    ps.add_argument("--card", required=True)
    ps.add_argument("--gpu-csv", default=None)
    ps.add_argument("--mem-csv", default=None)
    ps.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return dict(preflight=cmd_preflight, run=cmd_run, summarize=cmd_summarize)[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
