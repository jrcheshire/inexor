"""What is the HARDWARE doing while a step runs? Utilization, not phases.

Every wall instrument so far asks where the engine's seconds go
(`v2_m6_phase_time.py`); none asks what the node is doing during them. The
question this answers, per phase: how many of the 144 cores are busy, is the
memory system saturated or idle, does anything touch disk. The distinction it
exists to draw: a phase that saturates bandwidth is irreducible without a
traffic change (more cores buy nothing), while a phase running on 1-2 idle-node
cores is a parallelism candidate. Both look identical in a phase-time table.

Three cooperating pieces, joined on wall-clock time:

- `sample`: a sidecar process reading CUMULATIVE counters at an interval --
  per-cpu busy/idle/iowait jiffies (/proc/stat), disk sectors (/proc/diskstats),
  Lustre llite byte counters when the mount exposes them, PSI stall totals
  (/proc/pressure) and per-NUMA-node MemFree. Cumulative on purpose: a dropped
  tick costs resolution, never correctness, because the readout differences
  adjacent ticks itself.
- `canary`: ONE pinned core per socket running a small triad loop and logging
  its own achieved GB/s per ~0.4 s window. When the engine saturates a socket's
  memory bandwidth the canary slows down; its slowdown against its own in-job
  idle baseline is a MEASURED contention signal, per socket, per phase. This is
  the piece /proc cannot provide (there is no bandwidth counter without PMU
  access).
- `run`: drives the engine exactly as the phase probe does (same `_build`, same
  warmup discipline, same neutrality-gate shape) but records wall-clock
  TIMESTAMPS at each phase boundary, then attributes every sampler tick and
  canary window to the phase interval containing its midpoint.

The whole instrument is priced, not assumed: the control arm runs with no
callback and the canaries SIGSTOPped, and the gate is the phase probe's
two-term bound (2 sigma or 2% of the wall, whichever is larger). `stream` and
`disk` measure the node's own ceilings in the same job, so "fraction of
bandwidth" divides two same-node numbers rather than a spec sheet.

Bandwidth convention: the triad kernel counts 3 arrays x 8 B per element per
add (reads b,c + write a), the STREAM convention, write-allocate traffic
excluded. Canary and stream use the SAME kernel, so their ratio is clean even
if the absolute calibration is conservative.

    pixi run python scripts/v2_m6_c11_hw_util.py run --config cdev8 --k 3
"""

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))


# ---------------------------------------------------------------- proc seams
# Readers are module-level functions taking root paths so tests drive the
# arithmetic from fixture trees (the test_m6_peak_trace convention: a reading
# is a seam).

def parse_cpulist(text):
    """'0-71,144' -> [0..71, 144]."""
    out = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def read_proc_stat(proc_root="/proc"):
    """Per-cpu (busy, idle, iowait) jiffies. busy = everything but idle+iowait."""
    cpus = {}
    with open(os.path.join(proc_root, "stat")) as fh:
        for line in fh:
            if not line.startswith("cpu") or line[3] in (" ", "\t"):
                continue
            f = line.split()
            cid = int(f[0][3:])
            vals = [int(x) for x in f[1:]]
            # user nice system idle iowait irq softirq steal [guest guestnice]
            idle, iowait = vals[3], vals[4] if len(vals) > 4 else 0
            busy = sum(vals[:3]) + sum(vals[5:8])
            cpus[cid] = (busy, idle, iowait)
    return cpus


_DISK_OK = ("sd", "nvme", "vd", "xvd", "hd")


def _is_whole_disk(name):
    """sda yes, sda1 no; nvme0n1 yes, nvme0n1p1 no; loop/ram/dm never."""
    if not name.startswith(_DISK_OK):
        return False
    if name.startswith("nvme"):
        return "p" not in name.split("n", 1)[1]
    return not name[-1].isdigit()


def read_diskstats(proc_root="/proc"):
    """(read_bytes, write_bytes) summed over whole physical devices."""
    rd = wr = 0
    with open(os.path.join(proc_root, "diskstats")) as fh:
        for line in fh:
            f = line.split()
            if len(f) < 14 or not _is_whole_disk(f[2]):
                continue
            rd += int(f[5]) * 512
            wr += int(f[9]) * 512
    return rd, wr


def read_lustre():
    """(read_bytes, write_bytes) summed over llite mounts, or None."""
    total_r = total_w = 0
    found = False
    for pat in ("/proc/fs/lustre/llite/*/stats", "/sys/kernel/debug/lustre/llite/*/stats"):
        for path in glob.glob(pat):
            try:
                with open(path) as fh:
                    for line in fh:
                        f = line.split()
                        if f and f[0] in ("read_bytes", "write_bytes") and len(f) >= 7:
                            found = True
                            if f[0] == "read_bytes":
                                total_r += int(f[6])
                            else:
                                total_w += int(f[6])
            except OSError:
                continue
    return (total_r, total_w) if found else None


def read_psi(proc_root="/proc"):
    """{'cpu'|'memory'|'io': some-total microseconds} or None if no PSI."""
    out = {}
    for kind in ("cpu", "memory", "io"):
        path = os.path.join(proc_root, "pressure", kind)
        try:
            with open(path) as fh:
                for line in fh:
                    if line.startswith("some"):
                        out[kind] = int(line.rsplit("total=", 1)[1])
        except OSError:
            return None
    return out or None


def read_numa_free_kb(sys_root="/sys"):
    """{node_id: MemFree kB} or None on a machine without NUMA sysfs."""
    out = {}
    for path in glob.glob(os.path.join(sys_root, "devices/system/node/node*/meminfo")):
        nid = int(os.path.basename(os.path.dirname(path))[4:])
        try:
            with open(path) as fh:
                for line in fh:
                    if "MemFree:" in line:
                        out[nid] = int(line.split()[-2])
        except OSError:
            continue
    return out or None


def node_cpulists(sys_root="/sys"):
    """{node_id: [cpu ids]} or None."""
    out = {}
    for path in glob.glob(os.path.join(sys_root, "devices/system/node/node*/cpulist")):
        nid = int(os.path.basename(os.path.dirname(path))[4:])
        try:
            with open(path) as fh:
                out[nid] = parse_cpulist(fh.read())
        except OSError:
            continue
    return out or None


def mems_allowed(proc_root="/proc"):
    """The NUMA membind receipt: Mems_allowed_list from /proc/self/status."""
    try:
        with open(os.path.join(proc_root, "self/status")) as fh:
            for line in fh:
                if line.startswith("Mems_allowed_list:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


# ------------------------------------------------------------- triad kernel

def _alloc_triad(mb):
    n = int(mb * 2**20 / 8 / 3)
    rng = np.random.default_rng(0)
    a = rng.random(n)
    b = rng.random(n)
    c = rng.random(n)
    return a, b, c


def _triad_pass(a, b, c, reps):
    """`reps` add passes; returns nominal bytes touched (STREAM convention)."""
    for _ in range(reps):
        np.add(b, c, out=a)
    return reps * 3 * a.nbytes


# ------------------------------------------------------------------ sample

def cmd_sample(args):
    if not sys.platform.startswith("linux"):
        print("sample: /proc counters are Linux-only; refusing", file=sys.stderr)
        return 3
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    with open(args.out, "w") as fh:
        header = dict(
            header=dict(
                clk_tck=os.sysconf("SC_CLK_TCK"),
                interval_s=args.interval,
                node_cpus={str(k): v for k, v in (node_cpulists() or {}).items()},
                pid=os.getpid(),
            )
        )
        fh.write(json.dumps(header) + "\n")
        fh.flush()
        while not stop["flag"]:
            lus = read_lustre()
            line = dict(
                t=time.time(),
                cpu={str(k): list(v) for k, v in read_proc_stat().items()},
                disk=list(read_diskstats()),
                lustre=list(lus) if lus else None,
                psi=read_psi(),
                numa_free_kb={str(k): v for k, v in (read_numa_free_kb() or {}).items()} or None,
            )
            fh.write(json.dumps(line) + "\n")
            fh.flush()
            time.sleep(args.interval)
    return 0


# ------------------------------------------------------------------ canary

def cmd_canary(args):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {args.cpu})
    a, b, c = _alloc_triad(args.mb)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    reps = 4
    with open(args.out, "w") as fh:
        fh.write(json.dumps(dict(header=dict(cpu=args.cpu, mb=args.mb,
                                             pid=os.getpid()))) + "\n")
        fh.flush()
        while not stop["flag"]:
            t0 = time.time()
            nbytes = _triad_pass(a, b, c, reps)
            t1 = time.time()
            fh.write(json.dumps(dict(t0=t0, t1=t1, bytes=nbytes)) + "\n")
            fh.flush()
            el = max(t1 - t0, 1e-4)
            reps = max(1, min(int(reps * args.window / el), 10000))
    return 0


def load_canary_windows(path, drop_factor=3.0):
    """[(t0, t1, gb_s)] with stalled windows (SIGSTOP spans) dropped."""
    rows = []
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            if "header" in d:
                continue
            rows.append((d["t0"], d["t1"], d["bytes"]))
    if not rows:
        return []
    durs = sorted(t1 - t0 for t0, t1, _ in rows)
    med = durs[len(durs) // 2]
    return [(t0, t1, by / max(t1 - t0, 1e-9) / 1e9)
            for t0, t1, by in rows if (t1 - t0) <= drop_factor * max(med, 1e-4)]


# ------------------------------------------------------------------ stream

def _stream_worker(cpu, mb, seconds, q):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})
    a, b, c = _alloc_triad(mb)  # first-touch AFTER pinning: local under default policy
    total = 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        total += _triad_pass(a, b, c, 4)
    q.put((cpu, total, time.time() - t0))


def cmd_stream(args):
    import multiprocessing as mp

    cpus = parse_cpulist(args.cpus) if args.cpus else sorted(
        os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity")
        else range(os.cpu_count()))
    picked = [cpus[i % len(cpus)] for i in range(args.workers)]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_stream_worker, args=(c, args.mb, args.seconds, q))
             for c in picked]
    for p in procs:
        p.start()
    rows = [q.get() for _ in procs]
    for p in procs:
        p.join()
    agg = sum(by / el for _, by, el in rows) / 1e9
    res = dict(workers=args.workers, cpus=picked, mb_per_worker=args.mb,
               seconds=args.seconds, gb_s_total=agg,
               gb_s_per_worker=sorted(by / el / 1e9 for _, by, el in rows),
               mems_allowed=mems_allowed(),
               convention="triad add, 24 B/elem nominal, write-allocate excluded")
    print(json.dumps(res, indent=1))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
    return 0


# -------------------------------------------------------------------- disk

def cmd_disk(args):
    path = os.path.join(args.dir, f"c11_disk_probe_{os.getpid()}.bin")
    chunk = np.random.default_rng(0).integers(0, 255, 64 * 2**20, dtype=np.uint8)
    n_chunks = int(args.gb * 2**30 / chunk.nbytes)
    t0 = time.time()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        for _ in range(n_chunks):
            os.write(fd, chunk.data)
        os.fsync(fd)
    finally:
        os.close(fd)
    w_el = time.time() - t0
    fd = os.open(path, os.O_RDONLY)
    try:
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        buf = bytearray(chunk.nbytes)
        t0 = time.time()
        while os.readv(fd, [buf]):
            pass
        r_el = time.time() - t0
    finally:
        os.close(fd)
        os.unlink(path)
    total_gb = n_chunks * chunk.nbytes / 2**30
    res = dict(dir=args.dir, gb=total_gb,
               write_gb_s=total_gb / max(w_el, 1e-9),
               read_gb_s=total_gb / max(r_el, 1e-9),
               caveat="page cache defeated only by fadvise DONTNEED; on Lustre "
                      "that is advisory, so the read rate is an upper bound")
    print(json.dumps(res, indent=1))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
    return 0


# ------------------------------------------------- attribution (pure math)

def set_node_map(node_cpus):
    """Build {cpu_id_str: node_id} from a {node: [cpus]} map."""
    return {str(c): n for n, cs in (node_cpus or {}).items() for c in cs}


def attribute_ticks(intervals, ticks, clk_tck, node_cpus=None, max_gap_s=10.0):
    """Aggregate cumulative-counter ticks into labeled interval buckets.

    intervals: [(t0, t1, label)], non-overlapping. ticks: sampler dicts sorted
    by t, cumulative counters. Differences adjacent ticks, attributes each
    delta to the label containing the pair's midpoint, and returns
    {label: aggregate}. Pairs spanning more than max_gap_s are dropped (a
    sampler stall attributes garbage to whatever phase it lands in).
    """
    ivs = sorted(intervals)
    node_of = set_node_map(node_cpus)
    out = {}

    def bucket(label):
        return out.setdefault(label, dict(
            dt_s=0.0, busy_core_s=0.0, iowait_core_s=0.0,
            busy_core_s_by_node={}, disk_read_b=0, disk_write_b=0,
            lustre_read_b=0, lustre_write_b=0,
            psi_stall_us=dict(cpu=0, memory=0, io=0), n_ticks=0))

    def find(mid):
        for t0, t1, label in ivs:
            if t0 <= mid < t1:
                return label
        return None

    for prev, cur in zip(ticks, ticks[1:]):
        dt = cur["t"] - prev["t"]
        if dt <= 0 or dt > max_gap_s:
            continue
        label = find(prev["t"] + dt / 2.0)
        if label is None:
            continue
        b = bucket(label)
        b["dt_s"] += dt
        b["n_ticks"] += 1
        for cid, (busy1, _idle1, iow1) in cur["cpu"].items():
            if cid not in prev["cpu"]:
                continue
            busy0, _idle0, iow0 = prev["cpu"][cid]
            b["busy_core_s"] += (busy1 - busy0) / clk_tck
            b["iowait_core_s"] += (iow1 - iow0) / clk_tck
            nid = node_of.get(cid)
            if nid is not None:
                per = b["busy_core_s_by_node"]
                per[nid] = per.get(nid, 0.0) + (busy1 - busy0) / clk_tck
        b["disk_read_b"] += cur["disk"][0] - prev["disk"][0]
        b["disk_write_b"] += cur["disk"][1] - prev["disk"][1]
        if cur.get("lustre") and prev.get("lustre"):
            b["lustre_read_b"] += cur["lustre"][0] - prev["lustre"][0]
            b["lustre_write_b"] += cur["lustre"][1] - prev["lustre"][1]
        if cur.get("psi") and prev.get("psi"):
            for k in b["psi_stall_us"]:
                if k in cur["psi"] and k in prev["psi"]:
                    b["psi_stall_us"][k] += cur["psi"][k] - prev["psi"][k]
    return out


def summarize_label(agg):
    """Reduce one label's aggregate to the card row."""
    dt = agg["dt_s"]
    if dt <= 0:
        return None
    row = dict(
        wall_sampled_s=dt,
        n_ticks=agg["n_ticks"],
        busy_cores=agg["busy_core_s"] / dt,
        iowait_cores=agg["iowait_core_s"] / dt,
        busy_cores_by_node={str(k): v / dt for k, v in agg["busy_core_s_by_node"].items()},
        disk_read_mb=agg["disk_read_b"] / 2**20,
        disk_write_mb=agg["disk_write_b"] / 2**20,
        lustre_read_mb=agg["lustre_read_b"] / 2**20,
        lustre_write_mb=agg["lustre_write_b"] / 2**20,
        psi_stall_frac={k: v / 1e6 / dt for k, v in agg["psi_stall_us"].items()},
    )
    return row


def canary_by_label(intervals, windows):
    """{label: [gb_s...]} for windows whose midpoint falls in the label."""
    ivs = sorted(intervals)
    out = {}
    for t0, t1, gb_s in windows:
        mid = (t0 + t1) / 2.0
        label = next((la for a, b, la in ivs if a <= mid < b), None)
        if label is not None:
            out.setdefault(label, []).append(gb_s)
    return out


def neutrality(traced, control):
    """The phase probe's two-term bound, verbatim semantics."""
    t_med = float(np.median(traced))
    c_med = float(np.median(control))
    c_sd = float(np.std(control, ddof=1)) if len(control) > 1 else float("nan")
    sigma_bound = 2.0 * c_sd if c_sd == c_sd else float("nan")
    material_bound = 0.02 * c_med
    bound = max(sigma_bound, material_bound) if sigma_bound == sigma_bound else material_bound
    return dict(
        traced_median_s=t_med, control_median_s=c_med, control_sd_s=c_sd,
        instrument_overhead_s=t_med - c_med,
        gate_sigma_bound_s=sigma_bound, gate_material_bound_s=material_bound,
        gate_bound_used_s=bound,
        gate_bound_that_bound=("sigma" if sigma_bound == sigma_bound
                               and sigma_bound >= material_bound else "material"),
        instrument_neutral=abs(t_med - c_med) <= bound,
    )


# --------------------------------------------------------------------- run

class TsTimer:
    """Phase boundary timestamps: (wall-clock t, name) per callback."""

    def __init__(self):
        self.events = []

    def __call__(self, name):
        self.events.append((time.time(), name))


def _phase_intervals(t_start, events):
    """[(t0, t1, phase)] -- the interval ENDING at a boundary carries its name
    (the PhaseTimer semantics)."""
    out = []
    prev = t_start
    for t, name in events:
        out.append((prev, t, name))
        prev = t
    return out


def cmd_run(args):
    if not sys.platform.startswith("linux"):
        print("run: the sampler reads /proc; Linux only", file=sys.stderr)
        return 3
    import v2_m6_phase_time as pt

    nodes = node_cpulists() or {0: sorted(os.sched_getaffinity(0))}
    canary_cpus = {nid: cpus[-1] for nid, cpus in nodes.items() if cpus}
    raw_dir = os.path.join(REPO, "runs", "v2", f"c11_raw{args.out_suffix}")
    os.makedirs(raw_dir, exist_ok=True)

    me = [sys.executable, os.path.abspath(__file__)]
    sampler_path = os.path.join(raw_dir, "sample.jsonl")
    procs = {}
    procs["sample"] = subprocess.Popen(
        me + ["sample", "--out", sampler_path, "--interval", str(args.interval)])
    canary_paths = {}
    if args.canary:
        for nid, cpu in canary_cpus.items():
            p = os.path.join(raw_dir, f"canary_node{nid}.jsonl")
            canary_paths[nid] = p
            procs[f"canary{nid}"] = subprocess.Popen(
                me + ["canary", "--cpu", str(cpu), "--out", p, "--mb", str(args.canary_mb)])

    def canaries(sig):
        for name, p in procs.items():
            if name.startswith("canary"):
                os.kill(p.pid, sig)

    labeled = []   # (t0, t1, label) for baseline/control windows
    phase_iv = []  # (t0, t1, phase) across instrumented runs
    traced, control = [], []
    pool_steps = []

    def one(timed):
        engine, ec, st, cosmo, a_grid, bfc, bft = pt._build(
            args.config, args.slack, args.arena_frac, args.tile, args.buf,
            args.tile_workers, True, None)
        co = bfc(bft(a_grid(pt.m3.A_INIT, pt.m3.A_FINAL, args.k, pt.m3.SPACING), cosmo))
        timer = TsTimer() if timed else None
        seen = []
        t0 = time.time()
        engine.run(st, ec, co, phase=timer, collect=seen.append)
        wall = time.time() - t0
        return wall, t0, timer, seen

    try:
        # warmup: full K, untimed, sidecars running (the phase probe's lesson --
        # a shortened K primes the wrong `cap` shapes and drifts differently)
        for _ in range(args.warmup):
            one(False)

        t0 = time.time()
        time.sleep(args.baseline_s)
        labeled.append((t0, time.time(), "baseline"))

        for _r in range(args.repeats):
            wall, t_start, timer, seen = one(True)
            traced.append(wall)
            phase_iv.extend(_phase_intervals(t_start, timer.events))
            pool_steps.extend(s["pool"] for s in seen if "pool" in s)

            canaries(signal.SIGSTOP)
            wall, ts, _, _ = one(False)
            control.append(wall)
            labeled.append((ts, ts + wall, "control"))
            canaries(signal.SIGCONT)
    finally:
        for p in procs.values():
            try:
                os.kill(p.pid, signal.SIGCONT)
                p.terminate()
            except OSError:
                pass
        for p in procs.values():
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()

    # ------------------------------------------------------------- readout
    ticks, clk_tck = [], 100
    with open(sampler_path) as fh:
        for line in fh:
            d = json.loads(line)
            if "header" in d:
                clk_tck = d["header"]["clk_tck"]
                continue
            ticks.append(d)
    intervals = phase_iv + labeled
    agg = attribute_ticks(intervals, ticks, clk_tck, node_cpus=nodes)
    rows = {la: summarize_label(a) for la, a in agg.items()}
    rows = {la: r for la, r in rows.items() if r is not None}

    canary = {}
    if args.canary:
        for nid, path in canary_paths.items():
            windows = load_canary_windows(path)
            per = canary_by_label(intervals, windows)
            base = per.get("baseline", [])
            base_med = float(np.median(base)) if base else float("nan")
            canary[str(nid)] = dict(
                cpu=canary_cpus[nid],
                baseline_gb_s=base_med,
                n_windows=sum(len(v) for v in per.values()),
                by_label={la: dict(gb_s_median=float(np.median(v)), n=len(v),
                                   contention=(1.0 - float(np.median(v)) / base_med)
                                   if base_med == base_med and base_med > 0 else None)
                          for la, v in sorted(per.items()) if la != "baseline"},
            )

    res = dict(
        config=args.config, k=args.k, repeats=args.repeats,
        tile_workers=args.tile_workers, slack=args.slack, arena_frac=args.arena_frac,
        interval_s=args.interval, canary=bool(args.canary), canary_mb=args.canary_mb,
        canary_cpus={str(k): v for k, v in canary_cpus.items()},
        n_cpus=len(read_proc_stat()), node_cpus={str(k): len(v) for k, v in nodes.items()},
        mems_allowed=mems_allowed(),
        traced_wall_s=traced, control_wall_s=control,
        phases={la: rows[la] for la in sorted(rows, key=lambda x: -rows[x]["wall_sampled_s"])},
        canary_by_node=canary,
        eject_kernel=pt._RESOLVED_EJECT, eject_jax_calls=pt._eject_calls(),
        instrument_broken=(len(ticks) < 3
                           or (bool(args.canary) and not any(
                               c["n_windows"] for c in canary.values()))),
        hostname=os.uname().nodename,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    )
    res.update(neutrality(traced, control))
    if pool_steps:
        res["pool_concurrency_median"] = float(
            np.median([p["concurrency"] for p in pool_steps]))
    try:
        res["commit"] = subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        res["commit"] = None

    print(f"{args.config} K={args.k} W={args.tile_workers}: traced "
          f"{res['traced_median_s']:.1f} s, control {res['control_median_s']:.1f} s, "
          f"neutral={res['instrument_neutral']} "
          f"(overhead {res['instrument_overhead_s']:+.2f} s vs "
          f"{res['gate_bound_used_s']:.2f} s bound), mems_allowed={res['mems_allowed']}")
    for la, r in res["phases"].items():
        cn = ""
        for nid, c in canary.items():
            e = c["by_label"].get(la)
            if e and e["contention"] is not None:
                cn += f"  canary{nid} -{100 * e['contention']:.0f}%"
        print(f"  {la:<12} {r['wall_sampled_s']:7.1f} s  busy {r['busy_cores']:6.1f} cores  "
              f"iowait {r['iowait_cores']:5.2f}  disk r/w "
              f"{r['disk_read_mb']:.0f}/{r['disk_write_mb']:.0f} MB "
              f"lustre {r['lustre_read_mb']:.0f}/{r['lustre_write_mb']:.0f} MB{cn}")
    if res["instrument_broken"]:
        print("  INSTRUMENT BROKEN: no ticks or zero canary windows; "
              "the table above measured nothing")

    path = os.path.join(REPO, "runs", "v2", f"m6_c11_hw{args.out_suffix}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"  card -> {path}")
    if res["instrument_broken"]:
        return 4
    return 0 if res["instrument_neutral"] else 2


# -------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample")
    s.add_argument("--out", required=True)
    s.add_argument("--interval", type=float, default=0.5)

    c = sub.add_parser("canary")
    c.add_argument("--cpu", type=int, required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--mb", type=float, default=192.0)
    c.add_argument("--window", type=float, default=0.4)

    st = sub.add_parser("stream")
    st.add_argument("--workers", type=int, required=True)
    st.add_argument("--cpus", default=None, help="cpulist; default = all allowed")
    st.add_argument("--seconds", type=float, default=15.0)
    st.add_argument("--mb", type=float, default=192.0)
    st.add_argument("--out", default=None)

    d = sub.add_parser("disk")
    d.add_argument("--dir", required=True)
    d.add_argument("--gb", type=float, default=4.0)
    d.add_argument("--out", default=None)

    r = sub.add_parser("run")
    r.add_argument("--config", default="cdev8")
    r.add_argument("--k", type=int, default=3)
    r.add_argument("--repeats", type=int, default=2)
    r.add_argument("--warmup", type=int, default=1)
    r.add_argument("--slack", type=float, default=0.10)
    r.add_argument("--arena-frac", type=float, default=0.02)
    r.add_argument("--tile", type=int, default=None)
    r.add_argument("--buf", type=int, default=32)
    r.add_argument("--tile-workers", type=int, default=1)
    r.add_argument("--interval", type=float, default=0.5)
    r.add_argument("--baseline-s", type=float, default=12.0)
    r.add_argument("--canary", type=int, default=1, choices=(0, 1))
    r.add_argument("--canary-mb", type=float, default=192.0)
    r.add_argument("--out-suffix", default="")

    args = ap.parse_args(argv)
    return dict(sample=cmd_sample, canary=cmd_canary, stream=cmd_stream,
                disk=cmd_disk, run=cmd_run)[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
