"""D7: the device step on four GB200s at a preset's full shape, K steps from ICs on disk, instrumented to leave a readable record whatever happens.

    preflight  refuse early: devices, the allocator's receipt, host memory against
               the planner's load peak, the IC manifest, scratch space
    run        load the ICs, run K steps of the 40-step schedule on the device lane
               (window and fused pass on auto), the last or every step timed; then,
               optionally, a partial checkpoint write timed per part and a
               malloc_trim probe
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
- Each boundary also records where the host memory is and what the kernel did to
  get it: the process's anon / file / shmem / locked / pinned RSS, page faults
  (getrusage), `/proc/vmstat` counters (direct-reclaim stalls and scans, compaction
  stalls, NUMA misses), per NUMA node file pages, dirty and writeback, and with
  `--numa-maps` the process's pages per node. The instruments' own seconds are
  recorded per boundary and kept out of the phase's.
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


PROC_MEM_KEYS = ("VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem", "VmLck", "VmPin", "VmSwap")


def proc_memory():
    """This process's resident memory by kind, bytes: `PROC_MEM_KEYS` from
    /proc/self/status. Locked (VmLck) and pinned (VmPin) pages are the ones reclaim
    cannot take. {} off Linux."""
    status = _proc_kv(os.path.join(os.environ.get("D7_PROC_SELF", "/proc/self"), "status"),
                      1024)
    return {k: status[k] for k in PROC_MEM_KEYS if k in status}


def fault_counts():
    """(minor, major) page faults and (voluntary, involuntary) context switches of this
    process so far, all threads."""
    r = resource.getrusage(resource.RUSAGE_SELF)
    return dict(minflt=r.ru_minflt, majflt=r.ru_majflt, nvcsw=r.ru_nvcsw, nivcsw=r.ru_nivcsw)


# summed over every counter whose name starts with the key (allocstall_normal, ...)
VMSTAT_PREFIXES = ("allocstall", "pgscan_direct", "pgsteal_direct", "pgscan_kswapd",
                   "pgsteal_kswapd", "compact_stall", "compact_fail", "pgmajfault",
                   "thp_fault_alloc", "thp_fault_fallback", "numa_miss", "numa_foreign",
                   "pgmigrate_success", "workingset_refault_file", "pgpgout")


def vmstat():
    """System-wide /proc/vmstat counters summed by `VMSTAT_PREFIXES` (the node is the
    job's alone). Monotone, so a phase's value is a difference. {} off Linux."""
    out = {}
    try:
        with open(os.environ.get("D7_VMSTAT", "/proc/vmstat")) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) != 2 or not parts[1].isdigit():
                    continue
                for p in VMSTAT_PREFIXES:
                    # pgscan_direct_throttle counts throttling events, not pages
                    if parts[0].startswith(p) and not parts[0].endswith("_throttle"):
                        out[p] = out.get(p, 0) + int(parts[1])
    except OSError:
        pass
    return out


def _delta(now, before):
    return {k: v - before.get(k, 0) for k, v in (now or {}).items()}


# per-node meminfo fields recorded beside MemTotal/MemFree
NODE_KEYS = ("FilePages", "Active(file)", "Inactive(file)", "AnonPages", "Shmem", "Mlocked",
             "Unevictable", "Dirty", "Writeback", "AnonHugePages")


def numa_memory():
    """Per NUMA node (MemTotal, MemFree) in bytes, split into nodes with CPUs and nodes
    without (a GB200's HBM appears as CPU-less nodes, and `MemAvailable` sums both, which
    hid the CPU-side limit in gb 1002020), plus `detail[node]` = the `NODE_KEYS` fields.
    MemFree excludes page cache; FilePages is the cache. None off Linux."""
    root = os.environ.get("D7_NUMA_ROOT", "/sys/devices/system/node")
    if not os.path.isdir(root):
        return None
    cpu, gpu, detail = {}, {}, {}
    for d in sorted(os.listdir(root)):
        if not re.fullmatch(r"node\d+", d):
            continue
        vals = {}
        try:
            with open(os.path.join(root, d, "meminfo")) as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) >= 4 and parts[3].isdigit():
                        vals[parts[2][:-1]] = int(parts[3]) * 1024
            with open(os.path.join(root, d, "cpulist")) as fh:
                has_cpus = bool(fh.read().strip())
        except OSError:
            continue
        node = int(d[4:])
        (cpu if has_cpus else gpu)[node] = (vals.get("MemTotal"), vals.get("MemFree"))
        detail[node] = {k: vals[k] for k in NODE_KEYS if k in vals}
    return dict(cpu=cpu, gpu=gpu, detail=detail,
                cpu_total=sum(t for t, _ in cpu.values() if t),
                cpu_free=sum(f for _, f in cpu.values() if f),
                gpu_used={k: (t - f) if t is not None and f is not None else None
                          for k, (t, f) in gpu.items()})


def numa_maps():
    """This process's resident pages per NUMA node, bytes, split anon / file / other,
    from /proc/self/numa_maps (a page-table walk: slow at ~850 GB, so boundaries only,
    and the caller records its seconds). None where unreadable."""
    out = {}
    try:
        with open(os.path.join(os.environ.get("D7_PROC_SELF", "/proc/self"), "numa_maps")) as fh:
            for line in fh:
                toks = line.split()
                page = 4096
                for t in toks:
                    if t.startswith("kernelpagesize_kB="):
                        page = int(t.split("=")[1]) * 1024
                kind = ("anon" if any(t.startswith("anon=") for t in toks)
                        else "file" if any(t.startswith("file=") for t in toks) else "other")
                for t in toks:
                    m = re.fullmatch(r"N(\d+)=(\d+)", t)
                    if m:
                        e = out.setdefault(int(m.group(1)), dict(anon=0, file=0, other=0))
                        e[kind] += int(m.group(2)) * page
    except OSError:
        return None
    return out


def memory_policy():
    """This process's NUMA allocation policy per mapping, counted: `{"bind:0,1": n, ...}`
    from /proc/self/numa_maps. `Mems_allowed_list` is the CPUSET and does NOT show a
    `numactl --membind`, so the policy has to be read here."""
    out = {}
    try:
        with open(os.path.join(os.environ.get("D7_PROC_SELF", "/proc/self"), "numa_maps")) as fh:
            for line in fh:
                toks = line.split()
                if len(toks) > 1:
                    out[toks[1]] = out.get(toks[1], 0) + 1
    except OSError:
        return None
    return out


def membind_refusals(nodes, policy=None):
    """Refusals if this process is not bound to `nodes` (a set of NUMA node ids).

    gb 1003511 died here: with the CPU sockets full of page cache from the IC load, the
    kernel placed 197 GB of the process's memory on a card's HBM node rather than reclaim
    the cache, and a 2 GiB device allocation then failed on a card whose own allocator
    held 24 GB. The binding is what keeps host memory on the host, and a knob must prove
    it applied."""
    pol = memory_policy() if policy is None else policy
    if not pol:
        return ["no /proc/self/numa_maps: the memory policy cannot be read"]
    total = sum(pol.values())
    bound = 0
    bad = []
    for name, n in pol.items():
        if not name.startswith("bind:"):
            continue
        got = set()
        for part in name[5:].split(","):
            a, _, b = part.partition("-")
            got.update(range(int(a), int(b or a) + 1))
        if got <= set(nodes):
            bound += n
        else:
            bad.append(f"{n} mappings bound to {sorted(got)}, outside {sorted(nodes)}")
    if bad:
        return bad
    if bound < 0.9 * total:
        return [f"only {bound} of {total} mappings are bound to {sorted(nodes)}: run under "
                f"`numactl --membind={','.join(str(n) for n in sorted(nodes))}`"]
    return []


def pages_off_the_cpu_nodes(nmaps, nm):
    """Bytes of this process sitting on CPU-less (HBM) nodes, and which."""
    if not nmaps or not nm:
        return 0, {}
    off = {n: sum(e.values()) for n, e in nmaps.items() if n in nm["gpu"]}
    return sum(off.values()), {n: v for n, v in off.items() if v}


def _mem_text(pm, dflt, dvm, nm):
    """One line: the process's RSS by kind, this interval's faults and reclaim, and each
    CPU node's page cache."""
    def g(k):
        return round(pm.get(k, 0) / GB, 1)

    cache = ("" if not nm else " | cpu-node file pages " + str(
        {n: round(nm["detail"].get(n, {}).get("FilePages", 0) / GB, 1) for n in nm["cpu"]}))
    return (f"anon {g('RssAnon')} file {g('RssFile')} shmem {g('RssShmem')} lck {g('VmLck')} "
            f"pin {g('VmPin')} GB | faults min {dflt.get('minflt', 0)} maj "
            f"{dflt.get('majflt', 0)} | allocstall {dvm.get('allocstall', 0)} pgscan_direct "
            f"{dvm.get('pgscan_direct', 0)} compact_stall {dvm.get('compact_stall', 0)} "
            f"numa_miss {dvm.get('numa_miss', 0)}{cache}")


def _numa_text(nm):
    if not nm:
        return "numa -"
    return (f"cpu free {_gb(nm['cpu_free'])}/{_gb(nm['cpu_total']).strip()} GB, gpu-node used "
            f"{[None if v is None else round(v / GB, 1) for v in nm['gpu_used'].values()]} GB")


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

    def __init__(self, card_path, card, beat_s=60.0, fail_at=None, with_numa_maps=False):
        self.card_path = card_path
        self.with_numa_maps = bool(with_numa_maps)
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
        self._flt_last = fault_counts()
        self._vm_last = vmstat()
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

    def snapshot(self):
        """Memory, faults and reclaim counters now; faults and vmstat as the change since
        the previous snapshot."""
        flt, vm = fault_counts(), vmstat()
        dflt, dvm = _delta(flt, self._flt_last), _delta(vm, self._vm_last)
        self._flt_last, self._vm_last = flt, vm
        return dict(mem=proc_memory(), faults=dflt, vmstat=dvm)

    def __call__(self, name):
        now = time.time()
        rss, hwm, avail = host_memory()
        self.run_peak = max(self.run_peak, hwm or 0)
        cards = card_memory()
        nm = numa_memory()
        snap = self.snapshot()
        nmaps, nmaps_s = None, None
        if self.with_numa_maps:
            t0 = time.time()
            nmaps = numa_maps()
            nmaps_s = time.time() - t0
        if name == "coarse_paint":
            self.step += 1
        rec = dict(t=now, name=name, step=self.step, dt=now - self.t_last, rss=rss,
                   hwm=hwm, avail=avail, cards=cards, numa=nm, numa_maps=nmaps,
                   numa_maps_s=nmaps_s, **snap)
        print(f"[phase] {_stamp(now)} step {self.step:2d} {name:<16} {rec['dt']:9.1f} s | host "
              f"rss {_gb(rss)} peak {_gb(hwm)} | {_numa_text(nm)} | cards in_use "
              f"{[round((c['in_use'] or 0) / GB, 1) for c in cards]} peak "
              f"{[None if c['peak'] is None else round(c['peak'] / GB, 1) for c in cards]} GB",
              flush=True)
        print(f"[mem]   {_mem_text(snap['mem'], snap['faults'], snap['vmstat'], nm)}"
              + ("" if nmaps is None else
                 f" | process pages by node {_nodes_text(nmaps)} ({nmaps_s:.1f} s)"), flush=True)
        with self.lock:
            self.card["boundaries"].append(rec)
            self.last = name
        self.save()
        # the instruments' own seconds belong to no phase
        rec["instr_s"] = time.time() - now
        self.t_last = now + rec["instr_s"]
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
        # rchar/wchar: bytes through read()/write(), which is what a Lustre load is;
        # read_bytes counts block-device I/O only and read 0 through a 31 min load
        read = (io.get("rchar", 0) - self._io0.get("rchar", 0)) if io else None
        wrote = (io.get("wchar", 0) - self._io0.get("wchar", 0)) if io else None
        cards = card_memory()
        nm = numa_memory()
        with self.lock:
            last, since = self.last, now - self.t_last
        # absolute counters: the boundaries own the per-phase differences
        rec = dict(t=now, after=last, since=since, rss=rss, hwm=hwm, avail=avail,
                   read=read, wrote=wrote, cards=cards, numa=nm, mem=proc_memory(),
                   faults=fault_counts(), vmstat=vmstat())
        print(f"[beat]  {_stamp(now)} +{(now - self.t_start) / 60:6.1f} min, {since:7.0f} s "
              f"since '{last}' | host rss {_gb(rss)} | {_numa_text(nm)} | io read "
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
                host=host_memory(), numa=numa_memory(), mem=proc_memory(),
                vmstat=vmstat(), last_beat=self.last_beat)
        self.save()
        print(f"[FAIL] {_stamp()} after '{self.last}': {exc!r}; card {self.card_path}",
              flush=True)


def _node_list(text):
    """"0,1" or "0-1" -> {0, 1}."""
    out = set()
    for part in str(text).split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def _nodes_text(nmaps):
    return {n: {k: round(v / GB, 1) for k, v in e.items() if v} for n, e in sorted(nmaps.items())}


def _probe_record(mon, name):
    """A boundary-shaped record outside the engine's phases, for the post-run probes."""
    rss, hwm, avail = host_memory()
    return dict(t=time.time(), name=name, rss=rss, hwm=hwm, numa=numa_memory(),
                **mon.snapshot())


def checkpoint_probe(st, out_dir, n_slabs, mon):
    """Write the first `n_slabs` T9 slabs of the evolved state into `out_dir` (fresh,
    never a manifest) and time each part; the per-slab rate prices a full checkpoint."""
    from inexor import icgen

    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise RuntimeError(f"checkpoint probe dir {out_dir} is not empty; refusing to write "
                           "into it (the writer removes a manifest it finds)")
    io0 = _proc_kv("/proc/self/io")
    before = _probe_record(mon, "ckpt_probe_start")
    t, t0 = {}, time.time()
    icgen.write_t9_slabs(st, out_dir, timings=t, max_slabs=n_slabs)
    wall = time.time() - t0
    io1 = _proc_kv("/proc/self/io")
    after = _probe_record(mon, "ckpt_probe_end")
    wrote = io1.get("wchar", 0) - io0.get("wchar", 0) if io1 else None
    n = int(t.get("slabs", 0)) or 1
    nb = int(st.bricks_per_side)
    out = dict(slabs=t.get("slabs", 0), of=nb, wall_s=wall, parts_s=t, wrote=wrote,
               before=before, after=after, projected_full_s=wall / n * nb)
    print(f"== checkpoint probe: {out['slabs']} of {nb} slabs in {wall:.1f} s "
          f"({wall / n:.2f} s/slab -> {out['projected_full_s']:.0f} s for all), "
          + ", ".join(f"{k} {v:.1f} s" for k, v in t.items() if k != "slabs")
          + ("" if wrote is None else f"; wrote {wrote / GB:.1f} GB "
             f"({wrote / max(t.get('write', 0.0), 1e-9) / 1e6:.0f} MB/s over the write part)"),
          flush=True)
    print(f"[mem]   {_mem_text(after['mem'], after['faults'], after['vmstat'], after['numa'])}",
          flush=True)
    return out


def trim_probe(mon):
    """gc, then glibc `malloc_trim(0)`: how much of the host RSS was freed memory the C
    allocator still held. None off glibc."""
    import ctypes
    import gc

    gc.collect()
    before = _probe_record(mon, "trim_before")
    try:
        libc = ctypes.CDLL("libc.so.6")
    except OSError:
        print("== trim probe: no glibc here, skipped", flush=True)
        return None
    t0 = time.time()
    released = int(libc.malloc_trim(0))
    dt = time.time() - t0
    after = _probe_record(mon, "trim_after")
    out = dict(released_flag=released, seconds=dt, before=before, after=after)
    print(f"== trim probe: malloc_trim(0) returned {released} in {dt:.1f} s; RSS "
          f"{_gb(before['rss'])} -> {_gb(after['rss'])} GB, anon "
          f"{_gb(before['mem'].get('RssAnon'))} -> {_gb(after['mem'].get('RssAnon'))} GB",
          flush=True)
    return out


def _signals(card_path):
    faulthandler.enable(all_threads=True)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    print(f"  faulthandler: fatal signals + SIGUSR1 (pid {os.getpid()}); card {card_path}",
          flush=True)


def under(path, root):
    """Is `path` inside directory `root`? By PATH COMPONENT, never by string prefix:
    prefix matching called `.../smoke-ckpt-probe` a child of `.../smoke` and refused a
    write that was fine (gb 1003378, a gate leg)."""
    if not path:
        return False
    a, b = os.path.realpath(path), os.path.realpath(root)
    return a == b or a.startswith(b + os.sep)


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
    nm = numa_memory()
    card["host_available"] = avail
    card["numa"] = nm
    if nm and nm["cpu_total"]:
        # against the CPU-side nodes, not MemAvailable (which includes the cards' HBM)
        need = max(v for v in (plan["host_gb"], plan["load_gb"]) if v)
        if need > args.host_margin * nm["cpu_total"]:
            refusals.append(f"the planner's host peak {need / GB:.0f} GB > {args.host_margin} x "
                            f"the CPU nodes' {nm['cpu_total'] / GB:.0f} GB")
        print(f"  numa: {_numa_text(nm)}; planner host peak {_gb(need)} GB = "
              f"{need / nm['cpu_total']:.2f}x the CPU nodes", flush=True)
    card["memory_policy"] = memory_policy()
    if args.membind_nodes:
        want = _node_list(args.membind_nodes)
        refusals += [f"memory policy: {r}" for r in membind_refusals(want)]
        print(f"  memory policy: {card['memory_policy']} against --membind-nodes "
              f"{sorted(want)}", flush=True)
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
        if under(path, ics):
            raise SystemExit(f"FATAL: {what} would be written under the IC directory")
    if args.checkpoint_every and not args.checkpoint_dir:
        raise SystemExit("FATAL: --checkpoint-every needs --checkpoint-dir")
    if args.membind_nodes:
        bad = membind_refusals(_node_list(args.membind_nodes))
        if bad:
            raise SystemExit("FATAL: " + "; ".join(bad))
    _signals(args.card)
    card = _base_card("run", args)
    if args.ckpt_probe_slabs and not args.ckpt_probe_dir:
        raise SystemExit("FATAL: --ckpt-probe-slabs needs --ckpt-probe-dir")
    if under(args.ckpt_probe_dir, ics):
        raise SystemExit("FATAL: the checkpoint probe would be written under the IC directory")
    if args.ckpt_probe_dir and os.path.exists(args.ckpt_probe_dir) and os.listdir(
            args.ckpt_probe_dir):
        # here, not after the run: the writer removes a manifest it finds
        raise SystemExit(f"FATAL: checkpoint probe dir {args.ckpt_probe_dir} is not empty")
    mon = Monitor(args.card, card, beat_s=args.beat, fail_at=os.environ.get("D7_FAIL_AT"),
                  with_numa_maps=args.numa_maps)
    card["plan"] = dict(stop_at=args.stop_at, timed_last=args.timed_last,
                        timed_all=args.timed_all, numa_maps=args.numa_maps,
                        ckpt_probe_slabs=args.ckpt_probe_slabs, trim_probe=args.trim_probe,
                        cards=args.cards,
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
        if args.membind_nodes:
            # the state is on the host now: prove it is ON THE HOST'S nodes. Without a
            # binding the kernel put 197 GB of it on a card's HBM (gb 1003511), which
            # only shows up later as a device OOM on a card holding almost nothing.
            nmaps, nm = numa_maps(), numa_memory()
            off, where = pages_off_the_cpu_nodes(nmaps, nm)
            card["load_pages_off_cpu_nodes"] = dict(bytes=off, by_node=where)
            print(f"  after the load: {_gb(off)} GB of this process on CPU-less nodes "
                  f"{ {k: round(v / GB, 1) for k, v in where.items()} }", flush=True)
            if off > args.off_node_gb * GB:
                raise RuntimeError(
                    f"{off / GB:.1f} GB of the loaded state sits on CPU-less (HBM) nodes "
                    f"{sorted(where)}: those bytes are on the cards, which will OOM in the "
                    "step. The membind did not hold.")
        co, a_steps = _coeffs(_cosmo())
        timed = (tuple(range(args.stop_at)) if args.timed_all
                 else (args.stop_at - 1,) if args.timed_last else ())

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
        if args.ckpt_probe_slabs:
            card["ckpt_probe"] = checkpoint_probe(st, args.ckpt_probe_dir,
                                                  args.ckpt_probe_slabs, mon)
            mon.save()
        if args.trim_probe:
            card["trim_probe"] = trim_probe(mon)
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
    print("  " + f"{'(instruments)':<16}" + "".join(
        f"{sum(b.get('instr_s') or 0.0 for b in by[k]):11.1f}" for k in sorted(by)))
    if any("mem" in b for b in card["boundaries"]):
        print("\n== host memory at each boundary; faults and reclaim over the phase")
        print(f"  {'step':>4} {'phase':<16} {'anon':>6} {'file':>6} {'shmem':>6} {'lck':>6} "
              f"{'pin':>6} {'minflt':>10} {'majflt':>7} {'allocstall':>10} {'scan_direct':>12} "
              f"{'compact':>8} {'numa_miss':>10}  cpu-node file pages GB")
        for b in card["boundaries"]:
            m, f, v, nm = b.get("mem") or {}, b.get("faults") or {}, b.get("vmstat") or {}, \
                b.get("numa")
            cache = ({n: round(nm["detail"].get(n, {}).get("FilePages", 0) / GB, 1)
                      for n in nm["cpu"]} if nm and "detail" in nm else {})
            print(f"  {b['step']:>4} {b['name']:<16} {_gb(m.get('RssAnon'))} "
                  f"{_gb(m.get('RssFile'))} {_gb(m.get('RssShmem'))} {_gb(m.get('VmLck'))} "
                  f"{_gb(m.get('VmPin'))} {f.get('minflt', 0):>10} {f.get('majflt', 0):>7} "
                  f"{v.get('allocstall', 0):>10} {v.get('pgscan_direct', 0):>12} "
                  f"{v.get('compact_stall', 0):>8} {v.get('numa_miss', 0):>10}  {cache}")
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
    if not os.path.exists(args.card):
        # a leg that died before its first boundary leaves none; say so in one line
        # rather than a traceback that reads as a second, unrelated failure
        print(f"== no card at {args.card}: that leg wrote none, nothing to summarize")
        return 0
    with open(args.card) as fh:
        card = json.load(fh)
    gpu = _read_samples(args.gpu_csv, 4)      # epoch, index, memory.used MiB, util %
    # epoch, node, has_cpus (1/0), MemTotal kB, MemFree kB
    nodes = _read_samples(args.mem_csv, 5)
    cpu_free = {}
    for t, _node, has_cpus, _total, free in nodes:
        if has_cpus:
            cpu_free[t] = cpu_free.get(t, 0.0) + free * 1024
    mem = sorted(cpu_free.items())
    gpu_node_used = [(t, int(node), (total - free) * 1024)
                     for t, node, has_cpus, total, free in nodes if not has_cpus]
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
        m = [r[1] for r in mem if lo <= r[0] <= hi]
        gn = {}
        for t, node, used in gpu_node_used:
            if lo <= t <= hi:
                gn[node] = max(gn.get(node, 0.0), used)
        rows.append(dict(step=b["step"], name=b["name"], dt=b["dt"], host_phase_peak=b["hwm"],
                         host_avail_min=min(m) if m else None,
                         gpu_node_used_max_gb={k: v / GB for k, v in sorted(gn.items())},
                         card_max_gib={k: v["mib"] / 1024 for k, v in sorted(per.items())},
                         card_util_mean={k: sum(v["util"]) / len(v["util"])
                                         for k, v in sorted(per.items())},
                         samples=len(g)))
        prev = hi
    tail = None
    if "failure" in card or "finished" not in card:
        last = card["boundaries"][-1]["t"] if card["boundaries"] else t0
        g = [r for r in gpu if r[0] > last]
        m = [r[1] for r in mem if r[0] > last]
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
    print(f"  {'step':>4} {'phase':<16} {'s':>9} {'host peak':>10} {'cpu free':>10}  card max GiB"
          "  | gpu-node used max GB")
    for r in rows:
        print(f"  {r['step']:>4} {r['name']:<16} {r['dt']:9.1f} {_gb(r['host_phase_peak']):>10} "
              f"{_gb(r['host_avail_min']):>10}  "
              f"{[round(v, 1) for v in r['card_max_gib'].values()]}  | "
              f"{[round(v, 1) for v in r['gpu_node_used_max_gb'].values()]}")
    if tail:
        print(f"  UNFINISHED after '{tail['after']}': {tail['seconds']} s more sampled; card max "
              f"GiB {tail['card_max_gib']}; cpu free min {_gb(tail['host_avail_min'])} GB")
    print(f"  summary {args.out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "run"):
        p = sub.add_parser(name)
        p.add_argument("--membind-nodes", default=None,
                       help="the NUMA nodes this process must be bound to (e.g. 0,1): "
                            "refuse unless `numactl --membind` applied, and after the load "
                            "refuse if the state sits on CPU-less (HBM) nodes")
        p.add_argument("--preset", required=True)
        p.add_argument("--workdir", required=True, help="the IC directory (read only)")
        p.add_argument("--card", required=True)
        p.add_argument("--cards", type=int, default=4)
        p.add_argument("--slack", type=float, default=0.10)
        p.add_argument("--alloc-margin", type=float, default=0.10)
        p.add_argument("--arena-frac", type=float, default=0.01)
    pf = sub.choices["preflight"]
    pf.add_argument("--scratch", default=os.environ.get("SCRATCH", "/tmp"))
    pf.add_argument("--host-margin", type=float, default=0.95,
                    help="refuse if the planner's host or load peak exceeds this x the CPU "
                         "nodes' MemTotal")
    pf.add_argument("--allow-cpu", action="store_true")
    pr = sub.choices["run"]
    pr.add_argument("--stop-at", type=int, required=True, help="steps to run from the ICs")
    pr.add_argument("--timed-last", action="store_true",
                    help="synced per-phase breakdown of the device passes on the last step")
    pr.add_argument("--timed-all", action="store_true",
                    help="the same breakdown on every step")
    pr.add_argument("--numa-maps", action="store_true",
                    help="the process's pages per NUMA node at every boundary (slow walk)")
    pr.add_argument("--ckpt-probe-slabs", type=int, default=0,
                    help="after the run, write this many slabs of the state, timed per part")
    pr.add_argument("--ckpt-probe-dir", default=None)
    pr.add_argument("--off-node-gb", type=float, default=8.0,
                    help="with --membind-nodes: GB of this process allowed on CPU-less nodes")
    pr.add_argument("--trim-probe", action="store_true",
                    help="after the run (and the probe), gc + malloc_trim(0), RSS either side")
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
