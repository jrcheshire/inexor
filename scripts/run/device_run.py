"""Device-lane stepping driver: a preset on N GPUs from ICs (or a checkpoint), instrumented.

Subcommands:

    preflight  refuse early: device count and backend, the allocator's receipt, the
               planner's host/load peak against the CPU NUMA nodes' memory, the IC
               manifest (shape, slab count, growth2), the memory binding, scratch space
    run        load the ICs or resume a checkpoint, step to `--stop-at` on the device
               lane (coarse, tile and migrate on the cards; window and fused pass on
               the config's auto), optionally timed per pass; then optionally a
               partial checkpoint write timed per part and a malloc_trim probe
    ics        device ICs into --workdir (`icgen.generate_t9_slabs_device`), on one node or,
               under --comm mpi, across nodes (every rank writes its own slabs; rank 0 the
               manifest); one boundary per generator stage
    card       the P(k) card of a checkpoint, painted and transformed on the cards
               (`summary.pk_summary_card_cards`); rank 0 writes it to `--out`
    export     the (x, v) export of a checkpoint, decoded on the cards, one part per
               rank (`export.write_particle_parts`, format inexor-particles-2)
    summarize  after the fact, no jax: the run card against the job's sampler CSVs,
               per phase (card memory max per GPU, host MemAvailable min, utilization)

Usage (as the Vista job scripts call it):

    python scripts/run/device_run.py preflight --preset c-hero --workdir $ICS \
        --card pre.json --scratch $RUNS --membind-nodes 0,1
    python scripts/run/device_run.py run --preset c-hero --workdir $ICS --card run.json \
        --k-steps 120 --expect-step 0 --stop-at 20 --checkpoint-dir $CKPT \
        --checkpoint-every 20 --membind-nodes 0,1
    python scripts/run/device_run.py ics --preset c-hero --workdir $ICS --card ics.json
    python scripts/run/device_run.py card --preset c-hero --checkpoint-dir $CKPT \
        --k-steps 120 --expect-step 120 --card card-run.json --out pk.json
    python scripts/run/device_run.py export --preset c-hero --checkpoint-dir $CKPT \
        --k-steps 120 --expect-step 120 --card export-run.json --export-dir $EXPORT
    python scripts/run/device_run.py summarize --card run.json --gpu-csv gpu.csv \
        --mem-csv mem.csv --out summary.json

Across nodes, one process per node under MPI (`--comm mpi`, launched as `mpiexec -n N python
-m mpi4py scripts/run/device_run.py run ... --comm mpi`): each rank loads its own brick
slabs of the ICs or checkpoint, writes its own card (`run.rank<r>.json`) and prefixes its
log lines with its rank; the checkpoints are written by all ranks together and do not
depend on the rank count. `--comm-timeout` is the watchdog: a rank waiting longer at one
exchange aborts the job.

Refusals in `run`: a card or checkpoint under the IC directory; a checkpoint dir that
already holds a checkpoint unless `--expect-step` (resume) is given; a resume whose
newest checkpoint is not at `--expect-step`; a memory binding that did not apply. In `card`
and `export`: an output under the checkpoint directory, a non-empty export directory, a
newest checkpoint not at `--expect-step`, and (export) a checkpoint short of `--k-steps`
unless `--allow-partial`.

What survives a failure:
- Every engine phase boundary prints one line when it is crossed (wall clock, step,
  seconds, host VmRSS / per-phase VmHWM / MemAvailable, each card's allocator bytes),
  and the card JSON is rewritten atomically at every boundary.
- A heartbeat thread prints every `--beat` seconds: the last boundary and the time
  since it, host memory, bytes read/written, each card's allocator bytes, so a long
  compile, a slow phase and a hang look different.
- Each boundary also records where host memory is and what the kernel did to get it:
  RSS by kind (anon / file / shmem / locked / pinned), page faults, `/proc/vmstat`
  reclaim, compaction and NUMA counters, per-node file/dirty/writeback pages, and
  with `--numa-maps` the process's pages per node. The instruments' own seconds are
  recorded per boundary and kept out of the phase's.
- `STEP_JSON` per step: the engine's stats with every receipt.
- Any exception writes the traceback, every card's full `memory_stats()` and the last
  heartbeat into the card before re-raising.
- `faulthandler` on SIGUSR1 dumps every thread's stack (the sbatch sends it before the
  wall), and on a fatal signal.
- The job's own samplers (nvidia-smi and /proc/meminfo, 5 s, epoch-stamped) run
  outside this process, so an OOM kill or a hang cannot silence them; `summarize`
  lines them up against the boundaries recorded here.

`D7_FAIL_AT=<phase>` raises at that boundary, to exercise the failure path (with
`D7_FAIL_RANK=<r>`, on that rank only).
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

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    hides the CPU-side limit), plus `detail[node]` = the `NODE_KEYS` fields.
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

    Unbound, with the CPU nodes full of page cache from the IC load, the kernel has placed
    197 GB of host memory on a card's HBM node rather than reclaim the cache, and a later
    2 GiB device allocation failed on a card holding 24 GB. The binding keeps host memory
    on the host; this checks it actually applied. Refuses if under 90% of mappings are
    bound to `nodes`, or any is bound elsewhere."""
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
    # the arena grouping is ONE cost for the whole checkpoint. Folding it into a
    # per-slab rate and multiplying by nb charges it 32x over at an 8-slab probe.
    once = float(t.get("arena index", 0.0))
    per_slab = (wall - once) / n
    # bytes per second belongs against the WRITER's own seconds; `write` is the
    # main thread's blocked time, which the overlap makes smaller than the write
    write_s = float(t.get("write thread", t.get("write", 0.0)))
    out = dict(slabs=t.get("slabs", 0), of=nb, wall_s=wall, parts_s=t, wrote=wrote,
               before=before, after=after, once_s=once, per_slab_s=per_slab,
               projected_full_s=once + per_slab * nb)
    print(f"== checkpoint probe: {out['slabs']} of {nb} slabs in {wall:.1f} s "
          f"({per_slab:.2f} s/slab + {once:.1f} s once -> "
          f"{out['projected_full_s']:.0f} s for all), "
          + ", ".join(f"{k} {v:.1f} s" for k, v in t.items() if k != "slabs")
          + ("" if wrote is None else f"; wrote {wrote / GB:.1f} GB "
             f"({wrote / max(write_s, 1e-9) / 1e6:.0f} MB/s over the write itself)"),
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
    """Is `path` inside directory `root`? By PATH COMPONENT after realpath, never by
    string prefix (which would call `.../smoke-ckpt-probe` a child of `.../smoke`)."""
    if not path:
        return False
    a, b = os.path.realpath(path), os.path.realpath(root)
    return a == b or a.startswith(b + os.sep)


def timed_steps(k0, stop, *, timed_all=False, every=0, last=False):
    """The absolute steps of [k0, stop) that `engine.run` times: all of them, every `every`-th
    (step k with (k + 1) % every == 0, so resumed segments keep the cadence), or the last."""
    if timed_all:
        return tuple(range(k0, stop))
    if every:
        return tuple(k for k in range(k0, stop) if (k + 1) % every == 0)
    return (stop - 1,) if last else ()


def _y_blocks(value):
    """`--y-blocks`: a count, or "auto" (None: `decomp.auto_y_blocks`)."""
    return None if value == "auto" else int(value)


def _planner(preset, cards, slack, arena, alloc_margin=0.10, n_nodes=1, host_gb=1026.0,
             device_gb=199.0, y_blocks=None):
    cmd = [sys.executable, "-m", "inexor.plan", "--preset", preset, "--backend", "device",
           "--n-gpus", str(cards), "--host-gb", f"{host_gb:g}", "--device-gb", f"{device_gb:g}",
           "--arena-frac", str(arena), "--slack", str(slack),
           "--alloc-margin", str(alloc_margin), "--y-blocks", str(y_blocks or "auto")]
    if n_nodes != 1:
        cmd += ["--n-nodes", str(n_nodes)]
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
    # a manifest without the field predates it and was generated with EdS D2
    card["manifest"]["growth2"] = man.get("growth2", "eds")
    if card["manifest"]["growth2"] != args.growth2:
        refusals.append(f"the ICs were generated with growth2 = {card['manifest']['growth2']!r} "
                        f"and the run asks for {args.growth2!r}")

    plan = _planner(args.preset, args.cards, args.slack, args.arena_frac, args.alloc_margin,
                    n_nodes=args.n_nodes, host_gb=args.host_gb, device_gb=args.device_gb,
                    y_blocks=args.y_blocks)
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


class _RankLines:
    """A stream with every line prefixed by the rank, for an interleaved mpiexec log."""

    def __init__(self, out, rank):
        self._out, self._pre, self._bol = out, f"[rank {rank}] ", True

    def write(self, s):
        for part in s.splitlines(keepends=True):
            if self._bol:
                self._out.write(self._pre)
            self._out.write(part)
            self._bol = part.endswith("\n")
        return len(s)

    def flush(self):
        self._out.flush()

    def __getattr__(self, name):
        return getattr(self._out, name)


def _comm_up(args):
    """(comm, multi): MPI before jax; across ranks a per-rank card path and rank-prefixed
    log lines."""
    comm = None
    if args.comm == "mpi":
        from inexor.comm import MPIComm

        comm = MPIComm(timeout=args.comm_timeout)
    multi = comm is not None and comm.size > 1
    if multi:
        base, ext = os.path.splitext(args.card)
        args.card = f"{base}.rank{comm.rank}{ext}"
        sys.stdout = _RankLines(sys.stdout, comm.rank)
    return comm, multi


def _check_state_on_cpu_nodes(args, card):
    """After a load under --membind-nodes: refuse if the state sits on CPU-less (HBM) nodes.
    State on a card's HBM only shows up later as a device OOM on a nearly empty card."""
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


def cmd_run(args):
    comm, multi = _comm_up(args)
    if multi and args.ckpt_probe_slabs:
        raise SystemExit("FATAL: the checkpoint probe writes a whole state; one rank only")

    import jax

    from inexor import engine, icgen
    from inexor.decomp import Decomp
    from inexor.plan import engine_config
    from realization import _coeffs, _cosmo, _ic_linear_pk, _linear_pk, _require_ic_growth2

    jax.config.update("jax_enable_x64", True)
    ics = os.path.realpath(args.workdir)
    for what, path in (("the card", os.path.dirname(os.path.abspath(args.card))),
                       ("a checkpoint", args.checkpoint_dir)):
        if under(path, ics):
            raise SystemExit(f"FATAL: {what} would be written under the IC directory")
    if args.checkpoint_every and not args.checkpoint_dir:
        raise SystemExit("FATAL: --checkpoint-every needs --checkpoint-dir")
    if args.expect_step and not args.checkpoint_dir:
        raise SystemExit("FATAL: --expect-step needs --checkpoint-dir to resume from")
    if not args.expect_step and args.checkpoint_dir and any(
            os.path.exists(os.path.join(args.checkpoint_dir, f"gen{g}", "manifest.json"))
            for g in (0, 1)):
        raise SystemExit(f"FATAL: {args.checkpoint_dir} already holds a checkpoint; a run "
                         "from the ICs would overwrite it. Pass --expect-step to resume.")
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
    rank = 0 if comm is None else comm.rank
    fail_rank = os.environ.get("D7_FAIL_RANK")
    fail_at = (os.environ.get("D7_FAIL_AT")
               if fail_rank is None or int(fail_rank) == rank else None)
    mon = Monitor(args.card, card, beat_s=args.beat, fail_at=fail_at,
                  with_numa_maps=args.numa_maps)
    card["plan"] = dict(stop_at=args.stop_at, k_steps=args.k_steps,
                        expect_step=args.expect_step, timed_last=args.timed_last,
                        timed_all=args.timed_all, timed_every=args.timed_every,
                        numa_maps=args.numa_maps,
                        drop_ic_cache=args.drop_ic_cache,
                        ckpt_probe_slabs=args.ckpt_probe_slabs, trim_probe=args.trim_probe,
                        cards=args.cards, y_blocks=args.y_blocks or "auto",
                        slack=args.slack, alloc_margin=args.alloc_margin,
                        arena_frac=args.arena_frac, checkpoint_dir=args.checkpoint_dir,
                        checkpoint_every=args.checkpoint_every)
    try:
        devs = jax.devices()
        if len(devs) < args.cards:
            raise RuntimeError(f"{len(devs)} jax devices, {args.cards} asked")
        ec = engine_config(args.preset, coarse_backend="device", tile_backend="device",
                           migrate_backend="device", device_cards=args.cards, tile_workers=1,
                           device_y_blocks=args.y_blocks, brick_slack=args.slack,
                           checkpoint_dir=args.checkpoint_dir if args.checkpoint_every else None,
                           checkpoint_every=args.checkpoint_every)
        ec.validate()
        if multi and engine.rank_lane_refusal(ec):
            raise RuntimeError(engine.rank_lane_refusal(ec))
        decomp = Decomp.build(ec, n_ranks=1 if comm is None else comm.size, rank=rank)
        slabs = decomp.slabs if multi else None
        card["ranks"] = dict(rank=rank, n_ranks=decomp.n_ranks, slabs=list(decomp.slabs),
                             comm=args.comm, comm_timeout=args.comm_timeout)
        card["config"] = dict(tile_window=ec.tile_window, fused_pass=ec.fused_pass,
                              device_cards=ec.device_cards, y_blocks=ec.device_y_blocks,
                              checkpoint_dir=ec.checkpoint_dir,
                              coarse_dtype=ec.coarse_dtype, fine_dtype=ec.fine_dtype)
        src = (f"the step-{args.expect_step} checkpoint in {args.checkpoint_dir}"
               if args.expect_step else args.workdir)
        print(f"== device run {args.preset}: {args.cards} cards, steps {args.expect_step} -> "
              f"{args.stop_at} of {args.k_steps}, window={ec.tile_window} "
              f"fused={ec.fused_pass}, from {src}", flush=True)
        mon.start()

        cosmo = _cosmo()
        co, a_steps = _coeffs(cosmo, args.k_steps, growth2=args.growth2)
        resume = None
        if args.expect_step:
            # a resume job that silently restarted from the ICs would burn its whole
            # wall redoing finished steps, so the step it resumes from is stated
            st, resume = engine.load_checkpoint(
                args.checkpoint_dir, ec, co, brick_slack=args.slack,
                alloc_margin=args.alloc_margin, arena_frac=args.arena_frac, comm=comm,
                slabs=slabs)
            if int(resume["step"]) != args.expect_step:
                raise RuntimeError(
                    f"the newest checkpoint under {args.checkpoint_dir} is at step "
                    f"{resume['step']}, and this job was submitted to resume from step "
                    f"{args.expect_step}")
            linear_pk = _linear_pk(resume, cosmo)
        else:
            _require_ic_growth2(args.workdir, args.growth2)
            linear_pk = _ic_linear_pk(args.workdir, cosmo)
            st = icgen.load_slot_state(args.workdir, brick_slack=args.slack,
                                       alloc_margin=args.alloc_margin,
                                       arena_frac=args.arena_frac,
                                       drop_cache=args.drop_ic_cache, slabs=slabs)
        k0 = 0 if resume is None else int(resume["step"])
        card["start_step"] = k0
        card["state"] = dict(n_particles=st.n_particles, n_bricks=st.n_bricks,
                             rows=int(st.off.shape[0]), n_arena=int(st.n_arena))
        mon("load")
        if args.membind_nodes:
            _check_state_on_cpu_nodes(args, card)
        timed = timed_steps(k0, args.stop_at, timed_all=args.timed_all,
                            every=args.timed_every, last=args.timed_last)

        def collect(stats):
            s = {k: v for k, v in stats.items() if k not in ("pool", "busy", "loop_wall")}
            print("STEP_JSON " + json.dumps(s, default=str), flush=True)
            with mon.lock:
                card["steps"].append(s)

        # the ICs' linear P(k), if tabulated, rides along into every checkpoint
        epoch = (a_steps, cosmo, None if linear_pk is None else linear_pk.record())
        ic_prov = icgen.read_manifest(args.workdir).get("provenance") or {}
        source = dict(ics=ics, ics_provenance={k: ic_prov[k] for k in (
            "generator", "commit", "host", "when") if k in ic_prov})
        out = engine.run(st, ec, co, phase=mon, resume=resume, stop_at=args.stop_at,
                         collect=collect, timed_steps=timed, epoch=epoch,
                         comm=comm, decomp=decomp, source=source)
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


# ---------------------------------------------------------------------- ICs


def cmd_ics(args):
    """Device ICs into --workdir, one rank per node under --comm mpi."""
    comm, _multi = _comm_up(args)

    import jax

    from inexor import icgen
    from inexor.plan import engine_config
    from realization import GEN_FDTYPE, _cosmo, _geom, _ic_provenance, _pk_table_arg

    jax.config.update("jax_enable_x64", True)
    if os.path.exists(os.path.join(args.workdir, icgen.MANIFEST)):
        raise SystemExit(f"FATAL: {args.workdir} already holds an IC manifest")
    if under(os.path.dirname(os.path.abspath(args.card)), os.path.realpath(args.workdir)):
        raise SystemExit("FATAL: the run card would be written under the IC directory")
    if args.membind_nodes:
        bad = membind_refusals(_node_list(args.membind_nodes))
        if bad:
            raise SystemExit("FATAL: " + "; ".join(bad))
    _signals(args.card)
    card = _base_card("ics", args)
    rank = 0 if comm is None else comm.rank
    fail_rank = os.environ.get("D7_FAIL_RANK")
    fail_at = (os.environ.get("D7_FAIL_AT")
               if fail_rank is None or int(fail_rank) == rank else None)
    mon = Monitor(args.card, card, beat_s=args.beat, fail_at=fail_at)
    try:
        devs = jax.devices()
        if len(devs) < args.cards:
            raise RuntimeError(f"{len(devs)} jax devices, {args.cards} asked")
        devs = devs[:args.cards]
        g = _geom(args.preset)
        nb = g["n_fine"] // engine_config(args.preset).n_brick
        n_ranks = 1 if comm is None else comm.size
        card["ranks"] = dict(rank=rank, n_ranks=n_ranks, comm=args.comm,
                             comm_timeout=args.comm_timeout)
        card["plan"] = dict(seed=args.seed, a_init=args.a_init, growth2=args.growth2,
                            f_NL=args.f_nl, window=args.window, cards=args.cards,
                            batch_planes=args.batch_planes, y_blocks=args.y_blocks or "auto",
                            pencil_batch=args.pencil_batch, slab=args.slab,
                            pk_table=args.pk_table)
        print(f"== device ICs {args.preset}: n_part={g['n_part']} L={g['L']} "
              f"bricks_per_side={nb}, {args.cards} card(s) x {n_ranks} rank(s) -> "
              f"{args.workdir}", flush=True)
        mon.start()
        prov = _ic_provenance("device")
        prov.update(n_ranks=n_ranks, comm=args.comm)

        def log(line):
            print(line, flush=True)
            if line.strip().startswith("ic stage "):
                mon("ic_" + line.split(":")[0].split()[-1])

        cosmo = _cosmo()
        backend, table = _pk_table_arg(args, cosmo)
        t0 = time.time()
        man = icgen.generate_t9_slabs_device(
            args.workdir, jax.random.PRNGKey(args.seed), g["n_part"], g["L"], cosmo,
            args.a_init, nb, bucket_cells=args.bucket_cells, f_NL=args.f_nl,
            fdtype=GEN_FDTYPE, slab=args.slab, window=args.window, provenance=prov,
            growth2=args.growth2, devices=devs, pencil_batch=args.pencil_batch, log=log,
            comm=comm, batch_planes=args.batch_planes, emit_y_blocks=args.y_blocks,
            backend=backend, table=table)
        card["product"] = dict(wall_s=time.time() - t0, run_host_peak=mon.run_peak,
                               manifest={k: man.get(k) for k in (
                                   "n_particles", "bricks_per_side", "max_displacement",
                                   "vel_scale", "stage_s", "emission_s")})
        print(f"  ICs: {man['n_particles']:,} particles in {len(man['files'])} slab files, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        card["finished"] = time.time()
        mon.save()
        return 0
    except BaseException as e:
        mon.fail(e)
        raise
    finally:
        mon.stop()


# ---------------------------------------------------------------------- products


def cmd_card(args):
    return _product(args, "card")


def cmd_export(args):
    return _product(args, "export")


def _product(args, kind):
    """The P(k) card or the export of the newest checkpoint, on the cards, on every rank."""
    comm, multi = _comm_up(args)

    import jax
    import numpy as np

    from inexor import engine, export, summary
    from inexor.decomp import Decomp
    from inexor.plan import engine_config
    from realization import _coeffs, _cosmo, _linear_pk

    jax.config.update("jax_enable_x64", True)
    args.workdir = args.checkpoint_dir  # the base card's source field
    ckpt = os.path.realpath(args.checkpoint_dir)
    out = args.out if kind == "card" else args.export_dir
    for what, path in (("the run card", os.path.dirname(os.path.abspath(args.card))),
                       (f"the {kind}", out)):
        if under(path, ckpt):
            raise SystemExit(f"FATAL: {what} would be written under the checkpoint directory")
    if kind == "export" and os.path.isdir(out) and os.listdir(out):
        raise SystemExit(f"FATAL: export directory {out} is not empty")
    if args.membind_nodes:
        bad = membind_refusals(_node_list(args.membind_nodes))
        if bad:
            raise SystemExit("FATAL: " + "; ".join(bad))
    _signals(args.card)
    card = _base_card(kind, args)
    rank = 0 if comm is None else comm.rank
    fail_rank = os.environ.get("D7_FAIL_RANK")
    fail_at = (os.environ.get("D7_FAIL_AT")
               if fail_rank is None or int(fail_rank) == rank else None)
    mon = Monitor(args.card, card, beat_s=args.beat, fail_at=fail_at)
    try:
        devs = jax.devices()
        if len(devs) < args.cards:
            raise RuntimeError(f"{len(devs)} jax devices, {args.cards} asked")
        devs = devs[:args.cards]
        ec = engine_config(args.preset, coarse_backend="device", tile_backend="device",
                           migrate_backend="device", device_cards=args.cards, tile_workers=1,
                           brick_slack=args.slack)
        ec.validate()
        decomp = Decomp.build(ec, n_ranks=1 if comm is None else comm.size, rank=rank)
        slabs = decomp.slabs if multi else None
        card["ranks"] = dict(rank=rank, n_ranks=decomp.n_ranks, slabs=list(decomp.slabs),
                             comm=args.comm, comm_timeout=args.comm_timeout)
        print(f"== device {kind} {args.preset}: {args.cards} cards, the step-"
              f"{args.expect_step} checkpoint in {args.checkpoint_dir} -> {out}", flush=True)
        mon.start()

        cosmo = _cosmo()
        co, a_steps = _coeffs(cosmo, args.k_steps, growth2=args.growth2)
        t0 = time.time()
        st, resume = engine.load_checkpoint(
            args.checkpoint_dir, ec, co, brick_slack=args.slack,
            alloc_margin=args.alloc_margin, arena_frac=args.arena_frac, comm=comm, slabs=slabs)
        step = int(resume["step"])
        if step != args.expect_step:
            raise RuntimeError(f"the newest checkpoint under {args.checkpoint_dir} is at step "
                               f"{step}, and this job was submitted for step {args.expect_step}")
        load_s = time.time() - t0
        card["state"] = dict(n_particles=st.n_particles, n_bricks=st.n_bricks,
                             rows=int(st.off.shape[0]), n_arena=int(st.n_arena), step=step)
        mon("load")
        if args.membind_nodes:
            _check_state_on_cpu_nodes(args, card)
        a_out = float(a_steps[step])
        linear_pk = _linear_pk(resume, cosmo)
        prov = dict(preset=args.preset, step=step, commit=_git_commit(),
                    checkpoint=ckpt, n_ranks=decomp.n_ranks, cards=args.cards)
        if linear_pk is not None:
            prov["linear_pk"] = linear_pk.stamp()

        t0 = time.time()
        if kind == "card":
            edges = None
            if args.k_max:
                edges = np.linspace(0.0, float(args.k_max), int(args.n_bins) + 1)
            res = summary.pk_summary_card_cards(
                st, ec, cosmo, a_out, devices=devs, decomp=decomp, comm=comm, slab=args.slab,
                edges=edges, min_weight=args.min_weight, provenance=prov, phase=mon,
                linear_pk=linear_pk)
            wall = time.time() - t0
            card["product"] = dict(wall_s=wall, load_s=load_s, n_bins=res["n_bins"])
            if rank == 0:
                # the wrapper `realization.py card` writes, so the compare scripts read both
                _write_json(args.out, dict(
                    card="inexor-realization-pk-1", config=args.preset,
                    workdir=args.checkpoint_dir, commit=_git_commit(),
                    host=os.uname().nodename, k_steps=int(args.k_steps),
                    growth2=args.growth2, when=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    step=step, a_out=a_out, wall_s=wall, load_s=load_s,
                    n_ranks=decomp.n_ranks, cards=args.cards, summary=res))
            print(f"  card: {res['n_bins']} bins, {wall / 60:.1f} min; -> {args.out}",
                  flush=True)
        else:
            if step < args.k_steps and not args.allow_partial:
                raise RuntimeError(
                    f"the checkpoint is at step {step} of {args.k_steps}; exporting now would "
                    "produce a mock at the wrong epoch. Pass --allow-partial if that is "
                    "deliberate.")
            timings = {}
            epoch = {} if args.d_time else dict(a=a_out, cosmo=cosmo)
            n_all = (int(st.n_particles) if comm is None
                     else int(comm.allreduce(int(st.n_particles))))
            head = export.write_particle_parts(
                st, out, comm=comm, dtype=np.dtype(args.dtype), decode="cards", devices=devs,
                expect_total=n_all, provenance=prov, timings=timings, **epoch)
            wall = time.time() - t0
            mon("export")
            card["product"] = dict(wall_s=wall, load_s=load_s, timings=timings,
                                   rows=int(st.n_live), crc32=head["crc32"])
            print(f"  export: {int(st.n_live):,} rows on this rank of {head['n_particles']:,}, "
                  f"{wall / 60:.1f} min ({head['units']['velocity']}); crc32 {head['crc32']}",
                  flush=True)
        card["finished"] = time.time()
        mon.save()
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
        p.add_argument("--y-blocks", type=_y_blocks, default=None,
                       help="split each x-slab's card work into this many y-blocks "
                            "(EngineConfig.device_y_blocks); bitwise any count. Default "
                            "auto: units no larger than a 4096^3 x-slab")
        p.add_argument("--slack", type=float, default=0.10)
        p.add_argument("--alloc-margin", type=float, default=0.10)
        p.add_argument("--arena-frac", type=float, default=0.01)
        p.add_argument("--growth2", default="lcdm", choices=("lcdm", "eds"),
                       help="the second-order growth the ICs must have been generated with")
    pf = sub.choices["preflight"]
    pf.add_argument("--scratch", default=os.environ.get("SCRATCH", "/tmp"))
    pf.add_argument("--host-margin", type=float, default=0.95,
                    help="refuse if the planner's host or load peak exceeds this x the CPU "
                         "nodes' MemTotal")
    pf.add_argument("--allow-cpu", action="store_true")
    pf.add_argument("--n-nodes", type=int, default=1,
                    help="nodes the run spans; the planner prices the busiest node")
    pf.add_argument("--host-gb", type=float, default=1026.0,
                    help="the planner's host memory per node (default a gb node's CPU side)")
    pf.add_argument("--device-gb", type=float, default=199.0,
                    help="the planner's memory per card (default a GB200)")
    pr = sub.choices["run"]
    pr.add_argument("--stop-at", type=int, required=True,
                    help="absolute step to stop before (a checkpoint boundary when checkpointing)")
    pr.add_argument("--k-steps", type=int, default=40,
                    help="steps in the whole schedule; a segment runs part of it")
    pr.add_argument("--expect-step", type=int, default=0,
                    help="0 = start from the ICs; else resume from the newest checkpoint "
                         "under --checkpoint-dir, refusing unless it is at this step")
    pr.add_argument("--timed-last", action="store_true",
                    help="synced per-phase breakdown of the device passes on the last step")
    pr.add_argument("--timed-all", action="store_true",
                    help="the same breakdown on every step")
    pr.add_argument("--timed-every", type=int, default=0, metavar="N",
                    help="the same breakdown on every N-th step (absolute step k with "
                         "(k + 1) %% N == 0); the steps between are untimed, so their wall "
                         "is the step's cost without the syncs")
    pr.add_argument("--numa-maps", action="store_true",
                    help="the process's pages per NUMA node at every boundary (slow walk)")
    pr.add_argument("--ckpt-probe-slabs", type=int, default=0,
                    help="after the run, write this many slabs of the state, timed per part")
    pr.add_argument("--ckpt-probe-dir", default=None)
    pr.add_argument("--drop-ic-cache", action="store_true",
                    help="drop each IC slab's page cache as it is read, so the steps do "
                         "not run while the kernel drains it")
    pr.add_argument("--off-node-gb", type=float, default=8.0,
                    help="with --membind-nodes: GB of this process allowed on CPU-less nodes")
    pr.add_argument("--trim-probe", action="store_true",
                    help="after the run (and the probe), gc + malloc_trim(0), RSS either side")
    pr.add_argument("--beat", type=float, default=60.0, help="heartbeat seconds")
    pr.add_argument("--checkpoint-dir", default=None)
    pr.add_argument("--checkpoint-every", type=int, default=0,
                    help="0 = no checkpoints; else every this many steps (--stop-at a multiple)")
    pr.add_argument("--comm", default="serial", choices=("serial", "mpi"),
                    help="mpi: one rank per process across nodes (launch under mpiexec with "
                         "python -m mpi4py)")
    pr.add_argument("--comm-timeout", type=float, default=1800.0,
                    help="seconds a rank may wait at one exchange before aborting the job")
    pi = sub.add_parser("ics")
    pi.add_argument("--preset", required=True)
    pi.add_argument("--workdir", required=True, help="the IC directory (shared by the ranks)")
    pi.add_argument("--card", required=True, help="this process's run card")
    pi.add_argument("--cards", type=int, default=4)
    pi.add_argument("--seed", type=int, default=0)
    pi.add_argument("--a-init", type=float, default=0.1)
    pi.add_argument("--growth2", default="lcdm", choices=("lcdm", "eds"))
    pi.add_argument("--pk-table", default=None,
                    help="tabulated z = 0 linear P(k) (scripts/run/camb_linear_pk.py) instead "
                         "of EH98; embedded in the IC manifest, carried into the checkpoints")
    pi.add_argument("--f-nl", type=float, default=0.0, help="local f_NL of the ICs")
    pi.add_argument("--window", type=int, default=1,
                    help="emission window in brick slabs (refused if a displacement reaches it)")
    pi.add_argument("--bucket-cells", type=int, default=2)
    pi.add_argument("--slab", type=int, default=32)
    pi.add_argument("--pencil-batch", type=int, default=1)
    pi.add_argument("--batch-planes", type=int, default=16,
                    help="planes per rank in each plane <-> pencil exchange")
    pi.add_argument("--y-blocks", type=_y_blocks, default=None,
                    help="y-block units per destination slab in the emission (bitwise any). "
                         "Default auto: units no larger than a 4096^3 x-slab")
    pi.add_argument("--membind-nodes", default=None)
    pi.add_argument("--beat", type=float, default=60.0, help="heartbeat seconds")
    pi.add_argument("--comm", default="serial", choices=("serial", "mpi"))
    pi.add_argument("--comm-timeout", type=float, default=1800.0)
    for name in ("card", "export"):
        p = sub.add_parser(name)
        p.add_argument("--membind-nodes", default=None,
                       help="as `run`: refuse unless the binding applied and the loaded state "
                            "sits on the CPU nodes")
        p.add_argument("--off-node-gb", type=float, default=8.0)
        p.add_argument("--preset", required=True)
        p.add_argument("--checkpoint-dir", required=True, help="the run's checkpoint directory")
        p.add_argument("--k-steps", type=int, required=True,
                       help="steps in the run's schedule (the fingerprint and the epoch)")
        p.add_argument("--expect-step", type=int, required=True,
                       help="the step the newest checkpoint must be at")
        p.add_argument("--card", required=True, help="this process's run card")
        p.add_argument("--cards", type=int, default=4)
        p.add_argument("--slack", type=float, default=0.10)
        p.add_argument("--alloc-margin", type=float, default=0.10)
        p.add_argument("--arena-frac", type=float, default=0.01)
        p.add_argument("--growth2", default="lcdm", choices=("lcdm", "eds"))
        p.add_argument("--beat", type=float, default=60.0, help="heartbeat seconds")
        p.add_argument("--comm", default="serial", choices=("serial", "mpi"))
        p.add_argument("--comm-timeout", type=float, default=1800.0)
    pc = sub.choices["card"]
    pc.add_argument("--out", required=True, help="the P(k) card JSON (rank 0 writes it)")
    pc.add_argument("--slab", type=int, default=32)
    pc.add_argument("--min-weight", type=float, default=100.0)
    pc.add_argument("--k-max", type=float, default=None,
                    help="pin the top of the binning (as `realization.py card`)")
    pc.add_argument("--n-bins", type=int, default=64)
    pe = sub.choices["export"]
    pe.add_argument("--export-dir", required=True)
    pe.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    pe.add_argument("--d-time", action="store_true",
                    help="write the native dx/dD velocity instead of peculiar km/s")
    pe.add_argument("--allow-partial", action="store_true",
                    help="export a checkpoint short of --k-steps")
    ps = sub.add_parser("summarize")
    ps.add_argument("--card", required=True)
    ps.add_argument("--gpu-csv", default=None)
    ps.add_argument("--mem-csv", default=None)
    ps.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return dict(preflight=cmd_preflight, run=cmd_run, ics=cmd_ics, card=cmd_card,
                export=cmd_export,
                summarize=cmd_summarize)[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
