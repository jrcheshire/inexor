"""Run a Python script in this process with a memory sampler: host RSS and device bytes vs time.

    python scripts/compare/memtrace.py --out T.jsonl [--interval 0.02] -- script.py ARGS...

The script runs as `__main__` (via `runpy`) with `sys.argv = [script.py, ARGS...]`, so the
sampler sees exactly the process the script would have been. A daemon thread records, every
`--interval` seconds, one JSON line:

    {"t": epoch s, "rss": bytes, "anon": bytes, "file": bytes, "dev": [{bytes_in_use, ...}]}

`rss`/`anon`/`file` are `VmRSS`/`RssAnon`/`RssFile` from `/proc/self/status`; off Linux only
`rss` is sampled, through `ps`, at >= 0.5 s. `dev` is each local device's `memory_stats()`
(the allocator's own counters), queried only once the script has initialized a jax backend,
so the sampler never initializes one itself; it is null before that and on backends without
stats. The last line is a summary: wall, exit code, `VmHWM`, each device's
`peak_bytes_in_use`, and the trapezoid integrals of `rss` and of the summed `bytes_in_use`
over the samples (byte-seconds). Peaks come from the kernel's and the allocator's counters,
not from the samples, which can miss a short spike. Child processes are not counted.

Used by `gh_crossover_vista.sbatch` around each cost leg of both codes, so both are measured
by the same instrument.
"""

import argparse
import json
import os
import runpy
import subprocess
import sys
import threading
import time

DEV_KEYS = ("bytes_in_use", "peak_bytes_in_use", "bytes_reserved", "peak_bytes_reserved",
            "pool_bytes", "peak_pool_bytes", "bytes_limit")
PROC_STATUS = "/proc/self/status"


def host_sample():
    """(VmRSS, RssAnon, RssFile, VmHWM) in bytes from /proc; RSS alone via ps elsewhere."""
    if os.path.exists(PROC_STATUS):
        want = {"VmRSS": None, "RssAnon": None, "RssFile": None, "VmHWM": None}
        with open(PROC_STATUS) as fh:
            for line in fh:
                key = line.split(":", 1)[0]
                if key in want:
                    want[key] = int(line.split()[1]) * 1024
        return want["VmRSS"], want["RssAnon"], want["RssFile"], want["VmHWM"]
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                         capture_output=True, text=True).stdout.strip()
    return (int(out) * 1024 if out else None), None, None, None


def device_sample():
    """Each local device's allocator counters, or None if no backend is up yet."""
    if "jax" not in sys.modules:
        return None
    try:
        from jax._src import xla_bridge

        if not xla_bridge.backends_are_initialized():
            return None
        import jax

        out = []
        for d in jax.local_devices():
            s = d.memory_stats() or {}
            out.append({k: s[k] for k in DEV_KEYS if k in s} or None)
        return out
    except Exception as exc:  # the sampler must never take the run down
        return [{"error": repr(exc)}]


class Sampler(threading.Thread):
    def __init__(self, path, interval):
        super().__init__(daemon=True)
        self.fh = open(path, "w")
        self.interval = interval if os.path.exists(PROC_STATUS) else max(interval, 0.5)
        self.stop = threading.Event()
        self.t, self.rss, self.dev = [], [], []

    def sample(self):
        t = time.time()
        rss, anon, fil, _ = host_sample()
        dev = device_sample()
        self.t.append(t)
        self.rss.append(rss)
        used = [d.get("bytes_in_use") for d in dev or [] if d and "bytes_in_use" in d]
        self.dev.append(sum(used) if used else None)
        self.fh.write(json.dumps(dict(t=t, rss=rss, anon=anon, file=fil, dev=dev)) + "\n")

    def run(self):
        n = 0
        while not self.stop.is_set():
            self.sample()
            n += 1
            if n % 50 == 0:  # an OOM kill keeps all but the last second or so
                self.fh.flush()
            self.stop.wait(self.interval)

    def finish(self, summary):
        self.stop.set()
        self.join(timeout=5)
        self.sample()
        summary.update(n_samples=len(self.t), rss_byte_s=_trapz(self.t, self.rss),
                       dev_byte_s=_trapz(self.t, self.dev))
        self.fh.write(json.dumps(dict(summary=summary)) + "\n")
        self.fh.close()


def _trapz(t, y):
    """Trapezoid integral over the samples where y is defined (None where it is not)."""
    total = 0.0
    for i in range(1, len(t)):
        if y[i] is not None and y[i - 1] is not None:
            total += 0.5 * (y[i] + y[i - 1]) * (t[i] - t[i - 1])
    return total


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: memtrace.py --out T.jsonl [--interval S] -- script.py ARGS...")
    i = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=0.02)
    args = ap.parse_args(argv[:i])
    target = argv[i + 1:]
    if not target:
        raise SystemExit("memtrace.py: no script after --")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    sampler = Sampler(args.out, args.interval)
    t0 = time.time()
    sampler.start()
    sys.argv = list(target)
    sys.path.insert(0, os.path.dirname(os.path.abspath(target[0])))
    code, error = 0, None
    try:
        runpy.run_path(target[0], run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException as exc:
        code, error = 1, repr(exc)
        raise
    finally:
        _, _, _, hwm = host_sample()
        dev = device_sample()
        peaks = [d.get("peak_bytes_in_use") if d else None for d in dev or []]
        summary = dict(script=target[0], argv=target[1:], t_start=t0, t_end=time.time(),
                       wall_s=time.time() - t0, exit_code=code, error=error, vm_hwm=hwm,
                       dev_peak_bytes_in_use=peaks,
                       interval_s=sampler.interval, pid=os.getpid())
        sampler.finish(summary)
        print(f"memtrace: {target[0]} rc={code} wall {summary['wall_s']:.1f} s, "
              f"VmHWM {'n/a' if hwm is None else f'{hwm / 1e9:.2f} GB'}, "
              f"device peak {[None if p is None else round(p / 1e9, 2) for p in peaks]} GB, "
              f"{summary['n_samples']} samples -> {args.out}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
