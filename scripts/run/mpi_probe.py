"""Two-node MPI probe for the multi-node device lane: one rank per node, host MPI over mpi4py.

Each leg is its own launch (`ibrun ... python -m mpi4py mpi_probe.py <leg>`, or `mpiexec -n 2`
locally); `-m mpi4py` turns an uncaught exception on any rank into `MPI_Abort`. Every leg
records SIGUSR1 receipts (faulthandler stacks plus a stamped line) in `--out`, so a pre-wall
signal landing mid-leg is kept, not fatal. Rank 0 prints `GATE <name> PASS|FAIL <detail>` and
the leg exits non-zero if any gate failed.

  identity   MPI before jax; library, threading, placement, NUMA policy, affinity, devices,
             mapped libmpi / CUDA libraries, and the per-rank provenance allgathered
  bandwidth  sendrecv and alltoallv of host numpy buffers, 1 MiB - 1 GiB messages, reused and
             fresh buffers, after a jax host -> device -> host round trip; sampled crc32
  abort      both ranks arm after a Barrier; rank 1 raises while rank 0 blocks in Recv
  kill       as abort, rank 1 SIGKILLs itself (the OOM case: no exception, no MPI_Abort)
  signal     waits for SIGUSR1 (sent by the job's pre-wall trap), then gathers the receipts
  check-fail gates a finished abort/kill launch from its log, exit code and wall time
  host-attrib  one process, no MPI: per-phase host allocations of a device-lane run
"""

import argparse
import faulthandler
import glob
import json
import os
import signal
import socket
import subprocess
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)

MiB = 1 << 20
# /proc, sched_getaffinity and ldd: off Linux (a laptop rehearsal) those gates print SKIP
LINUX = sys.platform.startswith("linux")


# ------------------------------------------------------------------ common


def _receipt_path(out, rank):
    return os.path.join(out, f"usr1_rank{rank}.txt")


def _install_usr1(out, rank):
    """Stamp every SIGUSR1 into a per-rank file and dump all threads' stacks beside it."""
    path = _receipt_path(out, rank)
    stacks = open(os.path.join(out, f"usr1_stacks_rank{rank}.txt"), "a")

    def stamp(_sig, _frame):
        with open(path, "a") as fh:
            fh.write(f"{time.time():.3f} {os.environ.get('PROBE_LEG', '?')}\n")

    signal.signal(signal.SIGUSR1, stamp)
    # chain=True: the stack dump runs at C level, then the Python stamp above
    faulthandler.register(signal.SIGUSR1, file=stacks, all_threads=True, chain=True)


def _receipts(out, rank):
    try:
        with open(_receipt_path(out, rank)) as fh:
            return [float(ln.split()[0]) for ln in fh if ln.strip()]
    except OSError:
        return []


def _gate(results, name, ok, detail="", linux_only=False):
    if linux_only and not LINUX:
        print(f"GATE {name} SKIP (not Linux)", flush=True)
        return
    results.append((name, bool(ok)))
    print(f"GATE {name} {'PASS' if ok else 'FAIL'} {detail}", flush=True)


def _finish(results):
    return 0 if results and all(ok for _, ok in results) else 1


def _world():
    from mpi4py import MPI

    return MPI, MPI.COMM_WORLD


def _node_cpus(nodes):
    cpus = set()
    for n in nodes:
        with open(f"/sys/devices/system/node/node{n}/cpulist") as fh:
            for part in fh.read().strip().split(","):
                if part:
                    a, _, b = part.partition("-")
                    cpus.update(range(int(a), int(b or a) + 1))
    return cpus


def _mapped(patterns):
    """Paths of mapped shared objects whose basename starts with one of `patterns`."""
    got = {}
    try:
        with open("/proc/self/maps") as fh:
            for ln in fh:
                parts = ln.split()
                if len(parts) < 6 or not parts[5].startswith("/"):
                    continue
                path = parts[5]
                base = os.path.basename(path)
                for p in patterns:
                    if base.startswith(p):
                        got.setdefault(p, set()).add(os.path.realpath(path))
    except OSError:
        return None
    return {k: sorted(v) for k, v in got.items()}


def _git_commit():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance only
        return None


def _parents(n=6):
    out, pid = [], os.getppid()
    for _ in range(n):
        try:
            with open(f"/proc/{pid}/stat") as fh:
                f = fh.read().split()
            out.append(f"{pid}:{f[1]}")
            pid = int(f[3])
        except (OSError, IndexError, ValueError):
            break
        if pid <= 1:
            break
    return out


# ------------------------------------------------------------------ legs


def leg_identity(args):
    MPI, comm = _world()  # MPI is initialized before jax is imported
    rank, size = comm.Get_rank(), comm.Get_size()
    rec = dict(rank=rank, size=size, host=socket.gethostname(),
               library=MPI.Get_library_version().strip(),
               thread_level=int(MPI.Query_thread()),
               slurm={k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_STEP_ID",
                                                    "SLURM_PROCID", "SLURM_NODEID")},
               parents=_parents(),
               affinity=(sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                         else None))
    try:
        with open("/proc/self/cgroup") as fh:
            rec["cgroup"] = fh.read().strip()
    except OSError:
        rec["cgroup"] = None
    import device_run

    rec["numa_policy"] = device_run.memory_policy()
    rec["membind_refusals"] = (device_run.membind_refusals(set(args.expect_mems))
                               if args.expect_mems else None)
    mpi_so = glob.glob(os.path.join(os.path.dirname(MPI.__file__), "MPI*.so"))
    rec["ldd_libmpi"] = None
    if mpi_so and LINUX:
        ldd = subprocess.run(["ldd", mpi_so[0]], capture_output=True, text=True).stdout
        rec["ldd_libmpi"] = [ln.strip() for ln in ldd.splitlines() if "libmpi" in ln]

    import jax

    devs = jax.devices()
    rec["platform"] = devs[0].platform
    rec["n_gpus"] = sum(d.platform == "gpu" for d in devs)
    rec["device_kind"] = devs[0].device_kind
    rec["jaxlib"] = __import__("jaxlib").__version__
    try:
        with open("/proc/driver/nvidia/version") as fh:
            line = fh.readline().strip()
        rec["driver_line"] = line  # carries each node's module build host and date
        rec["driver"] = next((t for t in line.split() if t[:1].isdigit() and "." in t), line)
    except OSError:
        rec["driver"] = rec["driver_line"] = None
    rec["commit"] = _git_commit()
    rec["mpi4py"] = os.path.realpath(os.path.dirname(os.path.dirname(MPI.__file__)))
    rec["maps"] = _mapped(("libmpi", "libcudart.so.12", "libcudart.so.13", "libcuda.",
                           "libnccl", "libucp", "libfabric"))
    with open(os.path.join(args.out, f"identity_rank{rank}.json"), "w") as fh:
        json.dump(rec, fh, indent=1)

    all_rec = comm.gather(rec, root=0)
    if rank != 0:
        return 0
    res = []
    prefix = os.path.realpath(os.environ.get("CONDA_PREFIX", "/nonexistent"))
    _gate(res, "ranks", size == args.expect_ranks, f"{size} of {args.expect_ranks}")
    hosts = {r["host"] for r in all_rec}
    _gate(res, "distinct-hosts", len(hosts) == size or args.allow_one_host, sorted(hosts))
    libs = [r["library"].splitlines()[0] for r in all_rec]
    _gate(res, "library", all(args.expect_mpi.lower() in s.lower() for s in libs), libs)
    maps = [(r["maps"] or {}).get("libmpi", []) for r in all_rec]
    in_env = [any(p.startswith(prefix + os.sep) for p in m) for m in maps]
    _gate(res, "libmpi-from-module",
          all(m for m in maps) and (args.allow_env_mpi or not any(in_env)), maps,
          linux_only=True)
    if args.mpi4py_dir:
        want = os.path.realpath(args.mpi4py_dir)
        _gate(res, "mpi4py-built-for-the-module",
              all(r["mpi4py"].startswith(want) for r in all_rec),
              sorted({r["mpi4py"] for r in all_rec}))
    # jax's CUDA 12 runtime and NCCL must come from the env, whatever the MPI module maps
    cuda = [p for r in all_rec for k in ("libcudart.so.12", "libnccl")
            for p in (r["maps"] or {}).get(k, [])]
    # (none mapped is possible: a jaxlib may link its runtime statically or load NCCL late)
    _gate(res, "jax-cuda-libs-from-env", all(p.startswith(prefix + os.sep) for p in cuda),
          sorted(set(cuda)), linux_only=True)
    print("MPI-MAPPED CUDA 13", sorted({p for r in all_rec
                                        for p in (r["maps"] or {}).get("libcudart.so.13", [])}))
    gpus = [r["n_gpus"] for r in all_rec]
    _gate(res, "gpus", all(g == args.expect_gpus for g in gpus), gpus)
    if args.expect_mems:
        refs = [r["membind_refusals"] for r in all_rec]
        _gate(res, "membind", all(not x for x in refs), refs, linux_only=True)
        want = _node_cpus(args.expect_mems) if LINUX else set()
        aff = [len(set(r["affinity"] or ()) & want) for r in all_rec]
        _gate(res, "affinity", all(set(r["affinity"] or ()) >= want for r in all_rec),
              f"{aff} of {len(want)} cores of nodes {args.expect_mems}", linux_only=True)
    same = {k: {json.dumps(r[k]) for r in all_rec}
            for k in ("commit", "jaxlib", "driver", "device_kind")}
    _gate(res, "provenance-identical", all(len(v) == 1 for v in same.values()),
          {k: len(v) for k, v in same.items()})
    print("THREAD_LEVEL", [r["thread_level"] for r in all_rec], "(MULTIPLE = 3)")
    print("SLURM", [r["slurm"] for r in all_rec])
    return _finish(res)


def _crc(buf):
    return zlib.crc32(memoryview(buf)) & 0xFFFFFFFF


def leg_bandwidth(args):
    import numpy as np

    MPI, comm = _world()
    rank, size = comm.Get_rank(), comm.Get_size()
    if size != 2:
        raise SystemExit("bandwidth needs exactly 2 ranks")
    peer = 1 - rank
    import jax

    # pinned host memory in use, as in a production step, before any MPI traffic
    probe = np.arange(8 * MiB, dtype=np.uint8)
    back = np.asarray(jax.device_put(probe, jax.devices()[0]))
    assert int(back[-1]) == int(probe[-1])

    sizes = [int(s) * MiB for s in args.sizes_mib]
    rows, res = [], []
    for nbytes in sizes:
        reps = max(2, min(args.max_reps, args.volume_mib * MiB // nbytes))
        for fresh in (False, True):
            if fresh and nbytes < 64 * MiB:
                continue
            reps_here = min(reps, 4) if fresh else reps
            send = np.empty(nbytes, dtype=np.uint8)
            send[:] = rank + 1
            recv = np.empty(nbytes, dtype=np.uint8)
            crc_ok = True
            comm.Barrier()
            t0 = time.perf_counter()
            for i in range(reps_here):
                if fresh:
                    send = np.full(nbytes, rank + 1, dtype=np.uint8)
                    recv = np.empty(nbytes, dtype=np.uint8)
                sample = i in (0, reps_here - 1)
                if sample:
                    send[:8] = np.frombuffer(np.int64(i * 2 + rank).tobytes(), dtype=np.uint8)
                comm.Sendrecv(send, dest=peer, recvbuf=recv, source=peer)
                if sample:
                    mine = comm.sendrecv(_crc(send), dest=peer, source=peer)
                    crc_ok &= mine == _crc(recv)
            comm.Barrier()
            dt = time.perf_counter() - t0
            rows.append(dict(op="sendrecv", mib=nbytes // MiB, fresh=fresh, reps=reps_here,
                             s=dt, gbs_per_direction=nbytes * reps_here / dt / 1e9, crc=crc_ok))
        # alltoallv: nothing to self, the whole buffer to the peer
        counts = [0, 0]
        counts[peer] = nbytes
        displs = [0, 0]
        send = np.full(nbytes, rank + 1, dtype=np.uint8)
        recv = np.empty(nbytes, dtype=np.uint8)
        comm.Barrier()
        t0 = time.perf_counter()
        for _ in range(reps):
            comm.Alltoallv([send, (counts, displs), MPI.BYTE],
                           [recv, (counts, displs), MPI.BYTE])
        comm.Barrier()
        dt = time.perf_counter() - t0
        ok = comm.sendrecv(_crc(send), dest=peer, source=peer) == _crc(recv)
        rows.append(dict(op="alltoallv", mib=nbytes // MiB, fresh=False, reps=reps, s=dt,
                         gbs_per_direction=nbytes * reps / dt / 1e9, crc=ok))
    hca = sorted(os.listdir("/sys/class/infiniband")) if os.path.isdir(
        "/sys/class/infiniband") else []
    env = {k: v for k, v in os.environ.items()
           if k.startswith(("MV2_", "MVP_", "UCX_", "FI_", "MPIR_", "MPICH_"))}
    with open(os.path.join(args.out, f"bandwidth_rank{rank}.json"), "w") as fh:
        json.dump(dict(rows=rows, hca=hca, env=env), fh, indent=1)
    all_rows = comm.gather(rows, root=0)
    if rank != 0:
        return 0
    print(f"HCA {hca}")
    print(f"{'op':<10} {'MiB':>6} {'fresh':>5} {'reps':>5} {'GB/s/dir':>9} crc(both ranks)")
    for i, r in enumerate(all_rows[0]):
        crc = all(rr[i]["crc"] for rr in all_rows)
        print(f"{r['op']:<10} {r['mib']:>6} {str(r['fresh']):>5} {r['reps']:>5} "
              f"{r['gbs_per_direction']:>9.2f} {crc}")
    _gate(res, "crc", all(r["crc"] for rr in all_rows for r in rr))
    return _finish(res)


def leg_fail(args, how):
    import numpy as np

    _MPI, comm = _world()
    rank = comm.Get_rank()
    comm.Barrier()
    print(f"ARMED rank {rank}", flush=True)
    time.sleep(1.0)
    if rank == 1:
        if how == "kill":
            os.kill(os.getpid(), signal.SIGKILL)
        raise RuntimeError("probe: deliberate failure on rank 1")
    buf = np.empty(1, dtype=np.int64)
    comm.Recv(buf, source=1)  # never satisfied
    print("COMPLETED rank 0", flush=True)
    return 0


def leg_check_fail(args):
    """Gate a finished abort/kill launch: non-zero exit inside the bound, both ranks armed,
    rank 0 never completed. `--log` is the launch's combined output."""
    with open(args.log, errors="replace") as fh:
        text = fh.read()
    res = []
    armed = sorted({ln.split()[-1] for ln in text.splitlines() if ln.startswith("ARMED rank")})
    _gate(res, "armed-both", armed == ["0", "1"], armed)
    _gate(res, "exit-nonzero", args.rc != 0, f"rc {args.rc}")
    _gate(res, "ended-in-bound", args.elapsed <= args.bound, f"{args.elapsed:.1f} s")
    _gate(res, "rank0-never-completed", "COMPLETED rank 0" not in text)
    return _finish(res)


def leg_signal(args):
    _MPI, comm = _world()
    rank = comm.Get_rank()
    t_end = time.time() + args.wait
    while time.time() < t_end and not _receipts(args.out, rank):
        time.sleep(0.5)
    mine = _receipts(args.out, rank)
    got = comm.gather(dict(rank=rank, host=socket.gethostname(), receipts=mine), root=0)
    if rank != 0:
        return 0
    res = []
    batch = None
    if args.batch_stamp and os.path.exists(args.batch_stamp):
        with open(args.batch_stamp) as fh:
            batch = float(fh.read().split()[0])
    for g in got:
        first = min(g["receipts"]) if g["receipts"] else None
        lag = None if first is None or batch is None else first - batch
        print(f"rank {g['rank']} on {g['host']}: first receipt {first}, "
              f"lag after the batch shell {lag}")
    _gate(res, "every-rank-received", all(g["receipts"] for g in got))
    if batch is not None:
        _gate(res, "within-60s", all(g["receipts"] and min(g["receipts"]) - batch <= 60
                                     for g in got))
    return _finish(res)


# ------------------------------------------------------------------ host attribution


def leg_host_attrib(args):
    """Per-phase host allocations of a device-lane run on this node's cards (no MPI).

    A perturbed-lattice state at `--preset`; for each engine phase the RSS at its start and
    end and its high-water (VmHWM reset per phase), and the tracemalloc lines that grew most at
    the phase's traced peak. RSS rise minus traced rise is host memory numpy did not allocate
    (driver, XLA). Writes `host_attrib.json` in `--out`."""
    import threading
    import tracemalloc

    import jax
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    import device_run
    from inexor import engine, state
    from inexor.codec import T9Layout
    from inexor.config import Cosmology
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table
    from inexor.plan import PRESETS, engine_config

    g = PRESETS[args.preset]
    cards = len(jax.devices())
    cfg = engine_config(args.preset, coarse_backend="device", tile_backend="device",
                        migrate_backend="device", device_tile_window=True, device_cards=cards)
    n, L = g["n_part"], g["box"]
    rng = np.random.default_rng(3)
    q = (np.arange(n) + 0.5) * (L / n)
    x = np.stack(np.meshgrid(q, q, q, indexing="ij"), -1).reshape(-1, 3)
    x = np.mod(x + rng.normal(scale=0.25 * L / n, size=x.shape), L)
    v = np.random.default_rng(4).normal(scale=0.5, size=x.shape)
    st = state.SlotState.build(x, v, T9Layout(box_size=L, n_part=n, bucket_cells=2),
                               g["n_fine"] // cfg.n_brick, brick_slack=0.10, arena_frac=0.01)
    del x, v, q
    co = bullfrog_float_coeffs(bullfrog_table(a_grid(0.1, 1.0, args.steps, "log"),
                                              Cosmology()))

    def rss():
        m = device_run.proc_memory()
        return m.get("VmRSS", 0), m.get("VmHWM", 0)

    def reset_hwm():
        try:
            with open("/proc/self/clear_refs", "w") as fh:
                fh.write("5")
        except OSError:
            pass

    skip = [tracemalloc.Filter(False, tracemalloc.__file__),
            tracemalloc.Filter(False, "<frozen*")]
    tracemalloc.start(4)
    lock = threading.Lock()
    cur = dict(base=None, best=0, snap=None, rss0=None)
    phases, stop = [], [False]

    def sampler():
        while not stop[0]:
            c, _ = tracemalloc.get_traced_memory()
            with lock:
                if c > cur["best"]:
                    cur["best"] = c
                    cur["snap"] = tracemalloc.take_snapshot().filter_traces(skip)
            time.sleep(args.sample_s)

    def ph(name):
        c, _ = tracemalloc.get_traced_memory()
        r, hwm = rss()
        with lock:
            if cur["base"] is not None:
                top = []
                if cur["snap"] is not None:
                    for s in cur["snap"].compare_to(cur["base"][1], "lineno")[:10]:
                        if s.size_diff > 0:
                            fr = s.traceback[0]
                            top.append([os.path.relpath(fr.filename, REPO), fr.lineno,
                                        s.size_diff])
                phases.append(dict(name=name, rss_start=cur["rss0"], rss_end=r, hwm=hwm,
                                   traced_start=cur["base"][0], traced_peak=cur["best"],
                                   traced_end=c, top=top))
                print(f"  {name:<14} rss {cur['rss0'] / 1e9:8.2f} -> {r / 1e9:8.2f} GB, "
                      f"hwm {hwm / 1e9:8.2f}, traced +{(cur['best'] - cur['base'][0]) / 1e9:.2f}"
                      f" GB", flush=True)
            reset_hwm()
            cur.update(base=(c, tracemalloc.take_snapshot().filter_traces(skip)), best=0,
                       snap=None, rss0=rss()[0])

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    ph("setup")
    engine.run(st, cfg, co, phase=ph)
    ph("done")
    stop[0] = True
    with open(os.path.join(args.out, "host_attrib.json"), "w") as fh:
        json.dump(dict(preset=args.preset, cards=cards, n=n ** 3, rows=int(st.off.shape[0]),
                       n_buckets=int(st.n_buckets), phases=phases), fh, indent=1)
    return 0


# ------------------------------------------------------------------ entry


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("leg", choices=("identity", "bandwidth", "abort", "kill", "signal",
                                    "check-fail", "host-attrib"))
    ap.add_argument("--out", required=True, help="directory for per-rank records")
    ap.add_argument("--expect-ranks", type=int, default=2)
    ap.add_argument("--expect-gpus", type=int, default=0,
                    help="GPUs per rank (0 = a CPU jax, for a laptop rehearsal)")
    ap.add_argument("--expect-mems", type=int, nargs="*", default=[],
                    help="NUMA nodes host memory must be bound to (gh 0, gb 0 1)")
    ap.add_argument("--expect-mpi", default="MVAPICH",
                    help="substring of MPI_Get_library_version every rank must report")
    ap.add_argument("--mpi4py-dir", default=None,
                    help="identity: mpi4py must be imported from here (built for the module)")
    ap.add_argument("--allow-env-mpi", action="store_true",
                    help="accept a libmpi from the env (a laptop rehearsal)")
    ap.add_argument("--allow-one-host", action="store_true", help="a laptop rehearsal")
    ap.add_argument("--sizes-mib", type=int, nargs="+", default=[1, 4, 16, 64, 256, 1024])
    ap.add_argument("--volume-mib", type=int, default=16384,
                    help="bytes moved per direction per size class")
    ap.add_argument("--max-reps", type=int, default=4096)
    ap.add_argument("--wait", type=float, default=480.0, help="signal: seconds to wait")
    ap.add_argument("--batch-stamp", default=None,
                    help="signal: file holding the batch shell's receipt time")
    ap.add_argument("--log", help="check-fail: the launch's output")
    ap.add_argument("--rc", type=int, help="check-fail: the launch's exit code")
    ap.add_argument("--elapsed", type=float, help="check-fail: the launch's wall seconds")
    ap.add_argument("--bound", type=float, default=60.0, help="check-fail: seconds allowed")
    ap.add_argument("--preset", default="cgh64", help="host-attrib: geometry")
    ap.add_argument("--steps", type=int, default=2, help="host-attrib: steps to run")
    ap.add_argument("--sample-s", type=float, default=0.005, help="host-attrib: sampler period")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    os.environ["PROBE_LEG"] = args.leg
    if args.leg == "check-fail":
        return leg_check_fail(args)
    if args.leg == "host-attrib":
        _install_usr1(args.out, 0)
        return leg_host_attrib(args)
    _MPI, comm = _world()  # already initialized by `-m mpi4py`, before any jax import
    _install_usr1(args.out, comm.Get_rank())
    return dict(identity=leg_identity, bandwidth=leg_bandwidth, signal=leg_signal,
                abort=lambda a: leg_fail(a, "raise"), kill=lambda a: leg_fail(a, "kill"))[
                    args.leg](args)


if __name__ == "__main__":
    sys.exit(main())
