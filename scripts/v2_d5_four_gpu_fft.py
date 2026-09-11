"""D5 (pulled forward): does the transform's wall actually divide by four?

WHAT THIS DECIDES. The D0 budget charges each of a gb node's four GB200s an
exact quarter of the coarse solve, and 5y's 114 s/step floor assumes a perfect
4-way split. Neither is measured: every number in record secs. 7-9 is ONE
GB200. If the split is 2x rather than 4x, the budget's per-GPU column and the
step floor both move, and they move before any more of the pipeline is built on
them.

WHY IT IS NOT OBVIOUS EITHER WAY. The factorization is embarrassingly parallel
and has zero inter-device communication (pass 1 independent per plane, pass 2
independent per y-pencil-plane, spectrum host-resident throughout), so nothing
in the ALGORITHM stops a 4x split. But D1 measured this path to be 91% host
bus, and the bus is the one resource four GPUs share. So this is a contention
measurement wearing an FFT costume, and it is designed as one.

STRONG SCALING, FIXED TOTAL. The design charges a quarter of a FIXED transform,
so the total work is held and the width varies: speedup = T(1)/T(W). A weak
scaling read would answer a question nobody asked.

THE ARMS.
  identity   The W=4 spectrum must be BITWISE equal to the W=1 spectrum, at the
             size being timed. This is an identity (a partition of a loop
             cannot touch a value), not a tolerance, and without it a wall
             measured at W=4 is a wall for a different transform.
  timing     The real inverse at n, W in {1,2,4}. The INVERSE is the clean leg:
             it consumes a pre-built spectrum, generates no field, and is 3 of
             the 4 transforms per step. The forward runs too but with the FLAT
             source -- host RNG is 52 s at 2048^3, does NOT split, and would
             swamp ~4 s of device work at W=4.
  transfer   The traffic proxy at W in {1,2,4} x {pageable, pinned}. Sec. 9
             established this proxy accounts for 91% of the real inverse, and
             it is what carries the PINNED answer: a pinned FFT path is not
             built here (it needs the slab window itself to live in pinned host
             memory, which is pipeline work), so pinned-at-four-GPUs comes back
             as a measured scaling times a projected single-GPU rate. Said
             plainly rather than quietly.
  driver     Threads vs processes at W=4, on the proxy. If the GIL serializes
             dispatch, the threaded driver measures Python and reports a false
             "does not scale". These have opposite consequences for the
             engine's structure, so they are told apart rather than assumed.

TWO THINGS THE APPARATUS FIXES.
  1. D1 timed a 34.4 GB host `spec.copy()` INSIDE the inverse. It is serial
     host work that cannot split, so at W=4 it would cap the observed speedup
     near 2.8x however well the bus scaled. Hoisted out of the timed region and
     measured on its own -- three inverses per step makes it a step-floor term
     in its own right if it is large.
  2. Every leg records WHICH devices did work (per-device peak > 0 on exactly
     W of them). A W=4 leg that silently ran everything on device 0 looks
     exactly like "it did not scale".

All widths run in ONE job on ONE node: D1 measured 1.36x of node-to-node spread
on identical work, which is larger than some of the effects here.

Run (Vista gb node, gpu env, all four GPUs visible in every leg):
    pixi run -e gpu python scripts/v2_d5_four_gpu_fft.py --out-suffix _gb
Laptop smoke (4 forced CPU devices; exercises partition, threads, identity):
    XLA_FLAGS=--xla_force_host_platform_device_count=4 \
        pixi run python scripts/v2_d5_four_gpu_fft.py --smoke --out-suffix _smoke
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
OUT_DIR = os.path.join(REPO, "runs", "v2")

# The bar, DERIVED: gb MaxWall 12 h / K=40 steps; the coarse solve is four
# transforms per step (one forward, three inverse).
WALL_S = 12 * 3600.0
BAR_S = WALL_S / 40.0
TRANSFORMS_PER_STEP = 4


def _provenance():
    import jax

    devs = jax.devices()
    return dict(
        host=socket.gethostname(), python=platform.python_version(),
        jax=jax.__version__, platform=devs[0].platform,
        n_devices=len(devs), devices=[str(d) for d in devs],
        cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
        host_mem_limit_gb=os.environ.get("XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB"),
        preallocate=os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        xla_flags=os.environ.get("XLA_FLAGS"),
        argv=" ".join(sys.argv),
    )


def _per_device_peak():
    """High-water per device. Never resets, which is why each arm is a process."""
    import jax

    out = {}
    for d in jax.devices():
        try:
            out[str(d)] = int(d.memory_stats().get("peak_bytes_in_use") or 0)
        except Exception:
            out[str(d)] = None
    return out


def _participation(rec, width):
    """THE RECEIPT: did W devices actually do work, or did one do all of it?"""
    peaks = _per_device_peak()
    rec["device_peaks"] = peaks
    live = [k for k, v in peaks.items() if v]
    rec["devices_with_work"] = len(live)
    if all(v is None for v in peaks.values()):
        rec["participation"] = "unreported by backend"
    elif len(live) == width:
        rec["participation"] = "ok"
    else:
        rec["participation"] = (
            f"FAILED: {len(live)} device(s) did work at width {width}")
    return rec


def _devices_for(width):
    import jax

    devs = jax.devices()
    if len(devs) < width:
        raise RuntimeError(
            f"width {width} asked for on a backend with {len(devs)} device(s). "
            "A width the machine cannot realize is not a measurement; run with "
            "four GPUs visible, or --xla_force_host_platform_device_count=4.")
    return devs[:width]


def _timed_with_setup(setup, body, reps):
    """Time `body` only. `setup` runs between reps and is NOT counted."""
    ts, setup_ts = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        state = setup()
        setup_ts.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        body(state)
        ts.append(time.perf_counter() - t0)
    return dict(reps=reps, median_s=float(np.median(ts)), min_s=float(min(ts)),
                max_s=float(max(ts)), all_s=[float(t) for t in ts],
                setup_median_s=float(np.median(setup_ts)))


# ===========================================================================
# the arms, each in its own process
# ===========================================================================


def arm_roundtrip(n, slab, plane_batch, pencil_batch, seed):
    """The stack receipt: 512^3 must read 4.053e-06 for the fourth time."""
    from inexor import ooc_fft

    rec = dict(arm="roundtrip", n=n, width=1, slab=slab,
               plane_batch=plane_batch, pencil_batch=pencil_batch)
    try:
        rec.update(ooc_fft.roundtrip_residual(
            n, np.float32, seed=seed, slab=slab, plane_batch=plane_batch,
            pencil_batch=pencil_batch, device=True))
        rec["completed"] = True
    except Exception as exc:
        rec.update(completed=False, error=f"{type(exc).__name__}: {exc}")
    return _participation(rec, 1)


def arm_identity(n, slab, plane_batch, pencil_batch, width, seed):
    """W devices == 1 device, BITWISE, at the size being timed.

    Not a tolerance: partitioning a loop cannot change the arithmetic. ULPs are
    recorded rather than a bare pass, because a bitwise bar can pass by luck and
    a count of zero says nothing about how close it came.
    """
    from inexor import ooc_fft

    rec = dict(arm="identity", n=n, width=width, slab=slab,
               plane_batch=plane_batch, pencil_batch=pencil_batch)
    try:
        devs = _devices_for(width)
        one = ooc_fft.plane_noise(n, 0, np.float32, seed)

        def field(lo, hi):
            return np.broadcast_to(one, (hi - lo, n, n))

        kw = dict(slab=slab, plane_batch=plane_batch, pencil_batch=pencil_batch)
        ref = ooc_fft.forward_from_slabs_device(field, n, **kw)
        got = ooc_fft.forward_from_slabs_device(field, n, devices=devs, **kw)
        rec["dtype_ok"] = str(got.dtype) == str(ref.dtype)
        rv, gv = ref.view(np.float32), got.view(np.float32)
        diff = rv != gv
        rec["n_diff_forward"] = int(diff.sum())
        rec["n_total_forward"] = int(rv.size)
        if rec["n_diff_forward"]:
            rec["max_abs_delta_forward"] = float(np.max(np.abs(rv - gv)))
            rec["max_ulp_forward"] = int(np.max(np.abs(
                rv[diff].view(np.int32) - gv[diff].view(np.int32))))
        # the inverse, one slab deep: the whole field at 2048^3 is 34.4 GB per arm
        a = next(iter(ooc_fft.inverse_to_slabs_device(ref.copy(), n, **kw)))[1]
        b = next(iter(ooc_fft.inverse_to_slabs_device(
            got.copy(), n, devices=devs, **kw)))[1]
        rec["n_diff_inverse"] = int((a != b).sum())
        rec["n_total_inverse"] = int(a.size)
        rec["bitwise"] = (rec["n_diff_forward"] == 0
                          and rec["n_diff_inverse"] == 0 and rec["dtype_ok"])
        rec["completed"] = True
    except Exception as exc:
        rec.update(completed=False, error=f"{type(exc).__name__}: {exc}")
    return _participation(rec, width)


def arm_timing(n, slab, plane_batch, pencil_batch, width, seed, reps):
    """The wall at width W. Flat source: host RNG does not split and is 52 s."""
    from inexor import ooc_fft

    rec = dict(arm="timing", n=n, width=width, slab=slab,
               plane_batch=plane_batch, pencil_batch=pencil_batch, source="flat")
    try:
        devs = _devices_for(width)
        one = ooc_fft.plane_noise(n, 0, np.float32, seed)

        def field(lo, hi):
            return np.broadcast_to(one, (hi - lo, n, n))

        kw = dict(slab=slab, plane_batch=plane_batch, pencil_batch=pencil_batch,
                  devices=devs)

        t0 = time.perf_counter()
        spec = ooc_fft.forward_from_slabs_device(field, n, **kw)
        rec["first_fwd_s"] = time.perf_counter() - t0  # carries the compile
        rec["fwd"] = _timed_with_setup(
            lambda: None, lambda _: ooc_fft.forward_from_slabs_device(field, n, **kw),
            reps)

        # The 34.4 GB host copy D1 timed INSIDE the inverse, measured on its own.
        t0 = time.perf_counter()
        _scratch = spec.copy()
        rec["spec_copy_s"] = time.perf_counter() - t0
        rec["spec_gb"] = float(spec.nbytes / 1e9)
        del _scratch

        def body(s):
            for _lo, _out in ooc_fft.inverse_to_slabs_device(s, n, **kw):
                pass

        rec["inv"] = _timed_with_setup(spec.copy, body, reps)
        rec["per_step_s"] = rec["fwd"]["median_s"] + 3 * rec["inv"]["median_s"]
        rec["per_step_frac_of_bar"] = rec["per_step_s"] / BAR_S
        # what the same step costs if the unsplittable host copy is charged too
        rec["per_step_s_with_copy"] = rec["per_step_s"] + 3 * rec["spec_copy_s"]
        rec["completed"] = True
    except Exception as exc:
        rec.update(completed=False, error=f"{type(exc).__name__}: {exc}",
                   oom="MEMORY" in str(exc).upper() or "RESOURCE_EXHAUSTED" in str(exc))
    return _participation(rec, width)


def arm_transfer(n, mode, width, reps, seed=0):
    """The proxy: one transform's TRAFFIC and none of its arithmetic, split W ways.

    Strong scaling -- the total is one transform's traffic however wide it runs,
    because the design charges a quarter of a fixed transform. The unit stays
    one 16.8 MB plane, which is what the factorization moves; a rate quoted at a
    2 GiB chunk would be a different code path's best case (sec. 9).
    """
    import jax
    from concurrent.futures import ThreadPoolExecutor

    from inexor import ooc_fft

    rec = dict(arm="transfer", n=n, width=width, mode=mode, driver="threads")
    try:
        devs = _devices_for(width)
        rec["memory_kinds"] = sorted({m.kind for m in devs[0].addressable_memories()})
    except Exception as exc:
        rec.update(completed=False, error=f"{type(exc).__name__}: {exc}")
        return rec

    rng = np.random.default_rng(seed)
    real = np.ascontiguousarray(rng.standard_normal((n, n)), dtype=np.float32)
    spec = np.ascontiguousarray(
        (rng.standard_normal((n, n // 2 + 1))
         + 1j * rng.standard_normal((n, n // 2 + 1))).astype(np.complex64))
    # One transform's traffic: 2n plane-sized H2D and 2n plane-sized D2H.
    units = [real, spec, spec, spec]
    rec["bytes_total"] = int(sum(a.nbytes for a in units) * n)
    rec["planes"] = n

    parts = ooc_fft.partition_units(n, width, 1)
    rec["parts"] = [list(p) for p in parts]

    if mode == "pinned":
        try:
            staged = []
            for d in devs:
                hs = jax.sharding.SingleDeviceSharding(d, memory_kind="pinned_host")
                staged.append(([jax.device_put(a, hs) for a in units],
                               jax.sharding.SingleDeviceSharding(d)))
            rec["host_kind_used"] = "pinned_host"
        except Exception as exc:
            rec.update(completed=False,
                       error=f"pinned_host unavailable: {type(exc).__name__}: {exc}")
            return _participation(rec, width)

        def work(k, lo, hi):
            arrs, ds = staged[k]
            hs = jax.sharding.SingleDeviceSharding(devs[k], memory_kind="pinned_host")
            for a in arrs:
                for _ in range(hi - lo):
                    jax.block_until_ready(jax.device_put(a, ds))
                    jax.block_until_ready(jax.device_put(a, hs))
    elif mode == "pageable":
        rec["host_kind_used"] = "numpy (pageable)"

        def work(k, lo, hi):
            d = devs[k]
            for a in units:
                for _ in range(hi - lo):
                    jax.block_until_ready(jax.device_put(a, d))
                    np.asarray(jax.device_put(a, d))
    else:
        raise ValueError(f"unknown transfer mode {mode!r}")

    def once():
        if width == 1:
            work(0, parts[0][0], parts[0][1])
            return
        with ThreadPoolExecutor(max_workers=width) as ex:
            futs = [ex.submit(work, k, lo, hi) for k, (lo, hi) in enumerate(parts)]
            for f in futs:
                f.result()

    try:
        once()  # warm every device's compile/plan before timing
        rec["move"] = _timed_with_setup(lambda: None, lambda _: once(), reps)
        rec["gbps"] = rec["bytes_total"] / 1e9 / rec["move"]["median_s"]
        rec["completed"] = True
    except Exception as exc:
        rec.update(completed=False, error=f"{type(exc).__name__}: {exc}")
    return _participation(rec, width)


def _barrier(bdir, rank, width, timeout=600.0):
    """W processes start timing together. Without this, jax init skew (~30 s)
    would be the whole measurement."""
    os.makedirs(bdir, exist_ok=True)
    with open(os.path.join(bdir, f"ready_{rank}"), "w") as fh:
        fh.write("1")
    t0 = time.time()
    while time.time() - t0 < timeout:
        if len([f for f in os.listdir(bdir) if f.startswith("ready_")]) >= width:
            return True
        time.sleep(0.05)
    raise RuntimeError(f"barrier timed out at rank {rank}: not all {width} arrived")


def arm_transfer_child(n, mode, width, rank, reps, bdir, seed=0):
    """One process's slice of the process-driver arm. Prints its own card."""
    import jax

    from inexor import ooc_fft

    rec = dict(arm="transfer-child", n=n, width=width, mode=mode, rank=rank,
               driver="processes")
    try:
        dev = jax.devices()[0]  # one visible device per child, by CUDA_VISIBLE_DEVICES
        rec["device"] = str(dev)
        rec["n_visible"] = len(jax.devices())
        rng = np.random.default_rng(seed)
        real = np.ascontiguousarray(rng.standard_normal((n, n)), dtype=np.float32)
        spec = np.ascontiguousarray(
            (rng.standard_normal((n, n // 2 + 1))
             + 1j * rng.standard_normal((n, n // 2 + 1))).astype(np.complex64))
        units = [real, spec, spec, spec]
        lo, hi = ooc_fft.partition_units(n, width, 1)[rank]
        rec["part"] = [lo, hi]
        rec["bytes"] = int(sum(a.nbytes for a in units) * (hi - lo))

        if mode == "pinned":
            hs = jax.sharding.SingleDeviceSharding(dev, memory_kind="pinned_host")
            ds = jax.sharding.SingleDeviceSharding(dev)
            staged = [jax.device_put(a, hs) for a in units]
            rec["host_kind_used"] = "pinned_host"

            def work():
                for a in staged:
                    for _ in range(hi - lo):
                        jax.block_until_ready(jax.device_put(a, ds))
                        jax.block_until_ready(jax.device_put(a, hs))
        else:
            rec["host_kind_used"] = "numpy (pageable)"

            def work():
                for a in units:
                    for _ in range(hi - lo):
                        jax.block_until_ready(jax.device_put(a, dev))
                        np.asarray(jax.device_put(a, dev))

        work()  # warm before the barrier, so compile is not inside the window
        _barrier(bdir, rank, width)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            work()
            ts.append(time.perf_counter() - t0)
        rec["median_s"] = float(np.median(ts))
        rec["all_s"] = [float(t) for t in ts]
        rec["gbps"] = rec["bytes"] / 1e9 / rec["median_s"]
        rec["completed"] = True
    except Exception as exc:
        rec.update(completed=False, error=f"{type(exc).__name__}: {exc}")
    with open(os.path.join(bdir, f"card_{rank}.json"), "w") as fh:
        json.dump(rec, fh)
    return rec


def arm_transfer_processes(args, n, mode, width, reps):
    """THE CONTROL for the threaded driver: same traffic, W processes.

    If this beats the threaded arm materially, the GIL is serializing dispatch
    and the engine's structure is a design constraint rather than an
    implementation detail. If they agree, the driver is not the variable and the
    threaded number is the node's answer.
    """
    import tempfile

    rec = dict(arm="transfer", n=n, width=width, mode=mode, driver="processes")
    bdir = tempfile.mkdtemp(prefix=f"d5_barrier_w{width}_{mode}_")
    rec["barrier_dir"] = bdir
    procs = []
    try:
        t0 = time.perf_counter()
        for rank in range(width):
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(rank)
            cmd = [sys.executable, os.path.abspath(__file__), "--single",
                   "--arm", "transfer-child", "--n", str(n), "--mode", mode,
                   "--width", str(width), "--rank", str(rank),
                   "--reps", str(reps), "--barrier-dir", bdir]
            procs.append(subprocess.Popen(cmd, env=env,
                                          stdout=subprocess.DEVNULL,
                                          stderr=subprocess.PIPE))
        errs = []
        for p in procs:
            _, err = p.communicate()
            if p.returncode != 0:
                errs.append((p.returncode, (err or b"").decode()[-600:]))
        rec["wall_including_startup_s"] = time.perf_counter() - t0
        cards = []
        for rank in range(width):
            path = os.path.join(bdir, f"card_{rank}.json")
            if os.path.exists(path):
                with open(path) as fh:
                    cards.append(json.load(fh))
        rec["children"] = cards
        done = [c for c in cards if c.get("completed")]
        if len(done) != width or errs:
            rec.update(completed=False,
                       error=f"{len(done)}/{width} children completed; {errs}")
            return rec
        # The wall is the SLOWEST child: they start together and the transform
        # is not done until every part is.
        rec["median_s"] = max(c["median_s"] for c in done)
        rec["bytes_total"] = sum(c["bytes"] for c in done)
        rec["gbps"] = rec["bytes_total"] / 1e9 / rec["median_s"]
        rec["per_child_gbps"] = [c["gbps"] for c in done]
        rec["completed"] = True
    except Exception as exc:
        rec.update(completed=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    return rec


# ===========================================================================
# orchestration
# ===========================================================================


def spawn(args, arm, n, width, mode="pageable", extra=None):
    """Each arm in its own process: device peak high-water never resets."""
    cmd = [sys.executable, os.path.abspath(__file__), "--single",
           "--arm", arm, "--n", str(n), "--width", str(width), "--mode", mode,
           "--slab", str(args.slab), "--plane-batch", str(args.plane_batch),
           "--pencil-batch", str(args.pencil_batch), "--reps", str(args.reps),
           "--seed", str(args.seed)]
    cmd += list(extra or [])
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    for line in (p.stdout or "").splitlines():
        if line.startswith("__REC__"):
            rec = json.loads(line[len("__REC__"):])
            rec["subprocess_wall_s"] = time.perf_counter() - t0
            return rec
    return dict(arm=arm, n=n, width=width, mode=mode, completed=False,
                error=f"no record on stdout (rc={p.returncode})",
                stderr=(p.stderr or "")[-1200:])


def _line(rec):
    """ONE line per leg, streamed. A report that only prints at the end does not
    survive the failure it measures (job 923313)."""
    head = f"[{rec.get('arm'):>15}] n={rec.get('n')} W={rec.get('width')}"
    if rec.get("mode") and rec.get("arm") == "transfer":
        head += f" {rec['mode']:>8}/{rec.get('driver', '?')}"
    if not rec.get("completed"):
        return head + f"  FAILED: {str(rec.get('error'))[:150]}"
    bits = []
    if "max_abs" in rec:
        bits.append(f"roundtrip={rec['max_abs'] / rec['rms']:.3e}"
                    if rec.get("rms") else f"max_abs={rec['max_abs']:.3e}")
    if "bitwise" in rec:
        bits.append(f"bitwise={rec['bitwise']} "
                    f"n_diff={rec.get('n_diff_forward')}/{rec.get('n_diff_inverse')}")
    if "inv" in rec:
        bits.append(f"fwd={rec['fwd']['median_s']:.2f}s "
                    f"inv={rec['inv']['median_s']:.2f}s "
                    f"copy={rec.get('spec_copy_s', float('nan')):.2f}s "
                    f"step={rec['per_step_s']:.1f}s "
                    f"({100 * rec['per_step_frac_of_bar']:.1f}% of bar)")
    if "gbps" in rec:
        bits.append(f"{rec['gbps']:.1f} GB/s "
                    f"({rec.get('median_s', rec.get('move', {}).get('median_s', 0)):.2f}s)")
    if rec.get("participation") and rec["participation"] != "ok":
        bits.append(f"PARTICIPATION {rec['participation']}")
    return head + "  " + "  ".join(bits)


def _speedups(cards, key):
    """T(1)/T(W) per arm family, the number the design's quarter-charge needs."""
    out = {}
    for c in cards:
        if not c.get("completed"):
            continue
        fam = (c.get("arm"), c.get("mode") if c.get("arm") == "transfer" else None,
               c.get("driver"))
        t = c.get(key) or (c.get("move") or {}).get("median_s") \
            or (c.get("inv") or {}).get("median_s")
        if t:
            out.setdefault(fam, {})[c["width"]] = t
    rows = {}
    for fam, byw in out.items():
        if 1 in byw:
            rows["/".join(str(x) for x in fam if x)] = {
                f"W={w}": round(byw[1] / byw[w], 3) for w in sorted(byw)}
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2048,
                    help="the size the coarse solve needs (default 2048)")
    ap.add_argument("--receipt-n", type=int, default=512,
                    help="cheap stack receipt; must reproduce 4.053e-06")
    ap.add_argument("--widths", default="1,2,4",
                    help="device counts, strong scaling at fixed total work")
    ap.add_argument("--slab", type=int, default=32)
    ap.add_argument("--plane-batch", type=int, default=1)
    ap.add_argument("--pencil-batch", type=int, default=64,
                    help="D1's measured best on Grace (1.10x over 1)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-processes", action="store_true",
                    help="skip the process-driver control")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny sizes, every arm, for the laptop")
    ap.add_argument("--out-suffix", default="")
    # single-arm subprocess entry
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--arm", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--width", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--mode", default="pageable",
                    choices=("pageable", "pinned"), help=argparse.SUPPRESS)
    ap.add_argument("--rank", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--barrier-dir", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.single:
        if args.arm == "roundtrip":
            rec = arm_roundtrip(args.n, args.slab, args.plane_batch,
                                args.pencil_batch, args.seed)
        elif args.arm == "identity":
            rec = arm_identity(args.n, args.slab, args.plane_batch,
                               args.pencil_batch, args.width, args.seed)
        elif args.arm == "timing":
            rec = arm_timing(args.n, args.slab, args.plane_batch,
                             args.pencil_batch, args.width, args.seed, args.reps)
        elif args.arm == "transfer":
            rec = arm_transfer(args.n, args.mode, args.width, args.reps, args.seed)
        elif args.arm == "transfer-child":
            rec = arm_transfer_child(args.n, args.mode, args.width, args.rank,
                                     args.reps, args.barrier_dir, args.seed)
        else:
            raise SystemExit(f"unknown arm {args.arm!r}")
        print("__REC__" + json.dumps(rec))
        return 0

    if args.smoke:
        args.n, args.receipt_n = 64, 32
        args.slab, args.pencil_batch, args.reps = 16, 4, 1

    widths = [int(w) for w in args.widths.split(",") if w.strip()]
    os.makedirs(OUT_DIR, exist_ok=True)
    card_path = os.path.join(OUT_DIR, f"d5_four_gpu{args.out_suffix}.json")

    prov = None
    try:
        prov = _provenance()
    except Exception as exc:
        print(f"provenance unavailable: {exc}")
    print(f"=== D5 four-GPU reading: n={args.n} widths={widths} "
          f"slab={args.slab} pb={args.plane_batch} yb={args.pencil_batch} "
          f"reps={args.reps} ===")
    if prov:
        print(f"    {prov['host']}  jax {prov['jax']}  {prov['platform']}  "
              f"{prov['n_devices']} device(s)  CUDA_VISIBLE={prov['cuda_visible']}")
    if prov and prov["n_devices"] < max(widths):
        print(f"    !! only {prov['n_devices']} device(s) visible: widths above "
              f"that will REFUSE rather than silently run narrow")

    cards = []

    def record(rec):
        cards.append(rec)
        print(_line(rec), flush=True)
        # streamed: a card written only at the end dies with the job
        with open(card_path, "w") as fh:
            json.dump(dict(provenance=prov, args=vars(args), cards=cards), fh,
                      indent=1)

    record(spawn(args, "roundtrip", args.receipt_n, 1))
    for w in widths:
        record(spawn(args, "identity", args.n, w))
    for w in widths:
        record(spawn(args, "timing", args.n, w))
    for mode in ("pageable", "pinned"):
        for w in widths:
            record(spawn(args, "transfer", args.n, w, mode=mode))
    if not args.no_processes:
        for mode in ("pageable", "pinned"):
            w = max(widths)
            if w > 1:
                record(arm_transfer_processes(args, args.n, mode, w, args.reps))

    print("\n=== strong-scaling speedup, T(1)/T(W) ===")
    for fam, row in _speedups(cards, "median_s").items():
        print(f"  {fam:>28}: {row}")

    # THE DRIVER CONTROL: same traffic, same width, threads vs processes. If
    # these disagree the GIL is the variable, not the node.
    print("\n=== driver control at W=%d, threads vs processes ===" % max(widths))
    for mode in ("pageable", "pinned"):
        got = {}
        for c in cards:
            if (c.get("arm") == "transfer" and c.get("mode") == mode
                    and c.get("width") == max(widths) and c.get("completed")):
                got[c.get("driver")] = c.get("gbps")
        if len(got) == 2:
            ratio = got["processes"] / got["threads"] if got["threads"] else float("nan")
            print(f"  {mode:>8}: threads {got['threads']:.1f} GB/s  "
                  f"processes {got['processes']:.1f} GB/s  "
                  f"processes/threads = {ratio:.2f}x")
        else:
            print(f"  {mode:>8}: incomplete ({sorted(got)})")

    bad = [c for c in cards if not c.get("completed")]
    part = [c for c in cards if c.get("participation") not in (None, "ok",
                                                               "unreported by backend")]
    print(f"\n=== {len(cards) - len(bad)}/{len(cards)} legs completed; "
          f"{len(part)} participation failure(s); card -> {card_path} ===")
    for c in part:
        print(f"  !! {_line(c)}")
    # A width that only one device served is not a narrower measurement, it is a
    # WRONG one, so it fails the job rather than printing quietly.
    return 1 if (bad or part) else 0


if __name__ == "__main__":
    raise SystemExit(main())
