"""v2 G4b: node capability + fp64 health, for the C-hero / gb architecture call.

Two unknowns block V4's node-ladder decision and neither is written down
anywhere in this repo:

  1. WHAT IS ON A gb NODE. Vista's GB200 nodes are believed to carry ~192 GB
     of HBM per card, 4 cards, 2 Grace CPUs (144 cores), and an unrecorded
     amount of LPDDR. "Believed" is the problem: the 125-vs-48 GB precedent
     (pricing a CPU run with a device number) cost this project a dead job,
     so the numbers get measured. LPDDR matters most, because on a Grace
     part it is C2C-attached and therefore worth several times its size in
     PCIe-attached host memory.
  2. WHETHER GB200 fp64 IS USABLE. inexor's force and FFT path is f64
     throughout (`force_global` returns f64 whatever dtype the positions
     carry). Blackwell is widely reported to cut vector fp64 hard relative to
     Hopper. If so, capacity won on a gb node is handed straight back in
     wall, and that should be known BEFORE anyone contemplates the multi-GPU
     sharding a gb node would require.

THE MEASUREMENT IS A RATIO, ON PURPOSE. Comparing absolute f64 GB/s or
GFLOP/s across two different machines invites every mismatch this project has
already been bitten by (different clocks, different queues, different jaxlib
builds). So the headline is the WITHIN-MACHINE f64:f32 ratio for the same
op at the same shape. That is dimensionless, self-normalizing, and it is
exactly the quantity that exposes a crippled fp64 unit: a part with
full-rate fp64 lands near its architectural ratio, a part with throttled
fp64 lands far below it. The absolute numbers are recorded too, but the
ratio is what carries across the two nodes.

Ops chosen for relevance, not for peak numbers:
  gemm : clean FLOP-bound reference, the cleanest read on the fp64 unit.
  fft  : 3-D complex FFT, which is what `force_global` actually spends its
         time in, and is bandwidth- rather than FLOP-bound -- so a part can
         look fine on one and bad on the other. Both are reported.

Run (either node type, gpu env):
    pixi run -e gpu python scripts/v2_g4b_fp64_capability.py
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
OUT_DIR = os.path.join(REPO, "runs", "v2")


def capability():
    """Everything about the node we currently have to guess at."""
    import jax

    devs = jax.devices()
    info = dict(
        jax_version=jax.__version__,
        n_devices=len(devs),
        platform=devs[0].platform if devs else None,
        device_kind=devs[0].device_kind if devs else None,
        host_cpu_count=os.cpu_count(),
        uname=platform.platform(),
    )
    per_dev = []
    for d in devs:
        try:
            st = d.memory_stats() or {}
        except Exception:
            st = {}
        per_dev.append(
            dict(
                id=d.id,
                kind=d.device_kind,
                bytes_limit=st.get("bytes_limit"),
                memory_kinds=[m.kind for m in d.addressable_memories()],
            )
        )
    info["devices"] = per_dev

    # host memory: the number that decides whether gb can stream at all
    try:
        with open("/proc/meminfo") as fh:
            mi = {k.strip(): v.strip() for k, v in (ln.split(":", 1) for ln in fh)}
        info["mem_total_kb"] = int(mi.get("MemTotal", "0 kB").split()[0])
        info["mem_available_kb"] = int(mi.get("MemAvailable", "0 kB").split()[0])
    except Exception as exc:
        info["meminfo_error"] = repr(exc)

    # NUMA layout: on a 2x Grace part this is how the LPDDR is actually split,
    # and a single-process run may only reach one node's worth
    for cmd, key in (
        (["numactl", "--hardware"], "numactl"),
        (["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], "nvidia_smi"),
    ):
        try:
            info[key] = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception as exc:
            info[key] = f"unavailable: {exc!r}"
    return info


def _time_op(fn, reps):
    import jax

    jax.block_until_ready(fn())  # warmup + compile, never timed
    walls = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        walls.append(time.perf_counter() - t0)
    return float(np.median(walls)), [float(w) for w in walls]


def bench_gemm(n, dtype, reps):
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (n, n), dtype=jnp.float32).astype(dtype)
    b = jax.random.normal(key, (n, n), dtype=jnp.float32).astype(dtype)
    f = jax.jit(lambda x, y: x @ y)
    wall, walls = _time_op(lambda: f(a, b), reps)
    return dict(op="gemm", n=n, dtype=str(dtype.__name__), wall_median_s=wall,
                wall_all_s=walls, gflops=(2.0 * n**3 / wall) / 1e9)


def bench_fft(n, dtype, reps):
    """3-D complex FFT -- what force_global spends its time in."""
    import jax
    import jax.numpy as jnp

    cdtype = jnp.complex128 if dtype is np.float64 else jnp.complex64
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n, n, n), dtype=jnp.float32).astype(cdtype)
    f = jax.jit(jnp.fft.fftn)
    wall, walls = _time_op(lambda: f(x), reps)
    bytes_touched = x.size * (16 if cdtype is jnp.complex128 else 8)
    return dict(op="fft3d", n=n, dtype=str(dtype.__name__), wall_median_s=wall,
                wall_all_s=walls, gbytes_per_s=(bytes_touched / wall) / 1e9,
                array_gib=bytes_touched / 1024**3)


def run_single(args):
    import jax

    jax.config.update("jax_enable_x64", True)  # or every f64 request silently becomes f32
    dtype = np.float64 if args.dtype == "f64" else np.float32
    try:
        rec = bench_gemm(args.n, dtype, args.reps) if args.op == "gemm" else bench_fft(
            args.n, dtype, args.reps
        )
        rec["completed"] = True
    except Exception as exc:
        txt = str(exc)
        rec = dict(op=args.op, n=args.n, dtype=args.dtype, completed=False,
                   error=txt.strip().splitlines()[0][:200] if txt.strip() else repr(exc),
                   oom=("RESOURCE_EXHAUSTED" in txt or "out of memory" in txt.lower()))
    print("WORKER_JSON " + json.dumps(rec))


def spawn(op, n, dtype, reps):
    p = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--single", "--op", op,
         "--n", str(n), "--dtype", dtype, "--reps", str(reps)],
        capture_output=True, text=True,
    )
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("WORKER_JSON ")]
    if lines:
        return json.loads(lines[-1][len("WORKER_JSON ") :])
    el = (p.stderr or "").strip().splitlines()
    return dict(op=op, n=n, dtype=dtype, completed=False,
                error=(el[-1] if el else f"exit {p.returncode}")[:200])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gemm-sizes", default="4096,8192")
    ap.add_argument("--fft-sizes", default="256,512")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--op", default="gemm", choices=("gemm", "fft"), help=argparse.SUPPRESS)
    ap.add_argument("--n", type=int, default=4096, help=argparse.SUPPRESS)
    ap.add_argument("--dtype", default="f64", choices=("f32", "f64"), help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.single:
        run_single(args)
        return

    print("=== G4b: node capability + fp64 health ===")
    cap = capability()
    print(json.dumps({k: v for k, v in cap.items() if k not in ("numactl",)}, indent=1))
    if cap.get("mem_total_kb"):
        print(
            f"\nhost memory: {cap['mem_total_kb'] / 1024**2:.1f} GiB total, "
            f"{cap['mem_available_kb'] / 1024**2:.1f} GiB available"
        )
    print("\n--- numactl --hardware ---")
    print(cap.get("numactl", "")[:1500])

    recs = []
    for op, sizes in (("gemm", args.gemm_sizes), ("fft", args.fft_sizes)):
        for n in [int(v) for v in sizes.split(",") if v]:
            for dtype in ("f32", "f64"):
                print(f"[worker] {op:5s} n={n:5d} {dtype} ...", flush=True)
                recs.append(spawn(op, n, dtype, args.reps))

    print(f"\n{'op':6s} {'n':>6s} {'f32':>12s} {'f64':>12s} {'f64:f32':>9s}   unit")
    ratios = {}
    for op in ("gemm", "fft3d"):
        for n in sorted({r["n"] for r in recs if r["op"] == op}):
            got = {r["dtype"]: r for r in recs if r["op"] == op and r["n"] == n}
            f32, f64 = got.get("float32"), got.get("float64")
            unit = "GFLOP/s" if op == "gemm" else "GB/s"
            key = "gflops" if op == "gemm" else "gbytes_per_s"
            if not (f32 and f64 and f32.get("completed") and f64.get("completed")):
                bad = [d for d, r in got.items() if not r.get("completed")]
                print(f"{op:6s} {n:>6d} {'incomplete':>12s} {'':>12s} {'':>9s}   {bad}")
                continue
            r32, r64 = f32[key], f64[key]
            ratios[f"{op}_{n}"] = r64 / r32
            print(f"{op:6s} {n:>6d} {r32:12.1f} {r64:12.1f} {r64 / r32:9.3f}   {unit}")

    print(
        "\nREAD THE RATIO, not the absolutes: it is dimensionless and survives\n"
        "the two node types differing in clocks, jaxlib build and queue. A part\n"
        "with full-rate fp64 sits near its architectural f64:f32 ratio; a part\n"
        "with throttled fp64 sits far below it. inexor's force and FFT path is\n"
        "f64 throughout, so the fft row is the one that bears on wall."
    )
    print("Nothing self-ratifies: the architecture call is JC's at V4.")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"g4b_capability{args.out_suffix}.json")
    with open(path, "w") as fh:
        json.dump(dict(capability=cap, configs=recs, ratios=ratios), fh, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
