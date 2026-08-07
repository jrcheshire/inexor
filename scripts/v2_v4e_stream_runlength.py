"""V4e: does a TILE-shaped gather reach the sequential streaming rate?

WHY THIS EXISTS. D-v2-13 clause 1 measured host-resident state streaming at
359-367 GB/s, and the whole V4 plan to kill the tiled force's host plumbing
(75.7 ms `stage` + 57.9 ms `scatter` against 47.8 ms of device work at the C-gh
candidate geometry) assumes a brick-sorted layout gets close to that rate. But
G4 measured a SEQUENTIAL stream of 2 GiB chunks, and a tile does not read
sequentially: it reads the union of its bricks, which at C-gh T=256/b=32 is
~1000 brick spans of ~4096 particles each, i.e. runs of order 37 KB (9 B/p) to
100 KB (24 B/p). Nothing measured says a 37 KB run reaches the same rate as a
2 GiB one. If it does not, the redesign wins far less than the arithmetic
suggests and C-gh stays host-bound.

THE TWO ARMS ARE THE REAL DESIGN FORK.
  assemble : gather the runs into ONE contiguous staging buffer on the host,
             then a SINGLE device_put of the whole buffer. Transfer
             granularity is the buffer (~37 MB), not the run.
  per_run  : device_put each run separately. Transfer granularity IS the run.
The per-run arm is the design most likely to miss the measured rate, and it is
also the one you get by accident if you write the obvious loop.

PRE-REGISTERED READING (written before the first rung ran).
  Predicted: `assemble` is roughly flat in run length above ~64 KB and lands
  within ~3x of G4's 359 GB/s; `per_run` falls off sharply below ~1 MB runs.
  FALSIFIED IF: `assemble` at 37 KB runs is below ~40 GB/s, i.e. under ~10% of
  sequential. Then the brick-sorted redesign buys ~10x on `stage` rather than
  the ~20x the arithmetic allows, and the C-gh cost model needs redoing.
  This is NOT a gate. No bar is tested and nothing here is ratified.

It also does double duty as the only cost input to `n_brick`, which
v2_g5_core.choose_brick currently picks on divisibility alone: longer runs mean
a better rate but coarser bricks, and this curve is the trade.

Usage:
    python scripts/v2_v4e_stream_runlength.py --src-gib 8 --stage-mib 64
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
OUT_DIR = os.path.join(REPO, "runs", "v2")

# Run lengths in BYTES. 37 KB is the C-gh candidate's brick span at T9 (9 B/p x
# 4096 particles per brick); 98 KB is the same span at f64 (24 B/p); the tails
# bracket it by ~8x each way so the curve's shape is visible, not just a point.
RUN_BYTES = (4096, 16384, 37888, 98304, 262144, 1048576, 4194304)


def run_leg(src_gib, stage_mib, run_bytes, arm, reps=3):
    sys.path.insert(0, os.path.join(REPO, "src"))
    import jax

    dev = jax.devices()[0]
    dev_sh = jax.sharding.SingleDeviceSharding(dev)

    src_n = int(src_gib * 1024**3)
    stage_n = int(stage_mib * 1024**2)
    run_n = int(run_bytes)
    n_runs = stage_n // run_n
    if n_runs < 1:
        return dict(arm=arm, run_bytes=run_n, skipped="stage buffer smaller than one run")
    stage_n = n_runs * run_n  # exact multiple, so the arms move the same bytes

    # The source stands in for the host-resident state: big enough that the
    # gather misses cache the way a 77 GB state would, and touched once so the
    # pages are really resident before anything is timed.
    src = np.empty(src_n, dtype=np.uint8)
    src[::4096] = 1
    rng = np.random.default_rng(0)
    # Run starts in ASCENDING order: a brick-sorted layout reads a tile's spans
    # in increasing address order, so a random permutation would measure a
    # pattern the design does not produce.
    offs = np.sort(rng.integers(0, src_n - run_n, size=n_runs))

    stage = np.empty(stage_n, dtype=np.uint8)

    gather_s, h2d_s = [], []
    for _ in range(reps):
        if arm == "assemble":
            t0 = time.perf_counter()
            for i, o in enumerate(offs):
                np.copyto(stage[i * run_n : (i + 1) * run_n], src[o : o + run_n])
            t1 = time.perf_counter()
            out = jax.device_put(stage, dev_sh)
            jax.block_until_ready(out)
            t2 = time.perf_counter()
            del out
        elif arm == "per_run":
            t0 = time.perf_counter()
            t1 = t0  # no separate gather phase: each run transfers on its own
            parts = []
            for o in offs:
                p = jax.device_put(src[o : o + run_n], dev_sh)
                parts.append(p)
            jax.block_until_ready(parts[-1])
            t2 = time.perf_counter()
            del parts
        else:
            raise ValueError(arm)
        gather_s.append(t1 - t0)
        h2d_s.append(t2 - t1)

    # min-of-N, not mean: we want the achievable rate, and the tail is noise
    # from other tenants on the node (umbrella reference-kill-the-noise).
    g = float(np.min(gather_s))
    h = float(np.min(h2d_s))
    tot = g + h
    gb = stage_n / 1024.0**3
    return dict(
        arm=arm,
        run_bytes=run_n,
        n_runs=int(n_runs),
        bytes=int(stage_n),
        gather_s=g,
        h2d_s=h,
        total_s=tot,
        gather_gbs=(None if g <= 0 else gb / g),
        h2d_gbs=(None if h <= 0 else gb / h),
        effective_gbs=(None if tot <= 0 else gb / tot),
        platform=dev.platform,
    )


def spawn(args, run_bytes, arm):
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--single",
        "--src-gib",
        str(args.src_gib),
        "--stage-mib",
        str(args.stage_mib),
        "--run-bytes",
        str(run_bytes),
        "--arm",
        arm,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("WORKER_JSON "):
            return json.loads(line[len("WORKER_JSON ") :])
    return dict(arm=arm, run_bytes=run_bytes, failed=True, tail=(p.stderr or p.stdout)[-500:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-gib", type=float, default=8.0)
    ap.add_argument("--stage-mib", type=float, default=64.0)
    ap.add_argument("--tag", default="v4e")
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--run-bytes", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--arm", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.single:
        print(
            "WORKER_JSON "
            + json.dumps(run_leg(args.src_gib, args.stage_mib, args.run_bytes, args.arm))
        )
        return

    recs = []
    print(f"=== V4e run-length sweep (src {args.src_gib} GiB, stage {args.stage_mib} MiB) ===")
    print(f"{'run':>10s} {'n_runs':>7s} {'arm':>9s} {'gather':>10s} {'H2D':>10s} {'eff':>10s}")
    for rb in RUN_BYTES:
        for arm in ("assemble", "per_run"):
            r = spawn(args, rb, arm)
            recs.append(r)
            if r.get("failed") or r.get("skipped"):
                print(f"{rb:10d} {'':>7s} {arm:>9s}  {r.get('skipped') or 'FAILED'}", flush=True)
                continue
            gg = r["gather_gbs"]
            print(
                f"{rb:10d} {r['n_runs']:7d} {arm:>9s} "
                f"{('--' if gg is None else f'{gg:8.1f}'):>10s} "
                f"{r['h2d_gbs']:10.1f} {r['effective_gbs']:10.1f}   GB/s",
                flush=True,
            )

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"v4e_stream_runlength_{args.tag}.json")
    with open(path, "w") as f:
        json.dump(
            dict(
                what="V4e: achieved gather+H2D rate vs run length, tile-shaped access",
                reference="D-v2-13 clause 1 measured 359-367 GB/s on a 2 GiB SEQUENTIAL stream",
                pre_registered=(
                    "assemble is flat above ~64 KB and within ~3x of 359 GB/s; per_run "
                    "falls off below ~1 MB. FALSIFIED if assemble at 37 KB is below "
                    "~40 GB/s (< ~10% of sequential)."
                ),
                note="NOT a gate: no bar is tested and nothing here is ratified.",
                legs=recs,
            ),
            f,
            indent=2,
        )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
