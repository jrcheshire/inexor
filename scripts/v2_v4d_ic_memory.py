"""V4d: what does IC generation actually cost, on the HOST and on the DEVICE?

WHY THIS EXISTS. The only IC memory number on record is 76 B/p (v1 R6,
docs/m2-results.md:212), and it is `dev.memory_stats()["peak_bytes_in_use"]` --
DEVICE ONLY. Reading src/inexor/ic.py while planning V4 turned up a host term
nobody has ever measured:

    ic.py:49-58 builds |k| as a full (N, N, N//2+1) float64 numpy array and
    hands it to linear_power -> transfer_eh98 (cosmology.py:161), which
    materializes ~15 more arrays of that size plus expression temporaries.

At N=2048 one such array is 34.4 GB, against a ~116 GB host cliff (D-v2-13
clause 2). poisson_factor (ic.py:81) repeats the pattern. If that reading is
right, IC generation is the binding term at C-gh and it is nowhere in the
capacity table. If it is wrong, the IC problem is device-only and V4's IC ADR
shrinks accordingly. Either way the freeze should not rest on my arithmetic.

PRE-REGISTERED READING (written before the first rung ran).
  Predicted: host B/p GROWS with N (the transfer temporaries are ~15 half-grid
  f64 arrays = ~60 B/p on their own) and dominates device B/p at every rung;
  device B/p lands near the recorded 76 at the `full` phase.
  FALSIFIED IF: host B/p is flat and small (< ~20) across the ladder, or does
  not exceed device B/p. Then the host term is not real and only Tier-2 (the
  six simultaneous (N^3,3) arrays at lpt.py:153-158) matters.
  This is NOT a gate. No bar is tested and nothing here is ratified.

METHOD.
  * ONE SUBPROCESS PER (rung, phase). `peak_bytes_in_use` is a running max that
    never resets (umbrella reference-jax-peak-memory-no-reset), and
    ru_maxrss is a monotone high-water mark, so neither can be bracketed
    WITHIN a process. Phases are therefore CUMULATIVE PREFIXES, each run from a
    cold interpreter; the per-phase increment is the difference between
    adjacent prefixes.
  * ru_maxrss for the host, NOT a sampled RSS: a high-water mark cannot miss a
    transient, and every array of interest here is a transient. (There is a
    sampler at v2_g5_two_level_force.py:168; it is the wrong instrument for
    this measurement.)
  * Cheapest rung first, and the caller must not use `set -e`, so a top-rung
    OOM still yields the lower rungs.

The `fft` mode is a separate question that shares the plumbing: what is the
largest monolithic jnp.fft.rfftn that fits on this card? If 2048^3 fits, C-gh
needs no out-of-core FFT at all and that whole layer defers to C-hero.

Usage:
    python scripts/v2_v4d_ic_memory.py --mode ic  --rungs 256,512,1024
    python scripts/v2_v4d_ic_memory.py --mode fft --rungs 512,1024,1536,2048
"""

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
OUT_DIR = os.path.join(REPO, "runs", "v2")

# Cumulative prefixes of the IC path. Each is a superset of the one before, so
# the peak is monotone in this order and the increments are readable.
PHASES = ("colour", "linear_density", "psi1", "delta2", "full")


def _maxrss_bytes():
    """Host high-water mark. ru_maxrss is KB on Linux, BYTES on macOS."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(ru) if platform.system() == "Darwin" else int(ru) * 1024


def _device_peak(jax):
    try:
        dev = jax.devices()[0]
        if dev.platform != "gpu":
            return None, dev.platform
        return (dev.memory_stats() or {}).get("peak_bytes_in_use"), dev.platform
    except Exception:
        return None, "unknown"


# ===========================================================================
# worker
# ===========================================================================


def run_ic_phase(n, phase, seed=0):
    """Run the IC path up to `phase` and report the two high-water marks."""
    sys.path.insert(0, os.path.join(REPO, "src"))
    import jax
    import jax.numpy as jnp

    from inexor import ic, lpt
    from inexor.config import PLANCK

    base_host = _maxrss_bytes()
    n_p = float(n) ** 3
    key = jax.random.PRNGKey(seed)
    t0 = time.perf_counter()

    # NB fdtype f32 throughout: the production IC dtype. The host f64 term
    # under test is INDEPENDENT of fdtype -- it lives in the numpy colour /
    # transfer evaluation, which is float64 by design (the precision island).
    if phase == "colour":
        out = ic.gaussian_delta(key, n, float(n) * 0.5, PLANCK, fdtype=jnp.float32)
    elif phase == "linear_density":
        out = ic.linear_density(key, n, float(n) * 0.5, PLANCK, f_NL=0.0, fdtype=jnp.float32)
    else:
        d0 = ic.linear_density(key, n, float(n) * 0.5, PLANCK, f_NL=0.0, fdtype=jnp.float32)
        if phase == "psi1":
            out = lpt.zeldovich_displacement(d0, float(n) * 0.5, jnp.float32)
        elif phase == "delta2":
            out = lpt.lpt2_source(d0, float(n) * 0.5, jnp.float32)
        elif phase == "full":
            out = lpt.lpt_ics(d0, float(n) * 0.5, 0.1, PLANCK, order=2, fdtype=jnp.float32)
        else:
            raise ValueError(f"unknown phase {phase!r}")

    jax.block_until_ready(out)
    wall = time.perf_counter() - t0
    dev_peak, plat = _device_peak(jax)
    host_peak = _maxrss_bytes()
    del out

    return dict(
        mode="ic",
        n=int(n),
        phase=phase,
        wall_s=float(wall),
        platform=plat,
        host_peak_bytes=host_peak,
        host_base_bytes=base_host,
        host_bpp=float(host_peak) / n_p,
        # THE NUMBER TO READ. host_bpp carries the interpreter + jax import
        # baseline (~0.33 GB), which is 0.3 B/p at n=1024 and 1300 B/p at n=64.
        # Subtracting it makes the low rungs legible; at the rungs that decide
        # anything the two agree.
        host_delta_bpp=float(host_peak - base_host) / n_p,
        # ONLY MEANINGFUL ON GPU. On a CPU backend jax's own arrays land in host
        # RAM too, so host_peak conflates the numpy f64 transfer temporaries
        # under test with the field arrays that would live on the device. A CPU
        # run tests the plumbing, not the question.
        host_isolates_numpy=(plat == "gpu"),
        device_peak_bytes=dev_peak,
        device_bpp=(None if dev_peak is None else float(dev_peak) / n_p),
    )


def run_fft(n):
    """Largest monolithic rfftn that fits: this rung either completes or dies."""
    sys.path.insert(0, os.path.join(REPO, "src"))
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (n, n, n), dtype=jnp.float32)
    jax.block_until_ready(x)
    t0 = time.perf_counter()
    xk = jnp.fft.rfftn(x)
    jax.block_until_ready(xk)
    t_fwd = time.perf_counter() - t0
    t0 = time.perf_counter()
    xr = jnp.fft.irfftn(xk, s=(n, n, n))
    jax.block_until_ready(xr)
    t_inv = time.perf_counter() - t0
    dev_peak, plat = _device_peak(jax)
    return dict(
        mode="fft",
        n=int(n),
        platform=plat,
        fwd_s=float(t_fwd),
        inv_s=float(t_inv),
        host_peak_bytes=_maxrss_bytes(),
        device_peak_bytes=dev_peak,
        field_gb=float(4 * n**3) / 1024**3,
    )


# ===========================================================================
# orchestrator
# ===========================================================================


def spawn(mode, n, phase=None):
    cmd = [sys.executable, os.path.abspath(__file__), "--single", "--mode", mode, "--n", str(n)]
    if phase:
        cmd += ["--phase", phase]
    p = subprocess.run(cmd, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("WORKER_JSON "):
            return json.loads(line[len("WORKER_JSON ") :])
    return dict(
        mode=mode,
        n=int(n),
        phase=phase,
        failed=True,
        returncode=p.returncode,
        tail=(p.stderr or p.stdout)[-600:],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ic", choices=["ic", "fft"])
    ap.add_argument("--rungs", default="256,512,1024")
    ap.add_argument("--tag", default="v4d")
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--n", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--phase", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.single:
        rec = run_ic_phase(args.n, args.phase) if args.mode == "ic" else run_fft(args.n)
        print("WORKER_JSON " + json.dumps(rec))
        return

    rungs = [int(v) for v in args.rungs.split(",")]
    recs = []
    print(f"=== V4d {args.mode}: {rungs} ===", flush=True)

    for n in rungs:
        phases = PHASES if args.mode == "ic" else (None,)
        died = False
        for ph in phases:
            r = spawn(args.mode, n, ph)
            recs.append(r)
            if r.get("failed"):
                died = True
                print(f"  n={n:5d} {str(ph):15s} FAILED rc={r['returncode']}", flush=True)
                print(f"      {r['tail'].splitlines()[-1][:160] if r['tail'] else ''}", flush=True)
                break
            if args.mode == "ic":
                dev = r["device_bpp"]
                dev_s = "     --" if dev is None else f"{dev:7.1f}"
                warn = "" if r.get("host_isolates_numpy") else "  [CPU: host not isolated]"
                print(
                    f"  n={n:5d} {ph:15s} host {r['host_peak_bytes'] / 1024**3:8.2f} GB "
                    f"({r['host_delta_bpp']:7.1f} B/p net)  device {dev_s} B/p  "
                    f"{r['wall_s']:6.2f} s{warn}",
                    flush=True,
                )
            else:
                print(
                    f"  n={n:5d} field {r['field_gb']:7.2f} GB  fwd {r['fwd_s']:7.3f} s  "
                    f"inv {r['inv_s']:7.3f} s  peak "
                    f"{(r['device_peak_bytes'] or 0) / 1024**3:7.2f} GB",
                    flush=True,
                )
        if died and args.mode == "fft":
            print(f"  -> largest monolithic rfftn that fits is BELOW {n}", flush=True)
            break

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"v4d_ic_memory_{args.tag}_{args.mode}.json")
    with open(path, "w") as f:
        json.dump(
            dict(
                what="V4d: IC memory anatomy (host AND device) / monolithic FFT capacity",
                mode=args.mode,
                pre_registered=(
                    "host B/p grows with N and dominates device B/p; device lands "
                    "near the recorded 76 at `full`. FALSIFIED if host B/p is flat "
                    "and < ~20, or never exceeds device."
                ),
                note="NOT a gate: no bar is tested and nothing here is ratified.",
                legs=recs,
            ),
            f,
            indent=2,
        )
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
