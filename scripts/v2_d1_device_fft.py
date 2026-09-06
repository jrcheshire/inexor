"""D1: the plane-factorized device FFT, measured at the size the coarse solve needs.

WHAT THIS DECIDES. The coarse solve at 4096^3 is a 2048^3 transform. Three
forms exist and two are already disqualified: the monolithic device transform
does not fit (peak/field 8.0x, 275 GB against a GB200's 185 GiB) and at 1536^3
it is silently WRONG (record 5y finding 2); the host out-of-core form runs but
costs 417 s/step, 39% of a gb node's 1080 s per-step budget on its own. This
job measures the third -- `ooc_fft.forward_from_slabs_device` -- against the
0.4 s/step the gb probe projected for it from scaled n^3 log n.

Expect the projection to be optimistic and expect the reason to be transfer,
not compute: a 2048^3 f32 spectrum is 34.4 GB and every plane crosses the bus
twice per pass. The probe's eject arm read 14 ms on device against 0.528 s end
to end for exactly this reason. Both numbers are recorded separately here so
the gap is attributable rather than a surprise.

THREE WITNESSES, none of which is the wall.

1. CORRECTNESS AT SIZE. The roundtrip receipt (`ooc_fft.roundtrip_residual`)
   at each n. This is the instrument that catches the silent-wrong-transform
   class, and it is the reason the ladder includes 1024 as well as 2048: 1024
   is the size that read 2.9e-6 on the bad stack, so a 1024 leg that reads
   anything else says the stack moved under us, not that the code is wrong.
2. RESIDENCY. Device `peak_bytes_in_use` must stay O(plane), not O(field). A
   peak near 34 GB means something materialized the spectrum on the device and
   the whole memory argument for the design is void -- the same witness, and
   the same failure mode, as the streaming arm's `pk/set` ratio.
3. THE RATIO'S OTHER LEG, IN THIS JOB. The host out-of-core form runs here at
   the same n on the same node. Its 107.4 / 101.4 s is on record from job
   972737 and could have been imported -- but an imported constant is how a
   within-run digest ends up comparing two machines, so it runs.

Inputs are plane-keyed white noise (`ooc_fft.plane_noise`), so the field is
independent of the streaming decomposition and no leg is comparing two
different fields.

Run (Vista gb or gh node, gpu env, one GPU):
    CUDA_VISIBLE_DEVICES=0 pixi run -e gpu python scripts/v2_d1_device_fft.py --out-suffix _gb
Laptop smoke (CPU backend; exercises every path incl. the card write):
    pixi run python scripts/v2_d1_device_fft.py --smoke --out-suffix _smoke
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
ERR_DIR = os.path.join(OUT_DIR, "d1_fft_errs")

DEFAULT_SIZES = "512,1024,2048"
SMOKE_SIZES = "32,64"
DEFAULT_BATCHES = "1,4,16"
DEFAULT_PENCILS = "1,8,64,256"

# The bar this is measured against, DERIVED: gb MaxWall 12 h / K=40 steps.
# The coarse solve is four transforms per step (one forward, three inverse).
WALL_S = 12 * 3600.0
TRANSFORMS_PER_STEP = 4


def _provenance():
    import jax

    dev = jax.devices()[0]
    return dict(
        host=socket.gethostname(), python=platform.python_version(),
        jax=jax.__version__, platform=dev.platform,
        device=str(dev), n_devices=len(jax.devices()),
        cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
        host_mem_limit_gb=os.environ.get("XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB"),
        preallocate=os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        argv=" ".join(sys.argv),
    )


def _peak_bytes():
    """Device high-water, or None on a backend that does not report one.

    Never resets, which is why every arm here runs in its own process.
    """
    import jax

    try:
        return int(jax.devices()[0].memory_stats().get("peak_bytes_in_use"))
    except Exception:
        return None


def _timed(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return dict(reps=reps, median_s=float(np.median(ts)),
                min_s=float(min(ts)), max_s=float(max(ts)),
                all_s=[float(t) for t in ts])


# ===========================================================================
# the arms, each in its own process
# ===========================================================================


def arm_roundtrip(n, slab, plane_batch, pencil_batch, device, seed):
    """Witness 1 + 2: does it come back, and did it stay off the device."""
    from inexor import ooc_fft

    rec = dict(arm="roundtrip", n=n, device=bool(device), slab=slab,
               plane_batch=plane_batch, pencil_batch=pencil_batch)
    t0 = time.perf_counter()
    try:
        r = ooc_fft.roundtrip_residual(
            n, np.float32, seed=seed, slab=slab, plane_batch=plane_batch,
            pencil_batch=pencil_batch, device=bool(device))
    except Exception as exc:  # an OOM that reports itself is a RESULT
        rec.update(completed=False, error=f"{type(exc).__name__}: {exc}",
                   oom="MEMORY" in str(exc).upper() or "RESOURCE_EXHAUSTED" in str(exc),
                   wall_s=time.perf_counter() - t0,
                   peak_bytes_in_use=_peak_bytes() if device else None)
        return rec
    rec.update(r)
    rec.update(completed=True, wall_s=time.perf_counter() - t0,
               peak_bytes_in_use=_peak_bytes() if device else None)
    # RESIDENCY, as a ratio rather than a number to eyeball. The spectrum is
    # n*n*(n//2+1) complex64; a peak of that order means it was materialized on
    # the device and the design's memory argument is void.
    spec_bytes = n * n * (n // 2 + 1) * 8
    rec["spec_bytes"] = spec_bytes
    if rec["peak_bytes_in_use"]:
        rec["peak_over_spec"] = rec["peak_bytes_in_use"] / spec_bytes
    return rec


def arm_noise(n, slab, seed, reps):
    """Time FIELD GENERATION alone, because in 974643 it was inside the forward.

    That job's forward leg read 77.8 s against its own inverse's 31.3 s, and the
    only thing the forward has that the inverse does not is `plane_noise` for
    every plane -- 8.6e9 gaussians at 2048^3. The device and host legs put the
    excess at 46.5 and 44.1 s, agreeing to 5% as they must if it is the shared
    host RNG. That is a strong inference and it is still an inference; this arm
    makes it a reading.
    """
    from inexor import ooc_fft

    def gen():
        for lo in range(0, n, slab):
            hi = min(lo + slab, n)
            np.stack([ooc_fft.plane_noise(n, i, np.float32, seed)
                      for i in range(lo, hi)])

    rec = dict(arm="noise", n=n, device=False, slab=slab, plane_batch=1,
               pencil_batch=1, source="noise")
    rec["gen"] = _timed(gen, reps)
    rec["completed"] = True
    return rec


def arm_transfer(n, mode, reps, seed=0):
    """ATTRIBUTE the transform's wall: is it the bus, at the unit the FFT uses?

    974807 left ~31 s per transform unexplained. Pass 2's strided access is
    bounded at ~11% by its own ladder, so the leading suspect is that every
    transfer in this path is PAGEABLE -- numpy in, `jnp.asarray`, `np.asarray`
    out -- against the 201 GB/s this hardware streams from PINNED host (5z).
    One transform moves ~137.5 GB, which at 31.3 s is 4.4 GB/s: a 46x gap.

    This arm does the transform's TRAFFIC and none of its arithmetic, so what
    comes back is the bus and not the FFT. Both modes move identical bytes in
    identical units and differ only in the memory kind, and they run in ONE job
    on ONE node -- 974807's flat-source leg changed the data AND the node at
    once and its 1.41x is unattributable because of it.

    The unit is one plane, 16.8 MB, because that is what the factorization
    actually moves; a bandwidth quoted at a 2 GiB chunk would be measuring a
    different code path's best case.
    """
    import jax
    import jax.numpy as jnp

    dev = jax.devices()[0]
    kinds = []
    try:
        kinds = sorted(dev.memory_stats() and [] or [])
    except Exception:
        pass
    try:
        kinds = sorted({m.kind for m in dev.addressable_memories()})
    except Exception:
        kinds = []

    rec = dict(arm="transfer", n=n, device=True, mode=mode, plane_batch=1,
               pencil_batch=1, source="-", memory_kinds=kinds)

    real = np.ascontiguousarray(
        np.random.default_rng(seed).standard_normal((n, n)), dtype=np.float32)
    spec = np.ascontiguousarray(
        (np.random.default_rng(seed + 1).standard_normal((n, n // 2 + 1))
         + 1j * np.random.default_rng(seed + 2).standard_normal((n, n // 2 + 1)))
        .astype(np.complex64))
    # One transform's traffic, in the units the factorization moves it in:
    # 2n plane-sized H2D and 2n plane-sized D2H (module docstring / record 8).
    units = [real, spec, spec, spec]
    bytes_per_pass = sum(a.nbytes for a in units) * n
    rec["planes"] = n
    rec["bytes_total"] = int(bytes_per_pass)

    if mode == "pinned":
        try:
            host_kind = "pinned_host"
            hs = jax.sharding.SingleDeviceSharding(dev, memory_kind=host_kind)
            ds = jax.sharding.SingleDeviceSharding(dev)
            staged = [jax.device_put(a, hs) for a in units]
            rec["host_kind_used"] = host_kind
        except Exception as exc:
            rec.update(completed=False,
                       error=f"pinned_host unavailable: {type(exc).__name__}: {exc}")
            return rec

        def once():
            for a in staged:
                for _ in range(n):
                    d = jax.device_put(a, ds)
                    jax.block_until_ready(d)
                    b = jax.device_put(d, hs)
                    jax.block_until_ready(b)
    elif mode == "pageable":
        rec["host_kind_used"] = "numpy (pageable)"

        def once():
            for a in units:
                for _ in range(n):
                    d = jnp.asarray(a)
                    jax.block_until_ready(d)
                    np.asarray(d)
    else:
        raise ValueError(f"unknown transfer mode {mode!r}")

    try:
        once_small = None
        rec["move"] = _timed(once, reps)
        rec["completed"] = True
        rec["gbps"] = bytes_per_pass / 1e9 / rec["move"]["median_s"]
        _ = once_small
    except Exception as exc:
        rec.update(completed=False, error=f"{type(exc).__name__}: {exc}",
                   oom="MEMORY" in str(exc).upper())
    rec["peak_bytes_in_use"] = _peak_bytes()
    return rec


def arm_timing(n, slab, plane_batch, pencil_batch, device, seed, reps,
               source="noise"):
    """The wall, forward and inverse separately, at one batch setting.

    `source` selects what the forward transforms. "noise" regenerates plane-keyed
    white noise per plane and is what a correctness run wants; "flat" reuses ONE
    pre-drawn plane for every x and exists only to take the host RNG out of the
    forward leg, since an FFT's cost is data-independent but its input's
    GENERATION is not. The card records which was used -- a wall quoted without
    it is not comparable to one quoted with it, which is the defect this
    argument was added to fix.
    """
    from inexor import ooc_fft

    rec = dict(arm="timing", n=n, device=bool(device), slab=slab,
               plane_batch=plane_batch, pencil_batch=pencil_batch, source=source)

    if source == "flat":
        one = ooc_fft.plane_noise(n, 0, np.float32, seed)

        def field(lo, hi):
            return np.broadcast_to(one, (hi - lo, n, n))
    elif source == "noise":
        def field(lo, hi):
            return np.stack([ooc_fft.plane_noise(n, i, np.float32, seed)
                             for i in range(lo, hi)])
    else:
        raise ValueError(f"unknown source {source!r}")

    kw = dict(slab=slab)
    if device:
        kw.update(plane_batch=plane_batch, pencil_batch=pencil_batch)
        fwd_fn = ooc_fft.forward_from_slabs_device
        inv_fn = ooc_fft.inverse_to_slabs_device
    else:
        fwd_fn = ooc_fft.forward_from_slabs
        inv_fn = ooc_fft.inverse_to_slabs
    try:
        t0 = time.perf_counter()
        spec = fwd_fn(field, n, **kw)
        rec["first_fwd_s"] = time.perf_counter() - t0  # carries the compile
        rec["fwd"] = _timed(lambda: fwd_fn(field, n, **kw), reps)

        def one_inverse():
            s = spec.copy()
            for _lo, _out in inv_fn(s, n, **kw):
                pass

        rec["inv"] = _timed(one_inverse, reps)
        rec["completed"] = True
        # What the coarse solve actually pays: one forward + three inverses.
        rec["per_step_s"] = rec["fwd"]["median_s"] + 3 * rec["inv"]["median_s"]
    except Exception as exc:
        rec.update(completed=False, error=f"{type(exc).__name__}: {exc}",
                   oom="MEMORY" in str(exc).upper() or "RESOURCE_EXHAUSTED" in str(exc))
    rec["peak_bytes_in_use"] = _peak_bytes() if device else None
    return rec


# ===========================================================================
# orchestration
# ===========================================================================


def spawn(args, arm, n, plane_batch, device, pencil_batch=None, source="noise",
          mode="pageable"):
    pencil_batch = args.pencil_batch if pencil_batch is None else pencil_batch
    cmd = [sys.executable, os.path.abspath(__file__), "--single",
           "--arm", arm, "--n", str(n), "--slab", str(args.slab),
           "--plane-batch", str(plane_batch),
           "--pencil-batch", str(pencil_batch), "--source", source,
           "--mode", mode,
           "--seed", str(args.seed), "--reps", str(args.reps)]
    if not device:
        cmd.append("--host-path")
    tag = (f"{arm}_{n}_p{plane_batch}_y{pencil_batch}_{source}_{mode}"
           f"_{'dev' if device else 'host'}")
    print(f"[worker] {tag} ...", flush=True)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("WORKER_JSON ")]
    if p.stderr.strip():
        os.makedirs(ERR_DIR, exist_ok=True)
        with open(os.path.join(ERR_DIR, f"{tag}.err"), "w") as fh:
            fh.write(p.stderr)
    if not lines:
        return dict(arm=arm, n=n, plane_batch=plane_batch, device=device,
                    completed=False, died_without_report=True,
                    returncode=p.returncode, worker_wall_s=wall,
                    stderr_tail=p.stderr.strip()[-400:])
    rec = json.loads(lines[-1][len("WORKER_JSON "):])
    rec["worker_wall_s"] = wall
    rec["returncode"] = p.returncode
    return rec


def _line(rec):
    """ONE line per reading, printed as it lands.

    A digest that only prints at the end does not survive the failure it
    measures -- this project has lost four jobs' worth of readings to that.
    """
    who = "dev " if rec.get("device") else "host"
    tag = (f"{rec['arm']:9s} n={rec.get('n'):>5} {who} "
           f"p={rec.get('plane_batch'):>3} y={rec.get('pencil_batch'):>3} "
           f"{rec.get('source', '-'):5s}")
    if not rec.get("completed"):
        why = "OOM" if rec.get("oom") else ("DIED" if rec.get("died_without_report")
                                            else "ERR")
        return f"{tag}  {why}: {str(rec.get('error') or rec.get('stderr_tail',''))[:70]}"
    if rec["arm"] == "transfer":
        if not rec.get("completed"):
            return f"{tag}  {rec.get('mode')}: {str(rec.get('error'))[:60]}"
        return (f"{tag}  {rec['mode']:8s} {rec['bytes_total'] / 1e9:7.1f} GB in "
                f"{rec['move']['median_s']:7.2f} s = {rec['gbps']:6.1f} GB/s "
                f"[{rec.get('host_kind_used')}]")
    if rec["arm"] == "noise":
        return f"{tag}  field generation alone {rec['gen']['median_s']:8.3f} s"
    if rec["arm"] == "roundtrip":
        pk = rec.get("peak_bytes_in_use")
        pk_s = f"{pk / 1024**3:7.2f} GiB" if pk else "      n/a"
        ratio = rec.get("peak_over_spec")
        return (f"{tag}  residual {rec['residual']:.2e}  rms {rec['rms']:.3f}  "
                f"peak {pk_s}" + (f"  = {ratio:.3f} x spec" if ratio else ""))
    return (f"{tag}  fwd {rec['fwd']['median_s']:8.3f} s  "
            f"inv {rec['inv']['median_s']:8.3f} s  "
            f"per step (1 fwd + 3 inv) {rec['per_step_s']:8.3f} s  "
            f"first fwd {rec['first_fwd_s']:7.3f} s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", default=None,
                    help=f"mesh sizes, comma separated (default {DEFAULT_SIZES})")
    ap.add_argument("--batches", default=None,
                    help=f"plane_batch ladder (default {DEFAULT_BATCHES})")
    ap.add_argument("--slab", type=int, default=32)
    ap.add_argument("--pencil-batch", type=int, default=1)
    ap.add_argument("--pencil-batches", default=None,
                    help="pass-2 batch ladder. 974643 varied plane_batch (pass "
                         "1) over 1/4/16 and read 77.78/77.53/77.76 s -- a dead "
                         "knob -- while THIS one stayed pinned at 1 in every "
                         "leg, and pass 2 is the strided one: it reads "
                         "spec[:, y, :] across the whole 34.4 GB spectrum 2048 "
                         "times. An axis nothing varied is not an axis anything "
                         "was learned about.")
    ap.add_argument("--mode", default="pageable", choices=("pageable", "pinned"),
                    help=argparse.SUPPRESS)
    ap.add_argument("--transfer", action="store_true",
                    help="run the pageable-vs-pinned transfer A/B that "
                         "attributes the transform's wall to the bus or not")
    ap.add_argument("--source", default="noise", choices=("noise", "flat"),
                    help="what the forward transforms; 'flat' takes the host "
                         "RNG out of the timed region")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timing-n", type=int, default=None,
                    help="mesh size for the batch ladder and the host leg "
                         "(default: the largest that ROUNDTRIPPED)")
    ap.add_argument("--no-host-leg", action="store_true",
                    help="skip the host out-of-core comparison. NOT the default: "
                         "a ratio whose other leg is imported from another job "
                         "compares two machines and still prints as within-run.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-suffix", default="")
    # worker mode
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--arm", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--n", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--plane-batch", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--host-path", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.single:
        if args.arm == "roundtrip":
            rec = arm_roundtrip(args.n, args.slab, args.plane_batch,
                                args.pencil_batch, not args.host_path, args.seed)
        elif args.arm == "timing":
            rec = arm_timing(args.n, args.slab, args.plane_batch,
                             args.pencil_batch, not args.host_path, args.seed,
                             args.reps, source=args.source)
        elif args.arm == "noise":
            rec = arm_noise(args.n, args.slab, args.seed, args.reps)
        elif args.arm == "transfer":
            rec = arm_transfer(args.n, args.mode, args.reps, args.seed)
        else:
            raise ValueError(args.arm)
        print("WORKER_JSON " + json.dumps(rec), flush=True)
        return 0

    sizes = [int(s) for s in (args.sizes or
                              (SMOKE_SIZES if args.smoke else DEFAULT_SIZES)).split(",")]
    batches = [int(b) for b in (args.batches or DEFAULT_BATCHES).split(",")]
    pencils = [int(y) for y in (args.pencil_batches or DEFAULT_PENCILS).split(",")]
    if args.smoke:
        args.reps = min(args.reps, 2)
        batches, pencils = batches[:2], pencils[:2]

    print(f"D1: the plane-factorized device FFT. sizes={sizes} "
          f"plane_batches={batches} pencil_batches={pencils} "
          f"source={args.source} slab={args.slab} reps={args.reps}", flush=True)
    print(f"the bar: {WALL_S / 3600:.0f} h wall / 40 steps = "
          f"{WALL_S / 40:.0f} s per step; the coarse solve is "
          f"{TRANSFORMS_PER_STEP} transforms of it", flush=True)

    recs = []

    # --- witness 1 + 2, the correctness/residency ladder
    for n in sizes:
        rec = spawn(args, "roundtrip", n, 1, device=True)
        recs.append(rec)
        print(_line(rec), flush=True)

    ok = [r["n"] for r in recs if r["arm"] == "roundtrip" and r.get("completed")]
    timing_n = args.timing_n or (max(ok) if ok else None)
    if timing_n is None:
        print("no size roundtripped: NOTHING IS TIMED. A wall for a transform "
              "that did not come back is not a reading.", flush=True)
    else:
        # --- the bus, both memory kinds, same node, same job
        if args.transfer:
            for m in ("pageable", "pinned"):
                rec = spawn(args, "transfer", timing_n, 1, device=True, mode=m)
                recs.append(rec)
                print(_line(rec), flush=True)
        # --- field generation alone: the term 974643 timed inside its forward
        rec = spawn(args, "noise", timing_n, 1, device=False)
        recs.append(rec)
        print(_line(rec), flush=True)
        # --- the wall over pass 1's knob, then over pass 2's
        for b in batches:
            rec = spawn(args, "timing", timing_n, b, device=True,
                        source=args.source)
            recs.append(rec)
            print(_line(rec), flush=True)
        for y in pencils:
            rec = spawn(args, "timing", timing_n, batches[0], device=True,
                        pencil_batch=y, source=args.source)
            recs.append(rec)
            print(_line(rec), flush=True)
        # --- the ratio's other leg, in THIS job
        if not args.no_host_leg:
            rec = spawn(args, "timing", timing_n, 1, device=False,
                        source=args.source)
            recs.append(rec)
            print(_line(rec), flush=True)

    # --- the verdict, withheld when a load-bearing leg is missing
    print("\nAGAINST THE BAR", flush=True)
    dev = [r for r in recs if r["arm"] == "timing" and r.get("device")
           and r.get("completed")]
    host = [r for r in recs if r["arm"] == "timing" and not r.get("device")
            and r.get("completed")]
    if not dev:
        print("  VERDICT WITHHELD: no device timing leg completed.")
    else:
        best = min(dev, key=lambda r: r["per_step_s"])
        print(f"  best device coarse solve: {best['per_step_s']:.3f} s/step at "
              f"plane_batch={best['plane_batch']} (n={best['n']}), "
              f"{best['per_step_s'] / (WALL_S / 40) * 100:.2f}% of the bar")
        if host:
            h = host[0]
            print(f"  host out-of-core, same n, this job: {h['per_step_s']:.3f} "
                  f"s/step -> device is {h['per_step_s'] / best['per_step_s']:.1f}x")
        else:
            print("  no host leg in this job, so NO RATIO is printed: the "
                  "107.4/101.4 s on record is another job's node.")
        if best["n"] != 2048:
            print(f"  NB the timed size is {best['n']}^3, not the 2048^3 the "
                  "coarse solve runs. This is not a 4096^3 reading.")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"d1_device_fft{args.out_suffix}.json")
    with open(path, "w") as fh:
        json.dump(dict(sizes=sizes, batches=batches, slab=args.slab,
                       pencils=pencils, source=args.source,
                       reps=args.reps, bar_s_per_step=WALL_S / 40,
                       arms=recs, provenance=_provenance()), fh, indent=1)
    print(f"wrote {path}", flush=True)

    # A self-reported OOM is a result; dying without reporting is a failure.
    bad = [r for r in recs if r.get("died_without_report")]
    if bad:
        print(f"FAILED arms: {[(b['arm'], b.get('n')) for b in bad]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
