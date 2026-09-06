"""M-v2-6 gb probe: price the host-state / device-step design before building it.

THE DESIGN UNDER TEST. The CPU engine is at its bandwidth floor (tile loop +
coarse paint = 84% of the step, both within 1.5-2x of gg's ceiling) and 4096^3
fits no single node's host at ANY worker count (1487-1636 GB against gb's
1026). The only reading of the memory-floor thesis that still reaches 4096^3
on one node is: the host is a byte store for the T9 state (~721 GB of gb's
1026), and the four Blackwell GPUs do EVERYTHING per step -- decode, tile
force, kick, quantize, migrate, coarse paint, coarse solve -- with slabs
streamed over C2C. The partial port measured 1.28x (Vista 912457) because the
host plumbing stayed on the host; this design only works if nothing per-step
stays there. gb's wall is 12 h, so the design must be FAST, not long: K=40 in
one job means <= 1080 s per step.

WHAT THIS MEASURES, on one GPU, each arm in a fresh subprocess (device peak
never resets -- umbrella reference-jax-peak-memory-no-reset):

  eject   `eject_jax.eject_rows` -- the compiled drift + global stable
          partition, i.e. the migrate's eject half -- on ONE 4096^3-scale
          slab of rows (268,435,456 at the c-hero preset; nb=256 slabs per
          step). Two numbers: end to end through the shipped entry point
          (numpy in, numpy out = H2D + kernel + D2H, which is what a host-
          resident state pays) and the bare jitted kernel on device-resident
          inputs. The insert half is NOT here; it is named as unmeasured.
  paint   `painting.paint_tsc_int` of one slab's rows into the 2048^3 int32
          coarse mesh (34.4 GB on device) and, separately, into the x-slab
          SUB-BLOCK the engine's chunked path uses (`paint_tsc_int_subblock`,
          (span+3) x N x N). The sub-block is the design's realistic form (a
          slab's particles only touch 8+3 planes of the mesh); the full mesh
          is the pessimistic bound. Mass equality between the two is the
          containment receipt.
  fft     `jnp.fft.rfftn` / `irfftn` on f32 device fields at 1024^3, 1536^3
          and 2048^3. The 2048^3 attempt is a DESIGNED OOM (the workspace
          measured 7x the field on a GH200, 240 GB against 184 GiB of HBM);
          the two that fit give the device rate and the workspace ratio on
          this jax/Blackwell so the coarse solve can be priced against the
          host out-of-core path (`v2_m5_fft_gh.py`, run beside this).

WHAT IT DOES NOT ESTABLISH: the insert, repack and kick on device; the
efficiency of a four-GPU split (the readout ASSUMES 4.0 and says so); host
bookkeeping that a built pipeline would still do; anything about wall on a
node that is not gb. It prices the FLOOR of the design, and any term over the
bar kills the design cleanly.

Inputs are synthetic and uniform (a perturbed lattice has no clustering, so
nothing here says anything about arena residency or migrant fraction -- the
eject arm sets a ~5% leaver fraction by choosing the drift, and records what
it got). Row work is what is priced; uniform rows price row work.

Run (Vista gb node, gpu env, one GPU):
    CUDA_VISIBLE_DEVICES=0 pixi run -e gpu python scripts/v2_m6_gb_probe.py --out-suffix _gb
Laptop smoke (CPU backend; exercises every path incl. the card write):
    pixi run python scripts/v2_m6_gb_probe.py --smoke --out-suffix _smoke
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
OUT_DIR = os.path.join(REPO, "runs", "v2")
ERR_DIR = os.path.join(OUT_DIR, "gb_probe_errs")

ARMS = ("eject", "paint", "fft")
DEFAULT_FFT_SIZES = "1024,1536,2048"
SMOKE_FFT_SIZES = "32,64"

# eject arm: the drift is chosen so a few percent of rows leave their brick.
# sigma of the per-axis displacement in position QUANTA; a brick is
# per * 256 quanta wide (per = buckets per brick side, 8 at c-hero), so
# 40 quanta is ~2% of a brick per axis, ~5% of rows leaving over three axes.
SIGMA_QUANTA = 40.0
W_SIGMA = 3000.0  # int16 velocity code sigma; scale is derived from it


# ===========================================================================
# geometry, from the planner's preset table (one definition, not a copy)
# ===========================================================================


def geometry(preset):
    from inexor.layout import choose_brick
    from inexor.plan import PRESETS

    g = dict(PRESETS[preset])
    brick = int(choose_brick(g["tile"], g["buf"], g["n_fine"]))
    nb = int(g["n_fine"]) // brick
    n_total = int(g["n_part"]) ** 3
    rows_per_brick = (int(g["n_part"]) // nb) ** 3
    return dict(
        preset=preset,
        n_part=int(g["n_part"]),
        box=float(g["box"]),
        n_fine=int(g["n_fine"]),
        n_coarse=int(g["n_coarse"]),
        tile=int(g["tile"]),
        buf=int(g["buf"]),
        brick_fine=brick,
        nb=nb,
        n_total=n_total,
        rows_per_brick=rows_per_brick,
        rows_per_slab=rows_per_brick * nb * nb,
        tiles=(int(g["n_fine"]) // int(g["tile"])) ** 3,
        padded=int(g["tile"]) + 2 * int(g["buf"]),
        slab_thickness=float(g["box"]) / nb,
        coarse_cells_per_slab=int(g["n_coarse"]) // nb,
    )


# ===========================================================================
# workers
# ===========================================================================


def _device_info():
    import jax

    dev = jax.devices()[0]
    stats = None
    try:
        stats = dev.memory_stats()
    except Exception:
        stats = None
    return dev, dict(
        platform=str(dev.platform),
        device_kind=str(dev.device_kind),
        n_devices_visible=len(jax.devices()),
        jax_version=jax.__version__,
        x64=bool(jax.config.jax_enable_x64),
        memory_stats_available=stats is not None,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )


def _peak_bytes(dev):
    try:
        s = dev.memory_stats()
    except Exception:
        return None
    if not s:
        return None
    return int(s.get("peak_bytes_in_use", 0)) or None


def _timed(fn, reps):
    walls = []
    for _ in range(int(reps)):
        t0 = time.perf_counter()
        fn()
        walls.append(time.perf_counter() - t0)
    return dict(median_s=float(np.median(walls)), all_s=[float(w) for w in walls])


def _rows_used(g, row_frac):
    n_bricks = g["nb"] * g["nb"]
    used = max(1, int(round(float(row_frac) * n_bricks)))
    return min(used, n_bricks)


def arm_eject(g, row_frac, reps, seed):
    import jax
    from inexor import eject_jax
    from inexor.codec import LEVELS_PER_BUCKET, T9Layout

    dev, info = _device_info()
    t9 = T9Layout(box_size=g["box"], n_part=g["n_part"])
    nb = g["nb"]
    per = t9.n_buckets_side // nb
    assert per * nb == t9.n_buckets_side, (per, nb, t9.n_buckets_side)

    n_bricks_used = _rows_used(g, row_frac)
    rpb = g["rows_per_brick"]
    n = n_bricks_used * rpb
    rng = np.random.default_rng(int(seed))

    # brick-major within slab bx = 0, exactly as `_eject_slab` concatenates
    bidx = np.repeat(np.arange(n_bricks_used, dtype=np.int64), rpb)
    by, bz = bidx // nb, bidx % nb
    bijk = np.empty((n, 3), dtype=np.int64)
    bijk[:, 0] = rng.integers(0, per, n)
    bijk[:, 1] = by * per + rng.integers(0, per, n)
    bijk[:, 2] = bz * per + rng.integers(0, per, n)
    off = rng.integers(0, LEVELS_PER_BUCKET, (n, 3)).astype(np.uint8)
    w = np.clip(rng.normal(0.0, W_SIGMA, (n, 3)), -32767, 32767).astype(np.int16)
    brick_id = (0 * nb + by) * nb + bz  # global flat brick id, bx = 0
    c_drift = 1.0
    scale = np.full((n, 1), SIGMA_QUANTA * t9.quantum / (W_SIGMA * c_drift), dtype=np.float64)
    del bidx, by, bz

    def end_to_end():
        return eject_jax.eject_rows(t9, nb, off, bijk, w, None, scale, c_drift, brick_id)

    t0 = time.perf_counter()
    out = end_to_end()  # compile + first run
    first_s = time.perf_counter() - t0
    n_keep = int(out[5])
    # receipt: the kernel ran and the partition is a partition
    assert len(out[0]) == n and 0 <= n_keep <= n, (len(out[0]), n_keep, n)
    del out
    e2e = _timed(end_to_end, reps)

    # the bare kernel on device-resident inputs: the cached jit for this shape
    n_pad = eject_jax._padded(n)
    key = (int(t9.n_buckets_side), float(t9.quantum), int(nb), n_pad, False)
    fn = eject_jax._CACHE[key]
    pad = n_pad - n

    def _pad(a, fill):
        if pad == 0:
            return a
        tail = np.full((pad,) + a.shape[1:], fill, dtype=a.dtype)
        return np.concatenate([a, tail])

    real = np.zeros(n_pad, dtype=bool)
    real[:n] = True
    d_off = jax.device_put(_pad(off, 0))
    d_bijk = jax.device_put(_pad(bijk, 0))
    d_w = jax.device_put(_pad(w, 0))
    d_scale = jax.device_put(_pad(scale, 1.0))
    d_brick = jax.device_put(_pad(brick_id, -1))
    d_real = jax.device_put(real)

    def device_only():
        res = fn(d_off, d_bijk, d_w, None, d_scale, c_drift, d_brick, d_real)
        jax.block_until_ready(res)
        return res

    res = device_only()
    n_keep_dev = int(res[5])
    del res
    dev_t = _timed(device_only, reps)

    return dict(
        arm="eject",
        completed=True,
        rows=n,
        n_pad=n_pad,
        bricks_used=n_bricks_used,
        bricks_per_slab=nb * nb,
        slab_fraction=n / g["rows_per_slab"],
        rows_per_slab=g["rows_per_slab"],
        nb=nb,
        buckets_per_brick_side=per,
        sigma_quanta=SIGMA_QUANTA,
        w_sigma=W_SIGMA,
        leave_fraction=1.0 - n_keep / n,
        leave_fraction_device_run=1.0 - n_keep_dev / n,
        first_call_s=first_s,
        end_to_end=e2e,
        device_only=dev_t,
        per_slab_end_to_end_s=e2e["median_s"] * g["rows_per_slab"] / n,
        per_slab_device_only_s=dev_t["median_s"] * g["rows_per_slab"] / n,
        peak_bytes_in_use=_peak_bytes(dev),
        **info,
    )


def arm_paint(g, row_frac, reps, seed):
    import jax
    import jax.numpy as jnp
    from inexor.painting import (
        check_tsc_paint_headroom,
        paint_tsc_int,
        paint_tsc_int_subblock,
    )

    dev, info = _device_info()
    N = g["n_coarse"]
    box = g["box"]
    frac_bits = 12
    n_bricks_used = _rows_used(g, row_frac)
    n = n_bricks_used * g["rows_per_brick"]
    rng = np.random.default_rng(int(seed) + 1)

    headroom = dict(ok=True, message=None)
    try:
        check_tsc_paint_headroom(g["n_total"], frac_bits)
    except ValueError as exc:
        headroom = dict(ok=False, message=str(exc)[:300])

    # one slab's rows: x inside slab 0, y and z anywhere
    pos = np.empty((n, 3), dtype=np.float64)
    pos[:, 0] = rng.random(n) * g["slab_thickness"]
    pos[:, 1] = rng.random(n) * box
    pos[:, 2] = rng.random(n) * box

    span = g["coarse_cells_per_slab"]
    if span + 3 >= N:
        origin, extent = (0, 0, 0), (N, N, N)
    else:
        origin, extent = ((0 - 1) % N, 0, 0), (span + 3, N, N)

    full = jax.jit(lambda p: paint_tsc_int(p, N, box, frac_bits))
    sub = jax.jit(lambda p: paint_tsc_int_subblock(p, origin, extent, N, box, frac_bits))

    d_pos = jax.device_put(pos)

    def run_full():
        m = full(d_pos)
        jax.block_until_ready(m)
        return m

    def run_sub():
        m = sub(d_pos)
        jax.block_until_ready(m)
        return m

    t0 = time.perf_counter()
    m_sub = run_sub()
    sub_first = time.perf_counter() - t0
    mass_sub = int(jnp.sum(m_sub.astype(jnp.int64)))
    del m_sub
    sub_t = _timed(run_sub, reps)
    peak_after_sub = _peak_bytes(dev)

    full_t = None
    full_first = None
    mass_full = None
    full_error = None
    try:
        t0 = time.perf_counter()
        m_full = run_full()
        full_first = time.perf_counter() - t0
        mass_full = int(jnp.sum(m_full.astype(jnp.int64)))
        del m_full
        full_t = _timed(run_full, reps)
    except Exception as exc:  # the full mesh is the pessimistic arm; an OOM is a report
        full_error = str(exc).strip().splitlines()[0][:300] if str(exc).strip() else repr(exc)

    return dict(
        arm="paint",
        completed=True,
        rows=n,
        slab_fraction=n / g["rows_per_slab"],
        rows_per_slab=g["rows_per_slab"],
        nb=g["nb"],
        n_coarse=N,
        frac_bits=frac_bits,
        headroom=headroom,
        subblock_origin=[int(o) for o in origin],
        subblock_extent=[int(e) for e in extent],
        subblock_first_s=sub_first,
        subblock=sub_t,
        subblock_mass=mass_sub,
        subblock_peak_bytes_in_use=peak_after_sub,
        full_first_s=full_first,
        full=full_t,
        full_mass=mass_full,
        full_error=full_error,
        full_oom=bool(full_error and ("RESOURCE_EXHAUSTED" in full_error
                                      or "out of memory" in full_error.lower())),
        mass_equal=(mass_full is None) or (mass_full == mass_sub),
        expected_mass=n * (2**frac_bits),  # 27 quantized corner weights per row sum to ~2^frac_bits
        per_slab_subblock_s=sub_t["median_s"] * g["rows_per_slab"] / n,
        per_slab_full_s=(full_t["median_s"] * g["rows_per_slab"] / n) if full_t else None,
        peak_bytes_in_use=_peak_bytes(dev),
        **info,
    )


def arm_fft(n, reps):
    import jax
    import jax.numpy as jnp

    dev, info = _device_info()
    n = int(n)
    field_bytes = 4 * n**3
    rec = dict(arm="fft", n=n, field_bytes=field_bytes, **info)
    try:
        a = jnp.arange(n, dtype=jnp.float32)
        field = (jnp.sin(a[:, None, None] * 0.37) * jnp.cos(a[None, :, None] * 0.11)
                 + 0.5 * jnp.sin(a[None, None, :] * 0.23)).astype(jnp.float32)
        field = jax.block_until_ready(field)
        assert field.shape == (n, n, n) and field.dtype == jnp.float32

        def fwd():
            s = jnp.fft.rfftn(field)
            jax.block_until_ready(s)
            return s

        t0 = time.perf_counter()
        spec = fwd()
        first_fwd = time.perf_counter() - t0
        fwd_t = _timed(fwd, reps)

        def inv(spec=spec):
            b = jnp.fft.irfftn(spec, s=(n, n, n))
            jax.block_until_ready(b)
            return b

        t0 = time.perf_counter()
        back = inv()
        first_inv = time.perf_counter() - t0
        inv_t = _timed(inv, reps)
        # the peak is read HERE, after the transforms and before the receipt
        # below allocates its own temporaries, so peak/field is the FFT's
        pk = _peak_bytes(dev)
        # roundtrip receipt: the transform is a transform, not a no-op
        err = float(jnp.max(jnp.abs(back - field)))
        rms = float(jnp.sqrt(jnp.mean(field.astype(jnp.float64) ** 2)))
        rec.update(
            completed=True,
            oom=False,
            first_fwd_s=first_fwd,
            first_inv_s=first_inv,
            fwd=fwd_t,
            inv=inv_t,
            roundtrip_max_abs_over_rms=err / rms if rms else None,
            peak_bytes_in_use=pk,
            peak_over_field=(pk / field_bytes) if pk else None,
        )
    except Exception as exc:
        txt = str(exc)
        rec.update(
            completed=False,
            error=txt.strip().splitlines()[0][:300] if txt.strip() else repr(exc),
            oom=("RESOURCE_EXHAUSTED" in txt or "out of memory" in txt.lower()),
            peak_bytes_in_use=_peak_bytes(dev),
        )
    return rec


def run_single(args):
    import jax

    # x64 is the caller's choice by this repo's convention; the eject kernel
    # refuses without it, and the paint's weights are f64 in the engine.
    jax.config.update("jax_enable_x64", True)
    if args.arm == "fft":
        rec = arm_fft(args.fft_n, args.reps)
    else:
        g = geometry(args.preset)
        if args.arm == "eject":
            rec = arm_eject(g, args.row_frac, args.reps, args.seed)
        elif args.arm == "paint":
            rec = arm_paint(g, args.row_frac, args.reps, args.seed)
        else:
            raise ValueError(args.arm)
    print("WORKER_JSON " + json.dumps(rec), flush=True)


# ===========================================================================
# orchestration
# ===========================================================================


def spawn(arm, args, fft_n=None):
    cmd = [sys.executable, os.path.abspath(__file__), "--single", "--arm", arm,
           "--preset", args.preset, "--row-frac", str(args.row_frac),
           "--reps", str(args.reps), "--seed", str(args.seed)]
    if fft_n is not None:
        cmd += ["--fft-n", str(fft_n)]
    tag = arm if fft_n is None else f"{arm}_{fft_n}"
    print(f"[worker] {tag} ...", flush=True)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("WORKER_JSON ")]
    if p.stderr.strip():
        os.makedirs(ERR_DIR, exist_ok=True)
        with open(os.path.join(ERR_DIR, f"{tag}.err"), "w") as fh:
            fh.write(p.stderr)
    if lines:
        rec = json.loads(lines[-1][len("WORKER_JSON "):])
        rec["worker_wall_s"] = wall
        rec["worker_rc"] = p.returncode
        return rec
    el = p.stderr.strip().splitlines()
    exc = [t for t in el if ("Error" in t or "Exception" in t)]
    return dict(
        arm=arm, n=fft_n, completed=False, died_without_report=True,
        error=(exc[-1] if exc else (el[-1] if el else f"exit {p.returncode}"))[:300],
        oom=any("RESOURCE_EXHAUSTED" in t or "out of memory" in t.lower() for t in el),
        worker_wall_s=wall, worker_rc=p.returncode,
    )


def _provenance():
    try:
        commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                         text=True).strip()
    except Exception:
        commit = None
    return dict(
        hostname=socket.gethostname(),
        commit=commit,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        jax_platforms_env=os.environ.get("JAX_PLATFORMS"),
        argv=sys.argv[1:],
        time=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def _fmt_s(v):
    return "-" if v is None else f"{v:9.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", default="c-hero", help="planner preset (geometry source)")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--fft-sizes", default=DEFAULT_FFT_SIZES)
    ap.add_argument("--row-frac", type=float, default=1.0,
                    help="fraction of a slab's bricks to synthesize (1.0 = a whole slab)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="smoke preset, tiny FFTs: every path incl. the card write")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--arm", default=None, choices=ARMS, help=argparse.SUPPRESS)
    ap.add_argument("--fft-n", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.single:
        run_single(args)
        return 0

    if args.smoke:
        args.preset = "smoke"
        args.fft_sizes = SMOKE_FFT_SIZES
        args.reps = min(args.reps, 2)

    # the orchestrator never touches a device: it only spawns and reads
    g = geometry(args.preset)
    arms = [a for a in args.arms.split(",") if a]
    print(f"=== gb probe: preset {g['preset']} ({g['n_part']}^3, nb={g['nb']} slabs of "
          f"{g['rows_per_slab']:,} rows, {g['tiles']} tiles at P={g['padded']}) ===")
    recs = []
    for arm in arms:
        if arm == "fft":
            for n in [int(v) for v in args.fft_sizes.split(",") if v]:
                recs.append(spawn("fft", args, fft_n=n))
        else:
            recs.append(spawn(arm, args))

    print(f"\n{'arm':12s} {'rows/n':>12s} {'first s':>9s} {'median s':>9s} "
          f"{'per-slab s':>11s} {'peak GiB':>9s}  note")
    for r in recs:
        if r["arm"] == "eject":
            pk = r.get("peak_bytes_in_use")
            print(f"{'eject e2e':12s} {r['rows']:12,d} {_fmt_s(r['first_call_s'])} "
                  f"{_fmt_s(r['end_to_end']['median_s'])} {_fmt_s(r['per_slab_end_to_end_s'])} "
                  f"{(pk or 0) / 1024**3:9.2f}  leave {r['leave_fraction']:.3%}")
            print(f"{'eject dev':12s} {r['rows']:12,d} {'-':>9s} "
                  f"{_fmt_s(r['device_only']['median_s'])} {_fmt_s(r['per_slab_device_only_s'])} "
                  f"{'':>9s}  kernel only, inputs on device")
        elif r["arm"] == "paint":
            pk = r.get("subblock_peak_bytes_in_use")
            print(f"{'paint sub':12s} {r['rows']:12,d} {_fmt_s(r['subblock_first_s'])} "
                  f"{_fmt_s(r['subblock']['median_s'])} {_fmt_s(r['per_slab_subblock_s'])} "
                  f"{(pk or 0) / 1024**3:9.2f}  extent {r['subblock_extent']}")
            pk = r.get("peak_bytes_in_use")
            note = ("mass equal" if r["mass_equal"] else "MASS MISMATCH")
            if r.get("full_error"):
                note = ("OOM " if r.get("full_oom") else "ERR ") + r["full_error"][:60]
            print(f"{'paint full':12s} {r['rows']:12,d} {_fmt_s(r.get('full_first_s'))} "
                  f"{_fmt_s(r['full']['median_s'] if r.get('full') else None)} "
                  f"{_fmt_s(r.get('per_slab_full_s'))} {(pk or 0) / 1024**3:9.2f}  {note}")
        else:
            pk = r.get("peak_bytes_in_use")
            if r.get("completed"):
                print(f"{'fft ' + str(r['n']) + '^3':12s} {'':>12s} {_fmt_s(r['first_fwd_s'])} "
                      f"{_fmt_s(r['fwd']['median_s'])} {_fmt_s(r['inv']['median_s'])} "
                      f"{(pk or 0) / 1024**3:9.2f}  fwd / inv; peak/field "
                      f"{(r.get('peak_over_field') or 0):.2f}; rt {r['roundtrip_max_abs_over_rms']:.1e}")
            else:
                print(f"{'fft ' + str(r.get('n')) + '^3':12s} {'':>12s} {'-':>9s} {'-':>9s} "
                      f"{'-':>11s} {(pk or 0) / 1024**3:9.2f}  "
                      f"{'OOM' if r.get('oom') else 'ERR'} {str(r.get('error', ''))[:60]}")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"m6_gb_probe{args.out_suffix}.json")
    with open(path, "w") as fh:
        json.dump(dict(geometry=g, reps=args.reps, row_frac=args.row_frac,
                       arms=recs, provenance=_provenance()), fh, indent=1)
    print(f"wrote {path}")

    # an arm that died without reporting is a failure; an OOM that reported
    # itself (the designed 2048^3 outcome, or the pessimistic full-mesh paint)
    # is a result
    bad = [r for r in recs if r.get("died_without_report") and not r.get("oom")]
    bad += [r for r in recs if r["arm"] in ("eject", "paint") and not r.get("completed")]
    if bad:
        print(f"FAILED arms: {[b.get('arm') for b in bad]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
