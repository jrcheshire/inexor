"""M0 R3 probe: int32-mesh CIC paint -- determinism AND speed (roadmap R3).

Question: is the integer-accumulation paint (determinism by associativity,
architecture.md Sec. 5 / D-006) bit-deterministic under XLA scatter-add on
CUDA, at <= 2x the cost of ordinary f32 atomics?

Method: paint N ~ 1e8 particles into a 512^3 mesh, 10 repeated dispatches of
identical input + one fresh-trace instance -> pairwise bit-compare (device-
side). Three position flavors: (u)niform, (l)attice-quantized (the real
pipeline's input distribution -- CIC fractions take 2^16/n_mesh discrete
values), (c)lustered (atomic-contention worst case -- the sensitive flavor
for demonstrating f32 nondeterminism; a uniform pass does NOT clear f32).
Bench: median of 10 timed calls, chunked scatter INSIDE the jitted function
(honest bench; also the 8 GB consumer-GPU memory strategy). The
--xla_gpu_deterministic_ops comparison arm re-execs a subprocess (the flag is
process-global and must precede jax import).

Kill/pivot: nondeterministic (investigate XLA lowering) or > 2x slower ->
sort + segment_sum fallback, cost re-budgeted.

AUTHORITATIVE ON CUDA. CPU runs validate plumbing only (CPU scatter is
sequential, trivially deterministic).
Run:  pixi run python scripts/m0_r3_paint.py [--log2-n 27] [--mesh 512]
"""

# ruff: noqa: E402
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

import _m0_common as mc

F32 = jnp.float32


def make_painters(N, L, frac_bits, n_chunks):
    """Chunked paints, jitted whole (scan over chunks carrying the mesh)."""
    scale = jnp.float32(2.0**frac_bits)

    def chunk_body_int(mesh, pc):
        base, frac = mc._cic_pieces(pc, N, L)
        for corner in mc._CORNERS:
            flat, w = mc._corner_flat_weight(base, frac, corner, N)
            mesh = mesh.at[flat].add(mc.rint_i(w * scale), mode="promise_in_bounds")
        return mesh, None

    def chunk_body_f32(mesh, pc):
        base, frac = mc._cic_pieces(pc, N, L)
        for corner in mc._CORNERS:
            flat, w = mc._corner_flat_weight(base, frac, corner, N)
            mesh = mesh.at[flat].add(w, mode="promise_in_bounds")
        return mesh, None

    @jax.jit
    def paint_int_c(pos_chunks):
        mesh, _ = lax.scan(chunk_body_int, jnp.zeros((N**3,), jnp.int32), pos_chunks)
        return mesh

    @jax.jit
    def paint_f32_c(pos_chunks):
        mesh, _ = lax.scan(chunk_body_f32, jnp.zeros((N**3,), F32), pos_chunks)
        return mesh

    return paint_int_c, paint_f32_c, chunk_body_int


def positions(flavor, n_part, L, N, key):
    if flavor == "uniform":
        return jax.random.uniform(key, (n_part, 3), minval=0.0, maxval=L, dtype=F32)
    if flavor == "lattice":
        u = jax.random.uniform(key, (n_part, 3), minval=0.0, maxval=L, dtype=F32)
        s = L / 2.0**16
        return (mc.rint_i(u / s) & (2**16 - 1)).astype(F32) * jnp.float32(s)
    if flavor == "clustered":
        kc, kn, ka = jax.random.split(key, 3)
        n_blob = 512
        centers = jax.random.uniform(kc, (n_blob, 3), minval=0.0, maxval=L, dtype=F32)
        idx = jax.random.randint(ka, (n_part,), 0, n_blob)
        sigma = 2.0 * L / N  # 2 cells
        return jnp.mod(centers[idx] + sigma * jax.random.normal(kn, (n_part, 3), dtype=F32), L)
    raise ValueError(flavor)


def bench(fn, arg, reps=10):
    fn(arg).block_until_ready()  # compile
    fn(arg).block_until_ready()  # warmup / autotune paranoia
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn(arg).block_until_ready()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def determinism(fn, arg, reps=10):
    ref = fn(arg)
    ident = all(bool(jnp.array_equal(fn(arg), ref)) for _ in range(reps - 1))
    return ident, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log2-n", type=int, default=27)
    ap.add_argument("--mesh", type=int, default=512)
    ap.add_argument("--frac-bits", type=int, default=12)
    ap.add_argument("--log2-chunk", type=int, default=24)
    ap.add_argument("--outdir", default="runs/m0/r3")
    ap.add_argument("--mode", default="main", choices=["main", "detflag-bench"])
    ap.add_argument("--flavor", default=None, help="detflag-bench: single flavor")
    args = ap.parse_args()

    N, L = args.mesh, 500.0
    n_part = 2**args.log2_n
    n_chunks = max(1, 2 ** max(args.log2_n - args.log2_chunk, 0))
    chunk = n_part // n_chunks
    platform = jax.devices()[0].platform

    paint_int_c, paint_f32_c, _ = make_painters(N, L, args.frac_bits, n_chunks)

    if args.mode == "detflag-bench":
        pos = positions(args.flavor, n_part, L, N, jax.random.PRNGKey(1)).reshape(
            n_chunks, chunk, 3
        )
        t = bench(paint_f32_c, pos)
        print(json.dumps(dict(flavor=args.flavor, t_f32_detflag=t)))
        return

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    authoritative = platform == "gpu"
    if not authoritative:
        print("=" * 66)
        print(f"NON-AUTHORITATIVE: {platform!r} scatter is sequential; needs CUDA")
        print("=" * 66)
    print(f"R3: n={n_part:.3e} particles, mesh {N}^3, F={args.frac_bits}, "
          f"{n_chunks} chunks of 2^{args.log2_chunk}")

    results = dict(
        config=dict(n_part=n_part, mesh=N, frac_bits=args.frac_bits,
                    n_chunks=n_chunks, platform=platform, authoritative=authoritative),
        flavors={},
    )

    # --xla_gpu_deterministic_ops arm (GPU only; process-global -> re-exec).
    # Must run BEFORE the flavor loop: the parent's XLA pool keeps its high-water
    # mark once the big paints have run, and the child OOMs on a 6 GB card.
    detflag = {}
    if authoritative:
        for flavor in ("uniform", "clustered"):
            env = dict(os.environ)
            env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true")
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            cmd = [sys.executable, __file__, "--mode", "detflag-bench", "--flavor", flavor,
                   "--log2-n", str(args.log2_n), "--mesh", str(args.mesh),
                   "--frac-bits", str(args.frac_bits), "--log2-chunk", str(args.log2_chunk)]
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
            if line:
                detflag[flavor] = json.loads(line[-1])["t_f32_detflag"]
            else:
                print(f"  [{flavor:9s}] detflag subprocess failed: {r.stderr[-300:]}")

    for flavor in ("uniform", "lattice", "clustered"):
        pos = positions(flavor, n_part, L, N, jax.random.PRNGKey(1)).reshape(
            n_chunks, chunk, 3
        )
        det_i, ref_i = determinism(paint_int_c, pos)
        # fresh-trace instance (new closure/executable), same input
        paint_int_c2, _, _ = make_painters(N, L, args.frac_bits, n_chunks)
        det_retrace = bool(jnp.array_equal(paint_int_c2(pos), ref_i))
        det_f, ref_f = determinism(paint_f32_c, pos)
        n_diff_f32 = 0
        if not det_f:
            m2 = paint_f32_c(pos)
            n_diff_f32 = int(jnp.sum(m2 != ref_f))
            del m2
        t_int = bench(paint_int_c, pos)
        t_f32 = bench(paint_f32_c, pos)
        # mass conservation of the quantized weights (subsample)
        sub = pos.reshape(-1, 3)[: 2**20]
        base, frac = mc._cic_pieces(sub, N, L)
        wsum = jnp.zeros((sub.shape[0],), jnp.int32)
        for corner in mc._CORNERS:
            _, w = mc._corner_flat_weight(base, frac, corner, N)
            wsum = wsum + mc.rint_i(w * 2.0**args.frac_bits)
        mass_err = float(jnp.max(jnp.abs(wsum - 2**args.frac_bits)) / 2.0**args.frac_bits)
        # max cell occupancy (int32 overflow headroom check)
        max_cell = int(jnp.max(ref_i)) * 2.0**-args.frac_bits
        rec = dict(det_int=bool(det_i), det_int_retrace=det_retrace, det_f32=bool(det_f),
                   n_diff_f32=n_diff_f32, t_int=t_int, t_f32=t_f32,
                   slowdown=t_int / t_f32, mass_err=mass_err, max_cell=max_cell)
        results["flavors"][flavor] = rec
        print(f"  [{flavor:9s}] int: det={det_i} retrace={det_retrace} {t_int * 1e3:8.1f} ms | "
              f"f32: det={det_f} (ndiff={n_diff_f32}) {t_f32 * 1e3:8.1f} ms | "
              f"slowdown={rec['slowdown']:.2f}x mass_err={mass_err:.1e} "
              f"max_cell={max_cell:.0f}")
        # loop variables persist across iterations: without this, the next flavor's
        # position build double-buffers ~2.6 GB of dead arrays (OOM on a 6 GB card)
        del pos, ref_i, ref_f

    for flavor, t in detflag.items():
        results["flavors"][flavor]["t_f32_detflag"] = t
        print(f"  [{flavor:9s}] f32+detflag: {t * 1e3:8.1f} ms "
              f"({t / results['flavors'][flavor]['t_f32']:.2f}x f32)")

    with open(out / "r3_results.json", "w") as fh:
        json.dump(results, fh, indent=1,
                  default=lambda o: o.item() if isinstance(o, np.generic) else str(o))

    worst = max(r["slowdown"] for r in results["flavors"].values())
    all_det = all(r["det_int"] and r["det_int_retrace"] for r in results["flavors"].values())
    tag = "" if authoritative else " [NON-AUTHORITATIVE: CPU]"
    print(f"\nR3{tag}: det={'10/10 identical' if all_det else 'FAIL'} "
          f"worst slowdown={worst:.2f}x (trigger: >2x)")
    print(f"   outputs: {out.resolve()}")


if __name__ == "__main__":
    main()
