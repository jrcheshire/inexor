"""v2 gate G1: custom-kernel (Pallas) CIC paint/gather floor vs the XLA path.

The question (plan-plan Sec 5, seed V1): the v1 R6 profile put the enemy at
XLA's particle-op transients (the 8x (flat, w) corner materializations behind
`.at[].add`), not the state. Does a hand Pallas kernel -- chunked over
particles, atomic-add into ONE preallocated mesh via input_output_aliases --
recover the hand-managed memory floor inside JAX, and at what wall cost?

Kernels (probe code; promotion to package code is a V4/M-v2-1 decision):
  paint_int : int32 fixed-point atomic-add CIC paint, weight arithmetic
              matching painting.paint_int op-for-op (f32 frac product,
              rint -> int32 at frac_bits). Integer atomics are associative,
              so the result must be RUN-TO-RUN bit-stable on CUDA; whether it
              also bit-matches the XLA paint_int is measured, not assumed
              (two compiled programs can differ by ulps in the f32 weights ->
              rint tie-flips; the R4 lesson).
  paint_f32 : same shape, f32 atomics (wall/peak comparison only --
              nondeterministic by construction, never a primal candidate).
  gather    : 3 mesh fields read with ONE shared CIC stencil
              (cic_read_vector twin).
  composed  : paint_int -> gather(mesh, mesh, mesh) -- the particle-side ops
              of one force evaluation (FFTs excluded: mesh-side, identical
              for both impls).

Kill line (plan-plan V1): pallas floor > 2x the hand-managed estimate
(positions + mesh + output, computed analytically per shape below) -> X2
(JAX+kernels) dies; escalate X3 (CUDA core) vs A3 (tiles-in-plain-JAX) to JC.

Pallas GPU has no scatter primitive (jax#31876); plt.atomic_add IS the route.
NB Pallas interpret mode emulates atomic_add as a vector scatter --
last-write-wins on duplicate indices WITHIN one call (measured 2026-07-15) --
so CPU/interpret correctness checks only work on duplicate-free particle
sets (--sparse); the authoritative correctness + determinism gates run on
CUDA (deneb, scripts/v2_g1_deneb.sbatch).

Orchestration: one (op, impl, shape) per fresh subprocess (peak_bytes_in_use
is monotonic, retrospective Sec 5). Each worker reports the (peak B/p, wall)
pair for the cost-of-memory record.

Run (deneb):
    pixi run -e gpu python scripts/v2_g1_kernel_floor.py
CPU plumbing smoke (laptop; interpret mode, sparse correctness only):
    pixi run python scripts/v2_g1_kernel_floor.py --shapes 32:64 --interpret
"""

import argparse
import functools
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
OUT_DIR = os.path.join(REPO, "runs", "v2")

FRAC_BITS = 12
DEFAULT_SHAPES = "128:256,256:512,512:512"  # n_part:n_mesh; 256:512 = C-dev shape
OPS = ("paint_int", "paint_f32", "gather", "composed")
CORNERS = [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]


# ===========================================================================
# pallas kernels
# ===========================================================================


def _corner_flat_weight_k(px, py, pz, corner, n_mesh, cell):
    """Base cell + one corner's (flat index, weight), matching
    painting._cic_pieces / _corner_flat_weight arithmetic (f32).

    Component-wise 1D form: Triton block dimensions must be powers of 2, so
    the kernels take (chunk,) x/y/z arrays, never a (chunk, 3) block.
    """
    import jax.numpy as jnp

    dx, dy, dz = corner
    flat = None
    ws = []
    for p, d in ((px, dx), (py, dy), (pz, dz)):
        xp = p / cell
        base_f = jnp.floor(xp)
        frac = xp - base_f
        # no stop_gradient: nothing differentiates through the kernel, and the
        # extra primitive is one more thing for the Triton lowering to chew on
        idx = (base_f.astype(jnp.int32) + d) % n_mesh
        flat = idx if flat is None else flat * n_mesh + idx
        ws.append(frac if d else 1.0 - frac)
    return flat, ws[0] * ws[1] * ws[2]


def _rint_nonneg_halfeven(x):
    """Round-half-even for NON-NEGATIVE x, from floor primitives only.

    jnp.rint (lax round_p) has no Pallas Triton lowering (job 31:
    "Unimplemented primitive ... round"). CIC corner weights are >= 0 and
    w * 2^FRAC_BITS <= 4096 << 2^24, so floor/compare arithmetic below is
    EXACT in f32 and reproduces jnp.rint bit-for-bit on this domain.
    """
    import jax.numpy as jnp

    r = jnp.floor(x)
    f = x - r
    odd = r - 2.0 * jnp.floor(0.5 * r)  # exact parity for r < 2^24
    up = (f > 0.5) | ((f == 0.5) & (odd == 1.0))
    return r + up.astype(x.dtype)


def _paint_kernel(px_ref, py_ref, pz_ref, mesh_in_ref, mesh_out_ref, *, n_mesh, cell, int_paint):
    import jax.experimental.pallas.triton as plt
    import jax.numpy as jnp

    del mesh_in_ref  # aliased with mesh_out_ref; the zeros come in as input
    px, py, pz = px_ref[...], py_ref[...], pz_ref[...]
    scale = np.float32(2.0**FRAC_BITS)
    for corner in CORNERS:
        flat, w = _corner_flat_weight_k(px, py, pz, corner, n_mesh, cell)
        if int_paint:
            plt.atomic_add(
                mesh_out_ref, (flat,), _rint_nonneg_halfeven(w * scale).astype(jnp.int32)
            )
        else:
            plt.atomic_add(mesh_out_ref, (flat,), w)


def _gather_kernel(
    px_ref, py_ref, pz_ref, gx_ref, gy_ref, gz_ref, ox_ref, oy_ref, oz_ref, *, n_mesh, cell
):
    import jax.numpy as jnp

    px, py, pz = px_ref[...], py_ref[...], pz_ref[...]
    n = px.shape[0]
    ax = jnp.zeros((n,), jnp.float32)
    ay = jnp.zeros((n,), jnp.float32)
    az = jnp.zeros((n,), jnp.float32)
    for corner in CORNERS:
        flat, w = _corner_flat_weight_k(px, py, pz, corner, n_mesh, cell)
        ax = ax + w * gx_ref[flat]
        ay = ay + w * gy_ref[flat]
        az = az + w * gz_ref[flat]
    ox_ref[...] = ax
    oy_ref[...] = ay
    oz_ref[...] = az


def _pos_components(positions):
    """(n, 3) -> three (n,) arrays (Triton block dims must be powers of 2)."""
    import jax.numpy as jnp

    return tuple(jnp.asarray(positions[:, c]) for c in range(3))


def _triton_params(interpret):
    """Explicit Triton backend selection. jax 0.10 defaults pallas-GPU to
    Mosaic GPU (jax_pallas_use_mosaic_gpu=True); on stacks where the Mosaic
    backend fails to import (it targets Hopper+), compiler_params=None leaves
    NO backend and raises the misleading "install jaxlib GPU" error (jobs
    26-29). Ampere = Triton, requested explicitly."""
    if interpret:
        return {}
    import jax.experimental.pallas.triton as plt

    return dict(compiler_params=plt.CompilerParams())


def pallas_paint(positions, n_mesh, box_size, chunk, int_paint, interpret):
    """Chunked atomic-add CIC paint into ONE preallocated mesh (aliased)."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    n = positions.shape[0]
    cell = np.float32(box_size / n_mesh)
    dt = jnp.int32 if int_paint else jnp.float32
    mesh0 = jnp.zeros((n_mesh**3,), dtype=dt)
    kern = functools.partial(_paint_kernel, n_mesh=n_mesh, cell=cell, int_paint=int_paint)
    c = pl.BlockSpec((chunk,), lambda i: (i,))
    return pl.pallas_call(
        kern,
        grid=(n // chunk,),
        in_specs=[c, c, c, pl.BlockSpec((n_mesh**3,), lambda i: (0,))],
        out_specs=pl.BlockSpec((n_mesh**3,), lambda i: (0,)),
        out_shape=jax.ShapeDtypeStruct((n_mesh**3,), dt),
        input_output_aliases={3: 0},
        interpret=interpret,
        **_triton_params(interpret),
    )(*_pos_components(positions), mesh0)


def pallas_gather(gx, gy, gz, positions, n_mesh, box_size, chunk, interpret):
    """3-field CIC gather (one shared stencil), chunked over particles."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    n = positions.shape[0]
    cell = np.float32(box_size / n_mesh)
    kern = functools.partial(_gather_kernel, n_mesh=n_mesh, cell=cell)
    c = pl.BlockSpec((chunk,), lambda i: (i,))
    m = pl.BlockSpec((n_mesh**3,), lambda i: (0,))
    out = pl.pallas_call(
        kern,
        grid=(n // chunk,),
        in_specs=[c, c, c, m, m, m],
        out_specs=[c, c, c],
        out_shape=[jax.ShapeDtypeStruct((n,), jnp.float32)] * 3,
        interpret=interpret,
        **_triton_params(interpret),
    )(*_pos_components(positions), gx.reshape(-1), gy.reshape(-1), gz.reshape(-1))
    return jnp.stack(out, axis=1)


# ===========================================================================
# worker (fresh subprocess per (op, impl, shape))
# ===========================================================================


def _positions(n_part, n_mesh, sparse, seed=0):
    """Uniform random positions (or a duplicate-free sparse set for the
    interpret-mode correctness check: one particle per 4^3-cell block, jittered
    INSIDE its cell so no two particles share any CIC corner)."""
    rng = np.random.default_rng(seed)
    L = float(n_mesh)  # cell = 1.0; cell size is irrelevant to floor/wall
    if not sparse:
        return (rng.random((n_part**3, 3)) * L).astype(np.float32), L
    n_s = n_mesh // 4
    grid = np.stack(np.meshgrid(*(np.arange(n_s) * 4.0,) * 3, indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    jit = 0.1 + 0.8 * rng.random(grid.shape)
    return (grid + jit).astype(np.float32), L


def run_single(args):
    import jax
    import jax.numpy as jnp

    from inexor.painting import cic_read_vector, counts_from_int, paint_f32, paint_int

    n_part, n_mesh = args.np_, args.nm
    pos_np, L = _positions(n_part, n_mesh, args.sparse)
    pos = jnp.asarray(pos_np)
    dev = jax.devices()[0]

    def peak():
        try:
            return (dev.memory_stats() or {}).get("peak_bytes_in_use")
        except Exception:
            return None

    interp = args.interpret or dev.platform == "cpu"
    chunk = args.chunk
    n_tot = pos.shape[0]
    if n_tot % chunk:
        chunk = max(cs for cs in (256, 512, 1024, 2048, 4096) if n_tot % cs == 0)

    def op_fn():
        if args.impl == "xla":
            if args.op == "paint_int":
                return paint_int(pos, n_mesh, L, FRAC_BITS)
            if args.op == "paint_f32":
                return paint_f32(pos, n_mesh, L)
            if args.op == "gather":
                g = counts_from_int(paint_int(pos, n_mesh, L, FRAC_BITS))
                return cic_read_vector(g, g, g, pos, n_mesh, L)
            g = counts_from_int(paint_int(pos, n_mesh, L, FRAC_BITS))
            return cic_read_vector(g, g, g, pos, n_mesh, L)
        if args.op == "paint_int":
            return pallas_paint(pos, n_mesh, L, chunk, True, interp)
        if args.op == "paint_f32":
            return pallas_paint(pos, n_mesh, L, chunk, False, interp)
        mesh = counts_from_int(pallas_paint(pos, n_mesh, L, chunk, True, interp))
        m3 = mesh.reshape(n_mesh, n_mesh, n_mesh)
        if args.op == "gather":
            return pallas_gather(m3, m3, m3, pos, n_mesh, L, chunk, interp)
        return pallas_gather(m3, m3, m3, pos, n_mesh, L, chunk, interp)

    jax.block_until_ready(pos)
    peak_inputs = peak()
    out = jax.block_until_ready(op_fn())  # warmup + compile
    walls = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        jax.block_until_ready(op_fn())
        walls.append(time.perf_counter() - t0)
    peak_op = peak()

    rec = dict(
        op=args.op,
        impl=args.impl,
        n_part=n_part,
        n_mesh=n_mesh,
        chunk=chunk,
        sparse=args.sparse,
        platform=dev.platform,
        interpret=bool(interp),
        wall_median_s=float(np.median(walls)),
        peak_after_inputs=peak_inputs,
        peak_total=peak_op,
        peak_bytes_per_particle=(peak_op / n_tot) if peak_op else None,
    )

    # correctness + determinism (pallas arms only; authoritative on CUDA)
    if args.impl == "pallas":
        if args.op == "paint_int":
            ref = paint_int(pos, n_mesh, L, FRAC_BITS).reshape(-1)
            rec["n_diff_vs_xla"] = int(jnp.sum(out != ref))
            reps = [np.asarray(pallas_paint(pos, n_mesh, L, chunk, True, interp)) for _ in range(8)]
            rec["n_diff_run_to_run"] = int(sum(np.sum(r != reps[0]) for r in reps[1:]))
        elif args.op == "paint_f32":
            ref = paint_f32(pos, n_mesh, L).reshape(-1)
            d = np.abs(np.asarray(out) - np.asarray(ref))
            rec["max_absdiff_vs_xla"] = float(d.max())
        elif args.op in ("gather", "composed"):
            g = counts_from_int(paint_int(pos, n_mesh, L, FRAC_BITS))
            ref = cic_read_vector(g, g, g, pos, n_mesh, L)
            d = np.abs(np.asarray(out) - np.asarray(ref))
            denom = max(float(np.abs(np.asarray(ref)).max()), 1e-30)
            rec["max_reldiff_vs_xla"] = float(d.max() / denom)

    print("WORKER_JSON " + json.dumps(rec))


# ===========================================================================
# orchestration
# ===========================================================================


def hand_managed_floor(n_part, n_mesh, op):
    """Analytic hand-managed floor in B/particle: inputs + mesh + output only
    (the kill-line reference: pallas floor > 2x this -> X2 dies)."""
    n_tot = n_part**3
    pos = 12  # f32 (n,3)
    mesh = 4 * n_mesh**3 / n_tot
    if op in ("paint_int", "paint_f32"):
        return pos + mesh
    if op == "gather":
        return pos + 3 * mesh + 12  # 3 fields in + (n,3) out
    return pos + 2 * mesh + 12  # composed: painted mesh + counts + out


def spawn(op, impl, n_part, n_mesh, chunk, reps, sparse, interpret):
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--single",
        "--op",
        op,
        "--impl",
        impl,
        "--np",
        str(n_part),
        "--nm",
        str(n_mesh),
        "--chunk",
        str(chunk),
        "--reps",
        str(reps),
    ]
    if sparse:
        cmd.append("--sparse")
    if interpret:
        cmd.append("--interpret")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        lines = (p.stderr or "").strip().splitlines()
        # keep the WHOLE stderr on disk; surface the real exception line, not
        # JAX's trailing "For simplicity..." boilerplate (job 26 lesson)
        err_dir = os.path.join(OUT_DIR, "g1_errs")
        os.makedirs(err_dir, exist_ok=True)
        err_path = os.path.join(err_dir, f"{op}_{impl}_{n_part}_{n_mesh}.err")
        with open(err_path, "w") as fh:
            fh.write(p.stderr or "")
        exc = [t for t in lines if ("Error" in t or "Exception" in t) and "For simplicity" not in t]
        return dict(
            op=op,
            impl=impl,
            n_part=n_part,
            n_mesh=n_mesh,
            error=(exc[-1] if exc else (lines[-1] if lines else f"exit {p.returncode}")),
            error_file=err_path,
            oom=any("RESOURCE_EXHAUSTED" in t or "Out of memory" in t for t in lines),
        )
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("WORKER_JSON ")][-1]
    return json.loads(line[len("WORKER_JSON ") :])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shapes", default=DEFAULT_SHAPES, help="comma list of n_part:n_mesh")
    ap.add_argument("--chunk", type=int, default=4096, help="particles per pallas program")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument(
        "--sparse",
        action="store_true",
        help="duplicate-free particle set (interpret-mode correctness)",
    )
    ap.add_argument("--interpret", action="store_true", help="force pallas interpret mode")
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--op", default="paint_int", choices=OPS, help=argparse.SUPPRESS)
    ap.add_argument("--impl", default="pallas", choices=("xla", "pallas"), help=argparse.SUPPRESS)
    ap.add_argument("--np", dest="np_", type=int, default=128, help=argparse.SUPPRESS)
    ap.add_argument("--nm", type=int, default=256, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.single:
        run_single(args)
        return

    shapes = [tuple(int(v) for v in s.split(":")) for s in args.shapes.split(",")]
    recs = []
    print("=== G1 kernel floor: pallas vs XLA ===")
    for n_part, n_mesh in shapes:
        for op in OPS:
            for impl in ("xla", "pallas"):
                print(f"[worker] {op:9s} {impl:6s} np={n_part} nm={n_mesh} ...", flush=True)
                r = spawn(
                    op, impl, n_part, n_mesh, args.chunk, args.reps, args.sparse, args.interpret
                )
                r["hand_managed_bpp"] = hand_managed_floor(n_part, n_mesh, op)
                recs.append(r)

    print(
        f"\n{'op':9s} {'impl':6s} {'shape':>9s} {'peak B/p':>9s} {'hand B/p':>9s} "
        f"{'x hand':>7s} {'wall ms':>8s}   correctness"
    )
    by = {}
    for r in recs:
        if r.get("error"):
            print(
                f"{r['op']:9s} {r['impl']:6s} {r['n_part']:>4}:{r['n_mesh']:<4} "
                f"{'OOM' if r.get('oom') else 'ERR'}  {r['error'][:60]}"
            )
            continue
        key = (r["op"], r["n_part"], r["n_mesh"])
        by.setdefault(key, {})[r["impl"]] = r
        bpp = r.get("peak_bytes_per_particle")
        hand = r["hand_managed_bpp"]
        ratio = f"{bpp / hand:7.2f}" if bpp else "      -"
        chk = ""
        for k in ("n_diff_vs_xla", "n_diff_run_to_run", "max_absdiff_vs_xla", "max_reldiff_vs_xla"):
            if k in r:
                chk += f" {k}={r[k]:.3e}" if isinstance(r[k], float) else f" {k}={r[k]}"
        print(
            f"{r['op']:9s} {r['impl']:6s} {r['n_part']:>4}:{r['n_mesh']:<4} "
            f"{bpp:9.1f} {hand:9.1f} {ratio} {r['wall_median_s'] * 1e3:8.2f}  {chk}"
            if bpp
            else f"{r['op']:9s} {r['impl']:6s} {r['n_part']:>4}:{r['n_mesh']:<4} "
            f"{'-':>9s} {hand:9.1f} {ratio} {r['wall_median_s'] * 1e3:8.2f}  {chk}"
        )

    print(
        "\nKill line: pallas peak > 2x hand-managed -> X2 dies (JC escalation); "
        "verdict is JC's, nothing self-ratified."
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "g1_results.json")
    with open(path, "w") as fh:
        json.dump(dict(chunk=args.chunk, shapes=args.shapes, configs=recs), fh, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
