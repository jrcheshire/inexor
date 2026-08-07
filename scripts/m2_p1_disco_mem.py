"""P1: peak device memory of DISCO-DJ's adjoint (the CONTROL for inexor's transient).

Runs in DISCO-MOCKS' pixi env, never inexor's (the m1_run_disco.py pattern).
Invoke from the inexor repo root:

    pixi run --manifest-path ~/src/disco-mocks/pixi.toml -e gpu \
        python scripts/m2_p1_disco_mem.py --n 64,128 --steps 10

`-e gpu` is REQUIRED: disco-mocks' `default` env is CPU JAX, and a CPU backend
reports no memory_stats, so a default-env run silently measures nothing. The gpu
env (SPHEREx/disco-mocks#7) is CUDA JAX in its own prefix.

WHY THIS EXISTS (plan glistening-greeting-possum). S4 measured inexor's adjoint at
466 B/particle where architecture Sec. 9 budgeted ~68, so the 1024^3 flagship
projects ~6x over its 80 GB claim. But claim 2 is COMPARATIVE, and the real
question is not "do we win" -- it is whether the ~430 B/particle transient is OURS
or INHERENT to JAX PM adjoints. DISCO-DJ is the control that answers it:

  DISCO-DJ ~= inexor  -> the transient is inherent. inexor's state advantage
                         (12-24 B/particle) stays swamped and claim 2 needs
                         restating, not optimizing.
  DISCO-DJ << inexor  -> our fat is ours, the bisect has a named target, and a
                         mature code has proved the target reachable.

Do NOT read this as a "who is better" benchmark. Note especially that DISCO-DJ's
adjoint is ALSO O(1) in steps (a hand-rolled custom_vjp over lax.scan whose fwd
residual is the final state only -- discodj/nbody/nbody_scan_functions.py; its
Diffrax path is dead code, disco_dj.py:1103 raises NotImplementedError). Both
codes are backsolves. The paper's differentiators are replay FIDELITY (bit-exact
vs float drift) and state WIDTH, not step scaling.

FAIRNESS (each of these is load-bearing; see the plan):
  - f32. The M1 parity harness forces JAX_ENABLE_X64=1; for MEMORY that would be
    unfair (f64 state = 48 B/particle vs inexor's 12). Claim 2 is against float
    state at f32 -- pmwd's cited drift regime. This script therefore sets
    JAX_ENABLE_X64=0 explicitly and passes precision="single". --f64 runs the
    secondary row.
  - Matched config from _m1_common (BOX_SIZE, A_INIT, A_FINAL, shared a_steps),
    same N/K, `bullfrog` stepper -- which both codes have.
  - grad_kernel_order=0: DISCO-DJ DEFAULTS IT TO 4 (finite difference). The M1
    lesson from m1_run_disco.py. Left at 4 it would solve a different force and
    the memory would not be comparable.
  - Loss L = sum(x_f^2): deliberately trivial. No paint, no estimator, so the
    number isolates the adjoint and cannot be moved by our P(k) machinery vs
    theirs.
  - One config per FRESH SUBPROCESS: peak_bytes_in_use is monotonic within a
    process and has no reset API (the m2_mem_profile rule) -- a second config in
    the same process inherits the first's high-water mark.
  - Production config: XLA deterministic ops NOT set (S4 measured they cost
    memory and OOM'd 256^3).

Reports forward-only and full-adjoint peaks separately, mirroring inexor's
ICs/forward/adjoint split, so their adjoint cost is separable from their forward.

Compute: deneb (free, x86). Its 6 GB caps N, but B/particle is the comparison
quantity and it is measured scale-invariant for inexor (12.0-13.7x carry across
64^3..512^3); Vista confirms absolutes later if the answer warrants.
Outputs runs/m2/p1_disco_mem_{f32,f64}.json.
"""

import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUNS = os.path.join(REPO, "runs", "m2")
GIB = 1024.0**3


def _single(n_mesh, K, f64, forward_only=False, pm_ratio=1):
    # x64 BEFORE any jax import. Default OFF: f32 is the fair comparison, and
    # disco-mocks' own tasks set JAX_ENABLE_X64=1 in their env, so this must
    # override rather than inherit.
    os.environ["JAX_ENABLE_X64"] = "1" if f64 else "0"

    import jax
    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _m1_common as M  # numpy-only shared config; safe in a foreign env

    from discodj import DiscoDJ

    dev = jax.devices()[0]

    def peak():
        try:
            return (dev.memory_stats() or {}).get("peak_bytes_in_use")
        except Exception:
            return None

    def limit():
        try:
            return (dev.memory_stats() or {}).get("bytes_limit")
        except Exception:
            return None

    dtype = np.float64 if f64 else np.float32
    a_steps = M.a_grid(M.A_INIT, M.A_FINAL, K, "log")
    a_i, a_f = float(a_steps[0]), float(a_steps[-1])

    dj = DiscoDJ(
        dim=3,
        res=n_mesh,
        boxsize=M.BOX_SIZE,
        cosmo=dict(M.COSMO_DISCO),
        precision="double" if f64 else "single",
    )
    dj = dj.with_timetables()
    # Their built-in Eisenstein-Hu (the default) avoids needing our pk file: the
    # linear spectrum only sets the IC amplitude, and this measures MEMORY, not
    # parity, so the Tcmb=2.72548-vs-2.7255 quibble that forces `from_file` in
    # m1_run_disco.py is irrelevant here. Valid keys are "Eisenstein-Hu",
    # "BBKS", "GeneticAlgorithm", "DiscoEB", "from_file", "none" -- NOT "EH",
    # which is what job 21 died on.
    dj = dj.with_linear_ps()
    # NO with_lpt here: it belongs to the with_external_ics(delta=...) path and
    # needs 'fphi' in the IC dict. We inject pos/vel directly (m1_run_disco.py's
    # _evolve_injected), so LPT is not in the picture -- calling it raises
    # KeyError: 'fphi' not found in initial conditions dictionary.

    n_part = n_mesh**3
    fplus_i = float(dj.cosmo.Fplus(a_i))

    # A deterministic Lagrangian-grid IC in inexor's own convention, so both
    # codes carry the same array shapes/dtypes into the adjoint.
    d = M.BOX_SIZE / n_mesh
    c = np.arange(n_mesh, dtype=dtype) * d
    qx, qy, qz = np.meshgrid(c, c, c, indexing="ij")
    q = np.stack([qx.ravel(), qy.ravel(), qz.ravel()], axis=1).astype(dtype)
    rng = np.random.default_rng(0)
    x0 = np.mod(q + rng.normal(0.0, 0.05 * d, q.shape).astype(dtype), M.BOX_SIZE)
    v0 = np.zeros_like(x0)

    def evolve(pos, vel):
        dje = dj.with_external_ics(pos=pos, vel=vel * fplus_i)
        X, P, _ = dje.run_nbody(
            a_i,
            a_f,
            K,
            time_var=a_steps,
            stepper="bullfrog",
            method="pm",
            # res_pm/n_mesh is the mesh:particle ratio, and it MUST match the
            # arm it is compared against or the memory comparison is meaningless
            # (our config table runs n_fine = 2 x n_part per side, so pm_ratio=2
            # is the matched setting; pm_ratio=1 reproduces the v1 rows).
            res_pm=int(pm_ratio) * n_mesh,
            worder=2,
            antialias=0,
            grad_kernel_order=0,  # DISCO-DJ defaults to 4 (FD) -- M1 lesson
            laplace_kernel_order=0,
            deconvolve=False,
            convert_to_numpy=False,
        )
        return X, P

    x0j, v0j = jax.numpy.asarray(x0), jax.numpy.asarray(v0)
    jax.block_until_ready((x0j, v0j))
    peak_ic = peak()

    def loss(pos, vel):
        X, _P = evolve(pos, vel)
        return jax.numpy.sum(X.reshape(-1, 3) ** 2)  # trivial; isolates the adjoint

    if forward_only:
        # Capacity-ladder mode: the adjoint is not the comparison (v2 is a
        # forward engine), and running it would cap the ladder ~3x lower on
        # memory than the forward can actually reach.
        t0 = time.perf_counter()
        X, _P = evolve(x0j, v0j)
        jax.block_until_ready(X)
        wall_fwd = time.perf_counter() - t0
        peak_fwd = peak()
        peak_adj = None
        # Same guard shape as the adjoint's: a forward that silently returned
        # the inputs (or NaNs) would give a flatteringly small peak AND be
        # meaningless. A real evolve moves every particle a finite distance.
        Xn = np.asarray(X).reshape(-1, 3)
        moved = float(np.linalg.norm(Xn - x0))
        finite = bool(np.all(np.isfinite(Xn)))
        gnorm, ran = moved, bool(finite and moved > 0.0)
    else:
        t0 = time.perf_counter()
        jax.block_until_ready(loss(x0j, v0j))
        wall_fwd = time.perf_counter() - t0
        peak_fwd = peak()

        g = jax.grad(loss, argnums=(0, 1))(x0j, v0j)
        jax.block_until_ready(g)
        peak_adj = peak()

    # Guard against measuring a no-op. If the adjoint silently returned zeros
    # (a broken trace, convert_to_numpy severing the graph, ...) the peak would
    # be flatteringly small AND meaningless -- and it would look like a great
    # result for DISCO-DJ. A real gradient is finite and nonzero.
        gx = np.asarray(g[0])
        gnorm = float(np.linalg.norm(gx))
        finite = bool(np.all(np.isfinite(gx)))
        ran = bool(finite and gnorm > 0.0)

    return {
        "code": "discodj",
        "n_mesh": n_mesh,
        "n_particles": n_part,
        "K": K,
        "f64": f64,
        "forward_only": bool(forward_only),
        "pm_ratio": int(pm_ratio),
        "res_pm": int(pm_ratio) * n_mesh,
        "platform": dev.platform,
        "device": str(dev),
        "bytes_limit": limit(),
        "peak_after_ic": peak_ic,
        "peak_forward": peak_fwd,
        "peak_adjoint": peak_adj,
        "peak_forward_per_particle": (peak_fwd / n_part) if peak_fwd else None,
        "peak_adjoint_per_particle": (peak_adj / n_part) if peak_adj else None,
        "wall_forward_s": wall_fwd,
        "grad_norm": gnorm,
        "grad_finite": finite,
        "adjoint_ran": ran,
    }


def _spawn(n_mesh, K, f64, forward_only=False, pm_ratio=1):
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--single",
        "--n",
        str(n_mesh),
        "--steps",
        str(K),
        "--pm-ratio",
        str(pm_ratio),
    ]
    if f64:
        cmd.append("--f64")
    if forward_only:
        cmd.append("--forward-only")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        err = (p.stderr or "").strip().splitlines()
        # Skip JAX's traceback-filtering boilerplate, which is NOT the error --
        # taking the last stderr line reports it and hides the real exception
        # (the m2_mem_profile bug).
        real = [
            ln
            for ln in err
            if ln.strip()
            and "For simplicity, JAX has removed" not in ln
            and not ln.startswith(("  ", "Traceback"))
        ]
        return {
            "code": "discodj",
            "n_mesh": n_mesh,
            "K": K,
            "f64": f64,
            "error": real[-1][:200] if real else f"exit {p.returncode}",
            "oom": any("RESOURCE_EXHAUSTED" in ln or "Out of memory" in ln for ln in err),
        }
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", default="64,128", help="comma-separated n_mesh")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--f64", action="store_true", help="secondary row: f64 state")
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument(
        "--forward-only",
        action="store_true",
        help="capacity-ladder mode: skip the adjoint so the ladder finds the FORWARD ceiling",
    )
    ap.add_argument(
        "--pm-ratio",
        type=int,
        default=1,
        help="res_pm / n_mesh; 2 matches our config table's n_fine = 2 x n_part",
    )
    ap.add_argument("--tag", default=None, help="output suffix (default: f32/f64)")
    args = ap.parse_args()

    if args.single:
        print(
            json.dumps(
                _single(int(args.n), args.steps, args.f64, args.forward_only, args.pm_ratio)
            )
        )
        return

    recs = []
    for n in [int(x) for x in args.n.split(",") if x.strip()]:
        print(
            f"[p1_disco] n_mesh={n} K={args.steps} f64={args.f64} "
            f"pm_ratio={args.pm_ratio} forward_only={args.forward_only} ...",
            flush=True,
        )
        rec = _spawn(n, args.steps, args.f64, args.forward_only, args.pm_ratio)
        recs.append(rec)
        if rec.get("error"):
            # The ceiling IS the measurement: stop climbing once a rung dies,
            # and say whether it died of memory or of something else.
            print(
                f"  rung n_mesh={n} FAILED (oom={rec.get('oom')}): {rec.get('error')}",
                flush=True,
            )
            break

    print("\n===== DISCO-DJ adjoint peak (the control for inexor's transient) =====")
    lim = next((r.get("bytes_limit") for r in recs if r.get("bytes_limit")), None)
    print(
        f"  device: {next((r.get('device') for r in recs if r.get('device')), '?')}"
        f"   limit: {lim / GIB if lim else float('nan'):.3f} GiB   "
        f"precision: {'f64' if args.f64 else 'f32'}"
    )

    # A CPU backend exposes no memory_stats, so peaks are None there: the run is
    # an API/plumbing smoke, not a measurement. Format defensively rather than
    # dividing None by GIB (the m2_mem_profile _fmt_gib rule).
    def g(v):
        return "     -" if v is None else f"{v / GIB:6.3f}"

    print("\n  n_mesh    ICs      fwd  adjoint     B/particle (adjoint)")
    for r in recs:
        if r.get("error"):
            print(f"  {r['n_mesh']:>6}  {'OOM' if r.get('oom') else 'ERR'}: {r['error'][:56]}")
            continue
        a = r["peak_adjoint"]
        bpp = f"{a / r['n_particles']:8.1f}" if a else "       - (no device stats: CPU?)"
        flag = "" if r.get("adjoint_ran") else "   <-- ADJOINT DID NOT RUN (grad zero/non-finite)"
        print(
            f"  {r['n_mesh']:>6}  {g(r['peak_after_ic'])}  {g(r['peak_forward'])}  "
            f"{g(a)}      {bpp}{flag}"
        )

    ok = [r for r in recs if not r.get("error") and r.get("peak_adjoint")]
    if len(ok) > 1:
        pp = [r["peak_adjoint"] / r["n_particles"] for r in ok]
        spread = max(pp) / min(pp)
        print(
            f"\n  scale-invariance across n: {min(pp):.1f}-{max(pp):.1f} B/particle "
            f"({spread:.2f}x spread)"
        )
        if spread > 1.25:
            print("  WARNING: not scale-invariant -> the comparison is not per-particle")
            print("  and extrapolating it is invalid. This is a STOP condition (plan).")
    if ok:
        print("\n  inexor for comparison (deneb job 19, same card): 452.6 (64^3) / 488.2 (128^3)")
        print("  B/particle. Sanity floor: DISCO-DJ's f32 carry alone is ~48 B/particle")
        print("  (x, p, xi, pi x3 x4B); a peak below that means the adjoint did not run.")

    os.makedirs(RUNS, exist_ok=True)
    tag = args.tag or ("f64" if args.f64 else "f32")
    path = os.path.join(RUNS, f"p1_disco_mem_{tag}.json")
    with open(path, "w") as f:
        json.dump({"configs": recs}, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
