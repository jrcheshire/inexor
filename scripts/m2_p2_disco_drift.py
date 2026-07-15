"""P2/P3: DISCO-DJ's float-replay gradient drift vs K, and its adjoint wall-clock.

Runs in DISCO-MOCKS' pixi env (the m1_run_disco.py pattern):

    pixi run --manifest-path ~/src/disco-mocks/pixi.toml -e gpu \
        python scripts/m2_p2_disco_drift.py --n 64 --k 5,10,20,40

WHY THIS IS NOW THE DECIDING MEASUREMENT. P1 (deneb job 22) killed the memory
claim: DISCO-DJ f32 pays 436.6-447.3 B/particle against inexor's 452.6-488.2, so
the ~440 B/particle transient is INHERENT to JAX PM adjoints and inexor is
slightly WORSE at f32. Nor is inexor f64-class accurate -- D-015 measured its
gradient at medrel 2e-4..3.3e-3 against the smooth float path (quantization),
where f64 replay drifts ~1e-11. So on the two axes measured so far inexor sits
BETWEEN f32 and f64 on both memory and accuracy: a Pareto point, not a win.

What is not yet measured, and is the only thing that could still make exact
replay worth its complexity:

  P2 -- DOES FLOAT-REPLAY ERROR GROW WITH K? Float replay reverses the DKD
        leapfrog in float arithmetic, so roundoff accumulates over the backward
        sweep; exact replay cannot accumulate, by construction. If DISCO-DJ's
        f32 error grows with K while inexor's stays flat, exact replay has a real
        niche at long rollouts -- which is where f_NL work lives. If both are
        flat, inexor is a Pareto point of marginal interest and the honest answer
        is to shelve. A SINGLE K cannot distinguish these; that is the whole
        point of the sweep.

  P3 -- IS INEXOR ACTUALLY SLOWER? Never measured. "A slower DISCO-DJ with
        slightly less floating-point error" is the fair summary to beat, and the
        wall-clock half of it is currently an assumption on our side too.

METHOD. Each code is compared against ITS OWN f64 gradient, not against the other
code. That is deliberate: a cross-code gradient comparison would fold in
convention differences (LPT, kernels, units) that P2 is not about, whereas
"how far is your cheap gradient from your own exact-arithmetic gradient, as K
grows" is exactly the drift question and is directly comparable between codes.
inexor's half of the plot comes from `m2_grad_gate.py --k-sweep` (adjoint vs f64
float-twin), which is the same quantity and the D-015 machinery.

Metrics mirror m2_grad_gate._global_metrics EXACTLY (norm ratio, Pearson corr,
median-rel on the top-decile |ref|). Global, not per-component: S3 established
the per-particle gradient is noise-dominated where the reference is ~0, so a
per-component rel would measure noise, not drift. Copied rather than imported
because this file must import cleanly in a foreign env (numpy only, no inexor).

Compute: deneb (free). Outputs runs/m2/p2_disco_drift.json.
"""

import argparse
import json
import os
import statistics
import sys
import time

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUNS = os.path.join(REPO, "runs", "m2")


def _single(n_mesh, K, f64, repeats):
    os.environ["JAX_ENABLE_X64"] = "1" if f64 else "0"

    import jax
    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _m1_common as M

    from discodj import DiscoDJ

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
    dj = dj.with_timetables().with_linear_ps()
    fplus_i = float(dj.cosmo.Fplus(a_i))

    # Same deterministic IC at every K and precision: the ONLY thing varying
    # across the sweep must be K (and f32-vs-f64), or the drift signal is
    # confounded by the ICs.
    d = M.BOX_SIZE / n_mesh
    c = np.arange(n_mesh, dtype=np.float64) * d
    qx, qy, qz = np.meshgrid(c, c, c, indexing="ij")
    q = np.stack([qx.ravel(), qy.ravel(), qz.ravel()], axis=1)
    rng = np.random.default_rng(0)
    x0 = np.mod(q + rng.normal(0.0, 0.05 * d, q.shape), M.BOX_SIZE).astype(dtype)
    v0 = np.zeros_like(x0)

    def loss(pos, vel):
        dje = dj.with_external_ics(pos=pos, vel=vel * fplus_i)
        X, _P, _ = dje.run_nbody(
            a_i,
            a_f,
            K,
            time_var=a_steps,
            stepper="bullfrog",
            method="pm",
            res_pm=n_mesh,
            worder=2,
            antialias=0,
            grad_kernel_order=0,  # they default to 4 (FD) -- the M1 lesson
            laplace_kernel_order=0,
            deconvolve=False,
            convert_to_numpy=False,
        )
        return jax.numpy.sum(X.reshape(-1, 3) ** 2)

    gfn = jax.grad(loss, argnums=0)
    g = np.asarray(jax.block_until_ready(gfn(jax.numpy.asarray(x0), jax.numpy.asarray(v0))))

    # P3 timing: warm-up excluded (it is compile), median over repeats, blocked.
    xj, vj = jax.numpy.asarray(x0), jax.numpy.asarray(v0)
    jax.block_until_ready(gfn(xj, vj))
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(gfn(xj, vj))
        ts.append(time.perf_counter() - t0)

    return {
        "n_mesh": n_mesh,
        "K": K,
        "f64": f64,
        "grad": g.tolist(),
        "adjoint_s": statistics.median(ts),
        "device": str(jax.devices()[0]),
    }


def _metrics(a, b):
    """Mirrors m2_grad_gate._global_metrics exactly."""
    import numpy as np

    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    ratio = float(np.linalg.norm(a) / np.linalg.norm(b))
    corr = float(np.corrcoef(a, b)[0, 1])
    thr = np.quantile(np.abs(b), 0.9)
    m = np.abs(b) >= thr
    medrel = float(np.median(np.abs(a[m] - b[m]) / np.abs(b[m])))
    return dict(ratio=ratio, corr=corr, medrel=medrel)


def _spawn(n_mesh, K, f64, repeats):
    import subprocess

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--single",
        "--n",
        str(n_mesh),
        "--k",
        str(K),
        "--repeats",
        str(repeats),
    ] + (["--f64"] if f64 else [])
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        err = [
            ln
            for ln in (p.stderr or "").strip().splitlines()
            if ln.strip()
            and "For simplicity, JAX has removed" not in ln
            and not ln.startswith(("  ", "Traceback"))
        ]
        return {
            "n_mesh": n_mesh,
            "K": K,
            "f64": f64,
            "error": err[-1][:200] if err else f"exit {p.returncode}",
        }
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--k", default="5,10,20,40")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--f64", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.single:
        print(json.dumps(_single(args.n, int(args.k), args.f64, args.repeats)))
        return

    rows = []
    for K in [int(x) for x in args.k.split(",") if x.strip()]:
        print(f"[p2_disco] n={args.n} K={K} f32 + f64 ...", flush=True)
        lo = _spawn(args.n, K, False, args.repeats)
        hi = _spawn(args.n, K, True, args.repeats)
        if lo.get("error") or hi.get("error"):
            rows.append({"K": K, "error": lo.get("error") or hi.get("error")})
            continue
        m = _metrics(lo["grad"], hi["grad"])
        rows.append(
            {
                "K": K,
                **m,
                "f32_adjoint_s": lo["adjoint_s"],
                "f64_adjoint_s": hi["adjoint_s"],
                "device": lo["device"],
            }
        )

    print("\n===== P2: DISCO-DJ float-replay gradient error vs its OWN f64, by K =====")
    print("  (does float-replay error ACCUMULATE with steps? that is the question)")
    print("\n     K   median-rel      corr     |ratio-1|    f32 adjoint s")
    for r in rows:
        if r.get("error"):
            print(f"  {r['K']:>4}   ERR: {r['error'][:56]}")
            continue
        print(
            f"  {r['K']:>4}   {r['medrel']:9.3e}  {r['corr']:.7f}  "
            f"{abs(r['ratio'] - 1):9.3e}     {r['f32_adjoint_s']:7.3f}"
        )

    ok = [r for r in rows if not r.get("error")]
    if len(ok) > 1:
        g = ok[-1]["medrel"] / ok[0]["medrel"] if ok[0]["medrel"] > 0 else float("nan")
        print(f"\n  drift growth K={ok[0]['K']}->{ok[-1]['K']}: medrel x{g:.2f}")
        print("  compare inexor (m2_grad_gate --k-sweep, same quantity vs its own f64):")
        print("  D-015 measured 2e-4..3.3e-3 at K=6; if inexor is FLAT in K and this")
        print("  GROWS, exact replay has a niche at long rollouts. If both are flat,")
        print("  inexor is a Pareto point of marginal interest -- that is a shelve signal.")
    print("\n  Sets no gate. Numbers go to JC.")

    os.makedirs(RUNS, exist_ok=True)
    path = os.path.join(RUNS, "p2_disco_drift.json")
    with open(path, "w") as f:
        json.dump({"rows": rows}, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
