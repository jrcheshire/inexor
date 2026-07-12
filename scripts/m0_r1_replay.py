"""M0 R1 probe: bit-exact replay under XLA fusion (roadmap R1).

64^3, K=10 forward + K=10 reverse, exact integer equality with the initial
state, many seeds, FOUR drivers spanning the compilation spectrum:

  A1 -- ONE jitted program containing both scans (fwd + rev).
  A2 -- jit(scan(fwd)) and jit(scan(rev)) as separate programs: the realistic
        adjoint shape, most exposed to the Sec. 5 fusion hazard.
  B  -- per-step jit(step_fwd) / jit(step_rev), python loop (+ donate variant).
  C  -- eager integer ops around ONE shared jitted force executable (the
        hazard-free-by-construction pattern; R4's driver).

Session evidence (macOS CPU): two differently-compiled programs of the same
step can flip ~1 rint half-tie per ~1e5 components/step, so A2/B *may* fail
sparsely where C cannot -- that outcome selects the production driver, it
does not kill the design (roadmap kill column). Plus a wrap-adversarial arm
(s_w0/64: w wraps int16 mid-run) where replay must STILL be exact (D-007).

AUTHORITATIVE ON CUDA (atomics/fusion). CPU runs are smoke tests.
Run:  pixi run python scripts/m0_r1_replay.py [--seeds 100] [--outdir runs/m0/r1]
"""

# ruff: noqa: E402
import argparse
import json
import time
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

import _m0_common as mc

F32 = jnp.float32


def build(N, L, K, cosmo):
    force = mc.make_force_fn(N, L, N**3)
    force_j = jax.jit(force)
    s_x = L / 2.0**16
    lad_u = mc.ladder_constants(mc.a_grid(0.1, 1.0, K, "log"), cosmo, 1.0, 1.0)
    return force, force_j, s_x, lad_u


def make_drivers(force, force_j, s_x, K):
    def fwd_body(state, c):
        return mc.step_fwd(*state, c, force, s_x), None

    def rev_body(state, c):
        return mc.step_rev(*state, c, force, s_x), None

    @jax.jit
    def a1_roundtrip(x0, w0, cs):
        s, _ = lax.scan(fwd_body, (x0, w0), cs)
        s, _ = lax.scan(rev_body, s, cs, reverse=True)
        return s

    @jax.jit
    def scan_fwd(x0, w0, cs):
        s, _ = lax.scan(fwd_body, (x0, w0), cs)
        return s

    @jax.jit
    def scan_rev(x, w, cs):
        s, _ = lax.scan(rev_body, (x, w), cs, reverse=True)
        return s

    step_fwd_j = jax.jit(lambda x, w, c: mc.step_fwd(x, w, c, force, s_x))
    step_rev_j = jax.jit(lambda x, w, c: mc.step_rev(x, w, c, force, s_x))
    step_fwd_d = jax.jit(
        lambda x, w, c: mc.step_fwd(x, w, c, force, s_x), donate_argnums=(0, 1)
    )
    step_rev_d = jax.jit(
        lambda x, w, c: mc.step_rev(x, w, c, force, s_x), donate_argnums=(0, 1)
    )

    def loop_driver(fwd, rev):
        def go(x0, w0, cs):
            x, w = x0, w0
            for k in range(K):
                c = mc.StepConsts(cs.c1[k], cs.c2[k], cs.kappa[k])
                x, w = fwd(x, w, c)
            for k in reversed(range(K)):
                c = mc.StepConsts(cs.c1[k], cs.c2[k], cs.kappa[k])
                x, w = rev(x, w, c)
            return x, w

        return go

    def c_driver(x0, w0, cs):
        x, w = x0, w0
        for k in range(K):
            c = mc.StepConsts(cs.c1[k], cs.c2[k], cs.kappa[k])
            x, w = mc.step_fwd(x, w, c, force_j, s_x)
        for k in reversed(range(K)):
            c = mc.StepConsts(cs.c1[k], cs.c2[k], cs.kappa[k])
            x, w = mc.step_rev(x, w, c, force_j, s_x)
        return x, w

    return {
        "A1_one_program": lambda x0, w0, cs: a1_roundtrip(x0, w0, cs),
        "A2_two_scans": lambda x0, w0, cs: scan_rev(*scan_fwd(x0, w0, cs), cs),
        "B_perstep_jit": loop_driver(step_fwd_j, step_rev_j),
        "B_perstep_donate": loop_driver(step_fwd_d, step_rev_d),
        "C_eager_shared_force": c_driver,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--outdir", default="runs/m0/r1")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    platform = jax.devices()[0].platform
    authoritative = platform == "gpu"
    if not authoritative:
        print("=" * 66)
        print(f"NON-AUTHORITATIVE: running on {platform!r}; R1's verdict needs CUDA")
        print("(macOS-arm64 CPU has known run-to-run jit quirks; smoke test only)")
        print("=" * 66)

    N, L, K = args.n, 256.0, args.steps
    cosmo = mc.PLANCK
    force, force_j, s_x, lad_u = build(N, L, K, cosmo)
    drivers = make_drivers(force, force_j, s_x, K)

    t0 = time.time()
    results = dict(
        config=dict(N=N, L=L, K=K, seeds=args.seeds, platform=platform,
                    authoritative=authoritative),
        drivers={},
    )

    # per-seed ICs (encode once, shared by all drivers)
    def make_state(seed, s_w0_div=1.0):
        delta0 = mc.linear_delta0(jax.random.PRNGKey(seed), N, L, cosmo)
        xph, v0 = mc.za_ics(delta0, N, L, 0.1, cosmo)
        s_w0 = mc.s_w0_policy(float(jnp.max(jnp.abs(v0))), lad_u.P[-1]) / s_w0_div
        lad = mc.ladder_constants(mc.a_grid(0.1, 1.0, K, "log"), cosmo, s_w0, s_x)
        cs = mc.step_consts(lad)
        return mc.encode_x(xph, s_x), mc.encode_w(v0, s_w0), cs

    fails = {name: [] for name in drivers}
    detail = {}
    for seed in range(args.seeds):
        x0, w0, cs = make_state(seed)
        x0h, w0h = np.asarray(x0), np.asarray(w0)
        for name, drv in drivers.items():
            x, w = drv(x0, w0, cs)
            ok = bool(np.array_equal(np.asarray(x), x0h) and np.array_equal(np.asarray(w), w0h))
            if not ok:
                fails[name].append(seed)
                if name not in detail:  # diagnose the first failure per driver
                    dx = np.asarray(x).astype(np.int64) - x0h.astype(np.int64)
                    dw = np.asarray(w).astype(np.int64) - w0h.astype(np.int64)
                    nz = np.count_nonzero(dx) + np.count_nonzero(dw)
                    mx = int(max(np.abs(dx).max(), np.abs(dw).max()))
                    detail[name] = dict(seed=seed, n_diff=int(nz), max_abs_diff=mx)
                    np.savez(out / f"r1_fail_{name}_seed{seed}.npz", dx=dx, dw=dw)
        if seed % 20 == 19:
            print(f"  seed {seed + 1}/{args.seeds} "
                  f"({(time.time() - t0):.0f}s) fails so far: "
                  f"{ {k: len(v) for k, v in fails.items()} }")

    # wrap-adversarial arm (D-007): s_w0/64 -> w wraps int16 mid-run;
    # replay must still be exact. Drivers B and C.
    x0, w0, cs = make_state(0, s_w0_div=64.0)
    x0h, w0h = np.asarray(x0), np.asarray(w0)
    wrap_ok = {}
    for name in ("B_perstep_jit", "C_eager_shared_force"):
        x, w = drivers[name](x0, w0, cs)
        wrap_ok[name] = bool(
            np.array_equal(np.asarray(x), x0h) and np.array_equal(np.asarray(w), w0h)
        )

    results["drivers"] = {
        name: dict(passed=args.seeds - len(f), failed=len(f), fail_seeds=f[:10],
                   first_fail=detail.get(name))
        for name, f in fails.items()
    }
    results["wrap_adversarial"] = wrap_ok

    with open(out / "r1_results.json", "w") as fh:
        json.dump(results, fh, indent=1,
                  default=lambda o: o.item() if isinstance(o, np.generic) else str(o))

    tag = "" if authoritative else " [NON-AUTHORITATIVE: CPU]"
    parts = [f"{n}={args.seeds - len(f)}/{args.seeds}" for n, f in fails.items()]
    wraptxt = " ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in wrap_ok.items())
    print(f"\nR1{tag}: " + " ".join(parts))
    print(f"   wrap-adversarial: {wraptxt}")
    print(f"   total wall: {(time.time() - t0) / 60:.1f} min; outputs: {out.resolve()}")


if __name__ == "__main__":
    main()
