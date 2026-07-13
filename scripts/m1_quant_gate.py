"""M1 S6: Tier-B quantization gate measurement (CUBE-style floor comparison).

Question: is the int16 phase-space quantization error negligible RELATIVE TO
the PM method's own systematic floors at the same configuration? (The CUBE
argument: compression is free if it sits below the errors the method already
carries.) Three arms from the SAME injected ICs, all measured with the
neutral estimator:

  quant arm    : int16 evolve (production: int paint, f32 kernels) vs
                 evolve_float f64            -> the quantization error
  step arm     : evolve_float f64 at K vs 2K -> the stepping floor at K
  mesh arm     : evolve_float f64 with force mesh n vs 2n (same particles,
                 same steps)                 -> the PM force-resolution floor

Reported per k-band; the D-010 1e-4 relative-P(k) bar is quoted alongside
(the bar was ratified on M0-R2's like-for-like ladder budget, a different
measurement) -- the Tier-B gate NUMBER is JC's call at the S6 review.

    pixi run python scripts/m1_quant_gate.py --tag n128k40log_bullfrog_lpt2_s0
"""

import argparse
import json
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _m1_common as M  # noqa: E402

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="n128k40log_bullfrog_lpt2_s0")
    args = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from inexor import config, integrate

    ics = M.load_state(os.path.join(M.RUNS, f"ics_{args.tag}.npz"))
    cfg = ics["meta"]["config"]
    n, L, K = cfg["n_mesh"], cfg["box_size"], cfg["n_steps"]
    cosmo = config.Cosmology(**M.COSMO)
    quant = config.QuantConfig()
    x0 = jnp.asarray(ics["x"])
    v0 = jnp.asarray(ics["v_d"])

    def tcfg(n_steps):
        return config.TimeConfig(
            a_init=cfg["a_init"],
            a_final=cfg["a_final"],
            n_steps=n_steps,
            spacing=cfg["spacing"],
            integrator=cfg["integrator"],
        )

    def wrap(x):
        return np.mod(np.asarray(x), L)

    box = config.BoxConfig(n_mesh=n, box_size=L)
    box2 = config.BoxConfig(n_mesh=2 * n, box_size=L, n_particles=n)

    print(f"reference: evolve_float f64, {args.tag}")
    x_ref, _ = integrate.evolve_float(
        box, tcfg(K), cosmo, x0.astype(jnp.float64), v0.astype(jnp.float64), fdtype=jnp.float64
    )
    print("quant arm: int16 production path")
    x_q, _ = integrate.evolve(
        box,
        tcfg(K),
        quant,
        cosmo,
        x0.astype(jnp.float32),
        v0.astype(jnp.float32),
        driver="scan",
        paint="int",
        fdtype=jnp.float32,
    )
    print(f"step arm: evolve_float f64 at 2K = {2 * K}")
    x_2k, _ = integrate.evolve_float(
        box, tcfg(2 * K), cosmo, x0.astype(jnp.float64), v0.astype(jnp.float64), fdtype=jnp.float64
    )
    print(f"mesh arm: evolve_float f64, force mesh {2 * n}^3 (same particles/steps)")
    x_2m, _ = integrate.evolve_float(
        box2, tcfg(K), cosmo, x0.astype(jnp.float64), v0.astype(jnp.float64), fdtype=jnp.float64
    )

    out = dict(tag=args.tag, meta=M.make_meta("inexor", "quant-gate", cfg, REPO))
    print(f"\n{'arm':22s} {'rms [cells]':>12s} {'max |dP/P|':>12s} {'max 1-r':>12s}")
    for name, xa in [
        ("quant (int16 vs f64)", x_q),
        ("step (K vs 2K)", x_2k),
        ("mesh (n vs 2n force)", x_2m),
    ]:
        c = M.compare_states(wrap(x_ref), wrap(xa), n, L)
        out[name] = c
        print(
            f"{name:22s} {c['rms_cells']:12.3e} {c['ratio_max_absdev']:12.3e} "
            f"{c['one_minus_r_max']:12.3e}"
        )

    # k-band detail for the P-ratio (the D-010-bar-relevant statistic)
    print(
        f"\n|dP/P| by k-band {'':6s} {'low-k(4)':>10s} {'k<0.4':>10s} {'mid':>10s} {'Nyquist(8)':>10s}"
    )
    for name in ("quant (int16 vs f64)", "step (K vs 2K)", "mesh (n vs 2n force)"):
        r = np.abs(np.array(out[name]["p_ratio"]) - 1.0)
        nb = len(r)
        print(
            f"{name:22s} {r[:4].max():10.3e} {r[4:16].max():10.3e} "
            f"{r[16 : nb - 8].max():10.3e} {r[nb - 8 :].max():10.3e}"
        )

    jpath = os.path.join(M.RUNS, f"quant_gate_{args.tag}.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {jpath}")


if __name__ == "__main__":
    main()
