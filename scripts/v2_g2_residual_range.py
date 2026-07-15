"""v2 probe G2: dynamic range of LPT-residual phase space vs absolute state.

Run 2026-07-14 on the stored M1 parity artifacts (no new simulation); the
numbers are quoted in docs/plan-plan-v2.md Sec 4 and the design study Sec 3.
Findings: ~4.8x position dynamic-range reduction vs the 2LPT reference; ZA
beats 2LPT as a compression reference (smaller tails); velocities compress
only ~2-2.8x with 10-14 sigma tails (the binding component).

Requires runs/m1/ics_* and runs/m1/inexor_float_* from the M1 harness.
Run from the repo root:  JAX_ENABLE_X64=1 pixi run python scripts/v2_g2_residual_range.py
"""

import json
import os

import jax.numpy as jnp
import numpy as np

from inexor.config import Cosmology
from inexor.cosmology import growth_factor_2, growth_factor_a, growth_rate_2, growth_rate_a
from inexor.lpt import lagrangian_grid, second_order_displacement, zeldovich_displacement

RUNS = "runs/m1"
OUT_DIR = "runs/v2"
CASES = [
    ("n64k10log_bullfrog_lpt2_s0", 64),
    ("n128k40log_bullfrog_lpt2_s0", 128),
]


def wrap_min_image(d, L):
    return (d + 0.5 * L) % L - 0.5 * L


def stats(name, arr, L, n_mesh, out):
    """arr: (N,3) residual-like quantity in Mpc/h (or v_d units)."""
    cell = L / n_mesh
    flat = np.abs(np.asarray(arr)).ravel()
    s = {
        "rms": float(np.sqrt(np.mean(np.asarray(arr) ** 2))),
        "p50": float(np.percentile(flat, 50)),
        "p99": float(np.percentile(flat, 99)),
        "p999": float(np.percentile(flat, 99.9)),
        "p9999": float(np.percentile(flat, 99.99)),
        "max": float(flat.max()),
    }
    s["rms_cells"] = s["rms"] / cell
    s["max_cells"] = s["max"] / cell
    out[name] = s
    print(
        f"  {name:28s} rms {s['rms']:9.4f}  p99 {s['p99']:9.4f}  "
        f"p99.99 {s['p9999']:9.4f}  max {s['max']:9.4f}   (units as given)"
    )
    return s


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    for tag, n in CASES:
        ics = np.load(f"{RUNS}/ics_{tag}.npz", allow_pickle=True)
        fin = np.load(f"{RUNS}/inexor_float_{tag}.npz", allow_pickle=True)
        meta = json.loads(str(ics["meta"]))
        cfg = meta["config"]
        cosmo = Cosmology(**meta["cosmo"])
        L = float(cfg["box_size"])
        a_i, a_f = float(cfg["a_init"]), float(cfg["a_final"])
        assert cfg["n_mesh"] == n

        delta0 = jnp.asarray(ics["delta0"], dtype=jnp.float64)
        q = np.asarray(lagrangian_grid(n, L, fdtype=jnp.float64))

        # Psi fields from the code's own kernels (z=0 normalized).
        psi1 = np.asarray(zeldovich_displacement(delta0, L, fdtype=jnp.float64))
        psi2 = np.asarray(second_order_displacement(delta0, L, fdtype=jnp.float64))

        # Growth numbers (conventions per lpt.py docstring):
        # x(a)  = q + D1 psi1 - D2 psi2
        # v_d(a)= psi1 - (D2 f2)/(D1 f1) psi2
        D1i, D1f = growth_factor_a(a_i, cosmo), growth_factor_a(a_f, cosmo)
        D2i, D2f = growth_factor_2(a_i, cosmo), growth_factor_2(a_f, cosmo)
        f1i, f1f = growth_rate_a(a_i, cosmo), growth_rate_a(a_f, cosmo)
        f2i, f2f = growth_rate_2(a_i, cosmo), growth_rate_2(a_f, cosmo)

        # Sanity: reproduce stored ICs from delta0 (validates conventions here).
        x0_pred = q + D1i * psi1 - D2i * psi2
        v0_pred = psi1 - (D2i * f2i) / (D1i * f1i) * psi2
        dx0 = wrap_min_image(np.asarray(ics["x"]) - x0_pred, L)
        dv0 = np.asarray(ics["v_d"]) - v0_pred
        print(
            f"[{tag}] IC reproduction: max|dx0| {np.abs(dx0).max():.3e} Mpc/h, "
            f"max|dv0| {np.abs(dv0).max():.3e} (should be ~roundoff)"
        )

        x_f = np.asarray(fin["x"])
        v_f = np.asarray(fin["v_d"])

        x_za = q + D1f * psi1
        x_2lpt = q + D1f * psi1 - D2f * psi2
        v_za = psi1
        v_2lpt = psi1 - (D2f * f2f) / (D1f * f1f) * psi2

        out = {}
        print(f"[{tag}] L={L} n={n} cell={L / n:.3f} Mpc/h  D1(a_f)={D1f:.4f}")
        print("  -- positions (Mpc/h) --")
        stats("disp_full |x_f - q|", wrap_min_image(x_f - q, L), L, n, out)
        stats("resid vs ZA", wrap_min_image(x_f - x_za, L), L, n, out)
        stats("resid vs 2LPT", wrap_min_image(x_f - x_2lpt, L), L, n, out)
        print("  -- velocities (v_d units) --")
        stats("v_full |v_f|", v_f, L, n, out)
        stats("v resid vs ZA", v_f - v_za, L, n, out)
        stats("v resid vs 2LPT", v_f - v_2lpt, L, n, out)

        # Quantization arithmetic (positions).
        q16_global = L / 2**16  # D-014-validated code's quantum
        sig = out["resid vs 2LPT"]["rms"] / np.sqrt(3)  # per-component sigma
        for c in (6.0, 8.0):
            q8_resid = 2 * c * sig / 256.0
            out[f"quantum_int8_resid_c{int(c)}"] = q8_resid
            print(
                f"  int8-resid quantum (range +-{int(c)} sigma): {q8_resid:.5f} Mpc/h"
                f"  vs int16-global {q16_global:.5f}  -> ratio {q8_resid / q16_global:.2f}x"
            )
        out["quantum_int16_global"] = q16_global
        out["sigma_resid_percomp"] = float(sig)
        results[tag] = out

    with open(f"{OUT_DIR}/g2_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {OUT_DIR}/g2_results.json")


if __name__ == "__main__":
    main()
