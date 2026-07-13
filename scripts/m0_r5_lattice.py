"""M0 R5 probe: position-lattice physics (roadmap R5).

Question: is the uint16 box-lattice position quantization (64 sub-cell levels
at the 1024^3 flagship) below the PM algorithm's own error floor?

Method: f32 reference BullFrog at 256^3 vs the SAME sim with positions
re-quantized to a B-bit box lattice after EVERY position update (both
half-drifts); velocities stay float (R2 owns the velocity axis). What matters
physically is sub-cell levels = 2^B / n_mesh, so at 256^3 mesh:
B=14 -> 64 levels == the 1024-mesh-on-2^16-lattice flagship configuration.

Floors: F-2K (time stepping) + a 512^3-mesh force-resolution arm (the
CUBE-style criterion; P(k) always measured with the SAME 256^3 paint,
like-vs-like). Plus a static force cross-check separating instantaneous force
error from dynamically accumulated error.

Kill/pivot: P(k) deviation > PM error at k <= k_Nyq/2 -> larger mesh:lattice
ratio or int16-positions-only redesign.

Run:  pixi run python scripts/m0_r5_lattice.py [--quick] [--outdir runs/m0/r5]
"""

# ruff: noqa: E402
import argparse
import json
import time
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _m0_common as mc

F32 = jnp.float32


def make_step(force, L, B=None):
    """Float BullFrog step; if B is set, re-quantize positions to the B-bit
    box lattice after each half-drift (mask in lattice ints, never float mod)."""
    if B is not None:
        s_B = L / 2.0**B
        nlat = 2**B

        def rq(x):
            xi = jnp.rint(x / s_B).astype(jnp.int32) & (nlat - 1)
            return xi.astype(F32) * jnp.float32(s_B)
    else:

        def rq(x):
            return jnp.mod(x, L)

    @jax.jit
    def step(x, v, dD2, alpha, bcoef):
        x = rq(x + dD2 * v)
        g = force(x)
        v = alpha * v + bcoef * g
        x = rq(x + dD2 * v)
        return x, v

    return step


def run(x0, v0, lad, step):
    x, v = x0, v0
    for k in range(lad.n_steps):
        x, v = step(x, v, 0.5 * lad.dD[k], lad.alphas[k], lad.betas[k] / lad.D_mid[k])
    return x, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outdir", default="runs/m0/r5")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    N = 64 if args.quick else 256
    N_hi = 2 * N
    L, K = 500.0, 10
    cosmo = mc.PLANCK
    k_nyq = np.pi * N / L
    B_list = [16, 12] if args.quick else [16, 14, 13, 12]
    t0 = time.time()

    lad = mc.ladder_constants(mc.a_grid(0.1, 1.0, K, "log"), cosmo, 1.0, 1.0)
    lad2 = mc.ladder_constants(mc.a_grid(0.1, 1.0, 2 * K, "log"), cosmo, 1.0, 1.0)
    delta0 = mc.linear_delta0(jax.random.PRNGKey(args.seed), N, L, cosmo)
    x0, v0 = mc.za_ics(delta0, N, L, 0.1, cosmo)

    force = mc.make_force_fn(N, L, N**3)
    force_hi = mc.make_force_fn(N_hi, L, N**3)

    def pk_native(x):
        mesh = mc.paint_f32(jnp.asarray(x, F32), N, L)
        return mc.pk_estimator(np.asarray(mesh, np.float64) - 1.0, L, n_bins=28, k_max=k_nyq)

    print(
        f"R5: N={N}, L={L}, K={K}, B sweep {B_list} "
        f"(levels {[2**b // N for b in B_list]}; flagship-equivalent = 64)"
    )

    step_f = make_step(force, L, None)
    tt = time.time()
    xF, vF = run(x0, v0, lad, step_f)
    kkb, pkF, _ = pk_native(xF)
    print(f"  [F    ] {time.time() - tt:.0f}s")
    tt = time.time()
    xF2, _ = run(x0, v0, lad2, step_f)
    _, pkF2, _ = pk_native(xF2)
    print(f"  [F-2K ] {time.time() - tt:.0f}s")

    tt = time.time()
    step_hi = make_step(force_hi, L, None)
    xH, _ = run(x0, v0, lad, step_hi)
    _, pkH, _ = pk_native(xH)
    print(f"  [F-hi ] 512^3-mesh force arm {time.time() - tt:.0f}s")

    step_floor = np.abs(pkF / pkF2 - 1.0)
    res_floor = np.abs(pkF / pkH - 1.0)

    results = dict(
        config=dict(N=N, L=L, K=K, seed=args.seed, k_nyq=k_nyq, B_list=B_list),
        k=kkb.tolist(),
        step_floor=step_floor.tolist(),
        res_floor=res_floor.tolist(),
        arms=[],
    )

    g0 = np.asarray(force(xF))
    g0_rms = float(np.sqrt(np.mean(g0**2)))
    for B in B_list:
        tt = time.time()
        step_q = make_step(force, L, B)
        xq, vq = run(x0, v0, lad, step_q)
        _, pkq, _ = pk_native(xq)
        ratio = pkq / pkF - 1.0
        # static force cross-check at the final reference configuration
        s_B = L / 2.0**B
        xF_q = (jnp.rint(jnp.asarray(xF) / s_B).astype(jnp.int32) & (2**B - 1)).astype(
            F32
        ) * jnp.float32(s_B)
        dg = np.asarray(force(xF_q)) - g0
        arm = dict(
            B=B,
            levels=2**B // N,
            sigma_B=s_B / np.sqrt(12.0),
            ratio=ratio.tolist(),
            band_max=float(np.max(np.abs(ratio)[kkb <= 0.5 * k_nyq])),
            pos_rms=mc.min_image_rms(np.asarray(xq), np.asarray(xF), L),
            force_rms_rel=float(np.sqrt(np.mean(dg**2)) / g0_rms),
            wall_s=time.time() - tt,
        )
        results["arms"].append(arm)
        print(
            f"  [Xq B={B}] levels={arm['levels']:3d} band_max={arm['band_max']:.2e} "
            f"pos_rms={arm['pos_rms']:.3e} force_rms_rel={arm['force_rms_rel']:.2e} "
            f"({arm['wall_s']:.0f}s)"
        )

    # verdict at flagship-equivalent levels (64): below the PM floors?
    sel = kkb <= 0.5 * k_nyq
    floor = np.maximum(step_floor, res_floor)[sel]
    v_arm = [a for a in results["arms"] if a["levels"] == 64]
    verdict = "N/A"
    if v_arm:
        ratio = np.abs(np.array(v_arm[0]["ratio"]))[sel]
        ok_strict = bool((ratio <= floor).all())
        ok_margin = bool((ratio <= np.maximum(floor, 1e-3)).all())
        verdict = "PASS" if ok_strict else ("PASS-1e-3" if ok_margin else "FAIL")
        results["verdict"] = dict(strict=ok_strict, within_1e3=ok_margin, verdict=verdict)

    with open(out / "r5_results.json", "w") as fh:
        json.dump(
            results,
            fh,
            indent=1,
            default=lambda o: o.item() if isinstance(o, np.generic) else str(o),
        )

    # figure
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.cm.viridis
    for i, arm in enumerate(results["arms"]):
        ax.loglog(
            kkb,
            np.abs(arm["ratio"]),
            color=cmap(i / max(len(B_list) - 1, 1)),
            lw=1.4,
            label=f"B={arm['B']} ({arm['levels']} levels)",
        )
        guide = kkb**2 * arm["sigma_B"] ** 2
        ax.loglog(kkb, guide, color=cmap(i / max(len(B_list) - 1, 1)), ls=":", lw=0.9)
    ax.loglog(kkb, step_floor, "k:", lw=1.4, label="stepping floor (K vs 2K)")
    ax.loglog(kkb, res_floor, "k--", lw=1.4, label="resolution floor (256 vs 512 mesh)")
    ax.axvline(0.5 * k_nyq, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("|P_Xq / P_F - 1|")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "r5_fig1_lattice.png", dpi=150)
    plt.close(fig)

    print(
        f"\nR5 verdict at 64 levels: {verdict} "
        f"(strict = below max(stepping, resolution) floor at all k <= 0.5 k_Nyq)"
    )
    print(f"total wall: {(time.time() - t0) / 60:.1f} min; outputs: {out.resolve()}")


if __name__ == "__main__":
    main()
