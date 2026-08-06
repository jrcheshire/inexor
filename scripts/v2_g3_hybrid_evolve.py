"""Rung H3: evolve the hybrid arm and measure the density, not just the force.

WHAT IS COMPARED, three arms from one set of initial conditions:

  mono     the monolithic full-force reference (built by v2_g3_ladder.build)
  tiled    evolve_scola -- the FAILING scheme: the full kernel on each padded
           tile, no global long-range term at all
  hybrid   evolve_cola driven by the two-level force: a global coarse
           long-range solve plus force_short_tiled on fine tiles, exactly
           v2_g5_two_level_force.force_two_level

The headline is the auto amplitude ratio sqrt(P_arm/P_mono) per shell, which
involves no cross-realization comparison and is the quantity D-v2-12 clause 2
rests on. The squeezed-bispectrum ratios are printed too, but as PRELIMINARY:
the gate reading is D-v2-7's bar at the production box with an ensemble
comparison (D-v2-12 clause 4), and this is one seed at a development box.

WHAT THIS RUNG DOES NOT SHOW, and it matters. The hybrid is evolved in
LOCKSTEP: every particle's position is held and the global long-range solve is
done from the true current positions at each step. That answers whether the
PHYSICS is right. It does NOT demonstrate the memory saving that motivated
tiling in the first place, because a global solve per step is not a
tile-sequential schedule. Making it sequential -- e.g. by evaluating the
long-range force on the analytic background positions and precomputing one
coarse force mesh per step -- is a separate question with its own error to
measure, and nothing here licenses assuming it is free.

Usage:
    pixi run python scripts/v2_g3_hybrid_evolve.py --config cdev8 --tile 64 --buf 16
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT_DIR = os.path.join(os.path.dirname(HERE), "runs", "v2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cdev8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile", type=int, default=64, help="tile side in FINE cells")
    ap.add_argument("--buf", type=int, default=16, help="buffer in FINE cells")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--family", default="gauss")
    ap.add_argument("--assign-long", default="tsc")
    ap.add_argument("--nbins", type=int, default=10)
    ap.add_argument("--skip-tiled", action="store_true",
                    help="skip the failing arm (it is slow and already characterised)")
    ap.add_argument("--skip-bg", action="store_true",
                    help="skip the frozen-background arm (the tile-independence test)")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    import v2_g3_core as g3
    import v2_g3_floors as fl
    import v2_g3_ladder as lad
    from inexor import painting
    from v2_g5_core import force_global, force_short_tiled, padded_size

    B = lad.build(args.config, args.seed)
    g = B["g"]
    ell, n_fine, n_part = g["L"], g["n_fine"], g["n_part"]
    n_coarse, d_c, d_f = g["n_coarse"], g["coarse_cell"], g["fine_cell"]
    n_tot = n_part**3
    kf = 2.0 * np.pi / ell
    r_s = args.alpha * d_c
    p_side, b_real = padded_size(args.tile, args.buf, n_fine=n_fine)

    print(f"\n  config {args.config}  L={ell}  n_fine={n_fine}  n_coarse={n_coarse}")
    print(f"  tile T={args.tile} ({args.tile * d_f:.1f} Mpc/h)  "
          f"buffer b={b_real} ({b_real * d_f:.1f} Mpc/h = {b_real * d_f / r_s:.1f} r_s)  "
          f"P={p_side}  r_s={r_s:.3f}\n")

    def dens(x):
        return np.asarray(
            painting.density_contrast(x, n_fine, ell, n_tot, paint="int"), np.float64)

    def make_force(frozen_background):
        """The two-level force. frozen_background sources the long-range field
        from the ANALYTIC LPT positions instead of the true evolved ones.

        That is the whole architecture question. x_LPT(D) is known for every
        particle at every step without evolving anything, so if this arm agrees
        with the true-position one, all 20 coarse force fields can be computed
        UP FRONT and every tile can then run its full schedule independently --
        no lockstep, no per-step synchronisation. The gather point stays the
        particle's TRUE position: only the SOURCE of the field is approximated.
        """
        state = {"k": 0}

        def f(pos):
            pn = np.asarray(pos, np.float64)
            if frozen_background:
                d_mid = float(B["coeffs"][state["k"]][6])
                state["k"] += 1
                src = np.mod(np.asarray(
                    g3.x_lpt(B["q"], B["psi1"], B["psi2"], d_mid), np.float64), ell)
                gather = pn
            else:
                src, gather = pn, None
            gl, _ = force_global(jnp.asarray(src), n_coarse, ell, n_tot, "long",
                                 family=args.family, r_s=r_s, n_ref=n_fine,
                                 match=(d_c, d_f), assign=args.assign_long,
                                 pos_gather=gather)
            gs, _ = force_short_tiled(pn, n_fine, ell, n_tot, args.tile, args.buf,
                                      family=args.family, r_s=r_s)
            return jnp.asarray(np.asarray(gl, np.float64) + np.asarray(gs, np.float64))

        return f, state

    d_mono = dens(B["x_mono"])
    arms = {}

    if not args.skip_tiled:
        t0 = time.perf_counter()
        x_t, _, _ = g3.evolve_scola(B["q"], B["psi1"], B["psi2"], B["qi"], B["coeffs"],
                                    ell, n_fine, n_part, args.tile, args.buf,
                                    B["d_final"])
        arms["tiled"] = (dens(x_t), time.perf_counter() - t0)
        print(f"  tiled  (failing scheme)  evolved in {arms['tiled'][1]:.0f}s")

    zero = np.zeros_like(B["q"])
    for name, frozen in (("hybrid", False), ("hybrid_bg", True)):
        if frozen and args.skip_bg:
            continue
        fn, state = make_force(frozen)
        t0 = time.perf_counter()
        x_h, _, _ = g3.evolve_cola(zero, zero, B["q"], B["psi1"], B["psi2"],
                                   B["coeffs"], fn, ell, B["d_final"])
        arms[name] = (dens(np.asarray(x_h, np.float64)), time.perf_counter() - t0)
        if frozen and state["k"] != len(B["coeffs"]):
            raise SystemExit(
                f"frozen-background arm stepped the schedule {state['k']} times for "
                f"{len(B['coeffs'])} coefficients -- the background was NOT advancing "
                "with the integrator, so this arm measured something else entirely.")
        tag = "(proposal, lockstep)" if not frozen else "(frozen background)"
        print(f"  {name:8s} {tag:22s} evolved in {arms[name][1]:.0f}s")
    print()

    centers = [float(m * kf) for m in range(1, args.nbins + 1)]
    print("  AUTO amplitude ratio sqrt(P_arm/P_mono) -- realization-independent")
    print(f"    {'arm':10s} " + "".join(f"{m:>8d}" for m in range(1, args.nbins + 1))
          + "   <- k/k_f")
    rows = {}
    for name, (d_a, wall) in arms.items():
        t_sh, r_sh, a_sh = fl.shell_transfer(d_a, d_mono, ell, centers, kf)
        print(f"    {name:10s} " + "".join(f"{v:8.3f}" for v in a_sh))
        rows[name] = dict(wall=wall, A=[float(v) for v in a_sh],
                          r=[float(v) for v in r_sh], T=[float(v) for v in t_sh])
    print(f"    {'':10s} " + "".join(f"{'':>8s}" for _ in centers))
    print("  correlation r(k) with the monolithic arm")
    for name in arms:
        print(f"    {name:10s} " + "".join(f"{v:8.3f}" for v in rows[name]["r"]))

    # PRELIMINARY, not a gate reading: one seed, development box, and the
    # ratified reading is an ensemble comparison at the production box.
    tris, names = fl._triangles(kf, (1, 2, 3), 12)
    print("\n  squeezed-bispectrum ratios (PRELIMINARY -- one seed, dev box,")
    print("  realization-matched; the gate reading is an ensemble at cdev)")
    print(f"    {'arm':10s} {'max|R_B|':>10s} {'max|rho_auto|':>14s} {'bar':>6s}")
    for name, (d_a, _) in arms.items():
        s = fl.stats(d_a, d_mono, ell, tris)
        sq = [i for i, nm in enumerate(names) if nm != "equi"]
        rb = float(np.nanmax(np.abs(np.array(s["R_B"])[sq])))
        ra = float(np.nanmax(np.abs(np.array(s["rho_auto"])[sq])))
        print(f"    {name:10s} {rb:10.4f} {ra:14.4f} {0.15:6.2f}")
        rows[name].update(max_abs_R_B=rb, max_abs_rho_auto=ra)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"g3_hybrid_evolve_{args.config}_T{args.tile}_b{b_real}.json")
    with open(out, "w") as fh:
        json.dump(dict(config=args.config, seed=args.seed, tile=args.tile,
                       buf=int(b_real), p_side=int(p_side), alpha=args.alpha,
                       family=args.family, r_s=r_s, L=ell, n_fine=n_fine,
                       n_coarse=n_coarse, k_over_kf=list(range(1, args.nbins + 1)),
                       arms=rows), fh, indent=1)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
