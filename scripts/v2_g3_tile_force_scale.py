"""Where does the tiled arm lose its large-scale power? Measure the FORCE.

THE OBSERVATION THIS EXISTS TO EXPLAIN. At cdev the tiled arm carries only
15-40% of the monolithic LONG-MODE POWER (auto amplitude ratio sqrt(P_t/P_m) =
0.39-0.63 at the box fundamental), while pure 2LPT -- the analytic background
the tiles are built on top of -- carries 98%. The tiling destroys large-scale
power its own background already had right, which is the opposite of what a
COLA-style scheme is supposed to do (runs/v2/g3_stage5_record.md sec. 9, ADR
D-v2-12).

THE HYPOTHESIS, read off the integrator. cola_step_bullfrog updates the
residual velocity as

    u <- alpha*u + bcoef*g + c_k1*psi1 + c_k2*psi2

where g is the force and the psi terms are the analytic frame. COLA works
because at large scales the true force very nearly CANCELS the frame terms, so
the residual stays small there and the background carries the long modes
untouched. But a tile's g comes from force_global on a PERIODIC PADDED BOX,
which supports no mode longer than that box. If g is missing its
long-wavelength part while the frame terms keep theirs, the cancellation fails
and the residual grows a spurious large-scale piece that FIGHTS the background
displacement -- exactly the observed suppression.

WHAT IS MEASURED. One force evaluation per arm, no evolution: the monolithic
force and the assembled per-tile force on the SAME particles at the SAME
positions. Both are compared on the LAGRANGIAN grid, where the particle set is
a regular lattice and a per-shell transfer is well defined.

THE DISCRIMINATING PREDICTION, fixed before the run: if the hypothesis holds,
the tile force loses amplitude specifically BELOW the padded tile's own
fundamental, k_P = 2*pi/(P*cell), and tracks the monolithic force above it. A
loss that is flat in k, or that sets in somewhere unrelated to k_P, refutes it
and points at the reassembly instead.

Usage:
    pixi run python scripts/v2_g3_tile_force_scale.py --config smoke
    pixi run python scripts/v2_g3_tile_force_scale.py --config cdev8 --arms 64:16,64:32
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT_DIR = os.path.join(os.path.dirname(HERE), "runs", "v2")


def lagrangian_field(vec, qi, n_part):
    """Scatter a per-particle vector onto the (n_part^3, 3) Lagrangian lattice."""
    out = np.zeros((n_part, n_part, n_part, 3), np.float64)
    out[qi[:, 0], qi[:, 1], qi[:, 2], :] = np.asarray(vec, np.float64)
    return out


def shell_transfer_vec(f_t, f_m, box_size, n_bins=None):
    """Per-shell sqrt(P_t/P_m) and r, summed over the three vector components.

    Component-summed rather than per-component: the force is a vector field and
    only its total power is basis-independent.
    """
    n = f_m.shape[0]
    kf = 2.0 * np.pi / box_size
    k1 = 2.0 * np.pi * np.fft.fftfreq(n, d=box_size / n)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=box_size / n)
    kmag = np.sqrt(k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + kz[None, None, :] ** 2)
    idx = np.rint(kmag / kf).astype(int)
    nb = n_bins or (n // 2)
    p_tt = np.zeros(nb + 1)
    p_mm = np.zeros(nb + 1)
    p_tm = np.zeros(nb + 1)
    for c in range(3):
        tk = np.fft.rfftn(f_t[..., c])
        mk = np.fft.rfftn(f_m[..., c])
        sel = idx <= nb
        np.add.at(p_tt, idx[sel], (np.abs(tk) ** 2)[sel])
        np.add.at(p_mm, idx[sel], (np.abs(mk) ** 2)[sel])
        np.add.at(p_tm, idx[sel], np.real(tk * np.conj(mk))[sel])
    with np.errstate(invalid="ignore", divide="ignore"):
        amp = np.sqrt(p_tt / p_mm)
        r = p_tm / np.sqrt(p_tt * p_mm)
    return amp, r, p_mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="smoke")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default=None, help="T:b pairs in FINE cells, comma separated")
    ap.add_argument("--nbins", type=int, default=12)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    import v2_g3_core as g3
    import v2_g3_ladder as lad
    from v2_g3_stage5 import DEFAULT_ARMS
    from v2_g5_core import force_global, padded_size

    B = lad.build(args.config, args.seed)
    g = B["g"]
    ell, n_fine, n_part = g["L"], g["n_fine"], g["n_part"]
    kf = 2.0 * np.pi / ell
    q, psi1, psi2, qi = B["q"], B["psi1"], B["psi2"], B["qi"]

    # Evaluate every arm at the SAME configuration, the background's own final
    # positions. Any difference is then the force operator alone -- no evolution,
    # no divergence of trajectories, nothing to attribute except the solve.
    x_eval = np.mod(np.asarray(g3.x_lpt(q, psi1, psi2, B["d_final"]), np.float64), ell)
    g_mono, _ = force_global(jnp.asarray(x_eval), n_fine, ell, n_part**3, "mono", assign="cic")
    g_mono = np.asarray(g_mono, np.float64)
    f_mono = lagrangian_field(g_mono, qi, n_part)

    arms = ([tuple(int(v) for v in a.split(":")) for a in args.arms.split(",")]
            if args.arms else list(DEFAULT_ARMS[args.config]))
    print(f"\n  config {args.config}  L={ell}  n_fine={n_fine}  n_part={n_part}  "
          f"seed {args.seed}")
    print(f"  evaluating the force at the background's final positions "
          f"(no evolution)\n")

    rows = []
    for n_tile, b_fine in arms:
        p_side, b_real = padded_size(n_tile, b_fine, n_fine=n_fine)
        tiles, _, _ = g3.lagrangian_tiles(qi, n_part, n_fine, n_tile, b_real)
        g_tile = np.zeros_like(g_mono)
        filled = np.zeros(len(g_mono), np.int64)
        for tijk, idx, cim in tiles:
            force, _, _ = g3.make_tile_force(
                tijk, n_tile, b_fine, n_fine, ell, n_part**3, core_in_member=None)
            gt = np.asarray(force(jnp.asarray(x_eval[idx])), np.float64)
            core = idx[np.asarray(cim)]
            g_tile[core] = gt[np.asarray(cim)]
            filled[core] += 1
        assert filled.min() == 1 and filled.max() == 1, "tile partition broken"

        f_tile = lagrangian_field(g_tile, qi, n_part)
        amp, r, _ = shell_transfer_vec(f_tile, f_mono, ell, n_bins=args.nbins)
        # the padded tile's own fundamental, in units of the BOX fundamental
        x_p = float(n_fine) / float(p_side)
        print(f"  T={n_tile:4d} b={b_fine:3d} fine  ({b_fine * ell / n_fine:5.1f} Mpc/h)  "
              f"P={p_side:4d}  padded-tile fundamental = {x_p:.2f} k_f")
        print(f"     {'k/k_f':>7s} " + "".join(f"{k:>8d}" for k in range(1, args.nbins + 1)))
        print(f"     {'amp':>7s} " + "".join(f"{amp[k]:8.3f}" for k in range(1, args.nbins + 1)))
        print(f"     {'r':>7s} " + "".join(f"{r[k]:8.3f}" for k in range(1, args.nbins + 1)))
        print()
        rows.append(dict(n_tile=int(n_tile), b_fine=int(b_fine), p_side=int(p_side),
                         b_mpc=float(b_fine) * ell / n_fine,
                         tile_fundamental_kf=x_p,
                         k_over_kf=list(range(1, args.nbins + 1)),
                         amp=[float(amp[k]) for k in range(1, args.nbins + 1)],
                         r=[float(r[k]) for k in range(1, args.nbins + 1)]))

    print("  amp = sqrt(P_tileforce / P_monoforce) per shell on the Lagrangian grid.")
    print("  PREDICTION UNDER TEST: amp collapses below the padded-tile fundamental")
    print("  and tracks 1 above it. Flat loss, or a knee elsewhere, refutes it.")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"g3_tile_force_scale_{args.config}.json")
    with open(out, "w") as fh:
        json.dump(dict(config=args.config, seed=args.seed, L=ell, n_fine=n_fine,
                       n_part=n_part, kf=kf, rows=rows), fh, indent=1)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
