"""Rung H2: what buffer does the tiled SHORT-range force actually need?

THE QUESTION. The failing arm needed a buffer that grew with the box, because
it asked each tile for the whole 1/k^2 range and no buffer can supply a mode
longer than the padded tile. With the long range moved to a global coarse solve
the tile only owes the SHORT kernel, whose reach is set by the split scale r_s.
If the buffer requirement is set by r_s and not by the box, tiling is cheap; if
it still grows with the box, the proposal has no cost case.

THE ESTIMAND IS THE TILING ERROR ALONE. Comparing the hybrid against the
monolithic force would floor the measurement at the coarse solve's own ~1e-2
representation error and hide the thing being measured. So the reference here
is the UNTILED short force on the same fine mesh, `force_global(which="short")`,
and the estimand is

    rel(b) = rms|g_short_tiled(T,b) - g_short_exact| / rms|g_short_exact|

which is zero when tiling is exact and knows nothing about the coarse mesh.

HOW BOX-INDEPENDENCE IS TESTED, and why a one-box scan cannot do it. cdev8 and
cdev share the fine cell (0.25 Mpc/h) and the coarse cell (1.0 Mpc/h), so at
fixed alpha they share r_s exactly. Running the IDENTICAL (T, b) ladder at both
therefore varies the box and nothing else: same tile geometry, same padded size,
same kernel, 8x the volume. If rel(b) agrees between them the requirement is
box-independent; if it does not, it is not. This avoids arguing from a
curve-collapse over axes that were never matched.

SCAN b AT FIXED T, and read both axes. b and P = T + 2b move together, so a
buffer scan cannot separate an erfc-in-b/r_s decay from the recorded 2.0/P
power law in the padded size. Both are printed; the box comparison is what
carries the conclusion.

Usage:
    pixi run python scripts/v2_g3_buffer_criterion.py --config cdev8 --tile 64
    pixi run python scripts/v2_g3_buffer_criterion.py --config cdev  --tile 64
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
    ap.add_argument("--buffers", default="8,16,24,32,48,64",
                    help="buffer ladder in FINE cells")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--family", default="gauss")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    import v2_g3_core as g3
    import v2_g3_ladder as lad
    from v2_g5_core import force_global, force_short_tiled, padded_size

    B = lad.build(args.config, args.seed)
    g = B["g"]
    ell, n_fine, n_part = g["L"], g["n_fine"], g["n_part"]
    d_c, d_f = g["coarse_cell"], g["fine_cell"]
    n_tot = n_part**3
    r_s = args.alpha * d_c

    x = np.mod(np.asarray(
        g3.x_lpt(B["q"], B["psi1"], B["psi2"], B["d_final"]), np.float64), ell)

    # the untiled short force: the only reference that isolates tiling error
    g_ref, _ = force_global(jnp.asarray(x), n_fine, ell, n_tot, "short",
                            family=args.family, r_s=r_s, n_ref=n_fine, assign="cic")
    g_ref = np.asarray(g_ref, np.float64)
    rms_ref = float(np.sqrt((g_ref**2).sum(axis=1).mean()))

    print(f"\n  config {args.config}   L={ell}  n_fine={n_fine}  fine cell={d_f}  "
          f"coarse cell={d_c}")
    print(f"  r_s = {r_s:.4f} Mpc/h (alpha={args.alpha})   tile T={args.tile} fine cells "
          f"= {args.tile * d_f:.1f} Mpc/h")
    print(f"  reference = UNTILED short force, rms |g| = {rms_ref:.4e}\n")
    print(f"    {'b[fine]':>8s} {'b[Mpc/h]':>9s} {'b/r_s':>7s} {'P':>5s} "
          f"{'2.0/P':>8s} {'rel rms':>10s} {'rel max':>10s} {'wall[s]':>8s}")

    rows = []
    for b_fine in [int(v) for v in args.buffers.split(",")]:
        p_side, b_real = padded_size(args.tile, b_fine, n_fine=n_fine)
        if p_side > n_fine:
            print(f"    {b_fine:8d}  padded box {p_side} exceeds the mesh -- skipped")
            continue
        t0 = time.perf_counter()
        gs, _ = force_short_tiled(x, n_fine, ell, n_tot, args.tile, b_fine,
                                  family=args.family, r_s=r_s)
        wall = time.perf_counter() - t0
        d = np.asarray(gs, np.float64) - g_ref
        rel_rms = float(np.sqrt((d**2).sum(axis=1).mean()) / rms_ref)
        rel_max = float(np.abs(d).max() / np.abs(g_ref).max())
        b_mpc = b_real * d_f
        print(f"    {b_real:8d} {b_mpc:9.2f} {b_mpc / r_s:7.1f} {p_side:5d} "
              f"{2.0 / p_side:8.4f} {rel_rms:10.3e} {rel_max:10.3e} {wall:8.1f}")
        rows.append(dict(b_fine=int(b_real), b_mpc=float(b_mpc),
                         b_over_rs=float(b_mpc / r_s), p_side=int(p_side),
                         rel_rms=rel_rms, rel_max=rel_max, wall=float(wall)))

    print("\n  rel = tiled short force vs UNTILED short force. Zero means the tiling")
    print("  is exact; the coarse solve is not involved and cannot floor this.")
    print("  BOX-INDEPENDENCE IS READ ACROSS CONFIGS, not off one table: run the same")
    print("  T and b ladder at cdev8 and cdev (identical fine cell, coarse cell and")
    print("  r_s) and compare rel(b) row by row.")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"g3_buffer_criterion_{args.config}_T{args.tile}.json")
    with open(out, "w") as fh:
        json.dump(dict(config=args.config, seed=args.seed, tile=args.tile,
                       alpha=args.alpha, family=args.family, r_s=r_s, L=ell,
                       n_fine=n_fine, fine_cell=d_f, coarse_cell=d_c,
                       rms_ref=rms_ref, rows=rows), fh, indent=1)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
