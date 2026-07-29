"""G3 pricing probe: what buffer does an sCOLA tile need, and what does A3 cost?

Run BEFORE building the tile evolve machinery, because it prices the premise.
The design study's "A3 costs ~3.2x compute" implies b = 0.24 T (padded-volume
ratio 3.2 -> linear 1.47). Leclercq 2003.04925's recommended tile >= 50 /
buffer >= 25 Mpc/h is b = 0.5 T, i.e. 8x. Those differ by 2.5x in compute and
nothing in the repo measures which is right. This does.

TWO SEPARATE REQUIREMENTS set the buffer. The G3 plan conflated them.

  R1 CONTAINMENT. Every CORE particle must remain inside the padded box for the
     whole run. Otherwise it is aliased to the far side of the periodic padded
     box (tile_local_coords is exact mod P), its density lands in the wrong cell
     and tile_gather_vector reads the force there -- a wrong answer, not a
     crash. Pure geometry: needs a trajectory, no force solves.

  R2 FORCE FIDELITY. The mass that should be near a core particle must actually
     be present. Mass currently near the core wall came from Lagrangian
     positions within ~|displacement| of it, so this is a Lagrangian shell-depth
     requirement. Needs one padded-box force solve per (T, b).

WHY THE PLAN'S ARGUMENT WAS WRONG. It claimed "the only migration is the
residual y, which is small by construction". MEASURED at smoke (32^3, L=32,
a=1), L-infinity excursion in Mpc/h:

    total displacement   p50 3.39   p99 8.80   p999 10.22   max 11.26
    minus block mean     p50 2.27   p99 6.06   p999  8.57   max 10.48
    residual y           p50 0.77   p99 5.70   p999  7.81   max 10.39

y IS 4.4x smaller at the MEDIAN and only 1.1-1.3x smaller in the TAIL, and
containment is a TAIL requirement. So the residual-based optimism does not
survive. The same table settles a design fork: a frame-following (moving-origin)
padded box buys 16% at p999 and 7% at max, because the excursion is dominated by
internal shear within the tile rather than bulk motion, which no rigid box
motion removes. Hence FIXED ORIGIN -- the simpler choice -- and the buffer is
sized on the total-displacement tail.

Consequence for the cost model, and it is the useful output: the containment
buffer is an ABSOLUTE physical scale (set by the displacement tail, roughly
box-independent once the box holds the relevant bulk flows), so b/T shrinks as
the tile grows and the cost follows arithmetically:

    b = 12 Mpc/h ->  T=8: 64x   T=16: 15.6x   T=32: 5.8x   T=50: 3.4x   T=64: 2.7x

which is exactly the tension G3 has to price: bigger tiles are cheaper per
volume but give less memory saving AND less lever arm (n_fine/P smaller, so
fewer long modes below the tile fundamental to discriminate on).

smoke saturates and CANNOT answer this: a 12 Mpc/h shell around an 8 Mpc/h core
swallows the whole 32 Mpc/h box, which is why its force-fidelity error reads
exactly 0.0000 there rather than convergently small. Run cdev8 (L=64) and cdev
(L=128); cgh64 (L=256) is the only config that reaches the literature geometry.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from v2_g5_two_level_force import A_CONTROL, A_PIVOT, N_STEPS, geometry  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(HERE), "runs", "v2")

# Core sizes to price, in FINE cells. Filtered per config against the degeneracy
# guard (a padded tile >= the fine mesh is not a tile).
CORE_FINE = (32, 64, 128, 256)
# Buffer ladder, in FINE cells. Reaches 48 Mpc/h at the pinned 0.25 Mpc/h cell,
# so Leclercq's >= 25 Mpc/h recommendation is inside the ladder rather than an
# extrapolation off its end. padded_size rejects the combinations that exceed the
# fine mesh, so over-long entries are skipped rather than erroring.
BUF_FINE = (0, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192)
# Tiles sampled for R2, as a fraction along each axis. Four spread through the
# box rather than one: a single tile's force error is one draw.
R2_TILES = 4


# ===========================================================================
# the single-tile full-kernel force
# ===========================================================================


def tile_force_mono(pos_members, tijk, n_tile, b_fine, n_fine, box_size, n_global):
    """Full ik/k^2 force on ONE padded tile box, from the tile's own particles.

    Routed through force_global on the PADDED box rather than through a
    hand-rolled solve, so the tile arm and the monolithic arm share the identical
    kernel + paint + gather code path and the comparison is a test of the
    MISSING MASS AND THE PERIODIZATION, not of two harnesses. This is the same
    reasoning v2_g5_core's floor F1 is built on.

    Three conversions make that work:
      - positions are tile-LOCAL (mod(pos - origin, L)), so the padded box is
        the periodic box force_global assumes;
      - box_size becomes P * cell, n_mesh becomes P;
      - n_total is rescaled to n_global * (P/n_fine)^3 so density_f64's
        mean = n_total/n_mesh^3 stays the GLOBAL mean. Using the tile's own
        particle count instead rescales the force by mean_global/mean_tile, an
        O(1) error that reads as a catastrophic tiling failure
        (tile_paint_f64's docstring records this).

    Returns ((n_members, 3) f64 force, P, b_realized, n_outside).
    """
    import jax.numpy as jnp

    from v2_g5_core import force_global, padded_size, tile_local_coords, tile_origin_extent

    cell = box_size / n_fine
    p_side, b_realized = padded_size(n_tile, b_fine, n_fine=n_fine)
    origin, extent = tile_origin_extent(tijk, n_tile, b_realized, cell)
    u = np.asarray(tile_local_coords(jnp.asarray(pos_members), origin, box_size), np.float64)
    n_outside = int((u >= extent).any(axis=1).sum())
    n_total_arg = float(n_global) * (float(p_side) / float(n_fine)) ** 3
    g, _ = force_global(
        jnp.asarray(u), p_side, p_side * cell, n_total_arg, "mono", assign="cic"
    )
    return np.asarray(g, np.float64), p_side, b_realized, n_outside


# ===========================================================================
# the degenerate-limit identity -- runs FIRST, gates everything
# ===========================================================================


def identity_one_tile_is_box(x, g_mono, n_fine, box_size, n_global, tol=1e-10):
    """One tile with T = n_fine, b = 0 IS the whole box: must reproduce g_mono.

    P = n_fine, origin = 0, so tile_local_coords is the identity and the padded
    solve is literally the monolithic solve. A check that did not run reads
    exactly like one that passed, so this runs before any pricing number and
    hard-fails the probe.
    """
    g_tile, p_side, b_real, n_out = tile_force_mono(
        x, (0, 0, 0), n_fine, 0, n_fine, box_size, n_global
    )
    rms = float(np.sqrt((g_mono**2).sum(axis=1).mean()))
    d = g_tile - g_mono
    rel = float(np.sqrt((d**2).sum(axis=1).mean())) / rms
    ok = bool(rel < tol and p_side == n_fine and b_real == 0 and n_out == 0)
    return dict(
        rel=rel, tol=tol, p_side=int(p_side), b_realized=int(b_real), n_outside=n_out, ok=ok
    )


# ===========================================================================
# R1 / R2
# ===========================================================================


def lagrangian_index(q, n_part, box_size):
    """Integer Lagrangian grid index of each particle, C-order-consistent."""
    spacing = box_size / n_part
    return np.rint(np.asarray(q, np.float64) / spacing).astype(np.int64) % n_part


def containment(x_fin, qi, g, n_tile, b_fine):
    """R1: fraction of CORE particles that leave their padded box.

    Counted over ALL tiles and ALL particles -- not a sampled subset, so this is
    not one draw.

    VECTORIZED, one pass over particles. Every particle already knows which tile
    owns it (its Lagrangian block), so its own box origin follows from that block
    and the whole question is a single mod-and-compare. The obvious loop-over-
    tiles-masking-all-particles form is O(n_tiles x n_particles), which is merely
    slow at cdev (4096 tiles x 16.7M particles per row) but impossible at cgh64
    (32768 x 134M). Same arithmetic, same answer.
    """
    from v2_g5_core import padded_size

    n_fine, n_part, box_size = g["n_fine"], g["n_part"], g["L"]
    cell = box_size / n_fine
    p_side, b_real = padded_size(n_tile, b_fine, n_fine=n_fine)
    t_lag = n_tile * n_part // n_fine
    blk = qi // t_lag
    # Each particle's own tile origin, then its own tile-local coordinate. This
    # mirrors tile_origin_extent + tile_local_coords exactly, per particle.
    org = blk.astype(np.float64) * (n_tile * cell) - b_real * cell
    loc = np.mod(x_fin - org, box_size)
    n_out = int((loc >= p_side * cell).any(axis=1).sum())
    n_core = int(qi.shape[0])
    return dict(
        n_tile=int(n_tile),
        b_requested=int(b_fine),
        b_realized=int(b_real),
        padded_P=int(p_side),
        n_out_core=n_out,
        frac_out=n_out / n_core,
        vol_ratio=(p_side / n_tile) ** 3,
        **_degeneracy(p_side, n_fine, n_tile),
    )


def _degeneracy(p_side, n_fine, n_tile):
    """Flag rows where the "tile" is not meaningfully a tile.

    A row with padded_P == n_fine has the tile solving the WHOLE BOX, so its
    force error is exactly 0 by construction -- convergence-looking and
    completely uninformative. p_frac is reported so a nearly-degenerate row
    (P/n_fine -> 1) cannot be read as a converged measurement either. Both smoke
    and cdev8 saturate this way for every configuration that reaches a few-percent
    force error, which is why the COST factor has to be quoted at a larger box.
    """
    p_frac = float(p_side) / float(n_fine)
    return dict(
        p_frac=p_frac,
        n_tiles_total=(int(n_fine) // int(n_tile)) ** 3,
        degenerate=bool(p_side >= n_fine),
        near_degenerate=bool(p_frac > 0.5),
    )


def force_fidelity(x_fin, qi, g_mono, g, n_tile, b_fine, n_tiles=R2_TILES):
    """R2: tile force vs monolithic force on CORE particles, over n_tiles tiles."""
    from v2_g5_core import padded_size

    n_fine, n_part, box_size = g["n_fine"], g["n_part"], g["L"]
    n_global = n_part**3
    p_side, b_real = padded_size(n_tile, b_fine, n_fine=n_fine)
    t_lag = n_tile * n_part // n_fine
    b_lag = int(np.ceil(b_real * n_part / n_fine))
    n_side = n_fine // n_tile
    blk = qi // t_lag

    picks = []
    step = max(1, n_side // n_tiles)
    for i in range(0, n_side, step):
        picks.append((i, (i * 2) % n_side, (i * 3) % n_side))
        if len(picks) >= n_tiles:
            break

    rels = []
    for tijk in picks:
        core = np.ones(qi.shape[0], bool)
        member = np.ones(qi.shape[0], bool)
        for j in range(3):
            core &= blk[:, j] == tijk[j]
            # Lagrangian shell of depth b_lag around the core block, periodic.
            lo = tijk[j] * t_lag - b_lag
            member &= ((qi[:, j] - lo) % n_part) < (t_lag + 2 * b_lag)
        idx = np.where(member)[0]
        g_tile, _, _, n_out = tile_force_mono(
            x_fin[idx], tijk, n_tile, b_fine, n_fine, box_size, n_global
        )
        core_in_member = core[idx]
        d = g_tile[core_in_member] - g_mono[core]
        rms_core = float(np.sqrt((g_mono[core] ** 2).sum(axis=1).mean()))
        rels.append(float(np.sqrt((d**2).sum(axis=1).mean())) / rms_core)
    return dict(
        n_tile=int(n_tile),
        b_requested=int(b_fine),
        b_realized=int(b_real),
        b_lag=int(b_lag),
        padded_P=int(p_side),
        n_tiles_sampled=len(picks),
        rel_median=float(np.median(rels)),
        rel_max=float(np.max(rels)),
        rel_all=[float(r) for r in rels],
        vol_ratio=(p_side / n_tile) ** 3,
        **_degeneracy(p_side, n_fine, n_tile),
    )


# ===========================================================================
# main
# ===========================================================================


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="smoke", choices=("smoke", "cdev8", "cdev", "cgh64"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nsteps", type=int, default=N_STEPS)
    ap.add_argument("--skip-r2", action="store_true", help="containment only (no force solves)")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from inexor import ic, lpt
    from inexor.config import Cosmology
    from inexor.cosmology import growth_factor_a
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table
    from inexor.integrate import float_step_bullfrog

    import v2_g3_core as g3
    from v2_g5_core import force_global, padded_size

    g = geometry(args.config)
    L, n_fine, n_part = g["L"], g["n_fine"], g["n_part"]
    n_global, cell = n_part**3, g["fine_cell"]
    cosmo = Cosmology()

    print(f"=== G3 buffer cost: {args.config} ===")
    print(f"  L = {L} Mpc/h, n_part = {n_part}, n_fine = {n_fine}, fine cell = {cell} Mpc/h")

    t0 = time.perf_counter()
    d0 = ic.linear_density(
        jax.random.PRNGKey(args.seed), n_part, L, cosmo, f_NL=0.0, fdtype=jnp.float64
    )
    x_ic, v_ic = lpt.lpt_ics(d0, L, A_CONTROL, cosmo, order=2, fdtype=jnp.float64)
    q, psi1, psi2 = g3.lpt_frame(d0, L)
    d_final = growth_factor_a(A_PIVOT, cosmo)
    table = bullfrog_table(a_grid(A_CONTROL, A_PIVOT, args.nsteps, "log"), cosmo)

    def force_mono(pos):
        out, _ = force_global(pos, n_fine, L, n_global, "mono", assign="cic")
        return jnp.asarray(out)

    xd, vd = jnp.asarray(x_ic, jnp.float64), jnp.asarray(v_ic, jnp.float64)
    for c in bullfrog_float_coeffs(table):
        xd, vd = float_step_bullfrog(xd, vd, tuple(np.asarray(c, np.float64)), force_mono, L)
    x_fin = np.asarray(xd, np.float64)
    print(f"  monolithic evolve: {time.perf_counter() - t0:.1f} s")

    g_mono = np.asarray(force_mono(jnp.asarray(x_fin, jnp.float64)), np.float64)
    qi = lagrangian_index(q, n_part, L)

    # ---- degenerate limit FIRST -------------------------------------------
    ident = identity_one_tile_is_box(x_fin, g_mono, n_fine, L, n_global)
    print("\n=== degenerate limit: one tile (T = n_fine, b = 0) == the whole box ===")
    print(f"  rel = {ident['rel']:.3e} (tol {ident['tol']:.0e})   ok = {ident['ok']}")
    if not ident["ok"]:
        print("  REFUSING to report pricing: the single-tile force does not reproduce")
        print("  the monolithic force in the limit where they are the same solve.")
        sys.exit(1)

    # ---- excursion percentiles (the fork, and the absolute scale) ----------
    disp = g3.min_image(x_fin - q, L)
    y_fin = g3.min_image(x_fin - g3.x_lpt(q, psi1, psi2, d_final), L)
    blocks = [b for b in (2, 4, 8, 16, 32) if n_part % b == 0 and b < n_part]
    exc = {}
    for name, arr in (("total", disp), ("residual_y", y_fin)):
        e = np.abs(arr).max(axis=1)
        exc[name] = {
            k: float(np.percentile(e, v)) for k, v in (("p50", 50), ("p99", 99), ("p999", 99.9))
        }
        exc[name]["max"] = float(e.max())
    print("\n=== L-infinity excursion (Mpc/h) -- what the buffer must cover ===")
    for name in ("total", "residual_y"):
        d = exc[name]
        print(
            f"  {name:11s} p50 {d['p50']:6.2f}  p99 {d['p99']:6.2f}  "
            f"p999 {d['p999']:6.2f}  max {d['max']:6.2f}"
        )

    cores = [c for c in CORE_FINE if c < n_fine and n_fine % c == 0]
    r1, r2 = [], []
    print("\n=== R1 containment: core particles leaving the padded box ===")
    print(f"  {'T':>5s} {'T Mpc/h':>8s} {'b':>4s} {'b Mpc/h':>8s} {'P':>5s} {'vol x':>7s} {'frac_out':>10s}")
    for t in cores:
        for b in BUF_FINE:
            try:
                padded_size(t, b, n_fine=n_fine)
            except ValueError:
                continue
            rec = containment(x_fin, qi, g, t, b)
            r1.append(rec)
            flag = "  DEGENERATE" if rec["degenerate"] else ("  near-deg" if rec["near_degenerate"] else "")
            print(
                f"  {t:5d} {t * cell:8.1f} {b:4d} {rec['b_realized'] * cell:8.1f} "
                f"{rec['padded_P']:5d} {rec['vol_ratio']:7.2f} {rec['frac_out']:10.2e}{flag}"
            )

    if not args.skip_r2:
        print("\n=== R2 force fidelity: tile force vs monolithic on CORE particles ===")
        print(f"  {'T':>5s} {'b':>4s} {'b Mpc/h':>8s} {'P':>5s} {'vol x':>7s} {'rel med':>9s} {'rel max':>9s}")
        for t in cores:
            for b in BUF_FINE:
                try:
                    padded_size(t, b, n_fine=n_fine)
                except ValueError:
                    continue
                rec = force_fidelity(x_fin, qi, g_mono, g, t, b)
                r2.append(rec)
                flag = (
                    "  DEGENERATE (tile == box, 0 by construction)"
                    if rec["degenerate"]
                    else ("  near-deg" if rec["near_degenerate"] else "")
                )
                print(
                    f"  {t:5d} {b:4d} {rec['b_realized'] * cell:8.1f} {rec['padded_P']:5d} "
                    f"{rec['vol_ratio']:7.2f} {rec['rel_median']:9.4f} {rec['rel_max']:9.4f}{flag}"
                )

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"g3_buffer_cost_{args.config}{args.out_suffix}.json")
    with open(path, "w") as f:
        json.dump(
            dict(
                config=args.config,
                seed=args.seed,
                nsteps=args.nsteps,
                geometry={k: (v if not isinstance(v, np.generic) else float(v)) for k, v in g.items()},
                identity_one_tile_is_box=ident,
                excursion_linf=exc,
                blocks=blocks,
                containment=r1,
                force_fidelity=r2,
            ),
            f,
            indent=2,
        )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
