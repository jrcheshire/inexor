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


def _tile_picks(n_side, n_tiles):
    """The sampled tile indices. Factored out so R2 and R3 sample the SAME tiles.

    If the two statistics drew different tiles their ratio would mix a genuine
    frame effect with tile-to-tile scatter, which at n_tiles = 4 is large (the
    cdev T=32 rel_max ladder is non-monotone in b).
    """
    picks = []
    step = max(1, n_side // n_tiles)
    for i in range(0, n_side, step):
        picks.append((i, (i * 2) % n_side, (i * 3) % n_side))
        if len(picks) >= n_tiles:
            break
    return picks


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

    picks = _tile_picks(n_side, n_tiles)

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
# R3: the same error measured in the COLA residual force
# ===========================================================================


def _core_and_members(qi, n_part, n_fine, n_tile, b_real, tijk):
    """(core mask, member index) for one tile. Lagrangian, so FIELD-INDEPENDENT.

    Both R3 arms must use the identical particle set: the tile geometry is the
    thing held fixed, and the only difference between the arms is the
    configuration the force is solved on.
    """
    t_lag = n_tile * n_part // n_fine
    b_lag = int(np.ceil(b_real * n_part / n_fine))
    blk = qi // t_lag
    core = np.ones(qi.shape[0], bool)
    member = np.ones(qi.shape[0], bool)
    for j in range(3):
        core &= blk[:, j] == tijk[j]
        lo = tijk[j] * t_lag - b_lag
        member &= ((qi[:, j] - lo) % n_part) < (t_lag + 2 * b_lag)
    return core, np.where(member)[0], b_lag


def force_fidelity_cola(x_fin, kick_lpt, bcoef, qi, g_mono, g, n_tile, b_fine,
                        n_tiles=R2_TILES):
    """R3: tile-vs-mono error in the NET COLA KICK, on core particles.

    R2 compares the raw force, but the raw force is not what an sCOLA tile
    integrates. cola_step_bullfrog (v2_g3_core.py:242) applies

        u <- alpha u + bcoef g + c_k1 Psi1 + c_k2 Psi2

    so the quantity the residual state actually receives is the WHOLE
    right-hand side, and the question R2 cannot answer is whether the analytic
    Psi terms cancel the part of g that a truncated tile gets wrong. Here
    kick_lpt = c_k1 Psi1 + c_k2 Psi2 is passed in from the real BullFrog
    coefficients rather than reconstructed, and the statistic is

        rms(bcoef (g_tile - g_mono)) / rms(bcoef g_mono + kick_lpt).

    Note what did NOT change: the numerator is still the raw force difference,
    because Psi1 and Psi2 are sliced from the GLOBAL frame and are bit-identical
    between the arms. Only the denominator changes. That is the whole content of
    the test -- if the frame cancels most of g, the same absolute force error is
    a LARGER fraction of the kick, not a smaller one, and sCOLA is in worse
    shape than R2 suggested rather than better.

    A first attempt defined the residual as g - force(x_LPT), solving the LPT
    configuration on the same padded tile. That is a different quantity: the
    Psi terms are displacement fields (the LINEAR force), not the force on the
    LPT-displaced configuration, and the two agree only while the field is
    linear. The vacuity guard below caught it at smoke (frame_cancel 0.9997).

    VACUITY GUARD: frame_cancel = rms(kick)/rms(bcoef g), reported PER TILE.
    Near 1 means the analytic terms cancel nothing, R3 carries no information
    R2 did not, and NEITHER a better nor a worse rel may be read as a result.
    """
    n_fine, n_part, box_size = g["n_fine"], g["n_part"], g["L"]
    n_global = n_part**3
    p_side, b_real = padded_size_cached(n_tile, b_fine, n_fine)
    n_side = n_fine // n_tile

    rel_cola, rel_raw, frame_cancel = [], [], []
    uniform_frac, rel_fluc = [], []
    for tijk in _tile_picks(n_side, n_tiles):
        core, idx, _ = _core_and_members(qi, n_part, n_fine, n_tile, b_real, tijk)
        core_in_member = core[idx]
        g_tile, _, _, _ = tile_force_mono(
            x_fin[idx], tijk, n_tile, b_fine, n_fine, box_size, n_global
        )
        d_raw = g_tile[core_in_member] - g_mono[core]
        kick_mono = bcoef * g_mono[core] + kick_lpt[core]

        rms_kick = float(np.sqrt((kick_mono**2).sum(axis=1).mean()))
        rms_g = float(np.sqrt((g_mono[core] ** 2).sum(axis=1).mean()))
        rel_cola.append(float(np.sqrt(((bcoef * d_raw) ** 2).sum(axis=1).mean())) / rms_kick)
        rel_raw.append(float(np.sqrt((d_raw**2).sum(axis=1).mean())) / rms_g)
        frame_cancel.append(rms_kick / (abs(bcoef) * rms_g))

        # The part of the error that a TRANSLATION-INVARIANT statistic can see.
        # A force error uniform across the core moves the whole tile rigidly, and
        # R_Q (a bispectrum ratio) cannot see a rigid translation. Reported as a
        # decomposition, NOT as a replacement for rel_raw: the per-tile uniform
        # terms differ between tiles, so they still move tiles relative to one
        # another and land in the large-scale field that D-v2-9 gates separately.
        d_bar = d_raw.mean(axis=0)
        g_bar = g_mono[core].mean(axis=0)
        rms_d = float(np.sqrt((d_raw**2).sum(axis=1).mean()))
        uniform_frac.append(float(np.sqrt((d_bar**2).sum())) / rms_d if rms_d > 0 else 0.0)
        rel_fluc.append(
            float(np.sqrt(((d_raw - d_bar) ** 2).sum(axis=1).mean()))
            / float(np.sqrt(((g_mono[core] - g_bar) ** 2).sum(axis=1).mean()))
        )

    return dict(
        n_tile=int(n_tile),
        b_requested=int(b_fine),
        b_realized=int(b_real),
        padded_P=int(p_side),
        n_tiles_sampled=len(rel_cola),
        rel_cola_median=float(np.median(rel_cola)),
        rel_cola_max=float(np.max(rel_cola)),
        rel_raw_median=float(np.median(rel_raw)),
        rel_raw_max=float(np.max(rel_raw)),
        frame_cancel_median=float(np.median(frame_cancel)),
        uniform_frac_median=float(np.median(uniform_frac)),
        rel_fluc_median=float(np.median(rel_fluc)),
        rel_fluc_max=float(np.max(rel_fluc)),
        rel_cola_all=[float(v) for v in rel_cola],
        rel_raw_all=[float(v) for v in rel_raw],
        rel_fluc_all=[float(v) for v in rel_fluc],
        uniform_frac_all=[float(v) for v in uniform_frac],
        vol_ratio=(p_side / n_tile) ** 3,
        **_degeneracy(p_side, n_fine, n_tile),
    )


def padded_size_cached(n_tile, b_fine, n_fine):
    from v2_g5_core import padded_size

    return padded_size(n_tile, b_fine, n_fine=n_fine)


def identity_cola_one_tile_is_box(x_fin, kick_lpt, bcoef, g_mono, n_fine, box_size,
                                  n_global, tol=1e-10):
    """R3's degenerate limit: at T = n_fine, b = 0 the tile IS the box, so the
    kick error must be identically zero.

    R2 has this guard already, but R3 divides by a DIFFERENT denominator that is
    itself a computed quantity. A kick_lpt built at the wrong step's
    coefficients, or with Psi in the wrong units, would leave the numerator
    correct and silently rescale every row. This guard would not catch that on
    its own -- it is the frame_cancel column that has to be read for it -- so
    the denominator is also asserted finite and non-degenerate here.
    """
    g_tile, p_side, b_real, n_out = tile_force_mono(
        x_fin, (0, 0, 0), n_fine, 0, n_fine, box_size, n_global
    )
    kick = bcoef * g_mono + kick_lpt
    rms_kick = float(np.sqrt((kick**2).sum(axis=1).mean()))
    rel = float(np.sqrt(((bcoef * (g_tile - g_mono)) ** 2).sum(axis=1).mean())) / rms_kick
    ok = bool(
        rel < tol and p_side == n_fine and b_real == 0 and n_out == 0
        and np.isfinite(rms_kick) and rms_kick > 0.0
    )
    return dict(
        rel=rel, tol=tol, p_side=int(p_side), b_realized=int(b_real), n_outside=n_out,
        rms_kick=rms_kick, ok=ok
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
    ap.add_argument("--skip-r3", action="store_true", help="no COLA-residual force legs")
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

    # The analytic half of the COLA kick, at the LAST step -- the one whose force
    # evaluation sits closest to x_fin, where g_mono is measured. Taken from
    # bullfrog_cola_coeffs rather than rebuilt, so R3 cannot drift from the
    # stepper it is meant to describe.
    cola_c = g3.bullfrog_cola_coeffs(table)
    _, _, bcoef_k, _, c_k1, c_k2, _ = (float(v) for v in cola_c[-1])
    kick_lpt = c_k1 * psi1 + c_k2 * psi2

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

    r3, ident_cola = [], None
    if not args.skip_r3:
        ident_cola = identity_cola_one_tile_is_box(
            x_fin, kick_lpt, bcoef_k, g_mono, n_fine, L, n_global
        )
        print("\n=== R3 degenerate limit: net kick, one tile == the whole box ===")
        print(f"  rel = {ident_cola['rel']:.3e} (tol {ident_cola['tol']:.0e})   "
              f"ok = {ident_cola['ok']}")
        if not ident_cola["ok"]:
            print("  REFUSING to report R3: the single-tile residual force does not")
            print("  reproduce the monolithic residual force where they are the same solve.")
            sys.exit(1)

        gc_all = float(
            np.sqrt(((bcoef_k * g_mono + kick_lpt) ** 2).sum(axis=1).mean())
            / (abs(bcoef_k) * np.sqrt((g_mono**2).sum(axis=1).mean()))
        )
        print("\n=== R3 net COLA kick: tile vs monolithic on CORE particles ===")
        print(f"  whole-box frame cancellation rms|kick| / rms|bcoef F| = {gc_all:.4f}")
        print("  (near 1 => the frame cancels nothing here and R3 has no power over R2;")
        print("   well BELOW 1 => the same force error is a LARGER fraction of the kick)")
        print(f"  {'T':>5s} {'b':>4s} {'b Mpc/h':>8s} {'P':>5s} {'vol x':>7s} "
              f"{'rel med':>9s} {'rel max':>9s} {'raw max':>9s} {'cancel':>7s} "
              f"{'unif':>6s} {'fluc med':>9s}")
        for t in cores:
            for b in BUF_FINE:
                try:
                    padded_size(t, b, n_fine=n_fine)
                except ValueError:
                    continue
                rec = force_fidelity_cola(x_fin, kick_lpt, bcoef_k, qi, g_mono, g, t, b)
                r3.append(rec)
                flag = (
                    "  DEGENERATE (tile == box, 0 by construction)"
                    if rec["degenerate"]
                    else ("  near-deg" if rec["near_degenerate"] else "")
                )
                print(
                    f"  {t:5d} {b:4d} {rec['b_realized'] * cell:8.1f} {rec['padded_P']:5d} "
                    f"{rec['vol_ratio']:7.2f} {rec['rel_cola_median']:9.4f} "
                    f"{rec['rel_cola_max']:9.4f} {rec['rel_raw_max']:9.4f} "
                    f"{rec['frame_cancel_median']:7.4f} {rec['uniform_frac_median']:6.3f} "
                    f"{rec['rel_fluc_median']:9.4f}{flag}"
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
                identity_cola_one_tile_is_box=ident_cola,
                cola_force_fidelity=r3,
            ),
            f,
            indent=2,
        )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
