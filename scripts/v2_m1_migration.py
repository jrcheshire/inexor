"""M-v2-1 exit gate: the migrant distribution and the slack the layout needs.

WHAT IS BEING MEASURED, AND WHY IT IS OWED. D-v2-14 clause 2 prices the T9
tier at 10.15 B/p all-in, of which 0.90 is "slack and arena" -- and the clause
says in as many words that this term is an ESTIMATE, from a hand argument about
migration rates, and that measuring it is an exit condition of the codec
milestone. 7.7 GB of C-gh's 87.2 rests on it, and the headroom against the
~116 GB LPDDR cliff (D-v2-13) is 1.33x, so a factor here is not cosmetic.

WHAT MAKES THIS DIFFERENT FROM THE UNIT TEST. `tests/test_layout.py` measures
the same quantities under an incoherent per-particle random walk, which is a
pessimistic proxy: real PM displacement is COHERENT, so neighbouring particles
leave a bucket together and arrivals are not Poisson. That test pins the
MECHANISM (integer slot granularity plus the occupancy spread) and explicitly
does not claim an operating point. This script runs the real dynamics -- the
ratified two-level force, the ratified geometry -- and reports the operating
point.

WHAT IT REPORTS, per step:
  - migrant fraction at BUCKET level (the exchange the layout performs) and at
    BRICK level (the coarser exchange the streaming path sees)
  - the bucket occupancy distribution: mean, max, and upper percentiles, plus
    how close the max comes to the index dtype's ceiling. NB that headroom is
    now REPORTING rather than a decision input: the index was widened to uint32
    (0.50 B/p) precisely so that no measurement here has to establish a bound on
    the far tail. The tail columns stay because the distribution is worth
    knowing, not because a dtype waits on them.
  - for a LADDER of slack targets, the arena demand that target leaves --
    computed exactly against a capacity frozen at build time, which is the
    policy the layout implements (build once, never resize)
  - the realized B/p at each target, so the answer is in the units D-v2-14
    clause 2 is written in

NOT MEASURED HERE, deliberately: anything about accuracy. This changes no
physics; the trajectory is the ratified force's and the layout is a bystander
that watches it. If a number here moves D-v2-14 clause 2, that is an amendment
to a ratified ADR and JC's call, not this script's.

Probe code, not package code. Run:
    python scripts/v2_m1_migration.py --config smoke
    python scripts/v2_m1_migration.py --config cgh64 --steps 20
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO, "runs", "v2")

# Matches scripts/v2_g5_two_level_force.py CONFIGS exactly -- the cgh64 row is
# C-gh at 1/64 volume, same fine cell and same spacing, which is what makes a
# measurement here transferable rather than a different regime.
CONFIGS = {
    "smoke": dict(n_part=32, L=32.0, n_fine=64, n_coarse=16),
    "cdev8": dict(n_part=128, L=64.0, n_fine=256, n_coarse=64),
    "cdev": dict(n_part=256, L=128.0, n_fine=512, n_coarse=128),
    "cgh64": dict(n_part=512, L=256.0, n_fine=1024, n_coarse=256),
}
A_INIT, A_FINAL, SPACING = 0.1, 1.0, "log"
SEED = 0
ALPHA = 1.0  # r_s / coarse_cell, the ratified kernel (gauss + TSC + matching)



def geometry(cfg):
    c = CONFIGS[cfg]
    g = dict(c)
    g["fine_cell"] = c["L"] / c["n_fine"]
    g["coarse_cell"] = c["L"] / c["n_coarse"]
    g["spacing"] = c["L"] / c["n_part"]
    g["n_total"] = c["n_part"] ** 3
    return g


def make_ics(g, seed):
    """2LPT ICs at a_init, f64 on the CPU -- the same recipe as the G5 driver's
    config leg, so the trajectory is the ratified one."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from inexor.config import Cosmology
    from inexor.ic import linear_density
    from inexor.lpt import lpt_ics

    cosmo = Cosmology()
    delta0 = linear_density(
        jax.random.PRNGKey(seed), g["n_part"], g["L"], cosmo, fdtype=jnp.float64
    )
    x0, v0 = lpt_ics(delta0, g["L"], A_INIT, cosmo, order=2, fdtype=jnp.float64)
    return np.asarray(x0, dtype=np.float64), np.asarray(v0, dtype=np.float64), cosmo


def make_force(g, tile, buf, family="gauss", assign="tsc", match=True):
    """The ratified two-level force: coarse global long-range plus the tiled
    short-range. Imported from the probe, unmodified -- D-v2-16 clause 7 keeps
    v2_g5_core.py the oracle, and this script is a consumer of it."""
    import jax.numpy as jnp

    sys.path.insert(0, HERE)
    from v2_g5_core import force_global, force_short_tiled

    r_s = ALPHA * g["coarse_cell"]

    def force(pos):
        pn = np.asarray(pos, dtype=np.float64)
        gl, _ = force_global(
            jnp.asarray(pn),
            g["n_coarse"],
            g["L"],
            g["n_total"],
            "long",
            family=family,
            r_s=r_s,
            n_ref=g["n_fine"],
            match=((g["coarse_cell"], g["fine_cell"]) if match else None),
            assign=assign,
        )
        gs, _ = force_short_tiled(
            pn, g["n_fine"], g["L"], g["n_total"], tile, buf, family=family, r_s=r_s
        )
        return jnp.asarray(gl + gs)

    return force


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


# Bucket counts above each threshold, kept as a description of the occupancy
# distribution.
#
# THESE NUMBERS NO LONGER DECIDE ANYTHING, and the history is worth carrying.
# They were added to settle whether a uint16 index (ceiling 65535) survives at
# C-gh -- a question about the FAR TAIL, which percentiles cannot answer, since
# p99.9 is ~700 while the peak is ~14000. The plan was to exploit the measured
# volume-invariance of the distribution (p99 and p99.9 flat to ~2% across 64x)
# to extrapolate N(>x) per unit volume, with the extrapolation CHECKED against a
# bigger box's peak before being trusted. It was checked, and it failed: the fit
# predicted cgh64's peak at ~135700 against 13774 measured, 10x wrong, because
# it is dominated by well-populated low thresholds while the real tail falls far
# faster. The response was to widen the index to uint32 rather than buy a better
# extrapolation -- 0.25 B/p out of a 1.32x margin, against a bound that three
# points were never going to establish.
TAIL_THRESHOLDS = (100, 300, 1000, 3000, 10000, 30000, 65535)


def tail_counts(counts):
    return {str(t): int((counts > t).sum()) for t in TAIL_THRESHOLDS}


def occupancy_summary(counts, n, index_dtype=np.uint32):
    """Occupancy distribution. `uint16_headroom` is kept unchanged rather than
    renamed so cards written before the widening stay directly comparable (the
    `v2_g6b_calib_transport.py` precedent: fix additively, never in place);
    `index_headroom` is the one that tracks the dtype actually in force."""
    live = counts[counts > 0]
    hot = max(int(counts.max()), 1)
    return dict(
        tail=tail_counts(counts),
        mean=float(counts.mean()),
        mean_occupied=float(live.mean()) if len(live) else 0.0,
        max=int(counts.max()),
        p50=float(np.percentile(live, 50)) if len(live) else 0.0,
        p99=float(np.percentile(live, 99)) if len(live) else 0.0,
        p999=float(np.percentile(live, 99.9)) if len(live) else 0.0,
        n_empty=int((counts == 0).sum()),
        n_buckets=int(len(counts)),
        uint16_headroom=float(65535.0 / hot),
        index_dtype=np.dtype(index_dtype).name,
        index_headroom=float(int(np.iinfo(index_dtype).max) / hot),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="smoke", choices=sorted(CONFIGS))
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--tile", type=int, default=None, help="fine cells per tile side")
    ap.add_argument("--buf", type=int, default=32, help="buffer in fine cells")
    ap.add_argument("--bucket-cells", type=int, default=2, help="D-v2-14 ratifies 2")
    ap.add_argument(
        "--index-dtype",
        default="uint32",
        choices=("uint16", "uint32"),
        help="the per-bucket occupancy index; uint32 is the default and uint16 reproduces "
        "the cost D-v2-14 clause 2 was ratified with",
    )
    ap.add_argument("--slack", type=float, default=0.10, help="the live layout's target")
    # 1.0 by default, i.e. the arena can hold every particle. That is NOT an
    # operating point -- it is what makes this a measurement rather than a
    # refusal: overflow can never exceed N, so a full-size arena guarantees the
    # run completes and the ladder below reports the demand instead of the
    # script dying at the first bucket that outgrows its frozen capacity.
    ap.add_argument("--arena-frac", type=float, default=1.0)
    ap.add_argument(
        "--repack-every",
        type=int,
        default=0,
        help="redistribute capacity in place every N steps (0 = never, the "
        "frozen-capacity policy D-v2-14 clause 3 ratifies)",
    )
    ap.add_argument(
        "--alloc-margin",
        type=float,
        default=0.10,
        help="slot-array headroom above the BUILD requirement. The steady state "
        "under clustering sits a little above it by a setting-dependent amount, "
        "so a sweep needs one margin held fixed across every point -- otherwise "
        "the points differ in two variables at once",
    )
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    g = geometry(args.config)
    # default tile: the ratified T=256 where the mesh allows, else n_fine/4
    tile = args.tile or (256 if g["n_fine"] >= 512 else g["n_fine"] // 4)
    buf = args.buf if g["n_fine"] > 4 * args.buf else max(g["n_fine"] // 16, 1)

    print(f"=== M-v2-1 migration + slack [{args.config}] ===")
    print(
        f"    n_part {g['n_part']}^3, L {g['L']}, fine {g['n_fine']}^3, "
        f"tile {tile}/buf {buf}, bucket {args.bucket_cells} cells"
    )

    from inexor.codec import T9Layout
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table, float_step_bullfrog
    from inexor.layout import (
        BrickPackedLayout,
        bucket_ijk_from_key,
        bucket_order_key,
        choose_brick,
    )

    sys.path.insert(0, HERE)

    t9 = T9Layout(box_size=g["L"], n_part=g["n_part"], bucket_cells=args.bucket_cells)
    n_brick = choose_brick(tile, buf, g["n_fine"])
    bricks_per_side = g["n_fine"] // n_brick
    print(
        f"    bucket {t9.bucket_size:.3f} Mpc/h ({t9.n_buckets_side}^3), "
        f"brick {n_brick} fine cells ({bricks_per_side}^3), quantum {t9.quantum:.5f}"
    )

    # M-v2-3 instrument geometry: buckets per brick side, and per TILE side (the
    # engine takes the velocity scale per tile, so the ratio has to be measured
    # over the tiles the force actually uses -- not over bricks).
    # Derived from the bucket grid rather than from cell counts, so it is the
    # SAME `per` bucket_order_key uses and cannot drift from it.
    tiles_side = g["n_fine"] // tile
    per_brick_buckets = t9.n_buckets_side // bricks_per_side
    per_tile_buckets = t9.n_buckets_side // tiles_side

    x, v, cosmo = make_ics(g, args.seed)
    import jax.numpy as jnp

    force = make_force(g, tile, buf)
    a_steps = a_grid(A_INIT, A_FINAL, args.steps, SPACING)
    coeffs = bullfrog_float_coeffs(bullfrog_table(a_steps, cosmo))

    lay = BrickPackedLayout.build(
        x,
        t9,
        bricks_per_side,
        brick_slack=args.slack,
        arena_frac=args.arena_frac,
        alloc_margin=args.alloc_margin,
        index_dtype=np.dtype(args.index_dtype).type,
    )
    counts0 = lay.occupancy.astype(np.int64).copy()
    n = g["n_total"]
    bpp0 = lay.bytes_per_particle()
    print(f"    initial: {occupancy_summary(counts0, n, lay.index_dtype)}")
    print(f"    initial B/p at slack {args.slack:g}: {bpp0}")

    xj = jnp.asarray(x, jnp.float64)
    vj = jnp.asarray(v, jnp.float64)
    _k0, _, _ = bucket_order_key(np.asarray(x, dtype=np.float64), t9, bricks_per_side)
    bijk_prev = bucket_ijk_from_key(_k0, t9, bricks_per_side)
    per_step = []
    t_run = time.perf_counter()
    for k, c in enumerate(coeffs):
        t0 = time.perf_counter()
        xj, vj = float_step_bullfrog(xj, vj, tuple(np.asarray(c, np.float64)), force, g["L"])
        t_force = time.perf_counter() - t0

        x_np = np.asarray(xj, dtype=np.float64)
        t1 = time.perf_counter()
        stats = lay.migrate(x_np)
        t_mig = time.perf_counter() - t1
        lay.check()

        # Counts come from the POSITIONS, not from the layout's slots. Reading
        # them off `particle_to_slot` looked equivalent and is not: a particle
        # living in the arena has a slot past every bucket run, so searchsorted
        # files it under the last bucket or off the end -- which silently drops
        # exactly the overflowing particles the ladder exists to count, and made
        # the ladder report zero demand while the live layout was 47% in arena.
        key_now, _, _ = bucket_order_key(x_np, t9, bricks_per_side)
        counts = np.bincount(key_now, minlength=lay.n_buckets).astype(np.int64)
        assert int(counts.sum()) == n, "counts lost particles"

        # --- M-v2-3 (M1/M2/M-B1): three numbers the engine's design rests on and
        # which this probe computed or could compute and never recorded.
        #
        # M1. `migrate` has always computed brick_migrant_frac (layout.py:543)
        # and the card only ever carried the BUCKET one. They are different by
        # ~50x and it is the brick number that sizes the engine's in-flight
        # migrant buffer, because a bucket change inside a brick is a local
        # repack while a brick change is a payload move between runs.
        #
        # M2. How far a particle moves in BRICKS per step. The engine's slab
        # pipeline assumes at most one brick per step per axis; this is what
        # proves or kills that, and it sizes the far-jumper arena term.
        #
        # M-B1. The velocity scale. The engine takes it per TILE during the kick
        # (a reduction over a buffer already resident) and reconciles by a max
        # over tiles, which is exactly the global max because tile ownership is a
        # partition. The cost is one extra rounding, bounded by the ratio of the
        # tile's own scale to the global one -- so that ratio's distribution IS
        # the accuracy cost, and it has never been looked at.
        bijk_now = bucket_ijk_from_key(key_now, t9, bricks_per_side)
        d_brick = np.abs(bijk_now // per_brick_buckets - bijk_prev // per_brick_buckets)
        d_brick = np.minimum(d_brick, bricks_per_side - d_brick)  # periodic
        bijk_prev = bijk_now

        v_np = np.asarray(vj, dtype=np.float64)
        vabs = np.max(np.abs(v_np), axis=1)
        vmax_global = float(vabs.max())
        tijk = bijk_now // per_tile_buckets
        tile_flat = (tijk[:, 0] * tiles_side + tijk[:, 1]) * tiles_side + tijk[:, 2]
        tile_max = np.zeros(tiles_side**3)
        np.maximum.at(tile_max, tile_flat, vabs)
        ratio = tile_max[tile_max > 0] / vmax_global

        rep = None
        if args.repack_every and (k + 1) % args.repack_every == 0:
            t2 = time.perf_counter()
            rep = lay.repack(brick_slack=args.slack, chunk=1 << 16)
            rep["wall_s"] = time.perf_counter() - t2
            lay.check()

        rec = dict(
            step=k,
            repack=rep,
            a=float(a_steps[k + 1]),
            wall_force_s=t_force,
            wall_migrate_s=t_mig,
            migrant_frac=stats["migrant_frac"],
            # M1: computed since M-v2-1 and never written down. It is ~50x
            # smaller than the bucket figure and it is the one that sizes the
            # engine's in-flight migrant buffer.
            brick_migrant_frac=stats.get("brick_migrant_frac"),
            # M2: how far a particle moves in BRICKS. The engine's slab pipeline
            # assumes at most one per axis per step.
            d_brick=dict(
                max=int(d_brick.max()),
                frac_moving=float(np.mean(np.any(d_brick > 0, axis=1))),
                frac_over_one=float(np.mean(np.any(d_brick > 1, axis=1))),
            ),
            # M-B1: the velocity-scale ratio. The engine takes the scale per tile
            # during the kick and reconciles by a max over tiles, which is
            # exactly the global max because ownership is a partition; the extra
            # rounding costs sqrt(1 + r^2) on the RMS, so this distribution IS
            # the accuracy cost.
            vel=dict(
                vmax=vmax_global,
                ratio_p50=float(np.percentile(ratio, 50)),
                ratio_p99=float(np.percentile(ratio, 99)),
                ratio_max=float(ratio.max()),
                n_tiles_occupied=int(ratio.size),
            ),
            arena_used_frac=stats["arena_used"] / n,
            max_fill_frac=stats["max_fill_frac"],
            n_full_buckets=stats["n_full_buckets"],
            occupancy=occupancy_summary(counts, n, lay.index_dtype),
        )
        per_step.append(rec)
        # flush=True is not cosmetic. A buffered long run that dies leaves an
        # empty log and a zero exit code, which reads exactly like a clean run
        # that produced nothing -- the failure mode already on record for Vista
        # jobs 896055/896092. Progress has to be visible as it happens.
        print(
            f"  step {k:3d} a={rec['a']:.4f}  migrants {rec['migrant_frac']:7.3%}  "
            f"arena {rec['arena_used_frac']:7.3%}  max_occ {rec['occupancy']['max']:5d}  "
            f"force {t_force:7.2f}s  migrate {t_mig:6.2f}s"
            + (
                f"  REPACK {rep['wall_s'] * 1000:.0f}ms scratch "
                f"{rep['scratch_bytes'] / 1e6:.2f}MB slots {rep['slots_used'] / n:.3f}xN"
                + (f" fill {rep['max_brick_fill']:.1%}" if "max_brick_fill" in rep else "")
                if rep
                else ""
            ),
            flush=True,
        )

    wall = time.perf_counter() - t_run

    # the headline, priced the way D-v2-14 clause 2 is written. Peak arena, not
    # mean: the region has to be sized for the worst step of the run.
    reps = [r["repack"] for r in per_step if r.get("repack")]
    main = max(r["slots_used"] for r in reps) / n if reps else lay.n_slots / n
    peak_arena = max(r["arena_used_frac"] for r in per_step)
    terms = dict(
        payload=9.0,
        bucket_index=bpp0["bucket_index"],
        brick_start=bpp0["brick_start"],
        slack=(main - 1.0) * 9.0,
        arena=peak_arena * 9.0,
    )
    terms["total"] = sum(terms.values())

    print("\n--- all-in, in the units D-v2-14 clause 2 is written in ---")
    for k in ("payload", "bucket_index", "brick_start", "slack", "arena"):
        print(f"    {k:<14} {terms[k]:8.3f}")
    print(f"    {'TOTAL':<14} {terms['total']:8.3f} B/p   "
          f"vs the ratified 10.150 ({terms['total'] / 10.15 - 1:+.1%})")
    print(f"    peak arena {peak_arena:.3%} of particles;  "
          f"main {main:.3f} x N;  repack "
          f"{1000 * sum(r['wall_s'] for r in reps) / max(len(reps), 1):.0f} ms/step")
    print(f"    (probe scaffolding, NOT in the total and not shipped: "
          f"{bpp0['scaffold']:.1f} B/p of key + slot maps -- see bytes_per_particle)")
    print("\n    Any move to a ratified figure is JC's call, not this script's.")

    card = dict(
        config=args.config,
        geometry=g,
        tile=tile,
        buf=buf,
        n_brick=n_brick,
        bricks_per_side=bricks_per_side,
        bucket_cells=args.bucket_cells,
        bucket_size=t9.bucket_size,
        quantum=t9.quantum,
        steps=args.steps,
        seed=args.seed,
        live_slack=args.slack,
        live_arena_frac=args.arena_frac,
        initial_occupancy=occupancy_summary(counts0, n, lay.index_dtype),
        initial_bpp=bpp0,
        all_in_terms=terms,
        per_step=per_step,
        wall_s=wall,
        ratified_estimate=dict(slack_and_arena_bpp=0.90, all_in_bpp=10.15, source="D-v2-14 cl.2"),
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"m1_migration_{args.config}{args.out_suffix}.json")
    with open(path, "w") as f:
        json.dump(card, f, indent=1)
    print(f"\nwrote {path}  ({wall:.1f} s)")


if __name__ == "__main__":
    main()
