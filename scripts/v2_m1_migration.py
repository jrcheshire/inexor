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
    how close the max comes to the uint16 ceiling that the 0.25 B/p index
    depends on
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

# The slack targets priced. 0.10 is D-v2-14 clause 2's estimate; the rest
# bracket it. These are TARGETS -- the realized fraction is larger, because a
# whole spare slot is the smallest unit a bucket can be given.
SLACK_LADDER = (0.05, 0.10, 0.15, 0.25, 0.50, 1.00)


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


def slack_ladder_demand(counts_now, capacity_by_target, payload=9.0, n=None):
    """Arena demand left by each slack target, exactly.

    The layout freezes capacity at build time and never resizes, so a bucket
    overflowing at step k needs `count_k - capacity_0` arena slots. Summing that
    over buckets is the arena the run would have needed -- no simulation of the
    ladder required, and no approximation.
    """
    out = {}
    for target, cap in capacity_by_target.items():
        over = np.maximum(counts_now - cap, 0)
        out[f"{target:g}"] = dict(
            arena_slots=int(over.sum()),
            arena_frac_of_n=float(over.sum()) / max(n, 1),
            n_overflowing_buckets=int((over > 0).sum()),
            worst_overflow=int(over.max()) if len(over) else 0,
        )
    return out


def occupancy_summary(counts, n):
    live = counts[counts > 0]
    return dict(
        mean=float(counts.mean()),
        mean_occupied=float(live.mean()) if len(live) else 0.0,
        max=int(counts.max()),
        p50=float(np.percentile(live, 50)) if len(live) else 0.0,
        p99=float(np.percentile(live, 99)) if len(live) else 0.0,
        p999=float(np.percentile(live, 99.9)) if len(live) else 0.0,
        n_empty=int((counts == 0).sum()),
        n_buckets=int(len(counts)),
        uint16_headroom=float(65535.0 / max(int(counts.max()), 1)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="smoke", choices=sorted(CONFIGS))
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--tile", type=int, default=None, help="fine cells per tile side")
    ap.add_argument("--buf", type=int, default=32, help="buffer in fine cells")
    ap.add_argument("--bucket-cells", type=int, default=2, help="D-v2-14 ratifies 2")
    ap.add_argument("--slack", type=float, default=0.10, help="the live layout's target")
    # 1.0 by default, i.e. the arena can hold every particle. That is NOT an
    # operating point -- it is what makes this a measurement rather than a
    # refusal: overflow can never exceed N, so a full-size arena guarantees the
    # run completes and the ladder below reports the demand instead of the
    # script dying at the first bucket that outgrows its frozen capacity.
    ap.add_argument("--arena-frac", type=float, default=1.0)
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
    from inexor.layout import BrickLayout, bucket_order_key, choose_brick

    sys.path.insert(0, HERE)

    t9 = T9Layout(box_size=g["L"], n_part=g["n_part"], bucket_cells=args.bucket_cells)
    n_brick = choose_brick(tile, buf, g["n_fine"])
    bricks_per_side = g["n_fine"] // n_brick
    print(
        f"    bucket {t9.bucket_size:.3f} Mpc/h ({t9.n_buckets_side}^3), "
        f"brick {n_brick} fine cells ({bricks_per_side}^3), quantum {t9.quantum:.5f}"
    )

    x, v, cosmo = make_ics(g, args.seed)
    import jax.numpy as jnp

    force = make_force(g, tile, buf)
    a_steps = a_grid(A_INIT, A_FINAL, args.steps, SPACING)
    coeffs = bullfrog_float_coeffs(bullfrog_table(a_steps, cosmo))

    lay = BrickLayout.build(
        x, t9, bricks_per_side, slack_frac=args.slack, arena_frac=args.arena_frac
    )
    counts0 = lay.occupancy.astype(np.int64).copy()
    # capacity each ladder target would have frozen at build time
    cap_by_target = {}
    for s in SLACK_LADDER:
        extra = np.ceil(counts0 * s).astype(np.int64)
        extra = np.where(counts0 > 0, np.maximum(extra, 1), extra)
        cap_by_target[s] = counts0 + extra

    n = g["n_total"]
    bpp0 = lay.bytes_per_particle()
    print(f"    initial: {occupancy_summary(counts0, n)}")
    print(f"    initial B/p at slack {args.slack:g}: {bpp0}")

    xj = jnp.asarray(x, jnp.float64)
    vj = jnp.asarray(v, jnp.float64)
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

        rec = dict(
            step=k,
            a=float(a_steps[k + 1]),
            wall_force_s=t_force,
            wall_migrate_s=t_mig,
            migrant_frac=stats["migrant_frac"],
            arena_used_frac=stats["arena_used"] / n,
            max_fill_frac=stats["max_fill_frac"],
            n_full_buckets=stats["n_full_buckets"],
            occupancy=occupancy_summary(counts, n),
            slack_ladder=slack_ladder_demand(counts, cap_by_target, n=n),
        )
        per_step.append(rec)
        # flush=True is not cosmetic. A buffered long run that dies leaves an
        # empty log and a zero exit code, which reads exactly like a clean run
        # that produced nothing -- the failure mode already on record for Vista
        # jobs 896055/896092. Progress has to be visible as it happens.
        print(
            f"  step {k:3d} a={rec['a']:.4f}  migrants {rec['migrant_frac']:7.3%}  "
            f"arena {rec['arena_used_frac']:7.3%}  max_occ {rec['occupancy']['max']:5d}  "
            f"force {t_force:7.2f}s  migrate {t_mig:6.2f}s",
            flush=True,
        )

    wall = time.perf_counter() - t_run

    # the headline: what each slack target would have cost, and whether it held
    worst = {}
    for s in SLACK_LADDER:
        arena = max(r["slack_ladder"][f"{s:g}"]["arena_frac_of_n"] for r in per_step)
        cap = cap_by_target[s]
        slack_bpp = float((cap - counts0).sum()) * 9.0 / n
        worst[f"{s:g}"] = dict(
            slack_bpp=slack_bpp,
            peak_arena_frac_of_n=arena,
            arena_bpp=arena * 9.0,
            all_in_bpp=9.0 + bpp0["bucket_index"] + bpp0["brick_csr"] + slack_bpp + arena * 9.0,
        )

    print("\n--- slack ladder: what each target costs and what it leaves ---")
    print(f"    {'target':>8} {'slack B/p':>10} {'peak arena':>11} {'arena B/p':>10} {'all-in':>8}")
    for s, w in worst.items():
        print(
            f"    {s:>8} {w['slack_bpp']:>10.3f} {w['peak_arena_frac_of_n']:>10.3%} "
            f"{w['arena_bpp']:>10.3f} {w['all_in_bpp']:>8.3f}"
        )
    print(
        "\n    D-v2-14 clause 2 has slack+arena at 0.90 B/p and all-in at 10.15. "
        "Any move is JC's call, not this script's."
    )

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
        initial_occupancy=occupancy_summary(counts0, n),
        initial_bpp=bpp0,
        slack_ladder_worst=worst,
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
