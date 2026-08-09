"""M-v2-3 exit gate: the engine IS the ratified force plus a codec.

Probe code, NOT package code. Two questions, in the order that makes the second
readable:

**Leg A -- force parity, bitwise.** The engine reads positions out of
slot-ordered T9 state and drives the tiled force from brick SPANS; the ratified
path reads a global position array and drives it from the probe's own bucketing.
Those must produce the same force to the bit. That is possible at all only
because M-v2-3 made both paints INTEGER: with the f64 accumulators the two
membership ORDERS differ, and order changes an f64 scatter-add sequence, so the
comparison would be a coin flip. D-v2-16 clause 7's promotion gate had to pass
one membership to both arms for exactly that reason; this leg does not, and that
is the point.

**Leg B -- accumulated quantization, on the architecture that ships.** D-v2-14's
ratified 4.123e-4 was measured by `v2_g2c_accum_gate.py` against a MONOLITHIC
force with the compression faked in float space, never through the two-level
force, never through the real storage layout, and never with a reordering
particle sequence. This runs the real engine against a never-quantized float
driver of the SAME drift-synchronized shape and reads |dP/P| in D-v2-9's band.
Floors first (the m1_quant_gate discipline): the step floor K vs 2K, so the
codec error is read against a floor and not against zero.

PRE-REGISTERED, before the numbers exist. The engine adds three things the
ratified measurement never had -- the layout's reordering, the two-level force,
and integer paints on both arms -- and M0 measured the midpoint cadence at
2.311e-4 against the boundary's 2.821e-4 at cdev8. Expectation: cdev K=40 lands
within a factor of a few of 4.123e-4 and at least an order under D-v2-9's 3e-2.
A miss is a finding, not a failure of the run.
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

CONFIGS = {
    "smoke": dict(n_part=32, L=32.0, n_fine=64, n_coarse=16),
    "cdev8": dict(n_part=128, L=64.0, n_fine=256, n_coarse=64),
    "cdev": dict(n_part=256, L=128.0, n_fine=512, n_coarse=128),
    "cgh64": dict(n_part=512, L=256.0, n_fine=1024, n_coarse=256),
}
A_INIT, A_FINAL, SPACING, SEED = 0.1, 1.0, "log", 0
ALPHA = 1.0


def _geom(cfg, tile=None, buf=32):
    g = dict(CONFIGS[cfg])
    g["tile"] = tile or (256 if g["n_fine"] >= 512 else g["n_fine"] // 4)
    g["buf"] = min(buf, g["tile"] // 2)
    return g


def make_ics(g, seed=SEED):
    import jax

    from inexor.config import Cosmology
    from inexor.ic import linear_density
    from inexor.lpt import lpt_ics

    cosmo = Cosmology()
    key = jax.random.PRNGKey(seed)
    d0 = linear_density(key, g["n_part"], g["L"], cosmo, f_NL=0.0, fdtype=np.float64)
    x, v = lpt_ics(d0, g["L"], A_INIT, cosmo, order=2, fdtype=np.float64)
    return np.asarray(x, np.float64), np.asarray(v, np.float64), cosmo


def demonstrate_determinism(g):
    """Paint one tile twice and require bit equality BEFORE anything is compared.

    An env flag is a self-report and self-reported knobs have lied on this
    project before. The integer paints make this hold by associativity rather
    than by XLA's determinism flag, so on a correct build it is cheap and
    unconditional -- which means a failure here is a real finding about the
    build, not a configuration reminder.
    """
    import jax.numpy as jnp

    from inexor import forces

    rng = np.random.default_rng(1)
    n = 20000
    P, b_real = forces.padded_size(g["tile"], g["buf"], n_fine=g["n_fine"])
    cell = g["L"] / g["n_fine"]
    u = jnp.asarray(rng.random((n, 3)) * (P * cell))
    live = jnp.asarray(np.ones(n, dtype=bool))
    a, _ = forces.tile_paint_int(u, live, (P,) * 3, cell)
    b, _ = forces.tile_paint_int(jnp.asarray(np.asarray(u)[rng.permutation(n)]), live,
                                 (P,) * 3, cell)
    n_diff = int(np.count_nonzero(np.asarray(a) != np.asarray(b)))
    occupied = int(np.count_nonzero(np.asarray(a)))
    return dict(n_diff=n_diff, occupied_cells=occupied, ok=(n_diff == 0 and occupied > 0))


def leg_force_parity(cfg, g, slack=0.10, arena_frac=0.02):
    """Leg A: the engine's force against the ratified path's, bitwise."""
    import jax.numpy as jnp

    sys.path.insert(0, HERE)
    import v2_g5_core as probe

    from inexor import engine, forces, state
    from inexor.codec import T9Layout

    x, v, _ = make_ics(g)
    ec = engine.EngineConfig(
        box_size=g["L"], n_part=g["n_part"], n_fine=g["n_fine"], n_coarse=g["n_coarse"],
        n_tile=g["tile"], b_fine=g["buf"], alpha=ALPHA,
        brick_slack=slack,
    )
    ec.validate()
    t9 = T9Layout(box_size=g["L"], n_part=g["n_part"], bucket_cells=2)
    st = state.SlotState.build(
        x, v, t9, g["n_fine"] // ec.n_brick, brick_slack=slack, arena_frac=arena_frac
    )
    st.check()

    # The quantized positions the engine will actually see, compacted into one
    # array in brick order, plus the slot->row map. Slot indices run over the
    # ALLOCATION (runs plus spare plus arena), not over particles, so they cannot
    # index a compacted array -- this map is gate scaffolding and is exactly the
    # O(N) bookkeeping the engine exists to not have.
    xs, sl = [], []
    for b in range(st.n_bricks):
        s_b, xb, _ = st.decode_brick(b)
        xs.append(xb)
        sl.append(s_b)
    x_q = np.concatenate(xs)
    slot_of_row = np.concatenate(sl)
    row_of_slot = np.full(st.off.shape[0], -1, dtype=np.int64)
    row_of_slot[slot_of_row] = np.arange(len(slot_of_row))

    # --- the ratified path, driven from the PROBE's own bucketing
    b_real = ec._b_realized
    n_brick = probe.choose_brick(g["tile"], b_real, g["n_fine"])
    order, starts, nb = probe.brick_buckets(x_q, g["n_fine"], n_brick, g["L"] / g["n_fine"])
    tiles = ec.tiles
    cap_p, _ = probe.tile_capacity(order, starts, nb, tiles, g["tile"], b_real, n_brick)

    def member_fn(t):
        return probe.tile_members(order, starts, nb, t, g["tile"], b_real, n_brick)

    g_short_ref, diag = forces.force_short_tiled(
        x_q, g["n_fine"], g["L"], g["n_part"] ** 3, g["tile"], g["buf"],
        member_fn, cap_p, r_s=ec.r_s, paint="int",
    )

    # --- the engine's short arm, from slot spans
    one_tile, geom = forces.make_tile_force_fn(
        ec.n_fine, ec.box_size, ec.n_total, ec.n_tile, ec.b_fine, r_s=ec.r_s, paint="int"
    )
    members = {t: st.tile_bricks(t, ec.n_tile, b_real, ec.n_brick, ec.n_fine) for t in tiles}
    counts = [sum(st.brick_member_count(b) for b in members[t]) for t in tiles]
    cap_e = forces.tile_capacity(counts)
    g_short_eng = np.zeros_like(g_short_ref)
    # the engine's slots index x_q in slot order, so slot == row of x_q
    for t in tiles:
        slots, xt, _ = st.decode_bricks(members[t])
        m = len(slots)
        if m == 0:
            continue
        idx = np.resize(np.arange(m), cap_e)
        live = np.zeros(cap_e, dtype=bool)
        live[:m] = True
        origin, _ = forces.tile_origin_extent(t, ec.n_tile, b_real, geom["cell"])
        u = jnp.mod(jnp.asarray(xt[idx]) - jnp.asarray(origin), ec.box_size)
        out, owned, _ = one_tile(u, jnp.asarray(live))
        out, owned = np.asarray(out)[:m], np.asarray(owned)[:m]
        rows = row_of_slot[slots[owned]]
        assert rows.min() >= 0, "a tile owned a slot that holds no live particle"
        g_short_eng[rows] = out[owned]

    n_diff = int(np.count_nonzero(g_short_eng != g_short_ref))
    peak = float(np.max(np.abs(g_short_ref)))
    nonzero = int(np.count_nonzero(g_short_ref))
    return dict(
        config=cfg, n_particles=int(x.shape[0]), elements=int(g_short_ref.size),
        n_diff=n_diff, max_abs_delta=float(np.max(np.abs(g_short_eng - g_short_ref))),
        oracle_peak=peak, oracle_nonzero=nonzero,
        cap_probe=int(cap_p), cap_engine=int(cap_e),
        partition_ok=bool(diag["partition_ok"]), n_overhang=int(diag["n_overhang_total"]),
        vacuous=bool(peak < 1e-6 or nonzero < 0.5 * g_short_ref.size),
        ok=bool(n_diff == 0 and peak > 1e-6 and nonzero > 0.5 * g_short_ref.size),
    )


def _pk(x, n_mesh, L, n_total):
    import jax.numpy as jnp

    from inexor.diagnostics import _bin_edges, _k_grid
    from inexor.painting import density_contrast

    d = np.asarray(density_contrast(jnp.asarray(x), n_mesh, L, n_total, paint="int"))
    pm = (np.abs(np.fft.rfftn(d)) ** 2 * (L**3 / n_mesh**6)).ravel()
    _, _, kmag = _k_grid(n_mesh, L)
    km = kmag.ravel()
    edges = _bin_edges(n_mesh, L)
    cnt, _ = np.histogram(km, bins=edges)
    s, _ = np.histogram(km, bins=edges, weights=pm)
    c = 0.5 * (edges[1:] + edges[:-1])
    good = cnt > 0
    return c[good], s[good] / cnt[good]


def leg_accumulated(cfg, g, k_steps, slack=0.10, arena_frac=0.02):
    """Leg B: the engine against a never-quantized driver of the same shape."""
    import jax.numpy as jnp

    from inexor import engine, state
    from inexor.codec import T9Layout
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

    x, v, cosmo = make_ics(g)
    a_steps = a_grid(A_INIT, A_FINAL, k_steps, SPACING)
    co = bullfrog_float_coeffs(bullfrog_table(a_steps, cosmo))
    ec = engine.EngineConfig(
        box_size=g["L"], n_part=g["n_part"], n_fine=g["n_fine"], n_coarse=g["n_coarse"],
        n_tile=g["tile"], b_fine=g["buf"], alpha=ALPHA,
        brick_slack=slack,
    )
    ec.validate()

    # --- the engine
    t9 = T9Layout(box_size=g["L"], n_part=g["n_part"], bucket_cells=2)
    st = state.SlotState.build(
        x, v, t9, g["n_fine"] // ec.n_brick, brick_slack=slack, arena_frac=arena_frac
    )
    # flush=True and the instrument's cost beside the physics cost, per the
    # M-v2-1 lesson: a buffered long run that dies leaves an empty log and a zero
    # exit code, which reads exactly like a clean run that produced nothing.
    t_last = [time.perf_counter()]

    def _report(stats):
        now = time.perf_counter()
        print(f"    step wall {now - t_last[0]:7.2f}s  cap {stats['cap']:8d}  "
              f"arena {stats['arena_used']:6d}  scale {stats['vel_scale']:.4e}", flush=True)
        t_last[0] = now

    t0 = time.perf_counter()
    engine.run(st, ec, co, collect=_report)
    st.check()
    wall_engine = time.perf_counter() - t0
    xe = np.concatenate([st.decode_brick(b)[1] for b in range(st.n_bricks)])

    # --- the never-quantized reference.
    #
    # SAME drift-synchronized shape, and -- the part I got wrong first -- the SAME
    # TWO-LEVEL FORCE. Referencing the monolithic force instead measures the
    # split error PLUS the codec, and D-v2-9's 3e-2 bar is precisely the bar on
    # the split error, so that comparison reads a ratified quantity as if it were
    # this milestone's. It measured 6.96e-2 at `smoke`, which is a statement
    # about a tiny unvalidated geometry's split and not about T9 at all.
    #
    # Here the ONLY difference between the two arms is the codec, which is what
    # this leg is for.
    sys.path.insert(0, HERE)
    from inexor import forces as _f

    def force_fn(xx):
        xn = np.asarray(xx, dtype=np.float64)
        g_long, _ = _f.force_global(
            xn, g["n_coarse"], g["L"], g["n_part"] ** 3, "long", r_s=ec.r_s,
            match=(ec.coarse_cell, ec.fine_cell), assign="tsc", paint="int",
        )
        b_real = ec._b_realized
        n_brick = _f.k_components and ec.n_brick
        import v2_g5_core as probe

        order, starts, nbk = probe.brick_buckets(
            xn, g["n_fine"], n_brick, g["L"] / g["n_fine"]
        )
        capp, _ = probe.tile_capacity(
            order, starts, nbk, ec.tiles, g["tile"], b_real, n_brick
        )
        g_short, _ = _f.force_short_tiled(
            xn, g["n_fine"], g["L"], g["n_part"] ** 3, g["tile"], g["buf"],
            lambda t: probe.tile_members(order, starts, nbk, t, g["tile"], b_real, n_brick),
            capp, r_s=ec.r_s, paint="int",
        )
        return jnp.asarray(g_long + g_short)
    t0 = time.perf_counter()
    xr, _ = engine.float_run_bullfrog_sync(jnp.asarray(x), jnp.asarray(v), co, force_fn, g["L"])
    wall_ref = time.perf_counter() - t0
    xr = np.asarray(xr)

    k, p_e = _pk(xe, g["n_fine"], g["L"], g["n_part"] ** 3)
    _, p_r = _pk(xr, g["n_fine"], g["L"], g["n_part"] ** 3)
    k_nyq = np.pi * g["n_fine"] / g["L"]
    band = k <= 0.2 * k_nyq
    dpp = np.abs(p_e[band] / p_r[band] - 1.0)
    return dict(
        config=cfg, k=k_steps, k_gate=float(0.2 * k_nyq),
        max_dpp_in_band=float(dpp.max()),
        median_dpp_in_band=float(np.median(dpp)),
        dpp_vs_k=[[float(a), float(b)] for a, b in zip(k[band], dpp)],
        wall_engine_s=wall_engine, wall_ref_s=wall_ref,
        bar=3.0e-2, margin=float(3.0e-2 / max(dpp.max(), 1e-30)),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="smoke", choices=sorted(CONFIGS))
    ap.add_argument("--leg", default="force", choices=("force", "accum"))
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--buf", type=int, default=32)
    ap.add_argument("--slack", type=float, default=0.10,
                    help="brick slack; D-v2-19 measured 10% as the pooled figure")
    ap.add_argument("--arena-frac", type=float, default=0.02,
                    help="arena size as a fraction of N; D-v2-19 measured 1-2% at cdev8, "
                         "where a brick is 1/512 of the box. `smoke` has far fewer bricks, "
                         "so each is a much larger fraction and needs more")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)
    g = _geom(args.config, args.tile, args.buf)
    print(f"[m3-gate] {args.config} leg={args.leg} tile={g['tile']} buf={g['buf']} "
          f"backend={jax.devices()[0].platform}", flush=True)

    det = demonstrate_determinism(g)
    print(f"  determinism: {det['n_diff']} differing cells over {det['occupied_cells']} "
          f"occupied -> {'OK' if det['ok'] else 'FAIL'}", flush=True)
    if not det["ok"]:
        print("FATAL: the paint is not order-independent on this build; nothing below "
              "would be readable.", flush=True)
        return 1

    t0 = time.perf_counter()
    if args.leg == "force":
        res = leg_force_parity(args.config, g, args.slack, args.arena_frac)
        print(f"  force parity: {res['n_diff']} of {res['elements']} differ, "
              f"max |delta| {res['max_abs_delta']:.3e}, peak {res['oracle_peak']:.3f} -> "
              f"{'PASS' if res['ok'] else ('VACUOUS' if res['vacuous'] else 'FAIL')}", flush=True)
    else:
        res = leg_accumulated(args.config, g, args.k, args.slack, args.arena_frac)
        print(f"  accumulated: max |dP/P| in band {res['max_dpp_in_band']:.3e} against the "
              f"3e-2 bar ({res['margin']:.1f}x margin); engine {res['wall_engine_s']:.1f}s "
              f"ref {res['wall_ref_s']:.1f}s", flush=True)
    res["determinism"] = det
    res["slack"] = args.slack
    res["arena_frac"] = args.arena_frac
    res["wall_s"] = time.perf_counter() - t0

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"m3_gate_{args.config}_{args.leg}{args.out_suffix}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"  card -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
