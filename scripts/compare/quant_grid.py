"""Quantization of positions and velocities against never-quantized runs on the same ICs.

One arm per process; each writes a JSON card (P(k) on the fine mesh plus the arm's settings).

    engine   the engine (CPU lane): positions on the T9 lattice, velocities in per-brick int16
             codes, the two-level force with integer paints
    ref      a float run of the same drift-synchronized BullFrog shape and the same two-level
             force; nothing quantized
    refx     ref with positions rounded to the T9 lattice at the ICs and after every drift
    refv     ref with velocities round-tripped through per-brick int16 codes at the ICs and
             after every kick (scale = the brick's largest |v| component / 32767)
    refxv    both; reproduces `engine` up to the engine's own bookkeeping, which checks the
             two emulations

The float arms also record, per step, the drift in quanta (median |dx|/q over particles and
axes, and the fraction below q/2) and the kick in velocity codes (median, fraction below half
a code), the quantities that set the bias.

    python scripts/compare/quant_grid.py --config cdev8 --a-init 0.1 --k 120 --arm engine \
        --out OUT.json

Configs: `cdev8` (128^3 in 64 Mpc/h: 0.5 Mpc/h spacing, 0.25 fine cell, the production
resolution) and `cdev8-L128` (128^3 in 128 Mpc/h: 1.0 spacing, 0.5 fine cell). Every arm runs
in float64; ICs are 2LPT with the LCDM growth from seed 0. CPU backend.
"""

import argparse
import json
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

CONFIGS = {
    "smoke": dict(n_part=32, L=32.0, n_fine=64, n_coarse=16),
    "cdev8": dict(n_part=128, L=64.0, n_fine=256, n_coarse=64),
    "cdev8-L128": dict(n_part=128, L=128.0, n_fine=256, n_coarse=64),
}
A_FINAL, SPACING, SEED = 1.0, "log", 0
SLACK, ARENA = 0.20, 0.10   # layout carries no physics; room for large late steps from z = 49
MATCH_ORDER, INT16_MAX = 3, 32767.0


def geometry(name):
    g = dict(CONFIGS[name])
    g["tile"] = 256 if g["n_fine"] >= 512 else g["n_fine"] // 4
    g["buf"] = min(32, g["tile"] // 2)
    return g


def engine_config(g):
    from inexor import engine

    ec = engine.EngineConfig(
        box_size=g["L"], n_part=g["n_part"], n_fine=g["n_fine"], n_coarse=g["n_coarse"],
        n_tile=g["tile"], b_fine=g["buf"], alpha=1.0, brick_slack=SLACK,
        coarse_match_order=MATCH_ORDER, tile_workers=1,
    )
    ec.validate()
    return ec


def schedule(a_init, k):
    from inexor.config import Cosmology
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

    a_steps = a_grid(a_init, A_FINAL, k, SPACING)
    return bullfrog_float_coeffs(bullfrog_table(a_steps, Cosmology())), a_steps


def initial_conditions(g, a_init):
    import jax

    from inexor.config import Cosmology
    from inexor.ic import linear_density
    from inexor.lpt import lpt_ics

    d0 = linear_density(jax.random.PRNGKey(SEED), g["n_part"], g["L"], Cosmology(), f_NL=0.0,
                        fdtype=np.float64)
    x, v = lpt_ics(d0, g["L"], a_init, Cosmology(), order=2, fdtype=np.float64)
    return np.asarray(x, np.float64), np.asarray(v, np.float64)


def power_spectrum(x, n_mesh, L, n_total):
    """Integer-paint CIC density on `n_mesh`, |delta_k|^2 in the default linear bins."""
    import jax.numpy as jnp

    from inexor.diagnostics import _bin_edges, _k_grid
    from inexor.painting import density_contrast

    d = np.asarray(density_contrast(jnp.asarray(x), n_mesh, L, n_total, paint="int"))
    pm = (np.abs(np.fft.rfftn(d)) ** 2 * (L**3 / n_mesh**6)).ravel()
    km = _k_grid(n_mesh, L)[2].ravel()
    edges = _bin_edges(n_mesh, L)
    cnt, _ = np.histogram(km, bins=edges)
    s, _ = np.histogram(km, bins=edges, weights=pm)
    good = cnt > 0
    return 0.5 * (edges[1:] + edges[:-1])[good], s[good] / cnt[good]


# --- tile membership from float positions, for the float two-level force

def _brick_index(x, n_fine, n_brick, cell):
    nb = n_fine // n_brick
    b = np.mod(np.floor(x / (cell * n_brick)).astype(np.int64), nb)
    bid = (b[:, 0] * nb + b[:, 1]) * nb + b[:, 2]
    order = np.argsort(bid, kind="stable")
    return order, np.searchsorted(bid[order], np.arange(nb**3 + 1)), nb


def _tile_members(order, starts, nb, tijk, n_tile, b_fine, n_brick):
    """Particle indices of the brick union covering tile + buffer (refuses a wrapping span)."""
    pad = -(-b_fine // n_brick)
    span = n_tile // n_brick + 2 * pad
    if span > nb:
        raise ValueError(f"tile + buffer spans {span} bricks of {nb}: it would double-count")
    lo = np.asarray(tijk, np.int64) * (n_tile // n_brick) - pad
    idx = [order[starts[b]:starts[b + 1]]
           for i in range(span) for j in range(span) for k in range(span)
           for b in [(((lo[0] + i) % nb) * nb + (lo[1] + j) % nb) * nb + (lo[2] + k) % nb]]
    return np.concatenate(idx)


def two_level_force(g, ec):
    """The engine's split, in float64 with integer paints: coarse TSC long arm + tiled CIC."""
    import jax.numpy as jnp

    from inexor import forces as F

    n, cell = g["n_part"] ** 3, g["L"] / g["n_fine"]

    def force(x):
        x = np.asarray(x, np.float64)
        g_long, _ = F.force_global(x, g["n_coarse"], g["L"], n, "long", r_s=ec.r_s,
                                   match=ec.coarse_match, assign="tsc", paint="int")
        order, starts, nb = _brick_index(x, g["n_fine"], ec.n_brick, cell)

        def members(t):
            return _tile_members(order, starts, nb, t, g["tile"], ec._b_realized, ec.n_brick)

        cap = max(len(members(t)) for t in ec.tiles)
        cap = -(-cap // 1024) * 1024
        g_short, _ = F.force_short_tiled(x, g["n_fine"], g["L"], n, g["tile"], g["buf"],
                                         members, cap, r_s=ec.r_s, paint="int")
        return jnp.asarray(g_long + g_short)

    return force


def run_engine(g, ec, co, x, v):
    from inexor import engine, state
    from inexor.codec import T9Layout

    t9 = T9Layout(box_size=g["L"], n_part=g["n_part"], bucket_cells=2)
    st = state.SlotState.build(x, v, t9, g["n_fine"] // ec.n_brick, brick_slack=SLACK,
                               arena_frac=ARENA)
    engine.run(st, ec, co)
    st.check()
    return np.concatenate([st.decode_brick(b)[1] for b in range(st.n_bricks)]), {}


def run_float(g, ec, co, x, v, quant):
    """`engine.float_run_bullfrog_sync`'s shape, with the codecs named in `quant` emulated."""
    import jax.numpy as jnp

    from inexor.codec import T9Layout
    from inexor.engine import fused_drifts

    t9 = T9Layout(box_size=g["L"], n_part=g["n_part"], bucket_cells=2)
    q, L, levels = float(t9.quantum), g["L"], int(t9.n_levels)
    nb = g["n_fine"] // ec.n_brick
    brick = L / nb

    def lattice(xx):
        return np.mod(np.rint(xx / q).astype(np.int64), levels).astype(np.float64) * q

    def scales(xx, vv):
        b = np.mod(np.floor(np.mod(xx, L) / brick).astype(np.int64), nb)
        bid = (b[:, 0] * nb + b[:, 1]) * nb + b[:, 2]
        vmax = np.zeros(nb**3)
        np.maximum.at(vmax, bid, np.max(np.abs(vv), axis=1))
        s = vmax / INT16_MAX
        s[s <= 0] = 1.0
        return s[bid][:, None]

    def codes(xx, vv):
        s = scales(xx, vv)
        return np.rint(vv / s) * s

    force = two_level_force(g, ec)
    if "x" in quant:
        x = lattice(x)
    if "v" in quant:
        v = codes(x, v)
    stats = dict(drift_med_q=[], drift_frac_below_half_q=[], kick_med_code=[],
                 kick_frac_below_half_code=[])

    def drift(xx, dx):
        a = np.abs(dx).ravel() / q
        stats["drift_med_q"].append(float(np.median(a)))
        stats["drift_frac_below_half_q"].append(float(np.mean(a < 0.5)))
        xx = np.mod(xx + dx, L)
        return lattice(xx) if "x" in quant else xx

    lead, fused = fused_drifts(co)
    x = drift(x, lead * v)
    for k in range(len(fused)):
        v_new = co[k][1] * v + co[k][2] * np.asarray(force(jnp.asarray(x)))
        dv = np.abs(v_new - v) / scales(x, v_new)
        stats["kick_med_code"].append(float(np.median(dv)))
        stats["kick_frac_below_half_code"].append(float(np.mean(dv < 0.5)))
        v = codes(x, v_new) if "v" in quant else v_new
        x = drift(x, float(fused[k]) * v)
    return x, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="cdev8", choices=sorted(CONFIGS))
    ap.add_argument("--a-init", type=float, default=0.1)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--arm", required=True, choices=("engine", "ref", "refx", "refv", "refxv"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)

    g = geometry(args.config)
    ec = engine_config(g)
    co, _ = schedule(args.a_init, args.k)
    x, v = initial_conditions(g, args.a_init)
    print(f"[quant-grid] {args.config} a_init={args.a_init} K={args.k} arm={args.arm} "
          f"tile={g['tile']} buf={g['buf']} backend={jax.default_backend()}", flush=True)
    t0 = time.perf_counter()
    if args.arm == "engine":
        xf, stats = run_engine(g, ec, co, x, v)
    else:
        xf, stats = run_float(g, ec, co, x, v, args.arm[3:])
    wall = time.perf_counter() - t0
    k, p = power_spectrum(xf, g["n_fine"], g["L"], g["n_part"] ** 3)
    rec = dict(config=args.config, geometry=g, a_init=args.a_init, k_steps=args.k,
               arm=args.arm, coarse_match_order=MATCH_ORDER, seed=SEED, wall_s=wall,
               k_centres=np.asarray(k).tolist(), p=np.asarray(p).tolist(), **stats)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rec, fh)
    print(f"  wall {wall:.1f} s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
