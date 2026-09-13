"""D2e: one tile of the kick, on the device.

WHAT THIS REPLACES. `engine.tile_task` end to end: the host decode, the
tile-local shift and ownership, the short force (already jax), the host packing
of owned rows for the long gather, and the numpy kick and per-brick quantize.
`tile_task_device` takes the same arguments and returns the same dict, so
`engine.apply_result` consumes it unchanged.

The pieces are the ones earlier rungs put on the device, composed rather than
rewritten: `device.decode.decode_rows` (D2a), the caller's `one_tile`,
`forces.gather_coarse_subblock` with its deferred guard (D2c), and
`device.kick.quantize_per_brick` (D2b).

WHAT MOVES AND WHAT DOES NOT, relative to the host form.
- Padded rows decode slot 0 on the device. The short arm is handed the host's
  own padding instead -- rows cycled from the real ones, `np.resize` order -- so
  the tile paint sees exactly the input it sees on the host.
- The long gather reads the tile's rows IN ROW ORDER with `live=owned`, where the
  host packs owned rows to the front of a buffer. The gather is per row, so the
  packing is transport, not arithmetic.
- The kick is written with the host's operands and in its order, so the dtype
  promotion of an f32 force arm is the host's and not a cast of this module's
  choosing. `device.kick.kick_and_quantize` casts both arms to f64 first, which
  differs from the host whenever the fine arm is f32.
- The coarse sub-blocks are still staged on the host (`stage_coarse_subblock`)
  and copied per tile. That is a seam: where the coarse mesh lives is undecided.
- D-007's int16 range refusal is not on this path, as in D2b: by construction
  the extremes land on +-32767, and `kick.assert_int16_range_device` is the
  explicit check for gates.

SCOPE. Eager, and decoding against the whole `off` / `w` arrays; the tile's
bricks are not a contiguous slot range, so `device.paint.slab_window` does not
apply. Nothing here is wired into `engine.step`.
"""

from __future__ import annotations

import time

import numpy as np


def owned_rows_device(brick_of_row, tijk, n_tile, n_brick, nb):
    """The jnp twin of `forces.owned_mask_from_bricks`: integer, so exact."""
    import jax.numpy as jnp

    b = jnp.asarray(brick_of_row, dtype=jnp.int64)
    nb = int(nb)
    per = int(n_tile) // int(n_brick)
    bi, rem = jnp.divmod(b, nb * nb)
    bj, bk = jnp.divmod(rem, nb)
    t = [int(c) for c in tijk]
    return (bi // per == t[0]) & (bj // per == t[1]) & (bk // per == t[2])


def tile_task_device(st, one_tile, C, g_coarse, t, bricks):
    """One tile of the kick on the device; the `engine.tile_task` contract.

    `C` is `engine.step`'s per-step header and `g_coarse` its three coarse
    force meshes. Returns the dict `engine.apply_result` consumes.
    """
    import jax.numpy as jnp

    from ..forces import (
        COARSE_HALO,
        check_stencil_guard,
        coarse_subblock_origin_extent,
        gather_coarse_subblock,
        stage_coarse_subblock,
        tile_origin_extent,
    )
    from .decode import decode_rows, tile_decode_plan
    from .kick import quantize_per_brick

    t_dec = time.perf_counter()
    bricks = np.asarray(bricks, dtype=np.int64)
    cap = int(C["cap"])
    nb = int(C["n_fine"]) // int(C["n_brick"])
    plan = tile_decode_plan(st, bricks)
    m = int(plan["n_rows"])
    if m == 0:
        return dict(t=t, empty=True, n_owned=0, n_out=0)
    if m > cap:
        raise RuntimeError(f"tile {t}: {m} members > cap {cap}")
    dec = decode_rows(plan, st.off, st.w, st.vel_scale, st.arena_bucket,
                      st.arena_base, st.t9, nb, cap)
    live = dec["live"]
    origin, _ = tile_origin_extent(t, C["n_tile"], C["b_real"], C["cell"])
    # the host's padding: row r reads real row r mod m
    cyc = jnp.arange(cap, dtype=jnp.int64) % m
    u = jnp.mod(dec["x"][cyc] - jnp.asarray(origin), C["box"])
    own = owned_rows_device(dec["brick_of_row"], t, C["n_tile"], C["n_brick"], nb) & live

    t_short = time.perf_counter()
    g_short, owned, n_out = one_tile(u, live, own)
    n_own = int(jnp.sum(owned))
    if n_own == 0:
        return dict(t=t, empty=True, n_owned=0, n_out=int(n_out))

    t_long = time.perf_counter()
    o_cells, extent = coarse_subblock_origin_extent(
        t, C["n_tile"], C["n_coarse"], C["n_fine"], halo=COARSE_HALO)
    sub = [jnp.asarray(stage_coarse_subblock(g, o_cells, extent)) for g in g_coarse]
    guard = []
    g_long = gather_coarse_subblock(
        *sub, dec["x"], o_cells, C["coarse_cell"], C["n_coarse"], assign="tsc",
        live=owned, guard_out=guard)

    t_quant = time.perf_counter()
    g_tot = g_short + g_long
    v_new = C["alpha_k"] * dec["v"] + C["bcoef"] * g_tot
    q = quantize_per_brick(v_new, owned, dec["brick_index"], len(bricks))

    own_h = np.asarray(owned)
    counts = np.asarray(q["owned_counts"])
    check_stencil_guard(guard)  # before anything below is used
    written = counts > 0
    res = dict(
        t=t, empty=False,
        slots_o=np.asarray(dec["slots"])[own_h],
        w_codes=np.asarray(q["w_codes"])[own_h],
        run_bricks=bricks[written],
        run_scales=np.asarray(q["scales"])[written],
        n_owned=n_own, n_out=int(n_out),
    )
    t_end = time.perf_counter()
    res["busy"] = dict(decode=t_short - t_dec, short=t_long - t_short,
                       long=t_quant - t_long, quant=t_end - t_quant)
    return res
