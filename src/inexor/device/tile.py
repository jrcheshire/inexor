"""D2e: one tile of the kick, on the device.

WHAT THIS REPLACES. `engine.tile_task` end to end: the host decode, the
tile-local shift and ownership, the short force (already jax), the host packing
of owned rows for the long gather, and the numpy kick and per-brick quantize.
`tile_task_device` takes the same arguments and returns the same dict, so
`engine.apply_result` consumes it unchanged.

The pieces are the ones earlier rungs put on the device, composed rather than
rewritten: `device.decode.decode_core` (D2a), the caller's `one_tile`,
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

EAGER OR JITTED (`jit=`). Jitted, the decode, force, gather, kick and quantize
are one program at fixed per-step shapes (`tile_step_shapes`), with the tile
index, origins and kick coefficients as runtime values, so every tile of every
step reuses one executable. The coefficients enter as 0-d arrays cast to the
dtype the host's Python-float operand takes (a weak float adopts the array's
dtype), and the quantize's scale divisor is passed in: a divisor built inside
the program folds back to a scalar, which CPU XLA computes as a reciprocal
multiply (record sec. 16).

THE STATE ON THE DEVICE (`device_state=`). The jitted path decodes against the
whole `off` / `w` / `vel_scale` / `arena_bucket`. Unless the caller passes them
already on the device (`stage_state_on_device`), every call uploads them: 1.58 GB
per tile at a 512^3 state (Vista 993754). The 4096^3 design supplies a streamed
window here instead; the tile's bricks are not a contiguous slot range, so
`device.paint.slab_window` does not apply as is.

Nothing here is wired into `engine.step`.
"""

from __future__ import annotations

import time

import numpy as np

# One compiled tile program per set of static parameters, and a count of
# traces: the receipt that one executable served every tile.
_KERNELS = {}
_TRACES = [0]

STATE_FIELDS = ("off", "w", "vel_scale", "arena_bucket")


def owned_rows_device(brick_of_row, tijk, n_tile, n_brick, nb):
    """The jnp twin of `forces.owned_mask_from_bricks`: integer, so exact.

    `tijk` may be a traced int array.
    """
    import jax.numpy as jnp

    b = jnp.asarray(brick_of_row, dtype=jnp.int64)
    nb = int(nb)
    per = int(n_tile) // int(n_brick)
    bi, rem = jnp.divmod(b, nb * nb)
    bj, bk = jnp.divmod(rem, nb)
    t = jnp.asarray(tijk, dtype=jnp.int64)
    return (bi // per == t[0]) & (bj // per == t[1]) & (bk // per == t[2])


def tile_step_shapes(st, floor=None):
    """Fixed per-step shapes for the jitted tile: the arena rectangle's width.

    Every tile of a step has the same brick count and `cap`; only the widest
    per-brick arena varies, so it sits on `forces.capacity_shape`'s ladder and
    `floor` keeps it monotone across steps. O(bricks + arena).
    """
    from ..forces import capacity_shape
    from .paint import _arena_per_brick

    floor = floor or {}
    return dict(arena_rect=int(capacity_shape(
        max(int(_arena_per_brick(st).max()), 1),
        floor_shape=int(floor.get("arena_rect", 0)))))


def stage_state_on_device(st):
    """The state arrays the jitted tile decodes against, placed on the device
    once, for `tile_task_device(device_state=...)`. Blocks until placed."""
    import jax
    import jax.numpy as jnp

    return {k: jax.block_until_ready(jnp.asarray(getattr(st, k))) for k in STATE_FIELDS}


def _tile_kernel(one_tile, *, cap, n_b, p3, per, nb, n_tile, n_brick, n_coarse,
                 coarse_cell, box, t9, with_forces=False):
    key = (id(one_tile), cap, n_b, p3, per, nb, n_tile, n_brick, n_coarse,
           float(coarse_cell), float(box), float(t9.quantum), int(t9.n_buckets_side),
           bool(with_forces))
    fn = _KERNELS.get(key)
    if fn is not None:
        return fn
    import jax
    import jax.numpy as jnp

    from ..forces import gather_coarse_subblock
    from .decode import decode_core
    from .kick import quantize_per_brick

    def body(starts, occ, live_counts, arena_slots, row_offsets, bricks, off, w,
             vel_scale, arena_bucket, arena_base, n_rows, origin, tijk, o_cells,
             sub_x, sub_y, sub_z, alpha_k, bcoef, scale_div):
        _TRACES[0] += 1  # trace time only
        slots, x, bor, bi, live = decode_core(
            starts, occ, live_counts, arena_slots, row_offsets, bricks, off,
            arena_bucket, arena_base, n_rows, cap=cap, lift=cap + 1, p3=p3, per=per,
            bricks_per_side=nb, t9=t9, fdtype=jnp.float64)
        v = w[slots].astype(jnp.float64) * vel_scale[bor][:, None]
        cyc = jnp.arange(cap, dtype=jnp.int64) % n_rows
        u = jnp.mod(x[cyc] - origin, box)
        own = owned_rows_device(bor, tijk, n_tile, n_brick, nb) & live
        g_short, owned, n_out = one_tile(u, live, own)
        guard = []
        g_long = gather_coarse_subblock(
            sub_x, sub_y, sub_z, x, o_cells, coarse_cell, n_coarse, assign="tsc",
            live=owned, guard_out=guard)
        g_tot = g_short + g_long
        v_new = alpha_k.astype(v.dtype) * v + bcoef.astype(g_tot.dtype) * g_tot
        q = quantize_per_brick(v_new, owned, bi, n_b, scale_div=scale_div)
        lo, hi, _ext, _assign = guard[0]
        out = dict(slots=slots, owned=owned, n_out=n_out, n_own=jnp.sum(owned),
                   w_codes=q["w_codes"], scales=q["scales"],
                   counts=q["owned_counts"], lo=lo, hi=hi)
        if with_forces:
            out.update(g_short=g_short, g_long=g_long, v_new=v_new)
        return out

    fn = jax.jit(body)
    _KERNELS[key] = fn
    return fn


def tile_task_device(st, one_tile, C, g_coarse, t, bricks, jit=False, shapes=None,
                     with_forces=False, device_state=None, timings=None):
    """One tile of the kick on the device; the `engine.tile_task` contract.

    `C` is `engine.step`'s per-step header and `g_coarse` its three coarse
    force meshes. Returns the dict `engine.apply_result` consumes. `jit=True`
    needs `shapes` from `tile_step_shapes`.

    `with_forces=True` adds `forces`: `g_short`, `g_long` and `v_new` at the
    owned rows, in row order. A gate instrument; it costs a readback.

    `device_state` (jit only) is `stage_state_on_device(st)`; without it every
    call uploads the state arrays.

    `timings`, if a dict (jit only), receives per-phase seconds, each phase
    ended by a device sync: `plan`, `stage` (host), `h2d_tile`, `h2d_state`,
    `compute`, `d2h`, `result` (host). The syncs are taken only when it is
    passed, so the timed call is not the production call's scheduling.

    JIT IS NOT BITWISE THE EAGER PATH, and is accepted on a tolerance (record
    sec. 17): the compiled coarse gather differs by roundoff, a few eps of the
    long force. The eager path is the bitwise oracle against `engine.tile_task`.
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
    origin, _ = tile_origin_extent(t, C["n_tile"], C["b_real"], C["cell"])
    o_cells, extent = coarse_subblock_origin_extent(
        t, C["n_tile"], C["n_coarse"], C["n_fine"], halo=COARSE_HALO)

    if jit:
        if shapes is None:
            raise ValueError("jit=True needs the step's fixed shapes (tile_step_shapes)")
        if timings is not None:
            timings["plan"] = time.perf_counter() - t_dec
        return _tile_task_jit(st, one_tile, C, g_coarse, t, bricks, plan, origin,
                              o_cells, extent, shapes, with_forces, device_state, timings)
    if device_state is not None or timings is not None:
        raise ValueError("device_state and timings apply to the jitted path only")

    dec = decode_rows(plan, st.off, st.w, st.vel_scale, st.arena_bucket,
                      st.arena_base, st.t9, nb, cap)
    live = dec["live"]
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
    sub = [jnp.asarray(stage_coarse_subblock(g, o_cells, extent)) for g in g_coarse]
    guard = []
    g_long = gather_coarse_subblock(
        *sub, dec["x"], o_cells, C["coarse_cell"], C["n_coarse"], assign="tsc",
        live=owned, guard_out=guard)

    t_quant = time.perf_counter()
    g_tot = g_short + g_long
    v_new = C["alpha_k"] * dec["v"] + C["bcoef"] * g_tot
    q = quantize_per_brick(v_new, owned, dec["brick_index"], len(bricks))

    check_stencil_guard(guard)  # before anything below is used
    res = _result(t, bricks, np.asarray(dec["slots"]), np.asarray(owned),
                  np.asarray(q["w_codes"]), np.asarray(q["scales"]),
                  np.asarray(q["owned_counts"]), n_own, int(n_out))
    if with_forces:
        res["forces"] = _forces(owned, g_short, g_long, v_new)
    t_end = time.perf_counter()
    res["busy"] = dict(decode=t_short - t_dec, short=t_long - t_short,
                       long=t_quant - t_long, quant=t_end - t_quant)
    return res


def _forces(owned, g_short, g_long, v_new):
    o = np.asarray(owned)
    return dict(g_short=np.asarray(g_short)[o], g_long=np.asarray(g_long)[o],
                v_new=np.asarray(v_new)[o])


def _result(t, bricks, slots, owned, w_codes, scales, counts, n_own, n_out):
    written = counts > 0
    return dict(t=t, empty=False, slots_o=slots[owned], w_codes=w_codes[owned],
                run_bricks=bricks[written], run_scales=scales[written],
                n_owned=int(n_own), n_out=int(n_out))


def _tile_task_jit(st, one_tile, C, g_coarse, t, bricks, plan, origin, o_cells,
                   extent, shapes, with_forces=False, device_state=None, timings=None):
    import jax
    import jax.numpy as jnp

    from ..eject_jax import require_x64
    from ..forces import check_stencil_guard, stage_coarse_subblock

    require_x64()
    timed = timings is not None
    clock = [time.perf_counter()]

    def mark(name, *arrays):
        if timed:
            for a in arrays:
                jax.block_until_ready(a)
            now = time.perf_counter()
            timings[name] = now - clock[0]
            clock[0] = now

    nb = int(C["n_fine"]) // int(C["n_brick"])
    n_b = len(bricks)
    R = int(shapes["arena_rect"])
    a = plan["arena_slots"]
    if int(plan["arena_counts"].max()) > R:
        raise ValueError(
            f"tile {t}: {int(plan['arena_counts'].max())} arena residents in one "
            f"brick > the step's arena_rect {R}; rebuild shapes with tile_step_shapes")
    rect = np.full((n_b, R), -1, dtype=np.int64)
    rect[:, : min(a.shape[1], R)] = a[:, :R]
    fn = _tile_kernel(
        one_tile, cap=int(C["cap"]), n_b=n_b, p3=int(plan["p3"]),
        per=int(st.t9.n_buckets_side // nb), nb=nb, n_tile=int(C["n_tile"]),
        n_brick=int(C["n_brick"]), n_coarse=int(C["n_coarse"]),
        coarse_cell=C["coarse_cell"], box=C["box"], t9=st.t9, with_forces=with_forces)
    sub = [stage_coarse_subblock(g, o_cells, extent) for g in g_coarse]
    mark("stage")

    head = [jnp.asarray(plan["starts"]), jnp.asarray(plan["occ"]),
            jnp.asarray(plan["live_counts"]), jnp.asarray(rect),
            jnp.asarray(plan["row_offsets"]), jnp.asarray(bricks)]
    tail = [jnp.asarray(int(st.arena_base), dtype=jnp.int64),
            jnp.asarray(int(plan["n_rows"]), dtype=jnp.int64),
            jnp.asarray(origin, dtype=jnp.float64),
            jnp.asarray(np.asarray(t, dtype=np.int64)),
            jnp.asarray(np.asarray(o_cells), dtype=jnp.int32),
            *(jnp.asarray(s) for s in sub),
            jnp.asarray(C["alpha_k"], dtype=jnp.float64),
            jnp.asarray(C["bcoef"], dtype=jnp.float64),
            jnp.full((n_b,), 32767.0, dtype=jnp.float64)]
    mark("h2d_tile", *head, *tail)

    ds = device_state
    if ds is None:
        ds = {k: jnp.asarray(getattr(st, k)) for k in STATE_FIELDS}
    mark("h2d_state", *ds.values())

    out = fn(*head, ds["off"], ds["w"], ds["vel_scale"], ds["arena_bucket"], *tail)
    mark("compute", *out.values())

    host = {k: np.asarray(v) for k, v in out.items()}
    mark("d2h")

    n_own = int(host["n_own"])
    if n_own == 0:
        return dict(t=t, empty=True, n_owned=0, n_out=int(host["n_out"]))
    check_stencil_guard([(host["lo"], host["hi"], int(extent), "tsc")])
    res = _result(t, bricks, host["slots"], host["owned"], host["w_codes"],
                  host["scales"], host["counts"], n_own, int(host["n_out"]))
    if with_forces:
        res["forces"] = _forces(host["owned"], host["g_short"], host["g_long"],
                                host["v_new"])
    mark("result")
    res["busy"] = dict(decode=0.0, short=0.0, long=0.0, quant=0.0)
    return res
