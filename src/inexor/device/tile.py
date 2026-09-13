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
- The coarse sub-blocks are staged on the host (`stage_coarse_subblock`) and
  copied per tile, unless `coarse_shard=` holds the meshes on the card, where the
  design keeps them (record sec. 23).
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

THE RESULTS ON THE DEVICE (`tile_loop_device`). Returning a tile's owned slots
and codes to the host costs a readback of padded rows and two boolean-mask
gathers over them: 62 + 214 ms of a 389 ms tile at P=576 (Vista 993817). The
step-level loop instead writes each tile's codes and per-brick scales into the
device state inside the same program, returns only scalars, and copies the
state back to the host once, after every guard has been resolved and the
ownership partition checked. Order cannot matter: a tile writes only rows it
owns, and a later tile reads those rows only as buffer, whose velocities are
decoded and never used.

Nothing here is wired into `engine.step`.
"""

from __future__ import annotations

import threading
import time

import numpy as np

# One compiled tile program per set of static parameters, and a count of
# traces: the receipt that one executable served every tile. The lock guards
# the cache when one thread per card asks for the program first.
_KERNELS = {}
_TRACES = [0]
_KERNEL_LOCK = threading.Lock()

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


def stage_state_on_device(st, device=None):
    """The state arrays the jitted tile decodes against, COPIED onto `device`
    (None: jax's default device) once, for `device_state=`. Blocks until placed.

    A copy on every backend, never a view of the host arrays: `tile_loop_device`
    donates `w` and `vel_scale` to its program, and a donated buffer that aliased
    numpy memory would be overwritten under the host state.
    """
    import jax
    import jax.numpy as jnp

    if device is None:
        return {k: jax.block_until_ready(jnp.array(getattr(st, k), copy=True))
                for k in STATE_FIELDS}
    return {k: jax.block_until_ready(jax.device_put(np.array(getattr(st, k), copy=True),
                                                    device))
            for k in STATE_FIELDS}


def _tile_kernel(one_tile, *, cap, n_b, p3, per, nb, n_tile, n_brick, n_coarse,
                 coarse_cell, box, t9, with_forces=False, write=False, coarse_extent=None):
    key = (id(one_tile), cap, n_b, p3, per, nb, n_tile, n_brick, n_coarse,
           float(coarse_cell), float(box), float(t9.quantum), int(t9.n_buckets_side),
           bool(with_forces), bool(write), coarse_extent)
    fn = _KERNELS.get(key)
    if fn is not None:
        return fn
    import jax
    import jax.numpy as jnp

    from ..forces import gather_coarse_subblock
    from .coarse import subblock_device
    from .decode import decode_core
    from .kick import quantize_per_brick

    def tile(starts, occ, live_counts, arena_slots, row_offsets, bricks, off, w,
             vel_scale, arena_bucket, arena_base, n_rows, origin, tijk, o_cells,
             sub_x, sub_y, sub_z, alpha_k, bcoef, scale_div, *shard_x0):
        # With `coarse_extent` set, sub_x/y/z arrive as the card's coarse SHARD
        # meshes and the tile gathers its own blocks (`device.coarse`); otherwise
        # they are the host-staged blocks.
        _TRACES[0] += 1  # trace time only
        if coarse_extent is not None:
            sub_x, sub_y, sub_z = subblock_device((sub_x, sub_y, sub_z), shard_x0[0],
                                                  n_coarse, o_cells, coarse_extent)
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
        return dict(slots=slots, owned=owned, n_out=n_out, n_own=jnp.sum(owned),
                    w_codes=q["w_codes"], scales=q["scales"],
                    counts=q["owned_counts"], lo=lo, hi=hi,
                    g_short=g_short, g_long=g_long, v_new=v_new)

    def body(*args):
        out = tile(*args)
        keep = ("slots", "owned", "n_out", "n_own", "w_codes", "scales", "counts", "lo", "hi")
        if with_forces:
            keep += ("g_short", "g_long", "v_new")
        return {k: out[k] for k in keep}

    def body_write(*args):
        out = tile(*args)
        w, vel_scale, bricks = args[7], args[8], args[5]
        # codes: widen, subtract the stored code, mask to owned rows, narrow --
        # the codec's modular rule (`codec.isub`). Adding that difference back
        # gives the new code exactly, and a row this tile does not own (padding
        # included, all at slot 0) adds exactly zero, so repeated indices cannot
        # collide the way a scatter-SET would
        slots, owned = out["slots"], out["owned"]
        inc = jnp.where(owned[:, None],
                        out["w_codes"].astype(jnp.int32) - w[slots].astype(jnp.int32), 0)
        w_new = w.at[slots].add(inc.astype(w.dtype), mode="promise_in_bounds")
        # scales: a set on the tile's own bricks (unique within a tile), keeping
        # the stored scale where the tile owns no rows
        keep_old = out["counts"] == 0
        vs_new = vel_scale.at[bricks].set(
            jnp.where(keep_old, vel_scale[bricks], out["scales"]), mode="promise_in_bounds")
        return w_new, vs_new, {k: out[k] for k in ("n_own", "n_out", "lo", "hi")}

    fn = jax.jit(body_write, donate_argnums=(7, 8)) if write else jax.jit(body)
    with _KERNEL_LOCK:
        return _KERNELS.setdefault(key, fn)


def tile_task_device(st, one_tile, C, g_coarse, t, bricks, jit=False, shapes=None,
                     with_forces=False, device_state=None, timings=None, coarse_shard=None):
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

    `coarse_shard` (jit only) is a `device.coarse` shard of the coarse force
    meshes on the device; the program then gathers the tile's sub-blocks there
    and `g_coarse` is not read.

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
                              o_cells, extent, shapes, with_forces, device_state, timings,
                              coarse_shard)
    if device_state is not None or timings is not None or coarse_shard is not None:
        raise ValueError("device_state, timings and coarse_shard apply to the jitted "
                         "path only")

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


def _clock(timings, accumulate=False):
    """A phase marker: syncs on the arrays given and records the elapsed time
    under a name, only when `timings` is a dict."""
    import jax

    clock = [time.perf_counter()]

    def mark(name, *arrays):
        if timings is None:
            return
        for a in arrays:
            jax.block_until_ready(a)
        now = time.perf_counter()
        dt = now - clock[0]
        timings[name] = timings.get(name, 0.0) + dt if accumulate else dt
        clock[0] = now

    return mark


def _jit_inputs(st, one_tile, C, g_coarse, t, bricks, plan, origin, o_cells, extent,
                shapes, with_forces, write, mark, coarse_shard=None, device=None):
    """(program, leading args, trailing args) for one tile; the state arrays go
    between them. Host work and the per-tile upload, marked `stage` and
    `h2d_tile`. With `coarse_shard` (`device.coarse`), the shard meshes go in
    place of host-staged blocks and the program gathers them; they must already
    be on `device`, where every other input is placed (None: jax's default)."""
    from ..forces import stage_coarse_subblock
    from .coarse import check_covers
    from .paint import _on

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
        coarse_cell=C["coarse_cell"], box=C["box"], t9=st.t9, with_forces=with_forces,
        write=write, coarse_extent=None if coarse_shard is None else int(extent))
    d = device
    if coarse_shard is None:
        sub = [_on(stage_coarse_subblock(g, o_cells, extent), d) for g in g_coarse]
        extra = []
    else:
        if int(coarse_shard["n"]) != int(C["n_coarse"]):
            raise ValueError(f"coarse shard n={coarse_shard['n']} against n_coarse "
                             f"{C['n_coarse']}")
        check_covers(coarse_shard, o_cells, extent)
        sub = list(coarse_shard["meshes"])  # already resident; never pulled back
        extra = [_on(int(coarse_shard["x0"]), d, np.int64)]
    mark("stage")

    head = [_on(plan["starts"], d), _on(plan["occ"], d), _on(plan["live_counts"], d),
            _on(rect, d), _on(plan["row_offsets"], d), _on(bricks, d)]
    tail = [_on(int(st.arena_base), d, np.int64),
            _on(int(plan["n_rows"]), d, np.int64),
            _on(origin, d, np.float64),
            _on(np.asarray(t, dtype=np.int64), d),
            _on(np.asarray(o_cells), d, np.int32),
            *sub,
            _on(C["alpha_k"], d, np.float64),
            _on(C["bcoef"], d, np.float64),
            _on(np.full((n_b,), 32767.0), d, np.float64), *extra]
    mark("h2d_tile", *head, *tail)
    return fn, head, tail


def _tile_task_jit(st, one_tile, C, g_coarse, t, bricks, plan, origin, o_cells,
                   extent, shapes, with_forces=False, device_state=None, timings=None,
                   coarse_shard=None):
    import jax.numpy as jnp

    from ..eject_jax import require_x64
    from ..forces import check_stencil_guard

    require_x64()
    mark = _clock(timings)
    fn, head, tail = _jit_inputs(st, one_tile, C, g_coarse, t, bricks, plan, origin,
                                 o_cells, extent, shapes, with_forces, False, mark,
                                 coarse_shard)
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


def tile_loop_device(st, one_tile, C, g_coarse, members, shapes, tiles=None,
                     device_state=None, write_host=True, timings=None, coarse_shard=None,
                     device=None):
    """One step's tile loop on the device, writing the kick into the device state.

    `members` maps tile -> bricks (`SlotState.tile_bricks`) and `tiles` selects
    and orders them (default: every tile in `members`, which is then required to
    partition the particles). Each tile's codes and per-brick scales are written
    into `device_state` (default: a fresh `stage_state_on_device(st)`) inside
    its compiled program, which donates the old `w` and `vel_scale`; the dict is
    updated in place. Only scalars come back per tile.

    After the loop, every tile's stencil guard is resolved and, for a full step,
    the owned total checked against `st.n_particles` -- both BEFORE anything is
    used. Then, if `write_host`, `st.w` and `st.vel_scale` are overwritten from
    the device once.

    Returns `n_owned`, `n_out`, `tiles_run` and `device_state`. `timings`, if a
    dict, accumulates synced phase seconds over the loop (`plan`, `stage`,
    `h2d_tile`, `compute`) plus `d2h_state` once. `coarse_shard`, as in
    `tile_task_device`, gathers the coarse sub-blocks on the device.

    `device` places every per-tile input, and a default `device_state`, on that
    device (None: jax's default); a caller-supplied `device_state` and
    `coarse_shard` must already live there. One thread per card, each with its
    own `device`, state copy and tiles, is the four-card form.
    """
    import jax.numpy as jnp

    from ..eject_jax import require_x64
    from ..forces import (
        COARSE_HALO,
        check_stencil_guard,
        coarse_subblock_origin_extent,
        tile_origin_extent,
    )
    from .decode import tile_decode_plan

    require_x64()
    full_step = tiles is None
    tiles = list(members) if full_step else list(tiles)
    ds = stage_state_on_device(st, device) if device_state is None else device_state
    mark = _clock(timings, accumulate=True)
    cap = int(C["cap"])
    los, his, extents, owns, outs = [], [], [], [], []
    mark("h2d_state_once", *ds.values())
    for t in tiles:
        bricks = np.asarray(members[t], dtype=np.int64)
        plan = tile_decode_plan(st, bricks)
        m = int(plan["n_rows"])
        if m == 0:
            continue
        if m > cap:
            raise RuntimeError(f"tile {t}: {m} members > cap {cap}")
        origin, _ = tile_origin_extent(t, C["n_tile"], C["b_real"], C["cell"])
        o_cells, extent = coarse_subblock_origin_extent(
            t, C["n_tile"], C["n_coarse"], C["n_fine"], halo=COARSE_HALO)
        mark("plan")
        fn, head, tail = _jit_inputs(st, one_tile, C, g_coarse, t, bricks, plan, origin,
                                     o_cells, extent, shapes, False, True, mark,
                                     coarse_shard, device)
        w_new, vs_new, sc = fn(*head, ds["off"], ds["w"], ds["vel_scale"],
                               ds["arena_bucket"], *tail)
        ds["w"], ds["vel_scale"] = w_new, vs_new
        mark("compute", w_new, vs_new, *sc.values())
        los.append(sc["lo"])
        his.append(sc["hi"])
        extents.append(int(extent))
        owns.append(sc["n_own"])
        outs.append(sc["n_out"])

    if los:
        lo_h = np.asarray(jnp.stack(los))
        hi_h = np.asarray(jnp.stack(his))
        check_stencil_guard([(lo_h[i], hi_h[i], extents[i], "tsc") for i in range(len(los))])
        n_owned = int(np.asarray(jnp.sum(jnp.stack(owns))))
        n_out = int(np.asarray(jnp.sum(jnp.stack(outs))))
    else:
        n_owned = n_out = 0
    if full_step and n_owned != st.n_particles:
        raise AssertionError(
            f"ownership is not a partition: {n_owned} rows owned across the step's "
            f"tiles against {st.n_particles} particles; the host state was NOT written")
    if write_host:
        st.w[...] = np.asarray(ds["w"])
        st.vel_scale[...] = np.asarray(ds["vel_scale"])
        mark("d2h_state")
    return dict(n_owned=n_owned, n_out=n_out, tiles_run=len(los), device_state=ds)
