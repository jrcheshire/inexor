"""The device tile loop against a window of x-slabs, not the whole state.

Every tile of one tile plane (tiles sharing an x index) draws its rows from the same `span`
consecutive x-slabs (`layout.brick_span`), so the loop walks planes in x order and holds one
plane's slabs on the card at a time. With `y_blocks > 1` a plane is walked in y-blocks
(`decomp.y_blocks`, whole tile rows), and a block's window is only the brick-y range its
tiles reach, `[y_lo - pad, y_hi + pad)` of each slab (mod nb: one or two brick runs per
slab). Per (plane, block): stage the window (the runs' slot ranges of `off` and `w`, then
their bricks' arena residents), run its tiles through `tile._tile_kernel` with plans rebased
into window rows, resolve the stencil guards, check the block owned exactly the particles of
its core bricks, and only then copy back the rows it owns. `vel_scale` stays whole on the
card and returns once at the end.

Bitwise the whole-state loop at any block count: a tile writes only rows and bricks it owns,
a (plane, block) owns exactly its core bricks, `off` never changes, and buffer rows'
velocities are decoded but never used. Only the core returns because a buffer brick of one
window is a core brick of another: write-backs are disjoint, which makes concurrent per-card
threads safe.

Window shapes sit on `forces.capacity_shape`'s ladder (`window_shapes`), so every plane of a
step runs one program. With `census=c_drift`, each core slab's rows then go through the
migrate's rows program and eject kernel against the window and destination bricks are
counted on the card: the post-migrate per-brick membership, before the migrate runs.
"""

from __future__ import annotations

import numpy as np

#: Count of `tile_loop_windowed` calls (see `migrate.CALLS`).
CALLS = 0


def plane_geometry(st, n_tile, b_real, n_brick):
    """(bricks per side, pad, span, bricks per tile) for this state's tiles."""
    from ..layout import brick_span

    nb = int(st.bricks_per_side)
    pad, span = brick_span(int(n_tile), int(b_real), int(n_brick), nb)
    return nb, int(pad), int(span), int(n_tile) // int(n_brick)


def window_slabs(plane, per, pad, span, nb):
    """The `span` consecutive x-slabs (mod nb), in order, that tile plane `plane`
    draws from; its core slabs are `plane * per .. plane * per + per - 1`."""
    return [int((int(plane) * per - pad + s) % nb) for s in range(span)]


def tile_blocks(nb, per, y_blocks=1):
    """The brick-y blocks a tile plane is walked in (`decomp.y_blocks`)."""
    from ..decomp import y_blocks as _y_blocks

    return _y_blocks(int(nb) // int(per), int(per), int(y_blocks))


def window_runs(plane, block, per, pad, span, nb):
    """The brick runs `(s, lo_b, hi_b)` the tiles of tile plane `plane` whose rows lie in
    brick-y `block` draw from: for each of `window_slabs`, the brick-y range
    `[y_lo - pad, y_hi - per - pad + span)` mod nb, as one run (the whole slab when the range
    covers the grid) or two (ascending), each a contiguous brick range of that slab."""
    y_lo, y_hi = (int(v) for v in block)
    a, b = y_lo - pad, y_hi - per - pad + span
    if b - a >= nb:
        ys = [(0, nb)]
    elif a < 0:
        ys = [(0, b), (nb + a, nb)]
    elif b > nb:
        ys = [(0, b - nb), (a, nb)]
    else:
        ys = [(a, b)]
    nb2 = nb * nb
    return [(s, s * nb2 + y0 * nb, s * nb2 + y1 * nb)
            for s in window_slabs(plane, per, pad, span, nb) for y0, y1 in ys]


def _as_runs(slabs_or_runs, nb):
    """Runs from a list of runs, or of whole x-slabs."""
    nb2 = nb * nb
    return [tuple(int(v) for v in x) if np.ndim(x) else (int(x), int(x) * nb2,
                                                          (int(x) + 1) * nb2)
            for x in slabs_or_runs]


def window_shapes(st, n_tile, b_real, n_brick, planes=None, floor=None, y_blocks=1):
    """Fixed per-step window shapes: `rows` bounds every (plane, y-block) window's live slot
    span and `arena` its residents, each on `forces.capacity_shape`'s ladder with `floor`
    (a previous step's shapes) keeping them monotone. O(slabs x blocks + arena).

    `st` may be a `device.ghost.SlabView`; `planes` are then this rank's, and the shapes
    every rank compiles are the maxima over ranks."""
    from ..forces import capacity_shape
    from .ghost import as_view

    view = as_view(st)
    nb, pad, span, per = plane_geometry(view, n_tile, b_real, n_brick)
    planes = range(nb // per) if planes is None else planes
    res_cum = np.concatenate(([0], np.cumsum(view.brick_resident_counts())))
    rows = {}
    live_max = res_max = 0
    for p in planes:
        for block in tile_blocks(nb, per, y_blocks):
            runs = window_runs(p, block, per, pad, span, nb)
            for q in runs:
                if q not in rows:
                    lo, hi = view.range_rows(*q)
                    rows[q] = hi - lo
            # the runs go up as ladder-length chunks and the last one's pad must fit
            pads = max(_ladder(rows[q]) - rows[q] for q in runs)
            live_max = max(live_max, sum(rows[q] for q in runs) + pads)
            res_max = max(res_max, sum(int(res_cum[q[2]] - res_cum[q[1]]) for q in runs))
    floor = floor or {}
    return dict(rows=int(capacity_shape(max(live_max, 1), floor_shape=int(floor.get("rows", 0)))),
                arena=int(capacity_shape(max(res_max, 1), floor_shape=int(floor.get("arena", 0)))))


def _ladder(n):
    from ..forces import capacity_shape

    return int(capacity_shape(max(1, int(n))))


def _slab_fits(n_state_rows, lo, L):
    """True when `L` rows from slot `lo` lie inside the state: the slab chunk is a view."""
    return lo + L <= n_state_rows


def _window_programs(L, rows, trailing, dtype):
    """(place an `L`-row chunk into a `rows`-row buffer at a runtime row, donated; read
    `L` rows back from a runtime row)."""
    import jax

    from .migrate import _program

    def make_place():
        def place(buf, chunk, r):
            return jax.lax.dynamic_update_slice(buf, chunk, (r,) + (0,) * trailing)

        return jax.jit(place, donate_argnums=0)

    def make_take():
        return jax.jit(lambda buf, r: jax.lax.dynamic_slice(buf, (r,) + (0,) * trailing,
                                                            (L,) + buf.shape[1:]))

    key = (L, rows, trailing, str(dtype))
    return (_program(("window_place",) + key, make_place),
            _program(("window_take",) + key, make_take))


def stage_window(st, slabs, shapes, device=None, index=None):
    """One (plane, y-block) window on `device` (None: jax's default), from the host state.

    `slabs` are brick runs `(s, lo_b, hi_b)` (`window_runs`) or whole x-slabs. Rows
    `0 .. W-1` are the runs' live slot ranges in that order (a window across x = 0 or
    y = 0 holds ranges out of order), padded to `W = shapes["rows"]`; rows `W ..` are the
    arena residents of the runs' bricks in ascending slot, padded to `shapes["arena"]`.
    Returns the device arrays (`off`, `w`, `arena_bucket`; the tile program donates `w`)
    and the host maps `rebase_plan` and the write-back read. Refuses a window past its
    shapes. `index` is `migrate.pass_arena_index` of the state (computed if None).

    Each run goes up as a host view `capacity_shape(its rows)` long (copied only where it
    would run off the array); rows past the run are overwritten by the next, which is why
    `window_shapes` leaves room for the largest pad.

    `st` may be a `device.ghost.SlabView`: runs of slabs another rank owns are then staged
    from its ghost copies (their `w` as zeros), their residents after the owned ones.
    """
    import jax

    from .ghost import as_view
    from .migrate import _put, _zeros

    view = as_view(st)
    st = view.state
    nb = int(view.bricks_per_side)
    runs = _as_runs(slabs, nb)
    W, A = int(shapes["rows"]), int(shapes["arena"])
    off_d = _zeros((W + A, 3), st.off.dtype, device)
    w_d = _zeros((W + A, 3), st.w.dtype, device)
    run_rows, run_win = [], []
    r = copied = 0
    for q in runs:
        lo, hi = view.range_rows(*q)
        L = _ladder(hi - lo)
        if r + L > W:
            raise ValueError(f"window over runs {runs} holds more than rows={W} rows with "
                             "its runs' pads; rebuild the shapes with window_shapes")
        chunks = {name: view.range_chunk(name, *q, L) for name in ("off", "w")}
        copied += not view.range_fits(*q, L)
        for name, chunk in chunks.items():
            place, _take = _window_programs(L, W + A, 1, chunk.dtype)
            if name == "off":
                off_d = place(off_d, _put(chunk, device), _put(r, device, np.int64))
            else:
                w_d = place(w_d, _put(chunk, device), _put(r, device, np.int64))
        run_rows.append((lo, hi))
        run_win.append(r)
        r += hi - lo

    res_slots, brick_a, bucket_a = view.residents_of_runs(runs, index=index)
    n_res = len(res_slots)
    if n_res > A:
        raise ValueError(f"window over runs {runs} holds {n_res} arena residents > "
                         f"arena={A}; rebuild the shapes with window_shapes")
    arena_bucket = np.zeros(max(A, 1), dtype=st.arena_bucket.dtype)
    if n_res:
        arena_bucket[:n_res] = bucket_a
        if A:
            for name in ("off", "w"):
                dt = getattr(st, name).dtype
                res = np.zeros((A, 3), dtype=dt)
                view.resident_rows(name, res_slots, res)
                place, _take = _window_programs(A, W + A, 1, dt)
                if name == "off":
                    off_d = place(off_d, _put(res, device), _put(W, device, np.int64))
                else:
                    w_d = place(w_d, _put(res, device), _put(W, device, np.int64))
    ab_d = _put(arena_bucket, device)
    jax.block_until_ready((off_d, w_d, ab_d))
    slabs_x = list(dict.fromkeys(q[0] for q in runs))
    return dict(dev=dict(off=off_d, w=w_d, arena_bucket=ab_d),
                slabs=slabs_x, runs=runs, run_rows=run_rows, run_win=run_win,
                res_slots=res_slots, res_bricks=brick_a, W=W, A=A, live_rows=r,
                n_res=n_res, wrapped=slabs_x != sorted(slabs_x), copied_slabs=copied)


def _run_of(win, bricks):
    """Index into `win["runs"]` of each brick, -1 where no run holds it."""
    b = np.asarray(bricks, dtype=np.int64)
    lo = np.asarray([q[1] for q in win["runs"]], dtype=np.int64)
    hi = np.asarray([q[2] for q in win["runs"]], dtype=np.int64)
    order = np.argsort(lo, kind="stable")
    k = np.searchsorted(lo[order], b, side="right") - 1
    idx = order[np.maximum(k, 0)]
    return np.where((k >= 0) & (b < hi[idx]), idx, -1)


def _window_row(win, k, row):
    """Window row of state row id `row` in run `k`."""
    return int(win["run_win"][k]) + int(row) - int(win["run_rows"][k][0])


def rebase_plan(plan, win, bricks_per_side):
    """`device.decode.tile_decode_plan`'s dict with `starts` and `arena_slots`
    mapped into `win`'s rows. Refuses a brick outside the window or a resident
    the window did not stage."""
    k = _run_of(win, plan["bricks"])
    if (k < 0).any():
        raise ValueError(f"a tile brick lies outside the window's runs {win['runs']}")
    run_lo = np.asarray([a for a, _b in win["run_rows"]], dtype=np.int64)
    starts = plan["starts"] - run_lo[k] + np.asarray(win["run_win"], dtype=np.int64)[k]
    a = plan["arena_slots"]
    valid = a >= 0
    rect = np.full(a.shape, -1, dtype=np.int64)
    if valid.any():
        res = win["res_slots"]
        k = np.searchsorted(res, a[valid])
        kk = np.minimum(k, max(len(res) - 1, 0))
        if len(res) == 0 or not np.array_equal(res[kk], a[valid]):
            raise ValueError("a tile's arena resident was not staged in the window")
        rect[valid] = win["W"] + k
    return dict(plan, starts=starts, arena_slots=rect)


def _core_mask(bricks, plane, per, nb, block):
    """Which of `bricks` are core bricks of (tile plane `plane`, brick-y `block`)."""
    b = np.asarray(bricks, dtype=np.int64)
    y = (b // nb) % nb
    return ((b // (nb * nb)) // per == plane) & (y >= block[0]) & (y < block[1])


def _write_core(st, win, w_dev, plane, per, nb, device=None, block=None):
    """Copy the rows (tile plane `plane`, brick-y `block`) owns from the window's `w` on the
    card into the host state: each core slab's core bricks, downloaded as the ladder of
    rows its run was staged with (one run at a time), and their residents, gathered on the
    card. `block` None = the whole slab."""
    import jax.numpy as jnp

    from ..decomp import unit_bricks
    from .migrate import _host, _put

    block = (0, nb) if block is None else block
    rows = int(w_dev.shape[0])
    for s in range(plane * per, (plane + 1) * per):
        lo_b, hi_b = unit_bricks(s, block, nb)
        k = int(_run_of(win, [lo_b])[0])
        if k < 0 or win["runs"][k][2] < hi_b:
            raise AssertionError(f"core bricks [{lo_b}, {hi_b}) are not one run of the window")
        lo, hi = int(st.brick_start[lo_b]), int(st.brick_start[hi_b])
        if hi == lo:
            continue
        run_lo, run_hi = win["run_rows"][k]
        L = _ladder(run_hi - run_lo)
        _place, take = _window_programs(L, rows, 1, w_dev.dtype)
        got = _host(take(w_dev, _put(int(win["run_win"][k]), device, np.int64)),
                    "window: write-back")
        st.w[lo:hi] = got[lo - run_lo:hi - run_lo]
    if win["n_res"]:
        k = np.flatnonzero(_core_mask(win["res_bricks"], plane, per, nb, block))
        if len(k):
            idx = np.full(_ladder(len(k)), win["W"], dtype=np.int64)
            idx[:len(k)] = win["W"] + k
            got = _host(jnp.take(w_dev, _put(idx, device), axis=0), "window: write-back")
            st.w[win["res_slots"][k]] = got[:len(k)]


def _census_programs(n_bricks, nb2, cap):
    """(slice `nb2` bricks of the whole-grid scales at a runtime brick offset, add a unit's
    destination bricks into the card's per-brick accumulator)."""
    import jax
    import jax.numpy as jnp

    from .migrate import _program

    def make_slice():
        return jax.jit(lambda vs, lo: jax.lax.dynamic_slice(vs, (lo,), (nb2,)))

    def make_add():
        def add(acc, dest, n_rows, p3):
            real = jnp.arange(cap, dtype=jnp.int64) < n_rows
            return acc.at[jnp.where(real, dest // p3, n_bricks)].add(1, mode="drop")

        return jax.jit(add, donate_argnums=0)

    return (_program(("census_slice", nb2), make_slice),
            _program(("census_add", n_bricks, cap), make_add))


def _census_plane(st, win, wd, vs, plane, per, c_drift, ar_index, acc, device, block=None):
    """Count the destination bricks of (tile plane `plane`, brick-y `block`)'s core bricks
    into `acc`, one core unit (slab x block) at a time. Returns (acc, slabs with rows)."""
    import jax
    import jax.numpy as jnp

    from ..decomp import unit_bricks
    from . import migrate

    nb = int(st.bricks_per_side)
    p3 = int(st.buckets_per_brick)
    block = (0, nb) if block is None else block
    has_ids = st.ids is not None
    ar_slots, ar_bricks = ar_index
    rows_win = int(wd["off"].shape[0])
    ids_win = migrate._zeros(rows_win, st.ids.dtype, device) if has_ids else None
    no_clock = migrate._Clock(None)
    slabs = []
    for s in range(plane * per, (plane + 1) * per):
        lo_b, hi_b = unit_bricks(s, block, nb)
        ix = migrate._unit_index(st, lo_b, hi_b, ar_slots, ar_bricks)
        if ix["n_rows"] == 0:
            continue
        k = np.searchsorted(win["res_slots"], ix["rows_a"])
        if ix["n_ar"] and not np.array_equal(
                win["res_slots"][np.minimum(k, len(win["res_slots"]) - 1)], ix["rows_a"]):
            raise ValueError(f"unit [{lo_b}, {hi_b})'s arena residents were not staged in "
                             "its window")
        take = np.full(ix["a_cap"], win["W"], dtype=np.int64)
        take[:ix["n_ar"]] = win["W"] + k
        take_d = migrate._put(take, device)
        ar_rows = (jnp.take(wd["off"], take_d, axis=0), jnp.take(wd["w"], take_d, axis=0),
                   migrate._zeros(ix["a_cap"], st.ids.dtype, device) if has_ids else None)
        r = int(_run_of(win, [lo_b])[0])
        starts = (np.asarray(st.brick_start[lo_b:hi_b], dtype=np.int64)
                  - int(win["run_rows"][r][0]) + int(win["run_win"][r]))
        sl, add = _census_programs(int(st.n_bricks), hi_b - lo_b, ix["cap"])
        scales_d = sl(vs, migrate._put(ix["lo_b"], device, jnp.int64))
        dest, *_rest = migrate._eject_rows(
            st, ix, c_drift, wd["off"], wd["w"], ids_win, ar_rows, migrate._put(starts, device),
            scales_d, rows_win, no_clock, device)
        _rest = None
        acc = add(acc, dest, migrate._put(ix["n_rows"], device, jnp.int64),
                  migrate._put(p3, device, jnp.int64))
        slabs.append(s)
    return jax.block_until_ready(acc), slabs


def tile_loop_windowed(st, one_tile, C, g_coarse, members, shapes, planes=None,
                       coarse_shard=None, device=None, write_host=True, census=None,
                       timings=None, y_blocks=1):
    """`tile.tile_loop_device` over tile planes, one window on the card at a time. Bitwise
    that loop at any `y_blocks`; see the module docstring.

    `shapes` is `tile.tile_step_shapes` plus `window` from `window_shapes` (at the same
    `y_blocks`). `planes` selects tile planes (default all, in x order); each plane runs
    every tile of `members` with that x index, block by block. `st` may be a
    `device.ghost.SlabView`, whose ghost slabs the windows and decode plans read; everything
    written is owned. Per (plane, block) the stencil guards are resolved and the owned count
    checked against the particles stored in its core bricks BEFORE its rows return to the
    host. `coarse_shard` and `device` as in `tile_loop_device`; multi-card use is one thread
    per card, each with its own planes.

    `timings`, if a dict, accumulates synced wall per part of a plane (`window:
    stage`, `window: tiles`, `window: guards`, `window: census`, `window: write-back`)
    and `window: scales to host` (see `migrate._Clock`).

    `census`, if given, is the step's fused drift `c_drift` (see the module docstring).
    It adds `census_counts` (int64 per brick, this call's core slabs
    only) and `census_slabs`.

    Returns `n_owned`, `n_out`, `tiles_run`, `planes_run`, `vel_scale_kick_max`
    and the window receipts `wrapped`, `window_live_rows_max`, `residents_staged`,
    `window_slabs`, `window_units` (windows staged); with `census`, also `census_units`.
    """
    import jax
    import jax.numpy as jnp

    from ..eject_jax import require_x64
    from ..forces import (
        COARSE_HALO,
        check_stencil_guard,
        coarse_subblock_origin_extent,
        tile_origin_extent,
    )
    from ..decomp import unit_bricks
    from . import tile as dtile
    from .decode import tile_decode_plan
    from .ghost import as_view
    from .migrate import pass_arena_index

    global CALLS
    require_x64()
    CALLS += 1
    view = as_view(st)
    st = view.state
    nb, pad, span, per = plane_geometry(st, C["n_tile"], C["b_real"], C["n_brick"])
    planes = list(range(nb // per)) if planes is None else [int(p) for p in planes]
    blocks = tile_blocks(nb, per, y_blocks)
    ar_index = pass_arena_index(st)
    cap = int(C["cap"])
    stored = st.brick_member_counts()
    vs = (jax.block_until_ready(jnp.array(st.vel_scale, copy=True)) if device is None
          else jax.block_until_ready(jax.device_put(np.array(st.vel_scale, copy=True), device)))
    no_mark = dtile._clock(None)
    from .migrate import _Clock

    clock = _Clock(timings)
    if census is not None:
        from .migrate import _zeros

        census_acc = _zeros(int(st.n_bricks), jnp.int64, device)
        census_slabs, census_units = set(), 0
    n_owned = n_out = tiles_run = units = 0
    smax_all, wrapped, live_max, residents = [], False, 0, 0

    for i, block in ((i, b) for i in planes for b in blocks):
        rows_t = range(block[0] // per, block[1] // per)
        win = stage_window(view, window_runs(i, block, per, pad, span, nb), shapes["window"],
                           device, index=ar_index)
        units += 1
        wrapped |= win["wrapped"]
        live_max = max(live_max, win["live_rows"])
        residents += win["n_res"]
        wd = win["dev"]
        clock.mark("window: stage", *wd.values())
        los, his, extents, owns, outs = [], [], [], [], []
        for t in (t for t in members if int(t[0]) == i and int(t[1]) in rows_t):
            bricks = np.asarray(members[t], dtype=np.int64)
            plan = tile_decode_plan(view, bricks)
            m = int(plan["n_rows"])
            if m == 0:
                continue
            if m > cap:
                raise RuntimeError(f"tile {t}: {m} members > cap {cap}")
            origin, _ = tile_origin_extent(t, C["n_tile"], C["b_real"], C["cell"])
            o_cells, extent = coarse_subblock_origin_extent(
                t, C["n_tile"], C["n_coarse"], C["n_fine"], halo=COARSE_HALO)
            fn, head, tail = dtile._jit_inputs(
                st, one_tile, C, g_coarse, t, bricks, rebase_plan(plan, win, nb), origin,
                o_cells, extent, shapes, False, True, no_mark, coarse_shard, device,
                arena_base=win["W"])
            w_new, vs, sc = fn(*head, wd["off"], wd["w"], vs, wd["arena_bucket"], *tail)
            wd["w"] = w_new
            los.append(sc["lo"])
            his.append(sc["hi"])
            extents.append(int(extent))
            owns.append(sc["n_own"])
            outs.append(sc["n_out"])
            smax_all.append(sc["scale_max"])
            tiles_run += 1

        clock.mark("window: tiles", wd["w"], vs)
        # the plane's guards and partition, BEFORE anything of it reaches the host
        own_i = 0
        if los:
            lo_h, hi_h = np.asarray(jnp.stack(los)), np.asarray(jnp.stack(his))
            check_stencil_guard([(lo_h[k], hi_h[k], extents[k], "tsc") for k in range(len(los))])
            own_i = int(np.asarray(jnp.sum(jnp.stack(owns))))
            n_out += int(np.asarray(jnp.sum(jnp.stack(outs))))
        want = sum(int(stored[slice(*unit_bricks(s, block, nb))].sum())
                   for s in range(i * per, (i + 1) * per))
        if own_i != want:
            raise AssertionError(
                f"tile plane {i} (brick-y {block[0]}..{block[1]}) owned {own_i} rows against "
                f"{want} stored in its core bricks; nothing of it was written to the host")
        n_owned += own_i
        clock.mark("window: guards")
        if census is not None:
            census_acc, got = _census_plane(st, win, wd, vs, i, per, float(census),
                                            ar_index, census_acc, device, block=block)
            census_slabs.update(got)
            census_units += len(got)
            clock.mark("window: census", census_acc)
        if write_host:
            _write_core(st, win, wd["w"], i, per, nb, device, block=block)
            clock.mark("window: write-back")
        del win, wd

    nb2 = nb * nb
    if write_host and planes:
        vh = np.asarray(vs)
        for i in planes:
            st.vel_scale[i * per * nb2:(i + 1) * per * nb2] = vh[i * per * nb2:(i + 1) * per * nb2]
        clock.mark("window: scales to host")
    kick_max = None
    if smax_all:
        mx = float(np.asarray(jnp.max(jnp.stack(smax_all))))
        kick_max = mx if np.isfinite(mx) else None
    out = dict(n_owned=n_owned, n_out=n_out, tiles_run=tiles_run, planes_run=len(planes),
               vel_scale_kick_max=kick_max, wrapped=wrapped,
               window_live_rows_max=live_max, residents_staged=residents,
               window_slabs=span, window_units=units)
    if census is not None:
        from .migrate import _host

        out.update(census_counts=_host(census_acc, "window: destination counts"),
                   census_slabs=len(census_slabs), census_units=census_units)
    return out
