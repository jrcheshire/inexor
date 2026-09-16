"""E2: the tile loop against a window of x-slabs, not the whole state.

WHAT THIS REPLACES. `tile.tile_loop_device` decodes against the WHOLE state on
the card (`stage_state_on_device`): right at development scale, ~724 GB against a
199 GB card at 4096^3. A tile draws its rows from `span` consecutive x-slabs of
bricks (`layout.brick_span`: 18 of 256 at 4096^3), and every tile of one tile
PLANE -- tiles sharing an x index -- draws from the same `span` slabs. So the loop
walks tile planes in x order and holds one plane's slabs on the card at a time.

PER PLANE: stage the window (the slabs' slot ranges of `off` and `w`, then the
arena residents of their bricks), run the plane's tiles through
`tile._tile_kernel` with their decode plans rebased into window rows, resolve the
plane's stencil guards, check that the plane owned exactly the particles stored
in its core slabs, and only then copy back the rows it OWNS: its core slabs' runs
and their bricks' residents. `vel_scale` stays whole on the card for the loop
(8 B per brick, the kernel indexes it by global brick id) and the planes' core
bricks return once, at the end.

WHY IT IS BITWISE the whole-state loop. A tile writes only rows and bricks it
owns; the rows a plane owns are exactly those of its `per` core slabs, and every
slab belongs to one plane. `off` never changes in the loop, and buffer rows'
velocities are decoded but never used. So a window staged from the host reads, in
every row that matters, what the whole-state loop reads.

WHY ONLY THE CORE RETURNS. Planes on different cards run concurrently (E3). A
buffer slab of one plane is a core slab of another, so copying a whole window
back would race a neighbour's write-back with stale codes. Each plane's
write-back is disjoint from every other's, which is what makes the cards' threads
safe; a concurrent read of a neighbour's core rows while it writes lands only in
buffer velocities, which are never used.

SHAPES. The window is padded to per-step `rows` and `arena` on
`forces.capacity_shape`'s ladder (`window_shapes`), so every plane of a step runs
one program.

THE DESTINATION CENSUS (`census=c_drift`). After a plane's tiles have kicked and
before its write-back, every core slab's rows go through the migrate's own rows
program and compiled eject kernel (`migrate._slab_index`, `migrate._eject_rows`)
against the window, and each real row's destination brick is counted on the card.
Same executable, same values in every row it reads, so the counts are the
post-migrate per-brick membership the migrate will produce, before it runs: what
a fused migrate + repack needs for the repack's `new_start`.
"""

from __future__ import annotations

import numpy as np

#: RECEIPT: windowed tile loops run through this module (see `migrate.CALLS`).
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


def _slab_edges(st):
    """Slot index where each x-slab's live runs start, and the end: (nb + 1,)."""
    nb = int(st.bricks_per_side)
    return np.asarray(st.brick_start, dtype=np.int64)[::nb * nb][:nb + 1]


def _arena_rows(st):
    """(arena row indices, the brick of each) for occupied arena rows."""
    if not st.n_arena:
        e = np.empty(0, dtype=np.int64)
        return e, e
    ab = np.asarray(st.arena_bucket, dtype=np.int64)
    rows = np.flatnonzero(ab >= 0)
    return rows, ab[rows] // int(st.buckets_per_brick)


def window_shapes(st, n_tile, b_real, n_brick, planes=None, floor=None):
    """Fixed per-step window shapes: `rows` bounds every plane's live slot span and
    `arena` its residents, each on `forces.capacity_shape`'s ladder with `floor`
    (a previous step's shapes) keeping them monotone. O(slabs + arena)."""
    from ..forces import capacity_shape

    nb, pad, span, per = plane_geometry(st, n_tile, b_real, n_brick)
    planes = range(nb // per) if planes is None else planes
    rows = np.diff(_slab_edges(st))
    _r, res_brick = _arena_rows(st)
    res = (np.bincount(res_brick // (nb * nb), minlength=nb) if len(res_brick)
           else np.zeros(nb, dtype=np.int64))
    live_max = res_max = 0
    for p in planes:
        s = window_slabs(p, per, pad, span, nb)
        live_max = max(live_max, int(rows[s].sum()))
        res_max = max(res_max, int(res[s].sum()))
    floor = floor or {}
    return dict(rows=int(capacity_shape(max(live_max, 1), floor_shape=int(floor.get("rows", 0)))),
                arena=int(capacity_shape(max(res_max, 1), floor_shape=int(floor.get("arena", 0)))))


def stage_window(st, slabs, shapes, device=None):
    """One plane's window on `device` (None: jax's default), from the host state.

    Rows `0 .. W-1` are the slabs' live slot ranges in `slabs` order (a window
    across x = 0 is two ranges, concatenated), padded to `W = shapes["rows"]`;
    rows `W ..` are the arena residents of the slabs' bricks in ascending slot,
    padded to `shapes["arena"]`. Returns the device arrays (`off`, `w`,
    `arena_bucket`, copies: the tile program donates `w`) and the host maps
    `rebase_plan` and the write-back read. Refuses a window past its shapes.
    """
    import jax
    import jax.numpy as jnp

    nb = int(st.bricks_per_side)
    nb2 = nb * nb
    W, A = int(shapes["rows"]), int(shapes["arena"])
    edges = _slab_edges(st)
    off = np.zeros((W + A, 3), dtype=st.off.dtype)
    w = np.zeros((W + A, 3), dtype=st.w.dtype)
    slab_abs = np.full(nb, -1, dtype=np.int64)
    slab_win = np.full(nb, -1, dtype=np.int64)
    r = 0
    for s in slabs:
        lo, hi = int(edges[s]), int(edges[s + 1])
        if r + hi - lo > W:
            raise ValueError(f"window over slabs {slabs} holds more than rows={W} live rows; "
                             "rebuild the shapes with window_shapes")
        off[r:r + hi - lo] = st.off[lo:hi]
        w[r:r + hi - lo] = st.w[lo:hi]
        slab_abs[s], slab_win[s] = lo, r
        r += hi - lo

    rows_a, brick_a = _arena_rows(st)
    in_win = np.zeros(nb, dtype=bool)
    in_win[list(slabs)] = True
    keep = in_win[brick_a // nb2] if len(rows_a) else np.zeros(0, dtype=bool)
    rows_a, brick_a = rows_a[keep], brick_a[keep]
    n_res = len(rows_a)
    if n_res > A:
        raise ValueError(f"window over slabs {slabs} holds {n_res} arena residents > "
                         f"arena={A}; rebuild the shapes with window_shapes")
    res_slots = int(st.arena_base) + rows_a
    arena_bucket = np.zeros(max(A, 1), dtype=st.arena_bucket.dtype)
    if n_res:
        off[W:W + n_res] = st.off[res_slots]
        w[W:W + n_res] = st.w[res_slots]
        arena_bucket[:n_res] = st.arena_bucket[rows_a]

    def put(a):
        if device is None:
            return jax.block_until_ready(jnp.array(a, copy=True))
        return jax.block_until_ready(jax.device_put(a, device))

    return dict(dev=dict(off=put(off), w=put(w), arena_bucket=put(arena_bucket)),
                slabs=list(slabs), slab_abs=slab_abs, slab_win=slab_win, edges=edges,
                res_slots=res_slots, res_bricks=brick_a, W=W, A=A, live_rows=r,
                n_res=n_res, wrapped=list(slabs) != sorted(slabs))


def rebase_plan(plan, win, bricks_per_side):
    """`device.decode.tile_decode_plan`'s dict with `starts` and `arena_slots`
    mapped into `win`'s rows. Refuses a brick outside the window or a resident
    the window did not stage."""
    nb2 = int(bricks_per_side) ** 2
    s = np.asarray(plan["bricks"], dtype=np.int64) // nb2
    if (win["slab_win"][s] < 0).any():
        raise ValueError(f"a tile brick lies in an x-slab outside the window {win['slabs']}")
    starts = plan["starts"] - win["slab_abs"][s] + win["slab_win"][s]
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


def _write_core(st, win, wh, plane, per, nb):
    """Copy the rows tile plane `plane` owns from the window's `w` (host copy `wh`)
    into the host state: its core slabs' live runs and their bricks' residents."""
    nb2 = nb * nb
    edges = win["edges"]
    for s in range(plane * per, (plane + 1) * per):
        lo, hi, r = int(edges[s]), int(edges[s + 1]), int(win["slab_win"][s])
        st.w[lo:hi] = wh[r:r + hi - lo]
    if win["n_res"]:
        b = win["res_bricks"]
        core = (b >= plane * per * nb2) & (b < (plane + 1) * per * nb2)
        k = np.flatnonzero(core)
        st.w[win["res_slots"][k]] = wh[win["W"] + k]


def _census_programs(n_bricks, nb2, cap):
    """(slice the whole-grid scales at a runtime brick offset, add a slab's
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


def _census_plane(st, win, wd, vs, plane, per, c_drift, ar_index, acc, device):
    """Count the destination bricks of tile plane `plane`'s core slabs into `acc`."""
    import jax
    import jax.numpy as jnp

    from . import migrate

    nb = int(st.bricks_per_side)
    nb2, p3 = nb * nb, int(st.buckets_per_brick)
    has_ids = st.ids is not None
    ar_slots, ar_bricks = ar_index
    rows_win = int(wd["off"].shape[0])
    ids_win = migrate._zeros(rows_win, st.ids.dtype, device) if has_ids else None
    no_clock = migrate._Clock(None)
    n_slabs = 0
    for s in range(plane * per, (plane + 1) * per):
        ix = migrate._slab_index(st, s, ar_slots, ar_bricks)
        if ix["n_rows"] == 0:
            continue
        k = np.searchsorted(win["res_slots"], ix["rows_a"])
        if ix["n_ar"] and not np.array_equal(
                win["res_slots"][np.minimum(k, len(win["res_slots"]) - 1)], ix["rows_a"]):
            raise ValueError(f"slab {s}'s arena residents were not staged in its window")
        take = np.full(ix["a_cap"], win["W"], dtype=np.int64)
        take[:ix["n_ar"]] = win["W"] + k
        take_d = migrate._put(take, device)
        ar_rows = (jnp.take(wd["off"], take_d, axis=0), jnp.take(wd["w"], take_d, axis=0),
                   migrate._zeros(ix["a_cap"], st.ids.dtype, device) if has_ids else None)
        starts = (np.asarray(st.brick_start[ix["lo_b"]:ix["hi_b"]], dtype=np.int64)
                  - int(win["edges"][s]) + int(win["slab_win"][s]))
        sl, add = _census_programs(int(st.n_bricks), nb2, ix["cap"])
        scales_d = sl(vs, migrate._put(ix["lo_b"], device, jnp.int64))
        dest, *_rest = migrate._eject_rows(
            st, ix, c_drift, wd["off"], wd["w"], ids_win, ar_rows, migrate._put(starts, device),
            scales_d, rows_win, no_clock, device)
        _rest = None
        acc = add(acc, dest, migrate._put(ix["n_rows"], device, jnp.int64),
                  migrate._put(p3, device, jnp.int64))
        n_slabs += 1
    return jax.block_until_ready(acc), n_slabs


def tile_loop_windowed(st, one_tile, C, g_coarse, members, shapes, planes=None,
                       coarse_shard=None, device=None, write_host=True, census=None):
    """`tile.tile_loop_device` over tile planes, one window of x-slabs on the card
    at a time. Bitwise that loop; see the module docstring.

    `shapes` is `tile.tile_step_shapes` plus `window` from `window_shapes`.
    `planes` selects tile planes (default all, in x order); each plane runs every
    tile of `members` with that x index. Per plane the stencil guards are resolved
    and the owned count checked against the particles stored in its core slabs
    BEFORE its rows return to the host. `coarse_shard` and `device` as in
    `tile_loop_device`; one thread per card, each with its own planes, is the
    four-card form.

    `census`, if given, is the step's fused drift `c_drift`: see THE DESTINATION
    CENSUS above. It adds `census_counts` (int64 per brick, this call's core slabs
    only) and `census_slabs`.

    Returns `n_owned`, `n_out`, `tiles_run`, `planes_run`, `vel_scale_kick_max`
    and the window receipts `wrapped`, `window_live_rows_max`, `residents_staged`,
    `window_slabs`.
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
    from . import tile as dtile
    from .decode import tile_decode_plan
    from .paint import _arena_per_brick

    global CALLS
    require_x64()
    CALLS += 1
    nb, pad, span, per = plane_geometry(st, C["n_tile"], C["b_real"], C["n_brick"])
    nb2 = nb * nb
    planes = list(range(nb // per)) if planes is None else [int(p) for p in planes]
    cap = int(C["cap"])
    p3 = int(st.buckets_per_brick)
    stored = (np.asarray(st.occupancy).reshape(-1, p3).sum(axis=1, dtype=np.int64)
              + _arena_per_brick(st))
    vs = (jax.block_until_ready(jnp.array(st.vel_scale, copy=True)) if device is None
          else jax.block_until_ready(jax.device_put(np.array(st.vel_scale, copy=True), device)))
    no_mark = dtile._clock(None)
    if census is not None:
        from .migrate import _zeros, pass_arena_index

        census_acc = _zeros(int(st.n_bricks), jnp.int64, device)
        census_index = pass_arena_index(st)
        census_slabs = 0
    n_owned = n_out = tiles_run = 0
    smax_all, wrapped, live_max, residents = [], False, 0, 0

    for i in planes:
        win = stage_window(st, window_slabs(i, per, pad, span, nb), shapes["window"], device)
        wrapped |= win["wrapped"]
        live_max = max(live_max, win["live_rows"])
        residents += win["n_res"]
        wd = win["dev"]
        los, his, extents, owns, outs = [], [], [], [], []
        for t in (t for t in members if int(t[0]) == i):
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

        # the plane's guards and partition, BEFORE anything of it reaches the host
        own_i = 0
        if los:
            lo_h, hi_h = np.asarray(jnp.stack(los)), np.asarray(jnp.stack(his))
            check_stencil_guard([(lo_h[k], hi_h[k], extents[k], "tsc") for k in range(len(los))])
            own_i = int(np.asarray(jnp.sum(jnp.stack(owns))))
            n_out += int(np.asarray(jnp.sum(jnp.stack(outs))))
        want = int(stored[i * per * nb2:(i + 1) * per * nb2].sum())
        if own_i != want:
            raise AssertionError(
                f"tile plane {i} owned {own_i} rows against {want} stored in its core slabs; "
                "nothing of this plane was written to the host")
        n_owned += own_i
        if census is not None:
            census_acc, n_c = _census_plane(st, win, wd, vs, i, per, float(census),
                                            census_index, census_acc, device)
            census_slabs += n_c
        if write_host:
            _write_core(st, win, np.asarray(wd["w"]), i, per, nb)
        del win, wd

    if write_host and planes:
        vh = np.asarray(vs)
        for i in planes:
            st.vel_scale[i * per * nb2:(i + 1) * per * nb2] = vh[i * per * nb2:(i + 1) * per * nb2]
    kick_max = None
    if smax_all:
        mx = float(np.asarray(jnp.max(jnp.stack(smax_all))))
        kick_max = mx if np.isfinite(mx) else None
    out = dict(n_owned=n_owned, n_out=n_out, tiles_run=tiles_run, planes_run=len(planes),
               vel_scale_kick_max=kick_max, wrapped=wrapped,
               window_live_rows_max=live_max, residents_staged=residents,
               window_slabs=span)
    if census is not None:
        from .migrate import _host

        out.update(census_counts=_host(census_acc, "window: destination counts"),
                   census_slabs=census_slabs)
    return out
