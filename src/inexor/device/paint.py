"""D2d: the coarse paint, on the device.

WHAT THIS REPLACES. `engine.coarse_delta_streamed` decodes each chunk of bricks
on the host (`SlotState.decode_bricks`), pads it, checks stencil containment in
numpy and only then paints -- a per-particle host pass per chunk. Here the host
resolves per-brick INDICES and hands over one contiguous slice of the state;
the decode, the containment bounds and the integer sub-block paint all run on
the device.

THE UNIT IS A QUARTER OF AN X-SLAB OF BRICKS by default
(`default_chunk_bricks`). The host path paints `cfg.chunk_bricks` (64) bricks at
a time, which at 4096^3 is 262,144 chunks and so 262,144 device launches per
step; a quarter-slab is 1,024. A whole x-slab (256 launches) was the first
choice and does not fit a card EAGER: at 266 B per padded row (Vista 993139) it
is 90 GB at 4096^3, 1.15x a card. Jitted it is 62 B per row and fits; the
default stays a quarter-slab until chunk time is measured at 4096^3's block
shape (record sec. 13). Any chunk length whose bricks form a cuboid is accepted
(`engine._chunk_cuboid`).

BITWISE against the host paint, and it can be. Positions decode bitwise
(`device.decode`), the paint kernel is the same `paint_tsc_int_subblock` with
the same padding mask, and the accumulation is integer addition -- so neither
the chunk size nor the order chunks arrive in can move a bit.

EAGER OR JITTED (`jit=`). Eager runs the decode and paint op by op, which holds
every intermediate a Python name keeps alive: 266 B per padded row measured on
a card. `jit=True` compiles decode + containment + paint into ONE program per
step. For that the chunk's inputs are padded to fixed per-step shapes
(`step_shapes`) -- the live slice, the arena rectangle and the arena rows -- and
the sub-block origin is a traced value, so every chunk of a step reuses one
executable. XLA is free to fuse the TSC weight arithmetic, which could move a
rounded integer weight, so jit was adopted only on a bitwise gate: on a GB200
(Vista 993294) the jitted density hash-equals a CPU-only host engine at cdev,
plain and with 61,879 arena residents, and every jitted chunk block up to 268M
rows equals the host decode. It holds 62 B per padded row against eager's 266
and is the DEFAULT; `jit=False` is the eager path.

THE ACCUMULATOR IS A SEAM. Where the coarse mesh lives on a gb node is an open
design question: the budget (`plan.DEVICE_PLACEMENT`) shards it across the
cards, the code keeps it on the host. `HostInt64Accumulator` is the host form
the engine has today, and a card-resident accumulator replaces it without
touching the paint.

SCOPE. The containment check comes back as device scalars and is resolved at
the sync the accumulator performs anyway, before the block is added -- the same
arrangement as the tile gather's deferred guard. Nothing here is wired into
`engine.step`; the device executor that would call it does not exist yet.
"""

from __future__ import annotations

import numpy as np

# One compiled chunk program per distinct set of static parameters, and a count
# of how many times any of them was TRACED. The count is the receipt that one
# executable served a whole step: a per-chunk shape would retrace every chunk.
_KERNELS = {}
_TRACES = [0]


def default_chunk_bricks(bricks_per_side):
    """A quarter of an x-slab of bricks, or a whole x-slab where a quarter does
    not tile into cuboids (bricks per side not divisible by 4).

    A quarter-slab is (1, nb/4, nb) bricks, which `engine._chunk_cuboid`
    accepts whenever 4 divides nb -- at every production config nb is 32 or
    more and a power of two.
    """
    nb = int(bricks_per_side)
    return nb * nb // 4 if nb % 4 == 0 else nb * nb


def _arena_per_brick(st):
    p3 = int(st.buckets_per_brick)
    n_b = int(st.n_bricks)
    if not st.n_arena:
        return np.zeros(n_b, dtype=np.int64)
    res = st.arena_bucket[st.arena_bucket >= 0] // p3
    return np.bincount(res, minlength=n_b).astype(np.int64)


def chunk_rows(st, chunk_len):
    """Member rows (live + arena) in each chunk of `chunk_len` consecutive bricks.

    One vectorized pass over the index and the arena, not a Python loop over
    bricks: at 4096^3 there are 16.8M bricks.
    """
    p3 = int(st.buckets_per_brick)
    n_b = int(st.n_bricks)
    per_brick = st.occupancy.reshape(n_b, p3).sum(axis=1, dtype=np.int64)
    per_brick = per_brick + _arena_per_brick(st)
    return per_brick.reshape(-1, int(chunk_len)).sum(axis=1)


def step_shapes(st, chunk_len, pad, floor=None):
    """Fixed per-step input shapes for the jitted chunk, so one program serves it.

    `live_w` bounds every chunk's slot span, `arena_n` its arena residents and
    `arena_rect` the residents of any one brick; each sits on
    `forces.capacity_shape`'s ladder, as `pad` does, and `floor` (a previous
    step's shapes) keeps them monotone across steps for the same reason.
    O(bricks + arena) on the host.
    """
    from ..forces import capacity_shape

    L = int(chunk_len)
    n_b = int(st.n_bricks)
    bs = np.asarray(st.brick_start, dtype=np.int64)
    spans = bs[L::L] - bs[:n_b:L]
    ar = _arena_per_brick(st)
    floor = floor or {}

    def rung(v, key):
        return int(capacity_shape(max(int(v), 1), floor_shape=int(floor.get(key, 0))))

    return dict(pad=int(pad), live_w=rung(spans.max(), "live_w"),
                arena_n=rung(ar.reshape(-1, L).sum(axis=1).max(), "arena_n"),
                arena_rect=rung(ar.max(), "arena_rect"))


def slab_window(st, bricks):
    """(plan, off, arena_bucket, arena_base) re-based onto one contiguous slice.

    In slot order the live runs of consecutive bricks are one contiguous range,
    `[brick_start[b0], brick_start[b_last + 1])`, spare slots included. Their
    arena residents live elsewhere (slots >= `arena_base`) and are appended
    after it. The decode plan's slots are re-based so `device.decode.decode_rows`
    reads this window exactly as it would read the whole state: live starts
    shift by the window's origin, and arena residents become rows
    `span, span+1, ...` whose buckets sit at those offsets past `arena_base = span`.

    Host cost is O(bricks + arena) plus one slice; the whole state never goes
    to the device.
    """
    from .decode import tile_decode_plan

    bricks = _consecutive(bricks)
    plan = tile_decode_plan(st, bricks)
    s0 = int(st.brick_start[bricks[0]])
    s1 = int(st.brick_start[bricks[-1] + 1])
    span = s1 - s0

    a = plan["arena_slots"]
    valid = a >= 0
    flat = a[valid]
    remap = np.full(a.shape, -1, dtype=np.int64)
    remap[valid] = span + np.arange(len(flat), dtype=np.int64)
    if len(flat):
        off = np.concatenate([st.off[s0:s1], st.off[flat]])
        arena_bucket = st.arena_bucket[flat - int(st.arena_base)]
    else:
        off = st.off[s0:s1]
        # never read for a live row, but a gather needs something to index
        arena_bucket = np.zeros(1, dtype=np.int64)
    plan_w = dict(plan, starts=plan["starts"] - s0, arena_slots=remap)
    return plan_w, off, arena_bucket, span


def slab_window_fixed(st, bricks, shapes):
    """`slab_window` at the fixed shapes of `step_shapes`, for the jitted chunk.

    The live slice is padded to `live_w` rows and arena residents follow at row
    `live_w`, so `arena_base` is the same for every chunk of the step. The
    arena rectangle is padded to `arena_rect` columns and the arena rows to
    `arena_n`. Padding is never read for a live row. Refuses a chunk that
    exceeds the step's shapes rather than truncating it.
    """
    from .decode import tile_decode_plan

    bricks = _consecutive(bricks)
    plan = tile_decode_plan(st, bricks)
    b0, L = int(bricks[0]), len(bricks)
    s0 = int(st.brick_start[b0])
    s1 = int(st.brick_start[bricks[-1] + 1])
    span = s1 - s0
    W, A, R = shapes["live_w"], shapes["arena_n"], shapes["arena_rect"]
    a = plan["arena_slots"]
    valid = a >= 0
    flat = a[valid]
    n_ar = len(flat)
    a_cols = int(plan["arena_counts"].max()) if L else 0
    if span > W or n_ar > A or a_cols > R or plan["n_rows"] > shapes["pad"]:
        raise ValueError(
            f"chunk at brick {b0} exceeds the step's fixed shapes (span {span} of "
            f"{W}, arena {n_ar} of {A}, per-brick arena {a_cols} of {R}, rows "
            f"{plan['n_rows']} of {shapes['pad']}); rebuild them with step_shapes")

    off = np.zeros((W + A, 3), dtype=st.off.dtype)
    off[:span] = st.off[s0:s1]
    arena_bucket = np.zeros(A, dtype=np.int64)
    rect = np.full((L, R), -1, dtype=np.int64)
    if n_ar:
        off[W:W + n_ar] = st.off[flat]
        arena_bucket[:n_ar] = st.arena_bucket[flat - int(st.arena_base)]
        sub = np.full(a.shape, -1, dtype=np.int64)
        sub[valid] = W + np.arange(n_ar, dtype=np.int64)
        rect[:, : a.shape[1]] = sub
    p3 = int(st.buckets_per_brick)
    return dict(
        starts=plan["starts"] - s0,
        # the index itself, not the int64 copy the plan makes: a view, 4 B per
        # bucket over the bus, widened on the device
        occ=np.asarray(st.occupancy).reshape(-1, p3)[b0:b0 + L],
        live_counts=plan["live_counts"], arena_slots=rect,
        row_offsets=plan["row_offsets"], bricks=bricks, off=off,
        arena_bucket=arena_bucket, n_rows=int(plan["n_rows"]),
    )


def _consecutive(bricks):
    bricks = np.asarray(bricks, dtype=np.int64)
    if len(bricks) == 0 or not np.array_equal(
            bricks, np.arange(bricks[0], bricks[0] + len(bricks), dtype=np.int64)):
        raise ValueError(
            "a device paint chunk must be a run of CONSECUTIVE brick ids: only "
            "then are its live rows one contiguous slot range")
    return bricks


def chunk_origin_extent(chunk_index, chunk_len, bricks_per_side, n_coarse):
    """The chunk's sub-block, exactly as `engine.coarse_delta_streamed` derives it."""
    from ..engine import _chunk_cuboid

    n = int(n_coarse)
    cub = _chunk_cuboid(chunk_index, chunk_len, bricks_per_side, n)
    if cub is None:
        raise ValueError(
            f"chunk_bricks {chunk_len} does not tile a {bricks_per_side}^3 brick "
            "grid into cuboids. The device paint has no full-mesh fallback: a "
            "full-mesh int32 paint cannot address the meshes this path is for.")
    c0, span = cub
    origin = np.where(span + 3 >= n, 0, (c0 - 1) % n)
    extent = np.where(span + 3 >= n, n, span + 3)
    return origin, extent


def _containment_lohi(x, live, cell, origin, extent, n_coarse):
    """(axes, lo, hi): for each axis the block does not span in full, the lowest
    and highest wrapped local TSC base over live rows. Dead rows are pushed to
    sentinels that cannot win their own reduction. `origin` may be traced;
    `extent` fixes which axes are checked."""
    import jax.numpy as jnp

    n = int(n_coarse)
    base = jnp.rint(x / float(cell)).astype(jnp.int64)
    o = jnp.asarray(origin, dtype=jnp.int64)
    axes, lo, hi = [], [], []
    for ax in range(3):
        if int(extent[ax]) >= n:
            continue  # full axis: any index is in range by construction
        local = jnp.mod(base[:, ax] - o[ax], n)
        axes.append(ax)
        lo.append(jnp.min(jnp.where(live, local, n)))
        hi.append(jnp.max(jnp.where(live, local, -1)))
    return tuple(axes), lo, hi


def containment_bounds(x, live, cell, origin, extent, n_coarse, guard_out):
    """Append this chunk's containment bounds to `guard_out`, as device scalars.

    The device half of `engine._assert_stencil_contained`: the same rint, the
    same wrapped local index, over live rows only, so a chunk with no live row
    reports nothing to refuse.
    """
    axes, lo, hi = _containment_lohi(x, live, cell, origin, extent, n_coarse)
    _guard_entries(axes, lo, hi, origin, extent, n_coarse, guard_out)


def _guard_entries(axes, lo, hi, origin, extent, n_coarse, guard_out):
    for ax, a, b in zip(axes, lo, hi):
        guard_out.append((ax, a, b, int(origin[ax]), int(extent[ax]), int(n_coarse)))


def check_containment(guard_out):
    """Resolve deferred bounds and refuse. Call before USING the painted block."""
    try:
        for ax, lo, hi, o, e, n in guard_out:
            lo_i, hi_i = int(lo), int(hi)
            if hi_i < 0:
                continue  # no live rows
            if lo_i < 1 or hi_i > e - 2:
                raise ValueError(
                    f"sub-block paint containment violated on axis {ax}: a chunk "
                    f"row's TSC base falls outside [origin+1, origin+extent-2] "
                    f"(origin {o}, extent {e}, n {n}; live rows reach [{lo_i}, "
                    f"{hi_i}]). The cuboid derivation is wrong; refusing rather "
                    "than wrapping silently.")
    finally:
        guard_out.clear()


def _chunk_kernel(*, cap, lift, p3, per, nb, arena_base, extent, n_coarse, box,
                  frac_bits, t9, dead_rows="spread"):
    """The jitted decode + containment + paint for one set of static parameters."""
    key = (cap, lift, p3, per, nb, arena_base, tuple(int(e) for e in extent),
           n_coarse, float(box), frac_bits, float(t9.quantum), int(t9.n_buckets_side),
           dead_rows)
    fn = _KERNELS.get(key)
    if fn is not None:
        return fn
    import jax
    import jax.numpy as jnp

    from ..painting import paint_tsc_int_subblock
    from .decode import decode_core

    ext = tuple(int(e) for e in extent)
    cell = float(box) / float(n_coarse)

    def body(starts, occ, live_counts, arena_slots, row_offsets, bricks, off,
             arena_bucket, n_rows, origin):
        _TRACES[0] += 1  # runs at trace time only: the one-compile receipt
        _slots, x, _bor, _bi, live = decode_core(
            starts, occ, live_counts, arena_slots, row_offsets, bricks, off,
            arena_bucket, arena_base, n_rows, cap=cap, lift=lift, p3=p3, per=per,
            bricks_per_side=nb, t9=t9, fdtype=jnp.float64)
        _axes, lo, hi = _containment_lohi(x, live, cell, origin, ext, n_coarse)
        sub = paint_tsc_int_subblock(x, origin, ext, n_coarse, box, frac_bits,
                                     live=live, dead_rows=dead_rows)
        return sub, lo, hi

    fn = jax.jit(body)
    _KERNELS[key] = fn
    return fn


def paint_chunk(st, bricks, chunk_index, chunk_len, cfg, pad, guard_out, jit=False,
                shapes=None, dead_rows="spread"):
    """(sub, origin, extent) for one chunk, or None when it holds no rows.

    `sub` is the int32 sub-block, still on the device. Its containment bounds
    are appended to `guard_out`; `check_containment` must run before `sub` is
    used. `jit=True` needs `shapes` from `step_shapes`. `dead_rows` is
    `paint_tsc_int_subblock`'s switch for where padded rows scatter.
    """
    n = int(cfg.n_coarse)
    nb = int(st.bricks_per_side)
    origin, extent = chunk_origin_extent(chunk_index, chunk_len, nb, n)
    if jit:
        return _paint_chunk_jit(st, bricks, origin, extent, cfg, shapes, guard_out,
                                dead_rows)

    from ..painting import paint_tsc_int_subblock
    from .decode import decode_rows

    plan, off, arena_bucket, arena_base = slab_window(st, bricks)
    if plan["n_rows"] == 0:
        return None
    if plan["n_rows"] > int(pad):
        raise RuntimeError(f"chunk {chunk_index}: {plan['n_rows']} rows > pad {pad}")
    dec = decode_rows(plan, off, None, None, arena_bucket, arena_base, st.t9, nb,
                      int(pad), velocities=False)
    x, live = dec["x"], dec["live"]
    containment_bounds(x, live, cfg.box_size / float(n), origin, extent, n,
                       guard_out)
    sub = paint_tsc_int_subblock(
        x, tuple(int(o) for o in origin), tuple(int(e) for e in extent), n,
        cfg.box_size, cfg.frac_bits, live=live, dead_rows=dead_rows,
    )
    return sub, origin, extent


def _paint_chunk_jit(st, bricks, origin, extent, cfg, shapes, guard_out,
                     dead_rows="spread"):
    import jax.numpy as jnp

    from ..eject_jax import require_x64

    require_x64()
    if shapes is None:
        raise ValueError("jit=True needs the step's fixed shapes (step_shapes)")
    w = slab_window_fixed(st, bricks, shapes)
    if w["n_rows"] == 0:
        return None
    nb = int(st.bricks_per_side)
    cap = int(shapes["pad"])
    fn = _chunk_kernel(
        cap=cap, lift=cap + 1, p3=int(st.buckets_per_brick),
        per=int(st.t9.n_buckets_side // nb), nb=nb, arena_base=int(shapes["live_w"]),
        extent=extent, n_coarse=int(cfg.n_coarse), box=float(cfg.box_size),
        frac_bits=int(cfg.frac_bits), t9=st.t9, dead_rows=dead_rows)
    sub, lo, hi = fn(
        jnp.asarray(w["starts"]), jnp.asarray(w["occ"]), jnp.asarray(w["live_counts"]),
        jnp.asarray(w["arena_slots"]), jnp.asarray(w["row_offsets"]),
        jnp.asarray(w["bricks"]), jnp.asarray(w["off"]), jnp.asarray(w["arena_bucket"]),
        jnp.asarray(w["n_rows"], dtype=jnp.int64),
        jnp.asarray(origin, dtype=jnp.int32))
    axes = tuple(ax for ax in range(3) if int(extent[ax]) < int(cfg.n_coarse))
    _guard_entries(axes, lo, hi, origin, extent, cfg.n_coarse, guard_out)
    return sub, origin, extent


class HostInt64Accumulator:
    """The coarse mesh as the engine keeps it today: int64, on the host.

    `add` reads the device block back (the sync the containment check rides)
    and adds it through per-axis wrapped indices, exactly as
    `engine.coarse_delta_streamed` does.
    """

    def __init__(self, n_coarse):
        self.n = int(n_coarse)
        self.mesh = np.zeros((self.n,) * 3, dtype=np.int64)
        self.chunks = 0

    def add(self, sub, origin, extent):
        s = np.asarray(sub, dtype=np.int64)
        ax = [(np.arange(int(extent[a]), dtype=np.int64) + int(origin[a])) % self.n
              for a in range(3)]
        self.mesh[np.ix_(*ax)] += s
        self.chunks += 1

    def result(self):
        return self.mesh


def coarse_delta_device(st, cfg, stats=None, pad_shape=0, chunk_bricks=None,
                        accumulator=None, census=False, jit=True, shape_floor=None,
                        dead_rows="spread"):
    """delta on the coarse mesh, painted chunk by chunk on the device.

    Bitwise `engine.coarse_delta_streamed` at any `chunk_bricks` that tiles the
    brick grid into cuboids; the default is `default_chunk_bricks`. `pad_shape`
    carries the chunk row shape across steps as the host path does.
    `accumulator` defaults to `HostInt64Accumulator`. `jit` (default True)
    compiles each chunk as one program at fixed per-step shapes, and
    `shape_floor` carries those shapes from a previous step; `jit=False` runs
    the eager path.

    `stats`, if a dict, receives `coarse_pad`, `coarse_pad_true`,
    `coarse_peak_int`, `coarse_device_chunks` (the receipt that this path
    painted the mesh), `coarse_chunk_bricks`, `coarse_device_jit`, and with jit
    `coarse_jit_shapes` and `coarse_jit_traces` (compilations during this call),
    plus the census fields when `census`.
    """
    from ..engine import _delta_from_accumulated
    from ..forces import capacity_shape

    if cfg.paint_long != "int":
        raise ValueError(
            "the device coarse paint requires the integer accumulator: an f64 "
            "accumulation is order-dependent, so chunking it changes the result")
    nb = int(st.bricks_per_side)
    if nb != cfg.n_fine // cfg.n_brick:
        raise ValueError(f"state has {nb} bricks per side, config implies "
                         f"{cfg.n_fine // cfg.n_brick}")
    n_b = int(st.n_bricks)
    L = default_chunk_bricks(nb) if chunk_bricks is None else int(chunk_bricks)
    if L < 1 or n_b % L:
        raise ValueError(
            f"chunk_bricks {L} does not divide the {n_b} bricks, so the chunks "
            "cannot all be cuboids of one shape")

    rows = chunk_rows(st, L)
    pad_true = int(rows.max()) if len(rows) else 0
    pad = (capacity_shape(pad_true, rungs=cfg.cap_rungs, floor_shape=pad_shape)
           if cfg.pad_ladder else pad_true)
    shapes = step_shapes(st, L, pad, floor=shape_floor) if jit else None
    traces0 = _TRACES[0]
    acc = HostInt64Accumulator(cfg.n_coarse) if accumulator is None else accumulator
    guard = []
    n_dev = 0
    for gi in range(n_b // L):
        if rows[gi] == 0:
            continue
        res = paint_chunk(st, np.arange(gi * L, (gi + 1) * L, dtype=np.int64), gi,
                          L, cfg, pad, guard, jit=jit, shapes=shapes,
                          dead_rows=dead_rows)
        if res is None:
            continue
        check_containment(guard)  # before the block is used
        acc.add(*res)
        n_dev += 1

    out, peak, inexact = _delta_from_accumulated(acc.result(), cfg, census=census)
    if stats is not None:
        stats["coarse_pad"] = pad
        stats["coarse_pad_true"] = pad_true
        stats["coarse_peak_int"] = peak
        stats["coarse_device_chunks"] = n_dev
        stats["coarse_chunk_bricks"] = L
        stats["coarse_device_jit"] = bool(jit)
        stats["coarse_dead_rows"] = dead_rows
        if jit:
            stats["coarse_jit_shapes"] = shapes
            stats["coarse_jit_traces"] = _TRACES[0] - traces0
        if census:
            stats["coarse_cells_inexact_f32"] = inexact
            stats["coarse_exact_decode_ok"] = inexact == 0
    return out
