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
choice and does not fit a card: measured on a GB200 at 266 B per padded row
(Vista 993139), it is 90 GB at 4096^3 and puts a card at 1.15x its memory,
where a quarter-slab is 22.5 GB and 0.81x. Any chunk length whose bricks form a
cuboid is accepted (`engine._chunk_cuboid`).

BITWISE against the host paint, and it can be. Positions decode bitwise
(`device.decode`), the paint kernel is the same `paint_tsc_int_subblock` with
the same padding mask, and the accumulation is integer addition -- so neither
the chunk size nor the order chunks arrive in can move a bit.

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


def default_chunk_bricks(bricks_per_side):
    """A quarter of an x-slab of bricks, or a whole x-slab where a quarter does
    not tile into cuboids (bricks per side not divisible by 4).

    A quarter-slab is (1, nb/4, nb) bricks, which `engine._chunk_cuboid`
    accepts whenever 4 divides nb -- at every production config nb is 32 or
    more and a power of two.
    """
    nb = int(bricks_per_side)
    return nb * nb // 4 if nb % 4 == 0 else nb * nb


def chunk_rows(st, chunk_len):
    """Member rows (live + arena) in each chunk of `chunk_len` consecutive bricks.

    One vectorized pass over the index and the arena, not a Python loop over
    bricks: at 4096^3 there are 16.8M bricks.
    """
    p3 = int(st.buckets_per_brick)
    n_b = int(st.n_bricks)
    per_brick = st.occupancy.reshape(n_b, p3).sum(axis=1, dtype=np.int64)
    if st.n_arena:
        res = st.arena_bucket[st.arena_bucket >= 0] // p3
        per_brick = per_brick + np.bincount(res, minlength=n_b).astype(np.int64)
    return per_brick.reshape(-1, int(chunk_len)).sum(axis=1)


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

    bricks = np.asarray(bricks, dtype=np.int64)
    if len(bricks) == 0 or not np.array_equal(
            bricks, np.arange(bricks[0], bricks[0] + len(bricks), dtype=np.int64)):
        raise ValueError(
            "a device paint chunk must be a run of CONSECUTIVE brick ids: only "
            "then are its live rows one contiguous slot range")
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


def containment_bounds(x, live, cell, origin, extent, n_coarse, guard_out):
    """Append this chunk's containment bounds to `guard_out`, as device scalars.

    The device half of `engine._assert_stencil_contained`: the same rint, the
    same wrapped local index, over live rows only. Dead rows are pushed to
    sentinels that cannot win their own reduction, so a chunk with no live row
    reports nothing to refuse.
    """
    import jax.numpy as jnp

    n = int(n_coarse)
    base = jnp.rint(x / float(cell)).astype(jnp.int64)
    for ax in range(3):
        e = int(extent[ax])
        if e >= n:
            continue  # full axis: any index is in range by construction
        local = jnp.mod(base[:, ax] - int(origin[ax]), n)
        lo = jnp.min(jnp.where(live, local, n))
        hi = jnp.max(jnp.where(live, local, -1))
        guard_out.append((ax, lo, hi, int(origin[ax]), e, n))


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


def paint_chunk(st, bricks, chunk_index, chunk_len, cfg, pad, guard_out):
    """(sub, origin, extent) for one chunk, or None when it holds no rows.

    `sub` is the int32 sub-block, still on the device. Its containment bounds
    are appended to `guard_out`; `check_containment` must run before `sub` is
    used.
    """
    from ..painting import paint_tsc_int_subblock
    from .decode import decode_rows

    n = int(cfg.n_coarse)
    nb = int(st.bricks_per_side)
    origin, extent = chunk_origin_extent(chunk_index, chunk_len, nb, n)
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
        cfg.box_size, cfg.frac_bits, live=live,
    )
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
                        accumulator=None, census=False):
    """delta on the coarse mesh, painted chunk by chunk on the device.

    Bitwise `engine.coarse_delta_streamed` at any `chunk_bricks` that tiles the
    brick grid into cuboids; the default is `default_chunk_bricks`. `pad_shape`
    carries the chunk row shape across steps as the host path does.
    `accumulator` defaults to `HostInt64Accumulator`.

    `stats`, if a dict, receives `coarse_pad`, `coarse_pad_true`,
    `coarse_peak_int`, `coarse_device_chunks` (the receipt that this path
    painted the mesh) and `coarse_chunk_bricks`, plus the census fields when
    `census`.
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
    acc = HostInt64Accumulator(cfg.n_coarse) if accumulator is None else accumulator
    guard = []
    n_dev = 0
    for gi in range(n_b // L):
        if rows[gi] == 0:
            continue
        res = paint_chunk(st, np.arange(gi * L, (gi + 1) * L, dtype=np.int64), gi,
                          L, cfg, pad, guard)
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
        if census:
            stats["coarse_cells_inexact_f32"] = inexact
            stats["coarse_exact_decode_ok"] = inexact == 0
    return out
