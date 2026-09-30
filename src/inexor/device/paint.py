"""The coarse paint on the device: decode, containment bounds and integer TSC paint per chunk.

The host resolves per-brick indices and hands over one contiguous slice of the state
(`slab_window`); the chunk unit defaults to a quarter x-slab of bricks
(`default_chunk_bricks`), and any chunk length whose bricks form a cuboid is accepted.

Bitwise `engine.coarse_delta_streamed`: positions decode bitwise, the kernel is the same
`paint_tsc_int_subblock`, and accumulation is integer addition, so neither chunk size nor
chunk order can move a bit. `jit=True` (default) compiles decode + containment + paint into
one program per step at fixed shapes (`step_shapes`) with a traced sub-block origin.

`coarse_delta_cards` shards the mesh along x: each card paints the chunks starting in its
planes into a `CardInt64Accumulator` with ghost planes, folds ghosts into their owner, and
decodes its planes on the card. Containment bounds come back as device scalars and are
resolved (`check_containment`) before each block is added.
"""

from __future__ import annotations

import threading

import numpy as np

# Compiled chunk programs by static parameters, and a trace count (one per step means one
# executable served every chunk). The card path's small programs share the lock.
_KERNELS = {}
_TRACES = [0]
_CARD_KERNELS = {}
_KERNEL_LOCK = threading.Lock()

#: Count of `coarse_delta_cards` calls (see `migrate.CALLS`).
CALLS = 0


def default_chunk_bricks(bricks_per_side):
    """A quarter x-slab of bricks, (1, nb/4, nb), or a whole x-slab when 4 does not
    divide nb."""
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
    """Member rows (live + arena) in each chunk of `chunk_len` consecutive bricks (0 for a
    chunk of bricks this state does not own)."""
    return st.brick_member_counts().reshape(-1, int(chunk_len)).sum(axis=1)


def step_shapes(st, chunk_len, pad, floor=None):
    """Fixed per-step input shapes for the jitted chunk, so one program serves it.

    `live_w` bounds every chunk's slot span, `arena_n` its arena residents and
    `arena_rect` the residents of any one brick; each sits on
    `forces.capacity_shape`'s ladder, and `floor` (a previous step's shapes) keeps
    them monotone across steps.
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

    Consecutive bricks' live runs are one slot range `[brick_start[b0],
    brick_start[b_last + 1])`; their arena residents are appended after it at rows
    `span, span+1, ...` with `arena_base = span`. The plan is re-based so
    `device.decode.decode_rows` reads the window exactly as it reads the whole state.
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
    `arena_n`. Padding is never read for a live row. Refuses (never truncates) a
    chunk that exceeds the step's shapes.
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
        # the uint32 index view, not the plan's int64 copy; widened on the device
        occ=st._occ(b0, b0 + L).reshape(-1, p3),
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
        _TRACES[0] += 1  # trace time only
        _slots, x, _bor, _bi, live = decode_core(
            starts, occ, live_counts, arena_slots, row_offsets, bricks, off,
            arena_bucket, arena_base, n_rows, cap=cap, lift=lift, p3=p3, per=per,
            bricks_per_side=nb, t9=t9, fdtype=jnp.float64)
        _axes, lo, hi = _containment_lohi(x, live, cell, origin, ext, n_coarse)
        sub = paint_tsc_int_subblock(x, origin, ext, n_coarse, box, frac_bits,
                                     live=live, dead_rows=dead_rows)
        return sub, lo, hi

    with _KERNEL_LOCK:
        return _KERNELS.setdefault(key, jax.jit(body))


def paint_chunk(st, bricks, chunk_index, chunk_len, cfg, pad, guard_out, jit=False,
                shapes=None, dead_rows="spread", device=None):
    """(sub, origin, extent) for one chunk, or None when it holds no rows.

    `sub` is the int32 sub-block, still on the device. Its containment bounds
    are appended to `guard_out`; `check_containment` must run before `sub` is
    used. `jit=True` needs `shapes` from `step_shapes`. `dead_rows` is
    `paint_tsc_int_subblock`'s switch for where padded rows scatter. `device`
    (jit only) places the chunk's inputs, and so its program, on that device.
    """
    n = int(cfg.n_coarse)
    nb = int(st.bricks_per_side)
    origin, extent = chunk_origin_extent(chunk_index, chunk_len, nb, n)
    if jit:
        return _paint_chunk_jit(st, bricks, origin, extent, cfg, shapes, guard_out,
                                dead_rows, device)
    if device is not None:
        raise ValueError("device= places the jitted chunk; the eager path runs on "
                         "jax's default device")

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


def _on(a, device, dtype=None):
    """`a` as a jax array on `device`, or on jax's default device when None."""
    import jax
    import jax.numpy as jnp

    if device is None:
        return jnp.asarray(a, dtype=dtype)
    return jax.device_put(np.asarray(a, dtype=dtype), device)


def _paint_chunk_jit(st, bricks, origin, extent, cfg, shapes, guard_out,
                     dead_rows="spread", device=None):
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
    d = device
    sub, lo, hi = fn(
        _on(w["starts"], d), _on(w["occ"], d), _on(w["live_counts"], d),
        _on(w["arena_slots"], d), _on(w["row_offsets"], d),
        _on(w["bricks"], d), _on(w["off"], d), _on(w["arena_bucket"], d),
        _on(w["n_rows"], d, np.int64),
        _on(origin, d, np.int32))
    axes = tuple(ax for ax in range(3) if int(extent[ax]) < int(cfg.n_coarse))
    _guard_entries(axes, lo, hi, origin, extent, cfg.n_coarse, guard_out)
    return sub, origin, extent


class HostInt64Accumulator:
    """The coarse mesh as an int64 host array. `add` reads the device block back and
    adds it through per-axis wrapped indices, as `engine.coarse_delta_streamed` does."""

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
    `coarse_peak_int`, `coarse_device_chunks`, `coarse_chunk_bricks`,
    `coarse_device_jit`, `coarse_dead_rows`, and with jit
    `coarse_jit_shapes` and `coarse_jit_traces` (compilations during this call),
    plus the census fields when `census`.
    """
    from ..engine import _delta_from_accumulated
    from ..forces import capacity_shape

    n_b, L = _chunking(st, cfg, chunk_bricks)
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


def _chunking(st, cfg, chunk_bricks):
    """(n_bricks, chunk length) after the refusals both device paints share."""
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
    return n_b, L


# --- the coarse mesh on the cards ---

#: Ghost planes below/above a card's owned x-planes: a chunk block spans one plane before
#: its first cell to two after its last (`extent = span + 3`).
ACC_GHOST_LO = 1
ACC_GHOST_HI = 2


def _cached(key, build):
    with _KERNEL_LOCK:
        fn = _CARD_KERNELS.get(key)
        if fn is None:
            fn = _CARD_KERNELS[key] = build()
    return fn


def _zeros_on(shape, dtype, device):
    """A zero array built on `device` by a program, never staged from the host or the
    default device."""
    import jax
    import jax.numpy as jnp

    shape = tuple(int(s) for s in shape)

    def build():
        return jax.jit(lambda z: jnp.broadcast_to(z, shape))

    fn = _cached(("zeros", shape, np.dtype(dtype).str), build)
    return fn(_on(np.zeros((), dtype=dtype), device))


def card_x_ranges(n_coarse, n_cards, x_unit):
    """The [lo, hi) x-plane ranges each card owns: contiguous, tiling [0, n), with
    boundaries on `x_unit` -- a chunk's x thickness in coarse cells, so every
    chunk's cells start and end on one card."""
    from ..ooc_fft import partition_units

    return partition_units(int(n_coarse), int(n_cards), int(x_unit))


class CardInt64Accumulator:
    """One card's share of the coarse mesh: int64, resident on that card.

    Holds global x-planes `lo - ACC_GHOST_LO .. hi + ACC_GHOST_HI - 1` (mod n);
    local x index 0 is global plane `lo - ACC_GHOST_LO`. `add` scatter-adds a
    block in one compiled program that donates the mesh, so a card never holds
    two. After `fold_ghosts`, the owned planes are local `ACC_GHOST_LO ..
    ACC_GHOST_LO + hi - lo - 1`.
    """

    def __init__(self, n_coarse, lo, hi, device=None):
        self.n, self.lo, self.hi = int(n_coarse), int(lo), int(hi)
        if not 0 <= self.lo < self.hi <= self.n:
            raise ValueError(f"card range [{lo}, {hi}) is not inside [0, {n_coarse})")
        self.nx = self.hi - self.lo + ACC_GHOST_LO + ACC_GHOST_HI
        self.device = device
        self.mesh = _zeros_on((self.nx, self.n, self.n), np.int64, device)
        self.chunks = 0
        self.folded = False

    def local_x(self, origin_x, extent_x):
        """The block's first x index in this accumulator, or a refusal."""
        lx = (int(origin_x) - (self.lo - ACC_GHOST_LO)) % self.n
        if lx + int(extent_x) > self.nx:
            raise ValueError(
                f"block x planes from {int(origin_x)} (extent {int(extent_x)}) are not "
                f"inside this card's accumulator, planes [{self.lo - ACC_GHOST_LO}, "
                f"{self.hi + ACC_GHOST_HI}) mod {self.n}; the scatter would drop or "
                "misplace them silently")
        return lx

    def add(self, sub, origin, extent):
        if self.folded:
            raise RuntimeError("add after fold_ghosts: the ghost planes were already moved")
        ext = tuple(int(e) for e in extent)
        lx = self.local_x(origin[0], ext[0])
        fn = _card_add_kernel(self.n, ext)
        self.mesh = fn(self.mesh, _on(lx, self.device, np.int64),
                       _on(origin, self.device, np.int64), sub)
        self.chunks += 1


def _card_add_kernel(n, extent):
    import jax
    import jax.numpy as jnp

    def build():
        def body(mesh, lx, origin, sub):
            r = [jnp.arange(e, dtype=jnp.int64) for e in extent]
            xi = lx + r[0]
            yi = (origin[1] + r[1]) % n
            zi = (origin[2] + r[2]) % n
            # indices are unique per axis, so this is the host's `mesh[np.ix_] += sub`
            return mesh.at[xi[:, None, None], yi[None, :, None],
                           zi[None, None, :]].add(sub.astype(jnp.int64))
        return jax.jit(body, donate_argnums=0)

    return _cached(("add", int(n), tuple(extent)), build)


def fold_ghosts(accs, stats=None):
    """Add every card's ghost planes into the card that owns each plane.

    Exact (integer addition). Every ghost plane is read before any card is written,
    so one card owning its own wrapped ghosts folds the same as many. `stats`
    receives `coarse_ghost_planes_nonzero`.
    """
    import jax
    import jax.numpy as jnp

    n = accs[0].n

    def owner(g):
        for a in accs:
            if a.lo <= g < a.hi:
                return a
        raise ValueError(f"no card owns x-plane {g}: the ranges do not tile the mesh")

    take = _cached(("take",), lambda: jax.jit(lambda m, i: m[i]))
    put = _cached(("plane_add",),
                  lambda: jax.jit(lambda m, i, p: m.at[i].add(p), donate_argnums=0))
    ghosts = []
    for a in accs:
        if a.folded:
            raise RuntimeError("fold_ghosts called twice would add the ghosts twice")
        locs = list(range(ACC_GHOST_LO)) + list(range(a.nx - ACC_GHOST_HI, a.nx))
        for li in locs:
            g = (a.lo - ACC_GHOST_LO + li) % n
            ghosts.append((g, take(a.mesh, _on(li, a.device, np.int64))))
    nonzero = 0
    for g, plane in ghosts:
        t = owner(g)
        if stats is not None:
            nonzero += bool(jnp.any(plane != 0))
        p = plane if t.device is None else jax.device_put(plane, t.device)
        t.mesh = put(t.mesh, _on(g - t.lo + ACC_GHOST_LO, t.device, np.int64), p)
    for a in accs:
        a.folded = True
    if stats is not None:
        stats["coarse_ghost_planes_nonzero"] = nonzero


def _delta_on_cards(accs, cfg, census=False):
    """(deltas, peak, inexact): each card's owned planes as the coarse density,
    on that card, in `cfg`'s coarse dtype.

    The arithmetic of `engine._delta_from_accumulated`, plane by plane. The mean is a
    plane-shaped runtime array because CPU XLA turns a scalar divisor into a reciprocal
    multiply, one ulp off numpy.
    """
    import jax
    import jax.numpy as jnp

    for a in accs:
        if not a.folded:
            raise RuntimeError("decode before fold_ghosts would drop the ghost planes' mass")
    n = int(cfg.n_coarse)
    fdt = np.dtype(cfg.np_coarse_dtype)
    peak, inexact = 0, 0
    for a in accs:
        w = a.hi - a.lo

        def build(w=w):
            def body(mesh):
                own = mesh[ACC_GHOST_LO:ACC_GHOST_LO + w]
                pk = jnp.abs(own).max()
                ix = jnp.count_nonzero(own.astype(jnp.float32).astype(jnp.int64) != own)
                return pk, ix
            return jax.jit(body)

        pk, ix = _cached(("census", n, w), build)(a.mesh)
        peak = max(peak, int(pk))
        if census:
            inexact += int(ix)
    if peak >= 2**31:
        raise ValueError(
            f"the accumulated coarse paint reached {peak}, past int32. "
            "Lower frac_bits -- int32 overflow corrupts the paint; it is not imprecision."
        )
    mean = float(cfg.n_total) / float(n) ** 3
    scale = 2.0 ** -cfg.frac_bits

    def build_decode():
        def body(d, mesh, i, div):
            s = mesh[i + ACC_GHOST_LO]
            return d.at[i].set((s.astype(jnp.float64) * scale / div - 1.0).astype(fdt))
        return jax.jit(body, donate_argnums=0)

    decode = _cached(("decode", fdt.str, scale), build_decode)
    deltas = []
    for a in accs:
        w = a.hi - a.lo
        div = _on(np.full((n, n), mean), a.device, np.float64)
        d = _zeros_on((w, n, n), fdt, a.device)
        for i in range(w):
            d = decode(d, a.mesh, _on(i, a.device, np.int64), div)
        deltas.append(d)
    return deltas, peak, inexact


def coarse_delta_cards(st, cfg, devices=None, stats=None, pad_shape=0, chunk_bricks=None,
                       census=False, shape_floor=None, dead_rows="spread", fold=True):
    """The coarse density painted, accumulated and decoded on the cards.

    Returns one dict per card, `lo`, `hi`, `device` and `delta` (the density on
    x-planes [lo, hi), shape (hi - lo, n, n), on that card). BITWISE
    `engine.coarse_delta_streamed` over the whole mesh (`gather_card_delta`), at
    any card count and any `chunk_bricks` that tiles the brick grid into cuboids.

    `devices` is a sequence of jax devices, one per card (None: one card on jax's
    default device). Each card paints the chunks whose bricks start in its x range,
    jitted, on its own thread. `fold=False` drops the ghosts' mass (a test instrument).

    `stats`, if a dict, receives `coarse_device_chunks` in total and
    `coarse_card_chunks` per card, `coarse_cards`, `coarse_card_ranges`,
    `coarse_ghost_planes_nonzero`, and the fields `coarse_delta_device` reports.
    """
    from concurrent.futures import ThreadPoolExecutor

    from ..eject_jax import require_x64
    from ..engine import _chunk_cuboid
    from ..forces import capacity_shape

    global CALLS
    require_x64()  # before an int64 accumulator exists: without x64 it would be int32
    CALLS += 1
    n_b, L = _chunking(st, cfg, chunk_bricks)
    devs = [None] if devices is None else list(devices)
    if not devs:
        raise ValueError("devices= was an empty sequence; pass None for one card")
    W = len(devs)
    n = int(cfg.n_coarse)
    nb = int(st.bricks_per_side)
    chunk_origin_extent(0, L, nb, n)  # refuses a chunk length with no cuboid
    x_unit = int(_chunk_cuboid(0, L, nb, n)[1][0])
    if W > 1 and x_unit + 3 >= n:
        raise ValueError(
            f"chunk_bricks {L} paints the full x axis ({x_unit} + 3 >= {n} cells), so "
            "no chunk belongs to one card; use a chunk no thicker than n - 3 in x")
    ranges = card_x_ranges(n, W, x_unit)
    accs = [CardInt64Accumulator(n, lo, hi, dev) for (lo, hi), dev in zip(ranges, devs)]

    rows = chunk_rows(st, L)
    pad_true = int(rows.max()) if len(rows) else 0
    pad = (capacity_shape(pad_true, rungs=cfg.cap_rungs, floor_shape=pad_shape)
           if cfg.pad_ladder else pad_true)
    shapes = step_shapes(st, L, pad, floor=shape_floor)
    by_card = [[] for _ in range(W)]
    for gi in range(n_b // L):
        if rows[gi] == 0:
            continue
        c0 = int(_chunk_cuboid(gi, L, nb, n)[0][0])
        k = next(k for k, (lo, hi) in enumerate(ranges) if lo <= c0 < hi)
        by_card[k].append(gi)

    traces0 = _TRACES[0]

    def run(k):
        guard = []
        for gi in by_card[k]:
            res = paint_chunk(st, np.arange(gi * L, (gi + 1) * L, dtype=np.int64), gi, L,
                              cfg, pad, guard, jit=True, shapes=shapes,
                              dead_rows=dead_rows, device=devs[k])
            if res is None:
                continue
            check_containment(guard)  # before the block is used
            accs[k].add(*res)

    if W == 1:
        run(0)
    else:
        with ThreadPoolExecutor(max_workers=W) as ex:
            for f in [ex.submit(run, k) for k in range(W)]:
                f.result()

    fold_stats = {}
    if fold:
        fold_ghosts(accs, stats=fold_stats)
    else:
        for a in accs:
            a.folded = True  # test instrument: the ghosts' mass is dropped
    deltas, peak, inexact = _delta_on_cards(accs, cfg, census=census)
    if stats is not None:
        stats["coarse_pad"] = pad
        stats["coarse_pad_true"] = pad_true
        stats["coarse_peak_int"] = peak
        stats["coarse_card_chunks"] = [a.chunks for a in accs]
        stats["coarse_device_chunks"] = sum(a.chunks for a in accs)
        stats["coarse_cards"] = W
        stats["coarse_card_ranges"] = ranges
        stats["coarse_chunk_bricks"] = L
        stats["coarse_device_jit"] = True
        stats["coarse_dead_rows"] = dead_rows
        stats["coarse_jit_shapes"] = shapes
        stats["coarse_jit_traces"] = _TRACES[0] - traces0
        stats.update(fold_stats)
        if census:
            stats["coarse_cells_inexact_f32"] = inexact
            stats["coarse_exact_decode_ok"] = inexact == 0
    return [dict(lo=lo, hi=hi, device=dev, delta=d)
            for (lo, hi), dev, d in zip(ranges, devs, deltas)]


def gather_card_delta(shards):
    """The whole density on the host, from `coarse_delta_cards`' shards."""
    shards = sorted(shards, key=lambda s: s["lo"])
    return np.concatenate([np.asarray(s["delta"]) for s in shards], axis=0)
