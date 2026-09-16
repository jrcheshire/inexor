"""FROZEN copy of `inexor.device.migrate` as it stood at 6d882b1, for the M3 A/B only.

Vendored so the host-stall removal (the slab window uploaded without a host copy;
the census counts read once at eject) runs against its before-state in ONE job on
ONE node, as `v2_d3_insert_legacy.py` did for M2. Relative imports rewritten to
absolute; nothing else changed. The kernels (`eject_jax`, `insert_jax`) are the
live ones in both arms, so the A/B isolates the pass around them.

An ORACLE, not a path anything runs, and it does not move. Delete it once the
GB200 A/B is read out and recorded.

The original module docstring follows.

D3b R1: the migrate on the device, slab by slab, bitwise the serial numpy pass.

WHAT MOVES. `state.drift_and_migrate` ejects and inserts each x-slab with numpy
(or with the compiled kernels fed and emptied by per-row host work). Here each
slab's raw bytes -- its slot range, its arena residents, its occupancy slice --
go to the device once. A device program enumerates the slab's rows, the existing
compiled kernels (`eject_jax`, `insert_jax`, gated bitwise against numpy on a
GB200) run on device arrays, keepers and emigrants stay on the device until
their insert, and the written rows are scattered into the slab's original bytes,
which come back to the host as one slice. The host does O(bricks + arena) index
work per slab and replays the arena at the end.

WHY IT CAN BE BITWISE.
- An insert writes only inside its own slab's slot range (plus arena claims), and
  `brick_start` changes only in repack, so every slab's range is fixed for the
  pass and its original bytes are still valid when its insert writes them.
- Every order-dependent arena mutation (releases at ejects, claims at inserts) is
  deferred to `state._replay_arena_pass`, as the pooled migrate does. A resident
  of a slab not yet ejected is never overwritten by a claim, so reading residents
  from the pre-pass host state is exact.
- The kernels are called with exactly the padded inputs their public wrappers
  (`eject_rows`, `insert_rows`) build, including the pad fill values.

ROW ORDER is the reference's: per brick, the live run then arena residents in
ascending slot; insert input is the slab's keepers, then the emigrants of its
source slabs in `sorted({(d + o) % nb})` order.

WHAT IS HELD. A staged slab keeps its whole eject output only until its own
insert, then its emigrants alone; row buffers and the insert's compacted inputs
are released as soon as their kernel returns (retention probe
`scripts/v2_d3_retention.py`).

THE BUDGET. `device_budget_bytes` refuses a slab whose estimated footprint --
every array the pass already holds, plus the kernel's measured bytes per padded
row (`EJECT_B_PER_PADDED_ROW`, `INSERT_B_PER_PADDED_ROW`) -- exceeds it. The
largest estimate is on the receipt either way, so a device run can compare it
with the measured peak.

SHAPES. Row buffers use each kernel's `_padded`; the slab window, arena arrays and
staged emigrants use `forces.capacity_shape`, so slabs of nearby size share every
program.

SEVERAL CARDS (R3, `devices=`). Because every order-dependent arena change is
replayed at the end and a slab's work touches only its own rows, slabs may run on
any card in any order. Card k owns a contiguous run of slabs. It first ejects its
`r` lowest and `r` highest slabs and hands emigrant-only copies to the neighbour
that inserts from them; then each card sweeps its own slabs as the one-card pass
does, one thread per card, and the parent replays the arena once. A card holding
fewer than `2r + 1` slabs cannot separate its boundaries, and the pass falls back
to one card (on the receipt).
"""

from __future__ import annotations

import threading
import time

import numpy as np

INT16_MAX = 32767

#: Device bytes per PADDED row of one kernel call, its uploaded inputs included:
#: the peaks Vista 995435 measured at a 4096^3 slab (record sec. 29 -- eject 114.4,
#: insert 78.4-83.0 over the bracket, the largest taken). Estimate only.
EJECT_B_PER_PADDED_ROW = 114.4
INSERT_B_PER_PADDED_ROW = 83.0
#: The estimate read against the MEASURED device peak in one process, cgh64 on a
#: GB200 (Vista 995813, record sec. 33): estimate / measured = 0.762 and 0.760 on
#: two passes. The kernel coefficients above were measured alone; what the pass
#: holds around them under-reads by this factor. Applied to every estimate.
ESTIMATE_OVER_MEASURED = 0.76

_PROGRAMS: dict = {}
_PROGRAM_LOCK = threading.Lock()

#: RECEIPT: passes run through this module (see `eject_jax.CALLS`).
CALLS = 0


class _Clock:
    """Synced phase timer for `timings=`; a no-op when `timings` is None.

    `mark(key, *arrays)` blocks until the arrays are ready, then charges the wall
    since the previous mark to `key`. Syncing moves the wall, so a timed pass is
    a breakdown, not the pass's cost.
    """

    def __init__(self, timings):
        self.t = timings
        self.last = time.perf_counter()

    def mark(self, key, *arrays):
        if self.t is None:
            return
        if arrays:
            import jax

            jax.block_until_ready(arrays)
        now = time.perf_counter()
        self.t[key] = self.t.get(key, 0.0) + now - self.last
        self.last = now


class _Budget:
    """Per-slab device footprint estimate against `device_budget_bytes` (None: no limit)."""

    def __init__(self, limit, staged, fixed):
        self.limit = None if limit is None else int(limit)
        self.staged = staged
        self.fixed = fixed
        self.peak = 0

    def held(self):
        total = sum(int(a.nbytes) for a in self.fixed)
        for e in self.staged.values():
            for k in ("dest", "off", "w", "ids", "src"):
                a = e.get(k)
                if a is not None:
                    total += int(a.nbytes)
            for a in e.get("win", ()):
                if a is not None:
                    total += int(a.nbytes)
        return total

    def check(self, what, extra_bytes, coef, padded_rows):
        held = self.held()
        est = int((held + int(extra_bytes) + int(coef * padded_rows)) / ESTIMATE_OVER_MEASURED)
        self.peak = max(self.peak, est)
        if self.limit is not None and est > self.limit:
            raise ValueError(
                f"{what} needs an estimated {est / 1e9:.2f} GB on the device against a "
                f"budget of {self.limit / 1e9:.2f} GB: {held / 1e9:.2f} GB already held, "
                f"{extra_bytes / 1e9:.2f} GB of window, and {coef:g} B x {padded_rows:,} "
                "padded rows of kernel. The envelope is what a card has beside the tile "
                "loop's resident terms (record sec. 29). Use fewer rows per slab or a "
                "smaller drift (fewer staged slabs), or raise the budget deliberately."
            )


def _ladder(n):
    from inexor.forces import capacity_shape

    return int(capacity_shape(max(1, int(n))))


def _put(a, dev, dtype=None):
    """`a` as a device array on `dev`; None is jax's default device via
    `jnp.asarray`, exactly the one-card pass's call."""
    import jax
    import jax.numpy as jnp

    if dev is None:
        return jnp.asarray(a, dtype=dtype)
    if isinstance(a, jax.Array) and dtype is None:
        return jax.device_put(a, dev)
    return jax.device_put(np.asarray(a, dtype=dtype), dev)


def _zeros(shape, dtype, dev):
    """Zeros built on `dev` (None: `jnp.zeros`, the one-card pass's call)."""
    import jax.numpy as jnp

    if dev is None:
        return jnp.zeros(shape, dtype)
    from inexor.device.paint import _zeros_on

    shape = (int(shape),) if np.isscalar(shape) else tuple(shape)
    return _zeros_on(shape, dtype, dev)


def _arange(n, dev):
    """`jnp.arange(n, dtype=int64)` on `dev`, built by a program placed there."""
    import jax
    import jax.numpy as jnp

    if dev is None:
        return jnp.arange(n, dtype=jnp.int64)
    n = int(n)
    fn = _program(("arange", n), lambda: jax.jit(lambda z: jnp.arange(n, dtype=jnp.int64) + z))
    return fn(jax.device_put(np.int64(0), dev))


def pass_arena_index(st):
    """(slots, bricks): arena residents ordered by brick, ascending slot within.

    The grouping `SlotState._build_arena_index` makes, vectorized over the whole
    arena once. Valid for a pass whose releases and claims are deferred.
    """
    p3 = int(st.buckets_per_brick)
    ab = np.asarray(st.arena_bucket)
    live = np.flatnonzero(ab >= 0)
    bricks = ab[live] // p3
    order = np.argsort(bricks, kind="stable")
    return int(st.arena_base) + live[order], bricks[order]


def _program(key, make):
    with _PROGRAM_LOCK:
        fn = _PROGRAMS.get(key)
        if fn is None:
            fn = _PROGRAMS[key] = make()
    return fn


def _rows_program(cap, w_cap, a_cap, nb, p3, per, has_ids):
    """One slab's eject inputs at `cap` rows, built from its window on the device."""

    def make():
        import jax
        import jax.numpy as jnp

        nb2 = nb * nb
        lift = cap + 1  # above every prefix sum and rank (both <= cap)

        @jax.jit
        def rows(occ, live, starts_rel, row_offsets, ar_offsets, ar_bucket, off_win, w_win,
                 ids_win, scales_s, n_rows, span, lo_b):
            r = jnp.arange(cap, dtype=jnp.int64)
            real = r < n_rows
            bi = jnp.clip(jnp.searchsorted(row_offsets[1:], r, side="right"),
                          0, nb2 - 1).astype(jnp.int64)
            rank = r - row_offsets[bi]
            lc = live[bi]
            is_ar = rank >= lc
            # live rows: the bucket is how many of the row's own brick's prefix
            # sums are <= its rank, counted with one searchsorted over prefix sums
            # lifted brick by brick (the device decode's method, O(rows))
            occ_cum = jnp.cumsum(occ.astype(jnp.int64), axis=1)
            keys = (occ_cum + (jnp.arange(nb2, dtype=jnp.int64) * lift)[:, None]).reshape(-1)
            within = jnp.searchsorted(keys, bi * lift + rank, side="right").astype(jnp.int64)
            within = jnp.clip(within - bi * p3, 0, p3 - 1)
            bucket_live = (lo_b + bi) * p3 + within
            # arena rows: appended to the window after the slot range, in index order
            k = jnp.clip(ar_offsets[bi] + rank - lc, 0, a_cap - 1)
            slot = jnp.where(real, jnp.where(is_ar, span + k, starts_rel[bi] + rank), 0)
            bucket = jnp.where(is_ar, ar_bucket[k], bucket_live)
            # brick-major ordinal -> per-axis bucket index (`layout.bucket_ijk_from_key`)
            brick_flat, within_flat = jnp.divmod(bucket, per**3)
            bx, rem = jnp.divmod(brick_flat, nb2)
            by, bz = jnp.divmod(rem, nb)
            wx, rem = jnp.divmod(within_flat, per * per)
            wy, wz = jnp.divmod(rem, per)
            bijk = jnp.stack([bx * per + wx, by * per + wy, bz * per + wz], axis=-1)
            # pad rows carry the fills `eject_rows` pads with
            m = real[:, None]
            off = jnp.where(m, off_win[slot], jnp.zeros((), off_win.dtype))
            w = jnp.where(m, w_win[slot], jnp.zeros((), w_win.dtype))
            bijk = jnp.where(m, bijk, 0)
            scale = jnp.where(real, scales_s[bi], 1.0)[:, None]
            brick = jnp.where(real, lo_b + bi, -1)
            ids = jnp.where(real, ids_win[slot], jnp.zeros((), ids_win.dtype)) if has_ids else None
            return off, bijk, w, ids, scale, brick, real

        return rows

    return _program(("rows", cap, w_cap, a_cap, nb, p3, per, has_ids), make)


def _put_program(seg_cap, cap_i, has_ids):
    """Scatter rows [lo, hi) of one staged segment into the insert buffers at `base`."""

    def make():
        import jax
        import jax.numpy as jnp

        @jax.jit
        def put(bufs, dest, off, w, ids, src_brick, lo, hi, base):
            idx = jnp.arange(seg_cap, dtype=jnp.int64)
            pos = jnp.where((idx >= lo) & (idx < hi), base + idx - lo, cap_i)
            b_dest, b_off, b_w, b_ids, b_src = bufs
            b_dest = b_dest.at[pos].set(dest, mode="drop")
            b_off = b_off.at[pos].set(off, mode="drop")
            b_w = b_w.at[pos].set(w, mode="drop")
            b_ids = b_ids.at[pos].set(ids, mode="drop") if has_ids else None
            b_src = b_src.at[pos].set(src_brick, mode="drop")
            return b_dest, b_off, b_w, b_ids, b_src

        return put

    return _program(("put", seg_cap, cap_i, has_ids), make)


def _write_program(cap_i, w_cap, has_ids):
    """The insert's written rows scattered into the slab window's original bytes."""

    def make():
        import jax
        import jax.numpy as jnp

        @jax.jit
        def write(off_win, w_win, ids_win, pos, off, w, ids, n_write, s0):
            idx = jnp.arange(cap_i, dtype=jnp.int64)
            rel = jnp.where(idx < n_write, pos - s0, w_cap)
            off_win = off_win.at[rel].set(off, mode="drop")
            w_win = w_win.at[rel].set(w, mode="drop")
            ids_win = ids_win.at[rel].set(ids, mode="drop") if has_ids else None
            return off_win, w_win, ids_win

        return write

    return _program(("write", cap_i, w_cap, has_ids), make)


def _eject_kernel(t9, nb, cap, has_ids):
    from inexor import eject_jax

    key = (int(t9.n_buckets_side), float(t9.quantum), int(nb), int(cap), has_ids)
    with _PROGRAM_LOCK:
        fn = eject_jax._CACHE.get(key)
        if fn is None:
            fn = eject_jax._CACHE[key] = eject_jax._build(t9, int(nb), int(cap), has_ids)
    return fn


def _insert_kernel(p3, nb2, cap, has_ids):
    from inexor import insert_jax

    key = (int(p3), int(nb2), int(cap), has_ids)
    with _PROGRAM_LOCK:
        fn = insert_jax._CACHE.get(key)
        if fn is None:
            fn = insert_jax._CACHE[key] = insert_jax._build(int(p3), int(nb2), int(cap), has_ids)
    return fn


def _eject_slab(st, s, c_drift, scales, ar_slots, ar_bricks, clock, budget, dev=None):
    """Upload slab s's window to `dev`, build its rows and eject them. Returns the staged slab."""
    import jax.numpy as jnp

    from inexor import eject_jax

    nb, p3 = int(st.bricks_per_side), int(st.buckets_per_brick)
    nb2 = nb * nb
    per = int(st.t9.n_buckets_side) // nb
    has_ids = st.ids is not None
    lo_b, hi_b = st.slab_bricks(s)
    s0, s1 = int(st.brick_start[lo_b]), int(st.brick_start[hi_b])
    span = s1 - s0

    occ = np.asarray(st.occupancy)[lo_b * p3: hi_b * p3].reshape(nb2, p3)
    live = occ.sum(axis=1, dtype=np.int64)
    a0, a1 = np.searchsorted(ar_bricks, [lo_b, hi_b])
    rows_a = ar_slots[a0:a1]
    ar_counts = np.bincount(ar_bricks[a0:a1] - lo_b, minlength=nb2).astype(np.int64)
    ar_offsets = np.zeros(nb2 + 1, dtype=np.int64)
    np.cumsum(ar_counts, out=ar_offsets[1:])
    row_offsets = np.zeros(nb2 + 1, dtype=np.int64)
    np.cumsum(live + ar_counts, out=row_offsets[1:])
    n_rows = int(row_offsets[-1])
    n_ar = int(a1 - a0)

    a_cap = _ladder(n_ar)
    w_cap = _ladder(span + a_cap)
    cap = eject_jax._padded(n_rows)
    row_bytes = st.off.itemsize * 3 + st.w.itemsize * 3 + (st.ids.itemsize if has_ids else 0)
    budget.check(f"ejecting slab {s}", w_cap * row_bytes, EJECT_B_PER_PADDED_ROW, cap)

    ar_bucket = np.zeros(a_cap, dtype=np.int64)
    ar_bucket[:n_ar] = np.asarray(st.arena_bucket)[rows_a - int(st.arena_base)]

    def window(src):
        out = np.zeros((w_cap,) + src.shape[1:], dtype=src.dtype)
        out[:span] = src[s0:s1]
        out[span:span + n_ar] = src[rows_a]
        return out

    off_np, w_np = window(st.off), window(st.w)
    ids_np = window(st.ids) if has_ids else None
    clock.mark("eject: host index + window")

    off_win = _put(off_np, dev)
    w_win = _put(w_np, dev)
    ids_win = _put(ids_np, dev) if has_ids else None
    off_np = w_np = ids_np = None
    index_dev = (_put(occ, dev), _put(live, dev), _put(np.asarray(
        st.brick_start[lo_b:hi_b], dtype=np.int64) - s0, dev), _put(row_offsets, dev),
        _put(ar_offsets, dev), _put(ar_bucket, dev),
        _put(np.asarray(scales[lo_b:hi_b], dtype=np.float64), dev))
    clock.mark("eject: upload", off_win, w_win, ids_win, index_dev)

    rows = _rows_program(cap, w_cap, a_cap, nb, p3, per, has_ids)
    occ_d, live_d, starts_d, row_off_d, ar_off_d, ar_bucket_d, scales_d = index_dev
    off, bijk, w, ids, scale, brick, real = rows(
        occ_d, live_d, starts_d, row_off_d, ar_off_d, ar_bucket_d, off_win, w_win, ids_win,
        scales_d, _put(n_rows, dev, jnp.int64), _put(span, dev, jnp.int64),
        _put(lo_b, dev, jnp.int64))
    index_dev = occ_d = live_d = starts_d = row_off_d = ar_off_d = ar_bucket_d = scales_d = None
    clock.mark("eject: rows program", off, bijk, w, ids, scale, brick, real)

    fn = _eject_kernel(st.t9, nb, cap, has_ids)
    dest, off_new, w_out, ids_out, src_out, n_keep = fn(
        off, bijk, w, ids, scale, float(c_drift), brick, real)
    # the row buffers are the kernel's input only; release them before staging
    off = bijk = w = ids = scale = brick = real = None
    clock.mark("eject: kernel", dest, off_new, w_out, ids_out, src_out, n_keep)
    n_keep = int(n_keep)

    # realized x-reach over this slab's emigrants, as the serial pass reports it
    rr = 0
    if n_rows > n_keep:
        idx = _arange(cap, dev)
        em = (idx >= n_keep) & (idx < n_rows)
        disp = (dest // (p3 * nb2) - s + nb // 2) % nb - nb // 2
        rr = int(jnp.max(jnp.where(em, jnp.abs(disp), 0)))
    clock.mark("eject: reach + scalars")

    return dict(dest=dest, off=off_new, w=w_out, ids=ids_out, src=src_out, cap=cap,
                n_keep=n_keep, n_rows=n_rows, rr=rr,
                win=(off_win, w_win, ids_win), s0=s0, s1=s1, span=span, w_cap=w_cap)


def _insert_slab(st, d, reach, staged, scales_dev, clock, budget, dev=None):
    """Insert slab d on `dev` and write its slot range back. Returns insert_res."""
    import jax.numpy as jnp

    from inexor import insert_jax
    from inexor.layout import _to_index

    nb, p3 = int(st.bricks_per_side), int(st.buckets_per_brick)
    nb2 = nb * nb
    has_ids = st.ids is not None
    lo_b, hi_b = st.slab_bricks(d)
    sources = sorted({(int(d) + o) % nb for o in reach})

    consumed = {}
    segs = [(staged[d], 0, staged[d]["n_keep"], True)]
    for s in sources:
        e = staged[s]
        if e["n_rows"] > e["n_keep"]:
            idx = _arange(e["cap"], dev)
            em = (idx >= e["n_keep"]) & (idx < e["n_rows"])
            consumed[s] = int(jnp.sum(em & (e["dest"] // (p3 * nb2) == d)))
            segs.append((e, e["n_keep"], e["n_rows"], False))
    n_in = sum(hi - lo for _e, lo, hi, _k in segs)
    clock.mark("insert: census")

    cap_i = insert_jax._padded(n_in)
    budget.check(f"inserting slab {d}", 0, INSERT_B_PER_PADDED_ROW, cap_i)
    bufs = (_zeros(cap_i, jnp.int64, dev), _zeros((cap_i, 3), jnp.uint8, dev),
            _zeros((cap_i, 3), jnp.int16, dev),
            _zeros(cap_i, jnp.int32, dev) if has_ids else None, _zeros(cap_i, jnp.int64, dev))
    base = 0
    for e, lo, hi, is_keep in segs:
        src_brick = e["dest"] // p3 if is_keep else e["src"]
        put = _put_program(e["cap"], cap_i, has_ids)
        bufs = put(bufs, e["dest"], e["off"], e["w"], e["ids"], src_brick,
                   _put(lo, dev, jnp.int64), _put(hi, dev, jnp.int64),
                   _put(base, dev, jnp.int64))
        base += hi - lo
    b_dest, b_off, b_w, b_ids, b_src = bufs
    real = _arange(cap_i, dev) < n_in
    s_old = jnp.where(real, scales_dev[b_src], 1.0)
    clock.mark("insert: compact inputs", b_dest, b_off, b_w, b_ids, s_old, real)

    fn = _insert_kernel(p3, nb2, cap_i, has_ids)
    out = fn(b_dest, b_off, b_w, b_ids, s_old, real, _put(lo_b, dev, jnp.int64),
             _put(np.asarray(st.brick_start[lo_b:hi_b + 1], dtype=np.int64), dev),
             _put(np.full(nb2, 32767.0, dtype=np.float64), dev))
    clock.mark("insert: kernel", out)
    if float(out["abs_max"]) > INT16_MAX:
        raise ValueError(
            f"velocity code {float(out['abs_max']):.0f} escapes int16 under a rescale to a "
            "scale that does not cover it. Per-brick scales make this reachable where a "
            "global scale made it impossible; the caller must fix the destination scale "
            "over the rows it is about to write. D-007 forbids the clamp."
        )
    nw, ns = int(out["n_write"]), int(out["n_spill"])
    # the compacted inputs are not read past the kernel; only its outputs are
    bufs = b_dest = b_off = b_w = b_ids = b_src = s_old = real = None

    e = staged[d]
    write = _write_program(cap_i, e["w_cap"], has_ids)
    off_win, w_win, ids_win = write(*e["win"], out["pos"], out["off"], out["w"], out["ids"],
                                    _put(nw, dev, jnp.int64), _put(e["s0"], dev, jnp.int64))
    clock.mark("insert: scalars + write program", off_win, w_win, ids_win)
    s0, s1, span = e["s0"], e["s1"], e["span"]
    st.off[s0:s1] = np.asarray(off_win)[:span]
    st.w[s0:s1] = np.asarray(w_win)[:span]
    if has_ids:
        st.ids[s0:s1] = np.asarray(ids_win)[:span]
    clock.mark("insert: slot range to host")
    st.occupancy[lo_b * p3: hi_b * p3] = _to_index(np.asarray(out["occupancy"]),
                                                    st.index_dtype, "migrated")
    st.vel_scale[lo_b:hi_b] = np.asarray(out["scales"])
    clock.mark("insert: occupancy + scales to host")

    spills = []
    if ns:
        # rows [nw, nw + ns) gathered onto the ladder by the compaction program: a
        # device slice sized by the spill count keys a new op on every slab
        cap_s = _ladder(ns)
        sbufs = (_zeros(cap_s, jnp.int64, dev), _zeros((cap_s, 3), jnp.uint8, dev),
                 _zeros((cap_s, 3), jnp.int16, dev),
                 _zeros(cap_s, jnp.int32, dev) if has_ids else None,
                 _zeros(cap_s, jnp.int64, dev))
        take = _put_program(cap_i, cap_s, has_ids)
        g_dest, g_off, g_w, g_ids, _ = take(sbufs, out["dest"], out["off"], out["w"], out["ids"],
                                            out["dest"], _put(nw, dev, jnp.int64),
                                            _put(nw + ns, dev, jnp.int64),
                                            _put(0, dev, jnp.int64))
        sd = np.asarray(g_dest)[:ns]
        so = np.asarray(g_off)[:ns]
        sw = np.asarray(g_w)[:ns]
        si = np.asarray(g_ids)[:ns] if has_ids else None
        sb = sd // p3
        for grp in np.split(np.arange(ns), np.flatnonzero(np.diff(sb)) + 1):
            spills.append((int(sb[grp[0]]), sd[grp], so[grp], sw[grp],
                           None if si is None else si[grp]))
    clock.mark("insert: spills to host")
    return dict(consumed=consumed, spills=spills, n_over=ns)


def _emigrants_only(e, has_ids, dev=None):
    """A staged slab cut to its emigrants once its own insert has run.

    Its keepers and its window are read by that insert alone; later inserts read
    only rows [n_keep, n_rows). Those rows are compacted in order onto the ladder,
    so the staged slab shrinks to the emigrant share.
    """
    import jax.numpy as jnp

    n_emig = e["n_rows"] - e["n_keep"]
    if n_emig <= 0:
        return dict(dest=None, off=None, w=None, ids=None, src=None, cap=0, n_keep=0,
                    n_rows=0, rr=e["rr"])
    cap_e = _ladder(n_emig)
    bufs = (_zeros(cap_e, jnp.int64, dev), _zeros((cap_e, 3), jnp.uint8, dev),
            _zeros((cap_e, 3), jnp.int16, dev),
            _zeros(cap_e, jnp.int32, dev) if has_ids else None, _zeros(cap_e, jnp.int64, dev))
    put = _put_program(e["cap"], cap_e, has_ids)
    dest, off, w, ids, src = put(bufs, e["dest"], e["off"], e["w"], e["ids"], e["src"],
                                 _put(e["n_keep"], dev, jnp.int64),
                                 _put(e["n_rows"], dev, jnp.int64), _put(0, dev, jnp.int64))
    return dict(dest=dest, off=off, w=w, ids=ids, src=src, cap=cap_e, n_keep=0,
                n_rows=n_emig, rr=e["rr"])


def _moved(e, dev):
    """An emigrant-only staged slab copied onto `dev`; (copy, bytes moved)."""
    moved, nbytes = dict(e), 0
    for k in ("dest", "off", "w", "ids", "src"):
        a = e.get(k)
        if a is not None:
            moved[k] = _put(a, dev)
            nbytes += int(a.nbytes)
    return moved, nbytes


def _sweep(st, lo, hi, reach, c_drift, scales, ar_slots, ar_bricks, dev, staged, ejected,
           clock, budget):
    """One card's pass over its own slabs [lo, hi): today's schedule, with any
    slabs already in `staged` (its boundary ejects, the neighbours' emigrants)
    used as they are. Returns (insert_res, peak staged)."""
    nb = int(st.bricks_per_side)
    has_ids = st.ids is not None
    scales_dev = budget.fixed[0]
    own = range(lo, hi)
    inserted, insert_res, peak = set(), {}, 0
    for s in own:
        if s not in ejected:
            staged[s] = _eject_slab(st, s, c_drift, scales, ar_slots, ar_bricks, clock, budget,
                                    dev)
            ejected[s] = (staged[s]["n_rows"] - staged[s]["n_keep"], staged[s]["rr"])
        for d in own:
            if d in inserted:
                continue
            if all(((d + o) % nb) in staged for o in reach):
                insert_res[d] = _insert_slab(st, d, reach, staged, scales_dev, clock, budget, dev)
                inserted.add(d)
                staged[d] = _emigrants_only(staged[d], has_ids, dev)
                clock.mark("insert: shrink staged", staged[d]["dest"])
        # release what no insert of THIS card can still need
        for s2 in list(staged):
            if all(((s2 + o) % nb) in inserted for o in reach if lo <= (s2 + o) % nb < hi):
                del staged[s2]
        peak = max(peak, len(staged))
    return insert_res, peak


def drift_and_migrate_device(st, c_drift, max_staged_slabs=None, timings=None,
                             device_budget_bytes=None, devices=None):
    """`state.drift_and_migrate` with every slab's row work on the device.

    Same contract, mutations and stats dict (plus a `migrate_device` receipt),
    gated bitwise against the serial numpy pass (`tests/test_migrate_device.py`,
    and `tests/test_migrate_device_cards.py` across cards). Needs `jax_enable_x64`.
    Arena-full and census refusals raise from the end-of-pass replay rather than
    mid-pass, as the pooled migrate's do.

    `timings`, if a dict, accumulates synced wall per phase (see `_Clock`); on
    several cards each key is prefixed with its card.

    `device_budget_bytes` refuses a slab whose estimated device footprint exceeds
    it (see THE BUDGET above), per card. A refusal after the first slab has been
    written leaves the state invalid, as the serial pass's mid-pass refusals do.
    `migrate_device["peak_estimate_bytes"]` reports the largest estimate.

    `devices` is a sequence of jax devices, one per card (None: one card, jax's
    default device). See SEVERAL CARDS above; `migrate_device` then also reports
    `cards`, `slabs_per_card`, `cross_card_segments` / `_bytes` and `fallback`.
    """
    from concurrent.futures import ThreadPoolExecutor

    from inexor import state as _state
    from inexor.eject_jax import require_x64
    from inexor.ooc_fft import partition_units

    global CALLS
    require_x64()
    CALLS += 1
    nb = int(st.bricks_per_side)
    has_ids = st.ids is not None
    n_before = int(st.occupancy.astype(np.int64).sum()) + st.arena_used
    scales = np.array(st.vel_scale, dtype=np.float64, copy=True)
    r_raw = _state.brick_reach(st, c_drift, scales)
    r = min(r_raw, nb // 2)
    reach = range(-r, r + 1)
    ar_slots, ar_bricks = pass_arena_index(st)

    devs = [None] if devices is None else list(devices)
    if not devs:
        raise ValueError("devices= was an empty sequence; pass None for one card")
    fallback = None
    if len(devs) > 1 and nb // len(devs) < 2 * r + 1:
        fallback = (f"a card would hold {nb // len(devs)} slabs, fewer than 2r + 1 = "
                    f"{2 * r + 1}: one card")
        devs = devs[:1]
    W = len(devs)
    parts = [(0, nb)] if W == 1 else partition_units(nb, W, 1)

    t_card = [None] * W if timings is None else [({} if W > 1 else timings) for _ in range(W)]
    clocks = [_Clock(t) for t in t_card]
    staged = [dict() for _ in range(W)]
    ejected = [dict() for _ in range(W)]
    budgets = [_Budget(device_budget_bytes, staged[k], [_put(scales, devs[k])])
               for k in range(W)]
    clocks[0].mark("pass: setup", budgets[0].fixed[0])

    def run(fn):
        if W == 1:
            return [fn(0)]
        with ThreadPoolExecutor(max_workers=W) as ex:
            return list(ex.map(fn, range(W)))

    segments = moved_bytes = 0
    if W > 1:
        # PHASE 1: every card ejects its r lowest and r highest slabs, and cuts
        # emigrant-only copies for the neighbours that insert from them
        def boundary(k):
            lo, hi = parts[k]
            out = {}
            for s in sorted(set(range(lo, lo + r)) | set(range(hi - r, hi))):
                e = _eject_slab(st, s, c_drift, scales, ar_slots, ar_bricks, clocks[k],
                                budgets[k], devs[k])
                staged[k][s] = e
                ejected[k][s] = (e["n_rows"] - e["n_keep"], e["rr"])
                out[s] = _emigrants_only(e, has_ids, devs[k])
            return out

        exports = run(boundary)
        for k in range(W):
            lo, hi = parts[k]
            for s, e in exports[k].items():
                for j in {(k - 1) % W, (k + 1) % W} - {k}:
                    jlo, jhi = parts[j]
                    if any(jlo <= (s + o) % nb < jhi for o in reach):
                        staged[j][s], nbytes = _moved(e, devs[j])
                        segments += 1
                        moved_bytes += nbytes
        exports = None
        clocks[0].mark("pass: boundary ejects + hand-off")

    # PHASE 2: each card sweeps its own slabs
    swept = run(lambda k: _sweep(st, parts[k][0], parts[k][1], reach, c_drift, scales,
                                 ar_slots, ar_bricks, devs[k], staged[k], ejected[k],
                                 clocks[k], budgets[k]))
    staged = None
    insert_res, n_emig, rr_by_slab = {}, {}, {}
    for k in range(W):
        insert_res.update(swept[k][0])
        for s, (ne, rr) in ejected[k].items():
            n_emig[s], rr_by_slab[s] = ne, rr

    clock = clocks[0]
    rep = _state._replay_arena_pass(
        st, reach, r, r_raw, c_drift, scales, n_emig=n_emig, rr_by_slab=rr_by_slab,
        insert_res=insert_res, max_staged_slabs=max_staged_slabs,
        census_note=" (Device pass.)")
    clock.mark("pass: arena replay")
    n_after = int(st.occupancy.astype(np.int64).sum()) + st.arena_used
    if n_after != n_before:
        raise ValueError(
            f"the migration lost {n_before - n_after} particles ({n_before} -> {n_after} "
            f"against {st.n_particles} stored). D-007 forbids dropping, so this is "
            f"corruption, not imprecision. {rep['n_inserted']} of {nb} slabs inserted, "
            f"arena {st.arena_used}/{st.n_arena} (device pass)"
        )
    clock.mark("pass: final census")
    if timings is not None and W > 1:
        for k, t in enumerate(t_card):
            for key, v in t.items():
                timings[f"card {k}: {key}"] = timings.get(f"card {k}: {key}", 0.0) + v
    receipt = dict(slabs=nb, programs=len(_PROGRAMS), spill_rows=rep["spill_rows"],
                   peak_estimate_bytes=max(b.peak for b in budgets),
                   budget_bytes=device_budget_bytes)
    if devices is not None:
        receipt.update(cards=W, slabs_per_card=[hi - lo for lo, hi in parts],
                       cross_card_segments=segments, cross_card_bytes=moved_bytes,
                       fallback=fallback)
    return dict(n_arena_overflow=rep["n_over"], arena_used=st.arena_used,
                vel_scale=float(np.max(st.vel_scale)),
                vel_scale_min=float(np.min(st.vel_scale)),
                n_migrated_checked=n_after, brick_reach=r, brick_reach_raw=r_raw,
                brick_reach_realized=rep["realized_reach"],
                peak_staged_slabs=rep["peak_staged"],
                migrate_device=receipt)
