"""The migrate on the device, slab by slab, bitwise the serial numpy pass.

Each x-slab's slot range, arena residents and occupancy slice go to the device once; a
program enumerates the slab's rows, the compiled `eject_jax` / `insert_jax` kernels run on
device arrays, keepers and emigrants stay on the device until their insert, and the written
rows are scattered into the slab's window, which returns to the host as one slice.

Why it is bitwise:
- An insert writes only inside its own slab's slot range (plus arena claims) and
  `brick_start` changes only in repack, so each slab's range is fixed for the pass.
- Every order-dependent arena mutation is deferred to `state._replay_arena_pass`, so
  residents read from the pre-pass host state are exact.
- Kernels get exactly the padded inputs (and pad fills) of `eject_rows` / `insert_rows`.
Row order is the reference's: per brick, the live run then arena residents by slot; insert
input is the slab's keepers, then its sources' emigrants in `sorted({(d + o) % nb})` order.

A staged slab keeps its eject output until its own insert, then its emigrants alone.
`device_budget_bytes` refuses a slab whose estimated footprint (held arrays plus the
kernels' per-padded-row bytes) exceeds it. Row buffers use each kernel's `_padded`; other
shapes use `forces.capacity_shape`. The window is `w_cap` rows uploaded as a view of the
host state (copied only where it would run off the end); rows past the slab are never read.
Every device-to-host read goes through `_host`, counted in `READS`.

With several cards (`devices=`), card k owns a contiguous run of slabs: it first ejects its
`r` lowest and highest slabs and hands emigrant-only copies to neighbours, then sweeps its
own slabs on its own thread; the arena is replayed once. A card with fewer than `2r + 1`
slabs cannot separate its boundaries, and the pass falls back to one card.
"""

from __future__ import annotations

import threading
import time

import numpy as np

INT16_MAX = 32767

#: Device bytes per padded row of one kernel call, uploaded inputs included (measured
#: peaks at a 4096^3 slab). Estimate only.
EJECT_B_PER_PADDED_ROW = 114.4
INSERT_B_PER_PADDED_ROW = 83.0
#: Measured estimate / device-peak ratio of a whole pass: the kernel coefficients were
#: measured alone and the pass's surrounding arrays under-read by this factor.
ESTIMATE_OVER_MEASURED = 0.76

_PROGRAMS: dict = {}
_PROGRAM_LOCK = threading.Lock()

#: Count of device migrate passes (see `eject_jax.CALLS`).
CALLS = 0


class _Clock:
    """Synced phase timer for `timings=`; a no-op when `timings` is None.

    `mark(key, *arrays)` blocks until the arrays are ready, then charges the wall
    since the previous mark to `key`. Syncing moves the wall, so a timed pass is a
    breakdown, not the pass's cost.
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
                "loop's resident terms. Use fewer rows per slab or a "
                "smaller drift (fewer staged slabs), or raise the budget deliberately."
            )


def _ladder(n):
    from ..forces import capacity_shape

    return int(capacity_shape(max(1, int(n))))


def _put(a, dev, dtype=None):
    """`a` as a device array on `dev` (None: `jnp.asarray` on jax's default device)."""
    import jax
    import jax.numpy as jnp

    if dev is None:
        return jnp.asarray(a, dtype=dtype)
    if isinstance(a, jax.Array) and dtype is None:
        return jax.device_put(a, dev)
    return jax.device_put(np.asarray(a, dtype=dtype), dev)


def _zeros(shape, dtype, dev):
    """Zeros built on `dev` (None: `jnp.zeros`)."""
    import jax.numpy as jnp

    if dev is None:
        return jnp.zeros(shape, dtype)
    from .paint import _zeros_on

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


#: Device-to-host reads by phase.
READS: dict = {}
_READS_LOCK = threading.Lock()


def _host(x, what):
    """`np.asarray(x)`, counted in `READS[what]`; allowed under a disallowing transfer guard."""
    import jax

    with jax.transfer_guard_device_to_host("allow"):
        a = np.asarray(x)
    with _READS_LOCK:
        READS[what] = READS.get(what, 0) + 1
    return a


def pass_arena_index(st):
    """(slots, bricks): arena residents ordered by brick, ascending slot within.

    The grouping of `SlotState._build_arena_index`, vectorized; valid for a pass whose
    releases and claims are deferred.
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


def _rows_program(cap, w_cap, a_cap, nb, p3, per, has_ids, n_b=None):
    """One unit's eject inputs at `cap` rows, built from its window on the device. `n_b` is the
    unit's brick count (default a whole slab, `nb**2`)."""
    nb2 = nb * nb
    n_b = nb2 if n_b is None else int(n_b)

    def make():
        import jax
        import jax.numpy as jnp

        lift = cap + 1  # above every prefix sum and rank (both <= cap)

        @jax.jit
        def rows(occ, live, starts_rel, row_offsets, ar_offsets, ar_bucket, off_win, w_win,
                 ids_win, ar_off, ar_w, ar_ids, scales_s, n_rows, lo_b):
            r = jnp.arange(cap, dtype=jnp.int64)
            real = r < n_rows
            bi = jnp.clip(jnp.searchsorted(row_offsets[1:], r, side="right"),
                          0, n_b - 1).astype(jnp.int64)
            rank = r - row_offsets[bi]
            lc = live[bi]
            is_ar = rank >= lc
            # live rows: bucket = count of the brick's prefix sums <= rank, via one
            # searchsorted over per-brick-lifted prefix sums (as the device decode)
            occ_cum = jnp.cumsum(occ.astype(jnp.int64), axis=1)
            keys = (occ_cum + (jnp.arange(n_b, dtype=jnp.int64) * lift)[:, None]).reshape(-1)
            within = jnp.searchsorted(keys, bi * lift + rank, side="right").astype(jnp.int64)
            within = jnp.clip(within - bi * p3, 0, p3 - 1)
            bucket_live = (lo_b + bi) * p3 + within
            # arena rows: the residents array, in index order
            k = jnp.clip(ar_offsets[bi] + rank - lc, 0, a_cap - 1)
            slot = jnp.where(real & ~is_ar, starts_rel[bi] + rank, 0)
            a = is_ar[:, None]
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
            off = jnp.where(m, jnp.where(a, ar_off[k], off_win[slot]), jnp.zeros((), off_win.dtype))
            w = jnp.where(m, jnp.where(a, ar_w[k], w_win[slot]), jnp.zeros((), w_win.dtype))
            bijk = jnp.where(m, bijk, 0)
            scale = jnp.where(real, scales_s[bi], 1.0)[:, None]
            brick = jnp.where(real, lo_b + bi, -1)
            ids = (jnp.where(real, jnp.where(is_ar, ar_ids[k], ids_win[slot]),
                             jnp.zeros((), ids_win.dtype)) if has_ids else None)
            return off, bijk, w, ids, scale, brick, real

        return rows

    return _program(("rows", cap, w_cap, a_cap, nb, p3, per, has_ids, n_b), make)


def _eject_scalars_program(cap, nb, p3):
    """(n_keep, realized x-reach, emigrant count per destination slab) as one int64 vector."""

    def make():
        import jax
        import jax.numpy as jnp

        per_slab = p3 * nb * nb

        @jax.jit
        def scalars(dest, n_keep, n_rows, s):
            idx = jnp.arange(cap, dtype=jnp.int64)
            n_keep = n_keep.astype(jnp.int64)
            em = (idx >= n_keep) & (idx < n_rows)
            to = dest // per_slab
            disp = (to - s + nb // 2) % nb - nb // 2
            rr = jnp.max(jnp.where(em, jnp.abs(disp), 0))
            counts = jnp.zeros(nb, jnp.int64).at[jnp.where(em, to, nb)].add(1, mode="drop")
            return jnp.concatenate([n_keep[None], rr[None].astype(jnp.int64), counts])

        return scalars

    return _program(("eject_scalars", cap, nb, p3), make)


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


# These share `eject_jax._CACHE` / `insert_jax._CACHE` and their key and `_build`
# signatures: a change to either must land in both files.
def _eject_kernel(t9, nb, cap, has_ids):
    from .. import eject_jax

    key = (int(t9.n_buckets_side), float(t9.quantum), int(nb), int(cap), has_ids)
    with _PROGRAM_LOCK:
        fn = eject_jax._CACHE.get(key)
        if fn is None:
            fn = eject_jax._CACHE[key] = eject_jax._build(t9, int(nb), int(cap), has_ids)
    return fn


def _insert_kernel(p3, nb2, cap, has_ids):
    from .. import insert_jax

    key = (int(p3), int(nb2), int(cap), has_ids)
    with _PROGRAM_LOCK:
        fn = insert_jax._CACHE.get(key)
        if fn is None:
            fn = insert_jax._CACHE[key] = insert_jax._build(int(p3), int(nb2), int(cap), has_ids)
    return fn


def _window_fits(n_state_rows, s0, w_cap):
    """True when `w_cap` rows from slot `s0` lie inside the state: the window is a view."""
    return s0 + w_cap <= n_state_rows


def _slab_index(st, s, ar_slots, ar_bricks):
    """`_unit_index` of the whole x-slab `s`."""
    return _unit_index(st, *st.slab_bricks(s), ar_slots, ar_bricks)


def _unit_index(st, lo_b, hi_b, ar_slots, ar_bricks):
    """The host index of bricks [lo_b, hi_b) (a slab or a unit of one) for their eject rows,
    O(bricks + arena): the per-brick occupancy, live and resident counts and prefix sums, the
    arena residents in `pass_arena_index` order, and the padded shapes. Shared by the eject
    and the destination census (`device.window`), so both build identical kernel inputs."""
    from .. import eject_jax

    p3 = int(st.buckets_per_brick)
    lo_b, hi_b = int(lo_b), int(hi_b)
    n_b = hi_b - lo_b
    s0, s1 = int(st.brick_start[lo_b]), int(st.brick_start[hi_b])
    occ = st._occ(lo_b, hi_b).reshape(n_b, p3)
    live = occ.sum(axis=1, dtype=np.int64)
    a0, a1 = np.searchsorted(ar_bricks, [lo_b, hi_b])
    rows_a = ar_slots[a0:a1]
    ar_counts = np.bincount(ar_bricks[a0:a1] - lo_b, minlength=n_b).astype(np.int64)
    ar_offsets = np.zeros(n_b + 1, dtype=np.int64)
    np.cumsum(ar_counts, out=ar_offsets[1:])
    row_offsets = np.zeros(n_b + 1, dtype=np.int64)
    np.cumsum(live + ar_counts, out=row_offsets[1:])
    n_rows = int(row_offsets[-1])
    n_ar = int(a1 - a0)
    a_cap = _ladder(n_ar)
    ar_bucket = np.zeros(a_cap, dtype=np.int64)
    ar_bucket[:n_ar] = np.asarray(st.arena_bucket)[rows_a - int(st.arena_base)]
    return dict(lo_b=lo_b, hi_b=hi_b, s0=s0, s1=s1, span=s1 - s0, occ=occ, live=live,
                rows_a=rows_a, ar_offsets=ar_offsets, row_offsets=row_offsets, n_rows=n_rows,
                n_ar=n_ar, a_cap=a_cap, ar_bucket=ar_bucket, cap=eject_jax._padded(n_rows))


def _eject_rows(st, ix, c_drift, off_win, w_win, ids_win, ar_rows, starts_rel, scales_d, w_cap,
                clock, dev):
    """The rows program and the eject kernel on `dev` for one slab indexed by `ix`
    (`_slab_index`), against a window whose rows `starts_rel` are relative to and
    whose residents are `ar_rows`. `scales_d` is the slab's per-brick scales on
    `dev`. Returns the kernel's outputs."""
    import jax.numpy as jnp

    nb, p3 = int(st.bricks_per_side), int(st.buckets_per_brick)
    per = int(st.t9.n_buckets_side) // nb
    has_ids = st.ids is not None
    cap, lo_b = ix["cap"], ix["lo_b"]
    index_dev = (_put(ix["occ"], dev), _put(ix["live"], dev), starts_rel,
                 _put(ix["row_offsets"], dev), _put(ix["ar_offsets"], dev),
                 _put(ix["ar_bucket"], dev))
    clock.mark("eject: upload", index_dev, off_win, w_win, ids_win, ar_rows, scales_d)
    rows = _rows_program(cap, w_cap, ix["a_cap"], nb, p3, per, has_ids,
                         n_b=ix["hi_b"] - lo_b)
    off, bijk, w, ids, scale, brick, real = rows(
        *index_dev, off_win, w_win, ids_win, *ar_rows, scales_d,
        _put(ix["n_rows"], dev, jnp.int64), _put(lo_b, dev, jnp.int64))
    index_dev = None
    clock.mark("eject: rows program", off, bijk, w, ids, scale, brick, real)
    fn = _eject_kernel(st.t9, nb, cap, has_ids)
    out = fn(off, bijk, w, ids, scale, float(c_drift), brick, real)
    # the row buffers are the kernel's input only; release them before staging
    off = bijk = w = ids = scale = brick = real = None
    clock.mark("eject: kernel", *out)
    return out


def _upload_slab(st, s, ar_slots, ar_bricks, clock, dev=None, block=False):
    """Slab s's index and its window on `dev`: the slot range as a view of the host
    state and the arena residents as their own array. `block=True`, for a caller about
    to write host rows this window covers, copies the slot range (a CPU-backend upload
    of a view aliases the host) and waits for the transfer. Returns `_eject_slab(pre=)`."""
    import jax

    has_ids = st.ids is not None
    ix = _slab_index(st, s, ar_slots, ar_bricks)
    s0, s1, span = ix["s0"], ix["s1"], ix["span"]
    n_ar, a_cap, rows_a = ix["n_ar"], ix["a_cap"], ix["rows_a"]
    w_cap = _ladder(span + a_cap)
    clock.mark("eject: host index")

    direct = not block and _window_fits(len(st.off), s0, w_cap)

    def window(src):
        if direct:
            return src[s0:s0 + w_cap]
        out = np.zeros((w_cap,) + src.shape[1:], dtype=src.dtype)
        out[:span] = src[s0:s1]
        return out

    def residents(src):
        out = np.zeros((a_cap,) + src.shape[1:], dtype=src.dtype)
        out[:n_ar] = src[rows_a]
        return out

    off_np, w_np = window(st.off), window(st.w)
    ids_np = window(st.ids) if has_ids else None
    ar_np = (residents(st.off), residents(st.w), residents(st.ids) if has_ids else None)
    clock.mark("eject: window")

    win = (_put(off_np, dev), _put(w_np, dev), _put(ids_np, dev) if has_ids else None)
    ar_rows = tuple(None if a is None else _put(a, dev) for a in ar_np)
    if block:
        jax.block_until_ready((win, ar_rows))
    return dict(ix=ix, win=win, ar_rows=ar_rows, direct=direct, w_cap=w_cap)


def _eject_slab(st, s, c_drift, scales, ar_slots, ar_bricks, clock, budget, dev=None, pre=None,
                keep_window=True):
    """Upload slab s's window to `dev` (or take `pre`, an `_upload_slab`), build its
    rows and eject them. Returns the staged slab; `keep_window=False` releases the
    window with the rows, for a caller whose insert does not write into it."""
    import jax.numpy as jnp

    nb, p3 = int(st.bricks_per_side), int(st.buckets_per_brick)
    has_ids = st.ids is not None
    if pre is None:
        pre = _upload_slab(st, s, ar_slots, ar_bricks, clock, dev)
    ix, w_cap = pre["ix"], pre["w_cap"]
    lo_b, hi_b, s0, s1, span = ix["lo_b"], ix["hi_b"], ix["s0"], ix["s1"], ix["span"]
    n_rows, cap = ix["n_rows"], ix["cap"]
    row_bytes = st.off.itemsize * 3 + st.w.itemsize * 3 + (st.ids.itemsize if has_ids else 0)
    budget.check(f"ejecting slab {s}", w_cap * row_bytes, EJECT_B_PER_PADDED_ROW, cap)
    off_win, w_win, ids_win = pre["win"]
    ar_rows, direct = pre["ar_rows"], pre["direct"]
    pre = None
    starts_d = _put(np.asarray(st.brick_start[lo_b:hi_b], dtype=np.int64) - s0, dev)
    scales_d = _put(np.asarray(scales[lo_b:hi_b], dtype=np.float64), dev)
    dest, off_new, w_out, ids_out, src_out, n_keep = _eject_rows(
        st, ix, c_drift, off_win, w_win, ids_win, ar_rows, starts_d, scales_d, w_cap, clock, dev)
    ar_rows = starts_d = scales_d = None
    if not keep_window:
        off_win = w_win = ids_win = None

    # n_keep, realized x-reach and per-destination emigrant counts in one read
    sc = _host(_eject_scalars_program(cap, nb, p3)(
        dest, n_keep, _put(n_rows, dev, jnp.int64), _put(s, dev, jnp.int64)), "eject: scalars")
    n_keep, rr, to = int(sc[0]), int(sc[1]), sc[2:]
    clock.mark("eject: reach + scalars")

    return dict(dest=dest, off=off_new, w=w_out, ids=ids_out, src=src_out, cap=cap,
                n_keep=n_keep, n_rows=n_rows, rr=rr, to=to, direct=direct,
                win=(off_win, w_win, ids_win) if keep_window else (),
                s0=s0, s1=s1, span=span, w_cap=w_cap, lo_b=lo_b, hi_b=hi_b)


def _insert_on_card(st, d, reach, staged, scales_dev, clock, budget, dev=None):
    """Slab d's insert kernel on `dev` over its keepers and its sources' emigrants.
    Returns (kernel outputs, n_write, n_spill, consumed, cap_i); nothing is written."""
    import jax.numpy as jnp

    from .. import insert_jax

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
            consumed[s] = int(e["to"][d])
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
    if float(_host(out["abs_max"], "insert: scalars")) > INT16_MAX:
        raise ValueError(
            f"velocity code {float(out['abs_max']):.0f} escapes int16 under a rescale to a "
            "scale that does not cover it. Per-brick scales make this reachable where a "
            "global scale made it impossible; the caller must fix the destination scale "
            "over the rows it is about to write. Integer state is never clamped."
        )
    nw = int(_host(out["n_write"], "insert: scalars"))
    ns = int(_host(out["n_spill"], "insert: scalars"))
    # release the compacted inputs; only the kernel outputs are read from here
    bufs = b_dest = b_off = b_w = b_ids = b_src = s_old = real = None
    return out, nw, ns, consumed, cap_i


def _spills(st, out, nw, ns, cap_i, dev):
    """The kernel's spilled rows [nw, nw + ns): (host groups `((brick, dest, off, w,
    ids), ...)` in ascending brick order, the rows on `dev` on the ladder
    `(dest, off, w, ids)`, or `([], None)` with no spill."""
    import jax.numpy as jnp

    p3 = int(st.buckets_per_brick)
    has_ids = st.ids is not None
    spills = []
    if not ns:
        return spills, None
    # gather onto the ladder: a slice sized by the spill count would compile per slab
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
    sd = _host(g_dest, "insert: spills")[:ns]
    so = _host(g_off, "insert: spills")[:ns]
    sw = _host(g_w, "insert: spills")[:ns]
    si = _host(g_ids, "insert: spills")[:ns] if has_ids else None
    sb = sd // p3
    for grp in np.split(np.arange(ns), np.flatnonzero(np.diff(sb)) + 1):
        spills.append((int(sb[grp[0]]), sd[grp], so[grp], sw[grp],
                       None if si is None else si[grp]))
    return spills, (g_dest, g_off, g_w, g_ids)


def _insert_slab(st, d, reach, staged, scales_dev, clock, budget, dev=None):
    """Insert slab d on `dev` and write its slot range back. Returns insert_res."""
    import jax.numpy as jnp

    from ..layout import _to_index

    has_ids = st.ids is not None
    lo_b, hi_b = st.slab_bricks(d)
    out, nw, ns, consumed, cap_i = _insert_on_card(st, d, reach, staged, scales_dev, clock,
                                                   budget, dev)
    e = staged[d]
    write = _write_program(cap_i, e["w_cap"], has_ids)
    off_win, w_win, ids_win = write(*e["win"], out["pos"], out["off"], out["w"], out["ids"],
                                    _put(nw, dev, jnp.int64), _put(e["s0"], dev, jnp.int64))
    clock.mark("insert: scalars + write program", off_win, w_win, ids_win)
    s0, s1, span = e["s0"], e["s1"], e["span"]
    st.off[s0:s1] = _host(off_win, "insert: slot range")[:span]
    st.w[s0:s1] = _host(w_win, "insert: slot range")[:span]
    if has_ids:
        st.ids[s0:s1] = _host(ids_win, "insert: slot range")[:span]
    clock.mark("insert: slot range to host")
    st._occ(lo_b, hi_b)[...] = _to_index(_host(out["occupancy"], "insert: occupancy"),
                                         st.index_dtype, "migrated")
    st.vel_scale[lo_b:hi_b] = _host(out["scales"], "insert: scales")
    clock.mark("insert: occupancy + scales to host")
    spills, _rows = _spills(st, out, nw, ns, cap_i, dev)
    clock.mark("insert: spills to host")
    return dict(consumed=consumed, spills=spills, n_over=ns)


def _emigrants_only(e, has_ids, dev=None):
    """A staged slab cut to its emigrants once its own insert has run.

    Later inserts read only rows [n_keep, n_rows), so those are compacted in order
    onto the ladder.
    """
    import jax.numpy as jnp

    n_emig = e["n_rows"] - e["n_keep"]
    if n_emig <= 0:
        return dict(dest=None, off=None, w=None, ids=None, src=None, cap=0, n_keep=0,
                    n_rows=0, rr=e["rr"], to=e["to"])
    cap_e = _ladder(n_emig)
    bufs = (_zeros(cap_e, jnp.int64, dev), _zeros((cap_e, 3), jnp.uint8, dev),
            _zeros((cap_e, 3), jnp.int16, dev),
            _zeros(cap_e, jnp.int32, dev) if has_ids else None, _zeros(cap_e, jnp.int64, dev))
    put = _put_program(e["cap"], cap_e, has_ids)
    dest, off, w, ids, src = put(bufs, e["dest"], e["off"], e["w"], e["ids"], e["src"],
                                 _put(e["n_keep"], dev, jnp.int64),
                                 _put(e["n_rows"], dev, jnp.int64), _put(0, dev, jnp.int64))
    return dict(dest=dest, off=off, w=w, ids=ids, src=src, cap=cap_e, n_keep=0,
                n_rows=n_emig, rr=e["rr"], to=e["to"])


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
           clock, budget, insert=None, card=None, keep_window=True):
    """One card's pass over its own slabs [lo, hi), using any slabs already in
    `staged` (its boundary ejects, the neighbours' emigrants) as they are. `insert(st, d, reach, staged, scales_dev, clock, budget, dev,
    card)` runs each destination (default: `_insert_slab`); `card["pre"]` holds
    windows uploaded ahead of their eject. Returns (insert_res, peak staged)."""
    nb = int(st.bricks_per_side)
    has_ids = st.ids is not None
    scales_dev = budget.fixed[0]
    card = card if card is not None else dict(pre={})
    own = range(lo, hi)
    inserted, insert_res, peak = set(), {}, 0
    for s in own:
        if s not in ejected:
            staged[s] = _eject_slab(st, s, c_drift, scales, ar_slots, ar_bricks, clock, budget,
                                    dev, pre=card["pre"].pop(s, None), keep_window=keep_window)
            ejected[s] = (staged[s]["n_rows"] - staged[s]["n_keep"], staged[s]["rr"],
                          staged[s]["direct"])
        for d in own:
            if d in inserted:
                continue
            if all(((d + o) % nb) in staged for o in reach):
                if insert is None:
                    insert_res[d] = _insert_slab(st, d, reach, staged, scales_dev, clock, budget,
                                                 dev)
                else:
                    insert_res[d] = insert(st, d, reach, staged, scales_dev, clock, budget, dev,
                                           card)
                inserted.add(d)
                staged[d] = _emigrants_only(staged[d], has_ids, dev)
                clock.mark("insert: shrink staged", staged[d]["dest"])
        # release what no insert of THIS card can still need
        for s2 in list(staged):
            if all(((s2 + o) % nb) in inserted for o in reach if lo <= (s2 + o) % nb < hi):
                del staged[s2]
        peak = max(peak, len(staged))
    return insert_res, peak


def pass_reach(st, c_drift, comm=None, vel_scale=None):
    """(r_raw, r): the brick reach of this drift from the owned bricks' scales (non-owned
    bricks carry 1.0, which must not enter the max), maximized over ranks; `r` capped at
    half the grid. `vel_scale` overrides `st.vel_scale`."""
    from .. import state as _state

    nb = int(st.bricks_per_side)
    blo, bhi = st.owned_bricks
    vs = np.asarray(st.vel_scale if vel_scale is None else vel_scale)
    r_raw = int(_state.brick_reach(st, c_drift, vs[blo:bhi]))
    if comm is not None and comm.size > 1:
        r_raw = int(comm.allreduce(r_raw, "max"))
    return r_raw, min(r_raw, nb // 2)


_EMIGRANT_FIELDS = ("dest", "off", "w", "ids", "src")


def _pack_emigrants(slabs):
    """Host parcel of emigrant-only staged slabs `{s: entry}` for `comm.exchange_neighbours`:
    exactly the emigrant rows, plus each slab's count, realized reach and per-slab counts."""
    out = {}
    for s, e in slabs.items():
        n = int(e["n_rows"])
        out[f"{s}:meta"] = np.array([n, int(e["rr"])], dtype=np.int64)
        out[f"{s}:to"] = np.asarray(e["to"], dtype=np.int64)
        if n:
            for k in _EMIGRANT_FIELDS:
                if e.get(k) is not None:
                    out[f"{s}:{k}"] = np.ascontiguousarray(_host(e[k], "hand-off: to host")[:n])
    return out


def _unpack_emigrants(parcel, dev):
    """`{s: entry}` from `_pack_emigrants`, on `dev`, zero-padded to the sender's ladder (the
    same arrays `_emigrants_only` built there)."""
    out = {}
    for s in sorted({int(k.split(":")[0]) for k in parcel}):
        n, rr = (int(x) for x in parcel[f"{s}:meta"])
        e = dict(cap=0, n_keep=0, n_rows=n, rr=rr, to=parcel[f"{s}:to"],
                 **{k: None for k in _EMIGRANT_FIELDS})
        if n:
            e["cap"] = cap = _ladder(n)
            for k in _EMIGRANT_FIELDS:
                a = parcel.get(f"{s}:{k}")
                if a is not None:
                    buf = np.zeros((cap,) + a.shape[1:], dtype=a.dtype)
                    buf[:n] = a
                    e[k] = _put(buf, dev)
        out[s] = e
    return out

def _device_pass(st, c_drift, timings=None, device_budget_bytes=None, devices=None,
                 insert=None, keep_window=True, before_sweep=None, comm=None):
    """Every slab's eject and insert on the cards: the part of
    `drift_and_migrate_device` before its arena replay, shared with the fused
    migrate + repack. `insert` and `keep_window` as in `_sweep`; `before_sweep(ctx)`
    runs after the boundary ejects and before any card inserts. Returns the pass
    context (reach, cards, per-slab emigrant counts and reach, insert results, the
    receipt's terms).

    Across ranks (`comm`; a node-local `st`, every rank calls this): the reach is the max
    over ranks, the r slabs next to each end send their pre-pass scales to the neighbour
    (the inserts read their source bricks' scales), the first and last cards' boundary
    ejects send emigrant-only copies to the neighbour ranks, and each rank reports what its
    inserts took from the neighbours' slabs (`foreign_consumed`, for their replay)."""
    from concurrent.futures import ThreadPoolExecutor

    from .. import state as _state
    from ..comm import exchange_neighbours
    from ..eject_jax import require_x64
    from ..ooc_fft import partition_units

    global CALLS
    require_x64()
    CALLS += 1
    nb = int(st.bricks_per_side)
    nb2 = nb * nb
    has_ids = st.ids is not None
    multi = comm is not None and comm.size > 1
    lo_s, hi_s = st.owned_slabs
    n_before = _state.occupancy_total(st.occupancy) + st.arena_used
    scales = np.array(st.vel_scale, dtype=np.float64, copy=True)
    r_raw, r = pass_reach(st, c_drift, comm, scales)
    reach = range(-r, r + 1)
    if multi:
        thin = int(comm.allreduce(hi_s - lo_s, "min"))
        if thin < 2 * r + 1:
            raise ValueError(
                f"the drift reaches {r} brick slab(s) but a rank holds {thin}, fewer than "
                f"2r + 1 = {2 * r + 1}: particles would skip past a neighbour rank. Use "
                "fewer ranks or a smaller step.")
        # the neighbours' boundary slabs' pre-pass scales, for this pass only
        got_l, got_r = exchange_neighbours(
            comm, dict(s=scales[lo_s * nb2:(lo_s + r) * nb2]),
            dict(s=scales[(hi_s - r) * nb2:hi_s * nb2]))
        for got, first in ((got_l, lo_s - r), (got_r, hi_s)):
            for i in range(r):
                t = (first + i) % nb
                scales[t * nb2:(t + 1) * nb2] = got["s"][i * nb2:(i + 1) * nb2]
    ar_slots, ar_bricks = pass_arena_index(st)

    devs = [None] if devices is None else list(devices)
    if not devs:
        raise ValueError("devices= was an empty sequence; pass None for one card")
    fallback = None
    n_own = hi_s - lo_s
    if len(devs) > 1 and n_own // len(devs) < 2 * r + 1:
        fallback = (f"a card would hold {n_own // len(devs)} slabs, fewer than 2r + 1 = "
                    f"{2 * r + 1}: one card")
        devs = devs[:1]
    W = len(devs)
    parts = ([(lo_s, hi_s)] if W == 1
             else [(lo_s + a, lo_s + z) for a, z in partition_units(n_own, W, 1)])

    t_card = [None] * W if timings is None else [({} if W > 1 else timings) for _ in range(W)]
    clocks = [_Clock(t) for t in t_card]
    staged = [dict() for _ in range(W)]
    ejected = [dict() for _ in range(W)]
    budgets = [_Budget(device_budget_bytes, staged[k], [_put(scales, devs[k])])
               for k in range(W)]
    cards = [dict(k=k, lo=parts[k][0], hi=parts[k][1], pre={}, ejected=ejected[k],
                  staged=staged[k], dev=devs[k], clock=clocks[k], budget=budgets[k])
             for k in range(W)]
    clocks[0].mark("pass: setup", budgets[0].fixed[0])

    def run(fn):
        if W == 1:
            return [fn(0)]
        with ThreadPoolExecutor(max_workers=W) as ex:
            return list(ex.map(fn, range(W)))

    segments = moved_bytes = 0
    rank_sent = dict(rows=0, bytes=0)
    rank_recv = dict(rows=0, bytes=0)
    if W > 1 or multi:
        # every card ejects its r lowest and r highest slabs and cuts emigrant-only
        # copies for the neighbours that insert from them
        def boundary(k):
            lo, hi = parts[k]
            out = {}
            for s in sorted(set(range(lo, lo + r)) | set(range(hi - r, hi))):
                e = _eject_slab(st, s, c_drift, scales, ar_slots, ar_bricks, clocks[k],
                                budgets[k], devs[k], keep_window=keep_window)
                staged[k][s] = e
                ejected[k][s] = (e["n_rows"] - e["n_keep"], e["rr"], e["direct"])
                out[s] = _emigrants_only(e, has_ids, devs[k])
            return out

        exports = run(boundary)
        for k in range(W):
            lo, hi = parts[k]
            # the rank's cards form a ring on one rank; across ranks, a line whose ends hand
            # off to the neighbour ranks below
            nbrs = ({(k - 1) % W, (k + 1) % W} - {k}) if not multi else (
                {k - 1, k + 1} & set(range(W)))
            for s, e in exports[k].items():
                for j in nbrs:
                    jlo, jhi = parts[j]
                    if any(jlo <= (s + o) % nb < jhi for o in reach):
                        staged[j][s], nbytes = _moved(e, devs[j])
                        segments += 1
                        moved_bytes += nbytes
        if multi:
            to_l = _pack_emigrants({s: exports[0][s] for s in range(lo_s, lo_s + r)})
            to_r = _pack_emigrants({s: exports[W - 1][s] for s in range(hi_s - r, hi_s)})
            got_l, got_r = exchange_neighbours(comm, to_l, to_r)
            for got, k in ((got_l, 0), (got_r, W - 1)):
                for s, e in _unpack_emigrants(got, devs[k]).items():
                    staged[k][s] = e
                    rank_recv["rows"] += int(e["n_rows"])
                rank_recv["bytes"] += sum(int(v.nbytes) for v in got.values())
            rank_sent["rows"] = sum(int(e["n_rows"]) for e in (
                [exports[0][s] for s in range(lo_s, lo_s + r)]
                + [exports[W - 1][s] for s in range(hi_s - r, hi_s)]))
            rank_sent["bytes"] = sum(int(v.nbytes) for d in (to_l, to_r) for v in d.values())
        exports = None
        clocks[0].mark("pass: boundary ejects + hand-off")

    ctx = dict(nb=nb, r=r, r_raw=r_raw, reach=reach, scales=scales, n_before=n_before,
               devs=devs, W=W, parts=parts, cards=cards, clocks=clocks, budgets=budgets,
               t_card=t_card, fallback=fallback, ar_slots=ar_slots, ar_bricks=ar_bricks,
               run=run)
    if before_sweep is not None:
        before_sweep(ctx)

    # each card sweeps its own slabs
    swept = run(lambda k: _sweep(st, parts[k][0], parts[k][1], reach, c_drift, scales,
                                 ar_slots, ar_bricks, devs[k], staged[k], ejected[k],
                                 clocks[k], budgets[k], insert=insert, card=cards[k],
                                 keep_window=keep_window))
    insert_res, n_emig, rr_by_slab, n_direct = {}, {}, {}, 0
    for k in range(W):
        insert_res.update(swept[k][0])
        for s, (ne, rr, direct) in ejected[k].items():
            n_emig[s], rr_by_slab[s] = ne, rr
            n_direct += bool(direct)
        staged[k].clear()
    # rows this rank's inserts took from other ranks' slabs, reported to their owners
    foreign_consumed, taken = {}, 0
    if multi:
        side = {}
        for d, res in insert_res.items():
            for src, c in res["consumed"].items():
                if not lo_s <= src < hi_s:
                    left = (lo_s - src) % nb <= r
                    side.setdefault("l" if left else "r", []).append((d, src, int(c)))
                    taken += int(c)

        def parcel(rows):
            a = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
            return dict(dsc=a)

        got_l, got_r = exchange_neighbours(comm, parcel(side.get("l", [])),
                                           parcel(side.get("r", [])))
        for got in (got_l, got_r):
            for d, src, c in got["dsc"].tolist():
                foreign_consumed[(d, src)] = foreign_consumed.get((d, src), 0) + c
    given = sum(foreign_consumed.values())
    ctx.update(insert_res=insert_res, n_emig=n_emig, rr_by_slab=rr_by_slab, n_direct=n_direct,
               segments=segments, moved_bytes=moved_bytes, foreign_consumed=foreign_consumed,
               rows_in=taken, rows_out=given, rank_sent=rank_sent, rank_recv=rank_recv,
               multi=multi, comm=comm)
    return ctx


def _merge_card_timings(ctx, timings):
    if timings is not None and ctx["W"] > 1:
        for k, t in enumerate(ctx["t_card"]):
            for key, v in t.items():
                timings[f"card {k}: {key}"] = timings.get(f"card {k}: {key}", 0.0) + v


def _pass_receipt(ctx, devices, device_budget_bytes, spill_rows):
    n_own = sum(hi - lo for lo, hi in ctx["parts"])
    receipt = dict(slabs=n_own, programs=len(_PROGRAMS), spill_rows=spill_rows,
                   peak_estimate_bytes=max(b.peak for b in ctx["budgets"]),
                   budget_bytes=device_budget_bytes, window_direct_slabs=ctx["n_direct"],
                   window_copied_slabs=n_own - ctx["n_direct"],
                   rank_emigrant_rows_sent=ctx["rank_sent"]["rows"],
                   rank_emigrant_rows_received=ctx["rank_recv"]["rows"],
                   rank_hand_off_bytes_sent=ctx["rank_sent"]["bytes"],
                   rank_hand_off_bytes_received=ctx["rank_recv"]["bytes"],
                   rank_rows_in=int(ctx["rows_in"]), rank_rows_out=int(ctx["rows_out"]))
    if devices is not None:
        receipt.update(cards=ctx["W"], slabs_per_card=[hi - lo for lo, hi in ctx["parts"]],
                       cross_card_segments=ctx["segments"], cross_card_bytes=ctx["moved_bytes"],
                       fallback=ctx["fallback"])
    return receipt


def _migrate_stats(st, ctx, rep, n_after, receipt):
    blo, bhi = st.owned_bricks
    return dict(n_arena_overflow=rep["n_over"], arena_used=st.arena_used,
                vel_scale=float(np.max(st.vel_scale[blo:bhi])),
                vel_scale_min=float(np.min(st.vel_scale[blo:bhi])),
                n_migrated_checked=n_after, brick_reach=ctx["r"], brick_reach_raw=ctx["r_raw"],
                brick_reach_realized=rep["realized_reach"],
                peak_staged_slabs=rep["peak_staged"],
                migrate_device=receipt)


def conserve(st, ctx, n_after, what):
    """Refuse a pass that lost or made particles: on this rank against the rows it sent and
    received, and (across ranks) in total; then record a node-local state's new count."""
    want = ctx["n_before"] - ctx["rows_out"] + ctx["rows_in"]
    if n_after != want:
        raise ValueError(
            f"the {what} holds {n_after} particles on this rank against {want} expected "
            f"({ctx['n_before']} before, {ctx['rows_out']} sent, {ctx['rows_in']} received). "
            "Particles are never dropped, so this is corruption, not imprecision.")
    if ctx["multi"]:
        comm = ctx["comm"]
        before, after = comm.allreduce(int(ctx["n_before"])), comm.allreduce(int(n_after))
        if before != after:
            raise ValueError(f"the {what} holds {after} particles over all ranks against "
                             f"{before} before")
    if not st.is_whole:
        st.n_particles = int(n_after)


def drift_and_migrate_device(st, c_drift, max_staged_slabs=None, timings=None,
                             device_budget_bytes=None, devices=None, comm=None):
    """`state.drift_and_migrate` with every slab's row work on the device.

    Same contract, mutations and stats dict (plus a `migrate_device` receipt), bitwise
    the serial numpy pass. Needs `jax_enable_x64`. Arena-full and census refusals raise
    from the end-of-pass replay, as the pooled migrate's do.

    `timings`, if a dict, accumulates synced wall per phase (see `_Clock`); on
    several cards each key is prefixed with its card.

    `device_budget_bytes` refuses a slab whose estimated device footprint exceeds
    it, per card. A refusal after the first slab has been
    written leaves the state invalid, as the serial pass's mid-pass refusals do.
    `migrate_device["peak_estimate_bytes"]` reports the largest estimate.

    `devices` is a sequence of jax devices, one per card (None: one card, jax's
    default device); `migrate_device` then also reports
    `cards`, `slabs_per_card`, `cross_card_segments` / `_bytes` and `fallback`.

    `comm`, with a node-local `st` (every rank calls this): the pass across ranks
    (`_device_pass`); `st.n_particles` becomes the rank's new count.
    """
    from .. import state as _state

    ctx = _device_pass(st, c_drift, timings=timings, device_budget_bytes=device_budget_bytes,
                       devices=devices, comm=comm)
    rep = _state._replay_arena_pass(
        st, ctx["reach"], ctx["r"], ctx["r_raw"], c_drift, ctx["scales"], n_emig=ctx["n_emig"],
        rr_by_slab=ctx["rr_by_slab"], insert_res=ctx["insert_res"],
        max_staged_slabs=max_staged_slabs, census_note=" (Device pass.)",
        foreign_consumed=ctx["foreign_consumed"])
    clock = ctx["clocks"][0]
    clock.mark("pass: arena replay")
    n_after = _state.occupancy_total(st.occupancy) + st.arena_used
    conserve(st, ctx, n_after, f"migration ({rep['n_inserted']} slabs inserted, arena "
             f"{st.arena_used}/{st.n_arena}, device pass)")
    clock.mark("pass: final census")
    _merge_card_timings(ctx, timings)
    receipt = _pass_receipt(ctx, devices, device_budget_bytes, rep["spill_rows"])
    return _migrate_stats(st, ctx, rep, n_after, receipt)
