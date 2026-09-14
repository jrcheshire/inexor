"""D3b R1: the migrate on the device, slab by slab, bitwise the serial numpy pass.

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

SHAPES. Row buffers use each kernel's `_padded`; the slab window and arena arrays
use `forces.capacity_shape`, so slabs of nearby size share every program.

SCOPE (R1): one device, host-resident state, the serial slab schedule. The
memory-envelope refusal, a permuted schedule and the four-card split are later.
"""

from __future__ import annotations

import numpy as np

INT16_MAX = 32767

_PROGRAMS: dict = {}

#: RECEIPT: passes run through this module (see `eject_jax.CALLS`).
CALLS = 0


def _ladder(n):
    from ..forces import capacity_shape

    return int(capacity_shape(max(1, int(n))))


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
    from .. import eject_jax

    key = (int(t9.n_buckets_side), float(t9.quantum), int(nb), int(cap), has_ids)
    fn = eject_jax._CACHE.get(key)
    if fn is None:
        fn = eject_jax._CACHE[key] = eject_jax._build(t9, int(nb), int(cap), has_ids)
    return fn


def _insert_kernel(p3, nb2, cap, has_ids):
    from .. import insert_jax

    key = (int(p3), int(nb2), int(cap), has_ids)
    fn = insert_jax._CACHE.get(key)
    if fn is None:
        fn = insert_jax._CACHE[key] = insert_jax._build(int(p3), int(nb2), int(cap), has_ids)
    return fn


def _eject_slab(st, s, c_drift, scales, ar_slots, ar_bricks):
    """Upload slab s's window, build its rows and eject them. Returns the staged slab."""
    import jax.numpy as jnp

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
    ar_bucket = np.zeros(a_cap, dtype=np.int64)
    ar_bucket[:n_ar] = np.asarray(st.arena_bucket)[rows_a - int(st.arena_base)]

    def window(src):
        out = np.zeros((w_cap,) + src.shape[1:], dtype=src.dtype)
        out[:span] = src[s0:s1]
        out[span:span + n_ar] = src[rows_a]
        return out

    off_win = jnp.asarray(window(st.off))
    w_win = jnp.asarray(window(st.w))
    ids_win = jnp.asarray(window(st.ids)) if has_ids else None

    from .. import eject_jax

    cap = eject_jax._padded(n_rows)
    rows = _rows_program(cap, w_cap, a_cap, nb, p3, per, has_ids)
    off, bijk, w, ids, scale, brick, real = rows(
        jnp.asarray(occ), jnp.asarray(live), jnp.asarray(np.asarray(
            st.brick_start[lo_b:hi_b], dtype=np.int64) - s0),
        jnp.asarray(row_offsets), jnp.asarray(ar_offsets), jnp.asarray(ar_bucket),
        off_win, w_win, ids_win, jnp.asarray(np.asarray(scales[lo_b:hi_b], dtype=np.float64)),
        jnp.asarray(n_rows, dtype=jnp.int64), jnp.asarray(span, dtype=jnp.int64),
        jnp.asarray(lo_b, dtype=jnp.int64))
    fn = _eject_kernel(st.t9, nb, cap, has_ids)
    dest, off_new, w_out, ids_out, src_out, n_keep = fn(
        off, bijk, w, ids, scale, float(c_drift), brick, real)
    # the row buffers are the kernel's input only; release them before staging
    off = bijk = w = ids = scale = brick = real = None
    n_keep = int(n_keep)

    # realized x-reach over this slab's emigrants, as the serial pass reports it
    rr = 0
    if n_rows > n_keep:
        idx = jnp.arange(cap, dtype=jnp.int64)
        em = (idx >= n_keep) & (idx < n_rows)
        disp = (dest // (p3 * nb2) - s + nb // 2) % nb - nb // 2
        rr = int(jnp.max(jnp.where(em, jnp.abs(disp), 0)))

    return dict(dest=dest, off=off_new, w=w_out, ids=ids_out, src=src_out, cap=cap,
                n_keep=n_keep, n_rows=n_rows, rr=rr,
                win=(off_win, w_win, ids_win), s0=s0, s1=s1, span=span, w_cap=w_cap)


def _insert_slab(st, d, reach, staged, scales_dev):
    """Insert slab d on the device and write its slot range back. Returns insert_res."""
    import jax.numpy as jnp

    from ..layout import _to_index

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
            idx = jnp.arange(e["cap"], dtype=jnp.int64)
            em = (idx >= e["n_keep"]) & (idx < e["n_rows"])
            consumed[s] = int(jnp.sum(em & (e["dest"] // (p3 * nb2) == d)))
            segs.append((e, e["n_keep"], e["n_rows"], False))
    n_in = sum(hi - lo for _e, lo, hi, _k in segs)

    from .. import insert_jax

    cap_i = insert_jax._padded(n_in)
    bufs = (jnp.zeros(cap_i, jnp.int64), jnp.zeros((cap_i, 3), jnp.uint8),
            jnp.zeros((cap_i, 3), jnp.int16),
            jnp.zeros(cap_i, jnp.int32) if has_ids else None, jnp.zeros(cap_i, jnp.int64))
    base = 0
    for e, lo, hi, is_keep in segs:
        src_brick = e["dest"] // p3 if is_keep else e["src"]
        put = _put_program(e["cap"], cap_i, has_ids)
        bufs = put(bufs, e["dest"], e["off"], e["w"], e["ids"], src_brick,
                   jnp.asarray(lo, jnp.int64), jnp.asarray(hi, jnp.int64),
                   jnp.asarray(base, jnp.int64))
        base += hi - lo
    b_dest, b_off, b_w, b_ids, b_src = bufs
    real = jnp.arange(cap_i, dtype=jnp.int64) < n_in
    s_old = jnp.where(real, scales_dev[b_src], 1.0)

    fn = _insert_kernel(p3, nb2, cap_i, has_ids)
    out = fn(b_dest, b_off, b_w, b_ids, s_old, real, jnp.asarray(lo_b, dtype=jnp.int64),
             jnp.asarray(np.asarray(st.brick_start[lo_b:hi_b + 1], dtype=np.int64)),
             jnp.asarray(np.full(nb2, 32767.0, dtype=np.float64)))
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
                                    jnp.asarray(nw, jnp.int64), jnp.asarray(e["s0"], jnp.int64))
    s0, s1, span = e["s0"], e["s1"], e["span"]
    st.off[s0:s1] = np.asarray(off_win)[:span]
    st.w[s0:s1] = np.asarray(w_win)[:span]
    if has_ids:
        st.ids[s0:s1] = np.asarray(ids_win)[:span]
    st.occupancy[lo_b * p3: hi_b * p3] = _to_index(np.asarray(out["occupancy"]),
                                                    st.index_dtype, "migrated")
    st.vel_scale[lo_b:hi_b] = np.asarray(out["scales"])

    spills = []
    if ns:
        sd = np.asarray(out["dest"][nw:nw + ns])
        so = np.asarray(out["off"][nw:nw + ns])
        sw = np.asarray(out["w"][nw:nw + ns])
        si = np.asarray(out["ids"][nw:nw + ns]) if has_ids else None
        sb = sd // p3
        for grp in np.split(np.arange(ns), np.flatnonzero(np.diff(sb)) + 1):
            spills.append((int(sb[grp[0]]), sd[grp], so[grp], sw[grp],
                           None if si is None else si[grp]))
    return dict(consumed=consumed, spills=spills, n_over=ns)


def _emigrants_only(e, has_ids):
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
    bufs = (jnp.zeros(cap_e, jnp.int64), jnp.zeros((cap_e, 3), jnp.uint8),
            jnp.zeros((cap_e, 3), jnp.int16),
            jnp.zeros(cap_e, jnp.int32) if has_ids else None, jnp.zeros(cap_e, jnp.int64))
    put = _put_program(e["cap"], cap_e, has_ids)
    dest, off, w, ids, src = put(bufs, e["dest"], e["off"], e["w"], e["ids"], e["src"],
                                 jnp.asarray(e["n_keep"], jnp.int64),
                                 jnp.asarray(e["n_rows"], jnp.int64), jnp.asarray(0, jnp.int64))
    return dict(dest=dest, off=off, w=w, ids=ids, src=src, cap=cap_e, n_keep=0,
                n_rows=n_emig, rr=e["rr"])


def drift_and_migrate_device(st, c_drift, max_staged_slabs=None):
    """`state.drift_and_migrate` with every slab's row work on the device.

    Same contract, mutations and stats dict (plus a `migrate_device` receipt),
    gated bitwise against the serial numpy pass (`tests/test_migrate_device.py`).
    Needs `jax_enable_x64`. Arena-full and census refusals raise from the end-of-
    pass replay rather than mid-pass, as the pooled migrate's do.
    """
    import jax.numpy as jnp

    from .. import state as _state
    from ..eject_jax import require_x64

    global CALLS
    require_x64()
    CALLS += 1
    nb = int(st.bricks_per_side)
    n_before = int(st.occupancy.astype(np.int64).sum()) + st.arena_used
    scales = np.array(st.vel_scale, dtype=np.float64, copy=True)
    r_raw = _state.brick_reach(st, c_drift, scales)
    r = min(r_raw, nb // 2)
    reach = range(-r, r + 1)
    ar_slots, ar_bricks = pass_arena_index(st)
    scales_dev = jnp.asarray(scales)

    staged, inserted, insert_res, n_emig, rr_by_slab = {}, set(), {}, {}, {}
    for s in range(nb):
        staged[s] = _eject_slab(st, s, c_drift, scales, ar_slots, ar_bricks)
        n_emig[s] = staged[s]["n_rows"] - staged[s]["n_keep"]
        rr_by_slab[s] = staged[s]["rr"]
        for d in range(nb):
            if d in inserted:
                continue
            if all(((d + o) % nb) in staged for o in reach):
                insert_res[d] = _insert_slab(st, d, reach, staged, scales_dev)
                inserted.add(d)
                staged[d] = _emigrants_only(staged[d], st.ids is not None)
        for s2 in list(staged):
            if all(((s2 + o) % nb) in inserted for o in reach):
                del staged[s2]

    rep = _state._replay_arena_pass(
        st, reach, r, r_raw, c_drift, scales, n_emig=n_emig, rr_by_slab=rr_by_slab,
        insert_res=insert_res, max_staged_slabs=max_staged_slabs,
        census_note=" (Device pass.)")
    n_after = int(st.occupancy.astype(np.int64).sum()) + st.arena_used
    if n_after != n_before:
        raise ValueError(
            f"the migration lost {n_before - n_after} particles ({n_before} -> {n_after} "
            f"against {st.n_particles} stored). D-007 forbids dropping, so this is "
            f"corruption, not imprecision. {rep['n_inserted']} of {nb} slabs inserted, "
            f"arena {st.arena_used}/{st.n_arena} (device pass)"
        )
    return dict(n_arena_overflow=rep["n_over"], arena_used=st.arena_used,
                vel_scale=float(np.max(st.vel_scale)),
                vel_scale_min=float(np.min(st.vel_scale)),
                n_migrated_checked=n_after, brick_reach=r, brick_reach_raw=r_raw,
                brick_reach_realized=rep["realized_reach"],
                peak_staged_slabs=rep["peak_staged"],
                migrate_device=dict(slabs=nb, programs=len(_PROGRAMS),
                                    spill_rows=rep["spill_rows"]))
