"""D3b R1b: the repack on the device, slab by slab, bitwise the host `SlotState.repack`.

WHAT MOVES. `SlotState.repack` redistributes brick capacity in two directional
host passes over every row (~6-9 ns per row, projected 430-630 s/step at 4096^3,
record sec. 29). Here each x-slab's old slot range and its arena residents go to
the device once, one program computes every row's destination and scatters the
slab's whole NEW block (spare rows zeroed, ids -1) plus its new occupancy, and
the block comes back as one slice into the new range.

NO SORT. The reference order within a brick is the stable sort of [live run in
bucket order, then arena residents ascending slot] by bucket, and both parts are
already sorted, so every destination is a count:
  live row of rank i in bucket w:  new_start[b] + i + n_res(b, bucket < w)
  resident of rank j among the brick's residents sorted by (bucket, slot):
                                   new_start[b] + occ_cum[b, w] + j
`n_res` and `j` come from ONE searchsorted over the slab's residents sorted by
bucket (a stable argsort of the O(arena) resident list, whose pass order is
already ascending slot within brick).

WHY IT CAN BE BITWISE. Destinations are integer counts, the payload is copied,
and the new occupancy is a bincount -- nothing rounds.

THE HAZARD, handled by staging instead of by pass direction: a block write
into slab s's new range can overlap the OLD range of a later slab, so before
writing slab s every later slab whose old range intersects [new_lo, new_hi) is
uploaded first (a read-ahead window; its peak is on the receipt). Writes go in
ascending slab order, so an earlier slab's old range has always been read. The
arena rows cannot be overrun: the allocation refusal bounds `n_alloc_new` by the
old `arena_base`. Their payload is still lifted once up front, grouped by brick,
so each slab's window is two contiguous copies.

SEVERAL CARDS (R3, `devices=`). Card k owns a contiguous run of slabs and sweeps
them ascending as above, one thread per card; the new ranges are disjoint, so the
writes are. The hazard then crosses cards: a block write can overlap the old range
of a slab ANOTHER card has not read yet. So before any card writes, every slab
whose old range intersects a new range owned by a different card is uploaded by
its own card (`cross_card_early_uploads` on the receipt).
"""

from __future__ import annotations

import numpy as np

from .migrate import _Clock, _ladder, _program, _put, pass_arena_index

#: RECEIPT: passes run through this module.
CALLS = 0


def _repack_program(cap, w_cap, a_cap, out_cap, nb2, p3, has_ids, off_dtype, w_dtype,
                    ids_dtype):
    """One slab's new block from its window: (off, w, ids, new_occ)."""

    def make():
        import jax
        import jax.numpy as jnp

        lift = cap + 1  # above every prefix sum and rank
        nbk = nb2 * p3

        @jax.jit
        def block(occ, live, starts_rel, row_offsets, ar_offsets, ar_bucket, dest_base,
                  off_win, w_win, ids_win, n_rows, span):
            r = jnp.arange(cap, dtype=jnp.int64)
            real = r < n_rows
            bi = jnp.clip(jnp.searchsorted(row_offsets[1:], r, side="right"),
                          0, nb2 - 1).astype(jnp.int64)
            rank = r - row_offsets[bi]
            lc = live[bi]
            is_ar = rank >= lc
            # live rows: bucket = how many of the brick's prefix sums are <= rank
            # (the device decode's lifted searchsorted, O(rows))
            occ_cum = jnp.cumsum(occ.astype(jnp.int64), axis=1)
            keys = (occ_cum + (jnp.arange(nb2, dtype=jnp.int64) * lift)[:, None]).reshape(-1)
            w_live = jnp.searchsorted(keys, bi * lift + rank, side="right").astype(jnp.int64)
            w_live = jnp.clip(w_live - bi * p3, 0, p3 - 1)
            # residents: appended to the window after the slot range, in pass
            # order (grouped by brick, ascending slot). Sorted by bucket ONCE;
            # stable, so ties keep slot order.
            k = jnp.clip(ar_offsets[bi] + rank - lc, 0, a_cap - 1)
            order = jnp.argsort(ar_bucket, stable=True)
            sb = ar_bucket[order]
            rank_of = jnp.zeros((a_cap,), dtype=jnp.int64).at[order].set(
                jnp.arange(a_cap, dtype=jnp.int64))
            w_ar = jnp.clip(ar_bucket[k] - bi * p3, 0, p3 - 1)
            within = jnp.where(is_ar, w_ar, w_live)
            bucket = bi * p3 + within
            slot = jnp.where(is_ar, span + k, starts_rel[bi] + rank)
            slot = jnp.where(real, slot, 0)
            # residents of MY brick in buckets below mine; residents sorted by a
            # brick-major bucket ordinal are grouped by brick, so the count of
            # all residents below my bucket, less my brick's offset, is it
            n_res_lt = jnp.searchsorted(sb, bucket, side="left").astype(jnp.int64) - ar_offsets[bi]
            j = rank_of[k] - ar_offsets[bi]
            occ_cum_w = occ_cum.reshape(-1)[bucket]
            dest = jnp.where(is_ar, dest_base[bi] + occ_cum_w + j,
                             dest_base[bi] + rank + n_res_lt)
            dest = jnp.where(real, dest, out_cap)
            out_off = jnp.zeros((out_cap, 3), dtype=off_dtype).at[dest].set(
                off_win[slot], mode="drop")
            out_w = jnp.zeros((out_cap, 3), dtype=w_dtype).at[dest].set(
                w_win[slot], mode="drop")
            out_ids = None
            if has_ids:
                out_ids = jnp.full((out_cap,), -1, dtype=ids_dtype).at[dest].set(
                    ids_win[slot], mode="drop")
            new_occ = jnp.bincount(jnp.where(real, bucket, nbk), length=nbk + 1)[:-1]
            return out_off, out_w, out_ids, new_occ

        return block

    return _program(("repack", cap, w_cap, a_cap, out_cap, nb2, p3, has_ids,
                     np.dtype(off_dtype).str, np.dtype(w_dtype).str,
                     None if ids_dtype is None else np.dtype(ids_dtype).str), make)


def repack_geometry(st, brick_slack):
    """The host repack's capacity arithmetic, line for line: (run_counts, counts,
    new_start, n_alloc). Duplicated rather than shared so `SlotState.repack`
    stays the untouched oracle; `tests/test_repack_device.py` pins the two equal."""
    p3 = st.buckets_per_brick
    run_counts = st.occupancy.reshape(st.n_bricks, p3).sum(axis=1, dtype=np.int64)
    arena_live = np.nonzero(st.arena_bucket >= 0)[0]
    counts = run_counts.copy()
    if len(arena_live):
        counts += np.bincount(st.arena_bucket[arena_live] // p3, minlength=st.n_bricks)
    new_start, n_alloc = capacity_from_counts(st, counts, brick_slack)
    return run_counts, counts, new_start, n_alloc


def capacity_from_counts(st, counts, brick_slack):
    """`repack_geometry`'s capacity arithmetic from per-brick member counts:
    (new_start, n_alloc), with the index-ceiling and allocation refusals. The fused
    migrate + repack calls it with the destination census before any row moves."""
    _limit = int(np.iinfo(st.index_dtype).max)
    _hot = int(counts.max()) if counts.size else 0
    if _hot > _limit:
        raise ValueError(
            f"repacked brick holds {_hot} rows against the "
            f"{np.dtype(st.index_dtype).name} index ceiling {_limit}: a "
            "bucket in it cannot be stored. Widen index_dtype at build."
        )
    spare = np.ceil(counts * float(brick_slack)).astype(np.int64)
    spare = np.where(counts > 0, np.maximum(spare, 1), spare)
    new_start = np.zeros(st.n_bricks + 1, dtype=np.int64)
    np.cumsum(counts + spare, out=new_start[1:])
    n_alloc = int(new_start[-1])
    if n_alloc + st.n_arena > st.off.shape[0]:
        raise ValueError(
            f"repack needs {n_alloc} slots plus a {st.n_arena}-slot arena against an "
            f"allocation of {st.off.shape[0]}. Raise alloc_margin at build."
        )
    return new_start, n_alloc


def _cross_card_slabs(old_start, new_start, parts, nb2):
    """Per card, the slabs whose OLD range intersects a NEW range another card writes."""
    nb = len(parts) and parts[-1][1]
    owner = np.empty(nb, dtype=np.int64)
    for k, (lo, hi) in enumerate(parts):
        owner[lo:hi] = k
    e = np.arange(nb + 1) * nb2
    o_lo, o_hi = old_start[e[:-1]], old_start[e[1:]]
    n_lo, n_hi = new_start[e[:-1]], new_start[e[1:]]
    early = [[] for _ in parts]
    for t in range(nb):
        if o_hi[t] <= o_lo[t]:
            continue
        hit = (owner != owner[t]) & (n_lo < o_hi[t]) & (o_lo[t] < n_hi) & (n_hi > n_lo)
        if hit.any():
            early[owner[t]].append(t)
    return early


def repack_device(st, brick_slack=0.10, timings=None, devices=None):
    """`SlotState.repack` with every slab's row work on the device.

    Same contract, mutations and return dict (`scratch_bytes` is the HOST bytes
    this driver holds: the lifted arena payload, one slab's window and one
    block), plus a `repack_device` receipt. Gated bitwise against the host
    repack (`tests/test_repack_device.py`, and `tests/test_repack_device_cards.py`
    across cards). Needs `jax_enable_x64`.

    `devices` is a sequence of jax devices, one per card (None: one card, jax's
    default device); see SEVERAL CARDS above.
    """
    from concurrent.futures import ThreadPoolExecutor

    from ..eject_jax import require_x64
    from ..ooc_fft import partition_units

    global CALLS
    require_x64()
    CALLS += 1
    nb, p3 = int(st.bricks_per_side), int(st.buckets_per_brick)
    nb2 = nb * nb
    has_ids = st.ids is not None
    run_counts, counts, new_start, n_alloc = repack_geometry(st, brick_slack)
    devs = [None] if devices is None else list(devices)
    if not devs:
        raise ValueError("devices= was an empty sequence; pass None for one card")
    W = len(devs)
    parts = [(0, nb)] if W == 1 else partition_units(nb, W, 1)
    clock = _Clock(timings if W == 1 else None)

    # the residents' payload, lifted once and grouped by brick
    ar_slots, ar_bricks = pass_arena_index(st)
    a_off = st.off[ar_slots].copy()
    a_w = st.w[ar_slots].copy()
    a_ids = st.ids[ar_slots].copy() if has_ids else None
    a_bucket = st.arena_bucket[ar_slots - int(st.arena_base)]
    a_edge = np.searchsorted(ar_bricks, np.arange(st.n_bricks + 1))
    lifted = a_off.nbytes + a_w.nbytes + (0 if a_ids is None else a_ids.nbytes)
    k_per_brick = np.diff(a_edge)
    n_fast = int(np.count_nonzero((k_per_brick == 0) & (run_counts > 0)))
    n_merge = int(np.count_nonzero(k_per_brick > 0))
    old_start = np.asarray(st.brick_start, dtype=np.int64)
    new_occ = np.zeros(st.n_buckets, dtype=st.index_dtype)
    clock.mark("pass: setup + arena lift")
    programs0 = len(_repack_programs())

    def upload(t, windows, dev, acc):
        lo_b, hi_b = st.slab_bricks(t)
        s0, s1 = int(old_start[lo_b]), int(old_start[hi_b])
        span = s1 - s0
        a0, a1 = int(a_edge[lo_b]), int(a_edge[hi_b])
        n_ar = a1 - a0
        a_cap = _ladder(n_ar)
        w_cap = _ladder(span + a_cap)

        def window(src, src_a):
            out = np.zeros((w_cap,) + src.shape[1:], dtype=src.dtype)
            out[:span] = src[s0:s1]
            out[span:span + n_ar] = src_a[a0:a1]
            return out

        off_np, w_np = window(st.off, a_off), window(st.w, a_w)
        ids_np = window(st.ids, a_ids) if has_ids else None
        acc["scratch"] = max(acc["scratch"], lifted + off_np.nbytes + w_np.nbytes
                             + (0 if ids_np is None else ids_np.nbytes) - (
                                 0 if a_ids is None else a_ids.nbytes))
        off_win, w_win = _put(off_np, dev), _put(w_np, dev)
        ids_win = _put(ids_np, dev) if has_ids else None
        ar_bucket = np.full(a_cap, nb2 * p3, dtype=np.int64)
        ar_bucket[:n_ar] = a_bucket[a0:a1] - lo_b * p3
        windows[t] = dict(off=off_win, w=w_win, ids=ids_win, s0=s0, s1=s1, span=span,
                          n_ar=n_ar, a_cap=a_cap, w_cap=w_cap, ar_bucket=ar_bucket,
                          lo_b=lo_b, hi_b=hi_b)

    def sweep(k, windows, acc):
        lo, hi = parts[k]
        dev = devs[k]
        for s in range(lo, hi):
            if s not in windows:
                upload(s, windows, dev, acc)
            clock.mark("upload: own slab", windows[s]["off"])
            e = windows[s]
            lo_b, hi_b = e["lo_b"], e["hi_b"]
            n_lo, n_hi = int(new_start[lo_b]), int(new_start[hi_b])
            occ = np.asarray(st.occupancy)[lo_b * p3: hi_b * p3].reshape(nb2, p3)
            live = run_counts[lo_b:hi_b]
            ar_counts = k_per_brick[lo_b:hi_b]
            ar_offsets = np.zeros(nb2 + 1, dtype=np.int64)
            np.cumsum(ar_counts, out=ar_offsets[1:])
            row_offsets = np.zeros(nb2 + 1, dtype=np.int64)
            np.cumsum(live + ar_counts, out=row_offsets[1:])
            n_rows = int(row_offsets[-1])
            if n_rows == 0 and n_hi == n_lo:
                # an EMPTY slab: no row to place, no block to write, its occupancy
                # slice already zero -- exactly what the program would produce, at
                # the cost of nothing (a one-slab probe state has 255 of these)
                acc["empty"] += 1
                del windows[s]
                clock.mark("empty slab")
                continue
            cap = _ladder(n_rows)
            out_cap = _ladder(n_hi - n_lo)
            index_dev = (_put(occ, dev), _put(live, dev), _put(old_start[lo_b:hi_b] - e["s0"], dev),
                         _put(row_offsets, dev), _put(ar_offsets, dev),
                         _put(e["ar_bucket"], dev), _put(new_start[lo_b:hi_b] - n_lo, dev))
            clock.mark("upload: index", index_dev)
            block = _repack_program(cap, e["w_cap"], e["a_cap"], out_cap, nb2, p3, has_ids,
                                    st.off.dtype, st.w.dtype,
                                    None if not has_ids else st.ids.dtype)
            out_off, out_w, out_ids, occ_s = block(
                *index_dev, e["off"], e["w"], e["ids"], _put(n_rows, dev, np.int64),
                _put(e["span"], dev, np.int64))
            index_dev = None
            clock.mark("block program", out_off, out_w, out_ids, occ_s)
            # THE hazard: read every later slab of this card this write would overrun
            # (another card's were read before any card wrote)
            t = s + 1
            while t < hi and int(old_start[t * nb2]) < n_hi:
                if t not in windows:
                    upload(t, windows, dev, acc)
                    acc["readahead"] += 1
                t += 1
            acc["peak_windows"] = max(acc["peak_windows"], len(windows))
            acc["held_peak"] = max(acc["held_peak"], sum(
                int(v["off"].nbytes) + int(v["w"].nbytes)
                + (0 if v["ids"] is None else int(v["ids"].nbytes))
                for v in windows.values()) + int(out_off.nbytes) + int(out_w.nbytes)
                + (0 if out_ids is None else int(out_ids.nbytes)))
            clock.mark("upload: read-ahead", *[windows[u]["off"] for u in windows])
            m = n_hi - n_lo
            st.off[n_lo:n_hi] = np.asarray(out_off)[:m]
            st.w[n_lo:n_hi] = np.asarray(out_w)[:m]
            if has_ids:
                st.ids[n_lo:n_hi] = np.asarray(out_ids)[:m]
            new_occ[lo_b * p3: hi_b * p3] = np.asarray(occ_s)
            acc["scratch"] = max(acc["scratch"], lifted + m * (
                st.off.itemsize * 3 + st.w.itemsize * 3 + (st.ids.itemsize if has_ids else 0)))
            out_off = out_w = out_ids = occ_s = None
            del windows[s]
            clock.mark("write-back")

    wins = [dict() for _ in range(W)]
    accs = [dict(scratch=lifted, readahead=0, empty=0, peak_windows=0, held_peak=0)
            for _ in range(W)]
    early = [[] for _ in range(W)]

    def run(fn):
        if W == 1:
            return [fn(0)]
        with ThreadPoolExecutor(max_workers=W) as ex:
            return list(ex.map(fn, range(W)))

    if W > 1:
        early = _cross_card_slabs(old_start, new_start, parts, nb2)
        run(lambda k: [upload(t, wins[k], devs[k], accs[k]) for t in early[k]])
    run(lambda k: sweep(k, wins[k], accs[k]))

    # everything past the new allocation, arena included, reads empty
    st.off[n_alloc:] = 0
    st.w[n_alloc:] = 0
    if has_ids:
        st.ids[n_alloc:] = -1
    # CONTENTS, not bindings (the pool shares these arrays), as the host does
    st.brick_start[...] = new_start
    st.occupancy[...] = new_occ
    st.arena_base = n_alloc
    st.arena_bucket[:] = -1
    st._invalidate_arena_index()
    clock.mark("pass: tail")
    receipt = dict(slabs=nb, programs=len(_repack_programs()) - programs0,
                   windows_peak=int(max(a["peak_windows"] for a in accs)),
                   readahead_uploads=int(sum(a["readahead"] for a in accs)),
                   empty_slabs=int(sum(a["empty"] for a in accs)),
                   held_peak_bytes=int(max(a["held_peak"] for a in accs)))
    if devices is not None:
        receipt.update(cards=W, cross_card_early_uploads=int(sum(len(e) for e in early)))
    return dict(
        slots_used=n_alloc,
        slots_per_particle=n_alloc / max(st.n_particles, 1),
        scratch_bytes=int(max(a["scratch"] for a in accs)),
        bricks_fast=n_fast,
        bricks_merged=n_merge,
        repack_device=receipt,
    )


def _repack_programs():
    from . import migrate

    return {k: v for k, v in migrate._PROGRAMS.items() if k[0] == "repack"}
