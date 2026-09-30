"""T9 state stored in slot order: the engine's particle container.

Unlike `layout.BrickPackedLayout` (an index into caller-held arrays, costing ~21 B/p of
per-particle bookkeeping), `SlotState` stores the payload itself in slot order, so a
particle's bucket is implied by where it sits and no per-particle key or permutation exists.

Buckets are ordered brick-major. `occupancy` (uint32 per bucket) doubles as the bucket
boundaries via a prefix sum within a brick: a slot's brick is the `brick_start` run it falls
in, its bucket is `searchsorted(cumsum(occupancy[brick]), slot - lo)`, and its stored `off` is
relative to that bucket. A brick's live rows are its first `sum(occupancy[brick])` slots and
the rest is spare, so free slots need no sentinel. Overflow goes to a small arena whose rows
record their bucket in `arena_bucket` (-1 = free); arena residents still belong to their brick.

Wrap-never-clamp: no saturating op touches integer state; overflow escalates to spare, then
arena, then a loud refusal.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .codec import (
    INT16_MAX,
    LEVELS_PER_BUCKET,
    assert_int16_range,
    refuse_ids_above_int32,
)
from .layout import (
    DEFAULT_INDEX_DTYPE,
    _stable_sort_index,
    _to_index,
    _within_run_index,
    bucket_order_key,
)

__all__ = [
    "SlotState",
    "decode_positions_host",
    "drift_and_migrate",
    "encode_positions_host",
    "reconcile_velocity_scale",
    "require_whole",
]


# Host numpy mirrors of the jnp `codec.encode_positions` / `decode_positions` (the layout and
# exchange never put the global position array on the device). `tests/test_slot_state.py`
# asserts they are bitwise the codec's.


def encode_positions_host(x, t9):
    """Physical positions -> (uint8 offsets, int64 bucket ijk). Mirrors
    `codec.encode_positions`; the wrap is taken in the integer domain, where it is exact."""
    i = np.mod(np.rint(np.asarray(x, dtype=np.float64) / t9.quantum).astype(np.int64), t9.n_levels)
    b = i // LEVELS_PER_BUCKET
    return (i - b * LEVELS_PER_BUCKET).astype(np.uint8), b


def decode_positions_host(off, bucket_ijk, t9):
    """(offsets, bucket ijk) -> physical positions. Mirrors `codec.decode_positions`
    including its expression shape (global lattice index, then one multiply by the quantum),
    which is what makes the two bitwise equal by construction."""
    i = np.asarray(bucket_ijk, dtype=np.int64) * LEVELS_PER_BUCKET + np.asarray(
        off, dtype=np.int64
    )
    return i.astype(np.float64) * t9.quantum


def _alloc_geometry(brick_counts, n_particles, brick_slack, alloc_margin, arena_frac):
    """Capacity arithmetic: (spare, brick_start, n_alloc, n_arena).

    Spare is pooled per brick (at least one slot per non-empty brick). Shared by
    `SlotState.build` and the streamed loader so their geometry is bitwise identical.
    """
    spare = np.ceil(brick_counts * float(brick_slack)).astype(np.int64)
    spare = np.where(brick_counts > 0, np.maximum(spare, 1), spare)
    brick_start = np.zeros(len(brick_counts) + 1, dtype=np.int64)
    np.cumsum(brick_counts + spare, out=brick_start[1:])
    n_alloc = int(np.ceil(int(brick_start[-1]) * (1.0 + float(alloc_margin))))
    n_arena = int(np.ceil(n_particles * float(arena_frac)))
    return spare, brick_start, n_alloc, n_arena


def _bucket_flat_brick_major(bucket_ijk, t9, bricks_per_side):
    """Per-axis bucket -> brick-major flat ordinal (the tail of `layout.bucket_order_key`)."""
    per = t9.n_buckets_side // int(bricks_per_side)
    b = np.asarray(bucket_ijk, dtype=np.int64)
    brick = b // per
    within = b - brick * per
    bf = (brick[:, 0] * bricks_per_side + brick[:, 1]) * bricks_per_side + brick[:, 2]
    wf = (within[:, 0] * per + within[:, 1]) * per + within[:, 2]
    return bf * (per**3) + wf


def _rescale_w(w, s_old, s_new):
    """Re-express int16 velocity codes at a new scale; `s_old`/`s_new` are scalars or (n,).

    Returns `w` unchanged when both are equal scalars; otherwise decodes and re-rounds.
    With per-brick scales the result can escape int16 (a fast immigrant into a quiet brick),
    so the caller must pass an `s_new` covering every row (`_insert_slab` does); an escape
    raises rather than clamping.
    """
    s_old = np.asarray(s_old, dtype=np.float64)
    s_new = np.asarray(s_new, dtype=np.float64)
    if s_old.shape == () and s_new.shape == () and s_old == s_new:
        return w
    w = np.asarray(w, dtype=np.int16)
    if not len(w):
        return w
    ratio = np.divide(
        s_old, s_new, out=np.zeros(np.broadcast(s_old, s_new).shape), where=s_new > 0.0
    )
    if ratio.ndim:
        ratio = ratio[:, None]
    out = np.rint(w.astype(np.float64) * ratio)
    if np.abs(out).max(initial=0.0) > INT16_MAX:
        raise ValueError(
            f"velocity code {np.abs(out).max():.0f} escapes int16 under a rescale to a "
            "scale that does not cover it. Per-brick scales make this reachable where a "
            "global scale made it impossible; the caller must fix the destination scale "
            "over the rows it is about to write. Integer state is never clamped."
        )
    return out.astype(np.int16)


def _scales_from_sorted(absv_sorted, brick_counts):
    """Per-brick velocity scale, max|v|_inf / INT16_MAX, from a brick-major-sorted column.

    `reduceat` is given only non-empty bricks' starts, so it never meets equal consecutive
    indices (where it returns the element instead of the identity). Empty or at-rest bricks
    get 1.0, not 0.0, which would divide by zero on decode.
    """
    n_bricks = len(brick_counts)
    s = np.zeros(n_bricks, dtype=np.float64)
    starts = np.zeros(n_bricks, dtype=np.int64)
    np.cumsum(brick_counts[:-1], out=starts[1:])
    nz = brick_counts > 0
    if len(absv_sorted) and nz.any():
        s[nz] = np.maximum.reduceat(np.asarray(absv_sorted, dtype=np.float64), starts[nz])
    s /= INT16_MAX
    return np.where(s > 0.0, s, 1.0)


def _encode_at(v, scales_per_row):
    """Quantize float velocities at a per-row scale. Refuses rather than clips on escape."""
    v = np.asarray(v, dtype=np.float64)
    if not len(v):
        return np.zeros((0, 3), dtype=np.int16)
    w = np.rint(v / np.asarray(scales_per_row, dtype=np.float64)[:, None])
    assert_int16_range(w)
    return w.astype(np.int16)


def _cat(dest, off, w, ids, src=None):
    out = dict(
        dest=np.concatenate(dest) if dest else np.empty(0, np.int64),
        off=np.concatenate(off) if off else np.empty((0, 3), np.uint8),
        w=np.concatenate(w) if w else np.empty((0, 3), np.int16),
    )
    # Source brick, on emigrants only: their codes are at the source brick's scale. A keeper's
    # source is `dest // buckets_per_brick`.
    if src is not None:
        out["src"] = np.concatenate(src) if src else np.empty(0, np.int32)
    out["ids"] = np.concatenate(ids) if ids and ids[0] is not None else None
    return out


def _cat_dicts(ds):
    ds = [d for d in ds if len(d["dest"])]
    if not ds:
        return dict(
            dest=np.empty(0, np.int64),
            off=np.empty((0, 3), np.uint8),
            w=np.empty((0, 3), np.int16),
            ids=None,
            src=np.empty(0, np.int32),
        )
    has_ids = ds[0].get("ids") is not None
    has_src = ds[0].get("src") is not None
    return dict(
        dest=np.concatenate([d["dest"] for d in ds]),
        off=np.concatenate([d["off"] for d in ds]),
        w=np.concatenate([d["w"] for d in ds]),
        ids=np.concatenate([d["ids"] for d in ds]) if has_ids else None,
        src=np.concatenate([d["src"] for d in ds]) if has_src else None,
    )


def _stable_order(key, n_values):
    """`argsort(kind="stable")`, narrowed to uint8/uint16 so numpy takes its radix path.

    numpy's stable sort is radix only for 1- and 2-byte integers. The narrowing is
    order-preserving for non-negative in-range keys, so the permutation is identical to the
    wide sort's (downstream encodes are order-dependent). The range is checked because numpy
    narrows modularly; out-of-range keys fall back to the wide sort.
    """
    key = np.asarray(key)
    if not len(key):
        return np.argsort(key, kind="stable")
    hi = int(key.max())
    lo = int(key.min())
    if lo >= 0 and hi < min(int(n_values), np.iinfo(np.uint16).max + 1):
        narrow = np.uint8 if hi <= np.iinfo(np.uint8).max else np.uint16
        return np.argsort(key.astype(narrow), kind="stable")
    return np.argsort(key, kind="stable")


def _group_by_brick(brick_of_row, lo_b, hi_b):
    """Rows grouped by destination brick in `[lo_b, hi_b)`: (permutation, CSR offsets).

    One pass instead of a per-brick mask (which is O(N nb^2) per step). Rows outside the
    range are dropped, not refused: the immigrant buffer holds emigrants bound for other
    slabs too. Order within a brick is preserved (stable sort), which downstream encodes need.
    """
    n_b = int(hi_b) - int(lo_b)
    off = np.zeros(n_b + 1, dtype=np.int64)
    brick_of_row = np.asarray(brick_of_row, dtype=np.int64)
    if not len(brick_of_row) or n_b <= 0:
        return np.empty(0, dtype=np.int64), off
    within = brick_of_row - int(lo_b)
    idx = np.flatnonzero((within >= 0) & (within < n_b))
    if not len(idx):
        return np.empty(0, dtype=np.int64), off
    w = within[idx]
    np.cumsum(np.bincount(w, minlength=n_b), out=off[1:])
    # uint16 where it fits, for numpy's radix path (see `_stable_order`)
    key = w.astype(np.uint16) if n_b <= np.iinfo(np.uint16).max else w
    return idx[np.argsort(key, kind="stable")], off


def reconcile_velocity_scale(tile_scales):
    """Global velocity scale as the max of per-tile scales (1.0 if none are positive).

    Tile ownership is a partition, so the max over tiles is exactly the global max|v|/32767
    without an O(N) velocity buffer. Re-expressing a tile's codes at this scale is a second
    rounding: RMS grows by sqrt(1 + r^2), r = s_tile / s_global <= 1.
    """
    s = np.asarray(list(tile_scales), dtype=np.float64)
    s = s[s > 0.0]
    return float(s.max()) if len(s) else 1.0


def brick_reach(st, c_drift, vel_scale=None):
    """Upper bound on the bricks a particle can cross per axis during this drift. O(1).

    |v| <= max(vel_scale) * INT16_MAX, and each brick's scale is set by its fastest
    particle, so the bound is nearly tight. It is global because the slab schedule is.
    Peak migration staging is `2 * reach + 1` slabs. `vel_scale` overrides `st.vel_scale`.
    """
    s = float(np.max(st.vel_scale if vel_scale is None else vel_scale))
    nb = int(st.bricks_per_side)
    extent = float(st.t9.box_size) / nb
    if extent <= 0.0:
        return nb
    return int(np.ceil(abs(float(c_drift)) * s * INT16_MAX / extent))


def occupancy_total(occ):
    """Rows counted by a per-bucket occupancy index, as a Python int.

    Accumulates in int64 without materializing a widened copy of the index (68.7 GB at
    4096^3). Every whole-index count in a step goes through here.
    """
    return int(np.sum(occ, dtype=np.int64))


def drift_and_migrate(st, c_drift, max_staged_slabs=None, kernel="numpy", insert_kernel="numpy"):
    """Advance every particle by `c_drift * v` and re-home it, in one slab-scheduled pass.

    Drift and migration are inseparable once positions are bucket-relative. Every slab is
    ejected (read, leavers removed) before any slab that it can reach is inserted, so a
    destination's capacity includes its own departures. Staging holds `2 * reach + 1` slabs,
    with `reach = brick_reach(...)` clamped to nb // 2 (all-to-all, still correct);
    `max_staged_slabs` refuses a pass that exceeds a memory budget. A particle-count census
    raises at the point of any loss.

    `kernel` / `insert_kernel` select numpy or jax for `_eject_slab` / `_insert_slab`. Both
    default to "numpy" (unlike `EngineConfig`, which the engine always passes) because
    `tests/test_eject_jax.py` uses the bare call as its numpy reference arm.

    Returns a stats dict (arena overflow, scales, reach bound and realized, peak staging).
    """
    require_whole(st, "drift_and_migrate")
    nb = st.bricks_per_side
    # snapshot: inserts rewrite brick scales while later ejects must decode at pre-pass scales
    scales = np.array(st.vel_scale, dtype=np.float64, copy=True)
    n_before = occupancy_total(st.occupancy) + st.arena_used
    r_raw = brick_reach(st, c_drift, scales)
    # clamp, not refuse: beyond nb // 2 the periodic schedule is already all-to-all
    r = min(r_raw, nb // 2)
    reach = range(-r, r + 1)

    staged, emig, inserted = {}, {}, set()
    consumed = {}  # emig rows an insert actually took, per source slab
    n_over, peak_staged, realized_reach = 0, 0, 0
    for s in range(nb):
        staged[s], emig[s] = st._eject_slab(s, c_drift, scales, kernel=kernel)
        consumed[s] = 0
        # the realized x-reach, reported beside the bound
        if len(emig[s]["dest"]):
            d_slab = emig[s]["dest"] // (st.buckets_per_brick * nb * nb)
            disp = (d_slab - s + nb // 2) % nb - nb // 2
            realized_reach = max(realized_reach, int(np.abs(disp).max()))
        # a slab may be written once every slab that can REACH it has been ejected
        for d in range(nb):
            if d in inserted:
                continue
            if all(((d + o) % nb) in emig for o in reach):
                n_over += st._insert_slab(d, staged, emig, reach, consumed, scales=scales,
                                          kernel=insert_kernel)
                inserted.add(d)
        # release what no pending write can still need
        for s2 in list(staged):
            if s2 in inserted:
                del staged[s2]
        for s2 in list(emig):
            if all(((s2 + o) % nb) in inserted for o in reach):
                # census: releasing an emig slab with unconsumed rows would destroy particles
                n_rows = len(emig[s2]["dest"])
                if consumed[s2] != n_rows:
                    d_slab = (emig[s2]["dest"] // (st.buckets_per_brick * nb * nb))
                    disp = (d_slab - s2 + nb // 2) % nb - nb // 2
                    hist = {int(k): int(c) for k, c in zip(*np.unique(disp, return_counts=True))}
                    raise AssertionError(
                        f"releasing emig slab {s2} with {n_rows - consumed[s2]} of "
                        f"{n_rows} rows unconsumed (reach {r}, consumption offsets "
                        f"{sorted({int(o) for o in reach})}). Destination-slab "
                        f"displacement histogram for this slab's emigrants: {hist}. "
                        "Particles are never dropped; an unconsumed emigrant is a particle "
                        "about to be destroyed."
                    )
                del emig[s2]
        peak_staged = max(peak_staged, len(staged))
        if max_staged_slabs is not None and peak_staged > int(max_staged_slabs):
            raise ValueError(
                f"the migration is holding {peak_staged} slabs against a budget of "
                f"{max_staged_slabs}. The drift reaches {r_raw} bricks on a "
                f"{nb}-brick grid, so {2 * r + 1} slabs must be in flight.\n"
                f"  c_drift={c_drift:.6g}, max vel_scale={float(np.max(scales)):.6g}, "
                f"max |dx| = {abs(float(c_drift)) * float(np.max(scales)) * INT16_MAX:.6g} against a "
                f"brick of {float(st.t9.box_size) / nb:.6g}.\n"
                "  Reduce the step size, use a coarser brick, or raise the budget "
                "deliberately -- staging is bounded by (2 * reach + 1) slabs, so "
                "this is a real memory cost and not a formality."
            )
    if len(inserted) != nb:
        raise AssertionError(f"{nb - len(inserted)} slabs were never written back")
    n_after = occupancy_total(st.occupancy) + st.arena_used
    if n_after != n_before:
        left = sum(len(v.get("dest", ())) for v in staged.values()) if staged else 0
        raise ValueError(
            f"the migration lost {n_before - n_after} particles ({n_before} -> "
            f"{n_after} against {st.n_particles} stored). Particles are never dropped, "
            "so this is corruption, not imprecision.\n"
            f"  {len(staged)} slabs still staged at the end ({left} rows), "
            f"{len(inserted)} of {nb} slabs inserted, arena {st.arena_used}/"
            f"{st.n_arena}\n"
            "  LEADING CAUSE: a particle moved farther than the computed brick "
            "reach, or an insert did not consume every staged emigrant; compare "
            "brick_reach with brick_reach_realized in the step stats."
        )
    # every brick's scale was already fixed by its own insert
    return dict(n_arena_overflow=n_over, arena_used=st.arena_used,
                vel_scale=float(np.max(st.vel_scale)),
                vel_scale_min=float(np.min(st.vel_scale)),
                n_migrated_checked=n_after, brick_reach=r, brick_reach_raw=r_raw,
                brick_reach_realized=realized_reach, peak_staged_slabs=peak_staged)


def _replay_arena_pass(st, reach, r, r_raw, c_drift, scales, n_emig, rr_by_slab, insert_res,
                       max_staged_slabs=None, census_note=""):
    """Replay a migrate pass's arena mutations and bookkeeping in serial order.

    For a pass whose ejects and inserts ran out of order or elsewhere, with arena mutations
    deferred: releases (`_release_brick_arena`) at each slab's eject point, claims
    (`_to_arena`, spilled bricks ascending) at each insert point, and the census and staging
    accounting at the points `drift_and_migrate` runs them. Shared by the pooled and device
    migrates (`device/migrate.py`); both are gated bitwise against the serial pass.

    `n_emig[s]` is slab s's emigrant row count; `rr_by_slab[s]` its realized
    reach; `insert_res[d]` a dict with `consumed` (source slab -> rows taken),
    `spills` ((brick, dest, off, w, ids) in ascending brick order) and `n_over`.
    `census_note` is appended to the census failure message.

    Returns `n_over`, `peak_staged`, `realized_reach`, `spill_rows`, `spill_bytes`.
    """
    nb = st.bricks_per_side
    staged_sym, emig_sym, inserted = set(), set(), set()
    consumed = {}
    n_over, peak_staged, realized_reach = 0, 0, 0
    spill_rows = spill_bytes = 0
    for s in range(nb):
        lo_b, hi_b = st.slab_bricks(s)
        for b in range(lo_b, hi_b):
            st._release_brick_arena(b)
        staged_sym.add(s)
        emig_sym.add(s)
        consumed[s] = 0
        realized_reach = max(realized_reach, rr_by_slab[s])
        for d in range(nb):
            if d in inserted:
                continue
            if all(((d + o) % nb) in emig_sym for o in reach):
                res = insert_res[d]
                for src, c in res["consumed"].items():
                    consumed[src] += int(c)
                for _b, dest_r, off_r, w_r, ids_r in res["spills"]:
                    spill_rows += len(dest_r)
                    spill_bytes += dest_r.nbytes + off_r.nbytes + w_r.nbytes
                    spill_bytes += 0 if ids_r is None else ids_r.nbytes
                    st._to_arena(dest_r, off_r, w_r, ids_r)
                n_over += int(res["n_over"])
                inserted.add(d)
        for s2 in list(staged_sym):
            if s2 in inserted:
                staged_sym.discard(s2)
        for s2 in list(emig_sym):
            if all(((s2 + o) % nb) in inserted for o in reach):
                n_rows = n_emig[s2]
                if consumed[s2] != n_rows:
                    raise AssertionError(
                        f"releasing emig slab {s2} with {n_rows - consumed[s2]} of "
                        f"{n_rows} rows unconsumed (reach {r}, consumption offsets "
                        f"{sorted({int(o) for o in reach})}). Particles are never "
                        "dropped; an unconsumed emigrant is a particle about to "
                        "be destroyed." + census_note
                    )
                emig_sym.discard(s2)
        peak_staged = max(peak_staged, len(staged_sym))
        if max_staged_slabs is not None and peak_staged > int(max_staged_slabs):
            raise ValueError(
                f"the migration is holding {peak_staged} slabs against a budget of "
                f"{max_staged_slabs}. The drift reaches {r_raw} bricks on a "
                f"{nb}-brick grid, so {2 * r + 1} slabs must be in flight.\n"
                f"  c_drift={c_drift:.6g}, max vel_scale={float(np.max(scales)):.6g}, "
                f"max |dx| = {abs(float(c_drift)) * float(np.max(scales)) * INT16_MAX:.6g} against a "
                f"brick of {float(st.t9.box_size) / nb:.6g}.\n"
                "  Reduce the step size, use a coarser brick, or raise the budget "
                "deliberately -- staging is bounded by (2 * reach + 1) slabs, so "
                "this is a real memory cost and not a formality."
            )
    if len(inserted) != nb:
        raise AssertionError(f"{nb - len(inserted)} slabs were never written back")
    return dict(n_over=n_over, peak_staged=peak_staged, realized_reach=realized_reach,
                spill_rows=spill_rows, spill_bytes=spill_bytes, n_inserted=len(inserted))


def drift_and_migrate_pooled(st, c_drift, pool, kernel="numpy", window=None,
                             max_staged_slabs=None, eject_inflight=None):
    """`drift_and_migrate` on the worker pool, bitwise the serial pass.

    Workers eject and insert whole slabs, writing brick payloads straight into shared
    memory (disjoint per brick). Every order-dependent arena mutation is deferred: ejects
    run with `release_arena=False`, inserts return overflow via `spill_sink`, and the parent
    replays releases and claims in serial order (`_replay_arena_pass`). No pass reads an arena
    row before its own schedule point, so the deferral is invisible and the stats dict equals
    the serial one key for key; pool numbers go in `migrate_pool`. Differences from serial:
    an arena-full refusal raises at replay time, and the census omits the displacement
    histogram. Falls back to the serial pass when the reach is all-to-all.

    `window`: scratch slots in flight, floor 4r+2 (the wrap pins ~2r and the sliding span
    2r+2; below that the loop deadlocks); default workers + 4r+2.
    `eject_inflight`: max concurrent ejects (>= 1, never deadlocks). An eject materializes
    its whole slab privately (~35-129 B/row) while an insert reads scratch views, so
    bounding ejects caps host memory without shrinking the pool.
    """
    if eject_inflight is not None and int(eject_inflight) < 1:
        raise ValueError(
            f"eject_inflight must be >= 1, got {eject_inflight}: at zero no "
            "eject can ever launch and the dispatch loop blocks forever"
        )
    require_whole(st, "drift_and_migrate_pooled")
    nb = st.bricks_per_side
    p3 = st.buckets_per_brick
    n_before = occupancy_total(st.occupancy) + st.arena_used
    scales = np.array(st.vel_scale, dtype=np.float64, copy=True)
    r_raw = brick_reach(st, c_drift, scales)
    r = min(r_raw, nb // 2)
    reach = range(-r, r + 1)
    if 2 * r + 1 >= nb:
        out = drift_and_migrate(st, c_drift, max_staged_slabs=max_staged_slabs, kernel=kernel)
        out["migrate_pool"] = dict(workers=0, fallback="all-to-all reach")
        return out
    k_min = min(nb, 4 * r + 2)
    if window is not None and int(window) < k_min:
        raise ValueError(
            f"migrate window {window} is below the deadlock floor {k_min} "
            f"(reach {r}: the wrap pins ~{2 * r} slots and the sliding span "
            f"holds {2 * r + 2})"
        )
    K = min(nb, int(window) if window is not None else int(pool.workers) + 4 * r + 2)

    # worst-case eject rows per slab = live rows + arena residents, both from
    # the pre-pass state the workers will read
    occ_slab = st.occupancy.reshape(nb, -1).sum(axis=1, dtype=np.int64)
    arena_slab = np.zeros(nb, dtype=np.int64)
    if st.n_arena:
        keys = st.arena_bucket[st.arena_bucket >= 0]
        if len(keys):
            arena_slab = np.bincount((keys // p3) // (nb * nb), minlength=nb)
    slot_rows = int((occ_slab + arena_slab).max())
    pool.stage_migrate(c_drift, kernel, r, slot_rows, K)

    # backpressure: an eject needs a free slot; a slab's slot frees once every destination
    # its emig can feed has been inserted
    free_slots = list(range(K))
    e_cap = nb if eject_inflight is None else min(int(eject_inflight), nb)
    e_inflight = 0
    slot_of = {}
    ejected = {}  # s -> (slot, n_keep, n_emig)
    rr_by_slab = {}
    insert_res = {}
    dispatched = set()
    eject_busy = insert_busy = 0.0
    eject_jax_calls = 0
    next_eject = 0
    while len(insert_res) < nb:
        while next_eject < nb and free_slots and e_inflight < e_cap:
            slot = free_slots.pop()
            slot_of[next_eject] = slot
            pool.submit_eject(next_eject, slot)
            e_inflight += 1
            next_eject += 1
        res = pool.next_migrate_result()
        if res["kind"] == "eject":
            e_inflight -= 1
            s = res["s"]
            ejected[s] = (res["slot"], res["n_keep"], res["n_emig"])
            rr_by_slab[s] = int(res["realized_reach"])
            eject_busy += res["busy_s"]
            eject_jax_calls += int(res.get("eject_jax_calls", 0))
            for d in range(nb):
                if d in dispatched:
                    continue
                srcs = sorted({(d + o) % nb for o in reach})
                if all(sv in ejected for sv in srcs):
                    pool.submit_insert(d, [(sv,) + ejected[sv] for sv in srcs])
                    dispatched.add(d)
        else:
            insert_res[res["d"]] = res
            insert_busy += res["busy_s"]
            for s2 in list(slot_of):
                if all(((s2 + o) % nb) in insert_res for o in reach):
                    free_slots.append(slot_of.pop(s2))
    assert len(ejected) == nb, f"{nb - len(ejected)} slabs were never ejected"

    rep = _replay_arena_pass(
        st, reach, r, r_raw, c_drift, scales,
        n_emig={s: ejected[s][2] for s in ejected}, rr_by_slab=rr_by_slab,
        insert_res=insert_res, max_staged_slabs=max_staged_slabs,
        census_note=(" (Pooled pass: the displacement histogram the serial census "
                     "prints needs rows whose scratch slot may be reused -- re-run "
                     "serial for the full census.)"),
    )
    n_over, peak_staged = rep["n_over"], rep["peak_staged"]
    realized_reach = rep["realized_reach"]
    spill_rows, spill_bytes = rep["spill_rows"], rep["spill_bytes"]
    n_after = occupancy_total(st.occupancy) + st.arena_used
    if n_after != n_before:
        raise ValueError(
            f"the migration lost {n_before - n_after} particles ({n_before} -> "
            f"{n_after} against {st.n_particles} stored). Particles are never dropped, "
            "so this is corruption, not imprecision.\n"
            f"  {rep['n_inserted']} of {nb} slabs inserted, arena {st.arena_used}/"
            f"{st.n_arena} (pooled pass)\n"
            "  LEADING CAUSE: a particle moved farther than the computed brick "
            "reach, or an insert did not consume every staged emigrant; compare "
            "brick_reach with brick_reach_realized in the step stats."
        )
    row_b = 8 + 3 + 6 + 4 + (4 if st.ids is not None else 0)
    return dict(n_arena_overflow=n_over, arena_used=st.arena_used,
                vel_scale=float(np.max(st.vel_scale)),
                vel_scale_min=float(np.min(st.vel_scale)),
                n_migrated_checked=n_after, brick_reach=r, brick_reach_raw=r_raw,
                brick_reach_realized=realized_reach, peak_staged_slabs=peak_staged,
                migrate_pool=dict(workers=int(pool.workers), window=K,
                                  slot_rows=slot_rows,
                                  scratch_mb=K * slot_rows * row_b / 1e6,
                                  spill_rows=spill_rows, spill_bytes=spill_bytes,
                                  eject_busy_s=eject_busy, insert_busy_s=insert_busy,
                                  # summed from the workers; the parent's counter
                                  # cannot see them
                                  eject_jax_calls=eject_jax_calls))


# ===========================================================================
# the container
# ===========================================================================


def tile_brick_ids(tijk, n_tile, b_fine, n_brick, nb):
    """The bricks covering tile `tijk` plus its buffer on an nb^3 brick grid, as int64.

    Order is x outer, z inner, each axis ascending from the tile's low buffer edge (mod nb); the
    decode, and so every row index downstream, follows it.
    """
    from .layout import brick_span

    pad, span = brick_span(n_tile, b_fine, n_brick, nb)
    lo = np.asarray(tijk, dtype=np.int64) * (int(n_tile) // int(n_brick)) - pad
    bi, bj, bk = ((int(lo[a]) + np.arange(span, dtype=np.int64)) % nb for a in range(3))
    return ((bi[:, None, None] * nb + bj[None, :, None]) * nb + bk[None, None, :]).reshape(-1)


def tile_window_counts(grid, n_tile, b_fine, n_brick, planes=None):
    """Per-tile sums of a per-brick (nb, nb, nb) int grid over each tile's `tile_brick_ids`:
    an int64 (planes, s, s) array for tile x-planes `planes` = [lo, hi) (default all).

    Periodic window sums by wrap-extended cumulative sums along each axis; exact.
    """
    from .layout import brick_span

    g = np.asarray(grid, dtype=np.int64)
    nb = g.shape[0]
    pad, span = brick_span(n_tile, b_fine, n_brick, nb)
    bpt = int(n_tile) // int(n_brick)
    s = nb // bpt
    p_lo, p_hi = (0, s) if planes is None else (int(planes[0]), int(planes[1]))

    def window(a, axis, tiles):
        ext = np.concatenate([a, np.take(a, np.arange(span), axis=axis)], axis=axis)
        c = np.cumsum(ext, axis=axis, dtype=np.int64)
        c = np.concatenate([np.zeros_like(np.take(c, [0], axis=axis)), c], axis=axis)
        start = (np.asarray(tiles, dtype=np.int64) * bpt - pad) % nb
        return np.take(c, start + span, axis=axis) - np.take(c, start, axis=axis)

    return window(window(window(g, 0, range(p_lo, p_hi)), 1, range(s)), 2, range(s))


class TileMembers(Mapping):
    """Tile -> its bricks (`SlotState.tile_bricks`), built on access, not held.

    Keys are the tiles of tile x-planes `planes` = [lo, hi) (default all), in `cfg.tiles`
    order: one rank's tiles under `decomp.Decomp`.
    """

    def __init__(self, st, n_tile, b_fine, n_brick, n_fine, planes=None):
        s = int(n_fine) // int(n_tile)
        lo, hi = (0, s) if planes is None else (int(planes[0]), int(planes[1]))
        self._st = st
        self._geom = (n_tile, b_fine, n_brick, n_fine)
        self._keys = [(i, j, k) for i in range(lo, hi) for j in range(s) for k in range(s)]
        self._set = frozenset(self._keys)

    def __getitem__(self, t):
        t = tuple(int(q) for q in t)
        if t not in self._set:
            raise KeyError(t)
        return self._st.tile_bricks(t, *self._geom)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)


def require_whole(st, what):
    """Refuse a node-local state (`SlotState.slabs` set) in a pass that has no cross-rank path."""
    if not st.is_whole:
        lo, hi = st.owned_slabs
        raise NotImplementedError(
            f"{what} needs a whole-box state; this one holds brick slabs [{lo}, {hi}) of "
            f"{st.bricks_per_side} (one rank's share), and {what} has no cross-rank path"
        )


@dataclass
class SlotState:
    """T9 payload stored in slot order; the bucket is implied by the slot.

    A node-local state (`slabs = (lo, hi)`, one rank's brick x-slabs) holds only those slabs'
    particles: `occupancy` covers the owned buckets (read it through `_occ`), rows and arena
    are local, and `n_particles` is the local count. Per-brick arrays stay global length; a
    brick outside the owned range is an empty run with scale 1.0, exactly an empty brick.
    """

    t9: object
    bricks_per_side: int
    brick_start: np.ndarray  # int64 (n_bricks+1,) fixed allocation runs
    occupancy: np.ndarray  # uint32 (owned buckets,) THE index; also bucket bounds
    off: np.ndarray  # uint8 (n_alloc + n_arena, 3)
    w: np.ndarray  # int16 (n_alloc + n_arena, 3)
    # float64 (n_bricks,): one velocity scale per brick, fixed in `_insert_slab`
    vel_scale: np.ndarray
    arena_base: int
    arena_bucket: np.ndarray  # int64 (n_arena,) -1 where free
    n_particles: int
    ids: np.ndarray = None  # int32 (n_alloc + n_arena,) or None
    # owned brick x-slabs [lo, hi); None = the whole box
    slabs: tuple = None
    # the arena's fraction of the particles at build or load (None if unknown)
    arena_frac: float = None
    # caches over `arena_bucket` (None = dirty): brick -> absolute arena rows, and the
    # ascending free arena-relative rows
    _arena_by_brick: dict = None
    _arena_free: object = None

    # -------------------------------------------------------------- building

    @classmethod
    def build(
        cls,
        x,
        v,
        t9,
        bricks_per_side,
        brick_slack=0.10,
        alloc_margin=0.10,
        arena_frac=0.01,
        index_dtype=DEFAULT_INDEX_DTYPE,
        with_ids=False,
    ):
        """Encode (x, v) and place every particle in its bucket's slot.

        Spare is pooled per brick (`brick_slack`), total allocation grown by `alloc_margin`,
        plus an arena of `arena_frac * n` rows. `with_ids` stores int32 original indices.
        """
        x = np.asarray(x, dtype=np.float64)
        nbk = t9.n_buckets_side
        if nbk % int(bricks_per_side):
            raise ValueError(
                f"bricks_per_side {bricks_per_side} must divide the bucket grid {nbk}"
            )
        n = x.shape[0]
        per3 = (nbk // int(bricks_per_side)) ** 3
        n_bricks = int(bricks_per_side) ** 3

        key, _, _ = bucket_order_key(x, t9, int(bricks_per_side))
        brick = key // per3
        brick_counts = np.bincount(brick, minlength=n_bricks).astype(np.int64)
        occupancy = np.bincount(key, minlength=n_bricks * per3).astype(np.int64)

        _, brick_start, n_alloc, n_arena = _alloc_geometry(
            brick_counts, n, brick_slack, alloc_margin, arena_frac
        )

        order = _stable_sort_index(key)
        rank = _within_run_index(brick_counts)
        slots = brick_start[brick[order]] + rank

        n_rows = n_alloc + n_arena

        off_all, bijk = encode_positions_host(x, t9)
        # `order` is brick-major, so the per-brick scale reduction reuses the layout sort
        v = np.asarray(v, dtype=np.float64)
        scale = _scales_from_sorted(np.abs(v).max(axis=1)[order], brick_counts)
        w_all = _encode_at(v, scale[brick])

        off = np.zeros((n_rows, 3), dtype=np.uint8)
        w = np.zeros((n_rows, 3), dtype=np.int16)
        off[slots] = off_all[order]
        w[slots] = w_all[order]

        ids = None
        if with_ids:
            refuse_ids_above_int32(t9.n_part)
            ids = np.full(n_rows, -1, dtype=np.int32)
            ids[slots] = order.astype(np.int32)

        return cls(
            t9=t9,
            bricks_per_side=int(bricks_per_side),
            brick_start=brick_start,
            occupancy=_to_index(occupancy, index_dtype, "initial"),
            off=off,
            w=w,
            vel_scale=scale,
            arena_base=n_alloc,
            arena_bucket=np.full(n_arena, -1, dtype=np.int64),
            n_particles=n,
            ids=ids,
            arena_frac=float(arena_frac),
        )

    # ----------------------------------------------------------- geometry

    @property
    def owned_slabs(self):
        """The brick x-slabs [lo, hi) this state holds."""
        if self.slabs is None:
            return 0, int(self.bricks_per_side)
        return int(self.slabs[0]), int(self.slabs[1])

    @property
    def is_whole(self):
        return self.owned_slabs == (0, int(self.bricks_per_side))

    @property
    def owned_bricks(self):
        """The brick ordinals [lo, hi) this state holds (contiguous: slabs are x-major)."""
        lo, hi = self.owned_slabs
        nb2 = int(self.bricks_per_side) ** 2
        return lo * nb2, hi * nb2

    @property
    def bucket_lo(self):
        """Flat bucket ordinal of `occupancy[0]`."""
        return self.owned_bricks[0] * self.buckets_per_brick

    def _occ(self, lo_b, hi_b=None):
        """View of `occupancy` over bricks [lo_b, hi_b) (default the one brick `lo_b`).

        Raises for any brick outside the owned range, where a shifted slice would wrap or
        read another brick's buckets silently.
        """
        lo_b = int(lo_b)
        hi_b = lo_b + 1 if hi_b is None else int(hi_b)
        blo, bhi = self.owned_bricks
        if not blo <= lo_b <= hi_b <= bhi:
            raise IndexError(
                f"bricks [{lo_b}, {hi_b}) are outside this state's owned bricks "
                f"[{blo}, {bhi}) (slabs {self.owned_slabs})"
            )
        p3 = self.buckets_per_brick
        return self.occupancy[(lo_b - blo) * p3 : (hi_b - blo) * p3]

    @property
    def index_dtype(self):
        """The occupancy array's dtype."""
        return self.occupancy.dtype

    @property
    def buckets_per_brick(self):
        return (self.t9.n_buckets_side // self.bricks_per_side) ** 3

    @property
    def n_bricks(self):
        return self.bricks_per_side**3

    @property
    def n_buckets(self):
        """Buckets `occupancy` indexes: the owned ones."""
        return len(self.occupancy)

    @property
    def n_slots(self):
        return int(self.brick_start[-1])

    @property
    def n_arena(self):
        return len(self.arena_bucket)

    @property
    def arena_used(self):
        return int(np.sum(self.arena_bucket >= 0))

    @property
    def n_live(self):
        """Every particle the container holds: brick runs plus arena residents."""
        return occupancy_total(self.occupancy) + self.arena_used

    def brick_slot_range(self, brick_flat):
        """The brick's ALLOCATION span (live rows plus its spare)."""
        return int(self.brick_start[brick_flat]), int(self.brick_start[brick_flat + 1])

    def brick_live_count(self, brick_flat):
        return int(self._occ(brick_flat).astype(np.int64).sum())

    def brick_member_count(self, brick_flat):
        """Live rows plus arena residents: the brick's true membership.

        `brick_live_count` is the run length alone; use this wherever membership is meant.
        """
        return self.brick_live_count(brick_flat) + len(self.arena_slots_of_brick(brick_flat))

    def bucket_slot_starts(self, brick_flat):
        """Absolute slot boundaries (p3 + 1,) of the buckets in one brick, derived from
        `occupancy` by prefix sum rather than stored."""
        p3 = self.buckets_per_brick
        occ = self._occ(brick_flat).astype(np.int64)
        out = np.zeros(p3 + 1, dtype=np.int64)
        np.cumsum(occ, out=out[1:])
        return int(self.brick_start[brick_flat]) + out

    # ------------------------------------------------- the implied bucket

    def bucket_flat_of_live_slots(self, brick_flat):
        """Flat bucket ordinal for each of the brick's live rows, in slot order (a `repeat`
        over the brick's occupancy slice)."""
        p3 = self.buckets_per_brick
        occ = self._occ(brick_flat).astype(np.int64)
        return brick_flat * p3 + np.repeat(np.arange(p3, dtype=np.int64), occ)

    def bucket_ijk_of_live_slots(self, brick_flat):
        """Per-axis bucket index for each of the brick's live rows."""
        from .layout import bucket_ijk_from_key

        return bucket_ijk_from_key(
            self.bucket_flat_of_live_slots(brick_flat), self.t9, self.bricks_per_side
        )

    def _invalidate_arena_index(self):
        self._arena_by_brick = None
        self._arena_free = None

    def _build_arena_index(self):
        """Group the occupied arena rows by brick, ONCE."""
        idx = {}
        if self.n_arena:
            live = np.nonzero(self.arena_bucket >= 0)[0]
            if len(live):
                b = self.arena_bucket[live] // self.buckets_per_brick
                order = np.argsort(b, kind="stable")
                live, b = live[order], b[order]
                edges = np.nonzero(np.diff(b))[0] + 1
                for part in np.split(np.arange(len(b)), edges):
                    idx[int(b[part[0]])] = self.arena_base + live[part]
        self._arena_by_brick = idx
        return idx

    def arena_slots_of_brick(self, brick_flat):
        """Absolute arena rows belonging to this brick, ascending.

        An arena particle still belongs to its brick; omitting it silently drops it from the
        force. Served from the `_arena_by_brick` cache (built once, O(n_arena)); every write
        to `arena_bucket` must update or invalidate it.
        """
        if self.n_arena == 0:
            return np.empty(0, dtype=np.int64)
        idx = self._arena_by_brick
        if idx is None:
            idx = self._build_arena_index()
        return idx.get(int(brick_flat), np.empty(0, dtype=np.int64))

    def brick_member_slots(self, brick_flat):
        """Every slot holding one of this brick's particles: its live run, then
        its arena residents."""
        lo = int(self.brick_start[brick_flat])
        m = self.brick_live_count(brick_flat)
        run = np.arange(lo, lo + m, dtype=np.int64)
        a = self.arena_slots_of_brick(brick_flat)
        return np.concatenate([run, a]) if len(a) else run

    # ---------------------------------------------------------- decoding

    def decode_brick(self, brick_flat, scales=None):
        """(slots, x, v) for every particle of this brick, run rows then arena rows.

        O(brick) floats; nothing on the engine path is O(N) in floats. `scales` overrides
        `self.vel_scale`; the migration passes its pre-pass snapshot because `_insert_slab`
        rewrites scales in place.
        """
        lo = int(self.brick_start[brick_flat])
        m = self.brick_live_count(brick_flat)
        slots = np.arange(lo, lo + m, dtype=np.int64)
        bijk = self.bucket_ijk_of_live_slots(brick_flat)
        a = self.arena_slots_of_brick(brick_flat)
        if len(a):
            from .layout import bucket_ijk_from_key

            slots = np.concatenate([slots, a])
            a_b = bucket_ijk_from_key(
                self.arena_bucket[a - self.arena_base], self.t9, self.bricks_per_side
            )
            bijk = np.concatenate([bijk, a_b])
        x = decode_positions_host(self.off[slots], bijk, self.t9)
        s = (self.vel_scale if scales is None else scales)[brick_flat]
        v = self.w[slots].astype(np.float64) * s
        return slots, x, v

    # ---------------------------------------------------------- the check

    def check(self):
        """Structural consistency (counts, spans, aliasing, arena buckets). Raises or returns True.

        It does not decode and compare each slot's bucket with its position's: an offset is
        stored relative to its bucket, so `(bucket * 256 + off) // 256 == bucket` always holds
        and that check cannot fail. Placement against an external reference is
        `check_placement`.
        """
        p3 = self.buckets_per_brick
        blo, bhi = self.owned_bricks
        if len(self.occupancy) != (bhi - blo) * p3:
            raise ValueError(
                f"occupancy holds {len(self.occupancy)} buckets but slabs "
                f"{self.owned_slabs} own {(bhi - blo) * p3}"
            )
        occ = self.occupancy.astype(np.int64)

        # 1. no brick may hold more live rows than its allocation, and a brick this state
        # does not own has none
        live = occ.reshape(bhi - blo, p3).sum(axis=1)
        cap_all = np.diff(self.brick_start)
        foreign = np.nonzero(np.concatenate([cap_all[:blo], cap_all[bhi:]]))[0]
        if len(foreign):
            raise ValueError(
                f"{len(foreign)} brick(s) outside the owned bricks [{blo}, {bhi}) have an "
                "allocation: a non-owned brick must be an empty run"
            )
        cap = cap_all[blo:bhi]
        over = np.nonzero(live > cap)[0]
        if len(over):
            j = int(over[0])
            raise ValueError(
                f"brick {blo + j} holds {int(live[j])} live rows in an allocation of "
                f"{int(cap[j])} ({len(over)} bricks affected). Its run has overrun the next "
                "brick's slots, which silently reassigns particles rather than losing them."
            )

        # 2. nothing lost, nothing duplicated
        seen = int(live.sum()) + self.arena_used
        if seen != self.n_particles:
            raise ValueError(
                f"{seen} particles reachable through the layout against {self.n_particles} "
                "stored: the container has lost or duplicated state"
            )

        # 3. the arena cannot alias the brick runs
        if self.n_arena and self.arena_base < int(self.brick_start[-1]):
            raise ValueError(
                f"arena_base {self.arena_base} is inside the brick runs, which end at "
                f"{int(self.brick_start[-1])}: arena rows alias live slots"
            )

        # 4. every occupied arena row names an owned bucket
        if self.n_arena:
            used = self.arena_bucket[self.arena_bucket >= 0]
            lo_k, hi_k = self.bucket_lo, self.bucket_lo + self.n_buckets
            if len(used) and (int(used.min()) < lo_k or int(used.max()) >= hi_k):
                raise ValueError(
                    f"an arena row names bucket {int(used.max())} (or {int(used.min())}) "
                    f"outside the owned buckets [{lo_k}, {hi_k})"
                )
        return True

    def check_placement(self, x):
        """Did every particle land in the bucket its reference position `x` calls for?

        `x` is (n, 3) in original particle order, linked through the id tier. An O(N) float
        array, so test/build-time only; never call it on the engine path.
        """
        if self.ids is None:
            raise ValueError(
                "check_placement needs the opt-in id tier to map slots back to the "
                "reference array; rebuild with with_ids=True"
            )
        x = np.asarray(x, dtype=np.float64)
        from .layout import _bucket_ijk

        want = _bucket_ijk(x, self.t9)
        for b in range(self.n_bricks):
            slots = self.brick_member_slots(b)
            if not len(slots):
                continue
            ids = self.ids[slots]
            got = self._bucket_ijk_of_slots(b, slots)
            if not np.array_equal(got, want[ids]):
                bad = int(np.count_nonzero(np.any(got != want[ids], axis=1)))
                raise ValueError(
                    f"brick {b}: {bad} of {len(slots)} particles sit in a bucket other than "
                    "the one their reference position falls in"
                )
        return True

    def _bucket_ijk_of_slots(self, brick_flat, slots):
        """Per-axis bucket for an arbitrary set of this brick's slots (run rows
        then arena rows), in the order `brick_member_slots` returns them."""
        from .layout import bucket_ijk_from_key

        m = self.brick_live_count(brick_flat)
        out = self.bucket_ijk_of_live_slots(brick_flat)
        if len(slots) > m:
            a = slots[m:] - self.arena_base
            out = np.concatenate(
                [out, bucket_ijk_from_key(self.arena_bucket[a], self.t9, self.bricks_per_side)]
            )
        return out

    # ------------------------------------------------------------ the cost

    def tile_bricks(self, tijk, n_tile, b_fine, n_brick, n_fine):
        """The brick ordinals covering tile+buffer (`tile_brick_ids`); same union and wrap
        guard as `BrickPackedLayout.tile_members`, but returning bricks, not particle indices."""
        nb = self._check_brick_grid(n_fine, n_brick)
        return tile_brick_ids(tijk, n_tile, b_fine, n_brick, nb)

    def _check_brick_grid(self, n_fine, n_brick):
        nb = int(n_fine) // int(n_brick)
        if nb != self.bricks_per_side:
            raise ValueError(
                f"brick grid {nb} from (n_fine={n_fine}, n_brick={n_brick}) disagrees with "
                f"the layout's {self.bricks_per_side}"
            )
        return nb

    def brick_member_counts(self, workers=None):
        """`brick_member_count` for every brick at once: an int64 (n_bricks,) array.

        Occupancy summed per brick (x-slab chunks on `workers` threads, default the CPU count;
        integer sums, so the result is exact at any thread count) plus arena residents.
        Bricks this state does not own count 0.
        """
        import os
        from concurrent.futures import ThreadPoolExecutor

        nb, p3 = self.bricks_per_side, self.buckets_per_brick
        per_slab = nb * nb
        s_lo, s_hi = self.owned_slabs
        blo, bhi = self.owned_bricks
        out = np.empty(self.n_bricks, dtype=np.int64)
        out[:blo] = 0
        out[bhi:] = 0
        occ = self.occupancy.reshape(bhi - blo, p3)

        def slab(bx):
            lo, hi = bx * per_slab, (bx + 1) * per_slab
            np.sum(occ[lo - blo:hi - blo], axis=1, dtype=np.int64, out=out[lo:hi])

        n_s = s_hi - s_lo
        w = max(1, min(n_s, int(os.cpu_count() or 1) if workers is None else int(workers)))
        if w == 1:
            for bx in range(s_lo, s_hi):
                slab(bx)
        else:
            with ThreadPoolExecutor(max_workers=w) as ex:
                list(ex.map(slab, range(s_lo, s_hi)))
        if self.n_arena:
            b = self.arena_bucket[self.arena_bucket >= 0] // p3
            out += np.bincount(b, minlength=self.n_bricks).astype(np.int64, copy=False)
        return out

    def tile_member_counts(self, n_tile, b_fine, n_brick, n_fine, planes=None):
        """Member count of every tile of tile x-planes `planes` (default all), arena residents
        included: `tile_window_counts` over `brick_member_counts`."""
        nb = self._check_brick_grid(n_fine, n_brick)
        return tile_window_counts(self.brick_member_counts().reshape(nb, nb, nb), n_tile,
                                  b_fine, n_brick, planes=planes)

    def decode_bricks(self, bricks, scales=None):
        """(slots, x, v) over a list of bricks, concatenated.

        Rows stay grouped by brick in the order `bricks` gives; the kick relies on this to
        take one scale per brick without a sort.
        """
        s, xs, vs = [], [], []
        for b in bricks:
            sl, x, v = self.decode_brick(b, scales=scales)
            if len(sl):
                s.append(sl)
                xs.append(x)
                vs.append(v)
        if not s:
            return (
                np.empty(0, np.int64),
                np.empty((0, 3), np.float64),
                np.empty((0, 3), np.float64),
            )
        return np.concatenate(s), np.concatenate(xs), np.concatenate(vs)

    def write_velocities(self, slots, w):
        """Write int16 velocity codes back to given slots (the kick's only write)."""
        self.w[slots] = w

    # ------------------------------------------------- drift and re-home

    def slab_bricks(self, bx):
        """The brick ordinals [lo, hi) of x-slab `bx`; bricks are numbered
        `(bx * nb + by) * nb + bz`, so a slab is contiguous."""
        nb = self.bricks_per_side
        return int(bx) * nb * nb, (int(bx) + 1) * nb * nb

    def _eject_slab_jax(self, bx, c_drift, scales, release_arena=True):
        """`_eject_slab` with the drift and keep/leave partition compiled (`eject_jax`).

        Same contract, return value and mutations; gated elementwise against the numpy path
        (`tests/test_eject_jax.py`). Slot resolution and the arena release stay on the host,
        the release copied line for line from `_eject_slab`. The kernel goes from
        (off, bijk, w) straight to the new lattice index without a float decode, and runs
        once per slab, returning keepers then emigrants as slices of one buffer.
        """
        from .eject_jax import eject_rows
        from .layout import bucket_ijk_from_key

        lo_b, hi_b = self.slab_bricks(bx)
        offs, bijks, ws, ids_l, sc_l, bid_l = [], [], [], [], [], []
        for b in range(lo_b, hi_b):
            lo = int(self.brick_start[b])
            m = self.brick_live_count(b)
            slots = np.arange(lo, lo + m, dtype=np.int64)
            bijk = self.bucket_ijk_of_live_slots(b)
            a = self.arena_slots_of_brick(b)
            if len(a):
                slots = np.concatenate([slots, a])
                bijk = np.concatenate(
                    [bijk,
                     bucket_ijk_from_key(self.arena_bucket[a - self.arena_base],
                                         self.t9, self.bricks_per_side)]
                )
            if not len(slots):
                continue
            # the arena release, identical to `_eject_slab`'s
            if release_arena and len(a):
                self.arena_bucket[a - self.arena_base] = -1
                if self._arena_by_brick is not None:
                    self._arena_by_brick.pop(int(b), None)
                self._arena_free = None
            offs.append(self.off[slots])
            bijks.append(bijk)
            ws.append(self.w[slots])
            if self.ids is not None:
                ids_l.append(self.ids[slots])
            sc_l.append(np.full((len(slots), 1), float(scales[b]), dtype=np.float64))
            bid_l.append(np.full(len(slots), b, dtype=np.int64))

        if not offs:
            return (_cat([], [], [], []), _cat([], [], [], [], src=[]))

        dest, off_new, w_out, ids_out, src_out, n_keep = eject_rows(
            self.t9, self.bricks_per_side,
            np.concatenate(offs), np.concatenate(bijks), np.concatenate(ws),
            np.concatenate(ids_l) if ids_l else None,
            np.concatenate(sc_l), c_drift, np.concatenate(bid_l),
        )
        keep = dict(dest=dest[:n_keep], off=off_new[:n_keep], w=w_out[:n_keep],
                    ids=None if ids_out is None else ids_out[:n_keep])
        emig = dict(dest=dest[n_keep:], off=off_new[n_keep:], w=w_out[n_keep:],
                    ids=None if ids_out is None else ids_out[n_keep:],
                    src=src_out[n_keep:].astype(np.int32))
        return keep, emig

    def _eject_slab(self, bx, c_drift, scales, kernel="numpy", release_arena=True):
        """Drift one slab's particles and split them into keepers and leavers.

        Writes no payload back (only releases the slab's arena rows, unless
        `release_arena=False`, used by pooled workers whose parent replays the release), so a
        double drift is impossible. Returns (keep, emig): dicts of flat `dest` (flat bucket
        ordinal), `off`, `w` (still at the source brick's scale), `ids`, and `src` on emig.

        `kernel="jax"` routes to `_eject_slab_jax`. This numpy path is the reference it is
        gated against, so it must not be conditionally modified.
        """
        if kernel == "jax":
            return self._eject_slab_jax(bx, c_drift, scales, release_arena=release_arena)
        if kernel != "numpy":
            raise ValueError(f"unknown eject kernel {kernel!r}; expected 'numpy' or 'jax'")
        lo_b, hi_b = self.slab_bricks(bx)
        p3 = self.buckets_per_brick
        k_dest, k_off, k_w, k_id = [], [], [], []
        e_dest, e_off, e_w, e_id, e_src = [], [], [], [], []
        for b in range(lo_b, hi_b):
            slots, x, v = self.decode_brick(b, scales=scales)
            if not len(slots):
                continue
            # Arena residents are re-homed like any member, and their rows released here where
            # the payload is consumed (after the read, so no claim can take a live row).
            a_free = self.arena_slots_of_brick(b)
            if release_arena and len(a_free):
                self.arena_bucket[a_free - self.arena_base] = -1
                # drop exactly this brick's key; the free list goes dirty so freed rows
                # re-enter in ascending order via the rebuild scan
                if self._arena_by_brick is not None:
                    self._arena_by_brick.pop(int(b), None)
                self._arena_free = None
            # Drift in the integer lattice domain: the wrap is exactly modular there, and a small
            # step cannot be absorbed by a large coordinate.
            q = self.t9.quantum
            i_new = np.mod(
                np.rint(x / q + (float(c_drift) * v) / q).astype(np.int64), self.t9.n_levels
            )
            b_ijk = i_new // LEVELS_PER_BUCKET
            off_new = (i_new - b_ijk * LEVELS_PER_BUCKET).astype(np.uint8)
            dest = _bucket_flat_brick_major(b_ijk, self.t9, self.bricks_per_side)
            # No rescale: a row leaves at its own brick's scale and is re-expressed once, at the
            # destination's, in `_insert_slab`.
            w_cur = self.w[slots]
            stay = (dest // p3) == b
            ids_b = self.ids[slots] if self.ids is not None else None
            k_dest.append(dest[stay])
            k_off.append(off_new[stay])
            k_w.append(w_cur[stay])
            k_id.append(ids_b[stay] if ids_b is not None else None)
            e_dest.append(dest[~stay])
            e_off.append(off_new[~stay])
            e_w.append(w_cur[~stay])
            e_id.append(ids_b[~stay] if ids_b is not None else None)
            e_src.append(np.full(int((~stay).sum()), b, dtype=np.int32))
        return (
            _cat(k_dest, k_off, k_w, k_id),
            _cat(e_dest, e_off, e_w, e_id, src=e_src),
        )

    def _insert_slab(
        self, bx, staged, emig, reach=(-1, 0, 1), consumed=None, scales=None, spill_sink=None,
        kernel="numpy",
    ):
        """Write one slab's bricks back from its keepers and the immigrants from `reach` slabs.

        Overflow escalates to the arena (or `spill_sink`), then a refusal; nothing is
        clamped. This is the only place a brick's velocity scale can be fixed, since only here
        is its post-migration membership known: the scale is taken over keepers plus
        immigrants and every row re-expressed at it in one rounding. `scales` is the
        pre-migration snapshot (rows arrive coded at their source brick's scale). `consumed`
        accumulates emig rows taken per source slab. Returns the arena overflow count.

        `kernel="jax"` routes to `_insert_slab_jax`; this numpy path is its reference.
        """
        if kernel == "jax":
            return self._insert_slab_jax(bx, staged, emig, reach, consumed, scales, spill_sink)
        if kernel != "numpy":
            raise ValueError(f"unknown insert kernel {kernel!r}; expected 'numpy' or 'jax'")
        nb = self.bricks_per_side
        p3 = self.buckets_per_brick
        lo_b, hi_b = self.slab_bricks(bx)
        keep = staged[bx]
        # immigrants from every slab within reach: the set the schedule ejects first
        sources = sorted({(int(bx) + o) % nb for o in reach})
        if consumed is not None:
            for s in sources:
                if s in emig and len(emig[s]["dest"]):
                    d_slab = emig[s]["dest"] // (p3 * nb * nb)
                    consumed[s] += int(np.count_nonzero(d_slab == bx))
        imm = _cat_dicts([emig[s] for s in sources if s in emig])
        n_over = 0
        k_ord, k_off = _group_by_brick(keep["dest"] // p3, lo_b, hi_b)
        i_ord, i_off = _group_by_brick(imm["dest"] // p3, lo_b, hi_b)
        for j, b in enumerate(range(lo_b, hi_b)):
            sel_k = k_ord[k_off[j] : k_off[j + 1]]
            sel_i = i_ord[i_off[j] : i_off[j + 1]]
            # no arena term: arena residents were re-homed by the brick's own ejection
            has_i = len(sel_i) > 0
            dest = np.concatenate(
                [
                    keep["dest"][sel_k],
                    imm["dest"][sel_i] if has_i else np.empty(0, np.int64),
                ]
            )
            off = np.concatenate(
                [
                    keep["off"][sel_k],
                    imm["off"][sel_i] if has_i else np.empty((0, 3), np.uint8),
                ]
            )
            w_k = keep["w"][sel_k]
            w_i = imm["w"][sel_i] if has_i else np.empty((0, 3), np.int16)
            # the brick's new scale: max physical |v| over keepers (own old scale) and
            # immigrants (source brick's scale)
            s_k = float(scales[b])
            s_i = scales[imm["src"][sel_i]] if has_i and imm.get("src") is not None else None
            vmax = 0.0
            if len(w_k):
                vmax = max(vmax, float(np.abs(w_k).max()) * s_k)
            if len(w_i):
                vmax = max(vmax, float((np.abs(w_i).max(axis=1) * s_i).max()))
            s_b = vmax / INT16_MAX
            s_b = s_b if s_b > 0.0 else 1.0
            w = np.concatenate(
                [
                    _rescale_w(w_k, s_k, s_b),
                    _rescale_w(w_i, s_i, s_b) if len(w_i) else w_i,
                ]
            )
            self.vel_scale[b] = s_b
            ids = None
            if self.ids is not None:
                ids = np.concatenate(
                    [
                        keep["ids"][sel_k],
                        imm["ids"][sel_i] if has_i else np.empty(0, np.int32),
                    ]
                )
            n_over += self._write_brick(b, dest, off, w, ids, spill_sink=spill_sink)
        return n_over

    def _insert_slab_jax(self, bx, staged, emig, reach=(-1, 0, 1), consumed=None,
                         scales=None, spill_sink=None):
        """`_insert_slab` with the grouping, scale, rescale and within-brick order compiled
        (`insert_jax`). Same contract, return value and mutations.

        The census, old-scale gather, state writes and arena claims stay on the host; claims
        run per brick ascending because `_to_arena` takes the lowest free slots. An int16
        escape refuses before any write (the numpy path refuses at the offending brick).
        """
        from .insert_jax import insert_rows

        nb = self.bricks_per_side
        p3 = self.buckets_per_brick
        lo_b, hi_b = self.slab_bricks(bx)
        keep = staged[bx]
        sources = sorted({(int(bx) + o) % nb for o in reach})
        if consumed is not None:
            for s in sources:
                if s in emig and len(emig[s]["dest"]):
                    d_slab = emig[s]["dest"] // (p3 * nb * nb)
                    consumed[s] += int(np.count_nonzero(d_slab == bx))
        imm = _cat_dicts([emig[s] for s in sources if s in emig])
        has_ids = self.ids is not None

        def ids_of(d):
            return d["ids"] if d["ids"] is not None else np.empty(0, np.int32)

        dest = np.concatenate([keep["dest"], imm["dest"]])
        src_brick = np.concatenate([keep["dest"] // p3, np.asarray(imm["src"], dtype=np.int64)])
        res = insert_rows(
            dest, np.concatenate([keep["off"], imm["off"]]),
            np.concatenate([keep["w"], imm["w"]]),
            np.concatenate([ids_of(keep), ids_of(imm)]) if has_ids else None,
            np.asarray(scales, dtype=np.float64)[src_brick], lo_b,
            self.brick_start[lo_b:hi_b + 1], p3)
        if res["abs_max"] > INT16_MAX:
            raise ValueError(
                f"velocity code {res['abs_max']:.0f} escapes int16 under a rescale to a "
                "scale that does not cover it. Per-brick scales make this reachable where a "
                "global scale made it impossible; the caller must fix the destination scale "
                "over the rows it is about to write. Integer state is never clamped."
            )
        nw, ns = res["n_write"], res["n_spill"]
        pos = res["pos"][:nw]
        self.off[pos] = res["off"][:nw]
        self.w[pos] = res["w"][:nw]
        if has_ids:
            self.ids[pos] = res["ids"][:nw]
        self._occ(lo_b, hi_b)[...] = _to_index(res["occupancy"], self.index_dtype, "migrated")
        self.vel_scale[lo_b:hi_b] = res["scales"]
        if ns:
            sl = slice(nw, nw + ns)
            sd, so, sw = res["dest"][sl], res["off"][sl], res["w"][sl]
            si = res["ids"][sl] if has_ids else None
            sb = sd // p3
            for grp in np.split(np.arange(ns), np.flatnonzero(np.diff(sb)) + 1):
                args = (sd[grp], so[grp], sw[grp], None if si is None else si[grp])
                if spill_sink is None:
                    self._to_arena(*args)
                else:
                    spill_sink(int(sb[grp[0]]), *args)
        return int(ns)

    def _write_brick(self, b, dest, off, w, ids=None, spill_sink=None):
        """Stable-sort one brick's members by bucket and write its run and occupancy.

        Rows beyond the brick's allocation (the highest buckets) go to the arena, or to
        `spill_sink(b, dest, off, w, ids)` for a pooled worker, which must not claim because
        claim order is the arena layout. Returns the overflow count.
        """
        p3 = self.buckets_per_brick
        lo, hi = self.brick_slot_range(b)
        within = dest - b * p3
        counts = np.bincount(within, minlength=p3).astype(np.int64)
        cap = hi - lo
        n_over = 0
        if len(dest) > cap:
            # the run keeps a prefix of the bucket order; the tail spills
            order = _stable_order(within, p3)
            keep_n = cap
            spill = order[keep_n:]
            n_over = len(spill)
            if spill_sink is None:
                self._to_arena(
                    dest[spill], off[spill], w[spill], None if ids is None else ids[spill]
                )
            else:
                spill_sink(
                    b, dest[spill], off[spill], w[spill], None if ids is None else ids[spill]
                )
            order = order[:keep_n]
            within, dest, off, w = within[order], dest[order], off[order], w[order]
            if ids is not None:
                ids = ids[order]
            counts = np.bincount(within, minlength=p3).astype(np.int64)
        else:
            order = _stable_order(within, p3)
            within, off, w = within[order], off[order], w[order]
            if ids is not None:
                ids = ids[order]
        m = len(off)
        self.off[lo : lo + m] = off
        self.w[lo : lo + m] = w
        if ids is not None:
            self.ids[lo : lo + m] = ids
        self._occ(b)[...] = _to_index(counts, self.index_dtype, "migrated")
        return n_over

    def _to_arena(self, dest, off, w, ids=None):
        """Park overflow rows in the lowest free arena slots, or refuse. Never clamp or drop.

        The free list is cached in `_arena_free` and rebuilt by an ascending scan after a
        release dirties it; claims consume only its head, so the layout is the uncached one.
        """
        free = self._arena_free
        if free is None:
            free = np.nonzero(self.arena_bucket < 0)[0]
        if len(free) < len(dest):
            raise ValueError(
                f"{len(dest)} particles overflow their brick's capacity and the arena of "
                f"{self.n_arena} slots has only {len(free)} free. The layout does not clamp "
                "or drop. Raise brick_slack or arena_frac."
            )
        a = free[: len(dest)]
        self._arena_free = free[len(dest):]
        self.arena_bucket[a] = dest
        # update the brick index in place; sorting each touched key reproduces a rebuild exactly
        idx = self._arena_by_brick
        if idx is not None:
            bricks = np.asarray(dest, dtype=np.int64) // self.buckets_per_brick
            slots_abs = self.arena_base + a
            for b in np.unique(bricks):
                add = slots_abs[bricks == b]
                cur = idx.get(int(b))
                idx[int(b)] = np.sort(np.concatenate([cur, add])) if cur is not None else add
        self.off[self.arena_base + a] = off
        self.w[self.arena_base + a] = w
        if ids is not None:
            self.ids[self.arena_base + a] = ids

    def _release_brick_arena(self, b):
        """Release brick b's arena rows without ejecting it, for the parent-side replay of a
        pooled or device migrate. Must stay identical to the release in `_eject_slab` and
        `_eject_slab_jax`."""
        a_free = self.arena_slots_of_brick(b)
        if len(a_free):
            self.arena_bucket[a_free - self.arena_base] = -1
            if self._arena_by_brick is not None:
                self._arena_by_brick.pop(int(b), None)
            self._arena_free = None

    def _repack_reference(self, brick_slack=0.10):
        """Out-of-place repack: the elementwise reference `repack` is tested against.

        Never call it on the engine path: it allocates O(N) copies of `off` and `w`. Same
        result as `repack` (arena folded back, slot order == key order); a bucket that
        overflows the index dtype is refused when the counts are narrowed at the end.
        """
        p3 = self.buckets_per_brick
        blo, bhi = self.owned_bricks
        k0 = self.bucket_lo
        occ = self.occupancy.astype(np.int64)
        arena_live = np.nonzero(self.arena_bucket >= 0)[0]
        if len(arena_live):
            occ = occ + np.bincount(self.arena_bucket[arena_live] - k0,
                                    minlength=self.n_buckets)
        counts = np.zeros(self.n_bricks, dtype=np.int64)
        counts[blo:bhi] = occ.reshape(bhi - blo, p3).sum(axis=1)
        spare = np.ceil(counts * float(brick_slack)).astype(np.int64)
        spare = np.where(counts > 0, np.maximum(spare, 1), spare)
        new_start = np.zeros(self.n_bricks + 1, dtype=np.int64)
        np.cumsum(counts + spare, out=new_start[1:])
        n_alloc = int(new_start[-1])
        if n_alloc + self.n_arena > self.off.shape[0]:
            raise ValueError(
                f"repack needs {n_alloc} slots plus a {self.n_arena}-slot arena against an "
                f"allocation of {self.off.shape[0]}. Raise alloc_margin at build."
            )
        off = np.zeros_like(self.off)
        w = np.zeros_like(self.w)
        ids = None if self.ids is None else np.full_like(self.ids, -1)
        new_occ = np.zeros(self.n_buckets, dtype=np.int64)
        for b in range(blo, bhi):
            slots = self.brick_member_slots(b)
            if not len(slots):
                continue
            dest = self._bucket_flat_of_slots(b, slots)
            order = _stable_order(dest - b * p3, p3)
            lo = int(new_start[b])
            m = len(order)
            off[lo : lo + m] = self.off[slots[order]]
            w[lo : lo + m] = self.w[slots[order]]
            if ids is not None:
                ids[lo : lo + m] = self.ids[slots[order]]
            new_occ[b * p3 - k0 : (b + 1) * p3 - k0] = np.bincount(
                dest[order] - b * p3, minlength=p3
            )
        self.off, self.w = off, w
        if ids is not None:
            self.ids = ids
        self.brick_start = new_start
        self.occupancy = _to_index(new_occ, self.index_dtype, "repacked")
        self.arena_base = n_alloc
        self.arena_bucket[:] = -1
        self._invalidate_arena_index()
        return dict(slots_used=n_alloc, slots_per_particle=n_alloc / max(self.n_particles, 1))

    def repack(self, brick_slack=0.10):
        """Redistribute brick capacity to match current occupancy, in place.

        Required each step: frozen capacity overflows as structure collapses. Arena residents
        are folded back into their brick's run, so afterwards slot order is key order and the
        arena is empty. A monotone rearrangement in two directional passes:

          A. ascending, compact each brick's main run left to `main_pos[b]` (safe: a run never
             exceeds its allocation, so `sum(run_counts[:b]) <= brick_start[b]`);
          B. descending, write each brick's final block at `new_start[b] >= main_pos[b]`,
             merging its arena residents (lifted out first, since pass B can write past the
             old `arena_base`). Bricks without residents move as blocks (fast path).

        Arena rows cannot join pass A: they would push `sum(counts[:b])` past `brick_start[b]`
        onto unread rows. Spare and the tail are zeroed so the result is bytewise
        `_repack_reference`'s. Array contents are rewritten, not rebound (the pool shares them).
        Returns slots used, `scratch_bytes` (all transients), and `bricks_fast`/`bricks_merged`.
        """
        p3 = self.buckets_per_brick
        blo, bhi = self.owned_bricks
        k0 = self.bucket_lo
        # per-brick counts, accumulated in int64 without casting the per-bucket index
        run_counts = np.zeros(self.n_bricks, dtype=np.int64)
        run_counts[blo:bhi] = self.occupancy.reshape(bhi - blo, p3).sum(axis=1, dtype=np.int64)
        arena_live = np.nonzero(self.arena_bucket >= 0)[0]
        counts = run_counts.copy()
        if len(arena_live):
            counts += np.bincount(self.arena_bucket[arena_live] // p3,
                                  minlength=self.n_bricks)

        # Overflow refusal before `new_occ` is built at the index dtype: a brick total that
        # fits bounds every bucket in it. numpy narrows modularly, and one wrapped bucket
        # would shift every later bucket span in its brick.
        _limit = int(np.iinfo(self.index_dtype).max)
        _hot = int(counts.max()) if counts.size else 0
        if _hot > _limit:
            raise ValueError(
                f"repacked brick holds {_hot} rows against the "
                f"{np.dtype(self.index_dtype).name} index ceiling {_limit}: a "
                "bucket in it cannot be stored. Widen index_dtype at build."
            )
        spare = np.ceil(counts * float(brick_slack)).astype(np.int64)
        spare = np.where(counts > 0, np.maximum(spare, 1), spare)
        new_start = np.zeros(self.n_bricks + 1, dtype=np.int64)
        np.cumsum(counts + spare, out=new_start[1:])
        n_alloc = int(new_start[-1])
        if n_alloc + self.n_arena > self.off.shape[0]:
            raise ValueError(
                f"repack needs {n_alloc} slots plus a {self.n_arena}-slot arena against an "
                f"allocation of {self.off.shape[0]}. Raise alloc_margin at build."
            )

        scratch = 0
        row_bytes = 3 * self.off.itemsize + 3 * self.w.itemsize
        if self.ids is not None:
            row_bytes += self.ids.itemsize
        # ---- lift the live arena rows out, grouped by brick, before anything moves
        a_rows = self.arena_base + arena_live
        a_bucket = self.arena_bucket[arena_live]
        a_ord = np.argsort(a_bucket // p3, kind="stable")
        a_bucket = a_bucket[a_ord]
        a_off = self.off[a_rows[a_ord]].copy()
        a_w = self.w[a_rows[a_ord]].copy()
        a_ids = None if self.ids is None else self.ids[a_rows[a_ord]].copy()
        scratch += a_off.nbytes + a_w.nbytes + (0 if a_ids is None else a_ids.nbytes)
        a_edge = np.searchsorted(a_bucket // p3, np.arange(self.n_bricks + 1))

        # ---- pass A: compact the main runs leftward
        main_pos = np.zeros(self.n_bricks + 1, dtype=np.int64)
        np.cumsum(run_counts, out=main_pos[1:])
        for b in range(blo, bhi):
            m = int(run_counts[b])
            src, dst = int(self.brick_start[b]), int(main_pos[b])
            if m == 0 or src == dst:
                continue
            # No explicit copy: numpy buffers overlapping slice assignments itself (pinned by
            # `test_an_overlapping_slice_assignment_copies_before_it_writes`).
            scratch = max(scratch, m * row_bytes)
            self.off[dst : dst + m] = self.off[src : src + m]
            self.w[dst : dst + m] = self.w[src : src + m]
            if self.ids is not None:
                self.ids[dst : dst + m] = self.ids[src : src + m]

        # ---- pass B: expand rightward, merging the arena residents back in
        new_occ = np.zeros(self.n_buckets, dtype=self.index_dtype)
        bucket_ids = np.arange(p3, dtype=np.int64)
        n_fast = n_merge = 0
        for b in range(bhi - 1, blo - 1, -1):
            m = int(run_counts[b])
            k = int(a_edge[b + 1] - a_edge[b])
            if m + k == 0:
                continue
            mp, ns = int(main_pos[b]), int(new_start[b])
            if k == 0:
                # no residents: the run is already in bucket order, so move it as a block
                n_fast += 1
                scratch = max(scratch, m * row_bytes)
                self.off[ns : ns + m] = self.off[mp : mp + m]
                self.w[ns : ns + m] = self.w[mp : mp + m]
                if self.ids is not None:
                    self.ids[ns : ns + m] = self.ids[mp : mp + m]
                new_occ[b * p3 - k0 : (b + 1) * p3 - k0] = self._occ(b)
            else:
                n_merge += 1
                within = np.repeat(bucket_ids, occ_b := np.asarray(
                    self._occ(b), dtype=np.int64))
                del occ_b
                within = np.concatenate(
                    [within, a_bucket[a_edge[b] : a_edge[b + 1]] - b * p3])
                # stable over main-then-arena: `_repack_reference`'s exact order
                order = _stable_order(within, p3)
                cat_off = np.concatenate(
                    [self.off[mp : mp + m], a_off[a_edge[b] : a_edge[b + 1]]])
                cat_w = np.concatenate(
                    [self.w[mp : mp + m], a_w[a_edge[b] : a_edge[b + 1]]])
                scratch = max(
                    scratch, cat_off.nbytes + cat_w.nbytes + within.nbytes + order.nbytes)
                # `np.take(axis=0, out=)`: faster than advanced indexing on (n, 3), no temporary
                np.take(cat_off, order, axis=0, out=self.off[ns : ns + m + k])
                np.take(cat_w, order, axis=0, out=self.w[ns : ns + m + k])
                if self.ids is not None:
                    cat_i = np.concatenate(
                        [self.ids[mp : mp + m], a_ids[a_edge[b] : a_edge[b + 1]]])
                    np.take(cat_i, order, axis=0, out=self.ids[ns : ns + m + k])
                new_occ[b * p3 - k0 : (b + 1) * p3 - k0] = np.bincount(within[order],
                                                                      minlength=p3)
            # zero the spare (ids -1) so states compare bytewise; it lies above every
            # main block still to be read
            gap_lo, gap_hi = ns + m + k, int(new_start[b + 1])
            if gap_hi > gap_lo:
                self.off[gap_lo:gap_hi] = 0
                self.w[gap_lo:gap_hi] = 0
                if self.ids is not None:
                    self.ids[gap_lo:gap_hi] = -1

        # everything past the new allocation, arena included
        self.off[n_alloc:] = 0
        self.w[n_alloc:] = 0
        if self.ids is not None:
            self.ids[n_alloc:] = -1
        # contents, not bindings: worker processes share these arrays through shared memory;
        # `new_occ` is already bounded and at the index dtype
        self.brick_start[...] = new_start
        self.occupancy[...] = new_occ
        self.arena_base = n_alloc
        self.arena_bucket[:] = -1
        self._invalidate_arena_index()
        return dict(
            slots_used=n_alloc,
            slots_per_particle=n_alloc / max(self.n_particles, 1),
            scratch_bytes=int(scratch),
            bricks_fast=int(n_fast),
            bricks_merged=int(n_merge),
        )

    def _bucket_flat_of_slots(self, brick_flat, slots):
        """Flat bucket ordinal per slot, run rows then arena rows."""
        m = self.brick_live_count(brick_flat)
        out = self.bucket_flat_of_live_slots(brick_flat)
        if len(slots) > m:
            a = slots[m:] - self.arena_base
            out = np.concatenate([out, self.arena_bucket[a]])
        return out

    def bytes_per_particle(self, payload=9.0):
        """All-in bytes per particle by term. `scaffold` (the per-particle bookkeeping
        `BrickPackedLayout` reports) is kept as an explicit 0.0: those arrays do not exist here."""
        n = float(self.n_particles)
        index = self.n_buckets * self.occupancy.dtype.itemsize / n
        brick_csr = (len(self.brick_start)) * 8 / n
        scales = len(np.atleast_1d(self.vel_scale)) * 8 / n
        slack = (self.n_slots - self.n_particles) * payload / n
        arena = self.n_arena * (payload + 8) / n
        ids = (0.0 if self.ids is None else 4.0)
        return dict(
            payload=payload,
            bucket_index=index,
            brick_start=brick_csr,
            slack=slack,
            arena=arena,
            ids=ids,
            brick_scales=scales,
            total=payload + index + brick_csr + slack + arena + ids + scales,
            scaffold=0.0,
        )
