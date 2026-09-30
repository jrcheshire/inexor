"""A tile's decode on the device: the device twin of `SlotState.decode_bricks`.

The host resolves indices, all O(bricks + arena) (brick list, `brick_start`, `occupancy`, the
cached arena membership); the device turns them into slots, buckets, positions and velocities for
every row. The decode is elementwise integer arithmetic plus one multiply, with no reduction or
scatter, so it is gated on bitwise equality with `decode_bricks` on every backend.

Row order is a contract: brick by brick in the order `bricks` gives, live run first, then arena
residents in `arena_slots_of_brick` order. Downstream per-brick quantize and slot writes rely on a
brick's rows being contiguous.

`decode_core` is pure jnp (traceable under jit, used by `device.paint`); `decode_rows` is its
eager wrapper. `decode_rows` takes whole `off`/`w` arrays; at scale callers feed it a window.
"""

from __future__ import annotations

import numpy as np


def tile_decode_plan(st, bricks):
    """Host-side index metadata for one tile. O(bricks + arena), never O(rows).

    Returns a dict of small arrays. `arena_slots` is padded to a rectangle with
    -1; the kernel masks on `arena_counts`, not on the sentinel.
    """
    bricks = np.asarray(bricks, dtype=np.int64)
    p3 = int(st.buckets_per_brick)
    n_b = len(bricks)

    # slice before widening: widening first would copy the whole index
    occ = st.brick_occ(bricks).astype(np.int64)
    live_counts = occ.sum(axis=1)
    starts = np.asarray(st.brick_start, dtype=np.int64)[bricks]

    arena = [np.asarray(st.arena_slots_of_brick(int(b)), dtype=np.int64) for b in bricks]
    arena_counts = np.array([len(a) for a in arena], dtype=np.int64)
    a_max = int(arena_counts.max()) if n_b else 0
    arena_slots = np.full((n_b, max(a_max, 1)), -1, dtype=np.int64)
    for i, a in enumerate(arena):
        if len(a):
            arena_slots[i, : len(a)] = a

    member_counts = live_counts + arena_counts
    return dict(
        bricks=bricks, starts=starts, occ=occ, live_counts=live_counts,
        arena_slots=arena_slots, arena_counts=arena_counts,
        member_counts=member_counts, row_offsets=np.concatenate(
            ([0], np.cumsum(member_counts))),
        n_rows=int(member_counts.sum()), p3=p3,
    )


def decode_core(starts, occ, live_counts, arena_slots, row_offsets, bricks, off,
                arena_bucket, arena_base, n_rows, *, cap, lift, p3, per,
                bricks_per_side, t9, fdtype):
    """(slots, x, brick_of_row, brick_index, live) at `cap` rows. Pure jnp.

    Every array argument is a device array or a tracer; `arena_base` and
    `n_rows` may be Python ints or traced scalars. The keyword arguments fix
    shapes and constants. `lift` must exceed every value a per-brick occupancy
    prefix sum or a row rank can take -- `max(cap, n_rows) + 1` does.
    """
    import jax.numpy as jnp

    from ..codec import decode_positions

    n_b = bricks.shape[0]
    r = jnp.arange(cap, dtype=jnp.int64)
    live = r < n_rows

    # each row's brick: searchsorted on the running row total (the fixed-shape np.repeat)
    bi = jnp.clip(jnp.searchsorted(row_offsets[1:], r, side="right"), 0, n_b - 1)
    rank = r - row_offsets[bi]

    lc = live_counts[bi]
    is_arena = rank >= lc

    # live rows: slot is the brick's run, bucket is a prefix over occupancy
    slot_live = starts[bi] + rank
    # The bucket ordinal counts the own brick's prefix sums <= rank. To avoid a
    # (rows, buckets_per_brick) table, per-brick prefix sums are offset by b * lift into
    # one nondecreasing sequence and counted with one searchsorted. int64 throughout:
    # b * lift exceeds int32.
    occ_cum = jnp.cumsum(occ.astype(jnp.int64), axis=1)
    bi64 = bi.astype(jnp.int64)
    keys = (occ_cum
            + (jnp.arange(n_b, dtype=jnp.int64) * lift)[:, None]).reshape(-1)
    within = jnp.searchsorted(keys, bi64 * lift + rank, side="right") - bi64 * p3
    within = jnp.clip(within, 0, p3 - 1)
    bucket_flat_live = bricks[bi] * p3 + within

    # arena rows: slot and bucket both come from the arena's own records
    a_k = jnp.clip(rank - lc, 0, arena_slots.shape[1] - 1)
    slot_arena = arena_slots[bi, a_k]
    slot_arena = jnp.where(slot_arena < 0, 0, slot_arena)
    bucket_flat_arena = arena_bucket[jnp.clip(slot_arena - arena_base, 0, None)]

    slots = jnp.where(is_arena, slot_arena, slot_live)
    bucket_flat = jnp.where(is_arena, bucket_flat_arena, bucket_flat_live)
    slots = jnp.where(live, slots, 0)

    # brick-major ordinal -> per-axis bucket index: jnp twin of the numpy-only
    # `layout.bucket_ijk_from_key`
    brick_flat, within_flat = jnp.divmod(bucket_flat, per**3)
    bx, rem = jnp.divmod(brick_flat, bricks_per_side * bricks_per_side)
    by, bz = jnp.divmod(rem, bricks_per_side)
    wx, rem = jnp.divmod(within_flat, per * per)
    wy, wz = jnp.divmod(rem, per)
    bucket_ijk = jnp.stack([bx * per + wx, by * per + wy, bz * per + wz], axis=-1)

    # decode_positions owns LEVELS_PER_BUCKET; do not re-spell it here
    x = decode_positions(off[slots], bucket_ijk, t9, fdtype=fdtype)

    # `bi` is the row's position in the tile's brick list (a dense segment id for the
    # kick), not its global brick id
    return slots, x, bricks[bi], bi, live


def decode_rows(plan, off, w, vel_scale, arena_bucket, arena_base, t9,
                bricks_per_side, cap, fdtype=None, velocities=True):
    """Dict of slots, x, v, brick_of_row, brick_index, live, n_rows for a tile, padded to `cap`.

    Bitwise identical to `SlotState.decode_bricks` plus `np.repeat` of the brick
    ids, in the same row order. `velocities=False` skips `v` (`w` and `vel_scale`
    may then be None), for the positions-only coarse paint.

    `cap` is the padded row count the tile force uses (`forces.capacity_shape`), so
    per-tile row counts do not each compile a new shape. Padded rows carry slot 0
    and `live=False`; the caller masks them.
    """
    import jax.numpy as jnp

    from ..eject_jax import require_x64

    # without x64 slot indices narrow to int32 and wrap silently at scale, while
    # small-scale tests still pass
    require_x64()

    fdtype = jnp.float64 if fdtype is None else fdtype
    cap = int(cap)
    n_rows = int(plan["n_rows"])
    slots, x, brick_of_row, bi, live = decode_core(
        jnp.asarray(plan["starts"]), jnp.asarray(plan["occ"]),
        jnp.asarray(plan["live_counts"]), jnp.asarray(plan["arena_slots"]),
        jnp.asarray(plan["row_offsets"]), jnp.asarray(plan["bricks"]),
        jnp.asarray(off), jnp.asarray(arena_bucket), int(arena_base), n_rows,
        cap=cap, lift=max(cap, n_rows) + 1, p3=int(plan["p3"]),
        per=int(t9.n_buckets_side // bricks_per_side),
        bricks_per_side=int(bricks_per_side), t9=t9, fdtype=fdtype,
    )
    out = dict(slots=slots, x=x, brick_of_row=brick_of_row, brick_index=bi,
               live=live, n_rows=n_rows)
    if velocities:
        # `vel_scale` must be a snapshot: `_insert_slab` rewrites scales in place
        scale = jnp.asarray(vel_scale)[brick_of_row]
        out["v"] = jnp.asarray(w)[slots].astype(fdtype) * scale[:, None]
    return out
