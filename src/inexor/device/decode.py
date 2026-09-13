"""D2a: a tile's decode, on the device.

WHAT THIS REPLACES. `engine.tile_task` opens with `st.decode_bricks(bricks)` --
`SlotState.decode_brick` per brick, all numpy, ~4096 particles each at C-gh.
That is a per-particle HOST pass, and the gb probe measured that host plumbing
is 90% of a tile as the engine stands. It has to go, and it goes first because
every other phase of the tile pass consumes its output.

THE SPLIT. The host resolves INDICES and the device does ARITHMETIC. Everything
the host computes here is O(bricks) or O(arena), never O(rows): the brick list
is geometry, `brick_start` and `occupancy` are per brick, and the arena's
per-brick membership is a dict the state already caches. The device then turns
those into slots, buckets, positions and velocities for every row.

WHY THIS PHASE CAN CARRY A BITWISE GATE, which most device work cannot. The
decode is elementwise integer arithmetic followed by one multiply:
`(bucket * 256 + off) * quantum` and `w * scale`. No reduction, no scatter, no
transcendental -- so there is nothing for a GPU's ordering freedom to change,
and equality with `SlotState.decode_bricks` is exact rather than approximate on
both backends. `tests/conftest.py`'s `detflag` skip exists for the f32 scatter
in the paint; it does not apply here, and a tolerance here would be hiding
something rather than accommodating it.

THE ROW ORDER IS A CONTRACT, not an implementation detail. `decode_bricks`
concatenates brick by brick in the order `bricks` gives, live run first then
arena residents, and `tile_task`'s per-brick quantize depends on a brick's rows
being contiguous -- it replaces a sort with a run scan and asserts the
contiguity because "if a brick ever appeared in two runs, the second run would
silently overwrite the first one's scale and decode every row of it wrong".
This module reproduces that order exactly.

TWO ENTRY POINTS, ONE PROGRAM. `decode_core` is pure jnp with the row count and
the arena base as values, so it can be traced under `jax.jit`
(`device.paint`'s jitted chunk). `decode_rows` is the eager wrapper over it that
the tile pass and the tests call. Both run the same operations.

SCOPE, stated so it is not mistaken for more than it is. `decode_rows` takes
the WHOLE `off` and `w` arrays as device inputs, which is right at development
scale and is not the design at 4096^3, where the state is 794 GB on the host
and only a window of slabs is ever resident. `device.paint` feeds it a window.
"""

from __future__ import annotations

import numpy as np

# The row layout inside one brick, mirroring `SlotState.decode_brick`:
# live run [brick_start, brick_start + live_count), then arena residents in the
# order `arena_slots_of_brick` returns them.


def tile_decode_plan(st, bricks):
    """Host-side index metadata for one tile. O(bricks + arena), never O(rows).

    Returns a dict of small arrays. Nothing here is per particle: at C-gh a
    tile is ~5.3M rows and this is 512 buckets x the tile's bricks plus however
    many arena residents those bricks hold (measured peak 0.57% of N).

    `arena_slots` is padded to a rectangle because the device wants one shape;
    `-1` marks a pad and the kernel masks on `arena_counts`, not on the
    sentinel, so a legitimate slot can never be read as padding.
    """
    bricks = np.asarray(bricks, dtype=np.int64)
    p3 = int(st.buckets_per_brick)
    n_b = len(bricks)

    occ = np.asarray(st.occupancy, dtype=np.int64).reshape(-1, p3)[bricks]
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

    # WHICH BRICK each row belongs to. `searchsorted` on the running row total
    # is the device form of the host's `np.repeat` over member counts -- same
    # answer, fixed shape, and it does not need the ragged intermediate.
    bi = jnp.clip(jnp.searchsorted(row_offsets[1:], r, side="right"), 0, n_b - 1)
    rank = r - row_offsets[bi]

    lc = live_counts[bi]
    is_arena = rank >= lc

    # --- live rows: slot is the brick's run, bucket is a prefix over occupancy
    slot_live = starts[bi] + rank
    # The bucket ordinal is how many of the row's OWN brick's prefix sums are
    # <= its rank. Comparing every row against its brick's whole prefix sum
    # (`occ_cum[bi] <= rank[:, None]`) is the obvious spelling and it builds a
    # (rows, buckets_per_brick) table: ~100 GB for one 4096^3 tile and ~1.1 TB
    # for an x-slab of bricks. So the per-brick prefix sums are lifted into ONE
    # nondecreasing sequence -- brick b's by b * lift, with lift above any
    # value a prefix sum or a rank can take -- and one searchsorted counts the
    # same thing: every entry of the earlier bricks, plus exactly the own
    # brick's entries <= rank, and none of the later ones. Same integers, O(rows).
    #
    # int64 throughout: `searchsorted` returns int32, and b * lift at an x-slab
    # of bricks is ~2e13.
    occ_cum = jnp.cumsum(occ.astype(jnp.int64), axis=1)
    bi64 = bi.astype(jnp.int64)
    keys = (occ_cum
            + (jnp.arange(n_b, dtype=jnp.int64) * lift)[:, None]).reshape(-1)
    within = jnp.searchsorted(keys, bi64 * lift + rank, side="right") - bi64 * p3
    within = jnp.clip(within, 0, p3 - 1)
    bucket_flat_live = bricks[bi] * p3 + within

    # --- arena rows: slot and bucket both come from the arena's own records
    a_k = jnp.clip(rank - lc, 0, arena_slots.shape[1] - 1)
    slot_arena = arena_slots[bi, a_k]
    slot_arena = jnp.where(slot_arena < 0, 0, slot_arena)
    bucket_flat_arena = arena_bucket[jnp.clip(slot_arena - arena_base, 0, None)]

    slots = jnp.where(is_arena, slot_arena, slot_live)
    bucket_flat = jnp.where(is_arena, bucket_flat_arena, bucket_flat_live)
    slots = jnp.where(live, slots, 0)

    # --- the brick-major ordinal -> per-axis bucket index, the jnp twin of
    # `layout.bucket_ijk_from_key`. Written out rather than imported because
    # that one is numpy by design ("the layout never puts the global position
    # array on the device").
    brick_flat, within_flat = jnp.divmod(bucket_flat, per**3)
    bx, rem = jnp.divmod(brick_flat, bricks_per_side * bricks_per_side)
    by, bz = jnp.divmod(rem, bricks_per_side)
    wx, rem = jnp.divmod(within_flat, per * per)
    wy, wz = jnp.divmod(rem, per)
    bucket_ijk = jnp.stack([bx * per + wx, by * per + wy, bz * per + wz], axis=-1)

    # `decode_positions` owns the 256 (LEVELS_PER_BUCKET); it is deliberately
    # not re-spelled here, because a second copy of that constant is how the
    # decode and the encode drift apart.
    x = decode_positions(off[slots], bucket_ijk, t9, fdtype=fdtype)

    # `brick_index` is `bi` itself -- the row's position in the TILE's brick
    # list, not its global brick id. The kick's segmented reduction wants a
    # dense 0..n_b-1 segment id, and recovering one from the global id would
    # mean a searchsorted the decode has already done.
    return slots, x, bricks[bi], bi, live


def decode_rows(plan, off, w, vel_scale, arena_bucket, arena_base, t9,
                bricks_per_side, cap, fdtype=None, velocities=True):
    """(slots, x, v, brick_of_row, live) for a tile's rows, padded to `cap`.

    Device arithmetic, host indices. Bitwise identical to
    `SlotState.decode_bricks` followed by `np.repeat` of the brick ids, in the
    same row order.

    `velocities=False` skips the velocity decode and returns no `v`; `w` and
    `vel_scale` are then unused and may be None. The coarse paint needs
    positions only, and uploading `w` for it would be 6 B per row of bus
    traffic that buys nothing.

    `cap` is the padded row count the tile force already uses
    (`forces.tile_capacity` / `capacity_shape`), and it is padded here for the
    same reason it is padded there: a per-tile row count keys a new XLA shape,
    which was profiled at 2,107 compilations and 74% of a step before the fix.
    Padded rows carry slot 0 and `live=False`; they decode to a real position
    and are masked by the caller, exactly as `tile_task` does today.
    """
    import jax.numpy as jnp

    from ..eject_jax import require_x64

    # x64 OR NOTHING, the same contract and the same reason as the compiled
    # eject. With it off, `jnp.arange(cap, dtype=int64)` truncates to int32 and
    # every slot index narrows with it. At c-hero the slot space is 8.3e10 --
    # 38x past int32 -- so slots wrap silently, the gather reads the wrong rows,
    # and nothing raises. It is invisible at development scale, which is exactly
    # what makes it worth refusing rather than documenting: this file's own
    # bitwise gate passes at n_part=16 with x64 off.
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
        # velocity: the brick's own scale, which is why `decode_brick` takes a
        # SNAPSHOT -- `_insert_slab` rewrites a scale in place and a decode
        # reading the live array would be correct only while the schedule
        # happens to eject every brick before inserting it.
        scale = jnp.asarray(vel_scale)[brick_of_row]
        out["v"] = jnp.asarray(w)[slots].astype(fdtype) * scale[:, None]
    return out
