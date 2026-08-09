"""T9 state stored IN SLOT ORDER: the engine's container (M-v2-3).

## Why this exists next to `layout.BrickPackedLayout` rather than replacing it

`BrickPackedLayout` is an INDEX INTO ARRAYS THE CALLER HOLDS. `slot_to_particle`
maps a slot to a particle NUMBER, and the caller keeps the positions somewhere
else, in their original order. That costs three per-particle bookkeeping arrays --
`key` (int32), `particle_to_slot` and `slot_to_particle` (int64) -- which come to
**~21 B/p against a ~10.5 B/p state budget**, twice the thing they index. D-v2-20
measured that, reported it beside the total as `scaffold`, and refused to widen
`key` rather than fix it, because "the streamed engine stores state IN slot
order, where a particle's bucket is implied by where it sits and none of the
three exists". This module is that engine-side container, and removing the
ceiling is what D-v2-20 assigned to M-v2-3.

`BrickPackedLayout` stays exactly as it is. D-v2-19 and D-v2-20 are measurements
OF that code and `scripts/v2_m1_migration.py` is the probe that produced them --
the same relationship `scripts/v2_g5_core.py` has to `forces.py` under D-v2-16
clause 7. Deleting it would retire the M-v2-1 record's reproducibility to save a
class nothing on the engine path instantiates.

## What "the bucket is implied by where it sits" means concretely

Buckets are ordered brick-major, so one brick's buckets are contiguous, and
`occupancy` (the uint32 index D-v2-20 ratified) doubles as the bucket boundaries
via a prefix sum WITHIN a brick -- the trick D-v2-19 clause 2 introduced to
delete a stored 1.00 B/p `bucket_start`. So for a slot:

    brick   = the run of `brick_start` it falls in
    bucket  = searchsorted(prefix-sum of that brick's occupancy, slot - lo)

and the stored `off` is relative to exactly that bucket. Nothing per-particle is
stored to say so.

**Free slots need no sentinel.** A brick's live rows are its first
`sum(occupancy[brick])` slots and the rest is spare, so liveness is already a
function of `occupancy`. A `-1` marker would be a second source of truth beside
it, and the two could disagree.

## The invariant, and it is the whole correctness statement

For every occupied slot, the bucket DERIVED from the slot's position in the
layout equals the bucket the stored offset was encoded against. `check()` asserts
it for every particle rather than sampling: `BrickPackedLayout.check()` samples
three bricks, and the M-v2-1 record notes a sampled check very nearly missed a
real bug (`repack` scattering arena residents into wrong bucket spans -- clean
through step 5, fired at step 6). Here the full sweep is a decode and a compare,
O(N) time in O(brick) memory, so sampling buys nothing worth its risk.
"""

from dataclasses import dataclass

import numpy as np

from .codec import (
    LEVELS_PER_BUCKET,
    assert_int16_range,
    encode_velocities,
    refuse_ids_above_int32,
)
from .layout import (
    DEFAULT_INDEX_DTYPE,
    _stable_sort_index,
    _to_index,
    _within_run_index,
    bucket_order_key,
)

__all__ = ["SlotState", "decode_positions_host", "encode_positions_host"]


# ===========================================================================
# host-side codec mirrors
# ===========================================================================
#
# `codec.encode_positions` / `decode_positions` are jnp. The layout and the
# engine's exchange are host numpy by design -- `layout._bucket_ijk` says so:
# "the layout never puts the global position array on the device". These are the
# numpy mirrors, and `tests/test_slot_state.py` asserts they are BITWISE the
# codec's on shared inputs, because two implementations of one definition that
# are never compared is how the two drift apart.


def encode_positions_host(x, t9):
    """Physical positions -> (uint8 offsets, int64 bucket ijk). Mirrors
    `codec.encode_positions`; the wrap is taken in the integer domain, where it
    is exactly modular (D-007)."""
    i = np.mod(np.rint(np.asarray(x, dtype=np.float64) / t9.quantum).astype(np.int64), t9.n_levels)
    b = i // LEVELS_PER_BUCKET
    return (i - b * LEVELS_PER_BUCKET).astype(np.uint8), b


def decode_positions_host(off, bucket_ijk, t9):
    """(offsets, bucket ijk) -> physical positions. Mirrors
    `codec.decode_positions`, including its expression SHAPE: reconstruct the
    global lattice index, then multiply ONCE by the quantum. That form is what
    makes bitwise equality with the ratified probe hold by construction rather
    than by an exponent coincidence."""
    i = np.asarray(bucket_ijk, dtype=np.int64) * LEVELS_PER_BUCKET + np.asarray(
        off, dtype=np.int64
    )
    return i.astype(np.float64) * t9.quantum


# ===========================================================================
# the container
# ===========================================================================


@dataclass
class SlotState:
    """T9 payload stored in slot order; the bucket is implied by the slot."""

    t9: object
    bricks_per_side: int
    brick_start: np.ndarray  # int64 (n_bricks+1,) fixed allocation runs
    occupancy: np.ndarray  # uint32 (n_buckets,) THE index; also bucket bounds
    off: np.ndarray  # uint8 (n_alloc + n_arena, 3)
    w: np.ndarray  # int16 (n_alloc + n_arena, 3)
    vel_scale: float
    arena_base: int
    arena_bucket: np.ndarray  # int64 (n_arena,) -1 where free
    n_particles: int
    ids: np.ndarray = None  # int32 (n_alloc + n_arena,) or None

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

        Capacity policy is D-v2-19's, unchanged: spare pooled per BRICK (a bucket
        cannot be given a fraction of a slot, so per-bucket spare costs one whole
        slot per occupied bucket -- 12.5% of payload whatever the setting), plus a
        small arena for the rare brick that still overflows.
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

        spare = np.ceil(brick_counts * float(brick_slack)).astype(np.int64)
        spare = np.where(brick_counts > 0, np.maximum(spare, 1), spare)
        brick_start = np.zeros(n_bricks + 1, dtype=np.int64)
        np.cumsum(brick_counts + spare, out=brick_start[1:])

        order = _stable_sort_index(key)
        rank = _within_run_index(brick_counts)
        slots = brick_start[brick[order]] + rank

        n_alloc = int(np.ceil(int(brick_start[-1]) * (1.0 + float(alloc_margin))))
        n_arena = int(np.ceil(n * float(arena_frac)))
        n_rows = n_alloc + n_arena

        # the payload, written straight into slot order -- this is the whole point
        off_all, bijk = encode_positions_host(x, t9)
        w_all, scale = encode_velocities(np.asarray(v, dtype=np.float64))
        w_all = np.asarray(w_all)
        assert_int16_range(w_all)

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
            vel_scale=float(scale),
            arena_base=n_alloc,
            arena_bucket=np.full(n_arena, -1, dtype=np.int64),
            n_particles=n,
            ids=ids,
        )

    # ----------------------------------------------------------- geometry

    @property
    def index_dtype(self):
        """Read off the array rather than stored separately, so the two cannot
        disagree about what is in force."""
        return self.occupancy.dtype

    @property
    def buckets_per_brick(self):
        return (self.t9.n_buckets_side // self.bricks_per_side) ** 3

    @property
    def n_bricks(self):
        return self.bricks_per_side**3

    @property
    def n_buckets(self):
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
        return int(np.sum(self.occupancy.astype(np.int64))) + self.arena_used

    def brick_slot_range(self, brick_flat):
        """The brick's ALLOCATION span (live rows plus its spare)."""
        return int(self.brick_start[brick_flat]), int(self.brick_start[brick_flat + 1])

    def brick_live_count(self, brick_flat):
        p3 = self.buckets_per_brick
        return int(
            self.occupancy[brick_flat * p3 : (brick_flat + 1) * p3].astype(np.int64).sum()
        )

    def bucket_slot_starts(self, brick_flat):
        """Bucket boundaries inside ONE brick, DERIVED rather than stored.

        The array the per-bucket layout kept globally at 8 B per bucket -- 1.00
        B/p at C-gh, which D-v2-19 clause 2 found uncounted. Here it is a prefix
        sum over the brick's own occupancy slice, O(512) at C-gh, computed where
        it is needed and never resident.
        """
        p3 = self.buckets_per_brick
        occ = self.occupancy[brick_flat * p3 : (brick_flat + 1) * p3].astype(np.int64)
        out = np.zeros(p3 + 1, dtype=np.int64)
        np.cumsum(occ, out=out[1:])
        return int(self.brick_start[brick_flat]) + out

    # ------------------------------------------------- the implied bucket

    def bucket_flat_of_live_slots(self, brick_flat):
        """Flat bucket ordinal for each of the brick's live rows, in slot order.

        This is the inverse of "the bucket is implied by where it sits", and it
        is a `repeat` over the brick's occupancy slice rather than a search --
        the rows are already grouped by bucket.
        """
        p3 = self.buckets_per_brick
        occ = self.occupancy[brick_flat * p3 : (brick_flat + 1) * p3].astype(np.int64)
        return brick_flat * p3 + np.repeat(np.arange(p3, dtype=np.int64), occ)

    def bucket_ijk_of_live_slots(self, brick_flat):
        """Per-axis bucket index for each of the brick's live rows."""
        from .layout import bucket_ijk_from_key

        return bucket_ijk_from_key(
            self.bucket_flat_of_live_slots(brick_flat), self.t9, self.bricks_per_side
        )

    def arena_slots_of_brick(self, brick_flat):
        """Arena rows belonging to this brick.

        An arena particle still BELONGS to its brick -- it is only stored
        elsewhere because the brick was momentarily full -- and omitting it
        deletes it from the force with nothing raising. D-v2-19 clause 4 measured
        that at 98.4% loss on a stress fixture and 0.57% at the operating point.
        """
        if self.n_arena == 0:
            return np.empty(0, dtype=np.int64)
        # free arena rows carry -1, and -1 // p3 is -1, so they never match
        sel = np.nonzero(self.arena_bucket // self.buckets_per_brick == brick_flat)[0]
        return self.arena_base + sel

    def brick_member_slots(self, brick_flat):
        """Every slot holding one of this brick's particles: its live run, then
        its arena residents."""
        lo = int(self.brick_start[brick_flat])
        m = self.brick_live_count(brick_flat)
        run = np.arange(lo, lo + m, dtype=np.int64)
        a = self.arena_slots_of_brick(brick_flat)
        return np.concatenate([run, a]) if len(a) else run

    # ---------------------------------------------------------- decoding

    def decode_brick(self, brick_flat):
        """(slots, x, v) for every particle of this brick, arena included.

        O(brick) floats -- ~4096 particles at C-gh, about 100 KB. Nothing here
        is ever O(N) in floats; a global (n,3) f64 array is 206 GB at C-gh and
        deleting it is D-v2-16 clause 1.
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
        v = self.w[slots].astype(np.float64) * self.vel_scale
        return slots, x, v

    # ---------------------------------------------------------- the check

    def check(self):
        """Structural consistency of the container. Raises, or returns True.

        **WHAT THIS DELIBERATELY DOES NOT CHECK, and why the obvious version of
        it is worthless.** The tempting invariant is "decode each slot and assert
        the bucket its position falls in equals the bucket its slot implies".
        That is an IDENTITY, not a test. `decode` reconstructs the lattice index
        as `bucket * 256 + off` with `off` a uint8, so the recovered bucket is
        `(bucket * 256 + off) // 256 == bucket` for every value the byte can
        hold. The sweep passes on arbitrarily corrupted offsets -- verified by
        corrupting one and watching it pass -- which makes it a gate that cannot
        fail, on the module whose whole premise is that the bucket is implied.

        The offset genuinely CANNOT disagree with its bucket; it is stored
        relative to it. What can go wrong is structural, and that is what is
        checked here: counts, spans, aliasing and ownership. The placement
        question -- did this particle land in the bucket its POSITION calls for --
        needs an external reference and lives in `check_placement`.
        """
        p3 = self.buckets_per_brick
        occ = self.occupancy.astype(np.int64)

        # 1. no brick may hold more live rows than its allocation
        live = occ.reshape(self.n_bricks, p3).sum(axis=1)
        cap = np.diff(self.brick_start)
        over = np.nonzero(live > cap)[0]
        if len(over):
            b = int(over[0])
            raise ValueError(
                f"brick {b} holds {int(live[b])} live rows in an allocation of {int(cap[b])} "
                f"({len(over)} bricks affected). Its run has overrun the next brick's slots, "
                "which silently reassigns particles rather than losing them."
            )

        # 2. nothing lost, nothing duplicated (D-007 forbids either)
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

        # 4. every occupied arena row names a real bucket
        if self.n_arena:
            used = self.arena_bucket[self.arena_bucket >= 0]
            if len(used) and int(used.max()) >= self.n_buckets:
                raise ValueError(
                    f"an arena row names bucket {int(used.max())} of {self.n_buckets}"
                )

        # 5. the stored index dtype is the one the state was built with -- a
        #    silent widening would make D-v2-20's 0.50 B/p figure wrong
        if self.occupancy.dtype != self.index_dtype:
            raise ValueError("the occupancy index changed dtype under the container")
        return True

    def check_placement(self, x):
        """Did every particle land in the bucket its POSITION calls for?

        The question `check` cannot answer, because answering it needs the
        positions from OUTSIDE the container -- the stored state is otherwise its
        own authority. Requires an `(n, 3)` array in the ORIGINAL particle order
        and the id tier to connect the two, so it is a build-time and test-time
        instrument: at C-gh that array is 206 GB and D-v2-16 clause 1 deletes it.
        Never call this on the engine path.
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

    def bytes_per_particle(self, payload=9.0):
        """The all-in figure, with `scaffold` reported and EMPTY.

        D-v2-20 added a `scaffold` line to `BrickPackedLayout` reporting ~21 B/p
        of per-particle bookkeeping beside the total and deliberately not inside
        it, so the exclusion was visible rather than inferred. Here the line
        stays and reads 0.0: those arrays do not exist. Keeping it rather than
        deleting it is the point -- a term that vanishes from a table is
        indistinguishable from one that was never counted.
        """
        n = float(self.n_particles)
        index = self.n_buckets * self.occupancy.dtype.itemsize / n
        brick_csr = (len(self.brick_start)) * 8 / n
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
            total=payload + index + brick_csr + slack + arena + ids,
            scaffold=0.0,
        )
