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

__all__ = [
    "SlotState",
    "decode_positions_host",
    "drift_and_migrate",
    "encode_positions_host",
    "reconcile_velocity_scale",
]


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


def _bucket_flat_brick_major(bucket_ijk, t9, bricks_per_side):
    """Per-axis bucket -> brick-major flat ordinal. `layout.bucket_order_key`'s
    tail, split out so the exchange can key on a bucket it already has rather
    than re-deriving it from a position."""
    per = t9.n_buckets_side // int(bricks_per_side)
    b = np.asarray(bucket_ijk, dtype=np.int64)
    brick = b // per
    within = b - brick * per
    bf = (brick[:, 0] * bricks_per_side + brick[:, 1]) * bricks_per_side + brick[:, 2]
    wf = (within[:, 0] * per + within[:, 1]) * per + within[:, 2]
    return bf * (per**3) + wf


def _rescale_w(w, s_old, s_new):
    """Re-express an int16 velocity code at a new scale.

    Exact when the scales are equal, which is the common case within a step; the
    general path decodes and re-rounds. It CANNOT overflow, and that is a
    theorem rather than a margin: `s_new` is the max over tiles of each tile's
    own `max|v|/32767`, and tile ownership is a partition, so `s_new` is exactly
    the global `max|v|/32767` and `|rint(v/s_new)| <= 32767` for every particle.
    """
    if s_old == s_new:
        return w
    out = np.rint(np.asarray(w, dtype=np.float64) * (float(s_old) / float(s_new)))
    return out.astype(np.int16)


def _cat(dest, off, w, ids):
    out = dict(
        dest=np.concatenate(dest) if dest else np.empty(0, np.int64),
        off=np.concatenate(off) if off else np.empty((0, 3), np.uint8),
        w=np.concatenate(w) if w else np.empty((0, 3), np.int16),
    )
    # The id column rides with the payload or it is worse than useless: it would
    # keep pointing at whoever USED to occupy the slot, so every id-based check
    # silently compares the wrong particles. Found exactly that way.
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
        )
    has_ids = ds[0].get("ids") is not None
    return dict(
        dest=np.concatenate([d["dest"] for d in ds]),
        off=np.concatenate([d["off"] for d in ds]),
        w=np.concatenate([d["w"] for d in ds]),
        ids=np.concatenate([d["ids"] for d in ds]) if has_ids else None,
    )


def reconcile_velocity_scale(tile_scales):
    """The global velocity scale, from the per-tile scales the kick produced.

    The engine cannot know the new global `max|v|` before it encodes, and the
    two-pass alternative would need an O(N) float velocity buffer -- 206 GB at
    C-gh, which is exactly the array D-v2-16 clause 1 deletes. It does not need
    one: the kick already runs per tile in an O(cap) buffer, so each tile can
    take its own `max|v|/32767` there, and because ownership is a PARTITION the
    max over tiles is EXACTLY the global scale. The reduction is over a few
    thousand host scalars.

    The alternative of predicting the scale and refusing on overflow is not
    implementable here: the refusal is only detectable after the force has been
    consumed, and the force cannot be retained to retry with.

    **Cost, measured rather than assumed.** Re-expressing a tile's code at the
    global scale is a second rounding, so the RMS grows by `sqrt(1 + r^2)` with
    `r = s_tile / s_global`. The design predicted `r << 1`; at cdev8 the median
    tile sits at r = 0.40-0.80 and the 99th percentile at 0.90-0.99, because
    `max|v|` tracks the bulk flow rather than a halo core. So the cost is ~1.12x
    at the median and at most sqrt(2), against a velocity tier that passes its
    bar by ~3 orders.
    """
    s = np.asarray(list(tile_scales), dtype=np.float64)
    s = s[s > 0.0]
    return float(s.max()) if len(s) else 1.0


def drift_and_migrate(st, c_drift, vel_scale_new=None):
    """Advance every particle by `c_drift * v` and re-home it. ONE pass.

    Drift and migration are not separable once positions are bucket-relative:
    after a drift a particle may have left its bucket, and there is no valid way
    to store it where it sits, because D-007 forbids the saturating alternative.

    **Eject before insert, and it is not an optimization.** A brick is read (and
    its leavers removed) before any brick is written, so a destination's free
    capacity at write time includes its OWN departures. Inserting as leavers are
    found instead makes a brick absorb arrivals on top of a still-full run,
    which overflows to the arena in exactly the dense bricks where the arena is
    already under pressure.

    Staging is bounded by SLAB, not by N. Bricks are numbered brick-major, so a
    fixed `bx` is a contiguous block, and a particle moves at most one brick per
    axis per step -- measured at every step of a 20-step cdev8 run, max
    |delta brick| = 1 with 0.0000% over one -- so a slab's writes need only its
    own ejection and its two x-neighbours'. Peak staging is a handful of slabs,
    which scales as N^(2/3).
    """
    nb = st.bricks_per_side
    s_old = st.vel_scale
    s_new = float(vel_scale_new) if vel_scale_new else s_old

    staged, emig, inserted = {}, {}, set()
    n_over = 0
    for s in range(nb):
        staged[s], emig[s] = st._eject_slab(s, c_drift, s_old, s_new)
        # a slab may be written once it and both x-neighbours have been ejected
        for d in range(nb):
            if d in inserted:
                continue
            if all(((d + o) % nb) in emig for o in (-1, 0, 1)):
                n_over += st._insert_slab(d, staged, emig)
                inserted.add(d)
        # release what no pending write can still need
        for s2 in list(staged):
            if s2 in inserted:
                del staged[s2]
        for s2 in list(emig):
            if all(((s2 + o) % nb) in inserted for o in (-1, 0, 1)):
                del emig[s2]
    if len(inserted) != nb:
        raise AssertionError(f"{nb - len(inserted)} slabs were never written back")
    st.vel_scale = s_new
    return dict(n_arena_overflow=n_over, arena_used=st.arena_used, vel_scale=s_new)


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
    # brick -> arena rows, rebuilt on demand. NOT state: a pure function of
    # `arena_bucket`, cached because computing it per call is an O(n_arena) scan
    # and the callers are per-brick. See `arena_slots_of_brick`.
    _arena_by_brick: dict = None

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

    def brick_member_count(self, brick_flat):
        """Live rows PLUS arena residents -- the brick's true membership.

        `brick_live_count` is the run length alone, and using it where membership
        is meant undercounts by the arena population. That is the same class as
        the failure D-v2-19 clause 4 records: an arena particle still belongs to
        its brick, and forgetting it cost 98.4% of the force there on a stress
        fixture with nothing raising. Here it under-sized the tile capacity and
        the force refused to run, which is the good version of the same mistake.
        """
        return self.brick_live_count(brick_flat) + len(self.arena_slots_of_brick(brick_flat))

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

    def _invalidate_arena_index(self):
        self._arena_by_brick = None

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
        """Arena rows belonging to this brick.

        An arena particle still BELONGS to its brick -- it is only stored
        elsewhere because the brick was momentarily full -- and omitting it
        deletes it from the force with nothing raising. D-v2-19 clause 4 measured
        that at 98.4% loss on a stress fixture and 0.57% at the operating point.

        **Grouped once rather than scanned per brick.** The obvious form is
        `nonzero(arena_bucket // p3 == brick_flat)`, which is an O(n_arena) scan
        for ONE brick's answer, and the callers ask per brick: profiled at 10,240
        calls and 1.15 s of an 11.4 s step. That is the same shape as M-v2-1's
        third instrument defect, where `_to_arena` scanned the whole arena per
        particle and cost 91.00 s against a 3.97 s force. The grouping is a pure
        function of `arena_bucket`, so it is a cache and not state, and every
        write to `arena_bucket` invalidates it.
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

    def tile_bricks(self, tijk, n_tile, b_fine, n_brick, n_fine):
        """The brick ordinals covering tile+buffer. Same union and same wrap
        guard as `BrickPackedLayout.tile_members`; only the return differs, which
        is the point -- the engine wants SPANS, not particle indices."""
        from .layout import brick_span

        nb = int(n_fine) // int(n_brick)
        if nb != self.bricks_per_side:
            raise ValueError(
                f"brick grid {nb} from (n_fine={n_fine}, n_brick={n_brick}) disagrees with "
                f"the layout's {self.bricks_per_side}"
            )
        pad, span = brick_span(n_tile, b_fine, n_brick, nb)
        lo = np.asarray(tijk, dtype=np.int64) * (int(n_tile) // int(n_brick)) - pad
        out = []
        for i in range(span):
            bi = (lo[0] + i) % nb
            for j in range(span):
                bj = (lo[1] + j) % nb
                for k in range(span):
                    bk = (lo[2] + k) % nb
                    out.append((bi * nb + bj) * nb + bk)
        return out

    def decode_bricks(self, bricks):
        """(slots, x, v) over a list of bricks, concatenated.

        O(tile) floats. The tile is the largest float working set on the engine
        path, by design: a global (n,3) f64 array is 206 GB at C-gh and deleting
        both of them is D-v2-16 clause 1.
        """
        s, xs, vs = [], [], []
        for b in bricks:
            sl, x, v = self.decode_brick(b)
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
        """Write int16 velocity codes back to given slots. The kick's only
        write, and it is a scatter into contiguous spans rather than a global
        array."""
        self.w[slots] = w

    # ------------------------------------------------- drift and re-home

    def slab_bricks(self, bx):
        """The brick ordinals of one x-slab, as a contiguous range.

        Bricks are numbered `(bx * nb + by) * nb + bz`, so a fixed `bx` is a
        contiguous block -- which is what lets the pass below stage a slab as
        flat arrays with offsets instead of a dict of per-brick arrays.
        """
        nb = self.bricks_per_side
        return int(bx) * nb * nb, (int(bx) + 1) * nb * nb

    def _eject_slab(self, bx, c_drift, s_old, s_new):
        """Drift one slab's particles and split them into keepers and leavers.

        Reads the state; writes NOTHING back. That phase separation is what makes
        a double drift structurally impossible rather than merely unlikely: a
        brick is only written once every brick that can send to it has been read.

        Returns (keep, emig), each a dict of flat arrays plus the destination
        bucket ordinal, so a slab costs O(slab) rather than O(N).
        """
        lo_b, hi_b = self.slab_bricks(bx)
        p3 = self.buckets_per_brick
        k_dest, k_off, k_w, k_id = [], [], [], []
        e_dest, e_off, e_w, e_id = [], [], [], []
        for b in range(lo_b, hi_b):
            slots, x, v = self.decode_brick(b)
            if not len(slots):
                continue
            # `decode_brick` returns this brick's ARENA residents too, so they are
            # re-homed by this ejection like any other member. Their arena rows
            # are released here, where the payload is consumed -- releasing them
            # at insert instead double-counts them, which is how this was found
            # (32771 particles reachable against 32768 stored). A row is only
            # freed once its brick has been ejected, so a concurrent `_to_arena`
            # cannot claim a row that still holds live state.
            a_free = self.arena_slots_of_brick(b)
            if len(a_free):
                self.arena_bucket[a_free - self.arena_base] = -1
                self._invalidate_arena_index()
            # Drift in the INTEGER domain. The wrap is exactly modular there
            # (D-007), where `float_step_bullfrog`'s jnp.mod(x, L) is only
            # nearly so, and adding the displacement to the lattice index cannot
            # lose a small step to absorption in a large coordinate.
            q = self.t9.quantum
            i_new = np.mod(
                np.rint(x / q + (float(c_drift) * v) / q).astype(np.int64), self.t9.n_levels
            )
            b_ijk = i_new // LEVELS_PER_BUCKET
            off_new = (i_new - b_ijk * LEVELS_PER_BUCKET).astype(np.uint8)
            dest = _bucket_flat_brick_major(b_ijk, self.t9, self.bricks_per_side)
            # re-express the velocity at the new global scale (see `drift_and_migrate`)
            w_new = _rescale_w(self.w[slots], s_old, s_new)
            stay = (dest // p3) == b
            ids_b = self.ids[slots] if self.ids is not None else None
            k_dest.append(dest[stay])
            k_off.append(off_new[stay])
            k_w.append(w_new[stay])
            k_id.append(ids_b[stay] if ids_b is not None else None)
            e_dest.append(dest[~stay])
            e_off.append(off_new[~stay])
            e_w.append(w_new[~stay])
            e_id.append(ids_b[~stay] if ids_b is not None else None)
        return (
            _cat(k_dest, k_off, k_w, k_id),
            _cat(e_dest, e_off, e_w, e_id),
        )

    def _insert_slab(self, bx, staged, emig):
        """Write one slab's bricks back: keepers + immigrants + arena residents.

        Every brick's final membership passes through an O(brick) buffer here, so
        this is also where a bucket that outgrew its brick escalates -- spare,
        then arena, then a loud refusal, with no clamp anywhere (D-007).
        """
        nb = self.bricks_per_side
        p3 = self.buckets_per_brick
        lo_b, hi_b = self.slab_bricks(bx)
        keep = staged[bx]
        # immigrants can only come from this slab and its two x-neighbours: a
        # particle moves at most ONE brick per axis per step, measured at every
        # step of a 20-step cdev8 run (0.0000% over one, max |delta brick| = 1).
        sources = {(int(bx) + o) % nb for o in (-1, 0, 1)}
        imm = _cat_dicts([emig[s] for s in sorted(sources) if s in emig])
        n_over = 0
        for b in range(lo_b, hi_b):
            sel_k = keep["dest"] // p3 == b
            sel_i = imm["dest"] // p3 == b if len(imm["dest"]) else slice(0, 0)
            # NB no arena term: a brick's arena residents were decoded and
            # re-homed by its own ejection, and their rows released there.
            has_i = len(imm["dest"]) > 0
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
            w = np.concatenate(
                [
                    keep["w"][sel_k],
                    imm["w"][sel_i] if has_i else np.empty((0, 3), np.int16),
                ]
            )
            ids = None
            if self.ids is not None:
                ids = np.concatenate(
                    [
                        keep["ids"][sel_k],
                        imm["ids"][sel_i] if has_i else np.empty(0, np.int32),
                    ]
                )
            n_over += self._write_brick(b, dest, off, w, ids)
        return n_over

    def _write_brick(self, b, dest, off, w, ids=None):
        """Counting-sort one brick's members by bucket and write the run.

        Re-bucketing is a PERMUTATION, not the monotone rearrangement `repack`
        performs -- a particle can move from bucket 500 to bucket 3 -- so it needs
        an O(brick) scratch copy and a counting sort rather than a block shift.
        """
        p3 = self.buckets_per_brick
        lo, hi = self.brick_slot_range(b)
        within = dest - b * p3
        counts = np.bincount(within, minlength=p3).astype(np.int64)
        cap = hi - lo
        n_over = 0
        if len(dest) > cap:
            # the brick overflowed its allocation: the excess goes to the arena,
            # newest-bucket-first so the run stays a prefix of the bucket order
            order = np.argsort(within, kind="stable")
            keep_n = cap
            spill = order[keep_n:]
            n_over = len(spill)
            self._to_arena(
                dest[spill], off[spill], w[spill], None if ids is None else ids[spill]
            )
            order = order[:keep_n]
            within, dest, off, w = within[order], dest[order], off[order], w[order]
            if ids is not None:
                ids = ids[order]
            counts = np.bincount(within, minlength=p3).astype(np.int64)
        else:
            order = np.argsort(within, kind="stable")
            within, off, w = within[order], off[order], w[order]
            if ids is not None:
                ids = ids[order]
        m = len(off)
        self.off[lo : lo + m] = off
        self.w[lo : lo + m] = w
        if ids is not None:
            self.ids[lo : lo + m] = ids
        self.occupancy[b * p3 : (b + 1) * p3] = _to_index(counts, self.index_dtype, "migrated")
        return n_over

    def _to_arena(self, dest, off, w, ids=None):
        """Park overflow in the arena, or REFUSE. Never clamp, never drop."""
        free = np.nonzero(self.arena_bucket < 0)[0]
        if len(free) < len(dest):
            raise ValueError(
                f"{len(dest)} particles overflow their brick's capacity and the arena of "
                f"{self.n_arena} slots has only {len(free)} free. The layout does not clamp "
                "or drop (D-007). Raise brick_slack or arena_frac."
            )
        a = free[: len(dest)]
        self.arena_bucket[a] = dest
        self._invalidate_arena_index()
        self.off[self.arena_base + a] = off
        self.w[self.arena_base + a] = w
        if ids is not None:
            self.ids[self.arena_base + a] = ids

    def repack(self, brick_slack=0.10):
        """Redistribute BRICK capacity to match current occupancy.

        **Required, not an optimization.** D-v2-19 clause 3 measured that frozen
        capacity fails at every granularity -- per-brick, a collapsing halo
        outgrew even 50% spare by step 6 -- so pooling, repack and a small arena
        are all three needed and removing any one fails. Built without this, the
        engine ran ten steps at `smoke` and then hit the D-007 refusal with the
        arena full, which is the ladder doing its job and is exactly the failure
        clause 3 predicts.

        Arena residents are folded back into their brick's run here, so after a
        repack slot order IS key order with no exceptions -- which is what
        discharges the `argsort` D-v2-19's "what this does not establish" flags
        in `BrickPackedLayout.repack`.

        **The scratch is O(N), and the in-place form is owed.** D-v2-19 clause 3
        establishes that this is a MONOTONE rearrangement -- bucket order is a
        fixed spatial ordering, so restoring the layout is two in-place passes
        with O(chunk) scratch, measured at 0.13-0.52 MB independent of N. This
        implementation allocates instead, which is correct and is fine at the
        development configurations, and is 91 GB of transient at C-gh. Writing
        the in-place version is a named follow-up, not a design change.
        """
        p3 = self.buckets_per_brick
        occ = self.occupancy.astype(np.int64)
        # pull every arena resident back into its brick's count
        arena_live = np.nonzero(self.arena_bucket >= 0)[0]
        if len(arena_live):
            occ = occ + np.bincount(self.arena_bucket[arena_live], minlength=self.n_buckets)
        counts = occ.reshape(self.n_bricks, p3).sum(axis=1)
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
        for b in range(self.n_bricks):
            slots = self.brick_member_slots(b)
            if not len(slots):
                continue
            dest = self._bucket_flat_of_slots(b, slots)
            order = np.argsort(dest - b * p3, kind="stable")
            lo = int(new_start[b])
            m = len(order)
            off[lo : lo + m] = self.off[slots[order]]
            w[lo : lo + m] = self.w[slots[order]]
            if ids is not None:
                ids[lo : lo + m] = self.ids[slots[order]]
            new_occ[b * p3 : (b + 1) * p3] = np.bincount(
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

    def _bucket_flat_of_slots(self, brick_flat, slots):
        """Flat bucket ordinal per slot, run rows then arena rows."""
        m = self.brick_live_count(brick_flat)
        out = self.bucket_flat_of_live_slots(brick_flat)
        if len(slots) > m:
            a = slots[m:] - self.arena_base
            out = np.concatenate([out, self.arena_bucket[a]])
        return out

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
