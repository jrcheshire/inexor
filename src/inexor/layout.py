"""The brick-packed state layout.

WHAT THIS IS FOR. The T9 codec stores a position as a byte offset into a bucket
(`codec.py`), which only works if the bucket is known without storing it. That
is this module's whole job: keep particles ordered so that a bucket id is
implied by WHERE a particle sits, and a tile's members are contiguous to read.

THE TWO GRIDS, and why there are two.

  bucket   1.0 Mpc/h, the position codec's quantization cell. At C-gh that is
           1024^3 buckets holding ~8 particles each. Its per-bucket occupancy
           is the index the codec needs: uint32, 4.29 GB, 0.50 B/p.

           WHY uint32 AND NOT uint16, which would halve it. A uint16 ceiling of
           65535 is reachable in principle -- a 1 Mpc/h cell can sit inside a
           halo -- and nothing cheap establishes whether a production run gets
           there. Measured peaks are 5943 / 7581 / 13774 across 1x / 8x / 64x of
           volume, which is rising, but three points do not support an
           extrapolation to C-gh: a power-law fit through the tail overpredicted
           cgh64's own peak by 10x (M-v2-1, `runs/v2/m1_brick_packed_record.md`).
           So the ceiling is removed rather than measured. It costs 0.25 B/p out
           of a 1.32x margin under the GH200 host cliff, leaving 1.28x, and it
           makes the index independent of a bound nobody has established.

  brick    8 Mpc/h = 32 fine cells at C-gh, so 128^3 bricks of ~4096 particles.
           This is the FORCE's bucketing -- the union of a tile's bricks is a
           tight superset of tile+buffer -- and it is also the streaming unit:
           4096 particles x 9 B is the 37 KB run the V4e probe measured the
           host gather at (12.2 GB/s, D-v2-16 cl.6).

Buckets are ordered BRICK-MAJOR, so a brick's buckets are contiguous.

THE SLOT MODEL. A brick owns a contiguous run of slots. Its buckets are packed
TIGHT inside that run, and the spare sits at the end, pooled across the whole
brick. Bucket boundaries are therefore a prefix sum of `occupancy` within the
brick -- DERIVED, never stored.

WHY POOLED RATHER THAN PER-BUCKET, which is what D-v2-14 clause 3 ratified and
what this replaces (measurements in `runs/v2/m1_layout_record.md` and
`runs/v2/m1_brick_packed_record.md`):

  granularity   A bucket cannot be given a FRACTION of a slot, so per-bucket
                spare costs one whole slot per occupied bucket -- 12.5% of
                payload at ~8 particles per bucket, whatever the setting, since
                ceil() already returns >= 1. Pooled over a brick's ~4096
                particles, 10% means 10%.
  boundaries    A stored int64 slot boundary per bucket is 8.6 GB at C-gh, a
                full 1.00 B/p. Deriving them costs 0.002.
  measured      per-bucket ~13.4 B/p against brick-packed 10.20, on the same
                config, seed and force.

CAPACITY MUST STILL BE REDISTRIBUTED. Frozen capacity fails at EVERY
granularity -- a brick hosting a collapsing halo outgrew even 50% spare by step
6 at cdev8. `repack` handles that, and it is not a re-sort: bucket order is a
fixed spatial ordering, so restoring the layout is a monotone rearrangement,
doable in place with O(chunk) scratch. Clause 3's "a second 77 GB scatter target
is over the ceiling" objection does not apply to it.

AND A SMALL ARENA IS STILL REQUIRED. With repack every step a fixed brick
fraction still overflows, by MORE as the fraction rises (37 particles at 10%,
248 at 15%, 461 at 20%) because more spare reaches heavier clustering before
failing. But 461 of 2.1e6 is 0.02%: a rare-event problem, absorbed by a 1-2%
arena for ~0.05 B/p. Peak use measured 0.57%.

An arena particle still BELONGS to its brick, so `brick_members` returns it --
omitting it deletes it from the force with nothing raising.

OVERFLOW MAY NEVER CLAMP (D-007): spare, then arena, then a loud refusal.

Host-side numpy, deliberately: bucketing on the host is what a streamed engine
actually does, and it keeps the device's working set O(tile) rather than O(box).
"""

from dataclasses import dataclass

import numpy as np

from .codec import LEVELS_PER_BUCKET

DEFAULT_INDEX_DTYPE = np.uint32


# ============================================================================
# Brick geometry (promoted from scripts/v2_g5_core.py, unchanged in behaviour)
# ============================================================================


def choose_brick(n_tile, b_fine, n_fine):
    """Largest brick dividing n_tile, n_fine AND b_fine, with brick <= b_fine.

    Promoted verbatim from `scripts/v2_g5_core.py:821`. Bucketing on TILES and
    gathering the 27 neighbours would give a 27x superset of which only ~3.4x is
    live; bucketing on bricks makes the union of a tile's bricks a tight
    superset of tile+buffer. b_fine = 0 -> brick = n_tile.

    THE b_fine CONDITION. Without `c | b_fine` the brick union OVERSHOOTS the
    padded box, since brick_span pads by ceil(b/c) bricks and the union side
    becomes n_tile + 2*c*ceil(b/c) > n_tile + 2*b = P. Every member in the
    excess is gathered, staged, painted with zero weight and thrown away.
    Measured in the V4a card: overhang exactly 0 at all five legs where c | b
    holds, and 3,044,340,012 at the one where it does not (T128/b96 -> c = 64,
    union side 384 against P = 320), whose `cap` is inflated ~1.7x by this
    alone. With the condition enforced the union is EXACTLY the padded box,
    which is what promotes "no overhang" from a diagnostic to a contract.
    """
    if int(b_fine) <= 0:
        return int(n_tile)
    best = 1
    for c in range(1, int(b_fine) + 1):
        if int(n_tile) % c == 0 and int(n_fine) % c == 0 and int(b_fine) % c == 0:
            best = c
    return best


def brick_span(n_tile, b_fine, n_brick, nb):
    """Bricks per side covering tile+buffer, with the wrap guard.

    Promoted from `scripts/v2_g5_core.py:873`. MEASURED BUG (2026-07-15): tile
    membership walks bricks by MODULAR index, so once span > nb the same brick
    is visited twice and its particles are painted TWICE. At n_fine=64,
    n_tile=32, b=20 that gave span=6 against nb=4 and a 3.29 RELATIVE
    short-force error -- silent density corruption that reads like a
    catastrophic tiling failure rather than a bookkeeping bug. Refuse it.
    """
    pad = int(np.ceil(float(b_fine) / float(n_brick)))
    span = int(n_tile) // int(n_brick) + 2 * pad
    if span > nb:
        raise ValueError(
            f"brick span {span} > brick grid {nb}: tile+buffer wraps the box and would "
            f"double-count bricks (n_tile={n_tile}, b={b_fine}, n_brick={n_brick}). "
            "The buffer is too large for this box -- reduce beta or raise n_fine."
        )
    return pad, span


def assert_brick_divides_buffer(n_tile, b_fine, n_brick, n_fine):
    """The `c | b_fine` contract as a check rather than a comment.

    `choose_brick` guarantees it by construction; this exists so a hand-picked
    brick cannot quietly reintroduce the overhang, and so the invariant is
    testable on its own.
    """
    c = int(n_brick)
    bad = [
        name
        for name, v in (("n_tile", n_tile), ("n_fine", n_fine), ("b_fine", b_fine))
        if int(v) % c
    ]
    if bad:
        raise ValueError(
            f"brick {c} does not divide {', '.join(bad)} "
            f"(n_tile={n_tile}, b_fine={b_fine}, n_fine={n_fine}). The brick union would "
            "overshoot the padded box and every member in the excess would be gathered, "
            "staged, painted with zero weight and discarded -- measured at 3.04e9 overhang "
            "and a ~1.7x inflated cap on the one V4a leg where it held."
        )
    if c > int(b_fine) > 0:
        raise ValueError(f"brick {c} exceeds the buffer {b_fine}; the union stops being tight")


# ============================================================================
# The brick-major bucket ordering
# ============================================================================


def bucket_order_key(x, t9, bricks_per_side):
    """Per-particle sort key: the bucket's ordinal in BRICK-MAJOR order.

    Ordering buckets brick-major is what lets one array serve both grids: a
    brick's buckets land contiguously, so its CSR is a coarser prefix over the
    same slots instead of a second index.

    Returns (key, bucket_ijk, brick_flat).
    """
    nbk = t9.n_buckets_side
    per = nbk // bricks_per_side  # buckets per brick side
    b = _bucket_ijk(x, t9)
    brick = b // per
    within = b - brick * per
    brick_flat = (brick[:, 0] * bricks_per_side + brick[:, 1]) * bricks_per_side + brick[:, 2]
    within_flat = (within[:, 0] * per + within[:, 1]) * per + within[:, 2]
    return brick_flat * (per**3) + within_flat, b, brick_flat


def _bucket_ijk(x, t9):
    """Host-side per-axis bucket indices. Mirrors codec.bucket_indices but stays
    in numpy -- the layout never puts the global position array on the device."""
    i = np.mod(np.rint(np.asarray(x, dtype=np.float64) / t9.quantum).astype(np.int64), t9.n_levels)
    return (i // LEVELS_PER_BUCKET).astype(np.int64)


def bucket_ijk_from_key(key, t9, bricks_per_side):
    """Invert `bucket_order_key`: brick-major ordinal -> per-axis bucket index."""
    nbk = t9.n_buckets_side
    per = nbk // bricks_per_side
    key = np.asarray(key, dtype=np.int64)
    brick_flat, within_flat = np.divmod(key, per**3)
    bx, r = np.divmod(brick_flat, bricks_per_side * bricks_per_side)
    by, bz = np.divmod(r, bricks_per_side)
    wx, r = np.divmod(within_flat, per * per)
    wy, wz = np.divmod(r, per)
    return np.stack([bx * per + wx, by * per + wy, bz * per + wz], axis=-1)


# ============================================================================
# The layout
# ============================================================================


def _stable_sort_index(keys):
    """Stable sort permutation of non-negative integer `keys`, via a two-pass
    uint16 LSD radix. BITWISE the same permutation `np.argsort(kind="stable")`
    returns, and ~5x faster on the keys this layout actually sorts.

    WHY IT IS FASTER, measured rather than assumed. numpy's "stable" is a radix
    sort only for 1- and 2-byte integer types; for int32 and int64 it is a
    timsort/mergesort, so the bucket ordinal -- which needs 30 bits at C-gh --
    gets the O(M log M) path with poor locality. Splitting it into two uint16
    digits puts BOTH passes on numpy's radix implementation. At 16e6 keys:
    argsort int32 3063 ms, argsort int64 4423 ms, this 216 ms.

    On the layout's own inputs (2.1e6 particles, 512 bricks): `build` 87.4 ->
    17.5 ms and `migrate` 89.6 -> 17.0 ms across three steps, 4.9-5.3x, with the
    permutation identical every time.

    **NOT a general replacement, and `repack` deliberately still uses argsort.**
    Radix always does two full passes, while timsort detects existing runs and is
    O(M) on sorted input -- measured at 0.10x on an already-sorted array and
    0.29x on a constant one. `repack`'s input is slot order, which IS key order
    apart from arena residents (measured sortedness exactly 1.0000), so it is
    precisely timsort's best case and radix would make it ~10x slower. The
    merge that record calls for is still the right fix there.

    Equality with argsort was checked on seven adversarial patterns as well as
    random keys: all-identical, already-sorted, reverse-sorted, few-distinct,
    low-digit-constant, high-digit-constant, and the int32 maximum.
    """
    k = np.asarray(keys)
    if k.size == 0:
        return np.empty(0, dtype=np.int64)
    if not np.issubdtype(k.dtype, np.integer):
        raise TypeError(f"keys must be an integer dtype, got {k.dtype}")
    hi_max = int(k.max())
    if int(k.min()) < 0 or hi_max >= 2**32:
        raise ValueError(
            f"keys must lie in [0, 2^32) for the two-digit radix, got "
            f"[{int(k.min())}, {hi_max}]. A negative key means the bucket ordinal "
            "has already wrapped -- see _refuse_key_overflow."
        )
    order = np.argsort((k & 0xFFFF).astype(np.uint16), kind="stable")
    return order[np.argsort((k[order] >> 16).astype(np.uint16), kind="stable")]


def _within_run_index(counts):
    """0,1,..,c-1 concatenated over counts -- each particle's position inside
    its bucket's run. Written the obvious way rather than with the cumsum
    trick: this is correctness-critical bookkeeping and the clever form is one
    off-by-one away from silently permuting the state."""
    counts = np.asarray(counts, dtype=np.int64)
    starts = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    return np.arange(int(starts[-1]), dtype=np.int64) - np.repeat(starts[:-1], counts)


def _prefix_mask(counts, fits):
    """Boolean mask over group-sorted items selecting the first `fits[g]` of
    each group of size `counts[g]`."""
    within = _within_run_index(counts)
    return within < np.repeat(fits, counts)


def _to_index(counts, dtype, what):
    """Narrow bucket counts to the stored index dtype, REFUSING overflow.

    EVERY write to `occupancy` goes through here, which is the point. numpy
    narrows modularly, so a bare `.astype(uint16)` turns a bucket of 65,536 into
    an occupancy of 0 and one of 70,000 into 4,464 -- and because occupancy IS
    the bucket-boundary prefix sum, that does not merely misreport one bucket, it
    shifts the derived span of every LATER bucket in that brick. `check()`
    samples three bricks, so it would very likely pass.

    Until 2026-08-08 only `build` was guarded; `migrate` and `repack` narrowed
    bare, so the guarded path was the one where overflow is least likely and the
    two that run every step were silent. That is the defect this replaces, and
    it is why the refusal lives at the cast rather than beside it.
    """
    counts = np.asarray(counts)
    hot = int(counts.max()) if counts.size else 0
    limit = int(np.iinfo(dtype).max)
    if hot > limit:
        raise ValueError(
            f"{what} bucket count {hot} exceeds the {np.dtype(dtype).name} index ceiling "
            f"({limit}). The index does not wrap and does not clamp (D-007): occupancy "
            "doubles as the derived bucket boundaries, so a modular narrowing would "
            "silently relocate every later bucket in the brick. Widen index_dtype."
        )
    return counts.astype(dtype)


def _refuse_key_overflow(n_buckets):
    """`key` is int32, so the bucket grid must fit it. C-hero does not.

    The same silent-wrap class as `_to_index`, one config-table rung away: at
    C-hero (4096^3 particles, bucket_cells 2) the grid is 2048^3 = 8.59e9 buckets
    against an int32 max of 2.15e9, and a bare narrowing sends the high buckets
    to NEGATIVE ordinals.

    It is refused rather than widened, because widening is the wrong fix. `key`
    is a per-particle cache -- 34.4 GB at C-gh as int32, 68.7 as int64, against
    ~91 GB of state on a ~116 GB host -- so it cannot be resident at production
    scale in EITHER width, and neither can `particle_to_slot` (68.7 GB) or
    `slot_to_particle` (75.6 GB). All three are scaffolding for a probe that
    keeps positions in their original order and uses the layout as an index into
    them; the streamed engine stores state IN slot order, where a particle's
    bucket is implied by where it sits and none of the three exists. Making that
    ceiling loud is the fix available today; removing it is M-v2-3/M-v2-6's job.
    """
    if int(n_buckets) > np.iinfo(np.int32).max:
        raise ValueError(
            f"bucket grid {int(n_buckets)} exceeds int32 ({np.iinfo(np.int32).max}), which is "
            "what `key` is stored in -- the high buckets would narrow to negative ordinals "
            "silently. This bites at C-hero. Note the fix is NOT a wider key: at this scale "
            "the per-particle bookkeeping arrays (key, particle_to_slot, slot_to_particle) "
            "are ~21 B/p against ~10.5 B/p of state and cannot be resident at all, so the "
            "layout has to stop materializing them first."
        )


# ============================================================================
# Brick-packed layout: spare at BRICK granularity (the M-v2-1 alternative)
# ============================================================================
#
# WHY. The superseded per-bucket layout gave every bucket its own spare slots,
# and a bucket cannot
# be given a FRACTION of a slot -- so any nonzero slack costs one whole slot per
# occupied bucket, 12.5% of payload at the ratified ~8 particles per bucket.
# Measured, that floor is what makes the sweep bottom out at 12.38 B/p against
# a 10.15 budget, and it is granularity rather than a tunable: the `min_spare`
# rule that looked like the cause is INERT, since ceil() already returns >= 1.
#
# Here the spare belongs to the BRICK (~512 buckets, ~4096 particles at C-gh)
# and buckets are packed TIGHT inside it. One shared slot amortizes to 1/4096
# rather than 1/8. A bucket that grows takes room from the brick's pool by
# shifting its neighbours along -- at most a brick's worth of entries inside a
# 37 KB block, which is the same local move `repack` already performs.
#
# TWO TERMS FALL, NOT ONE.
#
#   granularity   12.5% -> ~0.02%, because the spare unit is 512x larger
#   bucket_start  GONE. The per-bucket layout carried an int64 slot boundary per
#                 bucket: 8 B per bucket is 8.6 GB at C-gh, a full 1.00 B/p
#                 that the record's all-in figures never counted. Here bucket
#                 boundaries are a prefix sum of `occupancy` WITHIN a brick, so
#                 the index we already pay 0.50 B/p for does both jobs.
#
# AND THE FLUCTUATION IS SMALLER, which is the physical reason to expect this to
# work at all. Bucket occupancy grew 76x over a run (8 -> 5943 at cdev8) because
# a 1 Mpc/h cell can sit inside a halo. A brick averages over 512x the volume, so
# its count cannot concentrate the same way -- and it is the brick's total, not
# any bucket's, that has to fit its allocation now.
#
# Both arrays that make this work are still O(1) per brick rather than per
# bucket, so the index cost does not come back somewhere else.


@dataclass
class BrickPackedLayout:
    """Buckets packed tight inside bricks; spare slots pooled per brick."""

    t9: object
    bricks_per_side: int
    brick_start: np.ndarray  # int64 (n_bricks+1,) fixed slot runs
    occupancy: np.ndarray  # uint32 (n_buckets,) THE index; also bucket bounds
    slot_to_particle: np.ndarray  # int64 (n_slots,) -1 where free
    particle_to_slot: np.ndarray  # int64 (n,)   transient bookkeeping
    key: np.ndarray  # int32 (n,)   transient: current bucket ordinal
    n_particles: int
    arena_base: int = 0
    # Explicit bucket id per arena resident, because an arena particle is by
    # definition NOT inside its bucket's derived span -- the one place the
    # occupancy-as-boundaries trick does not reach. 4 B each, for a population
    # measured at ~0.01% of N, so it does not disturb the tier.
    arena_bucket: np.ndarray = None

    @classmethod
    def build(cls, x, t9, bricks_per_side, brick_slack=0.10, alloc_margin=0.10,
              arena_frac=0.01, index_dtype=DEFAULT_INDEX_DTYPE):
        nbk = t9.n_buckets_side
        if nbk % int(bricks_per_side):
            raise ValueError(
                f"bricks_per_side {bricks_per_side} must divide the bucket grid {nbk}"
            )
        per3 = (nbk // int(bricks_per_side)) ** 3
        n_bricks = int(bricks_per_side) ** 3
        _refuse_key_overflow(n_bricks * per3)
        key, _, _ = bucket_order_key(x, t9, int(bricks_per_side))
        brick = key // per3

        brick_counts = np.bincount(brick, minlength=n_bricks).astype(np.int64)
        occupancy = np.bincount(key, minlength=n_bricks * per3).astype(np.int64)

        spare = np.ceil(brick_counts * float(brick_slack)).astype(np.int64)
        spare = np.where(brick_counts > 0, np.maximum(spare, 1), spare)
        brick_start = np.zeros(n_bricks + 1, dtype=np.int64)
        np.cumsum(brick_counts + spare, out=brick_start[1:])

        order = _stable_sort_index(key)
        rank = _within_run_index(brick_counts)  # position inside the brick's run
        slots = brick_start[brick[order]] + rank
        n_alloc = int(np.ceil(int(brick_start[-1]) * (1.0 + float(alloc_margin))))
        n_arena = int(np.ceil(len(key) * float(arena_frac)))
        slot_to_particle = np.full(n_alloc + n_arena, -1, dtype=np.int64)
        slot_to_particle[slots] = order
        particle_to_slot = np.empty(len(key), dtype=np.int64)
        particle_to_slot[order] = slots
        return cls(
            t9=t9,
            bricks_per_side=int(bricks_per_side),
            brick_start=brick_start,
            occupancy=_to_index(occupancy, index_dtype, "initial"),
            slot_to_particle=slot_to_particle,
            particle_to_slot=particle_to_slot,
            key=key.astype(np.int32),
            n_particles=len(key),
            arena_base=n_alloc,
            arena_bucket=np.full(n_arena, -1, dtype=np.int64),
        )

    @property
    def index_dtype(self):
        """The stored index dtype, read off the array rather than kept as a
        separate field so the two cannot disagree about what is in force."""
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

    def brick_slot_range(self, brick_flat):
        return int(self.brick_start[brick_flat]), int(self.brick_start[brick_flat + 1])

    def bucket_slot_starts(self, brick_flat):
        """Bucket boundaries inside ONE brick, derived rather than stored.

        This is the array the per-bucket layout kept globally at 8 B per bucket. Here it
        is a prefix sum over the brick's own occupancy slice -- O(512) at C-gh,
        computed where it is needed and never resident.
        """
        p3 = self.buckets_per_brick
        occ = self.occupancy[brick_flat * p3 : (brick_flat + 1) * p3].astype(np.int64)
        out = np.zeros(p3 + 1, dtype=np.int64)
        np.cumsum(occ, out=out[1:])
        return int(self.brick_start[brick_flat]) + out

    def brick_members(self, brick_flat):
        """Live particles belonging to a brick, INCLUDING its arena residents.

        The run alone is not the brick's membership. An arena particle still
        belongs to this brick -- it is only stored elsewhere because the brick
        was momentarily full -- and omitting it would silently drop it from the
        force. At the measured peak that is 0.57% of particles vanishing from
        gravity with nothing raising, which is why this is not an optimization
        detail.
        """
        lo, hi = self.brick_slot_range(brick_flat)
        run = self.slot_to_particle[lo:hi]
        out = run[run >= 0]
        if self.arena_bucket is not None and len(self.arena_bucket):
            # free arena slots carry -1, and -1 // p3 is -1, so they never match
            sel = np.nonzero(self.arena_bucket // self.buckets_per_brick == brick_flat)[0]
            if len(sel):
                out = np.concatenate([out, self.slot_to_particle[self.arena_base + sel]])
        return out

    def tile_members(self, tijk, n_tile, b_fine, n_brick, n_fine):
        """Live particle indices covering tile+buffer, as the brick union.

        Same union and same wrap guard as the ratified probe's
        `v2_g5_core.tile_members`; only the storage differs. Verified equal to it
        elementwise on matched (quantized) inputs.
        """
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
                    out.append(self.brick_members((bi * nb + bj) * nb + bk))
        return np.concatenate(out) if out else np.empty(0, dtype=np.int64)

    def migrate(self, x_new):
        """Rebuild every brick whose contents changed.

        A brick is the unit of work: within one, buckets are packed tight, so a
        particle changing bucket moves its neighbours' boundaries and the run has
        to be rewritten. Bricks nobody entered or left or moved inside are never
        touched.
        """
        key_new, _, _ = bucket_order_key(x_new, self.t9, self.bricks_per_side)
        key_new = key_new.astype(np.int32)
        changed = np.nonzero(key_new != self.key)[0]
        p3 = self.buckets_per_brick
        stats = dict(
            bucket_migrant_frac=float(len(changed)) / max(self.n_particles, 1),
            # aliases so a caller can record either layout with one code path
            migrant_frac=float(len(changed)) / max(self.n_particles, 1),
            arena_used=0,
            n_full_buckets=0,
            n_overflow=0,
            overflow_frac=0.0,
        )
        if len(changed) == 0:
            stats["brick_migrant_frac"] = 0.0
            stats["max_brick_fill"] = float(
                np.max(np.bincount(self.key // p3, minlength=self.n_bricks) / np.maximum(
                    np.diff(self.brick_start), 1))
            )
            stats["max_fill_frac"] = stats["max_brick_fill"]
            return stats

        old_brick, new_brick = self.key // p3, key_new // p3
        stats["brick_migrant_frac"] = float(np.sum(old_brick != new_brick)) / self.n_particles
        affected = np.unique(np.concatenate([old_brick[changed], new_brick[changed]]))

        # Membership by lookup table rather than np.isin. `affected` can be most
        # of the brick grid (measured: 512 of 512 by the first step), and isin
        # falls back to a sort-based path at that size; a bool LUT over n_bricks
        # is one fancy-index pass and 2.1 MB at C-gh. Measured 34.0 -> 9.8 ms at
        # 16e6 rows. NB this is the SMALL term -- the sort below is 5-10x it, so
        # the record's "argsort + isin" attribution overstates isin's share.
        is_affected = np.zeros(self.n_bricks, dtype=bool)
        is_affected[affected] = True
        claim = np.nonzero(is_affected[new_brick])[0]
        claim = claim[_stable_sort_index(key_new[claim])]
        cb = new_brick[claim]

        counts = np.bincount(cb, minlength=self.n_bricks)[affected]
        cap = np.diff(self.brick_start)[affected]
        over = counts - cap
        fits = np.minimum(counts, cap)
        stats["n_overflow"] = int(np.maximum(over, 0).sum())
        stats["overflow_frac"] = stats["n_overflow"] / self.n_particles
        stats["n_overflow_bricks"] = int(np.sum(over > 0))

        # clear the affected runs, then write them back packed
        lens = cap
        self.slot_to_particle[
            np.repeat(self.brick_start[affected], lens) + _within_run_index(lens)
        ] = -1
        # drop any stale arena residents belonging to affected bricks
        if self.arena_bucket is not None and len(self.arena_bucket):
            stale = np.isin(self.arena_bucket // p3, affected)
            if np.any(stale):
                idx = np.nonzero(stale)[0]
                self.slot_to_particle[self.arena_base + idx] = -1
                self.arena_bucket[idx] = -1

        keep = _prefix_mask(counts, fits)
        placed = claim[keep]
        slots = self.brick_start[cb[keep]] + _within_run_index(fits)
        self.slot_to_particle[slots] = placed
        self.particle_to_slot[placed] = slots
        spill, spill_b = claim[~keep], key_new[claim[~keep]]
        if len(spill):
            free = np.nonzero(self.arena_bucket < 0)[0]
            if len(free) < len(spill):
                raise ValueError(
                    f"{len(spill)} particles overflow their brick's capacity and the arena "
                    f"of {len(self.arena_bucket)} slots has only {len(free)} free. The layout "
                    "does not clamp or drop (D-007). Raise brick_slack or arena_frac."
                )
            a = free[: len(spill)]
            self.arena_bucket[a] = spill_b
            self.slot_to_particle[self.arena_base + a] = spill
            self.particle_to_slot[spill] = self.arena_base + a
        stats["arena_used"] = int(np.sum(self.arena_bucket >= 0))
        # `occupancy` must count what is IN the brick runs, because it is what
        # the derived bucket boundaries are built from -- an arena resident is
        # NOT in its bucket's span, so counting it would shift every later
        # bucket in that brick.
        occ = np.bincount(key_new, minlength=self.n_bricks * p3)
        if len(spill):
            np.subtract.at(occ, spill_b, 1)
        self.occupancy = _to_index(occ, self.index_dtype, "migrated")
        self.key = key_new
        fill = np.bincount(new_brick, minlength=self.n_bricks) / np.maximum(
            np.diff(self.brick_start), 1
        )
        stats["max_brick_fill"] = float(fill.max())
        stats["max_fill_frac"] = stats["max_brick_fill"]
        return stats

    def repack(self, brick_slack=0.10, chunk=1 << 20):
        """Redistribute BRICK capacity in place, by a monotone rearrangement.

        Needed for the same reason: capacity frozen at build time cannot track
        structure formation. Measured, the problem recurs at brick level -- a
        brick hosting a halo outgrew even 50% spare by step 6 at cdev8 -- so
        "bricks average over 512x the volume and therefore cannot concentrate"
        was too optimistic, and this is what corrects it.

        It is CHEAPER here than per-bucket, because only brick boundaries move:
        n_bricks is 512x smaller than n_buckets, and the particles inside a
        brick keep their relative order, so the rearrangement is a block shift.
        """
        counts = np.bincount(self.key // self.buckets_per_brick, minlength=self.n_bricks)
        counts = counts.astype(np.int64)
        spare = np.ceil(counts * float(brick_slack)).astype(np.int64)
        spare = np.where(counts > 0, np.maximum(spare, 1), spare)
        new_start = np.zeros(self.n_bricks + 1, dtype=np.int64)
        np.cumsum(counts + spare, out=new_start[1:])
        n_alloc = len(self.slot_to_particle)
        if int(new_start[-1]) > n_alloc:
            raise ValueError(
                f"repack needs {int(new_start[-1])} slots against {n_alloc} allocated"
            )

        live = np.nonzero(self.slot_to_particle >= 0)[0]
        parts = self.slot_to_particle[live]
        # ARENA RESIDENTS ARE OUT OF ORDER. The main runs are already in
        # (brick, bucket) order, so slot order is key order -- but an arena
        # particle sits past every brick run, so it arrives LAST regardless of
        # which bucket it belongs to. Placing by slot order would scatter it
        # into the wrong bucket's span, which is exactly what `check` caught at
        # the first step that overflowed. Re-order by key; stable, so particles
        # sharing a bucket keep their relative order.
        # argsort, NOT `_stable_sort_index`, and that is measured rather than an
        # oversight: slot order IS key order apart from the arena residents, so
        # this is timsort's best case (measured sortedness exactly 1.0000) where
        # radix is ~10x SLOWER because it always makes two full passes. The real
        # fix here is still a merge of the few out-of-order residents into an
        # already-sorted run, which is O(arena) rather than O(N log N).
        if np.any(self.slot_to_particle[self.arena_base :] >= 0):
            parts = parts[np.argsort(self.key[parts], kind="stable")]
        n_live = len(parts)
        scratch_peak = 0
        # compact forward, from the ordered particle list
        for i in range(0, n_live, chunk):
            j = min(i + chunk, n_live)
            buf = parts[i:j]
            scratch_peak = max(scratch_peak, buf.nbytes)
            self.slot_to_particle[i:j] = buf
        self.slot_to_particle[n_live:] = -1
        # expand backward into the new brick runs
        final = np.repeat(new_start[:-1], counts) + _within_run_index(counts)
        for i in range(n_live, 0, -chunk):
            lo = max(i - chunk, 0)
            buf = self.slot_to_particle[lo:i].copy()
            scratch_peak = max(scratch_peak, buf.nbytes)
            self.slot_to_particle[lo:i] = -1
            self.slot_to_particle[final[lo:i]] = buf
        self.brick_start = new_start
        self.particle_to_slot[parts] = final
        if self.arena_bucket is not None:
            self.arena_bucket[:] = -1  # everyone is back in a brick run
        self.occupancy = _to_index(
            np.bincount(self.key, minlength=self.n_bricks * self.buckets_per_brick),
            self.index_dtype,
            "repacked",
        )
        return dict(
            scratch_bytes=int(scratch_peak),
            slots_used=int(new_start[-1]),
            slots_allocated=int(n_alloc),
            max_brick_fill=float(np.max(counts / np.maximum(counts + spare, 1))),
        )

    def check(self):
        live = self.slot_to_particle >= 0
        if int(np.sum(live)) != self.n_particles:
            raise AssertionError(f"lost particles: {int(np.sum(live))} vs {self.n_particles}")
        p = self.slot_to_particle[live]
        if len(np.unique(p)) != len(p):
            raise AssertionError("a particle occupies two slots")
        if not np.array_equal(self.particle_to_slot[p], np.nonzero(live)[0]):
            raise AssertionError("particle_to_slot disagrees with slot_to_particle")
        # every particle must sit inside its OWN bucket's derived span
        p3 = self.buckets_per_brick
        for b in (0, self.n_bricks // 2, self.n_bricks - 1):
            starts = self.bucket_slot_starts(b)
            lo, _ = self.brick_slot_range(b)
            run = self.slot_to_particle[lo : starts[-1]]
            if np.any(run < 0):
                raise AssertionError(f"brick {b} has a hole inside its packed prefix")
            want = np.repeat(np.arange(b * p3, (b + 1) * p3), np.diff(starts))
            if not np.array_equal(self.key[run], want.astype(np.int32)):
                raise AssertionError(f"brick {b}: a particle is outside its bucket's span")
        return True

    def bytes_per_particle(self, payload=9.0):
        """The terms D-v2-14 clause 2's all-in figure is written in.

        `scaffold` is reported BESIDE the total and deliberately not inside it.
        It is what this object carries that a production engine must not: `key`,
        `particle_to_slot` and `slot_to_particle` exist because the probe keeps
        positions in their original order and treats the layout as an index into
        them, whereas the streamed engine stores state IN slot order, where a
        particle's bucket is implied by where it sits. It is reported because
        ~21 B/p of uncounted arrays sitting next to a 10.5 B/p budget should be
        visible rather than inferred -- the same reason the superseded record's
        missing 1.00 B/p `bucket_start` term mattered.
        """
        n = max(self.n_particles, 1)
        scaffold = (
            self.key.nbytes + self.particle_to_slot.nbytes + self.slot_to_particle.nbytes
        ) / n
        return dict(
            payload=payload,
            bucket_index=self.occupancy.nbytes / n,
            brick_start=self.brick_start.nbytes / n,
            slack=(self.n_slots - self.n_particles) * payload / n,
            total=payload
            + self.occupancy.nbytes / n
            + self.brick_start.nbytes / n
            + (self.n_slots - self.n_particles) * payload / n,
            scaffold=scaffold,
        )
