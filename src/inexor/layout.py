"""Brick-packed slot layout for the T9 codec's bucket-relative positions.

The codec (`codec.py`) stores a position as an offset inside a 1 Mpc/h bucket, so the bucket
must be implied by where a particle sits. This module keeps particles ordered so it is.

Two grids: the BUCKET (the codec's quantization cell; `occupancy` counts per bucket, uint32
so no halo can overflow it) and the BRICK (the force's tiling and streaming unit; the union
of a tile's bricks is exactly tile+buffer). Buckets are ordered brick-major, so a brick's
buckets are contiguous.

A brick owns a contiguous run of slots; its buckets are packed tight and the spare is pooled
at the end of the run. Bucket boundaries are the prefix sum of `occupancy` within the brick,
derived and never stored. `repack` redistributes brick capacity (a monotone rearrangement,
not a re-sort); a small arena absorbs brick overflow between repacks. An arena particle
still belongs to its brick and `brick_members` returns it.

Overflow never clamps (wrap-never-clamp): spare, then arena, then a ValueError. Host-side
numpy so the device working set stays O(tile).
"""

from dataclasses import dataclass

import numpy as np

from .codec import LEVELS_PER_BUCKET

DEFAULT_INDEX_DTYPE = np.uint32


# ============================================================================
# Brick geometry
# ============================================================================


def choose_brick(n_tile, b_fine, n_fine):
    """Largest brick side (fine cells) dividing n_tile, n_fine and b_fine, <= b_fine.

    Requiring `brick | b_fine` makes the union of a tile's bricks EXACTLY the padded box
    n_tile + 2*b_fine; otherwise it overshoots and the excess is gathered and painted with
    zero weight for nothing. b_fine = 0 -> brick = n_tile.
    """
    if int(b_fine) <= 0:
        return int(n_tile)
    best = 1
    for c in range(1, int(b_fine) + 1):
        if int(n_tile) % c == 0 and int(n_fine) % c == 0 and int(b_fine) % c == 0:
            best = c
    return best


def brick_span(n_tile, b_fine, n_brick, nb):
    """Return (pad, span): buffer bricks per side and bricks per side covering tile+buffer.

    Refuses span > nb: tile membership walks bricks by modular index, so a wrapping span
    would visit (and paint) the same brick twice.
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
    """Raise unless `n_brick` divides n_tile, n_fine and b_fine and does not exceed b_fine.

    `choose_brick` guarantees this; the check guards hand-picked bricks.
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
    """Per-particle sort key: the bucket's ordinal in brick-major order.

    Brick-major order makes a brick's buckets contiguous, so one slot array serves both
    grids. Returns (key, bucket_ijk, brick_flat).
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
    """Host-side per-axis bucket indices; numpy mirror of codec.bucket_indices."""
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
    """Stable sort permutation of integer `keys` in [0, 2^32), via two uint16 LSD radix passes.

    Bitwise the permutation `np.argsort(kind="stable")` returns. Faster on unsorted keys
    because numpy's stable sort is a radix sort only for 1- and 2-byte integers; slower on
    nearly sorted input, where timsort is O(M), which is why `repack` uses argsort.
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
    """0,1,..,c-1 concatenated over `counts`: each item's position inside its run."""
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
    """Narrow bucket counts to the index dtype, refusing overflow.

    Every write to `occupancy` must go through here: numpy narrows modularly, and since
    occupancy is the bucket-boundary prefix sum, a wrapped count would shift every later
    bucket in the brick.
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
    """Refuse a bucket grid that does not fit the int32 `key`.

    Narrowing would wrap high buckets to negative ordinals. Widening is not offered: `key`,
    `particle_to_slot` and `slot_to_particle` are per-particle bookkeeping that cannot be
    resident at production scale anyway; the streamed engine (`state.py`) stores state in
    slot order and has none of them.
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
# Brick-packed layout
# ============================================================================


@dataclass
class BrickPackedLayout:
    """Buckets packed tight inside bricks; spare slots pooled per brick.

    Host-side index over positions kept in their original order: `key`,
    `particle_to_slot` and `slot_to_particle` are per-particle bookkeeping. Use `build`.
    """

    t9: object
    bricks_per_side: int
    brick_start: np.ndarray  # int64 (n_bricks+1,) fixed slot runs
    occupancy: np.ndarray  # uint32 (n_buckets,) THE index; also bucket bounds
    slot_to_particle: np.ndarray  # int64 (n_slots,) -1 where free
    particle_to_slot: np.ndarray  # int64 (n,)   transient bookkeeping
    key: np.ndarray  # int32 (n,)   transient: current bucket ordinal
    n_particles: int
    arena_base: int = 0
    # bucket id per arena slot (-1 free): an arena resident is outside its bucket's derived span
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
        """The stored index dtype, read off `occupancy`."""
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
        """Slot boundaries of the buckets in one brick: brick start + prefix sum of occupancy."""
        p3 = self.buckets_per_brick
        occ = self.occupancy[brick_flat * p3 : (brick_flat + 1) * p3].astype(np.int64)
        out = np.zeros(p3 + 1, dtype=np.int64)
        np.cumsum(occ, out=out[1:])
        return int(self.brick_start[brick_flat]) + out

    def brick_members(self, brick_flat):
        """Live particles belonging to a brick, including its arena residents.

        Omitting arena residents would silently drop them from the force.
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


    def migrate(self, x_new):
        """Move particles to their buckets at positions `x_new`; return migration stats.

        Every brick a particle entered, left or moved within is rewritten packed; others
        are untouched. Brick overflow spills to the arena; an exhausted arena raises.
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

        # bool LUT rather than np.isin: `affected` is often most of the brick grid
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
        # occupancy counts only brick-run residents: it defines the derived bucket boundaries
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
        """Resize every brick run to count*(1+brick_slack) and empty the arena.

        Bucket order is a fixed spatial ordering, so this is a monotone rearrangement done in
        place in `chunk`-sized pieces (compact forward, expand backward). Returns stats.
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
        # Slot order is key order except for arena residents, which sit past every run; re-sort
        # stably by key. argsort, not the radix: on nearly sorted input timsort is O(N).
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
        """Per-particle byte budget: payload, index, brick_start, slack and their total.

        `scaffold` (key, particle_to_slot, slot_to_particle) is reported beside the total, not
        in it: a slot-ordered engine does not carry those arrays.
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
