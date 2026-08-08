"""The brick-sorted state layout (D-v2-14 clause 3).

WHAT THIS IS FOR. The T9 codec stores a position as a byte offset into a bucket
(`codec.py`), which only works if the bucket is known without storing it. That
is this module's whole job: keep particles ordered so that a bucket id is
implied by WHERE a particle sits, and a tile's members are contiguous to read.

THE TWO GRIDS, and why there are two.

  bucket   1.0 Mpc/h, the position codec's quantization cell. At C-gh that is
           1024^3 buckets holding ~8 particles each. Its per-bucket occupancy
           is the index the codec needs: uint16, 2.15 GB, the 0.25 B/p line of
           the 10.15 B/p all-in figure.

  brick    8 Mpc/h = 32 fine cells at C-gh, so 128^3 bricks of ~4096 particles.
           This is the FORCE's bucketing -- the union of a tile's bricks is a
           tight superset of tile+buffer -- and it is also the streaming unit:
           4096 particles x 9 B is the 37 KB run the V4e probe measured the
           host gather at (12.2 GB/s, the binding constraint, D-v2-16 cl.6).
           Its CSR is 16 B/brick, 0.004 B/p.

Buckets are ordered BRICK-MAJOR, so a brick's buckets are contiguous and the
brick CSR is a coarser prefix over the same slot array rather than a second
index. One ordering serves both grids.

THE SLOT MODEL, and why it is not a dense sort.

  Each bucket owns a contiguous run of `capacity` slots and fills them from the
  bottom; `occupancy` says how many are live, so free space is always the TAIL
  of a bucket's run and there are no tombstones and no free list. Capacity is
  per-bucket (D-v2-14 cl.3), not a global max over tiles.

  This is what makes migration affordable. A full re-sort every step needs a
  second scatter target the size of the state -- 77 GB at C-gh, over the ~116 GB
  host cliff (D-v2-13). So the operation is EJECT-AND-REINSERT: a particle
  leaving bucket b swaps with b's last live slot and b's occupancy drops; it
  then takes the first free slot of b'. Only the migrants move.

  The slack that absorbs migration and the `cap` padding the force already pays
  are THE SAME BUFFER -- a brick is streamed as one contiguous slot run,
  including the free tails, exactly as a padded tile is. D-v2-14 clause 2 prices
  it at 0.90 B/p, 10% of payload, and says plainly that this is an ESTIMATE from
  a hand argument about migration rates. Measuring it at cgh64 is the exit
  condition of M-v2-1; nothing here should be read as having measured it.

OVERFLOW MAY NEVER CLAMP (D-007). A bucket that will not fit escalates: slack,
then a shared arena, then a loud refusal. Dropping or clamping a particle would
silently delete mass, and the particles that overflow are the clustered ones --
precisely the ones a halo-grade mock exists to resolve.

Host-side numpy, deliberately, following the probe: bucketing on the host is
what a streamed engine actually does, and it is what keeps the device's working
set O(tile) rather than O(box).
"""

from dataclasses import dataclass

import numpy as np

from .codec import LEVELS_PER_BUCKET

UINT16_MAX = 65535


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


@dataclass
class BrickLayout:
    """Slotted, brick-sorted particle layout.

    Mutable by design: `migrate` edits it in place, which is the point -- a
    frozen layout would mean rebuilding, and rebuilding is the 77 GB scatter
    D-v2-14 clause 3 rules out.
    """

    t9: object  # codec.T9Layout
    bricks_per_side: int
    bucket_start: np.ndarray  # int64 (n_buckets+1,) slot range per bucket
    occupancy: np.ndarray  # uint16 (n_buckets,)  <- the 0.25 B/p index
    slot_to_particle: np.ndarray  # int64 (n_slots,) -1 where free
    particle_to_slot: np.ndarray  # int64 (n,)  transient; not resident state
    arena_slot_bucket: np.ndarray  # int64 (n_arena,) -1 where free
    arena_base: int
    n_particles: int

    # ---------------------------------------------------------------- build

    @classmethod
    def build(cls, x, t9, bricks_per_side, slack_frac=0.10, arena_frac=0.01):
        """Sort positions into the slot layout.

        slack_frac is D-v2-14 clause 2's 0.90 B/p estimate expressed as a
        fraction of payload; it is a knob here precisely because it is not yet
        measured. arena_frac sizes the shared overflow region.
        """
        t9_n_buckets_side = t9.n_buckets_side
        if t9_n_buckets_side % int(bricks_per_side):
            raise ValueError(
                f"bricks_per_side {bricks_per_side} must divide the bucket grid "
                f"{t9_n_buckets_side}; otherwise a brick's buckets are not contiguous in "
                "brick-major order and one array cannot serve both grids"
            )
        if slack_frac < 0.0 or arena_frac < 0.0:
            raise ValueError("slack_frac and arena_frac must be non-negative")

        key, _, _ = bucket_order_key(x, t9, int(bricks_per_side))
        n_buckets = t9.n_buckets
        counts = np.bincount(key, minlength=n_buckets).astype(np.int64)
        _refuse_uint16_overflow(counts, "initial")

        # per-bucket capacity: its own count plus slack, at least one spare slot
        # wherever anything lives, so a single arrival never needs the arena
        extra = np.ceil(counts * float(slack_frac)).astype(np.int64)
        extra = np.where(counts > 0, np.maximum(extra, 1), extra)
        capacity = counts + extra
        _refuse_uint16_overflow(capacity, "capacity")

        bucket_start = np.zeros(n_buckets + 1, dtype=np.int64)
        np.cumsum(capacity, out=bucket_start[1:])
        n_slots = int(bucket_start[-1])

        order = np.argsort(key, kind="stable")
        slot_to_particle = np.full(n_slots, -1, dtype=np.int64)
        particle_to_slot = np.empty(len(key), dtype=np.int64)
        # each bucket fills from the bottom of its run
        run_pos = np.repeat(bucket_start[:-1], counts) + _within_run_index(counts)
        slot_to_particle[run_pos] = order
        particle_to_slot[order] = run_pos

        n_arena = int(np.ceil(len(key) * float(arena_frac)))
        return cls(
            t9=t9,
            bricks_per_side=int(bricks_per_side),
            bucket_start=bucket_start,
            occupancy=counts.astype(np.uint16),
            slot_to_particle=np.concatenate(
                [slot_to_particle, np.full(n_arena, -1, dtype=np.int64)]
            ),
            particle_to_slot=particle_to_slot,
            arena_slot_bucket=np.full(n_arena, -1, dtype=np.int64),
            arena_base=n_slots,
            n_particles=len(key),
        )

    # ------------------------------------------------------------- geometry

    @property
    def n_buckets(self):
        return len(self.occupancy)

    @property
    def n_slots(self):
        return int(self.bucket_start[-1])

    @property
    def capacity(self):
        return np.diff(self.bucket_start)

    @property
    def buckets_per_brick_side(self):
        return self.t9.n_buckets_side // self.bricks_per_side

    def brick_slot_range(self, brick_flat):
        """(lo, hi) slot bounds of a brick -- ONE contiguous run, free tails
        included. That run is the streaming unit: 37 KB at C-gh, which is the
        span the pinned-gather measurement was taken at."""
        per3 = self.buckets_per_brick_side**3
        b0 = int(brick_flat) * per3
        return int(self.bucket_start[b0]), int(self.bucket_start[b0 + per3])

    def brick_members(self, brick_flat):
        """Live particle indices in a brick, read as one contiguous slot run
        and masked. The dead rows are the slack, and they are the same dead
        rows the force already pays for as `cap` padding."""
        lo, hi = self.brick_slot_range(brick_flat)
        run = self.slot_to_particle[lo:hi]
        return run[run >= 0]

    def tile_members(self, tijk, n_tile, b_fine, n_brick, n_fine):
        """Live particle indices covering tile+buffer, as the brick union.

        The probe's `tile_members` walks bricks by modular index and
        concatenates CSR slices; this walks the same bricks and concatenates
        slot runs. Same union, same wrap guard.
        """
        nb = int(n_fine) // int(n_brick)
        if nb != self.bricks_per_side:
            raise ValueError(
                f"brick grid {nb} from (n_fine={n_fine}, n_brick={n_brick}) disagrees with "
                f"the layout's {self.bricks_per_side}; the layout was built for a different "
                "brick geometry and its slot runs would not line up"
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

    # ------------------------------------------------------------ decoding

    def bucket_ordinal_of_slots(self, lo, hi):
        """Bucket ordinal for slots [lo, hi), from the index alone.

        This is how a position is decoded without any per-particle bucket id:
        `searchsorted` on the bucket run boundaries, costing O(span log
        n_buckets) rather than a pass over every slot in the box.
        """
        slots = np.arange(int(lo), int(hi), dtype=np.int64)
        return np.searchsorted(self.bucket_start, slots, side="right") - 1

    def bucket_ijk_of_slots(self, lo, hi):
        """Per-axis bucket indices for slots [lo, hi) -- what `decode_positions`
        needs for one brick's run."""
        return bucket_ijk_from_key(
            self.bucket_ordinal_of_slots(lo, hi), self.t9, self.bricks_per_side
        )

    def slot_payload_order(self):
        """Permutation putting particle-indexed payload into SLOT order.

        WHERE THE PAYLOAD ACTUALLY LIVES, stated plainly because the difference
        matters. In production the T9 arrays are indexed by SLOT, which is what
        makes a brick one contiguous 37 KB read; `slot_to_particle` is then
        bookkeeping for building and checking that array, not resident state
        (it is 8 B/p against the 0.25 B/p index, and holding it would eat the
        tier). This module provides the index and the migration bookkeeping;
        carrying the payload through a migration belongs to the engine, M-v2-3.

        Free slots come back as -1, so the caller decides what a dead row holds
        -- which is the same decision the force already makes for its `cap`
        padding, where cycling live indices beat zero-filling by 2.2x on device
        (the same-address atomic hot spot, eba91ab).
        """
        return self.slot_to_particle.copy()

    # ----------------------------------------------------------- migration

    def migrate(self, x_new):
        """Eject-and-reinsert every particle whose bucket changed.

        Returns migration statistics -- the raw material of M-v2-1's exit gate.

        Bulk, not per-particle. A python loop over migrants would be correct and
        useless: at cgh64 a few percent of 1.34e8 particles is millions of
        iterations per step, and the exit gate has to run there. So the work is
        done over AFFECTED BUCKETS only -- rewriting a touched bucket's live
        prefix from the particles that now claim it -- which is the same
        eject-and-reinsert, expressed as a scatter. Untouched buckets are never
        read, so this stays O(migrants) rather than O(N), which is the property
        D-v2-14 clause 3 needs.
        """
        key_new, _, _ = bucket_order_key(x_new, self.t9, self.bricks_per_side)
        cur = self._bucket_of_slot(self.particle_to_slot)
        moved = np.nonzero(key_new != cur)[0]
        arena_used_before = int(np.sum(self.arena_slot_bucket >= 0))

        if len(moved):
            affected = np.unique(np.concatenate([cur[moved], key_new[moved]]))
            # every particle that now claims an affected bucket, in bucket order
            claims = np.nonzero(np.isin(key_new, affected))[0]
            claims = claims[np.argsort(key_new[claims], kind="stable")]
            b_of_claim = key_new[claims]

            # Clear the affected buckets, then refill their prefixes. Vectorized:
            # under a real step most buckets exchange somebody, so `affected` is
            # O(n_buckets) -- 1.68e7 at cgh64 -- and a python loop here would
            # dominate everything. Cost is the affected capacity, not the box.
            lens = self.capacity[affected]
            self.slot_to_particle[
                np.repeat(self.bucket_start[affected], lens) + _within_run_index(lens)
            ] = -1
            self.occupancy[affected] = 0
            # arena entries belonging to affected buckets are re-placed too
            stale = np.isin(self.arena_slot_bucket, affected)
            if np.any(stale):
                idx = np.nonzero(stale)[0]
                self.slot_to_particle[self.arena_base + idx] = -1
                self.arena_slot_bucket[idx] = -1

            counts = np.bincount(b_of_claim, minlength=self.n_buckets)[affected]
            cap = self.capacity[affected]
            fits = np.minimum(counts, cap)
            _refuse_uint16_overflow(fits, "post-migration")

            # bulk-place the part that fits in each affected bucket's prefix
            keep = _prefix_mask(counts, fits)
            placed = claims[keep]
            slots = self.bucket_start[b_of_claim[keep]] + _within_run_index(fits)
            self.slot_to_particle[slots] = placed
            self.particle_to_slot[placed] = slots
            self.occupancy[affected] = fits.astype(np.uint16)

            # The remainder escalates to the arena, then to a refusal. In BULK:
            # placing them one at a time meant one `nonzero` scan of the whole
            # arena per particle, which measured 91 s a step at cdev8 against
            # 3.97 s for the force it was bookkeeping for -- an instrument 23x
            # the cost of the physics. One scan per step, not per migrant.
            self._to_arena_bulk(claims[~keep], b_of_claim[~keep])

        occ = self.occupancy.astype(np.int64)
        cap_all = self.capacity
        return dict(
            n_migrants=int(len(moved)),
            migrant_frac=float(len(moved)) / max(self.n_particles, 1),
            arena_used=int(np.sum(self.arena_slot_bucket >= 0)),
            arena_used_before=arena_used_before,
            arena_capacity=int(len(self.arena_slot_bucket)),
            max_occupancy=int(occ.max()),
            max_fill_frac=float(np.max(occ / np.maximum(cap_all, 1))),
            n_full_buckets=int(np.sum(occ >= cap_all)),
        )

    def _bucket_of_slot(self, slots):
        """Bucket ordinal for given slots, arena slots resolved by their tag."""
        slots = np.asarray(slots, dtype=np.int64)
        out = np.searchsorted(self.bucket_start, slots, side="right") - 1
        in_arena = slots >= self.arena_base
        if np.any(in_arena):
            out[in_arena] = self.arena_slot_bucket[slots[in_arena] - self.arena_base]
        return out

    def _to_arena_bulk(self, particles, buckets):
        """slack -> arena -> loud refusal. Never clamps, never drops.

        One scan of the arena free list per call, then a single scatter.
        """
        if len(particles) == 0:
            return
        free = np.nonzero(self.arena_slot_bucket < 0)[0]
        if len(free) < len(particles):
            raise ValueError(
                f"{len(particles)} particles overflowed their buckets' capacity and the arena "
                f"of {len(self.arena_slot_bucket)} slots has only {len(free)} free. The layout "
                "does not clamp or drop (D-007): a dropped particle deletes mass, and the "
                "particles that overflow are the clustered ones a halo-grade mock exists to "
                "resolve. Raise slack_frac or arena_frac -- and record the measured value, "
                "because D-v2-14 clause 2's 0.90 B/p slack is an estimate, not a measurement."
            )
        a = free[: len(particles)]
        self.arena_slot_bucket[a] = buckets
        s = self.arena_base + a
        self.slot_to_particle[s] = particles
        self.particle_to_slot[particles] = s

    # ------------------------------------------------------------ checking

    def check(self):
        """Every invariant the layout is supposed to hold, as one call.

        Cheap enough to assert in tests and after a migration; O(n_slots).
        """
        occ = self.occupancy.astype(np.int64)
        cap = self.capacity
        if np.any(occ > cap):
            raise AssertionError("occupancy exceeds capacity")
        live = self.slot_to_particle >= 0
        n_live = int(np.sum(live))
        if n_live != self.n_particles:
            raise AssertionError(f"lost particles: {n_live} live slots, {self.n_particles} known")
        p = self.slot_to_particle[live]
        if len(np.unique(p)) != len(p):
            raise AssertionError("a particle occupies two slots")
        if not np.array_equal(self.particle_to_slot[p], np.nonzero(live)[0]):
            raise AssertionError("particle_to_slot disagrees with slot_to_particle")

        # Each bucket's live slots are the PREFIX of its run: no holes, no
        # tombstones. Vectorized rather than looped over buckets -- at cgh64
        # that loop is 1.68e7 iterations and would make the exit-gate script
        # unusable, which is a slow way to lose an invariant check.
        n = self.n_slots
        ordinal = np.repeat(np.arange(self.n_buckets, dtype=np.int64), self.capacity)
        within = np.arange(n, dtype=np.int64) - self.bucket_start[ordinal]
        live_body = self.slot_to_particle[:n] >= 0
        expected = within < occ[ordinal]
        if np.any(expected & ~live_body):
            b = int(ordinal[np.argmax(expected & ~live_body)])
            raise AssertionError(f"bucket {b} has a hole inside its live prefix")
        if np.any(live_body & ~expected):
            b = int(ordinal[np.argmax(live_body & ~expected)])
            raise AssertionError(f"bucket {b} has a live slot past its occupancy")
        return True

    # ------------------------------------------------------------ accounting

    def bytes_per_particle(self, payload=9.0):
        """The D-v2-14 clause 2 table, computed from the layout that exists
        rather than quoted. `slack` here is MEASURED off this layout, so it is
        the number that replaces the 0.90 estimate."""
        n = max(self.n_particles, 1)
        index = self.occupancy.nbytes / n
        csr = (self.bricks_per_side**3) * 16.0 / n
        slack = (self.n_slots - self.n_particles) * payload / n
        arena = len(self.arena_slot_bucket) * payload / n
        return dict(
            payload=payload,
            bucket_index=index,
            brick_csr=csr,
            slack=slack,
            arena=arena,
            total=payload + index + csr + slack + arena,
        )


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


def _refuse_uint16_overflow(counts, what):
    hot = int(np.max(counts)) if len(counts) else 0
    if hot > UINT16_MAX:
        raise ValueError(
            f"{what} bucket count {hot} exceeds uint16 ({UINT16_MAX}). The per-bucket index "
            "is uint16 because that is what makes it 0.25 B/p; a wider index costs 0.5 B/p "
            "and moves D-v2-14 clause 2's all-in figure. This fires only under extreme "
            "clustering -- how close a real run gets is part of what M-v2-1's cgh64 "
            "measurement is for."
        )
