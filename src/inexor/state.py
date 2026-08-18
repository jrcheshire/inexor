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


def encode_velocities_host(v, scale):
    """D-time velocities -> int16 at a GIVEN scale. Mirrors
    `codec.encode_velocities`'s value path (rint at the scale, then int16),
    with the scale supplied rather than derived -- the streamed builder knows
    it from the partition-max over its staged slabs, exactly
    `reconcile_velocity_scale`'s theorem at build time. The pre-cast range is
    asserted (D-007): the theorem says it cannot fire, and the refusal is what
    proves that rather than assumes it."""
    w = np.rint(np.asarray(v, dtype=np.float64) / float(scale))
    assert_int16_range(w)
    return w.astype(np.int16)


def _alloc_geometry(brick_counts, n_particles, brick_slack, alloc_margin, arena_frac):
    """The D-v2-19 capacity arithmetic: (spare, brick_start, n_alloc, n_arena).

    Split out of `SlotState.build` so the streamed loader derives its geometry
    through THE SAME code path -- two implementations of one formula is how a
    loader and a builder drift apart bitwise.
    """
    spare = np.ceil(brick_counts * float(brick_slack)).astype(np.int64)
    spare = np.where(brick_counts > 0, np.maximum(spare, 1), spare)
    brick_start = np.zeros(len(brick_counts) + 1, dtype=np.int64)
    np.cumsum(brick_counts + spare, out=brick_start[1:])
    n_alloc = int(np.ceil(int(brick_start[-1]) * (1.0 + float(alloc_margin))))
    n_arena = int(np.ceil(n_particles * float(arena_frac)))
    return spare, brick_start, n_alloc, n_arena


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
    """Re-express int16 velocity codes at a new scale. Per-row scales allowed.

    `s_old` and `s_new` are scalars or (n,) arrays broadcast over rows. Exact
    when the scales are equal, which stays the common case; the general path
    decodes and re-rounds.

    **The no-overflow property is no longer free, and this is the function where
    that changed.** Under one global scale it was a theorem: `s_new` was the max
    over a PARTITION of the particles, so `|rint(v/s_new)| <= 32767` could not
    fail. Per brick it is false in general -- a fast particle drifting out of a
    dense brick into a quiet one needs more range than the quiet brick's own
    maximum provides -- so the caller must supply an `s_new` that already covers
    every row it passes, and `_insert_slab` is the only place that can, because
    it is the first point at which a brick's full post-migration membership is
    known. The assertion below is not a formality: D-007 forbids the saturating
    alternative, so a wrap here is silent corruption of the state.
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
            "over the rows it is about to write. D-007 forbids the clamp."
        )
    return out.astype(np.int16)


def _scales_from_sorted(absv_sorted, brick_counts):
    """Per-brick velocity scale from a brick-major-sorted |v|_inf column.

    `reduceat` rather than `np.maximum.at`: the latter is a ufunc.at loop and is
    orders slower at IC scale, and this runs over every particle. Passing only
    the starts of NON-EMPTY bricks is what makes it correct -- an empty brick has
    zero width, so the segment between two non-empty starts is exactly the first
    one's particles, and reduceat's documented misbehaviour on equal consecutive
    indices (it returns the element rather than the identity) is never reached.

    An empty brick, or one whose particles are all at rest, gets 1.0 rather than
    0.0. A zero scale is not a smaller scale, it is a division by zero on the
    next decode, and it would encode a genuinely-zero velocity no better.
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
    """Quantize float velocities at a per-row scale. No clip: D-007."""
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
    # The SOURCE brick, carried on emigrants only. An emigrant's code is written
    # at its old brick's scale and can only be re-expressed at the destination's
    # once that is fixed, so the reader needs to know which scale it is holding.
    # Keepers do not carry it because their source IS `dest // buckets_per_brick`
    # -- tagging them would put 4 B on the majority of staged rows to store what
    # is already there.
    if src is not None:
        out["src"] = np.concatenate(src) if src else np.empty(0, np.int32)
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
    """`argsort(kind="stable")` on numpy's RADIX path when the key range allows.

    **The fact this exists for.** numpy's stable sort is a radix sort only for 1-
    and 2-byte integer types; anything wider gets timsort. Bucket ordinals within
    a brick are tiny -- `buckets_per_brick` is 512 at cdev, cgh64, C-gh AND
    C-hero alike, because the config table holds the bucket grid and the brick
    grid in step -- but they are computed as int64 and so were sorted by
    comparison. The identical change bought `migrate`'s own sort 5.3x (89.6 ->
    17.0 ms at 2.1e6 rows) and `_group_by_brick` already did it locally, while
    four other sites in this module did not.

    **The permutation is IDENTICAL, not merely equivalent.** A narrowing cast is
    order-preserving for non-negative keys inside the target's range, and stable
    sorts agree on ties, so the returned order is the same one `argsort` on the
    wide key gives. That is what makes this bitwise neutral, which it has to be:
    the encode downstream is order-dependent through a float max.

    **The range is CHECKED, not assumed.** numpy narrows modularly, so a key of
    65536 would store as 0 and silently sort first -- exactly the failure mode
    D-v2-20 found in `migrate` and `repack`, where `.astype(uint16)` was called
    bare and a wrapped occupancy relocated every later bucket in the brick. A key
    out of range falls back to the wide sort rather than wrapping.
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
    """Rows grouped by destination brick: a permutation plus CSR offsets.

    **This replaces a scan per brick, and that scan was the engine's wall.**
    `_insert_slab` used to select each brick's rows with `dest // p3 == b`
    inside the brick loop, so every brick read every row of the slab. A slab
    holds ~N/nb rows and contains nb^2 bricks over nb slabs, which comes to
    `N x nb^2` comparisons per step -- N^(5/3), not N. Measured on deneb
    (job 456, particles fixed at 256^3, staging depth pinned at 1): insert time
    3.793 -> 10.756 -> 39.626 s as bricks per side went 8 -> 16 -> 32, against a
    prediction of 38.6 s at the last rung made before it ran. Carried to C-gh
    that term alone is ~87 h per step.

    One grouping pass instead. Rows outside `[lo_b, hi_b)` are dropped rather
    than refused: the immigrant buffer holds every emigrant from every reaching
    slab, and only some are bound for this one -- the old mask discarded them
    silently by never matching, and this reproduces that.

    **Order within a brick is preserved**, which is what makes the change
    bitwise neutral rather than merely equivalent: the boolean mask it replaces
    yielded rows in their original order, a stable sort does the same, and the
    encode that follows is order-dependent through a float max.
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
    # uint16 where it fits, because numpy's stable sort takes the RADIX path
    # only for 1- and 2-byte integer types -- the same fact that bought
    # `migrate` 5.3x on its own sort. nb^2 is 16,384 at C-gh, so it fits there;
    # above 65,535 this falls back to a comparison sort rather than pretending.
    key = w.astype(np.uint16) if n_b <= np.iinfo(np.uint16).max else w
    return idx[np.argsort(key, kind="stable")], off


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


def brick_reach(st, c_drift, vel_scale=None):
    """How many bricks a particle can cross in x during this drift. O(1).

    An upper bound that is nearly TIGHT, which is what makes bounded staging
    affordable: `w` is int16 so |v| <= vel_scale * INT16_MAX, and `vel_scale` is
    defined as the partition max of |v| divided by INT16_MAX, so the product is
    the fastest particle actually present rather than a pessimistic ceiling. No
    pass over the state, and no dependence on the realized displacement being
    small.

    Peak staging is `2 * reach + 1` slabs, so this is also the knob that prices
    the migration's memory: reach 1 reproduces the original three-slab schedule
    exactly.

    With per-brick scales the bound takes the MAX over bricks, which is still the
    fastest particle present and so still nearly tight -- the fastest particle
    sets its own brick's scale exactly. It is the whole grid's bound rather than
    a per-brick one on purpose: the schedule is global, so a per-brick reach
    would have to be reconciled into one number anyway, and taking the max is
    that reconciliation.
    """
    s = float(np.max(st.vel_scale if vel_scale is None else vel_scale))
    nb = int(st.bricks_per_side)
    extent = float(st.t9.box_size) / nb
    if extent <= 0.0:
        return nb
    return int(np.ceil(abs(float(c_drift)) * s * INT16_MAX / extent))


def drift_and_migrate(st, c_drift, max_staged_slabs=None, kernel="numpy"):
    # NB this default stays "numpy" while `EngineConfig.eject_kernel` defaults to
    # "jax", and the asymmetry is deliberate. The engine always passes the config
    # value, so nothing routes through this default in production; what DOES use
    # it is `tests/test_eject_jax.py`, whose reference arm calls this bare and
    # compares against `kernel="jax"`. Unifying the two defaults would make that
    # gate compare jax with jax -- a gate that cannot fail, in the one place the
    # compiled path is proved bitwise.
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
    # A SNAPSHOT, not a reference. `_insert_slab` rewrites a brick's scale as it
    # writes that brick, while ejections still to come must decode at the
    # PRE-migration scales. The schedule happens to eject a brick before
    # inserting it, so the live array would give the same answer today -- which
    # is exactly the kind of incidental correctness this module has been bitten
    # by, so the two arrays are separated by construction instead. 8 B per brick,
    # 16.8 MB at C-gh.
    scales = np.array(st.vel_scale, dtype=np.float64, copy=True)
    # D-007 SAYS NOTHING MAY BE DROPPED AND NOTHING CHECKED IT HERE. The slab
    # schedule above releases a staged row once its destination slab has been
    # written, which is only safe under the one-brick-per-axis-per-step assumption
    # the docstring states -- verified at cdev8 over 20 steps, and cdev at K=5
    # drifts much further per step than that. A particle that moves TWO bricks in
    # x finds its destination already inserted and released, and is silently lost.
    #
    # Measured (M-v2-6, antares 441, cdev): occupancy, brick_member_count and
    # decode all read 16,777,215 against n_particles 16,777,216 -- one particle
    # gone, with every census agreeing, no aliasing and ownership a perfect
    # partition. It surfaced ~200 lines away as the engine's ownership assertion,
    # which cost four wrong diagnoses. A loss must be loud AT THE POINT OF LOSS.
    n_before = int(st.occupancy.astype(np.int64).sum()) + st.arena_used

    # THE REACH, and it is a bound rather than an assumption. The schedule below
    # releases a staged row once its destination has been written, so it must know
    # how far a particle can travel. The old code hard-coded +-1 brick, which held
    # at cdev8 and lost a particle at cdev.
    #
    # The bound is O(1) and nearly TIGHT, which is what makes this cheap: `w` is
    # int16, so |v| <= vel_scale * INT16_MAX, and `vel_scale` is DEFINED as the
    # partition max of |v| over INT16_MAX -- so that product is the actual maximum
    # speed, not a pessimistic ceiling. No pass over the state is needed.
    r_raw = brick_reach(st, c_drift, scales)
    # CLAMP rather than refuse. On a periodic grid a reach of nb // 2 already
    # touches every slab, so beyond that the schedule is all-to-all and larger
    # values say nothing extra. Refusing here would confuse "needs more memory"
    # with "impossible": full staging is CORRECT, it just costs the whole state in
    # flight, and at a 2-brick test grid that is nothing at all. Correctness is not
    # the caller's choice; the memory budget is, and that is `max_staged_slabs`.
    r = min(r_raw, nb // 2)
    reach = range(-r, r + 1)

    staged, emig, inserted = {}, {}, set()
    consumed = {}  # emig rows an insert actually took, per source slab
    n_over, peak_staged, realized_reach = 0, 0, 0
    for s in range(nb):
        staged[s], emig[s] = st._eject_slab(s, c_drift, scales, kernel=kernel)
        consumed[s] = 0
        # the REALIZED x-reach, reported beside the bound: the bound said 2 at
        # cdev while every record claimed "at most one brick per axis", and the
        # gap between those two statements was a lost particle
        if len(emig[s]["dest"]):
            d_slab = emig[s]["dest"] // (st.buckets_per_brick * nb * nb)
            disp = (d_slab - s + nb // 2) % nb - nb // 2
            realized_reach = max(realized_reach, int(np.abs(disp).max()))
        # a slab may be written once every slab that can REACH it has been ejected
        for d in range(nb):
            if d in inserted:
                continue
            if all(((d + o) % nb) in emig for o in reach):
                n_over += st._insert_slab(d, staged, emig, reach, consumed, scales=scales)
                inserted.add(d)
        # release what no pending write can still need
        for s2 in list(staged):
            if s2 in inserted:
                del staged[s2]
        for s2 in list(emig):
            if all(((s2 + o) % nb) in inserted for o in reach):
                # THE CENSUS, at the point of loss. Releasing an emig slab whose
                # rows were not all consumed is exactly how the cdev particle
                # vanished (antares 442): the schedule staged a 2-brick x-mover
                # correctly, `_insert_slab` consumed only +-1 sources, and this
                # deletion destroyed the row with nothing raising. The final
                # n_before/n_after guard 200 lines of call stack away cost five
                # wrong causes; this one names the row's displacement.
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
                        "D-007 forbids dropping; an unconsumed emigrant is a particle "
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
    n_after = int(st.occupancy.astype(np.int64).sum()) + st.arena_used
    if n_after != n_before:
        left = sum(len(v.get("dest", ())) for v in staged.values()) if staged else 0
        raise ValueError(
            f"the migration lost {n_before - n_after} particles ({n_before} -> "
            f"{n_after} against {st.n_particles} stored). D-007 forbids dropping, "
            "so this is corruption, not imprecision.\n"
            f"  {len(staged)} slabs still staged at the end ({left} rows), "
            f"{len(inserted)} of {nb} slabs inserted, arena {st.arena_used}/"
            f"{st.n_arena}\n"
            "  LEADING CAUSE: the slab schedule releases a staged row once its "
            "destination is written, which assumes a particle moves at most ONE "
            "brick per axis per step (this function's docstring, measured at cdev8 "
            "over 20 steps). A larger drift breaks it -- reduce the step size, or "
            "generalize the staging to the realized brick displacement."
        )
    # NB no `st.vel_scale = ...` here: every brick's scale was fixed by its own
    # insert, over the membership that insert actually wrote.
    # reach and peak staging REPORTED, so the memory bound is a measurement
    # every step rather than a docstring claim that held at one configuration
    return dict(n_arena_overflow=n_over, arena_used=st.arena_used,
                vel_scale=float(np.max(st.vel_scale)),
                vel_scale_min=float(np.min(st.vel_scale)),
                n_migrated_checked=n_after, brick_reach=r, brick_reach_raw=r_raw,
                brick_reach_realized=realized_reach, peak_staged_slabs=peak_staged)


def drift_and_migrate_pooled(st, c_drift, pool, kernel="numpy", window=None, max_staged_slabs=None):
    """`drift_and_migrate` on the worker pool, BITWISE the serial pass.

    The division of labour the C13 census licenses: workers eject and insert
    whole slabs, writing brick payloads (off/w/ids/occupancy/vel_scale)
    straight into shared memory -- those writes are disjoint per brick and
    bricks partition into slabs -- while every ORDER-DEPENDENT arena mutation
    stays out of the workers entirely. Ejects run with `release_arena=False`
    and inserts hand their overflow rows back through `spill_sink`, so nothing
    writes the arena until every task has returned; the parent then re-runs
    the serial schedule's release/claim interleave (`_release_brick_arena` +
    `_to_arena`, the production claim path) at end of pass. Nothing in a pass
    reads an arena row the pass mutates before that row's own schedule point
    -- claims tag only already-inserted slabs' bricks, releases only the
    releasing slab's own rows -- so deferring the whole sequence is invisible
    and the free list every claim is ordered against evolves exactly as the
    serial pass's.

    The same replay walks the serial loop's bookkeeping symbolically (staged
    and emig sets, consumption census, peak staging), so the returned stats
    dict is equal KEY FOR KEY to the serial one; the pool's own numbers ride
    a separate `migrate_pool` entry.

    Failure semantics vs serial, accepted and deliberate: an arena-full
    refusal (D-007) raises from the same `_to_arena` line but at replay time
    rather than mid-pass, and the consumption census names counts without the
    displacement histogram when the scratch slot has been reused -- both arms
    leave invalid state on either path. A worker exception re-raises in the
    dispatch loop.

    `window` bounds the scratch slots in flight. The floor is 4r+2: the wrap
    pins ~2r slots for the whole pass and the sliding eject->insert span
    holds 2r+2 more, below which the dispatch loop cannot free a slot and
    would deadlock; the default adds the worker count so the pipeline can
    actually feed W workers.
    """
    nb = st.bricks_per_side
    p3 = st.buckets_per_brick
    n_before = int(st.occupancy.astype(np.int64).sum()) + st.arena_used
    scales = np.array(st.vel_scale, dtype=np.float64, copy=True)
    r_raw = brick_reach(st, c_drift, scales)
    r = min(r_raw, nb // 2)
    reach = range(-r, r + 1)
    if 2 * r + 1 >= nb:
        # the schedule is all-to-all: every slab reaches every other, the
        # window would be the whole state, and the serial path already handles
        # exactly this. Fall back rather than refuse -- correctness is not the
        # caller's choice -- and say so on the stats.
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
    occ_slab = st.occupancy.astype(np.int64).reshape(nb, -1).sum(axis=1)
    arena_slab = np.zeros(nb, dtype=np.int64)
    if st.n_arena:
        keys = st.arena_bucket[st.arena_bucket >= 0]
        if len(keys):
            arena_slab = np.bincount((keys // p3) // (nb * nb), minlength=nb)
    slot_rows = int((occ_slab + arena_slab).max())
    pool.stage_migrate(c_drift, kernel, r, slot_rows, K)

    # THE DISPATCH LOOP. Backpressure is the free-slot list: an eject may only
    # launch into a free slot, and a slab's slot frees once every destination
    # its emig can feed has been inserted -- the serial release condition.
    free_slots = list(range(K))
    slot_of = {}
    ejected = {}  # s -> (slot, n_keep, n_emig)
    rr_by_slab = {}
    insert_res = {}
    dispatched = set()
    eject_busy = insert_busy = 0.0
    eject_jax_calls = 0
    next_eject = 0
    while len(insert_res) < nb:
        while next_eject < nb and free_slots:
            slot = free_slots.pop()
            slot_of[next_eject] = slot
            pool.submit_eject(next_eject, slot)
            next_eject += 1
        res = pool.next_migrate_result()
        if res["kind"] == "eject":
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

    # THE REPLAY: the serial pass's arena interleave and bookkeeping, re-run
    # exactly. `_release_brick_arena` at each slab's eject point, `_to_arena`
    # at each destination's insert point (spilled bricks arrive ascending from
    # `_insert_slab`'s own loop), and the census/peak accounting at the same
    # schedule points the serial loop runs them.
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
                n_rows = ejected[s2][2]
                if consumed[s2] != n_rows:
                    raise AssertionError(
                        f"releasing emig slab {s2} with {n_rows - consumed[s2]} of "
                        f"{n_rows} rows unconsumed (reach {r}, consumption offsets "
                        f"{sorted({int(o) for o in reach})}). D-007 forbids "
                        "dropping; an unconsumed emigrant is a particle about to "
                        "be destroyed. (Pooled pass: the displacement histogram "
                        "the serial census prints needs rows whose scratch slot "
                        "may be reused -- re-run serial for the full census.)"
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
    n_after = int(st.occupancy.astype(np.int64).sum()) + st.arena_used
    if n_after != n_before:
        raise ValueError(
            f"the migration lost {n_before - n_after} particles ({n_before} -> "
            f"{n_after} against {st.n_particles} stored). D-007 forbids dropping, "
            "so this is corruption, not imprecision.\n"
            f"  {len(inserted)} of {nb} slabs inserted, arena {st.arena_used}/"
            f"{st.n_arena} (pooled pass)\n"
            "  LEADING CAUSE: the slab schedule releases a staged row once its "
            "destination is written, which assumes a particle moves at most ONE "
            "brick per axis per step. A larger drift breaks it -- reduce the "
            "step size, or generalize the staging to the realized displacement."
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
                                  # the compiled-kernel receipt, summed from the
                                  # workers (the parent's counter cannot see them)
                                  eject_jax_calls=eject_jax_calls))


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
    # float64 (n_bricks,) -- ONE SCALE PER BRICK, not per state. 8 B per brick is
    # 16.8 MB at C-gh against the 274.9 GB `kick_pending` array a single global
    # scale forces the engine to hold, because a global scale cannot be known
    # until every tile has been kicked. See `_insert_slab` for where a brick's
    # scale is fixed and why it cannot be fixed earlier.
    vel_scale: np.ndarray
    arena_base: int
    arena_bucket: np.ndarray  # int64 (n_arena,) -1 where free
    n_particles: int
    ids: np.ndarray = None  # int32 (n_alloc + n_arena,) or None
    # brick -> arena rows, rebuilt on demand. NOT state: a pure function of
    # `arena_bucket`, cached because computing it per call is an O(n_arena) scan
    # and the callers are per-brick. See `arena_slots_of_brick`.
    _arena_by_brick: dict = None
    # ascending free arena-relative rows, also a pure cache over `arena_bucket`.
    # None = dirty; the next claim rebuilds it with the same nonzero scan the
    # uncached path ran per call, so claim ORDER (lowest slots first) is
    # unchanged. See `_to_arena` for why this exists.
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

        _, brick_start, n_alloc, n_arena = _alloc_geometry(
            brick_counts, n, brick_slack, alloc_margin, arena_frac
        )

        order = _stable_sort_index(key)
        rank = _within_run_index(brick_counts)
        slots = brick_start[brick[order]] + rank

        n_rows = n_alloc + n_arena

        # the payload, written straight into slot order -- this is the whole point
        off_all, bijk = encode_positions_host(x, t9)
        # ONE SCALE PER BRICK, taken over that brick's own particles. `order` is
        # brick-major, so the reduction rides the sort the layout already needs.
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

    def decode_brick(self, brick_flat, scales=None):
        """(slots, x, v) for every particle of this brick, arena included.

        O(brick) floats -- ~4096 particles at C-gh, about 100 KB. Nothing here
        is ever O(N) in floats; a global (n,3) f64 array is 206 GB at C-gh and
        deleting it is D-v2-16 clause 1.

        `scales` overrides `self.vel_scale`, and the migration passes a SNAPSHOT
        taken before any brick was rewritten. That is structural rather than
        defensive: `_insert_slab` rewrites a brick's scale in place, so a decode
        that read the live array would be correct only for as long as the
        schedule happens to eject every brick before inserting it. Making the
        caller name the array it means removes the dependence on that ordering.
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

    def decode_bricks(self, bricks, scales=None):
        """(slots, x, v) over a list of bricks, concatenated.

        O(tile) floats. The tile is the largest float working set on the engine
        path, by design: a global (n,3) f64 array is 206 GB at C-gh and deleting
        both of them is D-v2-16 clause 1.

        Rows stay grouped by brick in the order `bricks` gives, which the kick
        relies on: it fixes one scale per brick, and a brick's rows being
        contiguous is what lets it do that without a sort.
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

    def _eject_slab_jax(self, bx, c_drift, scales, release_arena=True):
        """`_eject_slab` with the drift and the partition compiled.

        Same contract, same return value, same mutations, and gated elementwise
        against the numpy path (`tests/test_eject_jax.py`). The split of labour
        is section 5n's decomposition: the row arithmetic and the keep/leave
        partition go to XLA (58-65% of the call, 20-80x above the machine's
        traffic floor), while slot resolution and the arena splice stay here
        because they are pointer-chasing at 8-30x the floor.

        The two structural differences from the numpy path, both deliberate:

        1. **The decode is not done here.** The numpy path calls `decode_brick`,
           which builds `x` and `v` in float, and then re-derives the lattice
           index from them. The compiled kernel goes from `(off, bijk, w)`
           straight to the new index, so this resolves slots WITHOUT decoding.
           The arena splice is reproduced line for line because a divergence
           there is a lost particle, which this function has produced before.
        2. **One call for the whole slab, not one per brick.** `_cat` already
           concatenates all bricks' keepers into one array and all bricks'
           leavers into another, so the target order is global rather than
           per-brick, which makes both results contiguous slices of one buffer.
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
            # the release, identical to the numpy path INCLUDING the surgical
            # index drop (5g: invalidating the whole index was 51% of an
            # arena-occupied migrate). `release_arena=False` skips it for a
            # pooled worker, whose parent replays the release in serial order
            # (`_release_brick_arena`) -- a worker that released here would
            # change the free list other claims are ordered against.
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

        Reads the state; writes NOTHING back. That phase separation is what makes
        a double drift structurally impossible rather than merely unlikely: a
        brick is only written once every brick that can send to it has been read.

        Returns (keep, emig), each a dict of flat arrays plus the destination
        bucket ordinal, so a slab costs O(slab) rather than O(N).

        `kernel="jax"` routes to `_eject_slab_jax`, which is gated elementwise
        against this function. This one stays the reference and is never
        conditionally modified: an A/B whose arms share their lines cannot see a
        change to those lines (umbrella `ab_arms_sharing_code_are_policy_blind`).
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
            # `decode_brick` returns this brick's ARENA residents too, so they are
            # re-homed by this ejection like any other member. Their arena rows
            # are released here, where the payload is consumed -- releasing them
            # at insert instead double-counts them, which is how this was found
            # (32771 particles reachable against 32768 stored). A row is only
            # freed once its brick has been ejected, so a concurrent `_to_arena`
            # cannot claim a row that still holds live state.
            a_free = self.arena_slots_of_brick(b)
            # `release_arena=False`: a pooled worker must leave the free list
            # untouched -- the parent replays this release in serial order.
            if release_arena and len(a_free):
                self.arena_bucket[a_free - self.arena_base] = -1
                # Releasing brick b's rows changes the index by EXACTLY one key,
                # so drop that key instead of invalidating the whole cache. The
                # sledgehammer here was 51% of an arena-occupied migrate: each
                # invalidation forced the NEXT brick's decode to rebuild the
                # whole index, A x O(n_arena) per migrate (2,008 rebuilds of a
                # 3.4M-row arena in one profiled cdev call; the engine-scale
                # term is 5g of the scaling record). The free-list goes dirty
                # rather than maintained: freed rows must re-enter in ascending
                # slot order, which only the rebuild scan guarantees.
                if self._arena_by_brick is not None:
                    self._arena_by_brick.pop(int(b), None)
                self._arena_free = None
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
            # NO RESCALE HERE, and that is the change per-brick scales force. A
            # row leaves at its OWN brick's scale and is re-expressed once, in
            # `_insert_slab`, at the destination's -- which cannot be known here
            # because it depends on every other brick that sends to that
            # destination. Rescaling twice (out to a common scale, then in) would
            # round twice where this rounds once.
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
        self, bx, staged, emig, reach=(-1, 0, 1), consumed=None, scales=None, spill_sink=None
    ):
        """Write one slab's bricks back: keepers + immigrants + arena residents.

        Every brick's final membership passes through an O(brick) buffer here, so
        this is also where a bucket that outgrew its brick escalates -- spare,
        then arena, then a loud refusal, with no clamp anywhere (D-007).

        **It is also the only place a brick's velocity scale can be fixed**, and
        that is a consequence of deleting the global scale rather than a choice.
        A scale must cover every row it encodes. Under one global scale the kick
        could compute that by reducing over tiles, at the price of holding every
        tile's new velocities until the last tile was done -- 274.9 GB at C-gh.
        Per brick, the kick can only see the rows it owns NOW, and a brick's
        membership changes under it during the drift. Here, and only here, both
        halves are in hand: the keepers this brick retained and the immigrants
        every reaching slab sent it. So the scale is taken over the union and
        every row is expressed at it, in one rounding.

        `scales` is the pre-migration snapshot, needed because a row arrives
        holding a code written at its SOURCE brick's scale.
        """
        nb = self.bricks_per_side
        p3 = self.buckets_per_brick
        lo_b, hi_b = self.slab_bricks(bx)
        keep = staged[bx]
        # Immigrants come from every slab within REACH, the same set the caller's
        # schedule ejects before permitting this write. This was hard-coded to
        # +-1 ("a particle moves at most ONE brick per axis per step, measured at
        # cdev8") while the schedule in `drift_and_migrate` was generalized to the
        # realized reach -- so a 2-brick x-mover at cdev was staged correctly,
        # matched by NO insert, and destroyed by the release loop: the missing
        # particle of 16,777,216 (antares 442). At reach 1 the set is identical
        # to the old +-1, so every prior gate number is untouched.
        sources = sorted({(int(bx) + o) % nb for o in reach})
        if consumed is not None:
            for s in sources:
                if s in emig and len(emig[s]["dest"]):
                    d_slab = emig[s]["dest"] // (p3 * nb * nb)
                    consumed[s] += int(np.count_nonzero(d_slab == bx))
        imm = _cat_dicts([emig[s] for s in sources if s in emig])
        n_over = 0
        # GROUP ONCE, then slice. See `_group_by_brick` for what this replaces
        # and what it was measured to cost.
        k_ord, k_off = _group_by_brick(keep["dest"] // p3, lo_b, hi_b)
        i_ord, i_off = _group_by_brick(imm["dest"] // p3, lo_b, hi_b)
        for j, b in enumerate(range(lo_b, hi_b)):
            sel_k = k_ord[k_off[j] : k_off[j + 1]]
            sel_i = i_ord[i_off[j] : i_off[j + 1]]
            # NB no arena term: a brick's arena residents were decoded and
            # re-homed by its own ejection, and their rows released there.
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
            # THE SCALE, over the union and before anything is written. Keepers
            # hold codes at this brick's own pre-migration scale; immigrants hold
            # codes at whichever brick sent them. Both are turned back into
            # physical magnitudes to take the max, then everything is expressed
            # once at the result -- so the only rounding a row sees this step is
            # this one.
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

    def _write_brick(self, b, dest, off, w, ids=None, spill_sink=None):
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
            order = _stable_order(within, p3)
            keep_n = cap
            spill = order[keep_n:]
            n_over = len(spill)
            # `spill_sink`: a pooled worker must not claim -- slot assignment is
            # order-dependent (lowest-free-first), so the rows go back to the
            # parent, which claims via `_to_arena` at this brick's serial point.
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
        self.occupancy[b * p3 : (b + 1) * p3] = _to_index(counts, self.index_dtype, "migrated")
        return n_over

    def _to_arena(self, dest, off, w, ids=None):
        """Park overflow in the arena, or REFUSE. Never clamp, never drop.

        The free scan is CACHED (`_arena_free`), rebuilt only after a release
        dirtied it -- at most once per eject->claim transition instead of the
        per-call `np.nonzero` this ran before (1,988 calls, 2.99 s of a 13.8 s
        arena-occupied cdev migrate; scaling record 5g). Claims still take the
        LOWEST free slots, because the rebuild is the same ascending scan and
        claims only ever consume its head, so the arena layout is bitwise the
        uncached path's.
        """
        free = self._arena_free
        if free is None:
            free = np.nonzero(self.arena_bucket < 0)[0]
        if len(free) < len(dest):
            raise ValueError(
                f"{len(dest)} particles overflow their brick's capacity and the arena of "
                f"{self.n_arena} slots has only {len(free)} free. The layout does not clamp "
                "or drop (D-007). Raise brick_slack or arena_frac."
            )
        a = free[: len(dest)]
        self._arena_free = free[len(dest):]
        self.arena_bucket[a] = dest
        # Surgical index update, exact by construction: a rebuild groups live
        # rows by brick in ascending slot order, so appending the newly claimed
        # slots to their bricks' keys and re-sorting each touched key produces
        # the rebuild's own content without the O(n_arena) pass.
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
        """Release brick b's arena rows without ejecting it.

        The third copy of the eject-side release, and the duplication is as
        deliberate as the other two (`_eject_slab` / `_eject_slab_jax` carry it
        "line for line" because a divergence there is a lost particle). This
        one exists for the pooled migrate's parent-side replay: workers eject
        with `release_arena=False`, and the parent re-runs each slab's release
        at its serial schedule point so the free list every `_to_arena` claim
        is ordered against evolves exactly as the serial pass's.
        """
        a_free = self.arena_slots_of_brick(b)
        if len(a_free):
            self.arena_bucket[a_free - self.arena_base] = -1
            if self._arena_by_brick is not None:
                self._arena_by_brick.pop(int(b), None)
            self._arena_free = None

    def _repack_reference(self, brick_slack=0.10):
        """The out-of-place repack, KEPT AS THE IDENTITY ORACLE for `repack`.

        Not on the hot path and not to be called by the engine: it allocates
        `zeros_like` of `off` and `w`, measured at 11.1 B/row and ~115 GB at
        C-gh, which is what `repack` below exists to avoid. It stays because
        this container has lost a particle before and cost five sessions to
        find it, so the in-place version is gated on reproducing this one
        ELEMENTWISE rather than on a property that might hold for two different
        reasons. Deleting it would retire that gate to save nothing.

        Original docstring follows.

        Redistribute BRICK capacity to match current occupancy.

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
            order = _stable_order(dest - b * p3, p3)
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

    def repack(self, brick_slack=0.10):
        """Redistribute BRICK capacity to match current occupancy, IN PLACE.

        Required, not an optimization -- D-v2-19 clause 3 measured that frozen
        capacity fails at every granularity, so pooling, repack and a small
        arena are all three needed.

        **Why this is not the port D-v2-19 clause 3 named.** That clause points
        at `layout.BrickPackedLayout.repack` and its reported `scratch_bytes` of
        0.13-0.52 MB "independent of N". That figure counts only two chunk
        buffers; the function also allocates `live`, `parts` and `final` at one
        row each and MEASURES 39.4 B/row against this container's 11.1
        (`scripts/v2_m6_repack_bytes.py`, flat to 2.6% over 64x in N), so
        porting it would have multiplied the term by 3.55x. The clause's
        REASONING is what survives and is what this uses: a repack is a monotone
        rearrangement, not a sort.

        **The rearrangement in two directional passes.** Compaction only ever
        moves a row left and expansion only ever moves it right, so:

          A. ascending, move each brick's MAIN run down to `main_pos[b]`. Safe
             because a run's live rows cannot exceed its own allocation, so
             `sum(run_counts[:b]) <= brick_start[b]` with no exceptions.
          B. descending, write each brick's final block at `new_start[b]`. Safe
             because `new_start[b] >= main_pos[b]` -- the new starts include
             both the arena residents and the spare.

        Main runs are compacted WITHOUT their arena residents on purpose. Folding
        them in during pass A would break its guarantee: `counts` includes arena
        rows, so `sum(counts[:b])` can exceed `brick_start[b]` by the arena
        residents of earlier bricks, and the write would land on the next
        brick's unread rows. They are merged in pass B instead, where the
        destination already has room for them.

        **Scratch is one brick plus the live arena**, and both are reported. The
        arena lift is O(arena_used) rather than O(brick) -- 0.77 GB at C-gh at
        the default 1% arena against the 115 GB this replaces -- and it is
        needed because pass B's writes can reach past the OLD `arena_base` when
        the allocation grows, which would clobber residents before they are
        read. Reported rather than described, because a term omitted from a
        scratch figure is exactly what hid the 39.4 above.
        """
        p3 = self.buckets_per_brick
        run_counts = np.asarray(self.occupancy, dtype=np.int64).reshape(self.n_bricks, p3)
        run_counts = run_counts.sum(axis=1)
        occ = self.occupancy.astype(np.int64)
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

        scratch = 0
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
        for b in range(self.n_bricks):
            m = int(run_counts[b])
            src, dst = int(self.brick_start[b]), int(main_pos[b])
            if m == 0 or src == dst:
                continue
            # `.copy()` because source and destination overlap and numpy's slice
            # assignment gives no ordering guarantee across an overlap.
            buf_off = self.off[src : src + m].copy()
            buf_w = self.w[src : src + m].copy()
            scratch = max(scratch, buf_off.nbytes + buf_w.nbytes)
            self.off[dst : dst + m] = buf_off
            self.w[dst : dst + m] = buf_w
            if self.ids is not None:
                self.ids[dst : dst + m] = self.ids[src : src + m].copy()

        # ---- pass B: expand rightward, merging the arena residents back in
        new_occ = np.zeros(self.n_buckets, dtype=np.int64)
        bucket_ids = np.arange(p3, dtype=np.int64)
        for b in range(self.n_bricks - 1, -1, -1):
            m = int(run_counts[b])
            k = int(a_edge[b + 1] - a_edge[b])
            if m + k == 0:
                continue
            mp, ns = int(main_pos[b]), int(new_start[b])
            # the main rows' buckets are DERIVED from occupancy, never read back
            # off the array, so pass A moving them cannot desynchronize this
            within = np.repeat(bucket_ids, occ_b := np.asarray(
                self.occupancy[b * p3 : (b + 1) * p3], dtype=np.int64))
            del occ_b
            if k:
                within = np.concatenate([within, a_bucket[a_edge[b] : a_edge[b + 1]] - b * p3])
            # STABLE, and over the concatenation main-then-arena: that is the
            # exact order `_repack_reference` produces, and the identity gate
            # compares against it elementwise.
            order = _stable_order(within, p3)
            cat_off = self.off[mp : mp + m]
            cat_w = self.w[mp : mp + m]
            if k:
                cat_off = np.concatenate([cat_off, a_off[a_edge[b] : a_edge[b + 1]]])
                cat_w = np.concatenate([cat_w, a_w[a_edge[b] : a_edge[b + 1]]])
            else:
                cat_off, cat_w = cat_off.copy(), cat_w.copy()
            scratch = max(scratch, cat_off.nbytes + cat_w.nbytes + within.nbytes + order.nbytes)
            self.off[ns : ns + m + k] = cat_off[order]
            self.w[ns : ns + m + k] = cat_w[order]
            if self.ids is not None:
                cat_i = self.ids[mp : mp + m]
                if k:
                    cat_i = np.concatenate([cat_i, a_ids[a_edge[b] : a_edge[b + 1]]])
                else:
                    cat_i = cat_i.copy()
                self.ids[ns : ns + m + k] = cat_i[order]
            new_occ[b * p3 : (b + 1) * p3] = np.bincount(within[order], minlength=p3)
            # ZERO THE SPARE. The out-of-place form allocates `zeros_like` and
            # writes only live rows, so every non-live byte is 0 (ids -1). Left
            # alone, an in-place repack would carry stale payload in the gaps --
            # semantically dead, since liveness is derived from `occupancy`, but
            # it would make two states with identical live content differ
            # bytewise, and this project compares states bytewise. The gap sits
            # above `new_start[b] >= main_pos[b]`, so it cannot reach a main
            # block still waiting to be read.
            gap_lo, gap_hi = ns + m + k, int(new_start[b + 1])
            if gap_hi > gap_lo:
                self.off[gap_lo:gap_hi] = 0
                self.w[gap_lo:gap_hi] = 0
                if self.ids is not None:
                    self.ids[gap_lo:gap_hi] = -1

        # everything past the new allocation, arena included: the arena is empty
        # after a fold-in, so it must READ empty too
        self.off[n_alloc:] = 0
        self.w[n_alloc:] = 0
        if self.ids is not None:
            self.ids[n_alloc:] = -1
        # CONTENTS, not bindings: the pool executor shares these arrays with
        # worker processes through shared memory, so their addresses must
        # survive a repack. Both shapes are build-time-fixed (n_bricks + 1 and
        # n_buckets), so the copy-back is exact, and `_to_index` still guards
        # the narrowing. `arena_base` is a scalar and rides the per-step task
        # header instead.
        self.brick_start[...] = new_start
        self.occupancy[...] = _to_index(new_occ, self.index_dtype, "repacked")
        self.arena_base = n_alloc
        self.arena_bucket[:] = -1
        self._invalidate_arena_index()
        return dict(
            slots_used=n_alloc,
            slots_per_particle=n_alloc / max(self.n_particles, 1),
            # EVERYTHING transient, not just the largest buffer. The figure this
            # function replaces omitted three O(N) arrays and read as a constant.
            scratch_bytes=int(scratch),
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
        # what per-brick velocity scales cost, counted rather than described.
        # 8 B per brick against the 32 B PER PARTICLE the global scale forced the
        # engine to hold: 16.8 MB against 274.9 GB at C-gh.
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
