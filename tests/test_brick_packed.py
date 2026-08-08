"""The brick-packed layout: spare pooled per brick, buckets packed tight.

WHAT THIS DESIGN CHANGES, and therefore what has to be tested.

`BrickLayout` gives every bucket its own spare slots and its own stored slot
boundary. Both are per-bucket costs, and at the ratified ~8 particles per bucket
both are large: whole-slot granularity forces >= 12.5% of payload however small
the slack setting, and the int64 boundary array is 8 B per bucket, a full
1.00 B/p at C-gh.

Here the spare belongs to the brick (~512 buckets) and bucket boundaries are a
prefix sum of the occupancy index we already pay for. So the two properties that
need pinning are:

  1. bucket boundaries derived from `occupancy` agree with where particles
     actually are -- if they do not, the stored index is lying and every
     position in the box decodes against the wrong bucket origin;
  2. a brick's run stays packed with no holes, since a hole would make the
     derived boundaries wrong for every later bucket in that brick.

Plus the invariants both layouts share: particles are conserved, overflow raises
rather than clamping (D-007), and `check()` fails when the layout is broken.
"""

import numpy as np
import pytest

from inexor.codec import T9Layout
from inexor.layout import BrickPackedLayout

L_BOX = 128.0
N_PART = 32
# BRICKS_PER_SIDE = 2 is load-bearing, not arbitrary. It gives 512 buckets and
# 4096 particles per brick -- EXACTLY C-gh's ratios (1024^3 buckets / 128^3
# bricks, 8 particles per bucket). Every claim this design rests on scales with
# buckets-per-brick, so a fixture with small bricks silently tests a different
# design: at 8 buckets per brick the boundary array is 0.125 B/p rather than
# 0.002, the per-brick spare floor bites, and brick migration is only 1.9x below
# bucket migration instead of ~7x. All three assertions below failed on such a
# fixture and pass on this one, which is the tell that the geometry IS the test.
BRICKS_PER_SIDE = 2


def _t9(bucket_cells=2):
    return T9Layout(box_size=L_BOX, n_part=N_PART, bucket_cells=bucket_cells)


def _lattice(seed, jitter=0.35):
    rng = np.random.default_rng(seed)
    g = (np.arange(N_PART) + 0.5) * (L_BOX / N_PART)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    return np.mod(q + rng.normal(scale=jitter * L_BOX / N_PART, size=q.shape), L_BOX)


def _build(seed=0, **kw):
    t9 = _t9()
    x = _lattice(seed)
    return BrickPackedLayout.build(x, t9, BRICKS_PER_SIDE, **kw), x, t9


def test_bucket_boundaries_are_derived_not_stored():
    """Property 1. `bucket_slot_starts` reconstructs from occupancy alone; every
    particle in that span must actually belong to that bucket."""
    lay, _, _ = _build()
    lay.check()
    p3 = lay.buckets_per_brick
    for brick in (0, lay.n_bricks // 3, lay.n_bricks - 1):
        starts = lay.bucket_slot_starts(brick)
        assert starts[0] == lay.brick_slot_range(brick)[0]
        for i in range(p3):
            span = lay.slot_to_particle[starts[i] : starts[i + 1]]
            assert np.all(span >= 0), "hole inside a bucket's derived span"
            assert np.all(lay.key[span] == brick * p3 + i)


def test_no_per_bucket_boundary_array_exists():
    """The 1.00 B/p at C-gh that this design removes. If a bucket-length int64
    array reappears the saving is gone, so it is asserted rather than assumed."""
    lay, _, _ = _build()
    assert not hasattr(lay, "bucket_start")
    assert lay.brick_start.size == lay.n_bricks + 1
    b = lay.bytes_per_particle()
    assert b["brick_start"] < 0.01, "brick boundaries should be negligible per particle"
    assert b["bucket_index"] == pytest.approx(0.25, rel=0.01)


def test_spare_is_the_requested_fraction_with_no_granularity_floor():
    """The point of pooling. Per-bucket, ceil() forces one whole slot per
    occupied bucket -- 12.5% at ~8 particles per bucket -- however small the
    setting. Pooled over ~4096 particles, 10% means 10%."""
    lay, _, _ = _build(brick_slack=0.10)
    assert lay.n_slots / lay.n_particles == pytest.approx(1.10, abs=0.005)
    lay2, _, _ = _build(brick_slack=0.02)
    assert lay2.n_slots / lay2.n_particles == pytest.approx(1.02, abs=0.005)


def test_migration_conserves_particles_and_keeps_runs_packed():
    lay, x, t9 = _build(seed=1, brick_slack=0.50)
    rng = np.random.default_rng(2)
    for _ in range(4):
        x = np.mod(x + rng.normal(scale=0.3 * t9.spacing, size=x.shape), L_BOX)
        st = lay.migrate(x)
        lay.check()
    assert st["bucket_migrant_frac"] > 0.05


def test_brick_migration_is_far_below_bucket_migration():
    """The physical claim the design rests on: a brick is ~512 buckets, so a
    particle must travel much further to leave one. Measured here so the claim
    is a number rather than an argument -- but note it is NOT a stability
    guarantee, since a brick hosting a collapsing halo still outgrows a fixed
    capacity (which is why repack exists)."""
    lay, x, t9 = _build(seed=3, brick_slack=0.50)
    rng = np.random.default_rng(4)
    x = np.mod(x + rng.normal(scale=0.3 * t9.spacing, size=x.shape), L_BOX)
    st = lay.migrate(x)
    assert st["brick_migrant_frac"] < st["bucket_migrant_frac"] / 4, (
        f"brick {st['brick_migrant_frac']:.1%} vs bucket "
        f"{st['bucket_migrant_frac']:.1%} -- check buckets_per_brick"
    )


def test_overflow_escalates_to_the_arena_then_refuses():
    """D-007, both rungs. Measured on the real trajectory, a brick hosting a
    collapsing halo outgrows ANY fixed spare fraction -- 10%, 15% and 20% all
    overflowed, by 37, 248 and 461 particles respectively, the count rising
    only because more spare lets the run reach heavier clustering before it
    fails. But 461 of 2.1e6 is 0.02%, which is a rare-event problem rather than
    a sizing one, so a small arena absorbs it and only an exhausted arena is a
    refusal."""
    t9 = _t9()
    rng = np.random.default_rng(6)
    clump = np.mod(rng.normal(loc=L_BOX * 0.5, scale=L_BOX * 0.01, size=(N_PART**3, 3)), L_BOX)

    # with an arena: absorbed, and the layout stays consistent
    lay = BrickPackedLayout.build(
        _lattice(5), t9, BRICKS_PER_SIDE, brick_slack=0.0, arena_frac=1.0
    )
    st = lay.migrate(clump)
    lay.check()
    assert st["n_overflow"] > 0, "fixture did not overflow; it is not exercising the ladder"
    assert st["arena_used"] > 0

    # without one: a loud refusal, never a clamp or a drop
    lay2 = BrickPackedLayout.build(
        _lattice(5), t9, BRICKS_PER_SIDE, brick_slack=0.0, arena_frac=0.0
    )
    with pytest.raises(ValueError, match="overflow their brick"):
        lay2.migrate(clump)


def test_repack_redistributes_in_place_with_bounded_scratch():
    lay, x, t9 = _build(seed=7, brick_slack=0.50)
    rng = np.random.default_rng(8)
    for _ in range(3):
        x = np.mod(x + rng.normal(scale=0.3 * t9.spacing, size=x.shape), L_BOX)
        lay.migrate(x)
    r = lay.repack(brick_slack=0.10, chunk=1 << 12)
    lay.check()
    payload = lay.n_particles * 9.0
    assert r["scratch_bytes"] < 0.2 * payload, "scratch should be set by the chunk, not by N"
    assert r["slots_used"] <= r["slots_allocated"]


@pytest.mark.parametrize(
    "corrupt,match",
    [
        ("drop", "lost particles"),
        ("duplicate", "two slots"),
        ("hole", "hole inside its packed prefix"),
    ],
)
def test_check_catches_corruption(corrupt, match):
    lay, _, _ = _build(seed=9)
    lay.check()
    lo, _ = lay.brick_slot_range(0)
    if corrupt == "drop":
        lay.slot_to_particle[lo] = -1
    elif corrupt == "duplicate":
        lay.slot_to_particle[lo + 1] = lay.slot_to_particle[lo]
    elif corrupt == "hole":
        lay.slot_to_particle[lo] = -1
        lay.n_particles -= 1
    with pytest.raises(AssertionError, match=match):
        lay.check()
