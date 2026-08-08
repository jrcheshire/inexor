"""The brick-sorted slot layout (D-v2-14 clause 3).

WHAT HAS TO BE TRUE, and why each one is here.

  1. The bucket a particle sits in must be recoverable from its SLOT alone.
     That is the whole reason the T9 codec can store a byte instead of a
     coordinate; if it fails, positions decode to the wrong place and nothing
     downstream would necessarily raise.

  2. Migration must conserve particles exactly. The escalation ladder is
     slack -> arena -> loud refusal and it MAY NEVER CLAMP (D-007): a dropped
     particle deletes mass, and the ones that overflow are the clustered ones a
     halo-grade mock exists to resolve. So the refusal is tested by forcing it.

  3. The brick union must reproduce the ratified probe's. `scripts/v2_g5_core.py`
     is the oracle for the tiled force (D-v2-16 cl.7) and its `tile_members` is
     what every measured tiling number was taken with.

  4. `check()` must FAIL when the layout is broken. An invariant checker that
     passes on corrupted state is worse than none, so each class of corruption
     is injected and the checker is required to catch it.

FIXTURE OCCUPANCY IS LOAD-BEARING. Buckets hold ~8 particles at C-gh (1.0 Mpc/h
buckets, 0.5 Mpc/h spacing), and the layout's behaviour is qualitatively
different at low occupancy: integer slot granularity means a 10% slack target
on a bucket of 1 rounds to 100% slack, and capacity-2 buckets overflow on any
real displacement. So the fixture puts n_part^3 particles on a perturbed
lattice, which reproduces the ~8/bucket the ratified numbers assume. A sparse
fixture would have passed a weaker test and hidden that.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

from v2_g5_core import brick_buckets as probe_brick_buckets  # noqa: E402
from v2_g5_core import choose_brick as probe_choose_brick  # noqa: E402
from v2_g5_core import tile_members as probe_tile_members  # noqa: E402

from inexor.codec import (  # noqa: E402
    T9Layout,
    decode_positions,
    encode_positions,
    roundtrip_positions,
)
from inexor.layout import (  # noqa: E402
    BrickLayout,
    assert_brick_divides_buffer,
    brick_span,
    bucket_ijk_from_key,
    bucket_order_key,
    choose_brick,
)

L_BOX = 128.0
N_PART = 32  # -> n_fine 64, 16^3 buckets, 32768 particles, ~8 per bucket
N_FINE = 64
BRICKS_PER_SIDE = 8  # divides the 16^3 bucket grid: 2 buckets per brick side


@pytest.fixture(autouse=True)
def _x64():
    """Enable x64 for this module only, then restore (test_bispectrum.py pattern)."""
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _t9(bucket_cells=2):
    return T9Layout(box_size=L_BOX, n_part=N_PART, bucket_cells=bucket_cells)


def _lattice_positions(seed, jitter=0.35):
    """A perturbed Lagrangian lattice -- what LPT hands the stepper, and the
    occupancy regime the ratified numbers were taken in."""
    rng = np.random.default_rng(seed)
    g = (np.arange(N_PART) + 0.5) * (L_BOX / N_PART)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    return np.mod(q + rng.normal(scale=jitter * L_BOX / N_PART, size=q.shape), L_BOX)


def _clustered_positions(seed, n):
    """Half in one tight clump, half uniform: the regime that fills buckets."""
    rng = np.random.default_rng(seed)
    clump = rng.normal(loc=L_BOX * 0.5, scale=L_BOX * 0.004, size=(n // 2, 3))
    field = rng.uniform(0.0, L_BOX, size=(n - n // 2, 3))
    return np.mod(np.concatenate([clump, field]), L_BOX)


def _layout(seed=0, bucket_cells=2, **kw):
    t9 = _t9(bucket_cells)
    x = _lattice_positions(seed)
    return BrickLayout.build(x, t9, BRICKS_PER_SIDE, **kw), x, t9


def test_fixture_has_the_occupancy_the_ratified_numbers_assume():
    """Guards every test below: at ~1 particle per bucket the layout behaves
    qualitatively differently and the suite would be testing a regime we never
    run in."""
    lay, _, _ = _layout(seed=0)
    occ = lay.occupancy.astype(np.int64)
    assert lay.n_particles == N_PART**3
    assert occ.mean() == pytest.approx(8.0, rel=0.01)
    assert occ.max() < 40  # a perturbed lattice, not a clump


# ------------------------------------------------- the bucket is implied


def test_bucket_is_recoverable_from_the_slot_alone():
    """Property 1. No per-particle bucket id is stored anywhere; the ordinal
    comes from searchsorted on the run boundaries and must agree with the
    bucket the encoder derived from the position."""
    lay, x, t9 = _layout(seed=1)
    lay.check()

    _, direct, _ = bucket_order_key(x, t9, lay.bricks_per_side)
    from_slots = bucket_ijk_from_key(
        lay.bucket_ordinal_of_slots(0, lay.n_slots), t9, lay.bricks_per_side
    )
    live = lay.slot_to_particle[: lay.n_slots] >= 0
    p = lay.slot_to_particle[: lay.n_slots][live]
    assert np.array_equal(from_slots[live], direct[p])


def test_positions_decode_correctly_through_the_layout():
    """The end-to-end statement the tier rests on: a byte in a slot, plus the
    index, reproduces the position to within the quantum -- with no bucket id
    ever stored."""
    import jax.numpy as jnp

    lay, x, t9 = _layout(seed=2)
    off, _ = encode_positions(jnp.asarray(x), t9)

    b_ijk = lay.bucket_ijk_of_slots(0, lay.n_slots)
    live = lay.slot_to_particle[: lay.n_slots] >= 0
    p = lay.slot_to_particle[: lay.n_slots][live]

    xd = decode_positions(
        jnp.asarray(np.asarray(off)[p]), jnp.asarray(b_ijk[live]), t9, fdtype=jnp.float64
    )
    err = np.abs(np.asarray(xd) - x[p])
    err = np.minimum(err, L_BOX - err)
    assert err.max() <= 0.5 * t9.quantum * (1 + 1e-12)


def test_bucket_order_key_round_trips():
    lay, x, t9 = _layout(seed=3)
    key, ijk, _ = bucket_order_key(x, t9, lay.bricks_per_side)
    assert np.array_equal(bucket_ijk_from_key(key, t9, lay.bricks_per_side), ijk)


def test_a_bricks_buckets_are_contiguous_in_slot_order():
    """Brick-major ordering is what lets one array serve both grids: if a
    brick's buckets were not contiguous the brick CSR would have to be a second
    index rather than a coarser prefix over the same slots.

    Empty buckets own zero slots, so what is asserted is that the ordinals
    present in a brick's run are a subset of that brick's block and nothing
    else's -- not that all of them appear.
    """
    lay, _, _ = _layout(seed=4)
    per3 = lay.buckets_per_brick_side**3
    for brick in (0, 5, lay.bricks_per_side**3 - 1):
        lo, hi = lay.brick_slot_range(brick)
        ordinals = np.unique(lay.bucket_ordinal_of_slots(lo, hi))
        assert ordinals.min() >= brick * per3
        assert ordinals.max() < (brick + 1) * per3


# ----------------------------------------------- agreement with the probe


def test_choose_brick_matches_the_probe():
    for n_tile in (16, 32, 64):
        for b_fine in (0, 4, 8, 16):
            assert choose_brick(n_tile, b_fine, N_FINE) == probe_choose_brick(
                n_tile, b_fine, N_FINE
            )


def test_tile_membership_matches_the_probe_exactly_on_quantized_positions():
    """Property 3, on MATCHED INPUTS. The layout buckets the stored (quantized)
    position, because that is the only coordinate the state actually has; the
    probe buckets whatever float it is handed. Fed the same quantized
    positions, the two must agree exactly -- one reaching membership through
    concatenated CSR slices, the other through contiguous slot runs.
    """
    import jax.numpy as jnp

    n_tile, b_fine = 16, 8
    n_brick = choose_brick(n_tile, b_fine, N_FINE)
    assert_brick_divides_buffer(n_tile, b_fine, n_brick, N_FINE)
    nb = N_FINE // n_brick
    assert nb == BRICKS_PER_SIDE

    t9 = _t9()
    x = _lattice_positions(seed=5)
    lay = BrickLayout.build(x, t9, nb)

    xq = np.asarray(roundtrip_positions(jnp.asarray(x), t9))
    order, starts, nb_probe = probe_brick_buckets(xq, N_FINE, n_brick, L_BOX / N_FINE)
    assert nb_probe == nb

    for tijk in ((0, 0, 0), (1, 2, 3), (nb - 1, 0, nb - 1)):
        t = np.asarray(tijk)
        mine = lay.tile_members(t, n_tile, b_fine, n_brick, N_FINE)
        theirs = probe_tile_members(order, starts, nb, t, n_tile, b_fine, n_brick)
        assert np.array_equal(np.sort(mine), np.sort(theirs)), f"tile {tijk}"
        assert len(np.unique(mine)) == len(mine), "a particle appears twice in one tile"


def test_quantization_moves_only_particles_within_half_a_quantum_of_a_boundary():
    """The measured size of the effect above, pinned so it stays small and so
    the choice of input is not mistaken for a free parameter. Bucketing the raw
    float instead of the stored position disagrees only for particles sitting
    within half a quantum of a brick face -- a boundary effect, well under a
    percent, and following the STORED position is the correct side of it: a
    particle must be gathered into the tile its state says it is in."""
    import jax.numpy as jnp

    n_tile, b_fine = 16, 8
    n_brick = choose_brick(n_tile, b_fine, N_FINE)
    nb = N_FINE // n_brick
    t9 = _t9()
    x = _lattice_positions(seed=6)
    lay = BrickLayout.build(x, t9, nb)

    order, starts, _ = probe_brick_buckets(x, N_FINE, n_brick, L_BOX / N_FINE)
    xq = np.asarray(roundtrip_positions(jnp.asarray(x), t9))
    order_q, starts_q, _ = probe_brick_buckets(xq, N_FINE, n_brick, L_BOX / N_FINE)

    n_diff = n_total = 0
    for tijk in ((0, 0, 0), (1, 2, 3), (nb - 1, 0, nb - 1)):
        t = np.asarray(tijk)
        mine = lay.tile_members(t, n_tile, b_fine, n_brick, N_FINE)
        raw = probe_tile_members(order, starts, nb, t, n_tile, b_fine, n_brick)
        quant = probe_tile_members(order_q, starts_q, nb, t, n_tile, b_fine, n_brick)
        assert np.array_equal(np.sort(mine), np.sort(quant))  # exact on matched inputs
        n_diff += len(np.setxor1d(mine, raw))
        n_total += len(mine)
    assert 0 < n_diff, "no boundary particles at all -- the fixture cannot see the effect"
    assert n_diff / n_total < 0.01, f"boundary disagreement {n_diff}/{n_total} is not a rounding edge"


def test_brick_span_wrap_guard_still_refuses_the_measured_bug():
    """n_fine=64, n_tile=32, b=20 gave span 6 against a 4-brick grid and a 3.29
    relative short-force error -- silent double-painting. Pinned so it cannot
    come back through the promotion."""
    with pytest.raises(ValueError, match="double-count"):
        brick_span(32, 20, 8, 4)


def test_assert_brick_divides_buffer_catches_the_overhang_case():
    """T128/b96 -> brick 64, union side 384 against P=320: the one V4a leg with
    3.04e9 overhang and a ~1.7x inflated cap."""
    assert_brick_divides_buffer(128, 32, 32, 512)  # the operating geometry is fine
    with pytest.raises(ValueError, match="does not divide"):
        assert_brick_divides_buffer(128, 96, 64, 512)


# ---------------------------------------------------------- migration


def test_migration_conserves_particles_and_updates_buckets():
    lay, x, t9 = _layout(seed=7, slack_frac=0.10, arena_frac=0.10)
    rng = np.random.default_rng(8)
    x_new = np.mod(x + rng.normal(scale=0.4 * t9.spacing, size=x.shape), L_BOX)

    stats = lay.migrate(x_new)
    lay.check()
    assert stats["n_migrants"] > 0, "the displacement was too small to move anything"

    key_new, _, _ = bucket_order_key(x_new, t9, lay.bricks_per_side)
    got = lay._bucket_of_slot(lay.particle_to_slot)
    assert np.array_equal(got, key_new), "a particle is not in the bucket it now belongs to"


def test_ten_percent_slack_does_not_absorb_a_step_on_its_own():
    """A MEASUREMENT, pinned because it bears on a ratified number.

    D-v2-14 clause 2 prices slack at 0.90 B/p -- 10% of payload -- and says
    plainly that this is an estimate from a hand argument about migration
    rates. Measured on this fixture, a 10% target does not behave like 10%:

      realized slack        1.20 B/p, not 0.90
      migrants, 0.1 spacing  9.7% of particles, 1.6% of which need the arena
      migrants, 0.4 spacing 39.7% of particles, 6.6% of which need the arena
      buying it down        25% slack -> 2.65 B/p and still 0.3-3.4% arena
                            50% slack -> 4.78 B/p and still 0.05-1.1% arena

    WHAT THIS IS NOT. The displacement here is an incoherent per-particle
    random walk, which is a pessimistic proxy: real PM displacement is
    COHERENT, so neighbouring particles leave a bucket together and the arrival
    statistics are not Poisson. These numbers therefore bound the mechanism
    rather than measure the operating point -- which is precisely why M-v2-1's
    exit gate is a real evolution at cgh64 and not this.

    What is established here is the MECHANISM: integer slot granularity plus
    the occupancy spread (3 to 14 particles around a mean of 8) means a
    fractional slack target buys less headroom than its name suggests, because
    the sparse buckets carry a whole spare slot each.
    """
    lay, x, t9 = _layout(seed=19, slack_frac=0.10, arena_frac=0.50)
    assert lay.bytes_per_particle()["slack"] > 0.90

    rng = np.random.default_rng(20)
    x_new = np.mod(x + rng.normal(scale=0.1 * t9.spacing, size=x.shape), L_BOX)
    stats = lay.migrate(x_new)
    lay.check()
    assert stats["arena_used"] > 0, (
        "10% slack absorbed the whole step -- if this ever passes, the estimate "
        "is better than measured here and the reason should be understood"
    )
    assert stats["migrant_frac"] > 0.05


def test_migration_is_a_no_op_when_nothing_moves():
    lay, x, _ = _layout(seed=9)
    before = lay.slot_to_particle.copy()
    stats = lay.migrate(x)
    assert stats["n_migrants"] == 0
    assert np.array_equal(lay.slot_to_particle, before)
    lay.check()


def test_repeated_migration_stays_consistent():
    """The layout is edited in place and never re-sorted, so bookkeeping errors
    would accumulate silently rather than raise. Walk it several steps."""
    lay, x, t9 = _layout(seed=10, slack_frac=0.10, arena_frac=0.10)
    rng = np.random.default_rng(11)
    for _ in range(6):
        x = np.mod(x + rng.normal(scale=0.25 * t9.spacing, size=x.shape), L_BOX)
        lay.migrate(x)
        lay.check()
    key, _, _ = bucket_order_key(x, t9, lay.bricks_per_side)
    assert np.array_equal(lay._bucket_of_slot(lay.particle_to_slot), key)


def test_overflow_escalates_to_the_arena_before_refusing():
    """Collapse the box into a clump with no slack, so buckets must overflow.
    The arena has to absorb it and the layout stay consistent."""
    t9 = _t9()
    n = N_PART**3
    lay = BrickLayout.build(_lattice_positions(seed=12), t9, BRICKS_PER_SIDE,
                            slack_frac=0.0, arena_frac=0.9)
    stats = lay.migrate(_clustered_positions(seed=13, n=n))
    lay.check()
    assert stats["arena_used"] > 0, "nothing overflowed; the fixture is not exercising the ladder"


def test_overflow_refuses_loudly_rather_than_dropping():
    """D-007. With no slack and no arena there is nowhere to put a migrant, and
    the only acceptable behaviour is to raise. A silent drop deletes mass."""
    t9 = _t9()
    n = N_PART**3
    lay = BrickLayout.build(_lattice_positions(seed=14), t9, BRICKS_PER_SIDE,
                            slack_frac=0.0, arena_frac=0.0)
    with pytest.raises(ValueError, match="arena of 0 slots has only 0 free"):
        lay.migrate(_clustered_positions(seed=15, n=n))


# ------------------------------------------------------- refusals, checks


def test_build_refuses_an_incommensurate_brick_grid():
    with pytest.raises(ValueError, match="must divide the bucket grid"):
        BrickLayout.build(_lattice_positions(seed=16), _t9(), bricks_per_side=7)


@pytest.mark.parametrize(
    "corrupt,match",
    [
        ("drop_a_particle", "lost particles"),
        ("duplicate_a_particle", "two slots"),
        ("hole_in_prefix", "hole inside its live prefix"),
        ("live_past_occupancy", "live slot past its occupancy"),
        ("occupancy_over_capacity", "occupancy exceeds capacity"),
    ],
)
def test_check_catches_each_class_of_corruption(corrupt, match):
    """Property 4. Every invariant `check` claims is broken on purpose and it
    must notice. Corruptions are chosen so exactly ONE check fires -- a
    corruption that trips the particle count first would not tell us the
    prefix checks work."""
    lay, _, _ = _layout(seed=17)
    lay.check()
    b = int(np.argmax(lay.occupancy))
    lo = int(lay.bucket_start[b])
    occ = int(lay.occupancy[b])
    assert occ >= 2, "fixture bucket too small to corrupt meaningfully"

    if corrupt == "drop_a_particle":
        lay.slot_to_particle[lo] = -1
    elif corrupt == "duplicate_a_particle":
        lay.slot_to_particle[lo + 1] = lay.slot_to_particle[lo]
    elif corrupt == "hole_in_prefix":
        lay.slot_to_particle[lo] = -1
        lay.n_particles -= 1  # so the count check passes and the hole check fires
    elif corrupt == "live_past_occupancy":
        lay.occupancy[b] = occ - 1  # same live slots, one now past the occupancy
    elif corrupt == "occupancy_over_capacity":
        lay.occupancy[b] = int(lay.capacity[b]) + 1

    with pytest.raises(AssertionError, match=match):
        lay.check()


# ------------------------------------------------------------ accounting


def test_slack_accounting_is_self_consistent_and_exceeds_the_estimate():
    """The D-v2-14 clause 2 table computed off a layout that exists rather than
    quoted. It also records a real consequence of integer slots: a 10% slack
    TARGET does not give 0.90 B/p at 8 particles per bucket, because ceil(0.8)
    is 1 whole slot -- 12.5%. The exit-gate measurement at cgh64 is what
    settles the operating number; this pins the mechanism."""
    lay, _, _ = _layout(seed=18, slack_frac=0.10, arena_frac=0.0)
    acc = lay.bytes_per_particle()
    assert acc["payload"] == 9.0
    assert acc["slack"] == pytest.approx(9.0 * (lay.n_slots - lay.n_particles) / lay.n_particles)
    assert acc["total"] == pytest.approx(
        sum(acc[k] for k in ("payload", "bucket_index", "brick_csr", "slack", "arena"))
    )
    # Granularity plus the occupancy SPREAD, not a bug. One whole spare slot per
    # bucket would be 9/8 = 1.125 B/p at a uniform 8; the measured 1.20 is
    # higher because occupancy runs 3 to 14 and the sparse buckets pay the same
    # whole slot. Predicting from the mean alone understates it.
    assert acc["slack"] == pytest.approx(1.204, rel=0.02)
    assert acc["slack"] > 9.0 / 8.0, "the distribution effect vanished -- check the fixture"
    assert acc["slack"] > 0.90, "slack came in at or under the estimate -- check the fixture"
