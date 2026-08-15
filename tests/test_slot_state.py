"""SlotState: T9 payload stored in slot order, with the bucket implied (M-v2-3).

The property under test throughout is the one `state.py`'s docstring calls the
whole correctness statement: for every occupied slot, the bucket DERIVED from
where the slot sits equals the bucket the stored offset was encoded against.
Everything else here is either that invariant under a stress, or a guard against
the two codec implementations (jnp in `codec`, numpy in `state`) drifting apart.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

from inexor import layout, state  # noqa: E402
from inexor.codec import T9Layout  # noqa: E402

# Geometry: 32^3 particles, bucket 2 cells -> 16^3 buckets, 2 bricks/side -> 8
# bricks of 8^3 = 512 buckets each. 512 buckets per brick is C-gh's own figure,
# so the occupancy statistics this exercises are the shipped ones rather than a
# toy's. n_part/bucket_cells must be a power of two (D-007, so the wrap is
# modular and not saturating), which 32/2 = 16 satisfies.
L_BOX = 64.0
N_PART = 32
BRICKS = 2


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _t9():
    return T9Layout(box_size=L_BOX, n_part=N_PART, bucket_cells=2)


def _positions(seed=0, n_part=N_PART):
    """A perturbed Lagrangian lattice, as everywhere else in this suite."""
    rng = np.random.default_rng(seed)
    g = (np.arange(n_part) + 0.5) * (L_BOX / n_part)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    return np.mod(q + rng.normal(scale=0.3 * L_BOX / n_part, size=q.shape), L_BOX)


def _velocities(seed=1, n=N_PART**3):
    return np.random.default_rng(seed).normal(scale=2.0, size=(n, 3))


def _built(seed=0, **kw):
    x, v = _positions(seed), _velocities(seed + 1)
    return x, v, state.SlotState.build(x, v, _t9(), BRICKS, **kw)


# ----------------------------------------------- the two codec implementations


def test_the_host_codec_mirror_is_bitwise_the_device_one():
    """`state` re-implements encode/decode in numpy because the layout never puts
    the global position array on the device. Two implementations of one
    definition that are never compared is exactly how they drift apart."""
    import jax.numpy as jnp

    from inexor import codec

    t9 = _t9()
    x = _positions(2)
    off_h, b_h = state.encode_positions_host(x, t9)
    off_d, b_d = codec.encode_positions(jnp.asarray(x), t9)
    assert np.array_equal(off_h, np.asarray(off_d)), "host and device offsets differ"
    assert np.array_equal(b_h, np.asarray(b_d)), "host and device bucket indices differ"

    back_h = state.decode_positions_host(off_h, b_h, t9)
    back_d = np.asarray(codec.decode_positions(off_d, b_d, t9))
    assert np.array_equal(back_h, back_d), "host and device decode differ"
    # and the fixture must actually exercise the byte, or this proves nothing
    assert len(np.unique(off_h)) > 200, "offsets do not span the byte; fixture is degenerate"


def test_the_mirror_check_can_fail():
    """The paired can-it-fail test: perturb one offset by one quantum and the
    comparison above must notice."""
    t9 = _t9()
    x = _positions(3)
    off, b = state.encode_positions_host(x, t9)
    off2 = off.copy()
    off2[0, 0] = (int(off2[0, 0]) + 1) % 256
    a = state.decode_positions_host(off, b, t9)
    c = state.decode_positions_host(off2, b, t9)
    assert not np.array_equal(a, c)


# ------------------------------------------------------------- the invariant


def test_the_container_is_structurally_consistent_as_built():
    _, _, st = _built(0)
    assert st.check() is True
    assert st.n_live == N_PART**3


def test_the_self_consistency_sweep_would_have_been_a_gate_that_cannot_fail():
    """Pins the reason `check()` does NOT decode and compare buckets.

    The obvious invariant -- "decode each slot, assert its position falls in the
    bucket the slot implies" -- is an identity. `decode` rebuilds the lattice
    index as `bucket * 256 + off` with `off` a uint8, so the recovered bucket is
    `(bucket * 256 + off) // 256 == bucket` for EVERY value the byte can hold.
    This corrupts an offset by half a bucket and shows the comparison still
    agrees, which is what makes that sweep worthless on this container.

    Without this test, someone reads `check()`, thinks the strong invariant is
    missing, adds it, and ships a gate that cannot fail on the module whose whole
    premise is that the bucket is implied.
    """
    from inexor.layout import _bucket_ijk

    _, _, st = _built(4)
    b = next(i for i in range(st.n_bricks) if st.brick_live_count(i))
    slot = int(st.brick_start[b])
    st.off[slot, 0] = (int(st.off[slot, 0]) + 128) % 256

    _, x, _ = st.decode_brick(b)
    derived = st.bucket_ijk_of_live_slots(b)
    assert np.array_equal(_bucket_ijk(x, st.t9)[: len(derived)], derived), (
        "a corrupted offset DID move the derived bucket -- if that is real, the "
        "self-consistency sweep is worth having after all and this note is stale"
    )
    # the structural check is likewise blind to it, and honestly so
    assert st.check() is True


def test_a_lost_particle_is_caught():
    """D-007 forbids losing a particle as much as it forbids clamping one, and
    the count is the only thing that sees it."""
    _, _, st = _built(5)
    for b in range(st.n_bricks):
        p3 = st.buckets_per_brick
        occ = st.occupancy[b * p3 : (b + 1) * p3]
        hit = np.nonzero(occ > 0)[0]
        if len(hit):
            occ[hit[-1]] -= 1  # drop the LAST bucket's last row: no span shifts
            break
    with pytest.raises(ValueError, match="reachable through the layout"):
        st.check()


def test_a_brick_overrunning_its_allocation_is_caught():
    """The failure that silently reassigns particles to the next brick rather
    than losing them, so the count test cannot see it."""
    _, _, st = _built(13)
    b = next(i for i in range(st.n_bricks) if st.brick_live_count(i))
    p3 = st.buckets_per_brick
    lo, hi = st.brick_slot_range(b)
    st.occupancy[b * p3] += np.uint32(hi - lo)  # push the run past its allocation
    st.n_particles += int(hi - lo)  # keep the count consistent, so only test 1 fires
    with pytest.raises(ValueError, match="live rows in an allocation of"):
        st.check()


def test_placement_is_checkable_against_a_reference_and_that_check_can_fail():
    """The question `check` cannot answer: did the particle land in the bucket
    its POSITION calls for? Needs the reference array and the id tier."""
    x, v, st = _built(14, with_ids=True)
    assert st.check_placement(x) is True
    # move one reference position a whole bucket and the check must notice
    bad = x.copy()
    bad[0, 0] = np.mod(bad[0, 0] + st.t9.bucket_size, L_BOX)
    with pytest.raises(ValueError, match="other than the one their reference position"):
        st.check_placement(bad)


def test_placement_refuses_without_the_id_tier():
    _, _, st = _built(15)
    with pytest.raises(ValueError, match="needs the opt-in id tier"):
        st.check_placement(np.zeros((N_PART**3, 3)))


# --------------------------------------------------------------- round trip


def test_the_round_trip_is_exact_to_the_quantum_and_the_velocity_scale():
    x, v, st = _built(6)
    n = x.shape[0]
    got_x = np.zeros_like(x)
    got_v = np.zeros_like(v)
    seen = 0
    for b in range(st.n_bricks):
        slots, xb, vb = st.decode_brick(b)
        got_x[seen : seen + len(slots)] = xb
        got_v[seen : seen + len(slots)] = vb
        seen += len(slots)
    assert seen == n
    # order is NOT preserved -- that is the point of the container -- so compare
    # as multisets along each axis
    for ax in range(3):
        a = np.sort(got_x[:, ax])
        e = np.sort(np.mod(np.rint(x[:, ax] / st.t9.quantum), st.t9.n_levels) * st.t9.quantum)
        assert np.allclose(a, e, atol=0.51 * st.t9.quantum), f"axis {ax} positions moved"
    assert np.max(np.abs(np.sort(got_v.ravel()) - np.sort(v.ravel()))) <= st.vel_scale.max()


def test_ids_follow_their_particle_through_the_reordering():
    """The opt-in int32 id tier (D-v2-14 cl.5) earns its keep here: it is the
    handle that lets a test follow a NAMED particle through a layout that
    deliberately does not preserve order."""
    x, v, st = _built(7, with_ids=True)
    for b in range(st.n_bricks):
        slots, xb, _ = st.decode_brick(b)
        if not len(slots):
            continue
        ids = st.ids[slots]
        assert np.all(ids >= 0), "a live slot carries no id"
        want = np.mod(np.rint(x[ids] / st.t9.quantum), st.t9.n_levels) * st.t9.quantum
        assert np.allclose(xb, want, atol=1e-12), f"brick {b}: an id points at another particle"
    assert len(np.unique(st.ids[st.ids >= 0])) == N_PART**3, "ids are not a permutation"


# ----------------------------------------------------- the scaffolding is gone


def test_the_scaffolding_arrays_do_not_exist():
    """D-v2-20's whole point. `key`, `particle_to_slot` and `slot_to_particle` are
    ~21 B/p against a ~10.5 B/p budget and cannot be resident at production scale
    in EITHER integer width, which is why widening `key` was refused rather than
    done."""
    _, _, st = _built(8)
    for name in ("key", "particle_to_slot", "slot_to_particle"):
        assert not hasattr(st, name), f"{name} survived into the engine's container"
    assert st.bytes_per_particle()["scaffold"] == 0.0


def test_the_int32_key_ceiling_is_not_reachable_because_key_is_gone():
    """`layout._refuse_key_overflow` exists because C-hero's 2048^3 bucket grid
    (8.59e9) overflows the int32 `key` (2.15e9). With no `key`, the ceiling is
    not raised -- it is absent. Assert the refusal still guards the OLD layout,
    so this is a statement about the new container and not about a deleted
    guard."""
    with pytest.raises(ValueError, match="exceeds int32"):
        layout._refuse_key_overflow(2**31)


def test_the_all_in_cost_lands_near_the_ratified_figure():
    """D-v2-20's ~10.54 B/p: payload 9.00 + index 0.50 + brick CSR + slack +
    arena. The fixture's bucket grid is far coarser per particle than C-gh's, so
    this is a shape check on the accounting, not a reproduction of the number.

    `brick_scales` JOINED the sum when velocity scales went per brick, so the
    all-in figure moved -- on this fixture by 0.002 B/p, and at C-gh by 0.0020
    (16.8 MB over 8.59e9 particles). Recorded here rather than absorbed: the
    ratified number is a number of record, and it is now larger by a term that
    bought the deletion of 274.9 GB. The trade is the point, and a silent bump
    would hide both halves of it.
    """
    _, _, st = _built(9)
    bpp = st.bytes_per_particle()
    assert bpp["payload"] == 9.0
    assert bpp["ids"] == 0.0
    assert bpp["total"] == pytest.approx(
        sum(bpp[k] for k in ("payload", "bucket_index", "brick_start", "slack", "arena",
                             "ids", "brick_scales"))
    )
    assert bpp["brick_scales"] > 0.0, "the per-brick scales must be counted, not implied"
    # slack is the pooled 10%, and pooling per BRICK is what makes it 10% rather
    # than the 12.5% floor per-bucket granularity forces (D-v2-19 clause 1)
    assert 0.5 < bpp["slack"] < 1.6, f"slack {bpp['slack']:.3f} B/p is not the pooled 10%"


def test_ids_cost_exactly_four_bytes_when_asked_for():
    _, _, st = _built(10, with_ids=True)
    assert st.bytes_per_particle()["ids"] == 4.0


# ------------------------------------------------------- derived boundaries


def test_bucket_boundaries_are_derived_and_agree_with_the_stored_runs():
    """D-v2-19 clause 2 deleted a stored int64 slot boundary per bucket -- 1.00
    B/p at C-gh that the accounting had never counted -- by deriving it from the
    occupancy prefix sum. This asserts the derivation reproduces the runs."""
    _, _, st = _built(11)
    p3 = st.buckets_per_brick
    for b in range(st.n_bricks):
        starts = st.bucket_slot_starts(b)
        occ = st.occupancy[b * p3 : (b + 1) * p3].astype(np.int64)
        assert np.array_equal(np.diff(starts), occ)
        assert starts[0] == st.brick_start[b]
        assert starts[-1] == st.brick_start[b] + occ.sum()


def test_free_slots_need_no_sentinel():
    """Liveness is a function of `occupancy` alone: a brick's live rows are its
    first sum(occupancy) slots. A -1 marker would be a second source of truth
    beside the index, and the two could disagree."""
    _, _, st = _built(12)
    for b in range(st.n_bricks):
        lo, hi = st.brick_slot_range(b)
        m = st.brick_live_count(b)
        assert lo + m <= hi, "a brick holds more live rows than its allocation"
        assert len(st.brick_member_slots(b)) == m + len(st.arena_slots_of_brick(b))


# ==================================== drift and re-home (S3/S4, one fused pass)


def _drifted_reference(x, v, c, t9, scales, bricks_per_side):
    """Where every particle should land, computed independently of the pass.

    Starts from what the container actually HOLDS, not from the caller's inputs:
    the stored position is on the T9 lattice and the stored velocity is an int16
    code at ITS OWN BRICK's scale. Referencing the raw inputs instead makes
    particles within half a quantum of a bucket face disagree for a legitimate
    reason and reads as a bug in the exchange -- which is how this helper was
    first written.

    The brick is re-derived from the position through `bucket_order_key`, the
    same definition `SlotState.build` quantizes against, rather than read back
    out of the container. Reading it back would make this compare the pass with
    itself.
    """
    q = t9.quantum
    i0 = np.mod(np.rint(x / q).astype(np.int64), t9.n_levels)
    key, _, _ = layout.bucket_order_key(x, t9, int(bricks_per_side))
    per3 = (t9.n_buckets_side // int(bricks_per_side)) ** 3
    s = np.asarray(scales)[key // per3][:, None]
    v_q = np.rint(v / s).astype(np.int16).astype(np.float64) * s
    i = np.mod(np.rint(i0 + (c * v_q) / q).astype(np.int64), t9.n_levels)
    return i // 256


def test_the_pass_conserves_every_particle():
    """D-007 forbids losing a particle as much as clamping one, and a re-home is
    where one would go missing."""
    _, _, st = _built(20)
    n0 = st.n_live
    for _ in range(3):
        state.drift_and_migrate(st, 0.05)
        assert st.check() is True
        assert st.n_live == n0


def test_every_particle_lands_in_the_bucket_its_drifted_position_calls_for():
    """The pass's actual job, checked against an independent computation of the
    destination rather than against the pass's own arithmetic."""
    x, v, st = _built(21, with_ids=True)
    c = 0.05
    want = _drifted_reference(x, v, c, st.t9, st.vel_scale, st.bricks_per_side)
    state.drift_and_migrate(st, c)
    st.check()
    for b in range(st.n_bricks):
        slots = st.brick_member_slots(b)
        if not len(slots):
            continue
        got = st._bucket_ijk_of_slots(b, slots)
        assert np.array_equal(got, want[st.ids[slots]]), f"brick {b}: wrong destination bucket"


def test_particles_cross_bucket_brick_and_the_periodic_seam():
    """The fixture has to actually exercise the three crossings, or the
    destination test above passes on a pass that never moves anything.

    Needs a large drift: the bucket is 4 Mpc/h here and a brick side is 32, so
    the ordinary fixture's ~0.1 Mpc/h step per unit coefficient reaches neither.
    """
    x = _positions(22)
    v = np.random.default_rng(23).normal(scale=40.0, size=(N_PART**3, 3))
    st = state.SlotState.build(x, v, _t9(), BRICKS, with_ids=True, arena_frac=0.25)
    c = 1.0
    before_b = np.mod(np.rint(x / st.t9.quantum).astype(np.int64), st.t9.n_levels) // 256
    after_b = _drifted_reference(x, v, c, st.t9, st.vel_scale, st.bricks_per_side)
    per = st.t9.n_buckets_side // st.bricks_per_side
    nbk = st.t9.n_buckets_side

    moved_bucket = np.any(before_b != after_b, axis=1)
    moved_brick = np.any(before_b // per != after_b // per, axis=1)
    # a seam crossing: the drift is large enough that the SHORT way round the
    # box is not the way the coordinate moved, i.e. the wrap fired
    raw = np.rint(x / st.t9.quantum + (c * v) / st.t9.quantum).astype(np.int64)
    seam = np.any((raw < 0) | (raw >= st.t9.n_levels), axis=1)

    assert moved_bucket.sum() > 0.5 * len(x), "fixture barely changes bucket"
    assert moved_brick.sum() > 0.1 * len(x), "too few particles change BRICK"
    assert seam.sum() > 0, "no particle crosses the periodic seam"
    assert nbk == 16  # the geometry this reasoning is written against

    state.drift_and_migrate(st, c)
    assert st.check() is True
    assert st.n_live == len(x)
    for b in range(st.n_bricks):
        slots = st.brick_member_slots(b)
        if not len(slots):
            continue
        got = st._bucket_ijk_of_slots(b, slots)
        assert np.array_equal(got, after_b[st.ids[slots]]), (
            f"brick {b}: a particle crossing a brick or the seam landed wrong"
        )


def test_a_multi_brick_x_mover_survives_and_lands_right():
    """The missing particle of 16,777,216 (antares 442), at unit scale.

    `_insert_slab` consumed immigrants from hard-coded +-1 sources while the
    schedule staged the realized reach, so a particle crossing TWO bricks in x
    was staged correctly, matched by no insert, and destroyed by the release
    loop. Needs >= 4 bricks per side: at this suite's usual nb=2, {bx-1,bx,bx+1}
    mod 2 covers every slab and the defect is invisible -- which is why the
    suite was green while cdev lost a particle.
    """
    nb4 = 4  # brick extent 16.0 at this fixture's box of 64
    x = _positions(30)
    v = np.random.default_rng(31).normal(scale=0.05, size=(N_PART**3, 3))
    # mid-brick starts so the crossing count is unambiguous; movers in +x, -x
    # (across the periodic seam), +y, and a diagonal -- one variable per row
    x[0], v[0] = (8.0, 8.0, 8.0), (28.0, 0.0, 0.0)  # slab 0 -> 2
    x[1], v[1] = (8.0, 40.0, 8.0), (-28.0, 0.0, 0.0)  # slab 0 -> 2 the short way round
    x[2], v[2] = (40.0, 8.0, 8.0), (0.0, 28.0, 0.0)  # same slab, 2 bricks in y
    x[3], v[3] = (40.0, 40.0, 40.0), (28.0, 28.0, 28.0)  # 2 bricks on every axis
    st = state.SlotState.build(x, v, _t9(), nb4, with_ids=True, arena_frac=0.25)
    c = 1.0
    assert state.brick_reach(st, c) >= 2, "fixture does not reach 2 bricks"
    want = _drifted_reference(x, v, c, st.t9, st.vel_scale, st.bricks_per_side)
    state.drift_and_migrate(st, c)
    assert st.check() is True
    assert st.n_live == len(x)
    for b in range(st.n_bricks):
        slots = st.brick_member_slots(b)
        if not len(slots):
            continue
        got = st._bucket_ijk_of_slots(b, slots)
        assert np.array_equal(got, want[st.ids[slots]]), (
            f"brick {b}: a multi-brick mover landed in the wrong bucket"
        )


def test_the_release_census_fires_on_a_dropped_emigrant(monkeypatch):
    """The census must be able to FAIL, or it is a gate that cannot fail.

    Reinstate the old defect -- consumption pinned to +-1 sources regardless of
    the schedule's reach -- and require the release loop to refuse AT the release,
    naming the unconsumed rows, rather than let the count guard catch it 200
    lines later (or not at all).
    """
    orig = state.SlotState._insert_slab

    def pinned(self, bx, staged, emig, reach=(-1, 0, 1), consumed=None, scales=None):
        return orig(self, bx, staged, emig, (-1, 0, 1), consumed, scales=scales)

    monkeypatch.setattr(state.SlotState, "_insert_slab", pinned)
    x = _positions(30)
    v = np.random.default_rng(31).normal(scale=0.05, size=(N_PART**3, 3))
    x[0], v[0] = (8.0, 8.0, 8.0), (28.0, 0.0, 0.0)
    st = state.SlotState.build(x, v, _t9(), 4, with_ids=True, arena_frac=0.25)
    with pytest.raises(AssertionError, match="unconsumed"):
        state.drift_and_migrate(st, 1.0)


def test_a_particle_is_drifted_exactly_once():
    """The dangerous failure in a two-phase exchange: a record inserted into a
    brick that has not yet been ejected gets drifted again, producing a slightly
    wrong trajectory with nothing raising. Eject-before-insert makes it
    structurally impossible; this asserts the structure rather than trusting it.

    Two drifts of c would put a particle at 2c, so comparing against the
    single-drift reference catches it -- and the ids make it per particle."""
    x, v, st = _built(23, with_ids=True)
    c = 0.08
    once = _drifted_reference(x, v, c, st.t9, st.vel_scale, st.bricks_per_side)
    twice = _drifted_reference(x, v, 2 * c, st.t9, st.vel_scale, st.bricks_per_side)
    assert np.any(once != twice), "the fixture cannot tell one drift from two"
    state.drift_and_migrate(st, c)
    for b in range(st.n_bricks):
        slots = st.brick_member_slots(b)
        if not len(slots):
            continue
        got = st._bucket_ijk_of_slots(b, slots)
        assert np.array_equal(got, once[st.ids[slots]])


def test_the_velocity_scale_reconciliation_is_the_exact_global_max():
    """The claim the whole scheme rests on: tile ownership is a partition, so a
    max over per-tile scales IS the global scale, not an approximation."""
    rng = np.random.default_rng(24)
    v = rng.normal(size=(5000, 3))
    tile = rng.integers(0, 37, size=5000)
    scales = [np.max(np.abs(v[tile == t])) / 32767 for t in range(37) if np.any(tile == t)]
    assert state.reconcile_velocity_scale(scales) == np.max(np.abs(v)) / 32767


def test_rescaling_a_velocity_code_never_escapes_int16():
    """Not a margin -- a consequence of s_new being the exact global max."""
    from inexor.codec import assert_int16_range

    rng = np.random.default_rng(25)
    v = rng.normal(scale=3.0, size=(4000, 3))
    s_tile = np.max(np.abs(v[:1000])) / 32767  # a tile below the global max
    s_glob = np.max(np.abs(v)) / 32767
    w = np.rint(v[:1000] / s_tile).astype(np.int16)
    assert_int16_range(w)
    assert_int16_range(state._rescale_w(w, s_tile, s_glob))


def test_a_changed_velocity_scale_moves_no_particle_further_than_one_quantum():
    x, v, st = _built(26)
    s0 = st.vel_scale.copy()
    # There is no external scale setter any more: a brick's scale is fixed by
    # `_insert_slab` over the membership it writes. A zero drift keeps every
    # particle where it is, so every brick re-derives the SAME scale from the
    # same rows -- which is the degenerate case worth pinning, because it says
    # the re-derivation is idempotent rather than drifting a little each step.
    state.drift_and_migrate(st, 0.0)
    assert st.check() is True
    assert np.array_equal(st.vel_scale, s0), "a zero drift moved a brick's scale"
    seen = 0
    for b in range(st.n_bricks):
        slots, _, vb = st.decode_brick(b)
        seen += len(slots)
    assert seen == st.n_particles


def test_the_arena_absorbs_a_brick_overflow_and_then_refuses():
    """The D-007 ladder end to end: spare, then arena, then a loud refusal --
    and never a clamp or a drop.

    `brick_slack=0.0` leaves each brick exactly its build-time count, so any net
    inflow overflows. The drift converges every particle toward the box centre,
    which is the physical version of the failure: a collapsing halo outgrowing
    its brick.
    """
    x = _positions(27)
    # Converge on ONE brick's centre, not the box centre: with 2 bricks per side
    # the box centre is the corner where all eight meet, so convergence there is
    # roughly balanced and nothing overflows.
    v = (L_BOX * 0.25 - x) * 2.0
    st = state.SlotState.build(x, v, _t9(), BRICKS, brick_slack=0.0, arena_frac=0.30)
    stats = state.drift_and_migrate(st, 0.15)
    assert st.check() is True
    assert st.n_live == x.shape[0], "the ladder lost particles"
    assert stats["arena_used"] > 0, "fixture did not actually overflow a brick"

    tight = state.SlotState.build(x, v, _t9(), BRICKS, brick_slack=0.0, arena_frac=0.0)
    with pytest.raises(ValueError, match="does not clamp or drop"):
        state.drift_and_migrate(tight, 0.15)


def test_arena_residents_are_pulled_back_into_their_brick_run():
    """An arena particle still BELONGS to its brick. If the pass did not pull it
    back in, the arena would fill monotonically -- and D-v2-19 clause 4 measured
    the cost of forgetting an arena resident at 98.4% of the force in its brick.
    """
    x = _positions(29)
    v = (L_BOX * 0.25 - x) * 2.0
    st = state.SlotState.build(x, v, _t9(), BRICKS, brick_slack=0.0, arena_frac=0.30)
    state.drift_and_migrate(st, 0.15)
    crowded = st.arena_used
    assert crowded > 0
    # reverse the velocities and let it expand again: the arena must drain
    st.w[:] = -st.w
    state.drift_and_migrate(st, 0.15)
    assert st.check() is True
    assert st.n_live == x.shape[0]
    assert st.arena_used < crowded, f"arena never drains ({crowded} -> {st.arena_used})"


# ============================================ grouping rows by destination brick
# M-v2-6. `_insert_slab` used to select each brick's rows with a boolean mask
# inside the brick loop, so every brick scanned every row of its slab: N x nb^2
# comparisons per step, N^(5/3) rather than N. Measured on deneb (job 456,
# particles fixed at 256^3, staging depth pinned at 1) insert time went
# 3.793 -> 10.756 -> 39.626 s for bricks per side 8 -> 16 -> 32, against 38.6 s
# predicted for the last rung BEFORE it ran. At C-gh that term alone is ~87 h
# per step. `_group_by_brick` replaces it with one grouping pass.


def test_grouping_by_brick_reproduces_the_per_brick_mask_exactly():
    """The identity that makes the replacement bitwise neutral, not merely
    equivalent: same rows AND same order within every brick.

    Order is load-bearing downstream -- the destination scale is a max over the
    brick's rows and the encode that follows is order-dependent through it -- so
    membership alone would not be enough. `np.array_equal` on the index arrays
    asserts both at once where a set comparison would pass on a permutation.
    """
    rng = np.random.default_rng(5)
    lo_b, hi_b = 12, 28
    # deliberately spans OUTSIDE the slab: the immigrant buffer carries every
    # emigrant from every reaching slab and only some are bound for this one.
    # The mask discarded those by never matching; this must drop them the same
    # way rather than raising or mis-binning them.
    brick_of_row = rng.integers(lo_b - 6, hi_b + 6, size=2000)
    order, off = state._group_by_brick(brick_of_row, lo_b, hi_b)

    n_in = int(((brick_of_row >= lo_b) & (brick_of_row < hi_b)).sum())
    assert off[0] == 0 and off[-1] == n_in
    assert len(order) == n_in
    assert n_in < len(brick_of_row), "fixture has no out-of-slab rows to drop"
    for j, b in enumerate(range(lo_b, hi_b)):
        want = np.flatnonzero(brick_of_row == b)
        got = order[off[j] : off[j + 1]]
        assert np.array_equal(got, want), f"brick {b}: membership or order differs"
    assert (np.diff(off) > 0).any(), "every brick came out empty; the fixture is vacuous"


def test_grouping_by_brick_handles_the_empty_and_all_outside_cases():
    """Both reachable in a real run: a slab with no keepers, and an immigrant
    buffer none of whose rows are bound for this slab."""
    order, off = state._group_by_brick(np.empty(0, np.int64), 4, 9)
    assert len(order) == 0 and np.array_equal(off, np.zeros(6, dtype=np.int64))

    order, off = state._group_by_brick(np.array([0, 1, 2, 40, 41]), 4, 9)
    assert len(order) == 0 and off[-1] == 0


def test_grouping_by_brick_takes_the_radix_path_where_the_range_allows():
    """The key is cast to uint16 when the brick count fits, because numpy's
    stable sort is a RADIX sort only for 1- and 2-byte integer types -- the same
    fact that bought `migrate` 5.3x on its own sort. Asserted through behaviour
    at both sides of the boundary rather than by reading the cast: the result
    must be identical either way, which is what says the optimization is safe."""
    rng = np.random.default_rng(11)
    lo_b = 0
    for hi_b in (1 << 10, (1 << 16) + 4):  # under and over the uint16 ceiling
        brick_of_row = rng.integers(lo_b, hi_b, size=500)
        order, off = state._group_by_brick(brick_of_row, lo_b, hi_b)
        for j in np.unique(brick_of_row):
            want = np.flatnonzero(brick_of_row == j)
            assert np.array_equal(order[off[j] : off[j + 1]], want)


# ================================================== repack, in place (M-v2-6)
# The out-of-place form allocates 11.1 B/row (measured, flat over 64x in N) and
# ~115 GB at C-gh -- the largest single term left after the velocity array went.
# D-v2-19 clause 3 named `BrickPackedLayout.repack` as the in-place form to port
# on a reported scratch of 0.13-0.52 MB "independent of N", but that counts only
# its chunk buffers: measured it is 39.4 B/row, so the port would have been a
# 3.55x regression. The clause's REASONING (a monotone rearrangement, not a
# sort) is what the replacement uses.


def _repack_pair(seed, nb=4, arena_frac=0.25, drifts=(), brick_slack=0.10):
    """Two identical states, one repacked each way. Optional drifts first, so
    the arena is NON-EMPTY -- the arena fold-in is the part with no analogue in
    the layout module and the part most likely to be got wrong.

    `brick_slack=0.0` is how the arena is forced: spare then floors at ONE slot
    per occupied brick, so any brick that gains two particles spills. Reaching
    the arena by drifting harder does not work, because a larger drift moves
    particles between bricks without concentrating them."""
    x = _positions(seed)
    v = np.random.default_rng(seed + 1).normal(scale=0.05, size=(N_PART**3, 3))
    kw = dict(with_ids=True, arena_frac=arena_frac, brick_slack=brick_slack)
    a = state.SlotState.build(x, v, _t9(), nb, **kw)
    b = state.SlotState.build(x, v, _t9(), nb, **kw)
    for c in drifts:
        state.drift_and_migrate(a, c)
        state.drift_and_migrate(b, c)
    return a, b


def test_the_in_place_repack_is_elementwise_the_out_of_place_one():
    """The gate the plan asks for, and it compares against the REFERENCE
    implementation kept in the module rather than against a property.

    A property ("every particle is in its bucket") can hold for two different
    layouts; only elementwise equality says the rearrangement is the same one.
    `ids` is included because it is what makes the comparison per PARTICLE --
    without it, two states could agree on payload and still have permuted rows
    within a bucket.
    """
    a, b = _repack_pair(41)
    ra = a.repack()
    rb = b._repack_reference()

    assert np.array_equal(a.off, b.off), "position payload differs"
    assert np.array_equal(a.w, b.w), "velocity payload differs"
    assert np.array_equal(a.ids, b.ids), "ids differ: rows permuted within a bucket"
    assert np.array_equal(a.brick_start, b.brick_start)
    assert np.array_equal(a.occupancy, b.occupancy)
    assert a.occupancy.dtype == b.occupancy.dtype
    assert a.arena_base == b.arena_base
    assert np.array_equal(a.arena_bucket, b.arena_bucket)
    assert ra["slots_used"] == rb["slots_used"]
    assert a.check() is True


def test_the_in_place_repack_matches_with_a_NON_EMPTY_arena():
    """The arena fold-in is the part `layout.py` has no analogue for. A repack
    from a freshly built state never exercises it, so this drifts first and
    asserts the arena was actually populated -- otherwise the test above and
    this one are the same test."""
    a, b = _repack_pair(43, drifts=(0.4, 0.4), brick_slack=0.0)
    assert a.arena_used > 0, "fixture never spilled to the arena; the case is untested"
    a.repack()
    b._repack_reference()
    assert np.array_equal(a.off, b.off)
    assert np.array_equal(a.w, b.w)
    assert np.array_equal(a.ids, b.ids)
    assert np.array_equal(a.occupancy, b.occupancy)
    assert np.array_equal(a.brick_start, b.brick_start)
    assert a.arena_used == 0, "the fold-in must leave the arena empty"
    assert a.check() is True


def test_the_in_place_repack_conserves_particles_over_repeated_steps():
    """D-007 forbids dropping, and this container has lost exactly one particle
    before. Repack repeatedly, interleaved with drifts, and count."""
    a, _ = _repack_pair(45, drifts=())
    n0 = a.n_live
    for c in (0.3, 0.3, 0.3):
        state.drift_and_migrate(a, c)
        a.repack()
        assert a.check() is True
        assert a.n_live == n0, f"lost {n0 - a.n_live} particles"


def test_the_in_place_repack_reports_scratch_that_includes_everything():
    """The reference's `scratch_bytes` omitted three O(N) arrays and so read as
    a constant while the real cost was linear. This one must report a figure
    that actually bounds what it allocated, so the same mistake cannot be made
    twice on the same term.
    """
    a, _ = _repack_pair(47, drifts=(0.4,), brick_slack=0.0)
    n_rows = a.off.shape[0]
    r = a.repack()
    assert "scratch_bytes" in r
    # a brick plus the lifted arena, not a row-count-sized array
    assert r["scratch_bytes"] < 0.5 * n_rows * 9, (
        "scratch is a large fraction of the payload, so this is not the in-place form"
    )
    assert r["scratch_bytes"] > 0, "a scratch figure of zero is not credible"


# ------------------------------------------- the arena caches are pure caches


def test_the_arena_caches_are_pure_and_match_a_rebuild_under_migration():
    """The surgical index update + lazy free-list vs rebuild-on-every-read.

    Two identical arena-heavy states run the same drift chain. The oracle arm
    (`st_b`) gets an instance-level shim that INVALIDATES before every
    `arena_slots_of_brick` read, so every index it ever consumes is a fresh
    O(n_arena) rebuild and every `_to_arena` claim rescans the free list --
    exactly the pre-5g semantics, mid-migrate included. The fast arm (`st_a`)
    runs the maintained caches. An impure cache diverges the arena LAYOUT
    (the free list decides which slot a spilled particle lands in), so
    elementwise equality of every stored array is the whole claim.

    Built at brick_slack=0.0 because a 0.10-slack uniform state parks nothing
    in the arena and this would be a gate that cannot fail; the assert on
    `arena_used` enforces that the test is actually exercising the machinery.
    """
    x, v, st_a = _built(seed=3, brick_slack=0.0, arena_frac=0.30, with_ids=True)
    _, _, st_b = _built(seed=3, brick_slack=0.0, arena_frac=0.30, with_ids=True)

    orig = state.SlotState.arena_slots_of_brick

    def rebuild_every_read(brick_flat):
        st_b._invalidate_arena_index()
        return orig(st_b, brick_flat)

    st_b.arena_slots_of_brick = rebuild_every_read

    extent = L_BOX / BRICKS
    c = 0.6 * extent / (float(np.max(st_a.vel_scale)) * state.INT16_MAX)
    fields = ("off", "w", "occupancy", "brick_start", "vel_scale", "arena_bucket", "ids")
    for k in range(4):
        state.drift_and_migrate(st_a, c)
        state.drift_and_migrate(st_b, c)
        for f in fields:
            assert np.array_equal(getattr(st_a, f), getattr(st_b, f)), (k, f)
        assert st_a.arena_used > 0, "arena never exercised; the test is vacuous"
        # the maintained index must BE a fresh rebuild's content
        maintained = st_a._arena_by_brick
        if maintained is not None:
            fresh = dict(maintained)  # keep a handle; _build replaces the cache
            rebuilt = st_a._build_arena_index()
            assert set(fresh) == set(rebuilt)
            for b in rebuilt:
                assert np.array_equal(fresh[b], rebuilt[b]), b
        # the maintained free list must BE the scan's answer
        if st_a._arena_free is not None:
            assert np.array_equal(
                st_a._arena_free, np.nonzero(st_a.arena_bucket < 0)[0]
            )
        st_a.check()
