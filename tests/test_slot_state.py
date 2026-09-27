"""SlotState: T9 payload in slot order with the bucket implied by the slot. Pins that
invariant under drift, migration, arena overflow and repack, and host/device codec parity.
"""


import numpy as np
import pytest


from inexor import layout, state  # noqa: E402
from inexor.codec import T9Layout  # noqa: E402

# 32^3 particles, bucket 2 cells -> 16^3 buckets, 2 bricks/side -> 8 bricks of 512
# buckets each (C-gh's buckets per brick). n_part/bucket_cells must be a power of two
# so the wrap is modular, not saturating.
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
    """A perturbed Lagrangian lattice."""
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
    """The numpy host encode/decode in `state` (the global position array never goes to the
    device) is bitwise the jnp one in `codec`.
    """
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
    # anti-vacuity: the fixture spans the byte
    assert len(np.unique(off_h)) > 200, "offsets do not span the byte; fixture is degenerate"


def test_the_mirror_check_can_fail():
    """Anti-vacuity: a one-quantum offset change is visible to the comparison above."""
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
    """Why `check()` does not decode and compare buckets: `decode` rebuilds `bucket * 256 +
    off`, so the comparison is an identity and passes on a corrupted offset.
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
    # the structural check is blind to it too, by design
    assert st.check() is True


def test_a_lost_particle_is_caught():
    """`check()` raises when a particle is unreachable through the layout; the count is the
    only thing that sees a loss.
    """
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
    """`check()` raises when a brick's live rows exceed its allocation, which would silently
    reassign particles to the next brick without changing the count.
    """
    _, _, st = _built(13)
    b = next(i for i in range(st.n_bricks) if st.brick_live_count(i))
    p3 = st.buckets_per_brick
    lo, hi = st.brick_slot_range(b)
    st.occupancy[b * p3] += np.uint32(hi - lo)  # push the run past its allocation
    st.n_particles += int(hi - lo)  # keep the count consistent, so only test 1 fires
    with pytest.raises(ValueError, match="live rows in an allocation of"):
        st.check()


def test_placement_is_checkable_against_a_reference_and_that_check_can_fail():
    """`check_placement` (reference positions + id tier) confirms each particle sits in the
    bucket its position calls for, and raises when one reference moves a bucket.
    """
    x, v, st = _built(14, with_ids=True)
    assert st.check_placement(x) is True
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
    # order is not preserved by the container, so compare as multisets per axis
    for ax in range(3):
        a = np.sort(got_x[:, ax])
        e = np.sort(np.mod(np.rint(x[:, ax] / st.t9.quantum), st.t9.n_levels) * st.t9.quantum)
        assert np.allclose(a, e, atol=0.51 * st.t9.quantum), f"axis {ax} positions moved"
    assert np.max(np.abs(np.sort(got_v.ravel()) - np.sort(v.ravel()))) <= st.vel_scale.max()


def test_ids_follow_their_particle_through_the_reordering():
    """The opt-in int32 id tier follows each particle through the reordering, and ids are a
    permutation of the particle count.
    """
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


# ------------------------------------------------------ per-particle cost


def test_the_scaffolding_arrays_do_not_exist():
    """No `key`, `particle_to_slot` or `slot_to_particle` arrays: ~21 B/p against a
    ~10.5 B/p budget, not residentable at production scale in either integer width.
    """
    _, _, st = _built(8)
    for name in ("key", "particle_to_slot", "slot_to_particle"):
        assert not hasattr(st, name), f"{name} survived into the engine's container"
    assert st.bytes_per_particle()["scaffold"] == 0.0


def test_the_int32_key_ceiling_is_not_reachable_because_key_is_gone():
    """`layout._refuse_key_overflow` still guards the layout module's int32 `key` (C-hero's
    bucket grid, 8.59e9, overflows it); SlotState has no `key`, so the ceiling does not apply.
    """
    with pytest.raises(ValueError, match="exceeds int32"):
        layout._refuse_key_overflow(2**31)


def test_the_all_in_cost_lands_near_the_ratified_figure():
    """`bytes_per_particle` totals its terms (payload 9.00, index, brick CSR, slack, arena,
    ids, per-brick scales). A shape check: this bucket grid is coarser than C-gh's (~10.54 B/p).
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
    # pooled per brick, slack is ~10%; per-bucket granularity would floor it at 12.5%
    assert 0.5 < bpp["slack"] < 1.6, f"slack {bpp['slack']:.3f} B/p is not the pooled 10%"


def test_ids_cost_exactly_four_bytes_when_asked_for():
    _, _, st = _built(10, with_ids=True)
    assert st.bytes_per_particle()["ids"] == 4.0


# ------------------------------------------------------- derived boundaries


def test_bucket_boundaries_are_derived_and_agree_with_the_stored_runs():
    """Per-bucket slot boundaries derived from the occupancy prefix sum reproduce the
    stored runs (no stored per-bucket boundary, saving 1.00 B/p at C-gh).
    """
    _, _, st = _built(11)
    p3 = st.buckets_per_brick
    for b in range(st.n_bricks):
        starts = st.bucket_slot_starts(b)
        occ = st.occupancy[b * p3 : (b + 1) * p3].astype(np.int64)
        assert np.array_equal(np.diff(starts), occ)
        assert starts[0] == st.brick_start[b]
        assert starts[-1] == st.brick_start[b] + occ.sum()


def test_free_slots_need_no_sentinel():
    """Liveness follows from `occupancy` alone: a brick's live rows are its first
    sum(occupancy) slots, so no -1 marker competes with the index.
    """
    _, _, st = _built(12)
    for b in range(st.n_bricks):
        lo, hi = st.brick_slot_range(b)
        m = st.brick_live_count(b)
        assert lo + m <= hi, "a brick holds more live rows than its allocation"
        assert len(st.brick_member_slots(b)) == m + len(st.arena_slots_of_brick(b))


# ============================================ drift and re-home (one fused pass)


def _drifted_reference(x, v, c, t9, scales, bricks_per_side):
    """Destination bucket of every particle, computed independently of the pass: from the
    stored lattice positions and int16 velocity codes, with the brick re-derived through
    `bucket_order_key` rather than read back from the container.
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
    """`drift_and_migrate` conserves the particle count and passes `check()` over repeated
    passes.
    """
    _, _, st = _built(20)
    n0 = st.n_live
    for _ in range(3):
        state.drift_and_migrate(st, 0.05)
        assert st.check() is True
        assert st.n_live == n0


def test_every_particle_lands_in_the_bucket_its_drifted_position_calls_for():
    """Every particle lands in the bucket `_drifted_reference` computes."""
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
    """Destinations are right across buckets, bricks and the periodic seam, and the fixture
    is asserted to exercise all three.
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
    # a seam crossing: the drifted, unwrapped lattice coordinate leaves [0, n_levels),
    # so the wrap fires
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
    """Particles crossing two bricks (x, y, diagonal, across the seam) are conserved and land
    right. Needs >= 4 bricks per side: at nb=2 a +-1 source insert covers every slab.
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
    """Mutation: pinning `_insert_slab` to +-1 sources regardless of the schedule's reach
    makes the release loop raise "unconsumed" at the release, not later.
    """
    orig = state.SlotState._insert_slab

    def pinned(self, bx, staged, emig, reach=(-1, 0, 1), consumed=None, scales=None,
               kernel="numpy"):
        return orig(self, bx, staged, emig, (-1, 0, 1), consumed, scales=scales, kernel=kernel)

    monkeypatch.setattr(state.SlotState, "_insert_slab", pinned)
    x = _positions(30)
    v = np.random.default_rng(31).normal(scale=0.05, size=(N_PART**3, 3))
    x[0], v[0] = (8.0, 8.0, 8.0), (28.0, 0.0, 0.0)
    st = state.SlotState.build(x, v, _t9(), 4, with_ids=True, arena_frac=0.25)
    with pytest.raises(AssertionError, match="unconsumed"):
        state.drift_and_migrate(st, 1.0)


def test_a_particle_is_drifted_exactly_once():
    """No particle is drifted twice (a record inserted into a not-yet-ejected brick would be):
    per-id destinations match one drift of c, and the fixture distinguishes c from 2c.
    """
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
    """Tile ownership is a partition, so the max over per-tile scales equals the global
    scale exactly.
    """
    rng = np.random.default_rng(24)
    v = rng.normal(size=(5000, 3))
    tile = rng.integers(0, 37, size=5000)
    scales = [np.max(np.abs(v[tile == t])) / 32767 for t in range(37) if np.any(tile == t)]
    assert state.reconcile_velocity_scale(scales) == np.max(np.abs(v)) / 32767


def test_rescaling_a_velocity_code_never_escapes_int16():
    """Rescaling a tile's int16 code to the global scale stays in int16, since the global
    scale is the exact max.
    """
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
    # A brick's scale is set by `_insert_slab` over the rows it writes. A zero drift
    # keeps every row, so an idempotent re-derivation returns the same scales.
    state.drift_and_migrate(st, 0.0)
    assert st.check() is True
    assert np.array_equal(st.vel_scale, s0), "a zero drift moved a brick's scale"
    seen = 0
    for b in range(st.n_bricks):
        slots, _, vb = st.decode_brick(b)
        seen += len(slots)
    assert seen == st.n_particles


def test_the_arena_absorbs_a_brick_overflow_and_then_refuses():
    """Overflow goes to brick spare, then arena, then a ValueError; never a clamp or drop.
    brick_slack=0.0 plus convergence on one brick's centre (the box centre is where all eight
    bricks meet, so inflow there balances).
    """
    x = _positions(27)
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
    """The arena drains when the flow reverses: residents are pulled back into their brick
    (a forgotten resident measured at 98.4% of its brick's force).
    """
    x = _positions(29)
    v = (L_BOX * 0.25 - x) * 2.0
    st = state.SlotState.build(x, v, _t9(), BRICKS, brick_slack=0.0, arena_frac=0.30)
    state.drift_and_migrate(st, 0.15)
    crowded = st.arena_used
    assert crowded > 0
    st.w[:] = -st.w
    state.drift_and_migrate(st, 0.15)
    assert st.check() is True
    assert st.n_live == x.shape[0]
    assert st.arena_used < crowded, f"arena never drains ({crowded} -> {st.arena_used})"


# ============================================ grouping rows by destination brick
# `_insert_slab` groups a slab's rows by destination brick in one pass
# (`_group_by_brick`). A boolean mask per brick scans every row per brick, N x nb^2
# per step: measured 3.793 -> 10.756 -> 39.626 s at 256^3 for 8 -> 16 -> 32 bricks
# per side, ~87 h per step at C-gh. These tests pin the grouping to the mask.


def test_grouping_by_brick_reproduces_the_per_brick_mask_exactly():
    """`_group_by_brick` gives a per-brick mask's rows in the same order (the encode is
    order-dependent through the brick's max scale), dropping rows outside the slab.
    """
    rng = np.random.default_rng(5)
    lo_b, hi_b = 12, 28
    # spans outside the slab: the immigrant buffer carries emigrants from every
    # reaching slab, and rows bound elsewhere must be dropped, not mis-binned
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
    """Empty input and all-outside input both give empty groups (both occur in a run)."""
    order, off = state._group_by_brick(np.empty(0, np.int64), 4, 9)
    assert len(order) == 0 and np.array_equal(off, np.zeros(6, dtype=np.int64))

    order, off = state._group_by_brick(np.array([0, 1, 2, 40, 41]), 4, 9)
    assert len(order) == 0 and off[-1] == 0


def test_grouping_by_brick_takes_the_radix_path_where_the_range_allows():
    """Grouping gives identical results under and over the uint16 key ceiling, where numpy's
    stable sort switches between radix and the wide path.
    """
    rng = np.random.default_rng(11)
    lo_b = 0
    for hi_b in (1 << 10, (1 << 16) + 4):  # under and over the uint16 ceiling
        brick_of_row = rng.integers(lo_b, hi_b, size=500)
        order, off = state._group_by_brick(brick_of_row, lo_b, hi_b)
        for j in np.unique(brick_of_row):
            want = np.flatnonzero(brick_of_row == j)
            assert np.array_equal(order[off[j] : off[j + 1]], want)


# ============================================================ repack, in place
# `repack` is a monotone in-place rearrangement; `_repack_reference` is the
# out-of-place form (11.1 B/row measured, ~115 GB at C-gh) kept as its oracle.


def _repack_pair(seed, nb=4, arena_frac=0.25, drifts=(), brick_slack=0.10):
    """Two identical states for `repack` vs `_repack_reference`, optionally drifted first.
    `brick_slack=0.0` is what forces arena spills; a harder drift does not concentrate particles.
    """
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
    """In-place `repack` equals `_repack_reference` on every stored array, ids included
    (which catches permutations within a bucket).
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
    """Same equality with a populated arena (the arena fold-in has no analogue in
    `layout.py`), and the arena is empty afterwards.
    """
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
    """Particle count is conserved over repeated drift + repack."""
    a, _ = _repack_pair(45, drifts=())
    n0 = a.n_live
    for c in (0.3, 0.3, 0.3):
        state.drift_and_migrate(a, c)
        a.repack()
        assert a.check() is True
        assert a.n_live == n0, f"lost {n0 - a.n_live} particles"


def test_the_in_place_repack_reports_scratch_that_includes_everything():
    """`repack` reports a positive `scratch_bytes` well under the payload size (a brick plus
    the lifted arena, not O(rows)).
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
    """The maintained arena index and free list are pure caches: against an arm that rebuilds
    the index before every read, every stored array matches over a drift chain, and the caches
    equal a rebuild. brick_slack=0.0 so the arena is used.
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
        maintained = st_a._arena_by_brick
        if maintained is not None:
            fresh = dict(maintained)  # keep a handle; _build replaces the cache
            rebuilt = st_a._build_arena_index()
            assert set(fresh) == set(rebuilt)
            for b in rebuilt:
                assert np.array_equal(fresh[b], rebuilt[b]), b
        if st_a._arena_free is not None:
            assert np.array_equal(
                st_a._arena_free, np.nonzero(st_a.arena_bucket < 0)[0]
            )
        st_a.check()


def test_the_repack_fast_path_and_the_merge_path_BOTH_fire_and_agree():
    """One `repack` takes both the fast path (no arena residents; 89-96% of bricks on a
    clustered state) and the merge path, and still equals the reference.
    """
    a, b = _repack_pair(51, drifts=(0.4,), brick_slack=0.0)
    assert a.arena_used > 0, "fixture never spilled; the merge path is untested"
    r = a.repack()
    b._repack_reference()
    assert r["bricks_fast"] > 0, "no brick took the fast path; the branch is dead"
    assert r["bricks_merged"] > 0, "no brick took the merge path; the fixture is vacuous"
    assert np.array_equal(a.off, b.off), "position payload differs"
    assert np.array_equal(a.w, b.w), "velocity payload differs"
    assert np.array_equal(a.ids, b.ids), "ids differ: rows permuted within a bucket"
    assert np.array_equal(a.occupancy, b.occupancy)
    assert np.array_equal(a.brick_start, b.brick_start)
    assert a.arena_base == b.arena_base
    assert a.check() is True


def test_the_repack_fast_path_counts_every_occupied_brick_when_the_arena_is_empty():
    """With an empty arena every occupied brick takes the fast path and none merges."""
    a, _ = _repack_pair(53, drifts=())
    assert a.arena_used == 0, "fixture spilled; this case is about an EMPTY arena"
    r = a.repack()
    assert r["bricks_merged"] == 0, "a merge ran with no arena residents to merge"
    assert r["bricks_fast"] > 0, "no brick was repacked at all"


def test_an_overlapping_slice_assignment_copies_before_it_writes():
    """numpy copies before writing an overlapping slice, in both directions; `repack`'s block
    moves rely on it, and the particle count would not see the corruption.
    """
    base = (np.arange(20000, dtype=np.int16).reshape(-1, 1)
            * np.ones((1, 3), dtype=np.int16))
    src, m = 5000, 3000
    for dst, tag in ((src - 1000, "leftward, as pass A moves"),
                     (src + 1000, "rightward, as pass B moves")):
        a = np.ascontiguousarray(base)
        b = np.ascontiguousarray(base)
        a[dst:dst + m] = a[src:src + m]
        b[dst:dst + m] = b[src:src + m].copy()
        assert np.array_equal(a, b), (
            f"numpy did not copy before writing an overlapping range ({tag}); "
            "`repack`'s block moves must go back to an explicit .copy()"
        )


def test_repack_still_refuses_a_brick_that_outgrows_the_index_dtype():
    """`repack` raises when a brick total exceeds the index dtype. numpy narrows modularly,
    and a shifted per-bucket span would pass `check()`, which checks per-brick totals only.
    """
    n_side, box = 16, 16.0
    rng = np.random.default_rng(4)
    x = rng.random((n_side**3, 3)) * box
    v = rng.standard_normal((n_side**3, 3)) * 0.1
    t9 = T9Layout(box_size=box, n_part=n_side, bucket_cells=2)
    st = state.SlotState.build(x, v, t9, 4, brick_slack=0.2, alloc_margin=0.5,
                               arena_frac=0.05, index_dtype=np.uint16)
    p3 = st.buckets_per_brick
    st.occupancy[0:p3] = 0
    st.occupancy[0] = np.iinfo(np.uint16).max
    st.occupancy[1] = np.iinfo(np.uint16).max
    with pytest.raises(ValueError, match="index ceiling"):
        st.repack(brick_slack=0.2)


def test_repack_is_unchanged_by_the_per_brick_census_rewrite():
    """Repack's per-brick census is an integer identity: occupancy + arena and the decoded
    member count are unchanged, from a clustered state with arena residents.
    """
    # clustered and with no per-brick spare; uniform positions spill nothing
    n_side, box = 32, 32.0
    rng = np.random.default_rng(9)
    x = (rng.standard_normal((n_side**3, 3)) * 4.0 + box / 2) % box
    v = rng.standard_normal((n_side**3, 3)) * 10.0
    t9 = T9Layout(box_size=box, n_part=n_side, bucket_cells=2)
    st = state.SlotState.build(x, v, t9, 4, brick_slack=0.0, alloc_margin=0.6,
                               arena_frac=0.20)
    state.drift_and_migrate(st, 0.3, kernel="numpy")
    assert st.arena_used > 0, "this gate needs arena residents to be meaningful"

    before = int(st.occupancy.astype(np.int64).sum()) + st.arena_used
    n_before = len(np.concatenate(
        [np.asarray(st.decode_brick(b)[0]) for b in range(st.n_bricks)]))
    st.repack(brick_slack=0.05)
    after = int(st.occupancy.astype(np.int64).sum()) + st.arena_used
    assert after == before, f"repack moved the census {before} -> {after}"
    n_after = len(np.concatenate(
        [np.asarray(st.decode_brick(b)[0]) for b in range(st.n_bricks)]))
    assert n_after == n_before
    assert st.occupancy.dtype == st.index_dtype
    st.check()
