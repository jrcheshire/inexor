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
    assert np.max(np.abs(np.sort(got_v.ravel()) - np.sort(v.ravel()))) <= st.vel_scale


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
    this is a shape check on the accounting, not a reproduction of the number."""
    _, _, st = _built(9)
    bpp = st.bytes_per_particle()
    assert bpp["payload"] == 9.0
    assert bpp["ids"] == 0.0
    assert bpp["total"] == pytest.approx(
        sum(bpp[k] for k in ("payload", "bucket_index", "brick_start", "slack", "arena", "ids"))
    )
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
