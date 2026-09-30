"""Node-local `SlotState` (`slabs` set): one rank's brick x-slabs of a whole-box state.

`restrict_to_slabs` cuts a whole state into the node-local state a rank would hold. Every
per-brick read of the cut must equal the whole state's on the owned bricks, and refuse
outside them; the repack must produce the whole repack's owned block byte for byte; passes
with no cross-rank path refuse a node-local state.
"""

import math

import numpy as np
import pytest

from inexor import engine, state
from inexor.codec import T9Layout
from inexor.ooc_fft import partition_units

N_PART, BOX, NB = 32, 32.0, 8


def evolved_state(seed=3, steps=3, arena_frac=0.05, slack=0.02):
    """A whole state after a few migrations: spares in use and a populated arena."""
    rng = np.random.default_rng(seed)
    g = (np.arange(N_PART) + 0.5) * (BOX / N_PART)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    x = np.mod(q + rng.normal(scale=0.25 * BOX / N_PART, size=q.shape), BOX)
    v = rng.normal(scale=0.5, size=x.shape)
    st = state.SlotState.build(x, v, T9Layout(BOX, N_PART, 2), NB, brick_slack=slack,
                               arena_frac=arena_frac)
    for _ in range(steps):
        state.drift_and_migrate(st, 0.5)
    return st


def rank_slabs(nb, n_ranks):
    """Each rank's brick x-slabs [lo, hi) under an even split."""
    return [tuple(r) for r in partition_units(nb, n_ranks, 1)]


def restrict_to_slabs(st, slabs, alloc_margin=0.10):
    """The node-local state owning brick x-slabs `slabs`, cut from the whole state `st`.

    Owned bricks keep their allocation sizes (rows renumbered from 0); arena residents of
    owned buckets keep their relative arena-row order, the order the writer reads.
    """
    assert st.is_whole
    lo_s, hi_s = int(slabs[0]), int(slabs[1])
    nb2, p3 = st.bricks_per_side ** 2, st.buckets_per_brick
    blo, bhi = lo_s * nb2, hi_s * nb2
    r0, r1 = int(st.brick_start[blo]), int(st.brick_start[bhi])

    brick_start = np.zeros_like(st.brick_start)
    brick_start[blo:bhi + 1] = st.brick_start[blo:bhi + 1] - r0
    brick_start[bhi + 1:] = r1 - r0

    a_rows = np.nonzero((st.arena_bucket >= blo * p3) & (st.arena_bucket < bhi * p3))[0]
    n_local = st.brick_member_counts()[blo:bhi].sum()
    n_arena = max(len(a_rows), math.ceil(int(n_local) * st.arena_frac))
    # the runs, then alloc_margin headroom as `_alloc_geometry` gives it, then the arena
    n_alloc = math.ceil((r1 - r0) * (1.0 + alloc_margin))
    off = np.zeros((n_alloc + n_arena, 3), dtype=st.off.dtype)
    w = np.zeros((n_alloc + n_arena, 3), dtype=st.w.dtype)
    off[:r1 - r0] = st.off[r0:r1]
    w[:r1 - r0] = st.w[r0:r1]
    off[n_alloc:n_alloc + len(a_rows)] = st.off[st.arena_base + a_rows]
    w[n_alloc:n_alloc + len(a_rows)] = st.w[st.arena_base + a_rows]
    arena_bucket = np.full(n_arena, -1, dtype=np.int64)
    arena_bucket[:len(a_rows)] = st.arena_bucket[a_rows]
    vel_scale = np.ones_like(st.vel_scale)
    vel_scale[blo:bhi] = st.vel_scale[blo:bhi]
    return state.SlotState(
        t9=st.t9, bricks_per_side=st.bricks_per_side, brick_start=brick_start,
        occupancy=st.occupancy[blo * p3:bhi * p3].copy(), off=off, w=w, vel_scale=vel_scale,
        arena_base=n_alloc, arena_bucket=arena_bucket, n_particles=int(n_local),
        slabs=(lo_s, hi_s), arena_frac=st.arena_frac,
    )


@pytest.fixture(scope="module")
def whole():
    st = evolved_state()
    assert st.arena_used > 0, "vacuous: no arena residents"
    return st


@pytest.mark.parametrize("n_ranks", [1, 2, 4, 8])
def test_a_cut_reads_the_whole_states_bricks(whole, n_ranks):
    counts = whole.brick_member_counts()
    total = arena = 0
    for slabs in rank_slabs(NB, n_ranks):
        part = restrict_to_slabs(whole, slabs)
        assert part.is_whole == (n_ranks == 1)
        part.check()
        blo, bhi = part.owned_bricks
        got = part.brick_member_counts()
        np.testing.assert_array_equal(got[blo:bhi], counts[blo:bhi])
        assert not got[:blo].any() and not got[bhi:].any()
        for b in range(blo, bhi):
            _, x, v = part.decode_brick(b)
            _, xw, vw = whole.decode_brick(b)
            np.testing.assert_array_equal(x, xw)
            np.testing.assert_array_equal(v, vw)
            np.testing.assert_array_equal(part.bucket_slot_starts(b) - part.brick_start[b],
                                          whole.bucket_slot_starts(b) - whole.brick_start[b])
        total += part.n_live
        arena += part.arena_used
        assert part.n_live == part.n_particles
    assert total == whole.n_live and arena == whole.arena_used


def test_a_brick_outside_the_owned_range_refuses(whole):
    lo, hi = rank_slabs(NB, 4)[1]
    part = restrict_to_slabs(whole, (lo, hi))
    blo, bhi = part.owned_bricks
    for b in (blo - 1, bhi, 0, whole.n_bricks - 1):
        with pytest.raises(IndexError, match="outside this state's owned bricks"):
            part.brick_live_count(b)
    with pytest.raises(IndexError):
        part._occ(blo, bhi + 1)
    part.brick_live_count(blo)
    part.brick_live_count(bhi - 1)


@pytest.mark.parametrize("n_ranks", [2, 4])
@pytest.mark.parametrize("which", ["repack", "_repack_reference"])
def test_a_cuts_repack_is_the_whole_repacks_owned_block(whole, n_ranks, which):
    ref = restrict_to_slabs(whole, (0, NB))
    getattr(ref, which)(brick_slack=0.10)
    for slabs in rank_slabs(NB, n_ranks):
        part = restrict_to_slabs(whole, slabs)
        getattr(part, which)(brick_slack=0.10)
        part.check()
        blo, bhi = part.owned_bricks
        p3 = part.buckets_per_brick
        r0, r1 = int(ref.brick_start[blo]), int(ref.brick_start[bhi])
        np.testing.assert_array_equal(part.brick_start[blo:bhi + 1],
                                      ref.brick_start[blo:bhi + 1] - r0)
        np.testing.assert_array_equal(part.occupancy, ref.occupancy[blo * p3:bhi * p3])
        n_alloc = part.arena_base
        assert n_alloc == r1 - r0
        np.testing.assert_array_equal(part.off[:n_alloc], ref.off[r0:r1])
        np.testing.assert_array_equal(part.w[:n_alloc], ref.w[r0:r1])
        assert part.arena_used == 0 and not part.off[n_alloc:].any()


def test_check_refuses_a_malformed_cut(whole):
    slabs = rank_slabs(NB, 4)[1]
    bad = restrict_to_slabs(whole, slabs)
    bad.occupancy = bad.occupancy[:-1]
    with pytest.raises(ValueError, match="occupancy holds"):
        bad.check()

    bad = restrict_to_slabs(whole, slabs)
    bad.brick_start[1:] += 1       # brick 0 is not owned by rank 1
    with pytest.raises(ValueError, match="non-owned brick must be an empty run"):
        bad.check()

    bad = restrict_to_slabs(whole, slabs)
    a = int(np.nonzero(bad.arena_bucket < 0)[0][0])
    bad.arena_bucket[a] = 0        # a bucket of slab 0, owned by rank 0
    bad.n_particles += 1
    with pytest.raises(ValueError, match="outside the owned buckets"):
        bad.check()


def test_passes_without_a_cross_rank_path_refuse_a_cut(whole):
    part = restrict_to_slabs(whole, rank_slabs(NB, 2)[0])
    match = "needs a whole-box state"
    with pytest.raises(NotImplementedError, match=match):
        state.drift_and_migrate(part, 0.5)
    with pytest.raises(NotImplementedError, match=match):
        state.drift_and_migrate_pooled(part, 0.5, pool=None)
    with pytest.raises(NotImplementedError, match=match):
        engine.step(part, None, (0.0, 0.0), 0.0)
    with pytest.raises(NotImplementedError, match=match):
        engine.run(part, None, np.zeros((1, 3)))
    from inexor.executor import TilePool

    with pytest.raises(NotImplementedError, match=match):
        TilePool(part, None)
