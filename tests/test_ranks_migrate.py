"""The device migrate and the fused migrate + repack across ranks.

Each loopback rank migrates its own slabs: the reach is the max over ranks, the boundary
slabs' scales and emigrants go to the neighbour ranks, and the arena replay runs over the
rank's own slabs. Per owned slab, the occupancy, the runs' starts relative to the slab, the
live rows, the brick scales and the arena residents (in brick, slot order; arena slots are
rank-local) must equal the one-rank pass's, at 1-4 ranks x 1-2 cards on `cdev8-tile32`.
The fused pass is sized from each rank's tile-loop census (`tests/test_ranks_tile.py`).
"""

import numpy as np
import pytest

pytest.importorskip("jax")

from inexor import comm as cm  # noqa: E402
from inexor.comm import run_loopback  # noqa: E402
from inexor.decomp import Decomp  # noqa: E402
from inexor.device import migrate as dmig  # noqa: E402
from inexor.device.fused import migrate_repack_device  # noqa: E402
from tests.ranks_common import rank_cfg, rank_devices, whole_state  # noqa: E402
from tests.test_partial_state import restrict_to_slabs  # noqa: E402
from tests.test_ranks_tile import C_DRIFT, tile_pass  # noqa: E402

DRIFT = 0.45


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _slabs(st):
    """Per owned slab, its bytes in a rank-count-independent form."""
    ar_slots, ar_bricks = dmig.pass_arena_index(st)
    nb2 = st.bricks_per_side ** 2
    out = {}
    for s in range(*st.owned_slabs):
        lo_b, hi_b = st.slab_bricks(s)
        s0 = int(st.brick_start[lo_b])
        occ = st._occ(lo_b, hi_b).reshape(nb2, -1)
        live = occ.sum(axis=1, dtype=np.int64)
        starts = np.asarray(st.brick_start[lo_b:hi_b], dtype=np.int64) - s0
        runs = np.concatenate([np.arange(a, a + n) for a, n in zip(starts, live)]) + s0
        a0, a1 = np.searchsorted(ar_bricks, [lo_b, hi_b])
        rows = ar_slots[a0:a1]
        out[s] = (occ.tobytes(), starts.tobytes(), st.off[runs].tobytes(), st.w[runs].tobytes(),
                  st.vel_scale[lo_b:hi_b].tobytes(), ar_bricks[a0:a1].tobytes(),
                  st.off[rows].tobytes(), st.w[rows].tobytes(),
                  np.asarray(st.arena_bucket)[rows - int(st.arena_base)].tobytes())
    return out


def _check(want, got):
    seen = {}
    for g in got:
        seen.update(g)
    assert sorted(seen) == sorted(want)
    names = ("occupancy", "starts", "live off", "live w", "scales", "resident bricks",
             "resident off", "resident w", "resident buckets")
    for s in want:
        for name, a, b in zip(names, seen[s], want[s]):
            assert a == b, f"slab {s}: {name} differ"


def _migrate(st, cfg, decomp, comm, devs):
    dmig.drift_and_migrate_device(st, DRIFT, devices=devs, comm=comm)
    st.check()
    return _slabs(st)


def _fused(st, cfg, decomp, comm, devs):
    _cap, _shapes, census = tile_pass(st, cfg, decomp, comm, devs)
    migrate_repack_device(st, C_DRIFT, census, brick_slack=cfg.brick_slack, devices=devs,
                          comm=comm)
    st.check()
    assert st.arena_used == 0
    return _slabs(st)


PASSES = dict(migrate=_migrate, fused=_fused)


@pytest.fixture(scope="module")
def reference():
    import jax

    jax.config.update("jax_enable_x64", True)
    out = {}
    for name, fn in PASSES.items():
        cfg = rank_cfg(1)
        st = whole_state()
        out[name] = fn(st, cfg, Decomp.build(cfg), None, None)
    return out


def _ranks(fn, n_ranks, cards):
    cfg = rank_cfg(cards)
    whole = whole_state()

    def rank(c):
        d = Decomp.build(cfg, n_ranks=c.size, rank=c.rank)
        part = whole if c.size == 1 else restrict_to_slabs(whole, d.slabs)
        n0 = c.allreduce(int(part.n_particles))
        got = fn(part, cfg, d, c, rank_devices(c.rank, cards))
        assert c.allreduce(int(part.n_particles)) == n0 and part.n_live == part.n_particles
        return got

    return run_loopback(n_ranks, rank, timeout=300.0)


@pytest.mark.parametrize("cards", [1, 2])
@pytest.mark.parametrize("n_ranks", [1, 2, 3, 4])
@pytest.mark.parametrize("which", sorted(PASSES))
def test_rank_passes_are_the_one_rank_pass(reference, which, n_ranks, cards):
    _check(reference[which], _ranks(PASSES[which], n_ranks, cards))


def test_particles_cross_the_rank_boundaries(reference):
    """Not vacuous: at 4 ranks the hand-off carries rows both ways."""
    cfg = rank_cfg(1)
    whole = whole_state()

    def rank(c):
        d = Decomp.build(cfg, n_ranks=c.size, rank=c.rank)
        part = restrict_to_slabs(whole, d.slabs)
        out = dmig.drift_and_migrate_device(part, DRIFT, comm=c)
        return out["migrate_device"]

    for rec in run_loopback(4, rank, timeout=300.0):
        assert rec["rank_emigrant_rows_sent"] > 0 and rec["rank_emigrant_rows_received"] > 0


def test_skipping_the_scales_exchange_fails(reference, monkeypatch):
    real = cm.exchange_neighbours

    def own_scales(comm, to_left, to_right):
        got = real(comm, to_left, to_right)
        if set(to_left) == {"s"}:
            return tuple(dict(s=np.ones_like(d["s"])) for d in got)
        return got

    monkeypatch.setattr(cm, "exchange_neighbours", own_scales)
    with pytest.raises(AssertionError, match="differ"):
        _check(reference["migrate"], _ranks(_migrate, 2, 1))


def test_a_dropped_hand_off_row_is_caught_by_the_census(monkeypatch):
    """One emigrant row lost in transit, with its destination count lowered to match: the
    sender's unconsumed-emigrant census refuses."""
    real = dmig._pack_emigrants
    per_slab = None

    def drop_one(slabs):
        out = real(slabs)
        for key in [k for k in out if k.endswith(":meta")]:
            s = key.split(":")[0]
            n = int(out[key][0])
            if n:
                d = int(out[f"{s}:dest"][n - 1]) // per_slab
                out[key] = out[key].copy()
                out[key][0] = n - 1
                out[f"{s}:to"] = out[f"{s}:to"].copy()
                out[f"{s}:to"][d] -= 1
                for k in dmig._EMIGRANT_FIELDS:
                    if f"{s}:{k}" in out:
                        out[f"{s}:{k}"] = out[f"{s}:{k}"][:n - 1]
                break
        return out

    monkeypatch.setattr(dmig, "_pack_emigrants", drop_one)
    whole = whole_state()
    per_slab = whole.buckets_per_brick * whole.bricks_per_side ** 2
    with pytest.raises(AssertionError, match="unconsumed"):
        _ranks(_migrate, 2, 1)


def test_a_reach_past_a_rank_is_refused_on_every_rank():
    cfg = rank_cfg(1)
    whole = whole_state()

    def rank(c):
        d = Decomp.build(cfg, n_ranks=c.size, rank=c.rank)
        part = restrict_to_slabs(whole, d.slabs)
        try:
            dmig.drift_and_migrate_device(part, 40.0, comm=c)
        except ValueError as e:
            return str(e)
        return None

    msgs = run_loopback(4, rank, timeout=300.0)
    assert all(m is not None and "skip past a neighbour rank" in m for m in msgs), msgs
