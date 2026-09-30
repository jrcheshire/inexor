"""The windowed tile loop across ranks: ghost slabs, shapes, membership and the census.

Each loopback rank holds its brick slabs, receives its neighbours' boundary slabs as ghosts
(`device.ghost`), sizes its programs from shapes maximized over ranks, and runs its tile
planes against card shards of the same coarse force. Every owned row's `w`, every owned
brick's scale, the cap and window shapes, and the destination census summed over ranks must
equal the one-rank loop's, at 1-4 ranks x 1-2 cards on `cdev8-tile32` with arena residents.
"""

import functools

import numpy as np
import pytest

pytest.importorskip("jax")

from inexor import state  # noqa: E402
from inexor.comm import allreduce_shapes, run_loopback  # noqa: E402
from inexor.decomp import Decomp  # noqa: E402
from inexor.device import ghost as dghost  # noqa: E402
from inexor.device import tile as dtile  # noqa: E402
from inexor.device import window as dwin  # noqa: E402
from inexor.device.coarse import shard_coarse_meshes  # noqa: E402
from inexor.device.migrate import pass_arena_index  # noqa: E402
from inexor.forces import COARSE_HALO, capacity_shape, make_tile_force_fn, tile_capacity  # noqa: E402
from tests.ranks_common import rank_cfg, rank_devices, whole_state  # noqa: E402
from tests.test_partial_state import restrict_to_slabs  # noqa: E402

C_DRIFT = 0.4


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


@functools.lru_cache(maxsize=None)
def _tile_force(cards):
    cfg = rank_cfg(cards)
    return make_tile_force_fn(cfg.n_fine, cfg.box_size, cfg.n_total, cfg.n_tile, cfg.b_fine,
                              r_s=cfg.r_s, paint=cfg.paint_short, frac_bits=cfg.frac_bits,
                              fdtype=cfg.np_fine_dtype)


@functools.lru_cache(maxsize=None)
def _g_coarse():
    cfg = rank_cfg(1)
    rng = np.random.default_rng(6)
    return tuple(rng.normal(scale=0.3, size=(cfg.n_coarse,) * 3).astype(cfg.np_coarse_dtype)
                 for _ in range(3))


def tile_pass(st, cfg, decomp, comm, devs):
    """One rank's tile phase as `engine.step` runs it; returns (cap, shapes, census)."""
    b_real = cfg._b_realized
    g = (cfg.n_tile, b_real, cfg.n_brick, cfg.n_fine)
    ghosts, _receipt = dghost.exchange_ghosts(st, decomp, comm, decomp.pad)
    view = dghost.SlabView(st, ghosts)
    members = state.TileMembers(st, *g, planes=decomp.planes)
    counts = view.tile_member_counts(*g, planes=decomp.planes)
    cap = capacity_shape(allreduce_shapes(comm, dict(
        cap=tile_capacity(counts.reshape(-1))))["cap"], rungs=cfg.cap_rungs)
    shapes = allreduce_shapes(comm, dtile.tile_step_shapes(st))
    shapes["window"] = allreduce_shapes(comm, dwin.window_shapes(
        view, cfg.n_tile, b_real, cfg.n_brick, planes=range(*decomp.planes)))
    one_tile, geom = _tile_force(cfg.device_cards)
    C = dict(cap=int(cap), n_tile=cfg.n_tile, n_brick=cfg.n_brick, n_fine=cfg.n_fine,
             n_coarse=cfg.n_coarse, box=cfg.box_size, coarse_cell=cfg.coarse_cell,
             cell=geom["cell"], b_real=int(b_real), alpha_k=0.87, bcoef=1.31)
    census = np.zeros(st.n_bricks, dtype=np.int64)
    for k, ((a, z), (x0, nx)) in enumerate(zip(decomp.card_planes(),
                                                decomp.coarse_shards(COARSE_HALO))):
        dev = None if devs is None else devs[k]
        shard = shard_coarse_meshes(_g_coarse(), x0, nx, dev)
        lp = dwin.tile_loop_windowed(view, one_tile, C, None, members, shapes,
                                     planes=range(a, z), coarse_shard=shard, device=dev,
                                     census=C_DRIFT)
        census += lp["census_counts"]
    return cap, shapes, census


def _owned_rows(st):
    """Per owned slab: its slot range's `w` and its residents' `w` in (brick, slot) order."""
    ar_slots, ar_bricks = pass_arena_index(st)
    out = {}
    for s in range(*st.owned_slabs):
        lo_b, hi_b = st.slab_bricks(s)
        a0, a1 = np.searchsorted(ar_bricks, [lo_b, hi_b])
        out[s] = (st.w[int(st.brick_start[lo_b]):int(st.brick_start[hi_b])].tobytes(),
                  st.w[ar_slots[a0:a1]].tobytes(), st.vel_scale[lo_b:hi_b].tobytes())
    return out


@pytest.fixture(scope="module")
def reference():
    import jax

    jax.config.update("jax_enable_x64", True)
    st = whole_state()
    cfg = rank_cfg(1)
    cap, shapes, census = tile_pass(st, cfg, Decomp.build(cfg), None, None)
    assert census.sum() == st.n_particles
    return cap, shapes, census, _owned_rows(st)


def _run_ranks(n_ranks, cards):
    cfg = rank_cfg(cards)
    whole = whole_state()

    def rank(c):
        d = Decomp.build(cfg, n_ranks=c.size, rank=c.rank)
        part = whole if c.size == 1 else restrict_to_slabs(whole, d.slabs)
        cap, shapes, census = tile_pass(part, cfg, d, c, rank_devices(c.rank, cards))
        return cap, shapes, census, _owned_rows(part)

    return run_loopback(n_ranks, rank, timeout=300.0)


def _check(reference, got):
    cap_w, shapes_w, census_w, rows_w = reference
    census = np.zeros_like(census_w)
    seen = {}
    for cap, shapes, c, rows in got:
        assert cap == cap_w and shapes == shapes_w
        census += c
        seen.update(rows)
    np.testing.assert_array_equal(census, census_w)
    assert sorted(seen) == sorted(rows_w)
    for s, (live, res, scale) in rows_w.items():
        assert seen[s][0] == live, f"slab {s}: live rows' w differ"
        assert seen[s][1] == res, f"slab {s}: residents' w differ"
        assert seen[s][2] == scale, f"slab {s}: brick scales differ"


@pytest.mark.parametrize("cards", [1, 2])
@pytest.mark.parametrize("n_ranks", [1, 2, 3, 4])
def test_rank_tile_loops_are_the_one_rank_loop(reference, n_ranks, cards):
    _check(reference, _run_ranks(n_ranks, cards))


def test_ghost_velocities_are_never_read(reference, monkeypatch):
    """Ghost `w` staged as noise instead of zeros changes nothing: buffer rows' velocities
    are masked out of the kick."""
    real_chunk = dghost.SlabView.slab_chunk

    def noisy(self, name, s, L):
        out = real_chunk(self, name, s, L)
        if name == "w" and not self.owns(s):
            out = np.random.default_rng(s).integers(-3000, 3000, size=out.shape,
                                                    dtype=np.int64).astype(out.dtype)
        return out

    monkeypatch.setattr(dghost.SlabView, "slab_chunk", noisy)
    _check(reference, _run_ranks(2, 1))


def test_a_ghost_slab_without_its_occupancy_fails(reference, monkeypatch):
    """The gate can fail: ghost slabs that arrive with their buckets emptied lose the buffer
    rows' paint, and the owned rows' kicks move."""
    real_pack = dghost.pack_slabs

    def empty_occ(st, slabs):
        p = real_pack(st, slabs)
        for k in [k for k in p if k.endswith(":occ")]:
            p[k] = np.zeros_like(p[k])
        return p

    monkeypatch.setattr(dghost, "pack_slabs", empty_occ)
    with pytest.raises(AssertionError):
        _check(reference, _run_ranks(2, 1))
