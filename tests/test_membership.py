"""Tile membership: the vectorized bricks and counts are exactly the per-brick Python loops they
replace (kept here as the reference), arena residents included, and ranks that each build their
own tiles' membership agree with one rank on the tile set and the capacity."""

import numpy as np
import pytest

from inexor import state
from inexor.codec import T9Layout
from inexor.comm import run_loopback
from inexor.decomp import Decomp
from inexor.forces import tile_capacity
from inexor.layout import brick_span
from inexor.plan import PRESETS, engine_config


def _loop_bricks(tijk, n_tile, b_fine, n_brick, nb):
    """The former `SlotState.tile_bricks`: the reference order."""
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


def _geom(name):
    cfg = engine_config(name)
    return cfg, (cfg.n_tile, cfg._b_realized, cfg.n_brick), cfg.n_fine // cfg.n_brick


def _some_tiles(s, rng, n=40):
    """Every tile when few; else the corners (wrap on every axis) and a random sample."""
    if s**3 <= 512:
        return [(i, j, k) for i in range(s) for j in range(s) for k in range(s)]
    corners = [(i, j, k) for i in (0, s - 1) for j in (0, s - 1) for k in (0, s - 1)]
    return corners + [tuple(int(q) for q in rng.integers(0, s, 3)) for _ in range(n)]


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_tile_brick_ids_are_the_triple_loop(name):
    cfg, g, nb = _geom(name)
    for t in _some_tiles(cfg.tiles_side, np.random.default_rng(0)):
        got = state.tile_brick_ids(t, *g, nb)
        assert got.dtype == np.int64
        np.testing.assert_array_equal(got, _loop_bricks(t, *g, nb))


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_window_counts_are_the_per_brick_sums(name):
    cfg, g, nb = _geom(name)
    rng = np.random.default_rng(1)
    grid = rng.integers(0, 1000, size=(nb, nb, nb), dtype=np.int64)
    got = state.tile_window_counts(grid, *g)
    s = cfg.tiles_side
    assert got.shape == (s, s, s)
    flat = grid.reshape(-1)
    for t in _some_tiles(s, rng):
        assert got[t] == flat[_loop_bricks(t, *g, nb)].sum(), t
    lo, hi = s // 2, s
    np.testing.assert_array_equal(state.tile_window_counts(grid, *g, planes=(lo, hi)),
                                  got[lo:hi])


@pytest.fixture
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def test_counts_on_a_stepped_state_include_arena_residents(_x64):
    """Three device-lane steps with no repack leave residents in the arena."""
    from inexor import engine
    from tests.test_engine_device_backend import _coeffs, _state
    from tests.test_engine_device_step import _cfg

    cfg = _cfg(coarse_backend="device", tile_backend="device", migrate_backend="device",
               device_tile_window=True, repack_every=0)
    st = _state(cfg)
    engine.run(st, cfg, _coeffs(3))
    assert int((st.arena_bucket >= 0).sum()) > 0, "VACUOUS: no arena residents"
    want = np.array([st.brick_member_count(b) for b in range(st.n_bricks)], dtype=np.int64)
    for w in (1, 3, None):
        np.testing.assert_array_equal(st.brick_member_counts(workers=w), want)
    g = (cfg.n_tile, cfg._b_realized, cfg.n_brick, cfg.n_fine)
    counts = st.tile_member_counts(*g)
    for t in cfg.tiles:
        assert counts[t] == sum(st.brick_member_count(b) for b in st.tile_bricks(t, *g)), t
    members = state.TileMembers(st, *g)
    assert list(members) == list(cfg.tiles)
    for t in cfg.tiles:
        np.testing.assert_array_equal(members[t], _loop_bricks(t, *g[:3], st.bricks_per_side))
    with pytest.raises(KeyError):
        state.TileMembers(st, *g, planes=(0, 1))[(1, 0, 0)]


def _lattice_state(name, seed=3):
    """A jittered-lattice state of preset `name`'s geometry."""
    p = PRESETS[name]
    cfg = engine_config(name)
    n, box = p["n_part"], p["box"]
    rng = np.random.default_rng(seed)
    q = (np.arange(n) + 0.5) * (box / n)
    x = np.stack(np.meshgrid(q, q, q, indexing="ij"), axis=-1).reshape(-1, 3)
    x = np.mod(x + rng.normal(scale=0.3 * box / n, size=x.shape), box)
    v = rng.normal(scale=0.5, size=x.shape)
    t9 = T9Layout(box_size=box, n_part=n, bucket_cells=2)
    return cfg, state.SlotState.build(x, v, t9, cfg.n_fine // cfg.n_brick, brick_slack=0.1,
                                      arena_frac=0.05, with_ids=False)


@pytest.mark.parametrize("n_ranks", [1, 2, 4, 8])
def test_ranks_build_their_own_tiles_and_agree_on_cap(n_ranks):
    cfg, st = _lattice_state("cdev8-tile32")
    g = (cfg.n_tile, cfg._b_realized, cfg.n_brick, cfg.n_fine)
    cap_one = tile_capacity(st.tile_member_counts(*g).reshape(-1))

    def rank(c):
        d = Decomp.build(cfg, n_ranks=c.size, rank=c.rank)
        members = state.TileMembers(st, *g, planes=d.planes)
        counts = st.tile_member_counts(*g, planes=d.planes)
        return list(members), c.allreduce(tile_capacity(counts.reshape(-1)), "max")

    got = run_loopback(n_ranks, rank)
    tiles = [t for keys, _ in got for t in keys]
    assert tiles == list(cfg.tiles)
    assert all(cap == cap_one for _, cap in got)
