"""`decomp.Decomp`: at one rank it reproduces the single-node step's card split exactly, and
across ranks its ranges tile the box without overlap."""

import pytest

from inexor.decomp import Decomp
from inexor.forces import COARSE_HALO
from inexor.ooc_fft import partition_units
from inexor.plan import PRESETS, engine_config


def _cfg(name, cards=1):
    return engine_config(name, device_cards=cards)


def _tiles(ranges, lo, hi):
    """Contiguous, disjoint, non-empty, covering [lo, hi)."""
    assert ranges[0][0] == lo and ranges[-1][1] == hi
    assert all(a < b for a, b in ranges)
    assert all(ranges[k][1] == ranges[k + 1][0] for k in range(len(ranges) - 1))


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_one_rank_is_the_single_node_card_split(name):
    """The formulas `engine.step` used inline before `Decomp`, for every card count."""
    for cards in range(1, min(8, _cfg(name).tiles_side) + 1):
        cfg = _cfg(name, cards)
        d = Decomp.build(cfg)
        assert d.planes == (0, cfg.tiles_side)
        assert d.card_planes() == partition_units(cfg.tiles_side, cards, 1)
        per_tile = cfg.n_tile // (cfg.n_fine // cfg.n_coarse)
        assert d.coarse_shards(COARSE_HALO) == [
            (a * per_tile - COARSE_HALO, (b - a) * per_tile + 2 * COARSE_HALO)
            for a, b in partition_units(cfg.tiles_side, cards, 1)]
        assert d.card_pencils() == partition_units(cfg.n_coarse, cards, 1)


@pytest.mark.parametrize("name", sorted(PRESETS))
@pytest.mark.parametrize("n_ranks", [1, 2, 4, 8])
@pytest.mark.parametrize("cards", [1, 2])
def test_ranks_and_cards_tile_the_box(name, n_ranks, cards):
    cfg = _cfg(name, cards)
    try:
        ds = [Decomp.build(cfg, n_ranks=n_ranks, rank=r) for r in range(n_ranks)]
    except ValueError as e:
        # only the two refusals are allowed, and only where they are true
        bpt = cfg.n_tile // cfg.n_brick
        thin = min(b - a for a, b in partition_units(cfg.tiles_side, n_ranks, 1)) \
            if n_ranks <= cfg.tiles_side else 0
        assert n_ranks * cards > cfg.tiles_side or thin * bpt < 3, e
        return
    d0 = ds[0]
    _tiles([d.planes for d in ds], 0, cfg.tiles_side)
    _tiles([d.slabs for d in ds], 0, cfg.n_fine // cfg.n_brick)
    _tiles([d.coarse_planes for d in ds], 0, cfg.n_coarse)
    _tiles([d.pencils for d in ds], 0, cfg.n_coarse)
    for r, d in enumerate(ds):
        assert d.rank_planes == d0.rank_planes
        _tiles(d.card_planes(), *d.planes)
        _tiles(d.card_pencils(), *d.pencils)
        assert d.card_planes() == d0.card_planes(r)
        assert d.neighbours == ((r - 1) % n_ranks, (r + 1) % n_ranks)
        for p in range(*d.planes):
            assert d0.owner_of_plane(p) == r
        halo = COARSE_HALO
        for (x0, nx), (a, b) in zip(d.coarse_shards(halo), d.card_planes()):
            assert x0 == a * d.coarse_per_tile - halo
            assert nx == (b - a) * d.coarse_per_tile + 2 * halo


def test_refusals():
    cfg = _cfg("smoke")  # 4 tile planes, 2 brick slabs per plane
    with pytest.raises(ValueError, match="a card would run no tiles"):
        Decomp.build(_cfg("smoke", 2), n_ranks=3)
    with pytest.raises(ValueError, match="fewer than 2 \\* reach \\+ 1"):
        Decomp.build(cfg, n_ranks=4)
    Decomp.build(cfg, n_ranks=2)
    with pytest.raises(ValueError, match="fewer than 2 \\* reach \\+ 1"):
        Decomp.build(cfg, n_ranks=2, reach=2)
    with pytest.raises(ValueError, match="outside"):
        Decomp.build(cfg, n_ranks=2, rank=2)
    # one rank is never refused on slab count: its migrate stays inside the node
    Decomp.build(cfg, n_ranks=1, reach=4)
