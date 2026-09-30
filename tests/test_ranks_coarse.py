"""The coarse arm across ranks: paint, ghost-plane fold and the pencil-transposed solve.

Each loopback rank holds its brick slabs (`restrict_to_slabs`), paints its own chunks, folds
ghost planes into its neighbours, and solves with the spectrum split into y-pencils. Every
coarse force plane each rank's card shards hold (halo included) must equal the one-rank,
one-card solve's plane byte for byte, at 1-4 ranks x 1-2 cards on `cdev8-tile32` with arena
residents.
"""

import numpy as np
import pytest

pytest.importorskip("jax")

from inexor import comm as cm  # noqa: E402
from inexor import state  # noqa: E402
from inexor.codec import T9Layout  # noqa: E402
from inexor.comm import run_loopback  # noqa: E402
from inexor.decomp import Decomp  # noqa: E402
from inexor.device.coarse import CardShards  # noqa: E402
from inexor.device.paint import coarse_delta_cards  # noqa: E402
from inexor.forces import COARSE_HALO, coarse_force_meshes, coarse_kernel_parts  # noqa: E402
from inexor.plan import PRESETS, engine_config  # noqa: E402
from tests.test_partial_state import restrict_to_slabs  # noqa: E402

PRESET = "cdev8-tile32"


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _cfg(cards):
    return engine_config(PRESET, coarse_backend="device", tile_backend="device",
                         migrate_backend="device", device_cards=cards, tile_workers=1)


@pytest.fixture(scope="module")
def whole():
    """A jittered lattice at the preset's geometry, migrated into a populated arena."""
    import jax

    jax.config.update("jax_enable_x64", True)
    p = PRESETS[PRESET]
    n, box = p["n_part"], p["box"]
    rng = np.random.default_rng(11)
    q = (np.arange(n) + 0.5) * (box / n)
    x = np.stack(np.meshgrid(q, q, q, indexing="ij"), axis=-1).reshape(-1, 3)
    x = np.mod(x + rng.normal(scale=0.3 * box / n, size=x.shape), box)
    v = rng.normal(scale=0.5, size=x.shape)
    cfg = _cfg(1)
    st = state.SlotState.build(x, v, T9Layout(box_size=box, n_part=n, bucket_cells=2),
                               cfg.n_fine // cfg.n_brick, brick_slack=0.0, arena_frac=0.3,
                               with_ids=False)
    state.drift_and_migrate(st, 0.3)
    assert st.arena_used > 0, "vacuous: no arena residents"
    return st


def _devices(rank, cards):
    """Rank `rank`'s cards: disjoint devices while the backend has enough, else reused."""
    import jax

    if cards == 1:
        return None
    devs = jax.devices()
    return [devs[(rank * cards + k) % len(devs)] for k in range(cards)]


def _solve(st, cfg, decomp, comm, devs):
    """One rank's card force shards: `(x0, nx, (three host meshes))` per card."""
    kc = [(lo, hi, None if devs is None else devs[k])
          for k, (lo, hi) in enumerate(decomp.card_pencils())]
    parts = coarse_kernel_parts(cfg.n_coarse, cfg.box_size, "long", r_s=cfg.r_s,
                                match=cfg.coarse_match, fdtype=cfg.np_coarse_dtype, cards=kc)
    delta = coarse_delta_cards(st, cfg, devices=devs, decomp=decomp, comm=comm)
    out = CardShards([(x0, nx, None if devs is None else devs[k])
                      for k, (x0, nx) in enumerate(decomp.coarse_shards(COARSE_HALO))],
                     cfg.n_coarse)
    g = coarse_force_meshes(delta, cfg.n_coarse, cfg.box_size, "long", r_s=cfg.r_s,
                            match=cfg.coarse_match, fdtype=cfg.np_coarse_dtype, parts=parts,
                            out=out, fold_kernel=True, decomp=decomp, comm=comm)
    return [(s["x0"], s["nx"], tuple(np.asarray(m) for m in s["meshes"])) for s in g]


@pytest.fixture(scope="module")
def reference(whole):
    """plane -> its three force planes, from one rank on one card (a whole-mesh shard)."""
    cfg = _cfg(1)
    d = Decomp.build(cfg)
    ((x0, nx, meshes),) = _solve(whole, cfg, d, None, None)
    n = cfg.n_coarse
    assert float(np.abs(meshes[0]).max()) > 0.0, "vacuous: zero force"
    return {(x0 + i) % n: tuple(m[i] for m in meshes) for i in range(nx)}


def _check_against(reference, shards, n):
    for x0, nx, meshes in shards:
        for i in range(nx):
            want = reference[(x0 + i) % n]
            for a, b in zip(meshes, want):
                assert a[i].tobytes() == b.tobytes(), f"plane {(x0 + i) % n} differs"


@pytest.mark.parametrize("cards", [1, 2])
@pytest.mark.parametrize("n_ranks", [1, 2, 3, 4])
def test_rank_force_shards_are_the_one_rank_planes(whole, reference, n_ranks, cards):
    cfg = _cfg(cards)

    def rank(c):
        d = Decomp.build(cfg, n_ranks=c.size, rank=c.rank)
        part = whole if c.size == 1 else restrict_to_slabs(whole, d.slabs)
        return _solve(part, cfg, d, c, _devices(c.rank, cards))

    for shards in run_loopback(n_ranks, rank, timeout=120.0):
        _check_against(reference, shards, cfg.n_coarse)


def test_dropping_the_ghost_plane_exchange_fails(whole, reference, monkeypatch):
    """The gate can fail: ranks that keep their ghost planes' mass to themselves paint a
    different density."""
    real = cm.exchange_neighbours

    def no_planes(comm, to_left, to_right):
        got = real(comm, to_left, to_right)
        if "planes" in to_left:
            return tuple(dict(g=np.zeros(0, np.int64),
                              planes=np.zeros((0,) + d["planes"].shape[1:], np.int64))
                         for d in got)
        return got

    monkeypatch.setattr(cm, "exchange_neighbours", no_planes)
    cfg = _cfg(1)

    def rank(c):
        d = Decomp.build(cfg, n_ranks=c.size, rank=c.rank)
        return _solve(restrict_to_slabs(whole, d.slabs), cfg, d, c, None)

    with pytest.raises(AssertionError, match="differs"):
        for shards in run_loopback(2, rank, timeout=120.0):
            _check_against(reference, shards, cfg.n_coarse)
