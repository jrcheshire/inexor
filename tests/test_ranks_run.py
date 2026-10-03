"""`engine.run` across ranks: checkpoints byte-identical at any rank count.

From the same state (a `cdev8-tile32` lattice with arena residents; the lead drift runs
first), K = 6 steps with a checkpoint every 3 on N loopback ranks x 1-2 cards must write
generations whose every file (slab files and manifest) is the one-rank run's byte for byte.
A step-3 checkpoint from N ranks resumes on M ranks and writes the step-6 generation of the
uninterrupted run. A rank that raises mid-run ends every rank, and the generation in
progress has no manifest.
"""

import os
import shutil

import numpy as np
import pytest

pytest.importorskip("jax")

from inexor import engine  # noqa: E402
from inexor.comm import CommAborted, run_loopback  # noqa: E402
from inexor.config import Cosmology  # noqa: E402
from inexor.decomp import Decomp  # noqa: E402
from inexor.device import ghost as dghost  # noqa: E402
from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table  # noqa: E402
from inexor.plan import engine_config  # noqa: E402
from tests.ranks_common import (  # noqa: E402
    PRESET,
    RANKS_MARKS,
    hashes,
    need_devices,
    rank_cfg,
    rank_devices,
    whole_state,
)
from tests.test_partial_state import restrict_to_slabs  # noqa: E402

pytestmark = [pytest.mark.slow, *RANKS_MARKS]

K, EVERY = 6, 3


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _coeffs():
    return bullfrog_float_coeffs(bullfrog_table(a_grid(0.1, 1.0, K, "log"), Cosmology()))


def run_ranks(n_ranks, cards, ckpt_dir, resume=False, stop_at=None):
    """Run on `n_ranks` loopback ranks into `ckpt_dir`: from the whole state, or (`resume`)
    from the newest checkpoint there. Returns each rank's per-step stats."""
    need_devices(cards)
    cfg = rank_cfg(cards, checkpoint_dir=str(ckpt_dir), checkpoint_every=EVERY)
    co = _coeffs()
    whole = None if resume else whole_state()

    def rank(c):
        d = Decomp.build(cfg, n_ranks=c.size, rank=c.rank)
        res = None
        if resume:
            st, res = engine.load_checkpoint(str(ckpt_dir), cfg, co, comm=c,
                                             slabs=None if c.size == 1 else d.slabs)
        else:
            st = whole if c.size == 1 else restrict_to_slabs(whole, d.slabs)
        return engine.run(st, cfg, co, resume=res, stop_at=stop_at, comm=c, decomp=d,
                          devices=rank_devices(c.rank, cards))

    return run_loopback(n_ranks, rank, timeout=600.0)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    import jax

    jax.config.update("jax_enable_x64", True)
    d = tmp_path_factory.mktemp("ref")
    (out,) = run_ranks(1, 1, d)
    h = hashes(d)
    assert sorted({k.split("/")[0] for k in h}) == ["gen0", "gen1"]
    assert out[-1]["ranks"]["n_ranks"] == 1
    # one rank sends nothing to another rank
    assert all(e["bytes"] == 0 for e in out[-1]["ranks"]["comm"]["ops"].values())
    return d, h


@pytest.mark.parametrize("n_ranks,cards", [(1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1),
                                           (4, 2)])
def test_rank_checkpoints_are_the_one_rank_bytes(reference, tmp_path, n_ranks, cards):
    _d, want = reference
    outs = run_ranks(n_ranks, cards, tmp_path)
    assert hashes(tmp_path) == want
    if n_ranks > 1:
        rec = outs[0][-1]["ranks"]
        assert rec["n_ranks"] == n_ranks and rec["ghost_bytes"] > 0
        assert rec["forward_sent_bytes"] > 0 and rec["inverse_sent_bytes"] > 0
        assert sum(o[-1]["ranks"]["emigrant_rows_sent"] for o in outs) > 0
        # the ledger sees at least the spectrum transposes' bytes
        a2a = rec["comm"]["ops"]["Alltoallv"]
        assert a2a["calls"] > 0
        assert a2a["bytes"] >= rec["forward_sent_bytes"] + rec["inverse_sent_bytes"]


def test_the_rank_record_carries_each_ranks_particle_flux(tmp_path):
    outs = run_ranks(3, 1, tmp_path)
    total = sum(o[0]["n_migrated_checked"] for o in outs)
    moved = 0
    for k in range(K):
        assert sum(o[k]["n_migrated_checked"] for o in outs) == total
        assert sum(o[k]["ranks"]["rows_in"] - o[k]["ranks"]["rows_out"] for o in outs) == 0
    # each step's migrate changes a rank's count by exactly rows_in - rows_out
    for o in outs:
        for k in range(1, K):
            rec = o[k]["ranks"]
            assert (o[k]["n_migrated_checked"] - o[k - 1]["n_migrated_checked"]
                    == rec["rows_in"] - rec["rows_out"])
            moved += rec["rows_in"]
    assert moved > 0


@pytest.mark.parametrize("n_write,n_read", [(2, 3), (4, 1), (1, 4)])
def test_a_checkpoint_resumes_on_another_rank_count(reference, tmp_path, n_write, n_read):
    _d, want = reference
    first = tmp_path / "first"
    run_ranks(n_write, 1, first, stop_at=EVERY)
    second = tmp_path / "second"
    second.mkdir()
    shutil.copytree(first / "gen0", second / "gen0")
    run_ranks(n_read, 1, second, resume=True)
    got = hashes(second)
    assert {k: v for k, v in got.items() if k.startswith("gen1/")} == {
        k: v for k, v in want.items() if k.startswith("gen1/")}


def test_a_failing_rank_ends_every_rank(reference, tmp_path, monkeypatch):
    real = dghost.exchange_ghosts
    calls = {}
    ended = {}

    def failing(st, decomp, comm, pad):
        calls[comm.rank] = calls.get(comm.rank, 0) + 1
        if comm.rank == 1 and calls[comm.rank] == 5:
            raise RuntimeError("injected failure on rank 1 in step 5")
        return real(st, decomp, comm, pad)

    monkeypatch.setattr(dghost, "exchange_ghosts", failing)
    cfg = rank_cfg(1, checkpoint_dir=str(tmp_path), checkpoint_every=EVERY)
    co = _coeffs()
    whole = whole_state()

    def rank(c):
        d = Decomp.build(cfg, n_ranks=c.size, rank=c.rank)
        try:
            return engine.run(restrict_to_slabs(whole, d.slabs), cfg, co, comm=c, decomp=d)
        except BaseException as e:
            ended[c.rank] = type(e)
            raise

    with pytest.raises(RuntimeError, match="injected failure"):
        run_loopback(3, rank, timeout=600.0)
    assert ended[1] is RuntimeError
    assert all(issubclass(ended[r], CommAborted) for r in (0, 2)), ended
    assert os.path.exists(tmp_path / "gen0" / "manifest.json")
    assert not os.path.exists(tmp_path / "gen1" / "manifest.json")


def test_other_lanes_refuse_a_node_local_state():
    whole = whole_state()
    host = engine_config(PRESET, tile_workers=1)
    assert host.coarse_backend == host.tile_backend == host.migrate_backend == "host"
    d = Decomp.build(host, n_ranks=2, rank=0)
    part = restrict_to_slabs(whole, d.slabs)
    with pytest.raises(NotImplementedError, match="production device lane"):
        engine.run(part, host, _coeffs())
    with pytest.raises(NotImplementedError, match="production device lane"):
        engine.step(part, host, (0.1, 0.1), 0.1, repack_due=True)
    assert np.isfinite(float(part.vel_scale.max()))
