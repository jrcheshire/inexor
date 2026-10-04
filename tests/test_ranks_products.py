"""The products across ranks: the P(k) card on the cards and the export decoded on them.

Each loopback rank holds its brick slabs (`restrict_to_slabs`) of a `cdev8-tile32` state with
arena residents. At 1-4 ranks x 1-2 cards, the card painted and transformed on the cards must
be the one-rank, one-card card in every field, and the export's parts in rank order must be
the single-file host export (`write_particles`) byte for byte, with its crc32.
"""

import json
import os

import numpy as np
import pytest

pytest.importorskip("jax")

from inexor import export  # noqa: E402
from inexor.comm import run_loopback  # noqa: E402
from inexor.config import PLANCK, Cosmology  # noqa: E402
from inexor.decomp import Decomp  # noqa: E402
from inexor.summary import pk_summary_card_cards  # noqa: E402
from tests.ranks_common import RANKS_MARKS, rank_cfg, rank_devices, whole_state  # noqa: E402
from tests.test_partial_state import restrict_to_slabs  # noqa: E402

pytestmark = RANKS_MARKS

A_OUT = 0.5


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def whole():
    return whole_state()


def _edges(cfg):
    # 8 bins over half Nyquist of the 64^3 coarse mesh; min_weight=1 keeps every bin
    return np.linspace(0.0, 0.5 * np.pi * cfg.n_coarse / cfg.box_size, 9)


def _card(st, cfg, decomp, comm, devs, **kw):
    kw = dict(dict(edges=_edges(cfg), min_weight=1.0), **kw)
    return pk_summary_card_cards(st, cfg, Cosmology(), A_OUT, devices=devs, decomp=decomp,
                                 comm=comm, **kw)


def _ranks(whole, cfg, n_ranks, cards, **kw):
    def rank(c):
        d = Decomp.build(cfg, n_ranks=c.size, rank=c.rank)
        part = whole if c.size == 1 else restrict_to_slabs(whole, d.slabs)
        return _card(part, cfg, d, c, rank_devices(c.rank, cards), **kw)

    return run_loopback(n_ranks, rank, timeout=120.0)


@pytest.fixture(scope="module")
def reference(whole):
    import jax

    jax.config.update("jax_enable_x64", True)
    cfg = rank_cfg(1)
    card = _card(whole, cfg, Decomp.build(cfg), None, None)
    assert card["n_bins"] == 8 and card["transform"] == "cards"
    assert card["n_particles"] == whole.n_particles
    assert max(abs(p) for p in card["p"]) > 0.0, "vacuous: zero power"
    return card


@pytest.mark.parametrize("cards", [1, 2])
@pytest.mark.parametrize("n_ranks", [1, 2, 3, 4])
def test_rank_cards_are_the_one_rank_card(whole, reference, n_ranks, cards):
    want = json.dumps(reference, sort_keys=True)
    for card in _ranks(whole, rank_cfg(cards), n_ranks, cards):
        assert json.dumps(card, sort_keys=True) == want


def test_an_empty_card_is_refused_on_every_rank(whole):
    def rank(c):
        cfg = rank_cfg(1)
        d = Decomp.build(cfg, n_ranks=c.size, rank=c.rank)
        with pytest.raises(ValueError, match="no bin reached"):
            _card(restrict_to_slabs(whole, d.slabs), cfg, d, c, None, min_weight=1e12)
        return True

    assert all(run_loopback(2, rank, timeout=120.0))


@pytest.fixture(scope="module")
def host_export(whole, tmp_path_factory):
    d = str(tmp_path_factory.mktemp("host") / "e")
    head = export.write_particles(whole, d, dtype=np.float32, a=A_OUT, cosmo=PLANCK)
    assert whole.arena_used > 0, "vacuous: no arena residents"
    return d, head


def _bytes(path):
    return np.ascontiguousarray(np.load(path)).tobytes()


@pytest.mark.parametrize("cards", [1, 2])
@pytest.mark.parametrize("n_ranks", [1, 2, 3, 4])
def test_rank_exports_decoded_on_the_cards_are_the_host_export(whole, host_export, tmp_path,
                                                               n_ranks, cards):
    one, want = host_export
    cfg = rank_cfg(cards)
    d = str(tmp_path / "parts")

    def rank(c):
        dc = Decomp.build(cfg, n_ranks=c.size, rank=c.rank)
        part = whole if c.size == 1 else restrict_to_slabs(whole, dc.slabs)
        return export.write_particle_parts(part, d, comm=c, decode="cards",
                                           devices=rank_devices(c.rank, cards),
                                           dtype=np.float32, a=A_OUT, cosmo=PLANCK,
                                           expect_total=whole.n_live)

    heads = run_loopback(n_ranks, rank, timeout=120.0)
    head = heads[0]
    assert len(head["parts"]) == n_ranks and head["crc32"] == want["crc32"]
    for key in ("x", "v"):
        got = b"".join(_bytes(os.path.join(d, p["files"][key])) for p in head["parts"])
        assert got == _bytes(os.path.join(one, want["files"][key]))
