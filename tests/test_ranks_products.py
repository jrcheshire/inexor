"""The products across ranks: the P(k) card on the cards.

Each loopback rank holds its brick slabs (`restrict_to_slabs`) of a `cdev8-tile32` state with
arena residents. The card painted and transformed on the cards must be the one-rank,
one-card card in every field, at 1-4 ranks x 1-2 cards.
"""

import json

import numpy as np
import pytest

pytest.importorskip("jax")

from inexor.comm import run_loopback  # noqa: E402
from inexor.config import Cosmology  # noqa: E402
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
