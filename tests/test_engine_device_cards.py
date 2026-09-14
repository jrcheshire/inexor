"""E3: the step on several cards -- the coarse paint and solve split across them and
one thread per card walking its own tile planes through its own window -- gated
BITWISE against the same run on one card.

The multi-card tests need that many jax devices and SKIP otherwise: replicating
one device's handle would bypass `validate()`'s refusal of more cards than
devices. On the laptop run them under
`XLA_FLAGS=--xla_force_host_platform_device_count=4`.
"""

import pytest

pytest.importorskip("jax")

from inexor import engine  # noqa: E402
from tests.test_engine_device_backend import _coeffs, _same, _state  # noqa: E402
from tests.test_engine_device_step import _cfg, _same_stats  # noqa: E402

KW = dict(coarse_backend="device", tile_backend="device", migrate_backend="device",
          device_tile_window=True)
#: receipts that describe the card split rather than the step, and the paint's
#: compile count, which depends on what already ran in the process (the one-card
#: run compiles the program the wider run then reuses)
CARD_KEYS = ("coarse_card_chunks", "coarse_cards", "coarse_card_ranges",
             "coarse_ghost_planes_nonzero", "device_cards", "tile_cards",
             "coarse_jit_traces")


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _n_devices():
    import jax

    return len(jax.devices())


@pytest.mark.parametrize("w", [2, 4])
def test_a_run_on_w_cards_is_bitwise_the_run_on_one(w):
    if _n_devices() < w:
        pytest.skip(f"needs {w} jax devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    co = _coeffs(3)
    c1, cw = _cfg(**KW), _cfg(device_cards=w, **KW)
    s1, sw = _state(c1), _state(cw)
    o1 = engine.run(s1, c1, co)
    ow = engine.run(sw, cw, co)

    _same(s1, sw, f"{w} cards vs one")
    _same_stats(o1, ow, drop=CARD_KEYS)
    assert sum(o["n_arena_overflow"] for o in ow) > 0, "VACUOUS: nothing spilled"
    n_tiles = len(c1.tiles)
    for o in ow:
        assert o["device_cards"] == w
        cards = o["tile_cards"]
        assert len(cards) == w and all(c["tiles_run"] > 0 for c in cards), (
            "a card ran no tiles")
        assert sum(c["tiles_run"] for c in cards) == n_tiles
        assert o["coarse_cards"] == w and all(ch > 0 for ch in o["coarse_card_chunks"]), (
            "a card painted no chunk")
        # the migrate and repack split too, or say why not
        md = o["migrate_device"]
        assert md["cards"] == w or md["fallback"], "the migrate did not use the cards"
        assert o["repack"]["repack_device"]["cards"] == w
    assert all(o["device_cards"] == 1 and len(o["tile_cards"]) == 1 for o in o1)


def test_validate_refuses_the_card_counts_that_cannot_apply():
    with pytest.raises(ValueError, match="device_cards must be >= 1"):
        _cfg(device_cards=0, **KW).validate()
    for kw in (dict(KW, device_tile_window=False), dict(KW, coarse_backend="host")):
        with pytest.raises(ValueError, match="device_cards"):
            _cfg(device_cards=2, **kw).validate()
    with pytest.raises(ValueError, match="tile planes"):
        _cfg(device_cards=_cfg().tiles_side + 1, **KW).validate()
    if _n_devices() < _cfg().tiles_side:
        with pytest.raises(ValueError, match="jax devices"):
            _cfg(device_cards=_n_devices() + 1, **KW).validate()
    assert _cfg(device_cards=1, **KW).validate() is True
