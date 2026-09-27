"""The device migrate split across cards, gated BITWISE against the serial
numpy pass.

Every state array and the whole stats dict are compared after every pass, at 2, 3
(an uneven split) and 4 cards. Anti-vacuity: the drift spills into the arena and
emigrant segments actually cross cards.

The multi-card tests need that many jax devices and SKIP otherwise; on the laptop
run them under `XLA_FLAGS=--xla_force_host_platform_device_count=4`.
"""

import copy

import pytest

from inexor import state
from tests.test_migrate_device import _c_drift, _same_state, _state

jax = pytest.importorskip("jax")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _devices(w):
    if len(jax.devices()) < w:
        pytest.skip(f"needs {w} jax devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return list(jax.devices()[:w])


@pytest.mark.parametrize("w", [2, 3, 4])
def test_the_migrate_on_w_cards_is_bitwise_the_serial_numpy_pass(w):
    from inexor.device.migrate import drift_and_migrate_device

    devs = _devices(w)
    st_a = _state(n_part=64, nb=16, box=32.0)
    st_b = copy.deepcopy(st_a)
    c = _c_drift(st_a, 0.9)
    segments = spills = 0
    for step in range(2):
        r_a = state.drift_and_migrate(st_a, c)
        r_b = drift_and_migrate_device(st_b, c, devices=devs)
        rec = r_b.pop("migrate_device")
        _same_state(st_a, st_b, f"W={w} pass {step}")
        assert r_a == r_b, f"W={w} pass {step}: stats differ"
        assert rec["cards"] == w and rec["fallback"] is None
        assert sum(rec["slabs_per_card"]) == 16
        segments += rec["cross_card_segments"]
        spills += r_a["n_arena_overflow"]
        if step == 0:
            st_a.repack(brick_slack=0.0)
            st_b.repack(brick_slack=0.0)
    assert r_a["brick_reach"] >= 1
    assert spills > 0, "VACUOUS: nothing spilled into the arena"
    assert segments > 0, "VACUOUS: no emigrants crossed a card"


def test_a_card_too_narrow_for_the_reach_falls_back_to_one_card_bitwise():
    from inexor.device.migrate import drift_and_migrate_device

    devs = _devices(4)
    st_a = _state()  # 8 bricks per side: 2 slabs a card
    st_b = copy.deepcopy(st_a)
    c = _c_drift(st_a, 0.9)
    r_a = state.drift_and_migrate(st_a, c)
    r_b = drift_and_migrate_device(st_b, c, devices=devs)
    rec = r_b.pop("migrate_device")
    _same_state(st_a, st_b, "fallback")
    assert r_a == r_b
    assert rec["cards"] == 1 and "fewer than 2r + 1" in rec["fallback"]


def test_an_empty_device_list_is_refused():
    from inexor.device.migrate import drift_and_migrate_device

    st = _state()
    with pytest.raises(ValueError, match="empty"):
        drift_and_migrate_device(st, _c_drift(st, 0.9), devices=[])
