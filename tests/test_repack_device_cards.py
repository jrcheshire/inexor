"""The device repack split across cards, gated BITWISE against the host
`SlotState.repack`.

Every state array and the host's return dict are compared. Anti-vacuity: the arena
is populated (including residents planted below live rows), and some slab's old
range intersects a new range another card writes, so the cross-card early uploads
actually fire.

The multi-card tests need that many jax devices and SKIP otherwise; on the laptop
run them under `XLA_FLAGS=--xla_force_host_platform_device_count=4`.
"""

import copy

import pytest

from inexor import state
from tests.test_repack_device import (
    _c_drift,
    _plant_low_bucket_residents,
    _same_state,
    _same_stats,
    _state,
)

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
def test_the_repack_on_w_cards_is_bitwise_the_host_repack(w):
    from inexor.device.repack import repack_device

    devs = _devices(w)
    st_a = _state(n_part=64, nb=16, box=32.0)
    c = _c_drift(st_a, 1.3)
    for _ in range(2):
        state.drift_and_migrate(st_a, c)
    assert _plant_low_bucket_residents(st_a) > 0
    st_b = copy.deepcopy(st_a)
    assert st_a.arena_used > 0, "VACUOUS: no resident to fold in"

    r_a = st_a.repack(brick_slack=0.10)
    r_b = repack_device(st_b, brick_slack=0.10, devices=devs)
    rec = r_b.pop("repack_device")
    _same_state(st_a, st_b, f"W={w}")
    _same_stats(r_a, r_b, f"W={w}")
    assert rec["cards"] == w
    assert rec["cross_card_early_uploads"] > 0, (
        "VACUOUS: no slab's old range met another card's new range")
    assert st_b.check() is True


def test_repeated_migrate_and_repack_on_four_cards_stay_bitwise():
    from inexor.device.migrate import drift_and_migrate_device
    from inexor.device.repack import repack_device

    devs = _devices(4)
    st_a = _state(n_part=64, nb=16, box=32.0, seed=7, brick_slack=0.05, arena_frac=0.2,
                  with_ids=False)
    st_b = copy.deepcopy(st_a)
    c = _c_drift(st_a, 0.9)
    for step in range(3):
        r_a = state.drift_and_migrate(st_a, c)
        r_b = drift_and_migrate_device(st_b, c, devices=devs)
        r_b.pop("migrate_device")
        assert r_a == r_b, f"step {step}: migrate stats differ"
        p_a = st_a.repack(brick_slack=0.05)
        p_b = repack_device(st_b, brick_slack=0.05, devices=devs)
        p_b.pop("repack_device")
        _same_state(st_a, st_b, f"step {step}")
        _same_stats(p_a, p_b, f"step {step}")
