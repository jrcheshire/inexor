"""The IC generator on the cards (`icgen.generate_t9_slabs_device`).

The host generator is the oracle and stays untouched. What is exact here is gated
bitwise (card counts, the card noise stream on the CPU backend); what differs by
roundoff from the host generator is gated in code units. Card counts above the
backend's device count replicate handles; run with
`--xla_force_host_platform_device_count=4` for a true multi-device gate.
"""

import os

import jax
import numpy as np
import pytest

from inexor import ic, icgen, plan
from inexor.config import Cosmology

N, L, NB, A_INIT, SLAB = 32, 32.0, 4, 0.1, 5
PAYLOAD = ("brick_start", "occupancy", "off", "w", "vel_scale")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _devices(w):
    devs = jax.devices()
    return [devs[i % len(devs)] for i in range(w)]


def _gen(tmp_path, name, fn=icgen.generate_t9_slabs_device, seed=0, **kw):
    d = str(tmp_path / name)
    man = fn(d, jax.random.PRNGKey(seed), N, L, Cosmology(), A_INIT, NB, slab=SLAB, **kw)
    return man, icgen.load_slot_state(d), d


def _same_payload(a, b):
    return all(np.array_equal(getattr(a, k), getattr(b, k)) for k in PAYLOAD)


def test_the_device_generation_loads_checks_and_says_what_it_is(tmp_path):
    man, st, d = _gen(tmp_path, "g", noise="host", devices=_devices(1))
    st.check()
    assert st.n_particles == N**3 and st.arena_used == 0
    assert man["generator"] == "device" and man["n_devices"] == 1
    assert man["ic_stream"] == ic.IC_STREAM
    assert man["mean_phi2"] is None
    assert set(man["stage_s"]) == {"delta", "source", "source_forward", "velocities",
                                   "displacements", "emission"}
    assert man["stage_cleanup"]["dir_removed"]
    assert not os.path.isdir(os.path.join(d, icgen.STAGE_DIR))
    assert st.vel_scale.min() < st.vel_scale.max()


@pytest.mark.parametrize("f_NL", [0.0, 10.0])
def test_card_emission_is_bitwise_host_emission_in_the_generator(tmp_path, f_NL):
    if jax.devices()[0].platform != "cpu":
        pytest.skip("a CPU-backend gate")
    mh, host, _ = _gen(tmp_path, "host", f_NL=f_NL, noise="device", devices=_devices(4),
                       emission="host")
    mc, cards, _ = _gen(tmp_path, "cards", f_NL=f_NL, noise="device", devices=_devices(4))
    assert mh["emission"] == "host" and mc["emission"] == "cards"
    assert set(mc["emission_s"]) == {"upload_s", "source_s", "dest_s", "write_s"}
    assert _same_payload(host, cards)


def test_card_noise_carries_its_own_stream_tag(tmp_path):
    man, _st, _d = _gen(tmp_path, "g", noise="device", devices=_devices(1))
    assert man["ic_stream"] == ic.IC_STREAM_DEVICE != ic.IC_STREAM


@pytest.mark.parametrize("f_NL", [0.0, 10.0])
def test_four_cards_are_bitwise_one_card(tmp_path, f_NL):
    _m1, one, _ = _gen(tmp_path, "one", f_NL=f_NL, noise="device", devices=_devices(1))
    _m4, four, _ = _gen(tmp_path, "four", f_NL=f_NL, noise="device", devices=_devices(4))
    assert _same_payload(one, four)


def test_card_noise_is_the_cpu_stream_on_the_cpu_backend(tmp_path):
    if jax.devices()[0].platform != "cpu":
        pytest.skip("the card stream is a different stream off the CPU backend")
    _m, host_noise, _ = _gen(tmp_path, "h", noise="host", devices=_devices(1))
    _m, card_noise, _ = _gen(tmp_path, "c", noise="device", devices=_devices(1))
    assert _same_payload(host_noise, card_noise)


def test_a_different_seed_is_a_different_realization(tmp_path):
    _m, a, _ = _gen(tmp_path, "a", noise="device", devices=_devices(1))
    _m, b, _ = _gen(tmp_path, "b", seed=1, noise="device", devices=_devices(1))
    assert not np.array_equal(a.off, b.off)


# Bars from the floors measured at this fixture on four forced host devices: float32 moved at most 1 position code and ~200 of 120,033
# velocity codes, each by exactly 1, with per-brick scales at <= 5.1e-7 relative;
# float64 moved no code and scales at <= 5.6e-16. A code bar of 1 is a quantization
# statement (roundoff can move a rounding by one code, never two); the scale bars are
# the measured floor with 2x headroom.
SCALE_REL_BAR = {np.float32: 1.0e-6, np.float64: 1.2e-15}


@pytest.mark.parametrize("dt", [np.float32, np.float64])
@pytest.mark.parametrize("f_NL", [0.0, 10.0])
def test_the_device_generator_matches_the_host_generator_in_code_units(tmp_path, dt, f_NL):
    if jax.devices()[0].platform != "cpu":
        pytest.skip("these bars are CPU-backend floors")
    _mh, host, _ = _gen(tmp_path, "host", fn=icgen.generate_t9_slabs, fdtype=dt, f_NL=f_NL)
    _md, dev, _ = _gen(tmp_path, "dev", fdtype=dt, f_NL=f_NL, noise="host",
                       devices=_devices(4))
    assert np.array_equal(dev.brick_start, host.brick_start)
    assert np.array_equal(dev.occupancy, host.occupancy), (
        "a particle changed bucket; at this fixture none sits within roundoff of an edge")
    doff = np.abs(dev.off.astype(np.int64) - host.off.astype(np.int64))
    dw = np.abs(dev.w.astype(np.int64) - host.w.astype(np.int64))
    assert doff.max() <= 1 and dw.max() <= 1
    rel = np.max(np.abs(dev.vel_scale / host.vel_scale - 1.0))
    assert rel <= SCALE_REL_BAR[dt], f"per-brick scales moved {rel:.3e} relative"
    if dt is np.float64:
        assert doff.max() == 0 and dw.max() == 0
    else:
        assert 0 < int((dw > 0).sum()) < 0.01 * dw.size, (
            "no velocity code moved at float32, so this fixture cannot see roundoff and "
            "the code bar above is vacuous")


def test_the_parity_gate_can_fail(tmp_path):
    """Anti-vacuity: the host generator on another seed must break the code bar."""
    _mh, host, _ = _gen(tmp_path, "host", fn=icgen.generate_t9_slabs, fdtype=np.float32)
    _md, dev, _ = _gen(tmp_path, "dev", seed=1, fdtype=np.float32, noise="host",
                       devices=_devices(1))
    assert not np.array_equal(dev.occupancy, host.occupancy)


def test_refusals(tmp_path):
    with pytest.raises(ValueError, match="noise"):
        _gen(tmp_path, "x", noise="gpu")
    with pytest.raises(ValueError, match="multiple of the card count"):
        icgen.generate_t9_slabs_device(str(tmp_path / "y"), jax.random.PRNGKey(0), N, L,
                                       Cosmology(), A_INIT, NB, devices=_devices(3))
    jax.config.update("jax_enable_x64", False)
    with pytest.raises(RuntimeError, match="x64"):
        _gen(tmp_path, "z")


def test_host_allocation_is_what_the_planner_charges():
    """numpy's peak over a WARMED generation vs `plan.ic_device_stages`' host column.

    Measured on four forced host devices: 1.06x at 128^3, 1.02x at 256^3 (tracemalloc
    sees numpy, not jax arrays, which is the planner's host column); bar [0.9, 1.2]. RSS
    is not the quantity: on CPU devices it also holds the "cards".
    """
    import tempfile
    import tracemalloc

    n, nb = 128, 8
    devs = _devices(4)

    def once():
        icgen.generate_t9_slabs_device(tempfile.mkdtemp(), jax.random.PRNGKey(0), n, n / 2,
                                       Cosmology(), A_INIT, nb, slab=32, devices=devs)

    once()  # compile every program at these shapes before counting
    tracemalloc.start()
    try:
        once()
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    host, _card, _disk = plan.ic_device_stages(n, n_gpus=4, nb=nb, slab=32)
    ratio = peak / max(host.values())
    assert 0.9 <= ratio <= 1.2, f"numpy peak is {ratio:.2f}x the planner's host column"


def test_the_planner_stage_table_prices_4096_on_a_gb_node():
    host, card, disk = plan.ic_device_stages(4096, n_gpus=4, nb=256)
    field = 4096**3 * 4
    # the design: at most three full-size arrays on the host; on a card, a quarter field
    # through the transforms and, in card emission, the u_x halo shard plus the source
    # window and the larger emission program (charged, not measured: the gb emit-shape
    # leg reads it)
    assert 3 * field <= max(host.values()) < 3.3 * field
    halo = (256 // 4 + 2) * 16 * 4096**2 * 4
    assert card["6 emission"] >= halo + 3 * 16 * 4096**2 * plan.EMIT_KEPT_B_PER_ROW
    assert max(card[k] for k in card if not k.startswith("6")) < 0.27 * field
    assert disk["velocity staging (3 fields)"] == 3 * field
    assert max(host.values()) < 1026e9 and max(card.values()) < 0.9 * 199e9
    host_h, card_h, _ = plan.ic_device_stages(4096, n_gpus=4, nb=256, emission="host")
    assert card_h["6 emission"] < card["6 emission"] and host_h["6 emission"] > host["6 emission"]
