"""The device migrate must be BITWISE the serial numpy migrate, or it is not adoptable.

Every state array (spare and freed rows included) and the whole stats dict are
compared exactly, after every step. Anti-vacuity is asserted first: rows crossing
two bricks on a schedule that is not all-to-all, bricks overflowing into the
arena, arena residents re-homed, and a claim landing on a row a release freed in
the same pass -- a fixture without them would pass a broken window, a broken
spill order or a broken replay.
"""

import copy

import numpy as np
import pytest

from inexor import state
from inexor.codec import T9Layout

jax = pytest.importorskip("jax")

FIELDS = ("off", "w", "occupancy", "brick_start", "vel_scale", "arena_bucket", "ids")


def _state(n_part=32, nb=8, box=16.0, seed=5, brick_slack=0.0, arena_frac=0.25, with_ids=True):
    rng = np.random.default_rng(seed)
    n = n_part**3
    x = rng.uniform(0.0, box, size=(n, 3))
    v = rng.normal(scale=1.0, size=(n, 3))
    t9 = T9Layout(box, n_part, 2)
    return state.SlotState.build(x, v, t9, nb, brick_slack=brick_slack,
                                 arena_frac=arena_frac, with_ids=with_ids)


def _c_drift(st, fraction):
    extent = float(st.t9.box_size) / int(st.bricks_per_side)
    s_max = float(np.max(st.vel_scale))
    return fraction * extent / (s_max * float(np.iinfo(np.int16).max))


def _same_state(a, b, where):
    for name in FIELDS:
        x, y = getattr(a, name), getattr(b, name)
        if x is None:
            assert y is None, f"{where}: {name} None on one side only"
            continue
        n_diff = int(np.count_nonzero(np.asarray(x) != np.asarray(y)))
        assert n_diff == 0, f"{where}: state.{name}: {n_diff} of {np.asarray(x).size} differ"


@pytest.fixture(scope="module")
def x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _device(st, c):
    from inexor.device.migrate import drift_and_migrate_device

    out = drift_and_migrate_device(st, c)
    receipt = out.pop("migrate_device")
    assert receipt["slabs"] == int(st.bricks_per_side)
    return out


# ------------------------------------------------------------------ the gate


def test_two_migrations_on_the_device_are_bitwise_the_serial_numpy_pass(x64, monkeypatch):
    st_a, st_b = _state(), _state()
    c = _c_drift(st_a, 1.9)

    released, reclaimed = set(), 0
    real_release, real_claim = state.SlotState._release_brick_arena, state.SlotState._to_arena

    def release(self, b):
        if self is st_b:
            released.update(int(x) for x in self.arena_slots_of_brick(b))
        return real_release(self, b)

    def claim(self, dest, off, w, ids=None):
        nonlocal reclaimed
        if self is st_b:
            free = self._arena_free
            if free is None:
                free = np.nonzero(self.arena_bucket < 0)[0]
            got = int(self.arena_base) + free[: len(dest)]
            reclaimed += sum(int(x) in released for x in got)
        return real_claim(self, dest, off, w, ids)

    monkeypatch.setattr(state.SlotState, "_release_brick_arena", release)
    monkeypatch.setattr(state.SlotState, "_to_arena", claim)

    overflow, rehomed, reach = 0, 0, 0
    for step in range(2):
        rehomed += st_a.arena_used
        r_a = state.drift_and_migrate(st_a, c)
        r_b = _device(st_b, c)
        overflow += r_a["n_arena_overflow"]
        reach = max(reach, r_a["brick_reach_realized"])
        assert r_a == r_b, f"step {step}: stats differ: {r_a} vs {r_b}"
        _same_state(st_a, st_b, f"step {step}")
    assert reach >= 2, f"VACUOUS: realized reach {reach}; no row crossed two bricks"
    assert r_a["brick_reach"] < int(st_a.bricks_per_side) // 2, "VACUOUS: all-to-all"
    assert overflow > 0, "VACUOUS: no brick overflowed, so the spill order is untested"
    assert rehomed > 0, "VACUOUS: no arena resident was re-homed"
    assert reclaimed > 0, "VACUOUS: no claim reused a row released in the same pass"


def test_migrations_with_a_repack_between_and_no_ids_are_bitwise(x64):
    st_a = _state(seed=7, brick_slack=0.05, arena_frac=0.2, with_ids=False)
    st_b = copy.deepcopy(st_a)
    c = _c_drift(st_a, 1.3)
    overflow = 0
    for step in range(3):
        r_a = state.drift_and_migrate(st_a, c)
        r_b = _device(st_b, c)
        overflow += r_a["n_arena_overflow"]
        assert r_a == r_b, f"step {step}: stats differ"
        _same_state(st_a, st_b, f"step {step} migrate")
        st_a.repack(brick_slack=0.05)
        st_b.repack(brick_slack=0.05)
        _same_state(st_a, st_b, f"step {step} repack")
    assert st_b.ids is None
    assert overflow > 0, "VACUOUS: no brick overflowed across the repacked steps"


def test_an_arena_full_pass_refuses_like_the_serial_one(x64):
    st_a = _state(seed=9, arena_frac=0.002)
    st_b = copy.deepcopy(st_a)
    c = _c_drift(st_a, 1.9)
    with pytest.raises(ValueError, match="does not clamp"):
        state.drift_and_migrate(st_a, c)
    with pytest.raises(ValueError, match="does not clamp"):
        _device(st_b, c)


def test_the_device_pass_proves_it_ran(x64):
    from inexor.device import migrate

    st = _state(n_part=16, nb=4, box=8.0)
    before = migrate.CALLS
    out = migrate.drift_and_migrate_device(st, _c_drift(st, 0.9))
    assert migrate.CALLS - before == 1
    assert out["migrate_device"]["slabs"] == 4 and out["migrate_device"]["programs"] > 0


def test_a_pass_over_the_device_budget_refuses_before_writing(x64):
    from inexor.device.migrate import drift_and_migrate_device

    st = _state()
    before = copy.deepcopy(st)
    with pytest.raises(ValueError, match="budget of"):
        drift_and_migrate_device(st, _c_drift(st, 1.9), device_budget_bytes=1024)
    _same_state(before, st, "after the refusal")


def test_the_budget_estimate_is_reported_and_a_generous_budget_is_bitwise(x64):
    from inexor.device.migrate import drift_and_migrate_device

    st_a, st_b = _state(), _state()
    c = _c_drift(st_a, 1.9)
    r_a = state.drift_and_migrate(st_a, c)
    r_b = drift_and_migrate_device(st_b, c, device_budget_bytes=10**12)
    receipt = r_b.pop("migrate_device")
    assert receipt["budget_bytes"] == 10**12
    assert 0 < receipt["peak_estimate_bytes"] < 10**12
    assert r_a == r_b
    _same_state(st_a, st_b, "generous budget")


def test_pass_arena_index_is_the_state_grouping(x64):
    from inexor.device.migrate import pass_arena_index

    st = _state()
    state.drift_and_migrate(st, _c_drift(st, 1.9))
    slots, bricks = pass_arena_index(st)
    assert len(slots) == st.arena_used > 0, "VACUOUS: no arena residents"
    st._arena_by_brick = None
    for b in np.unique(bricks):
        np.testing.assert_array_equal(slots[bricks == b], st.arena_slots_of_brick(int(b)))
