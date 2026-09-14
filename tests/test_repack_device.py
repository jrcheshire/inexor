"""The device repack must be BITWISE the host `SlotState.repack`, or it is not adoptable.

Every state array (spare and freed rows included) and the host's return dict are
compared exactly. Anti-vacuity first: the arena is populated before the repack
(the fold-in is the part with no fast path), and at least one slab's new range
overruns a later slab's old range (the read-ahead is what makes the slab order
safe) -- a fixture without them would pass a driver that reads stale bytes.
"""

import copy

import numpy as np
import pytest

from inexor import state
from inexor.codec import T9Layout

jax = pytest.importorskip("jax")

FIELDS = ("off", "w", "occupancy", "brick_start", "vel_scale", "arena_bucket", "ids")
STATS = ("slots_used", "slots_per_particle", "bricks_fast", "bricks_merged")


def _state(n_part=32, nb=8, box=16.0, seed=5, brick_slack=0.0, arena_frac=0.25,
           with_ids=True, alloc_margin=0.5):
    rng = np.random.default_rng(seed)
    n = n_part**3
    x = rng.uniform(0.0, box, size=(n, 3))
    v = rng.normal(scale=1.0, size=(n, 3))
    t9 = T9Layout(box, n_part, 2)
    return state.SlotState.build(x, v, t9, nb, brick_slack=brick_slack,
                                 arena_frac=arena_frac, with_ids=with_ids,
                                 alloc_margin=alloc_margin)


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
    assert a.arena_base == b.arena_base, where
    assert a.occupancy.dtype == b.occupancy.dtype, where


def _same_stats(ra, rb, where):
    for k in STATS:
        assert ra[k] == rb[k], f"{where}: {k}: host {ra[k]} vs device {rb[k]}"


@pytest.fixture(scope="module")
def x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _device(st, brick_slack):
    from inexor.device.repack import repack_device

    out = repack_device(st, brick_slack=brick_slack)
    receipt = out.pop("repack_device")
    assert receipt["slabs"] == int(st.bricks_per_side)
    return out, receipt


# ------------------------------------------------------------------ the gate


def test_the_device_repack_is_bitwise_the_host_one_with_a_populated_arena(x64):
    st_a = _state()
    c = _c_drift(st_a, 1.3)
    for _ in range(2):
        state.drift_and_migrate(st_a, c)
    st_b = copy.deepcopy(st_a)
    assert st_a.arena_used > 0, "VACUOUS: no resident to fold in"
    old_starts = np.asarray(st_a.brick_start).copy()
    r_a = st_a.repack(brick_slack=0.10)
    r_b, receipt = _device(st_b, 0.10)
    _same_state(st_a, st_b, "repack")
    _same_stats(r_a, r_b, "repack")
    assert r_a["bricks_merged"] > 0 and r_a["bricks_fast"] > 0, "VACUOUS: one path only"
    nb2 = int(st_a.bricks_per_side) ** 2
    new = np.asarray(st_a.brick_start)
    overruns = sum(int(new[(s + 1) * nb2] > old_starts[(s + 1) * nb2])
                   for s in range(int(st_a.bricks_per_side) - 1))
    assert overruns > 0, "VACUOUS: no slab's new range overruns the next slab's old one"
    assert receipt["readahead_uploads"] > 0, "the read-ahead never fired on an overrun"
    assert receipt["windows_peak"] >= 2
    assert st_b.check() is True


def test_repeated_migrate_and_repack_steps_stay_bitwise_without_ids(x64):
    st_a = _state(seed=7, brick_slack=0.05, arena_frac=0.2, with_ids=False)
    st_b = copy.deepcopy(st_a)
    c = _c_drift(st_a, 1.1)
    folded = 0
    for step in range(3):
        state.drift_and_migrate(st_a, c)
        state.drift_and_migrate(st_b, c)
        folded += st_a.arena_used
        r_a = st_a.repack(brick_slack=0.05)
        r_b, _ = _device(st_b, 0.05)
        _same_state(st_a, st_b, f"step {step}")
        _same_stats(r_a, r_b, f"step {step}")
        assert st_b.arena_used == 0
    assert st_b.ids is None
    assert folded > 0, "VACUOUS: the arena was never populated across the steps"


def test_the_device_repack_proves_it_ran_and_reports_its_receipt(x64):
    from inexor.device import repack

    st = _state(n_part=16, nb=4, box=8.0)
    before = repack.CALLS
    out = repack.repack_device(st, brick_slack=0.1)
    assert repack.CALLS - before == 1
    rc = out["repack_device"]
    assert rc["slabs"] == 4 and rc["programs"] > 0 and rc["held_peak_bytes"] > 0
    assert out["scratch_bytes"] > 0
    assert st.check() is True


def test_the_geometry_helper_is_the_host_repacks_arithmetic(x64):
    from inexor.device.repack import repack_geometry

    st = _state(seed=11)
    state.drift_and_migrate(st, _c_drift(st, 1.3))
    _, _, new_start, n_alloc = repack_geometry(st, 0.10)
    r = st.repack(brick_slack=0.10)
    assert np.array_equal(new_start, st.brick_start)
    assert n_alloc == r["slots_used"] == st.arena_base


def test_a_brick_over_the_index_ceiling_refuses_before_writing(x64):
    from inexor.device.repack import repack_device

    st = _state(n_part=16, nb=2, box=8.0)
    before = copy.deepcopy(st)
    st.occupancy = st.occupancy.astype(np.uint8)  # a ceiling of 255 rows per brick
    with pytest.raises(ValueError, match="index ceiling"):
        repack_device(st, brick_slack=0.1)
    assert np.array_equal(before.off, st.off) and np.array_equal(before.w, st.w)


def _plant_low_bucket_residents(st, n_plant=3):
    """Residents in a bucket BELOW live rows of their brick.

    The engine's own migrate never makes one: an overflowing insert spills the
    TAIL of its bucket-sorted rows, so residents always sit at or above every
    live row's bucket and the live-row term `n_res(b, bucket < w)` is zero on
    every state the fixtures above can reach. The container allows it (the host
    repack folds any resident in), so plant some and keep the census honest.
    """
    p3 = int(st.buckets_per_brick)
    run = st.occupancy.reshape(st.n_bricks, p3).astype(np.int64)
    # bricks with live rows above their lowest bucket
    cands = np.flatnonzero(run[:, 1:].sum(axis=1) > 0)
    assert len(cands) >= n_plant, "fixture: no brick with rows above its lowest bucket"
    planted = 0
    for b in cands[:n_plant]:
        dest = np.array([int(b) * p3], dtype=np.int64)
        st._to_arena(dest, np.array([[7, 8, 9]], dtype=np.uint8),
                     np.array([[-3, 2, 1]], dtype=np.int16),
                     None if st.ids is None else np.array([10**6 + planted], dtype=np.int32))
        st.n_particles += 1
        planted += 1
    return planted


def test_a_resident_below_its_bricks_live_rows_is_folded_in_bitwise(x64):
    st_a = _state(seed=13)
    c = _c_drift(st_a, 1.3)
    state.drift_and_migrate(st_a, c)
    st_b = copy.deepcopy(st_a)
    n_a = _plant_low_bucket_residents(st_a)
    n_b = _plant_low_bucket_residents(st_b)
    assert n_a == n_b > 0
    _same_state(st_a, st_b, "planted")
    p3 = int(st_a.buckets_per_brick)
    live_above = 0
    for a in np.flatnonzero(st_a.arena_bucket >= 0):
        bk = int(st_a.arena_bucket[a])
        b = bk // p3
        live_above += int(st_a.occupancy[bk + 1:(b + 1) * p3].astype(np.int64).sum())
    assert live_above > 0, "VACUOUS: no live row sits above a resident's bucket"
    r_a = st_a.repack(brick_slack=0.10)
    r_b, _ = _device(st_b, 0.10)
    _same_state(st_a, st_b, "repack")
    _same_stats(r_a, r_b, "repack")
    assert st_b.check() is True
