"""The fused device migrate + repack, gated BITWISE against the two passes.

Every state array (spare, freed and arena rows included), `arena_base`, the migrate's
stats and the repack's stats are compared against BOTH the host pair
(`state.drift_and_migrate` + `SlotState.repack`) and the device pair
(`drift_and_migrate_device` + `repack_device`), after every step. The census is the
host migrate's own per-brick membership (the tile-loop census is gated against it in
`test_device_census.py`). Anti-vacuity: spills, reach 2, read-ahead uploads, and
cross-card early uploads are asserted where a fixture exists to make them happen.
"""

import copy

import numpy as np
import pytest

from inexor import state
from inexor.codec import T9Layout
from inexor.device import repack as drepack
from tests.test_migrate_device import _c_drift, _same_state, _state

jax = pytest.importorskip("jax")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _roomy_state(seed=5):
    """64^3 on 16 bricks built with no slack and an 80% allocation margin, so a repack
    at 50% slack moves blocks past a card's boundary slab into slabs no boundary
    eject has read."""
    rng = np.random.default_rng(seed)
    n = 64**3
    x = rng.uniform(0.0, 32.0, size=(n, 3))
    v = rng.normal(scale=1.0, size=(n, 3))
    return state.SlotState.build(x, v, T9Layout(32.0, 64, 2), 16, brick_slack=0.0,
                                 alloc_margin=0.8, arena_frac=0.25, with_ids=True)


def _census(st, c):
    ref = copy.deepcopy(st)
    state.drift_and_migrate(ref, c)
    return drepack.repack_geometry(ref, 0.0)[1]


def _strip_m(stats):
    d = dict(stats)
    d.pop("migrate_device", None)
    return d


def _strip_r(stats):
    d = dict(stats)
    d.pop("repack_device", None)
    d.pop("scratch_bytes", None)
    return d


def _devices(w):
    if w is None:
        return None
    if len(jax.devices()) < w:
        pytest.skip(f"needs {w} jax devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return list(jax.devices()[:w])


def _three_way(st0, c, slack, steps, devices=None):
    from inexor.device.fused import migrate_repack_device
    from inexor.device.migrate import drift_and_migrate_device

    st_h, st_d, st_f = copy.deepcopy(st0), copy.deepcopy(st0), copy.deepcopy(st0)
    receipts = []
    for step in range(steps):
        census = _census(st_f, c)
        m_h = state.drift_and_migrate(st_h, c)
        r_h = st_h.repack(brick_slack=slack)
        m_d = drift_and_migrate_device(st_d, c, devices=devices)
        r_d = drepack.repack_device(st_d, brick_slack=slack, devices=devices)
        m_f, r_f = migrate_repack_device(st_f, c, census, brick_slack=slack, devices=devices)
        receipts.append((m_f["migrate_device"], r_f["repack_device"], m_h))
        where = f"step {step}"
        _same_state(st_h, st_f, f"{where}: fused vs host")
        _same_state(st_d, st_f, f"{where}: fused vs device")
        assert st_h.arena_base == st_d.arena_base == st_f.arena_base, where
        assert _strip_m(m_f) == _strip_m(m_d) == m_h, f"{where}: migrate stats differ"
        assert _strip_r(r_f) == _strip_r(r_d) == _strip_r(r_h), f"{where}: repack stats differ"
    return receipts


def test_two_fused_steps_are_bitwise_both_pairs():
    st = _state()
    rec = _three_way(st, _c_drift(st, 1.9), 0.10, 2)
    assert sum(m["n_arena_overflow"] for _mr, _rr, m in rec) > 0, "VACUOUS: nothing spilled"
    assert max(m["brick_reach_realized"] for _mr, _rr, m in rec) >= 2, "VACUOUS: reach < 2"
    assert all(mr["fused"] and rr["fused"] for mr, rr, _m in rec)
    assert all(rr["blocks"] > 0 for _mr, rr, _m in rec)


def test_no_ids_and_a_slack_change_are_bitwise():
    st = _state(seed=7, brick_slack=0.05, arena_frac=0.2, with_ids=False)
    _three_way(st, _c_drift(st, 1.3), 0.05, 3)


def test_a_growing_allocation_reads_ahead_and_stays_bitwise():
    # built with no slack and repacked at 10%: new ranges drift right by a growing
    # prefix, past the next slab by the last of 16
    st = _state(n_part=64, nb=16, box=32.0, brick_slack=0.0)
    rec = _three_way(st, _c_drift(st, 0.9), 0.10, 1)
    assert rec[0][1]["readahead_uploads"] > 0, "VACUOUS: no block overran a later slab"


@pytest.mark.parametrize("w", [2, 3, 4])
def test_fused_on_w_cards_is_bitwise_both_pairs(w):
    devs = _devices(w)
    st = _state(n_part=64, nb=16, box=32.0)
    rec = _three_way(st, _c_drift(st, 0.9), 0.10, 2, devices=devs)
    assert all(mr["cards"] == w for mr, _rr, _m in rec)
    assert sum(mr["cross_card_segments"] for mr, _rr, _m in rec) > 0, "VACUOUS: no hand-off"


def test_cross_card_early_uploads_happen_and_stay_bitwise():
    import time

    from inexor.device import migrate
    from inexor.device.fused import migrate_repack_device
    from inexor.ooc_fft import partition_units

    devs = _devices(4)
    st = _roomy_state()
    c = _c_drift(st, 0.9)
    rec = _three_way(st, c, 0.5, 1, devices=devs)
    # boundary slabs are ejected before any card writes, so these are deeper ones
    assert rec[0][1]["cross_card_early_uploads"] > 0, "VACUOUS: no cross-card overrun"

    # THE WORST INTERLEAVING, forced: the cards are threads, and left alone a card
    # ejects its deep slabs long before its neighbour writes over their old range,
    # so a pass without the early uploads would pass this test by timing. Delay
    # exactly those ejects until the neighbours have written.
    census = _census(st, c)
    new_start, _ = drepack.capacity_from_counts(st, census, 0.5)
    parts = partition_units(16, 4, 1)
    early = drepack._cross_card_slabs(np.asarray(st.brick_start, dtype=np.int64), new_start,
                                      parts, 256)
    deep = {t for e in early for t in e} - {s for lo, hi in parts for s in (lo, hi - 1)}
    assert deep, "VACUOUS: every overrun slab is a boundary slab"
    real = migrate._eject_unit

    def late(st_, u, *args, **kw):
        if u[0] in deep and kw.get("pre") is None:
            time.sleep(2.0)
        return real(st_, u, *args, **kw)

    st_h, st_f = copy.deepcopy(st), copy.deepcopy(st)
    state.drift_and_migrate(st_h, c)
    st_h.repack(brick_slack=0.5)
    migrate._eject_unit = late
    try:
        migrate_repack_device(st_f, c, census, brick_slack=0.5, devices=devs)
    finally:
        migrate._eject_unit = real
    _same_state(st_h, st_f, "fused, deep ejects delayed, vs host")


def test_a_wrong_census_refuses_before_writing():
    from inexor.device.fused import migrate_repack_device

    st = _state()
    c = _c_drift(st, 1.9)
    census = _census(st, c)
    # the first slab the one-card sweep inserts is slab r, so its block is the first
    # write: a wrong count there must stop the pass before anything reaches the host
    r = state.brick_reach(st, c)
    lo_b, hi_b = st.slab_bricks(r)
    b = lo_b + int(np.flatnonzero(census[lo_b:hi_b])[0])
    bad = census.copy()
    bad[b] -= 1
    bad[hi_b] += 1  # same total: only the per-brick check can see it
    before = copy.deepcopy(st)
    with pytest.raises(AssertionError, match=f"brick {b} holds"):
        migrate_repack_device(st, c, bad)
    _same_state(before, st, "after the refusal")


def test_an_arena_full_pass_refuses_like_the_two_passes():
    from inexor.device.fused import migrate_repack_device

    roomy = _state(seed=9, arena_frac=0.25)
    c = _c_drift(roomy, 1.9)
    census = _census(roomy, c)
    st = _state(seed=9, arena_frac=0.002)
    with pytest.raises(ValueError, match="does not clamp"):
        migrate_repack_device(st, c, census)


def test_the_fused_pass_does_not_cross_twice():
    from inexor.device import migrate
    from inexor.device.fused import migrate_repack_device

    st = _state()
    c = _c_drift(st, 1.9)
    census = _census(st, c)
    before, calls = dict(migrate.READS), drepack.CALLS
    migrate_repack_device(st, c, census)
    reads = {k: v - before.get(k, 0) for k, v in migrate.READS.items()}
    nb = int(st.bricks_per_side)
    assert reads.get("insert: slot range", 0) == 0, reads
    assert reads.get("insert: occupancy", 0) == 0, reads
    assert reads.get("fused: occupancy") == nb, reads
    assert drepack.CALLS == calls, "the device repack ran"


# ------------------------------------------------------------------ through the engine


def _engine_kw(**kw):
    return dict(coarse_backend="device", tile_backend="device", migrate_backend="device", **kw)


@pytest.mark.parametrize("cards", [1, 4])
def test_a_fused_engine_run_is_bitwise_the_separate_passes(cards):
    from inexor import engine
    from inexor.device import fused
    from tests.test_engine_device_backend import _coeffs, _same
    from tests.test_engine_device_backend import _state as _estate
    from tests.test_engine_device_step import _cfg, _same_stats

    if cards > 1:
        _devices(cards)
    co = _coeffs(3)
    cfg_s = _cfg(**_engine_kw(migrate_repack_fused=False, device_cards=cards))
    cfg_f = _cfg(**_engine_kw(device_cards=cards))
    assert cfg_f.fused_pass and not cfg_s.fused_pass
    st_s, st_f = _estate(cfg_s), _estate(cfg_f)
    out_s = engine.run(st_s, cfg_s, co)
    c0 = fused.CALLS
    out_f = engine.run(st_f, cfg_f, co)
    assert fused.CALLS - c0 == len(co), "the fused pass did not run every step"
    _same(st_s, st_f, "fused vs separate")
    _same_stats(out_s, out_f, drop=("migrate_repack_fused", "census_slabs", "coarse_jit_traces"))
    assert all(o["census_slabs"] == st_f.bricks_per_side for o in out_f)
    assert all(o["migrate_repack_fused"] for o in out_f)
    assert not any(o["migrate_repack_fused"] for o in out_s)
    assert all(o["repack"]["repack_device"].get("fused") for o in out_f)
    assert sum(o["n_arena_overflow"] for o in out_f) > 0, "VACUOUS: nothing spilled"


def test_validate_refuses_fused_where_it_cannot_apply():
    from tests.test_engine_device_step import _cfg

    for kw in (dict(migrate_backend="host"), dict(device_tile_window=False)):
        cfg = _cfg(**dict(_engine_kw(migrate_repack_fused=True), **kw))
        with pytest.raises(ValueError, match="migrate_repack_fused=True"):
            cfg.validate()
        assert not cfg.fused_pass
