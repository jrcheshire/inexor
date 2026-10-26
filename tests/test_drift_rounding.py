"""Stochastic drift rounding: exact across implementations, unbiased, layout-independent.

Covered: the numpy and jax dither hashes agree bitwise and are uniform and uncorrelated across
axes, keys and neighbouring lattice points; a stochastic eject is bitwise between the numpy,
jax and device kernels; a drift below half a quantum moves no particle under nearest rounding
but moves them unbiasedly under stochastic rounding, with the variance E[f(1 - f)] it must
have; the result is the same particles at the same lattice points under a different brick
layout; the engine config refuses unknown modes, numbers drifts uniquely and fingerprints the
mode only when stochastic.
"""
import copy

import numpy as np
import pytest

from inexor import engine, state
from inexor.codec import T9Layout

jax = pytest.importorskip("jax")


def _state(n_part=32, nb=4, box=16.0, seed=3, brick_slack=0.0, arena_frac=0.25, perm=None):
    rng = np.random.default_rng(seed)
    n = n_part**3
    x = rng.uniform(0.0, box, size=(n, 3))
    v = rng.normal(scale=1.0, size=(n, 3))
    if perm is not None:
        x, v = x[perm], v[perm]
    t9 = T9Layout(box, n_part, 2)
    return state.SlotState.build(x, v, t9, nb, brick_slack=brick_slack,
                                 arena_frac=arena_frac, with_ids=True)


def _c_drift(st, fraction):
    extent = float(st.t9.box_size) / int(st.bricks_per_side)
    s_max = float(np.max(st.vel_scale))
    return fraction * extent / (s_max * float(np.iinfo(np.int16).max))


def _by_id(st):
    """(lattice index (n, 3) int64, velocity (n, 3)) ordered by particle id."""
    idx, vel, ids = [], [], []
    q = float(st.t9.quantum)
    for b in range(st.n_bricks):
        slots, x, v = st.decode_brick(b)
        if not len(slots):
            continue
        idx.append(np.rint(x / q).astype(np.int64))
        vel.append(v)
        ids.append(st.ids[slots])
    ids = np.concatenate(ids)
    order = np.argsort(ids)
    assert np.array_equal(ids[order], np.arange(len(ids))), "ids are not a permutation"
    return np.concatenate(idx)[order], np.concatenate(vel)[order]


@pytest.fixture(scope="module")
def x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.mark.parametrize("key", [0, 1, 239, 2**32 + 120])
def test_numpy_and_jax_dither_are_bitwise(x64, key):
    from inexor.eject_jax import _dither, drift_key

    rng = np.random.default_rng(key % 97)
    i = rng.integers(0, 2**20, size=(50_000, 3))
    a = state.dither(i, key)
    b = np.asarray(_dither(jax.numpy.asarray(i), drift_key(key)))
    assert a.dtype == b.dtype == np.float64
    assert int(np.count_nonzero(a != b)) == 0


def test_dither_is_uniform_and_uncorrelated():
    rng = np.random.default_rng(0)
    n = 1_000_000
    i = rng.integers(0, 2**14, size=(n, 3))
    u = state.dither(i, 7)
    assert u.min() >= 0.0 and u.max() < 1.0
    tol = 5 / np.sqrt(n)
    assert abs(u.mean() - 0.5) < tol * np.sqrt(1 / 12)
    assert abs(u.var() - 1 / 12) < tol * np.sqrt(1 / 180)
    others = [u[:, 1], u[:, 2], state.dither(i, 8)[:, 0], state.dither(i + [1, 0, 0], 7)[:, 0]]
    for other in others:
        assert abs(np.corrcoef(u[:, 0], other)[0, 1]) < tol


@pytest.mark.parametrize("fraction", [0.95, 2.85])
def test_stochastic_eject_is_bitwise_between_kernels(x64, fraction):
    st = _state()
    c = _c_drift(st, fraction)
    scales = np.array(st.vel_scale, dtype=np.float64, copy=True)
    bx = int(st.bricks_per_side) // 2
    a_keep, a_emig = copy.deepcopy(st)._eject_slab(bx, c, scales, dither_key=5)
    b_keep, b_emig = copy.deepcopy(st)._eject_slab(bx, c, scales, kernel="jax", dither_key=5)
    n_keep, n_emig = len(a_keep["dest"]), len(a_emig["dest"])
    assert n_keep > 0 and n_emig > 0, "the partition is untested"
    for a, b in ((a_keep, b_keep), (a_emig, b_emig)):
        for k in a:
            if a[k] is None:
                assert b[k] is None
                continue
            assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape
            assert int(np.count_nonzero(a[k] != b[k])) == 0, k
    # anti-vacuity: the key reaches the kernel
    c_keep, c_emig = copy.deepcopy(st)._eject_slab(bx, c, scales, kernel="jax", dither_key=6)
    assert len(c_emig["dest"]) != n_emig or np.any(c_keep["off"] != b_keep["off"])


def test_stochastic_migration_is_bitwise_on_the_device_kernel(x64):
    from inexor.device.migrate import drift_and_migrate_device

    st = _state(nb=8)
    c = _c_drift(st, 1.4)
    a, b = copy.deepcopy(st), copy.deepcopy(st)
    state.drift_and_migrate(a, c, kernel="numpy", dither_key=11)
    drift_and_migrate_device(b, c, dither_key=11)
    for name in ("off", "w", "occupancy", "brick_start", "vel_scale", "arena_bucket", "ids"):
        x, y = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        assert int(np.count_nonzero(x != y)) == 0, name


def test_a_sub_half_quantum_drift_moves_unbiasedly(x64):
    st = _state()
    q = float(st.t9.quantum)
    i0, v = _by_id(st)
    # every |dx| below q / 2: nearest rounding moves nobody
    c = 0.45 * q / float(np.max(np.abs(v)))
    dx = c * v / q
    n_levels = int(st.t9.n_levels)

    def moved(dither_key):
        s = copy.deepcopy(st)
        state.drift_and_migrate(s, c, kernel="jax", dither_key=dither_key)
        i1, _ = _by_id(s)
        return (i1 - i0 + n_levels // 2) % n_levels - n_levels // 2

    assert int(np.count_nonzero(moved(None))) == 0
    d = moved(3)
    err = d - dx
    n = err.size
    assert np.all(np.abs(err) < 1.0)
    assert abs(err.mean()) < 5 * err.std() / np.sqrt(n)
    f = np.abs(dx)
    assert abs(err.var() - np.mean(f * (1 - f))) < 5 * np.sqrt(2 / n) * 0.25


def test_stochastic_rounding_does_not_depend_on_the_layout(x64):
    """Same particles fed in another order, with other brick slack and arena: other slots and
    row order. Bricks stay the same, since per-brick velocity scales move the drift itself."""
    perm = np.random.default_rng(9).permutation(32**3)
    a = _state(brick_slack=0.0)
    b = _state(brick_slack=0.3, arena_frac=0.1, perm=perm)
    assert not np.array_equal(a.ids, b.ids), "the fixtures must differ in row order"
    ia, va = _by_id(a)
    ib, vb = _by_id(b)
    # b's particle j is a's particle perm[j]
    assert np.array_equal(ia[perm], ib) and np.array_equal(va[perm], vb)
    c = _c_drift(a, 0.9)
    state.drift_and_migrate(a, c, kernel="jax", dither_key=21)
    state.drift_and_migrate(b, c, kernel="jax", dither_key=21)
    ia, _ = _by_id(a)
    ib, _ = _by_id(b)
    assert int(np.count_nonzero(ia[perm] != ib)) == 0


def _cfg(**kw):
    return engine.EngineConfig(box_size=16.0, n_part=16, n_fine=64, n_coarse=16, n_tile=16,
                               b_fine=8, **kw)


def test_config_default_is_stochastic_and_unknown_modes_refuse():
    assert _cfg().drift_rounding == "stochastic"
    with pytest.raises(ValueError, match="drift_rounding"):
        _cfg(drift_rounding="round").validate()


def test_drift_keys_are_unique_and_off_under_nearest():
    cfg = _cfg()
    keys = [engine._dither_key(cfg, 0)]
    keys += [engine._dither_key(cfg, k + 1) for k in range(4096)]
    keys += [engine._dither_key(cfg, k + 1 + 2**32) for k in range(4096)]
    assert len(set(keys)) == len(keys)
    assert engine._dither_key(_cfg(drift_rounding="nearest"), 5) is None


def test_fingerprint_names_the_mode_only_when_stochastic():
    co = np.ones((4, 3))
    near = _cfg(drift_rounding="nearest")
    legacy = _cfg(drift_rounding="nearest")
    del legacy.drift_rounding
    assert engine.checkpoint_fingerprint(near, co) == engine.checkpoint_fingerprint(legacy, co)
    assert engine.checkpoint_fingerprint(_cfg(), co) != engine.checkpoint_fingerprint(near, co)
