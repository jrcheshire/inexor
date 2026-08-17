"""The compiled eject path must be BITWISE the numpy one, or it is not adoptable.

D-007 forbids a saturating op anywhere near integer state, and the whole M-v2-3
gate rests on the migration producing the same bits from different orderings. A
compiled twin that is merely *close* would move the state's physical layout,
which is why every assertion here is exact equality and none is a tolerance.

The three properties, in the order they can fail:

1. `_eject_slab_jax` returns exactly what `_eject_slab` returns, keys, dtypes,
   shapes and values, on a state where BOTH keepers and leavers exist and the
   arena is occupied. Anti-vacuity is asserted, not assumed: a slab with no
   leavers reproduces trivially and would pass a broken partition.
2. A whole `drift_and_migrate` through the compiled path leaves the state
   elementwise identical, which is the property `_insert_slab` actually depends
   on -- (1) can hold while the ORDER inside a run differs.
3. The x64 guard fires. With x64 off the lattice index narrows silently, so the
   refusal is the only thing between that and wrong destinations.
"""

import copy

import numpy as np
import pytest

from inexor import state
from inexor.codec import T9Layout

jax = pytest.importorskip("jax")


def _state(n_part=32, nb=4, box=16.0, seed=3, brick_slack=0.0, arena_frac=0.25):
    rng = np.random.default_rng(seed)
    n = n_part**3
    x = rng.uniform(0.0, box, size=(n, 3))
    v = rng.normal(scale=1.0, size=(n, 3))
    t9 = T9Layout(box, n_part, 2)
    return state.SlotState.build(x, v, t9, nb, brick_slack=brick_slack,
                                 arena_frac=arena_frac, with_ids=True)


def _c_drift(st, fraction):
    extent = float(st.t9.box_size) / int(st.bricks_per_side)
    s_max = float(np.max(st.vel_scale))
    return fraction * extent / (s_max * float(np.iinfo(np.int16).max))


@pytest.fixture(scope="module")
def x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.mark.parametrize("fraction", [0.95, 2.85])
def test_eject_slab_jax_is_bitwise(x64, fraction):
    st = _state()
    c = _c_drift(st, fraction)
    scales = np.array(st.vel_scale, dtype=np.float64, copy=True)
    bx = int(st.bricks_per_side) // 2

    a_keep, a_emig = copy.deepcopy(st)._eject_slab(bx, c, scales)
    b_keep, b_emig = copy.deepcopy(st)._eject_slab(bx, c, scales, kernel="jax")

    # ANTI-VACUITY, before any comparison: a slab with no leavers, or no
    # keepers, reproduces under a broken partition
    assert len(a_keep["dest"]) > 0, "no keepers -- the partition is untested"
    assert len(a_emig["dest"]) > 0, "no leavers -- the partition is untested"

    assert set(a_keep) == set(b_keep)
    assert set(a_emig) == set(b_emig)
    for name, (a, b) in [("keep", (a_keep, b_keep)), ("emig", (a_emig, b_emig))]:
        for k in a:
            if a[k] is None:
                assert b[k] is None, f"{name}.{k}: None on one side only"
                continue
            assert a[k].dtype == b[k].dtype, f"{name}.{k} dtype {a[k].dtype} vs {b[k].dtype}"
            assert a[k].shape == b[k].shape, f"{name}.{k} shape {a[k].shape} vs {b[k].shape}"
            n_diff = int(np.count_nonzero(a[k] != b[k]))
            assert n_diff == 0, f"{name}.{k}: {n_diff} of {a[k].size} elements differ"


def test_arena_is_actually_exercised(x64):
    """The arena splice is where this function lost a particle before.

    `brick_slack=0.0` plus a drift large enough to overflow bricks is the recipe
    that puts residents in the arena; the assertion is that they are THERE, so
    the comparison above is not silently testing the empty-arena path.
    """
    st = _state(brick_slack=0.0)
    c = _c_drift(st, 2.85)
    state.drift_and_migrate(st, c)
    assert st.arena_used > 0, "the arena stayed empty -- the splice is untested"

    scales = np.array(st.vel_scale, dtype=np.float64, copy=True)
    hits = [b for b in range(int(st.n_bricks))
            if len(st.arena_slots_of_brick(b))]
    assert hits, "no brick has arena residents"
    bx = int(hits[0]) // (int(st.bricks_per_side) ** 2)
    a = copy.deepcopy(st)._eject_slab(bx, c, scales)
    b = copy.deepcopy(st)._eject_slab(bx, c, scales, kernel="jax")
    for i, name in enumerate(("keep", "emig")):
        for k in a[i]:
            if a[i][k] is None:
                continue
            assert np.array_equal(a[i][k], b[i][k]), f"{name}.{k} differs with the arena live"


def test_whole_migration_is_bitwise(x64):
    """The property `_insert_slab` depends on: the STATE, not just the slabs."""
    st_a, st_b = _state(), _state()
    c = _c_drift(st_a, 2.85)
    r_a = state.drift_and_migrate(st_a, c)
    r_b = state.drift_and_migrate(st_b, c, kernel="jax")

    assert r_a == r_b, f"migration stats differ: {r_a} vs {r_b}"
    for name in ("off", "w", "occupancy", "brick_start", "vel_scale", "arena_bucket", "ids"):
        a, b = getattr(st_a, name), getattr(st_b, name)
        if a is None:
            assert b is None
            continue
        n_diff = int(np.count_nonzero(np.asarray(a) != np.asarray(b)))
        assert n_diff == 0, f"state.{name}: {n_diff} of {np.asarray(a).size} elements differ"


def test_padding_path_is_exercised_and_bitwise(x64, monkeypatch):
    """The pad was DEAD in every other test here, and a mutation proved it.

    A slab of this fixture holds exactly 8,192 rows, which is a multiple of
    `PAD_MULTIPLE`, so `pad` was 0 everywhere and corrupting the padded rows'
    brick id changed nothing. Shrinking the multiple to a value that cannot
    divide the row count forces a real pad, which is the configuration
    production hits constantly (occupancy moves the row count every step).
    """
    from inexor import eject_jax

    monkeypatch.setattr(eject_jax, "PAD_MULTIPLE", 3000)
    eject_jax._CACHE.clear()

    st = _state()
    c = _c_drift(st, 2.85)
    scales = np.array(st.vel_scale, dtype=np.float64, copy=True)
    bx = int(st.bricks_per_side) // 2

    a_keep, a_emig = copy.deepcopy(st)._eject_slab(bx, c, scales)
    n_rows = len(a_keep["dest"]) + len(a_emig["dest"])
    assert eject_jax._padded(n_rows) != n_rows, (
        f"{n_rows} rows still needs no padding at PAD_MULTIPLE=3000 -- this test "
        "would be asserting the same thing as the others"
    )

    b_keep, b_emig = copy.deepcopy(st)._eject_slab(bx, c, scales, kernel="jax")
    for name, a, b in (("keep", a_keep, b_keep), ("emig", a_emig, b_emig)):
        for k in a:
            if a[k] is None:
                continue
            n_diff = int(np.count_nonzero(a[k] != b[k]))
            assert n_diff == 0, f"{name}.{k}: {n_diff} differ with a live pad"
    eject_jax._CACHE.clear()


def test_engine_default_is_jax_and_actually_routes(x64):
    """The DEFAULT must reach the compiled path, not merely name it.

    Flipping `EngineConfig.eject_kernel` to "jax" changed nothing that ran,
    because `v2_m6_phase_time.py` carried its own `eject_kernel="numpy"` default
    that shadowed the library's -- and the full suite stayed green through the
    flip, because nothing asserted the default was exercised. So this asserts
    both halves: the declared default, and a CALL COUNT proving a default-config
    engine step went through `eject_jax`.
    """
    from inexor import eject_jax, engine

    cfg = engine.EngineConfig(
        box_size=8.0, n_part=16, n_fine=32, n_coarse=16, n_tile=16, b_fine=8,
        alpha=0.5,
    )
    assert cfg.eject_kernel == "jax", "the engine default is no longer the compiled path"

    st = _state(n_part=16, nb=2, box=8.0)
    before = eject_jax.CALLS
    state.drift_and_migrate(st, _c_drift(st, 0.95), kernel=cfg.eject_kernel)
    assert eject_jax.CALLS > before, (
        "the default named 'jax' but eject_jax was never called -- a knob that "
        "does not prove it applied"
    )


def test_x64_guard_fires():
    """With x64 off the compiled path must REFUSE, not narrow silently."""
    from inexor import eject_jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="x64"):
            eject_jax.require_x64()
    finally:
        jax.config.update("jax_enable_x64", prev)
    # and the guard passes in the other direction, so the test is two-sided
    jax.config.update("jax_enable_x64", True)
    try:
        eject_jax.require_x64()
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_unknown_kernel_refuses(x64):
    st = _state(n_part=16, nb=2, box=8.0)
    scales = np.array(st.vel_scale, dtype=np.float64, copy=True)
    with pytest.raises(ValueError, match="unknown eject kernel"):
        st._eject_slab(0, _c_drift(st, 0.95), scales, kernel="cuda-please")
