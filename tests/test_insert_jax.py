"""The compiled insert must be BITWISE the numpy one, or it is not adoptable.

`_insert_slab` fixes every brick's velocity scale, re-rounds every row's codes,
orders every run and claims the arena, so a compiled twin that is merely close
moves the state's physical layout. Every assertion here is exact equality.

Anti-vacuity is asserted before any comparison: particles crossing more than one
brick (reach > 1), bricks overflowing into the arena, and arena residents being
re-homed on a later step. A fixture without them would pass a broken grouping,
a broken spill order or a broken arena replay.
"""

import copy

import numpy as np
import pytest

from inexor import state
from inexor.codec import T9Layout

jax = pytest.importorskip("jax")

FIELDS = ("off", "w", "occupancy", "brick_start", "vel_scale", "arena_bucket", "ids")


def _state(n_part=32, nb=8, box=16.0, seed=5, brick_slack=0.0, arena_frac=0.25):
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


# ------------------------------------------------------------------ the gate


@pytest.mark.parametrize("eject_kernel", ["numpy", "jax"])
def test_two_migrations_through_the_compiled_insert_are_bitwise(x64, eject_kernel):
    st_a, st_b = _state(), _state()
    c = _c_drift(st_a, 1.9)
    overflow, residents_rehomed, reach = 0, 0, 0
    for step in range(2):
        residents_rehomed += st_a.arena_used
        r_a = state.drift_and_migrate(st_a, c)
        r_b = state.drift_and_migrate(st_b, c, kernel=eject_kernel, insert_kernel="jax")
        overflow += r_a["n_arena_overflow"]
        reach = max(reach, r_a["brick_reach_realized"])
        assert r_a == r_b, f"step {step}: migration stats differ: {r_a} vs {r_b}"
        _same_state(st_a, st_b, f"step {step}")
    assert reach >= 2, f"VACUOUS: realized reach {reach}; no row crossed two bricks"
    assert r_a["brick_reach"] < int(st_a.bricks_per_side) // 2, \
        "VACUOUS: the schedule was all-to-all"
    assert overflow > 0, "VACUOUS: no brick overflowed, so the spill order is untested"
    assert residents_rehomed > 0, "VACUOUS: no arena resident was re-homed"


def test_one_slab_insert_is_bitwise_including_its_arena_claims(x64):
    """Per slab, so a disagreement names the slab; staged from numpy ejects."""
    st = _state()
    c = _c_drift(st, 1.9)
    state.drift_and_migrate(st, c)  # arena residents for this pass to release
    nb = int(st.bricks_per_side)
    scales = np.array(st.vel_scale, dtype=np.float64, copy=True)
    r = min(state.brick_reach(st, c, scales), nb // 2)
    reach = range(-r, r + 1)
    staged, emig = {}, {}
    for s in range(nb):
        staged[s], emig[s] = st._eject_slab(s, c, scales)
    spilled = 0
    for bx in range(nb):
        a, b = copy.deepcopy(st), copy.deepcopy(st)
        ca, cb = dict.fromkeys(range(nb), 0), dict.fromkeys(range(nb), 0)
        na = a._insert_slab(bx, staged, emig, reach, ca, scales=scales)
        nb_ = b._insert_slab(bx, staged, emig, reach, cb, scales=scales, kernel="jax")
        assert na == nb_ and ca == cb, f"slab {bx}: overflow {na} vs {nb_}, census differs"
        _same_state(a, b, f"slab {bx}")
        spilled += na
    assert spilled > 0, "VACUOUS: no slab spilled into the arena"


def test_the_padding_path_is_exercised_and_bitwise(x64):
    from inexor import insert_jax

    insert_jax._CACHE.clear()
    st_a, st_b = _state(), _state()
    c = _c_drift(st_a, 1.9)
    n_slab = int(st_a.occupancy.astype(np.int64).sum()) // int(st_a.bricks_per_side)
    assert insert_jax._padded(n_slab) != n_slab, "VACUOUS: the rows need no padding"
    state.drift_and_migrate(st_a, c)
    state.drift_and_migrate(st_b, c, insert_kernel="jax")
    _same_state(st_a, st_b, "padded")
    insert_jax._CACHE.clear()


def test_the_sort_key_sees_rows_on_both_sides_of_the_slab(x64):
    """ANTI-VACUITY for the narrowed key: the sentinel path must be exercised.

    The key is slab-relative and every row bound elsewhere collapses onto ONE
    sentinel above the in-slab range. If a fixture's immigrant buffer only ever
    held rows for the slab being written, that collapse would never run and the
    narrowing would be untested -- so assert the buffer carries destinations
    BELOW `lo_b` and ABOVE `hi_b`, including the periodic wrap at slab 0.
    """
    st = _state()
    c = _c_drift(st, 1.9)
    nb = int(st.bricks_per_side)
    p3 = int(st.buckets_per_brick)
    scales = np.array(st.vel_scale, dtype=np.float64, copy=True)
    r = min(state.brick_reach(st, c, scales), nb // 2)
    staged, emig = {}, {}
    for s in range(nb):
        staged[s], emig[s] = st._eject_slab(s, c, scales)
    below = above = 0
    for bx in range(nb):
        lo_b, hi_b = st.slab_bricks(bx)
        sources = sorted({(bx + o) % nb for o in range(-r, r + 1)})
        dest = np.concatenate([emig[s]["dest"] for s in sources if len(emig[s]["dest"])])
        brick = dest // p3
        below += int((brick < lo_b).sum())
        above += int((brick >= hi_b).sum())
    assert below > 0, "VACUOUS: no immigrant was bound below its slab"
    assert above > 0, "VACUOUS: no immigrant was bound above its slab"


def test_the_narrow_key_guard_refuses_shapes_that_would_wrap():
    """The guarded branch is unrunnable, so pin the DECISION instead.

    Tripping the wide path needs `nb2 * p3 >= 2**32`, whose occupancy array alone
    is 34 GB, so no test can exercise it -- which is exactly why the predicate is
    separate and tested here. At `nb2 * p3 == 2**32` the sentinel wraps to 0 and
    out-of-slab rows sort FIRST, measured on a scratch reproduction.
    """
    from inexor.insert_jax import narrow_key_ok

    assert narrow_key_ok(512, 65536, 300_000_000), "production 4096^3 must narrow"
    assert narrow_key_ok(512, 1024, 5_000_000), "production cgh64 must narrow"
    assert not narrow_key_ok(512, 2**23, 1024), "nb2 * p3 == 2**32 must fall back"
    assert not narrow_key_ok(512, 2**24, 1024), "past the ceiling must fall back"
    assert not narrow_key_ok(512, 1024, 2**31), "n_pad at the int32 ceiling must fall back"
    assert narrow_key_ok(512, 1024, 2**31 - 1), "just under it must still narrow"


# ------------------------------------------------- the contracts around it


def test_the_compiled_insert_proves_it_ran(x64):
    from inexor import insert_jax

    st = _state(n_part=16, nb=4, box=8.0)
    before = insert_jax.CALLS
    state.drift_and_migrate(st, _c_drift(st, 0.95), insert_kernel="jax")
    assert insert_jax.CALLS - before == int(st.bricks_per_side), \
        "insert_kernel='jax' did not route every slab's insert through insert_jax"


def test_an_int16_escape_is_refused_before_any_write(x64, monkeypatch):
    from inexor import insert_jax

    real = insert_jax.insert_rows

    def escaping(*a, **k):
        out = real(*a, **k)
        out["abs_max"] = 40000.0
        return out

    monkeypatch.setattr(insert_jax, "insert_rows", escaping)
    st = _state(n_part=16, nb=4, box=8.0)
    w_before = st.w.copy()
    with pytest.raises(ValueError, match="escapes int16"):
        state.drift_and_migrate(st, _c_drift(st, 0.95), insert_kernel="jax")
    assert np.array_equal(st.w, w_before), "codes were written before the refusal"


def test_unknown_insert_kernel_refuses(x64):
    st = _state(n_part=16, nb=4, box=8.0)
    with pytest.raises(ValueError, match="unknown insert kernel"):
        state.drift_and_migrate(st, _c_drift(st, 0.95), insert_kernel="cuda-please")


def test_x64_off_is_refused():
    from inexor import insert_jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="x64"):
            insert_jax.insert_rows(np.zeros(1, np.int64), np.zeros((1, 3), np.uint8),
                                   np.zeros((1, 3), np.int16), None, np.ones(1), 0,
                                   np.array([0, 1]), 8)
    finally:
        jax.config.update("jax_enable_x64", prev)
