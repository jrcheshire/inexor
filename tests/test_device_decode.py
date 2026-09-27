"""The device tile decode, gated bitwise against the host decode.

The decode is elementwise integer arithmetic and one multiply (no reduction, scatter or
transcendental), so bit equality holds across backends and no tolerance is needed. Arena tests
first assert a nonzero arena resident count: nothing lands in the arena until a migrate
overflows a brick, so a plain fixture would skip that branch. The host plan is also gated to
O(bricks + arena) size and work, since a host loop over rows would pass every value check.
"""

import numpy as np
import pytest

from inexor import state
from inexor.codec import T9Layout

pytest.importorskip("jax")

L_BOX = 8.0
N_PART = 16
BRICKS = 2


@pytest.fixture(autouse=True)
def _x64():
    """The decode is f64 on both sides; with x64 off jax would silently narrow to f32."""
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _t9():
    return T9Layout(box_size=L_BOX, n_part=N_PART, bucket_cells=2)


def _positions(seed=0):
    rng = np.random.default_rng(seed)
    g = (np.arange(N_PART) + 0.5) * (L_BOX / N_PART)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    return np.mod(q + rng.normal(scale=0.3 * L_BOX / N_PART, size=q.shape), L_BOX)


def _built(**kw):
    x = _positions()
    v = np.random.default_rng(1).normal(scale=2.0, size=(len(x), 3))
    return state.SlotState.build(x, v, _t9(), BRICKS, **kw)


def _with_arena_residents():
    """A state with arena residents: `brick_slack=0.0` forces migrants into the arena.

    The caller asserts the resident count.
    """
    st = _built(brick_slack=0.0, arena_frac=0.30)
    state.drift_and_migrate(st, 0.35)
    return st


def _host(st, bricks):
    slots, x, v = st.decode_bricks(bricks)
    bor = np.repeat(np.asarray(bricks, np.int64),
                    [st.brick_member_count(b) for b in bricks])
    return slots, x, v, bor


def _device(st, bricks, cap):
    from inexor.device import decode as dev

    plan = dev.tile_decode_plan(st, bricks)
    out = dev.decode_rows(plan, st.off, st.w, st.vel_scale, st.arena_bucket,
                          st.arena_base, _t9(), BRICKS, cap)
    return plan, out


def _all_bricks():
    return list(range(BRICKS**3))


# ------------------------------------------------------------- the bitwise gate


def test_decode_is_bitwise_the_host_decode_with_no_arena_residents():
    st = _built(arena_frac=0.25)
    bricks = _all_bricks()
    assert sum(len(st.arena_slots_of_brick(b)) for b in bricks) == 0, \
        "this fixture is meant to be the no-arena path"
    slots, x, v, bor = _host(st, bricks)
    assert len(slots) > 0, "vacuous: nothing decoded"
    _plan, out = _device(st, bricks, len(slots) + 7)
    m = len(slots)
    assert np.array_equal(np.asarray(out["slots"])[:m], slots)
    assert np.array_equal(np.asarray(out["brick_of_row"])[:m], bor)
    assert np.array_equal(np.asarray(out["x"])[:m], x)
    assert np.array_equal(np.asarray(out["v"])[:m], v)


def test_decode_is_bitwise_the_host_decode_WITH_arena_residents():
    """The arena branch of the decode, with residents asserted present."""
    st = _with_arena_residents()
    bricks = _all_bricks()
    n_res = sum(len(st.arena_slots_of_brick(b)) for b in bricks)
    assert n_res > 0, (
        "VACUOUS: no brick overflowed into the arena, so the arena branch of "
        "the device decode did not run and this test proves nothing about it")
    slots, x, v, bor = _host(st, bricks)
    _plan, out = _device(st, bricks, len(slots) + 5)
    m = len(slots)
    assert np.array_equal(np.asarray(out["slots"])[:m], slots)
    assert np.array_equal(np.asarray(out["brick_of_row"])[:m], bor)
    assert np.array_equal(np.asarray(out["x"])[:m], x)
    assert np.array_equal(np.asarray(out["v"])[:m], v)


def test_the_bitwise_comparison_can_fail():
    """Control: one byte of `off` must move the decoded positions, or the gates above
    could be comparing two constants."""
    st = _built(arena_frac=0.25)
    bricks = _all_bricks()
    slots, x, _v, _bor = _host(st, bricks)
    _plan, out = _device(st, bricks, len(slots) + 3)
    before = np.asarray(out["x"])[: len(slots)].copy()
    st.off[slots[0], 0] = (int(st.off[slots[0], 0]) + 1) % 256
    _plan, out2 = _device(st, bricks, len(slots) + 3)
    assert not np.array_equal(np.asarray(out2["x"])[: len(slots)], before)


# ------------------------------------------------- the contracts around the values


def test_rows_stay_grouped_by_brick():
    """Each brick's rows are one contiguous run (order, not just the multiset).

    `tile_task` scans runs instead of sorting; a brick in two runs would get the wrong velocity
    scale.
    """
    st = _with_arena_residents()
    bricks = _all_bricks()
    assert sum(len(st.arena_slots_of_brick(b)) for b in bricks) > 0
    slots, _x, _v, _bor = _host(st, bricks)
    _plan, out = _device(st, bricks, len(slots) + 4)
    bor = np.asarray(out["brick_of_row"])[: len(slots)]
    runs = 1 + int(np.count_nonzero(np.diff(bor)))
    assert runs == len(np.unique(bor)), "a brick's rows are not contiguous"


def test_padding_is_marked_not_live_and_does_not_move_the_real_rows():
    st = _built(arena_frac=0.25)
    bricks = _all_bricks()
    slots, x, _v, _bor = _host(st, bricks)
    m = len(slots)
    _p, tight = _device(st, bricks, m)
    _p, padded = _device(st, bricks, m + 23)
    assert np.array_equal(np.asarray(tight["live"]), np.ones(m, bool))
    live = np.asarray(padded["live"])
    assert live[:m].all() and not live[m:].any()
    assert np.array_equal(np.asarray(padded["x"])[:m], np.asarray(tight["x"])[:m])


def test_every_output_has_the_padded_shape():
    """Every output has shape set by `cap`, so all tiles share one compiled XLA program."""
    st = _built(arena_frac=0.25)
    bricks = _all_bricks()
    cap = len(st.decode_bricks(bricks)[0]) + 11
    _plan, out = _device(st, bricks, cap)
    assert np.asarray(out["slots"]).shape == (cap,)
    assert np.asarray(out["x"]).shape == (cap, 3)
    assert np.asarray(out["v"]).shape == (cap, 3)
    assert np.asarray(out["brick_of_row"]).shape == (cap,)
    assert np.asarray(out["live"]).shape == (cap,)


def test_the_host_plan_is_not_per_particle():
    """The host plan's arrays are O(bricks + arena), never O(rows).

    A device decode fed by a host loop over rows would pass every value comparison here.
    """
    from inexor.device import decode as dev

    st = _with_arena_residents()
    bricks = _all_bricks()
    n_res = sum(len(st.arena_slots_of_brick(b)) for b in bricks)
    assert n_res > 0
    n_rows = int(len(st.decode_bricks(bricks)[0]))
    plan = dev.tile_decode_plan(st, bricks)
    assert n_rows > 4 * len(bricks), "vacuous: too few rows per brick to tell"
    budget = len(bricks) * (plan["p3"] + 4) + int(plan["arena_counts"].sum()) + 8
    for k, a in plan.items():
        if isinstance(a, np.ndarray):
            assert a.size <= budget, (
                f"plan['{k}'] has {a.size} entries against a budget of {budget}: "
                f"this is per-particle work on the host ({n_rows} rows)")


def test_the_host_plan_does_not_copy_the_whole_index():
    """The host plan's peak allocation is well under an int64 copy of the whole index.

    Widening the whole occupancy index before slicing passes the size gate above but costs
    8 B per bucket of the whole state (68.7 GB per chunk at 4096^3).
    """
    import tracemalloc

    from inexor.device import decode as dev

    n, box = 64, 32.0
    ax = (np.arange(n) + 0.5) * (box / n)
    x = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1).reshape(-1, 3)
    st = state.SlotState.build(x, np.zeros_like(x),
                               T9Layout(box_size=box, n_part=n, bucket_cells=2), 4)
    bricks = [0, 1]
    whole = st.n_buckets * 8
    assert whole >= 16 * len(bricks) * st.buckets_per_brick * 8, \
        "vacuous: the index is not much larger than the chunk's share of it"
    tracemalloc.start()
    tracemalloc.reset_peak()
    dev.tile_decode_plan(st, bricks)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < whole / 4, (
        f"tile_decode_plan allocated {peak:,} B for {len(bricks)} bricks against a "
        f"whole-index int64 copy of {whole:,} B: it is copying the state, not the chunk")


def test_x64_off_is_refused_rather_than_silently_narrowing_the_slots():
    """With x64 off the decode raises instead of narrowing slot indices to int32.

    Narrowing is invisible at test scale but wraps slots at production scale (8.3e10 slots, 38x
    past int32), so no runnable-size value test can catch it. Same contract as
    `eject_jax.require_x64`.
    """
    import jax

    st = _built(arena_frac=0.25)
    bricks = _all_bricks()
    cap = len(st.decode_bricks(bricks)[0]) + 2
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="jax_enable_x64"):
            _device(st, bricks, cap)
    finally:
        jax.config.update("jax_enable_x64", prev)


# ------------------------------------------------------- memory, by structure


def _largest_intermediate(jaxpr):
    """Element count of the largest equation output, sub-jaxprs included (inputs excluded)."""
    best = 0
    for eqn in jaxpr.eqns:
        for v in eqn.outvars:
            shape = getattr(v.aval, "shape", ())
            best = max(best, int(np.prod(shape)) if shape else 1)
        for p in eqn.params.values():
            for sub in p if isinstance(p, (tuple, list)) else (p,):
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    best = max(best, _largest_intermediate(inner))
    return best


def test_no_intermediate_scales_as_rows_times_buckets_per_brick():
    """The traced decode's largest intermediate is O(rows), not O(rows x buckets per brick).

    A (rows, 512) search table is invisible to value tests but ~100 GB per 4096^3 tile.
    """
    import jax

    from inexor.device import decode as dev

    st = _with_arena_residents()
    bricks = _all_bricks()
    assert sum(len(st.arena_slots_of_brick(b)) for b in bricks) > 0
    cap = len(st.decode_bricks(bricks)[0]) + 5
    plan = dev.tile_decode_plan(st, bricks)
    p3 = int(plan["p3"])
    bar = max(4 * cap, 2 * len(bricks) * p3)
    assert cap * p3 > 4 * bar, "vacuous: a rows x buckets table would not clear the bar"

    def fn(off, w):
        out = dev.decode_rows(plan, off, w, st.vel_scale, st.arena_bucket,
                              st.arena_base, _t9(), BRICKS, cap)
        return out["slots"], out["x"], out["v"]

    closed = jax.make_jaxpr(fn)(st.off, st.w)
    biggest = _largest_intermediate(closed.jaxpr)
    assert biggest <= bar, (
        f"the decode allocates a {biggest}-element intermediate against a bar of "
        f"{bar} ({cap} rows x {p3} buckets per brick = {cap * p3}): something "
        "scales as rows x buckets")
