"""D2a: the tile decode on the device, gated BITWISE against the host one.

Most device work in this project cannot carry a bitwise cross-backend gate --
`tests/conftest.py`'s `detflag` marker exists because CUDA's f32 scatter-add is
not reproducible. The decode can. It is elementwise integer arithmetic and one
multiply, with no reduction, no scatter and nothing transcendental, so there is
nothing for a GPU's ordering freedom to change. A tolerance here would be
hiding something rather than accommodating it.

Two things this file works hard at, because both have burned this project:

- **The arena branch must actually run.** `SlotState.build` allocates arena rows
  but nothing lands in them until a migrate overflows a brick, so the obvious
  fixture exercises the arena path zero times while looking like it covers it.
  Every arena test here asserts a nonzero resident count FIRST. (Measured while
  writing this: the natural fixture gave 0 residents and the gate passed.)
- **The host must not be doing per-particle work.** The whole point of the phase
  is deleting a per-particle host pass, so a test asserts the plan's arrays are
  O(bricks + arena) and never O(rows). A device decode fed by a host loop over
  rows would pass every value comparison here and defeat the purpose.
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
    """The decode is f64 on both sides; with x64 off jax narrows it to f32 and
    hands it back in an f64 container -- a wrong answer wearing the right
    dtype, and the comparison would be against the host's real f64."""
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
    """A state whose bricks have really overflowed into the arena.

    `brick_slack=0.0` leaves no spare in any brick, so the first migrate that
    moves a particle across a brick boundary has to put it in the arena. The
    caller asserts the count; this only sets it up.
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
    """The branch the obvious fixture never reaches."""
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
    """Anti-vacuity: one byte of `off` must move the decoded positions.

    Without this, every equality above could be comparing two constants.
    """
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
    """`tile_task` replaces a sort with a run scan and asserts this itself: if a

    brick appeared in two runs the second would overwrite the first's velocity
    scale and decode every row of it wrong. The device decode has to preserve
    the order, not merely the multiset.
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
    # the padding must not perturb the real rows: one XLA shape, same answer
    assert np.array_equal(np.asarray(padded["x"])[:m], np.asarray(tight["x"])[:m])


def test_every_output_has_the_padded_shape():
    """One shape per `cap`, which is why `cap` is padded at all: a per-tile row
    count keys a new XLA program, profiled at 74% of a step before the fix."""
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
    """The phase exists to DELETE a per-particle host pass. A device decode fed

    by a host loop over rows would pass every value comparison in this file and
    defeat the point, so the plan's own size is gated: O(bricks + arena), never
    O(rows).
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


def test_x64_off_is_refused_rather_than_silently_narrowing_the_slots():
    """The failure that is INVISIBLE at this file's own scale.

    With x64 off, `jnp.arange(cap, dtype=int64)` truncates to int32 and every
    slot index narrows with it. At n_part=16 that changes nothing and the
    bitwise gates above still pass; at c-hero the slot space is 8.3e10, 38x
    past int32, so slots wrap, the gather reads the wrong rows and nothing
    raises. Same contract and same reason as `eject_jax.require_x64`, and the
    reason it is a refusal rather than a note is that no test at a runnable
    size can catch it.
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
