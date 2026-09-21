"""The pooled P(k) card: bitwise the serial one, and cheap enough to be it.

The card was the one streamed-paint consumer nobody pooled. Pooling it is a
WALL knob and must move no number -- the workers return bounded sub-blocks and
the parent accumulates, so integer associativity makes any arrival order
bitwise the serial order, exactly as it does for the engine's own paint.

Two things here are about hero scale rather than about correctness, and both
are memory conditions the smoke config cannot fail on its own: a `paint_only`
pool must not allocate the three coarse force meshes (103.1 GB at c-hero) and
must not build a tile kernel per worker, and `close()` must not copy an
ADOPTED field back into private memory (754.6 GB at c-hero, on a 1026 GB node).
Both are pinned by inspection of what was allocated, not by a byte count the
smoke rung cannot make large.
"""

import numpy as np
import pytest

from inexor import engine, state, summary
from inexor.codec import T9Layout
from inexor.config import Cosmology
from inexor.executor import SharedAllocator, TilePool

L_BOX, N_PART, N_FINE, N_COARSE, N_TILE, B_FINE = 32.0, 32, 64, 16, 16, 8
FIELDS = ("off", "w", "occupancy", "brick_start", "vel_scale", "arena_bucket")


def _require_cpu_lane():
    """This file is the CPU lane, and it FAILS off it rather than skipping.

    `TilePool` voids a non-CPU parent by design and `cmd_card` calls
    `_require_cpu`, so every test here needs the same lane the card itself runs
    in. It does not skip: the bitwise gate is the entire basis for pooling the
    hero card, and a gate that goes inert on the node that runs the job is no
    gate -- gb 1011375 put this file in a GPU-backend process and spent 63 s
    emitting the same ValueError six times. The lane is a property of the
    PROCESS (`JAX_PLATFORMS` is read at backend init, so it cannot be set per
    test, same as `XLA_FLAGS` in conftest), which is why the remedy is the
    invocation and this is only here to name it.
    """
    import jax

    if jax.default_backend() != "cpu":
        raise RuntimeError(
            f"tests/test_card_pool.py is the CPU lane and this process is "
            f"{jax.default_backend()!r}: TilePool voids a non-CPU parent and "
            f"cmd_card calls _require_cpu. Run it as `env JAX_PLATFORMS=cpu "
            f"pixi run -e gpu python -m pytest tests/test_card_pool.py`."
        )


_require_cpu_lane()


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _cfg(**kw):
    return engine.EngineConfig(
        box_size=L_BOX, n_part=N_PART, n_fine=N_FINE, n_coarse=N_COARSE,
        n_tile=N_TILE, b_fine=B_FINE, **kw
    )


def _state(cfg, seed=3, steps=3, alloc=None):
    """A state that has been through the exchange, so the arena is populated --
    arena residents are the rows a chunked decode can silently drop."""
    rng = np.random.default_rng(seed)
    g = (np.arange(N_PART) + 0.5) * (L_BOX / N_PART)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    x = np.mod(q + rng.normal(scale=0.25 * L_BOX / N_PART, size=q.shape), L_BOX)
    v = rng.normal(scale=0.5, size=x.shape)
    st = state.SlotState.build(
        x, v, T9Layout(L_BOX, N_PART, 2), N_FINE // cfg.n_brick,
        brick_slack=0.10, arena_frac=0.05,
    )
    for _ in range(steps):
        state.drift_and_migrate(st, 0.5)
    if alloc is not None:
        # where the loader would have put the payload: `load_checkpoint(alloc=)`
        # writes into the allocator's segments, which is what makes the pool's
        # adoption zero-copy. Done AFTER the exchange, since migrate reallocates.
        for f in FIELDS:
            a = np.asarray(getattr(st, f))
            view = alloc.empty(a.shape, a.dtype, f)
            view[...] = a
            setattr(st, f, view)
        st._invalidate_arena_index()
    return st


def _edges(cfg):
    return np.linspace(0.0, 0.5 * np.pi * cfg.n_coarse / cfg.box_size, 5)


def _card(st, cfg, pool=None):
    return summary.pk_summary_card(st, cfg, Cosmology(), a_out=1.0,
                                   edges=_edges(cfg), min_weight=20.0, pool=pool)


def test_the_pooled_card_is_bitwise_the_serial_card():
    """The gate. Not `allclose`: the accumulation is integer and the claim is
    that arrival order cannot reach the answer."""
    serial = _card(_state(_cfg(tile_workers=1)), _cfg(tile_workers=1))
    assert serial["n_bins"] > 0, "vacuous: the serial card must have measured something"

    cfg = _cfg(tile_workers=2)
    alloc = SharedAllocator()
    st = _state(cfg, alloc=alloc)
    pool = TilePool(st, cfg, allocator=alloc, paint_only=True)
    # THE KNOB MUST PROVE IT APPLIED. Two identical cards are exactly what a
    # dropped `pool=` produces, so count the dispatch: without this the test
    # passes when the card never reaches the pool at all.
    dispatched = []
    real_imap = pool.imap_coarse
    pool.imap_coarse = lambda tasks: real_imap(dispatched.append(len(tasks)) or tasks)
    try:
        pooled = _card(st, cfg, pool=pool)
    finally:
        pool.close()
    assert dispatched and dispatched[0] > 0, "the card did not dispatch to the pool"

    for key in ("p", "p_oracle", "z_profile", "k_mean", "n_modes",
                "window_correction", "shot_fraction"):
        np.testing.assert_array_equal(
            np.asarray(pooled[key]), np.asarray(serial[key]),
            err_msg=f"{key} moved under pooling",
        )
    assert pooled["k_nonlinear"] == serial["k_nonlinear"]


def test_the_streamed_mesh_itself_is_bitwise_under_pooling():
    """One level below the card, so a failure says WHERE. The mesh is the
    integer accumulation; the card is everything downstream of it."""
    cfg1 = _cfg(tile_workers=1)
    s_serial = {}
    d_serial = engine.coarse_delta_streamed(_state(cfg1), cfg1, stats=s_serial)

    cfg = _cfg(tile_workers=2)
    alloc = SharedAllocator()
    st = _state(cfg, alloc=alloc)
    pool = TilePool(st, cfg, allocator=alloc, paint_only=True)
    s_pooled = {}
    try:
        d_pooled = engine.coarse_delta_streamed(st, cfg, pool=pool, stats=s_pooled)
    finally:
        pool.close()
    # THE KNOB MUST PROVE IT APPLIED. Without this the test passes just as
    # happily when `pool=` is dropped on the floor and both arms run serial,
    # which is the one failure it is here to catch.
    assert s_pooled["coarse_pooled_workers"] == 2
    assert s_serial["coarse_pooled_workers"] == 0
    assert s_pooled["coarse_subblock_chunks"] == s_serial["coarse_subblock_chunks"] > 0
    np.testing.assert_array_equal(d_pooled, d_serial)


def test_a_paint_only_pool_allocates_no_coarse_force_mesh():
    """103.1 GB of it at c-hero, for a consumer that computes no force. Read
    off what was SHARED, not off a total: at this rung the meshes are 32 KB and
    any byte-count bar would pass with them present."""
    cfg = _cfg(tile_workers=2)
    alloc = SharedAllocator()
    st = _state(cfg, alloc=alloc)
    pool = TilePool(st, cfg, allocator=alloc, paint_only=True)
    try:
        assert not [k for k in pool._names if k.startswith("g")]
        assert "coarse force g0,g1,g2" not in pool._shm_demand
    finally:
        pool.close()

    full = TilePool(st, cfg, allocator=alloc, paint_only=False)
    try:
        assert sorted(k for k in full._names if k.startswith("g")) == ["g0", "g1", "g2"]
        assert "coarse force g0,g1,g2" in full._shm_demand
    finally:
        full.close()


def test_the_force_entry_points_refuse_on_a_paint_only_pool():
    """A mesh that was never allocated must not surface as a KeyError three
    frames down, and a paint-only worker must not be handed a tile."""
    cfg = _cfg(tile_workers=2)
    alloc = SharedAllocator()
    st = _state(cfg, alloc=alloc)
    pool = TilePool(st, cfg, allocator=alloc, paint_only=True)
    try:
        with pytest.raises(RuntimeError, match="paint_only"):
            pool.g_views()
        with pytest.raises(RuntimeError, match="paint_only"):
            pool.stage_step([], {})
        with pytest.raises(RuntimeError, match="paint_only"):
            list(pool.imap([]))
    finally:
        pool.close()


def test_close_leaves_an_adopted_field_in_the_allocators_segment():
    """`close()` copied EVERY field back into private memory, including the
    ones the allocator owns and `close()` deliberately does not unlink. At
    c-hero that is a second 754.6 GB state against a 1026 GB node. The adopted
    fields must come back as the same shared arrays, with their values intact.
    """
    cfg = _cfg(tile_workers=2)
    alloc = SharedAllocator()
    st = _state(cfg, alloc=alloc)
    before = {f: np.asarray(getattr(st, f)).copy() for f in FIELDS}
    adopted = [f for f in FIELDS
               if alloc.segment_of(np.asarray(getattr(st, f))) is not None]
    assert adopted, "vacuous: the allocator placed none of these fields"

    pool = TilePool(st, cfg, allocator=alloc, paint_only=True)
    pool.close()

    for f in adopted:
        arr = np.asarray(getattr(st, f))
        assert alloc.segment_of(arr) is not None, (
            f"{f} was copied out of the allocator's segment by close()")
        np.testing.assert_array_equal(arr, before[f])


def test_close_still_privatises_what_the_pool_itself_shared():
    """The other half of the contract, and the reason the copy-back exists:
    with no allocator every field IS one of the pool's own segments, which
    close() unlinks, so those must be private afterwards or they are views
    into freed memory."""
    cfg = _cfg(tile_workers=2)
    st = _state(cfg)
    before = {f: np.asarray(getattr(st, f)).copy() for f in FIELDS}

    pool = TilePool(st, cfg, paint_only=True)
    pool.close()

    for f in FIELDS:
        arr = np.asarray(getattr(st, f))
        np.testing.assert_array_equal(arr, before[f])
        # it must still be usable after the segments are gone
        assert arr.sum() == before[f].sum()
    # and the state still decodes, which is what "usable" means here
    _, x, _ = st.decode_bricks(list(range(min(4, st.n_bricks))))
    assert np.isfinite(x).all()
