"""The pooled P(k) card is bitwise the serial card, and the pool stays cheap.

Workers return integer sub-blocks the parent accumulates, so arrival order cannot move a bit.
Two memory conditions matter only at scale and are pinned by inspecting what was allocated
(a byte count at this size could not fail): a `paint_only` pool allocates no coarse force
meshes, and `close()` does not copy an allocator-adopted field back into private memory.
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
    """Fail (not skip) at import off the CPU lane: `TilePool` and `cmd_card` require a CPU
    parent, and a bitwise gate must not go silently inert. The backend is fixed per process
    at init, so the remedy is the invocation named in the error.
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
    """A state after several exchanges, so the arena (rows a chunked decode could drop) is
    populated. With `alloc`, the fields are moved into the allocator's segments."""
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
        # as `load_checkpoint(alloc=)` would; after the exchange, since migrate reallocates
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
    """Every card output is bitwise equal pooled vs serial (integer accumulation)."""
    serial = _card(_state(_cfg(tile_workers=1)), _cfg(tile_workers=1))
    assert serial["n_bins"] > 0, "vacuous: the serial card must have measured something"

    cfg = _cfg(tile_workers=2)
    alloc = SharedAllocator()
    st = _state(cfg, alloc=alloc)
    pool = TilePool(st, cfg, allocator=alloc, paint_only=True)
    # count the dispatch: a dropped `pool=` would also give two identical cards
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
    """The streamed coarse mesh itself is bitwise under pooling, so a card failure localizes."""
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
    # the pooled arm must actually have pooled, or both arms ran serial
    assert s_pooled["coarse_pooled_workers"] == 2
    assert s_serial["coarse_pooled_workers"] == 0
    assert s_pooled["coarse_subblock_chunks"] == s_serial["coarse_subblock_chunks"] > 0
    np.testing.assert_array_equal(d_pooled, d_serial)


def test_a_paint_only_pool_allocates_no_coarse_force_mesh():
    """A paint_only pool shares no g0/g1/g2 meshes (a full pool does); checked by name, since
    the meshes are 32 KB here and a byte bar would pass with them present."""
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
    """Force entry points on a paint_only pool raise a clear RuntimeError."""
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
    """`close()` leaves allocator-owned fields in their shared segments, values intact; copying
    them back would double the state in host memory."""
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
    """With no allocator, close() unlinks the pool's own segments, so fields must be private
    copies afterwards, still usable and decodable. While open, each field is the pool's own
    view onto its segment (not owning its data); after close() each owns its data."""
    cfg = _cfg(tile_workers=2)
    st = _state(cfg)
    before = {f: np.asarray(getattr(st, f)).copy() for f in FIELDS}
    private = {f: getattr(st, f) for f in FIELDS}

    pool = TilePool(st, cfg, paint_only=True)
    try:
        for f in FIELDS:
            arr = getattr(st, f)
            assert arr is pool._views[f] and arr is not private[f], (
                f"{f} is not the pool's shared view while the pool is open")
            assert not arr.flags.owndata, f"{f} owns its data while shared"
    finally:
        pool.close()

    for f in FIELDS:
        arr = getattr(st, f)
        assert isinstance(arr, np.ndarray) and arr.flags.owndata and arr.base is None, (
            f"{f} still borrows its buffer after close(); the segment is unlinked")
        np.testing.assert_array_equal(arr, before[f])
        assert arr.sum() == before[f].sum()
    _, x, _ = st.decode_bricks(list(range(min(4, st.n_bricks))))
    assert np.isfinite(x).all()
