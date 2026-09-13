"""D2d: the coarse paint on the device, gated BITWISE against the host paint.

The whole-mesh oracle is the real host code, `engine.coarse_delta_streamed`,
not a transcription. Bitwise is achievable for the same reason the host path's
own streamed-vs-monolithic pin is: positions decode bitwise, the paint kernel is
shared, and integer addition does not care about chunk size or order.

As in `test_device_decode.py`, the arena branch must actually run: every arena
test asserts a nonzero resident count first.
"""

import numpy as np
import pytest

pytest.importorskip("jax")

from inexor import engine, state  # noqa: E402
from inexor.codec import T9Layout  # noqa: E402
from inexor.device import paint as dpaint  # noqa: E402

# `tests/test_engine.py`'s validated smoke geometry, verbatim.
L_BOX, N_PART, N_FINE, N_COARSE, N_TILE, B_FINE = 32.0, 32, 64, 16, 16, 8


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _cfg(**kw):
    return engine.EngineConfig(box_size=L_BOX, n_part=N_PART, n_fine=N_FINE,
                               n_coarse=N_COARSE, n_tile=N_TILE, b_fine=B_FINE, **kw)


def _t9():
    return T9Layout(box_size=L_BOX, n_part=N_PART, bucket_cells=2)


def _state(cfg, seed=0, arena=False):
    rng = np.random.default_rng(seed)
    g = (np.arange(N_PART) + 0.5) * (L_BOX / N_PART)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    x = np.mod(q + rng.normal(scale=0.25 * L_BOX / N_PART, size=q.shape), L_BOX)
    v = np.random.default_rng(seed + 1).normal(scale=0.5, size=x.shape)
    nb = N_FINE // cfg.n_brick
    if not arena:
        return state.SlotState.build(x, v, _t9(), nb, arena_frac=0.05)
    # no spare in any brick, so the first migrate across a brick boundary has to
    # land in the arena
    st = state.SlotState.build(x, v, _t9(), nb, brick_slack=0.0, arena_frac=0.30)
    state.drift_and_migrate(st, 2.0)
    return st


def _nontrivial(delta):
    assert float(np.abs(delta).max()) > 0.1, "degenerate density; comparison is vacuous"


# ------------------------------------------------------------------ the gate


def test_device_density_is_bitwise_the_host_streamed_density():
    cfg = _cfg()
    st = _state(cfg, 2)
    want = engine.coarse_delta_streamed(st, cfg)
    s = {}
    got = dpaint.coarse_delta_device(st, cfg, stats=s, jit=False)
    assert s["coarse_device_jit"] is False
    L = dpaint.default_chunk_bricks(st.bricks_per_side)
    assert L == st.bricks_per_side**2 // 4, "the default is not a quarter-slab here"
    assert s["coarse_chunk_bricks"] == L, "the default chunk length did not apply"
    assert s["coarse_device_chunks"] == st.n_bricks // L, \
        "the device path did not paint every chunk"
    assert np.array_equal(got, want), "the device paint moved a bit of the density"
    _nontrivial(got)


def test_device_density_is_bitwise_WITH_arena_residents():
    """The branch the obvious fixture never reaches."""
    cfg = _cfg()
    st = _state(cfg, 4, arena=True)
    assert st.arena_used > 0, (
        "VACUOUS: no brick overflowed into the arena, so the arena rows of the "
        "window were never painted")
    want = engine.coarse_delta_streamed(st, cfg)
    got = dpaint.coarse_delta_device(st, cfg, jit=False)
    assert np.array_equal(got, want)
    _nontrivial(got)


def test_the_default_is_the_jitted_paint():
    """Adopted on the GB200 bitwise gate (Vista 993294); eager is `jit=False`."""
    cfg = _cfg()
    st = _state(cfg, 2)
    s = {}
    got = dpaint.coarse_delta_device(st, cfg, stats=s)
    assert s["coarse_device_jit"] is True and s["coarse_jit_traces"] <= 1
    assert np.array_equal(got, engine.coarse_delta_streamed(st, cfg))


def test_a_chunk_block_is_bitwise_the_host_decode_and_paint():
    """Per chunk, including arena residents: host `decode_bricks` padded and
    painted with the same kernel against the device window path."""
    import jax.numpy as jnp

    from inexor.painting import paint_tsc_int_subblock

    cfg = _cfg()
    st = _state(cfg, 4, arena=True)
    nb = st.bricks_per_side
    L = nb * nb
    gi = next(g for g in range(nb)
              if sum(len(st.arena_slots_of_brick(b)) for b in range(g * L, (g + 1) * L)))
    bricks = np.arange(gi * L, (gi + 1) * L, dtype=np.int64)
    assert sum(len(st.arena_slots_of_brick(int(b))) for b in bricks) > 0

    _, x, _ = st.decode_bricks(list(bricks))
    m = len(x)
    pad = m + 9
    xp = np.zeros((pad, 3))
    xp[:m] = x
    lv = np.zeros(pad, dtype=bool)
    lv[:m] = True
    origin, extent = dpaint.chunk_origin_extent(gi, L, nb, cfg.n_coarse)
    host = np.asarray(paint_tsc_int_subblock(
        jnp.asarray(xp), tuple(int(o) for o in origin), tuple(int(e) for e in extent),
        cfg.n_coarse, cfg.box_size, cfg.frac_bits, live=lv))

    guard = []
    sub, o2, e2 = dpaint.paint_chunk(st, bricks, gi, L, cfg, pad, guard)
    dpaint.check_containment(guard)
    assert np.array_equal(o2, origin) and np.array_equal(e2, extent)
    assert np.array_equal(np.asarray(sub), host)
    assert int(np.abs(host).sum()) > 0, "vacuous: empty block"


def test_the_comparison_can_fail():
    """One quantum of one particle's position must move the painted density."""
    cfg = _cfg()
    st = _state(cfg, 2)
    before = dpaint.coarse_delta_device(st, cfg)
    b = next(b for b in range(st.n_bricks) if st.brick_live_count(b) > 0)
    slot = int(st.brick_start[b])
    o = int(st.off[slot, 0])
    st.off[slot, 0] = o + 1 if o < 255 else o - 1
    after = dpaint.coarse_delta_device(st, cfg)
    assert not np.array_equal(before, after)


# ------------------------------------------------- the contracts around it


def test_chunk_size_does_not_move_a_bit_and_the_knob_applies():
    cfg = _cfg()
    st = _state(cfg, 4, arena=True)
    nb = st.bricks_per_side
    s_slab, s_row = {}, {}
    a = dpaint.coarse_delta_device(st, cfg, stats=s_slab, chunk_bricks=nb * nb)
    b = dpaint.coarse_delta_device(st, cfg, stats=s_row, chunk_bricks=nb)
    assert s_slab["coarse_chunk_bricks"] == nb * nb and s_row["coarse_chunk_bricks"] == nb
    assert s_row["coarse_device_chunks"] > s_slab["coarse_device_chunks"], \
        "the chunk-size knob did not change the number of chunks"
    assert np.array_equal(a, b)


def test_a_wrong_cuboid_is_refused_not_wrapped():
    """Chunk 0's bricks painted as though they were the far slab: every stencil
    lands outside the block, and the deferred guard must refuse."""
    cfg = _cfg()
    st = _state(cfg, 2)
    nb = st.bricks_per_side
    L = nb * nb
    bricks = np.arange(0, L, dtype=np.int64)
    m = len(st.decode_bricks(list(bricks))[0])
    guard = []
    dpaint.paint_chunk(st, bricks, nb // 2, L, cfg, m + 3, guard)
    assert guard, "no containment bounds were recorded"
    with pytest.raises(ValueError, match="containment violated"):
        dpaint.check_containment(guard)


def test_a_chunk_length_that_is_not_a_cuboid_tiling_is_refused():
    cfg = _cfg()
    st = _state(cfg, 2)
    with pytest.raises(ValueError, match="does not divide"):
        dpaint.coarse_delta_device(st, cfg, chunk_bricks=3)


def test_the_window_is_one_slice_plus_the_arena_residents():
    cfg = _cfg()
    st = _state(cfg, 4, arena=True)
    nb = st.bricks_per_side
    L = nb * nb
    for gi in range(nb):
        bricks = np.arange(gi * L, (gi + 1) * L, dtype=np.int64)
        plan, off, _ab, base = dpaint.slab_window(st, bricks)
        n_ar = sum(len(st.arena_slots_of_brick(int(b))) for b in bricks)
        span = int(st.brick_start[bricks[-1] + 1] - st.brick_start[bricks[0]])
        assert base == span
        assert off.shape[0] == span + n_ar
    with pytest.raises(ValueError, match="CONSECUTIVE"):
        dpaint.slab_window(st, np.array([0, 2, 3]))


def test_the_accumulator_is_a_seam():
    """A caller-supplied accumulator receives every painted chunk."""
    cfg = _cfg()
    st = _state(cfg, 2)

    class Recording(dpaint.HostInt64Accumulator):
        pass

    acc = Recording(cfg.n_coarse)
    s = {}
    got = dpaint.coarse_delta_device(st, cfg, stats=s, accumulator=acc)
    assert acc.chunks == s["coarse_device_chunks"] > 0
    assert np.array_equal(got, engine.coarse_delta_streamed(st, cfg))


# ------------------------------------------------------- the jitted chunk
#
# jit is adopted only on a bitwise gate: XLA may fuse the TSC weight arithmetic,
# and a fused multiply-add rounds differently, which could move a rounded
# integer weight. These gates run on whatever backend the suite runs on; the
# GPU reading is its own job.


@pytest.mark.parametrize("arena", [False, True])
def test_jit_density_is_bitwise_the_host_density(arena):
    cfg = _cfg()
    st = _state(cfg, 4, arena=arena)
    if arena:
        assert st.arena_used > 0, "VACUOUS: no arena residents"
    want = engine.coarse_delta_streamed(st, cfg)
    s = {}
    got = dpaint.coarse_delta_device(st, cfg, stats=s, jit=True)
    assert s["coarse_device_jit"] is True and s["coarse_device_chunks"] > 1
    assert np.array_equal(got, want), "the jitted paint moved a bit of the density"
    _nontrivial(got)


def test_every_jitted_chunk_block_is_bitwise_the_eager_block():
    """Per chunk, arena residents present, so a disagreement names the chunk."""
    from inexor.forces import capacity_shape

    cfg = _cfg()
    st = _state(cfg, 4, arena=True)
    assert st.arena_used > 0
    nb = st.bricks_per_side
    L = dpaint.default_chunk_bricks(nb)
    pad = int(capacity_shape(int(dpaint.chunk_rows(st, L).max())))
    shapes = dpaint.step_shapes(st, L, pad)
    compared = 0
    for gi in range(st.n_bricks // L):
        bricks = np.arange(gi * L, (gi + 1) * L, dtype=np.int64)
        g_e, g_j = [], []
        e = dpaint.paint_chunk(st, bricks, gi, L, cfg, pad, g_e)
        j = dpaint.paint_chunk(st, bricks, gi, L, cfg, pad, g_j, jit=True, shapes=shapes)
        if e is None:
            assert j is None
            continue
        dpaint.check_containment(g_e)
        dpaint.check_containment(g_j)
        assert np.array_equal(np.asarray(e[0]), np.asarray(j[0])), f"chunk {gi}"
        compared += 1
    assert compared > 1


def test_one_compilation_serves_every_chunk_of_a_step():
    cfg = _cfg()
    st = _state(cfg, 4, arena=True)
    dpaint._KERNELS.clear()
    s = {}
    dpaint.coarse_delta_device(st, cfg, stats=s, jit=True)
    assert s["coarse_device_chunks"] > 1, "vacuous: one chunk cannot show reuse"
    assert s["coarse_jit_traces"] == 1, (
        f"{s['coarse_jit_traces']} traces for {s['coarse_device_chunks']} chunks: "
        "a per-chunk shape is keying a new program")
    s2 = {}
    dpaint.coarse_delta_device(st, cfg, stats=s2, jit=True,
                               shape_floor=s["coarse_jit_shapes"])
    assert s2["coarse_jit_traces"] == 0, "a second step at the same shapes retraced"


@pytest.mark.parametrize("jit", [False, True])
def test_spreading_the_dead_rows_moves_no_bit(jit):
    """Padded rows scatter a zero weight; WHERE they scatter it cannot matter.
    On a jitted run the spread kernel must also be a distinct program, which is
    the receipt that the switch reached the compiled paint."""
    cfg = _cfg()
    st = _state(cfg, 4, arena=True)
    s0, s1 = {}, {}
    a = dpaint.coarse_delta_device(st, cfg, stats=s0, jit=jit)
    b = dpaint.coarse_delta_device(st, cfg, stats=s1, jit=jit, dead_rows="spread")
    assert s1["coarse_pad"] > s1["coarse_pad_true"], "vacuous: no padded rows"
    assert s0["coarse_dead_rows"] == "cell0" and s1["coarse_dead_rows"] == "spread"
    if jit:
        assert s1["coarse_jit_traces"] == 1, "the spread kernel reused the cell-0 program"
    assert np.array_equal(a, b)


def test_an_unknown_dead_row_mode_is_refused():
    import jax.numpy as jnp

    from inexor.painting import paint_tsc_int_subblock

    with pytest.raises(ValueError, match="dead_rows"):
        paint_tsc_int_subblock(jnp.zeros((4, 3)), (0, 0, 0), (4, 4, 4), 16, 16.0,
                               live=np.ones(4, bool), dead_rows="elsewhere")


def test_the_jitted_guard_refuses_a_wrong_cuboid():
    from inexor.forces import capacity_shape

    cfg = _cfg()
    st = _state(cfg, 2)
    nb = st.bricks_per_side
    L = dpaint.default_chunk_bricks(nb)
    pad = int(capacity_shape(int(dpaint.chunk_rows(st, L).max())))
    shapes = dpaint.step_shapes(st, L, pad)
    far = (st.n_bricks // L) // 2
    guard = []
    dpaint.paint_chunk(st, np.arange(0, L, dtype=np.int64), far, L, cfg, pad, guard,
                       jit=True, shapes=shapes)
    assert guard, "no containment bounds were recorded"
    with pytest.raises(ValueError, match="containment violated"):
        dpaint.check_containment(guard)


def test_a_chunk_past_the_steps_fixed_shapes_is_refused():
    from inexor.forces import capacity_shape

    cfg = _cfg()
    st = _state(cfg, 4, arena=True)
    L = dpaint.default_chunk_bricks(st.bricks_per_side)
    pad = int(capacity_shape(int(dpaint.chunk_rows(st, L).max())))
    shapes = dict(dpaint.step_shapes(st, L, pad), live_w=1)
    with pytest.raises(ValueError, match="exceeds the step's fixed shapes"):
        dpaint.slab_window_fixed(st, np.arange(0, L, dtype=np.int64), shapes)


# ----------------------------------------------- memory, by structure


def _nbytes(v):
    shape = getattr(v.aval, "shape", ())
    size = int(np.prod(shape)) if shape else 1
    return size * np.dtype(v.aval.dtype).itemsize


def _live_peak(jaxpr):
    """Peak bytes alive at once under last-use freeing, sub-jaxprs included.

    Inputs and lifted constants are not counted -- only what the program
    allocates. Literals are skipped when recording uses: they are not buffers.
    """
    last = {}
    for i, e in enumerate(jaxpr.eqns):
        for v in e.invars:
            if type(v).__name__ != "Literal":
                last[v] = i
    for v in jaxpr.outvars:
        if type(v).__name__ != "Literal":
            last[v] = len(jaxpr.eqns)
    alive, cur, peak = {}, 0, 0
    for i, e in enumerate(jaxpr.eqns):
        for v in e.outvars:
            alive[v] = _nbytes(v)
            cur += alive[v]
        inner = 0
        for p in e.params.values():
            for s in p if isinstance(p, (tuple, list)) else (p,):
                sub = getattr(s, "jaxpr", s)
                if hasattr(sub, "eqns"):
                    inner = max(inner, _live_peak(sub))
        peak = max(peak, cur + inner)
        for v in [v for v in alive if last.get(v, -1) <= i]:
            cur -= alive.pop(v)
    return peak


def test_a_chunk_stays_inside_the_traced_per_row_floor():
    """`plan.PAINT_CHUNK_TRACED_B_PER_ROW` is read off this program, so the program
    is held to it: re-trace one chunk at two padded row counts, difference out the
    fixed terms, and fail if the live bytes per row exceed the charge. The floor
    is the decoded positions (24) plus the three per-axis TSC weight arrays (72),
    which are alive together through the whole corner loop -- a reading below
    that means the liveness pass is broken, not that the paint got cheaper."""
    import jax

    from inexor import plan as planner
    from inexor.device.decode import decode_rows
    from inexor.painting import paint_tsc_int_subblock

    cfg = _cfg()
    st = _state(cfg, 4, arena=True)
    nb = st.bricks_per_side
    L = nb * nb
    p, off, ab, base = dpaint.slab_window(st, np.arange(L, dtype=np.int64))
    origin, extent = dpaint.chunk_origin_extent(0, L, nb, cfg.n_coarse)
    n = cfg.n_coarse

    def peak_at(pad):
        def fn(off_w):
            dec = decode_rows(p, off_w, None, None, ab, base, st.t9, nb, pad,
                              velocities=False)
            guard = []
            dpaint.containment_bounds(dec["x"], dec["live"], cfg.box_size / float(n),
                                      origin, extent, n, guard)
            return paint_tsc_int_subblock(
                dec["x"], tuple(int(o) for o in origin), tuple(int(e) for e in extent),
                n, cfg.box_size, cfg.frac_bits, live=dec["live"])
        return _live_peak(jax.make_jaxpr(fn)(off).jaxpr)

    p1 = p["n_rows"] + 9
    p2 = 4 * p1
    per_row = (peak_at(p2) - peak_at(p1)) / (p2 - p1)
    assert per_row >= 96, f"liveness reads {per_row:.1f} B/row, below positions + weights"
    assert per_row <= planner.PAINT_CHUNK_TRACED_B_PER_ROW, (
        f"one paint chunk's traced program now holds {per_row:.1f} B/row against "
        f"the recorded floor of {planner.PAINT_CHUNK_TRACED_B_PER_ROW}: the program "
        "grew, so the card-measured PAINT_CHUNK_B_PER_ROW is stale too -- re-measure")


def test_x64_off_is_refused():
    import jax

    cfg = _cfg()
    st = _state(cfg, 2)
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="jax_enable_x64"):
            dpaint.coarse_delta_device(st, cfg)
    finally:
        jax.config.update("jax_enable_x64", prev)
