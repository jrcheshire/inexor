"""The v2 engine core: BullFrog PM on T9 state in slot order (M-v2-3, S5)."""

import numpy as np
import pytest

from inexor import engine, forces, painting, state
from inexor.codec import T9Layout
from inexor.config import Cosmology
from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table, float_step_bullfrog

# The `smoke` rung of the v2 config table, which is the smallest geometry whose
# tile+buffer decomposition is not degenerate.
L_BOX, N_PART, N_FINE, N_COARSE, N_TILE, B_FINE = 32.0, 32, 64, 16, 16, 8
BRICKS = None  # derived from choose_brick


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


def _positions(seed=0):
    rng = np.random.default_rng(seed)
    g = (np.arange(N_PART) + 0.5) * (L_BOX / N_PART)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    return np.mod(q + rng.normal(scale=0.25 * L_BOX / N_PART, size=q.shape), L_BOX)


def _state(cfg, seed=0, **kw):
    x = _positions(seed)
    v = np.random.default_rng(seed + 1).normal(scale=0.5, size=x.shape)
    t9 = T9Layout(box_size=L_BOX, n_part=N_PART, bucket_cells=2)
    nb = N_FINE // cfg.n_brick
    return x, v, state.SlotState.build(x, v, t9, nb, arena_frac=0.05, **kw)


def test_the_geometry_validates():
    cfg = _cfg()
    assert cfg.validate() is True
    assert cfg.n_tile % cfg.n_brick == 0, "a tile core must be a whole number of bricks"


# ------------------------------------------------- the streamed coarse paint


def test_the_streamed_coarse_paint_is_bitwise_the_monolithic_one():
    """The property that makes streaming the long arm possible at all, and it is
    a consequence of the paint being INTEGER: integer addition is associative, so
    chunking the accumulation cannot move a bit. With the f64 paint it would."""
    import jax.numpy as jnp

    cfg = _cfg()
    x, _, st = _state(cfg, 2)
    got = engine.coarse_delta_streamed(st, cfg)

    # the same paint over every particle at once, in slot order so the SET is
    # identical (order cannot matter, which is the point)
    slots_all = []
    for b in range(st.n_bricks):
        s, xb, _ = st.decode_brick(b)
        slots_all.append(xb)
    x_all = np.concatenate(slots_all)
    mono = np.asarray(
        painting.paint_tsc_int(jnp.asarray(x_all), cfg.n_coarse, cfg.box_size, cfg.frac_bits),
        dtype=np.int64,
    )
    mean = cfg.n_total / float(cfg.n_coarse) ** 3
    want = np.asarray(
        painting.counts_from_int(mono.astype(np.int32), cfg.frac_bits, fdtype=jnp.float64)
    ) / mean - 1.0
    assert np.array_equal(got, want), "chunked accumulation moved a bit"
    assert float(np.abs(got).max()) > 0.1, "degenerate density field; comparison is vacuous"


def test_the_streamed_paint_refuses_the_order_dependent_accumulator():
    cfg = _cfg(paint_long="f64")
    _, _, st = _state(cfg, 3)
    with pytest.raises(ValueError, match="requires the integer accumulator"):
        engine.coarse_delta_streamed(st, cfg)


# ------------------------------------------------------------- the schedule


def test_the_fused_schedule_covers_the_same_total_drift():
    """The fused form must be the SAME trajectory, not a similar one: the two
    half-drifts share a velocity, so h_k + h_{k+1} after each kick plus a leading
    h_0 reproduces the boundary form's total advance exactly."""
    a = a_grid(0.1, 1.0, 8, "log")
    co = bullfrog_float_coeffs(bullfrog_table(a, Cosmology()))
    lead, fused = engine.fused_drifts(co)
    h = co[:, 0]
    assert lead == pytest.approx(h[0])
    assert lead + fused.sum() == pytest.approx(2.0 * h.sum())
    assert len(fused) == len(h)


def test_the_synchronised_float_driver_matches_the_reference_stepper():
    """The matched oracle the parity gate uses. Algebraically identical to
    looping float_step_bullfrog and deliberately not bitwise it -- so this
    asserts agreement to roundoff, and that the difference is not zero, which
    would mean the fused form was never actually exercised."""
    import jax.numpy as jnp

    from inexor.config import BoxConfig

    x = jnp.asarray(_positions(4))
    v = jnp.asarray(np.random.default_rng(5).normal(scale=0.5, size=(N_PART**3, 3)))
    a = a_grid(0.1, 1.0, 6, "log")
    co = bullfrog_float_coeffs(bullfrog_table(a, Cosmology()))
    box = BoxConfig(n_mesh=N_FINE, box_size=L_BOX, n_particles=N_PART)
    force_fn = forces.make_force_fn(box, fdtype=jnp.float64, paint="int")

    xa, va = x, v
    for c in co:
        xa, va = float_step_bullfrog(xa, va, tuple(np.asarray(c, np.float64)), force_fn, L_BOX)
    xb, vb = engine.float_run_bullfrog_sync(x, v, co, force_fn, L_BOX)

    d = np.abs(np.asarray(xa) - np.asarray(xb))
    d = np.minimum(d, L_BOX - d)
    assert d.max() < 1e-6, f"the two float drivers are not the same trajectory (max {d.max():.2e})"
    assert d.max() > 0.0, "the drivers are bitwise identical; the fused form is not exercised"


# ------------------------------------------------------------------ the step


def test_one_step_conserves_particles_and_leaves_the_container_consistent():
    cfg = _cfg()
    _, _, st = _state(cfg, 6)
    n0 = st.n_live
    a = a_grid(0.1, 1.0, 4, "log")
    co = bullfrog_float_coeffs(bullfrog_table(a, Cosmology()))
    _, fused = engine.fused_drifts(co)
    stats = engine.step(st, cfg, (co[0][1], co[0][2]), float(fused[0]))
    assert st.check() is True
    assert st.n_live == n0
    assert stats["cap"] > 0


def test_the_step_asserts_ownership_is_a_partition():
    """`n_owned` is checked against the particle count every step, because the
    velocity-scale reconciliation is only exact if ownership partitions -- and
    because a broken partition would otherwise silently drop a tile's kick."""
    cfg = _cfg()
    _, _, st = _state(cfg, 7)
    a = a_grid(0.1, 1.0, 4, "log")
    co = bullfrog_float_coeffs(bullfrog_table(a, Cosmology()))
    st.n_particles += 1  # lie about the count; the partition assert must fire
    with pytest.raises(AssertionError, match="ownership"):
        engine.step(st, cfg, (co[0][1], co[0][2]), 0.01)


def test_a_short_run_advances_and_stays_consistent():
    cfg = _cfg()
    _, _, st = _state(cfg, 8)
    n0 = st.n_live
    before = st.decode_brick(0)[1].copy()
    a = a_grid(0.1, 1.0, 3, "log")
    co = bullfrog_float_coeffs(bullfrog_table(a, Cosmology()))
    out = engine.run(st, cfg, co)
    assert len(out) == len(co)
    assert st.check() is True
    assert st.n_live == n0
    after = st.decode_brick(0)[1]
    assert after.shape != before.shape or not np.array_equal(after, before), (
        "the run moved nothing; the step is not advancing the state"
    )


def test_the_velocity_scale_is_reconciled_and_never_clamps():
    from inexor.codec import assert_int16_range

    cfg = _cfg()
    _, _, st = _state(cfg, 9)
    a = a_grid(0.1, 1.0, 3, "log")
    co = bullfrog_float_coeffs(bullfrog_table(a, Cosmology()))
    engine.run(st, cfg, co)
    assert_int16_range(st.w)  # raises if any code escaped int16
    assert st.vel_scale > 0.0


def test_the_engine_defaults_to_the_order_independent_paints():
    """The choice `density_tsc`'s docstring defers to M-v2-3. The low-level
    defaults stay f64 so the probe-parity comparisons keep comparing like with
    like; the ENGINE is where the D-006-compliant path is selected."""
    cfg = _cfg()
    assert cfg.paint_short == "int"
    assert cfg.paint_long == "int"


# ------------------------------------------------- the capacity shape ladder
# M-v2-6 Stage 0. `cap` moves every step as occupancy shifts, and every buffer
# keyed on it keys a new XLA shape, so an unquantized cap leaks one executable
# family per step: measured at cdev8, ten distinct caps over ten steps
# (285,554 -> 397,319) and a peak host RSS LINEAR in K at 0.331 GB/step.


def test_the_capacity_ladder_is_monotone_and_never_shrinks_a_shape():
    """Both arguments monotone, and exactness on a rung.

    Monotonicity is the load-bearing property: an octave-relative ladder would
    map cap 1001 -> 1250 while cap 1024 -> 1024, and a shape schedule that can go
    DOWN as cap goes up reintroduces exactly the churn this removes.
    """
    caps = np.arange(1, 5000)
    shapes = np.array([forces.capacity_shape(int(c)) for c in caps])
    assert np.all(shapes >= caps), "a shape must never be smaller than the rows it holds"
    assert np.all(np.diff(shapes) >= 0), "the ladder must be monotone in cap"
    # exact on a rung: powers of two are rungs for any rungs-per-octave
    for e in range(1, 20):
        assert forces.capacity_shape(1 << e) == 1 << e
    # sticky: a dip in cap cannot shrink the shape
    assert forces.capacity_shape(100, floor_shape=4096) == 4096
    assert forces.capacity_shape(9000, floor_shape=4096) >= 9000


def test_the_capacity_ladder_bounds_both_padding_and_shape_count():
    """The two quantities the knob trades, asserted as bounds rather than checked
    by eye: worst-case padding is one rung, and a doubling of cap costs exactly
    `rungs` shapes."""
    for rungs in (1, 2, 3, 4, 8):
        ratio = 2.0 ** (1.0 / rungs)
        caps = np.arange(1000, 20000, 7)
        shapes = np.array([forces.capacity_shape(int(c), rungs=rungs) for c in caps])
        # +1 absorbs the integer ceil on small rungs; the claim is the RATIO
        assert np.all(shapes <= np.ceil(caps * ratio) + 1)
        # the HALF-OPEN octave (2^k, 2^(k+1)] is what costs `rungs` shapes; the
        # closed interval also contains the lower rung itself, which is where the
        # first version of this assertion was simply wrong
        lo, hi = 4096, 8192
        n = len({forces.capacity_shape(c, rungs=rungs) for c in range(lo + 1, hi + 1)})
        assert n == rungs, f"a half-open octave should cost {rungs} shapes, got {n}"


def test_quantizing_the_capacity_shape_is_bitwise_neutral():
    """THE GATE for the fix, and it is an identity rather than a threshold.

    Padded rows are MASKED, not filled: the short arm passes `live`, the long arm
    `live=lv`, masked rows contribute an integer paint weight of exactly zero, the
    gathers do not mix rows, and both results are sliced back to the real count.
    So a bigger buffer must give bit-identical state. Run with `cap_rungs=1`
    (coarsest ladder, largest padding, and it lands on a different shape than the
    fine ladder) against a run that pads as little as the ladder allows.
    """
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

    cosmo = Cosmology()
    co = bullfrog_float_coeffs(bullfrog_table(a_grid(0.1, 0.5, 3, "log"), cosmo))
    finals = []
    caps = []
    for rungs in (1, 16):
        cfg = _cfg(cap_rungs=rungs)
        _, _, st = _state(cfg)
        out = engine.run(st, cfg, co)
        caps.append([s["cap"] for s in out])
        finals.append((st.off.copy(), st.w.copy(), st.occupancy.copy(),
                       st.brick_start.copy(), st.vel_scale))
    assert caps[0] != caps[1], (
        "the two arms took the SAME shapes, so this asserts nothing -- the knob "
        "did not move (an arm must move its knob)"
    )
    a, b = finals
    assert np.array_equal(a[0], b[0]), "positions differ: padding is not neutral"
    assert np.array_equal(a[1], b[1]), "velocities differ: padding is not neutral"
    assert np.array_equal(a[2], b[2]), "occupancy differs"
    assert np.array_equal(a[3], b[3]), "brick_start differs"
    assert a[4] == b[4], "velocity scale differs"


def test_the_run_visits_few_shapes_and_they_never_decrease():
    """The behavioural claim: a run's shape family is small and monotone. Without
    the ladder, cdev8 took a distinct cap on all ten of ten steps."""
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

    cosmo = Cosmology()
    co = bullfrog_float_coeffs(bullfrog_table(a_grid(0.1, 1.0, 8, "log"), cosmo))
    cfg = _cfg()
    _, _, st = _state(cfg)
    out = engine.run(st, cfg, co)
    shapes = [s["cap"] for s in out]
    trues = [s["cap_true"] for s in out]
    assert all(s >= t for s, t in zip(shapes, trues)), "a shape held fewer rows than cap"
    assert shapes == sorted(shapes), "the shape schedule must be non-decreasing"
    assert len(set(shapes)) <= max(2, len(shapes) // 2), (
        f"{len(set(shapes))} distinct shapes over {len(shapes)} steps: the ladder "
        "is not collapsing the shape family"
    )
