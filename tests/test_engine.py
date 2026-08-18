"""The v2 engine core: BullFrog PM on T9 state in slot order (M-v2-3, S5)."""

import os

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


def test_the_velocity_scale_is_per_brick_and_never_clamps():
    """Per-brick scales removed the 32 B/p `pending` array, and with it the
    theorem that made overflow impossible. Under one global scale the encode
    could not escape int16 because the scale was a max over a PARTITION; per
    brick it can, whenever a fast particle drifts into a quiet brick. So the
    range assertion below is doing real work now where it was a formality
    before, and the spread assertion is what says the fixture could expose it."""
    from inexor.codec import assert_int16_range

    cfg = _cfg()
    _, _, st = _state(cfg, 9)
    a = a_grid(0.1, 1.0, 3, "log")
    co = bullfrog_float_coeffs(bullfrog_table(a, Cosmology()))
    engine.run(st, cfg, co)
    assert_int16_range(st.w)  # raises if any code escaped int16
    assert st.vel_scale.shape == (st.n_bricks,)
    assert st.vel_scale.min() > 0.0, "a zero scale is a division by zero on decode"
    assert st.vel_scale.min() < st.vel_scale.max(), (
        "every brick ended on the same scale, so this fixture cannot tell a "
        "per-brick scale from a global one"
    )


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
                       st.brick_start.copy(), st.vel_scale.copy()))
    assert caps[0] != caps[1], (
        "the two arms took the SAME shapes, so this asserts nothing -- the knob "
        "did not move (an arm must move its knob)"
    )
    a, b = finals
    assert np.array_equal(a[0], b[0]), "positions differ: padding is not neutral"
    assert np.array_equal(a[1], b[1]), "velocities differ: padding is not neutral"
    assert np.array_equal(a[2], b[2]), "occupancy differs"
    assert np.array_equal(a[3], b[3]), "brick_start differs"
    assert np.array_equal(a[4], b[4]), "velocity scale differs"


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


# ------------------------------------------------- the coarse chunk pad ladder
# M-v2-6 Stage 0b. Stage 0 put `cap` on the ladder and left the SECOND shape
# keyed on occupancy alone: `coarse_delta_streamed` sizes its chunk buffer from
# `max(rows)`, which moves every step for the same reason `cap` does. Measured at
# the smoke config, ten distinct pad values over ten steps against one for `cap`,
# and antares 446 measured the run peak still climbing +110 MB/step at cdev,
# linear over 15 steps and surviving malloc_trim (so it is live memory, and a
# retained executable family is live memory).


def _pad_shapes_of_a_run(cfg, co, seed=0):
    """The shapes as XLA SEES them, taken at the call rather than off the stats.

    Reading `coarse_pad` back out of the stats would pass if the field were
    quantized while the buffer stayed raw, which is the one defect this fix could
    plausibly have. `paint_tsc_int`'s first argument IS the buffer.
    """
    seen = []
    real_full = engine.paint_tsc_int
    real_sub = engine.paint_tsc_int_subblock

    def spy_full(xp, *a, **kw):
        seen.append(int(xp.shape[0]))
        return real_full(xp, *a, **kw)

    def spy_sub(xp, *a, **kw):
        seen.append(int(xp.shape[0]))
        return real_sub(xp, *a, **kw)

    # both entry points: since Stage 2c the chunk buffer reaches XLA through
    # `paint_tsc_int_subblock` on cuboid chunks and `paint_tsc_int` only on the
    # fallback path -- the buffer is the first argument of either
    engine.paint_tsc_int = spy_full
    engine.paint_tsc_int_subblock = spy_sub
    try:
        _, _, st = _state(cfg, seed)
        out = engine.run(st, cfg, co)
    finally:
        engine.paint_tsc_int = real_full
        engine.paint_tsc_int_subblock = real_sub
    return seen, out, st


def test_the_coarse_chunk_pad_collapses_to_a_small_shape_family():
    """The behavioural claim, and it is exact arithmetic: a shape COUNT is
    integer, so this is one of the few M-v2-6 numbers the laptop can settle.

    The bound is DERIVED from the ladder rather than picked -- over a pad range
    [lo, hi] the ladder offers `ceil(rungs * log2(hi/lo)) + 1` rungs, and the run
    may visit no more than that. The vacuity guard is the load-bearing half: if
    `coarse_pad_true` never moved at this config the assertion below would hold
    for a ladder that did nothing at all.
    """
    import math

    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

    cosmo = Cosmology()
    co = bullfrog_float_coeffs(bullfrog_table(a_grid(0.1, 1.0, 8, "log"), cosmo))
    cfg = _cfg()
    seen, out, _ = _pad_shapes_of_a_run(cfg, co)

    trues = [s["coarse_pad_true"] for s in out]
    shapes = [s["coarse_pad"] for s in out]
    assert len(set(trues)) > 1, (
        "the unquantized pad took ONE value at this config, so this test cannot "
        "see the churn it exists to remove (an arm must move its knob)"
    )
    assert set(seen) <= set(shapes), "a paint ran at a shape the stats never reported"
    assert all(s >= t for s, t in zip(shapes, trues)), "a pad held fewer rows than it must"
    assert shapes == sorted(shapes), "the pad schedule must be non-decreasing"
    rungs = math.ceil(cfg.cap_rungs * math.log2(max(trues) / min(trues))) + 1
    assert len(set(seen)) <= rungs, (
        f"{len(set(seen))} distinct pad shapes over {len(shapes)} steps against a "
        f"ladder that offers {rungs} across this pad range"
    )
    assert len(set(seen)) < len(set(trues)), (
        "the ladder collapsed nothing: as many shapes as unquantized pad values"
    )


def test_the_pad_ladder_knob_restores_the_churn_and_moves_no_bit():
    """`pad_ladder=False` is the A arm of the owed A/B, so two things have to
    hold or the job measures nothing: the knob must genuinely restore the churn
    (otherwise the arms differ in name only), and the two arms must evolve to
    bit-identical state (otherwise their peaks are not measurements of the same
    engine and the slope difference is unattributable).

    `cap` stays on its ladder in BOTH arms on purpose. Moving `cap_rungs` would
    move both shape families at once.
    """
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

    cosmo = Cosmology()
    co = bullfrog_float_coeffs(bullfrog_table(a_grid(0.1, 1.0, 8, "log"), cosmo))
    seen, caps, finals = {}, {}, {}
    for on in (False, True):
        cfg = _cfg(pad_ladder=on)
        s, out, st = _pad_shapes_of_a_run(cfg, co)
        seen[on] = s
        caps[on] = [x["cap"] for x in out]
        finals[on] = (st.off.copy(), st.w.copy(), st.occupancy.copy(),
                      st.brick_start.copy(), st.vel_scale.copy())
    n_off, n_on = len(set(seen[False])), len(set(seen[True]))
    assert n_off > n_on, (
        f"the knob did not move: {n_off} distinct pad shapes with the ladder off "
        f"against {n_on} with it on"
    )
    assert n_on == 1, f"the ladder should hold one shape over this run, got {n_on}"
    assert caps[False] == caps[True], "cap moved between arms: the A/B is confounded"
    a, b = finals[False], finals[True]
    assert np.array_equal(a[0], b[0]), "positions differ between the A/B arms"
    assert np.array_equal(a[1], b[1]), "velocities differ between the A/B arms"
    assert np.array_equal(a[2], b[2]), "occupancy differs between the A/B arms"
    assert np.array_equal(a[3], b[3]), "brick_start differs between the A/B arms"
    assert np.array_equal(a[4], b[4]), "velocity scale differs between the A/B arms"


def test_quantizing_the_coarse_pad_is_bitwise_neutral():
    """THE GATE, an identity like the `cap` one and for the same reason: the pad
    rows are masked (`live=lv`), a masked row contributes an integer weight of
    exactly zero, and integer addition is associative -- so a bigger chunk buffer
    must give a bit-identical mesh, not a nearly-identical one.

    Compared on the MESH rather than on evolved state so a failure localizes
    here; `test_quantizing_the_capacity_shape_is_bitwise_neutral` carries the
    end-to-end version for `cap`.
    """
    cfg = _cfg()
    _, _, st = _state(cfg, 3)
    stats = {}
    base = engine.coarse_delta_streamed(st, cfg, stats=stats)
    pad_true = stats["coarse_pad_true"]
    for floor in (pad_true, 2 * pad_true, 4 * pad_true + 7):
        s2 = {}
        got = engine.coarse_delta_streamed(st, cfg, stats=s2, pad_shape=floor)
        assert s2["coarse_pad"] >= floor, "the floor did not apply: the knob did not move"
        assert np.array_equal(got, base), (
            f"the mesh moved at pad {s2['coarse_pad']} vs {stats['coarse_pad']}: "
            "padding is not neutral"
        )


# ------------------------------------------------- ownership is a partition
# M-v2-6. The old rule tested the TILE-LOCAL coordinate,
# `all(u >= core_lo & u < core_hi)` with u = mod(x - origin, L) reached through a
# DIFFERENT subtraction per tile, so neighbouring tiles' decisions were not
# complementary in floating point. antares job 431 lost exactly one particle of
# 16,777,216 at cdev to a ~1 ulp gap at a core plane. It is realization-dependent,
# which is why it survived this long -- cdev passes on an Apple-arm64 realization
# and fails on an x86-64 one, the streams being per-machine-class (D-v2-23).


def _boundary_positions():
    """Positions engineered to sit ON and either side of every core plane, plus
    the wrap point -- the cases the old float rule could drop."""
    cell = L_BOX / N_FINE
    planes = np.arange(0, N_FINE + 1, N_TILE) * cell  # 0 .. L_BOX inclusive
    eps = np.spacing(L_BOX)  # ~1 ulp at the box scale
    coords = []
    for p in planes:
        coords += [p, np.nextafter(p, 0.0), np.nextafter(p, L_BOX), p - eps, p + eps]
    coords = np.array([c for c in coords if 0.0 <= c < L_BOX], dtype=np.float64)
    rng = np.random.default_rng(7)
    extra = rng.uniform(0.0, L_BOX, size=64)
    coords = np.concatenate([coords, extra])
    # every combination of a boundary-ish coordinate on each axis
    g = np.stack(np.meshgrid(coords, coords[:8], coords[:8], indexing="ij"), axis=-1)
    return g.reshape(-1, 3)


def test_ownership_is_an_exact_partition_including_on_the_core_planes():
    """Every row owned by EXACTLY one tile. Not 'almost always' -- the engine
    asserts this as a partition and the velocity-scale theorem depends on it."""
    cfg = _cfg()
    x = _boundary_positions()
    cell = L_BOX / N_FINE
    counts = np.zeros(len(x), dtype=np.int32)
    for t in cfg.tiles:
        counts += forces.owned_mask(x, t, cell, cfg.n_tile, cfg.n_fine).astype(np.int32)
    assert counts.min() == 1 and counts.max() == 1, (
        f"{int((counts == 0).sum())} rows owned by NO tile and "
        f"{int((counts > 1).sum())} owned by more than one, of {len(x)}"
    )


def test_the_old_tile_local_rule_is_the_one_that_leaks():
    """The regression's provenance, kept executable so the fix cannot be undone
    quietly: reproduce the retired rule and show it drops rows the new one keeps.

    If this ever stops finding a gap the test is vacuous, so it asserts that the
    old rule DOES leak -- which is what makes it evidence rather than decoration.
    """
    cfg = _cfg()
    x = _boundary_positions()
    cell = L_BOX / N_FINE
    b_real = cfg._b_realized
    core_lo, core_hi = b_real * cell, (b_real + cfg.n_tile) * cell
    old = np.zeros(len(x), dtype=np.int32)
    for t in cfg.tiles:
        origin, _ = forces.tile_origin_extent(t, cfg.n_tile, b_real, cell)
        u = np.mod(x - np.asarray(origin), L_BOX)
        old += np.all((u >= core_lo) & (u < core_hi), axis=1).astype(np.int32)
    assert old.min() == 0, (
        "the retired rule owned every row on this fixture, so it does not "
        "demonstrate the defect -- strengthen the fixture rather than delete this"
    )


def test_a_run_keeps_every_particle_owned_once_over_many_steps():
    """The engine's own partition assertion, exercised over a run rather than a
    step: `engine.step` raises if the tiles do not own exactly n_particles rows,
    so completing is the assertion passing at every step."""
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

    cosmo = Cosmology()
    co = bullfrog_float_coeffs(bullfrog_table(a_grid(0.1, 1.0, 8, "log"), cosmo))
    cfg = _cfg()
    _, _, st = _state(cfg)
    out = engine.run(st, cfg, co)
    assert len(out) == 8
    st.check()


def test_ownership_from_bricks_counts_every_stored_row_exactly_once():
    """The partition the engine's assertion actually counts: rows in STORAGE.

    `owned_mask` partitions space exactly and that is a different partition --
    membership comes from brick ordinals, so a row whose stored brick disagrees
    with its position by one cell is handed to one tile and assigned to another,
    and nobody claims it (antares 436: one row of 16,777,216). This asserts the
    storage-derived form over the real container, summed over every tile.
    """
    cfg = _cfg()
    _, _, st = _state(cfg, 11)
    b_real = cfg._b_realized
    nb = cfg.n_fine // cfg.n_brick
    total = 0
    for t in cfg.tiles:
        members = st.tile_bricks(t, cfg.n_tile, b_real, cfg.n_brick, cfg.n_fine)
        brick_of_row = np.repeat(
            np.asarray(members, dtype=np.int64),
            [st.brick_member_count(b) for b in members],
        )
        own = forces.owned_mask_from_bricks(brick_of_row, t, cfg.n_tile, cfg.n_brick, nb)
        total += int(own.sum())
    assert total == st.n_particles, (
        f"the tiles own {total} stored rows against {st.n_particles} particles"
    )


def test_every_brick_belongs_to_exactly_one_tile_core():
    """The arithmetic the storage-derived partition rests on, asserted rather than
    argued: the brick grid divides into tile cores with nothing left over."""
    cfg = _cfg()
    nb = cfg.n_fine // cfg.n_brick
    per = cfg.n_tile // cfg.n_brick
    assert nb % per == 0, "the brick grid does not divide into tile cores"
    assert nb // per == cfg.tiles_side
    claims = np.zeros(nb**3, dtype=np.int32)
    all_bricks = np.arange(nb**3, dtype=np.int64)
    for t in cfg.tiles:
        claims += forces.owned_mask_from_bricks(
            all_bricks, t, cfg.n_tile, cfg.n_brick, nb
        ).astype(np.int32)
    assert claims.min() == 1 and claims.max() == 1, (
        f"{int((claims != 1).sum())} bricks of {nb**3} are not claimed exactly once"
    )


# ------------------------------------------------- the phase hook (M-v2-6 Stage 0b)
# A peak is a max and a max carries no timestamp. Stage 0 attributed the engine's
# peak by differencing whole-run maxima between arms, and at cdev the terms it was
# separating (67-179 MB) sat inside the run-to-run scatter of the maximum itself
# (sigma 45-115 MB over five repeats of one leg, antares 445). The hook names the
# boundaries so a caller can take a high-water mark per phase instead. These tests
# pin the two properties the instrument rests on: the boundaries are where the
# docstring says, and the hook cannot move a number.


def _phase_names(cfg, seed, n_steps=2):
    _, _, st = _state(cfg, seed)
    a = a_grid(0.1, 1.0, n_steps, "log")
    co = bullfrog_float_coeffs(bullfrog_table(a, Cosmology()))
    seen = []
    out = engine.run(st, cfg, co, phase=seen.append)
    return seen, out, st


def test_the_phase_hook_names_every_boundary_in_order():
    """The per-step sequence, asserted exactly rather than by membership.

    Membership would pass if the hook fired the right names in the wrong places,
    which is the one failure that would silently misattribute a peak: a boundary
    after the wrong statement reports another phase's allocation as this one's.
    """
    cfg = _cfg()
    seen, out, _ = _phase_names(cfg, 21, n_steps=2)
    assert seen[0] == "kernel_build", "the once-per-run force build is unnamed"
    assert seen[1] == "lead_drift", "the drift onto the first midpoint is unnamed"
    per_step = seen[2:]
    n_tiles = len(cfg.tiles)
    # one step's worth: the three global phases, then the tile loop, then the tail
    head = per_step[: 3 + 4 * n_tiles + 3]
    assert head[:3] == ["coarse_paint", "coarse_solve", "membership"]
    tile_block = head[3 : 3 + 4 * n_tiles]
    assert tile_block == ["tile_decode", "tile_short", "tile_long", "tile_reduce"] * n_tiles, (
        "the per-tile boundaries are not one clean repeating group per tile"
    )
    assert head[3 + 4 * n_tiles :] == ["tile_loop_end", "reconcile", "migrate"]
    assert len(out) == 2


def test_the_tile_loop_end_boundary_falls_after_every_tile():
    """`pending` is largest at the end of the tile loop and nowhere else, so that
    boundary is the only place a high-water mark can price it. If it fired inside
    the loop it would price a fraction of the term and read as a smaller one."""
    cfg = _cfg()
    seen, _, _ = _phase_names(cfg, 22, n_steps=1)
    i = seen.index("tile_loop_end")
    assert seen[i - 1] == "tile_reduce", "the loop-end boundary is inside the loop"
    assert "tile_reduce" not in seen[i:], "a tile ran after the loop-end boundary"
    assert seen.count("tile_reduce") == len(cfg.tiles)


def test_the_repack_boundary_fires_only_when_the_repack_does():
    cfg_on = _cfg(repack_every=1)
    seen_on, _, _ = _phase_names(cfg_on, 23, n_steps=2)
    assert seen_on.count("repack") == 2
    cfg_off = _cfg(repack_every=0)
    seen_off, _, _ = _phase_names(cfg_off, 23, n_steps=2)
    assert "repack" not in seen_off


def test_the_phase_hook_cannot_move_a_number():
    """Neutrality, asserted on the STATE and on the per-step diagnostics.

    An instrument that perturbs what it measures is worse than no instrument,
    and this one runs inside the hot loop. Both runs start from the same seed,
    so every field must agree exactly -- not to a tolerance.
    """
    cfg = _cfg()
    _, _, st_a = _state(cfg, 24)
    _, _, st_b = _state(cfg, 24)
    a = a_grid(0.1, 1.0, 3, "log")
    co = bullfrog_float_coeffs(bullfrog_table(a, Cosmology()))
    out_a = engine.run(st_a, cfg, co)
    out_b = engine.run(st_b, cfg, co, phase=lambda _name: None)

    assert np.array_equal(st_a.off, st_b.off), "positions moved under the hook"
    assert np.array_equal(st_a.w, st_b.w), "velocity codes moved under the hook"
    assert np.array_equal(st_a.occupancy, st_b.occupancy)
    assert np.array_equal(st_a.vel_scale, st_b.vel_scale)
    for sa, sb in zip(out_a, out_b):
        for k in ("cap", "cap_true", "vel_scale", "coarse_peak_int", "arena_used"):
            assert sa.get(k) == sb.get(k), f"the hook moved `{k}`"


def test_the_default_hook_is_a_no_op_that_returns_nothing():
    """`_no_phase` is what the hot loop calls when no caller asked for a trace."""
    assert engine._no_phase("anything") is None


# ------------------------------------------------- the sub-block coarse paint


def test_the_subblock_paint_knob_is_bitwise_and_genuinely_applies():
    """Stage 2c's gate: the sub-block path must change NO bit of the coarse
    delta, and the A/B knob must prove it applied -- an arm whose knob did not
    move measures nothing (both stats fields assert it here)."""
    cfg_on = _cfg()
    cfg_off = _cfg(paint_subblock=False)
    _, _, st1 = _state(cfg_on, 5)
    _, _, st2 = _state(cfg_off, 5)
    s_on, s_off = {}, {}
    a = engine.coarse_delta_streamed(st1, cfg_on, stats=s_on)
    b = engine.coarse_delta_streamed(st2, cfg_off, stats=s_off)
    assert s_on["coarse_subblock_chunks"] > 0, "the sub-block path never fired: vacuous A/B"
    assert s_off["coarse_subblock_chunks"] == 0, "the OFF arm took the sub-block path"
    assert np.array_equal(a, b), "the sub-block paint moved a bit of the coarse delta"
    assert float(np.abs(a).max()) > 0.1, "degenerate density field; comparison is vacuous"


def test_the_subblock_containment_guard_fires_on_a_wrong_cuboid():
    """The host-side half of the containment contract must REFUSE, not wrap:
    a stencil corner leaving the block wraps silently inside the jit (the
    D-v2-21 failure class), so the guard in front of it is the safety."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, L_BOX, size=(64, 3))
    cell = L_BOX / N_COARSE
    with pytest.raises(ValueError, match="containment violated"):
        engine._assert_stencil_contained(
            x, cell, np.array([0, 0, 0]), np.array([4, 4, 4]), N_COARSE
        )
    # and the passing direction, so the test cannot rot into always-raising
    lo, hi = 5.5 * cell, 8.4 * cell  # bases 6..8 -> [origin+1, origin+extent-2]
    x_ok = rng.uniform(lo, hi, size=(64, 3))
    engine._assert_stencil_contained(
        x_ok, cell, np.array([5, 5, 5]), np.array([7, 7, 7]), N_COARSE
    )


# --------------------------------------------------------------------------
# M-v2-6 Stage 4(b): checkpoint and resume


def _ck_state(cfg, seed=0):
    _, _, st = _state(cfg, seed=seed)
    return st


def _rows(st):
    """Container content with allocation layout and intra-bucket order divided
    out -- the normal form the writer's round-trip gate uses. Raw array
    equality is the wrong invariant: a reloaded state's `brick_start` comes
    from `_alloc_geometry`, not from the run that produced it."""
    out = []
    for b in range(st.n_bricks):
        slots = st.brick_member_slots(b)
        live = st.brick_live_count(b)
        keys = np.concatenate([
            st.bucket_flat_of_live_slots(b),
            st.arena_bucket[slots[live:] - st.arena_base],
        ])
        out.append(np.column_stack([keys, st.off[slots], st.w[slots]]).astype(np.int64))
    r = np.concatenate(out)
    return r[np.lexsort(r.T[::-1])]


def _coeffs(k=6):
    return bullfrog_float_coeffs(bullfrog_table(a_grid(0.1, 1.0, k, "log"), Cosmology()))


def test_resumed_run_is_bitwise_the_uninterrupted_one(tmp_path):
    """THE Stage 4(b) gate: six steps straight through, against six steps
    interrupted after the third and resumed from disk, particle for particle.

    NB the interrupted arm runs the FULL schedule and stops, rather than
    running a truncated one. `fused_drifts` fuses the trailing half-drift of
    each step with the leading half of the next, so a run over `coeffs[:3]` is
    a different trajectory, not the first half of this one.

    What this gate CATCHES: the lead drift being reapplied on resume, which
    would move the whole box an extra half step (mutation-checked, fails).

    What it does NOT catch, stated because the docstring claimed otherwise
    first: `cap_shape` / `pad_shape` not being restored. Both are flat at this
    geometry (5161 at every step), so re-laddering from zero reaches the same
    value and the mutation is vacuous. Measured separately rather than assumed
    -- quadrupling both shapes on resume moves 0 of ~229k rows, since the
    padded rows are masked -- so on arm64 CPU at this scale the restore is a
    compile-count choice, not a correctness one. A GPU arm, where XLA
    reassociates by shape, is not covered by that measurement."""
    co = _coeffs(6)
    ref = _ck_state(_cfg())
    engine.run(ref, _cfg(), co)

    # the interrupted arm: same schedule, checkpoints at steps 3 and 6
    d = str(tmp_path / "ck")
    cfg_c = _cfg(checkpoint_dir=d, checkpoint_every=3)
    st = _ck_state(cfg_c)
    out = engine.run(st, cfg_c, co)
    assert [o["checkpoint"] is not None for o in out] == [False, False, True] * 2

    # drop the newer generation's manifest: what an interrupted write leaves,
    # and what makes step 3 the newest COMPLETE checkpoint
    os.remove(os.path.join(d, "gen1", "manifest.json"))
    st_r, resume = engine.load_checkpoint(d, _cfg(), co, arena_frac=0.05)
    assert int(resume["step"]) == 3, "fell back to the wrong generation"

    engine.run(st_r, _cfg(), co, resume=resume)
    np.testing.assert_array_equal(_rows(ref), _rows(st_r))


def test_load_checkpoint_refuses_a_foreign_run(tmp_path):
    """A resume under a different geometry or a different schedule does not
    fail loudly on its own -- it produces a run that is half one thing and half
    another. The fingerprint covers `coeffs` as bytes, so the cosmology, the
    a-grid and K are all in it."""
    d = str(tmp_path / "ck")
    cfg_c = _cfg(checkpoint_dir=d, checkpoint_every=1)
    engine.run(_ck_state(cfg_c), cfg_c, _coeffs(4))

    with pytest.raises(ValueError, match="different configuration or schedule"):
        engine.load_checkpoint(d, _cfg(), _coeffs(5))          # different schedule
    with pytest.raises(ValueError, match="different configuration or schedule"):
        engine.load_checkpoint(d, _cfg(frac_bits=11), _coeffs(4))   # different geometry
    # execution policy is deliberately NOT fingerprinted: resuming onto another
    # node with a different worker count is the point of having checkpoints
    engine.load_checkpoint(d, _cfg(tile_workers=2, eject_kernel="numpy"), _coeffs(4),
                           arena_frac=0.05)


def test_checkpointing_is_off_without_a_directory_and_disablable_with_zero(tmp_path):
    """`checkpoint_every=0` is the off switch, the `repack_every` idiom. With no
    directory the machinery is inert rather than a refusal, because the default
    config has none and every single-process caller would otherwise raise -- so
    the RECEIPT is what proves it applied."""
    co = _coeffs(3)
    out = engine.run(_ck_state(_cfg()), _cfg(), co)
    assert all(o["checkpoint"] is None for o in out)

    d = str(tmp_path / "off")
    cfg0 = _cfg(checkpoint_dir=d, checkpoint_every=0)
    out0 = engine.run(_ck_state(cfg0), cfg0, co)
    assert all(o["checkpoint"] is None for o in out0)
    assert not os.path.exists(d)


def test_checkpointing_refuses_a_state_carrying_ids(tmp_path):
    """The schema has no ids, and a checkpoint that dropped them would make the
    restart non-reproducible for anything id-dependent. It has to refuse before
    the first step, not 37 minutes into it."""
    cfg_c = _cfg(checkpoint_dir=str(tmp_path / "ck"), checkpoint_every=1)
    st = _ck_state(cfg_c, seed=1)
    st.ids = np.arange(len(st.off), dtype=np.int64)
    with pytest.raises(ValueError, match="carries ids"):
        engine.run(st, cfg_c, _coeffs(2))


def test_load_checkpoint_refuses_when_nothing_is_complete(tmp_path):
    d = str(tmp_path / "empty")
    os.makedirs(os.path.join(d, "gen0"))
    with pytest.raises(FileNotFoundError, match="no complete inexor checkpoint"):
        engine.load_checkpoint(d, _cfg(), _coeffs(2))
