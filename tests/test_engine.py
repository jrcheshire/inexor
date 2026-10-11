"""The engine core: BullFrog PM on T9 state in slot order -- streamed coarse paint, fused
schedule, ownership partition, shape ladders, phase hook, and checkpoint/resume."""

import json
import os

import numpy as np
import pytest

from inexor import engine, forces, icgen, painting, state
from inexor.codec import T9Layout
from inexor.config import Cosmology
from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table, float_step_bullfrog

# The smallest geometry whose tile+buffer decomposition is not degenerate.
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
    """Chunked coarse paint is bitwise the monolithic one: the paint is integer and integer
    addition is associative, so chunking cannot move a bit."""
    import jax.numpy as jnp

    cfg = _cfg()
    x, _, st = _state(cfg, 2)
    got = engine.coarse_delta_streamed(st, cfg)

    # the same paint over every particle at once
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
    """The fused drifts (leading h_0, then h_k + h_{k+1} after each kick) cover exactly the
    boundary form's total advance."""
    a = a_grid(0.1, 1.0, 8, "log")
    co = bullfrog_float_coeffs(bullfrog_table(a, Cosmology()))
    lead, fused = engine.fused_drifts(co)
    h = co[:, 0]
    assert lead == pytest.approx(h[0])
    assert lead + fused.sum() == pytest.approx(2.0 * h.sum())
    assert len(fused) == len(h)


def test_the_synchronised_float_driver_matches_the_reference_stepper():
    """The fused float driver matches looping float_step_bullfrog to float64 round-off, and
    is not bitwise it (a zero difference would mean the fused form was not exercised)."""
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
    """`engine.step` asserts owned rows == particle count; a broken partition would
    otherwise silently drop or double a kick."""
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
    """Velocity codes stay in int16 under per-brick scales, where a fast particle drifting
    into a quiet brick could overflow. The spread check shows the fixture has distinct
    per-brick scales, so it could expose that."""
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
    """The engine selects the integer (order-independent) paints for both arms, although the
    low-level `density_tsc` default stays f64."""
    cfg = _cfg()
    assert cfg.paint_short == "int"
    assert cfg.paint_long == "int"


# ------------------------------------------------- the capacity shape ladder
# `cap` moves every step with occupancy and each value keys a new XLA shape, so an
# unquantized cap retains one executable family per step (host RSS measured linear in steps,
# 0.331 GB/step). The ladder quantizes it.


def test_the_capacity_ladder_is_monotone_and_never_shrinks_a_shape():
    """The ladder is >= cap, monotone in cap and in the floor, and exact on a rung. A shape
    that could go down as cap goes up (e.g. an octave-relative ladder) reintroduces churn."""
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
    """Worst-case padding is one rung, and each half-open octave (2^k, 2^(k+1)] costs exactly
    `rungs` shapes."""
    for rungs in (1, 2, 3, 4, 8):
        ratio = 2.0 ** (1.0 / rungs)
        caps = np.arange(1000, 20000, 7)
        shapes = np.array([forces.capacity_shape(int(c), rungs=rungs) for c in caps])
        # +1 absorbs the integer ceil on small rungs; the claim is the RATIO
        assert np.all(shapes <= np.ceil(caps * ratio) + 1)
        # half-open: the closed interval would also contain the lower rung
        lo, hi = 4096, 8192
        n = len({forces.capacity_shape(c, rungs=rungs) for c in range(lo + 1, hi + 1)})
        assert n == rungs, f"a half-open octave should cost {rungs} shapes, got {n}"


def test_quantizing_the_capacity_shape_is_bitwise_neutral():
    """Padding `cap` is bitwise neutral on evolved state: padded rows are masked (integer
    paint weight exactly zero, no row mixing, results sliced back). Compares cap_rungs=1
    (largest padding) against 16, and checks the two arms really took different shapes."""
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
    """A run's cap shapes are few (at most half the steps) and non-decreasing."""
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
# `coarse_delta_streamed` sizes its chunk buffer from `max(rows)`, a second XLA shape that
# moves every step for the same reason `cap` does; it gets its own ladder.


def _pad_shapes_of_a_run(cfg, co, seed=0):
    """Buffer row counts as XLA sees them, spied at both paint entry points. Reading the
    stats alone would pass if the field were quantized while the buffer stayed raw."""
    seen = []
    real_full = engine.paint_tsc_int
    real_sub = engine.paint_tsc_int_subblock

    def spy_full(xp, *a, **kw):
        seen.append(int(xp.shape[0]))
        return real_full(xp, *a, **kw)

    def spy_sub(xp, *a, **kw):
        seen.append(int(xp.shape[0]))
        return real_sub(xp, *a, **kw)

    # cuboid chunks go through the sub-block paint, the fallback through the full one
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
    """The pad ladder collapses the run's pad shapes: no more than the
    `ceil(rungs * log2(hi/lo)) + 1` rungs the ladder offers over the pad range, and fewer
    than the raw values. The raw pad must vary here or the bound holds for a no-op ladder."""
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
    """`pad_ladder=False` restores the per-step pad churn while evolving bit-identical state,
    so a memory A/B between the arms compares the same engine. `cap` stays laddered in both
    arms so only one shape family moves."""
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
    """A larger chunk pad gives a bit-identical coarse mesh (masked rows weigh exactly zero,
    integer addition is associative). Compared on the mesh so a failure localizes here."""
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
# A tile-local float rule, `all(u >= core_lo & u < core_hi)` with u = mod(x - origin, L),
# uses a different subtraction per tile, so neighbouring tiles' decisions are not
# complementary and a row ~1 ulp from a core plane can be owned by no tile.


def _boundary_positions():
    """Positions on and 1 ulp either side of every core plane and the wrap point, plus
    uniform filler."""
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
    """Every row, including those on core planes, is owned by exactly one tile."""
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
    """Control: the tile-local float rule must drop a row on this fixture, or the fixture
    cannot discriminate the exact partition above from the leaky rule."""
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
    """The engine's per-step partition assertion holds over an 8-step run (completing is
    the assertion passing every step)."""
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

    cosmo = Cosmology()
    co = bullfrog_float_coeffs(bullfrog_table(a_grid(0.1, 1.0, 8, "log"), cosmo))
    cfg = _cfg()
    _, _, st = _state(cfg)
    out = engine.run(st, cfg, co)
    assert len(out) == 8
    st.check()


def test_ownership_from_bricks_counts_every_stored_row_exactly_once():
    """Ownership derived from stored brick ordinals counts every stored row exactly once over
    all tiles. This is the partition the engine counts; a spatial rule can disagree with it
    for a row whose stored brick is one cell off its position."""
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
    """The brick grid divides into tile cores with nothing left over or claimed twice."""
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


# ------------------------------------------------- the phase hook
# The hook names phase boundaries so a caller can take a per-phase memory high-water mark.
# Pinned: the boundaries fall where documented, and the hook cannot move a number.


def _phase_names(cfg, seed, n_steps=2):
    _, _, st = _state(cfg, seed)
    a = a_grid(0.1, 1.0, n_steps, "log")
    co = bullfrog_float_coeffs(bullfrog_table(a, Cosmology()))
    seen = []
    out = engine.run(st, cfg, co, phase=seen.append)
    return seen, out, st


def test_the_phase_hook_names_every_boundary_in_order():
    """The exact per-step boundary sequence; membership alone would pass names fired in the
    wrong places, which would misattribute a phase's peak."""
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
    """`tile_loop_end` fires once, after the last tile, so it separates the tile loop from the
    step tail."""
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
    """Passing a hook leaves state and per-step diagnostics bit-identical."""
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
    """The default hook, called when no trace was requested, is a no-op."""
    assert engine._no_phase("anything") is None


# ------------------------------------------------- the sub-block coarse paint


def test_the_subblock_paint_knob_is_bitwise_and_genuinely_applies():
    """The sub-block coarse paint is bitwise the full one, and the stats show each arm really
    took its path."""
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
    """The host-side containment guard raises on a stencil leaving its cuboid (inside the jit
    it would wrap silently) and passes a contained one."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, L_BOX, size=(64, 3))
    cell = L_BOX / N_COARSE
    with pytest.raises(ValueError, match="containment violated"):
        engine._assert_stencil_contained(
            x, cell, np.array([0, 0, 0]), np.array([4, 4, 4]), N_COARSE
        )
    # the passing direction, so the guard is not always-raising
    lo, hi = 5.5 * cell, 8.4 * cell  # bases 6..8 -> [origin+1, origin+extent-2]
    x_ok = rng.uniform(lo, hi, size=(64, 3))
    engine._assert_stencil_contained(
        x_ok, cell, np.array([5, 5, 5]), np.array([7, 7, 7]), N_COARSE
    )


# --------------------------------------------------------------------------
# checkpoint and resume


def _ck_state(cfg, seed=0):
    _, _, st = _state(cfg, seed=seed)
    return st


def _rows(st):
    """Container content with allocation layout and intra-bucket order divided out. Raw array
    equality is wrong here: a reloaded `brick_start` comes from `_alloc_geometry`."""
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
    """Six steps straight through equal six steps resumed from the step-3 checkpoint, row for
    row. Both arms get the full schedule (`fused_drifts` makes `coeffs[:3]` a different
    trajectory). Catches a lead drift reapplied on resume; does not catch unrestored
    `cap_shape`/`pad_shape`, which are flat at this geometry."""
    co = _coeffs(6)
    ref = _ck_state(_cfg())
    engine.run(ref, _cfg(), co)

    # the interrupted arm: same schedule, checkpoints at steps 3 and 6
    d = str(tmp_path / "ck")
    cfg_c = _cfg(checkpoint_dir=d, checkpoint_every=3)
    st = _ck_state(cfg_c)
    out = engine.run(st, cfg_c, co)
    assert [o["checkpoint"] is not None for o in out] == [False, False, True] * 2

    # drop the newer manifest, as an interrupted write would; step 3 is then newest complete
    os.remove(os.path.join(d, "gen1", "manifest.json"))
    st_r, resume = engine.load_checkpoint(d, _cfg(), co, arena_frac=0.05)
    assert int(resume["step"]) == 3, "fell back to the wrong generation"

    engine.run(st_r, _cfg(), co, resume=resume)
    np.testing.assert_array_equal(_rows(ref), _rows(st_r))


def test_load_checkpoint_refuses_a_foreign_run(tmp_path):
    """Resume refuses a different geometry or schedule (the fingerprint hashes `coeffs`, so
    cosmology, a-grid and K are covered) but accepts different execution policy."""
    d = str(tmp_path / "ck")
    cfg_c = _cfg(checkpoint_dir=d, checkpoint_every=1)
    engine.run(_ck_state(cfg_c), cfg_c, _coeffs(4))

    with pytest.raises(ValueError, match="different configuration or schedule"):
        engine.load_checkpoint(d, _cfg(), _coeffs(5))          # different schedule
    with pytest.raises(ValueError, match="different configuration or schedule"):
        engine.load_checkpoint(d, _cfg(frac_bits=11), _coeffs(4))   # different geometry
    # execution policy is not fingerprinted: resuming on another machine must work
    engine.load_checkpoint(d, _cfg(tile_workers=2, eject_kernel="numpy"), _coeffs(4),
                           arena_frac=0.05)


def test_checkpointing_is_off_without_a_directory_and_disablable_with_zero(tmp_path):
    """No directory, or `checkpoint_every=0`, writes nothing and reports no checkpoint (the
    default config has no directory, so this is inert rather than a refusal)."""
    co = _coeffs(3)
    out = engine.run(_ck_state(_cfg()), _cfg(), co)
    assert all(o["checkpoint"] is None for o in out)

    d = str(tmp_path / "off")
    cfg0 = _cfg(checkpoint_dir=d, checkpoint_every=0)
    out0 = engine.run(_ck_state(cfg0), cfg0, co)
    assert all(o["checkpoint"] is None for o in out0)
    assert not os.path.exists(d)


def test_a_timed_checkpoint_reports_its_own_parts(tmp_path):
    """A checkpoint on a timed step reports its write-cost parts in the step's timings, so a
    production run measures full-size writes directly; untimed checkpoints carry none."""
    co = _coeffs(4)
    cfg_c = _cfg(checkpoint_dir=str(tmp_path / "ck"), checkpoint_every=2)
    st = _ck_state(cfg_c)
    out = engine.run(st, cfg_c, co, timed_steps=(1, 3))

    wrote = [o for o in out if o["checkpoint"] is not None]
    assert len(wrote) == 2, "vacuous: no checkpoint landed on a timed step"
    for o in wrote:
        parts = o["timings"]["checkpoint"]
        assert set(parts) >= {"index", "gather", "crc32", "write", "slabs"}, parts
        assert parts["slabs"] == st.bricks_per_side
        assert all(parts[k] >= 0.0 for k in ("index", "gather", "crc32", "write"))

    out2 = engine.run(_ck_state(cfg_c), cfg_c, co)
    assert [o["checkpoint"] is not None for o in out2] == [False, True, False, True]
    assert all(o["timings"] is None for o in out2)


def test_checkpointing_refuses_a_state_carrying_ids(tmp_path):
    """The checkpoint schema has no ids, so a run carrying ids refuses before the first step
    rather than dropping them."""
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


def test_a_run_split_into_segments_is_bitwise_the_uninterrupted_one(tmp_path):
    """Three `stop_at` + `resume` segments of two steps, each reloading from disk, equal six
    steps straight through row for row (how a run longer than a queue limit executes).
    Each segment gets the full coefficient list, since `coeffs[:n]` is not a prefix."""
    co = _coeffs(6)
    ref = _ck_state(_cfg())
    engine.run(ref, _cfg(), co)

    d = str(tmp_path / "seg")
    cfg_c = _cfg(checkpoint_dir=d, checkpoint_every=2)
    st = _ck_state(cfg_c)
    out = engine.run(st, cfg_c, co, stop_at=2)
    assert len(out) == 2, "the first segment did not stop where it was told"
    assert out[-1]["checkpoint"] is not None, "segment ended without a checkpoint"
    del st

    for stop in (4, 6):
        st_r, resume = engine.load_checkpoint(d, cfg_c, co, arena_frac=0.05)
        assert int(resume["step"]) == stop - 2, (
            f"resumed at step {resume['step']}, expected {stop - 2}"
        )
        out = engine.run(st_r, cfg_c, co, resume=resume, stop_at=stop)
        assert len(out) == 2, "a middle segment did not advance exactly two steps"
        last = st_r

    np.testing.assert_array_equal(_rows(ref), _rows(last))


def test_a_resumed_segment_writes_first_to_the_generation_it_did_not_load(tmp_path):
    """A resumed segment writes first to the generation it did not load, so its resume point
    survives until a newer state exists (one checkpoint per segment is the hard case)."""
    co = _coeffs(6)
    d = str(tmp_path / "seg")
    cfg_c = _cfg(checkpoint_dir=d, checkpoint_every=2)
    st = _ck_state(cfg_c)
    out = engine.run(st, cfg_c, co, stop_at=2)
    assert out[-1]["checkpoint"].endswith("gen0")
    del st
    for stop, want_gen in ((4, "gen1"), (6, "gen0")):
        st_r, resume = engine.load_checkpoint(d, cfg_c, co, arena_frac=0.05)
        assert f"gen{resume['gen']}" != want_gen
        out = engine.run(st_r, cfg_c, co, resume=resume, stop_at=stop)
        assert out[-1]["checkpoint"].endswith(want_gen), (
            f"segment to step {stop} wrote {out[-1]['checkpoint']}, over its resume point")
    steps = {}
    for g in ("gen0", "gen1"):
        with open(os.path.join(d, g, icgen.MANIFEST)) as fh:
            steps[g] = json.load(fh)["provenance"]["step"]
    assert steps == {"gen0": 6, "gen1": 4}


def test_stop_at_refuses_to_discard_a_segments_work(tmp_path):
    """`stop_at` off a checkpoint boundary (or at 0) refuses: a batch caller reads exit 0 as
    segment done, so a silent partial segment would resume from the wrong step."""
    co = _coeffs(6)
    d = str(tmp_path / "seg")
    cfg_c = _cfg(checkpoint_dir=d, checkpoint_every=2)
    st = _ck_state(cfg_c)
    with pytest.raises(ValueError, match="not a multiple of checkpoint_every"):
        engine.run(st, cfg_c, co, stop_at=3)
    with pytest.raises(ValueError, match="advance nothing"):
        engine.run(st, cfg_c, co, stop_at=0)


def test_eject_inflight_refuses_zero():
    """eject_inflight=0 refuses; at zero the dispatch loop would block forever."""
    import pytest as _pytest

    from inexor.state import drift_and_migrate_pooled

    with _pytest.raises(ValueError, match="eject_inflight must be >= 1"):
        drift_and_migrate_pooled(None, 0.0, None, eject_inflight=0)


def _epoch(k=6):
    """The a-grid `_coeffs` is built from, paired with its cosmology."""
    return a_grid(0.1, 1.0, k, "log"), Cosmology()


def test_a_checkpoint_records_the_epoch_it_actually_sits_at(tmp_path):
    """Each checkpoint records a = a_steps[completed steps] (a_grid has n_steps+1 points). An
    off-by-one would silently scale exported velocities at a neighbouring epoch; checked at
    every retained checkpoint, not just the final one."""
    import json

    a_steps, cosmo = _epoch(6)
    d = str(tmp_path / "ck")
    cfg_c = _cfg(checkpoint_dir=d, checkpoint_every=1)
    engine.run(_ck_state(cfg_c), cfg_c, _coeffs(6), epoch=(a_steps, cosmo))

    # gen0/gen1 roll, so after six steps they hold steps 5 and 6
    seen = {}
    for gen in ("gen0", "gen1"):
        with open(os.path.join(d, gen, "manifest.json")) as fh:
            prov = json.load(fh)["provenance"]
        seen[int(prov["step"])] = prov
    assert sorted(seen) == [5, 6]
    for step, prov in seen.items():
        assert prov["a"] == float(a_steps[step]), f"step {step} recorded the wrong epoch"
        assert prov["cosmology"]["Omega_m"] == cosmo.Omega_m
    assert seen[6]["a"] == 1.0


def test_epoch_is_optional_and_absent_by_default(tmp_path):
    """Without an epoch the manifest carries no `a`/`cosmology` and the schema is unchanged:
    the epoch is provenance, not schema, so existing checkpoints keep loading."""
    import json

    d = str(tmp_path / "ck")
    cfg_c = _cfg(checkpoint_dir=d, checkpoint_every=1)
    engine.run(_ck_state(cfg_c), cfg_c, _coeffs(2))

    with open(os.path.join(d, "gen1", "manifest.json")) as fh:
        man = json.load(fh)
    assert man["schema"] == "t9-slabs-2"
    assert "a" not in man["provenance"] and "cosmology" not in man["provenance"]


def test_source_is_recorded_when_given_and_moves_nothing(tmp_path):
    """`run(source=)` names the run's ICs in every checkpoint's provenance; without it the key
    is absent, and with it neither the slabs nor any other manifest field moves."""
    import json

    src = {"ics": "/somewhere/ics", "ics_provenance": {"commit": "abc"}}
    dirs = [str(tmp_path / "a"), str(tmp_path / "b")]
    for d, source in zip(dirs, (None, src)):
        cfg_c = _cfg(checkpoint_dir=d, checkpoint_every=1)
        engine.run(_ck_state(cfg_c), cfg_c, _coeffs(2), source=source)
    man = [json.load(open(os.path.join(d, "gen1", "manifest.json"))) for d in dirs]
    assert "source" not in man[0]["provenance"]
    assert man[1]["provenance"].pop("source") == src
    assert man[0] == man[1]
    files = sorted(f for f in os.listdir(os.path.join(dirs[0], "gen1")) if f != "manifest.json")
    assert files == sorted(f for f in os.listdir(os.path.join(dirs[1], "gen1"))
                           if f != "manifest.json")
    for f in files:
        a, b = (open(os.path.join(d, "gen1", f), "rb").read() for d in dirs)
        assert a == b, f


def test_epoch_does_not_move_the_fingerprint_or_the_trajectory(tmp_path):
    """Recording the epoch moves neither the trajectory nor the fingerprint (which already
    hashes `coeffs`), and a checkpoint with an epoch resumes under a config without one."""
    co = _epoch(4)
    d0, d1 = str(tmp_path / "a"), str(tmp_path / "b")
    c0 = _cfg(checkpoint_dir=d0, checkpoint_every=4)
    c1 = _cfg(checkpoint_dir=d1, checkpoint_every=4)
    st0, st1 = _ck_state(c0), _ck_state(c1)
    engine.run(st0, c0, _coeffs(4))
    engine.run(st1, c1, _coeffs(4), epoch=co)

    np.testing.assert_array_equal(_rows(st0), _rows(st1))
    assert (engine.checkpoint_fingerprint(c0, _coeffs(4))
            == engine.checkpoint_fingerprint(c1, _coeffs(4)))
    engine.load_checkpoint(d1, _cfg(), _coeffs(4), arena_frac=0.05)


def test_a_mismatched_epoch_grid_refuses_before_the_first_step(tmp_path):
    """An epoch grid without len(coeffs)+1 points refuses at call time (coeffs is always the
    full schedule, so this holds on every segment)."""
    d = str(tmp_path / "ck")
    cfg_c = _cfg(checkpoint_dir=d, checkpoint_every=1)
    with pytest.raises(ValueError, match="epoch grid has"):
        engine.run(_ck_state(cfg_c), cfg_c, _coeffs(6), epoch=_epoch(5))
