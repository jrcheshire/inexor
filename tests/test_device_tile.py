"""D2e: one tile of the kick on the device, gated BITWISE against `tile_task`.

The oracle is the real host code with the real short force: `engine.tile_task`
and `tile_task_device` get the same state, the same jitted `one_tile`, the same
header and the same coarse meshes, and must return the same writes. On the CPU
jax backend both run the same FFTs, so equality is the bar, not a tolerance.

The arena branch must run: the state is built with no spare so a migrate forces
residents, and the test asserts they reached the tiles it compares.
"""

import copy

import numpy as np
import pytest

pytest.importorskip("jax")

from inexor import engine, forces, state  # noqa: E402
from inexor.codec import T9Layout  # noqa: E402
from inexor.device import kick as dkick  # noqa: E402
from inexor.device import tile as dtile  # noqa: E402
from inexor.device.decode import tile_decode_plan  # noqa: E402

# `tests/test_engine.py`'s validated smoke geometry, verbatim.
L_BOX, N_PART, N_FINE, N_COARSE, N_TILE, B_FINE = 32.0, 32, 64, 16, 16, 8

DTYPES = [("float64", "float64"), ("float64", "float32"),
          ("float32", "float32"), ("float32", "float64")]


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _cfg(fine="float64", coarse="float64"):
    return engine.EngineConfig(box_size=L_BOX, n_part=N_PART, n_fine=N_FINE,
                               n_coarse=N_COARSE, n_tile=N_TILE, b_fine=B_FINE,
                               fine_dtype=fine, coarse_dtype=coarse)


def _state(cfg):
    rng = np.random.default_rng(4)
    g = (np.arange(N_PART) + 0.5) * (L_BOX / N_PART)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    x = np.mod(q + rng.normal(scale=0.25 * L_BOX / N_PART, size=q.shape), L_BOX)
    v = np.random.default_rng(5).normal(scale=0.5, size=x.shape)
    t9 = T9Layout(box_size=L_BOX, n_part=N_PART, bucket_cells=2)
    st = state.SlotState.build(x, v, t9, N_FINE // cfg.n_brick, brick_slack=0.0,
                               arena_frac=0.30)
    state.drift_and_migrate(st, 2.0)
    return st


def _setup(fine="float64", coarse="float64"):
    cfg = _cfg(fine, coarse)
    st = _state(cfg)
    members = {t: st.tile_bricks(t, cfg.n_tile, cfg._b_realized, cfg.n_brick, cfg.n_fine)
               for t in cfg.tiles}
    counts = [sum(st.brick_member_count(b) for b in members[t]) for t in cfg.tiles]
    cap = forces.capacity_shape(forces.tile_capacity(counts) + 6)
    tile_force = forces.make_tile_force_fn(
        cfg.n_fine, cfg.box_size, cfg.n_total, cfg.n_tile, cfg.b_fine, r_s=cfg.r_s,
        paint=cfg.paint_short, frac_bits=cfg.frac_bits, fdtype=cfg.np_fine_dtype)
    one_tile, geom = tile_force
    C = dict(cap=int(cap), n_tile=cfg.n_tile, n_brick=cfg.n_brick, n_fine=cfg.n_fine,
             n_coarse=cfg.n_coarse, box=cfg.box_size, coarse_cell=cfg.coarse_cell,
             cell=geom["cell"], b_real=int(cfg._b_realized), alpha_k=0.87, bcoef=1.31)
    rng = np.random.default_rng(6)
    g_coarse = [rng.normal(scale=0.3, size=(N_COARSE,) * 3).astype(cfg.np_coarse_dtype)
                for _ in range(3)]
    return cfg, st, members, one_tile, C, g_coarse


def _same(a, b):
    assert a["empty"] == b["empty"] and a["n_owned"] == b["n_owned"]
    assert a["n_out"] == b["n_out"]
    if a["empty"]:
        return
    assert np.array_equal(a["slots_o"], b["slots_o"]), "owned slots differ"
    assert a["w_codes"].dtype == b["w_codes"].dtype
    assert np.array_equal(a["w_codes"], b["w_codes"]), "velocity codes differ"
    assert np.array_equal(a["run_bricks"], b["run_bricks"]), "written bricks differ"
    assert np.array_equal(a["run_scales"], b["run_scales"]), "per-brick scales differ"


def _arena_rows(st, bricks):
    return sum(len(st.arena_slots_of_brick(int(b))) for b in bricks)


# ------------------------------------------------------------------ the gate


@pytest.mark.parametrize("fine,coarse", DTYPES)
def test_one_tile_is_bitwise_the_host_tile_task(fine, coarse):
    cfg, st, members, one_tile, C, g_coarse = _setup(fine, coarse)
    tiles = [cfg.tiles[0], cfg.tiles[len(cfg.tiles) // 2], cfg.tiles[-1]]
    assert sum(_arena_rows(st, members[t]) for t in tiles) > 0, (
        "vacuous: no arena residents in the compared tiles")
    for t in tiles:
        want = engine.tile_task(st, one_tile, C, g_coarse, t, members[t])
        got = dtile.tile_task_device(st, one_tile, C, g_coarse, t, members[t])
        assert not want["empty"], f"vacuous: tile {t} owns nothing"
        _same(got, want)


def test_a_whole_step_of_tiles_writes_the_same_state():
    """Every tile, applied: the same velocity codes and scales everywhere, and
    ownership still a partition (every particle written exactly once)."""
    cfg, st, members, one_tile, C, g_coarse = _setup("float64", "float32")
    st_h, st_d = copy.deepcopy(st), copy.deepcopy(st)
    n_h = n_d = 0
    for t in cfg.tiles:
        rh = engine.tile_task(st, one_tile, C, g_coarse, t, members[t])
        rd = dtile.tile_task_device(st, one_tile, C, g_coarse, t, members[t])
        _same(rd, rh)
        engine.apply_result(st_h, rh)
        engine.apply_result(st_d, rd)
        n_h += rh["n_owned"]
        n_d += rd["n_owned"]
    assert n_d == n_h == st.n_particles
    assert not np.array_equal(st_h.w, st.w), "vacuous: the step wrote nothing"
    assert np.array_equal(st_d.w, st_h.w)
    assert np.array_equal(st_d.vel_scale, st_h.vel_scale)


def test_the_comparison_can_fail():
    """Anti-vacuity on each force arm: changing either must move the codes."""
    cfg, st, members, one_tile, C, g_coarse = _setup()
    t = cfg.tiles[0]
    base = dtile.tile_task_device(st, one_tile, C, g_coarse, t, members[t])
    bumped = [2.0 * g for g in g_coarse]
    long_moved = dtile.tile_task_device(st, one_tile, C, bumped, t, members[t])
    assert not np.array_equal(base["w_codes"], long_moved["w_codes"])

    def no_short(u, live, own):
        g, o, n = one_tile(u, live, own)
        return 0.0 * g, o, n

    short_moved = dtile.tile_task_device(st, no_short, C, g_coarse, t, members[t])
    assert not np.array_equal(base["w_codes"], short_moved["w_codes"])


def test_the_codes_use_the_whole_int16_range():
    """D-007 is off this path, so check what makes it unnecessary: every written
    brick's extreme lands on +-32767 and nothing exceeds it."""
    cfg, st, members, one_tile, C, g_coarse = _setup()
    res = dtile.tile_task_device(st, one_tile, C, g_coarse, cfg.tiles[0],
                                 members[cfg.tiles[0]])
    w = res["w_codes"].astype(np.int32)
    dkick.assert_int16_range_device(w)
    assert np.abs(w).max() == dkick.INT16_MAX


# ---------------------------------------------------------------- jit, tolerance
#
# The jitted tile is NOT bitwise the eager one: XLA compiles the coarse gather to
# a different rounding (record sec. 17). Measured over every tile of this state
# at all four dtype pairs, the long force differs by at most 4.48 eps x rms at an
# f64 coarse arm and 5.88 at f32, and nothing upstream of the gather differs.
# JIT_LONG_FORCE_EPS is that floor rounded up to a whole eps (JC, 2026-09-12).
# Everything else below is exact or an identity, not a tolerance.
JIT_LONG_FORCE_EPS = 6.0


def _jit_within_floor(j, e, st, bricks, t, cfg):
    """`j` (jit) against `e` (eager), both with `with_forces=True`."""
    assert j["empty"] == e["empty"]
    if e["empty"]:
        return
    for k in ("slots_o", "run_bricks"):
        assert np.array_equal(j[k], e[k]), f"{k} differ"
    assert j["n_owned"] == e["n_owned"] and j["n_out"] == e["n_out"]
    fj, fe = j["forces"], e["forces"]
    assert np.array_equal(fj["g_short"], fe["g_short"]), "the short force moved under jit"

    gl_e = fe["g_long"].astype(np.float64)
    rms = float(np.sqrt(np.mean(gl_e**2)))
    eps = float(np.finfo(fe["g_long"].dtype).eps)
    worst = float(np.abs(fj["g_long"].astype(np.float64) - gl_e).max()) / (eps * rms)
    assert worst <= JIT_LONG_FORCE_EPS, (
        f"tile {t}: jitted long force {worst:.2f} eps x rms from eager, floor "
        f"{JIT_LONG_FORCE_EPS}")

    dw = np.abs(j["w_codes"].astype(np.int32) - e["w_codes"].astype(np.int32))
    assert dw.max() <= 1, "a velocity code moved by more than one"

    # a max is 1-Lipschitz: a brick's scale cannot move further than its largest
    # velocity change / 32767, plus one ulp of the division
    plan = tile_decode_plan(st, np.asarray(bricks, dtype=np.int64))
    bor = np.repeat(plan["bricks"], plan["member_counts"])
    nb = N_FINE // cfg.n_brick
    bo = bor[forces.owned_mask_from_bricks(bor, t, cfg.n_tile, cfg.n_brick, nb)]
    dv = np.abs(fj["v_new"] - fe["v_new"]).max(axis=1)
    for b, se, sj in zip(e["run_bricks"], e["run_scales"], j["run_scales"]):
        assert abs(sj - se) <= dv[bo == b].max() / 32767 + np.spacing(max(se, sj)), (
            f"brick {b}: scale moved beyond what its velocities allow")


@pytest.mark.parametrize("fine,coarse", DTYPES)
def test_the_jitted_tile_is_within_the_floor_of_eager(fine, coarse):
    cfg, st, members, one_tile, C, g_coarse = _setup(fine, coarse)
    shapes = dtile.tile_step_shapes(st)
    for t in [cfg.tiles[0], cfg.tiles[len(cfg.tiles) // 2], cfg.tiles[-1]]:
        e = dtile.tile_task_device(st, one_tile, C, g_coarse, t, members[t],
                                   with_forces=True)
        j = dtile.tile_task_device(st, one_tile, C, g_coarse, t, members[t], jit=True,
                                   shapes=shapes, with_forces=True)
        _jit_within_floor(j, e, st, members[t], t, cfg)


def test_a_jitted_step_is_one_program_within_the_floor():
    """Every tile of a step through one compiled program; a second step with new
    kick coefficients must not retrace."""
    cfg, st, members, one_tile, C, g_coarse = _setup("float64", "float32")
    shapes = dtile.tile_step_shapes(st)
    dtile._KERNELS.clear()
    t0 = dtile._TRACES[0]
    for t in cfg.tiles:
        e = dtile.tile_task_device(st, one_tile, C, g_coarse, t, members[t],
                                   with_forces=True)
        j = dtile.tile_task_device(st, one_tile, C, g_coarse, t, members[t], jit=True,
                                   shapes=shapes, with_forces=True)
        _jit_within_floor(j, e, st, members[t], t, cfg)
    assert dtile._TRACES[0] - t0 == 1, "a per-tile value is keying a new program"
    C2 = dict(C, alpha_k=0.91, bcoef=1.07)
    t = cfg.tiles[1]
    e = dtile.tile_task_device(st, one_tile, C2, g_coarse, t, members[t], with_forces=True)
    j = dtile.tile_task_device(st, one_tile, C2, g_coarse, t, members[t], jit=True,
                               shapes=shapes, with_forces=True)
    _jit_within_floor(j, e, st, members[t], t, cfg)
    assert dtile._TRACES[0] - t0 == 1, "new kick coefficients retraced the program"


def test_a_staged_state_and_timings_change_no_bit():
    """The state placed on the device once, and the timed (synced) call, return
    exactly what the plain jitted call returns; every phase is timed."""
    cfg, st, members, one_tile, C, g_coarse = _setup("float64", "float32")
    shapes = dtile.tile_step_shapes(st)
    ds = dtile.stage_state_on_device(st)
    for t in (cfg.tiles[0], cfg.tiles[-1]):
        plain = dtile.tile_task_device(st, one_tile, C, g_coarse, t, members[t], jit=True,
                                       shapes=shapes)
        tm = {}
        staged = dtile.tile_task_device(st, one_tile, C, g_coarse, t, members[t], jit=True,
                                        shapes=shapes, device_state=ds, timings=tm)
        _same(staged, plain)
        assert set(tm) == {"plan", "stage", "h2d_tile", "h2d_state", "compute", "d2h",
                           "result"}
        assert all(v >= 0.0 for v in tm.values())
    with pytest.raises(ValueError, match="jitted path only"):
        dtile.tile_task_device(st, one_tile, C, g_coarse, cfg.tiles[0],
                               members[cfg.tiles[0]], device_state=ds)


# ------------------------------------------------ the step loop, results on device


def _host_apply_step(st, one_tile, C, g_coarse, members, shapes):
    """The jitted tile with host results and `engine.apply_result`, every tile."""
    n = 0
    for t in members:
        res = dtile.tile_task_device(st, one_tile, C, g_coarse, t, members[t], jit=True,
                                     shapes=shapes)
        engine.apply_result(st, res)
        n += res["n_owned"]
    return n


def test_a_device_step_writes_the_state_the_host_apply_path_writes():
    """The whole step's codes and scales written on the device and copied back
    once equal the jitted tile's host results applied tile by tile."""
    cfg, st, members, one_tile, C, g_coarse = _setup("float64", "float32")
    shapes = dtile.tile_step_shapes(st)
    st_h, st_d = copy.deepcopy(st), copy.deepcopy(st)
    n_h = _host_apply_step(st_h, one_tile, C, g_coarse, members, shapes)
    out = dtile.tile_loop_device(st_d, one_tile, C, g_coarse, members, shapes)
    assert n_h == out["n_owned"] == st.n_particles
    assert out["tiles_run"] == len(cfg.tiles)
    assert not np.array_equal(st_h.w, st.w), "vacuous: the step wrote nothing"
    assert np.array_equal(st_d.w, st_h.w), "velocity codes differ"
    assert np.array_equal(st_d.vel_scale, st_h.vel_scale), "per-brick scales differ"


def test_a_skipped_tile_leaves_its_rows_and_the_host_untouched():
    """Anti-vacuity on the write: a tile left out keeps its stored codes and
    scales, and the donated device buffers never alias the host state."""
    cfg, st, members, one_tile, C, g_coarse = _setup("float64", "float32")
    shapes = dtile.tile_step_shapes(st)
    st_d = copy.deepcopy(st)
    skip = cfg.tiles[3]
    w_before = st_d.w.copy()
    vs_before = st_d.vel_scale.copy()
    ds = dtile.stage_state_on_device(st_d)
    out = dtile.tile_loop_device(st_d, one_tile, C, g_coarse, members, shapes,
                                 tiles=[t for t in cfg.tiles if t != skip],
                                 device_state=ds, write_host=False)
    assert np.array_equal(st_d.w, w_before), "the donated program wrote into host memory"
    assert out["n_owned"] < st.n_particles
    w_dev, vs_dev = np.asarray(ds["w"]), np.asarray(ds["vel_scale"])
    assert not np.array_equal(w_dev, w_before), "vacuous: nothing was written"

    b = np.asarray(members[skip], dtype=np.int64)
    slots, _x, _v = st.decode_bricks(b)
    bor = np.repeat(b, [st.brick_member_count(x) for x in b])
    nb = N_FINE // cfg.n_brick
    own = forces.owned_mask_from_bricks(bor, skip, cfg.n_tile, cfg.n_brick, nb)
    assert own.sum() > 0
    assert np.array_equal(w_dev[slots[own]], w_before[slots[own]]), (
        "a skipped tile's owned rows were written")
    owned_bricks = np.unique(bor[own])
    assert np.array_equal(vs_dev[owned_bricks], vs_before[owned_bricks])


def test_a_partial_loop_refuses_to_pass_as_a_step():
    cfg, st, members, one_tile, C, g_coarse = _setup("float64", "float32")
    shapes = dtile.tile_step_shapes(st)
    st_d = copy.deepcopy(st)
    w_before = st_d.w.copy()
    part = {t: members[t] for t in cfg.tiles[:-1]}
    with pytest.raises(AssertionError, match="not a partition"):
        dtile.tile_loop_device(st_d, one_tile, C, g_coarse, part, shapes)
    assert np.array_equal(st_d.w, w_before), "the host state was written before the check"


# ------------------------------------------------ the coarse meshes on the device


def test_a_device_coarse_shard_is_bitwise_host_staging_per_tile():
    from inexor.device import coarse as dcoarse

    cfg, st, members, one_tile, C, g_coarse = _setup("float64", "float32")
    shapes = dtile.tile_step_shapes(st)
    sh = dcoarse.whole_mesh_shard(g_coarse)
    for t in [cfg.tiles[0], cfg.tiles[len(cfg.tiles) // 2], cfg.tiles[-1]]:
        host = dtile.tile_task_device(st, one_tile, C, g_coarse, t, members[t], jit=True,
                                      shapes=shapes)
        dev = dtile.tile_task_device(st, one_tile, C, g_coarse, t, members[t], jit=True,
                                     shapes=shapes, coarse_shard=sh)
        _same(dev, host)


def test_a_device_coarse_shard_step_writes_the_host_staged_state():
    from inexor.device import coarse as dcoarse

    cfg, st, members, one_tile, C, g_coarse = _setup("float64", "float32")
    shapes = dtile.tile_step_shapes(st)
    st_h, st_d = copy.deepcopy(st), copy.deepcopy(st)
    dtile.tile_loop_device(st_h, one_tile, C, g_coarse, members, shapes)
    sh = dcoarse.whole_mesh_shard(g_coarse)
    # g_coarse deliberately zeroed for the sharded run: it must not be read
    zeros = [np.zeros_like(g) for g in g_coarse]
    out = dtile.tile_loop_device(st_d, one_tile, C, zeros, members, shapes, coarse_shard=sh)
    assert out["n_owned"] == st.n_particles
    assert not np.array_equal(st_h.w, st.w), "vacuous: the step wrote nothing"
    assert np.array_equal(st_d.w, st_h.w), "velocity codes differ"
    assert np.array_equal(st_d.vel_scale, st_h.vel_scale), "per-brick scales differ"


def test_the_jit_floor_can_fail():
    """Anti-vacuity: coarse meshes moved by 1e-13 relative (~450 f64 eps) must
    exceed the floor."""
    cfg, st, members, one_tile, C, g_coarse = _setup("float64", "float64")
    shapes = dtile.tile_step_shapes(st)
    t = cfg.tiles[0]
    e = dtile.tile_task_device(st, one_tile, C, g_coarse, t, members[t], with_forces=True)
    nudged = [g * (1.0 + 1e-13) for g in g_coarse]
    j = dtile.tile_task_device(st, one_tile, C, nudged, t, members[t], jit=True,
                               shapes=shapes, with_forces=True)
    with pytest.raises(AssertionError, match="eps x rms"):
        _jit_within_floor(j, e, st, members[t], t, cfg)


def test_ownership_twin_is_the_host_rule():
    cfg = _cfg()
    nb = N_FINE // cfg.n_brick
    b = np.arange(nb**3, dtype=np.int64)
    for t in cfg.tiles[:: max(1, len(cfg.tiles) // 5)]:
        want = forces.owned_mask_from_bricks(b, t, cfg.n_tile, cfg.n_brick, nb)
        got = np.asarray(dtile.owned_rows_device(b, t, cfg.n_tile, cfg.n_brick, nb))
        assert want.any() and np.array_equal(got, want)


def test_a_stencil_outside_the_block_is_refused():
    """The deferred guard must still fire: a coarse halo of zero cannot contain a
    TSC stencil, so the tile must raise rather than read wrapped values."""
    cfg, st, members, one_tile, C, g_coarse = _setup()
    t = cfg.tiles[0]
    orig = forces.COARSE_HALO
    forces.COARSE_HALO = 0
    try:
        with pytest.raises(ValueError, match="stencil"):
            dtile.tile_task_device(st, one_tile, C, g_coarse, t, members[t])
    finally:
        forces.COARSE_HALO = orig
