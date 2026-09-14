"""The device step executor: `coarse_backend` / `tile_backend` route `engine.step`
through the device paint and the device tile loop.

ROUTING, gated bitwise. Each device phase has its own identity gate in its own
suite (`test_device_paint_cards.py`, `test_device_tile.py`); this file gates that
the engine calls them where it says it does, with the inputs the host lane would
have used. With the eager tile, every knob alone and all together must reproduce
the host engine particle for particle over a run with a lead drift, repacks and
arena spills.

THE COMPILED TILE is not bitwise the eager one (record sec. 17), so its run cannot
be compared with the host run field by field. What is gated instead: the step
writes exactly what `tile_loop_device` writes when called directly on the same
inputs; no code moves by more than one against the eager executor; one program
per shape; and a split run resumes bitwise into the uninterrupted one on the
same backend.
"""

import copy

import numpy as np
import pytest

from inexor import engine, forces
from tests.test_engine_device_backend import _coeffs, _same, _state

jax = pytest.importorskip("jax")

L_BOX, N_PART, N_FINE, N_COARSE, N_TILE, B_FINE = 32.0, 32, 64, 16, 16, 8

#: keys that NAME the lane rather than describe the step
RECEIPTS = ("migrate_backend", "migrate_device", "coarse_backend", "tile_backend",
            "device_shapes")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _cfg(**kw):
    return engine.EngineConfig(
        box_size=L_BOX, n_part=N_PART, n_fine=N_FINE, n_coarse=N_COARSE,
        n_tile=N_TILE, b_fine=B_FINE, **kw
    )


def _same_stats(host, dev, drop=()):
    """Every host key present on the device card with an equal value, receipts and
    `drop` aside. Keys only the device lane reports are its receipts."""
    assert len(host) == len(dev)
    for k, (h, d) in enumerate(zip(host, dev)):
        missing = set(h) - set(d)
        assert not missing, f"step {k}: the device card lacks {sorted(missing)}"
        for key in h:
            if key in RECEIPTS or key in drop:
                continue
            hv, dv = h[key], d[key]
            if key == "repack" and hv is not None:
                skip = ("repack_device", "scratch_bytes")
                hv = {x: y for x, y in hv.items() if x not in skip}
                dv = {x: y for x, y in dv.items() if x not in skip}
            assert hv == dv, f"step {k}: stats[{key!r}] {hv!r} != {dv!r}"


class _Spy:
    """Counts calls of a module function the engine imports at call time."""

    def __init__(self, monkeypatch, module, name):
        self.n = 0
        real = getattr(module, name)

        def wrapped(*a, **kw):
            self.n += 1
            return real(*a, **kw)

        monkeypatch.setattr(module, name, wrapped)


# ------------------------------------------------------------------ routing, bitwise

LANES = {
    "coarse": dict(coarse_backend="device"),
    "tile-eager": dict(tile_backend="device", device_tile_jit=False),
    "coarse+tile-eager": dict(coarse_backend="device", tile_backend="device",
                              device_tile_jit=False),
    "all-device-eager": dict(coarse_backend="device", tile_backend="device",
                             device_tile_jit=False, migrate_backend="device"),
}


@pytest.mark.parametrize("lane", sorted(LANES))
def test_each_device_lane_is_bitwise_the_host_engine_with_receipts(lane, monkeypatch):
    from inexor.device import paint as dpaint
    from inexor.device import tile as dtile

    K = 3
    co = _coeffs(K)
    kw = dict(LANES[lane])
    if kw.get("coarse_backend") == "device":
        # the host paint's chunk length, so the chunk-shaped stats compare too
        kw["device_paint_chunk_bricks"] = _cfg().chunk_bricks
    cfg_h, cfg_d = _cfg(), _cfg(**kw)
    st_h, st_d = _state(cfg_h), _state(cfg_d)
    out_h = engine.run(st_h, cfg_h, co)

    eager = _Spy(monkeypatch, dtile, "tile_task_device")
    p0, t0 = dpaint.CALLS, dtile.CALLS
    out_d = engine.run(st_d, cfg_d, co)

    _same(st_h, st_d, f"{lane}: after the run")
    _same_stats(out_h, out_d)
    assert sum(o["n_arena_overflow"] for o in out_d) > 0, "VACUOUS: nothing spilled"
    coarse_dev = cfg_d.coarse_backend == "device"
    assert dpaint.CALLS - p0 == (K if coarse_dev else 0), "the coarse lane did not apply"
    assert dtile.CALLS == t0, "the eager lane ran the compiled loop"
    n_tiles = len(cfg_d.tiles)
    assert eager.n == (K * n_tiles if cfg_d.tile_backend == "device" else 0)
    for o in out_d:
        assert o["coarse_backend"] == cfg_d.coarse_backend
        assert o["tile_backend"] == ("device-eager" if cfg_d.tile_backend == "device"
                                     else "host")
        assert (o.get("coarse_device_chunks", 0) > 0) == coarse_dev
    assert all(o["device_shapes"] == {} for o in out_h)


def test_the_device_paint_chunk_length_moves_no_state_bit():
    from inexor.device.paint import default_chunk_bricks

    co = _coeffs(2)
    cfg_h, cfg_d = _cfg(), _cfg(coarse_backend="device")
    st_h, st_d = _state(cfg_h), _state(cfg_d)
    out_h = engine.run(st_h, cfg_h, co)
    out_d = engine.run(st_d, cfg_d, co)
    L = default_chunk_bricks(st_d.bricks_per_side)
    assert L != cfg_d.chunk_bricks, "vacuous: the default chunk IS the host's"
    assert all(o["coarse_chunk_bricks"] == L for o in out_d)
    _same(st_h, st_d, "default device chunk")
    _same_stats(out_h, out_d,
                drop=("coarse_pad", "coarse_pad_true", "coarse_subblock_chunks"))


def test_the_comparison_can_fail(monkeypatch):
    """Anti-vacuity: one chunk's block dropped on the card must move the run."""
    from inexor.device import paint as dpaint

    real = dpaint.CardInt64Accumulator.add
    dropped = []

    def add_but_drop_one(self, sub, origin, extent):
        if not dropped:
            dropped.append(True)
            return None
        return real(self, sub, origin, extent)

    co = _coeffs(1)
    cfg_h = _cfg()
    cfg_d = _cfg(coarse_backend="device", device_paint_chunk_bricks=cfg_h.chunk_bricks)
    st_h, st_d = _state(cfg_h), _state(cfg_d)
    engine.run(st_h, cfg_h, co)
    monkeypatch.setattr(dpaint.CardInt64Accumulator, "add", add_but_drop_one)
    engine.run(st_d, cfg_d, co)
    assert dropped, "the mutant never ran"
    with pytest.raises(AssertionError):
        _same(st_h, st_d, "mutant")


# ------------------------------------------------------------------ the compiled tile


def _step_inputs(cfg, co):
    lead, fused = engine.fused_drifts(co)
    return (co[0][1], co[0][2]), float(fused[0])


def test_the_compiled_step_writes_what_the_tile_loop_writes_directly():
    """Routing identity for the compiled lane: at `tile_loop_end` the step's state
    is bitwise `tile_loop_device` called by hand on the same coarse meshes,
    header, shapes and card shard."""
    import jax.numpy as jnp

    from inexor.device import coarse as dcoarse
    from inexor.device import tile as dtile

    cfg = _cfg(tile_backend="device")
    st = _state(cfg)
    ref = copy.deepcopy(st)
    w0 = st.w.copy()
    coeff, c_drift = _step_inputs(cfg, _coeffs(1))
    snap = {}

    def ph(name):
        if name == "tile_loop_end":
            snap["w"], snap["vs"] = st.w.copy(), st.vel_scale.copy()

    engine.step(st, cfg, coeff, c_drift, phase=ph)

    delta = engine.coarse_delta_streamed(ref, cfg)
    g = forces.coarse_force_meshes(
        jnp.asarray(delta), cfg.n_coarse, cfg.box_size, "long", r_s=cfg.r_s,
        match=(cfg.coarse_cell, cfg.fine_cell), fdtype=cfg.np_coarse_dtype)
    b_real = cfg._b_realized
    members = {t: ref.tile_bricks(t, cfg.n_tile, b_real, cfg.n_brick, cfg.n_fine)
               for t in cfg.tiles}
    counts = [sum(ref.brick_member_count(b) for b in members[t]) for t in cfg.tiles]
    cap = forces.capacity_shape(forces.tile_capacity(counts), rungs=cfg.cap_rungs)
    one_tile, geom = forces.make_tile_force_fn(
        cfg.n_fine, cfg.box_size, cfg.n_total, cfg.n_tile, cfg.b_fine, r_s=cfg.r_s,
        paint=cfg.paint_short, frac_bits=cfg.frac_bits, fdtype=cfg.np_fine_dtype)
    C = dict(cap=int(cap), n_tile=cfg.n_tile, n_brick=cfg.n_brick, n_fine=cfg.n_fine,
             n_coarse=cfg.n_coarse, box=cfg.box_size, coarse_cell=cfg.coarse_cell,
             cell=geom["cell"], b_real=int(b_real), alpha_k=float(coeff[0]),
             bcoef=float(coeff[1]))
    dtile.tile_loop_device(ref, one_tile, C, g, members, dtile.tile_step_shapes(ref),
                           coarse_shard=dcoarse.whole_mesh_shard(g))

    assert not np.array_equal(ref.w, w0), "vacuous: the loop wrote nothing"
    assert np.array_equal(snap["w"], ref.w), "the step's codes are not the loop's"
    assert np.array_equal(snap["vs"], ref.vel_scale), "the step's scales are not the loop's"


def test_the_compiled_step_moves_no_code_by_more_than_one_against_eager():
    cfg_e = _cfg(tile_backend="device", device_tile_jit=False)
    cfg_j = _cfg(tile_backend="device")
    st_e, st_j = _state(cfg_e), _state(cfg_j)
    assert np.array_equal(st_e.w, st_j.w)
    coeff, c_drift = _step_inputs(cfg_j, _coeffs(1))
    snaps = {}

    def hook(tag, st):
        def ph(name):
            if name == "tile_loop_end":
                snaps[tag] = (st.w.copy(), st.vel_scale.copy())
        return ph

    out_e = engine.step(st_e, cfg_e, coeff, c_drift, phase=hook("e", st_e))
    out_j = engine.step(st_j, cfg_j, coeff, c_drift, phase=hook("j", st_j))
    w0 = _state(cfg_e).w
    assert not np.array_equal(snaps["e"][0], w0), "vacuous: the eager step wrote nothing"
    dw = np.abs(snaps["j"][0].astype(np.int32) - snaps["e"][0].astype(np.int32))
    assert dw.max() <= 1, "a velocity code moved by more than one"
    # a max over the same bricks' scales on both lanes, each within the floor
    assert out_j["n_tiles"] == out_e["n_tiles"]
    assert out_j["vel_scale_kick_max"] > 0 and out_e["vel_scale_kick_max"] > 0


def test_the_compiled_run_is_one_program_per_shape():
    from inexor.device import tile as dtile

    co = _coeffs(2)
    cfg = _cfg(tile_backend="device", coarse_backend="device")
    st = _state(cfg)
    dtile._KERNELS.clear()
    t0, c0 = dtile._TRACES[0], dtile.CALLS
    out = engine.run(st, cfg, co)
    assert dtile.CALLS - c0 == 2, "the compiled lane did not run every step"
    shapes = {(o["cap"], o["device_shapes"]["tile"]["arena_rect"]) for o in out}
    assert dtile._TRACES[0] - t0 == len(shapes), "a per-tile or per-step value keyed a program"
    floors = [o["device_shapes"]["tile"]["arena_rect"] for o in out]
    assert floors == sorted(floors), "the shape floor did not carry across steps"


def test_a_compiled_run_split_and_resumed_is_bitwise_the_uninterrupted_one(tmp_path):
    from tests.test_engine import _rows

    co = _coeffs(4)
    kw = dict(coarse_backend="device", tile_backend="device", migrate_backend="device")
    ref = _state(_cfg(**kw), with_ids=False)
    out_ref = engine.run(ref, _cfg(**kw), co)
    d = str(tmp_path / "ck")
    cfg_c = _cfg(checkpoint_dir=d, checkpoint_every=2, **kw)
    st = _state(cfg_c, with_ids=False)
    out = engine.run(st, cfg_c, co, stop_at=2)
    st_r, resume = engine.load_checkpoint(d, _cfg(**kw), co, arena_frac=0.05)
    assert int(resume["step"]) == 2
    assert resume["device_shapes"] == out[-1]["device_shapes"], (
        "the checkpoint did not record the shapes the programs were compiled at")
    out_r = engine.run(st_r, _cfg(**kw), co, resume=resume)
    assert out_r[0]["device_shapes"]["tile"]["arena_rect"] >= (
        resume["device_shapes"]["tile"]["arena_rect"]), "the restored floor did not apply"
    np.testing.assert_array_equal(_rows(ref), _rows(st_r))
    assert len(out_ref) == 4


# ------------------------------------------------------------------ refusals


def test_validate_refuses_the_lanes_that_cannot_apply():
    with pytest.raises(ValueError, match="coarse_backend"):
        _cfg(coarse_backend="gpu").validate()
    with pytest.raises(ValueError, match="tile_backend"):
        _cfg(tile_backend="cuda").validate()
    with pytest.raises(ValueError, match="paint_subblock"):
        _cfg(coarse_backend="device", paint_subblock=False).validate()
    with pytest.raises(ValueError, match="tile_workers"):
        _cfg(tile_backend="device", tile_workers=2).validate()
    with pytest.raises(ValueError, match="device_tile_jit"):
        _cfg(device_tile_jit=False).validate()
    with pytest.raises(ValueError, match="device_paint_chunk_bricks"):
        _cfg(coarse_backend="device", device_paint_chunk_bricks=0).validate()
    jax.config.update("jax_enable_x64", False)
    try:
        for kw in (dict(coarse_backend="device"), dict(tile_backend="device")):
            with pytest.raises(ValueError, match="jax_enable_x64"):
                _cfg(coarse_dtype="float32", fine_dtype="float32", **kw).validate()
    finally:
        jax.config.update("jax_enable_x64", True)
    assert _cfg(coarse_backend="device", tile_backend="device").validate() is True
