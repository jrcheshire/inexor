"""E2: the tile loop against a window of x-slabs, gated BITWISE against the
whole-state `tile_loop_device`.

At this geometry a tile draws from 4 of 8 x-slabs and tile plane 0's window
wraps x = 0, so the window genuinely slides; the state is built with no brick
slack and migrated, so arena residents land in the windows. Both are asserted.
"""

import copy

import numpy as np
import pytest

pytest.importorskip("jax")

from inexor import engine  # noqa: E402
from inexor.device import coarse as dcoarse  # noqa: E402
from inexor.device import tile as dtile  # noqa: E402
from inexor.device import window as dwin  # noqa: E402
from tests.test_device_tile import _setup  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _shapes(cfg, st):
    return dict(dtile.tile_step_shapes(st),
                window=dwin.window_shapes(st, cfg.n_tile, cfg._b_realized, cfg.n_brick))


@pytest.mark.parametrize("coarse_on_card", [False, True])
def test_the_windowed_loop_is_bitwise_the_whole_state_loop(coarse_on_card):
    cfg, st, members, one_tile, C, g_coarse = _setup("float64", "float32")
    shapes = _shapes(cfg, st)
    sh = dcoarse.whole_mesh_shard(g_coarse) if coarse_on_card else None
    st_whole, st_win = copy.deepcopy(st), copy.deepcopy(st)
    ref = dtile.tile_loop_device(st_whole, one_tile, C, g_coarse, members,
                                 dtile.tile_step_shapes(st), coarse_shard=sh)
    out = dwin.tile_loop_windowed(st_win, one_tile, C, None if sh else g_coarse, members,
                                  shapes, coarse_shard=sh)

    assert not np.array_equal(st_whole.w, st.w), "vacuous: the step wrote nothing"
    assert np.array_equal(st_win.w, st_whole.w), "velocity codes differ"
    assert np.array_equal(st_win.vel_scale, st_whole.vel_scale), "per-brick scales differ"
    assert out["n_owned"] == ref["n_owned"] == st.n_particles
    assert out["n_out"] == ref["n_out"]
    assert out["vel_scale_kick_max"] == ref["vel_scale_kick_max"]

    nb, _pad, span, _per = dwin.plane_geometry(st, cfg.n_tile, cfg._b_realized, cfg.n_brick)
    assert span < nb, "vacuous: the window is the whole brick grid"
    assert out["planes_run"] == cfg.tiles_side
    assert out["tiles_run"] == len(cfg.tiles)
    assert out["wrapped"], "vacuous: no window crossed x = 0"
    assert out["residents_staged"] > 0, "vacuous: no arena resident reached a window"
    assert out["window_live_rows_max"] < int(st.brick_start[-1]), (
        "vacuous: a window held every live row")


def test_a_windowed_step_is_one_program():
    cfg, st, members, one_tile, C, g_coarse = _setup("float64", "float32")
    shapes = _shapes(cfg, st)
    dtile._KERNELS.clear()
    t0 = dtile._TRACES[0]
    dwin.tile_loop_windowed(copy.deepcopy(st), one_tile, C, g_coarse, members, shapes)
    assert dtile._TRACES[0] - t0 == 1, "a plane or tile value keyed a new program"


def test_a_plane_missing_a_tile_refuses_before_writing_the_host():
    cfg, st, members, one_tile, C, g_coarse = _setup("float64", "float32")
    shapes = _shapes(cfg, st)
    st_d = copy.deepcopy(st)
    w0, vs0 = st_d.w.copy(), st_d.vel_scale.copy()
    part = {t: b for t, b in members.items() if t != cfg.tiles[1]}  # a tile of plane 0
    assert cfg.tiles[1][0] == 0
    with pytest.raises(AssertionError, match="tile plane 0"):
        dwin.tile_loop_windowed(st_d, one_tile, C, g_coarse, part, shapes)
    assert np.array_equal(st_d.w, w0), "plane 0's window reached the host"
    assert np.array_equal(st_d.vel_scale, vs0)


def test_rebasing_refuses_a_brick_outside_the_window():
    from inexor.device.decode import tile_decode_plan

    cfg, st, members, one_tile, C, g_coarse = _setup("float64", "float32")
    shapes = _shapes(cfg, st)
    nb, pad, span, per = dwin.plane_geometry(st, cfg.n_tile, cfg._b_realized, cfg.n_brick)
    win = dwin.stage_window(st, dwin.window_slabs(0, per, pad, span, nb), shapes["window"])
    far = next(t for t in cfg.tiles if t[0] == 2)
    with pytest.raises(ValueError, match="outside the window"):
        dwin.rebase_plan(tile_decode_plan(st, np.asarray(members[far])), win, nb)


def test_a_window_past_its_shapes_is_refused():
    cfg, st, members, one_tile, C, g_coarse = _setup("float64", "float32")
    nb, pad, span, per = dwin.plane_geometry(st, cfg.n_tile, cfg._b_realized, cfg.n_brick)
    with pytest.raises(ValueError, match="rows="):
        dwin.stage_window(st, dwin.window_slabs(0, per, pad, span, nb), dict(rows=8, arena=8))


# ------------------------------------------------------------------ through the engine


def _ecfg(**kw):
    from tests.test_engine_device_step import _cfg

    return _cfg(**kw)


def test_a_windowed_engine_run_is_bitwise_the_whole_state_run():
    from inexor.device import window
    from tests.test_engine_device_backend import _coeffs, _same, _state

    co = _coeffs(3)
    kw = dict(coarse_backend="device", tile_backend="device", migrate_backend="device")
    cfg_a, cfg_b = _ecfg(device_tile_window=False, **kw), _ecfg(device_tile_window=True, **kw)
    st_a, st_b = _state(cfg_a), _state(cfg_b)
    c0 = window.CALLS
    out_a = engine.run(st_a, cfg_a, co)
    assert window.CALLS == c0, "the whole-state run used the window"
    out_b = engine.run(st_b, cfg_b, co)
    assert window.CALLS - c0 == 3, "the window did not run every step"
    _same(st_a, st_b, "window vs whole state")
    assert sum(o["n_arena_overflow"] for o in out_b) > 0, "VACUOUS: nothing spilled"
    assert all("window" in o["device_shapes"] for o in out_b)
    assert all("window" not in o["device_shapes"] for o in out_a)


def test_a_windowed_run_resumes_bitwise_with_its_window_shapes(tmp_path):
    from tests.test_engine import _rows
    from tests.test_engine_device_backend import _coeffs, _state

    co = _coeffs(4)
    kw = dict(coarse_backend="device", tile_backend="device", migrate_backend="device",
              device_tile_window=True)
    ref = _state(_ecfg(**kw), with_ids=False)
    engine.run(ref, _ecfg(**kw), co)
    d = str(tmp_path / "ck")
    st = _state(_ecfg(checkpoint_dir=d, checkpoint_every=2, **kw), with_ids=False)
    out = engine.run(st, _ecfg(checkpoint_dir=d, checkpoint_every=2, **kw), co, stop_at=2)
    st_r, resume = engine.load_checkpoint(d, _ecfg(**kw), co, arena_frac=0.05)
    assert resume["device_shapes"]["window"] == out[-1]["device_shapes"]["window"]
    engine.run(st_r, _ecfg(**kw), co, resume=resume)
    np.testing.assert_array_equal(_rows(ref), _rows(st_r))


def test_validate_refuses_a_window_without_the_compiled_device_tile():
    with pytest.raises(ValueError, match="device_tile_window"):
        _ecfg(device_tile_window=True).validate()
    with pytest.raises(ValueError, match="device_tile_window"):
        _ecfg(device_tile_window=True, tile_backend="device", device_tile_jit=False).validate()


def test_the_window_is_the_default_for_the_compiled_device_tile_and_inert_elsewhere():
    assert _ecfg().validate() is True, "the default refused a host-lane config"
    assert _ecfg().tile_window is False
    assert _ecfg(tile_backend="device", device_tile_jit=False).tile_window is False
    assert _ecfg(tile_backend="device").tile_window is True
    assert _ecfg(tile_backend="device", device_tile_window=False).tile_window is False
    assert _ecfg(tile_backend="device", device_tile_window=True).validate() is True


# ------------------------------------------------------------------ no host copy of a window


def _traced_peak(fn):
    import tracemalloc

    fn()  # warm: the first call compiles, and compiling allocates on the host
    tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    try:
        out = fn()
        return tracemalloc.get_traced_memory()[1] - base, out
    finally:
        tracemalloc.stop()


def test_staging_a_window_allocates_no_host_window(monkeypatch):
    """gb 1002020 died with four cards each holding a 45 GiB numpy window on the host.
    Slabs now go up as views; the copied fallback is the control that must fail."""
    cfg, st, members, one_tile, C, g_coarse = _setup("float64", "float32")
    shapes = _shapes(cfg, st)
    nb, pad, span, per = dwin.plane_geometry(st, cfg.n_tile, cfg._b_realized, cfg.n_brick)
    slabs = dwin.window_slabs(0, per, pad, span, nb)
    W, A = shapes["window"]["rows"], shapes["window"]["arena"]
    window_bytes = (W + A) * (st.off.itemsize * 3 + st.w.itemsize * 3)
    peak, win = _traced_peak(lambda: dwin.stage_window(st, slabs, shapes["window"]))
    assert win["copied_slabs"] == 0 and win["n_res"] > 0
    assert peak < 0.25 * window_bytes, f"staging held {peak} B against a {window_bytes} B window"
    monkeypatch.setattr(dwin, "_slab_fits", lambda *a: False)
    ctl, win_c = _traced_peak(lambda: dwin.stage_window(st, slabs, shapes["window"]))
    assert win_c["copied_slabs"] == len(slabs)
    assert ctl >= 0.25 * window_bytes, f"CONTROL cannot fail: copied staging held {ctl} B"


def test_the_write_back_downloads_a_slab_at_a_time():
    from inexor.device import migrate

    cfg, st, members, one_tile, C, g_coarse = _setup("float64", "float32")
    shapes = _shapes(cfg, st)
    nb, pad, span, per = dwin.plane_geometry(st, cfg.n_tile, cfg._b_realized, cfg.n_brick)
    win = dwin.stage_window(st, dwin.window_slabs(0, per, pad, span, nb), shapes["window"])
    w_dev = win["dev"]["w"]
    whole = int(w_dev.shape[0]) * st.w.itemsize * 3
    st_a, st_b = copy.deepcopy(st), copy.deepcopy(st)
    before = migrate.READS.get("window: write-back", 0)
    peak, _ = _traced_peak(lambda: dwin._write_core(st_a, win, w_dev, 0, per, nb))
    reads = migrate.READS.get("window: write-back", 0) - before
    # the whole-window download this replaced, as the reference both must equal
    dwin_rows = np.asarray(w_dev)
    edges = win["edges"]
    for s in range(0, per):
        lo, hi, r = int(edges[s]), int(edges[s + 1]), int(win["slab_win"][s])
        st_b.w[lo:hi] = dwin_rows[r:r + hi - lo]
    b = win["res_bricks"]
    k = np.flatnonzero((b >= 0) & (b < per * nb * nb))
    st_b.w[win["res_slots"][k]] = dwin_rows[win["W"] + k]
    assert np.array_equal(st_a.w, st_b.w), "the per-slab write-back differs"
    assert len(k) > 0, "VACUOUS: no core resident was written back"
    # THE BOUND IS WHAT THE CODE DOWNLOADS, not a laptop reading: on the CPU backend a
    # download aliases the device buffer and allocates ~nothing, which is how a bound of
    # a quarter of the window passed on the laptop and failed on a GB200 (1002227, 43.8
    # KB). Per plane the write-back holds one core slab's ladder of `w`, then the core
    # residents' ladder of `w` and its int64 index; 1.5x covers the reading's own copies.
    slab = max(dwin._ladder(int(edges[s + 1] - edges[s])) for s in range(per))
    res = dwin._ladder(len(k))
    expected = slab * st.w.itemsize * 3 + res * (st.w.itemsize * 3 + 8)
    assert 1.5 * expected < 0.5 * whole, "VACUOUS: a slab is most of this window"
    assert peak <= 1.5 * expected, (
        f"write-back held {peak} B against the {expected} B one slab + residents download")
    assert reads == 2 * (per + 1), f"{reads} reads: two write-back passes of {per} slabs + residents"


def test_a_windowed_loop_with_every_slab_copied_is_bitwise_the_whole_state_loop(monkeypatch):
    cfg, st, members, one_tile, C, g_coarse = _setup("float64", "float32")
    shapes = _shapes(cfg, st)
    st_whole, st_win = copy.deepcopy(st), copy.deepcopy(st)
    dtile.tile_loop_device(st_whole, one_tile, C, g_coarse, members, dtile.tile_step_shapes(st))
    monkeypatch.setattr(dwin, "_slab_fits", lambda *a: False)
    dwin.tile_loop_windowed(st_win, one_tile, C, g_coarse, members, shapes)
    assert np.array_equal(st_win.w, st_whole.w)
    assert np.array_equal(st_win.vel_scale, st_whole.vel_scale)
