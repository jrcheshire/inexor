"""D3b R2: the engine with `migrate_backend="device"` is BITWISE the host engine.

The whole run -- lead drift, every step's migrate, the repack after each step,
a split-run resume -- against the host engine particle for particle, with the
receipts proving which pass ran. The device passes are gated bitwise against the
serial host passes in their own suites; this is the ROUTING gate: that the engine
calls them where it says it does and nowhere else.
"""


import numpy as np
import pytest

from inexor import engine, state
from inexor.codec import T9Layout
from inexor.config import Cosmology
from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

jax = pytest.importorskip("jax")

L_BOX, N_PART, N_FINE, N_COARSE, N_TILE, B_FINE = 32.0, 32, 64, 16, 16, 8
FIELDS = ("off", "w", "occupancy", "brick_start", "vel_scale", "arena_bucket", "ids")


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


def _state(cfg, seed=3, arena_frac=0.05, build_slack=0.10, with_ids=True):
    rng = np.random.default_rng(seed)
    g = (np.arange(N_PART) + 0.5) * (L_BOX / N_PART)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    x = np.mod(q + rng.normal(scale=0.25 * L_BOX / N_PART, size=q.shape), L_BOX)
    v = np.random.default_rng(seed + 1).normal(scale=0.5, size=x.shape)
    t9 = T9Layout(box_size=L_BOX, n_part=N_PART, bucket_cells=2)
    return state.SlotState.build(x, v, t9, N_FINE // cfg.n_brick, brick_slack=build_slack,
                                 arena_frac=arena_frac, with_ids=with_ids)


def _coeffs(k):
    a = a_grid(0.1, 1.0, k, "log")
    return bullfrog_float_coeffs(bullfrog_table(a, Cosmology()))


def _same(s1, s2, where):
    for f in FIELDS:
        a, b = getattr(s1, f), getattr(s2, f)
        if a is None:
            assert b is None, f"{where}: {f} None on one side only"
            continue
        n = int((np.asarray(a) != np.asarray(b)).sum())
        assert n == 0, f"{where}: {f}: {n} of {np.asarray(a).size} differ"
    assert s1.arena_base == s2.arena_base, where


def _strip(stats):
    """The per-step dict without the backend receipts, for equality."""
    out = []
    for o in stats:
        d = dict(o)
        d.pop("migrate_backend", None)
        d.pop("migrate_device", None)
        r = d.get("repack")
        if r is not None:
            r = dict(r)
            r.pop("repack_device", None)
            r.pop("scratch_bytes", None)
            d["repack"] = r
        out.append(d)
    return out


def test_a_device_backend_run_is_bitwise_the_host_run_with_receipts():
    co = _coeffs(3)
    cfg_h, cfg_d = _cfg(), _cfg(migrate_backend="device")
    st_h, st_d = _state(cfg_h), _state(cfg_d)
    from inexor.device import migrate, repack

    m0, r0 = migrate.CALLS, repack.CALLS
    out_h = engine.run(st_h, cfg_h, co)
    assert migrate.CALLS == m0 and repack.CALLS == r0, "the host run touched a device pass"
    out_d = engine.run(st_d, cfg_d, co)
    # lead drift + 3 steps; a repack after every step
    assert migrate.CALLS - m0 == 4 and repack.CALLS - r0 == 3
    _same(st_h, st_d, "after the run")
    assert _strip(out_h) == _strip(out_d)
    assert all(o["migrate_backend"] == "host" and o["migrate_device"] is None for o in out_h)
    assert all(o["migrate_backend"] == "device" for o in out_d)
    assert all(o["migrate_device"]["slabs"] == st_d.bricks_per_side for o in out_d)
    assert all("repack_device" in o["repack"] for o in out_d)
    assert sum(o["n_arena_overflow"] for o in out_d) > 0, "VACUOUS: nothing spilled"


def test_the_device_backend_beside_a_tile_pool_still_runs_the_device_pass():
    """A pool may drive the tile loop; the migrate is the device's, not the pool's."""
    co = _coeffs(2)
    cfg_h, cfg_d = _cfg(), _cfg(migrate_backend="device", tile_workers=2)
    st_h, st_d = _state(cfg_h), _state(cfg_d)
    engine.run(st_h, cfg_h, co)
    out_d = engine.run(st_d, cfg_d, co)
    _same(st_h, st_d, "pooled tiles + device migrate")
    assert all(o["migrate_pooled_workers"] == 0 for o in out_d)
    assert all(o["migrate_backend"] == "device" for o in out_d)
    assert all(o["tile_workers"] == 2 for o in out_d)


def test_a_device_backend_run_split_and_resumed_is_bitwise_the_uninterrupted_one(tmp_path):
    co = _coeffs(4)
    ref = _state(_cfg(migrate_backend="device"), with_ids=False)
    engine.run(ref, _cfg(migrate_backend="device"), co)
    d = str(tmp_path / "ck")
    cfg_c = _cfg(migrate_backend="device", checkpoint_dir=d, checkpoint_every=2)
    st = _state(cfg_c, with_ids=False)
    out = engine.run(st, cfg_c, co, stop_at=2)
    assert len(out) == 2 and out[-1]["checkpoint"] is not None
    st_r, resume = engine.load_checkpoint(d, _cfg(migrate_backend="device"), co,
                                          arena_frac=0.05)
    assert int(resume["step"]) == 2
    engine.run(st_r, _cfg(migrate_backend="device"), co, resume=resume)
    # rows, not raw arrays: a reloaded state's allocation geometry is
    # `_alloc_geometry`'s, not the producing run's (test_engine's `_rows`)
    from tests.test_engine import _rows

    np.testing.assert_array_equal(_rows(ref), _rows(st_r))
    # and the host engine agrees with both
    host = _state(_cfg(), with_ids=False)
    engine.run(host, _cfg(), co)
    _same(host, ref, "host vs device")


def test_the_device_budget_knob_reaches_the_pass_and_refuses():
    co = _coeffs(1)
    cfg = _cfg(migrate_backend="device", migrate_device_budget_bytes=10**12)
    st = _state(cfg)
    out = engine.run(st, cfg, co)
    assert out[0]["migrate_device"]["budget_bytes"] == 10**12
    assert 0 < out[0]["migrate_device"]["peak_estimate_bytes"] < 10**12
    cfg = _cfg(migrate_backend="device", migrate_device_budget_bytes=1024)
    st = _state(cfg)
    with pytest.raises(ValueError, match="budget of"):
        engine.run(st, cfg, co)


def test_validate_refuses_the_configurations_that_cannot_apply():
    with pytest.raises(ValueError, match="migrate_backend"):
        _cfg(migrate_backend="gpu").validate()
    with pytest.raises(ValueError, match="two different migrates"):
        _cfg(migrate_backend="device", migrate_pooled=True, tile_workers=2).validate()
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="jax_enable_x64"):
            _cfg(migrate_backend="device", coarse_dtype="float32",
                 fine_dtype="float32").validate()
    finally:
        jax.config.update("jax_enable_x64", True)
    assert _cfg(migrate_backend="device").validate() is True


def test_the_host_step_terms_move_to_windows_under_the_device_backend():
    n = N_PART**3
    h = _cfg().step_bytes(n)
    d = _cfg(migrate_backend="device").step_bytes(n)
    nb = N_FINE // _cfg().n_brick
    assert d["migrate_staging"] == d["repack_scratch"] == int(round(2 * 9.0 * n / nb))
    assert d["migrate_staging"] < h["migrate_staging"]
    assert set(d) == set(h)
