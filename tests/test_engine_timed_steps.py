"""D7: `engine.run(timed_steps=)` -- the synced per-phase breakdown of the device
passes on named steps only, and a timed run bitwise an untimed one."""

import pytest

jax = pytest.importorskip("jax")

from inexor import engine  # noqa: E402
from tests.test_engine_device_backend import _coeffs, _same  # noqa: E402
from tests.test_engine_device_backend import _state as _estate  # noqa: E402
from tests.test_engine_device_step import _cfg, _same_stats  # noqa: E402

KW = dict(coarse_backend="device", tile_backend="device", migrate_backend="device")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.mark.parametrize("cards,fused", [(1, None), (4, None), (1, False)])
def test_a_timed_step_is_bitwise_and_the_only_one_with_a_breakdown(cards, fused):
    if len(jax.devices()) < cards:
        pytest.skip(f"needs {cards} jax devices")
    co = _coeffs(3)
    cfg = _cfg(**KW, device_cards=cards, **({} if fused is None else
                                            dict(migrate_repack_fused=fused)))
    st_a, st_b = _estate(cfg), _estate(cfg)
    out_a = engine.run(st_a, cfg, co)
    out_b = engine.run(st_b, cfg, co, timed_steps=(2,))
    _same(st_a, st_b, "timed vs untimed")
    _same_stats(out_a, out_b, drop=("timings", "coarse_jit_traces"))
    assert [o["timings"] is not None for o in out_b] == [False, False, True]
    assert all(o["timings"] is None for o in out_a)
    t = out_b[2]["timings"]
    assert sorted(t["tile"]) == [f"card {k}" for k in range(cards)]
    for card in t["tile"].values():
        for key in ("window: stage", "window: tiles", "window: guards", "window: write-back"):
            assert card.get(key, 0) > 0, (key, card)
    for key in ("forward: pass1", "forward: pass2", "multiply", "inverse: pass1",
                "inverse: pass2"):
        assert t["solve"].get(key, 0) > 0, (key, t["solve"])
    census = [c.get("window: census", 0) for c in t["tile"].values()]
    if cfg.fused_pass:
        assert all(v > 0 for v in census), census
        assert any("fused: block program" in k for k in t["migrate"]), sorted(t["migrate"])
        assert "repack" not in t
    else:
        assert not any(census)
        assert t["migrate"] and t["repack"]
    tc = out_b[2]["tile_cards"][0]
    assert tc["window_slabs"] > 0 and "window_live_rows_max" in tc
