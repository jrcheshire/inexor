"""The destination census in the windowed tile loop.

The census counts, per brick, the rows the coming migrate will place there; a fused migrate +
repack sizes new brick ranges from it, so it must equal exactly the membership of host
`drift_and_migrate` + `repack_geometry`, on one card and split across cards. A wrong drift is
the control that must fail.
"""

import copy

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from inexor import state  # noqa: E402
from inexor.device import repack as drepack  # noqa: E402
from inexor.device import window as dwin  # noqa: E402
from tests.test_device_tile import _setup  # noqa: E402
from tests.test_device_window import _shapes  # noqa: E402
from tests.test_migrate_device import _c_drift  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _census(st, cfg, members, one_tile, C, g_coarse, c, devices=None, y_blocks=1):
    shapes = _shapes(cfg, st, y_blocks)
    if devices is None:
        out = dwin.tile_loop_windowed(st, one_tile, C, g_coarse, members, shapes, census=c,
                                      y_blocks=y_blocks)
        return out["census_counts"], out["census_slabs"], out
    from inexor.ooc_fft import partition_units

    parts = partition_units(cfg.tiles_side, len(devices), 1)
    counts, slabs = 0, 0
    for k, (a, b) in enumerate(parts):
        out = dwin.tile_loop_windowed(st, one_tile, C, g_coarse, members, shapes,
                                      planes=range(a, b), device=devices[k], census=c,
                                      y_blocks=y_blocks)
        counts = counts + out["census_counts"]
        slabs += out["census_slabs"]
    return counts, slabs, out


def _reference_counts(st, c):
    ref = copy.deepcopy(st)
    stats = state.drift_and_migrate(ref, c)
    return drepack.repack_geometry(ref, 0.10)[1], stats


@pytest.mark.parametrize("cards,y_blocks", [(None, 1), (2, 1), (4, 1), (None, 2), (None, 4),
                                            (2, 4)])
def test_the_census_is_the_membership_the_migrate_produces(cards, y_blocks):
    devices = None
    if cards is not None:
        if len(jax.devices()) < cards:
            pytest.skip(f"needs {cards} jax devices")
        devices = list(jax.devices()[:cards])
    cfg, st, members, one_tile, C, g_coarse = _setup("float64", "float32")
    c = _c_drift(st, 1.5)
    before = drepack.repack_geometry(st, 0.10)[1]
    counts, slabs, out = _census(st, cfg, members, one_tile, C, g_coarse, c, devices,
                                 y_blocks)
    want, stats = _reference_counts(st, c)

    assert slabs == int(st.bricks_per_side), f"{slabs} slabs counted"
    if devices is None:
        assert out["census_units"] == int(st.bricks_per_side) * y_blocks
    assert counts.dtype == np.int64 and counts.shape == want.shape
    n_diff = int(np.count_nonzero(counts != want))
    assert n_diff == 0, f"{n_diff} of {len(want)} bricks differ from the migrate's membership"
    assert int(counts.sum()) == st.n_particles
    assert np.count_nonzero(want != before) > 0, "VACUOUS: no brick's membership changed"
    assert stats["brick_reach_realized"] >= 2, "VACUOUS: no row crossed two slabs"
    assert stats["n_arena_overflow"] > 0, "VACUOUS: the migrate spilled nothing"
    assert st.arena_used > 0, "VACUOUS: no arena resident was in a census window"


def test_a_census_at_another_drift_does_not_match():
    cfg, st, members, one_tile, C, g_coarse = _setup("float64", "float32")
    c = _c_drift(st, 1.5)
    counts, _slabs, _out = _census(st, cfg, members, one_tile, C, g_coarse, 1.5 * c)
    want, _ = _reference_counts(st, c)
    assert np.count_nonzero(counts != want) > 0, "CONTROL cannot fail: a wrong drift matched"


def test_the_census_does_not_move_the_tile_loop():
    cfg, st, members, one_tile, C, g_coarse = _setup("float64", "float32")
    st_a, st_b = copy.deepcopy(st), copy.deepcopy(st)
    shapes = _shapes(cfg, st)
    ref = dwin.tile_loop_windowed(st_a, one_tile, C, g_coarse, members, shapes)
    out = dwin.tile_loop_windowed(st_b, one_tile, C, g_coarse, members, shapes,
                                  census=_c_drift(st, 1.5))
    for name in ("off", "w", "vel_scale", "occupancy", "brick_start", "arena_bucket"):
        assert np.array_equal(getattr(st_a, name), getattr(st_b, name)), name
    assert "census_counts" not in ref
    assert out["census_slabs"] > 0
