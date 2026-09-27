"""`state.occupancy_total`: whole-index counts without an index-sized host copy.

Widening the 34.4 GB uint32 bucket index (4096^3) to int64 to count it is a 68.7 GB
transient on top of `new_occ`. The helper is gated on value and allocation, and the
fused pass on its host peak against the index, with a widened count substituted as
the control that must fail.
"""

import copy
import tracemalloc

import numpy as np
import pytest

from inexor import state
from inexor.codec import T9Layout


def _widened(occ):
    return int(occ.astype(np.int64).sum())


def _traced_peak(fn):
    tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    try:
        out = fn()
        return tracemalloc.get_traced_memory()[1] - base, out
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
def test_the_count_equals_the_widened_sum_past_the_index_width(dtype):
    # every value at the dtype's max, so an accumulator of the index's own width wraps
    top = np.iinfo(dtype).max
    occ = np.full(1 << 16, top, dtype=dtype)
    occ[::7] = np.arange(len(occ[::7]), dtype=np.int64) % top
    got = state.occupancy_total(occ)
    assert type(got) is int
    assert got == _widened(occ) == sum(int(x) for x in occ)
    assert got > int(np.iinfo(dtype).max), "VACUOUS: the total fits the index's own width"


def test_the_count_allocates_nothing_index_sized():
    occ = np.random.default_rng(0).integers(0, 1 << 31, size=1 << 22, dtype=np.uint32)
    peak, got = _traced_peak(lambda: state.occupancy_total(occ))
    assert got == _widened(occ)
    assert peak < 0.01 * occ.nbytes, f"counting held {peak} B against a {occ.nbytes} B index"
    ctl, _ = _traced_peak(lambda: _widened(occ))
    assert ctl >= 2 * occ.nbytes, f"CONTROL cannot fail: the widened count held {ctl} B"


def _index_heavy_state():
    """64^3 on 16 bricks per side with one bucket per cell, a 3% arena and 30% slack:
    the bucket index (1 MB) is the largest whole-state term the fused pass holds, as
    it is at 4096^3, where a slab is 1/256 of the state rather than 1/16."""
    rng = np.random.default_rng(5)
    n = 64**3
    x = rng.uniform(0.0, 32.0, size=(n, 3))
    v = rng.normal(scale=1.0, size=(n, 3))
    return state.SlotState.build(x, v, T9Layout(32.0, 64, 1), 16, brick_slack=0.30,
                                 arena_frac=0.03, with_ids=True)


def test_the_fused_pass_holds_no_copy_of_the_whole_index(monkeypatch):
    """The fused pass keeps `new_occ` (one index) by design; everything else it holds
    is per slab, per brick or per arena row. The bar is two indexes: an int64 copy
    of the index is two more on top of `new_occ`."""
    jax = pytest.importorskip("jax")
    from inexor.device.fused import migrate_repack_device
    from tests.test_fused_migrate_repack import _census
    from tests.test_migrate_device import _c_drift

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        st0 = _index_heavy_state()
        c = _c_drift(st0, 0.9)
        census = _census(st0, c)
        index = st0.occupancy.nbytes

        def one_pass():
            st = copy.deepcopy(st0)
            peak, _ = _traced_peak(
                lambda: migrate_repack_device(st, c, census, brick_slack=0.10))
            return peak, st

        one_pass()  # warm: the first call compiles, and compiling allocates on the host
        peak, st = one_pass()
        assert state.occupancy_total(st.occupancy) + st.arena_used == st0.n_particles
        assert peak < 2 * index, f"the fused pass held {peak} B against a {index} B index"
        monkeypatch.setattr(state, "occupancy_total", _widened)
        ctl, _ = one_pass()
        assert ctl >= 2 * index, f"CONTROL cannot fail: the widened pass held {ctl} B"
    finally:
        jax.config.update("jax_enable_x64", prev)
