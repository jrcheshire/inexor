"""The host-byte instrument: what it can see, and what its gate does.

Written because the RSS probe's whole cost structure came from needing a cluster,
three repeats and a sigma. This instrument's claim is that host allocation is
measurable exactly, on any machine, in one run -- so the tests are about the two
places that claim can fail: the counter not seeing numpy data at all, and the
gate being either vacuous or impossible to satisfy.

The canary facts these pin were measured before the instrument was written, not
assumed, and three of them are limitations rather than capabilities.
"""

import os
import sys
import tracemalloc

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

import v2_m6_host_bytes as hb  # noqa: E402

MIB = 1024 * 1024


# ------------------------------------------------------- what the counter sees


def test_numpy_array_data_is_visible_and_exact():
    """The premise. numpy array DATA does not go through PyMem, so a plain
    tracemalloc would miss exactly the arrays that matter; numpy routes it through
    PyTraceMalloc_Track under its own domain, which is what makes this work."""
    tracemalloc.start(1)
    try:
        base = hb.HostByteTracer._numpy_bytes()
        a = np.ones(16 * MIB // 8, dtype=np.float64)  # exactly 16 MiB
        got = hb.HostByteTracer._numpy_bytes() - base
        assert abs(got - 16 * MIB) < MIB // 4, f"saw {got} B for a 16 MiB array"
        v = a[::2]  # a view allocates no data
        assert abs(hb.HostByteTracer._numpy_bytes() - base - 16 * MIB) < MIB // 4
        del a, v
        assert abs(hb.HostByteTracer._numpy_bytes() - base) < MIB // 4, "not freed"
    finally:
        tracemalloc.stop()


def test_the_all_domain_peak_survives_a_freed_transient():
    """The property the RSS probe could not have: the high-water is maintained
    continuously, so a transient allocated and freed BETWEEN two readings is
    still counted. Stage 0's 'a maximum carries no timestamp' cannot happen."""
    tracemalloc.start(1)
    try:
        tracemalloc.reset_peak()
        tmp = np.ones(32 * MIB // 8, dtype=np.float64)
        del tmp
        _, peak = tracemalloc.get_traced_memory()
        assert peak >= 32 * MIB, "a freed transient was not caught by the peak"
    finally:
        tracemalloc.stop()


def test_xla_scratch_is_invisible_and_that_is_recorded_not_hidden():
    """A LIMITATION, pinned so nobody later reads a host figure as a total.

    A jit whose temporary is 64 MiB moves neither counter. The instrument covers
    the numpy half only, and that is affordable solely because the terms binding
    at C-gh (kick_pending, repack scratch, state) are all numpy -- if this test
    ever starts failing, the instrument got BETTER and the docstring is stale.
    """
    import jax
    import jax.numpy as jnp

    n = 64 * MIB // 4

    @jax.jit
    def big_temp(x):
        return (x * 2.0 + 1.0).sum()

    x = jnp.ones(n, dtype=jnp.float32)
    x.block_until_ready()
    tracemalloc.start(1)
    try:
        base = hb.HostByteTracer._numpy_bytes()
        tracemalloc.reset_peak()
        big_temp(x).block_until_ready()
        _, peak = tracemalloc.get_traced_memory()
        assert hb.HostByteTracer._numpy_bytes() - base < 8 * MIB
        assert peak < 32 * MIB, (
            "XLA scratch became visible; the instrument's stated blind spot moved"
        )
    finally:
        tracemalloc.stop()


# --------------------------------------------------------------- the gate


def _rep(np_peak, run_peak, series=None):
    return dict(np_peak=np_peak, run_peak=run_peak,
                np_series=series if series is not None else [1, 2, 3])


def test_the_gate_fails_when_the_engine_allocates_differently():
    ok, msg = hb.compare_repeats([_rep(100, 200), _rep(101, 200)])
    assert ok is False and "numpy PEAK" in msg


def test_the_gate_passes_a_four_byte_boundary_wobble():
    """Measured at cdev8: 79-184 of 790 boundary readings differ, by exactly 4
    bytes at `tile_long`, while the peak is identical. Gating the series would
    fail forever on one scalar -- and the fix would have been a picked tolerance,
    which is the move this project keeps retracting."""
    ok, msg = hb.compare_repeats([_rep(100, 200, [10, 20, 30]),
                                  _rep(100, 200, [10, 24, 30])])
    assert ok is True
    assert "differ by at most 4 B" in msg and "not gated" in msg


def test_the_gate_reports_the_all_domain_spread_without_gating_it():
    """Python bookkeeping scatters it ~0.1%; that is quoted, never a pass/fail."""
    ok, msg = hb.compare_repeats([_rep(100, 1000), _rep(100, 1010)])
    assert ok is True and "1.00%" in msg


def test_the_gate_refuses_a_verdict_from_one_run():
    ok, msg = hb.compare_repeats([_rep(100, 200)])
    assert ok is None and "not evaluable" in msg


def test_a_warmup_is_the_default_because_the_first_run_compiles():
    """Measured 2.09x at smoke (8.79 vs 4.22 MB): run one carries the compiler's
    own allocations. A default of 0 would silently put the compiler in the
    budget, which is the compile-contamination that already cost a wall estimate
    in v4_pricing_record.md."""
    with open(hb.__file__) as fh:
        assert '"--warmup", type=int, default=1' in fh.read()


# --------------------------------------------------------------- the phases


def test_the_tracer_refuses_a_phase_name_it_does_not_know():
    """Same guard as the RSS probe: an incomplete decomposition that prints
    cleanly is the gate-that-cannot-fail class."""
    tracemalloc.start(1)
    try:
        t = hb.HostByteTracer()
        t("coarse_paint")
        t("a_phase_nobody_declared")
        assert t.report()["unknown_phases"] == ["a_phase_nobody_declared"]
    finally:
        tracemalloc.stop()


def test_the_tracer_keeps_a_max_over_visits_not_a_last_value():
    """`tile_short` is visited once per tile per step; the peak is set by one
    visit and reporting the last would report a moment no peak occurred at."""
    tracemalloc.start(1)
    try:
        t = hb.HostByteTracer()
        held = []
        for i in range(3):
            if i == 1:
                held.append(np.ones(8 * MIB // 8, dtype=np.float64))
            t("tile_short")
            if i == 1:
                held.clear()
        e = t.report()["phases"]["tile_short"]
        assert e["visits"] == 3
        assert e["peak"] >= 8 * MIB, "the expensive visit was not retained"
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("name", ["coarse_paint", "tile_short", "migrate"])
def test_the_phase_vocabulary_is_shared_with_the_rss_probe(name):
    """Both instruments must name phases identically or their tables cannot be
    read against each other, which is the whole point of having two."""
    assert name in hb.PHASES
