"""The M-v2-6 Stage 0b peak tracer: its arithmetic, and the gates on the instrument.

The probe itself only runs on Linux (procfs), but everything that decides what it
REPORTS is pure arithmetic over readings, and that is what goes wrong. Stage 0's
gates were unreadable not because the engine misbehaved but because the reduction
differenced two maxima with no scatter beside them, so these tests pin the
reduction: the tracer keeps a MAX over visits, a sigma is refused below three
points, and both gates are two-sided against a sigma measured in the same job.

The RSS readers are monkeypatched throughout. That is the point of their being
module-level functions: a reading is a seam, so the arithmetic can be tested on
sequences chosen to break it rather than on whatever the machine happened to do.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

import v2_m6_peak_trace as tr  # noqa: E402


@pytest.fixture
def fake_rss(monkeypatch):
    """Drive the tracer from a scripted sequence of (hwm, rss) pairs."""

    seq = {"pairs": [], "i": 0, "resets": 0}

    def _hwm():
        return seq["pairs"][seq["i"]][0]

    def _rss():
        return seq["pairs"][seq["i"]][1]

    def _advance():
        seq["resets"] += 1
        seq["i"] = min(seq["i"] + 1, len(seq["pairs"]) - 1)

    monkeypatch.setattr(tr, "_hwm", _hwm)
    monkeypatch.setattr(tr, "_rss", _rss)
    monkeypatch.setattr(tr, "_reset_hwm", _advance)
    return seq


# ------------------------------------------------------------------ the tracer


def test_a_phase_visited_many_times_keeps_its_MAXIMUM(fake_rss):
    """Every `tile_*` name is visited once per tile per step. The question the
    probe exists to answer is what the peak WAS, so a phase that is cheap on
    seven tiles and expensive on the eighth must report the eighth."""
    # (hwm, rss) read at each boundary; the tracer advances one pair per reset
    fake_rss["pairs"] = [(100, 100), (150, 120), (900, 130), (140, 130), (140, 130)]
    t = tr.PhaseTracer()
    for _ in range(4):
        t("tile_short")
    p = t.report()["phases"]["tile_short"]
    assert p["visits"] == 4
    assert p["peak"] == 900, "the tracer reported something other than the maximum"


def test_the_increment_is_measured_from_the_phase_s_own_start(fake_rss):
    """`peak` and `delta` answer different questions and a phase can have a big
    one and a zero other: a phase running while someone else's memory is
    resident has a high peak and allocates nothing. Collapsing them is exactly
    the confusion that made a difference of maxima look like an attribution."""
    fake_rss["pairs"] = [(1000, 1000), (1000, 1000), (1400, 1000)]
    t = tr.PhaseTracer()
    t("coarse_paint")   # peak 1000 over a start of 1000 -> allocates nothing
    t("tile_short")     # peak 1400 over a start of 1000 -> allocates 400
    ph = t.report()["phases"]
    assert ph["coarse_paint"]["peak"] == 1000
    assert ph["coarse_paint"]["delta"] == 0
    assert ph["tile_short"]["peak"] == 1400
    assert ph["tile_short"]["delta"] == 400


def test_an_unknown_phase_name_is_reported_rather_than_absorbed(fake_rss):
    """If the engine gains a phase the probe does not know, the decomposition is
    incomplete -- and an incomplete decomposition that prints cleanly is the
    "gate that cannot fail" class. The orchestrator fails the run on this."""
    fake_rss["pairs"] = [(10, 10), (20, 10)]
    t = tr.PhaseTracer()
    t("a_phase_nobody_declared")
    assert t.report()["unknown_phases"] == ["a_phase_nobody_declared"]


def test_every_phase_the_engine_emits_is_declared():
    """The other half of the same guard, from the engine's side: the probe's
    PHASES list must cover what `engine.step`/`engine.run` actually emit."""
    import inspect

    from inexor import engine

    src = inspect.getsource(engine.step) + inspect.getsource(engine.run)
    emitted = set()
    for line in src.splitlines():
        s = line.strip()
        if s.startswith('ph("'):
            emitted.add(s.split('"')[1])
    assert emitted, "no phase boundaries found in the engine -- the hook was removed"
    assert emitted <= set(tr.PHASES), (
        f"the engine emits phases the probe does not declare: {sorted(emitted - set(tr.PHASES))}"
    )


# ------------------------------------------------------------------ the reduction


def test_a_sigma_is_refused_below_three_points():
    """A sigma from two points is a number, not an estimate. Quoting one would
    reintroduce the false precision the probe exists to remove."""
    assert tr._stats([1.0, 2.0])["sigma"] is None
    assert tr._stats([1.0, 2.0, 3.0])["sigma"] == pytest.approx(1.0)
    s = tr._stats([5.0, 1.0, 3.0])
    assert (s["median"], s["min"], s["max"], s["spread"]) == (3.0, 1.0, 5.0, 4.0)


def _agg(trace_peaks, control_peaks, phases):
    runs = {
        "trace": [dict(maxrss=p, wall_s=1.0, phases=phases, unknown_phases=[])
                  for p in trace_peaks],
        "control": [dict(maxrss=p, wall_s=1.0) for p in control_peaks],
    }
    return tr._aggregate(runs)


def _phases(top):
    return {
        "coarse_paint": dict(peak=10, delta=1, visits=1),
        "tile_short": dict(peak=top, delta=top - 10, visits=8),
    }


def test_gate_a_fails_when_the_instrument_moves_the_thing_it_measures():
    """The traced and untraced arms must agree within the scatter, or the trace
    describes an instrumented engine and nothing it says transfers."""
    # control scatter sigma = 1.0; a 5-unit shift is 5 sigma
    agg = _agg([105.0, 105.0, 105.0], [99.0, 100.0, 101.0], _phases(105))
    v = tr._verdict(agg)
    assert v["gate_a_instrument_neutral"] is False
    assert v["instrument_delta_sigma"] == pytest.approx(5.0)

    agg = _agg([100.0, 100.0, 100.0], [99.0, 100.0, 101.0], _phases(100))
    assert tr._verdict(agg)["gate_a_instrument_neutral"] is True


def test_gate_b_fails_when_no_named_phase_reaches_the_run_peak():
    """A decomposition whose largest phase falls short of the run has the peak
    living in unnamed code, and everything attributed below it is attributed to
    the wrong thing."""
    agg = _agg([99.0, 100.0, 101.0], [99.0, 100.0, 101.0], _phases(60))
    v = tr._verdict(agg)
    assert v["gate_b_phases_reach_the_peak"] is False
    assert v["top_phase"] == "tile_short"
    assert v["run_peak_minus_top_phase"] == pytest.approx(40.0)

    agg = _agg([99.0, 100.0, 101.0], [99.0, 100.0, 101.0], _phases(100))
    assert tr._verdict(agg)["gate_b_phases_reach_the_peak"] is True


def test_the_gates_report_nothing_rather_than_passing_without_a_sigma():
    """One repeat gives no scatter, so neither gate is evaluable. `None` is the
    honest verdict; `True` would be a pass nobody measured."""
    agg = _agg([100.0], [100.0], _phases(100))
    v = tr._verdict(agg)
    assert v["gate_a_instrument_neutral"] is None
    assert v["gate_b_phases_reach_the_peak"] is None


# ------------------------------------------------------------------ the platform


def test_the_probe_refuses_a_platform_with_no_high_water_counter(monkeypatch):
    """macOS has no /proc/self/clear_refs, and the umbrella record already has
    Darwin peaks reading ~3x low and one point moving 6.484 -> 9.855 GB minutes
    apart. A fallback would produce numbers that look like measurements."""
    monkeypatch.setattr(tr.sys, "platform", "darwin")
    with pytest.raises(SystemExit, match="clear_refs"):
        tr._require_linux()


# ------------------------------------------------------------------ the growth series


def test_growth_is_the_last_visit_minus_the_first_per_phase():
    """A max over visits cannot carry a trend, and the trend is the open
    question: job 445's cdev K-ladder fit 0.157 GB per step + 6.200 GB fixed,
    so something accumulates. Attribution has to be per phase -- a run-level
    slope is what Stage 0 already had and it named nothing."""
    series = [
        ("coarse_paint", 100, 10), ("tile_short", 500, 50),
        ("coarse_paint", 180, 12), ("tile_short", 505, 51),
        ("coarse_paint", 260, 11), ("tile_short", 495, 49),
    ]
    g = tr.phase_growth(series)
    assert g["coarse_paint"]["growth"] == 160, "a rising phase must show its slope"
    assert g["coarse_paint"]["visits"] == 3
    assert g["tile_short"]["growth"] == -5, "a flat phase must not be credited a slope"


def test_growth_of_a_single_visit_phase_is_zero_not_missing():
    """`lead_drift` and `repack` are visited once or twice. They must appear
    with a zero rather than drop out of the table -- a term that vanishes is
    indistinguishable from one that was never counted."""
    g = tr.phase_growth([("lead_drift", 42, 42)])
    assert g["lead_drift"] == dict(first_visit_peak=42, last_visit_peak=42,
                                   growth=0, visits=1)


def test_the_tracer_records_a_series_when_asked_and_omits_it_when_not(fake_rss):
    fake_rss["pairs"] = [(10, 10), (20, 10), (30, 10), (40, 10)]
    t = tr.PhaseTracer(series=True)
    t("coarse_paint")
    t("coarse_paint")
    assert [s[0] for s in t.series] == ["coarse_paint", "coarse_paint"]
    assert t.report()["growth"]["coarse_paint"]["visits"] == 2

    fake_rss["i"] = 0
    t2 = tr.PhaseTracer(series=False)
    t2("coarse_paint")
    assert t2.report()["series"] is None
    assert t2.report()["growth"] == {}


def test_the_aggregate_carries_growth_across_repeats():
    """The aggregation drops a phase whose growth is missing from ANY repeat,
    so the branch needs its own test: a filter that silently emptied the table
    would print a clean run with nothing in it."""
    ph = _phases(100)
    gr = {"coarse_paint": dict(growth=50, visits=5),
          "tile_short": dict(growth=-2, visits=40)}
    runs = {"trace": [dict(maxrss=p, wall_s=1.0, phases=ph, growth=gr,
                           unknown_phases=[]) for p in (99.0, 100.0, 101.0)]}
    g = tr._aggregate(runs)["trace"]["growth"]
    assert set(g) == {"coarse_paint", "tile_short"}
    assert g["coarse_paint"]["median"] == 50
