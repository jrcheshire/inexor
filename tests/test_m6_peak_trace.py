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
    lad = [dict(start=1.0, peak=2.0)]
    runs = {
        "trace": [dict(run_peak=p, maxrss_raw=p, wall_s=1.0, phases=phases,
                       step_ladder=lad, unknown_phases=[]) for p in trace_peaks],
        "control": [dict(run_peak=p, maxrss_raw=p, wall_s=1.0) for p in control_peaks],
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
    assert v["control_peak_minus_top_phase"] == pytest.approx(40.0)

    agg = _agg([99.0, 100.0, 101.0], [99.0, 100.0, 101.0], _phases(100))
    assert tr._verdict(agg)["gate_b_phases_reach_the_peak"] is True


def test_gate_b_is_measured_against_the_CONTROL_not_against_itself():
    """The traced run peak is the max over boundary readings, hence the max over
    phase peaks by construction. Comparing the top phase against it returns zero
    for any engine, any config, any defect -- so the reference has to be the
    uninstrumented control. This test is the guard on that: the trace arm is
    given a run peak EQUAL to its top phase (as the real probe always will) while
    the control saw 40 units more, and the gate must still fail."""
    agg = _agg([60.0, 60.0, 60.0], [99.0, 100.0, 101.0], _phases(60))
    v = tr._verdict(agg)
    assert v["gate_b_phases_reach_the_peak"] is False, (
        "gate B compared the trace arm with itself and passed vacuously"
    )
    assert v["control_peak_minus_top_phase"] == pytest.approx(40.0)


def test_gate_b_is_not_evaluable_without_a_control_arm():
    agg = tr._aggregate({"trace": [
        dict(run_peak=p, maxrss_raw=p, wall_s=1.0, phases=_phases(100),
             step_ladder=[dict(start=1.0, peak=2.0)], unknown_phases=[])
        for p in (99.0, 100.0, 101.0)]})
    v = tr._verdict(agg)
    assert v["gate_b_phases_reach_the_peak"] is None
    assert "independent reference" in v["gate_b_note"]


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
    # a process whose resident set ratchets up: EVERY absolute peak rises, and
    # only `coarse_paint`'s own increment does. Reading the absolute peak is the
    # job 446 mistake and would credit both phases with a slope.
    series = [
        ("coarse_paint", 100, 10), ("tile_short", 500, 50),
        ("coarse_paint", 700, 30), ("tile_short", 900, 50),
        ("coarse_paint", 1300, 50), ("tile_short", 1500, 50),
    ]
    g = tr.phase_growth(series)
    assert g["coarse_paint"]["growth"] == 40, "a phase allocating more each visit"
    assert g["coarse_paint"]["visits"] == 3
    assert g["tile_short"]["growth"] == 0, (
        "a phase with a constant increment must not be credited a slope just "
        "because the process around it grew"
    )


def test_growth_of_a_single_visit_phase_is_zero_not_missing():
    """`lead_drift` and `repack` are visited once or twice. They must appear
    with a zero rather than drop out of the table -- a term that vanishes is
    indistinguishable from one that was never counted."""
    g = tr.phase_growth([("lead_drift", 42, 42)])
    assert g["lead_drift"] == dict(first_visit_delta=42, last_visit_delta=42,
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
    runs = {"trace": [dict(run_peak=p, maxrss_raw=p, wall_s=1.0, phases=ph,
                           growth=gr, step_ladder=[dict(start=1.0, peak=2.0)],
                           unknown_phases=[]) for p in (99.0, 100.0, 101.0)]}
    g = tr._aggregate(runs)["trace"]["growth"]
    assert set(g) == {"coarse_paint", "tile_short"}
    assert g["coarse_paint"]["median"] == 50


# ------------------------------------------------------------------ the run peak


def test_the_run_peak_is_accumulated_not_read_at_the_end(fake_rss):
    """`clear_refs` resets `mm->hiwater_rss`, which is what BOTH `VmHWM` and
    getrusage's `ru_maxrss` report -- so after a traced run `ru_maxrss` gives
    the peak since the last boundary, not the run's. Job 446 shipped that: it
    read cdev8's traced peak as 1.824 GB against 2.010 actual, BELOW an untraced
    control, which reads as the instrument lowering the peak and was the
    instrument mismeasuring it. The max over boundaries is the run's high-water
    by construction, and costs no extra syscall."""
    fake_rss["pairs"] = [(10, 10), (100, 10), (900, 10), (120, 10), (120, 10)]
    t = tr.PhaseTracer()
    for name in ("coarse_paint", "tile_short", "migrate"):
        t(name)
    assert t.run_peak == 900, "the run peak must survive a later, smaller phase"
    assert t.report()["run_peak"] == 900


def test_the_step_ladder_splits_the_run_and_carries_the_climb():
    """The process-wide climb, reported ONCE where it belongs rather than
    restated per phase. Two rungs cannot tell a slope from the start of a curve,
    which is the whole reason job 445's 0.157 GB/step fit (K=5 and K=10 alone)
    cannot yet be extrapolated to a production K."""
    series = [
        ("lead_drift", 5, 5),
        ("coarse_paint", 100, 40), ("tile_short", 300, 200), ("migrate", 250, 0),
        ("coarse_paint", 400, 50), ("tile_short", 700, 300), ("migrate", 650, 0),
    ]
    lad = tr.step_ladder(series)
    assert len(lad) == 2, "the ladder must split at each step, not at each phase"
    assert [d["peak"] for d in lad] == [300, 700]
    assert lad[0]["start"] == 60 and lad[1]["start"] == 350
    assert tr.step_ladder([]) == []


# ---------------------------------------------- the worker/orchestrator contract
# Job 447: the per-run print line was added after job 446 ran, so it had never
# executed. It read `maxrss`, a field name belonging to Stage 0's worker whose
# card this one is not, and the smoke leg died on a KeyError one run in. The
# fields the two halves share are now declared once and checked at both ends.


def test_the_orchestrator_reads_only_fields_the_worker_promises():
    """The contract is not empty and names the field the arms are compared on.

    `run_peak` is the load-bearing one: for a traced arm `ru_maxrss` is the peak
    since the LAST boundary, so an orchestrator reading a raw maxrss would print
    and aggregate a number that is not the run's peak.
    """
    assert tr.WORKER_FIELDS, "an empty contract checks nothing"
    assert "run_peak" in tr.WORKER_FIELDS
    for f in ("pad_ladder", "coarse_pad_distinct", "cap_distinct"):
        assert f in tr.WORKER_FIELDS, f"{f} proves an A/B knob applied; it must arrive"


def test_the_per_step_check_names_its_series_and_tolerates_the_float():
    """Job 451, five legs for five: the length check swept the card for keys
    ending `_per_step`, which also matched `s_per_step` -- a FLOAT, seconds per
    step, on the card since Stage 0b -- and `len()` of a float raised after each
    leg's full run and before its card. The check must name its series, pass a
    card that also carries the float, and still catch a short series by name."""
    card = {f"{n}_per_step": [1, 2, 3] for n in tr.PER_STEP_SERIES}
    card["s_per_step"] = 0.7
    tr.check_per_step_series(card, 3)  # the job-451 card shape; must not raise
    card["cap_per_step"] = [1, 2]
    with pytest.raises(AssertionError, match="cap_per_step has 2 entries for 3"):
        tr.check_per_step_series(card, 3)


def test_a_card_missing_a_contract_field_is_named_not_swallowed():
    full = {f: 1 for f in tr.WORKER_FIELDS}
    assert tr.missing_worker_fields(full) == []
    short = dict(full)
    del short["run_peak"]
    del short["pad_ladder"]
    assert tr.missing_worker_fields(short) == ["run_peak", "pad_ladder"]
    # a card carrying MORE than the contract is fine: the trace arm adds a series
    assert tr.missing_worker_fields({**full, "series": []}) == []


# ------------------------------------------------------------------ the printout
# Jobs 447 and 448 both died in the orchestrator's print path, on two different
# stale field names, and each cost a cluster job because nothing exercised it:
# `_print` and the per-run line are the only code here that no test called. A
# printout is not a result, but a printout that RAISES destroys one -- 448 died
# before `p0._write`, so the leg produced no card at all. These tests call it.


def _print_and_capture(capsys, agg):
    tr._print("cdev", agg, tr._verdict(agg), [])
    return capsys.readouterr().out


def test_the_printout_survives_a_full_three_arm_job(capsys):
    out = _print_and_capture(capsys, _agg([99.0, 100.0, 101.0], [99.0, 100.0, 101.0],
                                          _phases(100)))
    assert "the peak is set in `tile_short`" in out
    assert "the CONTROL arm's peak exceeds it by" in out
    assert "gate_a_instrument_neutral" in out and "gate_b_phases_reach_the_peak" in out


def test_the_printout_survives_a_single_arm_job(capsys):
    """The A/B legs run `--arms trace` alone, so gate B is not evaluable and the
    keys it would have written do not exist. That must read as a note, not as a
    KeyError: job 448's four A/B legs would each have run to completion and then
    thrown away their card."""
    agg = tr._aggregate({"trace": [
        dict(run_peak=p, maxrss_raw=p, wall_s=1.0, phases=_phases(100),
             step_ladder=[dict(start=1.0, peak=2.0)], unknown_phases=[])
        for p in (99.0, 100.0, 101.0)]})
    out = _print_and_capture(capsys, agg)
    assert "the peak is set in `tile_short`" in out
    assert "independent reference" in out, "the reason gate B is silent must be printed"
    assert "None" in out


def test_the_printout_survives_a_job_with_no_sigma(capsys):
    """One repeat: no scatter, so neither gate is evaluable and the sigma the
    phase table divides by does not exist."""
    out = _print_and_capture(capsys, _agg([100.0], [100.0], _phases(100)))
    assert "tile_short" in out
