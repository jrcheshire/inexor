"""`Heartbeat`: the throttle, and that the long loops actually call it.

The class is trivial; what is worth pinning is that it cannot go silent for an
arbitrarily long stretch (the failure it exists to prevent) and cannot flood a
log either, and that the four loops it was built for are wired to it. A
heartbeat nobody calls is the same job as no heartbeat.
"""

import io

import numpy as np
import pytest

from inexor.progress import Heartbeat, _hms


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _hb(every=60.0):
    out, clk = io.StringIO(), _Clock()
    return Heartbeat(every=every, out=out, clock=clk), out, clk


def test_the_first_and_last_call_of_a_stage_always_print():
    hb, out, clk = _hb(every=1e9)
    hb("paint", 0, 10)
    clk.t = 0.5
    hb("paint", 10, 10)
    lines = out.getvalue().strip().splitlines()
    assert len(lines) == 2, "a stage shorter than one interval must still leave its cost"
    assert "0/10" in lines[0]
    assert "10/10" in lines[1] and "DONE" in lines[1]


def test_the_middle_is_throttled_by_wall_time_not_iteration_count():
    hb, out, clk = _hb(every=10.0)
    for i in range(1, 100):
        clk.t = float(i)          # one second per unit
        hb("paint", i, 1000)
    n = len(out.getvalue().strip().splitlines())
    # 99 s at one line per 10 s, plus the opening line; NOT 99
    assert 9 <= n <= 12, n


def test_a_new_stage_gets_its_own_clock_and_its_own_rate():
    hb, out, clk = _hb(every=1e9)
    hb("paint", 0, 4)
    clk.t = 100.0
    hb("paint", 4, 4)
    hb("fft", 0, 4)
    clk.t = 102.0
    hb("fft", 4, 4)
    lines = out.getvalue().strip().splitlines()
    assert "0:01:40" in lines[1] and "paint" in lines[1]
    # the second stage reports 2 s, not 102: a rate over both describes neither
    assert "0:00:02" in lines[3] and "fft" in lines[3]


def test_no_rate_is_quoted_before_anything_has_completed():
    """`inf/s` on the opening line reads as a measurement."""
    hb, out, _ = _hb()
    hb("paint", 0, 10)
    assert "/s" not in out.getvalue()


def test_eta_is_the_current_stages_rate():
    hb, out, clk = _hb(every=1.0)
    hb("paint", 0, 100)
    clk.t = 10.0
    hb("paint", 10, 100)          # 1/s, 90 left
    assert "ETA 0:01:30" in out.getvalue().strip().splitlines()[-1]


def test_a_zero_total_does_not_divide_by_it():
    hb, out, _ = _hb()
    hb("paint", 0, 0)
    assert "?" in out.getvalue()


def test_hms():
    assert _hms(0) == "0:00:00"
    assert _hms(3725) == "1:02:05"
    assert _hms(-5) == "0:00:00"


# ------------------------------------------------- the loops are wired to it


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, stage, done, total):
        self.calls.append((stage, int(done), int(total)))

    def stages(self):
        out = []
        for s, _, _ in self.calls:
            if not out or out[-1] != s:
                out.append(s)
        return out

    def check_monotone_and_complete(self, stage):
        seen = [(d, t) for s, d, t in self.calls if s == stage]
        assert seen, f"{stage} never reported"
        done = [d for d, _ in seen]
        assert done == sorted(done), f"{stage} counted backwards: {done}"
        assert seen[-1][0] == seen[-1][1], f"{stage} never reached its total: {seen[-1]}"
        assert seen[-1][1] > 0


def test_the_card_reports_all_three_of_its_long_stages():
    from tests.test_summary import _smoke_card, _smoke_state

    rec = _Recorder()
    st, cfg = _smoke_state()
    card = _smoke_card(st, cfg, progress=rec)
    assert card["n_bins"] > 0, "vacuous: the card must have done real work"
    assert rec.stages() == ["coarse paint", "fft plane", "fft axis0", "bin power"]
    for s in rec.stages():
        rec.check_monotone_and_complete(s)


def test_the_export_reports_its_chunks():
    from tests.test_export import _evolved_state

    rec = _Recorder()
    st = _evolved_state()
    import tempfile

    from inexor import export

    with tempfile.TemporaryDirectory() as d:
        export.write_particles(st, d, chunk_bricks=1, progress=rec)
    assert rec.stages() == ["export chunk"]
    rec.check_monotone_and_complete("export chunk")
    assert rec.calls[-1][1] == st.n_bricks, "one chunk per brick at chunk_bricks=1"


def test_progress_none_is_the_default_and_changes_no_number():
    """The knob must be neutral: same card, with and without."""
    from tests.test_summary import _smoke_card, _smoke_state

    st, cfg = _smoke_state()
    a = _smoke_card(st, cfg)
    b = _smoke_card(st, cfg, progress=_Recorder())
    np.testing.assert_array_equal(np.asarray(a["z_profile"]), np.asarray(b["z_profile"]))
    np.testing.assert_array_equal(np.asarray(a["p"]), np.asarray(b["p"]))


@pytest.mark.parametrize("every", [0.0, 1e-9])
def test_a_tiny_interval_prints_every_call_rather_than_erroring(every):
    hb, out, clk = _hb(every=every)
    for i in range(5):
        clk.t = float(i)
        hb("paint", i, 5)
    assert len(out.getvalue().strip().splitlines()) == 5
