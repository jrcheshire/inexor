"""The GPU driver's pure helpers: `timed_steps` (which steps `--timed-all`, `--timed-every N`
and `--timed-last` time) and `snapshot_steps_for` (`--snapshot-z` to step boundaries)."""

import importlib.util
import os

DRIVER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "scripts", "run", "device_run.py")


def _driver():
    spec = importlib.util.spec_from_file_location("device_run_helpers", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _timed_steps():
    return _driver().timed_steps


def test_every_nth_step_is_timed_on_the_absolute_cadence():
    ts = _timed_steps()
    assert ts(0, 10, every=5) == (4, 9)
    # a segment resumed at step 7 keeps the whole run's cadence
    assert ts(7, 20, every=5) == (9, 14, 19)
    assert ts(0, 10, every=1) == tuple(range(10))


def test_all_last_and_none():
    ts = _timed_steps()
    assert ts(2, 6, timed_all=True) == (2, 3, 4, 5)
    assert ts(2, 6, last=True) == (5,)
    assert ts(2, 6) == ()
    # --timed-all wins over --timed-every
    assert ts(0, 4, timed_all=True, every=3) == (0, 1, 2, 3)


def test_snapshot_redshifts_land_on_the_nearest_interior_step():
    from inexor.integrate import a_grid

    a_steps = a_grid(0.1, 1.0, 120, "log")
    got = _driver().snapshot_steps_for([2, 1, 0.5], a_steps)
    assert [s for _, s, _ in got] == [63, 84, 99]
    for z, s, z_step in got:
        assert abs(z_step - (1 / a_steps[s] - 1)) < 1e-12
        # nearer than either neighbour
        assert all(abs(a_steps[s] - 1 / (1 + z)) <= abs(a_steps[t] - 1 / (1 + z))
                   for t in (s - 1, s + 1))


def test_snapshot_redshifts_outside_the_schedule_or_on_one_step_are_refused():
    import pytest

    from inexor.integrate import a_grid

    a_steps = a_grid(0.1, 1.0, 120, "log")
    snap = _driver().snapshot_steps_for
    for zs in ([0.0], [-0.1], [9.0], [12.0]):
        with pytest.raises(ValueError, match="outside"):
            snap(zs, a_steps)
    with pytest.raises(ValueError, match="two on one step"):
        snap([1.0, 1.001], a_steps)
