"""`device_run.timed_steps`: the steps `--timed-all`, `--timed-every N` and `--timed-last`
give a synced per-phase breakdown."""

import importlib.util
import os

DRIVER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "scripts", "run", "device_run.py")


def _timed_steps():
    spec = importlib.util.spec_from_file_location("device_run_timed", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.timed_steps


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
