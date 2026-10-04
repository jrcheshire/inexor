"""The device migrate makes no host copies or extra reads, proved directly.

The slab window is uploaded as a view of the host state (no host copy), arena
residents go up as their own array, and the inserts' census reads counts the
eject already brought back in its one read. Each is an identity on values, so
the bitwise gates in `test_migrate_device.py` cannot see whether it happened;
these tests watch the allocation, the read count and the census directly, each
with a control that must fail. The pass's own copied-window fallback is the
allocation control; a bare device scalar read is the transfer-guard control.
"""

import copy
import tracemalloc

import numpy as np
import pytest

from inexor import state
from tests.test_migrate_device import _c_drift, _same_state, _state

jax = pytest.importorskip("jax")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def test_the_window_fits_exactly_up_to_the_end_of_the_state():
    from inexor.device.migrate import _window_fits

    assert _window_fits(100, 60, 40)
    assert not _window_fits(100, 61, 40)


def test_a_pass_mixing_view_and_copied_windows_is_bitwise_the_serial_pass(monkeypatch):
    from inexor.device import migrate

    calls = []

    def alternate(n_state_rows, s0, w_cap):
        calls.append(s0)
        return len(calls) % 2 == 0 and s0 + w_cap <= n_state_rows

    monkeypatch.setattr(migrate, "_window_fits", alternate)
    st_a = _state()
    st_b = copy.deepcopy(st_a)
    c = _c_drift(st_a, 1.9)
    direct = copied = 0
    for step in range(2):
        r_a = state.drift_and_migrate(st_a, c)
        r_b = migrate.drift_and_migrate_device(st_b, c)
        rec = r_b.pop("migrate_device")
        assert r_a == r_b, f"step {step}: stats differ"
        _same_state(st_a, st_b, f"step {step}")
        direct += rec["window_direct_slabs"]
        copied += rec["window_copied_slabs"]
    assert direct > 0 and copied > 0, f"VACUOUS: direct {direct}, copied {copied}"
    assert r_a["n_arena_overflow"] > 0, "VACUOUS: no arena residents in the windows"


def test_an_undercounted_emigrant_is_refused_by_the_census(monkeypatch):
    from inexor.device import migrate

    real = migrate._eject_scalars_program
    done = []

    def undercount(cap, nb, p3, n_y=1):
        fn = real(cap, nb, p3, n_y)

        def wrapped(*args):
            v = fn(*args)
            counts = np.asarray(v[2:])
            if not done and counts.any():
                done.append(True)
                v = v.at[2 + int(np.flatnonzero(counts)[0])].add(-1)
            return v

        return wrapped

    monkeypatch.setattr(migrate, "_eject_scalars_program", undercount)
    st = _state()
    with pytest.raises(AssertionError, match="unconsumed"):
        migrate.drift_and_migrate_device(st, _c_drift(st, 1.9))
    assert done, "VACUOUS: no count was perturbed"


def _eject_host_peaks(n_part, monkeypatch, copy_window=False):
    """Largest traced host allocation inside one `_eject_unit` on a WARM pass (the first
    pass's compiles allocate on the host), and the smallest slab's slot range in bytes
    (a copied window holds at least that). `copy_window` forces the copied fallback."""
    from inexor.device import migrate as module

    if copy_window:
        monkeypatch.setattr(module, "_window_fits", lambda *a: False)
    warm = _state(n_part=n_part, nb=4, box=float(n_part) / 2)
    module.drift_and_migrate_device(warm, _c_drift(warm, 0.9))
    real = module._eject_unit
    peaks = []

    def traced(st, *args, **kw):
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        out = real(st, *args, **kw)
        peaks.append(tracemalloc.get_traced_memory()[1] - base)
        return out

    monkeypatch.setattr(module, "_eject_unit", traced)
    st = _state(n_part=n_part, nb=4, box=float(n_part) / 2)
    row_bytes = st.off.itemsize * 3 + st.w.itemsize * 3 + st.ids.itemsize
    window_bytes = row_bytes * min(
        int(st.brick_start[hi] - st.brick_start[lo]) for lo, hi in map(st.slab_bricks, range(4)))
    tracemalloc.start()
    try:
        module.drift_and_migrate_device(st, _c_drift(st, 0.9))
    finally:
        tracemalloc.stop()
        monkeypatch.undo()
    return max(peaks), window_bytes


def test_the_eject_allocates_no_host_window(monkeypatch):
    new, window = _eject_host_peaks(64, monkeypatch)
    copied, _ = _eject_host_peaks(64, monkeypatch, copy_window=True)
    small, window_small = _eject_host_peaks(32, monkeypatch)
    # control: the copied fallback holds at least one slab's rows per eject
    assert copied >= window, f"CONTROL cannot fail: copied peak {copied} < window {window}"
    assert new < 0.1 * window, f"eject host peak {new} B against a {window} B window"
    # 8x the rows per slab at the same brick count: no growth with the slab
    assert new < 2 * small + 0.1 * window_small, f"peak {small} -> {new} B at 8x the rows"


def test_one_read_per_eject_and_none_in_the_census():
    from inexor.device import migrate

    st = _state()
    nb = int(st.bricks_per_side)
    before = dict(migrate.READS)
    with jax.transfer_guard_device_to_host("disallow"):
        migrate.drift_and_migrate_device(st, _c_drift(st, 1.9))
    reads = {k: v - before.get(k, 0) for k, v in migrate.READS.items()}
    assert reads.get("eject: scalars") == nb, reads
    assert reads.get("insert: scalars") == 3 * nb, reads
    assert not any("census" in k for k in reads), reads


def test_the_transfer_guard_bites_on_an_unlabelled_read():
    """GPU only: on the CPU backend no read is a transfer and the guard is inert, so
    the read counts above are complete only where this control passes."""
    if jax.devices()[0].platform == "cpu":
        pytest.skip("the transfer guard does not fire on the CPU backend")
    x = jax.numpy.arange(4) + 1
    with pytest.raises(Exception, match="(?i)transfer"):
        with jax.transfer_guard_device_to_host("disallow"):
            int(x[0])
