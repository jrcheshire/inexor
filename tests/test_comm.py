"""`comm`: the same contract on `SerialComm` (one rank) and `LoopbackComm` (1-4 thread ranks):
buffer exchanges land exactly, collectives are deterministic and never alias, and a failing
or hanging rank ends the group instead of hanging it."""

import time

import numpy as np
import pytest

from inexor import comm as cm
from inexor.comm import CommAborted, CommTimeout, SerialComm, run_loopback
from tests import comm_checks

GROUPS = [("serial", 1), ("loopback", 1), ("loopback", 2), ("loopback", 3), ("loopback", 4)]


def _run(kind, n, fn, **kw):
    if kind == "serial":
        return [fn(SerialComm(**kw))]
    return run_loopback(n, fn, timeout=10.0, **kw)


@pytest.mark.parametrize("kind,n", GROUPS)
def test_ring_sendrecv(kind, n):
    def fn(c):
        right, left = (c.rank + 1) % c.size, (c.rank - 1) % c.size
        send = np.arange(1000, dtype=np.float64) + 1e6 * c.rank
        recv = np.full(1000, np.nan)
        c.Sendrecv(send, right, recv, left, tag=3)
        return recv

    for r, got in enumerate(_run(kind, n, fn)):
        np.testing.assert_array_equal(got, np.arange(1000.0) + 1e6 * ((r - 1) % n))


def _payload(src, dst, count):
    """Content that names its sender, receiver and index, so a misplaced chunk shows."""
    return (np.arange(count, dtype=np.int64) + 1_000_000 * src + 1_000 * dst).astype(np.int64)


def _count(src, dst):
    return ((src * 7 + dst * 3) % 5) * 300  # zeros included


@pytest.mark.parametrize("kind,n", GROUPS)
@pytest.mark.parametrize("chunk", [8, 1000, cm.DEFAULT_CHUNK_BYTES])
def test_ragged_alltoallv(kind, n, chunk):
    """Zero-length messages, and messages of many chunks (8 B chunks: one int64 each)."""
    assert any(_count(i, j) == 0 for i in range(5) for j in range(5))

    def fn(c):
        send = [_payload(c.rank, j, _count(c.rank, j)) for j in range(c.size)]
        recv = [np.full(_count(i, c.rank), -1, np.int64) for i in range(c.size)]
        c.Alltoallv(send, recv)
        return recv

    for r, got in enumerate(_run(kind, n, fn, chunk_bytes=chunk)):
        for i in range(n):
            np.testing.assert_array_equal(got[i], _payload(i, r, _count(i, r)))


def test_a_broken_chunk_offset_fails_the_exchange_test(monkeypatch):
    """The Alltoallv test can fail: chunks that skip a byte leave the sentinel behind."""
    good = cm._chunk_bounds

    def broken(nbytes, chunk):
        return [(lo + (1 if lo else 0), hi) for lo, hi in good(nbytes, chunk)]

    monkeypatch.setattr(cm, "_chunk_bounds", broken)
    with pytest.raises(AssertionError, match="Mismatched elements"):
        test_ragged_alltoallv("loopback", 3, 1000)


@pytest.mark.parametrize("kind,n", GROUPS)
def test_alltoallv_size_mismatch_raises_on_every_rank(kind, n):
    def fn(c):
        send = [np.zeros(4) for _ in range(c.size)]
        recv = [np.zeros(4 if c.rank else 5) for _ in range(c.size)]
        try:
            c.Alltoallv(send, recv)
        except ValueError as e:
            return str(e)
        return None

    got = _run(kind, n, fn)
    assert all(g is not None and "expects" in g for g in got), got


@pytest.mark.parametrize("kind,n", GROUPS)
def test_allreduce_is_a_rank_ordered_fold(kind, n):
    """Values whose float sum depends on order: every rank gets the rank-order fold."""
    vals = [1e16, 1.0, -1e16, 1.0][:n]

    def fn(c):
        return (c.allreduce(vals[c.rank], "sum"), c.allreduce(c.rank, "max"),
                c.allreduce(np.array([c.rank, -c.rank]), "min"))

    want = vals[0]
    for v in vals[1:]:
        want = want + v
    for s, mx, mn in _run(kind, n, fn):
        assert s == want
        assert mx == n - 1
        np.testing.assert_array_equal(mn, [0, -(n - 1)])


@pytest.mark.parametrize("kind,n", GROUPS)
def test_collectives_do_not_alias(kind, n):
    def fn(c):
        mine = dict(rank=c.rank, xs=[c.rank])
        got = c.allgather(mine)
        got[c.rank]["xs"].append("mutated")
        root = c.bcast(dict(v=[1, 2]) if c.rank == 0 else None, root=0)
        c.barrier()
        return mine, got, root

    for r, (mine, got, root) in enumerate(_run(kind, n, fn)):
        assert mine == dict(rank=r, xs=[r])
        assert [g["rank"] for g in got] == list(range(n))
        assert root == dict(v=[1, 2])


def test_a_raising_rank_ends_the_group_and_is_named():
    def fn(c):
        if c.rank == 1:
            raise KeyError("boom")
        c.barrier()

    t0 = time.monotonic()
    with pytest.raises(KeyError, match="boom") as ei:
        run_loopback(4, fn, timeout=30.0)
    assert time.monotonic() - t0 < 5.0
    assert any("rank 1 of 4" in note for note in ei.value.__notes__)


def test_abort_ends_the_group():
    def fn(c):
        if c.rank == 2:
            c.Abort(7)
        c.Sendrecv(np.zeros(1), (c.rank + 1) % c.size, np.zeros(1), (c.rank - 1) % c.size)

    with pytest.raises(CommAborted, match="Abort\\(7\\) on rank 2"):
        run_loopback(3, fn, timeout=30.0)


def test_a_hang_trips_the_watchdog():
    """Rank 0 waits on a message rank 1 never sends."""
    def fn(c):
        if c.rank == 0:
            c.Sendrecv(np.zeros(1), 1, np.zeros(1), 1)

    with pytest.raises(CommTimeout, match="rank 0 waited 0.5 s") as ei:
        run_loopback(2, fn, timeout=0.5)
    assert any("rank 0 of 2" in note for note in ei.value.__notes__)


def test_serial_refuses_other_ranks_and_mismatched_sizes():
    c = SerialComm()
    with pytest.raises(ValueError, match="must be 0"):
        c.Sendrecv(np.zeros(1), 1, np.zeros(1), 0)
    with pytest.raises(ValueError, match="sent into"):
        c.Sendrecv(np.zeros(2), 0, np.zeros(1), 0)
    with pytest.raises(ValueError, match="C-contiguous"):
        c.Sendrecv(np.zeros((4, 4))[:, 0], 0, np.zeros(4), 0)
    with pytest.raises(CommAborted):
        c.Abort()


@pytest.mark.parametrize("kind,n", GROUPS)
@pytest.mark.parametrize("check", comm_checks.CHECKS, ids=lambda f: f.__name__)
def test_shared_checks(kind, n, check):
    """The checks `tests/mpi_comm_check.py` runs on MPI processes, at two chunk sizes."""
    for chunk in (cm.DEFAULT_CHUNK_BYTES, 1000):
        _run(kind, n, check, chunk_bytes=chunk)


def test_mixed_neighbour_directions_fail_the_check(monkeypatch):
    """At two ranks each rank is both neighbours of the other: an exchange that sends the
    leftward arrays rightward must fail the neighbour check."""
    good = cm.exchange_neighbours

    def swapped(comm, to_left, to_right):
        return good(comm, to_right, to_left)

    monkeypatch.setattr(comm_checks, "exchange_neighbours", swapped)
    with pytest.raises(AssertionError):
        _run("loopback", 2, comm_checks.neighbour_arrays)


def test_user_tags_are_bounded():
    c = SerialComm()
    for bad in (-1, cm.MAX_TAG + 1):
        with pytest.raises(ValueError, match="tag"):
            c.Sendrecv(np.zeros(1), 0, np.zeros(1), 0, tag=bad)
