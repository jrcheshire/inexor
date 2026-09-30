"""Self-checking exchanges for any `Comm`: each function runs on one rank and asserts what that
rank received, so the same checks run on loopback ranks (`tests/test_comm.py`) and on MPI
processes (`tests/mpi_comm_check.py`)."""

import numpy as np

from inexor.comm import exchange_neighbours


def _payload(src, dst, count):
    """Content that names its sender, receiver and index, so a misplaced chunk shows."""
    return (np.arange(count, dtype=np.int64) + 1_000_000 * src + 1_000 * dst).astype(np.int64)


def _count(src, dst):
    return ((src * 7 + dst * 3) % 5) * 300  # zeros included


def ring_sendrecv(c):
    right, left = (c.rank + 1) % c.size, (c.rank - 1) % c.size
    recv = np.full(1000, np.nan)
    c.Sendrecv(np.arange(1000, dtype=np.float64) + 1e6 * c.rank, right, recv, left, tag=3)
    np.testing.assert_array_equal(recv, np.arange(1000.0) + 1e6 * left)


def ragged_alltoallv(c):
    """Zero-length messages, and messages of many chunks at small `chunk_bytes`."""
    send = [_payload(c.rank, j, _count(c.rank, j)) for j in range(c.size)]
    recv = [np.full(_count(i, c.rank), -1, np.int64) for i in range(c.size)]
    c.Alltoallv(send, recv)
    for i in range(c.size):
        np.testing.assert_array_equal(recv[i], _payload(i, c.rank, _count(i, c.rank)))


def collectives(c):
    got = c.allgather(dict(rank=c.rank, blob=list(range(c.rank))))
    assert got == [dict(rank=r, blob=list(range(r))) for r in range(c.size)]
    assert c.bcast(("root", c.rank), root=c.size - 1) == ("root", c.size - 1)
    # a float sum folded in rank order: the same bits on every rank and implementation
    x = 0.1 * (c.rank + 1) + 1e-17 * c.rank
    want = 0.1
    for r in range(1, c.size):
        want = want + (0.1 * (r + 1) + 1e-17 * r)
    assert c.allreduce(x) == want
    assert c.allreduce(c.rank, "max") == c.size - 1
    np.testing.assert_array_equal(c.allreduce(np.arange(3) * (c.rank + 1), "max"),
                                  np.arange(3) * c.size)


def object_sendrecv(c):
    right, left = (c.rank + 1) % c.size, (c.rank - 1) % c.size
    got = c.sendrecv(dict(src=c.rank, big="x" * (1000 * c.rank)), right, left, tag=5)
    assert got == dict(src=left, big="x" * (1000 * left))
    got = c.sendrecv(None, left, right)
    assert got is None


def _to_left(r):
    out = dict(x=np.arange(10 + r, dtype=np.int64) + 1000 * r,
               tag=np.full((2, 3), r, dtype=np.float32))
    if r % 2 == 0:
        out["even"] = np.array([r], dtype=np.int16)
    return out


def _to_right(r):
    return dict(x=np.arange(20 + r, dtype=np.int64) + 1000 * r + 500,
                y=np.full(r, r, dtype=np.uint8))


def _same(got, want):
    assert sorted(got) == sorted(want), (sorted(got), sorted(want))
    for k in want:
        assert got[k].dtype == want[k].dtype and got[k].shape == want[k].shape, k
        np.testing.assert_array_equal(got[k], want[k])


def neighbour_arrays(c):
    """Named arrays to each side; names present on some ranks only; empty arrays; at two ranks
    both neighbours are the same rank and the two directions must not mix."""
    left, right = (c.rank - 1) % c.size, (c.rank + 1) % c.size
    from_left, from_right = exchange_neighbours(c, _to_left(c.rank), _to_right(c.rank))
    _same(from_left, _to_right(left))
    _same(from_right, _to_left(right))


def ledger(c):
    """The ledger counts each public call once (not the calls nested inside it) with the bytes
    this rank sent to other ranks, and `take_ledger` resets it."""
    import pickle

    c.take_ledger()
    right, left = (c.rank + 1) % c.size, (c.rank - 1) % c.size
    c.Sendrecv(np.zeros(1000), right, np.zeros(1000), left)
    c.Alltoallv([_payload(c.rank, j, _count(c.rank, j)) for j in range(c.size)],
                [np.zeros(_count(i, c.rank), np.int64) for i in range(c.size)])
    c.allreduce(c.rank)
    obj = dict(src=c.rank)
    c.sendrecv(obj, right, left)
    got = c.take_ledger()
    other = c.size > 1
    want_bytes = dict(
        Sendrecv=8000 * other,
        Alltoallv=sum(8 * _count(c.rank, j) for j in range(c.size) if j != c.rank),
        allreduce=0,
        sendrecv=(len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)) + 8) * other)
    assert sorted(got["ops"]) == sorted(want_bytes), got
    for op, b in want_bytes.items():
        e = got["ops"][op]
        assert (e["calls"], e["bytes"]) == (1, b), (op, e)
        assert e["seconds"] >= 0.0
    assert got["wait_s"] >= 0.0
    assert c.take_ledger() == dict(ops={}, wait_s=0.0)


CHECKS = [ring_sendrecv, ragged_alltoallv, collectives, object_sendrecv, neighbour_arrays,
          ledger]
