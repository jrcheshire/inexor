"""The exchange interface between ranks (one process per node), and two implementations.

`Comm` carries every inter-rank exchange: small collectives on picklable objects (`barrier`,
`bcast`, `allgather`, `allreduce`) and buffer exchanges on contiguous numpy arrays
(`Sendrecv`, `Alltoallv`) with mpi4py's semantics: the receiver preallocates, sizes must
match, and messages between a pair arrive in the order sent. Call it from the rank's main
thread only.

- `SerialComm`: one rank; a send to itself is a copy. The single-node run.
- `LoopbackComm`: ranks as threads of one process (`run_loopback`), so N-rank gates run in
  pytest. Receives copy, so aliasing cannot hide a missing exchange.

`allreduce` is an allgather folded in rank order in every implementation, so a float sum is
deterministic and the same on every rank and every implementation. `Alltoallv` is a pairwise
exchange in `chunk_bytes` pieces (M0 measured 40-72 GB/s at 4-256 MiB between gb nodes and
2-3 GB/s into fresh buffers), built on `Sendrecv`.
"""

from __future__ import annotations

import pickle
import threading
import time
from collections import deque

import numpy as np

DEFAULT_CHUNK_BYTES = 64 << 20


class CommAborted(RuntimeError):
    """Another rank failed or aborted; this rank stops at its next exchange."""


class CommTimeout(CommAborted):
    """A blocking exchange waited longer than the watchdog allows."""


def _fold(values, op):
    if op not in ("sum", "max", "min"):
        raise ValueError(f"op must be 'sum', 'max' or 'min', got {op!r}")
    acc = values[0]
    for v in values[1:]:
        if op == "sum":
            acc = acc + v
        elif isinstance(acc, np.ndarray) or isinstance(v, np.ndarray):
            acc = np.maximum(acc, v) if op == "max" else np.minimum(acc, v)
        else:
            acc = max(acc, v) if op == "max" else min(acc, v)
    return acc


def _bytes_view(buf, what):
    """A flat uint8 view of a C-contiguous array; refuses anything that would need a copy."""
    a = np.asarray(buf)
    if not a.flags.c_contiguous:
        raise ValueError(f"{what} must be C-contiguous (a copy would detach it from the caller)")
    return a.reshape(-1).view(np.uint8)


def _chunk_bounds(nbytes, chunk):
    """[lo, hi) byte ranges of a message in `chunk`-byte pieces; one empty piece if empty."""
    n, c = int(nbytes), max(1, int(chunk))
    if n == 0:
        return [(0, 0)]
    return [(lo, min(lo + c, n)) for lo in range(0, n, c)]


class Comm:
    """The interface. Subclasses provide `rank`, `size`, `allgather`, `_sendrecv_bytes` and
    `Abort`; the rest is built on those."""

    rank = 0
    size = 1
    chunk_bytes = DEFAULT_CHUNK_BYTES

    def allgather(self, obj):
        """Every rank's `obj`, as a list in rank order."""
        raise NotImplementedError

    def _sendrecv_bytes(self, send, dest, recv, source, tag):
        raise NotImplementedError

    def Abort(self, code=1):
        raise NotImplementedError

    def barrier(self):
        self.allgather(None)

    def bcast(self, obj, root=0):
        """`root`'s `obj` on every rank."""
        return self.allgather(obj if self.rank == int(root) else None)[int(root)]

    def allreduce(self, x, op="sum"):
        """`op` ('sum', 'max', 'min') over ranks, folded in rank order."""
        return _fold(self.allgather(x), op)

    def Sendrecv(self, sendbuf, dest, recvbuf, source, tag=0):
        """Send `sendbuf` to `dest` and receive from `source` into `recvbuf` (same byte size as
        the sender's buffer, or the exchange raises)."""
        self._sendrecv_bytes(_bytes_view(sendbuf, "sendbuf"), int(dest),
                             _bytes_view(recvbuf, "recvbuf"), int(source), int(tag))

    def Alltoallv(self, sendbufs, recvbufs):
        """`sendbufs[j]` goes to rank j; `recvbufs[i]` is filled from rank i.

        Sizes are allgathered and checked on every rank first, so a mismatch raises on all
        ranks instead of hanging one. Pairwise exchange: at shift k, send to rank + k and
        receive from rank - k, each message in `chunk_bytes` pieces.
        """
        n = self.size
        if len(sendbufs) != n or len(recvbufs) != n:
            raise ValueError(f"need {n} send and {n} receive buffers, got "
                             f"{len(sendbufs)} and {len(recvbufs)}")
        sv = [_bytes_view(b, f"sendbufs[{j}]") for j, b in enumerate(sendbufs)]
        rv = [_bytes_view(b, f"recvbufs[{i}]") for i, b in enumerate(recvbufs)]
        sizes = self.allgather(([int(b.size) for b in sv], [int(b.size) for b in rv]))
        bad = [(i, j) for i in range(n) for j in range(n) if sizes[i][0][j] != sizes[j][1][i]]
        if bad:
            i, j = bad[0]
            raise ValueError(
                f"Alltoallv: rank {i} sends {sizes[i][0][j]} B to rank {j}, which expects "
                f"{sizes[j][1][i]} B ({len(bad)} mismatched pair(s))")
        for k in range(n):
            dest, source = (self.rank + k) % n, (self.rank - k) % n
            sc = _chunk_bounds(sv[dest].size, self.chunk_bytes)
            rc = _chunk_bounds(rv[source].size, self.chunk_bytes)
            # one piece count per shift on every rank, so each pair posts matching messages
            pieces = max(len(_chunk_bounds(sizes[i][0][(i + k) % n], self.chunk_bytes))
                         for i in range(n))
            for c in range(pieces):
                s = sv[dest][slice(*sc[c])] if c < len(sc) else sv[dest][:0]
                r = rv[source][slice(*rc[c])] if c < len(rc) else rv[source][:0]
                self._sendrecv_bytes(s, dest, r, source, -1 - k)


def _copy_obj(obj):
    return pickle.loads(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


class SerialComm(Comm):
    """One rank. Collectives return copies; a send to rank 0 lands in the receive buffer."""

    def __init__(self, chunk_bytes=DEFAULT_CHUNK_BYTES):
        self.chunk_bytes = int(chunk_bytes)

    def allgather(self, obj):
        return [_copy_obj(obj)]

    def _sendrecv_bytes(self, send, dest, recv, source, tag):
        if dest != 0 or source != 0:
            raise ValueError(f"one rank: dest {dest} / source {source} must be 0")
        if send.size != recv.size:
            raise ValueError(f"Sendrecv: {send.size} B sent into a {recv.size} B buffer")
        recv[...] = send

    def Abort(self, code=1):
        raise CommAborted(f"Abort({code}) on the only rank")


class _Hub:
    """State shared by a loopback group: mailboxes, collective slots, the first failure."""

    def __init__(self, n, timeout):
        self.n = int(n)
        self.timeout = float(timeout)
        self.cv = threading.Condition()
        self.box = {}
        self.slots = {}
        self.failed = None

    def fail(self, rank, exc):
        with self.cv:
            if self.failed is None:
                self.failed = (int(rank), exc)
            self.cv.notify_all()

    def wait(self, rank, ready, what):
        """Block until `ready()` (called under the lock); raise if a rank failed or the
        watchdog expires."""
        deadline = time.monotonic() + self.timeout
        while True:
            if self.failed is not None:
                r, exc = self.failed
                raise CommAborted(f"rank {r} failed ({type(exc).__name__}: {exc}); "
                                  f"rank {rank} stopped in {what}")
            if ready():
                return
            left = deadline - time.monotonic()
            if left <= 0:
                err = CommTimeout(f"rank {rank} waited {self.timeout:g} s in {what}")
                self.failed = (int(rank), err)
                self.cv.notify_all()
                raise err
            self.cv.wait(min(left, 0.05))


class LoopbackComm(Comm):
    """Rank `rank` of a thread group sharing `hub`; build the group with `run_loopback`."""

    def __init__(self, hub, rank, chunk_bytes=DEFAULT_CHUNK_BYTES):
        self._hub = hub
        self.rank = int(rank)
        self.size = hub.n
        self.chunk_bytes = int(chunk_bytes)
        self._generation = 0

    def allgather(self, obj):
        h = self._hub
        g = self._generation
        self._generation += 1
        payload = _copy_obj(obj)
        with h.cv:
            if h.failed is not None:
                h.wait(self.rank, lambda: False, "allgather")
            slot = h.slots.setdefault(g, dict(values={}, read=0))
            slot["values"][self.rank] = payload
            h.cv.notify_all()
            h.wait(self.rank, lambda: len(slot["values"]) == h.n, f"collective {g}")
            out = [_copy_obj(slot["values"][r]) for r in range(h.n)]
            slot["read"] += 1
            if slot["read"] == h.n:
                del h.slots[g]
        return out

    def _sendrecv_bytes(self, send, dest, recv, source, tag):
        h = self._hub
        for r, what in ((dest, "dest"), (source, "source")):
            if not 0 <= r < h.n:
                raise ValueError(f"{what} {r} is outside [0, {h.n})")
        msg = np.array(send, copy=True)
        key_in = (source, self.rank, tag)
        with h.cv:
            if h.failed is not None:
                h.wait(self.rank, lambda: False, "Sendrecv")
            h.box.setdefault((self.rank, dest, tag), deque()).append(msg)
            h.cv.notify_all()
            h.wait(self.rank, lambda: bool(h.box.get(key_in)),
                   f"Sendrecv from rank {source} (tag {tag})")
            got = h.box[key_in].popleft()
        if got.size != recv.size:
            raise ValueError(f"Sendrecv: rank {source} sent {got.size} B into rank "
                             f"{self.rank}'s {recv.size} B buffer")
        recv[...] = got

    def Abort(self, code=1):
        err = CommAborted(f"Abort({code}) on rank {self.rank}")
        self._hub.fail(self.rank, err)
        raise err


def run_loopback(n, fn, timeout=60.0, chunk_bytes=DEFAULT_CHUNK_BYTES):
    """Run `fn(comm)` on `n` loopback ranks (threads); return the results in rank order.

    If a rank raises, every rank blocked in an exchange raises `CommAborted`, and the first
    failure is re-raised here with a note naming its rank. `timeout` bounds each blocking
    wait (the watchdog), not the whole run.
    """
    hub = _Hub(n, timeout)
    comms = [LoopbackComm(hub, r, chunk_bytes=chunk_bytes) for r in range(int(n))]
    results = [None] * int(n)

    def body(r):
        try:
            results[r] = fn(comms[r])
        except BaseException as exc:  # noqa: BLE001 -- every failure must reach the hub
            hub.fail(r, exc)

    threads = [threading.Thread(target=body, args=(r,), daemon=True,
                                name=f"loopback-rank-{r}") for r in range(int(n))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if hub.failed is not None:
        r, exc = hub.failed
        exc.add_note(f"raised on loopback rank {r} of {n}")
        raise exc
    return results
