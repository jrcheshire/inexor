"""The exchange interface between ranks (one process per node), and two implementations.

`Comm` carries every inter-rank exchange: small collectives on picklable objects (`barrier`,
`bcast`, `allgather`, `allreduce`) and buffer exchanges on contiguous numpy arrays
(`Sendrecv`, `Alltoallv`) with mpi4py's semantics: the receiver preallocates, sizes must
match, and messages between a pair arrive in the order sent. Call it from the rank's main
thread only.

- `SerialComm`: one rank; a send to itself is a copy. The single-node run.
- `LoopbackComm`: ranks as threads of one process (`run_loopback`), so N-rank gates run in
  pytest. Receives copy, so aliasing cannot hide a missing exchange.
- `MPIComm`: one process per node over mpi4py, every wait polled against a watchdog.

`exchange_neighbours` moves named arrays to the left and right ranks of the x ring, the
exchange the step's ghost slabs, ghost planes and migrate hand-off use.

`allreduce` is an allgather folded in rank order in every implementation, so a float sum is
deterministic and the same on every rank and every implementation. `Alltoallv` is a pairwise
exchange in `chunk_bytes` pieces (M0 measured 40-72 GB/s at 4-256 MiB between gb nodes and
2-3 GB/s into fresh buffers), built on `Sendrecv`.

Every rank keeps a ledger of its exchanges (`take_ledger`): calls, bytes sent to other ranks
and seconds per public operation, and the seconds spent blocked waiting on other ranks.
"""

from __future__ import annotations

import pickle
import threading
import time
from collections import deque
from contextlib import contextmanager

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


#: Largest tag a caller may pass; `Alltoallv` uses negative tags internally, which MPI maps
#: past this range.
MAX_TAG = 9999


def _user_tag(tag):
    t = int(tag)
    if not 0 <= t <= MAX_TAG:
        raise ValueError(f"tag {t} is outside [0, {MAX_TAG}]")
    return t


def _chunk_bounds(nbytes, chunk):
    """[lo, hi) byte ranges of a message in `chunk`-byte pieces; one empty piece if empty."""
    n, c = int(nbytes), max(1, int(chunk))
    if n == 0:
        return [(0, 0)]
    return [(lo, min(lo + c, n)) for lo in range(0, n, c)]


class Comm:
    """The interface. Subclasses provide `rank`, `size`, `_allgather`, `_sendrecv_bytes` and
    `Abort`, and call `_reset_ledger()` on construction; the rest is built on those."""

    rank = 0
    size = 1
    chunk_bytes = DEFAULT_CHUNK_BYTES

    def _allgather(self, obj):
        raise NotImplementedError

    def _sendrecv_bytes(self, send, dest, recv, source, tag):
        raise NotImplementedError

    def Abort(self, code=1):
        raise NotImplementedError

    def _reset_ledger(self):
        self._ledger, self._wait_s, self._depth = {}, 0.0, 0

    @contextmanager
    def _entry(self, op, nbytes=0):
        """Ledger one public call; only the outermost, so nested calls count once."""
        self._depth += 1
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._depth -= 1
            if self._depth == 0:
                e = self._ledger.setdefault(op, [0, 0, 0.0])
                e[0] += 1
                e[1] += int(nbytes)
                e[2] += time.perf_counter() - t0

    def take_ledger(self):
        """This rank's exchanges since the previous call, then reset: `{"ops": {op: dict(calls,
        bytes, seconds)}, "wait_s": s}`. `bytes` counts buffer and header bytes sent to other
        ranks (object collectives count 0); `seconds` is wall time inside the call; `wait_s`
        is the time blocked waiting on other ranks, which includes both transfer and
        imbalance."""
        ops = {k: dict(calls=c, bytes=b, seconds=s) for k, (c, b, s) in self._ledger.items()}
        out = dict(ops=ops, wait_s=self._wait_s)
        self._ledger, self._wait_s = {}, 0.0
        return out

    def allgather(self, obj):
        """Every rank's `obj`, as a list in rank order."""
        with self._entry("allgather"):
            return self._allgather(obj)

    def barrier(self):
        with self._entry("barrier"):
            self._allgather(None)

    def bcast(self, obj, root=0):
        """`root`'s `obj` on every rank."""
        with self._entry("bcast"):
            return self._allgather(obj if self.rank == int(root) else None)[int(root)]

    def allreduce(self, x, op="sum"):
        """`op` ('sum', 'max', 'min') over ranks, folded in rank order."""
        with self._entry("allreduce"):
            return _fold(self._allgather(x), op)

    def Sendrecv(self, sendbuf, dest, recvbuf, source, tag=0):
        """Send `sendbuf` to `dest` and receive from `source` into `recvbuf` (same byte size as
        the sender's buffer, or the exchange raises). `tag` must be in [0, MAX_TAG]."""
        s = _bytes_view(sendbuf, "sendbuf")
        with self._entry("Sendrecv", s.size if int(dest) != self.rank else 0):
            self._sendrecv_bytes(s, int(dest), _bytes_view(recvbuf, "recvbuf"), int(source),
                                 _user_tag(tag))

    def sendrecv(self, obj, dest, source, tag=0):
        """Send the picklable `obj` to `dest`; return the object `source` sent. For small
        headers: one unchunked message each way, its size exchanged first."""
        payload = np.frombuffer(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL),
                                dtype=np.uint8)
        with self._entry("sendrecv", payload.size + 8 if int(dest) != self.rank else 0):
            n_in = np.zeros(1, dtype=np.int64)
            self.Sendrecv(np.array([payload.size], dtype=np.int64), dest, n_in, source, tag)
            buf = np.empty(int(n_in[0]), dtype=np.uint8)
            self.Sendrecv(payload, dest, buf, source, tag)
            return pickle.loads(buf.tobytes())

    def Alltoallv(self, sendbufs, recvbufs):
        """`sendbufs[j]` goes to rank j; `recvbufs[i]` is filled from rank i.

        Sizes are allgathered and checked on every rank first, so a mismatch raises on all
        ranks instead of hanging one. Pairwise exchange: at shift k, send to rank + k and
        receive from rank - k, each message in `chunk_bytes` pieces; a shift at which no
        rank sends is skipped.
        """
        n = self.size
        if len(sendbufs) != n or len(recvbufs) != n:
            raise ValueError(f"need {n} send and {n} receive buffers, got "
                             f"{len(sendbufs)} and {len(recvbufs)}")
        sv = [_bytes_view(b, f"sendbufs[{j}]") for j, b in enumerate(sendbufs)]
        rv = [_bytes_view(b, f"recvbufs[{i}]") for i, b in enumerate(recvbufs)]
        out = sum(int(b.size) for j, b in enumerate(sv) if j != self.rank)
        with self._entry("Alltoallv", out):
            self._alltoallv(sv, rv)

    def _alltoallv(self, sv, rv):
        n = self.size
        sizes = self._allgather(([int(b.size) for b in sv], [int(b.size) for b in rv]))
        bad = [(i, j) for i in range(n) for j in range(n) if sizes[i][0][j] != sizes[j][1][i]]
        if bad:
            i, j = bad[0]
            raise ValueError(
                f"Alltoallv: rank {i} sends {sizes[i][0][j]} B to rank {j}, which expects "
                f"{sizes[j][1][i]} B ({len(bad)} mismatched pair(s))")
        for k in range(n):
            if all(sizes[i][0][(i + k) % n] == 0 for i in range(n)):
                continue  # no rank sends at this shift; the sizes are shared, so all skip it
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
        self._reset_ledger()

    def _allgather(self, obj):
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
        self._reset_ledger()

    def _hub_wait(self, ready, what):
        t0 = time.perf_counter()
        try:
            self._hub.wait(self.rank, ready, what)
        finally:
            self._wait_s += time.perf_counter() - t0

    def _allgather(self, obj):
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
            self._hub_wait(lambda: len(slot["values"]) == h.n, f"collective {g}")
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
            self._hub_wait(lambda: bool(h.box.get(key_in)),
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


def allreduce_shapes(comm, shapes):
    """A dict of compiled-shape ints maximized over ranks, so every rank compiles the programs
    one rank would (every rank passes the same keys, in the same order). The key order is
    kept, since the shapes are recorded in checkpoints. Unchanged on one rank or `comm` None."""
    if comm is None or comm.size == 1:
        return dict(shapes)
    keys = list(shapes)
    top = comm.allreduce(np.array([int(shapes[k]) for k in keys], dtype=np.int64), "max")
    return {k: int(v) for k, v in zip(keys, top)}

def exchange_neighbours(comm, to_left, to_right):
    """Send the named arrays `to_left` to the left rank of the x ring and `to_right` to the
    right one; returns `(from_left, from_right)`: what the left rank sent right and what the
    right rank sent left, as fresh arrays.

    `to_left` / `to_right` map names to numpy arrays (sent C-contiguous; a name may be absent
    on some ranks, or different on every rank). Every rank must call this, in the same order.
    The names, dtypes and shapes are allgathered first; then the i-th name (sorted) of each
    side moves in two `Alltoallv` calls, one per direction, so the call count is the largest
    per-rank name count, not the union over ranks, and at two ranks (each rank both
    neighbours of the other) the directions never mix. On one rank everything is sent to
    itself as a copy; the step's callers skip it there.
    """
    n, r = comm.size, comm.rank
    left, right = (r - 1) % n, (r + 1) % n

    def head(d):
        return {k: (np.asarray(v).dtype.str, tuple(np.shape(v))) for k, v in d.items()}

    heads = comm.allgather((head(to_left), head(to_right)))
    n_slots = max(max(len(a), len(b)) for a, b in heads)
    empty = np.empty(0, dtype=np.uint8)
    out = dict(left={}, right={})
    # leftward: send to `left`, receive what `right` sent left; rightward the mirror
    sides = [(to_left, sorted(to_left), left, right, heads[right][0], sorted(heads[right][0]),
              out["right"]),
             (to_right, sorted(to_right), right, left, heads[left][1], sorted(heads[left][1]),
              out["left"])]
    for i in range(n_slots):
        for src, mine, dest, source, h, theirs, into in sides:
            sendbufs = [empty] * n
            if i < len(mine):
                sendbufs[dest] = np.ascontiguousarray(src[mine[i]])
            recvbufs = [empty] * n
            got = None
            if i < len(theirs):
                got = np.empty(h[theirs[i]][1], dtype=np.dtype(h[theirs[i]][0]))
                recvbufs[source] = got
            comm.Alltoallv(sendbufs, recvbufs)
            if got is not None:
                into[theirs[i]] = got
    return out["left"], out["right"]


class MPIComm(Comm):
    """The ranks of an MPI communicator (default `COMM_WORLD`), one process per node.

    Every wait is a nonblocking request polled against `timeout` seconds (the watchdog). A
    wait past it prints which exchange hung and calls `Abort`, because the other ranks may be
    blocked where no exception can reach them. A rank that raises ends the job through
    `python -m mpi4py`. mpi4py is imported on construction, not with this module.
    """

    def __init__(self, mpi_comm=None, chunk_bytes=DEFAULT_CHUNK_BYTES, timeout=1800.0,
                 poll_s=0.001):
        from mpi4py import MPI

        self._MPI = MPI
        self._c = MPI.COMM_WORLD if mpi_comm is None else mpi_comm
        self.rank = int(self._c.Get_rank())
        self.size = int(self._c.Get_size())
        self.chunk_bytes = int(chunk_bytes)
        self.timeout = float(timeout)
        self.poll_s = float(poll_s)
        self._reset_ledger()

    def _wait(self, reqs, what):
        """Statuses of `reqs` once all complete; aborts the job past the watchdog."""
        MPI = self._MPI
        t0 = time.monotonic()
        deadline = t0 + self.timeout
        statuses = [MPI.Status() for _ in reqs]
        while not MPI.Request.Testall(reqs, statuses):
            if time.monotonic() > deadline:
                import sys

                print(f"rank {self.rank}: waited {self.timeout:g} s in {what}; aborting the job",
                      file=sys.stderr, flush=True)
                self.Abort(3)
            time.sleep(self.poll_s)
        self._wait_s += time.monotonic() - t0
        return statuses

    def _allgather(self, obj):
        MPI = self._MPI
        payload = np.frombuffer(bytearray(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)),
                                dtype=np.uint8)
        sizes = np.zeros(self.size, dtype=np.int64)
        self._wait([self._c.Iallgather(np.array([payload.size], dtype=np.int64), sizes)],
                   "allgather (sizes)")
        displs = np.zeros(self.size, dtype=np.int64)
        np.cumsum(sizes[:-1], out=displs[1:])
        recv = np.empty(int(sizes.sum()), dtype=np.uint8)
        counts = (sizes.tolist(), displs.tolist())
        self._wait([self._c.Iallgatherv([payload, MPI.BYTE], [recv, counts, MPI.BYTE])],
                   "allgather")
        return [pickle.loads(recv[d:d + s].tobytes()) for d, s in zip(*counts[::-1])]

    def _sendrecv_bytes(self, send, dest, recv, source, tag):
        MPI = self._MPI
        # `Alltoallv`'s internal tags are negative; MPI tags are not
        t = int(tag) if tag >= 0 else MAX_TAG + 1 - int(tag)
        rreq = self._c.Irecv([recv, MPI.BYTE], source=int(source), tag=t)
        sreq = self._c.Isend([send, MPI.BYTE], dest=int(dest), tag=t)
        st_r, _st_s = self._wait([rreq, sreq], f"Sendrecv to {dest} / from {source} (tag {tag})")
        got = st_r.Get_count(MPI.BYTE)
        if got != recv.size:
            raise ValueError(f"Sendrecv: rank {source} sent {got} B into rank {self.rank}'s "
                             f"{recv.size} B buffer")

    def Abort(self, code=1):
        self._c.Abort(int(code))
