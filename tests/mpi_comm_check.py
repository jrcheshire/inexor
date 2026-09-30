"""`MPIComm` on real processes: `mpiexec -n N python -m mpi4py tests/mpi_comm_check.py LEG`.

Legs:
- `checks`: every `tests.comm_checks` check, at the default and a 1000 B chunk size; rank 0
  prints `MPI_COMM_CHECKS_OK <N>`.
- `timeout`: rank 0 never joins a barrier, so the others' 2 s watchdog aborts the job (a
  nonzero exit within seconds).
- `raise`: rank 1 raises; `python -m mpi4py` aborts the job (a nonzero exit) while the others
  wait in a barrier.

Run by `tests/test_comm_mpi.py` wherever mpi4py and `mpiexec` exist.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inexor.comm import DEFAULT_CHUNK_BYTES, MPIComm  # noqa: E402
from tests import comm_checks  # noqa: E402


def main(leg):
    if leg == "checks":
        for chunk in (DEFAULT_CHUNK_BYTES, 1000):
            c = MPIComm(chunk_bytes=chunk, timeout=60.0)
            for check in comm_checks.CHECKS:
                check(c)
        c.barrier()
        if c.rank == 0:
            print(f"MPI_COMM_CHECKS_OK {c.size}", flush=True)
    elif leg == "timeout":
        c = MPIComm(timeout=2.0)
        if c.rank == 0:
            time.sleep(60.0)
        else:
            c.barrier()
    elif leg == "raise":
        c = MPIComm(timeout=60.0)
        c.barrier()
        if c.rank == 1:
            raise RuntimeError("injected failure on rank 1")
        c.barrier()
    else:
        raise SystemExit(f"unknown leg {leg!r}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "checks")
