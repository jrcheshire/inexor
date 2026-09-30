"""`MPIComm` on real MPI processes (`tests/mpi_comm_check.py` under `mpiexec`).

Skipped where mpi4py or `mpiexec` is missing (the default pixi env); the laptop MPI lane and
the gpu env run it.
"""

import os
import shutil
import subprocess
import sys
import time

import pytest

pytest.importorskip("mpi4py")
MPIEXEC = shutil.which("mpiexec")
pytestmark = pytest.mark.skipif(MPIEXEC is None, reason="no mpiexec on PATH")

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mpi_comm_check.py")


def _launch(n, leg, timeout=120):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    t0 = time.monotonic()
    p = subprocess.run([MPIEXEC, "-n", str(n), sys.executable, "-m", "mpi4py", SCRIPT, leg],
                       capture_output=True, text=True, timeout=timeout, env=env)
    return p, time.monotonic() - t0


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_the_shared_checks_pass_on_mpi(n):
    p, _ = _launch(n, "checks")
    assert p.returncode == 0, p.stdout + p.stderr
    assert f"MPI_COMM_CHECKS_OK {n}" in p.stdout


def test_a_hung_exchange_aborts_the_job():
    p, wall = _launch(3, "timeout")
    assert p.returncode != 0
    assert "aborting the job" in p.stderr
    assert wall < 40.0, f"the watchdog took {wall:.1f} s against a 2 s timeout"


def test_a_raising_rank_ends_the_job():
    p, wall = _launch(3, "raise")
    assert p.returncode != 0
    assert "injected failure on rank 1" in p.stderr
    assert wall < 40.0, f"the abort took {wall:.1f} s"
