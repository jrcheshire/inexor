"""`device_run.py run --comm mpi` on real MPI processes: checkpoints byte-identical to one rank.

ICs for `cdev8-tile32` from `realization.py ics`; the driver runs K = 6 steps with a
checkpoint every 3 under `mpiexec -n N` (one card per rank), and every checkpoint file at
N = 2 and 4 must equal N = 1's. `D7_FAIL_AT` on rank 1 ends the whole launch. Skipped where
mpi4py or `mpiexec` is missing; `scripts/run/mpi_lane.sh` runs it on the laptop.
"""

import os
import shutil
import subprocess
import sys
import time

import pytest

pytest.importorskip("mpi4py")
MPIEXEC = shutil.which("mpiexec")
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(MPIEXEC is None, reason="no mpiexec on PATH")]

from tests.ranks_common import PRESET, hashes  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = os.path.join(HERE, "scripts", "run", "device_run.py")
REALIZATION = os.path.join(HERE, "scripts", "run", "realization.py")
ENV = dict(os.environ, OMP_NUM_THREADS="1", JAX_PLATFORMS="cpu",
           XLA_FLAGS="--xla_cpu_multi_thread_eigen=false "
                     "--xla_force_host_platform_device_count=1")


@pytest.fixture(scope="module")
def ics(tmp_path_factory):
    d = tmp_path_factory.mktemp("ics")
    p = subprocess.run([sys.executable, REALIZATION, "ics", "--config", PRESET, "--workdir",
                        str(d), "--tile-workers", "1"], capture_output=True, text=True,
                       env=ENV, timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return d


def _launch(n, ics, out, env=None, timeout=1200):
    cmd = [MPIEXEC, "-n", str(n), sys.executable, "-m", "mpi4py", DRIVER, "run",
           "--preset", PRESET, "--workdir", str(ics), "--card", str(out / "card.json"),
           "--cards", "1", "--k-steps", "6", "--stop-at", "6",
           "--checkpoint-dir", str(out / "ckpt"), "--checkpoint-every", "3",
           "--comm", "mpi", "--comm-timeout", "300", "--beat", "600"]
    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       env=dict(ENV, **(env or {})))
    return p, time.monotonic() - t0


@pytest.fixture(scope="module")
def one_rank(ics, tmp_path_factory):
    out = tmp_path_factory.mktemp("one")
    p, _ = _launch(1, ics, out)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    h = hashes(out / "ckpt")
    assert sorted({k.split("/")[0] for k in h}) == ["gen0", "gen1"]
    return h


@pytest.mark.parametrize("n", [2, 4])
def test_driver_checkpoints_are_the_one_rank_bytes(ics, one_rank, tmp_path, n):
    p, _ = _launch(n, ics, tmp_path)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert hashes(tmp_path / "ckpt") == one_rank
    assert all((tmp_path / f"card.rank{r}.json").exists() for r in range(n))
    assert "[rank 1] " in p.stdout


def test_a_failing_rank_ends_the_launch(ics, tmp_path):
    p, wall = _launch(2, ics, tmp_path, env=dict(D7_FAIL_AT="tile_loop", D7_FAIL_RANK="1"))
    assert p.returncode != 0
    assert "D7_FAIL_AT=tile_loop" in p.stdout + p.stderr
    assert wall < 600.0
