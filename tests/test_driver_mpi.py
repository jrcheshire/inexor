"""`device_run.py run / card / export --comm mpi` on real MPI processes, against one rank.

ICs for `cdev8-tile32` from `realization.py ics`; the driver runs the first 6 steps of a
40-step schedule with a checkpoint every 3 under `mpiexec -n N` (one card per rank), and
every checkpoint file at N = 2 and 4 must equal N = 1's. `D7_FAIL_AT` on rank 1 ends the
whole launch. The card and the export of the step-6 checkpoint at 2 ranks equal 1 rank's
(the card in every field; the export's parts concatenated, with the same crc32), and the
export decoded on the cards carries the host CLI export's crc32. Skipped where mpi4py or
`mpiexec` is missing; `scripts/run/mpi_lane.sh` runs it on the laptop.
"""

import json
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
    # a slightly early start: at this box's 2 Mpc/h brick slabs the a = 0.1 displacements
    # exceed the streamed generator's one-slab window (the driver's schedule still starts at
    # 0.1; the epoch does not matter to a byte comparison between rank counts)
    p = subprocess.run([sys.executable, REALIZATION, "ics", "--config", PRESET, "--workdir",
                        str(d), "--tile-workers", "1", "--a-init", "0.08"],
                       capture_output=True, text=True, env=ENV, timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return d


def _launch(n, ics, out, env=None, timeout=1200):
    cmd = [MPIEXEC, "-n", str(n), sys.executable, "-m", "mpi4py", DRIVER, "run",
           "--preset", PRESET, "--workdir", str(ics), "--card", str(out / "card.json"),
           # the first 6 steps of a 40-step schedule, as a production segment runs them
           "--cards", "1", "--k-steps", "40", "--stop-at", "6",
           # the CPU driver's roomier layout (not fingerprinted) for the early ICs
           "--slack", "0.20", "--arena-frac", "0.20",
           "--checkpoint-dir", str(out / "ckpt"), "--checkpoint-every", "3",
           "--comm", "mpi", "--comm-timeout", "300", "--beat", "600"]
    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       env=dict(ENV, **(env or {})))
    return p, time.monotonic() - t0


@pytest.fixture(scope="module")
def one_rank_run(ics, tmp_path_factory):
    out = tmp_path_factory.mktemp("one")
    p, _ = _launch(1, ics, out)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return out


@pytest.fixture(scope="module")
def one_rank(one_rank_run):
    h = hashes(one_rank_run / "ckpt")
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


def _product(kind, n, ckpt, out, extra):
    cmd = [MPIEXEC, "-n", str(n), sys.executable, "-m", "mpi4py", DRIVER, kind,
           "--preset", PRESET, "--checkpoint-dir", str(ckpt), "--k-steps", "40",
           "--expect-step", "6", "--card", str(out / f"{kind}-run.json"), "--cards", "1",
           "--slack", "0.20", "--arena-frac", "0.20", "--comm", "mpi",
           "--comm-timeout", "300", "--beat", "600"] + list(extra)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=1200, env=ENV)


def _card_args(out):
    # 8 bins up to half Nyquist of the 64^3 coarse mesh in a 64 Mpc/h box; step 6 of 40 is not
    # synchronized, which a rank-count comparison does not care about
    return ["--out", str(out / "pk.json"), "--k-max", "1.5707963267948966", "--n-bins", "8",
            "--min-weight", "1", "--allow-partial"]


def _export_bytes(d):
    from inexor import export

    import numpy as np

    head = json.load(open(os.path.join(d, export.HEADER)))
    return head, {k: b"".join(np.load(os.path.join(d, p["files"][k])).tobytes()
                              for p in head["parts"]) for k in ("x", "v")}


@pytest.fixture(scope="module")
def one_rank_products(one_rank_run, tmp_path_factory):
    out = tmp_path_factory.mktemp("prod1")
    for kind, extra in (("card", _card_args(out)),
                        ("export", ["--export-dir", str(out / "export"), "--allow-partial"])):
        p = _product(kind, 1, one_rank_run / "ckpt", out, extra)
        assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return out


def test_driver_products_at_two_ranks_are_the_one_rank_products(one_rank_run, one_rank_products,
                                                                tmp_path):
    for kind, extra in (("card", _card_args(tmp_path)),
                        ("export", ["--export-dir", str(tmp_path / "export"),
                                    "--allow-partial"])):
        p = _product(kind, 2, one_rank_run / "ckpt", tmp_path, extra)
        assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    one = json.load(open(one_rank_products / "pk.json"))
    two = json.load(open(tmp_path / "pk.json"))
    assert one["summary"]["transform"] == "cards" and one["n_ranks"] == 1
    assert two["n_ranks"] == 2
    strip = ("provenance",)
    assert ({k: v for k, v in two["summary"].items() if k not in strip}
            == {k: v for k, v in one["summary"].items() if k not in strip})
    h1, b1 = _export_bytes(one_rank_products / "export")
    h2, b2 = _export_bytes(tmp_path / "export")
    assert len(h1["parts"]) == 1 and len(h2["parts"]) == 2 and h2["decode"] == "cards"
    assert h2["crc32"] == h1["crc32"] and b2 == b1


def test_the_cards_export_carries_the_host_cli_crc(one_rank_run, one_rank_products, tmp_path):
    """`python -m inexor.export` decodes on the host at the checkpoint's recorded epoch."""
    gen = max((one_rank_run / "ckpt").glob("gen*"),
              key=lambda g: json.load(open(g / "manifest.json"))["provenance"]["step"])
    p = subprocess.run([sys.executable, "-m", "inexor.export", str(gen), str(tmp_path / "host")],
                       capture_output=True, text=True, timeout=600, env=ENV)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    host = json.load(open(tmp_path / "host" / "export.json"))
    cards = json.load(open(one_rank_products / "export" / "export.json"))
    assert host["a"] == cards["a"] and host["crc32"] == cards["crc32"]


def test_the_export_refuses_a_non_empty_directory(one_rank_run, tmp_path):
    (tmp_path / "export").mkdir()
    (tmp_path / "export" / "stray").write_text("")
    p = _product("export", 1, one_rank_run / "ckpt", tmp_path,
                 ["--export-dir", str(tmp_path / "export"), "--allow-partial"])
    assert p.returncode != 0 and "is not empty" in p.stdout + p.stderr


IC_IGNORED = ("provenance", "stage_s", "emission_s", "n_devices", "n_ranks", "stage_cleanup",
              "emission_y_blocks")


def _ic_dir(d):
    man = json.load(open(os.path.join(d, "manifest.json")))
    files = {f: open(os.path.join(d, f), "rb").read() for f in man["files"]}
    return files, {k: v for k, v in man.items() if k not in IC_IGNORED}, man


def _device_ics(n, out, extra=()):
    cmd = [MPIEXEC, "-n", str(n), sys.executable, "-m", "mpi4py", DRIVER, "ics",
           "--preset", PRESET, "--workdir", str(out / "ics"), "--card", str(out / "ics.json"),
           "--cards", "1", "--a-init", "0.08", "--batch-planes", "5", "--comm", "mpi",
           "--comm-timeout", "300", "--beat", "600", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=1200, env=ENV)


@pytest.fixture(scope="module")
def one_rank_device_ics(tmp_path_factory):
    out = tmp_path_factory.mktemp("dics1")
    p = _device_ics(1, out)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return out / "ics"


def test_driver_ics_at_two_ranks_are_the_one_rank_ics(one_rank_device_ics, tmp_path):
    p = _device_ics(2, tmp_path, ["--y-blocks", "2"])
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    f1, m1, _ = _ic_dir(one_rank_device_ics)
    f2, m2, raw2 = _ic_dir(tmp_path / "ics")
    assert raw2["n_ranks"] == 2 and raw2["emission_y_blocks"] == 2
    assert m2 == m1 and f2 == f1
    assert all((tmp_path / f"ics.rank{r}.json").exists() for r in range(2))
    assert "[rank 1] " in p.stdout and "ic_emission" in p.stdout


def test_driver_ics_are_the_realization_device_ics(one_rank_device_ics, tmp_path):
    p = subprocess.run([sys.executable, REALIZATION, "ics", "--config", PRESET, "--workdir",
                        str(tmp_path / "ics"), "--generator", "device", "--a-init", "0.08"],
                       capture_output=True, text=True, env=ENV, timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    f1, m1, _ = _ic_dir(one_rank_device_ics)
    f2, m2, _ = _ic_dir(tmp_path / "ics")
    assert m2 == m1 and f2 == f1


def test_driver_ics_refuse_an_existing_generation(one_rank_device_ics, tmp_path):
    out = tmp_path
    (out / "ics").symlink_to(one_rank_device_ics)
    p = _device_ics(1, out)
    assert p.returncode != 0 and "already holds an IC manifest" in p.stdout + p.stderr
