"""A tabulated linear P(k) through the drivers: ICs -> run -> resume -> card and export.

`device_run.py` (one process, CPU backend) makes `cdev8-tile32` ICs from a table, runs step
0 -> 1, resumes 1 -> 2, and cards and exports the step-2 checkpoint; `realization.py` makes
ICs from the same table and cards them at step 0. Every artifact must carry the table's
sha256: the IC manifests, both checkpoint generations (one written before the resume, one
after), both cards and the export header. `--pk-table` outside `ics` is refused.
"""

import dataclasses
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from inexor.config import Cosmology
from inexor.cosmology import K_TABLE_MAX, K_TABLE_MIN, LinearPkTable, linear_power, sigma_R

pytestmark = pytest.mark.slow

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = os.path.join(HERE, "scripts", "run", "device_run.py")
REALIZATION = os.path.join(HERE, "scripts", "run", "realization.py")
ENV = dict(os.environ, OMP_NUM_THREADS="1", JAX_PLATFORMS="cpu",
           XLA_FLAGS="--xla_cpu_multi_thread_eigen=false "
                     "--xla_force_host_platform_device_count=1")
PRESET = "cdev8-tile32"
# a slightly early start: this box's a = 0.1 displacements exceed the one-slab window
A_INIT = "0.08"
LAYOUT = ["--slack", "0.20", "--arena-frac", "0.20"]


def _run(cmd, ok=True):
    p = subprocess.run([sys.executable] + cmd, capture_output=True, text=True, env=ENV,
                       timeout=900)
    if ok:
        assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return p


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    """EH98 with a 3% wiggle in ln k, at the cosmology's sigma8: not EH98, so a lost
    table cannot pass for one."""
    c = Cosmology()
    k = np.geomspace(K_TABLE_MIN / 1.05, K_TABLE_MAX * 1.05, 2000)
    P = linear_power(k, c) * (1 + 0.03 * np.sin(3 * np.log(k)))
    t = LinearPkTable(k, P, "test-wiggle", dict(z=0.0, cosmology=dataclasses.asdict(c)))
    t = LinearPkTable(k, P * (c.sigma8 / sigma_R(8.0, c, backend="table", table=t)) ** 2,
                      "test-wiggle", t.meta)
    path = tmp_path_factory.mktemp("pk") / "pk.json"
    path.write_text(json.dumps(t.record()))
    return str(path), hashlib.sha256(t.k.tobytes() + t.P.tobytes()).hexdigest()


@pytest.fixture(scope="module")
def device_run(table, tmp_path_factory):
    path, _ = table
    d = tmp_path_factory.mktemp("dev")
    common = ["--preset", PRESET, "--cards", "1", "--beat", "600"]
    _run([DRIVER, "ics", *common, "--workdir", str(d / "ics"), "--card",
          str(d / "ics-card.json"), "--a-init", A_INIT, "--pk-table", path])
    run = [DRIVER, "run", *common, "--workdir", str(d / "ics"), "--k-steps", "40",
           *LAYOUT, "--checkpoint-dir", str(d / "ckpt"), "--checkpoint-every", "1"]
    _run(run + ["--card", str(d / "run0.json"), "--stop-at", "1"])
    _run(run + ["--card", str(d / "run1.json"), "--stop-at", "2", "--expect-step", "1"])
    prod = [*common, "--checkpoint-dir", str(d / "ckpt"), "--k-steps", "40",
            "--expect-step", "2", *LAYOUT]
    _run([DRIVER, "card", *prod, "--card", str(d / "card-run.json"),
          "--out", str(d / "pk.json"), "--min-weight", "1"])
    _run([DRIVER, "export", *prod, "--card", str(d / "export-run.json"),
          "--export-dir", str(d / "export"), "--allow-partial"])
    return d


def _sha(rec):
    return None if rec is None else rec.get("sha256")


def test_the_device_lane_carries_the_table_everywhere(table, device_run):
    _, sha = table
    d = device_run
    assert _sha(json.load(open(d / "ics" / "manifest.json")).get("linear_pk")) == sha
    steps = set()
    for g in ("gen0", "gen1"):
        prov = json.load(open(d / "ckpt" / g / "manifest.json"))["provenance"]
        steps.add(prov["step"])
        assert _sha(prov.get("linear_pk")) == sha, g
    assert steps == {1, 2}, "one generation from before the resume, one from after"
    card = json.load(open(d / "pk.json"))["summary"]
    assert card["linear_pk"] == dict(source="test-wiggle", sha256=sha)
    head = json.load(open(d / "export" / "export.json"))
    assert _sha(head["provenance"].get("linear_pk")) == sha


def test_the_cpu_lane_cards_its_ics_against_the_table(table, tmp_path):
    path, sha = table
    common = ["--config", PRESET, "--tile-workers", "1", "--a-init", A_INIT]
    _run([REALIZATION, "ics", *common, "--workdir", str(tmp_path / "ics"), "--pk-table", path])
    (tmp_path / "out").mkdir()
    _run([REALIZATION, "card", *common, "--workdir", str(tmp_path / "out"),
          "--ic-dir", str(tmp_path / "ics"), "--min-weight", "1"])
    card = json.load(open(tmp_path / "out" / "realization_pk_ics.json"))
    assert card["summary"]["linear_pk"] == dict(source="test-wiggle", sha256=sha)
    ic_card = json.load(open(tmp_path / "ics" / "realization_ics.json"))
    assert ic_card["manifest"]["linear_pk"] == dict(source="test-wiggle", sha256=sha)


def test_pk_table_is_refused_outside_ics(table, tmp_path):
    path, _ = table
    p = _run([REALIZATION, "run", "--config", PRESET, "--workdir", str(tmp_path),
              "--pk-table", path], ok=False)
    assert p.returncode != 0 and "--pk-table is an `ics` option" in p.stderr
