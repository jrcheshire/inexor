"""The multi-node job's shell pieces: a rank's end releases its launcher, every run leg has a
time cap and a silence limit, the gate refuses a missing or empty checkpoint, and a leg's
XLA_FLAGS is never a bare separator.

`scripts/run/rank_exec.sh` runs under a launcher that hands the rank one of its own pipes
and waits for it to close before reaping the rank, as hydra's proxy does, and under a real
`mpiexec` where there is one (`scripts/run/mpi_lane.sh`); `scripts/run/job_lib.sh` is sourced
into bash. A stub `nvidia-smi` turns the samplers on wherever the test runs. Skipped without
bash or pgrep.
"""

import json
import os
import random
import select
import shutil
import subprocess
import sys
import time

import pytest

BASH = shutil.which("bash")
MPIEXEC = shutil.which("mpiexec")
pytestmark = pytest.mark.skipif(BASH is None or shutil.which("pgrep") is None,
                                reason="needs bash and pgrep")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANK_EXEC = os.path.join(HERE, "scripts", "run", "rank_exec.sh")
JOB_LIB = os.path.join(HERE, "scripts", "run", "job_lib.sh")


def _survivors(tag):
    p = subprocess.run(["pgrep", "-f", tag], capture_output=True, text=True)
    return [int(x) for x in p.stdout.split()]


def _reap(tag):
    for pid in _survivors(tag):
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass


@pytest.fixture
def stub_bin(tmp_path):
    d = tmp_path / "bin"
    d.mkdir()
    smi = d / "nvidia-smi"
    smi.write_text("#!/bin/sh\necho '0, 1, 0'\n")
    smi.chmod(0o755)
    return d


def _read_to_eof(fd, timeout):
    """Read `fd` until every writer has closed it; False if that takes over `timeout` s."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.2)
        if ready and not os.read(fd, 65536):
            return True
    return False


def _launch_inherited(cmd, env):
    """A launcher that hands the rank one of its own pipes, as hydra's proxy does (its fds
    are not close-on-exec), and waits for that pipe and the rank's output to close."""
    r, w = os.pipe()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                         pass_fds=(w,))
    os.close(w)
    try:
        if not _read_to_eof(r, 30):
            p.kill()
            p.communicate()
            return None
    finally:
        os.close(r)
    p.communicate(timeout=30)
    return p.returncode


def _launch_mpiexec(cmd, env):
    p = subprocess.Popen([MPIEXEC, "-n", "1", *cmd], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, env=env)
    try:
        p.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        return None
    return p.returncode


@pytest.mark.parametrize("launch", [
    "inherited-fd",
    pytest.param("mpiexec", marks=pytest.mark.skipif(MPIEXEC is None, reason="no mpiexec"))])
@pytest.mark.parametrize("code", ["import os; os.abort()", "import sys; sys.exit(0)"],
                         ids=["abort", "exit0"])
def test_a_rank_that_ends_releases_its_launcher(tmp_path, stub_bin, code, launch):
    tag = str(tmp_path / "s_r@RANK@")
    env = dict(os.environ, PMI_RANK="0", PATH=f"{stub_bin}{os.pathsep}{os.environ['PATH']}")
    cmd = [BASH, RANK_EXEC, "--samples", tag, "--", sys.executable, "-c", code]
    try:
        t0 = time.monotonic()
        rc = (_launch_inherited if launch == "inherited-fd" else _launch_mpiexec)(cmd, env)
        assert rc is not None, "the launcher did not return within 30 s of the rank's end"
        assert time.monotonic() - t0 < 15
        assert (rc == 0) == code.endswith("exit(0)")
        deadline = time.monotonic() + 10
        while _survivors(str(tmp_path)) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not _survivors(str(tmp_path)), "a sampler outlived its rank"
        # the sampler ran, so the test saw the path it guards
        assert (tmp_path / "s_r0_gpu.csv").read_text().count("\n") >= 1
    finally:
        _reap(str(tmp_path))


def _lib(script, tmp_path, timeout=60, **env):
    full = f'LEG_DIR="{tmp_path}/legs"; rc_total=0; . "{JOB_LIB}"\n{script}'
    e = dict(os.environ, LEG_POLL_S="1", LEG_KILL_GRACE_S="2")
    e.pop("XLA_FLAGS", None)
    e.update(env)
    return subprocess.run([BASH, "-c", full], capture_output=True, text=True, env=e,
                          timeout=timeout)


@pytest.mark.parametrize("job,extra,seen", [
    (None, "", None), ("", "", None), (None, "--b", "--b"), ("--a", "", "--a"),
    ("--a", "--b", "--a --b")])
def test_xla_env_is_unset_or_flags(tmp_path, job, extra, seen):
    env = {} if job is None else {"XLA_FLAGS": job}
    show = f"{sys.executable} -c 'import os; print(repr(os.environ.get(\"XLA_FLAGS\")))'"
    r = _lib(f'xla_env "{extra}"; "${{XLA_ENV[@]}}" {show}', tmp_path, **env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == repr(seen)


def test_run_leg_passes_the_status_and_keeps_the_output(tmp_path):
    r = _lib('run_leg first bash -c "echo hello; exit 3"; echo "rc=$? total=$rc_total"',
             tmp_path)
    assert "rc=3 total=1" in r.stdout and "hello" in r.stdout
    assert (tmp_path / "legs" / "01-first.log").read_text() == "hello\n"
    r = _lib('run_leg ok true; echo "rc=$? total=$rc_total killed=$LEG_KILLED"', tmp_path)
    assert "rc=0 total=0 killed=0" in r.stdout


@pytest.fixture
def nap():
    """A `sleep` length no other process uses, to find a leg's survivors; any are killed."""
    n = random.randint(10_000, 99_999)
    yield n
    _reap(f"sleep {n}")


def test_a_silent_leg_is_killed_with_its_children(tmp_path, nap):
    t0 = time.monotonic()
    r = _lib(f'run_leg --quiet-limit 3 quiet bash -c "echo started; sleep {nap}; :"\n'
             'echo "rc=$? total=$rc_total killed=$LEG_KILLED"', tmp_path)
    assert time.monotonic() - t0 < 15
    assert "rc=124 total=1 killed=1" in r.stdout
    assert "printed nothing for 3 s" in r.stdout
    assert not _survivors(f"sleep {nap}")


def test_a_leg_past_its_cap_is_killed_even_while_printing(tmp_path):
    t0 = time.monotonic()
    r = _lib('run_leg --cap 3 --quiet-limit 100 busy '
             'bash -c "while :; do echo x; sleep 0.2; done"\n'
             'echo "rc=$? killed=$LEG_KILLED"', tmp_path)
    assert time.monotonic() - t0 < 15
    assert "rc=124 killed=1" in r.stdout and "ran past its 3 s cap" in r.stdout


def test_a_leg_that_ignores_term_is_killed(tmp_path, nap):
    t0 = time.monotonic()
    r = _lib(f'run_leg --cap 2 stubborn bash -c "trap \\"\\" TERM; sleep {nap}; :"\n'
             'echo "rc=$? killed=$LEG_KILLED"', tmp_path)
    assert time.monotonic() - t0 < 20
    assert "rc=124 killed=1" in r.stdout
    assert not _survivors(f"sleep {nap}")


def test_expect_fail_passes_only_a_failure_of_its_own(tmp_path, nap):
    r = _lib('run_leg --expect-fail a bash -c "exit 1"; echo "a=$?"\n'
             'run_leg --expect-fail b true; echo "b=$?"\n'
             f'run_leg --cap 2 --expect-fail c sleep {nap}; echo "c=$? total=$rc_total"',
             tmp_path)
    assert "a=0" in r.stdout and "b=1" in r.stdout and "c=1 total=2" in r.stdout


def test_the_usr1_trap_runs_during_a_leg(tmp_path):
    r = _lib("trap 'echo TRAPPED' USR1\n"
             "( sleep 1; kill -USR1 $$ ) &\n"
             'run_leg slow sleep 3; echo "rc=$?"', tmp_path)
    out = r.stdout
    assert "TRAPPED" in out and "rc=0" in out
    assert out.index("TRAPPED") < out.index("--- LEG slow")


def _ckpt(d, payload=b"x", manifest=True):
    g = d / "gen0"
    g.mkdir(parents=True)
    (g / "t9_slab_0000.npz").write_bytes(payload)
    if manifest:
        (g / "manifest.json").write_text("{}")
    return d


# each case: (first dir, second dir) as built by _ckpt, None (absent) or "" (an empty dir)
GATE_CASES = {
    "same": ({}, {}),
    "byte": ({}, {"payload": b"z"}),
    "extra": ({}, {"extra": True}),
    "missing": ({}, None),
    "empty": ({}, ""),
    "both-missing": (None, None),
    "both-empty": ("", ""),
    "no-manifest": ({"manifest": False}, {"manifest": False}),
}


def _gate_dir(d, spec):
    if spec is None:
        return d
    if spec == "":
        d.mkdir()
        return d
    _ckpt(d, payload=spec.get("payload", b"x"), manifest=spec.get("manifest", True))
    if spec.get("extra"):
        (d / "gen0" / "t9_slab_0001.npz").write_bytes(b"y")
    return d


@pytest.mark.parametrize("case", list(GATE_CASES))
def test_gate_passes_only_identical_checkpoints(tmp_path, case):
    a, b = (_gate_dir(tmp_path / n, spec) for n, spec in zip("ab", GATE_CASES[case]))
    r = _lib(f'gate test "{a}" "{b}"; echo "rc=$?"', tmp_path)
    if case == "same":
        assert "GATE test PASS" in r.stdout and "rc=0" in r.stdout
    else:
        assert "GATE test FAIL" in r.stdout and "rc=1" in r.stdout


def _cut_ckpt(d, window, step=3, payload=b"x"):
    g = d / "gen0"
    g.mkdir(parents=True)
    (g / "t9_slab_0000.npz").write_bytes(payload)
    m = {"n_particles": 8, "provenance": {"step": step, "device_shapes": {
        "tile": {"arena_rect": 4}, "window": window}}}
    (g / "manifest.json").write_text(json.dumps(m))
    return d


@pytest.mark.parametrize("case,ok", [
    ("window only", True), ("same", True), ("another manifest field", False),
    ("a slab byte", False), ("a tile shape", False)])
def test_the_any_cut_gate_ignores_only_the_window_shape(tmp_path, case, ok):
    w1 = {"rows": 100, "arena": 10}
    a = _cut_ckpt(tmp_path / "a", w1)
    b = _cut_ckpt(tmp_path / "b", {"rows": 40, "arena": 5, "y_blocks": 4}
                  if case != "same" else w1,
                  step=4 if case == "another manifest field" else 3,
                  payload=b"z" if case == "a slab byte" else b"x")
    if case == "a tile shape":
        f = b / "gen0" / "manifest.json"
        m = json.loads(f.read_text())
        m["provenance"]["device_shapes"]["tile"]["arena_rect"] = 8
        f.write_text(json.dumps(m))
    r = _lib(f'gate_any_cut test "{a}" "{b}"; echo "rc=$?"', tmp_path)
    want = ("GATE test PASS", "rc=0") if ok else ("GATE test FAIL", "rc=1")
    assert all(w in r.stdout for w in want), r.stdout + r.stderr
    plain = _lib(f'gate test "{a}" "{b}"; echo "rc=$?"', tmp_path)
    assert ("rc=0" in plain.stdout) == (case == "same"), "CONTROL: the plain gate"


def _gen(ckpt, g, step, n_particles=8):
    d = ckpt / g
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        f'{{"n_particles": {n_particles}, "provenance": {{"step": {step}, "n_steps": 120}}}}')


@pytest.mark.parametrize("steps,want", [
    ((100, 120), "gen0 100 120"), ((120, 100), "gen1 100 120"),
    ((100, None), None), ((None, None), None), ((100, 100), None)])
def test_ckpt_generations_names_the_older_generation_and_both_steps(tmp_path, steps, want):
    for g, s in zip(("gen0", "gen1"), steps):
        if s is not None:
            _gen(tmp_path / "ckpt", g, s)
    r = _lib(f'ckpt_generations "{tmp_path}/ckpt"; echo "rc=$?"', tmp_path,
             PY=sys.executable)
    if want is None:
        assert r.stdout.strip() == "rc=1", r.stdout
    else:
        assert r.stdout.split("\n")[:2] == [want, "rc=0"], r.stdout


@pytest.mark.parametrize("steps,want", [
    ((100, 120), "gen1 120"), ((120, 100), "gen0 120"), ((None, 20), "gen1 20"),
    ((None, None), None)])
def test_ckpt_newest_is_the_generation_at_the_higher_step(tmp_path, steps, want):
    for g, s in zip(("gen0", "gen1"), steps):
        if s is not None:
            _gen(tmp_path / "ckpt", g, s)
    r = _lib(f'ckpt_newest "{tmp_path}/ckpt"; echo "rc=$?"', tmp_path, PY=sys.executable)
    if want is None:
        assert r.stdout.strip() == "rc=1", r.stdout
    else:
        assert r.stdout.split("\n")[:2] == [want, "rc=0"], r.stdout


STEPS = os.path.join(HERE, "scripts", "run", "multinode_steps_vista.sbatch")


def _steps_job(tmp_path, **env):
    """The steps job script up to its first refusal; the inputs exist unless overridden.

    Its environment is only what is set here, so a job's own EXPECT_STEP or PLANT_* (this
    file runs as a guard inside the job) cannot reach it. Past a refusal it would start legs,
    this file among them, so `python`, `mpiexec` and `pixi` are stubs that fail at once, and
    the script runs in its own session, killed on timeout.
    """
    for d in ("ics", "ref", "ref_ics"):
        (tmp_path / d).mkdir(exist_ok=True)
    stubs = tmp_path / "stubs"
    stubs.mkdir(exist_ok=True)
    for name in ("python", "python3", "mpiexec", "pixi"):
        (stubs / name).write_text("#!/bin/sh\nexit 97\n")
        (stubs / name).chmod(0o755)
    e = {k: os.environ[k] for k in ("HOME", "TMPDIR") if k in os.environ}
    e["PATH"] = f"{stubs}:{os.environ.get('PATH', '')}"
    e.update(REHEARSAL="1", INEXOR_SRC=HERE, INEXOR_RUNS=str(tmp_path / "runs"),
             IC_DIR=str(tmp_path / "ics"), REAL_DIR=str(tmp_path / "real"),
             CONTROL_REF=str(tmp_path / "ref"), CONTROL_ICS=str(tmp_path / "ref_ics"))
    e.update(env)
    p = subprocess.Popen([BASH, STEPS], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=e, start_new_session=True)
    try:
        out, err = p.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9)
        p.communicate()
        pytest.fail("the steps job went past its refusals")
    return subprocess.CompletedProcess(p.args, p.returncode, out, err)


def test_the_steps_job_refuses_a_fresh_start_over_a_checkpoint(tmp_path):
    _gen(tmp_path / "real" / "ckpt", "gen0", 20)
    r = _steps_job(tmp_path)
    assert r.returncode == 1
    assert "already holds a checkpoint; set EXPECT_STEP" in r.stdout
    assert "=== LEG" not in r.stdout  # before any guard


@pytest.mark.parametrize("env,says", [
    ({"REAL_DIR": "{t}/ics/real"}, "REAL_DIR is under IC_DIR"),
    ({"CONTROL_REF": "{t}/absent"}, "set CONTROL_REF to an existing directory"),
    ({"IC_DIR": ""}, "set IC_DIR to an existing directory"),
])
def test_the_steps_job_refuses_inputs_it_cannot_use(tmp_path, env, says):
    r = _steps_job(tmp_path, **{k: v.format(t=tmp_path) for k, v in env.items()})
    assert r.returncode == 1 and says in r.stdout, r.stdout
    assert "=== LEG" not in r.stdout
