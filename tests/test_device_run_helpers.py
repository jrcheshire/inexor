"""The GPU driver's pure helpers: `timed_steps` (which steps `--timed-all`, `--timed-every N`
and `--timed-last` time), `snapshot_steps_for` (`--snapshot-z` to step boundaries), the
monitor's samplers, `smaps_summary` (`--smaps`) and `cache_probe` (`--cache-probe`)."""

import importlib.util
import os

DRIVER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "scripts", "run", "device_run.py")


def _driver():
    spec = importlib.util.spec_from_file_location("device_run_helpers", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _timed_steps():
    return _driver().timed_steps


def test_every_nth_step_is_timed_on_the_absolute_cadence():
    ts = _timed_steps()
    assert ts(0, 10, every=5) == (4, 9)
    # a segment resumed at step 7 keeps the whole run's cadence
    assert ts(7, 20, every=5) == (9, 14, 19)
    assert ts(0, 10, every=1) == tuple(range(10))


def test_all_last_and_none():
    ts = _timed_steps()
    assert ts(2, 6, timed_all=True) == (2, 3, 4, 5)
    assert ts(2, 6, last=True) == (5,)
    assert ts(2, 6) == ()
    # --timed-all wins over --timed-every
    assert ts(0, 4, timed_all=True, every=3) == (0, 1, 2, 3)


def test_snapshot_redshifts_land_on_the_nearest_interior_step():
    from inexor.integrate import a_grid

    a_steps = a_grid(0.1, 1.0, 120, "log")
    got = _driver().snapshot_steps_for([2, 1, 0.5], a_steps)
    assert [s for _, s, _ in got] == [63, 84, 99]
    for z, s, z_step in got:
        assert abs(z_step - (1 / a_steps[s] - 1)) < 1e-12
        # nearer than either neighbour
        assert all(abs(a_steps[s] - 1 / (1 + z)) <= abs(a_steps[t] - 1 / (1 + z))
                   for t in (s - 1, s + 1))


def test_snapshot_redshifts_outside_the_schedule_or_on_one_step_are_refused():
    import pytest

    from inexor.integrate import a_grid

    a_steps = a_grid(0.1, 1.0, 120, "log")
    snap = _driver().snapshot_steps_for
    for zs in ([0.0], [-0.1], [9.0], [12.0]):
        with pytest.raises(ValueError, match="outside"):
            snap(zs, a_steps)
    with pytest.raises(ValueError, match="two on one step"):
        snap([1.0, 1.001], a_steps)


def test_the_card_sampler_keeps_each_phases_own_peak(tmp_path):
    """`card_phase_peak` is the largest in-use sample since the previous boundary (or the
    boundary's own reading), and the next phase starts from the in-use at this boundary,
    not from the earlier peak."""
    mon = _driver().Monitor(str(tmp_path / "card.json"), {}, card_sample_s=1000.0)
    mon._card_max = [5, 7]
    assert mon._take_card_max([{"in_use": 6}, {"in_use": 1}]) == [6, 7]
    assert mon._take_card_max([{"in_use": 2}, {"in_use": 3}]) == [6, 3]
    mon._card_max = [9, 0]
    assert mon._take_card_max([{"in_use": None}, {"in_use": 4}]) == [9, 4]
    off = _driver().Monitor(str(tmp_path / "off.json"), {})
    assert off._take_card_max([{"in_use": 1}]) is None


def test_the_sampler_keeps_each_phases_host_rss_peak(tmp_path):
    """`rss_phase_peak` is the largest sampled RSS since the previous boundary (or this
    boundary's reading); the next phase starts from this boundary's RSS. None without the
    sampler or without an RSS reading."""
    drv = _driver()
    mon = drv.Monitor(str(tmp_path / "card.json"), {}, card_sample_s=1000.0)
    mon._rss_max = 9
    assert mon._take_rss_max(4) == 9
    assert mon._take_rss_max(3) == 4
    assert mon._take_rss_max(None) is None
    off = drv.Monitor(str(tmp_path / "off.json"), {})
    assert off._take_rss_max(5) is None
    rss = drv.rss_now()
    assert rss is None or rss > 0


_SMAPS = """\
00400000-00452000 r-xp 00000000 08:02 173521      /usr/bin/python3.14
Rss:                 100 kB
Anonymous:             0 kB
Shared_Clean:        100 kB
VmFlags: rd ex mr mw me dw
7f0000000000-7f0000100000 rw-p 00000000 00:00 0
Rss:                4000 kB
Anonymous:          4000 kB
Private_Dirty:      4000 kB
7f0000100000-7f0000200000 rw-p 00000000 00:00 0                          [anon:jemalloc]
Rss:                1000 kB
Anonymous:          1000 kB
Private_Dirty:      1000 kB
01000000-02000000 rw-p 00000000 00:00 0                                  [heap]
Rss:                2000 kB
Anonymous:          2000 kB
7f1000000000-7f1000400000 rw-s 00000000 00:1a 99        /dev/shm/mvp-shmem-1234 (deleted)
Rss:                3000 kB
Shared_Dirty:       3000 kB
7f2000000000-7f2000010000 rw-s 00000000 00:06 12        /dev/nvidiactl
Rss:                  64 kB
Locked:               64 kB
7f3000000000-7f3000100000 r-xp 00000000 08:02 777       /opt/lib/libcudart.so.12.9.37
Rss:                 500 kB
Shared_Clean:        500 kB
7f4000000000-7f4000100000 rw-s 00000000 00:01 5         /memfd:xla (deleted)
Rss:                 200 kB
Shared_Dirty:        200 kB
7ffd00000000-7ffd00021000 rw-p 00000000 00:00 0                          [stack]
Rss:                  12 kB
Anonymous:            12 kB
"""


def test_smaps_sums_resident_memory_by_mapping_kind(tmp_path, monkeypatch):
    """Every mapping lands in one kind by its pathname, its fields sum there (kB to bytes),
    and the largest mappings are listed with their names."""
    drv = _driver()
    (tmp_path / "smaps").write_text(_SMAPS)
    monkeypatch.setenv("D7_PROC_SELF", str(tmp_path))
    sm = drv.smaps_summary(top=3)
    kb = 1024
    k = sm["kinds"]
    assert sm["n_maps"] == 9
    assert set(k) == {"file", "anon", "heap", "shm", "nvidia", "lib", "stack"}
    assert k["anon"]["rss"] == 5000 * kb and k["anon"]["maps"] == 2
    assert k["anon"]["private_dirty"] == 5000 * kb
    assert k["shm"]["rss"] == 3200 * kb and k["shm"]["shared_dirty"] == 3200 * kb
    assert k["nvidia"]["locked"] == 64 * kb
    assert k["lib"]["shared_clean"] == 500 * kb and k["file"]["rss"] == 100 * kb
    assert sum(e["rss"] for e in k.values()) == 10876 * kb
    assert [m["name"] for m in sm["top"]] == ["", "/dev/shm/mvp-shmem-1234 (deleted)", "[heap]"]
    assert sm["top"][1]["shared"] == 3000 * kb and sm["top"][1]["kind"] == "shm"
    (tmp_path / "smaps").unlink()
    assert drv.smaps_summary() is None


def test_every_program_cache_the_probe_clears_exists():
    """`PROGRAM_CACHES` names live module dicts: a renamed cache would otherwise be skipped
    silently and its memory read as the state's."""
    import importlib

    from inexor import forces

    for mod, attr in _driver().PROGRAM_CACHES:
        assert isinstance(getattr(importlib.import_module(mod), attr), dict), (mod, attr)
    assert callable(forces._cached_force_fn.cache_clear)


def test_the_cache_probe_empties_the_program_caches_and_records_each_stage(tmp_path,
                                                                         monkeypatch):
    import importlib

    drv = _driver()
    (tmp_path / "smaps").write_text(_SMAPS)
    monkeypatch.setenv("D7_PROC_SELF", str(tmp_path))
    mig = importlib.import_module("inexor.device.migrate")
    mig._PROGRAMS[("probe-test",)] = object()
    mon = drv.Monitor(str(tmp_path / "card.json"), {}, with_smaps=True)
    out = drv.cache_probe(mon)
    assert [r["name"] for r in out["records"]] == ["cache_before", "after_jax_clear_caches",
                                                   "after_program_caches"]
    assert all(r["smaps"]["n_maps"] == 9 for r in out["records"])
    assert set(out["entries"]) == {f"{m}.{a}" for m, a in drv.PROGRAM_CACHES}
    assert out["entries"]["inexor.device.migrate._PROGRAMS"] >= 1
    assert not mig._PROGRAMS


def test_the_run_card_is_written_by_the_c_encoder(tmp_path, monkeypatch):
    """The card is rewritten whole at every boundary and heartbeat, so it goes through the C
    encoder (json's pure-Python one costs ~0.15 s per write at a 120-step card); the file
    reads back as the object, non-JSON values as their str."""
    import json
    import json.encoder

    from pathlib import Path

    real, calls = json.encoder.c_make_encoder, []
    assert real is not None, "this python's json has no C encoder"

    def spy(*a):
        calls.append(1)
        return real(*a)

    monkeypatch.setattr(json.encoder, "c_make_encoder", spy)
    card = dict(steps=[dict(step=k, hwm=1.5 * k) for k in range(3)], path=Path("/x"))
    out = tmp_path / "card.json"
    _driver()._write_json(str(out), card)
    assert calls, "the card went through the pure-Python encoder"
    assert json.loads(out.read_text()) == dict(card, path="/x")
    assert not os.path.exists(f"{out}.tmp")


def test_the_schedule_starts_at_the_ics_or_checkpoints_epoch(tmp_path):
    """`run` from ICs starts the schedule at the ICs' recorded a_init, a resume or a product
    at the checkpoint's (`source.a_init`); records without the field mean a = 0.1. An explicit
    --a-init that disagrees is refused: the ICs carry their epoch in D and f."""
    import argparse
    import json

    import pytest

    start = _driver()._start_epoch

    def ics(name, **man):
        d = tmp_path / name
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps(man))
        return argparse.Namespace(workdir=str(d), a_init=None)

    early, old = ics("early", a_init=0.08), ics("old")
    assert start(early) == 0.08
    assert start(old) == 0.1
    early.a_init = 0.1
    with pytest.raises(SystemExit, match="another epoch"):
        start(early)
    early.a_init = 0.08
    assert start(early) == 0.08

    def ckpt(name, step, **source):
        d = tmp_path / name / "gen0"
        d.mkdir(parents=True)
        prov = dict(kind="inexor-checkpoint", step=step)
        if source:
            prov["source"] = source
        (d / "manifest.json").write_text(json.dumps(dict(provenance=prov)))
        return str(d.parent)

    args = argparse.Namespace(workdir=None, a_init=None)
    assert start(args, ckpt("ck", 3, a_init=0.08)) == 0.08
    assert start(args, ckpt("ck_old", 3)) == 0.1
    args.a_init = 0.1
    with pytest.raises(SystemExit, match="another epoch"):
        start(args, str(tmp_path / "ck"))
