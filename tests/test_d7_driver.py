"""D7 driver instruments: the path guard and the /proc parsers, which only ever run on
a cluster. `under` is here because a string-prefix version refused a legitimate write
and cost gb 1003378's gate leg."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

import v2_d7_hero_smoke as d7  # noqa: E402


def test_under_is_containment_not_string_prefix(tmp_path):
    ics = tmp_path / "smoke"
    ics.mkdir()
    assert d7.under(str(ics), str(ics))
    assert d7.under(str(ics / "gen0"), str(ics))
    # the 1003378 case: a SIBLING whose name extends the IC directory's
    assert not d7.under(str(tmp_path / "smoke-ckpt-probe"), str(ics))
    assert not d7.under(str(tmp_path / "other"), str(ics))
    assert not d7.under(None, str(ics))


def _fake_node(root, node, cpus, **kw):
    d = root / f"node{node}"
    d.mkdir(parents=True)
    (d / "cpulist").write_text(cpus)
    (d / "meminfo").write_text("".join(
        f"Node {node} {k}: {v} kB\n" for k, v in kw.items()) + f"Node {node} HugePages_Total: 0\n")


def test_numa_memory_splits_cpu_and_hbm_nodes_and_reads_the_cache(tmp_path, monkeypatch):
    """A GB200's HBM appears as a CPU-less node; MemFree excludes page cache, so
    FilePages is read beside it."""
    root = tmp_path / "node"
    _fake_node(root, 0, "0-71", MemTotal=400, MemFree=100, FilePages=50, Dirty=7)
    _fake_node(root, 1, "", MemTotal=200, MemFree=200, FilePages=0)
    monkeypatch.setenv("D7_NUMA_ROOT", str(root))
    nm = d7.numa_memory()
    assert set(nm["cpu"]) == {0} and set(nm["gpu"]) == {1}
    assert nm["cpu_total"] == 400 * 1024 and nm["cpu_free"] == 100 * 1024
    assert nm["gpu_used"][1] == 0
    assert nm["detail"][0]["FilePages"] == 50 * 1024 and nm["detail"][0]["Dirty"] == 7 * 1024


def test_vmstat_sums_by_prefix_but_not_the_throttle_counter(tmp_path, monkeypatch):
    f = tmp_path / "vmstat"
    f.write_text("allocstall_normal 3\nallocstall_movable 4\npgscan_direct 100\n"
                 "pgscan_direct_throttle 50\ncompact_stall 2\nnr_dirty 9\n")
    monkeypatch.setenv("D7_VMSTAT", str(f))
    v = d7.vmstat()
    assert v["allocstall"] == 7 and v["pgscan_direct"] == 100 and v["compact_stall"] == 2
    assert "nr_dirty" not in v
    assert d7._delta({"allocstall": 10}, v) == {"allocstall": 3}


def test_numa_maps_totals_pages_per_node_by_kind(tmp_path, monkeypatch):
    p = tmp_path / "self"
    p.mkdir()
    (p / "numa_maps").write_text(
        "aaaa default anon=100 dirty=100 N0=60 N1=40 kernelpagesize_kB=64\n"
        "bbbb default file=/lib/x.so mapped=10 N1=10 kernelpagesize_kB=64\n")
    monkeypatch.setenv("D7_PROC_SELF", str(p))
    nm = d7.numa_maps()
    assert nm[0]["anon"] == 60 * 64 * 1024 and nm[1]["anon"] == 40 * 64 * 1024
    assert nm[1]["file"] == 10 * 64 * 1024 and nm[0]["file"] == 0


def test_proc_memory_reads_the_locked_and_pinned_rss(tmp_path, monkeypatch):
    p = tmp_path / "self"
    p.mkdir()
    (p / "status").write_text("VmRSS:\t100 kB\nRssAnon:\t80 kB\nVmPin:\t20 kB\nName:\tpython\n")
    monkeypatch.setenv("D7_PROC_SELF", str(p))
    pm = d7.proc_memory()
    assert pm == {"VmRSS": 100 * 1024, "RssAnon": 80 * 1024, "VmPin": 20 * 1024}


@pytest.mark.skipif(sys.platform != "linux", reason="glibc only")
def test_trim_probe_runs_where_there_is_a_glibc():
    class _Mon:
        def snapshot(self):
            return dict(mem=d7.proc_memory(), faults={}, vmstat={})

    out = d7.trim_probe(_Mon())
    assert out is not None and out["seconds"] >= 0


def _write_maps(tmp_path, monkeypatch, lines):
    p = tmp_path / "self"
    p.mkdir(exist_ok=True)
    (p / "numa_maps").write_text(lines)
    monkeypatch.setenv("D7_PROC_SELF", str(p))
    return p


def test_membind_refusals_accept_a_binding_to_the_cpu_nodes(tmp_path, monkeypatch):
    _write_maps(tmp_path, monkeypatch,
                "aaaa bind:0-1 anon=100 N0=100 kernelpagesize_kB=64\n" * 20)
    assert d7.membind_refusals({0, 1}) == []


def test_membind_refusals_catch_an_unbound_process_and_a_wrong_node(tmp_path, monkeypatch):
    """gb 1003511: with no binding the kernel put the state on a card's HBM node, and the
    symptom was a device OOM on a card whose own allocator held 24 GB."""
    _write_maps(tmp_path, monkeypatch, "aaaa default anon=100 N0=100 kernelpagesize_kB=64\n" * 20)
    assert any("numactl --membind" in r for r in d7.membind_refusals({0, 1}))
    _write_maps(tmp_path, monkeypatch, "aaaa bind:0-3 anon=100 N0=100 kernelpagesize_kB=64\n" * 20)
    assert any("outside" in r for r in d7.membind_refusals({0, 1}))
    assert d7.membind_refusals({0, 1}, policy={}) , "an unreadable policy must refuse"


def test_pages_off_the_cpu_nodes_names_the_hbm_nodes():
    nm = dict(cpu={0: (1, 1), 1: (1, 1)}, gpu={2: (1, 1), 3: (1, 1)})
    nmaps = {0: dict(anon=10, file=0), 2: dict(anon=197, file=3), 3: dict(anon=0, file=0)}
    off, where = d7.pages_off_the_cpu_nodes(nmaps, nm)
    assert off == 200 and where == {2: 200}


def test_node_list_parses_ranges_and_lists():
    assert d7._node_list("0,1") == {0, 1} and d7._node_list("0-3") == {0, 1, 2, 3}


def _slabs(d):
    return {f: open(os.path.join(d, f), "rb").read()
            for f in sorted(os.listdir(d)) if f != "manifest.json"}


def test_a_segmented_run_resumes_to_the_uninterrupted_state_and_refuses_misuse(tmp_path):
    """The driver's segment path, which a 120-step 4096^3 realization needs because
    the schedule is longer than a queue's wall: two segments with a resume between
    them land bitwise on the uninterrupted run, and the two ways a batch script can
    get a resume wrong -- restarting from the ICs onto a live checkpoint directory,
    or resuming from a step it was not submitted for -- refuse instead of burning
    the wall."""
    import json

    import v2_m6_realization as m6

    ics = str(tmp_path / "ics")
    m6.cmd_ics(m6.build_parser().parse_args(["ics", "--config", "smoke", "--workdir", ics]))

    def run(ckpt, stop, expect=0):
        return d7.main(["run", "--preset", "smoke", "--workdir", ics, "--cards", "1",
                        "--card", str(tmp_path / f"card_{stop}_{expect}.json"),
                        "--k-steps", "40", "--stop-at", str(stop), "--expect-step", str(expect),
                        "--checkpoint-dir", ckpt, "--checkpoint-every", "2", "--beat", "30"])

    full, seg = str(tmp_path / "full"), str(tmp_path / "seg")
    assert run(full, 4) == 0
    assert run(seg, 2) == 0
    with pytest.raises(SystemExit, match="already holds a checkpoint"):
        run(seg, 2)
    with pytest.raises(RuntimeError, match="submitted to resume from step 3"):
        run(seg, 4, expect=3)
    assert run(seg, 4, expect=2) == 0

    assert _slabs(os.path.join(full, "gen1")) == _slabs(os.path.join(seg, "gen1"))
    with open(os.path.join(seg, "gen1", "manifest.json")) as fh:
        prov = json.load(fh)["provenance"]
    assert (prov["step"], prov["n_steps"]) == (4, 40)
    assert prov.get("a") is not None, "the checkpoint does not record its epoch"


def test_the_driver_refuses_ics_generated_with_another_growth2(tmp_path):
    import json

    import v2_m6_realization as m6

    ics = str(tmp_path / "ics")
    m6.cmd_ics(m6.build_parser().parse_args(["ics", "--config", "smoke", "--workdir", ics]))
    p = os.path.join(ics, "manifest.json")
    with open(p) as fh:
        man = json.load(fh)
    man.pop("growth2")  # what every manifest written before the flag looks like
    with open(p, "w") as fh:
        json.dump(man, fh)
    with pytest.raises(SystemExit, match="growth2 = 'eds'"):
        d7.main(["run", "--preset", "smoke", "--workdir", ics, "--cards", "1",
                 "--card", str(tmp_path / "card.json"), "--stop-at", "2"])
