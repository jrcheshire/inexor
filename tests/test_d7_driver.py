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
