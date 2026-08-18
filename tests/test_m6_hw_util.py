"""The C11 hardware-utilization instrument: its arithmetic, on scripted inputs.

The sampler and canary only run on Linux, but everything that decides what the
card REPORTS -- counter differencing, midpoint attribution, the per-node busy
split, the canary drop rule, the neutrality bound -- is pure arithmetic over
readings, and that is what goes wrong (Stage 0's unreadable gates, job 451's
reader). Readers take root paths and the aggregators take plain lists, so every
test drives them from sequences chosen to break them rather than from whatever
the machine happened to do.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

import v2_m6_c11_hw_util as hw  # noqa: E402


def test_parse_cpulist():
    assert hw.parse_cpulist("0-3") == [0, 1, 2, 3]
    assert hw.parse_cpulist("0-2,8,10-11") == [0, 1, 2, 8, 10, 11]
    assert hw.parse_cpulist("144\n") == [144]


def test_whole_disk_filter():
    assert hw._is_whole_disk("sda")
    assert not hw._is_whole_disk("sda1")
    assert hw._is_whole_disk("nvme0n1")
    assert not hw._is_whole_disk("nvme0n1p1")
    assert not hw._is_whole_disk("loop0")
    assert not hw._is_whole_disk("ram2")
    assert not hw._is_whole_disk("dm-0")


def test_read_proc_stat_busy_split(tmp_path):
    # user nice system idle iowait irq softirq steal
    (tmp_path / "stat").write_text(
        "cpu  99 99 99 99 99 99 99 99 0 0\n"
        "cpu0 10 1 5 100 7 2 3 4 0 0\n"
        "cpu1 20 0 0 50 0 0 0 0 0 0\n"
        "intr 0\n")
    cpus = hw.read_proc_stat(str(tmp_path))
    # busy = user+nice+system+irq+softirq+steal = 10+1+5+2+3+4
    assert cpus[0] == (25, 100, 7)
    assert cpus[1] == (20, 50, 0)


def test_read_diskstats_sums_whole_devices_only(tmp_path):
    rows = [
        # major minor name reads rmerge rsect rtime writes wmerge wsect ...
        "8 0 sda 10 0 1000 0 5 0 2000 0 0 0 0",
        "8 1 sda1 99 0 999999 0 99 0 999999 0 0 0 0",
        "7 0 loop0 99 0 999999 0 99 0 999999 0 0 0 0",
        "259 0 nvme0n1 1 0 100 0 1 0 200 0 0 0 0",
    ]
    (tmp_path / "diskstats").write_text("\n".join(rows) + "\n")
    rd, wr = hw.read_diskstats(str(tmp_path))
    assert rd == (1000 + 100) * 512
    assert wr == (2000 + 200) * 512


def test_mems_allowed(tmp_path):
    os.makedirs(tmp_path / "self")
    (tmp_path / "self" / "status").write_text("Name:\tx\nMems_allowed_list:\t0-1\n")
    assert hw.mems_allowed(str(tmp_path)) == "0-1"


def _tick(t, cpu, disk=(0, 0), lustre=None, psi=None):
    return dict(t=t, cpu={k: list(v) for k, v in cpu.items()},
                disk=list(disk), lustre=lustre, psi=psi)


def test_attribute_midpoint_and_busy_math():
    # cpu0 gains 100 jiffies busy per 1 s tick at clk_tck=100 => 1.0 busy core.
    # Ticks at 0..4; intervals: A covers (0, 2.5), B covers (2.5, 4).
    # Pair (2,3) has midpoint 2.5, which is >= B's start => lands in B, not A.
    ticks = [_tick(float(i), {"0": (100 * i, 0, 10 * i)}) for i in range(5)]
    ivs = [(0.0, 2.5, "A"), (2.5, 4.0, "B")]
    agg = hw.attribute_ticks(ivs, ticks, clk_tck=100)
    assert agg["A"]["n_ticks"] == 2 and agg["B"]["n_ticks"] == 2
    a, b = hw.summarize_label(agg["A"]), hw.summarize_label(agg["B"])
    assert a["busy_cores"] == pytest.approx(1.0)
    assert b["busy_cores"] == pytest.approx(1.0)
    assert a["iowait_cores"] == pytest.approx(0.1)


def test_attribute_clk_tck_is_used():
    ticks = [_tick(float(i), {"0": (100 * i, 0, 0)}) for i in range(3)]
    ivs = [(0.0, 2.0, "A")]
    agg = hw.attribute_ticks(ivs, ticks, clk_tck=250)
    assert hw.summarize_label(agg["A"])["busy_cores"] == pytest.approx(100 / 250)


def test_attribute_drops_stalled_pairs():
    ticks = [_tick(0.0, {"0": (0, 0, 0)}),
             _tick(100.0, {"0": (10000, 0, 0)}),  # 100 s gap: dropped
             _tick(101.0, {"0": (10100, 0, 0)})]
    agg = hw.attribute_ticks([(0.0, 200.0, "A")], ticks, clk_tck=100, max_gap_s=10.0)
    assert agg["A"]["n_ticks"] == 1
    assert agg["A"]["busy_core_s"] == pytest.approx(1.0)


def test_attribute_per_node_split():
    ticks = [_tick(float(i), {"0": (100 * i, 0, 0), "1": (50 * i, 0, 0)})
             for i in range(3)]
    agg = hw.attribute_ticks([(0.0, 2.0, "A")], ticks, clk_tck=100,
                             node_cpus={0: [0], 1: [1]})
    row = hw.summarize_label(agg["A"])
    assert row["busy_cores_by_node"]["0"] == pytest.approx(1.0)
    assert row["busy_cores_by_node"]["1"] == pytest.approx(0.5)


def test_attribute_disk_and_psi_deltas():
    ticks = [_tick(0.0, {"0": (0, 0, 0)}, disk=(0, 0), psi=dict(cpu=0, memory=0, io=0)),
             _tick(1.0, {"0": (0, 100, 0)}, disk=(2**20 * 512, 0),
                   psi=dict(cpu=500000, memory=0, io=0))]
    agg = hw.attribute_ticks([(0.0, 1.0, "A")], ticks, clk_tck=100)
    row = hw.summarize_label(agg["A"])
    assert row["disk_read_mb"] == pytest.approx(512.0)
    assert row["psi_stall_frac"]["cpu"] == pytest.approx(0.5)


def test_summarize_refuses_empty():
    assert hw.summarize_label(dict(dt_s=0.0, busy_core_s=0, iowait_core_s=0,
                                   busy_core_s_by_node={}, disk_read_b=0,
                                   disk_write_b=0, lustre_read_b=0, lustre_write_b=0,
                                   psi_stall_us=dict(cpu=0, memory=0, io=0),
                                   n_ticks=0)) is None


def test_canary_drop_rule(tmp_path):
    path = tmp_path / "c.jsonl"
    rows = [dict(header=dict(cpu=0, mb=1, pid=1))]
    rows += [dict(t0=float(i), t1=float(i) + 0.4, bytes=int(4e9 * 0.4))
             for i in range(10)]
    rows.append(dict(t0=20.0, t1=40.0, bytes=int(4e9 * 0.4)))  # SIGSTOP span
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    wins = hw.load_canary_windows(str(path))
    assert len(wins) == 10  # the 20 s window is dropped
    assert all(gb == pytest.approx(4.0) for _, _, gb in wins)


def test_canary_by_label_midpoint():
    wins = [(0.0, 0.4, 4.0), (1.0, 1.4, 2.0)]
    per = hw.canary_by_label([(0.0, 1.0, "A"), (1.0, 2.0, "B")], wins)
    assert per == {"A": [4.0], "B": [2.0]}


def test_neutrality_material_bound_binds_when_sigma_small():
    v = hw.neutrality([101.0, 101.0, 101.0], [100.0, 100.0, 100.05])
    assert v["gate_bound_that_bound"] == "material"
    assert v["instrument_neutral"]  # 1.0 s overhead vs 2.0 s material bound


def test_neutrality_sigma_bound_binds_when_larger():
    v = hw.neutrality([110.0], [100.0, 90.0, 80.0])
    assert v["gate_bound_that_bound"] == "sigma"
    assert v["instrument_neutral"]  # 20 s sigma bound > 10 s overhead


def test_neutrality_single_control_uses_material():
    v = hw.neutrality([105.0], [100.0])
    assert v["gate_bound_that_bound"] == "material"
    assert not v["instrument_neutral"]  # 5 s overhead vs 2 s bound


def test_phase_intervals_carry_the_ending_name():
    ivs = hw._phase_intervals(0.0, [(1.0, "a"), (3.0, "b")])
    assert ivs == [(0.0, 1.0, "a"), (1.0, 3.0, "b")]


def test_triad_bytes_convention():
    a, b, c = hw._alloc_triad(3.0)
    nbytes = hw._triad_pass(a, b, c, reps=2)
    assert nbytes == 2 * 3 * a.nbytes
