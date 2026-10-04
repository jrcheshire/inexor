"""export.py: the portable (x, v) writer and its CLI.

The core gate is an identity: the export reproduces the state's own decode, row for row, in
the state's order. The rest covers ids, chunk invariance, f32 precision against the position
quantum, velocity units and epoch handling, refusals, and the streamed writer thread.
"""

import dataclasses
import json
import os

import zlib

import numpy as np
import pytest

from inexor import export, state
from inexor.codec import T9Layout
from inexor.config import PLANCK
from inexor.cosmology import E_of_a, growth_factor_a

N_PART, NB, BOX = 16, 4, 16.0


def _evolved_state(seed=3, steps=3, arena_frac=0.05, slack=0.02, with_ids=False):
    """A state after a few migrations: live spares and a populated arena (arena rows are not in
    any brick's contiguous run, so an export can silently drop them)."""
    rng = np.random.default_rng(seed)
    g = (np.arange(N_PART) + 0.5) * (BOX / N_PART)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    x = np.mod(q + rng.normal(scale=0.25 * BOX / N_PART, size=q.shape), BOX)
    v = rng.normal(scale=0.5, size=x.shape)
    st = state.SlotState.build(
        x, v, T9Layout(BOX, N_PART, 2), NB,
        brick_slack=slack, arena_frac=arena_frac, with_ids=with_ids,
    )
    for _ in range(steps):
        state.drift_and_migrate(st, 0.5)
    return st


def test_export_is_the_states_own_decode(tmp_path):
    """The export equals `decode_bricks` over every brick (what the force sees), in order.

    Written at f64 so the comparison is bitwise and only the writer is under test."""
    st = _evolved_state()
    assert st.arena_used > 0, "vacuous: no arena residents, so folding them in is untested"

    d = str(tmp_path / "e")
    head = export.write_particles(st, d, dtype=np.float64)
    _, x, v, ids = export.load_particles(d, mmap=False)

    _, xr, vr = st.decode_bricks(list(range(st.n_bricks)))
    assert head["n_particles"] == st.n_live == len(xr)
    np.testing.assert_array_equal(x, xr)
    np.testing.assert_array_equal(v, vr)
    assert ids is None and head["has_ids"] is False
    # positions are lattice index * quantum: [0, BOX), nothing on the far edge
    assert x.min() >= 0.0 and x.max() < BOX


def test_export_carries_ids_when_the_state_has_them(tmp_path):
    """Ids are exported when present (row order is spatial and recovers no Lagrangian index)."""
    st = _evolved_state(with_ids=True)
    d = str(tmp_path / "e")
    head = export.write_particles(st, d, dtype=np.float64)
    _, _, _, ids = export.load_particles(d, mmap=False)

    slots, _, _ = st.decode_bricks(list(range(st.n_bricks)))
    assert head["has_ids"] is True
    np.testing.assert_array_equal(ids, st.ids[slots])
    # every particle exactly once
    np.testing.assert_array_equal(np.sort(ids), np.arange(st.n_particles, dtype=np.int32))


@pytest.mark.parametrize("chunk", [1, 3, 7, 4096])
def test_chunking_is_invisible(tmp_path, chunk):
    """Output bytes are independent of `chunk_bricks`; sizes are ragged against 64 bricks and
    past it."""
    st = _evolved_state()
    ref = str(tmp_path / "ref")
    export.write_particles(st, ref, dtype=np.float64, chunk_bricks=4096)

    d = str(tmp_path / f"c{chunk}")
    export.write_particles(st, d, dtype=np.float64, chunk_bricks=chunk)
    for f in ("x.npy", "v.npy"):
        with open(os.path.join(ref, f), "rb") as a, open(os.path.join(d, f), "rb") as b:
            assert a.read() == b.read(), f"{f} changed with chunk_bricks={chunk}"


def test_f32_default_resolves_the_position_quantum():
    """The quantum-to-f32-spacing ratio (`quantum / (box * 2**-23)`) clears 2 at production sizes.

    Clearing 2 keeps an f32 cast within half a quantum of its lattice site. Values are 128x,
    32x, 16x for n_part 512/2048/4096; the ratio halves each time the box doubles at fixed
    cell, so the largest box is the binding case."""
    want = {512: 128.0, 2048: 32.0, 4096: 16.0}
    for n_part, box in ((512, 256.0), (2048, 1024.0), (4096, 2048.0)):
        t9 = T9Layout(box, n_part, 2)
        headroom = t9.quantum / (box * 2.0**-23)
        assert headroom == want[n_part], f"n_part={n_part}: {headroom}x, recorded {want[n_part]}x"
        assert headroom >= 2.0, f"n_part={n_part}: f32 cannot hold the quantum ({headroom:.1f}x)"


def test_f32_export_stays_inside_half_a_quantum(tmp_path):
    """An f32 export stays within half a quantum of the f64 decode on real rows."""
    st = _evolved_state()
    d = str(tmp_path / "e")
    export.write_particles(st, d, dtype=np.float32)
    _, x, _, _ = export.load_particles(d, mmap=False)
    _, xr, _ = st.decode_bricks(list(range(st.n_bricks)))
    assert np.abs(x.astype(np.float64) - xr).max() < 0.5 * st.t9.quantum


def test_peculiar_velocity_factor_against_a_finite_difference():
    """`peculiar_velocity_factor` (via f = dlnD/dlna) matches 100 a^2 E(a) dD/da with dD/da
    central-differenced, so the two paths share no algebra beyond D; 1e-6 is FD error scale."""
    for a in (0.25, 0.5, 1.0):
        h = 1e-5
        dDda = (growth_factor_a(a + h, PLANCK) - growth_factor_a(a - h, PLANCK)) / (2 * h)
        want = 100.0 * a * a * E_of_a(a, PLANCK) * dDda
        got = export.peculiar_velocity_factor(a, PLANCK)
        assert abs(got / want - 1.0) < 1e-6, f"a={a}: {got} vs {want}"


def test_km_per_second_output_is_the_dtime_file_times_the_factor(tmp_path):
    st = _evolved_state()
    d0, d1 = str(tmp_path / "dtime"), str(tmp_path / "kms")
    h0 = export.write_particles(st, d0, dtype=np.float64)
    h1 = export.write_particles(st, d1, dtype=np.float64, a=1.0, cosmo=PLANCK)

    _, x0, v0, _ = export.load_particles(d0, mmap=False)
    _, x1, v1, _ = export.load_particles(d1, mmap=False)
    np.testing.assert_array_equal(x0, x1)
    np.testing.assert_allclose(v1, v0 * export.peculiar_velocity_factor(1.0, PLANCK), rtol=0, atol=0)

    assert h0["velocity_is_dtime"] is True and h0["peculiar_velocity_factor"] is None
    assert h1["velocity_is_dtime"] is False
    assert h1["units"]["velocity"] == "km/s peculiar"


def test_refuses_half_a_cosmology(tmp_path):
    """A file whose header claims km/s and holds D-time is worse than a refusal."""
    st = _evolved_state()
    with pytest.raises(ValueError, match="BOTH"):
        export.write_particles(st, str(tmp_path / "a"), a=1.0)
    with pytest.raises(ValueError, match="BOTH"):
        export.write_particles(st, str(tmp_path / "b"), cosmo=PLANCK)


def test_header_is_written_last(tmp_path):
    """The header is the completeness marker: without it a load refuses, since truncated arrays
    would otherwise load without complaint."""
    st = _evolved_state()
    d = str(tmp_path / "e")
    os.makedirs(d)
    with open(os.path.join(d, "x.npy"), "wb") as fh:
        fh.write(b"stale")
    with pytest.raises(FileNotFoundError, match="written last"):
        export.load_particles(d)

    export.write_particles(st, d)
    export.load_particles(d)

    # a header alongside a missing array still refuses to load
    os.remove(os.path.join(d, "x.npy"))
    with pytest.raises(Exception):
        export.load_particles(d, mmap=False)


def test_refuses_crc_corruption(tmp_path):
    st = _evolved_state()
    d = str(tmp_path / "e")
    export.write_particles(st, d)
    with open(os.path.join(d, "v.npy"), "r+b") as fh:
        fh.seek(-4, os.SEEK_END)
        fh.write(b"\x00\x00\x00\x00")
    with pytest.raises(ValueError, match="crc mismatch"):
        export.load_particles(d, mmap=False)


def test_corruption_gate_can_fire_and_the_clean_file_passes(tmp_path):
    """Control for the crc test: an untouched export loads cleanly."""
    st = _evolved_state()
    d = str(tmp_path / "e")
    export.write_particles(st, d)
    export.load_particles(d, mmap=False)


def test_truncated_stream_refuses_at_close(tmp_path):
    """`_StreamedNpy` refuses to close with fewer rows than its declared shape."""
    s = export._StreamedNpy(str(tmp_path / "t.npy"), (10, 3), np.float32)
    s.append(np.zeros((4, 3)))
    with pytest.raises(RuntimeError, match="Rows were lost"):
        s.close()


def test_header_records_the_box_and_the_row_order(tmp_path):
    st = _evolved_state()
    d = str(tmp_path / "e")
    head = export.write_particles(st, d, provenance={"job": "test"})
    assert head["box_size"] == BOX and head["n_part"] == N_PART
    assert head["dtype"] == "float32"
    assert "no Lagrangian identity" in head["row_order"]
    assert head["provenance"] == {"job": "test"}
    with open(os.path.join(d, export.HEADER)) as fh:
        assert json.load(fh) == head


# ---------------------------------------------------------------------------
# CLI units: km/s is the default when the checkpoint records its epoch.


def _checkpoint(tmp_path, name, epoch=None, **prov):
    """A T9 slab directory, with or without a recorded epoch."""
    from inexor import icgen

    d = str(tmp_path / name)
    if epoch is not None:
        a, cosmo = epoch
        prov = dict(prov, a=float(a), cosmology=dataclasses.asdict(cosmo))
    icgen.write_t9_slabs(_evolved_state(), d, provenance=prov)
    return d


def test_cli_defaults_to_km_per_second_off_a_recorded_epoch(tmp_path, capsys):
    """With no flags, the CLI writes km/s at the checkpoint's epoch, pinned to the values of a
    direct km/s export and not merely to the header label."""
    ck = _checkpoint(tmp_path, "ck", epoch=(0.5, PLANCK))
    out = str(tmp_path / "out")
    assert export._main([ck, out]) == 0

    head = json.load(open(os.path.join(out, "export.json")))
    assert head["velocity_is_dtime"] is False
    assert head["units"]["velocity"] == "km/s peculiar"
    assert head["a"] == 0.5
    assert head["peculiar_velocity_factor"] == export.peculiar_velocity_factor(0.5, PLANCK)
    assert "checkpoint epoch" in head["provenance"]["epoch_source"]

    ref = str(tmp_path / "ref")
    export.write_particles(_evolved_state(), ref, a=0.5, cosmo=PLANCK)
    _, _, v_cli, _ = export.load_particles(out, mmap=False)
    _, _, v_ref, _ = export.load_particles(ref, mmap=False)
    np.testing.assert_array_equal(v_cli, v_ref)


def test_cli_falls_back_to_dtime_and_says_so_when_no_epoch_is_recorded(tmp_path, capsys):
    """A checkpoint with no epoch exports D-time velocities, and says so on stdout and in the
    header's provenance."""
    ck = _checkpoint(tmp_path, "ck", kind="inexor-checkpoint", step=3)
    out = str(tmp_path / "out")
    assert export._main([ck, out]) == 0

    head = json.load(open(os.path.join(out, "export.json")))
    assert head["velocity_is_dtime"] is True
    assert head["a"] is None
    assert "no epoch recorded" in head["provenance"]["epoch_source"]
    assert "no epoch recorded" in capsys.readouterr().out


def test_cli_flags_override_the_recorded_epoch(tmp_path):
    """`--a` overrides a recorded epoch and the header records the override; `--d-time` forces
    D-time."""
    ck = _checkpoint(tmp_path, "ck", epoch=(0.5, PLANCK))

    a_dir = str(tmp_path / "over")
    export._main([ck, a_dir, "--a", "1.0"])
    head = json.load(open(os.path.join(a_dir, "export.json")))
    assert head["a"] == 1.0
    assert "overriding the recorded a=0.5" in head["provenance"]["epoch_source"]

    d_dir = str(tmp_path / "dtime")
    export._main([ck, d_dir, "--d-time"])
    head = json.load(open(os.path.join(d_dir, "export.json")))
    assert head["velocity_is_dtime"] is True
    assert head["provenance"]["epoch_source"] == "--d-time"


def test_cli_refuses_epoch_flags_that_cannot_act(tmp_path):
    """Epoch/cosmology flags that cannot act refuse rather than being silently ignored."""
    with_epoch = _checkpoint(tmp_path, "with", epoch=(0.5, PLANCK))
    without = _checkpoint(tmp_path, "without")

    with pytest.raises(SystemExit, match="nothing to act on"):
        export._main([with_epoch, str(tmp_path / "o1"), "--d-time", "--a", "1.0"])
    with pytest.raises(SystemExit, match="need an epoch"):
        export._main([without, str(tmp_path / "o2"), "--omega-m", "0.3"])


def test_cli_cosmology_override_rides_on_the_recorded_epoch(tmp_path):
    """`--omega-m` alone keeps the recorded epoch and overrides one cosmology field; the other
    fields come from the checkpoint."""
    ck = _checkpoint(tmp_path, "ck", epoch=(0.5, PLANCK))
    out = str(tmp_path / "out")
    export._main([ck, out, "--omega-m", "0.25"])

    head = json.load(open(os.path.join(out, "export.json")))
    want = export.peculiar_velocity_factor(
        0.5, dataclasses.replace(PLANCK, Omega_m=0.25)
    )
    assert head["a"] == 0.5
    assert head["peculiar_velocity_factor"] == want
    assert want != export.peculiar_velocity_factor(0.5, PLANCK), "override did not apply"

    # epoch and cosmology come from different places; provenance must name both
    src = head["provenance"]["epoch_source"]
    assert "checkpoint epoch" in src and "Omega_m=0.25" in src, src
    assert head["cosmology"]["Omega_m"] == 0.25
    assert head["cosmology"]["h"] == PLANCK.h, "un-overridden fields must come from the file"


def test_cli_announces_the_epoch_it_converted_at(tmp_path, capsys):
    """The stdout line names the epoch, the cosmology, and where each came from.

    A velocity converted at the wrong scale factor is off by tens of percent yet looks
    plausible."""
    ck = _checkpoint(tmp_path, "ck", epoch=(0.5, PLANCK))
    export._main([ck, str(tmp_path / "out"), "--omega-m", "0.25"])

    line = [ln for ln in capsys.readouterr().out.splitlines() if "velocities:" in ln]
    assert len(line) == 1, line
    said = line[0]
    assert "a=0.5" in said, said
    assert "Omega_m=0.25" in said, said
    assert "checkpoint epoch" in said, said
    assert "cosmology overridden" in said, said


def test_dtime_export_records_no_cosmology(tmp_path):
    """`a`, factor and cosmology in the header are all None on a D-time export and set on km/s."""
    st = _evolved_state()
    d = str(tmp_path / "dtime")
    head = export.write_particles(st, d)
    assert head["a"] is None
    assert head["peculiar_velocity_factor"] is None
    assert head["cosmology"] is None

    k = str(tmp_path / "kms")
    head = export.write_particles(st, k, a=0.5, cosmo=PLANCK)
    assert head["cosmology"] == dataclasses.asdict(PLANCK)


# --- the streamed writer: byte views and the writer thread ---


def test_append_writes_the_bytes_the_tobytes_form_wrote(tmp_path):
    """`append` writes the same bytes and running crc32 as the `tobytes()` form, over three
    calls so the crc carries across appends."""
    rng = np.random.default_rng(0)
    block = rng.normal(size=(30, 3))
    path = tmp_path / "x.npy"
    s = export._StreamedNpy(str(path), (30, 3), np.float32)
    for lo in (0, 10, 20):
        s.append(block[lo : lo + 10])
    s.close()

    crc, raw = 0, b""
    for lo in (0, 10, 20):
        b = np.ascontiguousarray(block[lo : lo + 10], dtype=np.float32).tobytes()
        raw += b
        crc = zlib.crc32(b, crc)
    assert s.crc == crc
    assert path.read_bytes().endswith(raw)
    assert np.array_equal(np.load(path), block.astype(np.float32))


def test_the_write_runs_off_the_main_thread(tmp_path):
    """Appends run on the writer thread; value tests cannot see this, since a serial writer is
    also correct."""
    import threading

    st = _evolved_state()
    seen = []
    real = export._StreamedNpy.append

    def spy(self, block):
        seen.append(threading.current_thread().name)
        return real(self, block)

    export._StreamedNpy.append = spy
    try:
        export.write_particles(st, str(tmp_path), chunk_bricks=8)
    finally:
        export._StreamedNpy.append = real
    assert seen, "no block was ever appended"
    assert all(n.startswith("export-writer") for n in seen), sorted(set(seen))


def test_a_failure_on_the_writer_thread_surfaces(tmp_path):
    """An exception on the writer thread propagates and no header is written."""
    st = _evolved_state()
    real = export._StreamedNpy.append
    calls = []

    def boom(self, block):
        calls.append(1)
        if len(calls) > 2:
            raise OSError("no space left on device")
        return real(self, block)

    export._StreamedNpy.append = boom
    try:
        with pytest.raises(OSError, match="no space left"):
            export.write_particles(st, str(tmp_path), chunk_bricks=1)
    finally:
        export._StreamedNpy.append = real
    assert not os.path.exists(os.path.join(str(tmp_path), export.HEADER))


def test_timings_report_the_parts_and_the_chunk_count(tmp_path):
    """`timings` reports decode, write (main-thread blocked time), write thread (busy time) and
    the chunk count."""
    st = _evolved_state()
    t = {}
    export.write_particles(st, str(tmp_path), chunk_bricks=8, timings=t)
    assert set(t) >= {"decode", "write", "write thread", "chunks"}
    assert t["chunks"] == len(range(0, st.n_bricks, 8))
    assert t["write thread"] > 0.0
    assert t["decode"] > 0.0


def test_append_does_not_copy_the_block_to_write_it(tmp_path):
    """`append` peaks under 1.5x the f32 payload (the cast is its only copy).

    The `tobytes()` control must exceed 1.9x (a second copy), or the bound cannot discriminate.
    """
    import tracemalloc

    blk = np.zeros((200_000, 3), dtype=np.float64)   # 2.4 MB of f32 payload
    payload = blk.shape[0] * 3 * 4

    def _peak(fn):
        tracemalloc.start()
        try:
            fn()
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    def fixed():
        s = export._StreamedNpy(str(tmp_path / "a.npy"), blk.shape, np.float32)
        s.append(blk)
        s.close()

    def control():
        s = export._StreamedNpy(str(tmp_path / "b.npy"), blk.shape, np.float32)
        b = np.ascontiguousarray(blk, dtype=np.float32)
        raw = b.tobytes()                      # the extra copy
        s._fh.write(raw)
        s.crc = zlib.crc32(raw, s.crc)
        s.rows += b.shape[0]
        s.close()

    assert _peak(fixed) < 1.5 * payload
    assert _peak(control) > 1.9 * payload
    assert (tmp_path / "a.npy").read_bytes() == (tmp_path / "b.npy").read_bytes()


# --- multi-part exports (format inexor-particles-2) ---


@pytest.mark.parametrize("cut", [1, 7, 4096])
def test_crc32_combine_is_the_crc_of_the_concatenation(cut):
    rng = np.random.default_rng(cut)
    a = rng.integers(0, 256, size=5000, dtype=np.uint8).tobytes()
    b = rng.integers(0, 256, size=cut, dtype=np.uint8).tobytes()
    assert export._crc32_combine(zlib.crc32(a), zlib.crc32(b), len(b)) == zlib.crc32(a + b)
    assert export._crc32_combine(zlib.crc32(a), zlib.crc32(b""), 0) == zlib.crc32(a)
    assert export._crc32_combine(0, zlib.crc32(b), len(b)) == zlib.crc32(b)


def test_crc32_combine_past_four_gibibytes():
    """A part's byte length passes 2**32 at production scale (0.8 TB per node and array)."""
    head = b"inexor"
    n = 2**32 + 13
    block = bytes(64 << 20)
    crc_z = 0
    left = n
    while left:
        k = min(left, len(block))
        crc_z = zlib.crc32(block[:k] if k < len(block) else block, crc_z)
        left -= k
    want = crc_z_with_head = zlib.crc32(head)
    left = n
    while left:
        k = min(left, len(block))
        want = zlib.crc32(block[:k] if k < len(block) else block, want)
        left -= k
    assert export._crc32_combine(crc_z_with_head, crc_z, n) == want


def _parts_by_rank(st, n_ranks, workdir, **kw):
    from inexor.comm import run_loopback
    from tests.test_partial_state import rank_slabs, restrict_to_slabs

    cuts = list(rank_slabs(NB, n_ranks))

    def rank(c):
        part = st if c.size == 1 else restrict_to_slabs(st, cuts[c.rank])
        return export.write_particle_parts(part, workdir, comm=c, **kw)

    return run_loopback(n_ranks, rank, timeout=60.0)


def _array_bytes(path):
    a = np.load(path)
    return np.ascontiguousarray(a).tobytes()


@pytest.mark.parametrize("n_ranks", [1, 2, 4])
def test_parts_in_rank_order_are_the_single_file_bytes(tmp_path, n_ranks):
    """The gate: concatenated parts == `write_particles`' arrays, and the header's whole-array
    crc32 == the single-file crc32, at f32 km/s (the production output)."""
    st = _evolved_state()
    assert st.arena_used > 0, "vacuous: no arena residents"
    one = str(tmp_path / "one")
    want = export.write_particles(st, one, dtype=np.float32, a=0.5, cosmo=PLANCK,
                                  chunk_bricks=3)
    d = str(tmp_path / "parts")
    heads = _parts_by_rank(st, n_ranks, d, dtype=np.float32, a=0.5, cosmo=PLANCK,
                           chunk_bricks=5, expect_total=st.n_live)
    head = heads[0]
    assert all(h == head for h in heads)
    assert head["format"] == export.FORMAT_PARTS and len(head["parts"]) == n_ranks
    assert head["crc32"] == want["crc32"]
    assert head["n_particles"] == want["n_particles"] == st.n_live
    assert head["peculiar_velocity_factor"] == want["peculiar_velocity_factor"]
    for key in ("x", "v"):
        got = b"".join(_array_bytes(os.path.join(d, p["files"][key])) for p in head["parts"])
        assert got == _array_bytes(os.path.join(one, want["files"][key]))
    rows = [(p["row0"], p["rows"]) for p in head["parts"]]
    assert [r0 for r0, _ in rows] == list(np.cumsum([0] + [n for _, n in rows])[:-1])
    _, x, v, ids = export.load_particles(d, mmap=False)
    _, xw, vw, _ = export.load_particles(one, mmap=False)
    assert x.tobytes() == xw.tobytes() and v.tobytes() == vw.tobytes() and ids is None
    assert [r0 for r0, *_ in export.iter_particle_parts(d)] == [r0 for r0, _ in rows]


def test_one_part_carries_ids_and_loads_as_a_memmap(tmp_path):
    st = _evolved_state(with_ids=True)
    one = str(tmp_path / "one")
    want = export.write_particles(st, one, dtype=np.float64)
    d = str(tmp_path / "parts")
    head = export.write_particle_parts(st, d, dtype=np.float64, chunk_bricks=7)
    assert head["has_ids"] and head["crc32"] == want["crc32"]
    _, x, v, ids = export.load_particles(d, mmap=True)
    assert isinstance(x, np.memmap)
    _, xw, vw, idw = export.load_particles(one, mmap=False)
    np.testing.assert_array_equal(ids, idw)
    np.testing.assert_array_equal(x, xw)


def test_a_multi_part_export_refuses_one_memmap(tmp_path):
    d = str(tmp_path / "parts")
    _parts_by_rank(_evolved_state(), 2, d)
    with pytest.raises(ValueError, match="iter_particle_parts"):
        export.load_particles(d, mmap=True)


def test_a_short_part_refuses_at_close_and_writes_no_header(tmp_path):
    st = _evolved_state()
    lo, hi = st.owned_bricks
    chunks = [st.decode_bricks([b]) for b in range(lo, hi) if st.brick_member_count(b)]
    d = str(tmp_path / "short")
    with pytest.raises(RuntimeError, match="Rows were lost"):
        export.write_particle_parts(st, d, chunks=chunks[:-1])
    assert not os.path.exists(os.path.join(d, export.HEADER))


def test_an_unexpected_total_writes_no_header(tmp_path):
    st = _evolved_state()
    d = str(tmp_path / "e")
    with pytest.raises(RuntimeError, match="lost or doubled"):
        export.write_particle_parts(st, d, expect_total=st.n_live + 1)
    assert not os.path.exists(os.path.join(d, export.HEADER))


def test_a_failing_rank_leaves_no_header(tmp_path):
    from inexor.comm import run_loopback
    from tests.test_partial_state import rank_slabs, restrict_to_slabs

    st = _evolved_state()
    d = str(tmp_path / "e")
    export.write_particle_parts(st, d)  # a stale complete export: its header must go
    cuts = list(rank_slabs(NB, 2))

    def boom():
        raise OSError("disk full, on purpose")
        yield

    def rank(c):
        part = restrict_to_slabs(st, cuts[c.rank])
        return export.write_particle_parts(part, d, comm=c,
                                           chunks=boom() if c.rank == 1 else None)

    with pytest.raises(Exception):
        run_loopback(2, rank, timeout=60.0)
    assert not os.path.exists(os.path.join(d, export.HEADER))


# --- the export decoded on the cards ---


@pytest.mark.parametrize("kms,dtype,with_ids,chunk", [
    (True, np.float32, False, None),
    (False, np.float32, True, 16),
    (False, np.float64, False, 4),
    (True, np.float64, True, 64),
])
def test_the_cards_decode_writes_the_host_bytes(tmp_path, kms, dtype, with_ids, chunk):
    """Positions, velocities (D-time and km/s), ids and arena residents decoded on a card are
    the host export byte for byte."""
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        st = _evolved_state(with_ids=with_ids)
        assert st.arena_used > 0, "vacuous: no arena residents"
        epoch = dict(a=0.5, cosmo=PLANCK) if kms else {}
        one = str(tmp_path / "one")
        want = export.write_particles(st, one, dtype=dtype, **epoch)
        d = str(tmp_path / "cards")
        timings = {}
        head = export.write_particle_parts(st, d, dtype=dtype, decode="cards",
                                           chunk_bricks=chunk, timings=timings, **epoch)
    finally:
        jax.config.update("jax_enable_x64", prev)
    assert head["decode"] == "cards" and timings["card s"] > 0.0
    assert head["crc32"] == want["crc32"]
    for key, f in want["files"].items():
        assert _array_bytes(os.path.join(d, head["parts"][0]["files"][key])) == \
            _array_bytes(os.path.join(one, f))


def test_the_cards_decode_refuses_a_chunk_that_splits_the_range(tmp_path):
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        with pytest.raises(ValueError, match="does not tile"):
            export.write_particle_parts(_evolved_state(), str(tmp_path / "e"),
                                        decode="cards", chunk_bricks=3)
    finally:
        jax.config.update("jax_enable_x64", prev)
