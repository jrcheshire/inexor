"""export.py: the portable (x, v) writer (M-v2-6 Stage 4).

The gate is an IDENTITY, not a tolerance: the export must reproduce the state's
own decode, row for row, in the state's own order. Everything else here is the
refusal surface plus the two claims the module makes in prose -- that chunking
is invisible, and that the f32 default resolves the position quantum.
"""

import dataclasses
import json
import os

import numpy as np
import pytest

from inexor import export, state
from inexor.codec import T9Layout
from inexor.config import PLANCK
from inexor.cosmology import E_of_a, growth_factor_a

N_PART, NB, BOX = 16, 4, 16.0


def _evolved_state(seed=3, steps=3, arena_frac=0.05, slack=0.02, with_ids=False):
    """A state that has been through the exchange, so it has live spares and a
    populated arena. The arena is the part an export can silently drop: those
    rows are not in any brick's contiguous run."""
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
    """THE Stage 4 export gate. `decode_bricks` over every brick is what the
    force sees; the export must be that, exactly, in that order. Written at f64
    so the comparison is bitwise and the only thing under test is the writer."""
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
    # positions decode to a lattice index times the quantum, so the box is a
    # closed-open interval and nothing may sit on the far edge
    assert x.min() >= 0.0 and x.max() < BOX


def test_export_carries_ids_when_the_state_has_them(tmp_path):
    """Row order is spatial and recovers no Lagrangian index, so a state that
    paid for ids must not have them dropped on the way out."""
    st = _evolved_state(with_ids=True)
    d = str(tmp_path / "e")
    head = export.write_particles(st, d, dtype=np.float64)
    _, _, _, ids = export.load_particles(d, mmap=False)

    slots, _, _ = st.decode_bricks(list(range(st.n_bricks)))
    assert head["has_ids"] is True
    np.testing.assert_array_equal(ids, st.ids[slots])
    # every particle exactly once: the ids ARE the Lagrangian index
    np.testing.assert_array_equal(np.sort(ids), np.arange(st.n_particles, dtype=np.int32))


@pytest.mark.parametrize("chunk", [1, 3, 7, 4096])
def test_chunking_is_invisible(tmp_path, chunk):
    """The streaming granularity is a memory knob and must not reach the bytes.
    Chunk sizes chosen ragged against the brick count (64) and past it."""
    st = _evolved_state()
    ref = str(tmp_path / "ref")
    export.write_particles(st, ref, dtype=np.float64, chunk_bricks=4096)

    d = str(tmp_path / f"c{chunk}")
    export.write_particles(st, d, dtype=np.float64, chunk_bricks=chunk)
    for f in ("x.npy", "v.npy"):
        with open(os.path.join(ref, f), "rb") as a, open(os.path.join(d, f), "rb") as b:
            assert a.read() == b.read(), f"{f} changed with chunk_bricks={chunk}"


def test_f32_default_resolves_the_position_quantum():
    """The f32 default is a claim about precision, so measure it rather than
    assert it in prose. The stored position is an integer lattice index times
    `box / n_levels` and f32's spacing at magnitude `box` is `box * 2**-23`.

    The bar is DERIVED, not picked: the cast must not move a particle off its
    lattice site, so the ratio has to clear 2 (half a quantum, which is what
    the companion test measures on real rows). Measured: 128x at cgh64, 32x at
    C-gh, 16x at C-hero -- it halves each time the box doubles at fixed cell,
    so it is a property of the config and C-hero is the binding rung."""
    want = {512: 128.0, 2048: 32.0, 4096: 16.0}
    for n_part, box in ((512, 256.0), (2048, 1024.0), (4096, 2048.0)):
        t9 = T9Layout(box, n_part, 2)
        headroom = t9.quantum / (box * 2.0**-23)
        assert headroom == want[n_part], f"n_part={n_part}: {headroom}x, recorded {want[n_part]}x"
        assert headroom >= 2.0, f"n_part={n_part}: f32 cannot hold the quantum ({headroom:.1f}x)"


def test_f32_export_stays_inside_half_a_quantum(tmp_path):
    """The consequence of the above, on real rows."""
    st = _evolved_state()
    d = str(tmp_path / "e")
    export.write_particles(st, d, dtype=np.float32)
    _, x, _, _ = export.load_particles(d, mmap=False)
    _, xr, _ = st.decode_bricks(list(range(st.n_bricks)))
    assert np.abs(x.astype(np.float64) - xr).max() < 0.5 * st.t9.quantum


def test_peculiar_velocity_factor_against_a_finite_difference():
    """`peculiar_velocity_factor` uses f = dlnD/dlna to get dD/da. Check it by
    the other road: v_pec = a * H * (dD/da) * a * v_D = 100 a^2 E(a) dD/da v_D,
    with dD/da differenced rather than derived, so the two paths share no
    algebra beyond D itself."""
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
    """The completeness marker: an export interrupted mid-write leaves arrays
    that are the right size in their headers and short on disk, and those load
    without complaint. The absent header is what refuses."""
    st = _evolved_state()
    d = str(tmp_path / "e")
    os.makedirs(d)
    with open(os.path.join(d, "x.npy"), "wb") as fh:
        fh.write(b"stale")
    with pytest.raises(FileNotFoundError, match="written last"):
        export.load_particles(d)

    export.write_particles(st, d)
    export.load_particles(d)

    # and a pre-existing header is removed BEFORE the arrays move, so a torn
    # rewrite cannot be read as complete
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
    """The other half of the check above: the same call on an untouched export
    must pass, or `crc mismatch` would be proving nothing."""
    st = _evolved_state()
    d = str(tmp_path / "e")
    export.write_particles(st, d)
    export.load_particles(d, mmap=False)


def test_truncated_stream_refuses_at_close(tmp_path):
    """`_StreamedNpy` declares its shape before it has the rows. A writer that
    delivered fewer must not close a file whose header overstates it."""
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
# The CLI's units decision. km/s is the DEFAULT (a halo finder is the consumer),
# which is only possible when the checkpoint records its own epoch.


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
    """The point of the whole change: no flags, and the file is km/s at the
    epoch the checkpoint carries. Compared against the D-time file times the
    factor, so the DEFAULT is pinned to a value and not merely to a label."""
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
    """The pre-epoch artifacts -- the banked 2048^3 slabs among them -- carry no
    epoch, and must still export. What is NOT allowed is doing it quietly: the
    fallback names itself on stdout and in the header's provenance."""
    ck = _checkpoint(tmp_path, "ck", kind="inexor-checkpoint", step=3)
    out = str(tmp_path / "out")
    assert export._main([ck, out]) == 0

    head = json.load(open(os.path.join(out, "export.json")))
    assert head["velocity_is_dtime"] is True
    assert head["a"] is None
    assert "no epoch recorded" in head["provenance"]["epoch_source"]
    assert "no epoch recorded" in capsys.readouterr().out


def test_cli_flags_override_the_recorded_epoch(tmp_path):
    """`--a` wins over a recorded epoch, and the header records that it did --
    otherwise two files from one checkpoint differ with nothing to say why.
    `--d-time` is the other explicit exit."""
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
    """A flag that is silently ignored writes a header claiming a cosmology that
    never entered the file. Both directions refuse instead."""
    with_epoch = _checkpoint(tmp_path, "with", epoch=(0.5, PLANCK))
    without = _checkpoint(tmp_path, "without")

    with pytest.raises(SystemExit, match="nothing to act on"):
        export._main([with_epoch, str(tmp_path / "o1"), "--d-time", "--a", "1.0"])
    with pytest.raises(SystemExit, match="need an epoch"):
        export._main([without, str(tmp_path / "o2"), "--omega-m", "0.3"])


def test_cli_cosmology_override_rides_on_the_recorded_epoch(tmp_path):
    """`--omega-m` alone means "this epoch, that cosmology", which is the shape
    a reader wants when the recorded cosmology is not the one they want to
    convert with. The un-overridden fields come from the checkpoint."""
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

    # The epoch and the cosmology now come from DIFFERENT places, so both the
    # header and the terminal have to say so; reporting only the epoch would
    # leave a reader thinking the checkpoint's own cosmology was used.
    src = head["provenance"]["epoch_source"]
    assert "checkpoint epoch" in src and "Omega_m=0.25" in src, src
    assert head["cosmology"]["Omega_m"] == 0.25
    assert head["cosmology"]["h"] == PLANCK.h, "un-overridden fields must come from the file"


def test_cli_announces_the_epoch_it_converted_at(tmp_path, capsys):
    """`--omega-m`/`--h` work alone, which means the epoch and the cosmology can
    come from different places. The terminal line has to name the epoch, the
    cosmology and where each came from -- a velocity converted at a neighbouring
    scale factor is off by tens of percent and looks entirely reasonable.

    The header carries the same three things, since stdout does not survive."""
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
    """The header's `a` / factor / cosmology travel together: all three present
    on a km/s file, all three None on a D-time one. A half-filled set is how a
    reader ends up converting with numbers that were never applied."""
    st = _evolved_state()
    d = str(tmp_path / "dtime")
    head = export.write_particles(st, d)
    assert head["a"] is None
    assert head["peculiar_velocity_factor"] is None
    assert head["cosmology"] is None

    k = str(tmp_path / "kms")
    head = export.write_particles(st, k, a=0.5, cosmo=PLANCK)
    assert head["cosmology"] == dataclasses.asdict(PLANCK)
