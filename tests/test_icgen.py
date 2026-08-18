"""icgen.py: the streamed T9 generator and loader (M-v2-5, D-v2-15 clause 3).

THE THIRD M-v2-5 IDENTITY GATE at unit scale: generate_t9_slabs -> disk ->
load_slot_state must be bitwise SlotState.build fed the monolithic
linear_density -> lpt_ics chain on the same seed. Everything else here is the
refusal surface: the manifest-last contract, crc corruption, and the
displacement bound, each proven able to fire.
"""

import json
import os
import zlib

import jax
import numpy as np
import pytest

from inexor import ic, icgen, lpt, state
from inexor.codec import T9Layout
from inexor.config import Cosmology

N, L, NB, A_INIT = 32, 32.0, 4, 0.1
SLAB = 5  # deliberately ragged against both N and the brick-slab depth


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _monolithic(key, cosmo, f_NL=0.0):
    d0 = ic.linear_density(key, N, L, cosmo, f_NL=f_NL, fdtype=np.float64)
    x, v = lpt.lpt_ics(d0, L, A_INIT, cosmo, order=2, fdtype=np.float64)
    t9 = T9Layout(L, N, 2)
    return state.SlotState.build(x, v, t9, NB)


@pytest.mark.parametrize("f_NL", [0.0, 10.0])
def test_streamed_build_is_bitwise_the_monolithic_one(tmp_path, f_NL):
    cosmo = Cosmology()
    key = jax.random.PRNGKey(0)
    man = icgen.generate_t9_slabs(
        str(tmp_path), key, N, L, cosmo, A_INIT, NB, f_NL=f_NL, fdtype=np.float64, slab=SLAB
    )
    st = icgen.load_slot_state(str(tmp_path))
    ref = _monolithic(key, cosmo, f_NL=f_NL)

    assert np.array_equal(st.brick_start, ref.brick_start)
    assert np.array_equal(st.occupancy, ref.occupancy)
    assert st.occupancy.dtype == ref.occupancy.dtype
    assert np.array_equal(st.off, ref.off), "position payload moved bits"
    assert np.array_equal(st.w, ref.w), "velocity payload moved bits"
    assert np.array_equal(st.vel_scale, ref.vel_scale), (
        "the per-brick scales must be EXACT, brick for brick: the streamed\n"
        "generator and the monolithic build take the same max over the same rows"
    )
    assert st.n_particles == ref.n_particles == N**3
    assert st.arena_base == ref.arena_base and st.n_arena == ref.n_arena
    assert st.arena_used == 0
    st.check()

    # occupancy is non-degenerate (a flat field would pass vacuously)
    occ = np.asarray(st.occupancy, np.int64)
    assert occ.max() >= 2 * max(occ[occ > 0].mean(), 1)
    assert man["ic_stream"] == ic.IC_STREAM
    # the manifest keeps ONE number, and it must be the max over the per-brick
    # scales: bricks partition the particles, so the largest brick scale is the
    # global max|v|/INT16_MAX the manifest records.
    assert man["vel_scale"] == pytest.approx(st.vel_scale.max())
    assert st.vel_scale.shape == (NB**3,)
    assert st.vel_scale.min() < st.vel_scale.max(), (
        "every brick got the same scale, so this fixture cannot tell a per-brick\n"
        "scale from a global one and the comparison above is vacuous"
    )


def test_streamed_build_identity_can_fail(tmp_path):
    """Anti-vacuity: a different seed's slabs must NOT reproduce the build."""
    cosmo = Cosmology()
    icgen.generate_t9_slabs(
        str(tmp_path), jax.random.PRNGKey(1), N, L, cosmo, A_INIT, NB, fdtype=np.float64
    )
    st = icgen.load_slot_state(str(tmp_path))
    ref = _monolithic(jax.random.PRNGKey(0), cosmo)
    assert not np.array_equal(st.off, ref.off)


def test_loader_refuses_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest is written last"):
        icgen.load_slot_state(str(tmp_path))


def test_loader_refuses_crc_corruption(tmp_path):
    cosmo = Cosmology()
    icgen.generate_t9_slabs(
        str(tmp_path), jax.random.PRNGKey(0), N, L, cosmo, A_INIT, NB, fdtype=np.float64
    )
    victim = os.path.join(str(tmp_path), "t9_slab_0001.npz")
    with np.load(victim) as z:
        meta, occ, off, w, sc = (
            str(z["meta"]), z["occupancy"], z["off"], z["w"], z["scale"]
        )
    w = w.copy()
    w[0, 0] ^= 1  # one flipped bit in one velocity code
    np.savez(victim, meta=meta, occupancy=occ, off=off, w=w, scale=sc)
    with pytest.raises(ValueError, match="crc mismatch"):
        icgen.load_slot_state(str(tmp_path))
    # and the crc actually covers what it claims: restoring the byte loads clean
    w[0, 0] ^= 1
    np.savez(victim, meta=meta, occupancy=occ, off=off, w=w, scale=sc)
    icgen.load_slot_state(str(tmp_path)).check()

    # the SCALE array is covered too. It is new, it is per brick, and a silently
    # wrong scale decodes every velocity in that brick by a wrong factor without
    # touching a single payload byte -- so leaving it out of the crc would be the
    # one corruption the loader could not see.
    sc = sc.copy()
    sc[0] *= 1.5
    np.savez(victim, meta=meta, occupancy=occ, off=off, w=w, scale=sc)
    with pytest.raises(ValueError, match="crc mismatch"):
        icgen.load_slot_state(str(tmp_path))


def test_generator_refuses_displacement_over_the_window(tmp_path):
    """A sigma8 large enough to displace particles past one brick slab must
    trip the measured bound BEFORE emission, with the window in the message."""
    wild = Cosmology(sigma8=25.0)
    with pytest.raises(ValueError, match="sliding window"):
        icgen.generate_t9_slabs(
            str(tmp_path), jax.random.PRNGKey(0), N, L, wild, 1.0, 8, fdtype=np.float64
        )


def test_manifest_is_written_last(tmp_path):
    """Interrupting after slab files exist but before the manifest leaves a
    directory the loader refuses -- simulated by deleting the manifest."""
    cosmo = Cosmology()
    icgen.generate_t9_slabs(
        str(tmp_path), jax.random.PRNGKey(0), N, L, cosmo, A_INIT, NB, fdtype=np.float64
    )
    os.remove(os.path.join(str(tmp_path), icgen.MANIFEST))
    assert any(f.startswith("t9_slab_") for f in os.listdir(str(tmp_path)))
    with pytest.raises(FileNotFoundError):
        icgen.load_slot_state(str(tmp_path))


def test_slab_files_carry_their_own_integrity(tmp_path):
    """Every slab file's crc block matches its arrays (the loader checks this,
    but here it is asserted directly so a writer bug cannot hide behind a
    reader bug)."""
    cosmo = Cosmology()
    man = icgen.generate_t9_slabs(
        str(tmp_path), jax.random.PRNGKey(0), N, L, cosmo, A_INIT, NB, fdtype=np.float64
    )
    assert len(man["files"]) == NB
    for fname in man["files"]:
        with np.load(os.path.join(str(tmp_path), fname)) as z:
            meta = json.loads(str(z["meta"]))
            assert meta["schema"] == icgen.SCHEMA
            for name in ("occupancy", "off", "w"):
                assert zlib.crc32(z[name].tobytes()) == meta["crc32"][name]


# --------------------------------------------------------------------------
# M-v2-6 Stage 4(a): the writer, and the round trip on an EVOLVED state


def _evolved_state(seed=3, n_part=16, nb=4, box=16.0, steps=3, arena_frac=0.05, slack=0.02):
    """A state that has actually been through the engine's exchange: live
    spares occupied and, at this slack, a populated arena. Both are things a
    freshly loaded state never has, and both are what the writer has to
    compact away."""
    rng = np.random.default_rng(seed)
    g = (np.arange(n_part) + 0.5) * (box / n_part)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    x = np.mod(q + rng.normal(scale=0.25 * box / n_part, size=q.shape), box)
    v = rng.normal(scale=0.5, size=x.shape)
    t9 = T9Layout(box, n_part, 2)
    st = state.SlotState.build(
        x, v, t9, nb, brick_slack=slack, arena_frac=arena_frac, with_ids=False
    )
    for _ in range(steps):
        state.drift_and_migrate(st, 0.5)
    return st


def _all_rows(st):
    """Every particle as (bucket, off, w), sorted -- the container's content
    with its allocation layout and intra-bucket order divided out."""
    rows = []
    for b in range(st.n_bricks):
        slots = st.brick_member_slots(b)
        live = st.brick_live_count(b)
        keys = np.concatenate([
            st.bucket_flat_of_live_slots(b),
            st.arena_bucket[slots[live:] - st.arena_base],
        ])
        rows.append(np.column_stack([keys, st.off[slots], st.w[slots]]).astype(np.int64))
    out = np.concatenate(rows)
    return out[np.lexsort(out.T[::-1])]


def _crcs(workdir):
    with open(os.path.join(workdir, icgen.MANIFEST)) as fh:
        man = json.load(fh)
    out = {}
    for f in man["files"]:
        with np.load(os.path.join(workdir, f)) as z:
            out[f] = json.loads(str(z["meta"]))["crc32"]
    return man, out


def test_write_t9_slabs_round_trips_an_evolved_state(tmp_path):
    """The Stage 4(a) gate. Not array equality against `st`: the writer
    compacts, so `brick_start` and intra-bucket row order legitimately move,
    and D-v2-21 established that order carries no physics. The invariant is a
    FIXED POINT -- write, load, write again, byte for byte -- plus particle
    level conservation across the trip."""
    st = _evolved_state()
    assert st.arena_used > 0, "vacuous: this state has no arena residents to fold back"
    assert st.n_live == st.n_particles

    d1, d2 = str(tmp_path / "w1"), str(tmp_path / "w2")
    icgen.write_t9_slabs(st, d1)
    st2 = icgen.load_slot_state(d1)
    icgen.write_t9_slabs(st2, d2)

    man1, c1 = _crcs(d1)
    man2, c2 = _crcs(d2)
    assert c1 == c2, "the writer is not a fixed point: a second trip changed the bytes"
    assert man1["n_particles"] == man2["n_particles"] == st.n_live

    # the trip preserved the PARTICLES, not merely the byte layout
    np.testing.assert_array_equal(_all_rows(st), _all_rows(st2))
    np.testing.assert_array_equal(st.vel_scale, st2.vel_scale)
    st2.check()


def test_write_t9_slabs_does_not_recompute_vel_scale(tmp_path):
    """`generate_t9_slabs` derives each brick's scale from the velocities it is
    encoding. Doing that here would re-encode `w` against a new scale and lose
    bits on any brick whose membership changed, so the scales must ride out
    verbatim. Planting a perturbed scale proves the writer copies rather than
    derives."""
    st = _evolved_state()
    st.vel_scale[:] = st.vel_scale * 1.5
    icgen.write_t9_slabs(st, str(tmp_path))
    st2 = icgen.load_slot_state(str(tmp_path))
    np.testing.assert_array_equal(st.vel_scale, st2.vel_scale)
    np.testing.assert_array_equal(_all_rows(st), _all_rows(st2))


def test_write_t9_slabs_refuses_to_drop_ids_silently(tmp_path):
    """IDs are not in the schema, so writing a state that carries them loses
    data. It must say so rather than succeed quietly."""
    st = _evolved_state()
    st.ids = np.arange(len(st.off), dtype=np.int64)
    with pytest.raises(ValueError, match="drop_ids"):
        icgen.write_t9_slabs(st, str(tmp_path))
    icgen.write_t9_slabs(st, str(tmp_path), drop_ids=True)
    assert icgen.load_slot_state(str(tmp_path)).ids is None


def test_write_t9_slabs_manifest_is_written_last(tmp_path):
    """The completeness contract `load_slot_state` refuses on: slabs without a
    manifest are an interrupted write, not a loadable state."""
    st = _evolved_state()
    icgen.write_t9_slabs(st, str(tmp_path))
    os.remove(str(tmp_path / icgen.MANIFEST))
    with pytest.raises(FileNotFoundError, match="refusing to load"):
        icgen.load_slot_state(str(tmp_path))
