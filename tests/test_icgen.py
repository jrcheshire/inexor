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
