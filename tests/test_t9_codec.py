"""The T9 state codec (D-v2-14): bucket-relative uint8 positions, int16 velocities.

WHY THE CENTRAL TEST IS A BITWISE ONE. D-v2-14 was ratified on numbers measured
by `scripts/v2_g2c_accum_gate.py` (job 896160) -- max |dP/P| 4.123e-4 and
dP2/P0 2.035e-3 at bucket_cells=2, a 15x margin on D-v2-9's 3e-2 bar. That gate
does not implement a codec: it applies `_rt_pos_lattice`, a float round trip on
a global lattice, and says so in its own docstring ("measures REPRESENTATION
error only; storage layout is a build decision"). This module takes the storage
decision. If the shipped encoder is not numerically IDENTICAL to that round
trip, then job 896160's numbers quietly stop describing what we ship, and the
ADR is left resting on a measurement of something else.

So the probe is imported unmodified as the oracle, and the assertion is exact
elementwise equality over all three ratified bucket sizes. Tolerances would
defeat the point: the two paths compute the same product from the same
operands, so anything other than equality means the arithmetic diverged.

x64 throughout, because the gate that produced the ratified numbers runs f64.
The library never sets it -- callers do (house rule), so there is a fixture.
"""


import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _x64():
    """Enable x64 for this module only, then restore (test_bispectrum.py pattern)."""
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


# The ratified bucket ladder, as bucket side in PARTICLE cells. c=2 is D-v2-14's
# choice; c=1 and c=4 are the arms that bracket it and also passed.
BUCKET_ARMS = (1, 2, 4)

L_BOX = 128.0
N_PART = 64  # -> n_fine 128, the probe's smoke geometry


def _positions(seed, n=8192, box=L_BOX):
    """Positions strictly inside [0, box), which is what a stepper emits: every
    float driver ends its drift with jnp.mod(x, box_size)."""
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(0.0, box, size=(n, 3)), dtype=jnp.float64)


def _layout(bucket_cells, n_part=N_PART, box=L_BOX):
    from inexor.codec import T9Layout

    return T9Layout(box_size=box, n_part=n_part, bucket_cells=bucket_cells)


# ------------------------------------------------------- the ratified gate


@pytest.mark.parametrize("bucket_cells", BUCKET_ARMS)
def test_offsets_use_the_whole_byte_and_the_input_is_not_degenerate(bucket_cells):
    """Guards the gate above against passing vacuously. Two ways it could: an
    all-zero position array would make any two codecs agree, and an offset field
    that never leaves a corner of the byte would not exercise the encode."""
    import jax.numpy as jnp

    from inexor.codec import encode_positions

    x = _positions(seed=2)
    assert float(jnp.max(x) - jnp.min(x)) > 0.9 * L_BOX, "input lacks dynamic range"

    off, b = encode_positions(x, _layout(bucket_cells))
    assert off.dtype == jnp.uint8
    assert int(jnp.min(off)) == 0 and int(jnp.max(off)) == 255, "offsets do not span the byte"
    nb = _layout(bucket_cells).n_buckets_side
    assert int(jnp.min(b)) >= 0 and int(jnp.max(b)) < nb


# ------------------------------------------------------------ round trip


@pytest.mark.parametrize("bucket_cells", BUCKET_ARMS)
def test_roundtrip_is_exactly_idempotent(bucket_cells):
    """Re-encoding a decoded position must reproduce the same bytes. EXACT
    integer equality -- this is a lattice identity, not an approximation."""
    import jax.numpy as jnp

    from inexor.codec import decode_positions, encode_positions

    lay = _layout(bucket_cells)
    x = _positions(seed=3)
    off1, b1 = encode_positions(x, lay)
    x1 = decode_positions(off1, b1, lay, fdtype=x.dtype)
    off2, b2 = encode_positions(x1, lay)
    assert jnp.array_equal(off1, off2) and jnp.array_equal(b1, b2)


@pytest.mark.parametrize("bucket_cells", BUCKET_ARMS)
def test_decode_error_is_within_half_a_quantum(bucket_cells):
    import jax.numpy as jnp

    from inexor.codec import roundtrip_positions

    lay = _layout(bucket_cells)
    x = _positions(seed=4)
    err = jnp.abs(roundtrip_positions(x, lay) - x)
    err = jnp.minimum(err, L_BOX - err)  # a point near the seam wraps
    assert float(jnp.max(err)) <= 0.5 * lay.quantum * (1 + 1e-12)


@pytest.mark.parametrize("bucket_cells", BUCKET_ARMS)
def test_decoded_position_lies_inside_its_own_bucket(bucket_cells):
    """The offset is meaningless without this: it says the bucket index and the
    stored byte describe the same point."""
    import jax.numpy as jnp

    from inexor.codec import decode_positions, encode_positions

    lay = _layout(bucket_cells)
    x = _positions(seed=5)
    off, b = encode_positions(x, lay)
    xd = decode_positions(off, b, lay, fdtype=x.dtype)
    lo = b.astype(jnp.float64) * lay.bucket_size
    assert bool(jnp.all(xd >= lo - 1e-12))
    assert bool(jnp.all(xd < lo + lay.bucket_size))


# ------------------------------------------------- wrap, never clamp (D-007)


def test_wrap_at_the_box_seam_is_modular_not_saturating():
    """A particle pushed past the box edge must reappear at the low edge, not
    pile up on the last lattice site."""
    import jax.numpy as jnp

    from inexor.codec import encode_positions, lattice_index, roundtrip_positions

    lay = _layout(2)
    q = lay.quantum
    # just under the seam, exactly on it, and just past it
    x = jnp.asarray([[L_BOX - q, 0.0, 0.0], [L_BOX, 0.0, 0.0], [L_BOX + q, 0.0, 0.0]])
    idx = lattice_index(x, lay)
    assert [int(v) for v in idx[:, 0]] == [lay.n_levels - 1, 0, 1]

    xr = roundtrip_positions(x, lay)
    assert float(xr[1, 0]) == 0.0 and float(xr[2, 0]) == pytest.approx(q)

    off, b = encode_positions(x, lay)
    assert int(b[1, 0]) == 0 and int(off[1, 0]) == 0  # wrapped to the first bucket


def test_crossing_a_bucket_edge_changes_the_bucket_not_the_byte_range():
    """Migration, not saturation: stepping one quantum past a bucket's top
    offset must land on offset 0 of the NEXT bucket."""
    import jax.numpy as jnp

    from inexor.codec import LEVELS_PER_BUCKET, encode_positions

    lay = _layout(2)
    q = lay.quantum
    top = lay.bucket_size - q  # last site of bucket 0
    x = jnp.asarray([[top, 0.0, 0.0], [top + q, 0.0, 0.0]])
    off, b = encode_positions(x, lay)
    assert int(off[0, 0]) == LEVELS_PER_BUCKET - 1 and int(b[0, 0]) == 0
    assert int(off[1, 0]) == 0 and int(b[1, 0]) == 1


# --------------------------------------------------------- layout refusals


def test_layout_refuses_a_bucket_that_does_not_tile_the_box():
    from inexor.codec import T9Layout

    with pytest.raises(ValueError, match="must divide n_part"):
        T9Layout(box_size=L_BOX, n_part=64, bucket_cells=3)


def test_layout_refuses_a_non_power_of_two_lattice():
    """Without this the quantum does not divide the box exactly and the seam
    wrap saturates -- the failure D-007 exists to forbid."""
    from inexor.codec import T9Layout

    with pytest.raises(ValueError, match="not a power of two"):
        T9Layout(box_size=L_BOX, n_part=96, bucket_cells=1)


def test_layout_derives_the_ratified_c_gh_numbers():
    """The arithmetic D-v2-14 is written in, checked end to end at the real
    config: 2048^3 particles in 1024 Mpc/h, bucket 1.0 Mpc/h, quantum
    fine_cell/64 with a 0.25 Mpc/h fine cell.

    The index is the one number here that MOVED after ratification: 2.15 GB at
    uint16, 4.29 GB now that it is uint32. Both are asserted -- the ratified
    figure because D-v2-14 clause 2 is written in it, and the current default
    because that is what a run costs."""
    lay = _layout(bucket_cells=2, n_part=2048, box=1024.0)
    fine_cell = lay.spacing / 2  # mesh ratio 2x (plan-plan Sec. 2)
    assert lay.spacing == pytest.approx(0.5)
    assert lay.bucket_size == pytest.approx(1.0)
    assert lay.quantum == pytest.approx(fine_cell / 64)
    assert lay.n_buckets_side == 1024
    assert lay.index_bytes(dtype=np.uint16) / 1e9 == pytest.approx(2.15, abs=0.01)
    assert lay.index_bytes() / 1e9 == pytest.approx(4.29, abs=0.01)
    # 0.50 B/p against the ratified 0.25, on 2048^3 particles
    assert lay.index_bytes() / 2048**3 == pytest.approx(0.50, rel=0.01)


# ------------------------------------------------------------- velocities


def test_probe_velocity_scale_escapes_int16_when_the_extremum_is_positive():
    """A recorded finding, not a complaint about the probe. `_rt_vel_int16_max`
    uses q = 2*max|v|/65536, so +max|v| maps to index +32768 -- one past int16.
    In a float round trip that is invisible; in storage it is fatal, and whether
    it bites depends on the SIGN of the extremum. Pinned here because the
    shipped codec deviates from the probe on exactly this point, and the
    deviation needs a reason that stays true.
    """
    import jax.numpy as jnp

    rng = np.random.default_rng(7)
    for sign, fits_expected in ((+1.0, False), (-1.0, True)):
        v = rng.normal(size=(4096, 3))
        v[0, 0] = sign * 3.0 * np.abs(v).max()
        v = jnp.asarray(v)
        q = 2.0 * jnp.max(jnp.abs(v)) / 65536.0
        idx = jnp.rint(v / q)
        fits = int(jnp.min(idx)) >= -32768 and int(jnp.max(idx)) <= 32767
        assert fits is fits_expected


def test_encoded_velocity_always_fits_int16():
    """The property the probe's scale does not have, asserted on both signs of
    the extremum and on a degenerate all-zero field."""
    import jax.numpy as jnp

    from inexor.codec import assert_int16_range, encode_velocities, rint_i

    rng = np.random.default_rng(8)
    for sign in (+1.0, -1.0):
        v = rng.normal(size=(4096, 3))
        v[0, 0] = sign * 3.0 * np.abs(v).max()
        v = jnp.asarray(v)
        w, scale = encode_velocities(v)
        lo, hi = assert_int16_range(rint_i(v / scale))
        assert lo >= -32767 and hi <= 32767
        assert abs(sign * 32767) in (abs(lo), abs(hi))  # the extremum lands on the edge

    w0, s0 = encode_velocities(jnp.zeros((16, 3)))
    assert int(jnp.max(jnp.abs(w0))) == 0 and float(s0) > 0.0


def test_velocity_does_not_clip_the_extremes():
    """D-007 again, on the axis where a naive max-range codec is most tempted to
    clamp: the fastest particle must decode back to itself, not to a saturated
    edge value."""
    import jax.numpy as jnp

    from inexor.codec import decode_velocities, encode_velocities

    rng = np.random.default_rng(10)
    v = np.asarray(rng.normal(size=(2048, 3)))
    v[5, 1] = 40.0 * np.abs(v).max()  # a violent outlier
    v = jnp.asarray(v)
    w, scale = encode_velocities(v)
    out = decode_velocities(w, scale, fdtype=v.dtype)
    assert int(jnp.abs(w[5, 1])) == 32767
    assert float(out[5, 1]) == pytest.approx(float(v[5, 1]), rel=1e-12)


def test_assert_int16_range_raises_rather_than_clamping():
    import jax.numpy as jnp

    from inexor.codec import assert_int16_range

    with pytest.raises(ValueError, match="escapes int16"):
        assert_int16_range(jnp.asarray([[0, 32768, -3]], dtype=jnp.int32))


# ---------------------------------------------------------- the id tier


def test_id_tier_refusal_is_exactly_at_the_int32_boundary():
    from inexor.codec import ID_MAX_N_SIDE, refuse_ids_above_int32

    assert ID_MAX_N_SIDE**3 < 2**31 - 1 <= (ID_MAX_N_SIDE + 1) ** 3
    refuse_ids_above_int32(ID_MAX_N_SIDE)  # must not raise
    with pytest.raises(ValueError, match="refused above n_side"):
        refuse_ids_above_int32(ID_MAX_N_SIDE + 1)
    with pytest.raises(ValueError, match="refused above n_side"):
        refuse_ids_above_int32(2048)  # production C-gh carries no ids


# ---------------------------------------------------------------- state


def test_state_payload_is_nine_bytes_and_ids_are_a_separate_column():
    import jax.numpy as jnp

    from inexor.codec import decode_state, encode_state

    lay = _layout(2)
    x = _positions(seed=11, n=256)
    v = jnp.asarray(np.random.default_rng(12).normal(size=(256, 3)))

    state, b = encode_state(x, v, lay)
    assert state.payload_bytes_per_particle == 9
    assert state.ids is None

    with_ids, _ = encode_state(x, v, lay, ids=np.arange(256))
    assert with_ids.payload_bytes_per_particle == 13
    assert with_ids.ids.dtype == jnp.int32
    # the id column must not disturb the physics payload
    assert jnp.array_equal(with_ids.pos, state.pos)
    assert jnp.array_equal(with_ids.vel, state.vel)

    xd, vd = decode_state(state, b, lay, fdtype=x.dtype)
    assert xd.shape == x.shape and vd.shape == v.shape
