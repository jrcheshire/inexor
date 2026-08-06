"""Position-dependent P(k) (diagnostics.subvolume_response) -- the amplitude-side
squeezed statistic G3 Stage 5 reads where the bispectrum ratio has lost its
meaning.

TOLERANCES ARE MEASURED, NOT ASSUMED. The partition and guard checks are exact
(integer bookkeeping). The recovery check is statistical -- it regresses a
sub-volume scatter -- so its bound is set from the measured deviation on these
fixtures and the measured value is quoted next to it.

Pure numpy: this estimator has no JAX path, so unlike test_bispectrum.py there
is no x64 fixture to manage.
"""

import numpy as np
import pytest

from inexor.diagnostics import _straddle_fraction, _subvolume_blocks, subvolume_response

L_BOX = 128.0
N_MESH = 128
K_LONG_MAX = 4  # long field: |k| <= 4 k_f
K_SHORT_MIN = 8  # small-scale field: |k| >= 8 k_f
CENTERS = (0.6, 0.9)
DK = 0.2


def _split_fields(n, ell, seed):
    """A smooth long-wavelength field and a band-limited small-scale field."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n, n, n))
    wk = np.fft.rfftn(white)
    kf = 2.0 * np.pi / ell
    k1 = 2.0 * np.pi * np.fft.fftfreq(n, d=ell / n)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=ell / n)
    kmag = np.sqrt(k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + kz[None, None, :] ** 2)
    long_k = np.where(kmag <= K_LONG_MAX * kf, wk, 0.0)
    short_k = np.where(kmag >= K_SHORT_MIN * kf, wk, 0.0)
    long_f = np.fft.irfftn(long_k, s=(n, n, n), axes=(0, 1, 2))
    short_f = np.fft.irfftn(short_k, s=(n, n, n), axes=(0, 1, 2))
    long_f = 0.2 * long_f / long_f.std()
    short_f = short_f / short_f.std()
    return long_f, short_f


def _modulated(n, ell, seed, amp):
    """delta = long + short * (1 + amp * long).

    Inside a sub-volume the small-scale power is scaled by (1 + amp*long_bar)^2
    while the sub-volume mean overdensity IS long_bar (the small-scale part
    averages away), so the exact expected response is dlnP/ddelta_bar = 2*amp.
    No fitted constant, which is what makes this a real oracle.
    """
    long_f, short_f = _split_fields(n, ell, seed)
    return long_f + short_f * (1.0 + amp * long_f)


def _block_oracle(n, n_sub, seed, amp):
    """A field whose response is EXACT, not statistical.

    Every sub-volume gets the same small-scale realization (the pattern is
    periodic with the sub-volume), and is then multiplied by (1 + amp*c_i) with
    c_i a per-sub-volume constant that IS that sub-volume's mean overdensity. So
    P_sub = P_block * (1 + amp*c_i)^2 with no cosmic variance between blocks, and

        slope = 2*amp / (1 + amp^2 var(c))

    exactly, in the limit where the regression is linear. The realistic fixture
    above cannot do this job: with 64 sub-volumes of band-limited noise the
    fitted slope carries a 15-60% error bar, so it can confirm a null but cannot
    certify a scale.
    """
    rng = np.random.default_rng(seed)
    s = n // n_sub
    short = np.tile(rng.standard_normal((s, s, s)), (n_sub, n_sub, n_sub))
    c = rng.standard_normal((n_sub, n_sub, n_sub))
    c = 0.2 * (c - c.mean()) / c.std()
    c_field = np.repeat(np.repeat(np.repeat(c, s, axis=0), s, axis=1), s, axis=2)
    return c_field + short * (1.0 + amp * c_field), float(c.var())


def _run(delta, n_sub, **kw):
    return subvolume_response(delta, L_BOX, n_sub, CENTERS, dk=DK, **kw)


def test_blocks_are_an_exact_partition():
    """Every cell lands in exactly one sub-volume, for any offset."""
    rng = np.random.default_rng(0)
    d = rng.standard_normal((N_MESH, N_MESH, N_MESH))
    for n_sub in (2, 4, 8):
        for off in (0, 3, N_MESH // n_sub // 2, N_MESH - 1):
            blocks = _subvolume_blocks(d, n_sub, off)
            assert blocks.shape == (n_sub**3, N_MESH // n_sub, N_MESH // n_sub, N_MESH // n_sub)
            assert np.array_equal(np.sort(blocks.ravel()), np.sort(d.ravel()))
            assert blocks.sum() == pytest.approx(d.sum(), rel=0, abs=1e-9)


def test_straddle_fraction_matches_a_cell_level_count():
    """The guard's arithmetic against an independent brute-force count.

    The reference assigns every CELL a tile id and calls a sub-volume nested iff
    all of its cells share one id -- a different computation from the wall-
    interval arithmetic in _straddle_fraction, so agreement is evidence.

    VERIFIED TO DISCRIMINATE: closing the wall interval (`lo <= w`) gives 6
    disagreements and dropping its second period (`range(0, n_mesh, ...)`) gives
    8. A third mutation, shrinking the interval by one cell, is caught ONLY by
    the offsets 1 and s-1 below -- they are what put a tile wall on a block's
    first and last cell, and without them that mutation passes silently.
    """
    for n_sub in (2, 4, 8):
        for tps in (2, 4, 8):
            s = N_MESH // n_sub
            for off in (0, 1, s // 2, s - 1):
                tile_cells = N_MESH // tps
                ids = (np.arange(N_MESH) // tile_cells).astype(np.int64)
                ids = np.roll(ids, -off).reshape(n_sub, s)
                nested_axis = int(sum(1 for row in ids if len(set(row.tolist())) == 1))
                want = 1.0 - (nested_axis / float(n_sub)) ** 3
                got = _straddle_fraction(N_MESH, n_sub, off, tps)
                assert got == pytest.approx(want, abs=1e-12), (n_sub, tps, off)


def test_nested_lattice_raises():
    """A sub-volume lattice sitting inside the tiles is blind to the seams."""
    d = _modulated(N_MESH, L_BOX, 3, 0.0)
    with pytest.raises(ValueError, match="blind to the seams"):
        _run(d, 4, offset_frac=0.0, tiles_per_side=4)
    # the same lattice, offset off the tile walls, is fine and straddles
    res = _run(d, 4, offset_frac=0.5, tiles_per_side=4)
    assert res["straddle_frac"] == 1.0


def test_center_below_subvolume_fundamental_raises():
    d = _modulated(N_MESH, L_BOX, 4, 0.0)
    # n_sub=8 -> sub-box 16 Mpc/h -> k_f_sub = 0.393; 0.2 has no modes under it
    with pytest.raises(ValueError, match="sub-volume fundamental"):
        subvolume_response(d, L_BOX, 8, (0.2, 0.9), dk=DK)


def test_unmodulated_field_gives_zero_response():
    """The null: small-scale power that does not track the long mode.

    delta = long + short with no coupling, so the true slope is 0 and what the
    estimator returns is its own noise. Bounded against the fitted error bar,
    not against an absolute number, because the noise depends on the fixture.
    """
    d = _modulated(N_MESH, L_BOX, 5, 0.0)
    res = _run(d, 4, tiles_per_side=4)
    z = np.abs(res["slope"]) / res["slope_err"]
    assert (z < 3.0).all(), f"null slope {res['slope']} +- {res['slope_err']}"


@pytest.mark.parametrize("amp", [0.1, 0.25, 0.5])
def test_injected_modulation_is_recovered(amp):
    """The oracle: an injected coupling of known size comes back as 2*amp.

    Uses the block-periodic construction, where the expected slope is exact
    (see _block_oracle) rather than a statistical expectation. The lattice is
    unoffset here ON PURPOSE -- the construction defines its blocks on that
    lattice, and this test is about the estimator's arithmetic, not the seam
    guard, which test_nested_lattice_raises covers.

    MEASURED deviation from 2*amp/(1 + amp^2 var(c)) over seeds 11-14 and three
    amplitudes: worst 1.71e-2 relative, and it scales LINEARLY in amp
    (3.4e-3 / 8.6e-3 / 1.71e-2 at amp = 0.1 / 0.25 / 0.5 on seed 12). That is
    the O(amp^2) skewness term the expected-value formula truncates, not
    estimator error -- the two shell centers return bit-identical slopes, as
    they must when every block carries the same small-scale content. Bound set
    at 5e-2, ~3x the measured worst.
    """
    for seed in (11, 12, 13, 14):
        d, var_c = _block_oracle(N_MESH, 4, seed, amp)
        res = _run(d, 4, offset_frac=0.0)
        want = 2.0 * amp / (1.0 + amp**2 * var_c)
        rel = np.abs(res["slope"] - want) / want
        assert (rel < 5e-2).all(), f"seed {seed} amp {amp}: slope {res['slope']} vs {want}"


def test_response_is_stable_across_subvolume_count():
    """n_sub = 4 vs 8 must agree: the response is a property of the field, not
    of the binning. They share no sub-volume boundary and have different noise,
    so this is a genuine consistency check.
    """
    d = _modulated(N_MESH, L_BOX, 21, 0.5)
    r4 = _run(d, 4, tiles_per_side=4)
    r8 = _run(d, 8, tiles_per_side=4)
    err = np.sqrt(r4["slope_err"] ** 2 + r8["slope_err"] ** 2)
    assert (np.abs(r4["slope"] - r8["slope"]) < 3.0 * err).all(), (r4["slope"], r8["slope"])


def test_bookkeeping_fields_are_reported():
    d = _modulated(N_MESH, L_BOX, 31, 0.5)
    res = _run(d, 4, tiles_per_side=4)
    assert res["n_blocks"] == 64
    assert res["sub_box"] == pytest.approx(L_BOX / 4)
    assert res["k_f_sub"] == pytest.approx(2.0 * np.pi / (L_BOX / 4))
    assert res["offset_cells"] == (N_MESH // 4) // 2
    assert (res["n_modes"] > 0).all()
    assert res["p_sub"].shape == (64, len(CENTERS))
    # equal-volume blocks partitioning the box: the sub-volume means average to
    # the field mean exactly (which is NOT zero -- the fixture keeps its k=0 mode)
    assert float(res["delta_bar"].mean()) == pytest.approx(float(d.mean()), rel=0, abs=1e-14)
