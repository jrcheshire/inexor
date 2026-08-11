"""ooc_fft.py: the out-of-core FFT layer (M-v2-5, D-v2-15 clause 4).

The layer's whole license is that streamed and monolithic are the SAME
computation at different loop bounds, so nearly everything here is a bitwise
assertion plus the perturbation twin that proves it can fail. The one-plane
compute unit exists because the batched alternative FAILED this file's
invariance test on first contact (pocketfft results are batch-size dependent
at the bit level -- 341/68 differing elements on a 32^3 field for pass 1/2;
module docstring); these tests are what keep that from regressing.
"""

import numpy as np
import pytest

from inexor import ooc_fft

N = 32
L = 64.0


@pytest.fixture(scope="module")
def field64():
    rng = np.random.default_rng(11)
    return rng.standard_normal((N, N, N), dtype=np.float64)


@pytest.fixture(scope="module")
def field32():
    rng = np.random.default_rng(11)
    return rng.standard_normal((N, N, N), dtype=np.float32)


def _fwd(field, slab):
    return ooc_fft.forward_from_slabs(lambda lo, hi: field[lo:hi], N, slab=slab)


def test_forward_is_bitwise_invariant_to_slab_thickness(field64, field32):
    """Thickness 1 / ragged 7 / 8 / N all produce the identical spectrum --
    the decomposition-invariance half of the M-v2-5 exit gate, at the FFT."""
    for field in (field64, field32):
        ref = ooc_fft.rfftn_ooc(field)
        for slab in (1, 7, 8, N):
            got = _fwd(field, slab)
            assert got.dtype == ref.dtype
            assert np.array_equal(got, ref), f"slab={slab} moved bits ({field.dtype})"


def test_forward_is_bitwise_invariant_to_workers(field64):
    a = ooc_fft.forward_from_slabs(lambda lo, hi: field64[lo:hi], N, workers=1)
    b = ooc_fft.forward_from_slabs(lambda lo, hi: field64[lo:hi], N, workers=-1)
    assert np.array_equal(a, b), "worker count moved bits"


def test_inverse_is_bitwise_invariant_to_slab(field64):
    spec = ooc_fft.rfftn_ooc(field64)
    ref = ooc_fft.irfftn_ooc(spec.copy(), N)
    for slab in (1, 7, 8, N):
        out = np.empty_like(field64)
        for lo, s in ooc_fft.inverse_to_slabs(spec.copy(), N, slab=slab):
            out[lo : lo + s.shape[0]] = s
        assert np.array_equal(out, ref), f"inverse slab={slab} moved bits"


def test_invariance_can_fail(field64):
    """Anti-vacuity: a ulp-level perturbation of one plane must change the
    spectrum, or the bitwise comparisons above assert nothing."""
    ref = ooc_fft.rfftn_ooc(field64)
    bumped = field64.copy()
    bumped[13, 0, 0] = np.nextafter(bumped[13, 0, 0], np.inf)
    assert not np.array_equal(ooc_fft.rfftn_ooc(bumped), ref)


def test_inverse_mutates_its_input(field64):
    """The documented ownership handoff: inverse_to_slabs runs its axis-0 pass
    in place, so the caller's spectrum is consumed."""
    spec = ooc_fft.rfftn_ooc(field64)
    keep = spec.copy()
    list(ooc_fft.inverse_to_slabs(spec, N))
    assert not np.array_equal(spec, keep)


def test_roundtrip_hits_the_f64_floor(field64):
    spec = ooc_fft.rfftn_ooc(field64)
    back = ooc_fft.irfftn_ooc(spec, N)
    rel = np.max(np.abs(back - field64)) / np.max(np.abs(field64))
    assert rel < 1e-14, f"f64 roundtrip residual {rel:.2e}"


def test_roundtrip_f32_dtype_and_floor(field32):
    spec = ooc_fft.rfftn_ooc(field32)
    assert spec.dtype == np.complex64, "pocketfft must preserve single precision"
    back = ooc_fft.irfftn_ooc(spec, N)
    assert back.dtype == np.float32
    rel = np.max(np.abs(back - field32)) / np.max(np.abs(field32))
    assert rel < 1e-5, f"f32 roundtrip residual {rel:.2e}"


def test_matches_numpy_rfftn_at_tolerance(field64):
    """Tolerance ONLY, by design: a different transform order rounds
    differently, and claiming bitwise here would be claiming someone else's
    implementation detail."""
    mine = ooc_fft.rfftn_ooc(field64)
    theirs = np.fft.rfftn(field64)
    scale = np.max(np.abs(theirs))
    assert np.max(np.abs(mine - theirs)) / scale < 1e-13


# ---------------------------------------------------------------------------
# spectral multipliers
# ---------------------------------------------------------------------------


def _full_kgrid():
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)
    KX = kx.reshape(N, 1, 1)
    KY = kx.reshape(1, N, 1)
    KZ = kz.reshape(1, 1, -1)
    return KX, KY, KZ


def test_mul_radial_matches_full_grid_and_is_slab_invariant(field64):
    f = lambda k: 1.0 / (1.0 + k**2)  # noqa: E731
    KX, KY, KZ = _full_kgrid()
    kk = np.sqrt(KX**2 + KY**2 + KZ**2)
    kk_safe = kk.copy()
    kk_safe[0, 0, 0] = np.abs(2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)[1])
    vals = f(kk_safe)
    vals[0, 0, 0] = 0.0
    spec = ooc_fft.rfftn_ooc(field64)
    ref = spec * vals
    for slab in (1, 7, N):
        got = ooc_fft.mul_radial_inplace(spec.copy(), N, L, f, dc_value=0.0, slab=slab)
        assert np.array_equal(got, ref), f"mul_radial slab={slab} moved bits"
    kept = ooc_fft.mul_radial_inplace(spec.copy(), N, L, f, dc_value=1.0)
    assert kept[0, 0, 0] == spec[0, 0, 0], "dc_value=1.0 must leave DC untouched"


def test_grad_and_deriv2_match_full_grid_and_are_slab_invariant(field64):
    KX, KY, KZ = _full_kgrid()
    k2 = KX**2 + KY**2 + KZ**2
    k2[0, 0, 0] = 1.0  # forces.k_components convention
    spec = ooc_fft.rfftn_ooc(field64)
    comps = (KX, KY, KZ)
    for ax in range(3):
        ref = spec * ((1j * comps[ax]) / k2)
        for slab in (1, 7, N):
            got = ooc_fft.grad_invk2_spec(spec, ax, N, L, slab=slab)
            assert np.array_equal(got, ref), f"grad axis={ax} slab={slab} moved bits"
        assert got[0, 0, 0] == 0.0, "DC gradient must vanish"
    for i, j in ((0, 0), (0, 1), (1, 2), (2, 2)):
        ref = spec * (-(comps[i] * comps[j]) / k2)
        for slab in (1, 7, N):
            got = ooc_fft.deriv2_spec(spec, i, j, N, L, slab=slab)
            assert np.array_equal(got, ref), f"deriv2 ({i},{j}) slab={slab} moved bits"


# ---------------------------------------------------------------------------
# the accounting function
# ---------------------------------------------------------------------------


def test_plan_bytes_terms_and_policies():
    p = ooc_fft.plan_bytes(2048, np.float32, "forward")
    # one complex64 half-grid at 2048^3 is ~34.4 GB; forward peak is spec +
    # O(slab) buffers, i.e. the D-v2-15 design point of ~35.5 GB
    assert 34e9 < p["spec"] < 35e9
    assert p["peak"] < 37e9
    d = ooc_fft.plan_bytes(2048, np.float32, "derivative")
    assert d["peak"] > 2 * p["spec"]
    r = ooc_fft.plan_bytes(2048, np.float32, "roundtrip")
    assert r["field"] == 2048**3 * 4
    with pytest.raises(ValueError, match="unknown policy"):
        ooc_fft.plan_bytes(64, np.float32, "spill")


def test_require_fits_refuses_loudly():
    """The refusal fires with the arithmetic in the message (D-007 discipline),
    and f64 derivative streams at 2048^3 are over a 116 GB host by DESIGN."""
    with pytest.raises(MemoryError, match="refusing rather than paging"):
        ooc_fft.require_fits(2048, np.float64, "derivative", budget_bytes=116e9)
    plan = ooc_fft.require_fits(2048, np.float32, "derivative", budget_bytes=116e9)
    assert plan["peak"] < 116e9
