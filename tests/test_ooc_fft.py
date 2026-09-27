"""ooc_fft.py: the out-of-core FFT layer.

Streamed and monolithic transforms are the same computation at different loop bounds, so
most assertions are bitwise, each with a perturbation twin showing it can fail. The host
path transforms one plane at a time because pocketfft is batch-size dependent at the bit
level (341/68 differing elements on a 32^3 field for pass 1/2).
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
    """Slab thickness 1 / ragged 7 / 8 / N all give the bitwise-identical spectrum."""
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
    """A one-ulp perturbation of one element must change the spectrum."""
    ref = ooc_fft.rfftn_ooc(field64)
    bumped = field64.copy()
    bumped[13, 0, 0] = np.nextafter(bumped[13, 0, 0], np.inf)
    assert not np.array_equal(ooc_fft.rfftn_ooc(bumped), ref)


def test_inverse_mutates_its_input(field64):
    """inverse_to_slabs runs its axis-0 pass in place, consuming the caller's spectrum."""
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
    """Tolerance only (1e-13 of peak): a different transform order rounds differently."""
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
    # one complex64 half-grid at 2048^3 is ~34.4 GB; the forward peak is spec plus
    # O(slab) buffers, ~35.5 GB
    assert 34e9 < p["spec"] < 35e9
    assert p["peak"] < 37e9
    d = ooc_fft.plan_bytes(2048, np.float32, "derivative")
    assert d["peak"] > 2 * p["spec"]
    r = ooc_fft.plan_bytes(2048, np.float32, "roundtrip")
    assert r["field"] == 2048**3 * 4
    with pytest.raises(ValueError, match="unknown policy"):
        ooc_fft.plan_bytes(64, np.float32, "spill")


def test_require_fits_refuses_loudly():
    """require_fits raises MemoryError with the arithmetic in the message; f64 derivative
    streams at 2048^3 exceed a 116 GB host, f32 fits."""
    with pytest.raises(MemoryError, match="refusing rather than paging"):
        ooc_fft.require_fits(2048, np.float64, "derivative", budget_bytes=116e9)
    plan = ooc_fft.require_fits(2048, np.float32, "derivative", budget_bytes=116e9)
    assert plan["peak"] < 116e9


# ----------------------------------------------------------------- the device path
#
# Bitwise for anything invariant within a backend, tolerance across backends. Cross-backend
# bitwise is not asserted: CPU jax happens to reproduce scipy exactly, but the GPU
# legitimately need not. `MAX_DEVICE_TRANSFORM_ELEMENTS` is the one gate about correctness
# rather than reproducibility.

pytest.importorskip("jax")


def _fwd_dev(field, slab, **kw):
    return ooc_fft.forward_from_slabs_device(
        lambda lo, hi: field[lo:hi], N, slab=slab, **kw)


def test_device_forward_is_bitwise_invariant_to_slab_thickness(field32):
    """On device too the spectrum is bitwise independent of how many planes are resident,
    so resume or a different window size cannot change the physics."""
    ref = _fwd_dev(field32, N)
    for slab in (1, 7, 8, N):
        got = _fwd_dev(field32, slab)
        assert got.dtype == ref.dtype
        assert np.array_equal(got, ref), f"device slab={slab} moved bits"


def test_device_inverse_is_bitwise_invariant_to_slab(field32):
    spec = _fwd_dev(field32, N)
    ref = np.empty_like(field32)
    for lo, s in ooc_fft.inverse_to_slabs_device(spec.copy(), N, slab=N):
        ref[lo : lo + s.shape[0]] = s
    for slab in (1, 7, 8):
        out = np.empty_like(field32)
        for lo, s in ooc_fft.inverse_to_slabs_device(spec.copy(), N, slab=slab):
            out[lo : lo + s.shape[0]] = s
        assert np.array_equal(out, ref), f"device inverse slab={slab} moved bits"


def test_device_invariance_can_fail(field32):
    """A four-ulp bump of one element must change the device spectrum.

    Four is the smallest bump the f32 transform can see on a 32^3 field (measured ulps ->
    n_diff of 17,408: 1 -> 0, 4 -> 314, 64 -> 4,230, 1024 -> 17,344). The bump uses
    `np.float32(np.inf)`: a bare `np.inf` promotes to f64 and rounds back to a no-op on
    store, which the input-changed assert catches.
    """
    ref = _fwd_dev(field32, N)
    bumped = field32.copy()
    for _ in range(4):
        bumped[13, 0, 0] = np.nextafter(bumped[13, 0, 0], np.float32(np.inf))
    assert bumped[13, 0, 0] != field32[13, 0, 0], "the perturbation was a no-op"
    assert not np.array_equal(_fwd_dev(bumped, N), ref)


def test_device_matches_the_host_path_at_tolerance(field32):
    """Device vs host agree to 1e-5 of the spectrum's rms, a relative bar that cannot pass
    by the array being small."""
    host = ooc_fft.rfftn_ooc(field32.copy())
    dev = _fwd_dev(field32, 8)
    scale = np.sqrt(np.mean(np.abs(host) ** 2))
    assert scale > 1e-3, "vacuous: the reference spectrum is ~zero"
    assert np.max(np.abs(host - dev)) / scale < 1e-5


def test_device_preserves_single_precision(field32):
    """f32 in, complex64 spectrum, f32 back out on the device path."""
    spec = _fwd_dev(field32, 8)
    assert spec.dtype == np.complex64
    out = np.empty_like(field32)
    for lo, s in ooc_fft.inverse_to_slabs_device(spec, N, slab=8):
        out[lo : lo + s.shape[0]] = s
    assert out.dtype == np.float32


def test_f64_on_device_refuses_rather_than_narrowing(field64):
    """With x64 off an f64 field is refused, not transformed at f32 and returned in an f64
    container. The suite runs x64-off by default."""
    import jax

    if jax.config.read("jax_enable_x64"):
        pytest.skip("x64 is on in this session; the narrowing cannot occur")
    with pytest.raises(RuntimeError, match="jax_enable_x64"):
        _fwd_dev(field64, 8)


def test_a_transform_at_the_silent_wrong_bound_is_refused():
    """Transforms of >= 2^31 elements are refused: 1536^3 f32 (3.6e9 elements) came back
    silently wrong on jax 0.10.2 + GB200 (roundtrip 3.8e+3 vs 2.9e-6 at 1024^3)."""
    with pytest.raises(ValueError, match="silently"):
        ooc_fft.refuse_oversize_device_transform(2**31, "test")
    with pytest.raises(ValueError, match="silently"):
        ooc_fft.refuse_oversize_device_transform(1536**3, "the measured case")
    # the bound is not a blanket refusal: just under it passes, as does one 2048^2 plane
    # (the coarse solve's unit, three orders under)
    ooc_fft.refuse_oversize_device_transform(2**31 - 1, "just under")
    ooc_fft.refuse_oversize_device_transform(2048 * 2048, "one coarse plane")


def test_the_batch_knobs_cannot_reach_the_bound_unnoticed(field32):
    """A batch big enough to hit the silent regime is refused, not attempted."""
    with pytest.raises(ValueError, match="silently"):
        ooc_fft.rfft2_planes_device(field32, plane_batch=2**31 // (N * N) + 1)


def test_the_receipt_is_plane_keyed_so_the_field_is_slab_independent():
    """plane_noise is keyed by plane index and seed, so the receipt's field does not
    depend on the slab."""
    a = ooc_fft.plane_noise(N, 5, np.float32, seed=3)
    b = ooc_fft.plane_noise(N, 5, np.float32, seed=3)
    c = ooc_fft.plane_noise(N, 6, np.float32, seed=3)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_the_roundtrip_receipt_reads_the_f32_floor_on_both_paths():
    """roundtrip_residual reads the f32 floor (< 1e-4) on both paths at small n, on a
    non-trivial field, and carries its configuration beside the number."""
    dev = ooc_fft.roundtrip_residual(N, np.float32, slab=8, device=True)
    host = ooc_fft.roundtrip_residual(N, np.float32, slab=8, device=False)
    assert dev["residual"] < 1e-4, dev
    assert host["residual"] < 1e-4, host
    assert dev["rms"] > 0.5, "vacuous: the receipt transformed a ~zero field"
    for k in ("n_mesh", "dtype", "slab", "plane_batch", "pencil_batch"):
        assert k in dev, f"the receipt must carry {k} beside the number"


# ---------------------------------------------------------------------------
# splitting a pass across devices
# ---------------------------------------------------------------------------


def _devices(w):
    """`w` device handles, replicating if the backend has fewer.

    Replication still exercises the partition, threads and write-through views. With
    `--xla_force_host_platform_device_count=4` (or four GPUs) the tests are truly multi-device.
    """
    import jax

    devs = jax.devices()
    return [devs[i % len(devs)] for i in range(w)]


def test_partition_units_reproduces_the_unpartitioned_batch_sequence():
    """Interior boundaries land on `unit` multiples, parts tile [0, total)."""
    for total, w, unit in ((32, 4, 1), (2048, 4, 64), (10, 2, 4), (32, 3, 8)):
        parts = ooc_fft.partition_units(total, w, unit)
        assert len(parts) == w
        assert parts[0][0] == 0 and parts[-1][1] == total
        for (_, a), (b, _) in zip(parts, parts[1:]):
            assert a == b, f"{parts} is not contiguous"
            assert a % unit == 0, f"boundary {a} is not a multiple of {unit}"
        assert all(hi > lo for lo, hi in parts), f"{parts} has an empty part"


def test_partition_units_refuses_a_width_it_cannot_realize():
    """A width that would leave a part empty (a device silently not participating) is
    refused, as is n_parts=0."""
    with pytest.raises(ValueError, match="only 1 whole units"):
        ooc_fft.partition_units(32, 4, 64)
    with pytest.raises(ValueError, match="n_parts"):
        ooc_fft.partition_units(32, 0, 1)


@pytest.mark.parametrize("w", [2, 4])
def test_device_split_is_bitwise_identical_to_one_device(field32, w):
    """W devices == 1 device bitwise, forward and inverse: there is no inter-device
    communication, so a split only partitions a loop."""
    devs = _devices(w)
    for pb, yb in ((1, 1), (1, 8), (2, 4)):
        kw = dict(plane_batch=pb, pencil_batch=yb)
        ref = _fwd_dev(field32, 8, **kw)
        got = _fwd_dev(field32, 8, devices=devs, **kw)
        assert got.dtype == ref.dtype
        assert np.array_equal(got, ref), f"w={w} pb={pb} yb={yb} moved bits (forward)"

        def _inv(spec, **extra):
            out = np.empty_like(field32)
            for lo, s in ooc_fft.inverse_to_slabs_device(spec, N, slab=8, **kw, **extra):
                out[lo : lo + s.shape[0]] = s
            return out

        assert np.array_equal(_inv(ref.copy(), devices=devs), _inv(ref.copy())), (
            f"w={w} pb={pb} yb={yb} moved bits (inverse)")


def test_a_misaligned_split_really_does_move_bits():
    """A split that strands a size-1 batch moves bits, so `partition_units`' alignment rule
    guards something.

    jax's batch-size dependence is not monotone (CPU, N=64: plane_batch 1, 2, 3 each differ,
    4 and 8 agree), so some misaligned cuts happen not to bite. This one moved 1,062 f32 words.
    """
    n, t, pb = 64, 16, 2
    planes = np.stack([ooc_fft.plane_noise(n, i, np.float32, 3) for i in range(t)])
    whole = ooc_fft.rfft2_planes_device(planes, plane_batch=pb)
    mis = np.empty_like(whole)
    ooc_fft.rfft2_planes_device(planes[:1], plane_batch=pb, out=mis[:1])
    ooc_fft.rfft2_planes_device(planes[1:], plane_batch=pb, out=mis[1:])
    n_diff = int((whole.view(np.float32) != mis.view(np.float32)).sum())
    assert n_diff > 0, (
        "a split that strands a size-1 batch did NOT move bits, so the "
        "alignment rule in partition_units is guarding nothing on this backend")


@pytest.mark.parametrize("w", [1, 2])
def test_transfer_policy_changes_the_route_not_a_bit(field32, w):
    """`staged` (page-locked host buffer) and `pageable` (driver-staged copy) give bitwise
    identical results, forward and inverse. Accelerator-only: CPU jax refuses pinned_host."""
    if not ooc_fft.staging_supported():
        pytest.skip("backend has no host -> pinned_host -> device round trip "
                    "(jax CPU refuses it); this gate runs on the accelerator")
    devs = _devices(w)
    kw = dict(plane_batch=1, pencil_batch=4, devices=devs)
    ref = _fwd_dev(field32, 8, **kw)
    got = _fwd_dev(field32, 8, transfer="staged", **kw)
    assert got.dtype == ref.dtype
    assert np.array_equal(got, ref), "the staged transfer policy moved bits (forward)"

    def _inv(spec, **extra):
        out = np.empty_like(field32)
        for lo, s in ooc_fft.inverse_to_slabs_device(spec, N, slab=8, **kw, **extra):
            out[lo : lo + s.shape[0]] = s
        return out

    assert np.array_equal(_inv(ref.copy(), transfer="staged"), _inv(ref.copy())), (
        "the staged transfer policy moved bits (inverse)")


def test_an_unknown_transfer_policy_is_refused():
    """An unknown transfer policy is refused rather than falling through to the default."""
    with pytest.raises(ValueError, match="transfer must be one of"):
        ooc_fft.rfft2_planes_device(
            np.zeros((2, 8, 8), np.float32), transfer="pinned")
