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


# ------------------------------------------------------- the device path (D1)
#
# Same factorization on an accelerator, so the same discipline: BITWISE for
# anything that must be invariant WITHIN a backend, tolerance only across two.
# Cross-backend bitwise is deliberately NOT asserted -- on this laptop's CPU
# jax the device path happens to reproduce scipy exactly, and a test that
# pinned that would fail on the GPU this code exists for, for a legitimate
# reason. `MAX_DEVICE_TRANSFORM_ELEMENTS` is the one gate that is about
# correctness rather than reproducibility.

pytest.importorskip("jax")


def _fwd_dev(field, slab, **kw):
    return ooc_fft.forward_from_slabs_device(
        lambda lo, hi: field[lo:hi], N, slab=slab, **kw)


def test_device_forward_is_bitwise_invariant_to_slab_thickness(field32):
    """Streaming stays an OUTER LOOP BOUND on device too.

    This is the property the whole design rests on: the spectrum of a 2048^3
    field must not depend on how many planes were resident at a time, or a
    checkpoint/resume or a different window size silently changes the physics.
    """
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
    """Anti-vacuity for the two above.

    FOUR ulps, not one, and the number is MEASURED rather than picked. At f32 a
    one-ulp change to a single element of a 32^3 field is genuinely invisible
    through the transform -- n_diff 0 of 17,408 spectral elements -- because
    2.98e-08 against a spectrum of order 5 is below f32 eps. The ladder reads
    1 -> 0, 4 -> 314, 64 -> 4,230, 1024 -> 17,344. So four is the smallest
    perturbation that this comparison can actually see, which is the strongest
    honest form of the arm; a bigger bump would prove less.

    Two traps sit in writing it. `np.nextafter(x_f32, np.inf)` promotes to
    float64 (np.inf is a Python float) and storing it back into an f32 array
    rounds it to the value it started from, so the destination dtype has to be
    in the call -- the f64 twin higher in this file does not have that problem,
    which is why it reads `np.inf`. And the assert below that the perturbation
    changed the input is not decoration: without it, both forms of no-op above
    would have made the two invariance gates vacuous instead of failing here.
    """
    ref = _fwd_dev(field32, N)
    bumped = field32.copy()
    for _ in range(4):
        bumped[13, 0, 0] = np.nextafter(bumped[13, 0, 0], np.float32(np.inf))
    assert bumped[13, 0, 0] != field32[13, 0, 0], "the perturbation was a no-op"
    assert not np.array_equal(_fwd_dev(bumped, N), ref)


def test_device_matches_the_host_path_at_tolerance(field32):
    """Across backends, tolerance -- and the tolerance is checked against the

    spectrum's own scale, not an absolute number, so it cannot pass by the
    array being small.
    """
    host = ooc_fft.rfftn_ooc(field32.copy())
    dev = _fwd_dev(field32, 8)
    scale = np.sqrt(np.mean(np.abs(host) ** 2))
    assert scale > 1e-3, "vacuous: the reference spectrum is ~zero"
    assert np.max(np.abs(host - dev)) / scale < 1e-5


def test_device_preserves_single_precision(field32):
    """The dtype ledger on the seam where a silent precision change would hide."""
    spec = _fwd_dev(field32, 8)
    assert spec.dtype == np.complex64
    out = np.empty_like(field32)
    for lo, s in ooc_fft.inverse_to_slabs_device(spec, N, slab=8):
        out[lo : lo + s.shape[0]] = s
    assert out.dtype == np.float32


def test_f64_on_device_refuses_rather_than_narrowing(field64):
    """With x64 off jax would transform an f64 field at f32 and hand it back in

    an f64 container: a wrong answer wearing the right dtype. Same contract as
    `eject_jax.require_x64`, and the suite runs x64-off by default.
    """
    import jax

    if jax.config.read("jax_enable_x64"):
        pytest.skip("x64 is on in this session; the narrowing cannot occur")
    with pytest.raises(RuntimeError, match="jax_enable_x64"):
        _fwd_dev(field64, 8)


def test_a_transform_at_the_silent_wrong_bound_is_refused():
    """The measured failure this whole factorization exists to make unreachable.

    1536^3 f32 (3.6e9 elements) came back SILENTLY WRONG on jax 0.10.2 + GB200
    -- roundtrip 3.8e+3 against 1024^3's 2.9e-6, same peak ratio, plausible
    wall. Refuse the regime rather than report a receipt from inside it.
    """
    with pytest.raises(ValueError, match="silently"):
        ooc_fft.refuse_oversize_device_transform(2**31, "test")
    with pytest.raises(ValueError, match="silently"):
        ooc_fft.refuse_oversize_device_transform(1536**3, "the measured case")
    # anti-vacuity: the bound is a bound, not a blanket refusal. One 2048^2
    # plane -- the unit the coarse solve actually runs -- is three orders under.
    ooc_fft.refuse_oversize_device_transform(2**31 - 1, "just under")
    ooc_fft.refuse_oversize_device_transform(2048 * 2048, "one coarse plane")


def test_the_batch_knobs_cannot_reach_the_bound_unnoticed(field32):
    """A batch big enough to hit the silent regime is refused, not attempted."""
    with pytest.raises(ValueError, match="silently"):
        ooc_fft.rfft2_planes_device(field32, plane_batch=2**31 // (N * N) + 1)


def test_the_receipt_is_plane_keyed_so_the_field_is_slab_independent():
    """If the receipt's field depended on the slab, the invariance gates above

    would be comparing two different fields and would pass by construction.
    """
    a = ooc_fft.plane_noise(N, 5, np.float32, seed=3)
    b = ooc_fft.plane_noise(N, 5, np.float32, seed=3)
    c = ooc_fft.plane_noise(N, 6, np.float32, seed=3)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_the_roundtrip_receipt_reads_the_f32_floor_on_both_paths():
    """The instrument the 2048^3 measurement will be read off, exercised small.

    Reported as a number, not asserted into a verdict: what counts as a pass at
    a given n is a gate's decision. Here it must simply be at the f32 floor and
    nowhere near the 3.8e+3 the wrong regime produced.
    """
    dev = ooc_fft.roundtrip_residual(N, np.float32, slab=8, device=True)
    host = ooc_fft.roundtrip_residual(N, np.float32, slab=8, device=False)
    assert dev["residual"] < 1e-4, dev
    assert host["residual"] < 1e-4, host
    assert dev["rms"] > 0.5, "vacuous: the receipt transformed a ~zero field"
    for k in ("n_mesh", "dtype", "slab", "plane_batch", "pencil_batch"):
        assert k in dev, f"the receipt must carry {k} beside the number"


# ---------------------------------------------------------------------------
# splitting a pass across devices (owed off D1: every D1 number is one GB200)
# ---------------------------------------------------------------------------


def _devices(w):
    """`w` device handles, replicating if the backend has fewer.

    Replication is not a weaker gate for the BITWISE properties below: the
    partition, the threads and the write-through views are all exercised either
    way, and the arithmetic cannot depend on which handle it ran under. Run the
    suite with `--xla_force_host_platform_device_count=4` (or on a four-GPU
    node) and the same tests become a true multi-device gate.
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
    """A part with no work is a device that silently did not participate, which
    reads downstream as 'it did not scale'. Refuse instead."""
    with pytest.raises(ValueError, match="only 1 whole units"):
        ooc_fft.partition_units(32, 4, 64)
    with pytest.raises(ValueError, match="n_parts"):
        ooc_fft.partition_units(32, 0, 1)


@pytest.mark.parametrize("w", [2, 4])
def test_device_split_is_bitwise_identical_to_one_device(field32, w):
    """W devices == 1 device, to the bit, forward AND inverse.

    This is the identity the four-GPU reading is measured against: the
    factorization has no inter-device communication, so a split is a partition
    of a loop and cannot touch a value. Anything else and a wall measured at
    W=4 is a wall for a different transform.
    """
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
    """Anti-vacuity for the alignment rule, MEASURED rather than argued.

    jax's batch-size dependence is real but not monotone: on this backend at
    N=64, plane_batch 1, 2 and 3 each give a distinct spectrum while 4 and 8
    agree (n_diff 538 / 722 / 520 against batch 1, 0 between 4 and 8). So a
    misaligned cut sometimes happens not to bite -- splitting a 16-plane batch
    of 8 at plane 4 gives sizes (4, 8, 4) and reads n_diff 0 purely because
    4 and 8 agree here. That coincidence is exactly why the rule cannot be
    'align when it seems to matter': the case below strands a size-1 batch and
    moves 1,062 f32 words.
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
