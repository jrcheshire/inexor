"""k-space kernels folded into the device axis-0 pass (`ooc_fft.kspace_pass_device`).

Oracles are the host kernels (`grad_invk2_spec`, `deriv2_spec`, `mul_radial_inplace`)
and the existing device axis-0 pass. Under x64 every single kernel is bitwise its host
twin; a kernel product is bitwise ONE host pass with the product function. Card counts
above the backend's device count replicate handles; run with
`--xla_force_host_platform_device_count=4` for a true multi-device gate.
"""

import numpy as np
import pytest

pytest.importorskip("jax")

from inexor import ic, ooc_fft  # noqa: E402
from inexor.config import PLANCK  # noqa: E402
from inexor.cosmology import ic_k_table  # noqa: E402

N, L = 32, 16.0
K = ooc_fft.KSpaceKernel


@pytest.fixture
def x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def table():
    return ic_k_table(PLANCK, N, L)


def _spec(seed, dt=np.float32):
    f = np.random.default_rng(seed).standard_normal((N, N, N)).astype(dt)
    return ooc_fft.forward_from_slabs(lambda lo, hi: f[lo:hi], N, slab=8)


def _devices(w):
    import jax

    devs = jax.devices()
    return [devs[i % len(devs)] for i in range(w)]


def _radial_host(s, fn, dc):
    return ooc_fft.mul_radial_inplace(s.copy(), N, L, fn, dc)


def _host_bitwise_measured_here():
    """Skip off the CPU backend: kernel == numpy bitwise holds on CPU XLA; on a GPU the
    kernel parity is reported by `scripts/run/device_ics.py smoke`, not asserted."""
    import jax

    if jax.devices()[0].platform != "cpu":
        pytest.skip("kernel == numpy bitwise is a CPU-backend measurement; on a GPU, "
                    "scripts/run/device_ics.py smoke reports it")


@pytest.mark.parametrize("inverse", [False, True])
def test_the_identity_kernel_is_bitwise_the_device_axis0_pass(inverse):
    s = _spec(1)
    ref = ooc_fft.fft_axis0_device_inplace(s.copy(), inverse=inverse)
    got = ooc_fft.kspace_pass_device([(1.0, s)], N, L, inverse=inverse)
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("which", ["grad0", "grad1", "grad2", "d00", "d01", "d02", "d11",
                                   "d12", "d22"])
def test_derivative_kernels_are_bitwise_the_host_under_x64(x64, which):
    _host_bitwise_measured_here()
    s = _spec(2)
    if which.startswith("grad"):
        ax = int(which[-1])
        kern, ref = K.grad_invk2(ax), ooc_fft.grad_invk2_spec(s, ax, N, L)
    else:
        i, j = int(which[1]), int(which[2])
        kern, ref = K.deriv2(i, j), ooc_fft.deriv2_spec(s, i, j, N, L)
    got = ooc_fft.kspace_pass_device([(1.0, s)], N, L, kernel=kern, transform=False)
    assert np.array_equal(got, ref)


def test_radial_kernels_are_bitwise_the_host_under_x64(x64, table):
    _host_bitwise_measured_here()
    s = _spec(3)
    for kern, fn, dc in [
        (K.colour(table, N, L), ic._colour_fn(table, N, L), 0.0),
        (K.poisson(PLANCK, table, N, L), ic._poisson_fn(PLANCK, table), 1.0),
        (K.poisson(PLANCK, table, N, L, inverse=True),
         ic._poisson_fn(PLANCK, table, inverse=True), 1.0),
    ]:
        got = ooc_fft.kspace_pass_device([(1.0, s)], N, L, kernel=kern, transform=False)
        assert np.array_equal(got, _radial_host(s, fn, dc))


def test_a_kernel_product_is_bitwise_one_host_pass_with_the_product(x64, table):
    _host_bitwise_measured_here()
    s = _spec(4)
    col, ipo = ic._colour_fn(table, N, L), ic._poisson_fn(PLANCK, table, inverse=True)
    kern = K.colour(table, N, L) * K.poisson(PLANCK, table, N, L, inverse=True)
    got = ooc_fft.kspace_pass_device([(1.0, s)], N, L, kernel=kern, transform=False)
    ref = _radial_host(s, lambda kk: col(kk) * ipo(kk), 0.0 * 1.0)
    assert np.array_equal(got, ref)
    # and it is not the two-pass chain, which is why the oracle is one pass with the product
    chain = _radial_host(_radial_host(s, col, 0.0), ipo, 1.0)
    assert not np.array_equal(got, chain)


def test_the_kernel_lands_on_the_right_side_of_the_transform(x64):
    _host_bitwise_measured_here()
    s = _spec(5)
    kern = K.deriv2(0, 2)
    fwd = ooc_fft.kspace_pass_device([(1.0, s)], N, L, kernel=kern, inverse=False)
    assert np.array_equal(fwd, ooc_fft.deriv2_spec(ooc_fft.fft_axis0_device_inplace(s.copy()),
                                                   0, 2, N, L))
    inv = ooc_fft.kspace_pass_device([(1.0, s)], N, L, kernel=kern, inverse=True)
    assert np.array_equal(inv, ooc_fft.fft_axis0_device_inplace(
        ooc_fft.deriv2_spec(s, 0, 2, N, L), inverse=True))


def test_combination_in_place_and_card_counts_are_bitwise(table):
    a, b = _spec(6), _spec(7)
    ref = np.float32(0.7) * a + np.float32(-1.3) * b
    assert np.array_equal(
        ooc_fft.kspace_pass_device([(0.7, a), (-1.3, b)], N, L, transform=False), ref)

    kern = K.grad_invk2(1)
    out_of_place = ooc_fft.kspace_pass_device([(0.7, a), (-1.3, b)], N, L, kernel=kern,
                                              inverse=True)
    buf = a.copy()
    ooc_fft.kspace_pass_device([(0.7, buf), (-1.3, b)], N, L, kernel=kern, inverse=True, out=buf)
    assert np.array_equal(buf, out_of_place)

    radial = K.colour(table, N, L)
    one = ooc_fft.kspace_pass_device([(1.0, a)], N, L, kernel=radial, devices=_devices(1))
    for w, pb in [(2, 1), (4, 1), (4, 2)]:
        ref_pb = (one if pb == 1 else ooc_fft.kspace_pass_device(
            [(1.0, a)], N, L, kernel=radial, devices=_devices(1), pencil_batch=pb))
        got = ooc_fft.kspace_pass_device([(1.0, a)], N, L, kernel=radial, devices=_devices(w),
                                         pencil_batch=pb)
        assert np.array_equal(got, ref_pb), f"w={w} pencil_batch={pb} moved bits"


def _on_cpu():
    import jax

    return jax.devices()[0].platform == "cpu"


def _host_noise_spec(key, n=N):
    """The CPU stream through the existing device forward: the card path's oracle."""
    white = ic.white_slab(key, 0, n, n, np.float32)
    return ooc_fft.forward_from_slabs_device(lambda lo, hi: white[lo:hi], n, slab=8)


def test_card_noise_is_the_cpu_stream_on_the_cpu_backend_at_every_card_count():
    import jax

    if not _on_cpu():
        pytest.skip("the card stream is a different stream off the CPU backend (IC_STREAM_DEVICE)")
    key = jax.random.PRNGKey(5)
    ref = _host_noise_spec(key)
    for w in (1, 2, 4):
        got = ooc_fft.noise_forward_cards(key, N, _devices(w))
        assert np.array_equal(got, ref), f"w={w} moved bits"


def test_card_noise_is_invariant_to_card_count_on_any_backend(table):
    import jax

    key = jax.random.PRNGKey(6)
    kern = K.colour(table, N, L)
    one = ooc_fft.noise_forward_cards(key, N, _devices(1), kernel=kern, box_size=L)
    four = ooc_fft.noise_forward_cards(key, N, _devices(4), kernel=kern, box_size=L)
    assert np.array_equal(one, four)


def _host_accumulate(spec, kern, weight, acc, dt):
    """Oracle: the device inverse to host slabs, then numpy's weighted square."""
    s = ooc_fft.kspace_pass_device([(1.0, spec)], N, L, kernel=kern, transform=False)
    for lo, blk in ooc_fft.inverse_to_slabs_device(s, N, slab=8):
        acc[lo:lo + blk.shape[0]] += np.asarray(weight, dt) * np.square(blk.astype(dt))
    return acc


@pytest.mark.parametrize("w", [1, 4])
def test_card_accumulate_is_bitwise_the_host_square_and_leaves_sources_intact(x64, w):
    spec = _spec(9)
    keep = spec.copy()
    shards = ooc_fft.zeros_card_shards(N, _devices(w), np.float32)
    work = None
    ref = np.zeros((N, N, N), np.float32)
    for kern, wt in [(None, 0.5), (K.deriv2(0, 0), -0.5), (K.deriv2(1, 2), -1.0)]:
        shards, work = ooc_fft.inverse_accumulate_cards([(1.0, spec)], N, shards, wt,
                                                        kernel=kern, box_size=L, work=work)
        ref = _host_accumulate(spec, kern, wt, ref, np.float32)
    assert np.array_equal(spec, keep), "a source was mutated"
    got = np.concatenate([np.asarray(s["delta"]) for s in shards])
    assert np.array_equal(got, ref)
    for s in shards:
        assert s["delta"].devices() == {s["device"]}


def test_card_accumulate_feeds_the_card_forward():
    shards = ooc_fft.zeros_card_shards(N, _devices(2), np.float32)
    spec = _spec(10)
    shards, _ = ooc_fft.inverse_accumulate_cards([(1.0, spec)], N, shards, 1.0, box_size=L)
    host = np.concatenate([np.asarray(s["delta"]) for s in shards])
    got = ooc_fft.forward_from_card_planes(shards, N)
    ref = ooc_fft.forward_from_slabs_device(lambda lo, hi: host[lo:hi], N, slab=8)
    assert np.array_equal(got, ref)


def test_card_accumulate_refusals():
    spec = _spec(11)
    shards = ooc_fft.zeros_card_shards(N, _devices(2), np.float32)
    with pytest.raises(ValueError, match="aliases"):
        ooc_fft.inverse_accumulate_cards([(1.0, spec)], N, shards, 1.0, work=spec)
    with pytest.raises(ValueError, match="end at"):
        ooc_fft.inverse_accumulate_cards([(1.0, spec)], N, shards[:1], 1.0)
    with pytest.raises(ValueError, match="tile"):
        ooc_fft.inverse_accumulate_cards([(1.0, spec)], N, shards[1:], 1.0)


def test_refusals(table):
    s = _spec(8)
    with pytest.raises(ValueError, match="empty"):
        ooc_fft.kspace_pass_device([], N, L)
    with pytest.raises(ValueError, match="disagree"):
        ooc_fft.kspace_pass_device([(1.0, s), (1.0, s.astype(np.complex128))], N, L)
    with pytest.raises(TypeError, match="complex"):
        ooc_fft.kspace_pass_device([(1.0, s.real.copy())], N, L)
    with pytest.raises(ValueError, match="outside the table"):
        K.colour(table, 1024, 0.01)
