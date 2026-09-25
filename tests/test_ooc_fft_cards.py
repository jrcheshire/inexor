"""`ooc_fft.forward_from_card_planes`: the forward transform of a field already
resident on the cards, gated BITWISE against the host-slab device path.

Card counts above the backend's device count replicate device handles, as in
`test_ooc_fft._devices`; with `--xla_force_host_platform_device_count=4` the
same tests are a true multi-device gate.
"""

import numpy as np
import pytest

pytest.importorskip("jax")

from inexor import ooc_fft  # noqa: E402

N = 32


@pytest.fixture(scope="module")
def field32():
    return np.random.default_rng(11).standard_normal((N, N, N), dtype=np.float32)


def _devices(w):
    import jax

    devs = jax.devices()
    return [devs[i % len(devs)] for i in range(w)]


def _shards(field, w, unit):
    import jax

    return [dict(lo=lo, hi=hi, device=d, delta=jax.device_put(field[lo:hi], d))
            for (lo, hi), d in zip(ooc_fft.partition_units(N, w, unit), _devices(w))]


def _host_path(field, pb, yb):
    return ooc_fft.forward_from_slabs_device(lambda lo, hi: field[lo:hi], N, slab=8,
                                             plane_batch=pb, pencil_batch=yb)


@pytest.mark.parametrize("pb,yb", [(1, 1), (2, 4)])
@pytest.mark.parametrize("w", [1, 2, 4])
def test_forward_from_the_cards_is_bitwise_the_host_slab_path(field32, w, pb, yb):
    want = _host_path(field32, pb, yb)
    t = {}
    got = ooc_fft.forward_from_card_planes(_shards(field32, w, pb), N, plane_batch=pb,
                                           pencil_batch=yb, timings=t)
    assert got.dtype == want.dtype
    assert np.array_equal(got, want), f"w={w} pb={pb} yb={yb} moved bits"
    assert t["pass1_s"] > 0 and t["pass2_s"] > 0


def test_f64_forward_from_the_cards_is_bitwise_too():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        f = np.random.default_rng(12).standard_normal((N, N, N))
        want = _host_path(f, 1, 1)
        got = ooc_fft.forward_from_card_planes(_shards(f, 2, 1), N)
        assert got.dtype == np.complex128
        assert np.array_equal(got, want)
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_the_comparison_can_fail(field32):
    s = _shards(field32, 2, 1)
    s[1]["delta"] = s[1]["delta"].at[3, 5, 7].add(1.0)
    assert not np.array_equal(ooc_fft.forward_from_card_planes(s, N),
                              _host_path(field32, 1, 1))


def test_shards_that_do_not_tile_are_refused(field32):
    s = _shards(field32, 4, 1)
    with pytest.raises(ValueError, match="do not tile"):
        ooc_fft.forward_from_card_planes(s[:2] + s[3:], N)
    with pytest.raises(ValueError, match="do not tile"):
        ooc_fft.forward_from_card_planes(s[:3], N)


def test_a_card_boundary_off_the_plane_batch_is_refused(field32):
    with pytest.raises(ValueError, match="plane_batch"):
        ooc_fft.forward_from_card_planes(_shards(field32, 2, 1), N, plane_batch=3)


def test_a_shard_of_the_wrong_shape_is_refused(field32):
    s = _shards(field32, 2, 1)
    s[0]["delta"] = s[0]["delta"][:, :, :-1]
    with pytest.raises(ValueError, match="shape"):
        ooc_fft.forward_from_card_planes(s, N)
