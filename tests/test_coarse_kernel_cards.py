"""The coarse kernel's real half-grids built per y-block and kept on the cards.

A block is bitwise the corresponding rows of the full build (so each card, or each rank, can
build only its own rows), and the folded solve with card-resident blocks is bitwise the solve
with host arrays, at 1, 2 and 4 cards.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from inexor import forces, ooc_fft  # noqa: E402
from inexor.device.coarse import CardShards  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _former_parts(n_mesh, box_size, which, r_s, match, fdtype):
    """The full build as `coarse_kernel_parts` formed it before per-block builds: the
    reference."""
    fdtype = forces.field_dtype(fdtype)
    shape = (int(n_mesh),) * 3
    ikx, iky, ikz, k2_true, k2_safe = forces.kernel_grids(shape, box_size / n_mesh, np.float64)
    fac = forces.split_factor(k2_true, 0.0 if r_s is None else r_s, which)
    pref = (fac / k2_safe).astype(fdtype, copy=False)
    mf = None
    if match is not None:
        m, _ = forces.cic_match_factor(shape, match[0], match[1],
                                       **forces._match_orders(match))
        mf = m.astype(fdtype, copy=False)
    return (ikx, iky, ikz), pref, mf


def _bits(a):
    """The raw bytes, so -0.0 and NaN payloads count as differences."""
    return np.ascontiguousarray(a).view(np.uint8)


def _ranges(n):
    out = {(0, n), (0, 1), (n - 1, n), (n // 3, n // 3 + 5)}
    for k in (2, 3, 4):
        out.update(ooc_fft.partition_units(n, k, 1))
    return sorted(out)


@pytest.mark.parametrize("n", [16, 30, 64, 256])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("match", [False, True])
@pytest.mark.parametrize("which", ["long", "short"])
def test_a_block_is_bitwise_the_rows_of_the_full_build(n, dtype, match, which):
    box = 2.0 * n
    m = (box / n, box / (2 * n), 3, 3) if match else None
    iks, pref, mf = _former_parts(n, box, which, 1.5, m, dtype)
    full = forces.coarse_kernel_parts(n, box, which, r_s=1.5, match=m, fdtype=dtype)
    for a, b in zip(full["iks"], iks):
        assert a.dtype == b.dtype and np.array_equal(_bits(a), _bits(b))
    assert np.array_equal(_bits(full["pref"]), _bits(pref))
    assert (full["mf"] is None) == (mf is None)
    for lo, hi in _ranges(n):
        bp, bm = forces.coarse_kernel_block(n, box, which, r_s=1.5, match=m, fdtype=dtype,
                                            y=(lo, hi))
        assert bp.dtype == np.dtype(dtype)
        assert np.array_equal(_bits(bp), _bits(pref[:, lo:hi])), (lo, hi)
        if mf is not None:
            assert np.array_equal(_bits(bm), _bits(mf[:, lo:hi])), (lo, hi)


def _solve(d, n, box, cards, on_cards, dtype=np.float32):
    devs = jax.devices()[:cards]
    per = n // cards
    shards = CardShards([(k * per - 2, per + 4, devs[k]) for k in range(cards)], n)
    m = (box / n, box / (2 * n), 3, 3)
    placed = ([(lo, hi, dev) for (lo, hi), dev in zip(ooc_fft.partition_units(n, cards, 1),
                                                        devs)] if on_cards else None)
    parts = forces.coarse_kernel_parts(n, box, "long", r_s=1.5, match=m, fdtype=dtype,
                                       cards=placed)
    out = forces.coarse_force_meshes(d, n, box, "long", r_s=1.5, match=m, fdtype=dtype,
                                     parts=parts, out=shards, fold_kernel=True)
    return parts, [[np.asarray(x) for x in s["meshes"]] for s in out]


@pytest.mark.parametrize("cards", [1, 2, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_card_resident_kernel_is_bitwise_the_host_kernel(cards, dtype):
    if len(jax.devices()) < cards:
        pytest.skip(f"needs {cards} jax devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    n, box = 32, 64.0
    d = np.random.default_rng(5).normal(size=(n,) * 3).astype(dtype)
    # each arm compiles its own program: the cache is keyed by structure, and a card term keys
    # as the host term it replaces, so a shared program would hide the card arm's own trace
    ooc_fft._CARD_PROGRAMS.clear()
    card_parts, card = _solve(d, n, box, cards, True, dtype)
    ooc_fft._CARD_PROGRAMS.clear()
    host_parts, host = _solve(d, n, box, cards, False, dtype)
    assert host_parts["cards"] is None and card_parts["pref"] is None
    assert len(card_parts["cards"]) == cards
    for e, dev in zip(card_parts["cards"], jax.devices()[:cards]):
        assert e["pref"].devices() == {dev}
    for k in range(cards):
        for i in range(3):
            x, y = host[k][i], card[k][i]
            assert int(np.count_nonzero(_bits(x) != _bits(y))) == 0, (k, i)


def test_a_mismatched_placement_is_refused():
    n, box = 16, 32.0
    dev = jax.devices()[0]
    d = np.zeros((n,) * 3, np.float32)
    # resident on one card but only half the rows: the pass's second half has no block
    parts = forces.coarse_kernel_parts(n, box, "long", r_s=1.5, fdtype=np.float32,
                                       cards=[(0, n // 2, dev)])
    shards = CardShards([(-2, n + 4, dev)], n)
    with pytest.raises(ValueError, match="inside no resident kernel range"):
        forces.coarse_force_meshes(d, n, box, "long", r_s=1.5, fdtype=np.float32,
                                   parts=parts, out=shards, fold_kernel=True)
    with pytest.raises(ValueError, match="need fold_kernel=True"):
        forces.coarse_force_meshes(d, n, box, "long", r_s=1.5, fdtype=np.float32,
                                   parts=parts, out=shards, fold_kernel=False)
    with pytest.raises(ValueError, match="only the factorized solve"):
        forces.coarse_force_meshes(d, n, box, "long", r_s=1.5, fdtype=np.float32,
                                   parts=parts, transform="monolithic")
