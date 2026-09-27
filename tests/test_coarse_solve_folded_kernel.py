"""The coarse kernel multiply folded into the inverse FFT's axis-0 pass (`fold_kernel`).

The fold saves a host traversal of the half-grid per component (85 s of a 123 s coarse solve at
4096^3). `ooc_fft.ArrayKernel` keeps the association `(pref * ik) * mf`, so folded and
unfolded arms are compared bitwise.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from inexor import forces, ooc_fft  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _delta(n, seed=5, dtype=np.float32):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n,) * 3).astype(dtype)


def _kw(n, box, dtype=np.float32):
    parts = forces.coarse_kernel_parts(n, box, "long", r_s=1.5,
                                       match=(box / n, box / (2 * n)), fdtype=dtype)
    return parts, dict(box_size=box, which="long", r_s=1.5,
                       match=(box / n, box / (2 * n)), parts=parts, fdtype=dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_folded_kernel_is_bitwise_the_host_multiply_on_host_meshes(dtype):
    n, box = 16, 32.0
    d = _delta(n, dtype=dtype)
    _parts, kw = _kw(n, box, dtype)
    a = forces.coarse_force_meshes(d, n, fold_kernel=False, **kw)
    b = forces.coarse_force_meshes(d, n, fold_kernel=True, **kw)
    for i, (x, y) in enumerate(zip(a, b)):
        n_diff = int(np.count_nonzero(np.asarray(x) != np.asarray(y)))
        assert n_diff == 0, f"component {i}: {n_diff} of {np.asarray(x).size} differ"


def _on_card_shards(d, n, kw, cards, fold):
    from inexor.device.coarse import CardShards

    devs = jax.devices()
    per = n // cards
    shards = CardShards([(k * per, per, devs[k]) for k in range(cards)], n)
    return forces.coarse_force_meshes(d, n, out=shards, fold_kernel=fold, **kw)


@pytest.mark.parametrize("cards", [1, 4])
def test_folded_kernel_is_bitwise_on_card_shards_and_at_four_cards(cards):
    """Bitwise with meshes written onto card shards, at 1 and 4 cards (4 needs emulated
    devices, and only that case skips without them); at 4 cards the folded arm is also
    bitwise the 1-card one, so it is independent of card count."""
    if len(jax.devices()) < cards:
        pytest.skip(f"needs {cards} jax devices")
    n, box = 16, 32.0
    d = _delta(n)
    _parts, kw = _kw(n, box)
    outs = {fold: _on_card_shards(d, n, kw, cards, fold) for fold in (False, True)}
    for i in range(3):
        for k in range(cards):
            x = np.asarray(outs[False][k]["meshes"][i])
            y = np.asarray(outs[True][k]["meshes"][i])
            assert int(np.count_nonzero(x != y)) == 0, (cards, i, k)
    if cards > 1:
        one = np.concatenate(
            [np.asarray(s["meshes"][0]) for s in _on_card_shards(d, n, kw, 1, True)])
        many = np.concatenate([np.asarray(s["meshes"][0]) for s in outs[True]])
        assert int(np.count_nonzero(one != many)) == 0


def test_array_kernel_block_product_is_the_host_kernel_slab():
    """`ArrayKernel` blocks are bitwise `coarse_kernel_slab`, cut on the pencil axis."""
    n, box = 8, 16.0
    parts, _kwargs = _kw(n, box)
    cdtype = np.complex64
    for axis in range(3):
        k = ooc_fft.ArrayKernel.coarse(parts["pref"], parts["iks"][axis], axis,
                                       parts["mf"], cdtype)
        got = np.asarray(k.on_card([jax.numpy.asarray(b) for b in k.blocks(2, 5)]))
        whole = forces.coarse_kernel_slab(parts, axis, 0, n, cdtype)
        assert got.dtype == cdtype
        assert int(np.count_nonzero(got != whole[:, 2:5, :])) == 0, axis


def test_the_unfolded_arm_still_calls_the_shipped_kernel_builder():
    """The unfolded arm (the oracle) still calls `coarse_kernel_slab` on every axis."""
    n, box = 8, 16.0
    d = _delta(n)
    parts, kw = _kw(n, box)
    seen = []
    real = forces.coarse_kernel_slab
    try:
        forces.coarse_kernel_slab = lambda *a, **k: (seen.append(a[1]), real(*a, **k))[1]
        forces.coarse_force_meshes(d, n, fold_kernel=False, **kw)
    finally:
        forces.coarse_kernel_slab = real
    assert sorted(set(seen)) == [0, 1, 2]


def test_the_engine_step_folds_by_default_and_the_arms_agree():
    """Folding is the engine default, and a run lands on the same state with either arm."""
    from tests.test_engine_device_backend import _coeffs, _same
    from tests.test_engine_device_backend import _state as _estate
    from tests.test_engine_device_step import _cfg

    kw = dict(coarse_backend="device", tile_backend="device", migrate_backend="device")
    assert _cfg(**kw).coarse_fold_kernel is True
    co = _coeffs(2)
    cfg_a = _cfg(**kw, coarse_fold_kernel=False)
    cfg_b = _cfg(**kw)
    st_a, st_b = _estate(cfg_a), _estate(cfg_b)
    engine = pytest.importorskip("inexor.engine")
    engine.run(st_a, cfg_a, co)
    engine.run(st_b, cfg_b, co)
    _same(st_a, st_b, "unfolded vs folded coarse kernel")
