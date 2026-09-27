"""`ooc_fft.inverse_to_card_shards`: the coarse solve's inverse written onto the cards,
gated bitwise against the host-slab path followed by `device.coarse.shard_coarse_meshes`.

Card counts above the backend's device count replicate device handles, as in
`test_ooc_fft_cards`; with `--xla_force_host_platform_device_count=4` the same
tests are a true multi-device gate.
"""

import tracemalloc

import numpy as np
import pytest

pytest.importorskip("jax")

from inexor import forces, ooc_fft  # noqa: E402
from inexor.device import coarse as dcoarse  # noqa: E402

N = 32
HALO = forces.COARSE_HALO


@pytest.fixture(scope="module")
def field32():
    return np.random.default_rng(21).standard_normal((N, N, N), dtype=np.float32)


def _devices(w):
    import jax

    devs = jax.devices()
    return [devs[i % len(devs)] for i in range(w)]


def _spec(field, n=N):
    return ooc_fft.forward_from_slabs_device(lambda lo, hi: field[lo:hi], n, slab=8)


def _host_mesh(field, n=N):
    """The reference: the host-slab inverse assembled into one host mesh."""
    spec = _spec(field, n)
    out = None
    for lo, blk in ooc_fft.inverse_to_slabs_device(spec, n, slab=8):
        if out is None:
            out = np.empty((n, n, n), dtype=blk.dtype)
        out[lo:lo + blk.shape[0]] = blk
    return out


def _card_ranges(w, n=N):
    """One card per tile-plane group, with a halo on each side."""
    return [(lo - HALO, hi - lo + 2 * HALO, d)
            for (lo, hi), d in zip(ooc_fft.partition_units(n, w, 1), _devices(w))]


def _same_as_host(got, ranges, mesh):
    for (x0, nx, d), g in zip(ranges, got):
        ref = dcoarse.shard_coarse_meshes([mesh], x0, nx)["meshes"][0]
        assert g.shape == (nx, mesh.shape[0], mesh.shape[0])
        assert np.dtype(g.dtype) == mesh.dtype
        assert np.array_equal(np.asarray(g), np.asarray(ref)), f"card at x0={x0} moved bits"
        assert g.devices() == {d}, "the card's planes left its device"


@pytest.mark.parametrize("w", [1, 2, 4])
def test_the_card_inverse_is_bitwise_the_host_slab_path_sharded(field32, w):
    mesh = _host_mesh(field32)
    ranges = _card_ranges(w)
    t = {}
    got = ooc_fft.inverse_to_card_shards(_spec(field32), N, ranges, timings=t)
    _same_as_host(got, ranges, mesh)
    assert t["pass1_s"] > 0 and t["pass2_s"] > 0


def test_wrapped_and_overlapping_shards_are_bitwise_too(field32):
    mesh = _host_mesh(field32)
    d = _devices(3)
    ranges = [(-HALO, N + 2 * HALO, d[0]),   # the whole mesh, wrapped both ends
              (N - 3, 7, d[1]),               # straddles x = 0
              (5, 1, d[2])]                   # a single plane
    got = ooc_fft.inverse_to_card_shards(_spec(field32), N, ranges)
    _same_as_host(got, ranges, mesh)


def test_f64_card_inverse_is_bitwise_too():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        f = np.random.default_rng(22).standard_normal((N, N, N))
        mesh = _host_mesh(f)
        assert mesh.dtype == np.float64
        ranges = _card_ranges(2)
        _same_as_host(ooc_fft.inverse_to_card_shards(_spec(f), N, ranges), ranges, mesh)
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_the_comparison_can_fail(field32):
    """A spectrum moved at one element must move a card's planes."""
    mesh = _host_mesh(field32)
    ranges = _card_ranges(2)
    spec = _spec(field32)
    spec[N - 1, 3, 4] *= 1.0 + 1e-5
    got = ooc_fft.inverse_to_card_shards(spec, N, ranges)
    with pytest.raises(AssertionError, match="moved bits"):
        _same_as_host(got, ranges, mesh)


def test_refusals(field32):
    with pytest.raises(ValueError, match="plane_batch"):
        ooc_fft.inverse_to_card_shards(_spec(field32), N, _card_ranges(1), plane_batch=2)
    with pytest.raises(ValueError, match="holds 0 planes"):
        ooc_fft.inverse_to_card_shards(_spec(field32), N, [(0, 0, None)])
    with pytest.raises(ValueError, match="empty"):
        ooc_fft.inverse_to_card_shards(_spec(field32), N, [])


def test_no_host_mesh_is_assembled_and_the_bar_can_fail():
    """Host allocation during the card inverse stays under an eighth of one mesh (host work
    is per plane, O(n^2)); the host-slab path assembling its mesh must exceed the same bar.

    Both paths are warmed first: a first call's compile allocates ~350 KB of jax objects at
    n=64 (mostly retained by the program cache), which tracemalloc would count."""
    n = 64
    f = np.random.default_rng(23).standard_normal((n, n, n), dtype=np.float32)
    bar = n**3 * f.itemsize / 8
    ranges = [(-HALO, n + 2 * HALO, _devices(1)[0])]
    ooc_fft.inverse_to_card_shards(_spec(f, n), n, ranges)
    for _ in ooc_fft.inverse_to_slabs_device(_spec(f, n), n, slab=8):
        pass

    spec = _spec(f, n)
    tracemalloc.start()
    ooc_fft.inverse_to_card_shards(spec, n, ranges)
    _, peak_cards = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak_cards < bar, f"card inverse held {peak_cards} host bytes, bar {bar:.0f}"

    spec = _spec(f, n)
    tracemalloc.start()
    mesh = np.empty((n, n, n), dtype=np.float32)
    for lo, blk in ooc_fft.inverse_to_slabs_device(spec, n, slab=8):
        mesh[lo:lo + blk.shape[0]] = blk
    _, peak_host = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak_host >= bar, "vacuous: the bar does not catch a host mesh"


# ------------------------------------------------------------------ through the solve


def _solve_inputs(dtype):
    rng = np.random.default_rng(24)
    delta = rng.normal(scale=0.3, size=(N, N, N)).astype(dtype)
    box = 32.0
    kw = dict(r_s=1.25 * box / N, match=(box / N, box / (2 * N)), fdtype=np.dtype(dtype))
    return delta, box, kw


def test_the_solve_into_card_shards_is_bitwise_the_host_solve_sharded():
    import jax.numpy as jnp

    delta, box, kw = _solve_inputs(np.float32)
    want = forces.coarse_force_meshes(jnp.asarray(delta), N, box, "long", **kw)
    sink = dcoarse.CardShards.whole_mesh(N)
    got = forces.coarse_force_meshes(jnp.asarray(delta), N, box, "long", out=sink, **kw)
    ref = dcoarse.whole_mesh_shard(want)
    assert len(got) == 1 and (got[0]["x0"], got[0]["nx"], got[0]["n"]) == (
        ref["x0"], ref["nx"], ref["n"])
    for axis in range(3):
        assert np.array_equal(np.asarray(got[0]["meshes"][axis]),
                              np.asarray(ref["meshes"][axis])), f"axis {axis} moved bits"


def test_a_monolithic_solve_into_card_shards_is_refused():
    import jax.numpy as jnp

    delta, box, kw = _solve_inputs(np.float32)
    with pytest.raises(ValueError, match="factorized"):
        forces.coarse_force_meshes(jnp.asarray(delta), N, box, "long",
                                   out=dcoarse.CardShards.whole_mesh(N),
                                   transform="monolithic", **kw)
