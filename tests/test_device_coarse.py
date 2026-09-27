"""A tile's coarse sub-block gathered from a device-resident shard, gated bitwise against
`forces.stage_coarse_subblock`, which reads the global mesh without the shard's bookkeeping."""

import numpy as np
import pytest

pytest.importorskip("jax")

from inexor import forces  # noqa: E402
from inexor.device import coarse as dcoarse  # noqa: E402

N, EXTENT = 16, 6


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _meshes():
    rng = np.random.default_rng(0)
    return [rng.normal(size=(N, N, N)).astype(np.float32) for _ in range(3)]


def _host(g, o):
    return [forces.stage_coarse_subblock(m, o, EXTENT) for m in g]


@pytest.mark.parametrize("jit", [False, True])
def test_whole_mesh_shard_is_bitwise_host_staging(jit):
    import jax

    g = _meshes()
    sh = dcoarse.whole_mesh_shard(g, halo=2)
    fn = dcoarse.subblock_device
    if jit:
        fn = jax.jit(fn, static_argnums=(2, 4))
    origins = [(-2, -2, -2), (0, 5, 13), (N - 4, N - 1, 3), (7, -2, N - 3)]
    for o in origins:
        dcoarse.check_covers(sh, o, EXTENT)
        got = fn(sh["meshes"], sh["x0"], N, np.asarray(o), EXTENT)
        for a, b in zip(got, _host(g, o)):
            assert np.array_equal(np.asarray(a), b), f"origin {o}"


def test_a_partial_shard_is_bitwise_and_refuses_outside_planes():
    g = _meshes()
    sh = dcoarse.shard_coarse_meshes(g, x0=6, nx=9)
    for o in [(6, 0, 0), (9, 14, -1)]:  # x planes 6..11 and 9..14 lie in 6..14
        dcoarse.check_covers(sh, o, EXTENT)
        got = dcoarse.subblock_device(sh["meshes"], sh["x0"], N, np.asarray(o), EXTENT)
        for a, b in zip(got, _host(g, o)):
            assert np.array_equal(np.asarray(a), b), f"origin {o}"
    for o in [(5, 0, 0), (10, 0, 0)]:
        with pytest.raises(ValueError, match="not inside the shard"):
            dcoarse.check_covers(sh, o, EXTENT)


def test_a_mislabelled_shard_fails_the_comparison():
    """Control: the same planes labelled as starting one plane later read the wrong block."""
    g = _meshes()
    sh = dcoarse.shard_coarse_meshes(g, x0=6, nx=9)
    o = (8, 3, 3)
    wrong = dcoarse.subblock_device(sh["meshes"], 7, N, np.asarray(o), EXTENT)
    assert not np.array_equal(np.asarray(wrong[0]), _host(g, o)[0])
