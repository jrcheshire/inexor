"""Executor identity: the pool executor is bitwise the serial loop (W2).

The seam analysis says tile writes are disjoint (ownership is a partition) and
every value a worker reads while the parent applies another tile's writes is
discarded -- these tests are what CERTIFY that, at the smoke config, across
the machinery the pool actually adds: shm adoption, the per-step header, the
repack copy-back, worker-side arena decode. The cluster legs repeat the same
identity at cdev8/cdev scale.
"""

import numpy as np
import pytest

from inexor import engine, state
from inexor.codec import T9Layout
from inexor.config import Cosmology
from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

# the smoke rung: smallest geometry whose tile+buffer decomposition is not
# degenerate (same constants as test_engine.py)
L_BOX, N_PART, N_FINE, N_COARSE, N_TILE, B_FINE = 32.0, 32, 64, 16, 16, 8

FIELDS = ("off", "w", "occupancy", "brick_start", "vel_scale", "arena_bucket")


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _cfg(**kw):
    return engine.EngineConfig(
        box_size=L_BOX, n_part=N_PART, n_fine=N_FINE, n_coarse=N_COARSE,
        n_tile=N_TILE, b_fine=B_FINE, **kw
    )


def _positions(seed=0):
    rng = np.random.default_rng(seed)
    g = (np.arange(N_PART) + 0.5) * (L_BOX / N_PART)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    return np.mod(q + rng.normal(scale=0.25 * L_BOX / N_PART, size=q.shape), L_BOX)


def _run(tile_workers, seed=3, k=3, arena_frac=0.05, build_slack=0.10, **cfg_kw):
    cfg = _cfg(tile_workers=tile_workers, **cfg_kw)
    x = _positions(seed)
    v = np.random.default_rng(seed + 1).normal(scale=0.5, size=x.shape)
    t9 = T9Layout(box_size=L_BOX, n_part=N_PART, bucket_cells=2)
    st = state.SlotState.build(
        x, v, t9, N_FINE // cfg.n_brick, brick_slack=build_slack,
        arena_frac=arena_frac, with_ids=True,
    )
    a = a_grid(0.1, 1.0, k, "log")
    co = bullfrog_float_coeffs(bullfrog_table(a, Cosmology()))
    out = engine.run(st, cfg, co)
    return st, out


def _assert_states_identical(s1, s2):
    for f in FIELDS:
        a, b = np.asarray(getattr(s1, f)), np.asarray(getattr(s2, f))
        n = int((a != b).sum())
        assert n == 0, f"{f}: {n} of {a.size} elements differ between executors"
    assert s1.arena_base == s2.arena_base
    assert s1.ids is not None and s2.ids is not None, "ids must be in the comparison"
    assert int((s1.ids != s2.ids).sum()) == 0, "ids diverged between executors"


def test_the_pool_executor_is_bitwise_the_serial_loop():
    """W=2 against serial over a K=3 run with a repack every step, so the
    brick_start/occupancy copy-back is exercised at every step boundary."""
    s1, out1 = _run(1)
    s2, out2 = _run(2)
    _assert_states_identical(s1, s2)
    # same cap ladder: the pooled arm may not key different XLA shapes
    assert [o["cap"] for o in out1] == [o["cap"] for o in out2]
    assert out1[-1]["tile_workers"] == 1 and out2[-1]["tile_workers"] == 2
    assert "pool" in out2[-1] and "pool" not in out1[-1]
    assert out2[-1]["pool"]["workers"] == 2


def test_the_pool_executor_identity_with_a_resident_arena():
    """Zero BUILD slack concentrates overflow into the arena from the first
    migrate (the measured lever -- a larger drift only moves particles BETWEEN
    bricks), and `repack_every` past K keeps residents in place across every
    later step boundary, so the worker-side arena decode and the `arena_base`
    header are load-bearing here rather than idle."""
    kw = dict(build_slack=0.0, arena_frac=0.20, repack_every=4)
    s1, _ = _run(1, seed=5, **kw)
    assert (np.asarray(s1.arena_bucket) >= 0).any(), (
        "the serial arm never populated the arena: this leg is not exercising "
        "worker-side arena decode and needs a different configuration"
    )
    s2, _ = _run(2, seed=5, **kw)
    _assert_states_identical(s1, s2)


def test_the_pool_survives_and_restores_state_ownership():
    """After a pooled run the state must be backed by ordinary memory again
    (the shm segments are unlinked), and still usable."""
    s2, _ = _run(2, seed=7, k=2)
    # a post-run mutation must not touch shm (it is gone); repack exercises
    # the rebound arrays end to end, and check() raises on corruption
    s2.repack(brick_slack=0.10)
    s2.check()
