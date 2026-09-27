"""Shared pytest wiring: the `detflag` marker.

Tests marked `detflag` assert bit equality of float results. On GPU, f32 scatter-add accumulates
in a nondeterministic order, so these need `XLA_FLAGS=--xla_gpu_deterministic_ops=true`. The
flag is process-wide and read at backend init, so marked tests run in a separate process
(`pixi run test-det`); without the flag on GPU they skip with a visible reason rather than being
silently excluded. On CPU they always run (the CPU backend is deterministic).
"""

import os

import pytest


def _needs_detflag_skip():
    """True iff we are on a non-CPU backend without XLA's deterministic ops."""
    import jax

    if jax.devices()[0].platform == "cpu":
        return False  # CPU backend is deterministic; the flag is a no-op
    return "deterministic" not in os.environ.get("XLA_FLAGS", "")


def pytest_runtest_setup(item):
    if "detflag" in item.keywords and _needs_detflag_skip():
        pytest.skip(
            "bit-equality assertion on a GPU backend without XLA deterministic ops "
            "(f32 scatter-add is nondeterministic) -- run `pixi run test-det`"
        )
