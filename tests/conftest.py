"""Shared pytest wiring.

The `detflag` marker exists because of M2 S4 (deneb jobs 14/15). A handful of
tests assert BIT equality of float quantities. On CUDA the f32 CIC scatter-add
is nondeterministic -- its atomic accumulation order is not reproducible -- so
the STE twin's VJP, and therefore the gradient, differs run to run. Measured on
deneb's 3050: paint_f32 156/4096 elements differ across 8 identical calls, the
full bwd 18384/24576, while paint_int and the PRIMAL int force are bit-stable
(0/12288). The spread is f32 roundoff (median-rel ~1e-7, corr 1.000000000,
ratio 1.000000000) and threatens no ratified gate -- but it makes a bit-equality
assertion unpassable by default.

`XLA_FLAGS=--xla_gpu_deterministic_ops=true` removes it at every layer, so the
strong assertion is kept and run under that flag (JC, 2026-07-14) rather than
weakened to a tolerance, which would have cost us the canary: the assertion is
what would catch the drivers ACTUALLY diverging, and job 14 confirmed they do
not (cross-driver residual 0/12288).

XLA_FLAGS is process-wide and read at backend init, so it cannot be set per
test -- hence a separate process (`pixi run test-det`). Rather than silently
excluding these from `pixi run test`, they SKIP with a visible reason when the
flag is absent on a GPU: an omission you can see beats one you cannot. On CPU
they always run (the CPU backend is deterministic; the flag is a no-op there).
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
