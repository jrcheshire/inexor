"""Shared brick-geometry and bucket-ordering machinery.

The layout CLASS is tested in `test_brick_packed.py`. What lives here is the
module-level machinery both the class and the ratified probe depend on, plus one
measured finding about which positions to bucket.

The per-bucket `BrickLayout` these tests originally covered was removed when the
brick-packed design superseded it; its measurements survive in
`runs/v2/m1_layout_record.md` and its code in git history.
"""


import numpy as np
import pytest


from inexor.codec import T9Layout  # noqa: E402
from inexor.layout import (  # noqa: E402
    assert_brick_divides_buffer,
    brick_span,
    bucket_ijk_from_key,
    bucket_order_key,
)

L_BOX = 128.0
N_PART = 32
N_FINE = 64


@pytest.fixture(autouse=True)
def _x64():
    """Enable x64 for this module only, then restore (test_bispectrum.py pattern)."""
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _t9():
    return T9Layout(box_size=L_BOX, n_part=N_PART, bucket_cells=2)


def _lattice(seed, n_part=N_PART, jitter=0.35):
    rng = np.random.default_rng(seed)
    g = (np.arange(n_part) + 0.5) * (L_BOX / n_part)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    return np.mod(q + rng.normal(scale=jitter * L_BOX / n_part, size=q.shape), L_BOX)


def test_brick_span_wrap_guard_still_refuses_the_measured_bug():
    """n_fine=64, n_tile=32, b=20 gave span 6 against a 4-brick grid and a 3.29
    relative short-force error -- silent double-painting. Pinned so promotion
    cannot lose it."""
    with pytest.raises(ValueError, match="double-count"):
        brick_span(32, 20, 8, 4)


def test_assert_brick_divides_buffer_catches_the_overhang_case():
    """T128/b96 -> brick 64, union side 384 against P=320: the one V4a leg with
    3.04e9 overhang and a ~1.7x inflated cap."""
    assert_brick_divides_buffer(128, 32, 32, 512)  # the operating geometry is fine
    with pytest.raises(ValueError, match="does not divide"):
        assert_brick_divides_buffer(128, 96, 64, 512)


def test_bucket_order_key_round_trips():
    t9 = _t9()
    x = _lattice(3)
    key, ijk, _ = bucket_order_key(x, t9, 2)
    assert np.array_equal(bucket_ijk_from_key(key, t9, 2), ijk)


