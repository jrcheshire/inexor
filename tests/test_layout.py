"""Shared brick-geometry and bucket-ordering machinery.

The layout CLASS is tested in `test_brick_packed.py`. What lives here is the
module-level machinery both the class and the ratified probe depend on, plus one
measured finding about which positions to bucket.

The per-bucket `BrickLayout` these tests originally covered was removed when the
brick-packed design superseded it; its measurements survive in
`runs/v2/m1_layout_record.md` and its code in git history.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

from v2_g5_core import brick_buckets as probe_brick_buckets  # noqa: E402
from v2_g5_core import choose_brick as probe_choose_brick  # noqa: E402
from v2_g5_core import tile_members as probe_tile_members  # noqa: E402

from inexor.codec import T9Layout, roundtrip_positions  # noqa: E402
from inexor.layout import (  # noqa: E402
    BrickPackedLayout,
    assert_brick_divides_buffer,
    brick_span,
    bucket_ijk_from_key,
    bucket_order_key,
    choose_brick,
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


def test_choose_brick_matches_the_probe():
    for n_tile in (16, 32, 64):
        for b_fine in (0, 4, 8, 16):
            assert choose_brick(n_tile, b_fine, N_FINE) == probe_choose_brick(
                n_tile, b_fine, N_FINE
            )


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


def test_quantization_moves_only_particles_within_half_a_quantum_of_a_boundary():
    """WHICH positions to bucket, measured rather than assumed.

    The layout buckets the STORED (quantized) position, because that is the only
    coordinate the state has; the probe buckets whatever float it is handed. Fed
    matched inputs the two agree exactly. Fed raw floats they differ only for
    particles within half a quantum of a brick face -- a boundary effect well
    under a percent -- and following the stored position is the correct side of
    it: a particle must be gathered into the tile its state says it is in.
    """
    import jax.numpy as jnp

    n_part, n_fine, bricks = 64, 128, 8
    n_tile, b_fine = 32, 16
    t9 = T9Layout(box_size=L_BOX, n_part=n_part, bucket_cells=2)
    x = _lattice(6, n_part=n_part)
    lay = BrickPackedLayout.build(x, t9, bricks, brick_slack=0.10)
    n_brick = choose_brick(n_tile, b_fine, n_fine)
    assert n_fine // n_brick == bricks

    cell = L_BOX / n_fine
    xq = np.asarray(roundtrip_positions(jnp.asarray(x), t9))
    o_raw, s_raw, nb = probe_brick_buckets(x, n_fine, n_brick, cell)
    o_q, s_q, _ = probe_brick_buckets(xq, n_fine, n_brick, cell)

    n_diff = n_total = 0
    for tijk in ((0, 0, 0), (1, 2, 3), (nb - 1, 0, nb - 1)):
        t = np.asarray(tijk)
        mine = lay.tile_members(t, n_tile, b_fine, n_brick, n_fine)
        quant = probe_tile_members(o_q, s_q, nb, t, n_tile, b_fine, n_brick)
        raw = probe_tile_members(o_raw, s_raw, nb, t, n_tile, b_fine, n_brick)
        assert np.array_equal(np.sort(mine), np.sort(quant)), "exact on matched inputs"
        n_diff += len(np.setxor1d(mine, raw))
        n_total += len(mine)
    assert n_diff > 0, "no boundary particles -- the fixture cannot see the effect"
    assert n_diff / n_total < 0.01, f"boundary disagreement {n_diff}/{n_total} is not an edge"
