"""lagrangian_tiles: the vectorized form must equal the loop form exactly.

WHY THIS TEST EXISTS. The loop form -- one full-particle-array mask per tile --
is O(n_tiles x n_particles) and cannot run at cgh64 (32768 tiles x 134M
particles). It was replaced by a per-particle candidate-tile form. Membership
decides which particles a tile evolves, so a discrepancy would not raise; it
would silently change the physics of every tiled arm, and the identity ladder
(which runs at cdev8, where the loop form was affordable) would keep passing.

So the reference here is the ORIGINAL implementation, kept verbatim, and the
assertion is exact equality of all three returned fields over a spread of
geometries -- including b = 0, buffers wider than the tile (which clamps the
candidate count), and a PERMUTED particle order, since qi comes from
rint(q/spacing) % n_part and is not guaranteed to arrive in lattice order.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

from v2_g3_core import lagrangian_tiles  # noqa: E402


def reference_tiles(qi, n_part, n_fine, n_tile, b_realized):
    """The pre-vectorization implementation, unchanged."""
    t_lag = int(n_tile) * int(n_part) // int(n_fine)
    b_lag = int(np.ceil(float(b_realized) * int(n_part) / int(n_fine)))
    n_side = int(n_fine) // int(n_tile)
    blk = qi // t_lag
    out = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                tijk = (i, j, k)
                core = np.ones(qi.shape[0], bool)
                member = np.ones(qi.shape[0], bool)
                for ax in range(3):
                    core &= blk[:, ax] == tijk[ax]
                    lo = tijk[ax] * t_lag - b_lag
                    member &= ((qi[:, ax] - lo) % int(n_part)) < (t_lag + 2 * b_lag)
                idx = np.where(member)[0]
                out.append((tijk, idx, core[idx]))
    return out, t_lag, b_lag


def lattice_qi(n_part):
    a = np.arange(n_part, dtype=np.int64)
    g = np.meshgrid(a, a, a, indexing="ij")
    return np.stack([x.ravel() for x in g], axis=1)


# (n_part, n_fine, n_tile, b_realized). n_fine % n_tile == 0 throughout.
GEOMS = [
    (16, 32, 4, 0),    # no buffer: exactly one candidate tile per axis
    (16, 32, 4, 1),
    (16, 32, 4, 2),
    (16, 32, 8, 2),    # b_lag/t_lag = 1/4, the ratio EVERY GRID config runs at
    (16, 32, 8, 4),
    (16, 32, 8, 8),    # buffer == tile
    (16, 32, 16, 16),  # buffer wider than the tile -> candidate count clamps
    (8, 32, 8, 4),     # t_lag = 2, the coarsest Lagrangian block here
    (16, 16, 4, 3),    # n_fine == n_part, so fine cells == particle cells
    (12, 24, 6, 5),    # non-power-of-2, and b_lag rounds UP (5*12/24 = 2.5)
]


def _assert_same(got, ref):
    tiles_g, t_lag_g, b_lag_g = got
    tiles_r, t_lag_r, b_lag_r = ref
    assert (t_lag_g, b_lag_g) == (t_lag_r, b_lag_r)
    assert len(tiles_g) == len(tiles_r)
    for (tg, ig, cg), (tr, ir, cr) in zip(tiles_g, tiles_r):
        assert tg == tr, f"tile order diverged: {tg} vs {tr}"
        assert ig.dtype == ir.dtype, f"{tg}: idx dtype {ig.dtype} vs {ir.dtype}"
        assert cg.dtype == cr.dtype, f"{tg}: core dtype {cg.dtype} vs {cr.dtype}"
        np.testing.assert_array_equal(ig, ir, err_msg=f"members differ at {tg}")
        np.testing.assert_array_equal(cg, cr, err_msg=f"core flags differ at {tg}")


@pytest.mark.parametrize("n_part,n_fine,n_tile,b", GEOMS)
def test_matches_reference_on_lattice(n_part, n_fine, n_tile, b):
    qi = lattice_qi(n_part)
    _assert_same(lagrangian_tiles(qi, n_part, n_fine, n_tile, b),
                 reference_tiles(qi, n_part, n_fine, n_tile, b))


@pytest.mark.parametrize("n_part,n_fine,n_tile,b", GEOMS)
def test_matches_reference_under_permutation(n_part, n_fine, n_tile, b):
    """Particle order must not matter -- and idx must stay ASCENDING per tile."""
    qi = lattice_qi(n_part)
    qi = qi[np.random.default_rng(0).permutation(qi.shape[0])]
    got = lagrangian_tiles(qi, n_part, n_fine, n_tile, b)
    _assert_same(got, reference_tiles(qi, n_part, n_fine, n_tile, b))
    for tijk, idx, _ in got[0]:
        assert np.all(np.diff(idx) > 0), f"{tijk}: idx not strictly ascending"


def test_every_particle_lands_in_exactly_one_core():
    """The cores partition the particles -- the invariant the buffer must not break."""
    n_part, n_fine, n_tile = 16, 32, 4
    qi = lattice_qi(n_part)
    for b in (0, 2, 8):
        tiles, _, _ = lagrangian_tiles(qi, n_part, n_fine, n_tile, b)
        owned = np.concatenate([idx[core] for _, idx, core in tiles])
        np.testing.assert_array_equal(np.sort(owned), np.arange(qi.shape[0]))


def test_buffer_widens_membership_monotonically():
    """A bigger buffer can only ADD members, never drop one."""
    n_part, n_fine, n_tile = 16, 32, 4
    qi = lattice_qi(n_part)
    prev = None
    for b in (0, 1, 2, 4, 8):
        tiles, _, _ = lagrangian_tiles(qi, n_part, n_fine, n_tile, b)
        sets = [set(idx.tolist()) for _, idx, _ in tiles]
        if prev is not None:
            for a, c in zip(prev, sets):
                assert a <= c
        prev = sets
