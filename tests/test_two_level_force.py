"""M-v2-2: the promoted two-level force must be BITWISE the probe.

WHY BITWISE AND NOT A TOLERANCE. D-v2-10, D-v2-11 and D-v2-12 are measurements
OF `scripts/v2_g5_core.py`. If the promoted engine is merely close to it, those
three ratified records quietly stop describing the shipped artifact -- which is
the failure D-v2-16 clause 7 exists to prevent. So the probe stays UNMODIFIED
and is imported here as the oracle, and equality is exact.

WHAT MAKES A BITWISE ASSERTION WORTH ANYTHING. It has to be able to fail. Two
ways this class of check has already been caught passing vacuously in this
project:

  - on all-zero arrays, because `r_s=None` makes the short kernel identically
    zero (V4 parity check, 2026-08-07);
  - on a degenerate fixture, where the quantity under test is constant by
    construction (the equilateral control's window arm, deneb 361).

So every comparison here goes through `_agree`, which asserts dynamic range
FIRST, and there is an explicit test that a 1e-9 perturbation breaks equality.
An assertion that cannot discriminate is worse than an absent one: it reads as
evidence.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

import v2_g5_core as probe  # noqa: E402

from inexor import forces, painting  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    """The probe builds its kernels in f64 and the gate arms run x64; a promoted
    twin compared under f32 would differ for a reason that has nothing to do
    with the promotion."""
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


L_BOX = 32.0
N_MESH = 16
N_PART = 8
R_S = 1.5


def _positions(seed=0, n_part=N_PART):
    """A perturbed Lagrangian lattice -- structured like real ICs rather than
    uniform noise, so the CIC/TSC stencils see the correlated offsets they
    actually get."""
    rng = np.random.default_rng(seed)
    g = (np.arange(n_part) + 0.5) * (L_BOX / n_part)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    return np.mod(q + rng.normal(scale=0.3 * L_BOX / n_part, size=q.shape), L_BOX)


def _agree(mine, theirs, what, min_range=1e-6, min_nonzero_frac=0.5):
    """Bitwise equality, but only after proving the arrays carry signal.

    `min_range` is on the peak absolute value of the ORACLE, so an all-zero or
    constant array fails here rather than passing the equality trivially.

    `min_nonzero_frac` is the second half of that, and it has to be an argument
    rather than a constant because TILE-LOCAL quantities are legitimately mostly
    zero: a tile's padded box holds an eighth of the volume per side, so ~7/8 of
    the gather rows are out-of-box and zero BY CONSTRUCTION, and most cells of a
    sparse tile mesh are empty. Lowering it for those is fine; lowering it to 0
    is not, which is why it is stated per call site with a reason.
    """
    mine = np.asarray(mine)
    theirs = np.asarray(theirs)
    assert mine.shape == theirs.shape, f"{what}: shape {mine.shape} vs {theirs.shape}"
    peak = float(np.max(np.abs(theirs)))
    assert peak > min_range, (
        f"{what}: oracle peak |value| = {peak:.3e} is below {min_range:.1e} -- the "
        "comparison is vacuous, not passing"
    )
    nz = int(np.count_nonzero(theirs))
    assert nz > min_nonzero_frac * theirs.size, (
        f"{what}: oracle is {nz}/{theirs.size} nonzero, under the "
        f"{min_nonzero_frac:.0%} this comparison expects"
    )
    n_diff = int(np.count_nonzero(mine != theirs))
    assert n_diff == 0, (
        f"{what}: {n_diff}/{mine.size} elements differ, "
        f"max |delta| = {float(np.max(np.abs(mine - theirs))):.3e} against peak {peak:.3e}"
    )


# --------------------------------------------------------------- the kernels


def test_kernel_grids_is_bitwise_the_probe():
    """Non-cubic shape on purpose: the padded tile box is non-cubic in general,
    and a cubic-only fixture would not exercise the per-axis k build that is the
    whole reason this exists beside `k_components`."""
    shape, cell = (8, 12, 16), L_BOX / 16
    mine = forces.kernel_grids(shape, cell, np.float64)
    theirs = probe.kernel_grids(shape, cell, np.float64)
    for a, b, name in zip(mine, theirs, ("ikx", "iky", "ikz", "k2_true", "k2_safe")):
        _agree(a, b, f"kernel_grids/{name}")


def test_k2_true_keeps_a_genuine_dc_zero_and_k2_safe_does_not():
    """The distinction the two returns exist for: S built from k2_safe would
    give S(0) = exp(-r_s^2) != 1 and break the split at DC silently."""
    _, _, _, k2_true, k2_safe = forces.kernel_grids((8,) * 3, 1.0)
    assert k2_true[0, 0, 0] == 0.0
    assert k2_safe[0, 0, 0] == 1.0
    assert forces.s_of_k(k2_true, R_S)[0, 0, 0] == 1.0


@pytest.mark.parametrize("which", ["mono", "long", "short"])
def test_split_factor_is_bitwise_the_probe(which):
    _, _, _, k2_true, _ = forces.kernel_grids((8,) * 3, 1.0)
    mine = forces.split_factor(k2_true, R_S, which)
    theirs = probe.split_factor(k2_true, R_S, which)
    # `mono` is identically 1 and `long` is ~1 at low k, so the usual
    # dynamic-range guard would misfire; assert the shape of each instead.
    assert np.array_equal(mine, theirs)
    if which == "short":
        _agree(mine, theirs, "split_factor/short")


def test_long_plus_short_is_exactly_mono():
    """Floor F1, and the reason `split_factor` must keep its `1.0 - S` form. If
    this ever needs a tolerance, someone has rewritten it as -expm1 and every
    error this engine reports has stopped being attributable."""
    _, _, _, k2_true, _ = forces.kernel_grids((12,) * 3, 0.5)
    s_long = forces.split_factor(k2_true, R_S, "long")
    s_short = forces.split_factor(k2_true, R_S, "short")
    mono = forces.split_factor(k2_true, R_S, "mono")
    assert np.array_equal(s_long + s_short, mono), "long + short != mono BITWISE"
    assert float(np.max(s_short)) > 0.9, "fixture does not reach the short-dominated regime"


def test_split_kernels_is_bitwise_the_probe_gauss_family():
    shape, cell = (16, 16, 16), L_BOX / 16
    for which in ("mono", "long", "short"):
        mine = forces.split_kernels(shape, cell, which, r_s=R_S)
        theirs = probe.split_kernels(shape, cell, which, "gauss", r_s=R_S)
        for a, b, ax in zip(mine, theirs, "xyz"):
            _agree(a, b, f"split_kernels/{which}/{ax}")


def test_windowed_families_are_not_reachable_from_the_package():
    """They measured ~10x worse in the coarse arm and D-v2-10 froze the gaussian
    family. Asserted so a later caller cannot quietly resurrect a branch that
    the ratified records do not cover."""
    import inspect

    assert "family" not in inspect.signature(forces.split_kernels).parameters


@pytest.mark.parametrize("order", [2, 3])
def test_assignment_window_is_bitwise_the_probe(order):
    shape, cell = (8, 12, 16), 0.7
    _agree(
        forces.assignment_window(shape, cell, order),
        probe.assignment_window(shape, cell, order),
        f"assignment_window/order{order}",
    )


def test_cic_match_factor_is_bitwise_the_probe_including_the_clip():
    shape = (16, 16, 16)
    for clip in (None, 10.0):
        mine, mine_max = forces.cic_match_factor(shape, 2.0, 0.5, clip=clip)
        theirs, theirs_max = probe.cic_match_factor(shape, 2.0, 0.5, clip=clip)
        _agree(mine, theirs, f"cic_match_factor/clip={clip}")
        assert mine_max == theirs_max
    # the clip must actually bind on this fixture, or it is untested
    assert mine_max > 10.0, f"max_applied {mine_max} does not reach the clip"


# ------------------------------------------------------------------- the paint


def test_paint_tsc_f64_is_bitwise_the_probe():
    import jax.numpy as jnp

    pos = jnp.asarray(_positions(1))
    _agree(
        painting.paint_tsc_f64(pos, N_MESH, L_BOX, N_PART**3),
        probe.paint_tsc_f64(pos, N_MESH, L_BOX, N_PART**3),
        "paint_tsc_f64",
    )


def test_tsc_read_vector_is_bitwise_the_probe():
    import jax.numpy as jnp

    pos = jnp.asarray(_positions(2))
    rng = np.random.default_rng(3)
    g = [jnp.asarray(rng.normal(size=(N_MESH,) * 3)) for _ in range(3)]
    _agree(
        painting.tsc_read_vector(*g, pos, N_MESH, L_BOX),
        probe.tsc_read_vector(*g, pos, N_MESH, L_BOX),
        "tsc_read_vector",
    )


def test_tsc_weights_are_a_partition_of_unity():
    """Not a parity check but the property that makes TSC a mass-conserving
    assignment at all: if the three per-axis weights stop summing to 1, the
    paint silently loses or invents mass and every downstream number moves."""
    import jax.numpy as jnp

    pos = jnp.asarray(_positions(4))
    _, w = painting._tsc_pieces(pos, L_BOX / N_MESH)
    total = np.asarray(w[0] + w[1] + w[2])
    assert np.max(np.abs(total - 1.0)) < 1e-15, f"max |sum w - 1| = {np.max(np.abs(total - 1.0)):e}"


def test_density_f64_is_bitwise_the_probe():
    import jax.numpy as jnp

    pos = jnp.asarray(_positions(5))
    _agree(
        forces.density_f64(pos, N_MESH, L_BOX, N_PART**3),
        probe.density_f64(pos, N_MESH, L_BOX, N_PART**3),
        "density_f64",
    )


# ------------------------------------------------------------- the global arm


@pytest.mark.parametrize("which", ["mono", "long", "short"])
@pytest.mark.parametrize("assign", ["cic", "tsc"])
def test_force_global_is_bitwise_the_probe(which, assign):
    import jax.numpy as jnp

    pos = jnp.asarray(_positions(6))
    mine, mine_max = forces.force_global(
        pos, N_MESH, L_BOX, N_PART**3, which, r_s=R_S, assign=assign
    )
    theirs, theirs_max = probe.force_global(
        pos, N_MESH, L_BOX, N_PART**3, which, family="gauss", r_s=R_S, assign=assign
    )
    _agree(mine, theirs, f"force_global/{which}/{assign}")
    assert mine_max == theirs_max


def test_force_global_matching_arm_is_bitwise_the_probe():
    """The matched coarse arm is the one D-v2-10 ratified, so it is the one that
    most needs pinning -- and it is the only path where `cic_match_factor` runs
    inside the solve rather than standalone."""
    import jax.numpy as jnp

    pos = jnp.asarray(_positions(7))
    cell_c, cell_f = L_BOX / N_MESH, L_BOX / (N_MESH * 4)
    mine, mine_max = forces.force_global(
        pos, N_MESH, L_BOX, N_PART**3, "long", r_s=R_S,
        match=(cell_c, cell_f), clip=10.0, assign="tsc",
    )
    theirs, theirs_max = probe.force_global(
        pos, N_MESH, L_BOX, N_PART**3, "long", family="gauss", r_s=R_S,
        match=(cell_c, cell_f), clip=10.0, assign="tsc",
    )
    _agree(mine, theirs, "force_global/matched")
    assert mine_max == theirs_max > 1.0, "the matching factor did not bind"


def test_force_global_long_plus_short_recovers_mono_at_the_f1_floor():
    """The identity that makes the split's error attributable, end to end
    through paint, solve and gather rather than at the kernel alone."""
    import jax.numpy as jnp

    pos = jnp.asarray(_positions(8))
    g_long, _ = forces.force_global(pos, N_MESH, L_BOX, N_PART**3, "long", r_s=R_S)
    g_short, _ = forces.force_global(pos, N_MESH, L_BOX, N_PART**3, "short", r_s=R_S)
    g_mono, _ = forces.force_global(pos, N_MESH, L_BOX, N_PART**3, "mono", r_s=R_S)
    scale = float(np.max(np.abs(g_mono)))
    assert scale > 1e-6, "degenerate fixture: the monolithic force is ~zero"
    resid = float(np.max(np.abs(g_long + g_short - g_mono))) / scale
    assert resid < 1e-12, f"F1 relative residual {resid:.3e} exceeds 1e-12"


# --------------------------------------------------------- the anti-vacuity arm


def test_the_parity_check_can_actually_fail():
    """Break it on purpose. A 1e-9 relative perturbation -- far below any
    tolerance this project would have used -- must destroy bitwise equality, and
    `_agree` must be what reports it.

    This exists because the first V4 attempt at a parity check passed on all-zero
    arrays: `r_s=None` made the short kernel identically zero, so two different
    implementations agreed perfectly about nothing.
    """
    import jax.numpy as jnp

    pos = np.asarray(_positions(9))
    theirs, _ = probe.force_global(
        jnp.asarray(pos), N_MESH, L_BOX, N_PART**3, "short", family="gauss", r_s=R_S
    )
    nudged, _ = forces.force_global(
        jnp.asarray(pos * (1.0 + 1e-9)), N_MESH, L_BOX, N_PART**3, "short", r_s=R_S
    )
    with pytest.raises(AssertionError, match="elements differ"):
        _agree(nudged, theirs, "deliberately perturbed")


def test_the_vacuity_guard_rejects_an_all_zero_oracle():
    """The other half: `_agree` must refuse to pass on the degenerate arrays
    that made the original check meaningless, even when they are equal."""
    zeros = np.zeros((64,))
    with pytest.raises(AssertionError, match="vacuous"):
        _agree(zeros, zeros, "all-zero")


# =========================================================== the tile geometry


def test_padded_size_selection_is_bitwise_the_probe_over_the_frozen_range():
    """The <=512 entries fix every already-measured tile selection, so the whole
    frozen range is checked rather than sampled. This is the same exhaustive
    check that licensed appending the >512 block in the first place."""
    for want in range(1, 513):
        assert forces.padded_size(want, 0) == probe.padded_size(want, 0), f"want={want}"


def test_padded_size_refuses_a_degenerate_tile():
    """A padded tile at least as big as the box does more FFT work than the
    monolithic solve it replaces, and with the brick wrap it is where the
    double-count bug lives."""
    with pytest.raises(ValueError, match="degenerate"):
        forces.padded_size(64, 40, n_fine=64)


def test_tile_origin_may_be_negative_and_the_wrap_still_works():
    """The buffer of tile (0,0,0) hangs off the low edge, so its origin is
    negative by construction -- that is the case `mod` is relied on to handle,
    and a "fix" that clamped it would silently drop the wrapped buffer."""
    cell = L_BOX / N_MESH
    origin, extent = forces.tile_origin_extent((0, 0, 0), 8, 4, cell)
    assert np.all(origin < 0)
    o2, e2 = probe.tile_origin_extent((0, 0, 0), 8, 4, cell)
    assert np.array_equal(origin, o2) and extent == e2
    # a particle just inside the far edge must land in the low tile's buffer
    u = np.asarray(forces.tile_local_coords(np.array([[L_BOX - 0.1] * 3]), origin, L_BOX))
    assert np.all(u < extent), "the wrapped buffer particle was not captured"


# ------------------------------------------------------- the tile paint/gather


# The tile fixture needs its own geometry, and BOTH constraints below are
# load-bearing. FFT_FRIENDLY starts at 32, so a padded tile cannot be smaller
# than that. And P must be strictly LESS than n_fine, or the padded box covers
# the whole volume, no particle is ever outside a tile, and the `ok` mask plus
# the n_out contract go untested -- which is exactly how the first version of
# this fixture (n_fine=32, P=32) passed while exercising nothing.
# n_fine=64 with T=16/b=8 gives P=32 and n_side=4, so 64 tiles each covering an
# eighth of the box per side.
N_FINE_T, N_TILE_T, B_FINE_T, N_PART_T = 64, 16, 8, 16


def _tile_fixture(seed, n_tile=N_TILE_T, b_fine=B_FINE_T, n_fine=N_FINE_T):
    """Members of one tile, staged exactly as the driver stages them."""
    import jax.numpy as jnp

    pos = _positions(seed, N_PART_T)
    cell = L_BOX / n_fine
    P, b_real = forces.padded_size(n_tile, b_fine, n_fine=n_fine)
    origin, _ = forces.tile_origin_extent((0, 0, 0), n_tile, b_real, cell)
    u = jnp.asarray(np.mod(pos - origin, L_BOX))
    live = jnp.asarray(np.ones((pos.shape[0],), dtype=bool))
    return u, live, (P,) * 3, cell


def test_tile_paint_f64_is_bitwise_the_probe():
    u, live, shape, cell = _tile_fixture(10)
    mean = N_PART_T**3 / float(N_FINE_T) ** 3
    mine, n_out_mine = forces.tile_paint_f64(u, live, shape, cell, mean)
    theirs, n_out_theirs = probe.tile_paint_f64(u, live, shape, cell, mean)
    # a sparse tile mesh is mostly empty cells; the occupied ones are the test
    _agree(mine, theirs, "tile_paint_f64", min_nonzero_frac=0.02)
    assert int(n_out_mine) == int(n_out_theirs)
    assert int(n_out_mine) > 0, "fixture has no out-of-box rows; the mask is untested"


def test_tile_gather_vector_is_bitwise_the_probe():
    import jax.numpy as jnp

    u, live, shape, cell = _tile_fixture(11)
    rng = np.random.default_rng(12)
    g = [jnp.asarray(rng.normal(size=shape)) for _ in range(3)]
    mine, n_out_mine = forces.tile_gather_vector(*g, u, live, shape, cell)
    theirs, n_out_theirs = probe.tile_gather_vector(*g, u, live, shape, cell)
    # ~7/8 of rows are out-of-box and zero by construction; the in-box ones are
    # what is being compared, and they must all be nonzero
    _agree(mine, theirs, "tile_gather_vector", min_nonzero_frac=0.05)
    in_box = int(np.asarray(live).sum()) - int(n_out_mine)
    assert int(np.count_nonzero(np.asarray(theirs).any(axis=1))) == in_box, (
        "an in-box row gathered exactly zero, or an out-of-box row gathered nonzero"
    )
    assert int(n_out_mine) == int(n_out_theirs)


def test_tile_paint_conserves_mass_over_the_in_box_rows():
    """The property the CIC weights exist to have. If the corner weights stop
    summing to 1 the paint silently loses mass, which no parity check against a
    twin carrying the same bug would ever show."""
    u, live, shape, cell = _tile_fixture(13)
    mean = N_PART_T**3 / float(N_FINE_T) ** 3
    mesh, n_out = forces.tile_paint_f64(u, live, shape, cell, mean)
    n_in = int(np.asarray(live).sum()) - int(n_out)
    assert float(np.asarray(mesh).sum()) * mean == pytest.approx(n_in, rel=1e-12)


# ---------------------------------------------------------- the tiled arm


def _probe_membership(pos, n_fine, n_tile, b_fine):
    """The probe's own bucketing, so parity isolates the FORCE from the exchange.

    The layout's `tile_members` is verified elsewhere to return the same SET, but
    not the same ORDER -- and order changes the f64 scatter-add sequence, hence
    the bits. Driving both sides from one membership is what makes this a test of
    the promoted force rather than of two bucketings.
    """
    cell = L_BOX / n_fine
    _, b_real = forces.padded_size(n_tile, b_fine, n_fine=n_fine)
    n_brick = probe.choose_brick(n_tile, b_real, n_fine)
    order, starts, nb = probe.brick_buckets(pos, n_fine, n_brick, cell)
    n_side = n_fine // n_tile
    tiles = [(i, j, k) for i in range(n_side) for j in range(n_side) for k in range(n_side)]
    cap, _ = probe.tile_capacity(order, starts, nb, tiles, n_tile, b_real, n_brick)

    def member_fn(tijk):
        return probe.tile_members(order, starts, nb, tijk, n_tile, b_real, n_brick)

    return member_fn, cap


@pytest.mark.parametrize("pad_fill", ["cycle", "zero"])
def test_force_short_tiled_is_bitwise_the_probe(pad_fill):
    """The operating path, at the geometry the gates run. Both padding fills,
    because they must give bitwise-identical forces -- that equality is what
    makes the `zero` arm usable as a pure cost A/B."""
    n_tile, b_fine = N_TILE_T, B_FINE_T
    pos = _positions(14, N_PART_T)
    member_fn, cap = _probe_membership(pos, N_FINE_T, n_tile, b_fine)
    mine, diag = forces.force_short_tiled(
        pos, N_FINE_T, L_BOX, N_PART_T**3, n_tile, b_fine, member_fn, cap,
        r_s=R_S, pad_fill=pad_fill,
    )
    theirs, pdiag = probe.force_short_tiled(
        pos, N_FINE_T, L_BOX, N_PART_T**3, n_tile, b_fine,
        r_s=R_S, family="gauss", pad_fill=pad_fill,
    )
    _agree(mine, theirs, f"force_short_tiled/{pad_fill}")
    assert diag["partition_ok"] and pdiag["partition_ok"]
    assert diag["padded_P"] == pdiag["padded_P"]
    assert diag["n_overhang_total"] == pdiag["n_overhang_total"] == 0


def test_the_two_padding_fills_agree_bitwise():
    """Stated as its own assertion because it is the premise of the A/B: the
    fills differ only in which mesh addresses the zero-weight scatter-adds
    contend for, so any difference in the FORCE means a pad row is being counted.
    """
    n_tile, b_fine = N_TILE_T, B_FINE_T
    pos = _positions(15, N_PART_T)
    member_fn, cap = _probe_membership(pos, N_FINE_T, n_tile, b_fine)
    args = (pos, N_FINE_T, L_BOX, N_PART_T**3, n_tile, b_fine, member_fn, cap)
    a, _ = forces.force_short_tiled(*args, r_s=R_S, pad_fill="cycle")
    b, _ = forces.force_short_tiled(*args, r_s=R_S, pad_fill="zero")
    _agree(a, b, "pad_fill cycle vs zero")


def test_one_tile_equals_the_whole_box_identity():
    """The tile-identity rung: one tile covering the box with no buffer must
    reproduce the global short solve. This is the check that caught the dropped
    cell layer at 4.6e-1 when the paint required base < P-1."""
    import jax.numpy as jnp

    pos = _positions(16, N_PART_T)
    member_fn, cap = (lambda t: np.arange(pos.shape[0])), pos.shape[0]
    tiled, diag = forces.force_short_tiled(
        pos, N_FINE_T, L_BOX, N_PART_T**3, N_FINE_T, 0, member_fn, cap, r_s=R_S
    )
    glob, _ = forces.force_global(
        jnp.asarray(pos), N_FINE_T, L_BOX, N_PART_T**3, "short", r_s=R_S
    )
    scale = float(np.max(np.abs(glob)))
    assert scale > 1e-6, "degenerate fixture"
    resid = float(np.max(np.abs(tiled - glob))) / scale
    assert resid < 1e-13, f"tile identity relative residual {resid:.3e}"
    assert diag["partition_ok"]


def test_the_accumulate_sink_refuses_a_production_sized_box():
    """D-v2-16 clause 1: the global (n,3) array is 206 GB at C-gh and deleting
    two of them is what makes C-gh runnable. The test path must not become the
    production path by default."""
    pos = _positions(17, N_PART_T)
    member_fn, cap = _probe_membership(pos, N_FINE_T, N_TILE_T, B_FINE_T)
    with pytest.raises(ValueError, match="accumulate sink would allocate"):
        forces.force_short_tiled(
            pos, N_FINE_T, L_BOX, N_PART_T**3, N_TILE_T, B_FINE_T, member_fn, cap,
            r_s=R_S, max_accumulate_bytes=1024,
        )


def test_the_tile_local_sink_sees_every_particle_exactly_once():
    """The production path. Ownership is a partition, so a tile-local sink must
    receive each particle exactly once across all tiles and reconstruct exactly
    what the global accumulator would have built -- without ever holding it."""
    n_tile, b_fine = N_TILE_T, B_FINE_T
    pos = _positions(18, N_PART_T)
    member_fn, cap = _probe_membership(pos, N_FINE_T, n_tile, b_fine)
    args = (pos, N_FINE_T, L_BOX, N_PART_T**3, n_tile, b_fine, member_fn, cap)

    seen, rebuilt = [], np.zeros((pos.shape[0], 3))

    def sink(idx, g_owned):
        seen.append(idx)
        rebuilt[idx] = g_owned

    out, diag = forces.force_short_tiled(*args, r_s=R_S, sink=sink)
    assert out is None, "a tile-local sink must not also materialize the global array"
    seen = np.concatenate(seen)
    assert len(seen) == pos.shape[0] and len(np.unique(seen)) == pos.shape[0]
    reference, _ = forces.force_short_tiled(*args, r_s=R_S)
    _agree(rebuilt, reference, "tile-local sink vs accumulate")


# ------------------------------------------- regressions for the measured bugs


def test_regression_brick_span_refuses_the_wrapping_double_count():
    """Measured 2026-07-15: tile membership walks bricks by MODULAR index, so
    once span > nb the same brick is visited twice and its particles are painted
    TWICE. At n_fine=64, n_tile=32, b=20 that gave span=6 against nb=4 and a
    3.29 RELATIVE short-force error -- silent density corruption that reads like
    a catastrophic tiling failure rather than a bookkeeping bug."""
    from inexor.layout import brick_span

    with pytest.raises(ValueError, match="wraps the box and would double-count"):
        brick_span(32, 20, 16, 4)


def test_regression_the_last_cell_layer_of_the_padded_box_is_painted():
    """Measured: requiring base < P-1 silently discarded the last cell layer of
    every padded box, producing 4.6e-1 on the tile identity and a fake buffer
    plateau that mimicked kernel ringing. Pin it directly -- a particle in the
    final cell layer must carry weight."""
    import jax.numpy as jnp

    P, cell = 16, 1.0
    # sits inside the last cell along x, so its base is P-1 and it wraps to 0
    u = jnp.asarray([[float(P) - 0.5, 0.5, 0.5]])
    live = jnp.asarray([True])
    mesh, n_out = forces.tile_paint_f64(u, live, (P,) * 3, cell, 1.0)
    assert int(n_out) == 0, "a particle inside the padded box was counted as outside"
    assert float(np.asarray(mesh).sum()) == pytest.approx(1.0, rel=1e-13), (
        "the last cell layer lost its mass"
    )


def test_regression_padding_rows_do_not_funnel_onto_flat_index_zero():
    """Measured 2026-08-07 (`eba91ab`): `where(ok, flat, 0)` sent every padded
    row to flat index 0, so ~2e6 zero-weight f64 atomics per tile contended for
    ONE address -- 2.218x on the device phase. The fix is that `flat` is returned
    UNMASKED and only the weight is zeroed, which is asserted here directly
    because the cost is invisible to any correctness check."""
    import jax.numpy as jnp

    base = jnp.asarray([[5, 6, 7], [5, 6, 7]], dtype=jnp.int32)
    frac = jnp.asarray([[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]])
    ok = jnp.asarray([True, False])
    flat, w = forces._tile_corner(base, frac, 1.0 - frac, (0, 0, 0), (16, 16, 16), ok)
    assert int(flat[0]) == int(flat[1]) != 0, (
        "the masked row's index was rewritten -- that is the 2.2x contention bug"
    )
    assert float(w[0]) > 0.0 and float(w[1]) == 0.0, "masking must be on the WEIGHT"
