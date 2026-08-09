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


@pytest.mark.detflag
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


@pytest.mark.detflag
def test_density_f64_is_bitwise_the_probe():
    import jax.numpy as jnp

    pos = jnp.asarray(_positions(5))
    _agree(
        forces.density_f64(pos, N_MESH, L_BOX, N_PART**3),
        probe.density_f64(pos, N_MESH, L_BOX, N_PART**3),
        "density_f64",
    )


# ------------------------------------------------------------- the global arm


@pytest.mark.detflag
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


@pytest.mark.detflag
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


@pytest.mark.detflag
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


@pytest.mark.detflag
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


@pytest.mark.detflag
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


@pytest.mark.detflag
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


@pytest.mark.detflag
def test_force_short_tiled_is_bitwise_the_probe_under_heavy_clustering():
    """Gate geometry 3: the one that exercises `pad_fill` and `_tile_corner`
    under real contention.

    A clustered field makes `cap` (a max over tiles) far exceed the typical
    member count, so most rows in most tiles are PADDING -- which is the regime
    where the flat-index-0 funnel cost 2.218x, and the only one where the two
    pad_fill arms do meaningfully different work. A uniform fixture has pad_frac
    near zero and cannot see any of it.
    """
    n_tile, b_fine = N_TILE_T, B_FINE_T
    rng = np.random.default_rng(40)
    pos = np.mod(rng.normal(loc=L_BOX * 0.5, scale=L_BOX * 0.06, size=(N_PART_T**3, 3)), L_BOX)
    member_fn, cap = _probe_membership(pos, N_FINE_T, n_tile, b_fine)
    counts = [len(member_fn(t)) for t in
              [(i, j, k) for i in range(N_FINE_T // n_tile)
               for j in range(N_FINE_T // n_tile) for k in range(N_FINE_T // n_tile)]]
    pad_frac = 1.0 - float(np.mean(counts)) / cap
    assert pad_frac > 0.75, f"fixture is not padding-dominated (pad_frac {pad_frac:.2f})"

    args = (pos, N_FINE_T, L_BOX, N_PART_T**3, n_tile, b_fine, member_fn, cap)
    for pad_fill in ("cycle", "zero"):
        mine, diag = forces.force_short_tiled(*args, r_s=R_S, pad_fill=pad_fill)
        theirs, _ = probe.force_short_tiled(
            pos, N_FINE_T, L_BOX, N_PART_T**3, n_tile, b_fine,
            r_s=R_S, family="gauss", pad_fill=pad_fill,
        )
        _agree(mine, theirs, f"clustered/{pad_fill}")
        assert diag["partition_ok"] and diag["n_overhang_total"] == 0
        # a clustered field must produce a much larger force than a smooth one,
        # or the fixture is not actually clustered
        assert float(np.max(np.abs(mine))) > 1.0


# ================================ coarse sub-block staging (D-v2-16 clause 3)


def _coarse_setup(seed=30):
    """A global coarse force plus the geometry to slice it by tile."""
    import jax.numpy as jnp

    n_coarse = N_FINE_T // forces.COARSE_RATIO  # 16
    rng = np.random.default_rng(seed)
    g = [jnp.asarray(rng.normal(size=(n_coarse,) * 3)) for _ in range(3)]
    pos = _positions(seed + 1, N_PART_T)
    return g, pos, n_coarse, L_BOX / n_coarse


def test_the_staged_subblock_is_a_verbatim_periodic_slice():
    """The premise everything else rests on. If the slice is not exactly the
    cells the global mesh holds at those (wrapped) indices, the gather cannot be
    bitwise and the whole staging idea is a change in physics."""
    g, _, n_coarse, _ = _coarse_setup()
    origin, extent = forces.coarse_subblock_origin_extent(
        (0, 0, 0), N_TILE_T, n_coarse, N_FINE_T
    )
    assert np.any(np.asarray(origin) < 0), "tile 0's block should straddle the low boundary"
    sub = forces.stage_coarse_subblock(g[0], origin, extent)
    gg = np.asarray(g[0])
    for i in range(extent):
        for j in range(extent):
            for k in range(extent):
                want = gg[(origin[0] + i) % n_coarse,
                          (origin[1] + j) % n_coarse,
                          (origin[2] + k) % n_coarse]
                assert sub[i, j, k] == want


def test_the_subblock_is_far_smaller_than_the_global_mesh():
    """The reason it exists. Asserted as a ratio so a later halo change that
    quietly ate the saving shows up here."""
    n_coarse = 1024  # C-gh
    _, extent = forces.coarse_subblock_origin_extent((0, 0, 0), 256, n_coarse, 4096)
    assert extent == 64 + 2 * forces.COARSE_HALO
    assert (n_coarse / extent) ** 3 > 1000


@pytest.mark.parametrize("assign", ["cic", "tsc"])
def test_gathering_from_the_subblock_is_bitwise_the_global_gather(assign):
    """The contract: staging is a memory decision and must not move a number.

    Checked on OWNED rows of several tiles, including tile 0 whose block
    straddles the periodic boundary -- the case where an index-shift bug would
    hide.
    """
    import jax.numpy as jnp

    g, pos, n_coarse, cell_c = _coarse_setup()
    n_side = N_FINE_T // N_TILE_T
    glob = np.asarray(
        (painting.tsc_read_vector if assign == "tsc" else painting.cic_read_vector)(
            *g, jnp.asarray(pos), n_coarse, L_BOX
        )
    )
    tile_side = L_BOX * N_TILE_T / N_FINE_T
    checked = 0
    for tijk in ((0, 0, 0), (1, 2, 3), (n_side - 1, n_side - 1, n_side - 1)):
        lo = np.asarray(tijk) * tile_side
        owned = np.all((pos >= lo) & (pos < lo + tile_side), axis=1)
        if not owned.any():
            continue
        checked += int(owned.sum())
        origin, extent = forces.coarse_subblock_origin_extent(
            tijk, N_TILE_T, n_coarse, N_FINE_T
        )
        sub = [jnp.asarray(forces.stage_coarse_subblock(c, origin, extent)) for c in g]
        mine = forces.gather_coarse_subblock(
            *sub, jnp.asarray(pos[owned]), origin, cell_c, n_coarse, assign=assign
        )
        _agree(mine, glob[owned], f"coarse subblock/{assign}/{tijk}", min_nonzero_frac=0.9)
    assert checked > 100, f"only {checked} owned rows exercised; the fixture is too thin"


def test_the_subblock_gather_refuses_rows_it_cannot_serve():
    """A row outside the tile's core reads wrapped values from the far side of
    the block, silently and plausibly. That is the exact shape of bug the halo
    exists to prevent, so it raises instead."""
    import jax.numpy as jnp

    g, pos, n_coarse, cell_c = _coarse_setup()
    origin, extent = forces.coarse_subblock_origin_extent(
        (0, 0, 0), N_TILE_T, n_coarse, N_FINE_T
    )
    sub = [jnp.asarray(forces.stage_coarse_subblock(c, origin, extent)) for c in g]
    with pytest.raises(ValueError, match="reaches outside the staged sub-block"):
        forces.gather_coarse_subblock(*sub, jnp.asarray(pos), origin, cell_c, n_coarse)


def test_the_halo_is_wide_enough_for_tsc_rounding():
    """halo=2 is not decoration. TSC's base comes from round(), not floor(), so a
    core-edge particle reaches one cell further than a CIC bound suggests --
    checked by driving rows to both extremes of the core and confirming the
    stencil still fits."""
    import jax.numpy as jnp

    g, _, n_coarse, cell_c = _coarse_setup()
    tile_side = L_BOX * N_TILE_T / N_FINE_T
    edge = np.array([[1e-12, 1e-12, 1e-12], [tile_side - 1e-12] * 3])
    origin, extent = forces.coarse_subblock_origin_extent(
        (0, 0, 0), N_TILE_T, n_coarse, N_FINE_T
    )
    sub = [jnp.asarray(forces.stage_coarse_subblock(c, origin, extent)) for c in g]
    out = forces.gather_coarse_subblock(
        *sub, jnp.asarray(edge), origin, cell_c, n_coarse, assign="tsc"
    )
    assert np.all(np.isfinite(np.asarray(out)))


# ============================================= paint_tsc_int (D-v2-16 clause 2)


def test_paint_tsc_int_is_order_independent():
    """The whole point. `paint_tsc_f64` accumulates through order-dependent f64
    `.at[].add`, so the ratified coarse arm violates D-006 today; integer
    addition is associative, so this must be bit-identical under a permutation of
    particle order. That is not hypothetical for us -- the brick-sorted layout
    reorders particles every single step."""
    import jax.numpy as jnp

    pos = _positions(20, N_PART_T)
    perm = np.random.default_rng(21).permutation(pos.shape[0])
    a = painting.paint_tsc_int(jnp.asarray(pos), N_MESH, L_BOX)
    b = painting.paint_tsc_int(jnp.asarray(pos[perm]), N_MESH, L_BOX)
    assert np.array_equal(np.asarray(a), np.asarray(b)), (
        "the integer TSC paint is order-DEPENDENT, which defeats its only purpose"
    )
    assert int(np.asarray(a).max()) > 0, "degenerate fixture: nothing was painted"


def test_the_f64_tsc_paint_really_is_order_dependent_on_this_fixture():
    """The control for the test above, and it is not a formality.

    If the f64 twin happened to be order-invariant here, the integer test would
    be comparing two arrays that agree for a reason having nothing to do with
    integer arithmetic -- a pass proving nothing, which is the failure mode this
    file exists to avoid. Measured on THIS fixture, and it is visible even on
    CPU with no atomics involved: permuting the particles moves 770 of 4096 cells
    at 4.4e-16, purely from the changed accumulation order.

    So the defect D-v2-16 clause 2 names is real at f64 on any backend, and the
    GPU-atomics story is an amplifier rather than the cause.
    """
    import jax.numpy as jnp

    pos = _positions(22, N_PART_T)
    perm = np.random.default_rng(23).permutation(pos.shape[0])
    n_tot = N_PART_T**3
    a = np.asarray(painting.paint_tsc_f64(jnp.asarray(pos), N_MESH, L_BOX, n_tot))
    b = np.asarray(painting.paint_tsc_f64(jnp.asarray(pos[perm]), N_MESH, L_BOX, n_tot))
    n_diff = int(np.count_nonzero(a != b))
    assert n_diff > 0, (
        "the f64 TSC paint is order-INVARIANT on this fixture, so the integer "
        "test above proves nothing -- pick a fixture where the defect is visible"
    )
    # and it must be roundoff-scale, or something worse than ordering is wrong
    assert float(np.max(np.abs(a - b))) < 1e-12


def test_paint_tsc_int_matches_the_f64_twin_within_the_quantization_bound():
    """Bounded agreement, not equality: the two differ by the per-corner
    rounding, 27 corners at 2^-frac_bits each."""
    import jax.numpy as jnp

    pos = jnp.asarray(_positions(24, N_PART_T))
    n_tot = N_PART_T**3
    exact = np.asarray(painting.density_tsc(pos, N_MESH, L_BOX, n_tot, paint="f64"))
    quant = np.asarray(painting.density_tsc(pos, N_MESH, L_BOX, n_tot, paint="int"))
    mean = n_tot / float(N_MESH) ** 3
    bound = 27 * 2.0**-12 / mean  # per-particle corner rounding, in delta units
    err = float(np.max(np.abs(quant - exact)))
    assert err < bound, f"max |delta_int - delta_f64| = {err:.3e} exceeds {bound:.3e}"
    assert err > 0.0, "the two paints agree exactly, so the int path is not quantizing"


def test_tsc_headroom_bound_is_the_derived_one_and_refuses_a_real_overflow():
    """The 27-corner stencil's bound is 5.359375x the cell occupancy, not CIC's
    implicit 1x. Both directions asserted: the shipped configuration must pass,
    and a frac_bits that genuinely overflows must raise."""
    assert painting.TSC_CELL_WEIGHT_BOUND == pytest.approx(5.359375)
    painting.check_tsc_paint_headroom(N_PART_T**3, 12)  # production: ~9.8x headroom
    with pytest.raises(ValueError, match="TSC int-paint headroom"):
        painting.check_tsc_paint_headroom(10**9, 16)


def test_the_tsc_headroom_bound_is_not_below_a_measured_worst_case():
    """A derived bound is a claim. Check it against a construction that drives
    one cell as hard as the stencil allows: every particle at the same cell
    centre, where each contributes 0.75^3 to that cell."""
    import jax.numpy as jnp

    n = 500
    cell = L_BOX / N_MESH
    centre = np.full((n, 3), 4.0 * cell)  # exactly on a cell centre -> d = 0
    mesh = np.asarray(painting.paint_tsc_int(jnp.asarray(centre), N_MESH, L_BOX))
    hottest = int(mesh.max())
    allowed = painting.TSC_CELL_WEIGHT_BOUND * n * 2.0**12
    assert hottest <= allowed, f"measured peak {hottest} exceeds the derived bound {allowed:.3e}"
    assert hottest == pytest.approx(0.75**3 * n * 2**12, rel=1e-3), (
        "the fixture is not actually driving a cell to the single-cell maximum"
    )


# ========================== tile_paint_int: the SHORT arm's D-006 twin (M-v2-3)
#
# D-v2-16 clause 2 named only the coarse `paint_tsc_int`. The short arm has the
# same defect and no document said so: `tile_paint_f64` accumulates through
# order-dependent f64 `.at[].add`, and D-v2-14 clause 4 admits the brick-sorted
# layout ONLY because the paint is order-independent. The layout reorders every
# step, so the arm carrying most of the force was the non-reproducible one.
#
# WHICH FIXTURE CAN SEE THIS, measured before the tests below were written.
# Floating-point reassociation needs enough contributions per cell to bite:
# summing 8 CIC weights into one cell is bitwise identical under permutation,
# 64 is not (7.1e-15), 4096 is not (6.1e-12). The standard perturbed-lattice
# `_positions` fixture puts ~8 corner writes in each occupied tile cell, so the
# f64 tile paint is order-INVARIANT on it -- 0 of 32768 cells move under a
# shuffle. A shuffle test built on that fixture passes for BOTH paints and
# proves nothing.
#
# That is not true of the coarse arm, and the difference is the stencil: TSC's
# 27 corners on a dense mesh already clear the threshold, which is why
# `test_the_f64_tsc_paint_really_is_order_dependent_on_this_fixture` works on
# the ordinary fixture and its short-arm counterpart below needs a CLUMP.


def _clustered_tile_fixture(n=4096, seed=50):
    """A tight clump inside one padded tile: ~2000 particles' worth in one cell.

    This is the regime D-v2-19 measured as real (cdev8's peak bucket population
    was 5943 by the end of a run), and it is the only regime in which the f64
    short-arm paint's order dependence is visible at all. See the note above.
    """
    import jax.numpy as jnp

    cell = L_BOX / N_FINE_T
    _, b_real = forces.padded_size(N_TILE_T, B_FINE_T, n_fine=N_FINE_T)
    P, _ = forces.padded_size(N_TILE_T, B_FINE_T, n_fine=N_FINE_T)
    rng = np.random.default_rng(seed)
    centre = np.full(3, (b_real + 4.0) * cell)
    u = np.mod(centre + rng.normal(scale=0.25 * cell, size=(n, 3)), L_BOX)
    live = jnp.asarray(np.ones((n,), dtype=bool))
    return jnp.asarray(u), live, (P,) * 3, cell, rng.permutation(n)


def test_tile_paint_int_is_order_independent():
    """The whole point, and the precondition D-v2-14 clause 4 assumed."""
    import jax.numpy as jnp

    u, live, shape, cell, perm = _clustered_tile_fixture()
    a, _ = forces.tile_paint_int(u, live, shape, cell)
    b, _ = forces.tile_paint_int(jnp.asarray(np.asarray(u)[perm]), live, shape, cell)
    assert np.array_equal(np.asarray(a), np.asarray(b)), (
        "the integer tile paint is order-DEPENDENT, which defeats its only purpose"
    )
    assert int(np.asarray(a).max()) > 0, "degenerate fixture: nothing was painted"


def test_the_f64_tile_paint_really_is_order_dependent_on_this_fixture():
    """The control, and here it is doing more work than the coarse arm's.

    Measured on this clump: permuting the particles moves 23 of the 27 occupied
    cells at 1.5e-10. On the ordinary `_positions` fixture it moves NONE, which
    is why the clump exists -- without it the test above would be comparing two
    arrays that agree for a reason unrelated to integer arithmetic.
    """
    import jax.numpy as jnp

    u, live, shape, cell, perm = _clustered_tile_fixture(seed=51)
    mean = N_PART_T**3 / float(N_FINE_T) ** 3
    a = np.asarray(forces.tile_paint_f64(u, live, shape, cell, mean)[0])
    b = np.asarray(
        forces.tile_paint_f64(jnp.asarray(np.asarray(u)[perm]), live, shape, cell, mean)[0]
    )
    n_diff = int(np.count_nonzero(a != b))
    assert n_diff > 0, (
        "the f64 tile paint is order-INVARIANT on this fixture, so the integer "
        "test above proves nothing -- the clump is not dense enough"
    )
    assert float(np.max(np.abs(a - b))) < 1e-6, "the difference is larger than roundoff"


@pytest.mark.detflag
def test_a_uniform_fixture_cannot_discriminate_order_which_is_why_the_clump_exists():
    """Pins the measurement the two tests above are built on.

    If someone later 'simplifies' `_clustered_tile_fixture` to the ordinary
    perturbed lattice, the order-independence test keeps passing and silently
    stops testing anything. This fails first and says why.
    """
    import jax.numpy as jnp

    u, live, shape, cell = _tile_fixture(52)
    mean = N_PART_T**3 / float(N_FINE_T) ** 3
    perm = np.random.default_rng(53).permutation(np.asarray(u).shape[0])
    a = np.asarray(forces.tile_paint_f64(u, live, shape, cell, mean)[0])
    b = np.asarray(
        forces.tile_paint_f64(jnp.asarray(np.asarray(u)[perm]), live, shape, cell, mean)[0]
    )
    assert np.array_equal(a, b), (
        "the uniform fixture HAS become order-sensitive -- if that is real, the "
        "clustered fixture is no longer required and this note is stale"
    )


def test_tile_paint_int_matches_the_f64_twin_within_the_quantization_bound():
    """Bounded agreement, not equality: 8 corners at 2^-frac_bits each."""
    u, live, shape, cell, _ = _clustered_tile_fixture(seed=54)
    mean = N_PART_T**3 / float(N_FINE_T) ** 3
    exact = np.asarray(forces.tile_paint_f64(u, live, shape, cell, mean)[0])
    mesh_i, _ = forces.tile_paint_int(u, live, shape, cell)
    quant = np.asarray(forces.tile_delta_from_int(mesh_i, mean))
    n_in = int(np.asarray(live).sum())
    bound = 8 * 2.0**-12 * n_in / mean  # worst case per particle, summed
    err = float(np.max(np.abs(quant - exact)))
    assert err < bound, f"max |int - f64| = {err:.3e} exceeds {bound:.3e}"
    assert err > 0.0, "the two paints agree exactly, so the int path is not quantizing"


def test_tile_paint_int_conserves_mass_within_the_fixed_point_rounding():
    """`tile_paint_conserves_mass_over_the_in_box_rows`' integer counterpart. The
    8 corner weights sum to 1 exactly in f64; after per-corner rounding they sum
    to 1 +- 8 * 2^-(frac_bits+1), so mass is conserved to that, not exactly."""
    u, live, shape, cell, _ = _clustered_tile_fixture(seed=55)
    mesh, n_out = forces.tile_paint_int(u, live, shape, cell)
    n_in = int(np.asarray(live).sum()) - int(n_out)
    total = float(np.asarray(mesh).sum()) / 2.0**12
    assert total == pytest.approx(n_in, rel=8 * 2.0**-13)


def test_tile_paint_headroom_uses_the_strict_cic_bound_and_refuses_an_overflow():
    """CIC's strict factor is 8, not the 1 `check_int_paint_headroom` assumes by
    default. Both directions: production passes, a real overflow raises."""
    assert painting.CIC_CELL_WEIGHT_BOUND == 8.0
    forces.check_tile_paint_headroom(N_PART_T**3, 12)
    with pytest.raises(ValueError, match="int-paint headroom"):
        forces.check_tile_paint_headroom(10**9, 16)
    # and the strict bound must be STRICTER than the legacy default, or passing
    # it through changes nothing and the call site is decorative
    painting.check_int_paint_headroom(10**6, 15, bound=1.0)
    with pytest.raises(ValueError, match="int-paint headroom"):
        painting.check_int_paint_headroom(10**6, 15, bound=painting.CIC_CELL_WEIGHT_BOUND)


def test_the_int_tile_arm_is_reachable_through_force_short_tiled():
    """The knob is wired, the default is unchanged, and the two arms differ."""
    pos = _positions(56, N_PART_T)
    member_fn, cap = _probe_membership(pos, N_FINE_T, N_TILE_T, B_FINE_T)
    args = (pos, N_FINE_T, L_BOX, N_PART_T**3, N_TILE_T, B_FINE_T, member_fn, cap)
    g_f64, d_f64 = forces.force_short_tiled(*args, r_s=R_S)
    g_int, d_int = forces.force_short_tiled(*args, r_s=R_S, paint="int")
    assert d_f64["paint"] == "f64", "the default moved; the probe-parity tests now compare arms"
    assert d_int["paint"] == "int"
    err = float(np.max(np.abs(g_int - g_f64)))
    peak = float(np.max(np.abs(g_f64)))
    assert err > 0.0, "the int arm reproduced the f64 arm exactly, so it is not quantizing"
    assert err < 0.02 * peak, f"int-vs-f64 short force differs by {err / peak:.1%} of peak"


def test_force_global_can_reach_the_int_tsc_paint_and_refuses_int_cic():
    """`force_global(assign='tsc')` called `paint_tsc_f64` directly, so the
    D-006-compliant coarse paint existed and was unreachable from any force path.
    The default stays f64 so D-v2-10/11/12's oracle comparisons are untouched."""
    pos = _positions(57, N_PART_T)
    n_tot = N_PART_T**3
    a, _ = forces.force_global(pos, N_MESH, L_BOX, n_tot, "long", r_s=R_S, assign="tsc")
    b, _ = forces.force_global(
        pos, N_MESH, L_BOX, n_tot, "long", r_s=R_S, assign="tsc", paint="f64"
    )
    c, _ = forces.force_global(
        pos, N_MESH, L_BOX, n_tot, "long", r_s=R_S, assign="tsc", paint="int"
    )
    assert np.array_equal(a, b), "the paint= default is not the f64 path any more"
    err = float(np.max(np.abs(c - a)))
    assert err > 0.0, "the int coarse arm reproduced f64 exactly, so it is not quantizing"
    assert err < 0.02 * float(np.max(np.abs(a)))
    with pytest.raises(ValueError, match="no 'int' accumulator"):
        forces.force_global(pos, N_MESH, L_BOX, n_tot, "mono", assign="cic", paint="int")


def test_the_live_mask_is_a_no_op_when_every_row_is_live():
    """Padding must not move a number.

    The engine pads every tile's rows to a fixed capacity so that ONE XLA shape
    serves all of them -- without it, each tile keyed a new shape and the step
    spent 24.1 s of 32.7 s recompiling. That padding is only admissible because a
    masked run reproduces the unmasked one EXACTLY, which is what this asserts.
    """
    import jax.numpy as jnp

    g, pos, n_coarse, cell_c = _coarse_setup(61)
    tile_side = L_BOX * N_TILE_T / N_FINE_T
    tijk = (1, 2, 3)
    lo = np.asarray(tijk) * tile_side
    owned = np.all((pos >= lo) & (pos < lo + tile_side), axis=1)
    assert owned.sum() > 10, "fixture owns too few rows to be a test"
    core = pos[owned]
    origin, extent = forces.coarse_subblock_origin_extent(tijk, N_TILE_T, n_coarse, N_FINE_T)
    sub = [jnp.asarray(forces.stage_coarse_subblock(c, origin, extent)) for c in g]

    a = np.asarray(
        forces.gather_coarse_subblock(
            *sub, jnp.asarray(core), origin, cell_c, n_coarse, assign="tsc"
        )
    )
    n = core.shape[0]
    padded = np.zeros((2 * n, 3), dtype=np.float64)
    padded[:n] = core
    live = np.zeros(2 * n, dtype=bool)
    live[:n] = True
    b = np.asarray(
        forces.gather_coarse_subblock(
            *sub, jnp.asarray(padded), origin, cell_c, n_coarse, assign="tsc", live=live
        )
    )
    _agree(b[:n], a, "padded vs unpadded gather", min_nonzero_frac=0.9)
    assert np.count_nonzero(b[n:]) == 0, "a padded row gathered a nonzero force"


def test_jitting_the_subblock_gather_would_break_its_bitwise_contract():
    """A measured negative result, kept so it is not rediscovered.

    The gather is the largest remaining term in an engine step (4.9 s of 11.4 s)
    and it runs EAGER, so wrapping it in `jax.jit` is the obvious next
    optimization -- worth about 1.7x. It was tried and REFUSED: under jit, XLA
    fuses and reassociates the corner accumulation, and the result stopped being
    bitwise the global gather -- 86 of 189 elements at 2.220e-16.

    That is physically irrelevant and fatal to the parity gate, which is exactly
    the trade this function's docstring already records rejecting once at
    8.9e-16. Staging is a memory decision and must not move a number.

    This test does not re-run the jit; it pins the CONTRACT the jit broke, so
    that any future attempt fails here rather than silently shipping a 2e-16
    drift into three ratified records.
    """
    import jax.numpy as jnp

    g, pos, n_coarse, cell_c = _coarse_setup(62)
    tile_side = L_BOX * N_TILE_T / N_FINE_T
    tijk = (1, 2, 3)
    lo = np.asarray(tijk) * tile_side
    owned = np.all((pos >= lo) & (pos < lo + tile_side), axis=1)
    glob = np.asarray(painting.tsc_read_vector(*g, jnp.asarray(pos), n_coarse, L_BOX))
    origin, extent = forces.coarse_subblock_origin_extent(tijk, N_TILE_T, n_coarse, N_FINE_T)
    sub = [jnp.asarray(forces.stage_coarse_subblock(c, origin, extent)) for c in g]
    mine = forces.gather_coarse_subblock(
        *sub, jnp.asarray(pos[owned]), origin, cell_c, n_coarse, assign="tsc"
    )
    _agree(mine, glob[owned], "subblock gather stays bitwise", min_nonzero_frac=0.9)
