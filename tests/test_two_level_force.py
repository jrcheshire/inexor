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


def _agree(mine, theirs, what, min_range=1e-6):
    """Bitwise equality, but only after proving the arrays carry signal.

    `min_range` is on the peak absolute value of the ORACLE, so an all-zero or
    constant array fails here rather than passing the equality trivially.
    """
    mine = np.asarray(mine)
    theirs = np.asarray(theirs)
    assert mine.shape == theirs.shape, f"{what}: shape {mine.shape} vs {theirs.shape}"
    peak = float(np.max(np.abs(theirs)))
    assert peak > min_range, (
        f"{what}: oracle peak |value| = {peak:.3e} is below {min_range:.1e} -- the "
        "comparison is vacuous, not passing"
    )
    assert np.count_nonzero(theirs) > theirs.size // 2, (
        f"{what}: oracle is mostly zeros ({np.count_nonzero(theirs)}/{theirs.size} nonzero)"
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
