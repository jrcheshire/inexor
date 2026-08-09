"""M-v2-4 S1: the dtype ledger for the force path.

WHY THIS FILE EXISTS. Before it, NOTHING in the suite asserted the dtype of a
force, density, kernel or gather array. `grep '\\.dtype =='` over `tests/` hit
only the codec, the bispectrum, the integrator and the tile membership. So an
"f32 force mesh" that silently ran in f64 would have passed every test in the
repo, including all ~30 bitwise-parity tests in `test_two_level_force.py` --
because they compare VALUES against the probe, and an f32 arm that upcast back
to f64 produces exactly the f64 values they expect.

That is the failure mode this project has been bitten by repeatedly: a knob that
reports success without having applied. The ledger lands BEFORE any source
change in M-v2-4 so every later stage has a tripwire from its first commit, and
so the f64 defaults are pinned as a contract rather than surviving by accident.

TWO PROMOTION TRAPS are pinned here as measured facts rather than as prose,
because both are live in the current code and both would make an f32 arm
invisible. They are asserted on the ACTUAL library functions, not on synthetic
arrays, so a future change to those functions is what fails the test:

  1. `f64 * complex64 -> complex128`. Narrowing `kernel_grids`' dtype alone is
     not enough: `split_kernels` builds its real prefactor from `k2_true`, which
     is DELIBERATELY f64 so `s_of_k` sees a true DC zero, and f64 times a c64
     kernel promotes straight back to c128.
  2. `f32_accumulator + f64_weight * f32_field -> f64`. Three of the four
     gathers already set their accumulator from the field's dtype, but all four
     build their weights from f64 positions, so the accumulation promotes and
     the gather returns f64 no matter what the mesh is.

Both tests carry the reason in their assertion message, so a later
"simplification" that drops one of M-v2-4's casts fails here with an
explanation rather than with a bare dtype mismatch.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

from inexor import engine, forces, painting, state  # noqa: E402
from inexor.codec import T9Layout  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    """The ledger is only meaningful under x64.

    The library never enables x64 and callers opt in, so WITHOUT this fixture
    every `jnp.float64` in the force path is silently f32 and a "the default is
    f64" assertion would fail for a reason that has nothing to do with M-v2-4.
    That silent degradation is itself a defect -- M-v2-4 S6 makes
    `EngineConfig.validate()` refuse the f64-without-x64 combination -- but the
    refusal belongs there, and here the fixture just makes the ledger readable.
    """
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


L_BOX = 32.0
N_MESH = 16
N_PART = 8
R_S = 1.5

# The tile-arm geometry, matching test_two_level_force.py so a dtype read here
# and a value read there describe the same configuration.
N_FINE_T, N_TILE_T, B_FINE_T, N_PART_T = 64, 16, 8, 16


def _positions(seed=0, n_part=N_PART):
    rng = np.random.default_rng(seed)
    g = (np.arange(n_part) + 0.5) * (L_BOX / n_part)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    return np.mod(q + rng.normal(scale=0.3 * L_BOX / n_part, size=q.shape), L_BOX)


def _tile_fixture(seed):
    import jax.numpy as jnp

    pos = _positions(seed, N_PART_T)
    cell = L_BOX / N_FINE_T
    P, b_real = forces.padded_size(N_TILE_T, B_FINE_T, n_fine=N_FINE_T)
    origin, _ = forces.tile_origin_extent((0, 0, 0), N_TILE_T, b_real, cell)
    u = jnp.asarray(np.mod(pos - origin, L_BOX))
    live = jnp.asarray(np.ones((pos.shape[0],), dtype=bool))
    return u, live, (P,) * 3, cell


def _owned_subblock(seed, fdtype=None):
    """A staged sub-block plus the tile-0 rows it is allowed to serve.

    `gather_coarse_subblock` REFUSES rows outside the tile's core -- reading past
    the block would wrap to the far side of the mesh silently -- so a dtype read
    has to stage the same way the engine does rather than handing it every row.
    """
    import jax.numpy as jnp

    n_coarse = N_FINE_T // forces.COARSE_RATIO
    rng = np.random.default_rng(seed)
    g = [jnp.asarray(rng.normal(size=(n_coarse,) * 3)) for _ in range(3)]
    if fdtype is not None:
        g = [m.astype(fdtype) for m in g]
    pos = _positions(seed + 1, N_PART_T)
    tile_side = L_BOX * N_TILE_T / N_FINE_T
    owned = np.all((pos >= 0.0) & (pos < tile_side), axis=1)
    assert int(owned.sum()) > 20, "fixture has too few owned rows to read a dtype from"
    origin, extent = forces.coarse_subblock_origin_extent(
        (0, 0, 0), N_TILE_T, n_coarse, N_FINE_T
    )
    sub = [jnp.asarray(forces.stage_coarse_subblock(m, origin, extent)) for m in g]
    return g, sub, jnp.asarray(pos[owned]), pos, origin, n_coarse, L_BOX / n_coarse


def _name(a):
    return np.asarray(a).dtype.name


def _assert_dtypes(rows):
    """Report EVERY mismatch at once.

    A dtype regression usually moves several arrays along one path, and failing
    on the first hides how far it reached -- which is the diagnostic.
    """
    bad = [(what, got, want) for what, got, want in rows if got != want]
    assert not bad, "dtype ledger:\n" + "\n".join(
        f"  {what}: {got}, expected {want}" for what, got, want in bad
    )


# ============================================================ the f64 defaults
#
# Every row below passes TODAY. The file is the contract that they keep doing so
# while M-v2-4 threads an opt-in f32 through the same functions.


def test_the_kernel_defaults_are_f64():
    """`kernel_grids` takes an fdtype and `split_kernels` does not (yet); both
    build in f64 today, and `k2_true` must stay f64 at every fdtype because
    `s_of_k` needs a genuine DC zero."""
    shape, cell = (8, 12, 16), L_BOX / 16
    ikx, iky, ikz, k2_true, k2_safe = forces.kernel_grids(shape, cell, np.float64)
    kx, ky, kz = forces.split_kernels(shape, cell, "long", r_s=R_S)
    _assert_dtypes([
        ("kernel_grids/ikx", _name(ikx), "complex128"),
        ("kernel_grids/iky", _name(iky), "complex128"),
        ("kernel_grids/ikz", _name(ikz), "complex128"),
        ("kernel_grids/k2_true", _name(k2_true), "float64"),
        ("kernel_grids/k2_safe", _name(k2_safe), "float64"),
        ("split_kernels/Kx", _name(kx), "complex128"),
        ("split_kernels/Ky", _name(ky), "complex128"),
        ("split_kernels/Kz", _name(kz), "complex128"),
        ("s_of_k", _name(forces.s_of_k(k2_true, R_S)), "float64"),
        ("split_factor", _name(forces.split_factor(k2_true, R_S, "short")), "float64"),
        ("assignment_window", _name(forces.assignment_window(shape, cell, 1)), "float64"),
        ("cic_match_factor", _name(forces.cic_match_factor(shape, cell, cell)[0]), "float64"),
    ])


def test_the_low_rank_k_components_default_is_f32_and_is_a_different_builder():
    """`k_components` defaults to f32 while `kernel_grids` defaults to f64.

    That asymmetry is real and load-bearing: `k_components` serves
    `make_force_fn`, whose whole point is a dtype-parameterized single-level
    solve, while `kernel_grids` serves the two-level split where the f64 build is
    the precision island. Pinned so the two are not "harmonized" into one
    default by someone reading only one of them.
    """
    ikx, iky, ikz, inv_k2 = forces.k_components(N_MESH, L_BOX)
    _assert_dtypes([
        ("k_components/ikx", _name(ikx), "complex64"),
        ("k_components/iky", _name(iky), "complex64"),
        ("k_components/ikz", _name(ikz), "complex64"),
        ("k_components/inv_k2", _name(inv_k2), "float32"),
    ])
    ikx64, _, _, inv64 = forces.k_components(N_MESH, L_BOX, np.float64)
    _assert_dtypes([
        ("k_components(f64)/ikx", _name(ikx64), "complex128"),
        ("k_components(f64)/inv_k2", _name(inv64), "float64"),
    ])


def test_the_paint_and_decode_defaults_are_unchanged():
    """The accumulators are the part M-v2-4 must NOT touch: int32 for the
    deterministic primal paints, f64 for the differentiable twins. Only the
    DECODE dtype is a knob, and its two call sites disagree today by design --
    `density_contrast` takes f32 (every ratified v1 number ran through it) and
    `density_tsc` pins f64."""
    import jax.numpy as jnp

    pos = _positions(1)
    n_tot = N_PART**3
    _assert_dtypes([
        ("paint_f32 (default)", _name(painting.paint_f32(pos, N_MESH, L_BOX)), "float32"),
        ("paint_f32(f64)",
         _name(painting.paint_f32(pos, N_MESH, L_BOX, fdtype=jnp.float64)), "float64"),
        ("paint_int", _name(painting.paint_int(pos, N_MESH, L_BOX)), "int32"),
        ("paint_tsc_int", _name(painting.paint_tsc_int(pos, N_MESH, L_BOX)), "int32"),
        ("paint_tsc_f64", _name(painting.paint_tsc_f64(pos, N_MESH, L_BOX, n_tot)), "float64"),
        ("counts_from_int (default)",
         _name(painting.counts_from_int(painting.paint_int(pos, N_MESH, L_BOX))), "float32"),
        ("counts_from_int(f64)",
         _name(painting.counts_from_int(painting.paint_int(pos, N_MESH, L_BOX),
                                        fdtype=jnp.float64)), "float64"),
        ("density_contrast", _name(painting.density_contrast(pos, N_MESH, L_BOX, n_tot)),
         "float32"),
        ("density_f64", _name(forces.density_f64(pos, N_MESH, L_BOX, n_tot)), "float64"),
        ("density_tsc(int)",
         _name(painting.density_tsc(pos, N_MESH, L_BOX, n_tot, paint="int")), "float64"),
        ("density_tsc(f64)",
         _name(painting.density_tsc(pos, N_MESH, L_BOX, n_tot, paint="f64")), "float64"),
    ])


def test_the_global_arm_defaults_are_f64():
    pos = _positions(2)
    n_tot = N_PART**3
    g, _ = forces.force_global(pos, N_MESH, L_BOX, n_tot, "long", r_s=R_S)
    delta = forces.density_f64(pos, N_MESH, L_BOX, n_tot)
    meshes = forces.coarse_force_meshes(delta, N_MESH, L_BOX, "long", r_s=R_S)
    _assert_dtypes(
        [("force_global", _name(g), "float64")]
        + [(f"coarse_force_meshes[{i}]", _name(m), "float64") for i, m in enumerate(meshes)]
    )


def test_the_tile_arm_defaults_are_f64():
    import jax.numpy as jnp

    u, live, shape, cell = _tile_fixture(10)
    mean = N_PART_T**3 / float(N_FINE_T) ** 3
    mesh_f64, _ = forces.tile_paint_f64(u, live, shape, cell, mean)
    mesh_int, _ = forces.tile_paint_int(u, live, shape, cell)
    rng = np.random.default_rng(12)
    g = [jnp.asarray(rng.normal(size=shape)) for _ in range(3)]
    gathered, _ = forces.tile_gather_vector(*g, u, live, shape, cell)
    _assert_dtypes([
        ("tile_paint_f64", _name(mesh_f64), "float64"),
        ("tile_paint_int", _name(mesh_int), "int32"),
        ("tile_delta_from_int", _name(forces.tile_delta_from_int(mesh_int, mean)), "float64"),
        ("tile_gather_vector", _name(gathered), "float64"),
    ])


def test_the_gathers_and_subblock_staging_default_to_f64():
    g, sub, owned, pos, origin, n_coarse, cell_c = _owned_subblock(30)
    got = forces.gather_coarse_subblock(*sub, owned, origin, cell_c, n_coarse, assign="tsc")
    _assert_dtypes([
        ("stage_coarse_subblock", _name(sub[0]), "float64"),
        ("gather_coarse_subblock", _name(got), "float64"),
        ("tsc_read_vector", _name(painting.tsc_read_vector(*g, pos, n_coarse, L_BOX)), "float64"),
        ("cic_read_vector", _name(painting.cic_read_vector(*g, pos, n_coarse, L_BOX)), "float64"),
    ])


def test_the_engine_coarse_arm_defaults_to_f64():
    """The shipping path, end to end: the streamed integer accumulation decodes
    to f64 and the coarse solve stays there. This row is the one M-v2-4 moves."""
    cfg = engine.EngineConfig(
        box_size=L_BOX, n_part=32, n_fine=64, n_coarse=16, n_tile=16, b_fine=8
    )
    x = _positions(40, 32)
    v = np.random.default_rng(41).normal(scale=0.5, size=x.shape)
    t9 = T9Layout(box_size=L_BOX, n_part=32, bucket_cells=2)
    st = state.SlotState.build(x, v, t9, 64 // cfg.n_brick, arena_frac=0.05)
    delta = engine.coarse_delta_streamed(st, cfg)
    _assert_dtypes([("engine.coarse_delta_streamed", _name(delta), "float64")])


# ============================================== S2: the kernel seam, at f32


def test_split_kernels_narrows_the_kernel_and_keeps_the_build_in_f64():
    """The knob applies AND the precision island survives it."""
    shape, cell = (8, 12, 16), L_BOX / 16
    kx, ky, kz = forces.split_kernels(shape, cell, "long", r_s=R_S, fdtype=np.float32)
    _assert_dtypes([
        ("split_kernels(f32)/Kx", _name(kx), "complex64"),
        ("split_kernels(f32)/Ky", _name(ky), "complex64"),
        ("split_kernels(f32)/Kz", _name(kz), "complex64"),
    ])
    # the build is f64 whatever the output dtype: k2_true must keep a genuine
    # DC zero or S(0) != 1 and the split breaks at DC, silently
    assert _name(forces.kernel_grids(shape, cell, np.float32)[3]) == "float64"


def test_the_f32_kernel_is_the_f64_one_to_two_ulp_and_is_not_equal_to_it():
    """Both halves matter.

    Equal would mean the narrowing did not happen (the promotion trap). Further
    than 2 ulp would mean it narrowed something it should not have -- the build
    rather than the prefactor.
    """
    shape, cell = (32,) * 3, L_BOX / 32
    for which in ("long", "short"):
        k64 = forces.split_kernels(shape, cell, which, r_s=R_S)
        k32 = forces.split_kernels(shape, cell, which, r_s=R_S, fdtype=np.float32)
        for a32, a64, axis in zip(k32, k64, "xyz"):
            what = f"split_kernels/{which}/K{axis}"
            assert _name(a32) == "complex64", what
            assert not np.array_equal(a32, a64.astype(np.complex64)), (
                f"{what}: the f32 arm is bitwise the narrowed f64 one, so the prefactor "
                "cast did nothing -- check for a promotion back to complex128"
            )
            assert np.allclose(a32, a64, rtol=2.0**-22, atol=0.0), (
                f"{what}: further than 2 f32 ulp from the f64 kernel. The prefactor is the "
                "ONLY thing that may narrow; k2_true/k2_safe/S stay f64."
            )


def test_split_kernels_refuses_a_dtype_it_cannot_serve():
    """Loud, like the other setup-time refusals. A silently-ignored dtype is the
    failure mode the whole ledger exists to prevent."""
    with pytest.raises(ValueError, match="fdtype must be float32 or float64"):
        forces.split_kernels((8,) * 3, 1.0, "long", r_s=R_S, fdtype=np.float16)


# ====================================================== the two promotion traps


def test_narrowing_the_kernel_build_alone_does_not_narrow_the_kernel():
    """TRAP 1, asserted on the real functions.

    `kernel_grids(fdtype=f32)` genuinely returns c64 and f32. But
    `split_kernels`' prefactor comes from `split_factor(k2_true, ...)`, and
    `k2_true` is pinned f64 on purpose, so `fac / k2_safe` is f64 and
    `f64 * c64 -> c128`. An M-v2-4 that threads the dtype into `kernel_grids`
    and stops there produces a c128 kernel, which then promotes the f32 delta
    back to f64 at `dk * jnp.asarray(k)` and the milestone is invisible.

    The fix S2 lands is to narrow the REAL PREFACTOR instead. This test pins the
    trap so that fix cannot be undone by "simplifying" the cast away.
    """
    shape, cell = (8,) * 3, L_BOX / 8
    ikx, _, _, k2_true, k2_safe = forces.kernel_grids(shape, cell, np.float32)
    assert _name(ikx) == "complex64", "kernel_grids' own fdtype should reach ik"
    assert _name(k2_safe) == "float32"
    assert _name(k2_true) == "float64", (
        "k2_true must stay f64 at every fdtype -- s_of_k needs a genuine DC zero"
    )
    fac = forces.split_factor(k2_true, R_S, "long")
    assert _name(fac) == "float64", "the prefactor inherits k2_true's f64, by design"
    assert _name((fac / k2_safe) * ikx) == "complex128", (
        "f64 * complex64 promotes to complex128: narrowing kernel_grids alone leaves a "
        "c128 kernel, which re-promotes an f32 delta at the kernel multiply. M-v2-4 S2 "
        "narrows the real prefactor (fac / k2_safe) instead -- do not remove that cast."
    )


def test_a_gather_accumulator_is_defeated_by_an_f64_weight():
    """TRAP 2, asserted on the real function.

    `gather_coarse_subblock` sets its accumulators from the FIELD's dtype
    (forces.py, `dtype=fx.dtype`), which reads like it already follows the mesh.
    It does not: the corner weights are built from f64 positions, and
    `f32_acc + f64_w * f32_field -> f64`. So handing it an f32 sub-block today
    returns f64, and an f32 arm built without S4's weight cast would look like it
    worked while gathering in double precision.
    """
    import jax.numpy as jnp

    _, sub, owned, _, origin, n_coarse, cell_c = _owned_subblock(50, fdtype=jnp.float32)
    assert _name(sub[0]) == "float32", "staging is a slice and must not change dtype"
    got = forces.gather_coarse_subblock(*sub, owned, origin, cell_c, n_coarse, assign="tsc")
    assert _name(got) == "float64", (
        "an f32 field gathered through f64 weights returns f64 -- this is the trap M-v2-4 "
        "S4's weight cast exists for. When S4 lands, this expectation becomes float32 and "
        "the assertion below is what proves the cast is doing the work."
    )
