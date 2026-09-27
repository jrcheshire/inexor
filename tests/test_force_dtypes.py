"""Dtype ledger for the force path.

Kernel, density, decode, gather and engine arrays must carry the dtype asked for, and the f64
defaults are pinned. Value-parity tests cannot see this: an f32 arm that upcasts back to f64
reproduces the f64 values exactly. Two promotion traps are asserted on the real functions:

  1. `f64 * complex64 -> complex128`: `split_kernels`' prefactor comes from `k2_true`, kept f64
     so `s_of_k` sees a true DC zero, so the real prefactor itself must be narrowed.
  2. `f32_acc + f64_weight * f32_field -> f64`: gather weights come from f64 positions, so the
     weight product must be cast to the field dtype.
"""


import numpy as np
import pytest


from inexor import engine, forces, painting, state  # noqa: E402
from inexor.codec import T9Layout  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    """The ledger needs x64: without it every `jnp.float64` is silently f32 (a config that
    does this is refused by `EngineConfig.validate()`, tested below)."""
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


L_BOX = 32.0
N_MESH = 16
N_PART = 8
R_S = 1.5

# Tile-arm geometry, matching test_two_level_force.py.
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
    """A staged sub-block plus the tile-0 rows it may serve; `gather_coarse_subblock`
    refuses rows outside the tile core, which would otherwise wrap silently."""
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
    """Report every mismatch at once: a dtype regression usually moves several arrays along
    one path, and how far it reached is the diagnostic."""
    bad = [(what, got, want) for what, got, want in rows if got != want]
    assert not bad, "dtype ledger:\n" + "\n".join(
        f"  {what}: {got}, expected {want}" for what, got, want in bad
    )


# ============================================================ the f64 defaults


def test_the_kernel_defaults_are_f64():
    """Both kernel builders default to f64, and `k2_true` is f64 because `s_of_k` needs a
    genuine DC zero."""
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
    """`k_components` (for the dtype-parameterized `make_force_fn`) defaults to f32;
    `kernel_grids` (the two-level split's f64 build) defaults to f64. Pinned so the two are
    not harmonized by someone reading only one."""
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
    """Accumulators are int32 for the deterministic paints and f64 for the float twins. Only
    the decode dtype is a knob, and its defaults differ by design: `density_contrast` f32,
    `density_tsc` f64."""
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


def _engine_fixture(seed=40, **kw):
    """The `smoke` config of test_engine.py: the smallest non-degenerate decomposition."""
    cfg = engine.EngineConfig(
        box_size=L_BOX, n_part=32, n_fine=64, n_coarse=16, n_tile=16, b_fine=8, **kw
    )
    x = _positions(seed, 32)
    v = np.random.default_rng(seed + 1).normal(scale=0.5, size=x.shape)
    t9 = T9Layout(box_size=L_BOX, n_part=32, bucket_cells=2)
    st = state.SlotState.build(x, v, t9, 64 // cfg.n_brick, arena_frac=0.05)
    return cfg, st


def test_the_engine_coarse_arm_defaults_to_f64():
    """The streamed integer accumulation decodes to f64 by default."""
    cfg, st = _engine_fixture()
    delta = engine.coarse_delta_streamed(st, cfg)
    _assert_dtypes([("engine.coarse_delta_streamed", _name(delta), "float64")])


# ================================================== the kernel seam, at f32


def test_split_kernels_narrows_the_kernel_and_keeps_the_build_in_f64():
    """The knob applies AND the precision island survives it."""
    shape, cell = (8, 12, 16), L_BOX / 16
    kx, ky, kz = forces.split_kernels(shape, cell, "long", r_s=R_S, fdtype=np.float32)
    _assert_dtypes([
        ("split_kernels(f32)/Kx", _name(kx), "complex64"),
        ("split_kernels(f32)/Ky", _name(ky), "complex64"),
        ("split_kernels(f32)/Kz", _name(kz), "complex64"),
    ])
    # the build stays f64: k2_true needs a genuine DC zero or S(0) != 1
    assert _name(forces.kernel_grids(shape, cell, np.float32)[3]) == "float64"


def test_the_f32_kernel_is_the_f64_one_to_two_ulp_and_is_not_equal_to_it():
    """Equal would mean the narrowing did not happen (the promotion trap); further than 2 ulp
    would mean the build, not just the prefactor, was narrowed."""
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
    """An unsupported dtype raises: a silently ignored dtype is what this ledger guards."""
    with pytest.raises(ValueError, match="fdtype must be float32 or float64"):
        forces.split_kernels((8,) * 3, 1.0, "long", r_s=R_S, fdtype=np.float16)


# ================================================== the decode seam, at f32


def test_the_decode_seam_narrows_the_field():
    import jax.numpy as jnp

    pos = _positions(3)
    n_tot = N_PART**3
    u, live, shape, cell = _tile_fixture(14)
    mean_t = N_PART_T**3 / float(N_FINE_T) ** 3
    mesh_int, _ = forces.tile_paint_int(u, live, shape, cell)
    mesh32, _ = forces.tile_paint_f64(u, live, shape, cell, mean_t, fdtype=jnp.float32)
    _assert_dtypes([
        ("density_f64(f32)",
         _name(forces.density_f64(pos, N_MESH, L_BOX, n_tot, fdtype=jnp.float32)), "float32"),
        ("density_tsc(int, f32)",
         _name(painting.density_tsc(pos, N_MESH, L_BOX, n_tot, paint="int",
                                    fdtype=jnp.float32)), "float32"),
        ("density_tsc(f64 paint, f32)",
         _name(painting.density_tsc(pos, N_MESH, L_BOX, n_tot, paint="f64",
                                    fdtype=jnp.float32)), "float32"),
        ("tile_delta_from_int(f32)",
         _name(forces.tile_delta_from_int(mesh_int, mean_t, fdtype=jnp.float32)), "float32"),
        ("tile_paint_f64(f32)", _name(mesh32), "float32"),
    ])


def _hot_mesh(frac_bits=12, mean=8.0):
    """A coarse mesh at the table's exact mean of 8.0 with one hot cell.

    The hot cell is odd: `< 2^24` is sufficient but not necessary for an exact int->f32
    decode (significand width is what counts; 625 * 2^15 = 2.05e7 is exact), so it must need
    all 25 bits to lose anything.
    """
    m = np.full((4, 4, 4), int(mean * 2**frac_bits), dtype=np.int32)
    m[0, 0, 0] = 2**24 + 12345           # odd: needs all 25 bits
    m[1, 1, 1] = int(300 * 2**frac_bits)  # 512x under the bound
    assert int(m[0, 0, 0]) % 2 == 1 and int(m.max()) > 2**24 > int(m[1, 1, 1])
    return m, frac_bits, mean


def test_without_the_minus_one_the_decode_order_cannot_matter():
    """`tile_delta_from_int` gains nothing from an f64 decode, by construction: with no `- 1`
    (ik(0) = 0 kills DC), mean 8.0 and scale 2**-frac_bits are powers of two, so every step
    is an exact rescaling and both orders round the same integer once."""
    import jax.numpy as jnp

    m, frac_bits, mean = _hot_mesh()
    mi = jnp.asarray(m)
    ours = np.asarray(forces.tile_delta_from_int(mi, mean, frac_bits, fdtype=jnp.float32))
    direct = np.asarray(painting.counts_from_int(mi, frac_bits, fdtype=jnp.float32) / mean)
    assert np.array_equal(ours, direct), (
        "the two decode orders differ on a field with no mean subtraction, which they "
        "cannot do while mean and the frac_bits scale are both powers of two -- check "
        "whether a config broke mean == 8.0"
    )


def test_with_the_minus_one_the_f64_decode_helps_only_the_cells_that_matter_least():
    """With `counts/mean - 1.0`, the f64 decode helps only cells needing > 24 significand bits.

    Near-mean cells (raw sum ~2^15) are exact either way, so there is no cancellation to
    protect; only a cell >= 512x the mean differs, where delta >> 1 and the direct-f32 error
    is < 1e-6 relative. Decode margins must therefore be read relative, on those cells.
    """
    m, frac_bits, mean = _hot_mesh()
    exact = m.astype(np.float64) * 2.0**-frac_bits / mean - 1.0

    f64_then_narrow = (exact).astype(np.float32)
    direct_f32 = (m.astype(np.float32) * np.float32(2.0**-frac_bits)
                  / np.float32(mean) - np.float32(1.0))

    err_a = np.abs(f64_then_narrow.astype(np.float64) - exact)
    err_b = np.abs(direct_f32.astype(np.float64) - exact)

    # the near-mean and moderately-hot cells are exact BOTH ways
    ordinary = np.ones(m.shape, dtype=bool)
    ordinary[0, 0, 0] = False
    assert np.all(err_a[ordinary] == 0.0) and np.all(err_b[ordinary] == 0.0), (
        "a cell under the significand bound lost precision; the fork is not where "
        "this test says it is"
    )
    # only the hot cell moves: exact via f64, relatively negligible error direct
    assert err_a[0, 0, 0] == 0.0
    assert err_b[0, 0, 0] > 0.0, "the hot cell did not lose anything; fixture is vacuous"
    rel = float(err_b[0, 0, 0] / abs(exact[0, 0, 0]))
    assert rel < 1e-6, f"relative error at the hot cell is {rel:.2e}, larger than expected"


def test_counts_from_int_is_exact_where_we_claim_and_not_past_it():
    """The 2^24 exact-decode bound, asserted on `counts_from_int`: exact just below it,
    inexact just above."""
    import jax.numpy as jnp

    lo = np.asarray([[[2**24 - 1]]], dtype=np.int32)
    hi = np.asarray([[[2**24 + 1]]], dtype=np.int32)
    for m, exact in ((lo, True), (hi, False)):
        got = float(np.asarray(painting.counts_from_int(jnp.asarray(m), 0,
                                                        fdtype=jnp.float32))[0, 0, 0])
        assert (got == float(m[0, 0, 0])) is exact, (
            f"int32 {int(m[0, 0, 0])} -> f32 decoded to {got!r}; expected "
            f"{'exact' if exact else 'inexact'}"
        )


# ================================================== the gather seam, at f32


def test_every_gather_follows_the_field_dtype():
    import jax.numpy as jnp

    n_coarse = N_FINE_T // forces.COARSE_RATIO
    rng = np.random.default_rng(60)
    g32 = [jnp.asarray(rng.normal(size=(n_coarse,) * 3), dtype=jnp.float32) for _ in range(3)]
    pos = _positions(61, N_PART_T)
    u, live, shape, cell = _tile_fixture(62)
    t32 = [jnp.asarray(rng.normal(size=shape), dtype=jnp.float32) for _ in range(3)]
    tiled, _ = forces.tile_gather_vector(*t32, u, live, shape, cell)
    _, sub, owned, _, origin, nc, cell_c = _owned_subblock(63, fdtype=jnp.float32)
    _assert_dtypes([
        ("tsc_read_vector(f32)",
         _name(painting.tsc_read_vector(*g32, pos, n_coarse, L_BOX)), "float32"),
        ("cic_read_vector(f32)",
         _name(painting.cic_read_vector(*g32, pos, n_coarse, L_BOX)), "float32"),
        ("tile_gather_vector(f32)", _name(tiled), "float32"),
        ("gather_coarse_subblock(f32)",
         _name(forces.gather_coarse_subblock(*sub, owned, origin, cell_c, nc)), "float32"),
    ])


@pytest.mark.parametrize("assign", ["cic", "tsc"])
def test_the_subblock_gather_stays_bitwise_the_global_gather_at_f32(assign):
    """Staged sub-block gather is bitwise the global gather at f32 (test_two_level_force.py
    covers f64).

    This requires casting the three-factor weight product, not the per-axis weights:
    narrowing early makes 113 of 186 elements differ by 1.19e-7 on this fixture, unmasked
    rows included. Tile 0 straddles the periodic boundary, where an index shift would hide.
    """
    import jax.numpy as jnp

    g, sub, owned, _, origin, n_coarse, cell_c = _owned_subblock(64, fdtype=jnp.float32)
    glob = np.asarray(
        (painting.tsc_read_vector if assign == "tsc" else painting.cic_read_vector)(
            *g, owned, n_coarse, L_BOX
        )
    )
    mine = np.asarray(
        forces.gather_coarse_subblock(*sub, owned, origin, cell_c, n_coarse, assign=assign)
    )
    assert mine.dtype == np.float32 and glob.dtype == np.float32
    peak = float(np.max(np.abs(glob)))
    assert peak > 1e-6, f"oracle peak {peak:.3e} -- the comparison is vacuous, not passing"
    assert int(np.count_nonzero(glob)) > 0.9 * glob.size
    n_diff = int(np.count_nonzero(mine != glob))
    assert n_diff == 0, (
        f"{n_diff}/{mine.size} elements differ at f32, max |delta| "
        f"{float(np.max(np.abs(mine - glob))):.3e}. The staged gather is no longer bitwise "
        "the global one -- check whether the corner weight is being narrowed before the "
        "three-factor product rather than after it."
    )


def test_narrowing_the_weight_early_would_break_the_contract():
    """Control: per-axis narrowing differs from narrowing the product, so the gathers'
    cast-after-product rule matters."""
    rng = np.random.default_rng(70)
    a, b, c = (rng.random(4096) for _ in range(3))
    early = (a.astype(np.float32) * b.astype(np.float32) * c.astype(np.float32))
    late = (a * b * c).astype(np.float32)
    assert not np.array_equal(early, late), (
        "narrowing per-axis and narrowing the product agree on this sample, so the "
        "ordering rule in painting.py has no teeth -- re-derive it before relying on it"
    )


# ============================================== both arms wired, end to end


def test_the_global_arm_runs_f32_end_to_end():
    pos = _positions(80)
    n_tot = N_PART**3
    g32, _ = forces.force_global(pos, N_MESH, L_BOX, n_tot, "long", r_s=R_S,
                                 fdtype=np.float32)
    delta32 = forces.density_f64(pos, N_MESH, L_BOX, n_tot, fdtype=np.float32)
    meshes32 = forces.coarse_force_meshes(delta32, N_MESH, L_BOX, "long", r_s=R_S)
    _assert_dtypes(
        [("force_global(f32)", _name(g32), "float32")]
        + [(f"coarse_force_meshes(f32)[{i}]", _name(m), "float32")
           for i, m in enumerate(meshes32)]
    )


def test_the_match_factor_does_not_promote_the_kernel_back():
    """The matched coarse arm (the shipping configuration) stays f32: `cic_match_factor`
    returns an f64 half-grid, and an uncast `complex64 * float64` would promote to c128."""
    pos = _positions(81)
    cell = L_BOX / N_MESH
    g32, applied = forces.force_global(
        pos, N_MESH, L_BOX, N_PART**3, "long", r_s=R_S,
        match=(cell, cell / 2), assign="tsc", fdtype=np.float32,
    )
    assert _name(g32) == "float32", (
        "the matched arm came back f64 -- the match factor is promoting the kernel"
    )
    assert applied > 1.0, "the match factor was inert, so this test proved nothing"


def test_the_tile_arm_runs_f32_end_to_end_through_both_paints():
    """`fdtype` reaches `tile_delta_from_int` through `make_tile_force_fn`, on both paints."""
    u, live, shape, cell = _tile_fixture(82)
    for paint in ("f64", "int"):
        one_tile, geom = forces.make_tile_force_fn(
            N_FINE_T, L_BOX, N_PART_T**3, N_TILE_T, B_FINE_T, r_s=R_S,
            paint=paint, fdtype=np.float32,
        )
        # ownership is supplied by the caller (see forces.owning_tile); this fixture is
        # tile-local, so every live row is owned
        out, owned, _ = one_tile(u, live, live)
        assert geom["fdtype"] == "float32", f"geom does not carry the dtype ({paint})"
        assert _name(out) == "float32", (
            f"one_tile(paint={paint!r}) returned {_name(out)} at fdtype=float32"
        )
        assert int(np.asarray(owned).sum()) > 0, "no owned rows; the fixture is vacuous"


def test_force_short_tiled_carries_the_dtype_into_its_sink_and_diag():
    def member_fn(_):
        return np.arange(N_PART_T**3, dtype=np.int64)

    pos = _positions(83, N_PART_T)
    cap = N_PART_T**3
    g, diag = forces.force_short_tiled(
        pos, N_FINE_T, L_BOX, N_PART_T**3, N_TILE_T, B_FINE_T, member_fn, cap,
        r_s=R_S, fdtype=np.float32,
    )
    assert diag["fdtype"] == "float32"
    assert _name(g) == "float32", "the accumulate sink is still allocating f64"


def test_coarse_force_meshes_refuses_a_dtype_it_was_not_given():
    """A narrowed delta with f64 kernels would promote back silently (right numbers, double
    the memory), so a dtype disagreement raises rather than casting."""
    import jax.numpy as jnp

    pos = _positions(84)
    d32 = forces.density_f64(pos, N_MESH, L_BOX, N_PART**3, fdtype=jnp.float32)
    with pytest.raises(ValueError, match="delta is float32 but fdtype is float64"):
        forces.coarse_force_meshes(d32, N_MESH, L_BOX, "long", r_s=R_S, fdtype=np.float64)
    # and the agreeing call is fine
    assert _name(forces.coarse_force_meshes(d32, N_MESH, L_BOX, "long", r_s=R_S,
                                            fdtype=np.float32)[0]) == "float32"


# ==================================== the engine knobs, independent and live


def test_the_two_engine_knobs_move_independently():
    """The coarse and fine knobs move independently, so an accuracy reading can be
    attributed to one arm."""
    cfg, st = _engine_fixture(coarse_dtype="float32", fine_dtype="float64")
    assert (cfg.coarse_dtype, cfg.fine_dtype) == ("float32", "float64")
    delta = engine.coarse_delta_streamed(st, cfg)
    assert _name(delta) == "float32", "the coarse knob did not reach the streamed decode"
    _, geom = forces.make_tile_force_fn(
        cfg.n_fine, cfg.box_size, cfg.n_total, cfg.n_tile, cfg.b_fine,
        r_s=cfg.r_s, paint=cfg.paint_short, frac_bits=cfg.frac_bits,
        fdtype=cfg.np_fine_dtype,
    )
    assert geom["fdtype"] == "float64", "the fine arm followed the coarse knob"


def test_the_engine_reports_the_dtypes_it_actually_used():
    """The run summary reads dtypes off the arrays rather than echoing the config, which
    could not catch a knob that did not apply."""
    from inexor.config import Cosmology
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

    cfg, st = _engine_fixture(seed=44, coarse_dtype="float32", fine_dtype="float64")
    cosmo = Cosmology()
    a = a_grid(0.1, 1.0, 3, spacing="log")
    coeffs = bullfrog_float_coeffs(bullfrog_table(a, cosmo))
    out = engine.run(st, cfg, coeffs, census=True)
    s = out[0]
    assert s["coarse_dtype"] == "float32" and s["fine_dtype"] == "float64"
    assert isinstance(s["coarse_peak_int"], int) and s["coarse_peak_int"] > 0
    assert "coarse_cells_inexact_f32" in s and "coarse_exact_decode_ok" in s
    # and the census is OFF by default, since it costs two extra mesh passes
    plain = engine.step(st, cfg, (coeffs[0][1], coeffs[0][2]), 0.0)
    assert "coarse_peak_int" in plain, "the free statistic should always be reported"
    assert "coarse_cells_inexact_f32" not in plain, "the census is not opt-in any more"


def test_the_census_counts_round_trips_not_a_magnitude_threshold():
    """The decode census counts failed int->f32 round trips, not cells above 2^24:
    5000 * 2^12 = 625 * 2^15 (2.05e7) is past 2^24 yet exact."""
    m = np.array([[[5000 * 2**12, 2**24 + 12345]]], dtype=np.int64)
    above_threshold = int(np.count_nonzero(m > 2**24))
    round_trip_fails = int(np.count_nonzero(m.astype(np.float32).astype(np.int64) != m))
    assert above_threshold == 2, "fixture does not have two cells past 2^24"
    assert round_trip_fails == 1, (
        "the round-trip count agrees with the magnitude threshold here, so the census "
        "design makes no difference and the round-trip vs magnitude argument should be re-checked"
    )


def test_the_slabbed_decode_is_bitwise_the_whole_array_form():
    """The slabbed coarse decode (elementwise) is bitwise the whole-array expression. Asserted
    directly: a streamed-vs-monolithic paint test would pass if both forms changed."""
    cfg, st = _engine_fixture(seed=46)
    got = np.asarray(engine.coarse_delta_streamed(st, cfg))
    # whole-array reference
    import jax.numpy as jnp

    from inexor.painting import counts_from_int, paint_tsc_int

    n = cfg.n_coarse
    mesh = np.zeros((n, n, n), dtype=np.int64)
    _, x, _ = st.decode_bricks(list(range(st.n_bricks)))
    mesh += np.asarray(
        paint_tsc_int(jnp.asarray(x), n, cfg.box_size, cfg.frac_bits), dtype=np.int64
    )
    counts = counts_from_int(mesh.astype(np.int32), cfg.frac_bits, fdtype=jnp.float64)
    want = np.asarray(counts) / (float(cfg.n_total) / float(n) ** 3) - 1.0
    assert np.array_equal(got, want), (
        f"{int(np.count_nonzero(got != want))} cells differ between the slabbed and "
        "whole-array decodes; slabbing an elementwise expression must not move a bit"
    )


def test_an_f64_mesh_without_x64_is_refused():
    """f64 without x64 silently gives f32; on a reference arm that makes a comparison read
    zero, i.e. pass, so `validate()` refuses it."""
    import jax

    cfg, _ = _engine_fixture(seed=48)
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="ask for float64 while jax_enable_x64 is False"):
            cfg.validate()
        # and an explicitly-f32 config is fine without x64
        cfg32, _ = _engine_fixture(seed=48, coarse_dtype="float32", fine_dtype="float32")
        assert cfg32.validate() is True
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_a_bad_engine_dtype_is_refused_at_construction():
    """A dtype typo is refused at construction, not at first use."""
    with pytest.raises(ValueError, match="coarse_dtype must be float32 or float64"):
        engine.EngineConfig(box_size=L_BOX, n_part=32, n_fine=64, n_coarse=16,
                            n_tile=16, b_fine=8, coarse_dtype="float16")
    with pytest.raises(ValueError, match="fine_dtype must be float32 or float64"):
        engine.EngineConfig(box_size=L_BOX, n_part=32, n_fine=64, n_coarse=16,
                            n_tile=16, b_fine=8, fine_dtype=np.int32)


# ============================================ the mesh receipt

# The C-gh production geometry. EngineConfig is plain attributes, so this allocates nothing.
C_GH = dict(box_size=1024.0, n_part=2048, n_fine=4096, n_coarse=1024, n_tile=256, b_fine=32)
GB = 1024.0**3


def test_the_coarse_meshes_match_the_ratified_budget():
    """Resident coarse force at C-gh is 12.0 GiB (12.9 GB) at f32 and exactly double at f64.
    Asserted against the budget figure rather than a formula, so moving either fails."""
    f32 = engine.EngineConfig(**C_GH, coarse_dtype="float32").mesh_bytes()
    f64 = engine.EngineConfig(**C_GH, coarse_dtype="float64").mesh_bytes()
    assert f32["coarse_force_resident"] / GB == pytest.approx(12.0, abs=0.05), (
        f"{f32['coarse_force_resident'] / GB:.2f} GiB against the 12.9 GB "
        "(= 12.0 GiB) budget; the f32 coarse residency no longer matches it"
    )
    assert f64["coarse_force_resident"] == 2 * f32["coarse_force_resident"]


def test_the_kernel_build_does_not_shrink_with_the_knob():
    """`split_kernels` builds at f64 whatever it returns, so the build is its own accounting
    line; pricing an f32 arm off the kernel term alone would over-predict the saving."""
    f32 = engine.EngineConfig(**C_GH, coarse_dtype="float32").mesh_bytes()
    f64 = engine.EngineConfig(**C_GH, coarse_dtype="float64").mesh_bytes()
    assert f32["coarse_kernel_build_f64"] == f64["coarse_kernel_build_f64"]
    # the factorized solve's terms, all of which carry the complex width
    for k in ("coarse_spectrum", "coarse_solve_work", "coarse_kernel_slab",
              "coarse_device_planes"):
        assert f32[k] * 2 == f64[k], f"{k} did not follow the coarse knob"


def test_the_coarse_knob_does_not_move_the_fine_terms_or_the_accumulator():
    """Each dtype knob moves only its own terms in the memory accounting."""
    a = engine.EngineConfig(**C_GH, coarse_dtype="float32", fine_dtype="float64").mesh_bytes()
    b = engine.EngineConfig(**C_GH, coarse_dtype="float64", fine_dtype="float64").mesh_bytes()
    for k in ("tile_kernels", "tile_workspace", "coarse_accumulator", "coarse_decode_slab",
              "coarse_kernel_build_f64"):
        assert a[k] == b[k], f"{k} moved with the COARSE knob"
    for k in ("coarse_delta", "coarse_force_resident", "coarse_spectrum",
              "coarse_solve_work", "coarse_kernel_slab", "coarse_device_planes"):
        assert a[k] * 2 == b[k], f"{k} did not halve with the coarse knob"

    c = engine.EngineConfig(**C_GH, coarse_dtype="float64", fine_dtype="float32").mesh_bytes()
    assert c["tile_kernels"] * 2 == b["tile_kernels"]
    assert c["coarse_force_resident"] == b["coarse_force_resident"]


def test_the_slabbed_decode_transient_is_negligible_at_c_gh():
    """The slabbed decode transient is < 0.5 GiB at C-gh, > 50x under a whole-array decode
    (an int32 copy plus two full f64 meshes on top of the int64 accumulator)."""
    m = engine.EngineConfig(**C_GH, coarse_dtype="float32").mesh_bytes()
    old_transient = m["coarse_accumulator"] + 4 * 1024**3 + 2 * 8 * 1024**3
    assert m["coarse_decode_slab"] / GB < 0.5
    assert old_transient / m["coarse_decode_slab"] > 50


# ====================================================== the two promotion traps


def test_narrowing_the_kernel_build_alone_does_not_narrow_the_kernel():
    """Trap 1: `kernel_grids(fdtype=f32)` returns c64/f32, but the prefactor from f64
    `k2_true` makes `(fac / k2_safe) * ikx` c128, which would re-promote an f32 delta at the
    kernel multiply. Pins why `split_kernels` narrows the real prefactor."""
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


def test_a_gather_accumulator_would_be_defeated_by_an_f64_weight():
    """Trap 2: `f32_acc + f64_w * f32_field -> f64` still holds, so the gathers' weight cast
    is load-bearing; with it, an f32 sub-block gathers to f32."""
    import jax.numpy as jnp

    _, sub, owned, _, origin, n_coarse, cell_c = _owned_subblock(50, fdtype=jnp.float32)
    assert _name(sub[0]) == "float32", "staging is a slice and must not change dtype"

    # the raw promotion the cast exists for
    acc32 = jnp.zeros((4,), dtype=jnp.float32)
    w64 = jnp.ones((4,), dtype=jnp.float64)
    fld32 = jnp.ones((4,), dtype=jnp.float32)
    assert _name(acc32 + w64 * fld32) == "float64", (
        "f32_acc + f64_weight * f32_field no longer promotes; the weight cast in the four "
        "gathers may be removable, but check every one before touching it"
    )

    got = forces.gather_coarse_subblock(*sub, owned, origin, cell_c, n_coarse, assign="tsc")
    assert _name(got) == "float32", (
        "an f32 sub-block gathered back to f64 -- the gather's `ww.astype(dt)` is missing or has "
        "been removed, and an f32 arm built on this would silently gather in double"
    )
