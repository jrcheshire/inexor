"""Two-level (long/short split) force: kernels, tile paint/gather, coarse sub-block staging,
and the integer (order-independent) paints.

Tiling and staging are memory decisions and must not move a number, so parity is asserted
bitwise, not to a tolerance. Every bitwise comparison goes through `_agree`, which first
asserts the oracle carries signal: equality on all-zero or constant arrays passes vacuously
(e.g. `r_s=None` makes the short kernel identically zero).
"""


import numpy as np
import pytest


from inexor import forces, painting  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    """Run x64: the kernels are built in f64, and an f32 comparison would differ for
    reasons unrelated to the property under test."""
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


def test_tile_origin_may_be_negative_and_the_wrap_still_works():
    """The buffer of tile (0,0,0) hangs off the low edge, so its origin is
    negative by construction -- that is the case `mod` is relied on to handle,
    and a "fix" that clamped it would silently drop the wrapped buffer."""
    cell = L_BOX / N_MESH
    origin, extent = forces.tile_origin_extent((0, 0, 0), 8, 4, cell)
    assert np.all(origin < 0)
    # a particle just inside the far edge must land in the low tile's buffer
    u = np.asarray(forces.tile_local_coords(np.array([[L_BOX - 0.1] * 3]), origin, L_BOX))
    assert np.all(u < extent), "the wrapped buffer particle was not captured"


def _membership(pos, n_fine, n_tile, b_fine):
    """Brute-force tile membership: every particle inside a tile's extent plus buffer."""
    cell = L_BOX / n_fine
    _, b_real = forces.padded_size(n_tile, b_fine, n_fine=n_fine)
    n_side = n_fine // n_tile
    tiles = [(i, j, k) for i in range(n_side) for j in range(n_side) for k in range(n_side)]

    def member_fn(tijk):
        origin, extent = forces.tile_origin_extent(tijk, n_tile, b_real, cell)
        u = np.asarray(forces.tile_local_coords(pos, origin, L_BOX))
        return np.flatnonzero(np.all(u < extent, axis=1))

    cap = forces.tile_capacity(len(member_fn(t)) for t in tiles)
    return member_fn, cap


def _agree(mine, theirs, what, min_range=1e-6, min_nonzero_frac=0.5):
    """Bitwise equality, after asserting the oracle (`theirs`) carries signal.

    `min_range` bounds the oracle's peak |value| and `min_nonzero_frac` its nonzero
    fraction, so an all-zero or constant oracle fails instead of passing trivially. The
    fraction is per call site because tile-local quantities are legitimately mostly zero
    (out-of-box gather rows, empty cells); lower it with a reason, never to 0.
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


def test_k2_true_keeps_a_genuine_dc_zero_and_k2_safe_does_not():
    """The distinction the two returns exist for: S built from k2_safe would
    give S(0) = exp(-r_s^2) != 1 and break the split at DC silently."""
    _, _, _, k2_true, k2_safe = forces.kernel_grids((8,) * 3, 1.0)
    assert k2_true[0, 0, 0] == 0.0
    assert k2_safe[0, 0, 0] == 1.0
    assert forces.s_of_k(k2_true, R_S)[0, 0, 0] == 1.0


def test_long_plus_short_is_exactly_mono():
    """long + short == mono bitwise, which is why `split_factor` keeps its `1.0 - S` form;
    an -expm1 rewrite would need a tolerance and make the split error unattributable."""
    _, _, _, k2_true, _ = forces.kernel_grids((12,) * 3, 0.5)
    s_long = forces.split_factor(k2_true, R_S, "long")
    s_short = forces.split_factor(k2_true, R_S, "short")
    mono = forces.split_factor(k2_true, R_S, "mono")
    assert np.array_equal(s_long + s_short, mono), "long + short != mono BITWISE"
    assert float(np.max(s_short)) > 0.9, "fixture does not reach the short-dominated regime"


def test_windowed_families_are_not_reachable_from_the_package():
    """Only the gaussian split family is exposed; the windowed families measured ~10x
    worse in the coarse arm and are not reachable from `split_kernels`."""
    import inspect

    assert "family" not in inspect.signature(forces.split_kernels).parameters


# ------------------------------------------------------------------- the paint


def test_tsc_weights_are_a_partition_of_unity():
    """The three per-axis TSC weights sum to 1 (to 1e-15), which makes the paint
    mass-conserving."""
    import jax.numpy as jnp

    pos = jnp.asarray(_positions(4))
    _, w = painting._tsc_pieces(pos, L_BOX / N_MESH)
    total = np.asarray(w[0] + w[1] + w[2])
    assert np.max(np.abs(total - 1.0)) < 1e-15, f"max |sum w - 1| = {np.max(np.abs(total - 1.0)):e}"


# ------------------------------------------------------------- the global arm


def test_force_global_long_plus_short_recovers_mono_at_the_f1_floor():
    """long + short == mono to 1e-12 relative end to end (paint, solve, gather), not
    only at the kernel."""
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


def test_the_vacuity_guard_rejects_an_all_zero_oracle():
    """`_agree` refuses an all-zero oracle even when both arrays are equal."""
    zeros = np.zeros((64,))
    with pytest.raises(AssertionError, match="vacuous"):
        _agree(zeros, zeros, "all-zero")


# =========================================================== the tile geometry


def test_padded_size_refuses_a_degenerate_tile():
    """A padded tile at least as big as the box is refused: it does more FFT work than
    the monolithic solve, and with the brick wrap it double-counts particles."""
    with pytest.raises(ValueError, match="degenerate"):
        forces.padded_size(64, 40, n_fine=64)


# ------------------------------------------------------- the tile paint/gather


# Tile fixture geometry. A padded tile is at least 32 (FFT_FRIENDLY starts there), and P
# must be strictly less than n_fine: otherwise the padded box covers the whole volume, no
# particle is ever outside a tile, and the `ok` mask and n_out contract go untested.
# n_fine=64 with T=16, b=8 gives P=32 and 4^3 = 64 tiles, each an eighth of the box per side.
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


def test_tile_paint_conserves_mass_over_the_in_box_rows():
    """Painted mass equals the in-box particle count (rel 1e-12). A parity check against
    a twin with the same weight bug could not show a mass loss."""
    u, live, shape, cell = _tile_fixture(13)
    mean = N_PART_T**3 / float(N_FINE_T) ** 3
    mesh, n_out = forces.tile_paint_f64(u, live, shape, cell, mean)
    n_in = int(np.asarray(live).sum()) - int(n_out)
    assert float(np.asarray(mesh).sum()) * mean == pytest.approx(n_in, rel=1e-12)


# ---------------------------------------------------------- the tiled arm


@pytest.mark.detflag
def test_the_two_padding_fills_agree_bitwise():
    """pad_fill='cycle' and 'zero' give bitwise-identical forces: the fills differ only in
    which addresses the zero-weight scatter-adds hit, so a difference means a pad row counts.
    """
    n_tile, b_fine = N_TILE_T, B_FINE_T
    pos = _positions(15, N_PART_T)
    member_fn, cap = _membership(pos, N_FINE_T, n_tile, b_fine)
    args = (pos, N_FINE_T, L_BOX, N_PART_T**3, n_tile, b_fine, member_fn, cap)
    a, _ = forces.force_short_tiled(*args, r_s=R_S, pad_fill="cycle")
    b, _ = forces.force_short_tiled(*args, r_s=R_S, pad_fill="zero")
    _agree(a, b, "pad_fill cycle vs zero")


def test_one_tile_equals_the_whole_box_identity():
    """One tile covering the box with no buffer reproduces the global short solve (rel
    < 1e-13). A paint that drops the last cell layer fails this at ~4.6e-1."""
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
    """The accumulate sink materializes the global (n, 3) force (206 GB at C-gh), so it
    refuses above `max_accumulate_bytes` rather than becoming the production path."""
    pos = _positions(17, N_PART_T)
    member_fn, cap = _membership(pos, N_FINE_T, N_TILE_T, B_FINE_T)
    with pytest.raises(ValueError, match="accumulate sink would allocate"):
        forces.force_short_tiled(
            pos, N_FINE_T, L_BOX, N_PART_T**3, N_TILE_T, B_FINE_T, member_fn, cap,
            r_s=R_S, max_accumulate_bytes=1024,
        )


@pytest.mark.detflag
def test_the_tile_local_sink_sees_every_particle_exactly_once():
    """The production path: ownership is a partition, so a tile-local sink receives each
    particle exactly once and reconstructs the accumulate result bitwise."""
    n_tile, b_fine = N_TILE_T, B_FINE_T
    pos = _positions(18, N_PART_T)
    member_fn, cap = _membership(pos, N_FINE_T, n_tile, b_fine)
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


# --------------------------------------------------------- pinned failure modes


def test_regression_brick_span_refuses_the_wrapping_double_count():
    """Membership walks bricks by modular index, so span > nb visits a brick twice and
    paints its particles twice (n_fine=64, n_tile=32, b=20: span 6 vs nb 4, a 3.29
    relative short-force error). `brick_span` refuses it."""
    from inexor.layout import brick_span

    with pytest.raises(ValueError, match="wraps the box and would double-count"):
        brick_span(32, 20, 16, 4)


def test_regression_the_last_cell_layer_of_the_padded_box_is_painted():
    """A particle in the last cell layer of the padded box carries its full weight and
    is not counted outside. Dropping that layer gives 4.6e-1 on the tile identity and a
    buffer plateau that mimics kernel ringing."""
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
    """Masked rows keep their real `flat` index; only the weight is zeroed. Sending them
    all to index 0 makes ~2e6 zero-weight f64 atomics per tile contend for one address
    (2.2x on the device phase), a cost no correctness check sees."""
    import jax.numpy as jnp

    base = jnp.asarray([[5, 6, 7], [5, 6, 7]], dtype=jnp.int32)
    frac = jnp.asarray([[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]])
    ok = jnp.asarray([True, False])
    flat, w = forces._tile_corner(base, frac, 1.0 - frac, (0, 0, 0), (16, 16, 16), ok)
    assert int(flat[0]) == int(flat[1]) != 0, (
        "the masked row's index was rewritten -- that is the 2.2x contention bug"
    )
    assert float(w[0]) > 0.0 and float(w[1]) == 0.0, "masking must be on the WEIGHT"


# ===================================================== coarse sub-block staging


def _coarse_setup(seed=30):
    """A global coarse force plus the geometry to slice it by tile."""
    import jax.numpy as jnp

    n_coarse = N_FINE_T // forces.COARSE_RATIO  # 16
    rng = np.random.default_rng(seed)
    g = [jnp.asarray(rng.normal(size=(n_coarse,) * 3)) for _ in range(3)]
    pos = _positions(seed + 1, N_PART_T)
    return g, pos, n_coarse, L_BOX / n_coarse


def test_the_staged_subblock_is_a_verbatim_periodic_slice():
    """The staged sub-block is exactly the global mesh at the wrapped indices, including
    a block straddling the low boundary."""
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


def test_staging_allocates_the_block_and_not_a_slab():
    """Staging allocates at block scale, not slab scale (tracemalloc).

    An axis-at-a-time gather materializes (extent, n, n) first: 2.21 GB per call for a
    9.20 MB block at c-hero, 12,288 calls a step, invisible to value tests. The bar is a
    quarter of the slab (6.3 MB here), not a multiple of the output, because a correct
    `np.ix_` gather peaks at ~4.6x its output (int64 indices broadcast over the output;
    measured 0.25 MB for 0.055 MB). The bar sits an order clear of both.
    """
    import tracemalloc

    n, extent = 256, 24
    g = np.zeros((n, n, n), dtype=np.float32)
    out_bytes = extent**3 * g.itemsize
    slab_bytes = extent * n * n * g.itemsize
    assert slab_bytes > 50 * out_bytes, "the test size stopped separating the two regimes"

    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        sub = forces.stage_coarse_subblock(g, [3, 3, 3], extent)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert sub.nbytes == out_bytes
    transient = peak - before
    assert transient < slab_bytes / 4, (
        f"staging allocated {transient / 1e6:.2f} MB for a {out_bytes / 1e6:.2f} MB "
        f"block -- slab-scale, and an axis-at-a-time gather is ~{slab_bytes / 1e6:.1f} MB"
    )


def test_staging_matches_an_axis_at_a_time_gather_bitwise():
    """Bitwise identical to an axis-at-a-time gather (the oracle), including blocks that
    straddle the periodic boundary and negative origins; a one-ulp bump must break it."""
    rng = np.random.default_rng(4)
    n, extent = 32, 7
    g = rng.standard_normal((n, n, n)).astype(np.float64)

    def axis_at_a_time(mesh, origin, ext):
        out = mesh
        for axis, o in enumerate(np.asarray(origin, dtype=np.int64)):
            idx = (np.arange(int(ext), dtype=np.int64) + int(o)) % mesh.shape[axis]
            out = np.take(out, idx, axis=axis)
        return out

    for origin in ([5, 5, 5], [-3, 30, 1], [n - 1, n - 1, n - 1], [-n - 2, 0, n + 4]):
        got = forces.stage_coarse_subblock(g, origin, extent)
        want = axis_at_a_time(g, origin, extent)
        assert np.array_equal(got, want), f"origin {origin} moved bits"

    # anti-vacuity: the comparison can fail
    bumped = g.copy()
    bumped[5, 5, 5] = np.nextafter(bumped[5, 5, 5], np.inf)
    assert bumped[5, 5, 5] != g[5, 5, 5], "the perturbation was a no-op"
    assert not np.array_equal(forces.stage_coarse_subblock(bumped, [5, 5, 5], extent),
                              axis_at_a_time(g, [5, 5, 5], extent))


def test_the_deferred_guard_is_the_same_guard():
    """`guard_out` defers the stencil check without changing it: same refusals, same
    acceptances, bitwise-identical forces."""
    import jax.numpy as jnp

    g, _, n_coarse, cell = _coarse_setup()
    origin, extent = forces.coarse_subblock_origin_extent(
        (0, 0, 0), N_TILE_T, n_coarse, N_FINE_T
    )
    sub = [jnp.asarray(forces.stage_coarse_subblock(c, origin, extent)) for c in g]

    # a position whose stencil sits inside the block, and one half a box away
    inside = np.full((1, 3), (origin[0] + extent // 2) * cell, dtype=np.float64)
    outside = inside + np.array([[L_BOX / 2, 0.0, 0.0]])

    def call(pos, defer):
        guard = [] if defer else None
        out = forces.gather_coarse_subblock(
            *sub, jnp.asarray(pos), origin, cell, n_coarse, assign="tsc",
            guard_out=guard,
        )
        if defer:
            forces.check_stencil_guard(guard)
        return out

    # accepts what the eager form accepts, and to the BIT
    eager_ok = call(inside, defer=False)
    defer_ok = call(inside, defer=True)
    assert np.array_equal(np.asarray(eager_ok), np.asarray(defer_ok)), (
        "deferring the guard moved the forces")

    # refuses what the eager form refuses, with the same message
    with pytest.raises(ValueError, match="reaches outside the staged sub-block"):
        call(outside, defer=False)
    with pytest.raises(ValueError, match="reaches outside the staged sub-block"):
        call(outside, defer=True)


def test_the_deferred_guard_stays_quiet_when_every_row_is_padding():
    """With every row dead the deferred guard raises nothing: dead rows go to sentinels
    that land below zero for the max and above it for the min."""
    import jax.numpy as jnp

    g, _, n_coarse, cell = _coarse_setup()
    origin, extent = forces.coarse_subblock_origin_extent(
        (0, 0, 0), N_TILE_T, n_coarse, N_FINE_T
    )
    sub = [jnp.asarray(forces.stage_coarse_subblock(c, origin, extent)) for c in g]
    # positions that WOULD violate, but every row is dead
    pos = np.zeros((4, 3), dtype=np.float64) + L_BOX / 2
    guard = []
    forces.gather_coarse_subblock(
        *sub, jnp.asarray(pos), origin, cell, n_coarse, assign="tsc",
        live=np.zeros(4, dtype=bool), guard_out=guard,
    )
    forces.check_stencil_guard(guard)  # must not raise


def test_the_subblock_is_far_smaller_than_the_global_mesh():
    """At C-gh size the sub-block is >1000x smaller than the global coarse mesh; a halo
    change that ate the saving fails here."""
    n_coarse = 1024  # C-gh
    _, extent = forces.coarse_subblock_origin_extent((0, 0, 0), 256, n_coarse, 4096)
    assert extent == 64 + 2 * forces.COARSE_HALO
    assert (n_coarse / extent) ** 3 > 1000


@pytest.mark.parametrize("assign", ["cic", "tsc"])
def test_gathering_from_the_subblock_is_bitwise_the_global_gather(assign):
    """Gathering from the staged sub-block is bitwise the global gather on owned rows,
    including tile 0, whose block straddles the periodic boundary."""
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
    """A row outside the tile core would silently read wrapped values from the far side
    of the block, so the gather raises instead."""
    import jax.numpy as jnp

    g, pos, n_coarse, cell_c = _coarse_setup()
    origin, extent = forces.coarse_subblock_origin_extent(
        (0, 0, 0), N_TILE_T, n_coarse, N_FINE_T
    )
    sub = [jnp.asarray(forces.stage_coarse_subblock(c, origin, extent)) for c in g]
    with pytest.raises(ValueError, match="reaches outside the staged sub-block"):
        forces.gather_coarse_subblock(*sub, jnp.asarray(pos), origin, cell_c, n_coarse)


def test_the_halo_is_wide_enough_for_tsc_rounding():
    """halo=2 covers TSC: its base comes from round(), not floor(), so a core-edge
    particle reaches one cell further than a CIC bound suggests. Rows at both core
    extremes must still fit."""
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


# ================================================================ paint_tsc_int


def test_paint_tsc_int_is_order_independent():
    """`paint_tsc_int` is bitwise invariant under particle permutation (integer addition
    is associative), unlike `paint_tsc_f64`. The brick-sorted layout reorders particles
    every step."""
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
    """Control: the f64 TSC paint is order-dependent on this fixture, so the integer test
    above discriminates. Visible even on CPU without atomics: a permutation moves 770 of
    4096 cells at 4.4e-16, from accumulation order alone.
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
    """TSC's cell-weight bound is 5.359375x occupancy, not CIC's implicit 1x. The default
    frac_bits passes; a frac_bits that overflows int32 raises."""
    assert painting.TSC_CELL_WEIGHT_BOUND == pytest.approx(5.359375)
    painting.check_tsc_paint_headroom(N_PART_T**3, 12)  # the default frac_bits
    with pytest.raises(ValueError, match="TSC int-paint headroom"):
        painting.check_tsc_paint_headroom(10**9, 16)


def test_the_tsc_headroom_bound_is_not_below_a_measured_worst_case():
    """The derived bound holds against a worst-case construction: every particle at one
    cell centre, each contributing 0.75^3 to that cell."""
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


# ============================================ tile_paint_int: the short arm's integer paint
#
# `tile_paint_f64` accumulates through order-dependent f64 `.at[].add`; the brick-sorted
# layout reorders particles every step, so the short arm needs an order-independent paint.
#
# Reassociation only shows with enough contributions per cell: 8 CIC weights summed into one
# cell are permutation-invariant bitwise, 64 are not (7.1e-15), 4096 are not (6.1e-12).
# `_positions` puts ~8 corner writes per occupied tile cell, so on it the f64 tile paint is
# order-invariant (0 of 32768 cells move) and a shuffle test passes for both paints. The
# short-arm tests therefore use a clump. The coarse arm needs none: TSC's 27 corners on a
# dense mesh already clear the threshold.


def _clustered_tile_fixture(n=4096, seed=50):
    """A tight clump inside one padded tile, dense enough that the f64 tile paint's order
    dependence is visible (see the section note). Real runs reach such densities
    (measured peak bucket populations of ~5900).
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
    """`tile_paint_int` is bitwise invariant under particle permutation on the clump."""
    import jax.numpy as jnp

    u, live, shape, cell, perm = _clustered_tile_fixture()
    a, _ = forces.tile_paint_int(u, live, shape, cell)
    b, _ = forces.tile_paint_int(jnp.asarray(np.asarray(u)[perm]), live, shape, cell)
    assert np.array_equal(np.asarray(a), np.asarray(b)), (
        "the integer tile paint is order-DEPENDENT, which defeats its only purpose"
    )
    assert int(np.asarray(a).max()) > 0, "degenerate fixture: nothing was painted"


def test_the_f64_tile_paint_really_is_order_dependent_on_this_fixture():
    """Control: on the clump the f64 tile paint moves 23 of 27 occupied cells at 1.5e-10
    under a permutation, so the integer test above discriminates."""
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
    """Pins that the f64 tile paint is order-invariant on the ordinary `_tile_fixture`,
    which is why the clump fixture is needed. If this fails, the clump may be unnecessary.
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
    """Integer counterpart of the tile mass test: after per-corner rounding the 8 weights
    sum to 1 +- 8 * 2^-(frac_bits+1), so mass is conserved to that bound."""
    u, live, shape, cell, _ = _clustered_tile_fixture(seed=55)
    mesh, n_out = forces.tile_paint_int(u, live, shape, cell)
    n_in = int(np.asarray(live).sum()) - int(n_out)
    total = float(np.asarray(mesh).sum()) / 2.0**12
    assert total == pytest.approx(n_in, rel=8 * 2.0**-13)


def test_tile_paint_headroom_uses_the_strict_cic_bound_and_refuses_an_overflow():
    """The tile headroom check uses CIC's strict factor 8, not the default 1 of
    `check_int_paint_headroom`: the default frac_bits passes, a real overflow raises."""
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
    """`paint='int'` is wired through `force_short_tiled`, the default stays 'f64', and the
    arms differ by more than 0 and less than 2% of peak."""
    pos = _positions(56, N_PART_T)
    member_fn, cap = _membership(pos, N_FINE_T, N_TILE_T, B_FINE_T)
    args = (pos, N_FINE_T, L_BOX, N_PART_T**3, N_TILE_T, B_FINE_T, member_fn, cap)
    g_f64, d_f64 = forces.force_short_tiled(*args, r_s=R_S)
    g_int, d_int = forces.force_short_tiled(*args, r_s=R_S, paint="int")
    assert d_f64["paint"] == "f64", "the paint= default moved off 'f64'"
    assert d_int["paint"] == "int"
    err = float(np.max(np.abs(g_int - g_f64)))
    peak = float(np.max(np.abs(g_f64)))
    assert err > 0.0, "the int arm reproduced the f64 arm exactly, so it is not quantizing"
    assert err < 0.02 * peak, f"int-vs-f64 short force differs by {err / peak:.1%} of peak"


def test_force_global_can_reach_the_int_tsc_paint_and_refuses_int_cic():
    """`force_global(assign='tsc', paint='int')` reaches the integer TSC paint, the default
    stays 'f64', and `paint='int'` with CIC is refused."""
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
    """A masked gather reproduces the unmasked one bitwise and padded rows gather zero.

    The engine pads every tile's rows to a fixed capacity so one XLA shape serves all tiles
    (per-tile shapes cost 24.1 s of a 32.7 s step in recompiles); this is what admits that.
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
    """Pins the bitwise contract that jitting the sub-block gather would break.

    Under `jax.jit` XLA reassociates the corner accumulation and the gather stops matching
    the global gather bitwise (86 of 189 elements at 2.220e-16), so it runs eager despite
    being the largest step term (4.9 s of 11.4 s; jit would give ~1.7x). No jit is run here.
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
