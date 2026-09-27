"""BrickPackedLayout: spare slots pooled per brick, buckets packed with no holes.

Bucket boundaries are not stored; they are a prefix sum of `occupancy`. So the properties
pinned are: derived boundaries agree with where particles are, a brick's run has no holes (a
hole would shift every later bucket's span), particles are conserved, overflow raises rather
than clamping, the index and key dtypes refuse to wrap, and `check()` catches corruption.
"""


import numpy as np
import pytest


from inexor.codec import T9Layout  # noqa: E402
from inexor.layout import BrickPackedLayout  # noqa: E402

L_BOX = 128.0
N_PART = 32
# Load-bearing: 512 buckets and 4096 particles per brick, the production ratios (8 particles
# per bucket). With small bricks the brick_start cost, spare floor and brick-vs-bucket
# migration ratio all change, and the assertions below fail.
BRICKS_PER_SIDE = 2


def _t9(bucket_cells=2):
    return T9Layout(box_size=L_BOX, n_part=N_PART, bucket_cells=bucket_cells)


def _lattice(seed, jitter=0.35):
    rng = np.random.default_rng(seed)
    g = (np.arange(N_PART) + 0.5) * (L_BOX / N_PART)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    return np.mod(q + rng.normal(scale=jitter * L_BOX / N_PART, size=q.shape), L_BOX)


def _build(seed=0, **kw):
    t9 = _t9()
    x = _lattice(seed)
    return BrickPackedLayout.build(x, t9, BRICKS_PER_SIDE, **kw), x, t9


def test_bucket_boundaries_are_derived_not_stored():
    """`bucket_slot_starts` from occupancy alone: every particle in each derived span belongs
    to that bucket."""
    lay, _, _ = _build()
    lay.check()
    p3 = lay.buckets_per_brick
    for brick in (0, lay.n_bricks // 3, lay.n_bricks - 1):
        starts = lay.bucket_slot_starts(brick)
        assert starts[0] == lay.brick_slot_range(brick)[0]
        for i in range(p3):
            span = lay.slot_to_particle[starts[i] : starts[i + 1]]
            assert np.all(span >= 0), "hole inside a bucket's derived span"
            assert np.all(lay.key[span] == brick * p3 + i)


def test_no_per_bucket_boundary_array_exists():
    """No bucket-length boundary array (it would cost 1.00 B/p); only per-brick boundaries."""
    lay, _, _ = _build()
    assert not hasattr(lay, "bucket_start")
    assert lay.brick_start.size == lay.n_bricks + 1
    b = lay.bytes_per_particle()
    assert b["brick_start"] < 0.01, "brick boundaries should be negligible per particle"
    assert b["bucket_index"] == pytest.approx(0.50, rel=0.01)


def test_spare_is_the_requested_fraction_with_no_granularity_floor():
    """Pooled over ~4096 particles, the spare is the requested fraction (per-bucket ceil()
    would force >= 12.5% at 8 particles per bucket)."""
    lay, _, _ = _build(brick_slack=0.10)
    assert lay.n_slots / lay.n_particles == pytest.approx(1.10, abs=0.005)
    lay2, _, _ = _build(brick_slack=0.02)
    assert lay2.n_slots / lay2.n_particles == pytest.approx(1.02, abs=0.005)


def test_migration_conserves_particles_and_keeps_runs_packed():
    lay, x, t9 = _build(seed=1, brick_slack=0.50)
    rng = np.random.default_rng(2)
    for _ in range(4):
        x = np.mod(x + rng.normal(scale=0.3 * t9.spacing, size=x.shape), L_BOX)
        st = lay.migrate(x)
        lay.check()
    assert st["bucket_migrant_frac"] > 0.05


def test_brick_migration_is_far_below_bucket_migration():
    """A brick is ~512 buckets, so brick migration is well below bucket migration (not a
    capacity guarantee: a collapsing halo still outgrows a brick, hence repack)."""
    lay, x, t9 = _build(seed=3, brick_slack=0.50)
    rng = np.random.default_rng(4)
    x = np.mod(x + rng.normal(scale=0.3 * t9.spacing, size=x.shape), L_BOX)
    st = lay.migrate(x)
    assert st["brick_migrant_frac"] < st["bucket_migrant_frac"] / 4, (
        f"brick {st['brick_migrant_frac']:.1%} vs bucket "
        f"{st['bucket_migrant_frac']:.1%} -- check buckets_per_brick"
    )


def test_overflow_escalates_to_the_arena_then_refuses():
    """Brick overflow spills to the arena and the layout stays consistent; with no arena it
    raises, never clamps or drops. Overflow is a rare event (measured 461 of 2.1e6 particles
    at 20% spare), so a small arena absorbs it."""
    t9 = _t9()
    rng = np.random.default_rng(6)
    clump = np.mod(rng.normal(loc=L_BOX * 0.5, scale=L_BOX * 0.01, size=(N_PART**3, 3)), L_BOX)

    lay = BrickPackedLayout.build(
        _lattice(5), t9, BRICKS_PER_SIDE, brick_slack=0.0, arena_frac=1.0
    )
    st = lay.migrate(clump)
    lay.check()
    assert st["n_overflow"] > 0, "fixture did not overflow; it is not exercising the ladder"
    assert st["arena_used"] > 0

    lay2 = BrickPackedLayout.build(
        _lattice(5), t9, BRICKS_PER_SIDE, brick_slack=0.0, arena_frac=0.0
    )
    with pytest.raises(ValueError, match="overflow their brick"):
        lay2.migrate(clump)


def test_repack_redistributes_in_place_with_bounded_scratch():
    lay, x, t9 = _build(seed=7, brick_slack=0.50)
    rng = np.random.default_rng(8)
    for _ in range(3):
        x = np.mod(x + rng.normal(scale=0.3 * t9.spacing, size=x.shape), L_BOX)
        lay.migrate(x)
    r = lay.repack(brick_slack=0.10, chunk=1 << 12)
    lay.check()
    payload = lay.n_particles * 9.0
    assert r["scratch_bytes"] < 0.2 * payload, "scratch should be set by the chunk, not by N"
    assert r["slots_used"] <= r["slots_allocated"]


# -------------------------------------------------------------- the radix sort


@pytest.mark.parametrize(
    "name,make",
    [
        ("random", lambda r, m: r.integers(0, 2**30, m).astype(np.int32)),
        ("all identical", lambda r, m: np.zeros(m, np.int32)),
        ("already sorted", lambda r, m: np.sort(r.integers(0, 2**30, m).astype(np.int32))),
        ("reverse sorted",
         lambda r, m: np.sort(r.integers(0, 2**30, m).astype(np.int32))[::-1].copy()),
        ("few distinct", lambda r, m: r.integers(0, 512, m).astype(np.int32)),
        ("low digit zero", lambda r, m: (r.integers(0, 2**14, m) << 16).astype(np.int32)),
        ("high digit zero", lambda r, m: r.integers(0, 2**16, m).astype(np.int32)),
        ("int32 maximum", lambda r, m: np.full(m, 2**31 - 1, np.int32)),
        ("int64 keys", lambda r, m: r.integers(0, 2**30, m).astype(np.int64)),
    ],
)
def test_the_radix_sort_is_the_same_permutation_as_argsort(name, make):
    """The radix sort returns the identical permutation to a stable argsort, not merely a
    sorted one: within-bucket order sets slot placement and hence trajectory bits.
    Adversarial patterns probe the digit boundaries."""
    from inexor.layout import _stable_sort_index

    keys = make(np.random.default_rng(30), 20_000)
    assert np.array_equal(_stable_sort_index(keys), np.argsort(keys, kind="stable")), name


def test_the_radix_sort_refuses_keys_it_cannot_represent():
    """Keys outside [0, 2^32) raise (a negative key means the ordinal already wrapped)."""
    from inexor.layout import _stable_sort_index

    with pytest.raises(ValueError, match=r"\[0, 2\^32\)"):
        _stable_sort_index(np.array([-1, 0, 1], dtype=np.int64))
    with pytest.raises(ValueError, match=r"\[0, 2\^32\)"):
        _stable_sort_index(np.array([0, 2**32], dtype=np.int64))
    assert _stable_sort_index(np.empty(0, dtype=np.int32)).size == 0


def test_the_layout_is_unchanged_by_the_faster_sort():
    """After migrations, repack reproduces a fresh build at the same positions exactly:
    occupancy, slot map and particle map."""
    lay, x, t9 = _build(seed=31, brick_slack=0.50)
    rng = np.random.default_rng(32)
    for _ in range(3):
        x = np.mod(x + rng.normal(scale=0.3 * t9.spacing, size=x.shape), L_BOX)
        lay.migrate(x)
        lay.check()
    ref = BrickPackedLayout.build(x, t9, BRICKS_PER_SIDE, brick_slack=0.50)
    lay.repack(brick_slack=0.50)
    assert np.array_equal(lay.occupancy, ref.occupancy)
    assert np.array_equal(lay.slot_to_particle, ref.slot_to_particle)
    assert np.array_equal(lay.particle_to_slot, ref.particle_to_slot)


# ------------------------------------------------------- the index dtype ceiling


def test_index_defaults_to_uint32_and_costs_half_a_byte():
    """The bucket index defaults to uint32 (0.50 B/p): nothing cheap bounds the occupancy of
    a cell inside a halo."""
    lay, _, _ = _build()
    assert lay.index_dtype == np.uint32
    assert lay.bytes_per_particle()["bucket_index"] == pytest.approx(0.50, rel=0.01)


def test_uint16_index_is_still_reachable_and_reproduces_the_ratified_cost():
    """A uint16 index remains selectable, at 0.25 B/p."""
    lay, _, _ = _build(index_dtype=np.uint16)
    assert lay.index_dtype == np.uint16
    assert lay.bytes_per_particle()["bucket_index"] == pytest.approx(0.25, rel=0.01)
    lay.check()


def test_a_bare_narrowing_would_have_wrapped_silently():
    """A bare numpy narrowing wraps (65536 -> 0), which is what `_to_index` guards: occupancy
    is the boundary prefix sum, so a wrap relocates every later bucket in the brick. The
    helper must refuse exactly the counts numpy would wrap, and pass the ceiling unchanged."""
    from inexor.layout import _to_index

    counts = np.array([70000, 65536, 65535], dtype=np.int64)
    assert list(counts.astype(np.uint16)) == [4464, 0, 65535]
    for over in ([70000], [65536], [3, 65536, 7]):
        with pytest.raises(ValueError, match="exceeds the uint16 index ceiling"):
            _to_index(np.array(over, dtype=np.int64), np.uint16, "test")
    ok = _to_index(np.array([0, 65535, 12], dtype=np.int64), np.uint16, "test")
    assert ok.dtype == np.uint16 and list(ok) == [0, 65535, 12]


def test_build_refuses_an_index_overflow():
    """Build refuses an index overflow. uint8, because 32,768 particles cannot reach the
    uint16 ceiling and the test could not fail."""
    t9 = _t9()
    bad = np.uint8
    rng = np.random.default_rng(20)
    clump = np.mod(rng.normal(loc=L_BOX * 0.5, scale=t9.quantum, size=(N_PART**3, 3)), L_BOX)
    with pytest.raises(ValueError, match="initial bucket count .* exceeds"):
        BrickPackedLayout.build(clump, t9, BRICKS_PER_SIDE, index_dtype=bad)


def test_migrate_refuses_an_index_overflow():
    """Migrate (the every-step path) refuses an index overflow."""
    t9 = _t9()
    lay = BrickPackedLayout.build(
        _lattice(21), t9, BRICKS_PER_SIDE, brick_slack=1.0, arena_frac=1.0,
        index_dtype=np.uint8,
    )
    rng = np.random.default_rng(22)
    clump = np.mod(rng.normal(loc=L_BOX * 0.5, scale=t9.quantum, size=(N_PART**3, 3)), L_BOX)
    with pytest.raises(ValueError, match="migrated bucket count .* exceeds"):
        lay.migrate(clump)


def test_repack_refuses_an_index_overflow_migrate_cannot_see():
    """Repack refuses an overflow migrate cannot see: migrate counts only the brick run
    (arena spill excluded), repack counts arena residents too. Small bricks (capacity ~64)
    put the hot bucket under the ceiling in migrate and over it in repack.
    """
    t9 = _t9()
    many_bricks = 8
    lay = BrickPackedLayout.build(
        _lattice(23), t9, many_bricks, brick_slack=0.0, arena_frac=1.0,
        index_dtype=np.uint8,
    )
    rng = np.random.default_rng(24)
    clump = np.mod(rng.normal(loc=L_BOX * 0.5, scale=t9.quantum, size=(N_PART**3, 3)), L_BOX)
    st = lay.migrate(clump)
    assert st["arena_used"] > 0, "fixture did not park the excess in the arena"
    assert int(lay.occupancy.max()) <= np.iinfo(np.uint8).max
    with pytest.raises(ValueError, match="repacked bucket count .* exceeds"):
        lay.repack(brick_slack=0.10, chunk=1 << 12)


def test_build_refuses_a_bucket_grid_past_the_int32_key():
    """A 2048^3 bucket grid overflows the int32 key and is refused; 1024^3 fits and must not
    raise. Both directions, so the guard can neither always pass nor always fail."""
    from inexor.layout import _refuse_key_overflow

    _refuse_key_overflow(1024**3)  # C-gh: fits, must not raise
    with pytest.raises(ValueError, match="exceeds int32"):
        _refuse_key_overflow(2048**3)  # C-hero

    hero = T9Layout(box_size=L_BOX, n_part=4096, bucket_cells=2)
    with pytest.raises(ValueError, match="exceeds int32"):
        BrickPackedLayout.build(np.zeros((1, 3)), hero, 128)


def test_scaffolding_is_reported_beside_the_total_not_inside_it():
    """Scaffolding bytes (~21 B/p, not part of the shipped state) are reported beside the
    total, not inside it."""
    lay, _, _ = _build()
    b = lay.bytes_per_particle()
    assert b["scaffold"] > 2 * b["total"], "scaffolding should dwarf the state it indexes"
    assert b["total"] == pytest.approx(
        b["payload"] + b["bucket_index"] + b["brick_start"] + b["slack"]
    ), "scaffold must not be inside the total"


@pytest.mark.parametrize(
    "corrupt,match",
    [
        ("drop", "lost particles"),
        ("duplicate", "two slots"),
        ("hole", "hole inside its packed prefix"),
    ],
)
def test_check_catches_corruption(corrupt, match):
    lay, _, _ = _build(seed=9)
    lay.check()
    lo, _ = lay.brick_slot_range(0)
    if corrupt == "drop":
        lay.slot_to_particle[lo] = -1
    elif corrupt == "duplicate":
        lay.slot_to_particle[lo + 1] = lay.slot_to_particle[lo]
    elif corrupt == "hole":
        lay.slot_to_particle[lo] = -1
        lay.n_particles -= 1
    with pytest.raises(AssertionError, match=match):
        lay.check()


# --------------------------------------------------- membership for the force


# Own geometry: at 2 bricks per side tile+buffer wraps the brick grid, which brick_span refuses.
TM_N_PART, TM_N_FINE, TM_BRICKS = 64, 128, 8
TM_TILE, TM_BUF = 32, 16


def _tm_setup(seed=11):
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from inexor.codec import T9Layout, roundtrip_positions
    from inexor.layout import choose_brick

    t9 = T9Layout(box_size=L_BOX, n_part=TM_N_PART, bucket_cells=2)
    rng = np.random.default_rng(seed)
    g = (np.arange(TM_N_PART) + 0.5) * (L_BOX / TM_N_PART)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    x = np.mod(q + rng.normal(scale=0.35 * L_BOX / TM_N_PART, size=q.shape), L_BOX)
    xq = np.asarray(roundtrip_positions(jnp.asarray(x), t9))
    n_brick = choose_brick(TM_TILE, TM_BUF, TM_N_FINE)
    jax.config.update("jax_enable_x64", prev)
    return t9, x, xq, n_brick


def test_arena_residents_are_not_dropped_from_membership():
    """Arena residents belong to their brick's membership: with the arena populated, the union
    over all bricks holds every particle exactly once (omission is silent mass loss)."""
    t9, x, _, n_brick = _tm_setup(seed=12)
    lay = BrickPackedLayout.build(x, t9, TM_BRICKS, brick_slack=0.0, arena_frac=1.0)
    rng = np.random.default_rng(13)
    clump = np.mod(
        rng.normal(loc=L_BOX * 0.5, scale=L_BOX * 0.02, size=(TM_N_PART**3, 3)), L_BOX
    )
    st = lay.migrate(clump)
    lay.check()
    assert st["arena_used"] > 0, "fixture did not populate the arena"

    seen = np.concatenate([lay.brick_members(b) for b in range(lay.n_bricks)])
    assert len(seen) == lay.n_particles, (
        f"brick union holds {len(seen)} of {lay.n_particles} particles -- "
        f"{lay.n_particles - len(seen)} would vanish from the force"
    )
    assert len(np.unique(seen)) == lay.n_particles, "a particle is in two bricks"
