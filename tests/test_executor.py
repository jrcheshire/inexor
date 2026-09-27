"""The pool executor is bitwise the serial loop.

Tile writes are disjoint and any value a worker reads while the parent applies another tile's
writes is discarded; these tests check that at the smoke config across what the pool adds (shm
adoption, the per-step header, repack copy-back, worker-side arena decode, the pooled migrate),
plus the shared-memory budget checks and backing-store (memfd/posix) selection.
"""

import os

import numpy as np
import pytest

from inexor import engine, state
from inexor.codec import T9Layout
from inexor.config import Cosmology
from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

# smallest geometry whose tile+buffer decomposition is not degenerate (as in test_engine.py)
L_BOX, N_PART, N_FINE, N_COARSE, N_TILE, B_FINE = 32.0, 32, 64, 16, 16, 8

FIELDS = ("off", "w", "occupancy", "brick_start", "vel_scale", "arena_bucket")


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _cfg(**kw):
    return engine.EngineConfig(
        box_size=L_BOX, n_part=N_PART, n_fine=N_FINE, n_coarse=N_COARSE,
        n_tile=N_TILE, b_fine=B_FINE, **kw
    )


def _positions(seed=0):
    rng = np.random.default_rng(seed)
    g = (np.arange(N_PART) + 0.5) * (L_BOX / N_PART)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    return np.mod(q + rng.normal(scale=0.25 * L_BOX / N_PART, size=q.shape), L_BOX)


def _run(tile_workers, seed=3, k=3, arena_frac=0.05, build_slack=0.10, **cfg_kw):
    cfg = _cfg(tile_workers=tile_workers, **cfg_kw)
    x = _positions(seed)
    v = np.random.default_rng(seed + 1).normal(scale=0.5, size=x.shape)
    t9 = T9Layout(box_size=L_BOX, n_part=N_PART, bucket_cells=2)
    st = state.SlotState.build(
        x, v, t9, N_FINE // cfg.n_brick, brick_slack=build_slack,
        arena_frac=arena_frac, with_ids=True,
    )
    a = a_grid(0.1, 1.0, k, "log")
    co = bullfrog_float_coeffs(bullfrog_table(a, Cosmology()))
    out = engine.run(st, cfg, co)
    return st, out


def _assert_states_identical(s1, s2):
    for f in FIELDS:
        a, b = np.asarray(getattr(s1, f)), np.asarray(getattr(s2, f))
        n = int((a != b).sum())
        assert n == 0, f"{f}: {n} of {a.size} elements differ between executors"
    assert s1.arena_base == s2.arena_base
    assert s1.ids is not None and s2.ids is not None, "ids must be in the comparison"
    assert int((s1.ids != s2.ids).sum()) == 0, "ids diverged between executors"


def test_the_pool_executor_is_bitwise_the_serial_loop():
    """W=2 equals serial over a K=3 run with a repack every step (copy-back at every boundary)."""
    s1, out1 = _run(1)
    s2, out2 = _run(2)
    _assert_states_identical(s1, s2)
    # same cap ladder: the pooled arm may not key different XLA shapes
    assert [o["cap"] for o in out1] == [o["cap"] for o in out2]
    assert out1[-1]["tile_workers"] == 1 and out2[-1]["tile_workers"] == 2
    assert "pool" in out2[-1] and "pool" not in out1[-1]
    assert out2[-1]["pool"]["workers"] == 2


def test_the_pool_executor_identity_with_a_resident_arena():
    """Pool identity with arena residents live across every step boundary.

    Zero build slack spills into the arena on the first migrate (a larger drift only moves
    particles between bricks), and `repack_every` > K keeps residents in place."""
    kw = dict(build_slack=0.0, arena_frac=0.20, repack_every=4)
    s1, _ = _run(1, seed=5, **kw)
    assert (np.asarray(s1.arena_bucket) >= 0).any(), (
        "the serial arm never populated the arena: this leg is not exercising "
        "worker-side arena decode and needs a different configuration"
    )
    s2, _ = _run(2, seed=5, **kw)
    _assert_states_identical(s1, s2)


def test_the_pooled_coarse_paint_is_bitwise_the_serial_one():
    """The pooled coarse paint gives exactly the serial mesh (integer adds are associative in
    any arrival order), compared on the decoded float delta the solver consumes."""
    from inexor.executor import TilePool

    cfg = _cfg(tile_workers=2)
    x = _positions(11)
    v = np.random.default_rng(12).normal(scale=0.5, size=x.shape)
    t9 = T9Layout(box_size=L_BOX, n_part=N_PART, bucket_cells=2)
    st = state.SlotState.build(x, v, t9, N_FINE // cfg.n_brick, arena_frac=0.05)
    stats_ser, stats_pool = {}, {}
    ref = engine.coarse_delta_streamed(st, cfg, stats=stats_ser)
    pool = TilePool(st, cfg)
    try:
        got = engine.coarse_delta_streamed(st, cfg, stats=stats_pool, pool=pool)
    finally:
        pool.close()
    n = int((np.asarray(ref) != np.asarray(got)).sum())
    assert n == 0, f"{n} of {ref.size} coarse cells differ between executors"
    assert stats_ser["coarse_pooled_workers"] == 0
    assert stats_pool["coarse_pooled_workers"] == 2
    assert stats_pool["coarse_subblock_chunks"] == stats_ser["coarse_subblock_chunks"] > 0
    # census always routes serial, so it never reads a pooled mesh
    pool2 = TilePool(st, cfg)
    stats_census = {}
    try:
        engine.coarse_delta_streamed(st, cfg, stats=stats_census, census=True, pool=pool2)
    finally:
        pool2.close()
    assert stats_census["coarse_pooled_workers"] == 0


def test_the_pool_survives_and_restores_state_ownership():
    """After a pooled run the state is back in ordinary memory and usable: repack exercises the
    rebound arrays and check() raises on corruption."""
    s2, _ = _run(2, seed=7, k=2)
    s2.repack(brick_slack=0.10)
    s2.check()


# --------------------------------------- the pooled migrate: drift_and_migrate_pooled vs serial

C_DRIFT = 1.0  # sized so brick_reach == 1 at this geometry (asserted in-test)


def _mig_state(seed=3, build_slack=0.10, arena_frac=0.05, with_ids=True):
    cfg = _cfg(tile_workers=2)
    x = _positions(seed)
    v = np.random.default_rng(seed + 1).normal(scale=0.5, size=x.shape)
    t9 = T9Layout(box_size=L_BOX, n_part=N_PART, bucket_cells=2)
    st = state.SlotState.build(
        x, v, t9, N_FINE // cfg.n_brick, brick_slack=build_slack,
        arena_frac=arena_frac, with_ids=with_ids,
    )
    return st, cfg


def _pooled_vs_serial(st_pool, st_ser, cfg, kernel, window=6):
    """Run the two arms and return their stats dicts."""
    from inexor.executor import TilePool

    pool = TilePool(st_pool, cfg)
    try:
        out_p = state.drift_and_migrate_pooled(st_pool, C_DRIFT, pool, kernel=kernel, window=window)
    finally:
        pool.close()
    out_s = state.drift_and_migrate(st_ser, C_DRIFT, kernel=kernel)
    return out_p, out_s


@pytest.mark.parametrize("kernel", ["numpy", "jax"])
def test_the_pooled_migrate_is_bitwise_the_serial_one(kernel):
    """At W=2 every state array (ids included) and the full stats dict match serial.

    The window (6) is below nb (8) so slot reuse is exercised; reach 1 means the reach window
    does not cover every slab, and particles are asserted to have crossed bricks."""
    import copy

    st_p, cfg = _mig_state()
    st_s = copy.deepcopy(st_p)
    out_p, out_s = _pooled_vs_serial(st_p, st_s, cfg, kernel)
    mp = out_p.pop("migrate_pool")
    assert mp["workers"] == 2 and mp["window"] == 6
    # eject_jax calls summed over workers: nb on jax, 0 on numpy
    want_calls = st_s.bricks_per_side if kernel == "jax" else 0
    assert mp["eject_jax_calls"] == want_calls
    assert out_p == out_s
    assert out_s["brick_reach"] == 1, "C_DRIFT no longer gives reach 1 here"
    assert 2 * out_s["brick_reach"] + 1 < st_s.bricks_per_side
    assert out_s["brick_reach_realized"] >= 1, "no particle crossed a brick"
    _assert_states_identical(st_s, st_p)


def test_pooled_migrate_window_floor_refuses():
    """A window below the deadlock floor refuses before any task is dispatched."""
    import copy

    st_p, cfg = _mig_state()
    st_s = copy.deepcopy(st_p)
    with pytest.raises(ValueError, match="deadlock floor"):
        # the check precedes any pool use, so pool=None suffices
        state.drift_and_migrate_pooled(st_p, C_DRIFT, pool=None, window=2)
    del st_s


@pytest.mark.parametrize("kernel", ["numpy", "jax"])
def test_the_pooled_migrate_identity_with_a_resident_arena(kernel):
    """Pooled migrate identity through both arena replays, each guarded non-vacuous.

    A zero-slack build primed by one serial pass has arena residents, so the pooled pass
    replays releases of resident rows and claims for fresh spills."""
    import copy

    st_p, cfg = _mig_state(seed=5, build_slack=0.0, arena_frac=0.20)
    st_s = copy.deepcopy(st_p)
    state.drift_and_migrate(st_p, C_DRIFT, kernel=kernel)
    state.drift_and_migrate(st_s, C_DRIFT, kernel=kernel)
    assert (np.asarray(st_p.arena_bucket) >= 0).any(), (
        "the priming pass never populated the arena: the release replay is "
        "not exercised and this leg needs a different configuration"
    )
    out_p, out_s = _pooled_vs_serial(st_p, st_s, cfg, kernel)
    assert out_s["n_arena_overflow"] > 0, (
        "the pooled pass never spilled: the claim replay is not exercised"
    )
    mp = out_p.pop("migrate_pool")
    assert mp["spill_rows"] == out_s["n_arena_overflow"]
    assert out_p == out_s
    _assert_states_identical(st_s, st_p)


def test_the_pooled_migrate_without_ids():
    """Pooled migrate identity with with_ids=False; ids stay None on both arms."""
    import copy

    st_p, cfg = _mig_state(seed=7, with_ids=False)
    st_s = copy.deepcopy(st_p)
    out_p, out_s = _pooled_vs_serial(st_p, st_s, cfg, "numpy")
    out_p.pop("migrate_pool")
    assert out_p == out_s
    for f in FIELDS:
        a, b = np.asarray(getattr(st_s, f)), np.asarray(getattr(st_p, f))
        assert int((a != b).sum()) == 0, f"{f} diverged"
    assert st_p.ids is None and st_s.ids is None


def test_migrate_pooled_knob_refuses_without_a_pool():
    """Explicit migrate_pooled=True without a pool refuses at validate() (None means auto)."""
    with pytest.raises(ValueError, match="needs a pool"):
        _cfg(tile_workers=1, migrate_pooled=True).validate()


def test_migrate_pooled_auto_default_validates_without_a_pool():
    """The auto default validates at tile_workers=1."""
    assert _cfg(tile_workers=1).validate()


def test_migrate_pooled_auto_default_pools_and_is_bitwise_the_serial_arm():
    """The default pools wherever a pool exists, False forces serial, and the two are bitwise.

    Both directions are checked by per-step worker counts."""
    s_auto, out_auto = _run(2)
    s_ser, out_ser = _run(2, migrate_pooled=False)
    _assert_states_identical(s_ser, s_auto)
    assert all(o["migrate_pooled_workers"] == 2 for o in out_auto), (
        "the auto default did not pool at tile_workers=2 -- the flip is inert"
    )
    assert all(o["migrate_pooled_workers"] == 0 for o in out_ser), (
        "migrate_pooled=False did not force the serial path -- the A/B "
        "baseline arm is unreachable and every serial reference is void"
    )


def test_the_pooled_migrate_engine_run_is_bitwise_the_serial_one():
    """A K=3 engine run with migrate_pooled=True (repack every step, kick/migrate sharing one
    pool, ids copy-back) is bitwise the serial run, with worker counts checked both ways."""
    s1, out1 = _run(1)
    s2, out2 = _run(2, migrate_pooled=True)
    _assert_states_identical(s1, s2)
    assert [o["cap"] for o in out1] == [o["cap"] for o in out2]
    assert all(o["migrate_pooled_workers"] == 0 for o in out1)
    assert all(o["migrate_pooled_workers"] == 2 for o in out2), (
        "the pooled migrate did not apply on every step (a 0 here can also "
        "mean the reach fallback fired at this geometry)"
    )


def test_pooled_migrate_arena_full_refuses():
    """When the arena cannot absorb a spill, both arms raise the same ValueError (never clamp);
    the pooled raise comes from the parent's claim replay."""
    import copy

    from inexor.executor import TilePool

    st_p, cfg = _mig_state(seed=9, build_slack=0.0, arena_frac=0.002)
    st_s = copy.deepcopy(st_p)
    with pytest.raises(ValueError, match="does not clamp"):
        state.drift_and_migrate(st_s, C_DRIFT)
    pool = TilePool(st_p, cfg)
    try:
        with pytest.raises(ValueError, match="does not clamp"):
            state.drift_and_migrate_pooled(st_p, C_DRIFT, pool, window=6)
    finally:
        pool.close()


# --------------------------------------------------------------- shm budget
# `SharedMemory` sizes lazily: `create=True` succeeds for a segment the tmpfs cannot back and
# the process dies with SIGBUS on first touch. The budget check turns that into a refusal.


def test_the_shm_model_reproduces_what_the_pool_actually_allocates():
    """`shm_terms` (used by the planner) sums to exactly what `TilePool` allocates.

    The planner prices unbuilt configurations, so the two implementations must agree.
    """
    from inexor.executor import TilePool, shm_terms

    cfg = _cfg(tile_workers=2)
    x = _positions(0)
    v = np.random.default_rng(1).normal(scale=0.5, size=x.shape)
    t9 = T9Layout(box_size=L_BOX, n_part=N_PART, bucket_cells=2)
    st = state.SlotState.build(
        x, v, t9, N_FINE // cfg.n_brick, brick_slack=0.10,
        arena_frac=0.05, with_ids=False,
    )
    pool = TilePool(st, cfg)
    try:
        measured = pool._shm_demand
    finally:
        pool.close()

    modelled = shm_terms(
        n_rows=st.off.shape[0], index_bytes=st.occupancy.nbytes,
        n_arena=st.arena_bucket.shape[0], n_bricks=st.vel_scale.shape[0],
        n_coarse=cfg.n_coarse,
        coarse_itemsize=np.dtype(cfg.np_coarse_dtype).itemsize,
    )
    assert sum(modelled.values()) == sum(measured.values()), (
        f"model {sum(modelled.values())} B against the pool's actual "
        f"{sum(measured.values())} B\n  model:    {modelled}\n  measured: {measured}"
    )


def test_the_shm_check_refuses_a_demand_that_cannot_fit(tmp_path):
    """An oversized demand refuses, naming the terms (largest first) and the arena_frac lever."""
    from inexor.executor import check_shm_budget

    huge = {"w (n_rows,3) int16": 8 * 10**18, "vel_scale": 17}
    with pytest.raises(MemoryError) as e:
        check_shm_budget(huge, path=str(tmp_path))
    msg = str(e.value)
    assert "w (n_rows,3) int16" in msg
    assert "arena_frac" in msg, "the refusal must name the lever, not just the miss"
    # ordered by size
    assert msg.index("w (n_rows,3) int16") < msg.index("vel_scale")


def test_a_missing_tmpfs_reports_that_it_could_not_check(tmp_path):
    """A missing tmpfs reports capacity None (could not check), never a number.

    macOS has no /dev/shm, so local runs take this path.
    """
    from inexor.executor import check_shm_budget, shm_capacity

    assert shm_capacity(str(tmp_path / "nope")) == (None, None)
    demand, avail = check_shm_budget({"x": 8 * 10**18}, path=str(tmp_path / "nope"))
    assert demand == 8 * 10**18
    assert avail is None, "unknown capacity must be None, never a number"


def test_the_pilot_configuration_would_now_be_refused():
    """A 2048^3 geometry (slack 0.20, arena_frac 0.20, alloc_margin 0.10) demands 148.5 GB,
    above a 127.6 GB measured /dev/shm, so the budget check would refuse it.

    If a layout change brings the demand under that ceiling this test fails, prompting a re-read.
    """
    from inexor.executor import shm_terms

    n = 2048**3
    n_bricks = 128**3
    per_brick = n // n_bricks
    spare = -(-int(per_brick * 20) // 100)  # ceil(per_brick * 0.20)
    n_alloc = -(-(per_brick + spare) * n_bricks * 11 // 10)
    n_arena = n // 5
    terms = shm_terms(
        n_rows=n_alloc + n_arena, index_bytes=1024**3 * 4, n_arena=n_arena,
        n_bricks=n_bricks, n_coarse=1024, coarse_itemsize=4,
    )
    gg_shm = 249116032 * 1024 // 2  # measured /dev/shm capacity
    demand = sum(terms.values())
    assert demand / 1e9 == pytest.approx(148.5, abs=0.5)
    assert demand > gg_shm, (
        f"2048^3 geometry asks {demand / 1e9:.1f} GB of a {gg_shm / 1e9:.1f} GB tmpfs"
    )


# ------------------------------------------------------------ the backing store
# Shared arrays sit on `memfd` where the kernel has it, POSIX `/dev/shm` otherwise. memfd has
# no mount size cap. INEXOR_SHM_BACKEND lets one machine exercise both paths.

from inexor.executor import has_memfd  # noqa: E402

# has_memfd() probes the syscall rather than reading `os.memfd_create`, which some Python
# builds lack on kernels that support it; otherwise this list would silently drop memfd.
BACKENDS = ["posix"] + (["memfd"] if has_memfd() else [])


def _child_reads(handle, nbytes):
    """Run in a SPAWNED interpreter: no fork, no inherited descriptors."""
    from inexor.executor import open_segment

    seg = open_segment(handle)
    got = bytes(seg.buf[:8]), bytes(seg.buf[nbytes - 8:nbytes])
    seg.buf[8:16] = b"CHILDWRT"
    seg.close()
    return got


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_spawned_process_maps_the_same_pages(backend):
    """A spawned process sees the parent's writes and vice versa (one direction alone would pass
    on a private copy)."""
    import multiprocessing as mp

    from inexor.executor import create_segment

    n = 1 << 20
    seg = create_segment(n, "roundtrip", backend=backend)
    try:
        seg.buf[:8] = b"PARENT!!"
        seg.buf[n - 8:n] = b"THEBTAIL"
        with mp.get_context("spawn").Pool(1) as pool:
            head, tail = pool.apply(_child_reads, (seg.handle, n))
        assert head == b"PARENT!!", "child did not see the parent's write"
        assert tail == b"THEBTAIL", "child's mapping is short"
        assert bytes(seg.buf[8:16]) == b"CHILDWRT", (
            "parent did not see the child's write: this is a COPY, not a shared "
            "mapping, and every bitwise identity gate above would still pass"
        )
    finally:
        seg.close()
        seg.unlink()


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_backend_override_is_what_it_says(backend, monkeypatch):
    from inexor.executor import create_segment, shm_backend

    monkeypatch.setenv("INEXOR_SHM_BACKEND", backend)
    assert shm_backend() == backend
    seg = create_segment(1 << 16, "kind")
    try:
        assert seg.kind == backend
        assert seg.handle[0] == backend
    finally:
        seg.close()
        seg.unlink()


def test_a_bad_backend_override_refuses():
    from inexor.executor import shm_backend

    os.environ["INEXOR_SHM_BACKEND"] = "tmpfs"
    try:
        with pytest.raises(ValueError, match="expected 'memfd' or 'posix'"):
            shm_backend()
    finally:
        del os.environ["INEXOR_SHM_BACKEND"]


@pytest.mark.skipif(has_memfd(), reason="needs a machine without memfd")
def test_memfd_is_not_silently_downgraded():
    """Requesting memfd where it is unavailable refuses rather than falling back to the capped
    /dev/shm mount."""
    from inexor.executor import shm_backend

    os.environ["INEXOR_SHM_BACKEND"] = "memfd"
    try:
        with pytest.raises(ValueError, match="cannot create one"):
            shm_backend()
    finally:
        del os.environ["INEXOR_SHM_BACKEND"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_pool_is_bitwise_the_serial_loop_on_either_backend(backend, monkeypatch):
    """Pool (W=4) is bitwise serial on each backing store."""
    monkeypatch.setenv("INEXOR_SHM_BACKEND", backend)
    s_serial, _ = _run(1)
    s_pool, _ = _run(4)
    _assert_states_identical(s_serial, s_pool)


def test_memfd_is_detected_by_probe_not_by_attribute():
    """memfd is chosen by probing the syscall, not by `hasattr(os, "memfd_create")`.

    `os.memfd_create` depends on CPython's build sysroot and can be absent on a kernel that has
    the syscall; the interesting case is capability present, attribute absent.
    """
    from inexor import executor

    if not has_memfd():
        pytest.skip("no memfd here either way")
    if not hasattr(os, "memfd_create"):
        assert executor.shm_backend() == "memfd", (
            "has_memfd() is True but the backend chose posix, which is exactly "
            "the silent downgrade back onto the capped mount"
        )
    seg = executor.create_segment(1 << 16, "probe", backend="memfd")
    try:
        assert seg.kind == "memfd"
    finally:
        seg.close()
        seg.unlink()


def test_adoption_peak_is_not_the_total():
    """`adoption_peak` is the larger of the biggest adopted field and the fresh total, not the sum.

    Adopted fields release their private copies as they are copied, so only the in-flight
    duplicate and fresh segments are new.
    """
    from inexor.executor import adoption_peak

    terms = {"w": 78_346_000_000, "off": 39_173_000_000,
             "arena_bucket": 13_744_000_000, "occupancy": 4_295_000_000,
             "coarse force g0,g1,g2": 12_885_000_000}
    adopted = {"w", "off", "arena_bucket", "occupancy"}
    assert sum(terms.values()) == pytest.approx(148.4e9, rel=1e-3)
    assert adoption_peak(terms, adopted) == 78_346_000_000
    # nothing adopted: the total (the posix case)
    assert adoption_peak(terms, ()) == sum(terms.values())
    assert adoption_peak({"a": 5, "big_fresh": 100}, {"a"}) == 100


def test_the_memfd_preflight_checks_the_peak_not_the_demand(monkeypatch):
    """A demand far above RAM still passes when it is nearly all adopted."""
    from inexor import executor

    monkeypatch.setattr(executor, "available_ram", lambda: 100_000_000_000)
    terms = {"w": 78_000_000_000, "off": 39_000_000_000, "fresh": 1_000_000_000}
    # 118 GB of demand against 100 GB of RAM, but the peak is 78 GB
    demand, avail, what = executor.preflight_shared_memory(
        terms, backend="memfd", adopted={"w", "off"})
    assert demand == 118_000_000_000 and what == "MemAvailable"
    # and it still refuses when the PEAK genuinely does not fit
    with pytest.raises(MemoryError, match="peak ABOVE baseline"):
        executor.preflight_shared_memory(
            {"w": 120_000_000_000, "fresh": 1_000_000_000},
            backend="memfd", adopted={"w"})


def test_posix_is_still_checked_on_the_total(monkeypatch):
    """posix is checked on the total demand, since a tmpfs holds every segment at once.

    `shm_capacity` and `available_ram` are stubbed to the same 100 GB, because macOS has no
    /dev/shm or /proc/meminfo (either would report "could not check"). The same terms pass
    memfd (peak 78 GB) and are refused by posix (total 117 GB).
    """
    from inexor import executor

    monkeypatch.setattr(executor, "shm_capacity",
                        lambda path=None: (100_000_000_000, 100_000_000_000))
    monkeypatch.setattr(executor, "available_ram", lambda: 100_000_000_000)
    terms = {"w": 78_000_000_000, "off": 39_000_000_000}
    demand, avail, what = executor.preflight_shared_memory(
        dict(terms), backend="memfd", adopted={"w", "off"})
    assert (demand, avail, what) == (117_000_000_000, 100_000_000_000, "MemAvailable")
    with pytest.raises(MemoryError, match="does not fit /dev/shm"):
        executor.preflight_shared_memory(terms, backend="posix",
                                         adopted={"w", "off"})


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_preshared_state_is_adopted_without_a_second_copy(backend, monkeypatch):
    """A state already in shared memory is adopted, not copied: object identity is checked,
    since equality would pass on a copy, and only the coarse meshes are charged.
    """
    monkeypatch.setenv("INEXOR_SHM_BACKEND", backend)
    from inexor.executor import SharedAllocator, TilePool

    cfg = _cfg(tile_workers=2)
    x = _positions(0)
    v = np.random.default_rng(1).normal(scale=0.5, size=x.shape)
    t9 = T9Layout(box_size=L_BOX, n_part=N_PART, bucket_cells=2)
    st = state.SlotState.build(x, v, t9, N_FINE // cfg.n_brick, brick_slack=0.10,
                               arena_frac=0.05)
    alloc = SharedAllocator()
    try:
        for f in FIELDS:
            a = np.asarray(getattr(st, f))
            view = alloc.empty(a.shape, a.dtype, f)
            view[...] = a
            setattr(st, f, view)
        before = {f: id(np.asarray(getattr(st, f))) for f in FIELDS}
        pool = TilePool(st, cfg, allocator=alloc)
        try:
            after = {f: id(np.asarray(getattr(st, f))) for f in FIELDS}
            assert before == after, "the pool copied an array that was already shared"
            assert pool._preshared == set(FIELDS)
            assert set(pool._shm_demand) == {"coarse force g0,g1,g2"}
        finally:
            pool.close()
    finally:
        alloc.close()


def test_the_loader_fills_shared_memory_and_is_otherwise_unchanged(tmp_path):
    """Loading into shared memory gives the same state as a plain load, placed in shm."""
    from inexor import icgen
    from inexor.executor import SharedAllocator

    cfg = _cfg(tile_workers=1)
    x = _positions(2)
    v = np.random.default_rng(3).normal(scale=0.5, size=x.shape)
    t9 = T9Layout(box_size=L_BOX, n_part=N_PART, bucket_cells=2)
    st = state.SlotState.build(x, v, t9, N_FINE // cfg.n_brick, brick_slack=0.10,
                               arena_frac=0.05)
    icgen.write_t9_slabs(st, str(tmp_path))

    plain = icgen.load_slot_state(str(tmp_path), brick_slack=0.10, arena_frac=0.05)
    alloc = SharedAllocator()
    try:
        shared = icgen.load_slot_state(str(tmp_path), brick_slack=0.10,
                                       arena_frac=0.05, alloc=alloc)
        for f in FIELDS:
            a, b = np.asarray(getattr(plain, f)), np.asarray(getattr(shared, f))
            assert a.dtype == b.dtype and a.shape == b.shape, f
            assert int((a != b).sum()) == 0, f"{f} differs between the two loaders"
            assert alloc.segment_of(b) is not None, f"{f} is not in shared memory"
        assert alloc.segment_of(np.asarray(getattr(plain, "off"))) is None
        assert plain.arena_base == shared.arena_base
    finally:
        alloc.close()


def test_engine_run_carries_the_allocator_all_the_way_to_the_pool(tmp_path):
    """`engine.run(..., allocator=)` hands the allocator to the pool, so a loaded shared state
    is not shared a second time (the allocator does not grow during the run).

    The adoption unit test above builds the pool directly and cannot see this hand-off.
    """
    from inexor import icgen
    from inexor.executor import SharedAllocator
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

    cfg = _cfg(tile_workers=2)
    x = _positions(5)
    v = np.random.default_rng(6).normal(scale=0.5, size=x.shape)
    t9 = T9Layout(box_size=L_BOX, n_part=N_PART, bucket_cells=2)
    seed_st = state.SlotState.build(x, v, t9, N_FINE // cfg.n_brick,
                                    brick_slack=0.20, arena_frac=0.20)
    icgen.write_t9_slabs(seed_st, str(tmp_path))

    alloc = SharedAllocator()
    try:
        st = icgen.load_slot_state(str(tmp_path), brick_slack=0.20,
                                   arena_frac=0.20, alloc=alloc)
        held_after_load = alloc.bytes_held()
        a = a_grid(0.1, 1.0, 3, "log")
        co = bullfrog_float_coeffs(bullfrog_table(a, Cosmology()))
        engine.run(st, cfg, co, allocator=alloc)
        state_bytes = sum(np.asarray(getattr(st, f)).nbytes for f in FIELDS)
        assert alloc.bytes_held() == held_after_load, (
            "the allocator grew during the run: the pool re-shared the state, "
            "which doubles shared memory for the state"
        )
        assert held_after_load >= state_bytes * 0.99
    finally:
        alloc.close()


def test_bounding_ejects_in_flight_is_bitwise_the_unbounded_pass():
    """`eject_inflight` (None, 2, and the floor 1) does not change the migrated state.

    It only sets when a slab's eject is dispatched; arena bookkeeping runs in the parent's
    serial replay at fixed points.
    """
    import jax

    from inexor.executor import TilePool
    from inexor.plan import engine_config
    from inexor.state import SlotState, drift_and_migrate_pooled

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        n_side, box = 32, 32.0
        rng = np.random.default_rng(5)
        x = rng.random((n_side**3, 3)) * box
        v = rng.standard_normal((n_side**3, 3)) * 2.0
        t9 = T9Layout(box_size=box, n_part=n_side, bucket_cells=2)
        ec = engine_config("smoke", tile_workers=2)
        nb = 64 // ec.n_brick if 64 // ec.n_brick else 8

        out = {}
        for cap in (None, 2, 1):
            st = SlotState.build(x, v, t9, nb, brick_slack=0.2,
                                 alloc_margin=0.1, arena_frac=0.10)
            pool = TilePool(st, ec)
            try:
                drift_and_migrate_pooled(st, 0.02, pool, kernel="jax",
                                         eject_inflight=cap)
                out[cap] = (np.asarray(st.off).copy(), np.asarray(st.w).copy())
            finally:
                pool.close()
        for cap in (2, 1):
            assert np.array_equal(out[None][0], out[cap][0]), f"off differs at {cap}"
            assert np.array_equal(out[None][1], out[cap][1]), f"w differs at {cap}"
    finally:
        jax.config.update("jax_enable_x64", prev)
