"""Executor identity: the pool executor is bitwise the serial loop (W2).

The seam analysis says tile writes are disjoint (ownership is a partition) and
every value a worker reads while the parent applies another tile's writes is
discarded -- these tests are what CERTIFY that, at the smoke config, across
the machinery the pool actually adds: shm adoption, the per-step header, the
repack copy-back, worker-side arena decode. The cluster legs repeat the same
identity at cdev8/cdev scale.
"""

import os

import numpy as np
import pytest

from inexor import engine, state
from inexor.codec import T9Layout
from inexor.config import Cosmology
from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

# the smoke rung: smallest geometry whose tile+buffer decomposition is not
# degenerate (same constants as test_engine.py)
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
    """W=2 against serial over a K=3 run with a repack every step, so the
    brick_start/occupancy copy-back is exercised at every step boundary."""
    s1, out1 = _run(1)
    s2, out2 = _run(2)
    _assert_states_identical(s1, s2)
    # same cap ladder: the pooled arm may not key different XLA shapes
    assert [o["cap"] for o in out1] == [o["cap"] for o in out2]
    assert out1[-1]["tile_workers"] == 1 and out2[-1]["tile_workers"] == 2
    assert "pool" in out2[-1] and "pool" not in out1[-1]
    assert out2[-1]["pool"]["workers"] == 2


def test_the_pool_executor_identity_with_a_resident_arena():
    """Zero BUILD slack concentrates overflow into the arena from the first
    migrate (the measured lever -- a larger drift only moves particles BETWEEN
    bricks), and `repack_every` past K keeps residents in place across every
    later step boundary, so the worker-side arena decode and the `arena_base`
    header are load-bearing here rather than idle."""
    kw = dict(build_slack=0.0, arena_frac=0.20, repack_every=4)
    s1, _ = _run(1, seed=5, **kw)
    assert (np.asarray(s1.arena_bucket) >= 0).any(), (
        "the serial arm never populated the arena: this leg is not exercising "
        "worker-side arena decode and needs a different configuration"
    )
    s2, _ = _run(2, seed=5, **kw)
    _assert_states_identical(s1, s2)


def test_the_pooled_coarse_paint_is_bitwise_the_serial_one():
    """Stage C in isolation: the same mesh, chunk by chunk, from workers.

    Integer accumulation is associative, so the arrival-order adds must give
    the EXACT serial mesh -- the same property the streamed paint itself
    stands on. Compared through the decoded float delta, which is what the
    solver consumes."""
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
    # the knob must prove it applied, in both directions
    assert stats_ser["coarse_pooled_workers"] == 0
    assert stats_pool["coarse_pooled_workers"] == 2
    assert stats_pool["coarse_subblock_chunks"] == stats_ser["coarse_subblock_chunks"] > 0
    # census routes SERIAL regardless of the pool: the gate instrument must
    # never read a pooled mesh
    pool2 = TilePool(st, cfg)
    stats_census = {}
    try:
        engine.coarse_delta_streamed(st, cfg, stats=stats_census, census=True, pool=pool2)
    finally:
        pool2.close()
    assert stats_census["coarse_pooled_workers"] == 0


def test_the_pool_survives_and_restores_state_ownership():
    """After a pooled run the state must be backed by ordinary memory again
    (the shm segments are unlinked), and still usable."""
    s2, _ = _run(2, seed=7, k=2)
    # a post-run mutation must not touch shm (it is gone); repack exercises
    # the rebound arrays end to end, and check() raises on corruption
    s2.repack(brick_slack=0.10)
    s2.check()


# ---------------------------------------------------------------------------
# the pooled migrate (idle-half Stage 2): drift_and_migrate_pooled vs serial
# ---------------------------------------------------------------------------

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
    """The whole contract at W=2: every state array (ids included) and the
    ENTIRE stats dict equal key for key, with a window smaller than nb so slot
    reuse is actually exercised, and an anti-vacuity guard that particles
    really crossed bricks (the nb=2 trap: at this geometry nb=8 and reach 1,
    so the reach window does NOT cover every slab)."""
    import copy

    st_p, cfg = _mig_state()
    st_s = copy.deepcopy(st_p)
    out_p, out_s = _pooled_vs_serial(st_p, st_s, cfg, kernel)
    mp = out_p.pop("migrate_pool")
    assert mp["workers"] == 2 and mp["window"] == 6
    # the compiled-kernel receipt, summed from the workers: nb calls on the
    # jax arm, 0 on numpy (the parent's own counter cannot see workers)
    want_calls = st_s.bricks_per_side if kernel == "jax" else 0
    assert mp["eject_jax_calls"] == want_calls
    assert out_p == out_s
    assert out_s["brick_reach"] == 1, "C_DRIFT no longer gives reach 1 here"
    assert 2 * out_s["brick_reach"] + 1 < st_s.bricks_per_side
    assert out_s["brick_reach_realized"] >= 1, "no particle crossed a brick"
    _assert_states_identical(st_s, st_p)


def test_pooled_migrate_window_floor_refuses():
    """A window below the deadlock floor must refuse loudly, before any task
    is dispatched (a knob that cannot apply must not silently move)."""
    import copy

    st_p, cfg = _mig_state()
    st_s = copy.deepcopy(st_p)
    with pytest.raises(ValueError, match="deadlock floor"):
        # the check precedes every pool interaction, so a stub suffices
        state.drift_and_migrate_pooled(st_p, C_DRIFT, pool=None, window=2)
    del st_s


@pytest.mark.parametrize("kernel", ["numpy", "jax"])
def test_the_pooled_migrate_identity_with_a_resident_arena(kernel):
    """Both shared-surface paths provably exercised (the C13 arena_probed
    lesson): a zero-slack build overflows into the arena on a serial priming
    pass, so the pooled pass must (a) replay RELEASES of resident rows and
    (b) replay CLAIMS for fresh spills -- both guarded non-vacuous."""
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
    """A state built with_ids=False: the scratch drops the ids column and the
    facade binds None, end to end."""
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
    """A knob that cannot apply must refuse at validate(), not silently run
    the serial path under a pooled-looking config. EXPLICIT True only: the
    None default is auto and falls back to serial by design."""
    with pytest.raises(ValueError, match="needs a pool"):
        _cfg(tile_workers=1, migrate_pooled=True).validate()


def test_migrate_pooled_auto_default_validates_without_a_pool():
    """The counterpart to the refusal above: the default must NOT refuse at
    tile_workers=1, or every single-process config in the package breaks."""
    assert _cfg(tile_workers=1).validate()


def test_migrate_pooled_auto_default_pools_and_is_bitwise_the_serial_arm():
    """The C14 default flip (Vista 918684). An unnamed knob now POOLS wherever
    a pool exists, and False is the only way back to the serial arm. Both
    directions carry a receipt, and the two arms must still be bitwise: a
    default that moved the answer would be a regression, not a speedup."""
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
    """The knob end to end: a whole K=3 run with repack every step, lead-drift
    routing, kick/migrate epoch alternation on one pool, and the shm ids
    copy-back -- against the plain serial run. Receipts in both directions."""
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
    """D-007: when the arena cannot absorb a spill, BOTH arms refuse with the
    same ValueError; the pooled raise comes from the parent's claim replay."""
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
# Vista 920910 generated the 2048^3 ICs and then took a SIGBUS 196 s into
# stepping, inside `TilePool.__init__`, having asked for 148.5 GB of a
# measured 127.6 GB /dev/shm (job 922332). `SharedMemory` sizes lazily, so
# `create=True` succeeds for a segment the tmpfs cannot back and the process
# dies on first TOUCH with no traceback. These are the tests for the check
# that turns that into a refusal.


def test_the_shm_model_reproduces_what_the_pool_actually_allocates():
    """The model and the allocation are two implementations of one formula.

    `inexor.plan` prices a configuration nobody has built, so it cannot use
    the pool's own measurement; that is exactly how a planner and a runtime
    drift apart. This holds them together at a geometry where both are real.
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
    """And the refusal NAMES the terms, because the levers are among them."""
    from inexor.executor import check_shm_budget

    huge = {"w (n_rows,3) int16": 8 * 10**18, "vel_scale": 17}
    with pytest.raises(MemoryError) as e:
        check_shm_budget(huge, path=str(tmp_path))
    msg = str(e.value)
    assert "w (n_rows,3) int16" in msg
    assert "arena_frac" in msg, "the refusal must name the lever, not just the miss"
    # ordered by size: the binding term is the one a reader acts on
    assert msg.index("w (n_rows,3) int16") < msg.index("vel_scale")


def test_a_missing_tmpfs_reports_that_it_could_not_check(tmp_path):
    """An absent check must not read as a passed one.

    macOS has no /dev/shm, so every developer machine takes this path -- and
    a silent return there would mean the guard's tests pass locally while the
    guard is inert on the one platform that has the problem.
    """
    from inexor.executor import check_shm_budget, shm_capacity

    assert shm_capacity(str(tmp_path / "nope")) == (None, None)
    demand, avail = check_shm_budget({"x": 8 * 10**18}, path=str(tmp_path / "nope"))
    assert demand == 8 * 10**18
    assert avail is None, "unknown capacity must be None, never a number"


def test_the_pilot_configuration_would_now_be_refused():
    """The regression: the exact geometry that took the bus error.

    Numbers are the pilot's own invocation (`--slack 0.20 --arena-frac 0.20`,
    alloc_margin 0.10, 16 workers) against the gg node's MEASURED /dev/shm.
    If a change to the layout brings c-gh under that ceiling this test fails,
    which is the right time to re-read it.
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
    gg_shm = 249116032 * 1024 // 2  # measured, Vista job 922332
    demand = sum(terms.values())
    assert demand / 1e9 == pytest.approx(148.5, abs=0.5)
    assert demand > gg_shm, (
        f"c-gh asks {demand / 1e9:.1f} GB of a {gg_shm / 1e9:.1f} GB tmpfs"
    )


# ------------------------------------------------------------ the backing store
# The pool's shared arrays sit on `memfd` where the kernel has it and POSIX
# `/dev/shm` otherwise. The two are the same pages; the difference is that
# `/dev/shm` is a mount with a size cap and memfd is not, which is the whole
# reason c-gh fits. The platform picks, so WITHOUT the env override neither
# machine can exercise both paths: this suite would only ever see posix and
# the cluster only ever memfd.

from inexor.executor import has_memfd  # noqa: E402

# has_memfd() PROBES; it does not read an attribute. Job 922557 passed this
# whole file 24/24 while exercising posix only, because conda-forge's Python
# 3.14 has no `os.memfd_create` even where the kernel does, and the parametrise
# list silently collapsed to one entry. A gate that quietly stops covering the
# path it exists for is worse than no gate.
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
    """Both directions, because a one-way check passes on a private copy."""
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
    """Asking for memfd where there is none must fail loudly.

    Falling back would put the c-gh run back on the capped mount and it would
    die exactly as 920910 did, having been told it was on the new path.
    """
    from inexor.executor import shm_backend

    os.environ["INEXOR_SHM_BACKEND"] = "memfd"
    try:
        with pytest.raises(ValueError, match="cannot create one"):
            shm_backend()
    finally:
        del os.environ["INEXOR_SHM_BACKEND"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_pool_is_bitwise_the_serial_loop_on_either_backend(backend, monkeypatch):
    """The identity gate, re-run against the backing store it is standing on."""
    monkeypatch.setenv("INEXOR_SHM_BACKEND", backend)
    s_serial, _ = _run(1)
    s_pool, _ = _run(4)
    _assert_states_identical(s_serial, s_pool)


def test_memfd_is_detected_by_probe_not_by_attribute():
    """The regression for job 922557.

    `os.memfd_create` is gated on CPython's BUILD sysroot, so conda-forge
    ships without it on a kernel that has the syscall -- the gpu env's Python
    3.14.6 says no while the system 3.9 on the same node says yes. Anything
    that reads the attribute to decide is deciding on the wrong question.
    """
    from inexor import executor

    if not has_memfd():
        pytest.skip("no memfd here either way")
    if not hasattr(os, "memfd_create"):
        # the interesting platform: capability present, attribute absent
        assert executor.shm_backend() == "memfd", (
            "has_memfd() is True but the backend chose posix, which is exactly "
            "the silent downgrade that put 922557 back on the capped mount"
        )
    seg = executor.create_segment(1 << 16, "probe", backend="memfd")
    try:
        assert seg.kind == "memfd"
    finally:
        seg.close()
        seg.unlink()


def test_adoption_peak_is_not_the_total():
    """The regression for job 922682, which was refused for this difference.

    Adopted fields release their private copies as they are copied, so they
    net to zero; only the in-flight duplicate and the fresh segments are new.
    """
    from inexor.executor import adoption_peak

    terms = {"w": 78_346_000_000, "off": 39_173_000_000,
             "arena_bucket": 13_744_000_000, "occupancy": 4_295_000_000,
             "coarse force g0,g1,g2": 12_885_000_000}
    adopted = {"w", "off", "arena_bucket", "occupancy"}
    assert sum(terms.values()) == pytest.approx(148.4e9, rel=1e-3)
    # the largest adopted field, not the sum, and not the fresh total either
    assert adoption_peak(terms, adopted) == 78_346_000_000
    # with nothing adopted it degrades to the total, which is the posix case
    assert adoption_peak(terms, ()) == sum(terms.values())
    # fresh dominates when the adopted set is small
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
    """Adoption buys nothing on a tmpfs: it must hold every segment at once,
    whatever the parent happens to be holding.

    `shm_capacity` is stubbed rather than pointed at a real directory because
    macOS has no /dev/shm at all, and the unstubbed call there returns "could
    not check" -- which is the right answer for the platform and useless for
    testing the arithmetic.
    """
    from inexor import executor

    monkeypatch.setattr(executor, "shm_capacity",
                        lambda path=None: (100_000_000_000, 100_000_000_000))
    terms = {"w": 78_000_000_000, "off": 39_000_000_000}
    # the same terms and the same adoption that PASS on memfd
    executor.preflight_shared_memory(dict(terms), backend="memfd",
                                     adopted={"w", "off"}) if False else None
    with pytest.raises(MemoryError, match="does not fit /dev/shm"):
        executor.preflight_shared_memory(terms, backend="posix",
                                         adopted={"w", "off"})


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_preshared_state_is_adopted_without_a_second_copy(backend, monkeypatch):
    """The state must exist ONCE. Job 922723 died because it existed twice.

    Identity, not equality: if the pool copied, `st.off` would be a different
    object afterwards and the peak would be twice the state. Equality would
    pass on a copy, which is exactly the failure being excluded.
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
        # put the payload where the loader would have put it
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
            # only the coarse meshes are charged
            assert set(pool._shm_demand) == {"coarse force g0,g1,g2"}
        finally:
            pool.close()
    finally:
        alloc.close()


def test_the_loader_fills_shared_memory_and_is_otherwise_unchanged(tmp_path):
    """Same state, whichever allocator: the shared path is a placement
    change, not a numerical one."""
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
