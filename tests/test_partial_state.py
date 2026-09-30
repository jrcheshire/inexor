"""Node-local `SlotState` (`slabs` set): one rank's brick x-slabs of a whole-box state.

`restrict_to_slabs` cuts a whole state into the node-local state a rank would hold. Every
per-brick read of the cut must equal the whole state's on the owned bricks, and refuse
outside them; the repack must produce the whole repack's owned block byte for byte; passes
with no cross-rank path refuse a node-local state.
"""

import math

import numpy as np
import pytest

from inexor import engine, state
from inexor.codec import T9Layout
from inexor.ooc_fft import partition_units

N_PART, BOX, NB = 32, 32.0, 8


def evolved_state(seed=3, steps=3, arena_frac=0.05, slack=0.02):
    """A whole state after a few migrations: spares in use and a populated arena."""
    rng = np.random.default_rng(seed)
    g = (np.arange(N_PART) + 0.5) * (BOX / N_PART)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    x = np.mod(q + rng.normal(scale=0.25 * BOX / N_PART, size=q.shape), BOX)
    v = rng.normal(scale=0.5, size=x.shape)
    st = state.SlotState.build(x, v, T9Layout(BOX, N_PART, 2), NB, brick_slack=slack,
                               arena_frac=arena_frac)
    for _ in range(steps):
        state.drift_and_migrate(st, 0.5)
    return st


def rank_slabs(nb, n_ranks):
    """Each rank's brick x-slabs [lo, hi) under an even split."""
    return [tuple(r) for r in partition_units(nb, n_ranks, 1)]


def restrict_to_slabs(st, slabs, alloc_margin=0.10):
    """The node-local state owning brick x-slabs `slabs`, cut from the whole state `st`.

    Owned bricks keep their allocation sizes (rows renumbered from 0); arena residents of
    owned buckets keep their relative arena-row order, the order the writer reads.
    """
    assert st.is_whole
    lo_s, hi_s = int(slabs[0]), int(slabs[1])
    nb2, p3 = st.bricks_per_side ** 2, st.buckets_per_brick
    blo, bhi = lo_s * nb2, hi_s * nb2
    r0, r1 = int(st.brick_start[blo]), int(st.brick_start[bhi])

    brick_start = np.zeros_like(st.brick_start)
    brick_start[blo:bhi + 1] = st.brick_start[blo:bhi + 1] - r0
    brick_start[bhi + 1:] = r1 - r0

    a_rows = np.nonzero((st.arena_bucket >= blo * p3) & (st.arena_bucket < bhi * p3))[0]
    n_local = st.brick_member_counts()[blo:bhi].sum()
    n_arena = max(len(a_rows), math.ceil(int(n_local) * st.arena_frac))
    # the runs, then alloc_margin headroom as `_alloc_geometry` gives it, then the arena
    n_alloc = math.ceil((r1 - r0) * (1.0 + alloc_margin))
    off = np.zeros((n_alloc + n_arena, 3), dtype=st.off.dtype)
    w = np.zeros((n_alloc + n_arena, 3), dtype=st.w.dtype)
    off[:r1 - r0] = st.off[r0:r1]
    w[:r1 - r0] = st.w[r0:r1]
    off[n_alloc:n_alloc + len(a_rows)] = st.off[st.arena_base + a_rows]
    w[n_alloc:n_alloc + len(a_rows)] = st.w[st.arena_base + a_rows]
    arena_bucket = np.full(n_arena, -1, dtype=np.int64)
    arena_bucket[:len(a_rows)] = st.arena_bucket[a_rows]
    vel_scale = np.ones_like(st.vel_scale)
    vel_scale[blo:bhi] = st.vel_scale[blo:bhi]
    return state.SlotState(
        t9=st.t9, bricks_per_side=st.bricks_per_side, brick_start=brick_start,
        occupancy=st.occupancy[blo * p3:bhi * p3].copy(), off=off, w=w, vel_scale=vel_scale,
        arena_base=n_alloc, arena_bucket=arena_bucket, n_particles=int(n_local),
        slabs=(lo_s, hi_s), arena_frac=st.arena_frac,
    )


@pytest.fixture(scope="module")
def whole():
    st = evolved_state()
    assert st.arena_used > 0, "vacuous: no arena residents"
    return st


@pytest.mark.parametrize("n_ranks", [1, 2, 4, 8])
def test_a_cut_reads_the_whole_states_bricks(whole, n_ranks):
    counts = whole.brick_member_counts()
    total = arena = 0
    for slabs in rank_slabs(NB, n_ranks):
        part = restrict_to_slabs(whole, slabs)
        assert part.is_whole == (n_ranks == 1)
        part.check()
        blo, bhi = part.owned_bricks
        got = part.brick_member_counts()
        np.testing.assert_array_equal(got[blo:bhi], counts[blo:bhi])
        assert not got[:blo].any() and not got[bhi:].any()
        for b in range(blo, bhi):
            _, x, v = part.decode_brick(b)
            _, xw, vw = whole.decode_brick(b)
            np.testing.assert_array_equal(x, xw)
            np.testing.assert_array_equal(v, vw)
            np.testing.assert_array_equal(part.bucket_slot_starts(b) - part.brick_start[b],
                                          whole.bucket_slot_starts(b) - whole.brick_start[b])
        total += part.n_live
        arena += part.arena_used
        assert part.n_live == part.n_particles
    assert total == whole.n_live and arena == whole.arena_used


def test_a_brick_outside_the_owned_range_refuses(whole):
    lo, hi = rank_slabs(NB, 4)[1]
    part = restrict_to_slabs(whole, (lo, hi))
    blo, bhi = part.owned_bricks
    for b in (blo - 1, bhi, 0, whole.n_bricks - 1):
        with pytest.raises(IndexError, match="outside this state's owned bricks"):
            part.brick_live_count(b)
    with pytest.raises(IndexError):
        part._occ(blo, bhi + 1)
    part.brick_live_count(blo)
    part.brick_live_count(bhi - 1)


@pytest.mark.parametrize("n_ranks", [2, 4])
@pytest.mark.parametrize("which", ["repack", "_repack_reference"])
def test_a_cuts_repack_is_the_whole_repacks_owned_block(whole, n_ranks, which):
    ref = restrict_to_slabs(whole, (0, NB))
    getattr(ref, which)(brick_slack=0.10)
    for slabs in rank_slabs(NB, n_ranks):
        part = restrict_to_slabs(whole, slabs)
        getattr(part, which)(brick_slack=0.10)
        part.check()
        blo, bhi = part.owned_bricks
        p3 = part.buckets_per_brick
        r0, r1 = int(ref.brick_start[blo]), int(ref.brick_start[bhi])
        np.testing.assert_array_equal(part.brick_start[blo:bhi + 1],
                                      ref.brick_start[blo:bhi + 1] - r0)
        np.testing.assert_array_equal(part.occupancy, ref.occupancy[blo * p3:bhi * p3])
        n_alloc = part.arena_base
        assert n_alloc == r1 - r0
        np.testing.assert_array_equal(part.off[:n_alloc], ref.off[r0:r1])
        np.testing.assert_array_equal(part.w[:n_alloc], ref.w[r0:r1])
        assert part.arena_used == 0 and not part.off[n_alloc:].any()


def test_check_refuses_a_malformed_cut(whole):
    slabs = rank_slabs(NB, 4)[1]
    bad = restrict_to_slabs(whole, slabs)
    bad.occupancy = bad.occupancy[:-1]
    with pytest.raises(ValueError, match="occupancy holds"):
        bad.check()

    bad = restrict_to_slabs(whole, slabs)
    bad.brick_start[1:] += 1       # brick 0 is not owned by rank 1
    with pytest.raises(ValueError, match="non-owned brick must be an empty run"):
        bad.check()

    bad = restrict_to_slabs(whole, slabs)
    a = int(np.nonzero(bad.arena_bucket < 0)[0][0])
    bad.arena_bucket[a] = 0        # a bucket of slab 0, owned by rank 0
    bad.n_particles += 1
    with pytest.raises(ValueError, match="outside the owned buckets"):
        bad.check()


def test_passes_without_a_cross_rank_path_refuse_a_cut(whole):
    part = restrict_to_slabs(whole, rank_slabs(NB, 2)[0])
    match = "needs a whole-box state"
    with pytest.raises(NotImplementedError, match=match):
        state.drift_and_migrate(part, 0.5)
    with pytest.raises(NotImplementedError, match=match):
        state.drift_and_migrate_pooled(part, 0.5, pool=None)
    # the step has a cross-rank path in the production device lane only (tests/test_ranks_run)
    lane = "production device lane"
    with pytest.raises(NotImplementedError, match=lane):
        engine.step(part, None, (0.0, 0.0), 0.0)
    with pytest.raises(NotImplementedError, match=lane):
        engine.run(part, None, np.zeros((1, 3)))
    from inexor.executor import TilePool

    with pytest.raises(NotImplementedError, match=match):
        TilePool(part, None)


# ------------------------------------------------------------------ the node-local loader


def _same_state(a, b):
    for f in ("brick_start", "occupancy", "off", "w", "vel_scale", "arena_bucket"):
        x, y = getattr(a, f), getattr(b, f)
        assert x.dtype == y.dtype and x.shape == y.shape, f
        np.testing.assert_array_equal(x, y, err_msg=f)
    assert (a.arena_base, a.n_particles, a.owned_slabs, a.arena_frac) == \
        (b.arena_base, b.n_particles, b.owned_slabs, b.arena_frac)


@pytest.fixture(scope="module")
def written(whole, tmp_path_factory):
    from inexor import icgen

    d = str(tmp_path_factory.mktemp("whole"))
    icgen.write_t9_slabs(whole, d)
    return d


@pytest.mark.parametrize("n_ranks", [1, 2, 4, 8])
def test_a_node_local_load_is_the_cut_of_the_whole_load(written, n_ranks):
    from inexor import icgen

    full = icgen.load_slot_state(written, arena_frac=0.05)
    assert full.is_whole and full.slabs is None
    for slabs in rank_slabs(NB, n_ranks):
        part = icgen.load_slot_state(written, arena_frac=0.05, slabs=slabs)
        part.check()
        _same_state(part, restrict_to_slabs(full, slabs))
        assert part.occupancy.nbytes * n_ranks == full.occupancy.nbytes


def test_the_loader_maps_files_by_slab_and_refuses_a_bad_list(written, tmp_path):
    import json
    import os
    import shutil

    from inexor import icgen

    d = str(tmp_path / "c")
    shutil.copytree(written, d)
    mpath = f"{d}/{icgen.MANIFEST}"
    man = json.load(open(mpath))
    files = man["files"]
    # list order is the writer's: a rotated list loads the same state
    json.dump(dict(man, files=files[1:] + files[:1]), open(mpath, "w"))
    _same_state(icgen.load_slot_state(d), icgen.load_slot_state(written))
    json.dump(dict(man, files=files[:-1]), open(mpath, "w"))
    with pytest.raises(ValueError, match="needs every slab once"):
        icgen.load_slot_state(d)
    json.dump(dict(man, files=files[:-1] + files[:1]), open(mpath, "w"))
    with pytest.raises(ValueError, match="not a distinct"):
        icgen.load_slot_state(d)
    # a file whose content is another slab's
    os.replace(f"{d}/{files[1]}", f"{d}/{files[0]}")
    shutil.copy(f"{written}/{files[1]}", f"{d}/{files[1]}")
    json.dump(man, open(mpath, "w"))
    with pytest.raises(ValueError, match="not the slab 0 its name gives"):
        icgen.load_slot_state(d)
    shutil.copy(f"{written}/{files[0]}", f"{d}/{files[0]}")
    json.dump(dict(man, n_particles=man["n_particles"] + 1), open(mpath, "w"))
    with pytest.raises(ValueError, match="the manifest records"):
        icgen.load_slot_state(d)
    json.dump(man, open(mpath, "w"))
    for bad in ((2, 2), (-1, 3), (0, NB + 1)):
        with pytest.raises(ValueError, match="not a non-empty range"):
            icgen.load_slot_state(d, slabs=bad)
    icgen.load_slot_state(d).check()


# ------------------------------------------------------------ the per-rank writer


def _digests(d):
    """sha256 of every file in `d` (the slab files and the manifest)."""
    import hashlib
    import os

    return {f: hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest()
            for f in sorted(os.listdir(d))}


def _write_on_ranks(n_ranks, make_state, d):
    """`make_state(slabs)` on each of `n_ranks` loopback ranks, then the collective write."""
    from inexor import icgen
    from inexor.comm import run_loopback

    def body(comm):
        st = make_state(rank_slabs(NB, n_ranks)[comm.rank])
        return icgen.write_t9_slabs(st, d, comm=comm)

    mans = run_loopback(n_ranks, body)
    assert all(m == mans[0] for m in mans)
    return mans[0]


@pytest.mark.parametrize("n_ranks", [1, 2, 4, 8])
def test_load_n_then_write_n_is_the_one_rank_bytes(written, n_ranks, tmp_path):
    from inexor import icgen

    ref = str(tmp_path / "ref")
    icgen.write_t9_slabs(icgen.load_slot_state(written, arena_frac=0.05), ref)
    assert _digests(ref) == _digests(written), "the one-rank writer is not a fixed point"
    d = str(tmp_path / "n")
    man = _write_on_ranks(
        n_ranks, lambda s: icgen.load_slot_state(written, arena_frac=0.05, slabs=s), d)
    assert _digests(d) == _digests(ref)
    assert man == icgen.read_manifest(ref)


@pytest.mark.parametrize("n_ranks", [2, 4, 8])
def test_cuts_holding_arena_residents_write_the_whole_states_bytes(whole, written, n_ranks,
                                                                   tmp_path):
    d = str(tmp_path / "n")
    _write_on_ranks(n_ranks, lambda s: restrict_to_slabs(whole, s), d)
    assert _digests(d) == _digests(written)


@pytest.mark.parametrize("n_w", [1, 2, 4])
@pytest.mark.parametrize("n_r", [1, 2, 4])
def test_n_ranks_write_and_m_ranks_load(whole, written, n_w, n_r, tmp_path):
    from inexor import icgen

    dw, dr = str(tmp_path / "w"), str(tmp_path / "r")
    _write_on_ranks(n_w, lambda s: restrict_to_slabs(whole, s), dw)
    _write_on_ranks(n_r, lambda s: icgen.load_slot_state(dw, arena_frac=0.05, slabs=s), dr)
    assert _digests(dw) == _digests(dr) == _digests(written)


def test_a_rank_failing_mid_write_leaves_no_manifest(whole, written, tmp_path, monkeypatch):
    import shutil

    from inexor import icgen
    from inexor.comm import CommAborted, run_loopback

    d = str(tmp_path / "g")
    shutil.copytree(written, d)            # a complete generation being overwritten
    real = icgen._save_slab

    def failing(path, *a):
        if path.endswith("t9_slab_0005.npz"):
            raise OSError("disk full (injected)")
        return real(path, *a)

    monkeypatch.setattr(icgen, "_save_slab", failing)
    seen = {}

    def body(comm):
        st = restrict_to_slabs(whole, rank_slabs(NB, 4)[comm.rank])
        try:
            return icgen.write_t9_slabs(st, d, comm=comm)
        except BaseException as e:
            seen[comm.rank] = type(e)
            raise

    with pytest.raises(OSError, match="injected"):
        run_loopback(4, body, timeout=20.0)
    assert seen[2] is OSError                              # slab 5 is rank 2's
    assert all(issubclass(seen[r], CommAborted) for r in (0, 1, 3)), seen
    with pytest.raises(FileNotFoundError, match="manifest is written last"):
        icgen.load_slot_state(d)


def test_the_writer_refuses_ranks_that_do_not_tile_the_box(whole, tmp_path):
    from inexor import icgen
    from inexor.comm import run_loopback

    part = restrict_to_slabs(whole, rank_slabs(NB, 2)[0])
    with pytest.raises(ValueError, match="do not tile"):
        icgen.write_t9_slabs(part, str(tmp_path / "a"))            # one rank, half the box

    def swapped(comm):
        st = restrict_to_slabs(whole, rank_slabs(NB, 2)[1 - comm.rank])
        return icgen.write_t9_slabs(st, str(tmp_path / "b"), comm=comm)

    with pytest.raises(ValueError, match="do not tile"):
        run_loopback(2, swapped)

    def probe(comm):
        st = restrict_to_slabs(whole, rank_slabs(NB, 2)[comm.rank])
        return icgen.write_t9_slabs(st, str(tmp_path / "c"), comm=comm, max_slabs=1)

    with pytest.raises(ValueError, match="run it on one rank"):
        run_loopback(2, probe)


# ------------------------------------------------------------------- checkpoints


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    """A 6-step run checkpointed at steps 3 (gen0) and 6 (gen1), and its coefficients."""
    import jax

    from tests.test_engine import _cfg, _ck_state, _coeffs

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        d = str(tmp_path_factory.mktemp("ck"))
        co = _coeffs(6)
        cfg = _cfg(checkpoint_dir=d, checkpoint_every=3)
        engine.run(_ck_state(cfg), cfg, co)
    finally:
        jax.config.update("jax_enable_x64", prev)
    return d, co


def _reload_and_rewrite(src, co, n_ranks, out, torn=None):
    """Load the newest checkpoint of `src` on `n_ranks` ranks, rewrite it as gen0 of `out`;
    returns every rank's `resume`."""
    from inexor.comm import run_loopback
    from tests.test_engine import _cfg

    nb = _cfg().n_fine // _cfg().n_brick

    def body(comm):
        slabs = None if n_ranks == 1 else rank_slabs(nb, n_ranks)[comm.rank]
        st, res = engine.load_checkpoint(src, _cfg(), co, comm=comm, slabs=slabs)
        cfg_out = _cfg(checkpoint_dir=out, checkpoint_every=3)
        engine._write_checkpoint(st, cfg_out, co, res["step"], res["cap_shape"],
                                 res["pad_shape"], 0, device_shapes=res["device_shapes"],
                                 comm=comm)
        return res

    return run_loopback(n_ranks, body)


@pytest.mark.parametrize("n_ranks", [1, 2, 4, 8])
def test_an_n_rank_checkpoint_is_the_one_rank_bytes(ckpt, n_ranks, tmp_path):
    import os

    src, co = ckpt
    ref, out = str(tmp_path / "ref"), str(tmp_path / "out")
    r1 = _reload_and_rewrite(src, co, 1, ref)[0]
    res = _reload_and_rewrite(src, co, n_ranks, out)
    assert all(r == r1 for r in res) and r1["step"] == 6 and r1["gen"] == 1
    assert _digests(f"{out}/gen0") == _digests(f"{ref}/gen0")
    # writing is a fixed point: the rewrite is the run's own step-6 generation
    assert _digests(f"{ref}/gen0") == _digests(f"{src}/gen1")
    assert os.listdir(out) == ["gen0"]


def test_every_rank_takes_rank_zeros_generation(ckpt, tmp_path):
    import os
    import shutil

    src, co = ckpt
    d = str(tmp_path / "ck")
    shutil.copytree(src, d)
    os.remove(f"{d}/gen1/manifest.json")        # a torn step-6 generation
    res = _reload_and_rewrite(d, co, 4, str(tmp_path / "out"))
    assert [r["step"] for r in res] == [3] * 4 and {r["gen"] for r in res} == {0}
    assert _digests(f"{tmp_path}/out/gen0") == _digests(f"{src}/gen0")


@pytest.mark.parametrize("n_w", [1, 2, 4])
@pytest.mark.parametrize("n_r", [1, 2, 4, 8])
def test_a_checkpoint_from_n_ranks_resumes_on_m(ckpt, n_w, n_r, tmp_path):
    src, co = ckpt
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _reload_and_rewrite(src, co, n_w, a)
    _reload_and_rewrite(a, co, n_r, b)
    assert _digests(f"{b}/gen0") == _digests(f"{src}/gen1")


def test_checkpoint_refusals_across_ranks(ckpt):
    from inexor.comm import CommAborted, run_loopback
    from tests.test_engine import _cfg, _coeffs

    src, co = ckpt
    nb = _cfg().n_fine // _cfg().n_brick
    seen = {}

    def foreign(comm):
        try:
            engine.load_checkpoint(src, _cfg(), _coeffs(5), comm=comm,
                                   slabs=rank_slabs(nb, 2)[comm.rank])
        except BaseException as e:
            seen[comm.rank] = e
            raise

    with pytest.raises(ValueError, match="different configuration or schedule"):
        run_loopback(2, foreign)
    assert all(isinstance(e, (ValueError, CommAborted)) for e in seen.values()) and \
        len(seen) == 2

    def overlapping(comm):                       # both ranks load slabs [0, 4)
        return engine.load_checkpoint(src, _cfg(), co, comm=comm,
                                      slabs=rank_slabs(nb, 2)[0])

    with pytest.raises(ValueError, match="do not cover the box exactly once"):
        run_loopback(2, overlapping)

    with pytest.raises(ValueError, match="each must load its own"):
        run_loopback(2, lambda comm: engine.load_checkpoint(src, _cfg(), co, comm=comm))
