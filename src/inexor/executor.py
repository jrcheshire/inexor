"""The pool executor: the engine's tile loop over persistent worker processes.

Promoted from `scripts/v2_m6_c2_pool.py` (canary C2 of the wall plan), which
keeps its own copy as the ratified skeleton. The design constraints, each
measured or asserted rather than assumed:

- **spawn, never fork** -- jax is loaded in the parent.
- **Affinity BEFORE jax exists in the worker.** XLA-CPU sizes its spin-waiting
  pool by the VISIBLE cores and ignores every thread env var (umbrella:
  thread-count-is-part-of-the-pin). Job 466 measured the un-pinned
  consequence on antares: walls GROWING with W (9.05 -> 11.50 s, W=2 -> 16)
  as 16 workers each spun a 28-core pool. This is why nothing in this module
  imports jax (or anything that imports jax) at module level: the spawned
  child imports this module before `_worker_init` runs.
- **The parent's `SlotState` arrays are ADOPTED into shared memory** at pool
  creation and the parent's fields rebound to the shm views, so every
  in-place mutation the parent makes (apply, migrate, the in-place repack) is
  visible to workers with zero per-step copies. `repack` writes `brick_start`
  and `occupancy` by contents for exactly this reason; the scalar
  `arena_base` it moves rides the per-step task header instead.
- **Why concurrent parent writes cannot change a consumed value:** the kick
  consumes velocities only at OWNED rows, a brick is written only by the tile
  that owns it, and positions are untouched by the kick -- so the
  buffer-brick velocities a worker may decode mid-application are discarded
  in every arm. The bitwise executor-identity gate is what certifies this.
- **One backend per run.** cuFFT and XLA-CPU FFTs are not bitwise
  interchangeable, so the pool is the CPU lane by construction: creation
  REFUSES a non-CPU parent backend, and workers are spawned with
  `JAX_PLATFORMS=cpu`. A device lane is a different executor (wall plan W3).
- **Results are applied in ARRIVAL order**, deliberately: the canary's
  bitwise gate passed that way on three architectures, which is the strong
  form of the disjointness claim, and buffering for a fixed order would cost
  W x cap x 14 B for auditability the partition assertion already provides.
- **Transport is pickled results** (canary transport A, the configuration the
  gg 10.4x was measured at); the shm-slab return path is the priced fallback
  if a readout's idle numbers implicate pickling.
"""

import multiprocessing as mp
import os
import time
from multiprocessing import shared_memory

import numpy as np

__all__ = ["SHM_FIELDS", "TilePool"]

# Every array field of SlotState the step reads or writes. `ids` is absent on
# purpose: the tile loop never touches ids, so they stay parent-side and the
# executor-identity gate proves they evolve identically.
SHM_FIELDS = ("off", "w", "occupancy", "brick_start", "vel_scale", "arena_bucket")

_G = {}


def _rss_mb():
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1e3  # kB -> MB
    except OSError:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # macOS bytes
    return -1.0


def _worker_init(shm_names, shapes, dtypes, small, fn_args, x64, core_sets, rank_counter):
    """Pin affinity, then attach shm views and build the jitted tile program.

    `x64` replicates the PARENT's jax_enable_x64 into the worker -- the
    library's never-touch-jax.config rule is about not overriding the caller's
    choice, and a spawned worker starts without the caller's runtime, so
    replicating it is how the choice propagates rather than being made here.
    """
    if core_sets is not None and hasattr(os, "sched_setaffinity"):
        with rank_counter.get_lock():
            rank = rank_counter.value
            rank_counter.value += 1
        os.sched_setaffinity(0, set(core_sets[rank % len(core_sets)]))
    t0 = time.perf_counter()
    import jax

    jax.config.update("jax_enable_x64", bool(x64))
    from inexor.forces import make_tile_force_fn

    views, segs = {}, []
    for name in SHM_FIELDS:
        seg = shared_memory.SharedMemory(name=shm_names[name])
        segs.append(seg)
        views[name] = np.ndarray(shapes[name], dtype=dtypes[name], buffer=seg.buf)
    g_coarse = []
    for i in range(3):
        seg = shared_memory.SharedMemory(name=shm_names[f"g{i}"])
        segs.append(seg)
        g_coarse.append(np.ndarray(shapes[f"g{i}"], dtype=dtypes[f"g{i}"], buffer=seg.buf))
    one_tile, _ = make_tile_force_fn(**fn_args)
    _G.update(views=views, small=small, g_coarse=g_coarse, one_tile=one_tile,
              segs=segs, st=None, step=None, init_s=time.perf_counter() - t0)


def _ensure_facade(C):
    """The worker's read-only SlotState, rebuilt once per DISPATCH EPOCH.

    Not per task: the facade's lazy brick->arena index would otherwise be
    reconstructed O(n_arena) per tile. And it must not survive an epoch
    boundary, because migrate moves the arena and repack moves `arena_base`
    between dispatches -- the header carries both the epoch and the scalar.
    """
    if _G["step"] != C["step"]:
        from inexor.state import SlotState

        v, small = _G["views"], _G["small"]
        _G["st"] = SlotState(
            t9=small["t9"], bricks_per_side=small["bricks_per_side"],
            brick_start=v["brick_start"], occupancy=v["occupancy"],
            off=v["off"], w=v["w"], vel_scale=v["vel_scale"],
            arena_base=C["arena_base"], arena_bucket=v["arena_bucket"],
            n_particles=small["n_particles"], ids=None,
        )
        _G["step"] = C["step"]
    return _G["st"]


def _worker_task(arg):
    """One tile of the kick; the writes ride back for the parent to apply."""
    t, bricks, C = arg
    st = _ensure_facade(C)
    from inexor.engine import tile_task

    res = tile_task(st, _G["one_tile"], C, _G["g_coarse"], tuple(t), bricks)
    res["worker"] = os.getpid()
    res["rss_mb"] = _rss_mb()
    return res


def _worker_coarse_task(arg):
    """One coarse-paint chunk: decode -> sub-block integer paint (W2 Stage C).

    The ACCUMULATION stays in the parent, where integer associativity makes
    any application order bitwise the serial one; a worker only ever returns
    its chunk's bounded sub-block. Mirrors the sub-block branch of
    `engine.coarse_delta_streamed` line for line."""
    gi, bricks, H = arg
    st = _ensure_facade(H)
    import jax.numpy as jnp

    from inexor.engine import _assert_stencil_contained, _chunk_cuboid
    from inexor.painting import paint_tsc_int_subblock

    n = H["n_coarse"]
    _, x, _ = st.decode_bricks(bricks)
    m = len(x)
    if m == 0:
        return dict(gi=gi, empty=True)
    xp = np.zeros((H["pad"], 3), dtype=np.float64)
    xp[:m] = x
    lv = np.zeros(H["pad"], dtype=bool)
    lv[:m] = True
    c0, span = _chunk_cuboid(gi, H["chunk_bricks"], H["nb_side"], n)
    origin = np.where(span + 3 >= n, 0, (c0 - 1) % n)
    extent = np.where(span + 3 >= n, n, span + 3)
    _assert_stencil_contained(x, H["box"] / float(n), origin, extent, n)
    sub = np.asarray(
        paint_tsc_int_subblock(
            jnp.asarray(xp), tuple(int(o) for o in origin),
            tuple(int(e) for e in extent), n, H["box"], H["frac_bits"], live=lv,
        ),
        dtype=np.int64,
    )
    return dict(gi=gi, empty=False, origin=origin, extent=extent, sub=sub)


class TilePool:
    """W persistent workers sharing the caller's state through shared memory.

    Lifecycle: `run` creates one per pooled run and MUST `close()` it (run's
    `finally` does) -- creation rebinds the caller's `SlotState` fields to shm
    views, and `close()` gives them regular memory back before unlinking, so
    the state object outlives the pool either way.
    """

    def __init__(self, st, cfg):
        import jax

        if jax.default_backend() != "cpu":
            # a CUDA parent with CPU workers would put the two force arms on
            # backends whose FFTs are not bitwise-interchangeable -- and the
            # mesh-ladder precedent (antares 409) is that a wrong backend must
            # VOID the run loudly, not degrade it
            raise ValueError(
                f"TilePool is the CPU lane and the parent backend is "
                f"{jax.default_backend()!r}; the device lane is a different executor"
            )
        self.st = st
        self.workers = int(cfg.tile_workers)
        self._step = 0
        self._C = None
        self._segs = []
        self._names, self._shapes, self._dtypes, self._views = {}, {}, {}, {}
        for f in SHM_FIELDS:
            view = self._share(f, np.asarray(getattr(st, f)))
            setattr(st, f, view)
        # the parent's cached brick->arena index maps into the OLD array;
        # values are equal but the invariant is identity, so rebuild lazily
        st._invalidate_arena_index()
        n = int(cfg.n_coarse)
        for i in range(3):
            self._share(f"g{i}", shape=(n, n, n), dtype=cfg.np_coarse_dtype)
        fn_args = dict(
            n_fine=cfg.n_fine, box_size=cfg.box_size, n_particles_total=cfg.n_total,
            n_tile=cfg.n_tile, b_fine=cfg.b_fine, r_s=cfg.r_s, paint=cfg.paint_short,
            frac_bits=cfg.frac_bits, fdtype=cfg.np_fine_dtype,
        )
        small = dict(t9=st.t9, bricks_per_side=st.bricks_per_side,
                     n_particles=st.n_particles)
        core_sets = None
        if cfg.worker_affinity and hasattr(os, "sched_getaffinity"):
            cores = sorted(os.sched_getaffinity(0))
            per = max(1, len(cores) // self.workers)
            core_sets = [cores[i * per:(i + 1) * per] or cores[-per:]
                         for i in range(self.workers)]
        ctx = mp.get_context("spawn")
        rank_counter = ctx.Value("i", 0)
        # inherited at spawn, which is the only reliable channel: the child
        # imports numpy (module import) before any initializer code runs, so
        # setting these inside the worker would be too late. Restored after
        # the workers exist so the parent's environment is not repainted.
        saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "JAX_PLATFORMS")}
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["JAX_PLATFORMS"] = "cpu"
        try:
            self._pool = ctx.Pool(
                self.workers, initializer=_worker_init,
                initargs=(self._names, self._shapes, self._dtypes, small, fn_args,
                          bool(jax.config.jax_enable_x64), core_sets, rank_counter),
            )
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def _share(self, key, arr=None, shape=None, dtype=None):
        if arr is not None:
            shape, dtype = arr.shape, arr.dtype
        nbytes = max(int(np.dtype(dtype).itemsize * np.prod(shape, dtype=np.int64)), 1)
        seg = shared_memory.SharedMemory(create=True, size=nbytes)
        view = np.ndarray(shape, dtype=dtype, buffer=seg.buf)
        if arr is not None:
            view[...] = arr
        self._segs.append(seg)
        self._names[key], self._shapes[key], self._dtypes[key] = (
            seg.name, tuple(shape), str(np.dtype(dtype)))
        self._views[key] = view
        return view

    def stage_step(self, g_coarse, C):
        """Publish this step's coarse force meshes and per-step header."""
        for i, g in enumerate(g_coarse):
            a = np.asarray(g)
            buf = self._views[f"g{i}"]
            if a.shape != buf.shape or a.dtype != buf.dtype:
                raise ValueError(
                    f"coarse mesh {i}: {a.shape}/{a.dtype} does not match the "
                    f"pool's {buf.shape}/{buf.dtype} -- the mesh geometry moved "
                    "under a live pool"
                )
            buf[...] = a
        self._step += 1
        self._C = dict(C, step=self._step, arena_base=int(self.st.arena_base))

    def imap(self, tasks):
        """Arrival-order iterator of tile results for the staged step."""
        if self._C is None:
            raise RuntimeError("imap before stage_step: the workers have no header")
        C = self._C
        return self._pool.imap_unordered(_worker_task, [(t, b, C) for t, b in tasks])

    def stage_coarse(self, H):
        """Publish the coarse-paint header (W2 Stage C). No arrays move: the
        workers read state through shm and the mesh accumulates parent-side.
        Its own epoch, distinct from the tile loop's, so the facade is rebuilt
        after the migrate that ended the previous step."""
        self._step += 1
        self._H = dict(H, step=self._step, arena_base=int(self.st.arena_base))

    def imap_coarse(self, tasks):
        """Arrival-order iterator of coarse-chunk sub-blocks."""
        if getattr(self, "_H", None) is None:
            raise RuntimeError("imap_coarse before stage_coarse: no header")
        H = self._H
        return self._pool.imap_unordered(
            _worker_coarse_task, [(gi, b, H) for gi, b in tasks]
        )

    def close(self):
        if getattr(self, "_pool", None) is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        # give the caller's state regular memory back BEFORE unlinking, or the
        # arrays would be views into freed segments
        for f in SHM_FIELDS:
            cur = getattr(self.st, f, None)
            if cur is not None:
                setattr(self.st, f, np.array(cur, copy=True))
        self.st._invalidate_arena_index()
        for seg in self._segs:
            seg.close()
            try:
                seg.unlink()
            except FileNotFoundError:
                pass
        self._segs = []
