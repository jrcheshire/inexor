"""The pool executor: the engine's tile loop over persistent worker processes.

- spawn, never fork (jax is loaded in the parent).
- Affinity is set before jax loads in a worker: XLA-CPU sizes its spin-wait pool by the
  visible cores and ignores thread env vars. Hence nothing here imports jax at module level
  (a spawned child imports this module before `_worker_init` runs).
- The parent's `SlotState` arrays are adopted into shared memory and its fields rebound to
  the views, so parent in-place mutations (apply, migrate, repack) reach workers with no
  per-step copy. `repack` writes `brick_start`/`occupancy` by contents for this reason;
  the scalar `arena_base` rides the per-step header.
- Concurrent parent writes cannot change a consumed value: the kick reads velocities only
  at owned rows, each brick is written only by its owning tile, and positions are untouched,
  so buffer-brick velocities a worker may see mid-apply are discarded. Pinned by the
  bitwise executor-identity test.
- CPU only: cuFFT and XLA-CPU FFTs are not bitwise interchangeable, so a non-CPU parent is
  refused and workers run with `JAX_PLATFORMS=cpu`.
- Results are applied in arrival order (tile writes are disjoint) and returned pickled.
"""

import mmap
import multiprocessing as mp
import os
import queue
import time
from multiprocessing import shared_memory

import numpy as np

__all__ = ["SHM_FIELDS", "TilePool", "shm_backend", "has_memfd", "shm_capacity",
           "shm_terms", "preflight_shared_memory", "create_segment", "open_segment",
           "SHM_DIR"]

# POSIX shared memory lives on a tmpfs whose size is a separate budget from host RAM (half
# of it by default). It must be checked up front: `SharedMemory` sizes lazily, so an
# over-budget segment is created fine and the process dies with SIGBUS on first touch.
SHM_DIR = "/dev/shm"

# The SlotState array fields the step reads or writes. `ids` is not here (the tile loop
# never reads it); a pool on a state with ids adds it to its per-instance `_fields`, since
# the pooled migrate reads and writes ids.
SHM_FIELDS = ("off", "w", "occupancy", "brick_start", "vel_scale", "arena_bucket")

# The migrate scratch (`stage_migrate`) is a rolling window of K slab slots, each holding a
# slab's ejected rows as keep-prefix + emig-suffix in eject output order (brick-major; the
# order `_insert_slab` relies on). `mig_scales`, the pass-start vel_scale snapshot, is in
# shm rather than the pickled header.
# Bytes per (K, R) entry: int64 dest + 3 x uint8 off + 3 x int16 w + int32 src; +4 for ids.
MIG_BYTES_PER_ENTRY = 8 + 3 * 1 + 3 * 2 + 4
MIG_IDS_BYTES_PER_ENTRY = 4


def shm_capacity(path=SHM_DIR):
    """(total, available) bytes of the tmpfs backing POSIX shared memory.

    Available is `f_bavail` (what an unprivileged process can get). Returns (None, None)
    where there is no such mount (e.g. macOS).
    """
    try:
        s = os.statvfs(path)
    except (OSError, AttributeError):
        return None, None
    return s.f_blocks * s.f_frsize, s.f_bavail * s.f_frsize


def shm_terms(*, n_rows, index_bytes, n_arena, n_bricks, n_coarse,
              coarse_itemsize, has_ids=False, mig_window=None, mig_rows=None):
    """The model of what `TilePool` puts in shared memory, term by term.

    Geometry in, bytes out, for `inexor.plan`. The pool itself checks its measured arrays
    (`_shm_demand`), not this model; tests hold the two against each other.
    """
    t = {
        "off (n_rows,3) uint8": n_rows * 3 * 1,
        "w (n_rows,3) int16": n_rows * 3 * 2,
        "occupancy (the bucket index)": index_bytes,
        "arena_bucket": n_arena * 8,
        "brick_start": (n_bricks + 1) * 8,
        "vel_scale": n_bricks * 8,
        "coarse force g0,g1,g2": 3 * n_coarse**3 * coarse_itemsize,
    }
    if has_ids:
        t["ids (n_rows,) int32"] = n_rows * 4
    if mig_window and mig_rows:
        per = MIG_BYTES_PER_ENTRY + (MIG_IDS_BYTES_PER_ENTRY if has_ids else 0)
        t["migrate scratch (K x R)"] = mig_window * mig_rows * per
        t["migrate scratch (scales)"] = n_bricks * 8
    return t


def _fmt_gb(b):
    return f"{b / 1e9:.3f} GB"


def check_shm_budget(terms, path=SHM_DIR, headroom=1.0):
    """Raise `MemoryError` (with a table) if `terms` do not fit the /dev/shm tmpfs.

    `headroom` multiplies the demand (the unpriced migrate scratch scales with the run).
    Returns `(demand, available)`; `available` is None where the tmpfs cannot be read, so an
    unperformed check is distinguishable from a passed one.
    """
    demand = sum(terms.values())
    total, avail = shm_capacity(path)
    if total is None:
        return demand, None
    if demand * headroom <= avail:
        return demand, avail
    width = max(len(k) for k in terms)
    table = "\n".join(
        f"    {k:<{width}}  {_fmt_gb(v):>12}"
        for k, v in sorted(terms.items(), key=lambda kv: -kv[1])
    )
    raise MemoryError(
        f"the pool's shared memory does not fit {path}.\n"
        f"{table}\n"
        f"    {'-' * width}  {'-' * 12}\n"
        f"    {'demand':<{width}}  {_fmt_gb(demand):>12}"
        + (f"  (x{headroom:g} headroom = {_fmt_gb(demand * headroom)})"
           if headroom != 1.0 else "")
        + f"\n    {'available':<{width}}  {_fmt_gb(avail):>12}"
        f"\n    {'total':<{width}}  {_fmt_gb(total):>12}\n"
        f"  POSIX shared memory is a tmpfs and its size is a SEPARATE budget "
        f"from host RAM,\n  by default half of it -- so a configuration that "
        f"fits the node's memory can still\n  fail here. `SharedMemory` sizes "
        f"lazily, so without this check the run allocates\n  cleanly and takes "
        f"a SIGBUS on first touch, with no traceback (Vista 920910).\n"
        f"  Levers, largest first: `arena_frac` and `brick_slack` both add rows "
        f"to off/w;\n  `arena_frac` also sets `arena_bucket` outright. "
        f"`python -m inexor.plan --shm-gb` prices\n  a configuration before it "
        f"is run."
    )


_MEMFD_FN = None


def _memfd_fn():
    """`memfd_create` via `os` or, failing that, the runtime libc; None if absent (macOS).

    `os.memfd_create` depends on CPython's build-time sysroot, so it can be missing on a
    kernel that has the syscall; libc has the symbol regardless. Cached.
    """
    global _MEMFD_FN
    if _MEMFD_FN is None:
        if hasattr(os, "memfd_create"):
            _MEMFD_FN = os.memfd_create
        else:
            try:
                import ctypes

                libc = ctypes.CDLL(None, use_errno=True)
                raw = libc.memfd_create
                raw.argtypes = [ctypes.c_char_p, ctypes.c_uint]
                raw.restype = ctypes.c_int
            except (OSError, AttributeError):
                _MEMFD_FN = False
            else:
                def _call(name, flags=0, _raw=raw, _ct=ctypes):
                    fd = _raw(name.encode(), flags)
                    if fd < 0:
                        err = _ct.get_errno()
                        raise OSError(err, os.strerror(err), "memfd_create")
                    return fd
                _MEMFD_FN = _call
    return _MEMFD_FN or None


_HAS_MEMFD = None


def has_memfd():
    """Whether this process can create a memfd, tested by creating one. Cached."""
    global _HAS_MEMFD
    if _HAS_MEMFD is None:
        fn = _memfd_fn()
        if fn is None:
            _HAS_MEMFD = False
        else:
            try:
                fd = fn("inexor-probe", 0)
            except (OSError, ValueError):
                _HAS_MEMFD = False
            else:
                os.close(fd)
                _HAS_MEMFD = True
    return _HAS_MEMFD


def available_ram():
    """`MemAvailable` in bytes, or None where /proc/meminfo is not readable.

    The budget for the `memfd` path (not `MemFree`, which counts reclaimable page cache).
    """
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def adoption_peak(terms, adopted=()):
    """Bytes needed above what the process already holds: `max(largest adopted, sum fresh)`.

    An adopted field is copied into a segment and its private array released, netting zero
    except for the one field mid-copy; fresh segments (the coarse meshes) are all new.
    Assumes nothing outside `st` still references the adopted arrays.
    """
    fresh = {k: v for k, v in terms.items() if k not in adopted}
    biggest_adopted = max((v for k, v in terms.items() if k in adopted), default=0)
    return max(biggest_adopted, sum(fresh.values()))


def preflight_shared_memory(terms, backend=None, headroom=1.0, adopted=()):
    """Check the demand against the backend's own ceiling; raises `MemoryError` if over.

    `posix`: the total against the /dev/shm tmpfs cap (which must hold every segment).
    `memfd`: exempt from that cap, so `adoption_peak` against `MemAvailable`.
    Returns `(demand, budget, what)`; `what` is None when no budget could be read.
    """
    backend = backend or shm_backend()
    demand = sum(terms.values())
    if backend == "posix":
        d, avail = check_shm_budget(terms, headroom=headroom)
        return d, avail, (None if avail is None else "/dev/shm")
    avail = available_ram()
    if avail is None:
        return demand, None, None
    need = adoption_peak(terms, adopted)
    if need * headroom > avail:
        width = max(len(k) for k in terms)
        table = "\n".join(
            f"    {k:<{width}}  {_fmt_gb(v):>12}"
            f"{'  (adopted, frees its own)' if k in adopted else ''}"
            for k, v in sorted(terms.items(), key=lambda kv: -kv[1])
        )
        raise MemoryError(
            f"the pool's shared memory does not fit available RAM.\n{table}\n"
            f"    {'-' * width}  {'-' * 12}\n"
            f"    {'demand (total)':<{width}}  {_fmt_gb(demand):>12}\n"
            f"    {'peak ABOVE baseline':<{width}}  {_fmt_gb(need):>12}"
            f"  <- what is checked\n"
            f"    {'MemAvailable':<{width}}  {_fmt_gb(avail):>12}\n"
            f"  An adopted field releases its private copy as it is copied, so "
            f"the total is\n  NOT the cost: the peak is the largest single "
            f"adopted field, or the sum of the\n  fresh segments, whichever is "
            f"bigger. These are memfd pages, so this is a real\n  RAM shortage "
            f"and not the /dev/shm cap that stopped 920910. Levers, largest\n"
            f"  first: `arena_frac` and `brick_slack` both add rows to off/w, "
            f"and `arena_frac`\n  also sets `arena_bucket` outright; "
            f"`coarse_dtype` halves the mesh segments."
        )
    return demand, avail, "MemAvailable"


def shm_backend():
    """`memfd` where the kernel has it, `posix` otherwise.

    Both give anonymous RAM-backed shared pages, but `shm_open` places them on the size-capped
    `/dev/shm` mount while `memfd_create` uses the kernel's internal, uncapped shm mount.
    `posix` is the fallback (e.g. macOS) and the only path the /dev/shm check applies to.
    `INEXOR_SHM_BACKEND=memfd|posix` forces one, so each path can be exercised on any platform
    (and as an escape hatch where `/proc/<pid>/fd` is not exposed).
    """
    forced = os.environ.get("INEXOR_SHM_BACKEND")
    if forced:
        if forced not in ("memfd", "posix"):
            raise ValueError(
                f"INEXOR_SHM_BACKEND={forced!r}; expected 'memfd' or 'posix'"
            )
        if forced == "memfd" and not has_memfd():
            raise ValueError(
                "INEXOR_SHM_BACKEND=memfd but this process cannot create one"
            )
        return forced
    return "memfd" if has_memfd() else "posix"


class _Seg:
    """One shared mapping: created by the parent, opened by each worker.

    The handle crosses the process boundary: the segment name for `posix`; `(pid, fd)` for the
    nameless `memfd`, which a worker opens via `/proc/<pid>/fd/<n>` (same file, not a copy).
    """

    __slots__ = ("kind", "buf", "handle", "nbytes", "_fd", "_shm")

    def __init__(self, kind, buf, handle, fd=None, shm=None):
        self.kind, self.buf, self.handle = kind, buf, handle
        # recorded, not probed: a released memoryview raises on len()
        self.nbytes = int(handle[-1])
        self._fd, self._shm = fd, shm

    def close(self):
        if self.kind == "memfd":
            self.buf.close()
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
        else:
            self._shm.close()

    def unlink(self):
        """Unlink a posix segment; no-op for memfd (anonymous, freed with its last reference)."""
        if self.kind == "posix":
            self._shm.unlink()


def create_segment(nbytes, tag, backend=None):
    """Parent side. `nbytes` is the mapped length; `tag` is for `/proc` only."""
    backend = backend or shm_backend()
    if backend == "memfd":
        fd = _memfd_fn()(f"inexor-{tag}", 0)
        try:
            os.ftruncate(fd, nbytes)
            buf = mmap.mmap(fd, nbytes, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE)
        except BaseException:
            os.close(fd)
            raise
        return _Seg("memfd", buf, ("memfd", os.getpid(), fd, nbytes), fd=fd)
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    return _Seg("posix", shm.buf, ("posix", shm.name, nbytes), shm=shm)


def open_segment(handle):
    """Worker side. Takes what `_Seg.handle` produced, gives back a mapping."""
    kind = handle[0]
    if kind == "memfd":
        _, pid, fd, nbytes = handle
        # a new descriptor, closed once mapped (the mapping outlives it)
        dup = os.open(f"/proc/{pid}/fd/{fd}", os.O_RDWR)
        try:
            buf = mmap.mmap(dup, nbytes, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(dup)
        return _Seg("memfd", buf, handle)
    _, name, _nbytes = handle
    shm = shared_memory.SharedMemory(name=name)
    return _Seg("posix", shm.buf, handle, shm=shm)


def malloc_trim():
    """Hand glibc's retained free arenas back to the OS. True if it did work.

    Needed because shared segments are fresh kernel pages that cannot reuse the arenas glibc
    retains after the loader frees its per-slab payload. glibc only; False elsewhere.
    """
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        libc.malloc_trim.argtypes = [ctypes.c_size_t]
        libc.malloc_trim.restype = ctypes.c_int
    except (OSError, AttributeError):
        return False
    libc.malloc_trim(0)
    return True


class SharedAllocator:
    """ndarray views backed by shared segments, so a loader writes the state directly into the
    memory the workers read (the state then exists once, not private + shared copy).

    Registered by `id()` of the view, sound because the allocator keeps every view alive.
    """

    def __init__(self, backend=None):
        self.backend = backend or shm_backend()
        self._segs = []
        self._by_id = {}

    def empty(self, shape, dtype, tag="alloc"):
        dtype = np.dtype(dtype)
        nbytes = max(int(dtype.itemsize * np.prod(shape, dtype=np.int64)), 1)
        seg = create_segment(nbytes, tag, backend=self.backend)
        view = np.ndarray(shape, dtype=dtype, buffer=seg.buf)
        self._segs.append(seg)
        self._by_id[id(view)] = (seg, view)
        return view

    def zeros(self, shape, dtype, tag="alloc"):
        view = self.empty(shape, dtype, tag)
        view[...] = 0
        return view

    def segment_of(self, arr):
        """The segment backing `arr` (by identity), or None if this allocator did not make it."""
        hit = self._by_id.get(id(arr))
        return None if hit is None or hit[1] is not arr else hit[0]

    def bytes_held(self):
        return sum(s.nbytes for s in self._segs)

    def close(self):
        for seg in self._segs:
            seg.close()
            try:
                seg.unlink()
            except FileNotFoundError:
                pass
        self._segs, self._by_id = [], {}


_G = {}


def worker_rss_bytes(pool):
    """Summed RSS of a Pool's worker processes, read from /proc.

    Needs no dispatch, so it works before any task has run (`ru_maxrss` is per-process).
    Returns (n_workers, total_bytes), or (n, None) off Linux.
    """
    procs = [p for p in getattr(pool, "_pool", []) if p.pid]
    total = 0
    for proc in procs:
        try:
            with open(f"/proc/{proc.pid}/statm") as fh:
                total += int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, IndexError, ValueError):
            return len(procs), None
    return len(procs), total


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


def _worker_init(shm_handles, shapes, dtypes, fields, small, fn_args, x64,
                 core_sets, rank_counter, paint_only=False):
    """Pin affinity, then attach shm views and build the jitted tile program.

    `x64` replicates the parent's jax_enable_x64 (propagating the caller's choice into a
    fresh spawned process, not making one). `paint_only` skips the coarse-force views and
    the tile kernel build, which `_worker_coarse_task` does not use.
    """
    if core_sets is not None and hasattr(os, "sched_setaffinity"):
        with rank_counter.get_lock():
            rank = rank_counter.value
            rank_counter.value += 1
        os.sched_setaffinity(0, set(core_sets[rank % len(core_sets)]))
    t0 = time.perf_counter()
    import jax

    jax.config.update("jax_enable_x64", bool(x64))
    from inexor.forces import make_tile_force_fn  # noqa: F401  (force arm only)

    views, segs = {}, []
    for name in fields:
        seg = open_segment(shm_handles[name])
        segs.append(seg)
        views[name] = np.ndarray(shapes[name], dtype=dtypes[name], buffer=seg.buf)
    g_coarse = []
    if not paint_only:
        for i in range(3):
            seg = open_segment(shm_handles[f"g{i}"])
            segs.append(seg)
            g_coarse.append(
                np.ndarray(shapes[f"g{i}"], dtype=dtypes[f"g{i}"], buffer=seg.buf))
    one_tile = None if paint_only else make_tile_force_fn(**fn_args)[0]
    _G.update(views=views, small=small, g_coarse=g_coarse, one_tile=one_tile,
              segs=segs, st=None, step=None, init_s=time.perf_counter() - t0)


def _ensure_facade(C):
    """The worker's read-only SlotState, rebuilt once per DISPATCH EPOCH.

    Not per task (its lazy brick->arena index is O(n_arena) to build), and never across an
    epoch, since migrate and repack move the arena and `arena_base` between dispatches.
    """
    if _G["step"] != C["step"]:
        from inexor.state import SlotState

        v, small = _G["views"], _G["small"]
        _G["st"] = SlotState(
            t9=small["t9"], bricks_per_side=small["bricks_per_side"],
            brick_start=v["brick_start"], occupancy=v["occupancy"],
            off=v["off"], w=v["w"], vel_scale=v["vel_scale"],
            arena_base=C["arena_base"], arena_bucket=v["arena_bucket"],
            n_particles=small["n_particles"], ids=v.get("ids"),
        )
        _G["step"] = C["step"]
    return _G["st"]


def _ensure_scratch(M):
    """Attach the migrate scratch segments, re-attached when they are rebuilt.

    Keyed on `mig_id`, which the parent bumps only when it recreates the segments."""
    if _G.get("mig_id") != M["mig_id"]:
        for seg in _G.get("mig_segs", ()):
            seg.close()
        views, segs = {}, []
        for name in M["mig_handles"]:
            seg = open_segment(M["mig_handles"][name])
            segs.append(seg)
            views[name] = np.ndarray(
                M["mig_shapes"][name], dtype=M["mig_dtypes"][name], buffer=seg.buf
            )
        _G["mig"] = views
        _G["mig_segs"] = segs
        _G["mig_id"] = M["mig_id"]
    return _G["mig"]


def _worker_migrate_eject(arg):
    """Eject one slab into its scratch slot, touching no shared bookkeeping.

    `release_arena=False` leaves the arena free list to the parent's serial replay; the slot
    layout preserves the eject output order `_insert_slab` depends on."""
    s, slot, M = arg
    st = _ensure_facade(M)
    scr = _ensure_scratch(M)
    from inexor import eject_jax

    t0 = time.perf_counter()
    calls0 = eject_jax.CALLS
    keep, emig = st._eject_slab(
        s, M["c_drift"], scr["mig_scales"], kernel=M["kernel"], release_arena=False
    )
    nk, ne = len(keep["dest"]), len(emig["dest"])
    cap = scr["mig_dest"].shape[1]
    if nk + ne > cap:
        raise ValueError(
            f"slab {s} ejected {nk + ne} rows against a scratch slot of {cap} -- "
            "the parent sized the window from a stale occupancy"
        )
    scr["mig_dest"][slot, :nk] = keep["dest"]
    scr["mig_dest"][slot, nk : nk + ne] = emig["dest"]
    scr["mig_off"][slot, :nk] = keep["off"]
    scr["mig_off"][slot, nk : nk + ne] = emig["off"]
    scr["mig_w"][slot, :nk] = keep["w"]
    scr["mig_w"][slot, nk : nk + ne] = emig["w"]
    scr["mig_src"][slot, nk : nk + ne] = emig["src"]
    if M["has_ids"]:
        scr["mig_ids"][slot, :nk] = keep["ids"]
        scr["mig_ids"][slot, nk : nk + ne] = emig["ids"]
    rr = 0
    if ne:
        nb = st.bricks_per_side
        d_slab = emig["dest"] // (st.buckets_per_brick * nb * nb)
        disp = (d_slab - s + nb // 2) % nb - nb // 2
        rr = int(np.abs(disp).max())
    return dict(
        kind="eject", s=int(s), slot=int(slot), n_keep=nk, n_emig=ne,
        realized_reach=rr, busy_s=time.perf_counter() - t0, worker=os.getpid(),
        # the parent's eject_jax.CALLS cannot see a worker's calls
        eject_jax_calls=eject_jax.CALLS - calls0,
    )


def _worker_migrate_insert(arg):
    """Insert one destination slab from scratch views.

    Brick payloads are written to shm directly (disjoint per slab); arena spills are returned
    for the parent to claim in serial order, since arena slot assignment is lowest-free-first
    and therefore order-dependent."""
    d, slot_map, M = arg
    st = _ensure_facade(M)
    scr = _ensure_scratch(M)
    t0 = time.perf_counter()
    has_ids = M["has_ids"]
    staged, emig = {}, {}
    for src, slot, nk, ne in slot_map:
        if src == d:
            staged[d] = dict(
                dest=scr["mig_dest"][slot, :nk],
                off=scr["mig_off"][slot, :nk],
                w=scr["mig_w"][slot, :nk],
                ids=scr["mig_ids"][slot, :nk] if has_ids else None,
            )
        emig[src] = dict(
            dest=scr["mig_dest"][slot, nk : nk + ne],
            off=scr["mig_off"][slot, nk : nk + ne],
            w=scr["mig_w"][slot, nk : nk + ne],
            ids=scr["mig_ids"][slot, nk : nk + ne] if has_ids else None,
            src=scr["mig_src"][slot, nk : nk + ne],
        )
    r = int(M["reach_r"])
    consumed = {src: 0 for src, _, _, _ in slot_map}
    spills = []

    def sink(b, dest, off, w, ids):
        spills.append((int(b), dest, off, w, ids))

    n_over = st._insert_slab(
        d, staged, emig, reach=range(-r, r + 1), consumed=consumed,
        scales=scr["mig_scales"], spill_sink=sink,
    )
    return dict(
        kind="insert", d=int(d), n_over=int(n_over), consumed=consumed,
        spills=spills, busy_s=time.perf_counter() - t0, worker=os.getpid(),
    )


def _worker_alive(_):
    """No-op task; its completion proves the initializer finished."""
    return os.getpid()


def _worker_task(arg):
    """One tile of the kick; the writes ride back for the parent to apply."""
    t, bricks, C = arg
    if _G["one_tile"] is None:
        raise RuntimeError(
            "tile task on a paint_only worker: this pool was built without the "
            "force arm, so there is no tile kernel and no coarse mesh here"
        )
    st = _ensure_facade(C)
    from inexor.engine import tile_task

    res = tile_task(st, _G["one_tile"], C, _G["g_coarse"], tuple(t), bricks)
    res["worker"] = os.getpid()
    res["rss_mb"] = _rss_mb()
    return res


def _worker_coarse_task(arg):
    """One coarse-paint chunk: decode -> sub-block integer paint; returns the sub-block.

    The parent accumulates, where integer associativity makes any arrival order bitwise the
    serial one. Mirrors the sub-block branch of `engine.coarse_delta_streamed`."""
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

    Construction rebinds the caller's `SlotState` fields to shm views; `close()` (which `run`
    always calls) gives pool-created fields regular memory back before unlinking, so the state
    outlives the pool. `allocator` fields already in shared memory are adopted without copy.

    `paint_only=True` builds only the coarse-paint half (no coarse force meshes, no tile
    kernels), for consumers like the P(k) card; at large n_coarse the three meshes would not
    fit beside the state. Force entry points then raise.
    """

    def __init__(self, st, cfg, allocator=None, paint_only=False):
        import jax

        if jax.default_backend() != "cpu":
            # CPU workers under a GPU parent would mix non-bitwise-interchangeable FFT backends
            raise ValueError(
                f"TilePool is the CPU lane and the parent backend is "
                f"{jax.default_backend()!r}; the device lane is a different executor"
            )
        self.st = st
        self.workers = int(cfg.tile_workers)
        self._step = 0
        self._C = None
        self._segs = []
        self._adopted = []
        self._names, self._shapes, self._dtypes, self._views = {}, {}, {}, {}
        # ids are shared when present: the pooled migrate reads and writes them
        self._fields = SHM_FIELDS + (("ids",) if st.ids is not None else ())
        self._mig_segs = {}
        self._mig_views = {}
        self._mig_shape = None
        self._mig_id = 0
        self._M = None
        n = int(cfg.n_coarse)
        # Preflight before any segment is created (after, an overrun is a SIGBUS), from the
        # actual arrays. Arrays the allocator already placed in shared memory cost nothing.
        self._alloc = allocator
        self._preshared = {
            f for f in self._fields
            if allocator is not None
            and allocator.segment_of(np.asarray(getattr(st, f))) is not None
        }
        self._shm_demand = {
            f: int(np.asarray(getattr(st, f)).nbytes)
            for f in self._fields if f not in self._preshared
        }
        self.paint_only = bool(paint_only)
        if not self.paint_only:
            self._shm_demand["coarse force g0,g1,g2"] = int(
                3 * n**3 * np.dtype(cfg.np_coarse_dtype).itemsize
            )
        # margin for the migrate scratch, whose (K, R) is unknown until `stage_migrate`
        self._shm_headroom = 1.10 if cfg.tile_workers > 1 else 1.0
        self._shm_backend = shm_backend()
        # fields free their private copy as they are shared; the coarse meshes are fresh
        self._shm_receipt = preflight_shared_memory(
            self._shm_demand, backend=self._shm_backend,
            headroom=self._shm_headroom,
            adopted=set(self._fields) - self._preshared,
        )
        for f in self._fields:
            arr = np.asarray(getattr(st, f))
            seg = None if self._alloc is None else self._alloc.segment_of(arr)
            if seg is not None:
                # already in shared memory: register, do not copy
                self._adopt(f, seg, arr)
                continue
            view = self._share(f, np.asarray(getattr(st, f)))
            setattr(st, f, view)
        # the cached brick->arena index refers to the old arrays
        st._invalidate_arena_index()
        if not self.paint_only:
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
        # Set for the children via the inherited environment (they import numpy before the
        # initializer runs), then restored in the parent.
        saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "JAX_PLATFORMS")}
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["JAX_PLATFORMS"] = "cpu"
        try:
            self._pool = ctx.Pool(
                self.workers, initializer=_worker_init,
                initargs=(self._names, self._shapes, self._dtypes, self._fields,
                          small, fn_args, bool(jax.config.jax_enable_x64),
                          core_sets, rank_counter, self.paint_only),
            )
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        if os.environ.get("INEXOR_LOAD_TRACE"):
            # Barrier first: `ctx.Pool()` returns before `_worker_init` finishes, and a task
            # cannot run until it has.
            self._pool.map(_worker_alive, range(4 * self.workers))
            n, tot = worker_rss_bytes(self._pool)
            avail = available_ram()
            rss = ("RSS unreadable" if tot is None else
                   f"RSS {tot / 1e9:.1f} GB ({tot / 1e9 / max(n, 1):.2f} GB each)")
            mem = "" if avail is None else f"  MemAvailable {avail / 1e9:.1f} GB"
            print(f"  [pool] {n} workers up, {rss}{mem}", flush=True)

    def _adopt(self, key, seg, arr):
        """Register an array the allocator already put in shared memory.

        Zero-copy; `st` already holds the shared view. Not added to `self._segs`: the segment
        belongs to the allocator, and `close()` must not unlink state a caller still uses."""
        self._adopted.append(seg)
        self._names[key], self._shapes[key], self._dtypes[key] = (
            seg.handle, tuple(arr.shape), str(arr.dtype))
        self._views[key] = arr
        return arr

    def _share(self, key, arr=None, shape=None, dtype=None):
        if arr is not None:
            shape, dtype = arr.shape, arr.dtype
        nbytes = max(int(np.dtype(dtype).itemsize * np.prod(shape, dtype=np.int64)), 1)
        seg = create_segment(nbytes, key)
        view = np.ndarray(shape, dtype=dtype, buffer=seg.buf)
        if arr is not None:
            view[...] = arr
        self._segs.append(seg)
        self._names[key], self._shapes[key], self._dtypes[key] = (
            seg.handle, tuple(shape), str(np.dtype(dtype)))
        self._views[key] = view
        return view

    def g_views(self):
        """The three coarse-force shm views, for a caller that can solve INTO them.

        Solving into these saves a parent-side copy of the triple: `stage_step` then sees
        `a is buf` and copies nothing. A list in component order.
        """
        self._refuse_force("g_views")
        return [self._views[f"g{i}"] for i in range(3)]

    def _refuse_force(self, what):
        """Raise if this is a paint-only pool (no force meshes, no tile kernel)."""
        if self.paint_only:
            raise RuntimeError(
                f"{what} on a paint_only TilePool: this pool has no coarse "
                "force meshes and its workers have no tile kernel. Build it "
                "with paint_only=False for the force arm."
            )

    def stage_step(self, g_coarse, C):
        """Publish this step's coarse force meshes and per-step header."""
        self._refuse_force("stage_step")
        for i, g in enumerate(g_coarse):
            a = np.asarray(g)
            buf = self._views[f"g{i}"]
            if a.shape != buf.shape or a.dtype != buf.dtype:
                raise ValueError(
                    f"coarse mesh {i}: {a.shape}/{a.dtype} does not match the "
                    f"pool's {buf.shape}/{buf.dtype} -- the mesh geometry moved "
                    "under a live pool"
                )
            # identity: a solve into `g_views()` already wrote the segment
            if a is not buf:
                buf[...] = a
        self._step += 1
        self._C = dict(C, step=self._step, arena_base=int(self.st.arena_base))

    def imap(self, tasks):
        """Arrival-order iterator of tile results for the staged step."""
        self._refuse_force("imap")
        if self._C is None:
            raise RuntimeError("imap before stage_step: the workers have no header")
        C = self._C
        return self._pool.imap_unordered(_worker_task, [(t, b, C) for t, b in tasks])

    def stage_coarse(self, H):
        """Publish the coarse-paint header. No arrays move; the mesh accumulates parent-side.
        Starts a new epoch so the worker facade is rebuilt after the previous migrate."""
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

    def _dispose_scratch(self):
        for seg in self._mig_segs.values():
            seg.close()
            try:
                seg.unlink()
            except FileNotFoundError:
                pass
        self._mig_segs = {}
        self._mig_views = {}
        self._mig_shape = None

    def stage_migrate(self, c_drift, kernel, r, slot_rows, window):
        """Publish one migrate pass: scratch window, scales snapshot, header.

        Scratch is sized per pass (slab capacities change with `repack`) and recreated only when
        it must grow; `mig_id` tells workers to re-attach. The `vel_scale` snapshot is taken
        before dispatch because inserts rewrite the live array while later ejects must decode
        at pre-pass scales (as in the serial `drift_and_migrate`)."""
        has_ids = self.st.ids is not None
        K, R = int(window), int(slot_rows)
        n_bricks = int(np.atleast_1d(self.st.vel_scale).shape[0])
        cur = self._mig_shape
        if cur is None or cur[0] < K or cur[1] < R or (has_ids and "mig_ids" not in self._mig_views):
            K = max(K, 0 if cur is None else cur[0])
            R = max(R, 0 if cur is None else cur[1])
            self._dispose_scratch()
            spec = dict(
                mig_dest=((K, R), np.int64), mig_off=((K, R, 3), np.uint8),
                mig_w=((K, R, 3), np.int16), mig_src=((K, R), np.int32),
                mig_scales=((n_bricks,), np.float64),
            )
            if has_ids:
                spec["mig_ids"] = ((K, R), np.int32)
            for name, (shape, dtype) in spec.items():
                nbytes = max(int(np.dtype(dtype).itemsize * np.prod(shape, dtype=np.int64)), 1)
                seg = create_segment(nbytes, name)
                self._mig_segs[name] = seg
                self._mig_views[name] = np.ndarray(shape, dtype=dtype, buffer=seg.buf)
            self._mig_shape = (K, R)
            self._mig_id += 1
        self._mig_views["mig_scales"][...] = np.asarray(self.st.vel_scale, dtype=np.float64)
        self._step += 1
        self._M = dict(
            step=self._step, arena_base=int(self.st.arena_base),
            c_drift=float(c_drift), kernel=str(kernel), reach_r=int(r),
            has_ids=has_ids, mig_id=self._mig_id,
            mig_handles={k: s.handle for k, s in self._mig_segs.items()},
            mig_shapes={k: v.shape for k, v in self._mig_views.items()},
            mig_dtypes={k: str(v.dtype) for k, v in self._mig_views.items()},
        )
        self._mig_q = queue.Queue()
        return self._mig_shape

    def submit_eject(self, s, slot):
        self._pool.apply_async(
            _worker_migrate_eject, ((int(s), int(slot), self._M),),
            callback=self._mig_q.put, error_callback=self._mig_q.put,
        )

    def submit_insert(self, d, slot_map):
        self._pool.apply_async(
            _worker_migrate_insert, ((int(d), list(slot_map), self._M),),
            callback=self._mig_q.put, error_callback=self._mig_q.put,
        )

    def next_migrate_result(self):
        """Blocking arrival-order get; re-raises a worker exception here rather than hanging."""
        item = self._mig_q.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        if getattr(self, "_pool", None) is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        # Copy pool-shared fields back to regular memory before unlinking. Allocator-owned
        # fields stay in their (not unlinked) segments; copying them would double the state.
        for f in self._fields:
            if f in self._preshared:
                continue
            cur = getattr(self.st, f, None)
            if cur is not None:
                setattr(self.st, f, np.array(cur, copy=True))
        self.st._invalidate_arena_index()
        self._dispose_scratch()
        for seg in self._segs:
            seg.close()
            try:
                seg.unlink()
            except FileNotFoundError:
                pass
        self._segs = []
