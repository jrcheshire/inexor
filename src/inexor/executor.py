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

# POSIX shared memory lives on a tmpfs, and its size is a SEPARATE budget from
# host RAM -- the kernel default is half of it. `inexor.plan` priced the run
# against the node's RAM and said FITS at 0.74x while the pool's share alone
# was 1.16x of the tmpfs, because no table had ever named this budget. Vista
# 920910 found it the expensive way: the 2048^3 ICs generated fine (rc=0, 90
# min) and stepping took a SIGBUS 196 s later, inside pool construction,
# having asked for 148.5 GB of a measured 127.6 GB /dev/shm (job 922332).
#
# The failure mode is why it has to be checked UP FRONT: `SharedMemory` sizes
# lazily, so `create=True` succeeds for a segment the tmpfs cannot back and
# the process dies on first TOUCH, with a bus error and no traceback.
SHM_DIR = "/dev/shm"

# Every array field of SlotState the step reads or writes. `ids` is absent on
# purpose: the tile loop never touches ids, so they stay parent-side and the
# executor-identity gate proves they evolve identically. The POOLED MIGRATE
# does touch ids (eject reads them, insert writes them), so a pool built on a
# state that carries ids shares them too -- that is the per-instance `_fields`
# list, not this constant, which keeps its meaning (and its script consumers).
SHM_FIELDS = ("off", "w", "occupancy", "brick_start", "vel_scale", "arena_bucket")

# The migrate scratch (`stage_migrate`): one rolling window of K slab slots,
# each holding a slab's ejected rows as keep-prefix + emig-suffix in the eject
# output order (global brick-major -- ORDER IS CORRECTNESS, `_insert_slab`
# walks it). `mig_scales` is the pass-start vel_scale snapshot: it rides shm
# because the header is pickled per task and the snapshot is 8 B x n_bricks.

# Bytes per (K, R) entry of the migrate scratch: int64 dest + 3 x uint8 off +
# 3 x int16 w + int32 src. `mig_scales` is n_bricks, not K x R, so it is
# priced separately; `mig_ids` adds 4 when the state carries ids.
MIG_BYTES_PER_ENTRY = 8 + 3 * 1 + 3 * 2 + 4
MIG_IDS_BYTES_PER_ENTRY = 4


def shm_capacity(path=SHM_DIR):
    """(total, available) bytes of the tmpfs backing POSIX shared memory.

    `f_bavail`, not `f_bfree`: the unprivileged figure is the one this process
    can actually get, and a run that dies for want of the root reserve dies
    just as hard. Returns (None, None) where there is no such mount, which is
    every non-Linux developer machine -- the caller then cannot check, and
    says so rather than inventing a budget.
    """
    try:
        s = os.statvfs(path)
    except (OSError, AttributeError):
        return None, None
    return s.f_blocks * s.f_frsize, s.f_bavail * s.f_frsize


def shm_terms(*, n_rows, index_bytes, n_arena, n_bricks, n_coarse,
              coarse_itemsize, has_ids=False, mig_window=None, mig_rows=None):
    """The model of what `TilePool` puts in shared memory, term by term.

    Geometry in, bytes out, so `inexor.plan` can price a configuration that
    has never been built. The pool itself does NOT use this: it measures its
    own arrays (`_shm_demand`), because a model standing between the check
    and the allocation is a model that can be wrong in the direction that
    matters. `test_executor.py` holds the two against each other.
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
    """Refuse now, with a table, rather than take a bus error on first touch.

    `headroom` is a MULTIPLIER on the demand, not a subtracted constant: the
    thing left out of `terms` (the migrate scratch, when the caller prices
    only construction) scales with the run, not with the machine.

    Returns the (demand, available) pair when it fits, so a caller can record
    what it was standing on. Raises `MemoryError` when it does not, and does
    nothing at all where the tmpfs cannot be read -- an absent check must not
    read as a passed one, so it says which of the two happened.
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
    """`memfd_create`, from wherever this interpreter can reach it.

    **`hasattr(os, "memfd_create")` IS NOT THE CAPABILITY.** CPython gates
    that attribute on a configure-time check against the BUILD sysroot, and
    conda-forge builds against an old one: the gpu env's Python 3.14.6 has no
    `os.memfd_create` while the system Python 3.9 on the same Vista node
    does, and the kernel has had the syscall throughout. Trusting the
    attribute silently put job 922557 back on the capped mount after the
    whole point of the change was to leave it.

    The runtime libc has the symbol regardless, so go through it. Cached
    because the miss path is a `CDLL` load. Returns None where there really
    is no memfd, which is macOS.
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
    """Whether this process can actually create a memfd, by CREATING one.

    A probe, not an attribute test, for the reason above: the attribute lied
    on the exact machine this runs on. One syscall, cached.
    """
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

    The budget for the `memfd` path. Not `MemFree`: page cache is reclaimable
    and counting it as spoken-for would refuse runs that fit.
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
    """Bytes this process must find ON TOP of what it already holds.

    **The demand is not the cost, and job 922682 was refused for the
    difference.** The state arrays already exist in the parent when the pool
    is built: `_share` copies each into a segment and rebinds `st`, dropping
    the last reference to the private array, so an ADOPTED field releases its
    own bytes as it is copied and nets to zero. Only two things are new --
    the one field duplicated while its copy is in progress, and the segments
    with no private counterpart (the coarse meshes).

    So the peak above baseline is `max(largest adopted field, sum of fresh)`,
    not `sum(everything)`. At c-gh that is the difference between 78.3 GB and
    161.4 GB, and the second number refused a run the node could hold.

    This is a MODEL of the release, and it is the one assumption here worth
    naming: it holds only while nothing outside `st` still references those
    arrays. `TilePool` is constructed before the step loop for exactly that
    reason.
    """
    fresh = {k: v for k, v in terms.items() if k not in adopted}
    biggest_adopted = max((v for k, v in terms.items() if k in adopted), default=0)
    return max(biggest_adopted, sum(fresh.values()))


def preflight_shared_memory(terms, backend=None, headroom=1.0, adopted=()):
    """Check the demand against whichever budget the backend actually has.

    THE TWO BACKENDS HAVE DIFFERENT CEILINGS, and using the wrong one is how
    this check would become decoration: `/dev/shm` is a hard tmpfs cap that
    `memfd` is exempt from, while `memfd` is bounded by RAM, which `/dev/shm`
    charges against as well but is not the first thing it hits.

    They also differ in what ADOPTION buys. The tmpfs must hold every segment
    at once no matter what the parent is holding, so posix is checked on the
    total. RAM is not: an adopted field frees its private copy as it goes, so
    memfd is checked on `adoption_peak`.

    Returns (demand, budget, what) so a caller can record what it stood on;
    `what` is None when nothing could be read, because an absent check must
    not be recorded as a passed one.
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

    **`memfd` is the reason the c-gh run fits at all.** Both mechanisms hand
    out the same thing -- anonymous RAM-backed shared pages, no device, no
    I/O -- but `shm_open` puts them on the `/dev/shm` mount, whose size is
    capped (half of RAM by default: 127.55 GB on a Vista gg node), while
    `memfd_create` uses the kernel's internal shm mount, which has no size
    limit at all. MEASURED on Vista: 2.147 GB of memfd raised `Shmem` by
    2.139 GB (1.00x, so it really is those pages) and moved `/dev/shm` used
    by 0.000. The state was never too big for the node; it was too big for
    one doorway.

    `posix` remains for every machine without `memfd_create`, which is every
    macOS developer box, so the local suite exercises the fallback and the
    cluster exercises the production path. The `/dev/shm` budget check
    applies to that path ONLY -- see `check_shm_budget`.

    `INEXOR_SHM_BACKEND` forces one. It exists because the choice is
    otherwise made by the platform, which means the memfd path CANNOT be
    exercised where this is developed and the posix path cannot be exercised
    where it runs -- a gate neither machine can fail. It is also the escape
    hatch if a node turns out not to expose `/proc/<pid>/fd` to a job.
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

    The handle is what crosses the process boundary. For `posix` it is the
    segment's name, as before. For `memfd` it is `(pid, fd)`, because the
    object HAS no name -- a worker reaches it through `/proc/<pid>/fd/<n>`,
    which is a second reference to the same file, not a copy. Verified with a
    spawned interpreter: the child read what the parent wrote and the parent
    saw the child's write back.
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
        """No-op for memfd: it is ANONYMOUS, so it dies with its last
        reference and there is no name left behind to remove. 920910 leaked
        six `/dev/shm` segments on the way down; this kind cannot."""
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
        # a NEW descriptor onto the same object; the mapping outlives it, and
        # holding it open would pin a descriptor per worker per segment
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

    `free()` does not shrink the process. After the slab loader releases ~86
    GB of per-slab payload the allocator keeps those arenas, and the pool's
    segments are FRESH KERNEL PAGES that cannot be served from them -- so the
    two costs stack instead of cancelling. Job 922723 was OOM-killed 41 s into
    pool construction with exactly that shape.

    Not a no-op by assumption: it reports whether the call happened, so a
    caller can record "trimmed" rather than infer it. glibc only; returns
    False on macOS, where the allocator is different and this question does
    not arise.
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
    """Hands out ndarray views backed by shared segments, so a producer can
    write the state STRAIGHT into the memory the workers will read.

    **This is what stops the state existing twice.** Before it, the loader
    built ~135 GB of private arrays and `TilePool` then copied them into
    another ~135 GB of segments, so a node had to hold both at once; at c-gh
    on a 255 GB box, with the loader's own peak on top, it did not.

    The registry is by `id()` of the view, which is sound only because the
    allocator OWNS those buffers and keeps them alive for its lifetime -- an
    id is reusable once its object dies, and none of these die early.
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
        """The segment backing `arr`, or None if this allocator did not make
        it. Identity, not equality: a copy of a shared array is not shared."""
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

    THE PARENT CANNOT SEE THEM ANY OTHER WAY -- `ru_maxrss` is per-process,
    and at cdev8 the workers held 21x the parent. The existing path rides
    `rss_mb` back on tile results, which is a step too late: job 922905 died
    in the nineteen seconds between the workers spawning and the first tile,
    with MemAvailable going 74.1 -> 0.0 GB. Reading /proc needs no dispatch,
    so it works before any task has run.

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
                 core_sets, rank_counter):
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
    for name in fields:
        seg = open_segment(shm_handles[name])
        segs.append(seg)
        views[name] = np.ndarray(shapes[name], dtype=dtypes[name], buffer=seg.buf)
    g_coarse = []
    for i in range(3):
        seg = open_segment(shm_handles[f"g{i}"])
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
            n_particles=small["n_particles"], ids=v.get("ids"),
        )
        _G["step"] = C["step"]
    return _G["st"]


def _ensure_scratch(M):
    """Attach the migrate scratch segments, re-attached when they are rebuilt.

    Keyed on `mig_id`, which the parent bumps only when a segment is recreated
    (the slot window or the slab capacity grew) -- names are stable otherwise,
    so the common pass re-uses the open handles."""
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
    """Eject one slab into its scratch slot. NO shared bookkeeping is touched:
    `release_arena=False` leaves the arena free list to the parent's serial
    replay, and the slot's keep-prefix + emig-suffix layout preserves the eject
    output order that `_insert_slab` depends on."""
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
        # the compiled-kernel receipt travels WITH the result: the parent's own
        # eject_jax.CALLS cannot see a worker's, so a pooled card reading the
        # parent counter would always say 0 and look like a broken instrument
        eject_jax_calls=eject_jax.CALLS - calls0,
    )


def _worker_migrate_insert(arg):
    """Insert one destination slab from scratch views. Brick payloads land in
    shm directly (the C13-censused disjoint writes); arena SPILLS come back as
    rows for the parent to claim at this brick's serial point, because slot
    assignment is lowest-free-first and therefore order-dependent."""
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
    """A no-op whose only job is to prove the initializer finished."""
    return os.getpid()


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

    def __init__(self, st, cfg, allocator=None):
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
        self._adopted = []
        self._names, self._shapes, self._dtypes, self._views = {}, {}, {}, {}
        # ids join the shared set only when the state carries them: the tile
        # loop never touches ids, but the pooled migrate reads them at eject
        # and writes them at insert, and a facade with `ids=None` would strand
        # the parent's column silently.
        self._fields = SHM_FIELDS + (("ids",) if st.ids is not None else ())
        self._mig_segs = {}
        self._mig_views = {}
        self._mig_shape = None
        self._mig_id = 0
        self._M = None
        n = int(cfg.n_coarse)
        # BEFORE the first `SharedMemory(create=True)`, because after it the
        # failure is a bus error on touch rather than an exception. Measured
        # from the arrays themselves, not modelled: this is the check, and a
        # check that prices something other than what it is about to allocate
        # is the shape of a gate that cannot fail.
        # Arrays the allocator already made are ALREADY in shared memory: the
        # loader wrote them there. They cost nothing to adopt and must not be
        # priced as if they did, or the check refuses a run whose whole point
        # was to avoid the second copy.
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
        self._shm_demand["coarse force g0,g1,g2"] = int(
            3 * n**3 * np.dtype(cfg.np_coarse_dtype).itemsize
        )
        # The migrate scratch is staged per step and its (K, R) is not known
        # until the first `stage_migrate`, so it cannot be measured here. It
        # is real all the same, so the construction demand is held to a
        # margin rather than to the bare ceiling.
        self._shm_headroom = 1.10 if cfg.tile_workers > 1 else 1.0
        self._shm_backend = shm_backend()
        # `self._fields` are ADOPTED: they exist in the parent right now and
        # each releases its private copy as `_share` rebinds it. The coarse
        # meshes are fresh. That distinction is the whole check on the memfd
        # path -- see `adoption_peak`.
        self._shm_receipt = preflight_shared_memory(
            self._shm_demand, backend=self._shm_backend,
            headroom=self._shm_headroom,
            adopted=set(self._fields) - self._preshared,
        )
        for f in self._fields:
            arr = np.asarray(getattr(st, f))
            seg = None if self._alloc is None else self._alloc.segment_of(arr)
            if seg is not None:
                # already in shared memory; register its handle and DO NOT copy
                self._adopt(f, seg, arr)
                continue
            view = self._share(f, np.asarray(getattr(st, f)))
            setattr(st, f, view)
        # the parent's cached brick->arena index maps into the OLD array;
        # values are equal but the invariant is identity, so rebuild lazily
        st._invalidate_arena_index()
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
                initargs=(self._names, self._shapes, self._dtypes, self._fields,
                          small, fn_args, bool(jax.config.jax_enable_x64),
                          core_sets, rank_counter),
            )
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        if os.environ.get("INEXOR_LOAD_TRACE"):
            # AFTER A BARRIER. `ctx.Pool()` returns when the processes exist,
            # not when `_worker_init` has finished importing jax and building
            # kernels, so reading /proc here measured newborn processes: job
            # 922991 reported 0.01 GB per worker and then died as they grew.
            # A dispatched task cannot run until the initializer has returned.
            self._pool.map(_worker_alive, range(4 * self.workers))
            n, tot = worker_rss_bytes(self._pool)
            avail = available_ram()
            rss = ("RSS unreadable" if tot is None else
                   f"RSS {tot / 1e9:.1f} GB ({tot / 1e9 / max(n, 1):.2f} GB each)")
            mem = "" if avail is None else f"  MemAvailable {avail / 1e9:.1f} GB"
            print(f"  [pool] {n} workers up, {rss}{mem}", flush=True)

    def _adopt(self, key, seg, arr):
        """Register an array the allocator already put in shared memory.

        The zero-copy path. `st` is not rebound because the array it holds IS
        the shared view.

        Deliberately NOT added to `self._segs`: that list is what `close()`
        closes and unlinks, and these segments belong to the allocator. The
        pool freeing them would pull the state out from under a caller that
        is still using it -- and a segmented run resumes into exactly that."""
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

        `TilePool` allocates these at construction and the workers read them
        every tile, so a caller that writes its solve straight here saves the
        whole parent-side triple: `stage_step` then finds `a is buf` and copies
        nothing. At C-gh that is 12.9 GB of the 25.8 the old jax-list-then-numpy
        -copy shape carried. Returned as a list in component order.
        """
        return [self._views[f"g{i}"] for i in range(3)]

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
            # `is`, not `==`: a caller that solved into `g_views()` has already
            # written the segment, and `buf[...] = buf` would be a full
            # self-copy of a 4.3 GB mesh for nothing. Identity is the right test
            # because it is exactly the question being asked -- did this array
            # come from here.
            if a is not buf:
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

        Scratch is sized PER PASS (`repack` moves `brick_start`, so slab
        capacities change between steps) and recreated only when the
        requirement grows; `mig_id` tells workers when to re-attach. The
        `vel_scale` snapshot is written here, before any task is dispatched --
        inserts rewrite the live array while later ejects must decode at
        pre-pass scales, exactly the serial pass's copy at state.py's
        `drift_and_migrate`."""
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
        """Blocking arrival-order get; a worker exception is re-raised HERE, in
        the driver's loop, so a failed pass dies loudly instead of hanging the
        backpressure window."""
        item = self._mig_q.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        if getattr(self, "_pool", None) is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        # give the caller's state regular memory back BEFORE unlinking, or the
        # arrays would be views into freed segments
        for f in self._fields:
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
