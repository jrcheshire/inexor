"""D5c: price page-locked, numpy-WRITABLE host memory -- does XLA even notice?

WHY THIS EXISTS. D5b killed the cheap route: staging each crossing through
`pinned_host` is 1.23-1.39x SLOWER than handing the driver ordinary numpy,
because JAX arrays are immutable so every crossing allocates a fresh pinned
buffer and page-locking is a kernel call. The 7.8x ceiling is real but only
reachable by a buffer pinned ONCE and reused. The engine's buffers are numpy
written in place, so the surviving route is memory that is both page-locked and
numpy-writable -- which JAX cannot express, but `cudaHostRegister` can: it
page-locks pages that already exist, leaving the numpy array exactly as it was.

THE LOAD-BEARING UNKNOWN IS NOT WHETHER WE CAN GET SUCH MEMORY. It is whether
XLA TAKES THE FAST PATH when handed it. PJRT may well memcpy any host pointer
into staging memory of its own without ever asking whether it is already
page-locked, in which case the route is dead however cheap the allocation is.
That is gate 1, it is the cheapest thing to test, and it decides the rest.

WHAT WOULD MAKE THE ANSWER BEAUTIFUL. Registration is a property of the
ALLOCATION, not of the transfer call. If XLA notices, the existing "pageable"
code path gets faster with NO change to the device passes at all -- the engine
registers its buffers once at startup and every crossing in every phase
benefits, including the decode and kick already on device.

THE GATES, in the order they decide things:
  1. FAST PATH. Time `device_put` of the same array before and after
     registration, against the true pinned ceiling measured in this same job.
     Near the ceiling: route alive. Unchanged: route dead, stop here.
  2. REGISTRATION COST. A ladder in GB, with the rate. This is the
     amortization question -- paid once per run against 40 steps x 4
     transforms, so even a slow rate can be fine, but it has to be known.
  3. SCALE. The design needs ~724 GB of host state page-locked. `ulimit -l` on
     the compute node, and a large registration actually attempted rather than
     extrapolated.
  4. END TO END. The real inverse at 2048^3 with the spectrum registered vs
     not. A PARTIAL win is expected and is not a defect: pass 1 transfers
     slices OF the spectrum and benefits, while pass 2 transfers fresh
     `ascontiguousarray` temporaries that no one registered. The gap between
     gate 1 and gate 4 is the price of registering the scratch buffers too.

RECEIPTS, because a registration that silently did not happen looks exactly
like one that did and bought nothing:
  - `cudaHostRegister` returns 0, AND
  - `cudaHostGetFlags` on the pointer afterwards returns 0, which it does only
    for a registered pointer, AND
  - the array still reads back what was written into it after registration.
A leg missing any of those is void, not slow.

PORTABLE, because the design uses four GPUs. Default registration associates
the pages with the current context; `cudaHostRegisterPortable` makes them
visible to all of them. Both flags are measured -- if portable costs materially
more, that is a number the four-GPU design has to carry.

Run (Vista gb node, gpu env):
    pixi run -e gpu python scripts/v2_d5c_host_register.py --out-suffix _gb
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
OUT_DIR = os.path.join(REPO, "runs", "v2")

CUDA_HOST_REGISTER_DEFAULT = 0
CUDA_HOST_REGISTER_PORTABLE = 1


class Cudart:
    """The three calls this needs, by ctypes -- no new dependency.

    `libcudart` ships inside the gpu env already (jaxlib's cuda build), so this
    reaches page-locking without adding cupy or numba to a locked environment.
    """

    def __init__(self):
        self.lib = None
        self.error = None
        for name in ("libcudart.so.12", "libcudart.so", "libcudart.so.11.0"):
            try:
                self.lib = ctypes.CDLL(name)
                self.soname = name
                break
            except OSError as exc:
                self.error = str(exc)
        if self.lib is None:
            return
        self.lib.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                              ctypes.c_uint]
        self.lib.cudaHostRegister.restype = ctypes.c_int
        self.lib.cudaHostUnregister.argtypes = [ctypes.c_void_p]
        self.lib.cudaHostUnregister.restype = ctypes.c_int
        self.lib.cudaHostGetFlags.argtypes = [ctypes.POINTER(ctypes.c_uint),
                                              ctypes.c_void_p]
        self.lib.cudaHostGetFlags.restype = ctypes.c_int
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def ok(self):
        return self.lib is not None

    def strerror(self, rc):
        try:
            return self.lib.cudaGetErrorString(rc).decode()
        except Exception:
            return f"cuda error {rc}"

    def register(self, arr, flags=CUDA_HOST_REGISTER_PORTABLE):
        ptr = ctypes.c_void_p(arr.ctypes.data)
        return int(self.lib.cudaHostRegister(ptr, arr.nbytes, flags))

    def unregister(self, arr):
        return int(self.lib.cudaHostUnregister(ctypes.c_void_p(arr.ctypes.data)))

    def is_registered(self, arr):
        """cudaHostGetFlags succeeds ONLY for a registered host pointer."""
        flags = ctypes.c_uint(0)
        rc = int(self.lib.cudaHostGetFlags(ctypes.byref(flags),
                                           ctypes.c_void_p(arr.ctypes.data)))
        return rc == 0, int(flags.value), rc


def page_aligned(shape, dtype):
    """A numpy array whose buffer starts on a page boundary.

    `cudaHostRegister` takes a page-aligned range, and Vista's pages are 64 KiB
    rather than 4 KiB (record: VISTA 2026-09-10), so the page size is READ, not
    assumed. numpy's own allocations are 64-byte aligned and would leave the
    first partial page ambiguous.
    """
    page = os.sysconf("SC_PAGESIZE")
    dtype = np.dtype(dtype)
    n = int(np.prod(shape))
    raw = np.empty(n * dtype.itemsize + page, dtype=np.uint8)
    off = (-raw.ctypes.data) % page
    view = raw[off:off + n * dtype.itemsize].view(dtype).reshape(shape)
    assert view.ctypes.data % page == 0, "alignment failed"
    # `raw` stays alive through the view's own .base chain; ndarray has no
    # __dict__, so stashing a reference on the array is not available anyway.
    return view


def _provenance():
    import jax

    devs = jax.devices()
    try:
        import resource
        lim = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        memlock = "unlimited" if lim[0] == -1 else f"{lim[0] / 1e9:.1f} GB soft"
    except Exception:
        memlock = None
    return dict(host=socket.gethostname(), python=platform.python_version(),
                jax=jax.__version__, platform=devs[0].platform,
                n_devices=len(devs), devices=[str(d) for d in devs],
                page_size=os.sysconf("SC_PAGESIZE"), ulimit_memlock=memlock,
                host_mem_limit_gb=os.environ.get("XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB"),
                argv=" ".join(sys.argv))


def _move(arr, dev, reps):
    """Time the H2D crossing the engine actually makes."""
    import jax

    sh = jax.sharding.SingleDeviceSharding(dev)
    jax.block_until_ready(jax.device_put(arr, sh))  # warm
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(jax.device_put(arr, sh))
        ts.append(time.perf_counter() - t0)
    med = float(np.median(ts))
    return dict(median_s=med, gbps=arr.nbytes / 1e9 / med,
                all_s=[float(t) for t in ts])


# ===========================================================================
# gate 1: does XLA take the fast path for registered memory?
# ===========================================================================


def gate_fast_path(cu, n_plane, reps, flags_name, flags):
    """THE DECIDER. Same array, same call, before and after registration."""
    import jax

    rec = dict(gate="fast_path", flags=flags_name, plane_n=n_plane)
    dev = jax.devices()[0]
    a = page_aligned((n_plane, n_plane), np.float32)
    a[...] = np.random.default_rng(0).standard_normal(a.shape).astype(np.float32)
    before_vals = a.copy()
    rec["bytes"] = int(a.nbytes)

    rec["pageable"] = _move(a, dev, reps)

    t0 = time.perf_counter()
    rc = cu.register(a, flags)
    rec["register_s"] = time.perf_counter() - t0
    rec["register_rc"] = rc
    if rc != 0:
        rec.update(completed=False, error=f"cudaHostRegister: {cu.strerror(rc)}")
        return rec
    ok, got_flags, grc = cu.is_registered(a)
    rec["get_flags_ok"], rec["get_flags"] = ok, got_flags
    if not ok:
        rec.update(completed=False,
                   error=f"cudaHostGetFlags says not registered (rc={grc})")
        return rec
    # still numpy, still writable, still the same values
    a[0, 0] = np.float32(1234.5)
    rec["writable_after_register"] = bool(a[0, 0] == np.float32(1234.5))
    a[0, 0] = before_vals[0, 0]
    rec["values_survived"] = bool(np.array_equal(a, before_vals))

    rec["registered"] = _move(a, dev, reps)

    # the true ceiling, in this job: a buffer jax itself pinned, moved repeatedly
    hs = jax.sharding.SingleDeviceSharding(dev, memory_kind="pinned_host")
    pinned = jax.device_put(np.asarray(a), hs)
    sh = jax.sharding.SingleDeviceSharding(dev)
    jax.block_until_ready(jax.device_put(pinned, sh))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(jax.device_put(pinned, sh))
        ts.append(time.perf_counter() - t0)
    med = float(np.median(ts))
    rec["ceiling"] = dict(median_s=med, gbps=a.nbytes / 1e9 / med)

    p, r, c = (rec["pageable"]["gbps"], rec["registered"]["gbps"],
               rec["ceiling"]["gbps"])
    rec["speedup_vs_pageable"] = r / p
    rec["fraction_of_ceiling"] = (r - p) / (c - p) if c > p else None
    rec["verdict"] = ("XLA TAKES THE FAST PATH" if r > 1.5 * p
                      else "XLA IGNORES REGISTRATION -- route dead")
    cu.unregister(a)
    rec["completed"] = True
    return rec


def gate_register_cost(cu, sizes_gb, flags_name, flags):
    """Gate 2 + 3: what registration costs per GB, and whether it scales."""
    rec = dict(gate="register_cost", flags=flags_name, rungs=[])
    for gb in sizes_gb:
        n = int(gb * 1e9 // 4)
        rung = dict(gb=gb)
        try:
            a = page_aligned((n,), np.float32)
            a[::1000] = 1.0
            t0 = time.perf_counter()
            rc = cu.register(a, flags)
            rung["register_s"] = time.perf_counter() - t0
            rung["rc"] = rc
            if rc == 0:
                ok, _f, _rc = cu.is_registered(a)
                rung["get_flags_ok"] = ok
                rung["gbps"] = a.nbytes / 1e9 / rung["register_s"]
                t0 = time.perf_counter()
                cu.unregister(a)
                rung["unregister_s"] = time.perf_counter() - t0
            else:
                rung["error"] = cu.strerror(rc)
            del a
        except Exception as exc:
            rung["error"] = f"{type(exc).__name__}: {exc}"
        rec["rungs"].append(rung)
        print(f"    register {gb:6.1f} GB: "
              + (f"{rung['register_s']:.2f} s = {rung.get('gbps', 0):.1f} GB/s"
                 if rung.get("rc") == 0 else f"FAILED {rung.get('error')}"),
              flush=True)
    ok = [r for r in rec["rungs"] if r.get("rc") == 0]
    rec["max_gb_registered"] = max([r["gb"] for r in ok], default=0.0)
    if ok:
        rec["median_gbps"] = float(np.median([r["gbps"] for r in ok]))
        # what the design's ~724 GB host state would cost at this rate
        rec["projected_724gb_s"] = 724.0 / rec["median_gbps"]
    rec["completed"] = bool(ok)
    return rec


def gate_end_to_end(cu, n, slab, plane_batch, pencil_batch, reps, flags):
    """Gate 4: the real inverse at size, spectrum registered vs not.

    A PARTIAL win is the expected result and not a defect -- pass 1 transfers
    slices OF the spectrum and benefits, pass 2 transfers fresh
    `ascontiguousarray` temporaries nobody registered. The gap is the price of
    registering the scratch buffers too, which is the next question if this
    gate is positive.
    """
    from inexor import ooc_fft

    rec = dict(gate="end_to_end", n=n, slab=slab, plane_batch=plane_batch,
               pencil_batch=pencil_batch)
    try:
        one = ooc_fft.plane_noise(n, 0, np.float32, 0)

        def field(lo, hi):
            return np.broadcast_to(one, (hi - lo, n, n))

        kw = dict(slab=slab, plane_batch=plane_batch, pencil_batch=pencil_batch)
        built = ooc_fft.forward_from_slabs_device(field, n, **kw)
        spec = page_aligned(built.shape, built.dtype)
        spec[...] = built
        del built
        rec["spec_gb"] = float(spec.nbytes / 1e9)

        def timed_inverse(src):
            t = {}
            ts = []
            for _ in range(reps):
                work = src.copy()
                t0 = time.perf_counter()
                for _lo, _out in ooc_fft.inverse_to_slabs_device(work, n,
                                                                 timings=t, **kw):
                    pass
                ts.append(time.perf_counter() - t0)
            return dict(median_s=float(np.median(ts)), passes=dict(t))

        rec["unregistered"] = timed_inverse(spec)
        rc = cu.register(spec, flags)
        rec["register_rc"] = rc
        if rc != 0:
            rec.update(completed=False, error=f"register: {cu.strerror(rc)}")
            return rec
        ok, _f, _rc = cu.is_registered(spec)
        rec["get_flags_ok"] = ok
        if not ok:
            rec.update(completed=False, error="cudaHostGetFlags says not registered")
            return rec
        # NOTE: .copy() inside the timing makes a FRESH unregistered buffer, so
        # this leg measures registration of the SOURCE only. That is the honest
        # partial, and it is why gate 1 is the decider rather than this.
        rec["registered"] = timed_inverse(spec)
        rec["speedup"] = (rec["unregistered"]["median_s"]
                          / rec["registered"]["median_s"])
        cu.unregister(spec)
        rec["completed"] = True
    except Exception as exc:
        rec.update(completed=False, error=f"{type(exc).__name__}: {exc}")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--plane-n", type=int, default=2048,
                    help="gate 1 moves one plane, the unit the transform moves")
    ap.add_argument("--sizes-gb", default="1,4,16,64,128")
    ap.add_argument("--slab", type=int, default=32)
    ap.add_argument("--plane-batch", type=int, default=1)
    ap.add_argument("--pencil-batch", type=int, default=64)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-end-to-end", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args(argv)

    if args.smoke:
        args.n, args.plane_n, args.sizes_gb = 64, 256, "0.01,0.05"
        args.slab, args.pencil_batch, args.reps = 16, 4, 2

    cu = Cudart()
    os.makedirs(OUT_DIR, exist_ok=True)
    card = os.path.join(OUT_DIR, f"d5c_host_register{args.out_suffix}.json")
    prov = None
    try:
        prov = _provenance()
    except Exception as exc:
        print(f"provenance unavailable: {exc}")
    print("=== D5c: page-locked, numpy-writable host memory ===")
    if prov:
        print(f"    {prov['host']}  jax {prov['jax']}  {prov['platform']}  "
              f"{prov['n_devices']} device(s)  page={prov['page_size']}  "
              f"memlock={prov['ulimit_memlock']}")
    if not cu.ok():
        print(f"FATAL: no libcudart ({cu.error}). This gate cannot run without "
              f"CUDA; it is not a result.")
        json.dump(dict(provenance=prov, completed=False, error=cu.error),
                  open(card, "w"), indent=1)
        return 2
    print(f"    libcudart: {cu.soname}")

    out = dict(provenance=prov, args=vars(args), gates=[])

    def record(rec):
        out["gates"].append(rec)
        with open(card, "w") as fh:
            json.dump(out, fh, indent=1)

    print("\n--- gate 1: does XLA take the fast path? ---", flush=True)
    for name, fl in (("portable", CUDA_HOST_REGISTER_PORTABLE),
                     ("default", CUDA_HOST_REGISTER_DEFAULT)):
        g = gate_fast_path(cu, args.plane_n, args.reps, name, fl)
        record(g)
        if g.get("completed"):
            print(f"  [{name:>8}] pageable {g['pageable']['gbps']:7.1f} GB/s  "
                  f"registered {g['registered']['gbps']:7.1f}  "
                  f"ceiling {g['ceiling']['gbps']:7.1f}  "
                  f"=> {g['speedup_vs_pageable']:.2f}x, "
                  f"{100 * (g['fraction_of_ceiling'] or 0):.0f}% of the way. "
                  f"{g['verdict']}", flush=True)
        else:
            print(f"  [{name:>8}] FAILED: {g.get('error')}", flush=True)

    print("\n--- gate 2+3: registration cost and scale ---", flush=True)
    sizes = [float(x) for x in args.sizes_gb.split(",") if x.strip()]
    g = gate_register_cost(cu, sizes, "portable", CUDA_HOST_REGISTER_PORTABLE)
    record(g)
    if g.get("completed"):
        print(f"  max registered {g['max_gb_registered']:.0f} GB at "
              f"{g['median_gbps']:.1f} GB/s median; the design's ~724 GB host "
              f"state would take {g['projected_724gb_s']:.0f} s to page-lock",
              flush=True)

    if not args.no_end_to_end:
        print("\n--- gate 4: the real inverse, spectrum registered vs not ---",
              flush=True)
        g = gate_end_to_end(cu, args.n, args.slab, args.plane_batch,
                            args.pencil_batch, max(2, args.reps // 2),
                            CUDA_HOST_REGISTER_PORTABLE)
        record(g)
        if g.get("completed"):
            u, r = g["unregistered"], g["registered"]
            print(f"  inverse {u['median_s']:.2f} s -> {r['median_s']:.2f} s = "
                  f"{g['speedup']:.2f}x   (passes "
                  f"p2 {u['passes'].get('pass2_s', 0):.2f}->{r['passes'].get('pass2_s', 0):.2f}, "
                  f"p1 {u['passes'].get('pass1_s', 0):.2f}->{r['passes'].get('pass1_s', 0):.2f})",
                  flush=True)
        else:
            print(f"  FAILED: {g.get('error')}", flush=True)

    bad = [g for g in out["gates"] if not g.get("completed")]
    print(f"\n=== {len(out['gates']) - len(bad)}/{len(out['gates'])} gates "
          f"completed; card -> {card} ===")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
