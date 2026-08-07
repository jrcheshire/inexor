"""v2 gate G4: does a Vista GH200 reach LPDDR-resident state coherently?

The question (plan-plan Sec V3): C-gh is 2048^3 particles. At the ratified
T9 state tier (D-v2-8, 9 B/p) that is 77.3 GB of compressed state alone --
81% of the GH200's 96 GB HBM before a single mesh exists. So C-gh is only
real if the 116 GB of Grace LPDDR is reachable at useful bandwidth. The
design study's E2 card measured the ceilings (HBM 3.4 TB/s, LPDDR 486 GB/s,
C2C 375/297 GB/s) and noted that JAX documents `pinned_host` offload only,
with the coherent/ATS path undocumented. This measures which of the three
regimes we actually get.

Three arms, ONE estimator (same op, same chunking, same warmup, same reps):
  hbm       : chunks device-resident. The ceiling control. EXPECTED to OOM
              once the working set passes HBM -- that failure is a designed
              output, not an accident (it calibrates where the cliff is).
  staged    : chunks live in `pinned_host`; each is device_put to HBM, used,
              and dropped. The documented path. Should land near C2C.
  coherent  : chunks live in host memory and are handed straight to jnp with
              no explicit staging. The undocumented path under test.

THE TRAP THIS IS BUILT AROUND: `coherent` can silently BE `staged`. If XLA
inserts a copy to HBM behind the scenes, a naive reading reports "the
coherent path works" while having measured a staged copy. So the arm carries
independent witnesses rather than a self-report (umbrella
reference-knob-must-prove-it-applied):

  (1) CAPACITY (primary, and hard to fake): run the ladder PAST 96 GB. An arm
      that completes a 128 GB working set cannot have had it all resident in
      96 GB of HBM. Binary, and it needs no bandwidth model to interpret.
  (2) HBM RESIDENCY: peak_bytes_in_use over the arm. If it tracks the whole
      working set, the data was copied wholesale; if it tracks ONE chunk, the
      set streamed.
  (3) BANDWIDTH vs the three known ceilings. A rate above C2C did not cross
      C2C per byte; a rate at HBM speed means the data was resident and the
      arm is lying.

Any one of these can be argued with. Together they pin the regime.

Sizing note: the ladder is a bandwidth-bound streaming reduction, so bytes
touched == working-set bytes EXACTLY and GB/s needs no model. `--real` then
runs the G1 CIC paint at the C-gh particle count, which is the claim itself
rather than a proxy.

Orchestration: one (arm, size) per fresh subprocess -- `peak_bytes_in_use` is
a running max that never resets, so two arms in one process report the same
number (umbrella reference-jax-peak-memory-no-reset).

Also runs on Stampede3 h100 nodes, where the same ladder measures a
PCIe-attached host instead of a C2C-attached one -- the comparison that
decides C-hero vs gb at V4. HBM is DETECTED per node, not assumed.

Run (Vista gh node, gpu env):
    pixi run -e gpu python scripts/v2_g4_gh_memory.py
Gate leg / reachability (small ladder under the cliff, used as leg 1):
    pixi run -e gpu python scripts/v2_g4_gh_memory.py --smoke
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
OUT_DIR = os.path.join(REPO, "runs", "v2")

ARMS = ("hbm", "staged", "coherent")

# Measured GH200 ceilings, design study E2 card (Schieffer+ 2407.07850,
# Fusco+ 2408.11556). Reference points for reading a rate, NOT gates.
CEILING_GBS = {"hbm": 3400.0, "lpddr": 486.0, "c2c_read": 375.0, "c2c_write": 297.0}

# The cliff the ladder must cross. DETECTED, not assumed: this probe now runs
# on GH200 (96 GB HBM3) and on Stampede3 H100 nodes, and a hardcoded 96 GiB
# would silently answer `exceeds_hbm` and the device-cap precondition against
# the WRONG card -- i.e. the capacity witness would be measured against a
# reference that is not the hardware. Overridable with --hbm-gib.
HBM_BYTES_FALLBACK = 96 * 1024**3


def physical_hbm_bytes(override_gib=None):
    """Physical HBM of device 0, from nvidia-smi; fallback is GH200's 96 GiB."""
    if override_gib:
        return int(override_gib * 1024**3)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip().splitlines()
        if out:
            return int(float(out[0].strip()) * 1024**2)  # MiB -> bytes
    except Exception:
        pass
    return HBM_BYTES_FALLBACK


HBM_BYTES = physical_hbm_bytes()

# Working-set ladder in GiB. Straddles HBM deliberately: 64/88 below, 104/128
# above, so the hbm arm's OOM brackets the cliff instead of merely reporting it.
DEFAULT_LADDER = "64,88,104,128"
CHUNK_GIB = 2.0  # one streamed chunk; also the residency witness's unit


def _mem_of_kind(dev, kind):
    """The device's Memory object for `kind`, or None if unsupported here."""
    for m in dev.addressable_memories():
        if m.kind == kind:
            return m
    return None


def _sharding(dev, kind):
    import jax

    if kind == "device":
        return jax.sharding.SingleDeviceSharding(dev)
    return jax.sharding.SingleDeviceSharding(dev, memory_kind=kind)


def _host_kind(dev):
    """Which host memory kind this stack exposes.

    `unpinned_host` is the one that can sit in plain pageable Grace LPDDR;
    `pinned_host` is the documented offload target (page-locked staging).

    MEASURED 2026-08-06 on Vista (job 894005, jax 0.10.2, GH200 120GB):
    this CUDA stack exposes ONLY ['device', 'pinned_host'] -- there is no
    `unpinned_host`, though CPU JAX on the laptop reports all three. So the
    `coherent` arm falls back to pinned_host here, and what it then tests is
    IMPLICIT vs EXPLICIT staging out of page-locked host memory, NOT ATS/
    coherent access to ordinary LPDDR. The seed's "coherent vs staged"
    dichotomy is not expressible through memory kinds on this jax. Callers
    must read `host_kind_used` before calling anything "coherent"; the
    summary prints the caveat when it fires.
    """
    for kind in ("unpinned_host", "pinned_host"):
        if _mem_of_kind(dev, kind) is not None:
            return kind
    return None


# ===========================================================================
# worker: one arm at one working-set size, in its own process
# ===========================================================================


def _make_jit_sum(dev, dtype):
    """A sum whose INPUT stays in host memory and whose OUTPUT is on device.

    This is the coherent arm, and it has to be spelled this way. MEASURED
    (job 894005): calling jnp.sum directly on a pinned_host array raises
        INVALID_ARGUMENT: E1200: CompileTimeHostOffloadOutputLocationMismatch
    because the result of an op on host-resident data defaults to host memory
    space while the caller wants it on device. Declaring out_shardings on the
    device memory space resolves the mismatch and hands XLA the whole
    transfer decision -- which is precisely the thing under test: if this
    compiles and runs, XLA moved the bytes without us staging them, and the
    residency + capacity witnesses say whether it streamed or copied.
    """
    import jax
    import jax.numpy as jnp

    return jax.jit(
        lambda c: jnp.sum(c, dtype=dtype),
        out_shardings=_sharding(dev, "device"),
    )


def _alloc_chunks(dev, kind, n_chunks, chunk_elems):
    """Allocate the working set as a LIST of chunks in memory space `kind`.

    Chunked rather than one contiguous array on purpose: it is what a streamed
    state actually looks like, and it avoids ever holding a second full-size
    host copy (a 128 GiB numpy staging buffer plus its device_put would exceed
    the node). Each chunk is generated on device and placed, so the host never
    materializes the whole set either.
    """
    import jax
    import jax.numpy as jnp

    sh = _sharding(dev, kind)
    chunks = []
    for i in range(n_chunks):
        # iota+offset: cheap, deterministic, and it defeats any zero-page or
        # compression trick that would make the transfer unrepresentative
        c = (jnp.arange(chunk_elems, dtype=jnp.float32) + np.float32(i)) * np.float32(1.0000001)
        chunks.append(jax.block_until_ready(jax.device_put(c, sh)))
        del c
    return chunks


def _stream_reduce(chunks, arm, dev, jit_sum=None):
    """Sum every chunk. Bytes touched == working-set bytes, exactly.

    Identical arithmetic in all three arms, so a wall difference is a memory-
    path difference and nothing else. What differs is WHO moves the bytes:
      hbm      : nothing to move, the chunk is already in HBM.
      staged   : we device_put each chunk, use it, and drop it. MEASURED
                 (job 894005): without forcing completion per chunk, XLA's
                 async dispatch keeps every staged copy alive and device peak
                 equals the WHOLE working set (pk/set read exactly 1.000 at
                 both smoke sizes). That is not a slow streamer, it is not a
                 streamer at all -- above the HBM cliff it would OOM for a
                 reason unrelated to the memory path and the ladder would
                 discriminate nothing. So block per chunk, then drop.
      coherent : XLA moves them. The chunk stays in host memory and a jitted
                 sum with its OUTPUT declared on device does the transfer
                 internally (see _make_jit_sum).
    """
    import jax
    import jax.numpy as jnp

    dev_sh = _sharding(dev, "device")
    total = jnp.zeros((), jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)
    for c in chunks:
        if arm == "staged":
            on_dev = jax.device_put(c, dev_sh)
            part = jax.block_until_ready(jnp.sum(on_dev, dtype=total.dtype))
            del on_dev  # the block above is what makes this release effective
            total = total + part
        elif arm == "coherent":
            total = total + jit_sum(c)
        else:
            total = total + jnp.sum(c, dtype=total.dtype)
    return jax.block_until_ready(total)


def run_single(args):
    import jax

    dev = jax.devices()[0]
    host_kind = _host_kind(dev)
    kinds = [m.kind for m in dev.addressable_memories()]

    rec = dict(
        arm=args.arm,
        working_set_gib=args.gib,
        chunk_gib=args.chunk_gib,
        platform=dev.platform,
        device=str(dev.device_kind),
        memory_kinds_available=kinds,
        host_kind_used=None,
        jax_version=jax.__version__,
    )

    # arm -> memory space the WORKING SET lives in
    if args.arm == "hbm":
        kind = "device"
    else:
        kind = host_kind
        rec["host_kind_used"] = kind
        if kind is None:
            rec["error"] = "no host memory kind exposed by this jax/jaxlib"
            rec["unsupported"] = True
            print("WORKER_JSON " + json.dumps(rec))
            return

    chunk_elems = int(args.chunk_gib * 1024**3) // 4  # f32
    n_chunks = max(1, int(round(args.gib / args.chunk_gib)))
    set_bytes = n_chunks * chunk_elems * 4
    rec.update(n_chunks=n_chunks, chunk_elems=chunk_elems, working_set_bytes=set_bytes)

    def peak():
        try:
            return (dev.memory_stats() or {}).get("peak_bytes_in_use")
        except Exception:
            return None

    # --- the caps, recorded BEFORE anything is allocated ---
    # Job 894010 was invalidated by exactly this: both ceilings it "measured"
    # were software defaults, not the GH200. The host arms hit
    # XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB (default 64 GB) on a node with 192 GB
    # free, and the device arm died at a 68 GiB peak against 95.6 GiB of HBM.
    # An env var that is SET is not a cap that APPLIED, so the limits are read
    # back off the device and the run is refused if they cannot cover the
    # ladder (umbrella reference-knob-must-prove-it-applied).
    stats = {}
    try:
        stats = dev.memory_stats() or {}
    except Exception:
        pass
    rec["device_bytes_limit"] = stats.get("bytes_limit")
    rec["env_host_limit_gb"] = os.environ.get("XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB")
    rec["env_mem_fraction"] = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")
    rec["env_preallocate"] = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")

    try:
        import jax.numpy as jnp

        t0 = time.perf_counter()
        chunks = _alloc_chunks(dev, kind, n_chunks, chunk_elems)
        rec["alloc_s"] = time.perf_counter() - t0
        rec["peak_after_alloc"] = peak()

        jit_sum = _make_jit_sum(dev, jnp.float32) if args.arm == "coherent" else None
        _stream_reduce(chunks, args.arm, dev, jit_sum)  # warmup + compile
        rec["peak_after_warmup"] = peak()

        walls = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            _stream_reduce(chunks, args.arm, dev, jit_sum)
            walls.append(time.perf_counter() - t0)

        wall = float(np.median(walls))
        rec.update(
            completed=True,
            wall_median_s=wall,
            wall_min_s=float(np.min(walls)),
            wall_all_s=[float(w) for w in walls],
            gbytes_per_s=(set_bytes / wall) / 1e9,
            peak_total=peak(),
        )
        # --- witness 2: HBM residency ---
        # whole set resident -> ~1.0; one chunk at a time -> ~chunk/set
        pk = rec["peak_total"]
        if pk:
            rec["hbm_peak_over_working_set"] = pk / set_bytes
            rec["hbm_peak_over_chunk"] = pk / (chunk_elems * 4)
        # --- witness 1: capacity ---
        rec["exceeds_hbm"] = bool(set_bytes > HBM_BYTES)
        rec["capacity_witness"] = bool(set_bytes > HBM_BYTES)  # completed AND over HBM
    except Exception as exc:  # OOM here is a RESULT, not a crash
        txt = str(exc)
        rec.update(
            completed=False,
            error=txt.strip().splitlines()[0][:300] if txt.strip() else repr(exc),
            oom=("RESOURCE_EXHAUSTED" in txt or "Out of memory" in txt or "out of memory" in txt),
            peak_total=peak(),
        )

    print("WORKER_JSON " + json.dumps(rec))


# ===========================================================================
# the real point: G1 CIC paint at the C-gh particle count
# ===========================================================================


def run_real(args):
    """Paint C-gh's particle load from host-resident positions.

    This is the claim itself rather than a proxy: 2048^3 particles is 77.3 GB
    of T9 state, and the positions streamed here are the f32 stand-in for it.
    Same three arms, same witnesses; the op is the G1 paint so the number is
    commensurate with the cost-of-memory record.
    """
    import jax
    import jax.numpy as jnp

    from inexor.painting import paint_int

    dev = jax.devices()[0]
    host_kind = _host_kind(dev)
    kind = "device" if args.arm == "hbm" else host_kind

    n_side, n_mesh = args.n_side, args.n_mesh
    n_tot = n_side**3
    per_chunk = int(args.chunk_gib * 1024**3) // 12  # f32 (n,3)
    n_chunks = max(1, (n_tot + per_chunk - 1) // per_chunk)
    L = float(n_mesh)

    rec = dict(
        arm=args.arm,
        real=True,
        n_side=n_side,
        n_mesh=n_mesh,
        n_particles=n_tot,
        n_chunks=n_chunks,
        chunk_gib=args.chunk_gib,
        positions_bytes=n_tot * 12,
        t9_state_bytes=n_tot * 9,  # the quantity C-gh actually has to hold
        memory_kinds_available=[m.kind for m in dev.addressable_memories()],
        host_kind_used=None if args.arm == "hbm" else kind,
        jax_version=jax.__version__,
    )
    if kind is None:
        rec.update(error="no host memory kind exposed", unsupported=True)
        print("WORKER_JSON " + json.dumps(rec))
        return

    def peak():
        try:
            return (dev.memory_stats() or {}).get("peak_bytes_in_use")
        except Exception:
            return None

    try:
        sh = _sharding(dev, kind)
        dev_sh = _sharding(dev, "device")
        rng = np.random.default_rng(0)
        chunks = []
        remaining = n_tot
        for _ in range(n_chunks):
            m = min(per_chunk, remaining)
            arr = (rng.random((m, 3), dtype=np.float32) * np.float32(L)).astype(np.float32)
            chunks.append(jax.block_until_ready(jax.device_put(arr, sh)))
            del arr
            remaining -= m
        rec["peak_after_alloc"] = peak()

        mesh = jnp.zeros((n_mesh**3,), jnp.int32)
        # same three spellings as the ladder: we stage, or XLA does. The
        # coherent arm must declare its output on device or it hits the E1200
        # host-offload location mismatch (job 894005).
        jit_paint = jax.jit(
            lambda p: paint_int(p, n_mesh, L, 12).reshape(-1),
            out_shardings=_sharding(dev, "device"),
        )

        def one_pass():
            acc = mesh
            for c in chunks:
                if args.arm == "staged":
                    pos = jax.device_put(c, dev_sh)
                    part = jax.block_until_ready(paint_int(pos, n_mesh, L, 12).reshape(-1))
                    del pos  # block first, or every staged chunk stays resident
                elif args.arm == "coherent":
                    part = jit_paint(c)
                else:
                    part = paint_int(c, n_mesh, L, 12).reshape(-1)
                acc = acc + part
            return jax.block_until_ready(acc)

        one_pass()  # warmup + compile
        rec["peak_after_warmup"] = peak()
        walls = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            one_pass()
            walls.append(time.perf_counter() - t0)
        wall = float(np.median(walls))
        rec.update(
            completed=True,
            wall_median_s=wall,
            wall_all_s=[float(w) for w in walls],
            positions_gbytes_per_s=(n_tot * 12 / wall) / 1e9,
            particles_per_s=n_tot / wall,
            peak_total=peak(),
        )
        pk = rec["peak_total"]
        if pk:
            rec["hbm_peak_over_positions"] = pk / (n_tot * 12)
        rec["exceeds_hbm"] = bool(n_tot * 12 > HBM_BYTES)
    except Exception as exc:
        txt = str(exc)
        rec.update(
            completed=False,
            error=txt.strip().splitlines()[0][:300] if txt.strip() else repr(exc),
            oom=("RESOURCE_EXHAUSTED" in txt or "Out of memory" in txt or "out of memory" in txt),
            peak_total=peak(),
        )
    print("WORKER_JSON " + json.dumps(rec))


# ===========================================================================
# orchestration
# ===========================================================================


def spawn(arm, gib, chunk_gib, reps, real, n_side, n_mesh):
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--single",
        "--arm",
        arm,
        "--gib",
        str(gib),
        "--chunk-gib",
        str(chunk_gib),
        "--reps",
        str(reps),
    ]
    if real:
        cmd += ["--real", "--n-side", str(n_side), "--n-mesh", str(n_mesh)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("WORKER_JSON ")]
    if lines:
        return json.loads(lines[-1][len("WORKER_JSON ") :])
    # the worker died before it could report (a kernel-level OOM kills the
    # process outright); keep the whole stderr and surface the real line
    err_dir = os.path.join(OUT_DIR, "g4_errs")
    os.makedirs(err_dir, exist_ok=True)
    tag = f"{arm}_{'real' if real else int(gib)}"
    with open(os.path.join(err_dir, f"{tag}.err"), "w") as fh:
        fh.write(p.stderr or "")
    el = (p.stderr or "").strip().splitlines()
    exc = [t for t in el if ("Error" in t or "Exception" in t) and "For simplicity" not in t]
    return dict(
        arm=arm,
        working_set_gib=gib,
        real=real,
        completed=False,
        error=(exc[-1] if exc else (el[-1] if el else f"exit {p.returncode}"))[:300],
        error_file=os.path.join(err_dir, f"{tag}.err"),
        oom=any("RESOURCE_EXHAUSTED" in t or "out of memory" in t.lower() for t in el),
        died_without_report=True,
    )


def _read_regime(r):
    """Name the regime a completed arm landed in, from the witnesses alone."""
    if not r.get("completed"):
        return "OOM" if r.get("oom") else "ERR"
    # the ceilings are GH200 numbers; on any other platform the label would be
    # a category error dressed up as a measurement
    if r.get("platform") != "gpu":
        return f"n/a ({r.get('platform', '?')})"
    gbs = r.get("gbytes_per_s") or r.get("positions_gbytes_per_s")
    resid = r.get("hbm_peak_over_working_set") or r.get("hbm_peak_over_positions")
    if gbs is None:
        return "?"
    if resid is not None and resid > 0.9:
        return "HBM-resident"  # it copied the set wholesale
    if gbs > CEILING_GBS["c2c_read"] * 1.15:
        return "above-C2C"  # did not cross C2C per byte
    if gbs > CEILING_GBS["lpddr"] * 0.5:
        return "LPDDR-class"
    return "C2C-class"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ladder", default=DEFAULT_LADDER, help="working-set sizes in GiB")
    ap.add_argument("--chunk-gib", type=float, default=CHUNK_GIB)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--smoke", action="store_true", help="tiny ladder: reachability only")
    ap.add_argument("--out-suffix", default="", help="suffix for the card, e.g. _smoke")
    ap.add_argument("--hbm-gib", type=float, default=None,
                    help="override detected physical HBM (GiB); default reads nvidia-smi")
    ap.add_argument("--real", action="store_true", help="C-gh paint point")
    ap.add_argument("--n-side", type=int, default=2048, help="C-gh particle side")
    ap.add_argument("--n-mesh", type=int, default=1024)
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--arm", default="hbm", choices=ARMS, help=argparse.SUPPRESS)
    ap.add_argument("--gib", type=float, default=8.0, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.single:
        run_real(args) if args.real else run_single(args)
        return

    global HBM_BYTES
    HBM_BYTES = physical_hbm_bytes(args.hbm_gib)
    arms = [a for a in args.arms.split(",") if a]
    if args.smoke:
        # under HBM on purpose: the smoke asks "is the path expressible and
        # does it stream", not "how fast" -- and it has to fit gh-dev's cap
        ladder = [4.0, 8.0]
        args.reps = min(args.reps, 3)
    else:
        ladder = [float(v) for v in args.ladder.split(",")]

    recs = []
    print("=== G4: GH200 memory-path reality check ===")
    print(f"HBM cliff at {HBM_BYTES / 1024**3:.1f} GiB (detected); ladder {ladder} GiB; arms {arms}")
    for gib in ladder:
        for arm in arms:
            print(f"[worker] {arm:9s} {gib:6.1f} GiB ...", flush=True)
            recs.append(spawn(arm, gib, args.chunk_gib, args.reps, False, 0, 0))

    if args.real:
        for arm in arms:
            print(f"[worker] {arm:9s} REAL C-gh {args.n_side}^3 ...", flush=True)
            recs.append(spawn(arm, 0.0, args.chunk_gib, max(1, args.reps // 2), True,
                              args.n_side, args.n_mesh))

    print(
        f"\n{'arm':9s} {'set GiB':>8s} {'GB/s':>8s} {'HBM pk GiB':>11s} {'pk/set':>7s} "
        f"{'>HBM':>5s}  regime / note"
    )
    for r in recs:
        gib = "REAL" if r.get("real") else f"{r.get('working_set_gib', 0):8.1f}"
        pk = r.get("peak_total")
        pk_s = f"{pk / 1024**3:11.1f}" if pk else f"{'-':>11s}"
        ratio = r.get("hbm_peak_over_working_set") or r.get("hbm_peak_over_positions")
        ratio_s = f"{ratio:7.3f}" if ratio else f"{'-':>7s}"
        gbs = r.get("gbytes_per_s") or r.get("positions_gbytes_per_s")
        gbs_s = f"{gbs:8.1f}" if gbs else f"{'-':>8s}"
        over = "yes" if r.get("exceeds_hbm") else "no"
        note = _read_regime(r)
        if not r.get("completed"):
            note += f"  {str(r.get('error', ''))[:70]}"
        if r.get("unsupported"):
            note = "UNSUPPORTED  " + str(r.get("error", ""))[:60]
        print(f"{r['arm']:9s} {gib:>8s} {gbs_s} {pk_s} {ratio_s} {over:>5s}  {note}")

    # --- precondition: did the caps clear the ladder? ---
    # This runs BEFORE the witnesses because a ladder run under a cap below
    # its own top measures the cap, not the machine (job 894010). A capacity
    # claim read off such a run is void, so say so here rather than let the
    # witness section imply otherwise.
    top_bytes = max(ladder) * 1024**3
    dev_lim = next((r.get("device_bytes_limit") for r in recs if r.get("device_bytes_limit")), None)
    host_lim_gb = next((r.get("env_host_limit_gb") for r in recs if r.get("env_host_limit_gb")), None)
    host_lim = float(host_lim_gb) * 1e9 if host_lim_gb else 64 * 1e9  # 64 GB is the XLA default
    print("\n--- preconditions (a cap below the ladder top voids the capacity witness) ---")
    print(
        f"device bytes_limit: {dev_lim / 1024**3:.1f} GiB"
        if dev_lim
        else "device bytes_limit: UNKNOWN"
    )
    print(
        f"host limit: {host_lim / 1024**3:.1f} GiB "
        f"({'XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=' + host_lim_gb if host_lim_gb else 'XLA DEFAULT 64 GB -- NOT SET'})"
    )
    voided = []
    # The two caps are checked against DIFFERENT references, on purpose. The
    # host arms must be free to run the whole ladder, so their cap is read
    # against the ladder top. The hbm arm is DESIGNED to die above the cliff,
    # so its cap is read against physical HBM instead -- the requirement is
    # that the cliff it finds is the CARD's, not a fraction of it. Comparing
    # the device cap to the ladder top would void every possible run.
    if dev_lim and dev_lim < 0.9 * HBM_BYTES:
        voided.append(
            f"device cap {dev_lim / 1024**3:.1f} GiB is only "
            f"{dev_lim / HBM_BYTES:.2f} of HBM -- the hbm arm's OOM point is a "
            f"software fraction, not the card's cliff"
        )
    if host_lim < top_bytes:
        voided.append(
            f"host cap {host_lim / 1024**3:.1f} GiB < ladder top {max(ladder):g} GiB -- "
            f"the host arms cannot reach the rungs that carry the capacity witness"
        )
    if voided:
        print("*** CAPACITY WITNESS VOID ***")
        for v in voided:
            print("    " + v)
        print(
            "    An OOM under these caps says nothing about the GH200. Raise\n"
            "    XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB (and the device fraction),\n"
            "    confirm the readback above moved, and re-run."
        )
    else:
        print("caps clear the ladder top: capacity witness is readable")

    # the verdict is JC's; this only reports whether the witnesses fired
    coh_over = [
        r for r in recs if r["arm"] == "coherent" and r.get("completed") and r.get("exceeds_hbm")
    ]
    print("\n--- witnesses ---")
    print(
        f"capacity: coherent completed {len(coh_over)} working set(s) larger than HBM"
        f"{' -> the set was NOT all HBM-resident' if coh_over else ' -> no capacity evidence'}"
    )
    print(
        "residency: read pk/set above -- ~1.0 means the set was copied wholesale; "
        "one chunk at a time gives chunk/set, e.g. "
        + ", ".join(f"{args.chunk_gib / g:.3f} at {g:g} GiB" for g in ladder)
    )
    print(
        f"bandwidth ceilings (E2 card): HBM {CEILING_GBS['hbm']:.0f}, "
        f"LPDDR {CEILING_GBS['lpddr']:.0f}, C2C r/w "
        f"{CEILING_GBS['c2c_read']:.0f}/{CEILING_GBS['c2c_write']:.0f} GB/s"
    )
    print("\nNothing here self-ratifies: G4's exit is JC's call on the regime.")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"g4_gh_memory{args.out_suffix}.json")
    with open(path, "w") as fh:
        json.dump(
            dict(
                ladder=ladder,
                chunk_gib=args.chunk_gib,
                reps=args.reps,
                hbm_bytes=HBM_BYTES,
                ceilings_gbs=CEILING_GBS,
                configs=recs,
            ),
            fh,
            indent=1,
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
