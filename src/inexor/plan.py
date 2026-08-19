"""Will this configuration fit this machine? `python -m inexor.plan`.

**Every number here already existed and nothing put them in one place.** That is
not a convenience complaint: M-v2-6 Stage 0 measured the engine's end-to-end peak
for the first time and found 8.2 GB at cdev8 where the accounted terms came to
~0.25 GB, because a 32 B/p host array in the kick was in no table. A budget that
omits a term cannot trade against it, and `EngineConfig.mesh_bytes()` -- the one
function that could have caught the analogous mesh case -- was called by nothing
outside its own test.

So this is a front end over four existing sources, not new arithmetic:
`EngineConfig.mesh_bytes` and `.step_bytes` (engine), `T9Layout.index_bytes`
(codec), `ooc_fft.plan_bytes` (the IC stage), plus the T9 payload itself.

**No site knowledge, by design.** Budgets are ARGUMENTS. The 116 GB figure that
appears throughout this project's records is a property of one Vista GH200 host,
not of inexor, and nothing in the package should know it -- pass `--host-gb 116`
if that is your machine, `--host-gb 237` for a Vista gg node, `--host-gb 1007`
for a Stampede3 h100 node. See `docs/running-elsewhere.md`.

What this does NOT do: measure anything. It is arithmetic over a config, and
where the arithmetic is known to be incomplete it says so rather than implying a
total is a peak. The engine's true peak has been measured at exactly two
configurations and both are development-scale.

Examples:
  python -m inexor.plan --preset cdev
  python -m inexor.plan --preset c-gh --host-gb 116
  python -m inexor.plan --n-part 2048 --box 1024 --n-fine 4096 --n-coarse 1024 \
      --tile 256 --buf 32 --host-gb 237 --disk-gb 2000
"""

import argparse
import sys

import numpy as np

# The ratified config table (`docs/plan-plan-v2.md`). Presets are a CONVENIENCE:
# the fine cell is held fixed across the ladder and volume grows, so these differ
# in box and particle count, not in resolution.
PRESETS = {
    "smoke": dict(n_part=32, box=32.0, n_fine=64, n_coarse=16, tile=16, buf=8),
    "cdev8": dict(n_part=128, box=64.0, n_fine=256, n_coarse=64, tile=64, buf=32),
    "cdev": dict(n_part=256, box=128.0, n_fine=512, n_coarse=128, tile=256, buf=32),
    "cgh64": dict(n_part=512, box=256.0, n_fine=1024, n_coarse=256, tile=256, buf=32),
    "c-gh": dict(n_part=2048, box=1024.0, n_fine=4096, n_coarse=1024, tile=256, buf=32),
    "c-hero": dict(n_part=4096, box=2048.0, n_fine=8192, n_coarse=2048, tile=512, buf=32),
}

GB = 1e9

# One pool worker's own footprint, MEASURED: Vista gg, 8 workers, 9.0 GB total
# read from /proc/<pid>/statm after a barrier task proved `_worker_init` had
# finished. Mostly the interpreter plus jax; the state itself is shared and is
# already in the tables above, so this is what each worker adds on top.
#
# It is a floor twice over -- taken at pool startup before any tile has been
# forced, and only ever at W=8 -- and it is in this module rather than in
# `executor` because it is a property of a machine, not of the code. See
# `docs/running-elsewhere.md` on why nothing in the package knows a node size.
WORKER_STARTUP_BYTES = 1.12 * GB

# The knobs the PRODUCTION path runs, in one place, because the alternative
# is what happened: `EngineConfig` defaults coarse_dtype to float64 for the
# benefit of the f64 reference arms, `v2_m6_realization.py` did not override
# it, and this planner defaulted to float32 -- so the table said FITS while
# pricing a configuration nobody was running, and the miss (12.885 GB of
# shared memory) took three separate rediscoveries and a dead job to find.
#
# Anything here is a RATIFIED choice with a decision behind it. The gate
# oracles deliberately do NOT use this: their whole job is to vary these.
RATIFIED = dict(
    coarse_dtype="float32",   # D-v2-22 / M-v2-4: 1.83x peak host, 1e-5 error
    fine_dtype="float64",     # unchanged by M-v2-4; the fine arm stays f64
    alpha=1.0,                # r_s / coarse_cell, the ratified kernel
)
BUCKET_CELLS = 2              # D-v2-20's layout


def engine_config(preset, **overrides):
    """The `EngineConfig` the production driver builds, from ONE definition.

    Both `inexor.plan` and `scripts/v2_m6_realization.py` go through this, so
    a planner verdict is about the run that will actually happen. Overrides
    are for the driver's per-invocation knobs (workers, slack, checkpointing),
    not for quietly reinstating a default this dict exists to pin.
    """
    from .engine import EngineConfig

    g = PRESETS[preset] if isinstance(preset, str) else dict(preset)
    kw = dict(
        box_size=g["box"], n_part=g["n_part"], n_fine=g["n_fine"],
        n_coarse=g["n_coarse"], n_tile=g["tile"], b_fine=g["buf"],
        **RATIFIED,
    )
    kw.update(overrides)
    return EngineConfig(**kw)


def _fmt(b):
    return f"{b / GB:10.3f} GB"


def _table(title, terms, total_label="total", reduce=sum):
    print(f"\n{title}")
    width = max(len(k) for k in terms) if terms else 1
    for k, v in sorted(terms.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<{width}}  {_fmt(v)}")
    print(f"  {'-' * width}  {'-' * 13}")
    print(f"  {total_label:<{width}}  {_fmt(reduce(terms.values()))}")


def load_stages(*, n, n_rows, n_buckets, index_itemsize, n_arena, n_bricks,
                n_slabs, shared):
    """Peak resident bytes at each stage of `icgen.load_slot_state`.

    THE LOAD PATH WAS IN NO TABLE, and it is where two jobs died. It is also
    the stage where `shared` changes the answer completely: with an allocator
    the payload is written straight into the segments the pool will use and
    the state exists ONCE; without one it is built privately and copied,
    which is 2x at the moment of copying.

    A slab is one x-slice of bricks, so its payload is n/n_slabs rows of the
    9 B T9 record plus its share of the index.
    """
    slab = n // max(n_slabs, 1) * 9 + n_buckets // max(n_slabs, 1) * 8
    occ64 = n_buckets * 8            # the int64 accumulator, both passes
    payload = n_rows * 9             # off + w
    index = n_buckets * index_itemsize
    stages = {
        "pass 1 (index, one slab live)": occ64 + slab,
        "allocate off/w": occ64 + payload,
        "pass 2 (payload, one slab live)": occ64 + payload + slab,
        # `_to_index` makes the uint32 beside the int64, and a shared build
        # copies that into a segment before the int64 goes away
        "build SlotState": occ64 + payload + index * (2 if shared else 1),
    }
    if not shared:
        # TilePool then copies every field into segments, one at a time
        stages["adopt into the pool"] = payload + index + n_arena * 8 + max(
            n_rows * 6, n_rows * 3)
    return stages


def build(args):
    from .codec import T9Layout
    from .engine import EngineConfig

    ec = EngineConfig(
        box_size=args.box, n_part=args.n_part, n_fine=args.n_fine,
        n_coarse=args.n_coarse, n_tile=args.tile, b_fine=args.buf,
        coarse_dtype=args.coarse_dtype, fine_dtype=args.fine_dtype,
    )
    t9 = T9Layout(box_size=args.box, n_part=args.n_part, bucket_cells=args.bucket_cells)
    return ec, t9


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Memory anatomy of an inexor configuration, and whether it fits.",
    )
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--n-part", type=int, default=None, help="particles per side")
    ap.add_argument("--box", type=float, default=None, help="box size, Mpc/h")
    ap.add_argument("--n-fine", type=int, default=None)
    ap.add_argument("--n-coarse", type=int, default=None)
    ap.add_argument("--tile", type=int, default=None)
    # None, not 32: the preset fill below only writes a field that is still None,
    # so a non-None default here silently OUTRANKS the preset table. Every preset
    # but `smoke` carries buf=32, which is why the shadowing was invisible -- and
    # `smoke` (buf=8) was left unevaluable, T=16 + 2*32 = 80 against a 64 mesh
    # tripping the degeneracy guard. Same class as a decayed default at a call
    # site that omits the argument.
    ap.add_argument("--buf", type=int, default=None)
    ap.add_argument("--coarse-dtype", default="float32")
    ap.add_argument("--fine-dtype", default="float64")
    ap.add_argument("--bucket-cells", type=int, default=2)
    ap.add_argument("--slack", type=float, default=0.10, help="per-brick spare fraction")
    ap.add_argument("--alloc-margin", type=float, default=0.10)
    ap.add_argument("--arena-frac", type=float, default=0.01,
                    help="arena rows as a fraction of N. NOTE the default is "
                         "`SlotState.build`'s, but `scripts/v2_m6_realization.py` "
                         "runs 0.20, which is 28 GB of shared memory at c-gh.")
    # No default, and deliberately not derived: geometry alone UNDERSTATES it.
    # `cap` is the max over tiles of the padded per-tile row count, so it carries
    # (P/T)^3 x N/tiles, the geometric ladder's <=26.0%, AND the clustering spread
    # of per-tile occupancy -- and that last part is what geometry cannot give.
    # Measured excess over (P/T)^3 x N/tiles is 1.290x at cdev and 1.587x at cdev8,
    # shrinking as per-tile occupancy grows, so a derived cap would look like a
    # measurement and read low.
    ap.add_argument("--cap", type=int, default=None,
                    help="measured per-tile capacity, for the tile_buffers term. "
                         "cdev/cgh64/C-gh share N/tile and P, so cdev's measured "
                         "5284492 is the anchor for all three.")
    # BUDGETS ARE ARGUMENTS. No default host size: a wrong default is worse than
    # an absent one, because it silently makes a verdict up.
    ap.add_argument("--host-gb", type=float, default=None,
                    help="host RAM budget. A property of YOUR machine: Vista gh "
                         "~116 (a hard cliff), Vista gg 237, S3 h100 ~1007.")
    ap.add_argument("--device-gb", type=float, default=None, help="accelerator HBM budget")
    # A SEPARATE budget, and the reason this argument exists at all: the pool
    # holds the state in POSIX shared memory, which is a tmpfs sized by default
    # at HALF the node's RAM. This table said "FITS (0.74x)" for the c-gh run
    # that then died in pool construction (Vista 920910), because the host
    # total it checked was never the budget that bound. Read your node's with
    # `df -B1 /dev/shm`; a Vista gg node measures 127.6 GB (job 922332).
    ap.add_argument("--shm-gb", type=float, default=None,
                    help="/dev/shm budget, for the pooled (tile_workers>1) lane. "
                         "Vista gg measures 127.6. Kernel default is half of RAM.")
    ap.add_argument("--workers", type=int, default=None,
                    help="tile_workers. >1 (or unset) means the pooled lane: the "
                         "loader writes into shared memory and the state exists once")
    ap.add_argument("--slabs", type=int, default=128,
                    help="T9 slab files the ICs were written as (c-gh: 128)")
    ap.add_argument("--disk-gb", type=float, default=None, help="scratch budget for IC staging")
    args = ap.parse_args(argv)

    if args.preset:
        for k, v in PRESETS[args.preset].items():
            if getattr(args, k.replace("-", "_")) is None:
                setattr(args, k.replace("-", "_"), v)
    if args.buf is None:
        args.buf = 32
    missing = [k for k in ("n_part", "box", "n_fine", "n_coarse", "tile")
               if getattr(args, k) is None]
    if missing:
        ap.error(f"need {', '.join(missing)} (or --preset)")

    ec, t9 = build(args)
    n = args.n_part**3
    print(f"config: n_part={args.n_part}^3 = {n:,} particles, box={args.box} Mpc/h, "
          f"n_fine={args.n_fine}, n_coarse={args.n_coarse}, T={args.tile}, b={args.buf}")
    print(f"        fine cell {ec.fine_cell:.4f} Mpc/h, particle spacing "
          f"{args.box / args.n_part:.4f} Mpc/h, {len(ec.tiles)} tiles, "
          f"coarse {ec.coarse_dtype} / fine {ec.fine_dtype}")

    # ---- state, the thing the architecture is about
    rows = int(np.ceil(np.ceil(n * (1.0 + args.slack)) * (1.0 + args.alloc_margin)))
    arena = int(np.ceil(n * args.arena_frac))
    state = {
        "t9_payload (9 B/p)": n * 9,
        "slack + alloc_margin": (rows - n) * 9,
        "bucket_index": t9.index_bytes(),
        # The arena is n_arena EXTRA ROWS of off/w (`state.SlotState`: off is
        # (n_alloc + n_arena, 3)), and this table priced only `arena_bucket`,
        # its int64 side. At the realization's `--arena-frac 0.20` that is
        # 15.5 GB of payload the total did not carry -- the omitted-term fault
        # this module's docstring is about, in this module.
        "arena rows in off/w": arena * 9,
        "arena_bucket": arena * 8,
        "brick_start": (ec.n_brick and (args.n_fine // ec.n_brick) ** 3 + 1) * 8,
        # THE ARRAY THAT REPLACED `kick_pending`. One f64 per brick, resident,
        # against 32 B per PARTICLE held per-step: 16.8 MB against 274.9 GB at
        # C-gh. It is listed rather than folded into the noise because the term
        # it replaced was the binding one, and a reader comparing this table to
        # an older card needs to see where that went.
        "brick_scales": (ec.n_brick and (args.n_fine // ec.n_brick) ** 3) * 8,
    }
    _table("STATE (resident for the whole run)", state)
    print(f"  {'':<20}  {sum(state.values()) / n:6.2f} B/p")

    mesh = ec.mesh_bytes()
    # The split is `engine.MESH_PHASE`'s, not this module's: a term's phase is a
    # property of the code that allocates it, so the accounting and the engine
    # cannot drift apart the way they did over `coarse_dtype`.
    from .engine import MESH_PHASE, STEP_PHASE

    resident = {k: v for k, v in mesh.items() if MESH_PHASE[k] == "resident"}
    transient = {k: v for k, v in mesh.items() if k not in resident}
    _table("MESH, resident through the tile loop", resident)
    _table("MESH, transient (peak while that phase runs)", transient)

    step = ec.step_bytes(n, n_rows=rows, cap=args.cap)
    _table("PER-STEP HOST TERMS THAT SCALE WITH PARTICLES", step)

    # PHASES, which is the line the C-gh budget was missing. Summing every
    # transient overstates (the tile loop does not run during the coarse solve);
    # taking the largest single one understates, and understated is how a run
    # that does not fit gets a FITS. Terms in a phase are co-resident by
    # construction, so sum within and max across.
    phases = {}
    for src, phase_of in ((mesh, MESH_PHASE), (step, STEP_PHASE)):
        for k, v in src.items():
            p = phase_of[k]
            # a term may name several phases: it is charged to each, because
            # what a phase's line answers is "how much is live while this runs"
            for one in (p,) if isinstance(p, str) else p:
                if one != "resident":
                    phases[one] = phases.get(one, 0) + v
    _table("BY PHASE (transients summed within, because they ARE co-resident)",
           phases, total_label="sum of the in-step phases", reduce=sum)
    print("  the total is a SUM over the phases INSIDE a step and excludes "
          "kernel_build,\n  which runs once before the loop. Summing rather than "
          "maxing is deliberate:\n  glibc does not return freed arenas between "
          "phases, so a step's high-water\n  accumulates -- `malloc_trim` recovered "
          "12-41% of a run's peak here.")

    # ---- the sub-budget, for the pooled lane only
    if args.workers is None or args.workers > 1:
        from .executor import shm_terms

        shm = shm_terms(
            n_rows=rows + arena, index_bytes=t9.index_bytes(), n_arena=arena,
            n_bricks=ec.n_brick and (args.n_fine // ec.n_brick) ** 3,
            n_coarse=args.n_coarse,
            coarse_itemsize=np.dtype(args.coarse_dtype).itemsize,
        )
        _table("SHARED MEMORY (/dev/shm), the pooled lane's sub-budget", shm)
        print("  the migrate scratch is staged per step at a (K, R) this table "
              "cannot know;\n  `TilePool` holds construction to a 1.10x margin "
              "for it")
        if args.shm_gb is not None:
            sb = args.shm_gb * GB
            r = sum(shm.values()) / sb
            print(f"  against --shm-gb {args.shm_gb}: "
                  f"{'FITS' if r < 1.0 else 'DOES NOT FIT'} ({r:.2f}x the budget)")
        else:
            print("  no --shm-gb given, so no verdict. This is NOT the host "
                  "budget above: it is a\n  tmpfs, by default half the node's "
                  "RAM, and it bound the c-gh run that the host\n  line called "
                  "a fit. `df -B1 /dev/shm` on the node you will run on.")

    # ---- the load path, which is where two jobs actually died
    idx_itemsize = t9.index_bytes() // max(t9.n_buckets_side**3, 1)
    ld = load_stages(
        n=n, n_rows=rows + arena, n_buckets=t9.n_buckets_side**3,
        index_itemsize=max(idx_itemsize, 1), n_arena=arena,
        n_bricks=ec.n_brick and (args.n_fine // ec.n_brick) ** 3,
        n_slabs=args.slabs, shared=(args.workers is None or args.workers > 1),
    )
    _table("LOADING THE STATE, peak resident at each stage", ld,
           total_label="PEAK (max, not sum)", reduce=max)
    print("  the total line above is a MAX: these stages do not coexist")

    # ---- IC stage
    try:
        from .ooc_fft import plan_bytes

        ic = plan_bytes(args.n_part, np.dtype(args.coarse_dtype), "derivative")
        print(f"\nIC STAGE (out-of-core, 'derivative' policy): peak "
              f"{_fmt(ic['peak'])}")
    except Exception as exc:  # pragma: no cover - informational only
        print(f"\nIC STAGE: not evaluable here ({exc})")

    # ---- the verdict, with the binding term NAMED
    print("\nBINDING TERMS")
    # THE WORST PHASE, not the largest single transient. The old form was
    #     sum(state) + sum(resident) + max(transient) + sum(step)
    # which charged 12.9 GB of a 56 GB C-gh mesh transient while summing the
    # per-step host terms as though repack and migrate ran at the same instant.
    # It said FITS (0.75x) for the run that OOM-killed in its coarse solve, and
    # the phase it under-charged is the one that died: Vista 923139 lost ~79 GB
    # of MemAvailable inside it, against 12.9 charged.
    from .engine import ONCE_PER_RUN_PHASES

    in_step = sum(v for k, v in phases.items() if k not in ONCE_PER_RUN_PHASES)
    once = max((v for k, v in phases.items() if k in ONCE_PER_RUN_PHASES),
               default=0)
    worst_phase = max(in_step, once)
    # THE WORKERS ARE PROCESSES AND THIS TABLE ONLY EVER PRICED ONE. Measured
    # 1.12 GB each on a Vista gg node (8 workers, 9.0 GB total, read from
    # /proc/<pid>/statm AFTER a barrier task -- `ctx.Pool()` returns before
    # `_worker_init` has imported jax, and the first version of that probe
    # reported 0.01 GB each and then watched them grow). It is a FLOOR: the
    # reading is at pool startup, before any tile has been forced, and it is the
    # measurement this budget most needs repeating at W=16.
    n_workers = 0 if args.workers is None else max(0, int(args.workers))
    workers_b = int(n_workers * WORKER_STARTUP_BYTES) if n_workers > 1 else 0
    peak_est = (sum(state.values()) + sum(resident.values()) + worst_phase
                + workers_b)
    load_peak = max(ld.values())
    # TRANSIENTS ARE CANDIDATES. They were excluded here, so the line could not
    # name a transient however large -- at cdev it reported `tile_kernels` (0.791
    # GB) while `tile_workspace` (1.443) was bigger and the phase MEASURED to set
    # the peak is the tile force (job 446: `tile_short` increments 3.432 GB).
    # A "largest term" line that structurally cannot name the measured winner is
    # the shape of a gate that cannot fail.
    biggest = max(list(state.items()) + list(resident.items()) + list(step.items())
                  + [(f"{k} (transient)", v) for k, v in transient.items()],
                  key=lambda kv: kv[1])
    if workers_b:
        print(f"  {n_workers} pool workers at {_fmt(WORKER_STARTUP_BYTES)} each "
              f"(measured, gg): {_fmt(workers_b)}")
    print(f"  a lower bound on the run's peak: {_fmt(peak_est)}")
    print(f"  the LOAD stage peaks at:          {_fmt(load_peak)}"
          f"   {'<- BINDING' if load_peak > peak_est else ''}")
    print(f"  largest single term: {biggest[0]} at {_fmt(biggest[1])}")
    if "tile_buffers" not in step:
        print("  NB `tile_buffers` is NOT in the total above: it needs a measured "
              "`cap`,\n     and it is the per-tile host set that sits inside the phase "
              "measured to\n     SET the peak at cdev. Pass --cap to include it "
              "(cdev measured 5,284,492).")
    if args.host_gb is not None:
        budget = args.host_gb * GB
        ratio = peak_est / budget
        verdict = "FITS" if ratio < 1.0 else "DOES NOT FIT"
        print(f"  against --host-gb {args.host_gb}: {verdict} "
              f"({ratio:.2f}x the budget)")
    else:
        print("  no --host-gb given, so no verdict: pass your machine's host RAM")
    if args.device_gb is not None:
        dev = sum(resident.values()) + max(transient.values())
        print(f"  device-resident mesh against --device-gb {args.device_gb}: "
              f"{dev / (args.device_gb * GB):.2f}x")
    print("\n  NB this is a LOWER BOUND from arithmetic, not a measurement. It "
          "charges every\n  in-step phase, not the largest single term -- the largest-"
          "term form is what\n  called the c-gh run a fit twice. What it still cannot "
          "see is XLA's intra-jit\n  scratch, which is invisible to tracemalloc, to "
          "`live_arrays` and to\n  `memory_stats()` alike on CPU. Vista 923139 lost "
          "~79 GB inside a phase this\n  prices at 30, so treat it as a sizing floor "
          "and never as a peak.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
