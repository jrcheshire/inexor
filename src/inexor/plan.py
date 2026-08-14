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


def _fmt(b):
    return f"{b / GB:10.3f} GB"


def _table(title, terms, total_label="total"):
    print(f"\n{title}")
    width = max(len(k) for k in terms) if terms else 1
    for k, v in sorted(terms.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<{width}}  {_fmt(v)}")
    print(f"  {'-' * width}  {'-' * 13}")
    print(f"  {total_label:<{width}}  {_fmt(sum(terms.values()))}")


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
    ap.add_argument("--arena-frac", type=float, default=0.01)
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
    # `tile_kernel_build_f64`/`tile_kernel_pref` are TRANSIENT: they are live only
    # while `split_kernels` runs, and it runs inside the `membership` phase every
    # step because `make_tile_force_fn` is not cached. They still set the peak
    # there -- M-v2-6 measured that phase as the largest single term at both cdev8
    # and cdev -- so classifying them as transient is about WHEN, not whether.
    resident = {k: v for k, v in mesh.items() if k in (
        "coarse_delta", "coarse_force_resident", "tile_kernels")}
    transient = {k: v for k, v in mesh.items() if k not in resident}
    _table("MESH, resident through the tile loop", resident)
    _table("MESH, transient (peak while that phase runs)", transient)

    step = ec.step_bytes(n, n_rows=rows, cap=args.cap)
    _table("PER-STEP HOST TERMS THAT SCALE WITH PARTICLES", step)

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
    peak_est = sum(state.values()) + sum(resident.values()) + max(transient.values()) \
        + sum(step.values())
    # TRANSIENTS ARE CANDIDATES. They were excluded here, so the line could not
    # name a transient however large -- at cdev it reported `tile_kernels` (0.791
    # GB) while `tile_workspace` (1.443) was bigger and the phase MEASURED to set
    # the peak is the tile force (job 446: `tile_short` increments 3.432 GB).
    # A "largest term" line that structurally cannot name the measured winner is
    # the shape of a gate that cannot fail.
    biggest = max(list(state.items()) + list(resident.items()) + list(step.items())
                  + [(f"{k} (transient)", v) for k, v in transient.items()],
                  key=lambda kv: kv[1])
    print(f"  a lower bound on the run's peak: {_fmt(peak_est)}")
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
          "assumes\n  one transient peaks at a time, and the engine's true peak has "
          "been measured\n  at development scale only. Treat it as a sizing floor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
