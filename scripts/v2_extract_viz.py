"""Pull small, portable visualization products out of a 1.65 TB particle export.

    pixi run python scripts/v2_extract_viz.py --export $RUNS/d7-1009133-product/export \
        --out runs/v2/viz-1009133

Run this ONCE on a node that can see the export; everything afterwards is
laptop work on a few hundred MB.

WHY IT IS CHEAP. `x.npy` is a plain `(n, 3)` float32 array behind a 128-byte
header, so it memory-maps and a read costs only the bytes asked for. The 1 TB
node this realization needed was for BUILDING the state, not for looking at it.

WHAT THE ROW ORDER BUYS AND WHAT IT DOES NOT. The header records
`row_order: brick-major`, so rows are ordered by the engine's spatial brick
index and any contiguous row range is a COMPACT REGION of the box. What the
header does NOT carry is a per-brick row count, and at a = 1 occupancy varies
with clustering, so a row offset cannot be turned into a brick coordinate. This
script never needs one: the positions are in the file, so every window reports
the bounding box it actually landed in, measured rather than assumed. (Worth
adding to `write_particles` next time it is touched: 16,384 per-chunk counts,
~100 kB, would make spatial random access exact.)

Products, all with a `viz.json` recording the row range and measured extent of
each so any of them can be re-cut:

  box_sub.npy   whole box, block-subsampled -- many small windows spread through
                the file, which in brick-major order spreads them through the box
  slab.npy      one contiguous 1/`--slab-frac` of the rows: a thick slab of the
                box, the classic cosmic-web frame
  zoom_N.npy    a ladder of nested CUBES centered on the densest region found,
                each one a factor `--zoom-step` smaller on a side

A contiguous row range is compact in x and spans the full y-z face, so it is a
SLAB, not a cube -- the first version of this script shipped a "zoom ladder"
that was four nested slabs. Cubes need a strided gather that brick-major order
cannot give without the per-brick counts the header lacks, so instead one slab
is read ONCE in full and every cube is cut out of it in memory: the slab is
histogrammed on a coarse grid, the densest cell picked, and the ladder nested
there. Deeper zooms cost no additional reads at all.
"""

import argparse
import json
import os
import time

import numpy as np

XNPY = "x.npy"
VNPY = "v.npy"


def _open(export, name):
    """Memory-map one export array without reading it."""
    path = os.path.join(export, name)
    return np.load(path, mmap_mode="r")


def _bbox(p):
    lo = p.min(axis=0)
    hi = p.max(axis=0)
    return lo, hi, float(np.prod(hi - lo))


def _take(x, lo, n, stride):
    """Rows [lo, lo+n) of a memmap, thinned by `stride`, as a real array."""
    block = x[lo : lo + n]
    if stride > 1:
        block = block[::stride]
    return np.ascontiguousarray(block, dtype=np.float32)


def _record(out, name, arr, lo, n_rows, note):
    bl, bh, vol = _bbox(arr)
    np.save(os.path.join(out, name), arr)
    rec = dict(file=name, points=int(arr.shape[0]), row_lo=int(lo),
               row_span=int(n_rows), bytes=int(arr.nbytes),
               bbox_lo=[float(v) for v in bl], bbox_hi=[float(v) for v in bh],
               extent=[float(v) for v in (bh - bl)],
               mpc_per_particle=float((vol / max(arr.shape[0], 1)) ** (1 / 3)),
               note=note)
    print(f"  {name:16s} {arr.shape[0]:>10,} pts  {arr.nbytes / 1e6:7.1f} MB  "
          f"extent {bh[0] - bl[0]:7.1f} x {bh[1] - bl[1]:7.1f} x {bh[2] - bl[2]:7.1f} Mpc/h")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export", required=True, help="the directory holding x.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--box-windows", type=int, default=8192,
                    help="windows spread through the file for the whole-box sample")
    ap.add_argument("--box-rows", type=int, default=256,
                    help="rows per whole-box window")
    ap.add_argument("--slab-frac", type=int, default=256,
                    help="the slab is 1/this of the rows (256 ~ one brick slab)")
    ap.add_argument("--slab-points", type=int, default=1_500_000)
    ap.add_argument("--zoom-levels", type=int, default=4)
    ap.add_argument("--zoom-thickness", type=float, default=64.0,
                    help="Mpc/h thickness of the source slab, which is also the "
                         "widest cube it can hold. Rows scale with it: at "
                         "c-hero this is 402 MB of reads per Mpc/h before thinning")
    ap.add_argument("--zoom-read-points", type=int, default=8_000_000,
                    help="cap on points read for the source slab; sets its stride, "
                         "so the read stays bounded however thick the slab is")
    ap.add_argument("--zoom-at", type=float, default=0.5,
                    help="where in the file the source slab is centered, 0-1")
    ap.add_argument("--zoom-step", type=float, default=3.0,
                    help="each zoom level is this factor smaller on a side")
    ap.add_argument("--zoom-points", type=int, default=600_000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.export, "export.json")) as fh:
        head = json.load(fh)
    x = _open(args.export, XNPY)
    n = x.shape[0]
    print(f"export {args.export}: {n:,} rows, {x.dtype}, box {head['box_size']} Mpc/h, "
          f"a={head.get('a')}, row_order={head['row_order']!r}")
    t0 = time.perf_counter()
    recs = []
    read_bytes = 0

    # --- whole box. Many small windows, spread evenly; in brick-major order
    # that spreads them through the volume. One big contiguous read would give
    # a slab instead, which is what `slab.npy` is for.
    step = n // args.box_windows
    parts = [_take(x, i * step, args.box_rows, 1) for i in range(args.box_windows)]
    box = np.concatenate(parts)
    read_bytes += box.nbytes
    recs.append(_record(args.out, "box_sub.npy", box, 0, n,
                        f"{args.box_windows} windows of {args.box_rows} rows, evenly spread"))
    del parts, box

    # --- one slab: a single contiguous range, so a compact region of the box.
    span = n // args.slab_frac
    lo = (n - span) // 2
    stride = max(1, span // args.slab_points)
    slab = _take(x, lo, span, stride)
    read_bytes += span * 12
    recs.append(_record(args.out, "slab.npy", slab, lo, span,
                        f"contiguous 1/{args.slab_frac} of rows, thinned {stride}x"))
    del slab

    # --- the zoom ladder. Read ONE slab in full, then cut cubes from it.
    # A contiguous range spans the whole y-z face, so nesting row ranges gives
    # thinner slabs, never a cube; the cubes have to come from a spatial cut.
    # THE KNOB IS THICKNESS, NOT ROWS. The widest cube is the slab's thickness,
    # and rows scale with it: at c-hero 40e6 rows is a 1.19 Mpc/h sliver, which
    # is what the first sizing of this asked for by accident.
    zrows = min(n, max(1, int(round(args.zoom_thickness / float(head["box_size"]) * n))))
    zlo = max(0, min(int(args.zoom_at * n) - zrows // 2, n - zrows))
    zstride = max(1, zrows // max(args.zoom_read_points, 1))
    src = _take(x, zlo, zrows, zstride)
    read_bytes += zrows * 12
    print(f"  zoom slab: {args.zoom_thickness:g} Mpc/h -> {zrows:,} rows "
          f"({zrows * 12 / 1e9:.2f} GB), thinned {zstride}x to {src.shape[0]:,} pts")
    slo, shi, _ = _bbox(src)
    print(f"  zoom slab extent "
          f"{shi[0] - slo[0]:.1f} x {shi[1] - slo[1]:.1f} x {shi[2] - slo[2]:.1f} Mpc/h")

    # densest cell of a coarse histogram over the slab. Cells are cubic and no
    # wider than the slab is thick, so the winner is a real 3-D overdensity
    # rather than a column picked by the slab's own geometry.
    cell = max(float(shi[0] - slo[0]) / 2.0, 1e-3)
    grid = np.maximum(((shi - slo) / cell).astype(int), 1)
    idx = np.minimum(((src - slo) / cell).astype(int), grid - 1)
    flat = (idx[:, 0] * grid[1] + idx[:, 1]) * grid[2] + idx[:, 2]
    counts = np.bincount(flat, minlength=int(np.prod(grid)))
    top = int(counts.argmax())
    ci = np.array([top // (grid[1] * grid[2]), (top // grid[2]) % grid[1], top % grid[2]])
    center = slo + (ci + 0.5) * cell
    print(f"  densest cell at {np.array2string(center, precision=1)} Mpc/h, "
          f"{counts[top]:,} of {src.shape[0]:,} pts in a {cell:.1f} Mpc/h cube")

    side = float(np.min(shi - slo))   # the widest cube the slab can hold
    for lvl in range(args.zoom_levels):
        h = side / 2.0
        # a cube whose centre sits within h of a slab face comes back CLIPPED,
        # so it is a box and not the cube its record claims. Pull the centre in
        # rather than shrinking the level.
        c = np.clip(center, slo + h, shi - h)
        m = np.all(np.abs(src - c) <= h, axis=1)
        cube = src[m]
        if cube.shape[0] > args.zoom_points:
            sel_i = np.linspace(0, cube.shape[0] - 1, args.zoom_points).astype(np.int64)
            cube = cube[sel_i]
        if cube.shape[0] < 100:
            print(f"  zoom_{lvl}: only {cube.shape[0]} pts at {side:.1f} Mpc/h; stopping")
            break
        recs.append(_record(args.out, f"zoom_{lvl}.npy", cube, zlo, zrows,
                            f"cube of side {side:.2f} Mpc/h at "
                            f"{np.array2string(c, precision=1)}, densest cell of the slab"))
        side /= args.zoom_step

    man = dict(card="inexor-viz-1", source=args.export, n_particles=int(n),
               box_size=float(head["box_size"]), a=head.get("a"),
               dtype=str(x.dtype), export_provenance=head.get("provenance", {}),
               wall_s=time.perf_counter() - t0,
               approx_bytes_read=int(read_bytes), products=recs)
    with open(os.path.join(args.out, "viz.json"), "w") as fh:
        json.dump(man, fh, indent=2)
    total = sum(r["bytes"] for r in recs)
    print(f"-> {args.out}/viz.json | {len(recs)} products, {total / 1e6:.0f} MB written, "
          f"~{read_bytes / 1e9:.2f} GB read, {man['wall_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()
