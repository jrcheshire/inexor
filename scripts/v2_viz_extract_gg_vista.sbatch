#!/bin/bash
#SBATCH -J viz-extract
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o viz-extract-%j.log
# Cut the visualization products out of gb 1009133's (x, v) export, ONCE, so
# every plot after this is laptop work.
#
# gg, because this job has no device work in it at all: it memory-maps a plain
# (n, 3) float32 array and copies byte ranges out. What it needs is Lustre and
# a few GB of RAM.
#
# WHY IT IS SMALL. The export is 1.65 TB but `x.npy` is a flat npy behind a
# 128-byte header, so a read costs the bytes asked for and nothing else. The
# header records `row_order: brick-major`, the engine's spatial layout, so any
# contiguous row range is a compact region of the box. The 1 TB host this
# realization needed was for BUILDING the state.
#
# There is no per-brick row count in the header and occupancy varies with
# clustering at a = 1, so a row offset cannot be turned into a brick
# coordinate. The extractor never needs one -- the positions are in the file,
# so every window reports the bounding box it measured.
#
# TIME, from the code's own read sizes at c-hero (68.7e9 rows, 2048 Mpc/h, so
# 402 MB of rows per Mpc/h of slab):
#   whole-box sample   8,192 windows x 256 rows, page-granular ........  ~0.03 GB
#   slab               1/256 of the rows, contiguous ..................   3.2  GB
#   zoom slab          64 Mpc/h thick, the widest cube it can hold .....  25.8  GB
#                                                                        -------
#                                                                         ~29   GB
# The zoom slab is thinned to 8e6 points but the READ is the full 25.8 GB: a
# 268-row stride is 3.2 kB, under a page, so every page is touched anyway. The
# table counts what the kernel reads, not what numpy keeps.
#
# Serial and I/O-bound; ~29 GB off Lustre is under a minute of transfer, and the
# 8,192 scattered windows are latency rather than bandwidth. Expected a few
# minutes. Peak RAM is the thinned slab, ~96 MB, against the node's ~255 GB.
# 30 min requested because TACC's minimum charge is 15 and asking under it buys
# nothing.
#
# BUDGET: gg is 0.33 SU/node-hr, so 0.17 SU worst case, well under 0.05 expected.
#
# OPS (load-bearing on gg): pixi on PATH; JAX_PLATFORMS=cpu (jax 0.10
# hard-raises on cuInit on a GPU-less node); CONDA_OVERRIDE_CUDA (the gpu env's
# activation gate). The extractor itself imports only numpy and json, so no jax
# is loaded either way -- these are the partition's contract, not this script's.

set -u

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export CONDA_OVERRIDE_CUDA=12.0
export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }

RUNS=/scratch/10303/jrc4/inexor_runs
EXPORT=$RUNS/d7-1009133-product/export
OUT=$RUNS/viz-1009133
export EXPORT

echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
free -g | head -2
df -h "$RUNS" | tail -1
[ -e "$EXPORT/export.json" ] || { echo "FATAL: no complete export at $EXPORT"; exit 1; }

# The export's header is written LAST and removed FIRST, so its presence is the
# completeness contract -- but it does not say the arrays are the ones it
# describes. Check the declared size against the file before reading 5 GB of it.
pixi run -e gpu python -c '
import json, os, sys
d = os.environ["EXPORT"]
h = json.load(open(d + "/export.json"))
want = h["n_particles"] * 3 * 4 + 128
got = os.path.getsize(d + "/x.npy")
print(f"  x.npy {got:,} bytes, header declares {want:,}")
print(f"  {h[\"n_particles\"]:,} particles, a={h[\"a\"]}, box={h[\"box_size\"]}, "
      f"step {h[\"provenance\"].get(\"step\")}, commit {h[\"provenance\"].get(\"commit\")}")
sys.exit(0 if got == want else 1)
' || { echo "FATAL: x.npy is not the size its header declares"; exit 1; }

pixi run -e gpu python scripts/v2_extract_viz.py \
  --export "$EXPORT" --out "$OUT" \
  --zoom-thickness 64 --zoom-read-points 8000000 --zoom-levels 4
rc=$?

echo ""
echo "=== products ==="
ls -la "$OUT" 2>/dev/null
du -sh "$OUT" 2>/dev/null
echo "=== end $(date) rc=$rc ==="
exit $rc
