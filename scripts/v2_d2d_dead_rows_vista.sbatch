#!/bin/bash
#SBATCH -J d2d-dead
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d2d-dead-%j.log
# TIME ASK: ~4 min measured from 993350's arms, 10 min requested, one gb node,
# ~0.1 SU.
#
#   phase                         count   measured each (993350)   total
#   smoke leg (32^3, 2 arms)        1      ~45 s                     ~45 s
#   512^3 chunk arm                 5      ~40 s (29 s state build)  ~200 s
#                                                                    ~4 min
#
# ONE QUESTION: is the jitted paint's padding penalty on a GB200 the padded rows'
# shared scatter index? Record sec. 14 (Vista 993350): unpadded chunks cost
# 1.1 ns per row; 26% padding costs ~5.3x the device time and 100% ~17x. Every
# padded row scatters a zero weight into block cell 0 on all 27 corners.
# `paint_tsc_int_subblock(dead_rows="spread")` sends row i to cell i mod the
# block instead -- still weight zero, so bitwise the same block (each arm's block
# is checked against host decode through the shipped cell-0 kernel).
#
# ARMS, all jitted, 16.78M rows, 512^3 state, each its own subprocess:
#   512:4096:jit                          unpadded control (pad 2^24, 16 rows)
#   512:4096:jit:21139272                 26% padding, dead rows -> cell 0
#   512:4096:jit:21139272:spread          26% padding, dead rows spread
#   512:4096:jit:33554432                 100% padding, dead rows -> cell 0
#   512:4096:jit:33554432:spread          100% padding, dead rows spread
#
# PRE-REGISTERED READINGS (device time = chunk time minus host prep):
#   - spread arms near the unpadded rate per padded row (~1.1 ns): the shared
#     index is the cost; spread goes into the paint behind its bitwise gate.
#   - spread arms still ~5x / ~17x: the index is not it, and the cost is
#     elsewhere in how padded rows are carried (the decode's gathers for them
#     are the next suspect).
#   - the cell-0 arms reproduce 993350 within node spread.
#
# ENV copied verbatim from the D5 gb sbatch (v2_d5_four_gpu_vista.sbatch).

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$SCRATCH/src/inexor" 2>/dev/null || cd "$HOME/src/inexor" || {
  echo "FATAL: no inexor checkout"; exit 1; }
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=900
unset JAX_PLATFORMS
unset CUDA_VISIBLE_DEVICES

echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"

rc_total=0
run_leg () {
  local name="$1"; shift
  echo ""; echo "=== LEG $name ==="
  "$@"; local rc=$?
  echo "--- LEG $name rc=$rc ---"
  [ $rc -ne 0 ] && rc_total=$((rc_total + 1))
  return $rc
}

run_leg smoke \
  pixi run -e gpu python scripts/v2_d2d_device_paint.py --skip-xback --reps 1 \
    --chunks 32:64:jit:8192,32:64:jit:8192:spread --out-suffix _deadsmoke
if [ $rc_total -ne 0 ]; then
  echo "FATAL: smoke failed, not spending the node on the long arms"
  exit 1
fi

run_leg dead \
  pixi run -e gpu python scripts/v2_d2d_device_paint.py --skip-xback --reps 3 \
    --out-suffix _dead_gb \
    --chunks 512:4096:jit,512:4096:jit:21139272,512:4096:jit:21139272:spread,512:4096:jit:33554432,512:4096:jit:33554432:spread

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
