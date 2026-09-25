#!/bin/bash
#SBATCH -J d2d-shape
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d2d-shape-%j.log
# D2d follow-up: does the jitted chunk's TIME depend on block shape or on padding?
# Submit from the Vista checkout:
#   sbatch scripts/v2_d2d_chunk_shape_vista.sbatch
#
# WHY. Job 993294 (record sec. 13) measured the jitted paint's device time at
# 1.1 ns per padded row for a 16.8M-row chunk and 4.8 / 4.7 for 84.6M / 338M,
# which taken at face value makes small chunks 3x faster per 4096^3 step. Two
# things confound it, and this job separates them:
#   1. BLOCK SHAPE. Every 4096^3 chunk paints a block 11 cells thick in x; the
#      test chunks were 35-131 cells thick. A 1024^3 state tiles 4096 bricks as
#      1 x 64 x 64, painting an 11 x 512 x 512 block from the same 16.8M rows the
#      512^3 arm paints into 35 x 256 x 256.
#   2. PADDING / POWER OF TWO. The fast arm was the only one padded to an exact
#      power of two (2^24, 16 rows of padding; the others carry 26%). At fixed
#      rows, pad is overridden (an instrument, never an operating point) to
#      separate the fraction from the power of two.
#
# ARMS (all jitted, each its own subprocess, one node):
#   1024:4096:jit                  thin block, 16.8M rows, default pad 2^24
#   512:4096:jit                   thick block, 16.8M rows, default pad 2^24 (993294 repeat)
#   512:4096:jit:21139272          thick, 1.26x pad, not a power of two
#   512:4096:jit:33554432          thick, 2x pad, 2^25
#   512:16384:jit                  67M rows, default pad 84551871 (26%, not pow2; 993294 repeat)
#   512:16384:jit:67109009         67M rows, zero padding, not pow2
#   512:16384:jit:134217728        67M rows, 2x pad, 2^27
# Every arm's block must equal host decode through the eager kernel (rc 3
# otherwise). The cross-backend identity is not re-run: it passed on two nodes
# (993139, 993294) and nothing in the paint has changed since.
#
# PRE-REGISTERED READINGS (device time = chunk time minus host window prep, per
# padded row):
#   - thin ~= thick at 16.8M rows: block shape does not move time, and sec. 12-13's
#     per-step arithmetic stands on shape.
#   - pad arms track padded rows (time per padded row flat across 2^24 / 1.26x /
#     2^25 and across zero / 26% / 2^27): padding is just rows, and the anomaly is
#     elsewhere (row count itself).
#   - the pow2 arms fast and the others slow at the same rows: power of two is
#     the variable, which would make pad sizing a performance knob.
#   - repeats of 993294's two arms should land within node spread (1.28-1.36x).
#
# ENV copied verbatim from the D5 gb sbatch (v2_d5_four_gpu_vista.sbatch).
#
# BUDGET: one 1024^3 state build (~4 min) plus six 512^3 (~30 s each) plus short
# arms; ~12-15 min expected, 45 min requested, <= 0.75 SU.

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
free -g | head -2

rc_total=0
run_leg () {
  local name="$1"; shift
  echo ""; echo "=== LEG $name ==="
  "$@"; local rc=$?
  echo "--- LEG $name rc=$rc ---"
  [ $rc -ne 0 ] && rc_total=$((rc_total + 1))
  return $rc
}

# Smoke ON THE GPU at 32^3, including a pad override, before the long arms.
run_leg smoke \
  pixi run -e gpu python scripts/v2_d2d_device_paint.py --skip-xback --reps 1 \
    --chunks 32:64:jit,32:64:jit:8192 --out-suffix _shapesmoke
if [ $rc_total -ne 0 ]; then
  echo "FATAL: smoke failed, not spending the node on the long arms"
  exit 1
fi

run_leg shape \
  pixi run -e gpu python scripts/v2_d2d_device_paint.py --skip-xback --reps 3 \
    --out-suffix _shape_gb \
    --chunks 1024:4096:jit,512:4096:jit,512:4096:jit:21139272,512:4096:jit:33554432,512:16384:jit,512:16384:jit:67109009,512:16384:jit:134217728

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
