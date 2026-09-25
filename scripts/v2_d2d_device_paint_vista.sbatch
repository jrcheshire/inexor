#!/bin/bash
#SBATCH -J d2d-paint
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d2d-paint-%j.log
# D2d on a GB200: the device coarse paint, bitwise across backends, and what one
# chunk costs a card.
# Submit from the Vista checkout:
#   sbatch scripts/v2_d2d_device_paint_vista.sbatch
#
# THREE QUESTIONS.
#   1. Is the device paint BITWISE the host engine's paint across backends? The
#      laptop suite shows device == host on one backend only. Here a CPU-only
#      process computes `engine.coarse_delta_streamed` at cdev (plain state, and
#      one with arena residents) and the GPU must hash-equal it.
#   2. What does one chunk hold on the card? The planner charges 124 B per padded
#      row, read off the traced program under last-use freeing -- a FLOOR, since
#      the paint runs eagerly. Chunks are built at production brick geometry so
#      their rows are 4096^3 chunks': 1/16, 1/4 and a whole x-slab.
#   3. How long does a chunk take, and how much of that is the host's window prep?
#
# ENV TAKEN FROM THE SIBLING gb SBATCH (v2_d5_four_gpu_vista.sbatch), verbatim,
# including XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB. All four GPUs stay visible as
# there; every device reading is jax.devices()[0].
#
# PRE-REGISTERED:
#   - Identity gates everything. If either cross-backend hash differs, the job
#     exits 1 before any chunk arm, and no timing or memory is read as the
#     device paint's.
#   - Each chunk arm's block must equal host decode_bricks through the same
#     kernel (rc 3 otherwise).
#   - Memory, read per padded row and projected to 4096^3 (no pass/fail here,
#     the reading picks the chunk size): the per-card budget has ~19 GB of
#     headroom with a whole x-slab charged at 41.9 GB, so an x-slab chunk fits a
#     card only if the measured rate is <= ~180 B/row; a quarter-slab has room
#     up to ~715 B/row. Expected: above the 124 floor, because eager Python keeps
#     the decode's intermediates alive until it returns.
#
# BUDGET: ~30-45 min expected (the 1024^3 state build is the longest host step,
# a few minutes by a 128^3 laptop timing); 1:30 requested; <= 1.5 SU at gb's
# 1 SU/node-hr, one node.

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
echo "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=$XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB"
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

# Smoke ON THE GPU at 32^3: every arm and the whole reporting path, in minutes,
# before the long legs. It resolves the apparatus, not the question.
run_leg smoke \
  pixi run -e gpu python scripts/v2_d2d_device_paint.py --smoke --out-suffix _smoke
if [ $rc_total -ne 0 ]; then
  echo "FATAL: smoke failed, not spending the node on the long legs"
  exit 1
fi

run_leg paint \
  pixi run -e gpu python scripts/v2_d2d_device_paint.py \
    --xback-n 256 --chunks 512:4096,512:16384,1024:65536 --reps 3 \
    --out-suffix _gb

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
