#!/bin/bash
#SBATCH -J d5-4gpu
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d5-4gpu-%j.log
# D5 pulled forward: does the transform's wall actually divide by FOUR?
# Submit from the Vista checkout:
#   sbatch scripts/v2_d5_four_gpu_vista.sbatch
#
# WHERE THIS SITS. D0 charges each of a gb node's four GB200s an exact quarter
# of the coarse solve and 5y's 114 s/step floor assumes a perfect 4-way split.
# Neither is measured: every number in record secs. 7-9 is ONE GB200. D2c..D2e
# would be built on top of that assumption, so it is tested before they are.
#
# WHY IT IS GENUINELY OPEN. The factorization is embarrassingly parallel with
# zero inter-device communication -- pass 1 is independent per plane, pass 2 per
# y-pencil-plane, the spectrum is host-resident throughout -- so nothing in the
# ALGORITHM stops a 4x split. But D1 measured this path at 91% host bus, and the
# bus is the one thing four GPUs share. This is a contention measurement.
#
# STRONG SCALING: the total work is held at one 2048^3 transform and the width
# varies, because the design charges a quarter of a FIXED transform.
#
# TWO DEFECTS IN D1's APPARATUS, FIXED HERE.
#   1. D1 timed a 34.4 GB host `spec.copy()` INSIDE the inverse. That is serial
#      host work which cannot split, and left in place it would have capped the
#      observed speedup near 2.8x however well the bus scaled. Hoisted out and
#      measured on its own -- at three inverses per step it is a step-floor term
#      in its own right if it is large. (It also means D1's 19.7-23.06 s inverse
#      figures each contain it.)
#   2. Every leg records WHICH devices did work. A W=4 leg that quietly ran
#      everything on device 0 is indistinguishable from "it did not scale", and
#      it now FAILS the job rather than printing a number.
#
# ALL WIDTHS IN ONE JOB ON ONE NODE. D1 measured 1.36x of node-to-node spread on
# identical work, which is bigger than some of the effects being looked for.
#
# PRE-REGISTERED:
#   - The identity arm gates everything else. If the W=4 spectrum is not BITWISE
#     equal to the W=1 spectrum, every timing number in this job is a wall for a
#     different transform and the job is read as FAILED, not as slow.
#   - Both modes scale >=3.4x: the quarter-charge holds, the coarse solve on
#     four GPUs is ~20-30 s/step as built, D0's premise stands.
#   - Pinned scales, pageable saturates: pinning is a REQUIREMENT for using the
#     node, not an optimization. Changes D2's priority, not the budget.
#   - Pinned also saturates near 2x: the exact-quarter charge is WRONG. D0's
#     per-GPU column and 5y's 114 s/step floor both need restating, and the
#     margin falls. This is the outcome the reading is owed for.
#   - threads vs processes at W=4 differing by more than ~1.3x means the GIL is
#     the variable rather than the node, and the engine's process structure
#     becomes a design constraint. Same traffic, same width, one node.
#   - A pinned FFT PATH IS NOT BUILT HERE (it needs the slab window itself to
#     live in pinned host memory, which is pipeline work). So pinned-at-four-
#     GPUs is a measured scaling times a PROJECTED single-GPU rate, and is
#     reported as such -- not as a measured wall.
#
# BUDGET: ~40 min expected against 1:30; ~1.5 SU worst case at gb's 1 SU/node-hr.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$SCRATCH/src/inexor" 2>/dev/null || cd "$HOME/src/inexor" || {
  echo "FATAL: no inexor checkout"; exit 1; }
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=900
unset JAX_PLATFORMS
# ALL FOUR GPUs stay visible in every leg: the width is chosen in code, so
# driver init does not vary with W. Only the process-driver control sets
# CUDA_VISIBLE_DEVICES, per child, which is the point of that arm.
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

# Smoke on the node's CPU with four FORCED devices: exercises the partition, the
# threads, the process barrier and the bitwise identity before a GPU is spent.
# It resolves the apparatus, not the question -- nothing here is a prediction of
# the GPU numbers.
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  run_leg smoke \
  pixi run -e gpu python scripts/v2_d5_four_gpu_fft.py --smoke --out-suffix _d5smoke
if [ $rc_total -ne 0 ]; then
  echo "FATAL: smoke failed, not spending four GPUs on it"
  exit 1
fi

run_leg four-gpu \
  pixi run -e gpu python scripts/v2_d5_four_gpu_fft.py \
    --n 2048 --receipt-n 512 --widths 1,2,4 \
    --slab 32 --plane-batch 1 --pencil-batch 64 --reps 2 \
    --out-suffix _gb

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
