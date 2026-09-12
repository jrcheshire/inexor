#!/bin/bash
#SBATCH -J d5b-staged
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d5b-staged-%j.log
# D5b: does staging through pinned memory beat what the driver does for us?
# Submit from the Vista checkout:
#   sbatch scripts/v2_d5b_staged_transfer_vista.sbatch
#
# THE DECISION THIS MAKES. JC chose (2026-09-12) to pay for pinning ONCE, at the
# pipeline level, rather than retrofit it into each device phase -- D5 measured
# 7.8x between the two memory kinds at width, and the two phases already on the
# device hand the GPU ordinary numpy, so they are on the slow path too. What is
# undecided is the ROUTE, and one number decides it.
#
# WHY THE ROUTE IS NOT OBVIOUS. Memory a program allocates normally can be moved
# by the OS, so a GPU cannot DMA from it: the driver copies it into staging
# memory of its own on every crossing. Pinned memory is page-locked and the DMA
# engine reads it directly. But the engine's host buffers are numpy arrays that
# host code writes IN PLACE, and JAX's pinned arrays are IMMUTABLE, so the
# allocation cannot simply be swapped. That leaves staging each crossing through
# pinned memory -- which pays an EXPLICIT host copy. The pageable path already
# pays that copy inside the driver, so the question is whether the driver is
# doing something worse than a plain memcpy. If it is, staging is nearly free
# and the route is cheap. If it is not, the explicit copy eats the gain and
# pinning is only worth having if the buffers themselves can be made pinned AND
# numpy-writable, which is a much larger change needing its own pricing.
#
# MEASURED ON THE REAL TRANSFORM, NOT ON A PROXY. 991236 showed why: the
# synthetic traffic proxy took 1.69x LONGER than the inverse whose traffic it
# claims to replicate (36.60 s against 21.68), so it cannot carry an attribution
# and must not carry a route decision. The transfer policy is therefore a
# parameter of `ooc_fft`'s device passes and the timing arm measures it end to
# end. The proxy legs still run, as the memory-kind CEILING only.
#
# PRE-REGISTERED:
#   - The bitwise identity gates it: `staged` and `pageable` differ in WHERE the
#     host buffer lives and in nothing else, so any bit that moves means the
#     policy did something it was not asked to and the job reads as FAILED.
#   - staged lands near the pinned ceiling: route settled and cheap. Keep numpy
#     buffers everywhere, stage every crossing, apply it to the decode and kick
#     that are already on device, and D2c..D2e inherit it.
#   - staged lands near pageable: the explicit copy eats the gain. Do NOT build
#     it; price the pinned-and-writable allocation instead, as its own question.
#   - staged lands between: report the fraction recovered and let JC choose --
#     that is a cost/benefit call, not a measurement.
#   - A staged leg on a backend that cannot stage is a FAILURE here, not a skip.
#     `--allow-no-staging` is passed to the CPU smoke and to nothing else: a
#     staged arm that silently ran pageable would report that staging buys
#     nothing, which is the one wrong answer this job could produce.
#   - CONTROL: one leg runs the package's uncommitted path (jax places the
#     buffers rather than being told a device). It is the only code difference
#     between this apparatus and D1's, and it is the standing suspect for why
#     the proxy disagrees with the transform.
#
# BUDGET: ~25 min expected against 1 h; ~1 SU worst case at gb's 1 SU/node-hr.

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

# CPU smoke: four forced devices, and staged legs are allowed to SKIP here
# because jax's CPU backend refuses the pinned_host -> device move. That flag is
# deliberately absent from the GPU leg below.
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  run_leg smoke \
  pixi run -e gpu python scripts/v2_d5_four_gpu_fft.py --smoke \
    --widths 1,4 --no-processes --allow-no-staging --out-suffix _d5bsmoke
if [ $rc_total -ne 0 ]; then
  echo "FATAL: smoke failed, not spending four GPUs on it"
  exit 1
fi

run_leg staged-ab \
  pixi run -e gpu python scripts/v2_d5_four_gpu_fft.py \
    --n 2048 --receipt-n 512 --widths 1,4 --transfers pageable,staged \
    --slab 32 --plane-batch 1 --pencil-batch 64 --reps 2 \
    --no-processes --out-suffix _gb

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
