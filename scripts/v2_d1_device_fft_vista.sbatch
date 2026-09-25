#!/bin/bash
#SBATCH -J d1-device-fft
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d1-device-fft-%j.log
# D1: the plane-factorized device FFT, at the size the coarse solve runs.
# Submit from the Vista checkout:
#   sbatch scripts/v2_d1_device_fft_vista.sbatch
#
# WHY. The coarse solve at 4096^3 is a 2048^3 transform and two of its three
# possible forms are already dead: the monolithic device transform does not fit
# (peak/field 8.0x, 275 GB against 185 GiB) and at 1536^3 it is silently WRONG
# (record 5y finding 2); the host out-of-core form costs 417 s/step, 39% of the
# 1080 s bar on its own. The third form is what this job measures, against the
# 0.4 s/step the gb probe projected for it from scaled n^3 log n. That
# projection is device COMPUTE only and this transform moves 34.4 GB across the
# bus twice per pass, so expect the reading to be transfer-dominated and expect
# the gap to be attributable rather than a surprise -- the probe's eject arm
# read 14 ms on device against 0.528 s end to end for the same reason.
#
# LEGS (one process per arm: a device high-water never resets):
#   1  smoke on the CPU backend. Gates the card write, the verdict path and
#      every worker seam before a GPU is touched. Hard exit on failure.
#   2  the roundtrip ladder, 512 / 1024 / 2048^3 f32, on one GB200. This is the
#      correctness witness AND the residency witness.
#   3  the wall at the largest size that roundtripped, over plane_batch 1/4/16.
#   4  the HOST out-of-core form at the same n, in this job. Its 107.4/101.4 s
#      is on record from 972737 and could have been imported -- but an imported
#      leg turns a within-run ratio into a cross-machine one that still prints
#      as within-run, so it runs here.
#
# PRE-REGISTERED, so the readout cannot be chosen after the fact:
#   - 1024^3 f32 must roundtrip at ~1e-6. It read 2.9e-6 on this stack in
#     972737, so anything else means the STACK moved, not the code.
#   - 2048^3 must roundtrip at the same order. If it does not, the factorized
#     form inherits the monolithic form's silent-wrong class and D1 has
#     FAILED -- that is a real possible outcome of this job and it kills the
#     device coarse solve, not just this implementation.
#   - Device peak must stay O(plane): `peak_over_spec` well under 1. A peak of
#     order the 34.4 GB spectrum means it materialized on the device and the
#     design's memory argument is void.
#   - The wall is a reading, not a gate. There is no threshold on it here: the
#     bar is 1080 s/step for the WHOLE step and the coarse solve is one term.
#
# BUDGET: ~45 min expected against a 2 h request (the 2048^3 host leg is ~12
# min of it and the noise regeneration at 8.6e9 gaussians per pass is
# untimed); ~2.1 SU worst case if it runs the full wall.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$SCRATCH/src/inexor" 2>/dev/null || cd "$HOME/src/inexor" || {
  echo "FATAL: no inexor checkout"; exit 1; }
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache
# Taken from the sibling gb sbatch rather than reasoned about: this job does not
# ask for a pinned host memory space, but the one time a gb sbatch here omitted
# this flag it measured the 64 GB default instead of the node and lost its
# load-bearing leg. Set and echoed as a receipt.
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=900
unset JAX_PLATFORMS
unset CUDA_VISIBLE_DEVICES

echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID ==="
echo "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=$XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
free -g | head -2

rc_total=0
run_leg () {  # run_leg <name> <cmd...>
  local name="$1"; shift
  echo ""
  echo "=== LEG $name ==="
  "$@"
  local rc=$?
  echo "--- LEG $name rc=$rc ---"
  [ $rc -ne 0 ] && rc_total=$((rc_total + 1))
  return $rc
}

# LEG 1 -- the smoke gates everything. A device leg that runs after a broken
# worker seam produces a card nobody can read.
JAX_PLATFORMS=cpu run_leg smoke \
  pixi run -e gpu python scripts/v2_d1_device_fft.py --smoke --out-suffix _d1smoke
if [ $rc_total -ne 0 ]; then
  echo "FATAL: smoke failed, not spending a GPU on it"
  exit 1
fi

# LEGS 2-4 -- one GPU. The script spawns each arm in its own process.
CUDA_VISIBLE_DEVICES=0 run_leg device-fft \
  pixi run -e gpu python scripts/v2_d1_device_fft.py \
    --sizes 512,1024,2048 --batches 1,4,16 --slab 32 --reps 3 \
    --out-suffix _gb

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
