#!/bin/bash
#SBATCH -J d1-fft2
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d1-fft2-%j.log
# D1 follow-up: the axis 974643 never varied, and the term it timed by mistake.
# Submit from the Vista checkout:
#   sbatch scripts/v2_d1_device_fft2_vista.sbatch
#
# WHAT 974643 SETTLED, so this job does not re-litigate it: the factorized
# device transform is CORRECT at 2048^3 (roundtrip 6.68e-6, the same order as
# 512's 4.05e-6 and 1024's 6.20e-6 -- it does NOT inherit the monolithic form's
# silent-wrong class), and it STREAMS (device peak 117 MB against a 34.4 GB
# spectrum, 0.0034x). Those are the two witnesses that decided the design.
#
# WHAT IT GOT WRONG, both mine, both found by reading the card:
#
# 1. THE FORWARD LEG TIMED THE HOST RNG. Its forward read 77.8 s against its own
#    inverse's 31.3 s, and the only thing a forward has that an inverse does not
#    is generating the field -- 8.6e9 gaussians at 2048^3. The device and host
#    legs put the excess at 46.5 and 44.1 s, agreeing to 5% as they must if it
#    is the shared host RNG. So the honest coarse-solve figure is ~4 x 31.3 =
#    125 s/step on ONE GPU, not the 170 that job printed. That is an inference
#    and this job makes it a reading: `--source flat` takes generation out of
#    the timed region, and a `noise` arm times generation on its own.
#
# 2. THE LADDER VARIED THE WRONG AXIS. plane_batch (pass 1) read
#    77.78 / 77.53 / 77.76 s over 1 / 4 / 16 -- identical digits, a dead knob --
#    while pencil_batch (pass 2) sat pinned at 1 in every leg. Pass 2 is the
#    strided one: it reads spec[:, y, :] across the whole 34.4 GB spectrum, 2048
#    times, with 16.8 MB between consecutive x rows. An axis nothing varied is
#    not an axis anything was learned about.
#
# THE KNOB IS ALIVE, on the laptop: the same numpy pass-2 code at n=512 reads
# 0.403 s at pencil_batch=1 against 0.118 at 8. That is a SMALL CONFIG on
# different hardware and it bounds the payoff here in NEITHER direction -- it is
# why the ladder is worth running, not a prediction of what it will say.
#
# PRE-REGISTERED:
#   - The coarse solve's honest one-GPU figure is `4 x inv` at source=flat. If
#     the pencil ladder is flat to within a few percent, pass 2's access pattern
#     is NOT the cost and the next suspect is the per-plane host<->device
#     transfer, which is a different fix.
#   - `noise` must land near 45 s or the attribution above is wrong and the
#     forward/inverse gap needs another explanation.
#   - No threshold on the wall. 1080 s is the bar for a WHOLE step; the coarse
#     solve is one term, and after 974643 it is a term worth ~11% of the bar
#     rather than the 0.04% the probe projected.
#
# BUDGET: ~30 min expected against 1.5 h; ~1.6 SU worst case.

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

rc_total=0
run_leg () {
  local name="$1"; shift
  echo ""; echo "=== LEG $name ==="
  "$@"; local rc=$?
  echo "--- LEG $name rc=$rc ---"
  [ $rc -ne 0 ] && rc_total=$((rc_total + 1))
  return $rc
}

JAX_PLATFORMS=cpu run_leg smoke \
  pixi run -e gpu python scripts/v2_d1_device_fft.py --smoke --out-suffix _d1smoke2
if [ $rc_total -ne 0 ]; then
  echo "FATAL: smoke failed, not spending a GPU on it"
  exit 1
fi

# 512 is a cheap receipt that the stack has not moved under us; 2048 is the
# load-bearing one and is re-run because this commit changed the script.
CUDA_VISIBLE_DEVICES=0 run_leg fft-pencil-ladder \
  pixi run -e gpu python scripts/v2_d1_device_fft.py \
    --sizes 512,2048 --batches 1 --pencil-batches 1,8,64,256 \
    --source flat --slab 32 --reps 2 --out-suffix _gb2

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
