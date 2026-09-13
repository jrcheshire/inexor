#!/bin/bash
#SBATCH -J d2f-cards
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d2f-cards-%j.log
# D2f on a gb node: the coarse mesh on the cards (record sec. 23, JC's placement).
# Submit from the Vista checkout:
#   sbatch scripts/v2_d2f_cards_vista.sbatch
#
# TIME ASK: ~8 min estimated, 30 min wall, one gb node; billed at the 15 min
# minimum unless it overruns, <= 0.5 SU.
#
#   phase                          each (source)                               total
#   smoke, 32^3, 5 processes       ~10 s start each + compiles (laptop smoke)  ~1.5 min
#   xback-host, cdev               builds 3.7 + 7.4 s, host paints 8.9 + 8.4 s
#                                  (993294), CPU jit paint on Grace unmeasured  ~1 min
#   xback-cards, cdev, 1 + 4 cards two builds (as above), jitted paint 1.2 s per
#                                  state (993294) x 2 widths, one chunk-program
#                                  compile per card (~5-7 s each, 993139 warm
#                                  calls), 128^3 solves                        ~2 min
#   cards-scale, 4096^3 shards     start 10 s; four 17.3 GB accumulators;
#                                  1024 synced adds + fold + 2048 per-plane
#                                  decodes (all UNMEASURED); density to host
#                                  34.4 GB at 3.8-13.9 GB/s (sec. 10) 3-9 s;
#                                  two forward transforms at W=4, 6.86 s each
#                                  (sec. 10); a 34.4 GB spectrum comparison
#                                  ~15 s                                       ~3 min
#
# The unmeasured terms sit in the last arm, so an overrun costs only it.
#
# QUESTIONS, one card per arm:
#   xback-cards  Is the density painted, accumulated, folded and decoded on the
#                cards (`device.paint.coarse_delta_cards`) bitwise the CPU host
#                engine's, on one GB200 and across four? Are the force meshes
#                solved from the cards bitwise the solve of the same density from
#                host memory on the GPU?
#   cards-scale  At 4096^3 shard shapes (synthetic blocks; the paint's own cost is
#                sec. 13-15): what do the adds, the ghost fold and the decode cost a
#                card, what does a card hold, and is the forward transform from the
#                cards faster than from host slabs at the same width -- with both
#                legs of that ratio in this job.
#
# PRE-REGISTERED:
#   - Identity gates everything. Any xback-cards miss -- a hash at either width
#     or state, a four-card leg served by fewer than four devices, no ghost plane
#     carrying mass, or the solve from the cards differing from the host
#     density's solve -- exits 1 before cards-scale.
#   - cards-scale's receipts: the owned mass equals blocks x block cells exactly;
#     sampled planes (including a folded card edge) decode bitwise against numpy;
#     the two spectra are bitwise equal. Any miss is rc 3 and its timings are not
#     read. The sampled planes are also the first GPU reading of the decode's
#     plane-shaped divisor (CPU XLA divides by a scalar as a reciprocal multiply).
#   - Per-card peaks are read against the planner's 17.28 GB accumulator plus
#     8.59 GB density per card.
#   - Walls are reported, not predicted.
#
# ENV copied verbatim from the D2d jit gb sbatch (v2_d2d_jit_paint_vista.sbatch).

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

# Smoke ON THE GPUs at 32^3: every arm and the reporting path.
run_leg smoke \
  pixi run -e gpu python scripts/v2_d2d_device_paint.py --smoke --skip-xback-dev \
    --card-widths 1+4 --scale --out-suffix _d2f_smoke
if [ $rc_total -ne 0 ]; then
  echo "FATAL: smoke failed, not spending the node on the long legs"
  exit 1
fi

run_leg d2f \
  pixi run -e gpu python scripts/v2_d2d_device_paint.py \
    --xback-n 256 --reps 3 --skip-xback-dev --card-widths 1+4 --chunks none --scale \
    --out-suffix _d2f_gb

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
