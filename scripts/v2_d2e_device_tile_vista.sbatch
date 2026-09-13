#!/bin/bash
#SBATCH -J d2e-tile
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d2e-tile-%j.log
# TIME ASK: ~5 min of measured-rate phases plus the CPU cross-backend arm, whose
# per-tile rate on Grace is UNMEASURED; wall is set once that is priced.
#
#   phase                     count  each (source)                          total
#   smoke, 4 arms at 32^3       1    ~45 s (993600's smoke leg)            ~45 s
#   xback-cpu, cdev             1    10 s start + 7.6 s build (993139)
#                                    + 8 tiles x UNMEASURED CPU tile        ?
#   xback-gpu, cdev             1    10 s + 7.6 s + ~20 s compile (972737
#                                    P=576: 21 s) + 8 tiles < 1 s          ~45 s
#   short, P=576                1    10 s + 29.5 s build (993139) + 21 s
#                                    compile + 4 x 6 s host decode (0.25
#                                    us/row, B200 cdev card)               ~1.5 min
#   tile, P=576                 1    10 s + 29.5 s + ~25 s compile
#                                    + 4 tiles x ~1 s                      ~1.2 min
#
# TWO QUESTIONS, one card:
#   1. The device tile's GPU-vs-CPU floor, before any cross-backend tolerance is
#      set: `one_tile` alone on identical inputs, then the jitted tile's short
#      force, long force and kicked velocity, in eps x rms; layout must be exact.
#      cdev (P=320, 8 tiles, arena residents), ratified dtypes (f64 fine, f32
#      coarse). The CPU arm also re-runs check A (eager device tile == host
#      tile_task) on this machine.
#   2. What one tile costs a card at the 4096^3 tile shape: cgh64 with tile 512 /
#      buffer 32 gives P=576 and ~23.9M rows per tile. `one_tile` alone (like the
#      gb probe's 94.2 ms/tile) and the jitted tile, each in its own process,
#      device time and peak.
#
# NO BAR is applied to question 1; the card is the floor a bar is set from.
#
# ENV copied verbatim from the D2d gb sbatch (v2_d2d_dead_rows_vista.sbatch).

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

WORK="$SCRATCH/inexor_runs/d2e_xback_$SLURM_JOB_ID"
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
  pixi run -e gpu python scripts/v2_d2e_device_tile.py --smoke \
    --workdir "$WORK/smoke" --out-suffix _smoke
if [ $rc_total -ne 0 ]; then
  echo "FATAL: smoke failed, not spending the node on the long arms"
  exit 1
fi

run_leg d2e \
  pixi run -e gpu python scripts/v2_d2e_device_tile.py --workdir "$WORK/cdev" \
    --out-suffix _gb

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
