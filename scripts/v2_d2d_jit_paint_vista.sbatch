#!/bin/bash
#SBATCH -J d2d-jit
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d2d-jit-%j.log
# D2d follow-up on a GB200: the JITTED device coarse paint -- still bitwise, and
# what it saves against the eager paint in memory and time.
# Submit from the Vista checkout:
#   sbatch scripts/v2_d2d_jit_paint_vista.sbatch
#
# WHY. Job 993139 measured the eager paint at 266 B per padded row on the card
# (a whole 4096^3 x-slab chunk does not fit, 1.15x) and ~0.8 s per quarter-slab
# chunk, which puts the paint at 19-26% of the 1080 s/step bar on four cards.
# The traced program holds 121 B/row under last-use freeing, so a single compiled
# chunk has room to fall. `device.paint` now compiles decode + containment +
# paint into one program per step (`jit=True`, fixed per-step shapes, traced
# sub-block origin).
#
# THE RISK THE GATE IS FOR. XLA may fuse the TSC weight arithmetic; a fused
# multiply-add rounds differently and can move a rounded integer weight. The
# laptop gate (CPU jit == host) paints few rows, so a rare tie is unlikely to show
# there. Here the jitted paint must hash-equal a CPU-only host engine at cdev,
# plain and with arena residents -- tens of millions of rows x 27 corners -- and
# every jitted chunk block (up to 268M rows) must equal host decode_bricks through
# the eager kernel.
#
# EAGER AND JIT IN ONE JOB, at each chunk size, because a jit/eager ratio across
# two jobs carries 1.28-1.36x of node-to-node spread. Each arm is its own
# subprocess, so each device high-water is its own.
#
# ENV copied verbatim from the D5 gb sbatch (v2_d5_four_gpu_vista.sbatch),
# including XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB.
#
# PRE-REGISTERED:
#   - Identity gates everything. Any cross-backend hash mismatch (eager or jit,
#     plain or arena, or CPU-jit vs CPU-host) exits 1 before any chunk arm. A jit
#     miss means jit is NOT adoptable as built, and that is the answer.
#   - Each chunk arm's block equals host decode through the eager kernel (rc 3
#     otherwise), and a jit arm compiles exactly once (`jit_traces` 1).
#   - Memory, per padded row, read against the per-card budget: a whole x-slab
#     chunk fits a card only at <= ~180 B/row, a quarter-slab at <= ~715. The
#     eager arms should reproduce 993139's 265-272 on this node.
#   - Wall is reported, not predicted.
#
# BUDGET: two 1024^3 state builds (~4 min each) plus four 512^3 (~30 s) plus the
# arms; ~25-40 min expected against 1:30, <= 1.5 SU at gb's 1 SU/node-hr.

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

# Smoke ON THE GPU at 32^3, eager and jit: every arm and the reporting path.
run_leg smoke \
  pixi run -e gpu python scripts/v2_d2d_device_paint.py --smoke --out-suffix _jitsmoke
if [ $rc_total -ne 0 ]; then
  echo "FATAL: smoke failed, not spending the node on the long legs"
  exit 1
fi

run_leg paint \
  pixi run -e gpu python scripts/v2_d2d_device_paint.py \
    --xback-n 256 --reps 3 --out-suffix _jit_gb \
    --chunks 512:4096:eager,512:4096:jit,512:16384:eager,512:16384:jit,1024:65536:eager,1024:65536:jit

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
