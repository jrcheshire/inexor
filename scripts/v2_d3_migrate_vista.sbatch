#!/bin/bash
#SBATCH -J d3-migrate
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d3-migrate-%j.log
# D3 on a gb node: the compiled migrate on one GB200 -- bitwise against numpy, the
# step's split between device calls and host bookkeeping at cgh64, and one 4096^3
# slab through the compiled eject and insert, phase by phase.
# Submit from the Vista checkout:
#   sbatch scripts/v2_d3_migrate_vista.sbatch
#
# TIME ASK: ~9 min estimated, 45 min wall, one gb node, billed at the 15 min
# minimum (~0.25 SU).
#
#   phase                          each (source)                               total
#   smoke, 3 arms at 32-64^3       3 processes, ~10 s start each (993849);
#                                  laptop smoke ran in ~2 s of compute         ~0.5 min
#   xback, cdev, 2 steps           build ~4 s (1024^3 built in 244 s, 994675,
#                                  /64); numpy migrate on an arena-occupied
#                                  cdev ~14 s/step (scaling record 5g) plus
#                                  the compiled pass and its compiles, same
#                                  order (UNMEASURED on a GB200)               ~1.5 min
#   slab-real, cgh64, 2 steps      build ~30 s (/8 of 1024^3); numpy serial
#                                  migrate 24.8 s/step (gg, scaling record
#                                  5s card); compiled pass + compiles same order
#                                  (UNMEASURED)                                ~2.5 min
#   slab-shape, 4096^3 slab, x2    generate 268M rows (UNMEASURED, ~1 min);
#   (drift 0.5 and 0.13)           eject first call 10.6 s + 6 calls x 0.53 s
#                                  (gb probe, scaling record 5y); insert first
#                                  call and 6 calls UNMEASURED, taken as the
#                                  eject's x3                                  ~4 min
#
# QUESTIONS, one card per invocation:
#   xback       Is `drift_and_migrate` with `eject_jax` + `insert_jax` on a GB200
#               bitwise the numpy migrate, field by field and stats dict, over
#               two steps with rows crossing two bricks, arena overflow and
#               re-homed residents? (The laptop suite proves it on CPU XLA only.)
#   slab-real   At cgh64 (nb=32, 4.19M rows per slab), how does one compiled
#               migrate step divide between the device calls (upload + compute +
#               readback, synchronous) and the host (slot resolution, census,
#               writes, arena), against the numpy migrate of the same state? How
#               many kernel shapes compile per step?
#   slab-shape  At a true 4096^3 slab (65,536 bricks, 268,435,456 rows), what do
#               the compiled eject and insert each cost, split into host prep,
#               upload, compute and readback, and what device peak? Run at the
#               probe's default drift (0.5 brick, ~18% leavers on the laptop
#               smoke) and at 0.13 brick, near the gb probe's 4.6% leavers, whose
#               eject read 14 ms compute / 0.528 s end to end -- a reproduction
#               receipt for the eject half.
#
# PRE-REGISTERED:
#   - Every smoke arm passes on the GPU or the long legs do not run.
#   - xback: all fields equal and stats equal at both steps (rc 3), and every
#     anti-vacuity condition met (rc 4). A failure here voids every timing: the
#     orchestrator stops before slab-real and slab-shape.
#   - slab-real: equal to the numpy migrate at both steps (rc 3); eject and
#     insert call receipts = nb per step (rc 4). Host share reported, not gated.
#   - slab-shape: the split program equals the public call and repeats are
#     equal (rc 3). Phases reported, not predicted; this job prices where the
#     migrate lives, it does not choose.
#
# ENV copied verbatim from the D2f bundle sbatch (v2_d2f_bundle_vista.sbatch).

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

# the smoke runs on the GPU (no --allow-cpu), so a CPU fallback fails it here
run_leg smoke \
  pixi run -e gpu python scripts/v2_d3_device_migrate.py \
    --xback-n 32 --real-n 64 --shape-nb 8 --reps 1 --out-suffix _gbsmoke
if [ $rc_total -ne 0 ]; then
  echo "FATAL: smoke failed, not spending the node on the long legs"
  exit 1
fi

run_leg migrate \
  pixi run -e gpu python scripts/v2_d3_device_migrate.py \
    --arms xback,slab-real,slab-shape --reps 3 --out-suffix _gb
if [ $rc_total -ne 0 ]; then
  echo "FATAL: the migrate leg failed; the second slab reading is not run"
  exit 1
fi

run_leg slab-lowdrift \
  pixi run -e gpu python scripts/v2_d3_device_migrate.py \
    --arms slab-shape --shape-frac 0.13 --reps 3 --out-suffix _gb_lowdrift

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
