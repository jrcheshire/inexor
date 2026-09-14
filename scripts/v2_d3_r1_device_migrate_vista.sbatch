#!/bin/bash
#SBATCH -J d3-r1-migrate
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d3-r1-migrate-%j.log
# D3b R1 on a gb node: is `device.migrate.drift_and_migrate_device` bitwise the
# serial numpy migrate ON A GB200, and what does a cgh64 step cost on one card?
# The laptop gate (CPU XLA) passed with five mutants caught; the int16 fault of
# 995067 is why no device gate closes on CPU alone.
# Submit from the Vista checkout:
#   sbatch scripts/v2_d3_r1_device_migrate_vista.sbatch
#
# TIME ASK: ~10 min estimated, 45 min wall, one gb node, billed at the 15 min
# minimum (~0.25 SU).
#
#   phase                          each (source)                               total
#   pytest on the GPU: device      laptop 8 s + 2 s (CPU); GPU compiles of the
#   migrate + insert/eject gates   rows/put/write/eject/insert programs at
#                                  32^3 shapes, ~1-3 s each (995264 smoke)     ~1.5 min
#   probe smoke, device arms       32^3 and 64^3, ~10 s start each             ~0.5 min
#   device-xback, cdev, 2 steps    build ~4 s; numpy 4.5 s/step (995264);
#                                  device pass UNMEASURED on a GPU: 16 slabs,
#                                  compiles dominate step 0                    ~2 min
#   device-real, cgh64, 2 steps    build 30.8 s; numpy 27 s/step (995264);
#                                  device pass UNMEASURED: 32 slabs            ~4 min
#
# QUESTIONS:
#   - tests/test_migrate_device.py (and the insert/eject kernel gates) pass on the
#     GPU backend.
#   - device-xback (cdev, zero slack, 30% arena, reach 2): state and stats bitwise
#     the numpy migrate at both steps.
#   - device-real (cgh64, slack 0.10, 1% arena): bitwise at both steps; device
#     step wall against numpy's 27 s and sec. 27's compiled-kernel host path
#     15.2 s; programs compiled per step; device peak.
#
# PRE-REGISTERED:
#   - A pytest failure or a non-bitwise step is a failed gate (rc 3), read before
#     any wall.
#   - Walls and peak are reported, not gated. Step 0 carries compilation; step 1
#     is the steady reading.
#
# ENV copied verbatim from the D3 migrate sbatch (v2_d3_migrate_vista.sbatch).

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

run_leg pytest-gpu \
  pixi run -e gpu python -c "import jax; print('jax devices:', jax.devices())"
run_leg pytest-gpu-gates \
  pixi run -e gpu python -m pytest -q tests/test_migrate_device.py tests/test_insert_jax.py \
    tests/test_eject_jax.py
run_leg smoke \
  pixi run -e gpu python scripts/v2_d3_device_migrate.py --arms device-xback,device-real \
    --xback-n 32 --real-n 64 --steps 2 --reps 1 --out-suffix _r1_gbsmoke
if [ $rc_total -ne 0 ]; then
  echo "FATAL: a GPU gate or the smoke failed, not spending the node on the long legs"
  exit 1
fi

run_leg device-migrate \
  pixi run -e gpu python scripts/v2_d3_device_migrate.py --arms device-xback,device-real \
    --xback-n 256 --real-n 512 --steps 2 --out-suffix _r1_gb

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
