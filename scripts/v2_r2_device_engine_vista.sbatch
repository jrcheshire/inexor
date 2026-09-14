#!/bin/bash
#SBATCH -J r2-device-engine
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o r2-device-engine-%j.log
# D3b R1b + R2 on a gb node, ONE bundled job (JC, 2026-09-13: consolidate):
#   1. the device test set on the GPU backend (migrate, repack, engine backend,
#      insert/eject kernels);
#   2. engine A/B at cgh64: `migrate_backend="device"` vs the host engine, K=2 +
#      lead drift, repack every step -- every state field hashed, stripped stats
#      compared: the R2 gate ON A GB200; per-boundary walls for both;
#   3. migrate-budget: the R1 owed reading -- `peak_estimate_bytes` against the
#      measured device peak in one process, cgh64;
#   4. repack-slab: a 4096^3-shape slab (nb=256, 268M rows) repacked on the
#      device, bitwise the host repack, synced phase split, device peak.
# The laptop smoke (CPU XLA) passed all four arms with the R2 gate green; the
# int16 fault of 995067 is why no device gate closes on CPU alone.
# Submit from the Vista checkout:
#   sbatch scripts/v2_r2_device_engine_vista.sbatch
#
# TIME ASK: ~25 min estimated, 45 min wall, one gb node (~0.75 SU at the wall,
# ~0.4 at the estimate). The tile loop at cgh64 on one card is UNMEASURED and is
# the largest term; bounded from cdev's serial 22.1 s/step on a B200 (sec. 5y) x8
# in particles. The unpriced leg (repack-slab) runs LAST.
#
#   phase                          each (source)                                total
#   pytest on the GPU              995638: migrate+insert+eject ~1.5 min; the
#                                  repack and engine-backend gates add ~1 min
#                                  of laptop time, compiles on top             ~3 min
#   probe smoke, 4 arms            laptop 25 s; GPU compiles                   ~1 min
#   engine-device, cgh64, K=2      build 31 s (995602); lead drift 6.4 s (device,
#                                  compiles); 2 x (tile loop <= ~180 s bound +
#                                  device migrate 1.6 s + device repack ~1 s)  ~6.5 min
#   engine-host, cgh64, K=2        same tile loop; numpy migrate 27 s x 3;
#                                  host repack 0.85 s x 2 (C16)               ~7.5 min
#   migrate-budget, cgh64          build 31 s; 2 device passes 6.4 + 1.6 s     ~1 min
#   repack-slab, nb=256            build of a 268M-row state UNMEASURED
#                                  (~3-4 min: a 268M stable sort + an 8.6e9-
#                                  bucket bincount); host repack ~1 min (65,536
#                                  live bricks, 16.8M empty); device repack x2
#                                  UNMEASURED (the R1b reading)                ~6 min
#
# QUESTIONS:
#   - the device gates pass on the GPU backend;
#   - engine A/B: hashes and stripped stats equal at cgh64 on a GB200 (rc 3 if
#     not); migrate and repack walls per step, device vs host;
#   - migrate-budget: estimate / measured peak (a ratio < 1 means the estimate
#     under-reads and the envelope refusal cannot be trusted at c-hero);
#   - repack-slab: bitwise the host at production shape; wall and device peak
#     of one 268M-row slab (x256 slabs is the c-hero step's repack bound).
#
# PRE-REGISTERED:
#   - A pytest failure or a smoke failure stops the job before the long legs.
#   - Walls and peaks are reported, not gated.
#
# ENV copied verbatim from v2_d3_r1_device_migrate_vista.sbatch.

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

run_leg jax-devices \
  pixi run -e gpu python -c "import jax; print('jax devices:', jax.devices())"
run_leg pytest-gpu-gates \
  pixi run -e gpu python -m pytest -q tests/test_migrate_device.py tests/test_repack_device.py \
    tests/test_engine_device_backend.py tests/test_insert_jax.py tests/test_eject_jax.py
run_leg smoke \
  pixi run -e gpu python scripts/v2_r2_device_engine.py --smoke --out-suffix _gbsmoke
if [ $rc_total -ne 0 ]; then
  echo "FATAL: a GPU gate or the smoke failed, not spending the node on the long legs"
  exit 1
fi

run_leg engine-and-budget \
  pixi run -e gpu python scripts/v2_r2_device_engine.py \
    --arms engine-device,engine-host,migrate-budget --engine-n 512 --steps 2 \
    --out-suffix _gb
run_leg repack-slab \
  pixi run -e gpu python scripts/v2_r2_device_engine.py \
    --arms repack-slab --shape-nb 256 --out-suffix _slab_gb

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
