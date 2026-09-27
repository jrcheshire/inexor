#!/bin/bash
#SBATCH -J d8-hero-ics
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -o d8-hero-ics-%j.log
# 4096^3 ICs (LCDM second-order growth) by the device generator on four GB200s, into a
# new directory. Legs:
#   1. preflight: devices, scratch space, planner verdicts (a refusal stops the job);
#   2. smoke at 32^3: device generator vs host generator, 4 cards == 1 card (stops the job);
#   3. the 4096^3 generation -> $HERO_IC_DIR (refuses if a manifest is already there);
#   4. manifest check: growth2 = lcdm, 4096^3 particles in 256 slabs;
#   5. the GPU pytest set for the generator.
#
# Requires: INEXOR_RUNS (run root), HERO_IC_DIR (IC output directory), and INEXOR_SRC or
# submission from the inexor checkout. The account is passed with `sbatch -A`:
#   INEXOR_RUNS=$SCRATCH/inexor_runs HERO_IC_DIR=$SCRATCH/inexor_runs/c-hero-lcdm \
#     sbatch -A <account> scripts/run/hero_ics_vista.sbatch
#
# Wall: the generation takes ~67 min, the smoke ~1 min, the pytest ~3; 3 h leaves margin.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
# Run from the inexor checkout: INEXOR_SRC, or the directory sbatch was called from.
cd "${INEXOR_SRC:-$SLURM_SUBMIT_DIR}" && [ -f src/inexor/engine.py ] || {
  echo "FATAL: not an inexor checkout: ${INEXOR_SRC:-$SLURM_SUBMIT_DIR} (set INEXOR_SRC)"; exit 1; }
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=900
unset JAX_PLATFORMS
unset CUDA_VISIBLE_DEVICES

RUNS=${INEXOR_RUNS:?set INEXOR_RUNS to the run root, e.g. $SCRATCH/inexor_runs}
HERO=${HERO_IC_DIR:?set HERO_IC_DIR to the directory the ICs are written to}
CARDS=runs/v2
TAG=d8_ics_$SLURM_JOB_ID
GPU_LOG=$CARDS/${TAG}_gpu.csv

echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
free -g | head -2
df -h "$RUNS" | tail -1
env | grep -E '^XLA_' | sort

rc_total=0
run_leg () {
  local name="$1"; shift
  echo ""; echo "=== LEG $name ($(date +%H:%M:%S)) ==="
  "$@"; local rc=$?
  echo "--- LEG $name rc=$rc ($(date +%H:%M:%S)) ---"
  [ $rc -ne 0 ] && rc_total=$((rc_total + 1))
  return $rc
}

if [ -e "$HERO/manifest.json" ]; then
  echo "FATAL: $HERO already holds a manifest; this job does not overwrite ICs"; exit 1
fi

run_leg preflight \
  pixi run -e gpu python scripts/run/device_ics.py preflight --root "$RUNS" --need-gb 1500 \
  || { echo "FATAL: preflight refused"; exit 1; }
run_leg smoke \
  pixi run -e gpu python scripts/run/device_ics.py smoke --out $CARDS/${TAG}_smoke.json \
  || { echo "FATAL: the smoke missed, not spending the node on the generation"; exit 1; }

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 10 > "$GPU_LOG" 2>/dev/null &
SAMPLER=$!
run_leg ics-4096 \
  pixi run -e gpu python scripts/run/realization.py ics --config c-hero --workdir "$HERO" \
    --generator device
kill $SAMPLER 2>/dev/null
[ -f "$HERO/realization_ics.json" ] && cp "$HERO/realization_ics.json" $CARDS/${TAG}_c-hero.json

run_leg manifest-check \
  python3 -c "
import json, sys
m = json.load(open('$HERO/manifest.json'))
print('growth2', m.get('growth2'), 'n_particles', m['n_particles'], 'slabs', len(m['files']),
      'ic_stream', m.get('ic_stream'))
sys.exit(0 if (m.get('growth2') == 'lcdm' and m['n_particles'] == 4096 ** 3
               and len(m['files']) == 256) else 1)
"

run_leg pytest-gpu \
  pixi run -e gpu python -m pytest -q tests/test_icgen_device.py tests/test_device_emit.py \
    tests/test_icgen.py

du -sh "$HERO" 2>/dev/null
echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
