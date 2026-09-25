#!/bin/bash
#SBATCH -J d8-hero-ics
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d8-hero-ics-%j.log
# D8, job 1 of the 120-step 4096^3 realization: the ICs, with LCDM second-order
# growth (D-v2-25), by the device generator on four GB200s, into a NEW directory.
# c-hero-r0 (EdS D2) is kept untouched; the run driver refuses it for an LCDM run.
#   1. preflight: devices, scratch space, planner verdicts (stops the job);
#   2. smoke at 32^3: device generator vs host generator, 4 cards == 1 card (stops the job);
#   3. the 4096^3 generation -> $HERO (refuses if a manifest is already there);
#   4. the GPU pytest set for the generator.
# The leg set is v2_d6_device_ics_vista.sbatch's minus its 2048^3 probe and allocator
# A/B, which were measurements that are done; env copied from it verbatim.
#
# TIME: 998798 wrote c-hero-r0 in 66.7 min with this generator and allocator; the
# smoke is ~1 min and the pytest ~3. Nothing in D-v2-25 touches the IC stage's cost
# (D2/f2 are two scalars). CENTRAL ~1.2 h; REQUESTED 3:00 (~1.2 SU, max 3).
#
# PRE-REGISTERED:
#   - preflight or smoke failure stops the job before the generation;
#   - the manifest must record growth2 = lcdm and 4096^3 particles in 256 slabs;
#   - wall and host peak are reported against 998798 (66.7 min), not gated.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$SCRATCH/src/inexor" 2>/dev/null || cd "$HOME/src/inexor" || {
  echo "FATAL: no inexor checkout"; exit 1; }
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=900
unset JAX_PLATFORMS
unset CUDA_VISIBLE_DEVICES

RUNS=/scratch/10303/jrc4/inexor_runs
HERO=${HERO_IC_DIR:-$RUNS/c-hero-lcdm-r1}
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
  pixi run -e gpu python scripts/v2_d6_device_ics.py preflight --root "$RUNS" --need-gb 1500 \
  || { echo "FATAL: preflight refused"; exit 1; }
run_leg smoke \
  pixi run -e gpu python scripts/v2_d6_device_ics.py smoke --out $CARDS/${TAG}_smoke.json \
  || { echo "FATAL: the smoke missed, not spending the node on the generation"; exit 1; }

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 10 > "$GPU_LOG" 2>/dev/null &
SAMPLER=$!
run_leg ics-4096 \
  pixi run -e gpu python scripts/v2_m6_realization.py ics --config c-hero --workdir "$HERO" \
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
