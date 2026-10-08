#!/bin/bash
#SBATCH -J d8-hero-ics
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -o d8-hero-ics-%j.log
# A preset's ICs (LCDM second-order growth) by the device generator on four GB200s, into a
# new directory. Legs:
#   1. preflight: devices, scratch space, planner verdicts (a refusal stops the job);
#   2. smoke at 32^3: device generator vs host generator, 4 cards == 1 card (stops the job);
#   3. the generation -> IC_DIR (refuses if a manifest is already there);
#   4. manifest check: growth2 = lcdm, the preset's n_part^3 particles in its brick slabs;
#   5. the GPU pytest set for the generator.
#
# Requires: INEXOR_RUNS (run root), IC_DIR (IC output directory; HERO_IC_DIR is read when it
# is unset), and INEXOR_SRC or submission from the inexor checkout. Optional: PRESET
# (c-hero), NEED_GB (1500, the free scratch the preflight requires), PK_TABLE (a tabulated
# linear P(k) from scripts/run/camb_linear_pk.py instead of EH98). The account is passed
# with `sbatch -A`:
#   INEXOR_RUNS=$SCRATCH/inexor_runs IC_DIR=$SCRATCH/inexor_runs/c-hero-lcdm \
#     sbatch -A <account> scripts/run/hero_ics_vista.sbatch
#
# Wall: the c-hero generation takes ~67 min, the smoke ~1 min, the pytest ~3; 3 h leaves
# margin. c-gh has 1/8 the particles.

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
HERO=${IC_DIR:-${HERO_IC_DIR:?set IC_DIR to the directory the ICs are written to}}
PRESET=${PRESET:-c-hero}
NEED_GB=${NEED_GB:-1500}
CARDS=runs/v2
mkdir -p "$CARDS"
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
  pixi run -e gpu python scripts/run/device_ics.py preflight --root "$RUNS" --need-gb "$NEED_GB" \
  || { echo "FATAL: preflight refused"; exit 1; }
run_leg smoke \
  pixi run -e gpu python scripts/run/device_ics.py smoke --out $CARDS/${TAG}_smoke.json \
  || { echo "FATAL: the smoke missed, not spending the node on the generation"; exit 1; }

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 10 > "$GPU_LOG" 2>/dev/null &
SAMPLER=$!
run_leg "ics-$PRESET" \
  pixi run -e gpu python scripts/run/realization.py ics --config "$PRESET" --workdir "$HERO" \
    --generator device ${PK_TABLE:+--pk-table "$PK_TABLE"}
kill $SAMPLER 2>/dev/null
[ -f "$HERO/realization_ics.json" ] && cp "$HERO/realization_ics.json" "$CARDS/${TAG}_$PRESET.json"

# the expected counts come from the preset, not from the manifest under test
run_leg manifest-check \
  env JAX_PLATFORMS=cpu pixi run -e gpu python -c "
import json, sys
from inexor.plan import PRESETS, engine_config
ec = engine_config('$PRESET')
want_n, want_slabs = PRESETS['$PRESET']['n_part'] ** 3, ec.n_fine // ec.n_brick
m = json.load(open('$HERO/manifest.json'))
print('growth2', m.get('growth2'), 'n_particles', m['n_particles'], 'of', want_n,
      'slabs', len(m['files']), 'of', want_slabs, 'ic_stream', m.get('ic_stream'))
sys.exit(0 if (m.get('growth2') == 'lcdm' and m['n_particles'] == want_n
               and len(m['files']) == want_slabs) else 1)
"

run_leg pytest-gpu \
  pixi run -e gpu python -m pytest -q tests/test_icgen_device.py tests/test_device_emit.py \
    tests/test_icgen.py

du -sh "$HERO" 2>/dev/null
echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
