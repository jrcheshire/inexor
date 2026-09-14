#!/bin/bash
#SBATCH -J d6-device-ics
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d6-device-ics-%j.log
# D6 on a gb node, ONE bundled job: the IC stage on the cards, timed at 2048^3,
# then the 4096^3 ICs.
#   1. preflight: devices, scratch space + quota (~1.5 TB), planner verdicts (stops the job);
#   2. smoke at 32^3 on the GPUs: device vs host generator, 4 cards == 1 card (stops the job);
#   3. 2048^3 device generation, per-stage wall + host peak; nvidia-smi sampled throughout;
#   4. projection to 4096^3 (stops the job: wall > 9 h or host peak > 0.9 x 1026 GB);
#   5. the 2048^3 output removed by name;
#   6. the 4096^3 device generation -> /scratch/10303/jrc4/inexor_runs/c-hero-r0 (UNPRICED);
#   7. the GPU pytest set for the new files (LAST: unpriced).
# Submit from the Vista checkout, after JC approves the time ask:
#   sbatch scripts/v2_d6_device_ics_vista.sbatch
#
# TIME ASK (DRAFT, for JC): ~3.5 h estimated, 10 h wall, one gb node. Every generation
# term is ARITHMETIC; nothing about the device generator has run on a GB200.
#
#   phase                   each (source)                                          total
#   preflight + smoke       996685's GPU smoke ~4.5 min (compiles); 32^3 x 6 gens     ~8 min
#   2048^3 transforms       15 passes x ~9 s (D5: 2048^3 inverse at W=4, 28.2 s/3)    ~3 min
#   2048^3 source/noise     on the cards, one plane at a time; UNPRICED              ~3 min
#   2048^3 V staging        103 GB at 0.96 GB/s write + 1.76 GB/s read (C11)         ~3 min
#   2048^3 emission         host numpy, 8.6e9 rows; UNPRICED, 30-60 ns/row guessed  ~5-9 min
#   projection + cleanup                                                             ~1 min
#   4096^3 generation       2048^3 x 8.73 (n^3 log n); the projection leg reprices  ~2.5 h
#   pytest, 2 files, GPU    UNPRICED; 996857's 6 files took 10:18                    ~10 min
#
# PRE-REGISTERED:
#   - preflight or smoke failure stops the job before any generation;
#   - smoke gates: 4 cards == 1 card bitwise; occupancy exact, position and velocity
#     codes within 1 of the host generator (JC-proposed bars); scales reported, not gated;
#   - projected 4096^3 wall > 9 h or host peak > 923 GB stops the job before leg 6;
#   - walls, peaks and per-stage splits are reported, not gated.
#
# ENV copied verbatim from v2_e2e4_device_cards_vista.sbatch.

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

RUNS=/scratch/10303/jrc4/inexor_runs
PROBE=$RUNS/d6-c-gh-probe
HERO=$RUNS/c-hero-r0
CARDS=runs/v2
GPU_LOG=$CARDS/d6-gpu-mem-$SLURM_JOB_ID.csv

echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
free -g | head -2

rc_total=0
run_leg () {
  local name="$1"; shift
  echo ""; echo "=== LEG $name ($(date +%H:%M:%S)) ==="
  "$@"; local rc=$?
  echo "--- LEG $name rc=$rc ($(date +%H:%M:%S)) ---"
  [ $rc -ne 0 ] && rc_total=$((rc_total + 1))
  return $rc
}

run_leg preflight \
  pixi run -e gpu python scripts/v2_d6_device_ics.py preflight --root "$RUNS" --need-gb 1500 \
  || { echo "FATAL: preflight refused"; exit 1; }
run_leg smoke \
  pixi run -e gpu python scripts/v2_d6_device_ics.py smoke --out $CARDS/d6_smoke_gb.json \
  || { echo "FATAL: the smoke missed, not spending the node on the generations"; exit 1; }

# card memory, sampled for the whole generation (inside this job; dies with it)
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 10 > "$GPU_LOG" 2>/dev/null &
SAMPLER=$!

run_leg ics-2048 \
  pixi run -e gpu python scripts/v2_m6_realization.py ics --config c-gh --workdir "$PROBE" \
    --generator device \
  || { kill $SAMPLER; echo "FATAL: the 2048^3 generation failed"; exit 1; }
run_leg project \
  pixi run -e gpu python scripts/v2_d6_device_ics.py project \
    --card "$PROBE/realization_ics.json" --gpu-log "$GPU_LOG" --out $CARDS/d6_projection_gb.json \
  || { kill $SAMPLER; cp "$PROBE/realization_ics.json" $CARDS/d6_ics_c-gh_gb.json;
       echo "STOP: the projection is past the pre-registered stop"; exit 1; }
cp "$PROBE/realization_ics.json" $CARDS/d6_ics_c-gh_gb.json
run_leg cleanup-2048 \
  pixi run -e gpu python scripts/v2_d6_device_ics.py cleanup --workdir "$PROBE"

run_leg ics-4096 \
  pixi run -e gpu python scripts/v2_m6_realization.py ics --config c-hero --workdir "$HERO" \
    --generator device
[ -f "$HERO/realization_ics.json" ] && cp "$HERO/realization_ics.json" $CARDS/d6_ics_c-hero_gb.json
kill $SAMPLER 2>/dev/null

run_leg pytest-gpu \
  pixi run -e gpu python -m pytest -q tests/test_ooc_fft_device_kernels.py tests/test_icgen_device.py

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
