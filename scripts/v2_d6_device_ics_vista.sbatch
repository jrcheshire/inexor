#!/bin/bash
#SBATCH -J d6-device-ics
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d6-device-ics-%j.log
# D6 on a gb node, ONE bundled job: the IC stage on the cards, timed at 2048^3,
# then the 4096^3 ICs.
#   1. preflight: devices, scratch space + quota (~1.5 TB), planner verdicts (stops the job);
#   2. smoke at 32^3 on the GPUs: device generator (card emission) vs host generator,
#      4 cards == 1 card (stops the job);
#   2b. emit-shape: ONE 4096^3 emission slab on one card -- card peak (memory_stats) and
#      per-slab seconds, projected to four cards (stops the job: card peak > 0.9 x 199 GB);
#   3. 2048^3 device generation, per-stage wall + host peak; nvidia-smi sampled throughout;
#   4. projection to 4096^3 (stops the job: wall > 6 h or host peak > 0.9 x 1026 GB);
#   5. the 2048^3 output removed by name;
#   5b. the allocator A/B: the same 2048^3 generation again under BFC (the leg above is
#      the cuda_async control), both arms in ONE job on ONE node, then the per-stage
#      ratio. Its own receipt gates it and a refusal skips the arm, not the job;
#   6. the 4096^3 device generation -> /scratch/10303/jrc4/inexor_runs/c-hero-r0 (UNPRICED);
#   7. the GPU pytest set for the new files (LAST: unpriced).
# Submit from the Vista checkout, after JC approves the time ask:
#   sbatch scripts/v2_d6_device_ics_vista.sbatch
#
# RERUN after 997814 (OOM at the 4096^3 u_x shard: BFC capped at the default 0.75 of
# the card, card 2 fragmented). Changes: XLA_CLIENT_MEM_FRACTION=0.95 and the
# cuda_async allocator; code unchanged. Approved by JC at the same 8 h wall.
# Estimate ~1.3 h from 997814 + 997280 (legs 1-5 ~9 min, 4096^3 before emission ~52 min,
# 4096^3 emission ~14 min projected), ~1.4 SU; ~8.6 SU at the wall.
#
# ORIGINAL TIME ASK (997814): ~2 h estimated, 8 h wall, one gb node (~1.08 SU per
# node-hour from 996685/996857: ~2.2 SU at the estimate, ~8.6 at the wall). Everything
# before emission is MEASURED (997280); card emission has not run on a GB200.
#
#   phase                   each (source)                                          total
#   preflight + smoke       997280: 33 s (32^3 x 6 generations on the cards)          ~2 min
#   emit-shape, 4096^3 slab 16 GB of host fields + 3 sources + 1 destination; UNPRICED ~10 min
#   2048^3 before emission  997280 MEASURED: 7.8 + 41 + 7 + 196 + 48 s                 ~5 min
#   2048^3 emission         cards; UNPRICED (host was 2115 s); emit-shape reprices    ~5 min
#   projection + cleanup                                                             ~1 min
#   4096^3 before emission  997280 MEASURED: 58 + 370 + 93 + 1979 + 549 s             ~51 min
#   4096^3 emission         cards; UNPRICED (host ~6.6 h); the projection reprices   ~30 min
#   pytest, 5 files, GPU    997280's 2 files took 35 s                               ~3 min
#
# PRE-REGISTERED:
#   - preflight or smoke failure stops the job before any generation;
#   - smoke gates: 4 cards == 1 card bitwise; occupancy exact, position and velocity
#     codes within 1 of the host generator (JC-proposed bars); scales reported, not gated;
#   - projected 4096^3 wall > 9 h or host peak > 923 GB stops the job before leg 6;
#   - walls, peaks and per-stage splits are reported, not gated;
#   - allocator slowdown, reported not gated: smoke leg, emit-shape split_s, ics-2048
#     stage_s and ics-4096 delta/source/source_forward/velocities vs 997814 (BFC).
#   - the allocator A/B (leg 5b) is REPORTED, NOT GATED and decides nothing on its
#     own: it measures a wall at 2048^3, where per-card peak is ~18 GB and BFC
#     fragmentation cannot be exercised. 998798 read 1.15x across jobs; this reads
#     it within one. The arm identifies itself from manifest provenance, not from
#     its label, and the comparison REFUSES (rc 3) on two same-allocator arms or on
#     any other axis moving -- a vacuous A/B must not read as a null.
#
# ENV copied verbatim from v2_e2e4_device_cards_vista.sbatch.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$SCRATCH/src/inexor" 2>/dev/null || cd "$HOME/src/inexor" || {
  echo "FATAL: no inexor checkout"; exit 1; }
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# The allocator caps its growth at XLA_CLIENT_MEM_FRACTION of the card even with
# preallocation off, and the compiled default is 0.75 (138.8 GiB of a GB200):
# 997814 OOMed on a 66 GiB u_x shard at that cap. Preallocation stays off.
export XLA_CLIENT_MEM_FRACTION=0.95
# CUDA's own pool rather than BFC: freed blocks can coalesce, where BFC never merges
# across its growth regions (997814's card 2 held two 64 GiB regions and could not
# serve 66 GiB).
# 998798 measured the cost of that choice at 1.15x on the 2048^3 leg, but against
# 997814 -- a different job on a different allocation, which is the comparison shape
# that cost record 5h a chase. ALLOC is per leg so both arms can run in ONE job;
# the default stays cuda_async (no-worse-defaults: it is what wrote c-hero-r0).
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=900
unset JAX_PLATFORMS
unset CUDA_VISIBLE_DEVICES

RUNS=/scratch/10303/jrc4/inexor_runs
PROBE=$RUNS/d6-c-gh-probe
PROBE_BFC=$RUNS/d6-c-gh-probe-bfc
HERO=$RUNS/c-hero-r0
CARDS=runs/v2
GPU_LOG=$CARDS/d6-gpu-mem-$SLURM_JOB_ID.csv

echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
free -g | head -2
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

# An allocator is chosen when jax initialises, so an arm needs its own PROCESS with
# its own environment -- setting it inside the child is too late. `bfc` is spelled by
# UNSETTING the variable, which is what the compiled default does, so passing the
# literal string would be a different (and unsupported) request.
run_leg_alloc () {
  local alloc="$1"; local name="$2"; shift 2
  if [ "$alloc" = "bfc" ]; then
    run_leg "$name" env -u XLA_PYTHON_CLIENT_ALLOCATOR "$@"
  else
    run_leg "$name" env XLA_PYTHON_CLIENT_ALLOCATOR="$alloc" "$@"
  fi
}

run_leg preflight \
  pixi run -e gpu python scripts/v2_d6_device_ics.py preflight --root "$RUNS" --need-gb 1500 \
  || { echo "FATAL: preflight refused"; exit 1; }
run_leg smoke \
  pixi run -e gpu python scripts/v2_d6_device_ics.py smoke --out $CARDS/d6_smoke_gb.json \
  || { echo "FATAL: the smoke missed, not spending the node on the generations"; exit 1; }
run_leg emit-shape \
  pixi run -e gpu python scripts/v2_d6_device_ics.py emit-shape --n 4096 --nb 256 \
    --out $CARDS/d6b_emit_shape_gb.json --tmp "$RUNS" \
  || { echo "STOP: one 4096^3 emission slab projects past the card budget"; exit 1; }

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

# --- the allocator A/B, both arms in THIS job on THIS node -------------------
# 998798 bought its capacity with cuda_async and paid 1.15x on the 2048^3 leg for
# it, measured against 997814 across two jobs. The 4096^3 peak has since been read
# at 100-105 GiB per card -- under the 138.8 GiB the DEFAULT fraction allows -- so
# BFC at 0.95 has ~75 GiB of headroom over the working set and the wall may be
# recoverable. The arm above is the cuda_async control; this is the BFC one.
# What this leg does NOT settle: BFC ratchets toward whatever ceiling it is given,
# which is how 997814 died, and 2048^3 peaks ~18 GB/card -- far too low to
# exercise fragmentation. This measures the WALL only. The safety question is a
# 4096^3 generation and nothing smaller answers it.
run_leg_alloc bfc preflight-bfc \
  pixi run -e gpu python scripts/v2_d6_device_ics.py preflight --root "$RUNS" --need-gb 100
AB_OK=$?
if [ $AB_OK -ne 0 ]; then
  echo "SKIP allocator A/B: the BFC arm's own receipt refused (rc=$AB_OK); the arm"
  echo "cannot prove which allocator ran, so its seconds are not attributable."
else
  run_leg_alloc bfc ics-2048-bfc \
    pixi run -e gpu python scripts/v2_m6_realization.py ics --config c-gh \
      --workdir "$PROBE_BFC" --generator device
  if [ -f "$PROBE_BFC/realization_ics.json" ]; then
    cp "$PROBE_BFC/realization_ics.json" $CARDS/d6_ics_c-gh_bfc_gb.json
    run_leg allocator-ab \
      pixi run -e gpu python scripts/v2_d6_device_ics.py allocator-ab \
        --a $CARDS/d6_ics_c-gh_gb.json --b $CARDS/d6_ics_c-gh_bfc_gb.json \
        --out $CARDS/d6_allocator_ab_gb.json
    run_leg cleanup-2048-bfc \
      pixi run -e gpu python scripts/v2_d6_device_ics.py cleanup --workdir "$PROBE_BFC"
  fi
fi

run_leg ics-4096 \
  pixi run -e gpu python scripts/v2_m6_realization.py ics --config c-hero --workdir "$HERO" \
    --generator device
[ -f "$HERO/realization_ics.json" ] && cp "$HERO/realization_ics.json" $CARDS/d6_ics_c-hero_gb.json
kill $SAMPLER 2>/dev/null

run_leg pytest-gpu \
  pixi run -e gpu python -m pytest -q tests/test_ooc_fft_device_kernels.py tests/test_icgen_device.py \
    tests/test_device_emit.py tests/test_icgen.py tests/test_plan.py

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
