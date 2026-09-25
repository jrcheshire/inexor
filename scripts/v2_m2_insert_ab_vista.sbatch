#!/bin/bash
#SBATCH -J m2-insert-ab
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o m2-insert-ab-%j.log
# M2 on a gb node: the rewritten compiled insert against the frozen pre-cba30c3
# one, ON A GB200, at production slab shape.
#
# cba30c3 narrowed the sort key to slab-relative uint32 with an int32 index and
# replaced a second `argsort` -- an eight-pass int64 radix sort spent separating
# THREE values -- with an exclusive-prefix partition. 1ecae62 narrowed the index
# family that the partition had widened. Both are permutation-identical by
# construction and both were checked standalone (132 partition cases, 300
# randomised key cases) and gated against numpy on CPU. None of that is a GPU
# reading, and the one fault this kernel has actually suffered (record sec. 26:
# the jitted int16 row magnitude wrong in 5,293 of 15,898 rows) was GB200-only.
# Submit from the Vista checkout:
#   sbatch scripts/v2_m2_insert_ab_vista.sbatch
#
# TIME ASK: ~40 min estimated, 1 h 30 wall, one gb node (~1.08 SU/node-hour:
# ~0.7 SU at the estimate, ~1.6 SU at the wall).
#
#   phase                          each (source)                               total
#   jax devices + GPU pytest,      995813's 5-file set took 5:24; this is 7
#   7 files                        files and the class has been under-priced
#                                  twice (E1 asked 7 min, took 17:40)          ~12 min
#   probe smoke, small shapes      ~10 s start each (995638)                   ~1 min
#   xback, cdev, cross-backend     995638 ran xback+device-real in 4:04 total   ~2 min
#   device-real, cgh64 split       995638 MEASURED 4:04 for this pair           ~4 min
#   insert-parity, --shape-nb 256  995435 MEASURED 3:56 including its
#                                  production-shape arms; 2 compiles + 2 calls  ~5 min
#   peak A/B, legacy then current, 995435 same source; separate PROCESSES so
#   --shape-nb 256                 each device high-water belongs to one arm    ~8 min
#
# QUESTIONS, in the order the job answers them:
#   1. Do the insert/eject/migrate/repack gates pass on the GPU backend? (gate)
#   2. Is the rewritten kernel bitwise the numpy reference on a GB200 at cdev
#      (xback) and cgh64 (device-real)? (gate)
#   3. Is it bitwise the FROZEN pre-cba30c3 kernel at production slab shape --
#      268,435,456 slab rows -- field by field? (gate: rc 3 on disagreement)
#   4. What did it cost, and what did it save? Walls and device peak per padded
#      row for both kernels, same node, same job, back to back.
#   5. Does the sort lower to a CUB radix path? The whole rung rests on the key's
#      WIDTH being what mattered, and a CPU-only jaxlib cannot answer it. The
#      parity arm dumps optimized HLO and reports which tokens it finds.
#
# PRE-REGISTERED:
#   - A pytest failure, a non-bitwise step, or an insert-parity disagreement is a
#     failed gate, read BEFORE any wall. The smoke gates the long legs.
#   - Walls and peaks are REPORTED, NOT GATED. This job measures; whether the
#     change is worth keeping is a separate call with the numbers in hand.
#   - The HLO receipt is reported and CANNOT fail the leg: if the token is absent
#     the change is still permutation-identical, it only means the win has a
#     different explanation than the one on record.
#   - Expectation, falsifiable, written before the run: current should be FASTER
#     than legacy on the insert wall, and LOWER on device peak per padded row.
#     CPU XLA's buffer assignment put temp at 63.0 B/padded row against the
#     legacy 71.0 (0.887x) at p3=512 nb2=1024 n_pad=2^20 -- a different backend
#     and a different quantity from the card's peak, so it sets the DIRECTION
#     expected and not the number. `INSERT_B_PER_PADDED_ROW` charges 83.0.
#   - NOT in this job: the per-op split inside the kernel (which of the sort, the
#     rescale or the occupancy scatter dominates). It needs ablated kernels,
#     which are instruments that by construction return wrong answers, and one is
#     not going into the same job as the gate untested.
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

run_leg devices \
  pixi run -e gpu python -c "import jax; print('jax devices:', jax.devices())"
run_leg pytest-gpu \
  pixi run -e gpu python -m pytest -q tests/test_insert_jax.py tests/test_eject_jax.py \
    tests/test_migrate_device.py tests/test_migrate_device_cards.py \
    tests/test_slot_state.py tests/test_repack_device.py \
    tests/test_engine_device_backend.py
run_leg smoke \
  pixi run -e gpu python scripts/v2_d3_device_migrate.py \
    --arms xback,insert-parity --xback-n 32 --shape-nb 8 --steps 2 --reps 1 \
    --out-suffix _m2ab_gbsmoke
if [ $rc_total -ne 0 ]; then
  echo "FATAL: a GPU gate or the smoke failed, not spending the node on the long legs"
  exit 1
fi

# the identity legs first: no wall is read as this kernel's until they hold
run_leg identity \
  pixi run -e gpu python scripts/v2_d3_device_migrate.py --arms xback,device-real \
    --xback-n 256 --real-n 512 --steps 2 --out-suffix _m2ab_gb

# production slab shape: 268,435,456 rows per slab at nb=256
run_leg insert-parity \
  pixi run -e gpu python scripts/v2_d3_device_migrate.py --arms insert-parity \
    --shape-nb 256 --out-suffix _m2ab_parity_gb

# the A/B. Separate processes per impl, legacy FIRST so the live kernel is not
# the arm that pays for a cold compilation cache.
run_leg peak-ab \
  pixi run -e gpu python scripts/v2_d3_device_migrate.py --arms "" --shape-nb 256 \
    --peaks "insert:0.05:1:legacy+insert:0.05:1:current" --out-suffix _m2ab_peak_gb

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
