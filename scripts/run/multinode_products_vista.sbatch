#!/bin/bash
#SBATCH -J inexor-mn-products
#SBATCH -p gh
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:00:00
#SBATCH -o inexor-mn-products-%j.log
#SBATCH --signal=B:USR1@600
# The P(k) card and the (x, v) export of a realization's checkpoint, across the nodes, one
# GH200 and one rank per node (`device_run.py card` / `export`): the card painted and
# transformed on the cards, the export decoded on them and written as one part per rank.
#
# Guards first (job-script tests, the MPI launch line on every node, a dying rank ends the
# launch; all fatal). With SMALL_CKPT, a byte check before the products: that checkpoint's
# card and export on one rank (node 0) and on every rank must be the same (the card in every
# field, the export's crc32 of every array); a mismatch stops the job. Then the card and the
# export of CKPT_DIR into PROD_DIR. With REF_EXPORT (an export of the same checkpoint made
# another way, e.g. hero_export_vista.sbatch's host decode) the export's crc32 must equal it;
# with REF_CARD (a card of the same checkpoint, e.g. hero_card_vista.sbatch's) the per-bin
# difference is printed, never gated (the transforms differ).
#
# Every leg has a time cap and a silence limit; a failed leg stops the job. Each leg's
# output is also in PROD_DIR/job-<id>/legs/.
#
# Requires INEXOR_RUNS, CKPT_DIR (the run's checkpoint dir), PROD_DIR (not under CKPT_DIR;
# the card lands in PROD_DIR/pk.json, the export in PROD_DIR/export, which must not hold
# files), and INEXOR_SRC or submission from the checkout. Optional: PRESET (c-gh); K and
# EXPECT_STEP (the newest checkpoint's n_steps and step); ALLOW_PARTIAL (0: refuse an
# export short of K); SMALL_CKPT + SMALL_PRESET (c-1024); REF_EXPORT; REF_CARD; SLACK /
# ALLOC_MARGIN (0.10), ARENA_FRAC (0.01); CARD_CAP_S (2400), EXPORT_CAP_S (3600), QUIET_S (900).
#   R=$SCRATCH/inexor_runs; INEXOR_RUNS=$R CKPT_DIR=$R/c-gh-k120/ckpt PROD_DIR=$R/c-gh-k120-mn \
#     SMALL_CKPT=$R/mn-gate-c-1024-1043437/mpi/ckpt REF_EXPORT=$R/c-gh-k120-product/export \
#     REF_CARD=<the c-gh card json> sbatch -A <account> scripts/run/multinode_products_vista.sbatch
#
# REHEARSAL=1 runs the same legs on a laptop CPU at cdev8-tile32, from a rehearsal of the
# gate script (the small check against the same checkpoint):
#   G=runs/rehearsal/mn-gate-cdev8-tile32-<tag>; REHEARSAL=1 INEXOR_RUNS=runs/rehearsal \
#     CKPT_DIR=$G/mpi/ckpt SMALL_CKPT=$G/mpi/ckpt PROD_DIR=runs/rehearsal/products-r1 \
#     scripts/run/mpi_lane.sh --run bash scripts/run/multinode_products_vista.sbatch
# PLANT_CARD_MISMATCH=1 / PLANT_EXPORT_MISMATCH=1 run the small check's every-rank card with
# bins pinned by --k-max / export with D-time velocities, so its gate must stop the job;
# PLANT_LEG_FAIL=1 gives the first rank leg an unknown XLA flag, which aborts every rank.
#
# Wall (c-gh on 2 gh): per node the load is ~4.3e9 particles (~55 GB), the card paints and
# transforms a 1024^3 mesh, the export writes ~100 GB; with the c-1024 check and ~15 min of
# guards, ~1 h central, 2 h requested.

export PATH="$HOME/.pixi/bin:$PATH"
cd "${INEXOR_SRC:-${SLURM_SUBMIT_DIR:-$PWD}}" && [ -f src/inexor/engine.py ] || {
  echo "FATAL: not an inexor checkout (set INEXOR_SRC)"; exit 1; }
SRC=$(pwd)
RUNS=${INEXOR_RUNS:?set INEXOR_RUNS to the run root, e.g. \$SCRATCH/inexor_runs}
mkdir -p "$RUNS" && RUNS=$(cd "$RUNS" && pwd)
export PYTHONUNBUFFERED=1

abspath () { (cd "$1" 2>/dev/null && pwd); }
[ -n "${CKPT_DIR:-}" ] && [ -d "$CKPT_DIR" ] || { echo "FATAL: set CKPT_DIR to an existing directory"; exit 1; }
CKPT_DIR=$(abspath "$CKPT_DIR")
for v in SMALL_CKPT REF_EXPORT; do
  if [ -n "${!v:-}" ]; then
    [ -d "${!v}" ] || { echo "FATAL: $v is not a directory: ${!v}"; exit 1; }
    printf -v "$v" %s "$(abspath "${!v}")"
  fi
done
[ -z "${REF_EXPORT:-}" ] || [ -f "$REF_EXPORT/export.json" ] || {
  echo "FATAL: REF_EXPORT holds no export.json (an incomplete export)"; exit 1; }
[ -z "${REF_CARD:-}" ] || [ -f "$REF_CARD" ] || { echo "FATAL: REF_CARD is not a file"; exit 1; }
[ -n "${PROD_DIR:-}" ] || { echo "FATAL: set PROD_DIR (the products' dir)"; exit 1; }
mkdir -p "$PROD_DIR" && PROD_DIR=$(abspath "$PROD_DIR")
case "$PROD_DIR/" in "$CKPT_DIR"/*) echo "FATAL: PROD_DIR is under CKPT_DIR"; exit 1 ;; esac
EXPORT=$PROD_DIR/export
if [ -d "$EXPORT" ] && [ -n "$(ls -A "$EXPORT")" ]; then
  echo "FATAL: $EXPORT already holds files"; exit 1
fi

if [ "${REHEARSAL:-0}" = 1 ]; then
  PRESET=${PRESET:-cdev8-tile32} SMALL_PRESET=${SMALL_PRESET:-cdev8-tile32}
  SLACK=${SLACK:-0.20} ARENA=${ARENA_FRAC:-0.20} ALLOW_PARTIAL=${ALLOW_PARTIAL:-1}
fi
PRESET=${PRESET:-c-gh} SMALL_PRESET=${SMALL_PRESET:-c-1024} ALLOW_PARTIAL=${ALLOW_PARTIAL:-0}
SLACK=${SLACK:-0.10} ALLOC_MARGIN=${ALLOC_MARGIN:-0.10} ARENA=${ARENA:-${ARENA_FRAC:-0.01}}
CARD_CAP_S=${CARD_CAP_S:-2400} EXPORT_CAP_S=${EXPORT_CAP_S:-3600} QUIET_S=${QUIET_S:-900}
N_RANKS=${SLURM_NTASKS:-${N_RANKS:-2}}

. "$SRC/scripts/run/mpi_env_vista.sh"
. "$SRC/scripts/run/job_lib.sh"

# the newest generation's step and schedule, unless given
read -r _gen STEP < <(ckpt_newest "$CKPT_DIR") || { echo "FATAL: no checkpoint in $CKPT_DIR"; exit 1; }
EXPECT_STEP=${EXPECT_STEP:-$STEP}
K=${K:-$(manifest_field "$CKPT_DIR/$_gen" "m['provenance']['n_steps']")}
if [ -n "${SMALL_CKPT:-}" ]; then
  read -r _sgen SMALL_STEP < <(ckpt_newest "$SMALL_CKPT") || {
    echo "FATAL: no checkpoint in $SMALL_CKPT"; exit 1; }
  SMALL_K=$(manifest_field "$SMALL_CKPT/$_sgen" "m['provenance']['n_steps']")
fi

WORK=$PROD_DIR/job-$SLURM_JOB_ID
# shellcheck disable=SC2034  # read by job_lib.sh
LEG_DIR=$WORK/legs
CARDS=$SRC/runs/v2
TAG=mnprod_${PRESET}_$SLURM_JOB_ID
LAYOUT=(--slack "$SLACK" --alloc-margin "$ALLOC_MARGIN" --arena-frac "$ARENA")
PARTIAL=()
[ "$ALLOW_PARTIAL" = 1 ] && PARTIAL=(--allow-partial)
mkdir -p "$WORK" "$CARDS"

echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
echo "=== $PRESET: card + export of the step-$EXPECT_STEP checkpoint (K=$K) in $CKPT_DIR ==="
echo "=== -> $PROD_DIR; small check ${SMALL_CKPT:-none}${SMALL_CKPT:+ ($SMALL_PRESET, step $SMALL_STEP of $SMALL_K)} ==="
echo "=== references: export ${REF_EXPORT:-none}, card ${REF_CARD:-none}; layout ${LAYOUT[*]} ==="
echo "=== nodes ${SLURM_JOB_NODELIST:-local}, launch '$LAUNCH', work $WORK ==="
for t in mpiexec mpicc "$PY"; do echo "  $t -> $(command -v "$t" || echo none)"; done
if [ "${REHEARSAL:-0}" != 1 ]; then
  echo "  modules $(module -t list 2>&1 | tr '\n' ' ')"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null
  free -g | head -2
fi
df -h "$PROD_DIR" | tail -1
env | grep -E '^(XLA_|JAX_|INEXOR_|MV2_|MVP_|HYDRA_)' | sort

rc_total=0
finish () {
  echo ""; du -sh "$PROD_DIR"/* 2>/dev/null; df -h "$PROD_DIR" | tail -1
  echo "=== sacct ==="
  sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,ExitCode -P 2>/dev/null
  echo "=== end $(date) rc_total=$rc_total ==="
}
trap finish EXIT
trap forward_usr1 USR1

stop_job () {
  echo "=== stopping: $1; every output is kept under $PROD_DIR ==="
  exit $((rc_total > 0 ? rc_total : 1))
}

# prod_ranks KIND ARM PRESET CKPT K STEP XLA_EXTRA ARGS...: one product across the nodes
prod_ranks () {
  local kind=$1 arm=$2 preset=$3 ckpt=$4 k=$5 step=$6 extra=$7; shift 7
  on_ranks "$CARDS/${TAG}_${arm}_s${step}_r@RANK@" "$extra" "${MPIPATH[@]}" \
    "$PY" -m mpi4py "$DRIVER" "$kind" --preset "$preset" --checkpoint-dir "$ckpt" \
    --k-steps "$k" --expect-step "$step" --cards 1 --beat 60 "${MEMBIND_CHECK[@]}" \
    "${LAYOUT[@]}" --card "$CARDS/${TAG}_${arm}_s${step}.json" --comm mpi \
    --comm-timeout 1800 "$@"
}

# prod_one KIND ARM PRESET CKPT K STEP ARGS...: the same product on node 0 alone
prod_one () {
  local kind=$1 arm=$2 preset=$3 ckpt=$4 k=$5 step=$6; shift 6
  "${BIND0[@]}" "${ON0[@]}" "$PY" "$DRIVER" "$kind" --preset "$preset" --checkpoint-dir "$ckpt" \
    --k-steps "$k" --expect-step "$step" --cards 1 --beat 60 "${MEMBIND_CHECK[@]}" \
    "${LAYOUT[@]}" --card "$CARDS/${TAG}_${arm}_s${step}.json" "$@"
}

# --- guards (node 0)
run_leg --cap 1800 build-mpi4py build_mpi4py || stop_job "mpi4py did not build"
# its mpiexec cases are left to the launch guards below, on the launcher the legs use
run_leg --cap 600 guard-job-scripts \
  "${CPU[@]}" "$PY" -m pytest -q -rs -p no:cacheprovider tests/test_job_scripts.py \
    -k "not mpiexec" \
  || stop_job "the job-script tests failed on this machine"
run_leg --cap 300 guard-launch-mpi \
  on_ranks "$CARDS/${TAG}_guard-mpi_r@RANK@" "" "${MPIPATH[@]}" \
    "$PY" -m mpi4py -c "from mpi4py import MPI; import jax; c = MPI.COMM_WORLD
assert c.Get_size() == $N_RANKS, c.Get_size()
print('rank', c.Get_rank(), 'of', c.Get_size(), jax.devices())" \
  || stop_job "the MPI launch line does not start every rank"
run_leg --cap 300 --expect-fail guard-dead-rank \
  on_ranks "$CARDS/${TAG}_guard-dead_r@RANK@" "" "$PY" -c "import os; os.abort()" \
  || stop_job "a rank that dies does not end the launch"

PLANT=""
[ "${REHEARSAL:-0}" = 1 ] && [ "${PLANT_LEG_FAIL:-0}" = 1 ] && PLANT=--xla_planted_unknown_flag

# --- the small check: one rank vs every rank, card and export
if [ -n "${SMALL_CKPT:-}" ]; then
  S=$WORK/small
  mkdir -p "$S"
  PLANT_CARD=() PLANT_EXPORT=()
  if [ "${REHEARSAL:-0}" = 1 ]; then
    [ "${PLANT_CARD_MISMATCH:-0}" = 1 ] && PLANT_CARD=(--k-max 1.0)
    [ "${PLANT_EXPORT_MISMATCH:-0}" = 1 ] && PLANT_EXPORT=(--d-time)
  fi
  run_leg --cap "$CARD_CAP_S" --quiet-limit "$QUIET_S" small-card-1 \
    prod_one card small-card-1 "$SMALL_PRESET" "$SMALL_CKPT" "$SMALL_K" "$SMALL_STEP" \
    --out "$S/pk-1.json" || stop_job "the small one-rank card failed"
  run_leg --cap "$CARD_CAP_S" --quiet-limit "$QUIET_S" "small-card-$N_RANKS" \
    prod_ranks card small-card "$SMALL_PRESET" "$SMALL_CKPT" "$SMALL_K" "$SMALL_STEP" \
    "$PLANT" --out "$S/pk-$N_RANKS.json" "${PLANT_CARD[@]}"
  r=$?; summarize small-card "$SMALL_STEP"
  [ $r -eq 0 ] || stop_job "the small every-rank card failed"
  gate_with small-card same_card "$S/pk-1.json" "$S/pk-$N_RANKS.json" \
    || { rc_total=$((rc_total + 1)); stop_job "the small card differs across rank counts"; }
  run_leg --cap "$EXPORT_CAP_S" --quiet-limit "$QUIET_S" small-export-1 \
    prod_one export small-export-1 "$SMALL_PRESET" "$SMALL_CKPT" "$SMALL_K" "$SMALL_STEP" \
    --export-dir "$S/export-1" --allow-partial || stop_job "the small one-rank export failed"
  run_leg --cap "$EXPORT_CAP_S" --quiet-limit "$QUIET_S" "small-export-$N_RANKS" \
    prod_ranks export small-export "$SMALL_PRESET" "$SMALL_CKPT" "$SMALL_K" "$SMALL_STEP" \
    "" --export-dir "$S/export-$N_RANKS" --allow-partial "${PLANT_EXPORT[@]}"
  r=$?; summarize small-export "$SMALL_STEP"
  [ $r -eq 0 ] || stop_job "the small every-rank export failed"
  gate_with small-export same_export "$S/export-1" "$S/export-$N_RANKS" \
    || { rc_total=$((rc_total + 1)); stop_job "the small export differs across rank counts"; }
  PLANT=""
fi

# --- the products
run_leg --cap "$CARD_CAP_S" --quiet-limit "$QUIET_S" card \
  prod_ranks card card "$PRESET" "$CKPT_DIR" "$K" "$EXPECT_STEP" "$PLANT" --out "$PROD_DIR/pk.json"
r=$?; summarize card "$EXPECT_STEP"
[ $r -eq 0 ] || stop_job "the card failed"
if [ -n "${REF_CARD:-}" ]; then
  echo ""; echo "=== CARD vs $REF_CARD (reported, not gated) ==="
  card_diff "$REF_CARD" "$PROD_DIR/pk.json"
fi
run_leg --cap "$EXPORT_CAP_S" --quiet-limit "$QUIET_S" export \
  prod_ranks export export "$PRESET" "$CKPT_DIR" "$K" "$EXPECT_STEP" "" \
  --export-dir "$EXPORT" "${PARTIAL[@]}"
r=$?; summarize export "$EXPECT_STEP"
[ $r -eq 0 ] || stop_job "the export failed"
if [ -n "${REF_EXPORT:-}" ]; then
  gate_with ref-export same_export "$REF_EXPORT" "$EXPORT" || rc_total=$((rc_total + 1))
fi
exit $rc_total
