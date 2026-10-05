#!/bin/bash
#SBATCH -J inexor-mn-ics
#SBATCH -p gh
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --ntasks-per-node=1
#SBATCH -t 02:00:00
#SBATCH -o inexor-mn-ics-%j.log
#SBATCH --signal=B:USR1@600
# Device ICs across the nodes, one GH200 and one rank per node (`device_run.py ics --comm
# mpi`): every rank generates its own brick slabs into the shared IC_DIR, rank 0 writes the
# manifest.
#
# Guards first (job-script tests, the MPI launch line on every node, a dying rank ends the
# launch; all fatal). With SMALL_PRESET, a byte check before the production generation: that
# preset's ICs on one rank (node 0) and on every rank must be the same files (slab bytes,
# manifest minus provenance, timings and rank / card / y-block counts); a mismatch stops the
# job. Then PRESET's ICs into IC_DIR, and a check that the manifest holds the preset's
# particle count in its slab count.
#
# Every leg has a time cap and a silence limit; a failed leg stops the job. Each leg's
# output is also in IC_DIR/../<name>-job-<id>/legs/.
#
# Requires INEXOR_RUNS, IC_DIR (a new directory: one holding a manifest is refused), and
# INEXOR_SRC or submission from the checkout. Optional: PRESET (c-gh), SEED (0), A_INIT
# (0.1), F_NL (0), WINDOW (1), Y_BLOCKS (auto, emission units), BATCH_PLANES (16), SMALL_PRESET
# (none; c-1024 for the cluster check), GEN_CAP_S (3600), QUIET_S (900).
#   R=$SCRATCH/inexor_runs; INEXOR_RUNS=$R IC_DIR=$R/c-gh-ics-mn SMALL_PRESET=c-1024 \
#     sbatch -A <account> scripts/run/multinode_ics_vista.sbatch
#
# REHEARSAL=1 runs the same legs on a laptop CPU at cdev8-tile32 (a = 0.08, whose
# displacements fit that box's one-slab window):
#   REHEARSAL=1 INEXOR_RUNS=runs/rehearsal IC_DIR=runs/rehearsal/ics-r1/ics SMALL_PRESET=cdev8-tile32 \
#     scripts/run/mpi_lane.sh --run bash scripts/run/multinode_ics_vista.sbatch
# PLANT_MISMATCH=1 gives the small check's every-rank leg another seed, so its gate must stop
# the job; PLANT_LEG_FAIL=1 gives the first rank leg an unknown XLA flag, which aborts every
# rank.
#
# Wall (c-gh on 2 gh): the 2048^3 ICs took 7.0 min on one gb node's four GB200s (1045956),
# mostly the emission's uploads and writes; on two GH200s ~15-30 min, plus the c-1024 check
# (~5 min) and ~15 min of guards: ~1 h at most, 2 h requested.

export PATH="$HOME/.pixi/bin:$PATH"
cd "${INEXOR_SRC:-${SLURM_SUBMIT_DIR:-$PWD}}" && [ -f src/inexor/engine.py ] || {
  echo "FATAL: not an inexor checkout (set INEXOR_SRC)"; exit 1; }
SRC=$(pwd)
RUNS=${INEXOR_RUNS:?set INEXOR_RUNS to the run root, e.g. \$SCRATCH/inexor_runs}
mkdir -p "$RUNS" && RUNS=$(cd "$RUNS" && pwd)
export PYTHONUNBUFFERED=1

abspath () { (cd "$1" 2>/dev/null && pwd); }
[ -n "${IC_DIR:-}" ] || { echo "FATAL: set IC_DIR (the new IC directory)"; exit 1; }
[ -f "$IC_DIR/manifest.json" ] && { echo "FATAL: $IC_DIR already holds an IC manifest"; exit 1; }
mkdir -p "$IC_DIR" && IC_DIR=$(abspath "$IC_DIR")

if [ "${REHEARSAL:-0}" = 1 ]; then
  PRESET=${PRESET:-cdev8-tile32} A_INIT=${A_INIT:-0.08}
fi
PRESET=${PRESET:-c-gh} SEED=${SEED:-0} A_INIT=${A_INIT:-0.1} F_NL=${F_NL:-0} WINDOW=${WINDOW:-1}
Y_BLOCKS=${Y_BLOCKS:-auto} BATCH_PLANES=${BATCH_PLANES:-16}
GEN_CAP_S=${GEN_CAP_S:-3600} QUIET_S=${QUIET_S:-900}
N_RANKS=${SLURM_NTASKS:-${N_RANKS:-2}}

. "$SRC/scripts/run/mpi_env_vista.sh"
. "$SRC/scripts/run/job_lib.sh"

WORK=$(dirname "$IC_DIR")/$(basename "$IC_DIR")-job-$SLURM_JOB_ID
# shellcheck disable=SC2034  # read by job_lib.sh
LEG_DIR=$WORK/legs
CARDS=$SRC/runs/v2
TAG=mnics_${PRESET}_$SLURM_JOB_ID
GEN=(--seed "$SEED" --a-init "$A_INIT" --f-nl "$F_NL" --window "$WINDOW"
     --y-blocks "$Y_BLOCKS" --batch-planes "$BATCH_PLANES")
mkdir -p "$WORK" "$CARDS"

echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
echo "=== $PRESET ICs -> $IC_DIR: seed $SEED, a_init $A_INIT, f_NL $F_NL, window $WINDOW, y-blocks $Y_BLOCKS, batch $BATCH_PLANES ==="
echo "=== small check ${SMALL_PRESET:-none}; nodes ${SLURM_JOB_NODELIST:-local}, launch '$LAUNCH', work $WORK ==="
for t in mpiexec mpicc "$PY"; do echo "  $t -> $(command -v "$t" || echo none)"; done
if [ "${REHEARSAL:-0}" != 1 ]; then
  echo "  modules $(module -t list 2>&1 | tr '\n' ' ')"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null
  free -g | head -2
fi
df -h "$IC_DIR" | tail -1
env | grep -E '^(XLA_|JAX_|INEXOR_|MV2_|MVP_|HYDRA_)' | sort

rc_total=0
finish () {
  echo ""; du -sh "$IC_DIR" "$WORK"/* 2>/dev/null; df -h "$IC_DIR" | tail -1
  echo "=== sacct ==="
  sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,ExitCode -P 2>/dev/null
  echo "=== end $(date) rc_total=$rc_total ==="
}
trap finish EXIT
trap forward_usr1 USR1

stop_job () {
  echo "=== stopping: $1; every output is kept under $WORK and $IC_DIR ==="
  exit $((rc_total > 0 ? rc_total : 1))
}

# ics_ranks ARM PRESET DIR XLA_EXTRA ARGS...: the ICs across the nodes
ics_ranks () {
  local arm=$1 preset=$2 dir=$3 extra=$4; shift 4
  on_ranks "$CARDS/${TAG}_${arm}_s0_r@RANK@" "$extra" "${MPIPATH[@]}" \
    "$PY" -m mpi4py "$DRIVER" ics --preset "$preset" --workdir "$dir" --cards 1 --beat 60 \
    "${MEMBIND_CHECK[@]}" --card "$CARDS/${TAG}_${arm}_s0.json" --comm mpi \
    --comm-timeout 1800 "$@"
}

# ics_one ARM PRESET DIR ARGS...: the same ICs on node 0 alone
ics_one () {
  local arm=$1 preset=$2 dir=$3; shift 3
  "${BIND0[@]}" "${ON0[@]}" "$PY" "$DRIVER" ics --preset "$preset" --workdir "$dir" --cards 1 \
    --beat 60 "${MEMBIND_CHECK[@]}" --card "$CARDS/${TAG}_${arm}_s0.json" "$@"
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

# --- the small check: one rank vs every rank
if [ -n "${SMALL_PRESET:-}" ]; then
  S=$WORK/small
  mkdir -p "$S"
  SMALL_SEED=$SEED
  [ "${REHEARSAL:-0}" = 1 ] && [ "${PLANT_MISMATCH:-0}" = 1 ] && SMALL_SEED=$((SEED + 1))
  run_leg --cap "$GEN_CAP_S" --quiet-limit "$QUIET_S" small-ics-1 \
    ics_one small-ics-1 "$SMALL_PRESET" "$S/ics-1" "${GEN[@]}" \
    || stop_job "the small one-rank ICs failed"
  run_leg --cap "$GEN_CAP_S" --quiet-limit "$QUIET_S" "small-ics-$N_RANKS" \
    ics_ranks small-ics "$SMALL_PRESET" "$S/ics-$N_RANKS" "$PLANT" "${GEN[@]}" \
    --seed "$SMALL_SEED"
  r=$?; summarize small-ics 0
  [ $r -eq 0 ] || stop_job "the small every-rank ICs failed"
  gate_with small-ics same_ics "$S/ics-1" "$S/ics-$N_RANKS" \
    || { rc_total=$((rc_total + 1)); stop_job "the small ICs differ across rank counts"; }
  PLANT=""
fi

# --- the ICs
run_leg --cap "$GEN_CAP_S" --quiet-limit "$QUIET_S" ics \
  ics_ranks ics "$PRESET" "$IC_DIR" "$PLANT" "${GEN[@]}"
r=$?; summarize ics 0
[ $r -eq 0 ] || stop_job "the ICs failed"

# --- the manifest holds the preset's particles in its slab count
want=$("${CPU[@]}" "$PY" -c "from inexor.plan import PRESETS, engine_config
p = PRESETS['$PRESET']; print(p['n_part'] ** 3, p['n_fine'] // engine_config('$PRESET').n_brick)")
read -r want_n want_nb <<< "$want"
got_n=$(manifest_field "$IC_DIR" "m['n_particles']")
got_nb=$(manifest_field "$IC_DIR" "len(m['files'])")
echo ""; echo "=== CHECK: ${got_n:-?} of $want_n particles in ${got_nb:-?} of $want_nb slab files ==="
if [ "$got_n" = "$want_n" ] && [ "$got_nb" = "$want_nb" ]; then
  echo "CHECK PASS"
else
  echo "CHECK FAIL"; rc_total=$((rc_total + 1))
fi
exit $rc_total
