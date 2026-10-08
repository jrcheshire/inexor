#!/bin/bash
#SBATCH -J inexor-mn-scaling
#SBATCH -p gh
#SBATCH -N 16
#SBATCH -n 16
#SBATCH --ntasks-per-node=1
#SBATCH -t 05:00:00
#SBATCH -o inexor-mn-scaling-%j.log
#SBATCH --signal=B:USR1@900
# Strong scaling inside one allocation: one segment of a finished realization re-run at each
# rank count in RANK_COUNTS, one rank per node, every leg on the allocation's first nodes (the
# 2-rank leg on nodes 0-1, the 4-rank leg on nodes 0-3, ...), so the legs differ in rank count
# and in nothing else. The segment is REF_CKPT's older generation resumed to its newer one's
# step; each leg's checkpoint dir must then equal REF_CKPT byte for byte (the run does not
# depend on the rank count).
#
# Before the legs (all fatal): the MPI launch line at every rank count (that many ranks, on
# that many distinct nodes), a rank that dies ending the launch, the planner and the preflight
# at every rank count. A leg that fails, or whose checkpoints differ from REF_CKPT, is counted
# and the next leg still runs. A leg is killed once its output (the driver's 60 s heartbeat)
# has been silent; nothing else bounds a leg but the wall. Each leg's output is also in
# OUT_DIR/job-<id>/legs/; its checkpoints in OUT_DIR/job-<id>/n<ranks>/ckpt.
#
# Requires INEXOR_RUNS, REF_CKPT (a run's ckpt/, two generations at different steps), REF_ICS
# (that run's ICs), OUT_DIR, and INEXOR_SRC or submission from the checkout. Optional: PRESET
# (c-gh), RANK_COUNTS ("2 4 8 16"; each at most the allocation's tasks), Y_BLOCKS (auto),
# SLACK / ALLOC_MARGIN (0.10), ARENA_FRAC (0.01) -- REF_CKPT's own, or the gate fails on
# the manifests -- and QUIET_S (600).
# 2048^3, steps 100 -> 120 of the 2-gh-node realization, at 2, 4, 8 and 16 gh nodes:
#   R=$SCRATCH/inexor_runs; INEXOR_RUNS=$R REF_CKPT=$R/c-gh-k120/ckpt REF_ICS=$R/c-gh-lcdm-r1 \
#     OUT_DIR=$R/c-gh-scaling sbatch -A <account> scripts/run/multinode_scaling_vista.sbatch
#
# REHEARSAL=1 runs the same legs on a laptop CPU at cdev8-tile32 (1 and 2 ranks), from a
# rehearsal of the gate script:
#   G=runs/rehearsal/mn-gate-cdev8-tile32-<tag>; REHEARSAL=1 INEXOR_RUNS=runs/rehearsal \
#     REF_CKPT=$G/mpi/ckpt REF_ICS=$G/ics OUT_DIR=runs/rehearsal/scaling-r1 \
#     scripts/run/mpi_lane.sh --run bash scripts/run/multinode_scaling_vista.sbatch
# PLANT_MISMATCH=1 flips one byte of the first leg's copy before its resume, so its gate must
# fail and the next leg still run.
#
# Wall (c-gh, 20 steps late in the run): 1045958 ran its last 20 steps at ~126 s/step on 2 gh
# nodes, so the 2-rank leg is ~42 min; at 4 / 8 / 16 ranks ~20-30 / 12-20 / 7-14 min if the
# exchange and the per-step fixed costs keep pace (unmeasured). Each leg also copies a
# generation (~80 GB) and loads it; with ~20 min of guards, ~2-2.5 h central, 5 h requested.
# Every node is charged for every leg. USR1 900 s before the wall reaches every rank through
# the local mpiexec and dumps every thread's stack.

export PATH="$HOME/.pixi/bin:$PATH"
cd "${INEXOR_SRC:-${SLURM_SUBMIT_DIR:-$PWD}}" && [ -f src/inexor/engine.py ] || {
  echo "FATAL: not an inexor checkout (set INEXOR_SRC)"; exit 1; }
SRC=$(pwd)
RUNS=${INEXOR_RUNS:?set INEXOR_RUNS to the run root, e.g. \$SCRATCH/inexor_runs}
mkdir -p "$RUNS" && RUNS=$(cd "$RUNS" && pwd)
export PYTHONUNBUFFERED=1

abspath () { (cd "$1" 2>/dev/null && pwd); }
for v in REF_CKPT REF_ICS; do
  [ -n "${!v:-}" ] && [ -d "${!v}" ] || { echo "FATAL: set $v to an existing directory"; exit 1; }
  printf -v "$v" %s "$(abspath "${!v}")"
done
[ -n "${OUT_DIR:-}" ] || { echo "FATAL: set OUT_DIR (the legs' dir)"; exit 1; }
mkdir -p "$OUT_DIR" && OUT_DIR=$(abspath "$OUT_DIR")
case "$OUT_DIR/" in "$REF_CKPT"/*|"$REF_ICS"/*)
  echo "FATAL: OUT_DIR is under REF_CKPT or REF_ICS"; exit 1 ;; esac

if [ "${REHEARSAL:-0}" = 1 ]; then
  PRESET=${PRESET:-cdev8-tile32} RANK_COUNTS=${RANK_COUNTS:-1 2}
  SLACK=${SLACK:-0.20} ARENA=${ARENA_FRAC:-0.20}
  N_TASKS=${N_TASKS:-2}
fi
PRESET=${PRESET:-c-gh} RANK_COUNTS=${RANK_COUNTS:-2 4 8 16} Y_BLOCKS=${Y_BLOCKS:-auto}
SLACK=${SLACK:-0.10} ALLOC_MARGIN=${ALLOC_MARGIN:-0.10} ARENA=${ARENA:-${ARENA_FRAC:-0.01}}
QUIET_S=${QUIET_S:-600}
N_TASKS=${SLURM_NTASKS:-${N_TASKS:-16}}
read -ra COUNTS <<< "$RANK_COUNTS"
for n in "${COUNTS[@]}"; do
  [[ "$n" =~ ^[1-9][0-9]*$ ]] && [ "$n" -le "$N_TASKS" ] || {
    echo "FATAL: rank count '$n' is not between 1 and the allocation's $N_TASKS tasks"; exit 1; }
done
N_RANKS=${COUNTS[${#COUNTS[@]}-1]}

. "$SRC/scripts/run/mpi_env_vista.sh"
. "$SRC/scripts/run/job_lib.sh"

read -r OLD FROM TO < <(ckpt_generations "$REF_CKPT") \
  || { echo "FATAL: $REF_CKPT does not hold two generations at different steps"; exit 1; }
K=$(manifest_field "$REF_CKPT/$OLD" "m['provenance']['n_steps']")
[ $((TO % (TO - FROM))) = 0 ] \
  || { echo "FATAL: a leg cannot checkpoint at step $TO every $((TO - FROM)) from $FROM"; exit 1; }

WORK=$OUT_DIR/job-$SLURM_JOB_ID
# shellcheck disable=SC2034  # read by job_lib.sh
LEG_DIR=$WORK/legs
CARDS=$SRC/runs/v2
TAG=mnscale_${PRESET}_$SLURM_JOB_ID
LAYOUT=(--slack "$SLACK" --alloc-margin "$ALLOC_MARGIN" --arena-frac "$ARENA")
mkdir -p "$WORK" "$CARDS"

# on N: the launch line of an N-rank leg, on the allocation's first N nodes
on () {
  N_RANKS=$1
  if [ "${REHEARSAL:-0}" = 1 ]; then LAUNCH="mpiexec -n $1"; else LAUNCH="ibrun -n $1 -o 0"; fi
}

echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
echo "=== $PRESET: $OLD (step $FROM) -> $TO of $K at ranks ${COUNTS[*]}, y-blocks $Y_BLOCKS, layout ${LAYOUT[*]} ==="
echo "=== reference $REF_CKPT, ICs $REF_ICS ==="
echo "=== nodes ${SLURM_JOB_NODELIST:-local}, work $WORK ==="
for t in mpiexec mpicc ptxas "$PY"; do echo "  $t -> $(command -v "$t" || echo none)"; done
if [ "${REHEARSAL:-0}" != 1 ]; then
  echo "  modules $(module -t list 2>&1 | tr '\n' ' ')"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null
  free -g | head -2; numactl --hardware | head -6
fi
df -h "$OUT_DIR" | tail -1
env | grep -E '^(XLA_|JAX_|INEXOR_|MV2_|MVP_|HYDRA_)' | sort

rc_total=0
LEG_S=()
finish () {
  echo ""; echo "=== SCALING: ranks, leg wall s (copy and load included; per-step times are in the cards) ==="
  local i
  for i in "${!LEG_S[@]}"; do echo "  ${LEG_S[$i]}"; done
  du -sh "$WORK"/n* 2>/dev/null; df -h "$OUT_DIR" | tail -1
  echo "=== sacct ==="
  sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,ExitCode -P 2>/dev/null
  echo "=== end $(date) rc_total=$rc_total ==="
}
trap finish EXIT
trap forward_usr1 USR1

stop_job () {
  echo "=== stopping: $1; every output is kept under $WORK ==="
  exit $((rc_total > 0 ? rc_total : 1))
}

# ranks ARM CKPT [ARGS...]: the segment FROM -> TO across the current N_RANKS, from CKPT
ranks () {
  local arm=$1 ckpt=$2; shift 2
  on_ranks "$CARDS/${TAG}_${arm}_s${FROM}_r@RANK@" "" "${MPIPATH[@]}" \
    "$PY" -m mpi4py "$DRIVER" run --preset "$PRESET" --workdir "$REF_ICS" --cards "$N_CARDS" \
    --k-steps "$K" --stop-at "$TO" --expect-step "$FROM" --timed-all --beat 60 \
    "${MEMBIND_CHECK[@]}" "${LAYOUT[@]}" --y-blocks "$Y_BLOCKS" \
    --checkpoint-every $((TO - FROM)) --card "$CARDS/${TAG}_${arm}_s${FROM}.json" \
    --checkpoint-dir "$ckpt" --comm mpi --comm-timeout 1800 "$@"
}

# --- guards, every rank count before any leg
DISTINCT=1
[ "${REHEARSAL:-0}" = 1 ] && DISTINCT=0  # one laptop: every rank on one host
run_leg --quiet-limit 1800 build-mpi4py build_mpi4py || stop_job "mpi4py did not build"
for n in "${COUNTS[@]}"; do
  on "$n"
  run_leg --quiet-limit 300 "guard-launch-mpi-$n" \
    on_ranks "$CARDS/${TAG}_guard-mpi-${n}_r@RANK@" "" "${MPIPATH[@]}" \
      "$PY" -m mpi4py -c "from mpi4py import MPI; import jax, socket; c = MPI.COMM_WORLD
h = c.allgather(socket.gethostname())
assert c.Get_size() == $n, c.Get_size()
assert not $DISTINCT or len(set(h)) == $n, h
print('rank', c.Get_rank(), 'of', c.Get_size(), 'on', h[c.Get_rank()], jax.devices())" \
    || stop_job "the $n-rank launch line does not start $n ranks on $n nodes"
  run_leg --quiet-limit 300 "plan-$n-node" \
    "${CPU[@]}" "$PY" -m inexor.plan --preset "$PRESET" --backend device \
      "${PLAN_MACHINE[@]}" --n-nodes "$n" --y-blocks "$Y_BLOCKS" --slack "$SLACK" \
      --alloc-margin "$ALLOC_MARGIN" --arena-frac "$ARENA"
  run_leg --quiet-limit 600 "preflight-$n-node" \
    "${BIND0[@]}" "${ON0[@]}" "$PY" "$DRIVER" preflight --preset "$PRESET" --workdir "$REF_ICS" \
      --cards "$N_CARDS" --card "$CARDS/${TAG}_preflight_n$n.json" --scratch "$OUT_DIR" \
      --n-nodes "$n" --y-blocks "$Y_BLOCKS" "${MACHINE[@]}" "${MEMBIND_CHECK[@]}" "${LAYOUT[@]}" \
    || stop_job "preflight refused at $n nodes"
done
on "${COUNTS[${#COUNTS[@]}-1]}"
run_leg --quiet-limit 300 --expect-fail guard-dead-rank \
  on_ranks "$CARDS/${TAG}_guard-dead_r@RANK@" "" "$PY" -c "import os, signal; os.kill(os.getpid(), signal.SIGKILL)" \
  || stop_job "a rank that dies does not end the launch"

# --- the legs
PLANT=${PLANT_MISMATCH:-0}
[ "${REHEARSAL:-0}" = 1 ] || PLANT=0
for n in "${COUNTS[@]}"; do
  on "$n"
  D=$WORK/n$n/ckpt
  t0=$SECONDS
  mkdir -p "$D"
  if ! run_leg "copy-$n" cp -a "$REF_CKPT/$OLD" "$D/"; then
    LEG_S+=("$n FAILED (copy)"); continue
  fi
  if [ "$PLANT" = 1 ]; then
    f=$(cd "$D/$OLD" && find . -type f ! -name manifest.json | sort | head -1)
    echo "=== REHEARSAL: flipping one byte of n$n/ckpt/$OLD/$f ==="
    "$PY" - "$D/$OLD/$f" <<'EOF'
import sys
with open(sys.argv[1], "r+b") as fh:
    fh.seek(-1, 2)
    b = fh.read(1)
    fh.seek(-1, 2)
    fh.write(bytes([b[0] ^ 1]))
EOF
    PLANT=0
  fi
  run_leg --quiet-limit "$QUIET_S" "n$n $FROM->$TO" ranks "n$n" "$D"
  r=$?; summarize "n$n" "$FROM"
  if [ $r -ne 0 ]; then
    LEG_S+=("$n FAILED after $((SECONDS - t0)) s"); continue
  fi
  LEG_S+=("$n $((SECONDS - t0))")
  gate "n$n" "$REF_CKPT" "$D" || rc_total=$((rc_total + 1))
done
exit $rc_total
