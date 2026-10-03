#!/bin/bash
#SBATCH -J inexor-mn-steps
#SBATCH -p gh
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00
#SBATCH -o inexor-mn-steps-%j.log
#SBATCH --signal=B:USR1@900
# One realization stepped across the nodes, one GH200 and one rank per node, from ICs made
# elsewhere (hero_ics_vista.sbatch on gb), with checkpoints in REAL_DIR/ckpt. A lost job is
# resubmitted with EXPECT_STEP at its newest checkpoint.
#
# Before the steps: the job-script tests, the MPI launch line on every node (an MPI start and
# a rank that dies at once, which must end the launch), the planner at 1 and N nodes, and the
# preflight against IC_DIR on node 0 (all fatal). Then the control, also fatal: the older
# generation of CONTROL_REF (a run that passed the byte gate, multinode_gate_vista.sbatch) is
# copied into a fresh dir and resumed to the newer one's step with this commit and launch
# line, and the dir must equal CONTROL_REF byte for byte.
#
# Every leg has a time cap; a run leg is also killed once its output (the driver's 60 s
# heartbeat) has been silent for QUIET_S. A failed leg stops the job. At the end the newest
# checkpoint must be at step STOP and hold the preset's particle count. Each leg's output is
# also in REAL_DIR/job-<id>/legs/.
#
# Requires INEXOR_RUNS, IC_DIR, REAL_DIR (not under IC_DIR), CONTROL_REF (a gated run's
# ckpt/, two generations) and CONTROL_ICS (its ICs), and INEXOR_SRC or submission from the
# checkout. Optional: PRESET (c-gh), K (120), EXPECT_STEP (0), STOP (K), EVERY (20),
# CONTROL_PRESET (c-1024), SLACK / ALLOC_MARGIN (0.10), ARENA_FRAC (0.01), LEG_FIXED_S (600) +
# LEG_STEP_S (300) per step (a run leg's cap), QUIET_S (600).
#   R=$SCRATCH/inexor_runs; INEXOR_RUNS=$R IC_DIR=$R/c-gh-ics REAL_DIR=$R/c-gh-r1 \
#     CONTROL_REF=$R/mn-gate-c-1024-1043437/mpi/ckpt CONTROL_ICS=$R/mn-gate-c-1024-1043437/ics \
#     sbatch -A <account> scripts/run/multinode_steps_vista.sbatch
#
# REHEARSAL=1 runs the same legs on a laptop CPU at cdev8-tile32 (steps 0 -> 6 of 40), with
# IC_DIR and CONTROL_REF from a rehearsal of the gate script:
#   G=runs/rehearsal/mn-gate-cdev8-tile32-<tag>; REHEARSAL=1 INEXOR_RUNS=runs/rehearsal \
#     IC_DIR=$G/ics CONTROL_REF=$G/mpi/ckpt CONTROL_ICS=$G/ics REAL_DIR=runs/rehearsal/steps-r1 \
#     scripts/run/mpi_lane.sh --run bash scripts/run/multinode_steps_vista.sbatch
# PLANT_CONTROL_MISMATCH=1 flips one byte of the control copy before its resume, so the gate
# must stop the job before production; PLANT_LEG_FAIL=1 gives the first rank leg an unknown
# XLA flag, which aborts every rank at startup.
#
# Wall (c-gh on 2 gh): ~155-165 s/step from 8x 1043437's 19.2 s/step per rank at c-1024, so
# 120 steps ~5.3 h, plus the load and 6 checkpoints (~6 min each) and ~40 min of guards and
# control: ~6.5 h central, 12 h requested. USR1 900 s before the wall reaches every rank
# through the local mpiexec and dumps every thread's stack.

export PATH="$HOME/.pixi/bin:$PATH"
cd "${INEXOR_SRC:-${SLURM_SUBMIT_DIR:-$PWD}}" && [ -f src/inexor/engine.py ] || {
  echo "FATAL: not an inexor checkout (set INEXOR_SRC)"; exit 1; }
SRC=$(pwd)
RUNS=${INEXOR_RUNS:?set INEXOR_RUNS to the run root, e.g. \$SCRATCH/inexor_runs}
mkdir -p "$RUNS" && RUNS=$(cd "$RUNS" && pwd)
export PYTHONUNBUFFERED=1

abspath () { (cd "$1" 2>/dev/null && pwd); }
for v in IC_DIR CONTROL_REF CONTROL_ICS; do
  [ -n "${!v:-}" ] && [ -d "${!v}" ] || { echo "FATAL: set $v to an existing directory"; exit 1; }
  printf -v "$v" %s "$(abspath "${!v}")"
done
[ -n "${REAL_DIR:-}" ] || { echo "FATAL: set REAL_DIR (the realization's dir)"; exit 1; }
mkdir -p "$REAL_DIR" && REAL_DIR=$(abspath "$REAL_DIR")
case "$REAL_DIR/" in "$IC_DIR"/*) echo "FATAL: REAL_DIR is under IC_DIR"; exit 1 ;; esac

if [ "${REHEARSAL:-0}" = 1 ]; then
  PRESET=${PRESET:-cdev8-tile32} K=${K:-40} STOP=${STOP:-6} EVERY=${EVERY:-3}
  CONTROL_PRESET=${CONTROL_PRESET:-cdev8-tile32}
  SLACK=${SLACK:-0.20} ARENA=${ARENA_FRAC:-0.20}
fi
PRESET=${PRESET:-c-gh} K=${K:-120} EXPECT_STEP=${EXPECT_STEP:-0} STOP=${STOP:-$K}
EVERY=${EVERY:-20} CONTROL_PRESET=${CONTROL_PRESET:-c-1024}
SLACK=${SLACK:-0.10} ALLOC_MARGIN=${ALLOC_MARGIN:-0.10} ARENA=${ARENA:-${ARENA_FRAC:-0.01}}
LEG_FIXED_S=${LEG_FIXED_S:-600} LEG_STEP_S=${LEG_STEP_S:-300} QUIET_S=${QUIET_S:-600}
N_RANKS=${SLURM_NTASKS:-${N_RANKS:-2}}

CKPT=$REAL_DIR/ckpt
# here, before any guard: the driver refuses the same, but only after the guards' ~30 min
if [ "$EXPECT_STEP" = 0 ] && compgen -G "$CKPT/gen*/manifest.json" >/dev/null; then
  echo "FATAL: $CKPT already holds a checkpoint; set EXPECT_STEP to resume it"; exit 1
fi

. "$SRC/scripts/run/mpi_env_vista.sh"
. "$SRC/scripts/run/job_lib.sh"

WORK=$REAL_DIR/job-$SLURM_JOB_ID
# shellcheck disable=SC2034  # read by job_lib.sh
LEG_DIR=$WORK/legs
CARDS=$SRC/runs/v2
TAG=mnsteps_${PRESET}_$SLURM_JOB_ID
LAYOUT=(--slack "$SLACK" --alloc-margin "$ALLOC_MARGIN" --arena-frac "$ARENA")
mkdir -p "$WORK" "$CARDS"

echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
echo "=== $PRESET, K=$K, steps $EXPECT_STEP -> $STOP, every $EVERY, layout ${LAYOUT[*]} ==="
echo "=== ICs $IC_DIR, checkpoints $CKPT, control $CONTROL_REF ($CONTROL_PRESET) ==="
echo "=== nodes ${SLURM_JOB_NODELIST:-local}, launch '$LAUNCH', work $WORK ==="
for t in mpiexec mpicc ptxas "$PY"; do echo "  $t -> $(command -v "$t" || echo none)"; done
if [ "${REHEARSAL:-0}" != 1 ]; then
  echo "  modules $(module -t list 2>&1 | tr '\n' ' ')"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null
  free -g | head -2; numactl --hardware | head -6
fi
df -h "$REAL_DIR" | tail -1
env | grep -E '^(XLA_|JAX_|INEXOR_|MV2_|MVP_|HYDRA_)' | sort

rc_total=0
finish () {
  echo ""; du -sh "$CKPT"/* "$WORK"/* 2>/dev/null; df -h "$REAL_DIR" | tail -1
  echo "=== sacct ==="
  sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,ExitCode -P 2>/dev/null
  echo "=== end $(date) rc_total=$rc_total ==="
}
trap finish EXIT
trap forward_usr1 USR1

# a failed leg ends the job
stop_job () {
  echo "=== stopping: $1; every output is kept under $REAL_DIR ==="
  exit $((rc_total > 0 ? rc_total : 1))
}

# leg_cap FROM STOP: a run leg's time cap
leg_cap () { echo $((LEG_FIXED_S + LEG_STEP_S * ($2 - $1))); }

# ranks ARM PRESET ICS CKPT K FROM STOP EVERY [XLA_FLAGS]: one run across the nodes, one rank
# each, from the ICs or (FROM > 0) the step-FROM checkpoint in CKPT
ranks () {
  local arm=$1 preset=$2 ics=$3 ckpt=$4 k=$5 from=$6 stop=$7 every=$8 extra=${9:-}
  on_ranks "$CARDS/${TAG}_${arm}_s${from}_r@RANK@" "$extra" "${MPIPATH[@]}" \
    "$PY" -m mpi4py "$DRIVER" run --preset "$preset" --workdir "$ics" --cards 1 \
    --k-steps "$k" --stop-at "$stop" --expect-step "$from" --timed-all --beat 60 \
    "${MEMBIND_CHECK[@]}" "${LAYOUT[@]}" --checkpoint-every "$every" \
    --card "$CARDS/${TAG}_${arm}_s${from}.json" \
    --checkpoint-dir "$ckpt" --comm mpi --comm-timeout 1800
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
for n in 1 "$N_RANKS"; do
  run_leg --cap 300 "plan-$n-node" \
    "${CPU[@]}" "$PY" -m inexor.plan --preset "$PRESET" --backend device \
    --n-gpus 1 --n-nodes "$n" --host-gb 116 --device-gb 96 --slack "$SLACK" \
    --alloc-margin "$ALLOC_MARGIN" --arena-frac "$ARENA"
done
run_leg --cap 600 "preflight-$N_RANKS-node" \
  "${BIND0[@]}" "${ON0[@]}" "$PY" "$DRIVER" preflight --preset "$PRESET" --workdir "$IC_DIR" \
    --cards 1 --card "$CARDS/${TAG}_preflight_n$N_RANKS.json" --scratch "$REAL_DIR" \
    --n-nodes "$N_RANKS" "${MACHINE[@]}" "${MEMBIND_CHECK[@]}" "${LAYOUT[@]}" \
  || stop_job "preflight refused"

# --- control: CONTROL_REF's older generation resumed to its newer one, byte for byte
CTRL=$WORK/control/ckpt
read -r old C_FROM C_TO < <(ckpt_generations "$CONTROL_REF") \
  || stop_job "$CONTROL_REF does not hold two generations at different steps"
CK=$(manifest_field "$CONTROL_REF/$old" "m['provenance']['n_steps']")
[ $((C_TO % (C_TO - C_FROM))) = 0 ] \
  || stop_job "the control cannot checkpoint at step $C_TO every $((C_TO - C_FROM)) from $C_FROM"
echo "=== control: $CONTROL_PRESET $old (step $C_FROM) -> $C_TO of $CK, in $CTRL ==="
mkdir -p "$CTRL"
run_leg --cap 1800 control-copy cp -a "$CONTROL_REF/$old" "$CTRL/" \
  || stop_job "the control generation did not copy"
if [ "${REHEARSAL:-0}" = 1 ] && [ "${PLANT_CONTROL_MISMATCH:-0}" = 1 ]; then
  f=$(cd "$CTRL/$old" && find . -type f ! -name manifest.json | sort | head -1)
  echo "=== REHEARSAL: flipping one byte of control/ckpt/$old/$f ==="
  "$PY" - "$CTRL/$old/$f" <<'EOF'
import sys
with open(sys.argv[1], "r+b") as fh:
    fh.seek(-1, 2)
    b = fh.read(1)
    fh.seek(-1, 2)
    fh.write(bytes([b[0] ^ 1]))
EOF
fi
PLANT=""
[ "${REHEARSAL:-0}" = 1 ] && [ "${PLANT_LEG_FAIL:-0}" = 1 ] && PLANT=--xla_planted_unknown_flag
run_leg --cap "$(leg_cap "$C_FROM" "$C_TO")" --quiet-limit "$QUIET_S" \
  "control $C_FROM->$C_TO" ranks control "$CONTROL_PRESET" "$CONTROL_ICS" "$CTRL" "$CK" \
  "$C_FROM" "$C_TO" $((C_TO - C_FROM)) "$PLANT"
r=$?; summarize control "$C_FROM"
[ $r -eq 0 ] || stop_job "the control leg failed"
gate control "$CONTROL_REF" "$CTRL" \
  || { rc_total=$((rc_total + 1)); stop_job "the control differs from $CONTROL_REF"; }

# --- production: EXPECT_STEP -> STOP
run_leg --cap "$(leg_cap "$EXPECT_STEP" "$STOP")" --quiet-limit "$QUIET_S" \
  "steps $EXPECT_STEP->$STOP" ranks steps "$PRESET" "$IC_DIR" "$CKPT" "$K" "$EXPECT_STEP" \
  "$STOP" "$EVERY"
r=$?; summarize steps "$EXPECT_STEP"
[ $r -eq 0 ] || stop_job "the steps leg failed"

# --- the newest checkpoint is at STOP with the preset's particles
want=$("${CPU[@]}" "$PY" -c "from inexor.plan import PRESETS; print(PRESETS['$PRESET']['n_part'] ** 3)")
newest="" best=""
read -r newest best < <(ckpt_newest "$CKPT")
got=$( [ -n "$newest" ] && manifest_field "$CKPT/$newest" "m['n_particles']")
echo ""; echo "=== CHECK: newest ${newest:-none} at step $best of $STOP, ${got:-?} of $want particles ==="
if [ "$best" = "$STOP" ] && [ "$got" = "$want" ]; then
  echo "CHECK PASS"
else
  echo "CHECK FAIL"; rc_total=$((rc_total + 1))
fi
exit $rc_total
