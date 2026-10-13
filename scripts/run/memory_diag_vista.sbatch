#!/bin/bash
#SBATCH -J inexor-mem-diag
#SBATCH -p gh
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --ntasks-per-node=1
#SBATCH -t 03:00:00
#SBATCH -o inexor-mem-diag-%j.log
#SBATCH --signal=B:USR1@900
# Where host memory goes, in two parts on two gh nodes:
#   one   a one-GPU run on node 0, outside MPI, from ONE_ICS to step ONE_STOP of ONE_K, with
#         the memory by mapping kind at every boundary (--smaps) and, after the run, as the
#         compilation and program caches are cleared and freed memory is trimmed
#         (--cache-probe --trim-probe). With ONE_CKPT (a run of the same ICs and schedule to
#         the same step, one generation), its checkpoint must equal ONE_CKPT's byte for byte.
#   mpi   STEPS steps of REF_CKPT's realization from its older generation on two ranks, twice:
#         as the multi-node jobs run (arm `base`), and with MPI_OFF in every rank's environment
#         (arm `off`), both with --smaps and the probes. The two checkpoints must be equal: MPI
#         settings do not change the numbers.
# Every leg's output is also in OUT_DIR/job-<id>/legs/, its cards in runs/cards/.
#
# Requires INEXOR_RUNS, ONE_ICS, REF_CKPT (two generations at different steps), REF_ICS (its
# ICs), OUT_DIR, MPI_OFF (space-separated VAR=value), and INEXOR_SRC or submission from the
# checkout. Optional: ONE_CKPT, ONE_PRESET (cgh64), ONE_K (120), ONE_STOP (ONE_K), ONE_FINE
# (float64), PRESET (c-gh), STEPS (10, or the generations' gap if smaller), SLACK /
# ALLOC_MARGIN (0.10), ARENA_FRAC (0.01), QUIET_S (600).
#   R=$SCRATCH/inexor_runs; S=$R/gh-same-ics-<job>/cgh64; INEXOR_RUNS=$R ONE_ICS=$S/ics \
#     ONE_CKPT=$S/inexor-float64/ckpt REF_CKPT=$R/c-gh-k120/ckpt REF_ICS=$R/c-gh-lcdm-r1 \
#     OUT_DIR=$R/mem-diag MPI_OFF="..." sbatch -A <account> scripts/run/memory_diag_vista.sbatch
#
# REHEARSAL=1 runs the same legs on a laptop CPU at cdev8-tile32, from a rehearsal of the
# gate script:
#   G=runs/rehearsal/mn-gate-cdev8-tile32-<tag>; REHEARSAL=1 INEXOR_RUNS=runs/rehearsal \
#     ONE_ICS=$G/ics REF_CKPT=$G/mpi/ckpt REF_ICS=$G/ics OUT_DIR=runs/rehearsal/mem-diag-r1 \
#     MPI_OFF="MVP_NOLOCAL=1" scripts/run/mpi_lane.sh --run \
#     bash scripts/run/memory_diag_vista.sbatch
#
# Wall: one ~16 min (a 512^3 run takes ~965 s); each mpi arm copies a generation, loads
# it (~1.5 min) and runs 10 steps at ~127 s (2048^3 on 2 gh), ~25 min; with the guards ~75 min,
# 3 h requested. Node 1 idles through `one`.

export PATH="$HOME/.pixi/bin:$PATH"
cd "${INEXOR_SRC:-${SLURM_SUBMIT_DIR:-$PWD}}" && [ -f src/inexor/engine.py ] || {
  echo "FATAL: not an inexor checkout (set INEXOR_SRC)"; exit 1; }
SRC=$(pwd)
RUNS=${INEXOR_RUNS:?set INEXOR_RUNS to the run root, e.g. \$SCRATCH/inexor_runs}
mkdir -p "$RUNS" && RUNS=$(cd "$RUNS" && pwd)
export PYTHONUNBUFFERED=1

abspath () { (cd "$1" 2>/dev/null && pwd); }
for v in ONE_ICS REF_CKPT REF_ICS; do
  [ -n "${!v:-}" ] && [ -d "${!v}" ] || { echo "FATAL: set $v to an existing directory"; exit 1; }
  printf -v "$v" %s "$(abspath "${!v}")"
done
if [ -n "${ONE_CKPT:-}" ]; then
  [ -d "$ONE_CKPT" ] || { echo "FATAL: ONE_CKPT $ONE_CKPT is not a directory"; exit 1; }
  ONE_CKPT=$(abspath "$ONE_CKPT")
fi
[ -n "${OUT_DIR:-}" ] || { echo "FATAL: set OUT_DIR (the legs' dir)"; exit 1; }
mkdir -p "$OUT_DIR" && OUT_DIR=$(abspath "$OUT_DIR")
for d in "$REF_CKPT" "$REF_ICS" "$ONE_ICS" ${ONE_CKPT:+"$ONE_CKPT"}; do
  case "$OUT_DIR/" in "$d"/*) echo "FATAL: OUT_DIR is under $d"; exit 1 ;; esac
done
read -ra OFF <<< "${MPI_OFF:-}"
[ ${#OFF[@]} -gt 0 ] || { echo "FATAL: set MPI_OFF (VAR=value ..., the off arm's MPI settings)"; exit 1; }
for kv in "${OFF[@]}"; do
  [[ "$kv" =~ ^[A-Za-z_][A-Za-z0-9_]*=[^=]*$ ]] || {
    echo "FATAL: MPI_OFF item '$kv' is not VAR=value"; exit 1; }
done

if [ "${REHEARSAL:-0}" = 1 ]; then
  # the gate rehearsal's 40-step schedule, 4 steps of it
  ONE_PRESET=${ONE_PRESET:-cdev8-tile32} PRESET=${PRESET:-cdev8-tile32}
  ONE_K=${ONE_K:-40} ONE_STOP=${ONE_STOP:-4}
  SLACK=${SLACK:-0.20} ARENA=${ARENA_FRAC:-0.20}
fi
ONE_PRESET=${ONE_PRESET:-cgh64} ONE_K=${ONE_K:-120} ONE_FINE=${ONE_FINE:-float64}
ONE_STOP=${ONE_STOP:-$ONE_K}
[[ "$ONE_STOP" =~ ^[1-9][0-9]*$ ]] && [ "$ONE_STOP" -le "$ONE_K" ] || {
  echo "FATAL: ONE_STOP $ONE_STOP is not a step of the $ONE_K-step schedule"; exit 1; }
PRESET=${PRESET:-c-gh}
SLACK=${SLACK:-0.10} ALLOC_MARGIN=${ALLOC_MARGIN:-0.10} ARENA=${ARENA:-${ARENA_FRAC:-0.01}}
QUIET_S=${QUIET_S:-600}
N_RANKS=2

. "$SRC/scripts/run/mpi_env_vista.sh"
. "$SRC/scripts/run/job_lib.sh"

read -r OLD FROM TO < <(ckpt_generations "$REF_CKPT") \
  || { echo "FATAL: $REF_CKPT does not hold two generations at different steps"; exit 1; }
K=$(manifest_field "$REF_CKPT/$OLD" "m['provenance']['n_steps']")
STEPS=${STEPS:-$((TO - FROM < 10 ? TO - FROM : 10))}
STOP=$((FROM + STEPS))
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] && [ "$STOP" -le "$K" ] && [ $((STOP % STEPS)) = 0 ] || {
  echo "FATAL: $STEPS steps from step $FROM cannot end on a checkpoint of a $K-step run"; exit 1; }

WORK=$OUT_DIR/job-$SLURM_JOB_ID
# shellcheck disable=SC2034  # read by job_lib.sh
LEG_DIR=$WORK/legs
CARDS=$SRC/runs/cards
TAG=memdiag_$SLURM_JOB_ID
LAYOUT=(--slack "$SLACK" --alloc-margin "$ALLOC_MARGIN" --arena-frac "$ARENA")
PROBES=(--smaps --cache-probe --trim-probe)
mkdir -p "$WORK" "$CARDS"
if [ "${REHEARSAL:-0}" = 1 ]; then LAUNCH="mpiexec -n 2"; else LAUNCH="ibrun -n 2 -o 0"; fi

echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
echo "=== one: $ONE_PRESET from $ONE_ICS to step $ONE_STOP of $ONE_K ($ONE_FINE), reference ${ONE_CKPT:-none} ==="
echo "=== mpi: $PRESET $OLD (step $FROM) -> $STOP of $K from $REF_CKPT; off arm: ${OFF[*]} ==="
echo "=== nodes ${SLURM_JOB_NODELIST:-local}, work $WORK, layout ${LAYOUT[*]} ==="
if [ "${REHEARSAL:-0}" != 1 ]; then
  echo "  modules $(module -t list 2>&1 | tr '\n' ' ')"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null
  free -g | head -2
  # which network layer MPI runs over (UCX or libfabric) decides which settings reach it
  mpichversion 2>/dev/null | head -12
  ucx_info -v 2>/dev/null | head -3
  # the installed UCX's registration-cache settings, by name (UCX warns on an unknown one)
  ucx_info -f 2>/dev/null | grep -E '^UCX_[A-Z_]*(RCACHE|REG_METHODS)' | head -20
  fi_info --version 2>/dev/null | head -3
fi
env | grep -E '^(XLA_|JAX_|INEXOR_|MV2_|MVP_|MPIR_|UCX_|FI_|HYDRA_)' | sort

rc_total=0
finish () {
  echo ""; echo "=== sacct ==="
  sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,ExitCode -P 2>/dev/null
  echo "=== end $(date) rc_total=$rc_total ==="
}
trap finish EXIT
trap forward_usr1 USR1

stop_job () {
  echo "=== stopping: $1; every output is kept under $WORK ==="
  exit $((rc_total > 0 ? rc_total : 1))
}

# --- guards
run_leg --quiet-limit 1800 build-mpi4py build_mpi4py || stop_job "mpi4py did not build"
DISTINCT=1
[ "${REHEARSAL:-0}" = 1 ] && DISTINCT=0  # one laptop: both ranks on one host
run_leg --quiet-limit 300 guard-launch-mpi-2 \
  on_ranks "$CARDS/${TAG}_guard-mpi_r@RANK@" "" "${MPIPATH[@]}" \
    "$PY" -m mpi4py -c "from mpi4py import MPI; import jax, socket; c = MPI.COMM_WORLD
h = c.allgather(socket.gethostname())
assert c.Get_size() == 2, c.Get_size()
assert not $DISTINCT or len(set(h)) == 2, h
print('rank', c.Get_rank(), 'of', c.Get_size(), 'on', h[c.Get_rank()], jax.devices())" \
  || stop_job "the 2-rank launch line does not start 2 ranks on 2 nodes"

# --- one: node 0, no MPI, no persistent compilation cache (as the reference ran)
run_leg --quiet-limit "$QUIET_S" one \
  "${BIND0[@]}" env -u JAX_COMPILATION_CACHE_DIR "$PY" "$DRIVER" run --preset "$ONE_PRESET" \
    --workdir "$ONE_ICS" --cards "$N_CARDS" --card "$CARDS/${TAG}_one.json" \
    --k-steps "$ONE_K" --stop-at "$ONE_STOP" --timed-last --beat 60 "${MEMBIND_CHECK[@]}" \
    "${LAYOUT[@]}" --fine-dtype "$ONE_FINE" --checkpoint-dir "$WORK/one/ckpt" \
    --checkpoint-every "$ONE_STOP" "${PROBES[@]}"
if [ -n "${ONE_CKPT:-}" ]; then
  gate "one against $ONE_CKPT" "$ONE_CKPT" "$WORK/one/ckpt" || rc_total=$((rc_total + 1))
fi

# --- mpi: the same segment under two MPI environments
for arm in base off; do
  D=$WORK/$arm/ckpt
  mkdir -p "$D"
  run_leg "copy-$arm" cp -a "$REF_CKPT/$OLD" "$D/" || continue
  pre=()
  [ "$arm" = off ] && pre=(env "${OFF[@]}")
  run_leg --quiet-limit "$QUIET_S" "mpi-$arm $FROM->$STOP" \
    on_ranks "$CARDS/${TAG}_${arm}_s${FROM}_r@RANK@" "" "${pre[@]}" "${MPIPATH[@]}" \
      "$PY" -m mpi4py "$DRIVER" run --preset "$PRESET" --workdir "$REF_ICS" --cards "$N_CARDS" \
      --k-steps "$K" --stop-at "$STOP" --expect-step "$FROM" --beat 60 \
      "${MEMBIND_CHECK[@]}" "${LAYOUT[@]}" --checkpoint-every "$STEPS" \
      --card "$CARDS/${TAG}_${arm}_s${FROM}.json" --checkpoint-dir "$D" \
      --comm mpi --comm-timeout 1800 "${PROBES[@]}"
  summarize "$arm" "$FROM"
done
gate "MPI settings do not change the numbers" "$WORK/base/ckpt" "$WORK/off/ckpt" \
  || rc_total=$((rc_total + 1))
exit $rc_total
