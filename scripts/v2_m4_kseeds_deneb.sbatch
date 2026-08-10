#!/bin/bash
#SBATCH -J inexor-m4-kseeds
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o %x-%j.log
#
# M-v2-4 leg 1c: put an ERROR BAR on the K-ladder.
#
# WHY. Job 399 ran one seed at each of K = 20/40/80 and read E_max = 3.365e-5 /
# 8.457e-5 / 2.800e-5, with medians 3.98e-6 / 3.62e-5 / 2.96e-6. That refutes the
# linear-in-K hypothesis outright -- the error does not grow with step count --
# but it leaves a worse question open: K=20 and K=80 agree closely while K=40
# sits ~10x above both on the median, and with ONE seed per rung there is no way
# to tell "K=40 is special" from "the per-realization scatter is ~10x and three
# points landed this way".
#
# That matters beyond the ladder, because the GATED reading (Vista 899070, cdev
# K=40, E = 4.002e-05, 4x outside its own pre-registration) was taken at K=40
# and also has no error bar. A milestone cannot report a number 4x outside its
# prediction with n=1 and call it measured.
#
# WHAT THIS RUNS. Replicates at every rung rather than at one, so the output is
# three DISTRIBUTIONS rather than three points. Answering "does E scale with K"
# with error bars is strictly better than answering "is K=40 special" twice, and
# costs the same.
#
#   K=20  seeds 1,2,3      (seed 0 from job 399)   ->  n=4
#   K=40  seeds 1,2,3,4,5  (seed 0 from job 399)   ->  n=6, the rung under suspicion
#   K=80  seeds 1,2,3      (seed 0 from job 399)   ->  n=4
#
# Seeds change the IC REALIZATION (`make_ics(g, seed=)` keys jax.random), which
# is the scatter under test. Both arms of a pair still share their ICs exactly --
# the pairing is inside the gate script and is untouched by this.
#
# CAPACITY held at 0.20/0.08 for every rung, identical to job 399, so these
# cards are directly comparable to its three. Job 899070's cdev8 anchor died on
# the D-007 arena refusal at the default 0.10/0.02, and a SMALLER K needs MORE
# arena because fewer, larger drifts migrate more particles per step.
#
# BUDGET from job 399's MEASURED paired walls (K=20 285 s, K=40 584 s, K=80
# 1189 s on this node): 3(285) + 5(584) + 3(1189) = 7342 s ~ 2.0 h. 12 h is
# generous; deneb is unbilled. Note job 399 came in 14x under my a-priori
# estimate because deneb runs cdev8 at ~7.8 s/step against a Vista gh node's
# ~16.7 s/step -- the engine is host-bound and single-threaded, so Grace loses.

export PATH="$HOME/.pixi/bin:$PATH"

echo "=== host $(hostname)  job ${SLURM_JOB_ID}  $(date -Is) ==="

if ! command -v pixi >/dev/null 2>&1; then
    echo "FATAL: pixi not on PATH -- this job would run zero science and exit 0"
    exit 1
fi

cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
echo "commit: $(git rev-parse --short HEAD)  branch: $(git rev-parse --abbrev-ref HEAD)"

pixi run -e gpu python -c "
import jax, sys
d = jax.devices()
print('jax', jax.__version__, 'devices', d)
if d[0].platform == 'cpu':
    sys.exit('FATAL: CPU backend')
" || { echo "FATAL: jax/GPU precondition failed"; exit 1; }

STAMP="$(mktemp)"; touch "$STAMP"

# No `set -e`, and the SUSPECT rung first: if the job dies partway, the readings
# that constrain the open question should already be on disk. Within a rung the
# seeds are independent, so a failure costs one point rather than the rung.
rc=0
G=scripts/v2_m4_f32_mesh_gate.py
CAP="--slack 0.20 --arena-frac 0.08"

run_seed () {
  local K="$1" S="$2"
  echo ""; echo "--- K=${K} seed=${S} ---"; date -Is
  # shellcheck disable=SC2086
  pixi run -e gpu python "$G" --config cdev8 --leg accuracy --k "$K" --seed "$S" $CAP \
      --out-suffix "_k${K}_s${S}" || rc=1
}

for S in 1 2 3 4 5; do run_seed 40 "$S"; done
for S in 1 2 3;       do run_seed 20 "$S"; done
for S in 1 2 3;       do run_seed 80 "$S"; done

echo ""
echo "=== CARDS WRITTEN (this job only) ==="
n=0
while IFS= read -r f; do
  echo "  $f"; n=$((n+1))
done < <(find runs/v2 -maxdepth 1 -name 'm4_gate_cdev8_accuracy_k*_s*.json' -newer "$STAMP" 2>/dev/null)
[ "$n" -eq 0 ] && echo "  (none)"
rm -f "$STAMP"

echo ""
echo "=== LADDER WITH ERROR BARS (this job + job 399's seed-0 rungs) ==="
pixi run -e gpu python scripts/v2_m4_kladder_readout.py

echo ""
echo "seeds rc=${rc};  fresh cards: ${n}"
[ "$n" -eq 0 ] && { echo "FATAL: zero cards written by THIS job"; exit 1; }
exit $rc
