#!/bin/bash
#SBATCH -J inexor-m4-kladder
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o %x-%j.log
#
# M-v2-4 leg 1b: does the f32 coarse-mesh error ACCUMULATE with step count?
#
# WHY THIS EXISTS, and why it is now the most diagnostic leg of the milestone.
# Vista 899070's gated reading came in at E = 4.002e-05 at cdev K=40 -- a PASS
# on the tier-1 bar (3.0e-3, 75x margin) but OUTSIDE the pre-registered tier-2
# expectation of <= 1e-5, by 4x. The pre-registration was anchored on
# `v2_g3_floors`' f32 rung (R_Q 1.1e-6), which perturbs only the INITIAL state's
# rounding -- a one-off. This perturbs the coarse force on EVERY step, and that
# difference was under-weighted.
#
# The arithmetic that makes this urgent: 40 x 1e-6 = 4e-5, landing exactly on the
# measurement, while a random walk would give sqrt(40) x 1e-6 ~ 6e-6. So the
# headline number is consistent with error LINEAR in K, and M-v2-4's own
# pre-registration says what that would mean: "growth linear in K or faster means
# a systematic bias rather than roundoff, and that blocks adoption regardless of
# the bar." One point cannot distinguish the two. Three can.
#
# WHAT IT ANSWERS
#   E(K) at cdev8 for K = 20, 40, 80, each a PAIRED run (f32 coarse vs f64
#   coarse, same ICs, everything else identical). Read the slope:
#     ~K^0.5  roundoff accumulating as a random walk       -- expected, benign
#     ~K^1.0  a systematic bias                            -- blocks adoption
#     ~K^0.0  a one-off, and the cdev reading needs another explanation
#
# WHAT IT DOES NOT ANSWER
#   Anything about the absolute size at cdev: this is cdev8, an 8x smaller
#   volume, and the box-size axis is 899070's cgh64 leg. The SLOPE is the
#   deliverable here, not the intercept.
#
# CAPACITY, and a defect this leg exists partly to route around. 899070's
# cdev8 K=10 anchor died on the D-007 arena refusal:
#   "433 particles overflow their brick's capacity and the arena of 41944 slots
#    has only 427 free."
# because the sbatch gave that leg the DEFAULT --slack 0.10 --arena-frac 0.02
# while its neighbours got 0.15/0.05. The direction is counterintuitive and
# worth carrying: a SMALLER K needs MORE arena, because fewer, larger drifts
# migrate more particles per step. M-v2-3 ran cdev8 at the defaults happily --
# at K=40. So all three rungs here get the same generous 0.20/0.08, held FIXED
# across rungs so the slope is not confounded by a storage knob (slack and arena
# are capacity, not physics: they change where a particle is stored, not where
# it is).
#
# BUDGET. cdev8 K=40 ran 16.7 s/step on a Vista gh node and the step is
# host-bound (load ~1.4, GPU 3-4%, and the layout is single-threaded numpy), so
# deneb's weaker GPU costs less than it looks. Budgeting 3x that rate: paired
# K=20+40+80 = 280 engine-steps x 2 arms x ~50 s ~ 7.8 h. 12 h is deliberately
# generous; deneb is unbilled, and the last several runtime estimates on this
# project were all low.

export PATH="$HOME/.pixi/bin:$PATH"

echo "=== host $(hostname)  job ${SLURM_JOB_ID}  $(date -Is) ==="

if ! command -v pixi >/dev/null 2>&1; then
    echo "FATAL: pixi not on PATH -- this job would run zero science and exit 0"
    exit 1
fi

cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
echo "commit: $(git rev-parse --short HEAD)  branch: $(git rev-parse --abbrev-ref HEAD)"

# The backend is DEMONSTRATED, not inferred.
pixi run -e gpu python -c "
import jax, sys
d = jax.devices()
print('jax', jax.__version__, 'devices', d)
if d[0].platform == 'cpu':
    sys.exit('FATAL: CPU backend')
" || { echo "FATAL: jax/GPU precondition failed"; exit 1; }

STAMP="$(mktemp)"; touch "$STAMP"

# No `set -e`: the rungs are independent and CHEAPEST FIRST, so if the longest
# one dies the shorter ones still constrain the slope. Two points give a slope;
# one gives nothing.
rc=0
G=scripts/v2_m4_f32_mesh_gate.py
CAP="--slack 0.20 --arena-frac 0.08"

for K in 20 40 80; do
  echo ""
  echo "--- rung K=${K} (paired) ---"; date -Is
  # shellcheck disable=SC2086
  pixi run -e gpu python "$G" --config cdev8 --leg accuracy --k "$K" $CAP \
      --out-suffix "_k${K}" || rc=1
done

echo ""
echo "=== CARDS WRITTEN (this job only) ==="
n=0
while IFS= read -r f; do
  echo "  $f"; n=$((n+1))
done < <(find runs/v2 -maxdepth 1 -name 'm4_gate_cdev8_accuracy_k*.json' -newer "$STAMP" 2>/dev/null)
[ "$n" -eq 0 ] && echo "  (none)"
rm -f "$STAMP"

echo ""
echo "=== SLOPE ==="
# The readout used to be inline here and fitted ENDPOINTS ONLY, which reported a
# tidy -0.13 for data whose middle rung is 10x both ends on the median. It is a
# script now, it least-squares over all rungs, it shows residuals, and it checks
# monotonicity SEPARATELY -- a slope through non-monotone data is not a result.
pixi run -e gpu python scripts/v2_m4_kladder_readout.py

echo ""
echo "rungs rc=${rc};  fresh cards: ${n}"
[ "$n" -eq 0 ] && { echo "FATAL: zero cards written by THIS job"; exit 1; }
exit $rc
