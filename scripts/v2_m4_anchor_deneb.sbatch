#!/bin/bash
#SBATCH -J inexor-m4-anchor
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o %x-%j.log
#
# M-v2-4: the cdev8 K=10 ANCHOR. Owed twice over.
#
# WHY IT IS OWED. Vista 899070's `accuracy-cdev8-k10` leg exited 1 on the D-007
# arena refusal: that sbatch handed it the default --slack 0.10 --arena-frac
# 0.02 while its neighbours ran 0.20/0.08, and a SMALLER K needs MORE arena
# because fewer, larger drifts migrate more particles per step. Without this
# rung the cgh64 K=10 reading has no matched cdev8 baseline, and the charter's
# "within 3x, not 16x" expectation for cgh64-vs-cdev8 has nothing to compare.
#
# WHY IT IS OWED AGAIN. It was run locally on 2026-08-09 and read E = 1.145e-05,
# median 1.400e-06 -- but on the CPU backend. That card is NOT the anchor. cgh64
# is 134M particles and can only run on a cluster GPU, so a CPU-measured cdev8
# reading fails the single axis the anchor exists to hold fixed. XLA-CPU is a
# different computation, not the same one more slowly, and its reduction order
# additionally follows the host core count.
#
# That is not a theoretical objection. Globbed in with the 14 GPU cards, the
# local anchor read as a clean fourth rung and moved the readout's least-squares
# slope from +0.32 to +0.80 -- against a pre-registration where ~1.0 means a
# systematic bias that blocks adoption. Nothing in the output flagged the rungs
# as incommensurable. Cards now carry a `provenance` block and the readout
# refuses to pool across backends, which is how that gets caught rather than
# discovered.
#
# EXPECTED, and pre-registered here before the GPU number exists: E within the
# per-K scatter of the local CPU reading. The two backends are not required to
# agree bitwise and will not, but a reading that lands outside the 41-73%
# per-rung scatter measured at K=20/40/80 would say the backend moves the
# ESTIMAND rather than the last bits, which is a finding in its own right.
#
# CAPACITY 0.20/0.08, identical to jobs 399 and 400, so this rung is directly
# comparable to theirs. This is exactly what 899070's leg got wrong.
#
# BUDGET from deneb's MEASURED cdev8 rate. A 3-rung fit over job 399/400's
# paired walls (K=20 290 s, K=40 592 s, K=80 1229 s) gives 15.7 s per K paired,
# i.e. 7.83 s/step with no fixed overhead. K=10 paired ~= 157 s, call it 3 min.
# 1 h is generous and deneb is unbilled -- but note that a generous wall is only
# free on a job that COMPLETES. Vista 899070 timed out and paid its full 8 h for
# 1 h 19 min of banked work, which is why this is ONE LEG PER JOB.

export PATH="$HOME/.pixi/bin:$PATH"

echo "=== host $(hostname)  job ${SLURM_JOB_ID}  $(date -Is) ==="

if ! command -v pixi >/dev/null 2>&1; then
    echo "FATAL: pixi not on PATH -- this job would run zero science and exit 0"
    exit 1
fi

cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
echo "commit: $(git rev-parse --short HEAD)  branch: $(git rev-parse --abbrev-ref HEAD)"

# The provenance block is the POINT of this re-run, so refuse to produce a card
# that lacks it rather than write a second anchor nobody can place.
pixi run -e gpu python -c "
import sys
sys.path.insert(0, 'scripts')
import v2_m4_f32_mesh_gate as G
if not hasattr(G, '_provenance'):
    sys.exit('FATAL: this checkout predates the provenance block. git pull first -- '
             'a card without a backend field is the exact defect this job exists to fix.')
print('provenance: present')
" || exit 1

pixi run -e gpu python -c "
import jax, sys
d = jax.devices()
print('jax', jax.__version__, 'devices', d)
if d[0].platform == 'cpu':
    sys.exit('FATAL: CPU backend -- a CPU card is not the anchor, it is the thing being replaced')
" || { echo "FATAL: jax/GPU precondition failed"; exit 1; }

STAMP="$(mktemp)"; touch "$STAMP"

pixi run -e gpu python scripts/v2_m4_f32_mesh_gate.py \
    --config cdev8 --leg accuracy --k 10 --slack 0.20 --arena-frac 0.08 \
    --out-suffix _k10
rc=$?

echo ""
echo "=== CARDS WRITTEN (this job only) ==="
n=0
while IFS= read -r f; do
  echo "  $f"; n=$((n+1))
done < <(find runs/v2 -maxdepth 1 -name 'm4_gate_cdev8_accuracy_k10.json' -newer "$STAMP" 2>/dev/null)
[ "$n" -eq 0 ] && echo "  (none)"
rm -f "$STAMP"

echo ""
echo "anchor rc=${rc};  fresh cards: ${n}"
[ "$n" -eq 0 ] && { echo "FATAL: zero cards written by THIS job"; exit 1; }
exit $rc
