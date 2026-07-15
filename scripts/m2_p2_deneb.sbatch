#!/bin/bash
#SBATCH -J inexor-p2
#SBATCH -w deneb
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 4:00:00
#SBATCH --exclusive
#SBATCH -o %x-%j.log
# P2/P3: the last measurement before the continue/shelve call.
#
# WHERE THIS SITS. P1 (job 22) killed the memory claim: DISCO-DJ f32 pays
# 436.6-447.3 B/particle vs inexor's 452.6-488.2, so the ~440 B/particle
# transient is INHERENT and inexor is slightly WORSE at f32. And inexor is not
# f64-class accurate either -- D-015 puts its gradient at medrel 2e-4..3.3e-3 vs
# the smooth path (quantization) where f64 replay drifts ~1e-11. On both measured
# axes inexor sits BETWEEN f32 and f64: a Pareto point, not a win.
#
# So the fair summary to beat is "a slower DISCO-DJ with slightly less
# floating-point error", and exactly two things could still overturn it:
#
#   P2: does float-replay error GROW with K? Roundoff accumulates over a float
#       backward sweep; exact replay cannot, by construction. GROWING => exact
#       replay has a real niche at long rollouts (where f_NL work lives).
#       BOTH FLAT => the summary stands and shelving is the honest call.
#   P3: is inexor actually slower? Never measured -- the "slower" half of that
#       summary is an assumption on our side too.
#
# Both arms measure each code against ITS OWN f64 gradient, which is directly
# comparable across codes without folding in convention differences.
#
# --exclusive: P3 is a benchmark; a shared card corrupts the timings.
# --mem REQUIRED (partition default 124000M > deneb's 56 GB -> pends forever).
#
# Submit from ~/spherex/inexor on deneb:  sbatch scripts/m2_p2_deneb.sbatch

set -e
cd ~/spherex/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export PYTEST_XDIST_AUTO_NUM_WORKERS=1

echo "=== env ==="
git log --oneline -1
nvidia-smi --query-gpu=name,driver_version --format=csv || true

echo
echo "########## arm 1: inexor -- adjoint vs its own f64 float path, by K ##########"
# D-015's machinery, K axis opened. x64 is set inside the script (it needs a
# clean f64 reference); this is a FIDELITY measurement, not the memory one.
pixi run -e gpu python scripts/m2_grad_gate.py --k-sweep 5,10,20,40

echo
echo "########## arm 2: DISCO-DJ -- f32 vs its own f64, by K (+ P3 timing) ##########"
pixi run --manifest-path ~/src/disco-mocks/pixi.toml -e gpu \
  python scripts/m2_p2_disco_drift.py --n 64 --k 5,10,20,40

echo "=== DONE ==="
