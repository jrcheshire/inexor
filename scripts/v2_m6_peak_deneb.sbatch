#!/bin/bash
#SBATCH -J m6-peak-cdev
#SBATCH -p compute
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH -t 01:30:00
#SBATCH -o m6-peak-cdev-%j.log
# M-v2-6 Stage 0 at cdev, the anchor config: the laptop ran cdev8, where every
# term is tens of MB against a 0.31 GB backend baseline, so the nets are a small
# difference of large numbers. At cdev the terms separate -- `pending` alone is
# 537 MB there against 67 MB at cdev8.
#
# THIS JOB MEASURES THE PRE-FIX TREE ON PURPOSE. cdev8 showed the step peak is
# LINEAR IN K (0.331 GB/step, fit residuals <= 0.143 GB) because `cap` takes a
# new value every step (285,554 -> 397,319 over ten) and every eager jnp op keyed
# on it keys a new XLA shape. The fix is in development; this is the baseline it
# gets judged against, and it also answers whether the slope scales with cap or
# with N.
#
# --mem=48G, not the 32G first proposed: deneb-the-node allows 56G, the quantity
# being measured is an unbounded accumulation, and an OOM kill is not a
# measurement (the standing rule is no doomed jobs, and express --mem not -w).
# CPU env by construction (`default` is CPU-only jax, so the backend cannot
# silently become CUDA) -- which this probe REQUIRES: ru_maxrss is HOST memory
# and on CUDA the meshes are in VRAM, where every ratio reads 1.0 (deneb 409).
# NO set -e, deliberately: independent legs, cheapest first, per-leg exit codes.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
pixi run -e default python -c "
import jax
d = jax.devices()
print('jax', jax.__version__, d)
assert d[0].platform == 'cpu', 'expected the CPU backend in the default env'
" || { echo "FATAL: jax import/backend check failed"; exit 1; }

export TMPDIR="${TMPDIR:-/tmp}/m6_peak_$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
echo "=== $(hostname): $(nproc) cores, TMPDIR $TMPDIR ==="
free -g | head -2
STAMP="$TMPDIR/.stamp"
touch "$STAMP"

rc_total=0
names=(); codes=()
run_leg() {
  local name="$1"; shift
  echo "=== $name ==="
  "$@"
  local rc=$?
  echo "=== $name rc=$rc ==="
  names+=("$name"); codes+=("$rc")
  [ $rc -ne 0 ] && rc_total=1
  return $rc
}

# cheapest first: the ladder reuses the staged slabs the full set generates
run_leg "full-legs-cdev" pixi run -e default python scripts/v2_m6_engine_peak.py \
  --configs cdev --k 10 --workdir-root "$TMPDIR" --out-suffix _cdev
run_leg "kladder-cdev" pixi run -e default python scripts/v2_m6_engine_peak.py \
  --configs cdev --k-ladder 5 10 --workdir-root "$TMPDIR" --out-suffix _kladder_cdev

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "CARDS WRITTEN BY THIS JOB:"
n=0
for f in runs/v2/m6_peak_cdev.json runs/v2/m6_peak_kladder_cdev.json; do
  if [ -f "$f" ] && [ "$f" -nt "$STAMP" ]; then echo "  $f"; n=$((n + 1)); fi
done
[ "$n" -eq 0 ] && { echo "  (none)"; echo "FATAL: zero fresh cards"; exit 1; }
rmdir "$TMPDIR/cdev" 2>/dev/null
echo "TMPDIR left for inspection if non-empty: $TMPDIR"
exit $rc_total
