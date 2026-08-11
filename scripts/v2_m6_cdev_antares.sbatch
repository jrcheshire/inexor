#!/bin/bash
#SBATCH -J m6-cdev
#SBATCH -p compute
#SBATCH --mem=100G
#SBATCH -c 8
#SBATCH -t 02:30:00
#SBATCH -o m6-cdev-%j.log
# M-v2-6 Stage 0, closing it out on ONE glibc machine. Resubmit of job 431, which
# died on the ownership assertion at cdev (one particle of 16,777,216 owned by no
# tile) -- fixed at 031b3fc, ownership now decided from the global position in
# integers.
#
# --mem=100G is PLACEMENT expressed as a resource: only antares (28 c, 124 GB)
# satisfies it, deneb-the-node has 56 GB, and deneb is occupied anyway. The `-c 8`
# is honest about what the step uses: job 433 measured s/step FLAT across 1..32
# threads (12.43-12.51, a 0.6% spread), so the step is effectively serial and more
# cores buy nothing.
#
# WHY EVERYTHING RUNS HERE RATHER THAN SPLIT WITH THE LAPTOP. macOS retains freed
# buffers where glibc munmaps them (M-v2-5 leg V), and the same cdev8 K=20 point
# read 6.484 GB and 9.855 GB on this laptop minutes apart. So the before/after
# comparison for the shape ladder has to live on one machine, and that machine has
# to be the glibc one. Same reason the cdev8 ladder is repeated here even though
# the laptop already ran it.
#
# CPU env by construction (`default` is CPU-only jax), which the peak probe
# REQUIRES: ru_maxrss is HOST memory and on CUDA the meshes are in VRAM, where
# every ratio reads 1.0 (deneb 409).
# NO set -e, deliberately: independent legs, cheapest first, per-leg exit codes.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
pixi run -e default python -c "
import jax, platform, sys
d = jax.devices()
print('jax', jax.__version__, d, platform.machine(), sys.platform)
assert d[0].platform == 'cpu', 'expected the CPU backend in the default env'
" || { echo "FATAL: jax import/backend check failed"; exit 1; }

export TMPDIR="${TMPDIR:-/tmp}/m6_cdev_$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
echo "=== $(hostname): $(nproc) cores, commit $(git rev-parse --short HEAD) ==="
free -g | head -2
STAMP="$TMPDIR/.stamp"; touch "$STAMP"

rc_total=0
names=(); codes=()
run_leg() {
  local name="$1"; shift
  echo "=== $name ==="
  "$@"; local rc=$?
  echo "=== $name rc=$rc ==="
  names+=("$name"); codes+=("$rc")
  [ $rc -ne 0 ] && rc_total=1
  return 0
}

# ---- cheapest first, and the PARITY REGRESSIONS lead: if the ownership change
# broke the two arms' agreement, nothing downstream is worth reading. NB this is
# NOT a reproduction of the ratified cgh64 card (Vista 897904, 0 of 402,653,184):
# that is a different machine and a different backend, and bitwise parity is
# backend-specific. What these legs test is that the engine arm and the probe arm
# still agree WITH EACH OTHER after the ownership derivation changed.
run_leg "parity-cdev8" pixi run -e default python scripts/v2_m3_engine_gate.py \
  --config cdev8 --leg force --out-suffix _m6own
run_leg "parity-cdev" pixi run -e default python scripts/v2_m3_engine_gate.py \
  --config cdev --leg force --out-suffix _m6own

# ---- the shape-ladder before/after, on glibc, one machine
run_leg "kladder-cdev8" pixi run -e default python scripts/v2_m6_engine_peak.py \
  --configs cdev8 --k-ladder 5 10 20 --workdir-root "$TMPDIR" --out-suffix _kladder_cdev8_antares

# ---- the cdev anchor: the config where the terms separate (`pending` 537 MB
# against cdev8's 67 MB)
run_leg "full-legs-cdev" pixi run -e default python scripts/v2_m6_engine_peak.py \
  --configs cdev --k 5 --workdir-root "$TMPDIR" --out-suffix _cdev
run_leg "kladder-cdev" pixi run -e default python scripts/v2_m6_engine_peak.py \
  --configs cdev --k-ladder 5 10 --workdir-root "$TMPDIR" --out-suffix _kladder_cdev

# ---- REPEATS of one fixed point, so the run-to-run scatter is MEASURED rather
# than assumed. Without this the before/after peak deltas have no error bar and
# the laptop already showed 1.5x scatter is possible.
for r in 1 2 3; do
  run_leg "repeat-cdev-$r" pixi run -e default python scripts/v2_m6_engine_peak.py \
    --configs cdev --k-ladder 5 --workdir-root "$TMPDIR" --out-suffix "_rep${r}"
done

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "CARDS WRITTEN BY THIS JOB:"
n=0
for f in runs/v2/m3_gate_cdev8_force_m6own.json runs/v2/m3_gate_cdev_force_m6own.json \
         runs/v2/m6_peak_kladder_cdev8_antares.json runs/v2/m6_peak_cdev.json \
         runs/v2/m6_peak_kladder_cdev.json \
         runs/v2/m6_peak_rep1.json runs/v2/m6_peak_rep2.json runs/v2/m6_peak_rep3.json; do
  if [ -f "$f" ] && [ "$f" -nt "$STAMP" ]; then echo "  $f"; n=$((n + 1)); fi
done
[ "$n" -eq 0 ] && { echo "  (none)"; echo "FATAL: zero fresh cards"; exit 1; }
echo "($n fresh cards)"
echo "TMPDIR left for inspection: $TMPDIR"
exit $rc_total
