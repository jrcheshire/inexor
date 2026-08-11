#!/bin/bash
#SBATCH -J m6-threads
#SBATCH -p compute
#SBATCH --mem=32G
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH -o m6-threads-%j.log
# M-v2-6 Stage 1a: does THREAD COUNT explain the deneb-beats-GH200 2x?
#
# THE ANOMALY. On the same engine at cdev, deneb reads 24.89 s/step (job 411)
# against a Vista gh node's 49.12 s/step (899070) -- a 6 GB desktop GPU node
# beating a GH200 by 2x on a GH200's own workload. It is the premise the whole
# CPU-vs-GPU pricing question rests on and nothing on record explains it.
#
# THE CHEAP HYPOTHESIS, and why it is plausible: the two readings did not match
# thread count. deneb ran `-c 8` with no OMP_NUM_THREADS export; every Vista job
# exports OMP_NUM_THREADS=72. The dominant op is a memory-bound gather rather
# than GEMM (measured locally: 2.71 mean cores of 16, where a matmul reaches
# 13.6), and XLA-CPU runs a SPIN-WAITING thread pool per process, so 72 threads
# can plausibly cost more than 8 on this workload rather than less. If that is
# the whole story, the "GH200 loses" claim evaporates before any TACC SU is spent
# on the three-arm pricing job.
#
# PRE-REGISTERED READING. If s/step is FLAT across the sweep, the env vars are
# not the knob and the next test is core count via the cgroup (separate `-c`
# jobs), not more of this. If s/step has an interior minimum, thread count is a
# real knob and Stage 1b must match it across arms rather than inherit each
# site's habit. Either outcome is publishable into the record; a flat read is
# not a null result, it relocates the variable.
#
# The sweep sets OMP/OPENBLAS/MKL together because the wall is split between
# host numpy and eager jnp and both honour different ones. `-c 32` is requested
# once so every point has the cores available to use or ignore.
# NO set -e, deliberately: each point is independent and a failure at one thread
# count must not delete the others.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
pixi run -e default python -c "
import jax
d = jax.devices()
print('jax', jax.__version__, d)
assert d[0].platform == 'cpu', 'expected the CPU backend in the default env'
" || { echo "FATAL: jax import/backend check failed"; exit 1; }

export TMPDIR="${TMPDIR:-/tmp}/m6_threads_$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
echo "=== $(hostname): $(nproc) cores visible, TMPDIR $TMPDIR ==="
STAMP="$TMPDIR/.stamp"
touch "$STAMP"

rc_total=0
names=(); codes=()
for T in 1 2 4 8 16 32; do
  export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
  echo "=== threads=$T ==="
  pixi run -e default python scripts/v2_m6_engine_peak.py \
    --configs cdev8 --k-ladder 5 --workdir-root "$TMPDIR" --out-suffix "_t${T}"
  rc=$?
  echo "=== threads=$T rc=$rc ==="
  names+=("t$T"); codes+=("$rc")
  [ $rc -ne 0 ] && rc_total=1
done

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "CARDS WRITTEN BY THIS JOB:"
n=0
for T in 1 2 4 8 16 32; do
  f="runs/v2/m6_peak_t${T}.json"
  if [ -f "$f" ] && [ "$f" -nt "$STAMP" ]; then echo "  $f"; n=$((n + 1)); fi
done
[ "$n" -eq 0 ] && { echo "  (none)"; echo "FATAL: zero fresh cards"; exit 1; }
echo "s/step by thread count:"
pixi run -e default python -c "
import glob, json
for f in sorted(glob.glob('runs/v2/m6_peak_t*.json')):
    d = json.load(open(f))
    kl = d.get('k_ladder') or {}
    for k, r in sorted((kl.get('rungs') or {}).items()):
        print(f\"  {f.split('_t')[-1][:-5]:>3} threads: {r['s_per_step']:.2f} s/step, \"
              f\"peak {r['peak'] / 1e9:.3f} GB\")
"
echo "TMPDIR left for inspection if non-empty: $TMPDIR"
exit $rc_total
