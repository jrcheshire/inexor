#!/bin/bash
#SBATCH -J m2-pinned-gh
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o m2-pinned-gh-%j.log
# The pinned-gather probe on the PRODUCTION fabric. deneb 398 measured the
# gather/copy/transfer split on an RTX 3050 over ~13 GB/s PCIe and its record
# says so: the transfer columns there are statements about PCIe, not about
# C-gh's NVLink-C2C (176-221 GB/s pinned, ~15x). This job re-runs the same two
# legs on a GH200 so the ffi-gather build item (D-v2-16 cl.6) is priced against
# the fabric it would actually run on.
#
# PRE-REGISTERED (before submission):
#   a. the transfer legs land in the 176-221 GB/s class D-v2-16 measured, so
#      the host gather (~12 GB/s, per-ROW bound) becomes >=90% of the staged
#      path -- the ffi case gets STRONGER on the fast fabric, not weaker.
#   b. the deneb sign flips: the pinned path is net FASTER here (on PCIe it
#      lost 20.5 vs 19.4 ms because the extra copy cost more than the faster
#      transfer saved; at ~15x fabric the record's own arithmetic says it wins).
#   c. the 6.9x-vs-18x discrepancy (deneb 398 vs 896408) resolves toward the
#      LOWER figure at T9: the gather is per-row bound, so the narrower record
#      cannot win MORE than f64. If 18x reappears here, 896408's derivation was
#      right and deneb's gather was the anomaly -- either way it is settled on
#      the fabric that matters.
#
# BUDGET: probe legs are ~2 min each on deneb; ~10 min expected end to end
# against a 30 min request. gh is 1.0 SU/node-hr: ~0.2 SU expected, 0.5 worst.
#
# OPS: pixi on PATH; CONDA_OVERRIDE_CUDA for the gpu env activation gate; NO
# JAX_PLATFORMS=cpu here -- this job exists to run on the GPU, and the
# precondition DEMONSTRATES the backend (an M-v2-1 gh leg once got read as GPU
# from a speedup alone). NO set -e: legs independent, preconditions abort.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$SCRATCH/src/inexor" 2>/dev/null || cd "$HOME/src/inexor" || {
  echo "FATAL: no inexor checkout"; exit 1; }
export CONDA_OVERRIDE_CUDA=12.0

echo "=== host $(hostname)  job ${SLURM_JOB_ID}  commit $(git rev-parse --short HEAD) ==="

pixi run -e gpu python -c "
import jax, sys
d = jax.devices()
print('jax', jax.__version__, 'devices', d)
if d[0].platform == 'cpu':
    sys.exit('FATAL: CPU backend -- this probe measures a GPU-only distinction')
try:
    print('memory kinds:', sorted(m.kind for m in d[0].addressable_memories()))
except Exception as e:
    print('memory kinds: unavailable:', e)
" || { echo "FATAL: jax/GPU precondition failed"; exit 1; }

rc=0
echo ""
echo "--- leg 1: brick-span gather at the ratified 4096-particle run ---"
pixi run -e gpu python scripts/v2_m2_pinned_gather.py \
    --n-rows 40000000 --runs 512 --run-len 4096 \
    --out runs/v2/m2_pinned_gather_gh_brickspan.json || rc=1

echo ""
echo "--- leg 2: the run-length control (pinned H2D is flat in run length; the gather is not) ---"
pixi run -e gpu python scripts/v2_m2_pinned_gather.py \
    --n-rows 40000000 --runs 32 --run-len 65536 \
    --out runs/v2/m2_pinned_gather_gh_longrun.json || rc=1

echo ""
echo "=== LEG SUMMARY ==="
ls -la runs/v2/m2_pinned_gather_gh_*.json 2>/dev/null || echo "  (no cards written)"
n=$(ls runs/v2/m2_pinned_gather_gh_*.json 2>/dev/null | wc -l)
echo "cards written: $n"
if [ "$n" -eq 0 ]; then
    echo "FATAL: zero legs produced a card"
    exit 1
fi
echo "exit rc=$rc"
exit $rc
