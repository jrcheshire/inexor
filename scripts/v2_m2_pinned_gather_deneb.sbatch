#!/bin/bash
#SBATCH --job-name=inexor-m2-pinned
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=inexor-m2-pinned-%j.log
#
# M-v2-2: does jax express the pinned host gather, or is an ffi extension owed?
#
# --mem IS MANDATORY on deneb, not boilerplate. The partition default is 124G
# against a 56G allocatable node, so a job without it PENDS FOREVER (M1 job 9 was
# scrapped for exactly this). 24G is generous for a probe whose largest array is
# ~1 GB.
#
# NO `set -e`: the legs are independent and cheapest-first, so an expected
# failure on one must not take the rest of the run with it (M1 job 11's lesson).
# The preconditions below ABORT deliberately, because the failure they catch is a
# job that runs zero science and reports COMPLETED 0:0 (Vista 896055).

export PATH="$HOME/.pixi/bin:$PATH"

echo "=== host $(hostname)  job ${SLURM_JOB_ID}  $(date -Is) ==="

if ! command -v pixi >/dev/null 2>&1; then
    echo "FATAL: pixi not on PATH -- this job would run zero science and exit 0"
    exit 1
fi

cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
echo "commit: $(git rev-parse --short HEAD)  branch: $(git rev-parse --abbrev-ref HEAD)"

# The backend must be DEMONSTRATED, not inferred. M-v2-1's gh leg recorded the
# jax version and no device, and the GPU was concluded from a 6.83x speedup --
# which is exactly how a CPU-only run gets read as a GPU one. A CPU run here
# would be worse than useless: the pinned-vs-pageable distinction it exists to
# measure does not exist on CPU, so it would return plausible numbers about
# nothing.
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
    --out runs/v2/m2_pinned_gather_brickspan.json || rc=1

echo ""
echo "--- leg 2: a LONGER run, to separate 'pinned is faster' from 'long runs are' ---"
# D-v2-16 clause 6 records pinned H2D as FLAT in run length while the gather is
# not, so a single run length cannot tell the two apart. This leg is the control
# for that, not an operating point.
pixi run -e gpu python scripts/v2_m2_pinned_gather.py \
    --n-rows 40000000 --runs 32 --run-len 65536 \
    --out runs/v2/m2_pinned_gather_longrun.json || rc=1

echo ""
echo "=== LEG SUMMARY ==="
ls -la runs/v2/m2_pinned_gather_*.json 2>/dev/null || echo "  (no cards written)"
n=$(ls runs/v2/m2_pinned_gather_*.json 2>/dev/null | wc -l)
echo "cards written: $n"
if [ "$n" -eq 0 ]; then
    echo "FATAL: zero legs produced a card"
    exit 1
fi
echo "exit rc=$rc  $(date -Is)"
exit $rc
