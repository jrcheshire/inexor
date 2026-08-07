#!/bin/bash
#SBATCH -J inexor-v4b-disco-cap
#SBATCH -o %x-%j.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A JPL-SPHEREx
#SBATCH -t 02:00:00
# V4b: DISCO-DJ's FORWARD capacity ceiling on one GH200. Submit from
# ~/src/inexor:
#   sbatch scripts/v2_v4b_disco_capacity_vista.sbatch
#
# WHY THIS EXISTS. v2's claim is "maximum volume x resolution per node", and
# the only competitor number we own is an ADJOINT one at 128^3 (DISCO-DJ f32
# 437 B/p, f64 816, against inexor v1's 488). That measures the wrong thing
# twice over: v2 is a FORWARD engine, and matched-N bytes-per-particle is not
# the claim. The claim is the largest box each code fits on one card. This
# walks DISCO-DJ up until it dies and records where.
#
# WHAT IT IS NOT. It is NOT a wall comparison, and nothing here should be read
# as one. Our tiled force currently runs far above its FFT floor (see the V4a
# anatomy job), so timing it against DISCO-DJ's production path today would
# measure our probe's plumbing, not the architecture. `wall_forward_s` is
# recorded for provenance only.
#
# MATCHING, which is the part that decides whether the number means anything:
#   - FORWARD ONLY (`--forward-only`). The adjoint caps the ladder ~3x lower on
#     memory and is not what v2 competes on. Without this flag the ceiling
#     found would be DISCO-DJ's adjoint ceiling and the comparison would be
#     wrong in our favour.
#   - `--pm-ratio 2`, so res_pm = 2 x n_part per side, matching our config
#     table (C-dev 256^3 particles / 512^3 fine, C-gh 2048^3 / 4096^3).
#     pm_ratio=1 is the v1 setting and would give DISCO-DJ an 8x smaller mesh
#     than the arm it is being compared against.
#   - f32, DISCO-DJ's default precision. That is its favourable case and how
#     it would be run in production; handicapping it to f64 would be a
#     comparison we won by choosing the axis.
# The rung that OOMs IS the measurement, so the ladder stops climbing at the
# first failure and records whether it died of memory or of something else.
#
# LEG 1 IS A GATE (D-v2-13's pattern): DISCO-DJ lives in the disco-mocks pixi
# env, not ours, and has never been imported on Vista. If it will not import,
# the ladder cannot mean anything, so leg 1 aborts the job before it starts.
#
# NO `set -e` inside the ladder: an OOM at a rung is a result, not a crash.

cd ~/src/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache

echo "=== provenance ==="
git log --oneline -1
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo
echo "=== leg 1 (GATE): can DISCO-DJ import in the disco-mocks gpu env here? ==="
cd ~/src/disco-mocks || { echo "FATAL: ~/src/disco-mocks missing"; exit 1; }
pixi run -e gpu python -c "
import jax, discodj
from discodj import DiscoDJ
print('discodj OK      ', discodj.__file__)
print('jax             ', jax.__version__)
print('devices         ', jax.devices())
"
if [ $? -ne 0 ]; then
  echo "FATAL: DISCO-DJ does not import in disco-mocks -e gpu on this node."
  echo "The ladder is meaningless without it. Aborting before the expensive legs."
  exit 1
fi

echo
echo "=== leg 2: forward-only capacity ladder, pm_ratio=2, f32 ==="
# n_part per side; res_pm is twice each. Climbs until a rung dies.
pixi run -e gpu python ~/src/inexor/scripts/m2_p1_disco_mem.py \
  --n 64,128,256,384,512,640,768,1024 \
  --steps 10 \
  --forward-only \
  --pm-ratio 2 \
  --tag v4b_fwd_capacity

echo
echo "=== DONE ==="
