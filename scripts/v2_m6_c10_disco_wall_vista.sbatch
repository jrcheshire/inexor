#!/bin/bash
#SBATCH -J inexor-c10-disco-wall
#SBATCH -o %x-%j.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A JPL-SPHEREx
#SBATCH -t 00:40:00
# C10: DISCO-DJ's STEADY-STATE per-step wall on one GH200. Submit from
# ~/src/inexor:
#   sbatch scripts/v2_m6_c10_disco_wall_vista.sbatch
#
# Budget: <= 0.67 SU worst case at gh's 1 SU/node-hr, ~0.25 expected. The wall
# request is deliberately generous because a TIMEOUT bills the whole wall.
#
# WHY THIS EXISTS. M-v2-6 turned the project's cost function from bytes into
# SU, and SU is node-count x wall x rate. We own DISCO-DJ's CAPACITY (V4b: the
# forward ceiling on one GH200 sits between 512^3 and 640^3 at ~280 B/p) but we
# have never owned its WALL. V4b's `wall_forward_s` column was recorded as
# provenance only and says so.
#
# The gap matters because a fit through V4b's three usable rungs gives
#
#     10-step forward wall = 1.63 s + 0.0476 s per 1e6 particles
#
# (predicting the 384^3 rung to 1.4%), and that 1.63 s intercept is COMPILE. It
# is why the small rungs looked sublinear: 64^3 took 1.85 s and 128^3 took
# 1.73 s, which is a measurement of jit, not of gravity. Every per-particle
# cost anyone derives from that table inherits the contamination. This job
# replaces the FITTED intercept with a MEASURED one.
#
# METHOD. The same ladder at two step counts. With wall(K) linear in K,
#
#     steady per step = (wall_K40 - wall_K10) / 30
#     compile         =  wall_K10 - 10 x steady
#
# so compile is solved for rather than assumed, and the two K legs at the same
# n are a consistency check on the linearity the subtraction depends on. Three
# n values test whether steady-per-step is linear in N across 8x in particle
# count; if it is not, the slope is the result and the intercept is a nuisance.
#
# WHAT IT IS NOT, and this is load-bearing. It is NOT a wall comparison against
# inexor. Our tiled force runs far above its FFT floor with ~88% of a tile in
# host plumbing, so timing the two codes head to head today measures our probe.
# This job characterizes ONE side, so that the reference implementation's cost
# per particle per step is a number we own rather than one we fit.
#
# It is also NOT multi-node, and cannot be. discodj at upstream HEAD (a687972,
# 2026-05-16, the same commit our env pins) contains no `jax.distributed`, no
# `NamedSharding`, no `shard_map`, no `pmap` and no `Mesh(`; every transform is
# a plain `jnp.fft.rfftn`. The package ships no distribution machinery, so its
# ceiling is one device. Any 2048^3 DISCO-DJ run is a caller-side construction
# on top of it and is out of scope for this project.
#
# MATCHING is inherited verbatim from V4b, where each choice is argued:
#   - `--forward-only`. v2 is a forward engine; the adjoint caps the ladder ~3x
#     lower and is not what we compete on.
#   - `--pm-ratio 2`, so res_pm = 2 x n_part per side, matching the config
#     table's n_fine = 2 x n_part. pm_ratio=1 would hand DISCO-DJ an 8x smaller
#     mesh than the arm it is compared against.
#   - f32, DISCO-DJ's default and its favourable case. The risk to us is the
#     reference being CHEAP, so it is run in the configuration most favourable
#     to it and we take the answer.
#
# LADDER CHOICE. 256/384/512 only. Below 256^3 the compile term dominates the
# signal (see the 64^3 vs 128^3 inversion above) and above 512^3 V4b already
# measured the OOM. Climbing into a known ceiling would spend wall to rediscover
# a result we own.
#
# LEG 1 IS A GATE (D-v2-13's pattern): DISCO-DJ lives in the disco-mocks pixi
# env, not ours. If it will not import, nothing downstream can mean anything.
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
  echo "Aborting before the ladder."
  exit 1
fi

# NB the compilation cache is shared across the two K legs deliberately. If it
# warms, K=10's compile term shrinks toward zero and the SUBTRACTION still
# returns the right steady-state slope -- the intercept is what degrades, and
# it is reported rather than gated. Disable the cache export above to measure a
# cold compile at both K.

rc_all=0
for K in 10 40; do
  echo
  echo "=== leg 2.K${K}: forward-only, pm_ratio=2, f32, steps=${K} ==="
  pixi run -e gpu python ~/src/inexor/scripts/m2_p1_disco_mem.py \
    --n 256,384,512 \
    --steps "${K}" \
    --forward-only \
    --pm-ratio 2 \
    --tag "c10_wall_k${K}"
  rc=$?
  echo "--- leg 2.K${K} rc=${rc} ---"
  [ $rc -ne 0 ] && rc_all=1
done

echo
echo "=== LEG SUMMARY ==="
ls -la ~/src/inexor/runs/m2/p1_disco_mem_c10_wall_k*.json 2>/dev/null \
  || echo "WARNING: no cards written"
n_cards=$(ls ~/src/inexor/runs/m2/p1_disco_mem_c10_wall_k*.json 2>/dev/null | wc -l)
echo "cards written: ${n_cards}"
if [ "${n_cards}" -eq 0 ]; then
  echo "FATAL: zero cards. The job ran no usable science."
  exit 1
fi

echo "=== DONE (rc_all=${rc_all}) ==="
