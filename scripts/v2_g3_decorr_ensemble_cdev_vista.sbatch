#!/bin/bash
#SBATCH -J inexor-g3-decorr-ens
#SBATCH -o %x-%A_%a.log
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A JPL-SPHEREx
#SBATCH -t 8:00:00
#SBATCH --array=1-8
#
# Is the T=128 advantage above seed scatter? An ensemble at cdev.
#
# THE QUESTION. Job 297 (the cdev decorrelation map, seed 0) found that the
# T=128 lead over T=64 at matched 3.38x cost fell from 1.84x at cdev8 to 1.29x
# at cdev. Neither pre-registered outcome fired: the triple did not converge,
# and T=128 does not still lead by ~2x. Before that 1.29x can pin a Stage 5
# geometry it has to be shown to exceed the realization scatter, and seed 0
# alone cannot show that.
#
# WHY WE SUSPECT THE SCATTER IS NOT SMALL. Grouped by ABSOLUTE buffer rather
# than by cost, seed 0's cdev map goes non-monotone in T: at b = 4 Mpc/h,
# k(r=0.5) = 0.4088 / 0.3670 / 0.4517 for T = 32 / 64 / 128. There is no
# mechanism that makes the middle tile size the worst of three at fixed absolute
# buffer, so that ~20% spread is most likely realization noise -- against a 29%
# effect we are trying to rank. G6 measured 100-500x shared-IC cancellation for
# P(k), but NOT for r(k) at fixed (T, b), so it does not transfer here.
#
# WHY THE CONFOUND CANNOT BE BROKEN INSIDE ONE BOX (read this before proposing
# a cleverer single-box design). At fixed cost, cost = (P/T)^3 fixes b/T, so
#     P/n_fine = T (1 + 2b/T) / n_fine   is EXACTLY proportional to T.
# Seed 0's 3.38x row confirms it: P/n_fine = 0.094 / 0.188 / 0.375, a clean
# doubling. Tile size and box fraction are the same variable at fixed cost. Only
# the box ladder separates them, which is what 297 bought and what the cgh64
# rung would extend. THIS JOB DOES NOT ADDRESS THE CONFOUND -- it measures the
# error bar on the ranking, which is a different and prerequisite question.
#
# WHAT IS RUN. Seeds 1-8 (seed 0 already exists from job 297), at cdev, over the
# SIX configurations forming the two cost-matched triples that Stage 5 could
# choose between:
#     cost 3.38:  T=32/b=8    T=64/b=16   T=128/b=32
#     cost 8.00:  T=32/b=16   T=64/b=32   T=128/b=64
# The 8.00 row is included deliberately: T=128/b=64 is the ONLY cdev config
# reaching k(r=0.5) ~ 0.9, and its T-lead there is LARGER (1.74x) than at 3.38x,
# so if Stage 5 wants that usable-k it is the operative row. The other four GRID
# entries are dropped -- at 4.4 h/seed the full grid would be 35 h of ensemble
# for rows no decision turns on.
#
# HOW TO READ IT. The comparison is PAIRED: rank the configs WITHIN each seed,
# then look at the distribution of the per-seed ratio k(r=0.5)|T=128 over
# k(r=0.5)|T=64 at matched cost. Pairing is the point -- comparing independent
# per-config means throws away exactly the shared-IC cancellation that makes 8
# seeds enough. Report the mean ratio and its scatter, NOT a ratio of means.
# If the per-seed ratio brackets 1.0, the cdev8 T-advantage did not survive the
# box and Stage 5 takes the cheapest geometry at its target usable-k.
#
# RESOURCE FIT (no doomed jobs).
#   Memory: peak is the MONOLITHIC arm at n_fine=512 -- f64 mesh 1.07 GB,
#   complex half-grid 1.08 GB, x+v state 0.81 GB. deneb job 282 and job 297 both
#   ran that arm to completion at --mem=48G. The largest padded solve in these
#   six configs is P=256, an eighth of the monolithic mesh volume, so the tiled
#   legs do not move the peak. Against a gg node's 237 GB (TACC Vista docs
#   Table 1; Slurm RealMemory=1 here reports and enforces nothing) this is under
#   3% fill. No --mem flag: Vista does not enforce it, unlike deneb where it is
#   mandatory or the job pends forever.
#   Wall: measured per-config at cdev by job 297 -- 1759 + 2165 + 568 + 1061 +
#   442 + 1103 = 7098 s, plus the monolithic build (~214 s evolve + IC/LPT), so
#   ~2.1 h/seed at deneb's 64 threads. 8 h requested: generous per house rule
#   (billed on actual use, and a tight wall only forces a resubmit of the
#   expensive monolithic arm), and it carries ~3.8x headroom against the
#   possibility that a Grace socket is slower per-core than deneb.
#   Placement: one seed per node as an ARRAY, not two seeds packed onto the two
#   sockets. Packing would halve the node-hours, but it puts two XLA-CPU
#   processes on one node and the 72-vs-144 thread question is still untested
#   (G6c declined it for the same reason); an array keeps every task in the
#   single-socket configuration every prior Vista rung was measured in.
#
# THREADS. OMP_NUM_THREADS, *not* G5_CPU_THREADS. The latter is read only by
# v2_g5_two_level_force.py:718 when it spawns workers, and this script runs the
# physics IN-PROCESS -- setting it here would configure nothing while XLA-CPU
# spun a machine-sized (144) spin-waiting pool on a job using 72. That is the
# same class of error as deneb's -c 32 strangling a 64-sized pool.
#
# NO `set -e`: independent seeds, and each writes its own card, so a failure in
# one config should not discard the ones already measured. The env is a
# PREREQUISITE, not a leg, so it aborts explicitly below (deneb 272, Vista
# 865844 both buried one env cause under one copy per leg).
#
# OUTPUT (runs/ is gitignored; pull it back explicitly):
#   runs/v2/g3_decorrelation_cdev_seed{1..8}.json
#   runs/v2/figs/g3_decorrelation_cdev_seed{1..8}.png
# One card per seed by design: the array is readable incrementally and a wall
# overrun loses only the task it hit.

SEED=${SLURM_ARRAY_TASK_ID}

cd ~/src/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=72

# The `gpu` feature declares system-requirements cuda = "12" and pixi enforces
# that at ENV ACTIVATION, before any python runs -- and a gg node has no GPU.
# Job 865844 died this way 11 times. Mocking the virtual package is pixi's own
# documented remedy: `gpu` is the ONLY env installed on Vista, it is locked for
# linux-aarch64, and it is the env behind every rung this will be compared
# against. The value asserts something about the MACHINE, not a device request.
export CONDA_OVERRIDE_CUDA=12.0

echo "=== env ==="
date
hostname
echo "seed ${SEED}"
git log --oneline -1
nproc
free -g | head -2

# Env install runs ON THE COMPUTE NODE (login-node pixi gets killed).
# --locked: fail loudly on a stale lock rather than silently re-solving.
pixi install --locked -e gpu
if ! pixi run -e gpu python -c "import jax" >/dev/null 2>&1; then
  echo "!! ABORT: the 'gpu' env will not activate/import on this node -- fix the"
  echo "!! env before spending the monolithic arm. Diagnostic:"
  pixi run -e gpu python -c "import jax; print(jax.__version__, jax.devices())"
  exit 1
fi
pixi run -e gpu python -c "import jax; print('jax', jax.__version__, jax.devices())"

echo
echo "=== cdev decorrelation map, seed ${SEED}, two cost-matched triples ==="
pixi run -e gpu python scripts/v2_g3_decorrelation_map.py \
  --config cdev --seed "${SEED}" \
  --grid 32:8,32:16,64:16,64:32,128:32,128:64 \
  --out-suffix "_seed${SEED}"

echo
echo "=== done ==="
date
