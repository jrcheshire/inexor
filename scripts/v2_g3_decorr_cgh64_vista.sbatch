#!/bin/bash
#SBATCH -J inexor-g3-decorr-cgh64
#SBATCH -o %x-%j.log
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A JPL-SPHEREx
#SBATCH -t 18:00:00
#
# The third rung: does the T=128 advantage survive another 8x of volume?
#
# THE QUESTION, AND WHY ONLY THE BOX LADDER CAN ANSWER IT. At matched cost the
# T=128 lead over T=64 in k(r=0.5) read 1.84x at cdev8 (job 296) and 1.29x at
# cdev (job 297). Neither of 297's pre-registered outcomes fired -- the triple
# did not converge, and T=128 does not still lead by ~2x. The reason a cleverer
# single-box design cannot settle it is arithmetic: at fixed cost,
# cost = (P/T)^3 fixes b/T, so
#     P/n_fine = T (1 + 2b/T) / n_fine   is EXACTLY proportional to T.
# Job 297's 3.38x row reads P/n_fine = 0.094 / 0.188 / 0.375, a clean doubling.
# Tile size and box fraction are ONE variable inside a box. cgh64 halves P/n_fine
# again at fixed absolute geometry (cdev8/cdev/cgh64 share the 0.25 Mpc/h fine
# cell exactly), so the same (T, b) is the same tile in Mpc/h at all three rungs.
#
# WHAT EACH OUTCOME MEANS (fixed before the run, so the readout is not post-hoc):
#   - Lead CONTINUES TO DECAY toward 1.0 (the two-point trend 0.84 -> 0.29 in
#     excess, a factor 0.34 per 8x volume, extrapolates to ~1.10x here): the
#     cdev8 T-advantage was box fraction throughout, and Stage 5 takes the
#     cheapest geometry that reaches its target usable-k.
#   - Lead HOLDS near 1.29x: tile size matters on its own and Stage 5 takes
#     T=128/b=32.
#   - Lead GROWS: neither story is right and the P/L law needs revisiting before
#     any geometry is pinned.
# The trend is computed on the T=128-over-T=64 ratio, but all three T are run:
# T=32 is what makes the row a curve rather than a pair, and it is now cheap.
#
# SEED 0 ONLY, DELIBERATELY. The companion cdev ensemble (seeds 1-8,
# v2_g3_decorr_ensemble_cdev_vista.sbatch) measures the scatter this ratio
# carries. If that ensemble shows the per-seed ratio brackets 1.0 at cdev, this
# rung's single realization cannot be read as a ranking either -- read the two
# together, and do not pin a Stage 5 geometry on this job alone.
#
# RESOURCE FIT (no doomed jobs).
#   Memory, all measured rather than estimated:
#     cgh64 mono evolve, n_fine=1024 f64   125.4 GB  job 861849 peak_host_rss_bytes
#     frame q + Psi1 + Psi2 (512^3 x3 x8B x3)  9.7 GB  resident across it
#     x_ic, v_ic, plus the jnp evolve state   12.9 GB
#     d_mono, an f64 1024^3 density            8.6 GB  this script, not the ladder
#     tile member arrays, worst config         4.1 GB  measured at cdev, x8
#   Peak ~160 GB against a gg node's 237 GB (TACC Vista docs Table 1) = 68% fill.
#   Slurm's RealMemory=1 here reports and enforces nothing and FreeMem on a busy
#   node reads ~85 GB, so neither is usable for a fit argument -- the doc figure
#   is. No --mem flag: Vista does not enforce it.
#   Wall: per-tile work is IDENTICAL across boxes at fixed (T, b) in fine cells --
#   same padded P, same solve -- so only the tile count changes, 8x with volume.
#   Job 297's cdev walls minus their (old, slow) membership cost give the physics:
#     T= 32/b= 8   1759 s - 395 s membership = 1364 s  -> x8 = 10912 s
#     T= 64/b=16    568 s -  49 s            =  519 s  -> x8 =  4152 s
#     T=128/b=32    442 s -   6 s            =  436 s  -> x8 =  3488 s
#   plus ~22 s membership per config now, plus the 937 s monolithic evolve
#   (measured, 861849, 64 threads on gh) and the IC/LPT build: ~5.7 h. 18 h
#   requested -- generous per house rule (billed on actual use, and a tight wall
#   only forces a resubmit of the expensive monolithic arm), covering the
#   possibility that a Grace socket is slower per-core than deneb.
#
# REQUIRES e31c570 OR LATER. Before it, lagrangian_tiles masked the whole
# particle array once per tile: O(n_tiles x n_particles), which is 7.0 h for the
# T=32 config ALONE at cgh64's 32768 tiles x 134M particles. Running this on an
# earlier commit would not fail, it would simply spend the wall on membership.
# The vectorized form is asserted bit-identical against the loop form in
# tests/test_lagrangian_tiles.py, so this rung stays comparable to cdev8/cdev.
#
# THREADS. OMP_NUM_THREADS, *not* G5_CPU_THREADS. The latter is read only by
# v2_g5_two_level_force.py:718 when it spawns workers, and this script runs the
# physics IN-PROCESS -- setting it here would configure nothing while XLA-CPU
# spun a machine-sized (144) spin-waiting pool on a job using 72. 72 = ONE Grace
# socket, following G6c: a gg node is two NUMA domains, and first-touch pages on
# socket 0 with half the pool on socket 1 can lose more to remote bandwidth than
# 2x the cores gains on a bandwidth-bound f64 FFT. Still untested either way;
# worth TIMING against this job's numbers rather than assuming.
#
# NO `set -e`: independent legs. The env is a PREREQUISITE, not a leg, so it
# aborts explicitly below -- deneb 272 and Vista 865844 both buried one env cause
# under one copy per leg.
#
# OUTPUT (runs/ is gitignored; pull it back explicitly):
#   runs/v2/g3_decorrelation_cgh64.json
#   runs/v2/figs/g3_decorrelation_cgh64.png
# Read: k(r=0.5) for the three configs, and the T=128-over-T=64 ratio against
# 1.84x (cdev8) and 1.29x (cdev).

cd ~/src/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=72

# The `gpu` feature declares system-requirements cuda = "12" and pixi enforces
# that at ENV ACTIVATION, before any python runs -- and a gg node has no GPU.
# Job 865844 died this way 11 times. Mocking the virtual package is pixi's own
# documented remedy: `gpu` is the ONLY env installed on Vista, it is locked for
# linux-aarch64, and it is the env behind every rung this is compared against.
# The value asserts something about the MACHINE, not a device request.
export CONDA_OVERRIDE_CUDA=12.0

echo "=== env ==="
date
hostname
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
echo "=== cgh64 decorrelation map, seed 0, the 3.38x cost-matched triple ==="
pixi run -e gpu python scripts/v2_g3_decorrelation_map.py \
  --config cgh64 --seed 0 --grid 32:8,64:16,128:32

echo
echo "=== done ==="
date
