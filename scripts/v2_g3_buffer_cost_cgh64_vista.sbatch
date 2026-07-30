#!/bin/bash
#SBATCH -J inexor-g3-bufcost-cgh64
#SBATCH -o %x-%j.log
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A JPL-SPHEREx
#SBATCH -t 12:00:00
#
# G3 pricing probe at cgh64: the ONLY config where sCOLA's own recommended
# geometry is non-degenerate. See scripts/v2_g3_buffer_cost.py's docstring for
# the two requirements (containment, force fidelity) and why the plan's
# residual-based buffer argument did not survive measurement.
#
# WHY cgh64 AND WHY THIS IS OWED. Leclercq 2003.04925 recommends tile >= 50 and
# buffer >= 25 Mpc/h. At the pinned 0.25 Mpc/h cell that is a padded tile of
# ~100 Mpc/h, which against cdev's 128 Mpc/h box is 1.3 tiles per side -- not a
# tiling. MEASURED: smoke (L=32) and cdev8 (L=64) both SATURATE, every geometry
# reaching a few-percent force error having padded_P ~ n_fine (cdev8's cheapest
# is T=128/b=48 -> P=240 of 256, 8 tiles each covering 82% of the box volume).
# cgh64's L=256 at the same cell is the first box where T=256 fine (64 Mpc/h)
# with b ~ 100 fine (25 Mpc/h) gives P ~ 456 of 1024 -- a real tiling, and the
# first place A3's compute factor is quotable rather than extrapolated.
#
# MEMORY FIT (affirmative). A gg node has 237 GB LPDDR (TACC Vista docs, Table 1);
# Slurm's RealMemory=1 here enforces nothing and FreeMem on a busy node reads
# ~85 GB, so neither is usable for a fit argument -- the doc figure is.
#   cgh64 mono evolve, n_fine=1024 f64   125.4 GB   MEASURED, job 861849's own
#                                                   peak_host_rss_bytes
#   frame q + Psi1 + Psi2 (512^3 x 3 x 8 B x 3)     9.7 GB   resident across it
#   x_ic, v_ic, plus the jnp evolve state          12.9 GB
#   post-evolve g_mono, disp, y_fin                 9.7 GB   after the peak
# Peak ~148 GB against 237 GB = 63% fill. No --mem flag: Vista does not enforce
# it (unlike deneb, where it is mandatory or the job pends forever).
#
# WALL. Mono evolve measured 937 s at 64 threads on gh (861849); the R2 leg is
# ~176 padded-box force solves with P from 32 to 640, which dominates at ~4-6 h.
# -t 12:00:00 matches G6c and is deliberately generous: billing is on actual use,
# and a tight wall only forces a resubmit of the expensive evolve.
#
# G5_CPU_THREADS=72 (ONE Grace socket), not 144, following G6c's reasoning
# verbatim: the wall calibration comes from 64-thread gh numbers, and a gg node
# is two NUMA domains, so first-touch pages on socket 0 with half the pool on
# socket 1 can lose more to remote bandwidth than 2x the cores gains on a
# bandwidth-bound f64 FFT. Untested either way; worth TIMING against this job's
# numbers rather than assuming.
#
# REQUIRES cea3942 OR LATER. Containment is vectorized there. The prior
# loop-over-tiles form is O(n_tiles x n_particles), which is merely slow at cdev
# but impossible at cgh64 (32768 tiles x 134M particles per row). Verified
# identical on 23/23 cdev8 rows, so this is a speedup and not a change of
# answer -- but running the old code here would simply never finish.
#
# NO `set -e`: independent legs, cheapest first (M1 job 11). The env is a
# PREREQUISITE, not a leg, so it aborts explicitly below -- deneb job 272 and
# Vista job 865844 both buried one env cause under one copy per leg.
#
# OUTPUT (runs/ is gitignored; pull it back explicitly):
#   runs/v2/g3_buffer_cost_cgh64.json
# Read: the largest T where force error reaches a few percent at a buffer that
# is NOT flagged degenerate/near_degenerate, and its vol_ratio -- that is A3's
# measured compute factor. Rows carrying `degenerate: true` have the tile solving
# the whole box and read 0.0000 by construction; they are not convergence.

cd ~/src/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export G5_CPU_THREADS=72
# The `gpu` feature declares system-requirements cuda = "12" and pixi enforces
# that at ENV ACTIVATION, before any python runs -- and a gg node has no GPU.
# Job 865844 died this way 11 times. Mocking the virtual package is pixi's own
# documented remedy and the right one here: `gpu` is the ONLY env installed on
# Vista, it is locked for linux-aarch64, and it is the env behind every rung this
# measurement will be compared against. The value asserts something about the
# MACHINE, not a request for a device.
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
  echo "!! env before spending the evolve. Diagnostic:"
  pixi run -e gpu python -c "import jax; print(jax.__version__, jax.devices())"
  exit 1
fi
pixi run -e gpu python -c "import jax; print('jax', jax.__version__, jax.devices())"

echo
echo "=== cgh64: containment + force fidelity ==="
pixi run -e gpu python scripts/v2_g3_buffer_cost.py --config cgh64

echo
echo "=== done ==="
date
