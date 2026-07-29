#!/bin/bash
#SBATCH --job-name=g3-bufcost
#SBATCH --output=inexor-g3-bufcost-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#
# G3 pricing probe at cdev: what buffer does an sCOLA tile need, and what does
# A3 cost? See scripts/v2_g3_buffer_cost.py's docstring for the two requirements
# (containment and force fidelity) and why the plan's residual-based argument
# does not survive measurement.
#
# WHY cdev AND NOT cdev8/smoke. Both smaller configs SATURATE: every geometry
# that reaches a few-percent force error has padded_P ~ n_fine, i.e. the "tile"
# is solving most of the box, so its error is small by construction rather than
# by convergence. MEASURED at cdev8 (L=64): the cheapest few-percent point is
# T=128/b=48 -> P=240 of 256, 8 tiles each covering 82% of the box volume. No
# cost factor can be read off that. cdev's L=128 at the same 0.25 Mpc/h cell
# doubles the box side at fixed resolution, which is exactly the lever needed.
# cgh64 (L=256) is still the only config that reaches Leclercq's recommended
# tile >= 50 / buffer >= 25 Mpc/h non-degenerately; that leg is Vista's.
#
# MEMORY FIT (affirmative, not hopeful). cdev is n_part=256, n_fine=512, f64:
#   force_global peak      delta 1.07 GB + rfftn half-grid complex128 1.08 GB
#                          + one g component at a time 1.07 GB
#                          + 3 half-grid f64 kernels 1.61 GB      ~= 4.9 GB
#   persistent host arrays x_fin, q, psi1, psi2, g_mono, disp, y_fin
#                          = 7 x (256^3 x 3 x 8 B) = 7 x 0.40 GB  ~= 2.8 GB
#   padded tile solves     P <= 512, so bounded by the same 4.9 GB
# Peak ~10-12 GB against deneb's 56 GB allocatable. --mem=48G is deliberately
# generous rather than tight: the partition default (124G) exceeds the node and
# pends forever (M1 job 9's lesson), so the flag is mandatory, and an
# over-request costs nothing here because billing is on actual use.
#
# -c 64 matches the cgroup to XLA-CPU's machine-sized spin-waiting thread pool
# (umbrella reference-xla-cpu-thread-pool). This is also the first exercise of
# the -c 64 fix from e38c4fe, which had been recorded as owed on a g5b rerun.
#
# NO `set -e`, deliberately: the legs are independent and the cheapest
# (containment, pure arithmetic) runs first, so a failure in the expensive
# force-fidelity leg must not discard it. M1 job 11's lesson.

cd ~/src/inexor || exit 1

# pixi is NOT on a batch shell's PATH on deneb -- a login shell picks it up but
# sbatch does not. Job 272 died on `pixi: command not found` twice in a row for
# exactly this. v2_g6_deneb.sbatch:44 carries the same line.
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export G5_CPU_THREADS=64
export OMP_NUM_THREADS=64
export JAX_PLATFORMS=cpu

echo "=== env ==="
date
hostname
git log --oneline -1
free -g | head -2
nproc

# PREREQUISITE GUARD, despite the deliberate absence of `set -e` above. The
# no-set -e rule is about independent physics LEGS; the env is a prerequisite,
# not a leg. Job 272 let a broken env through and buried one cause under two
# identical copies of the same error -- the same shape as Vista job 865844,
# which buried it under eleven. A repeat now exits in seconds with the real
# diagnostic instead of a log that has to be read backwards.
if ! command -v pixi >/dev/null 2>&1; then
    echo "FATAL: pixi not on PATH after adding ~/.pixi/bin -- aborting before any leg"
    exit 1
fi
if ! pixi run python -c "import jax; print('jax', jax.__version__); print(jax.devices())"; then
    echo "FATAL: the pixi env cannot import jax -- aborting before any leg"
    exit 1
fi

echo
echo "=== cdev: containment + force fidelity ==="
pixi run python scripts/v2_g3_buffer_cost.py --config cdev

echo
echo "=== done ==="
date
