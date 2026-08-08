#!/bin/bash
#SBATCH -J inexor-v4f-t9-quantum
#SBATCH -o %x-%j.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A JPL-SPHEREx
#SBATCH -t 06:00:00
# V4f on Vista: is the RATIFIED 9 B/p state tier implementable at the fidelity
# it was gated at? Submit from ~/src/inexor:
#   sbatch scripts/v2_v4f_t9_quantum_vista.sbatch
#
# WHY VISTA AND NOT DENEB. Two reasons, one of them ratified.
#   1. D-v2-10 already called it: "the full C-dev G2c rerun (where t9/t12
#      actually separate at 2^17 levels) runs MONOLITHICALLY on a GH200 with no
#      new machinery." deneb's 6 GB card is what forced G2c's cdev8 fallback in
#      the first place, and that fallback is a contortion D-v2-10 deleted.
#      So there is NO cdev8 fallback in this job: 96 GB runs cdev outright, and
#      an OOM here would be a real finding rather than a known card limit.
#   2. deneb's queue is backed up behind a pico-prod array.
#
# This is an ACCURACY measurement, not a performance one, so D-v2-10's "no
# architecture decisions from deneb rows" rule is not what is at stake --
# running it on a GH200 is about capacity and turnaround, and the numbers would
# be equally valid on either card. Nothing here reads a wall clock.
#
# WHY IT EXISTS. v2_g2c_accum_gate.py:35 says the gate that ratified T9
# "measures REPRESENTATION error only (storage layout is a build decision)".
# V4 takes that decision and the arithmetic does not work: the gated t9 arm
# quantizes at fine_cell/256, which an int8 carries only if its bucket is ONE
# FINE CELL (verified -- t9's bucket is 0.25 Mpc/h, exactly the fine cell). At
# C-gh that is 4096^3 = 6.9e10 buckets against 8.6e9 particles, so the index
# costs ~69 GB, more than the 77 GB of state it indexes. CUBE's sorted-by-cell
# layout works because CUBE runs one particle per cell; our fine mesh is 2x the
# particle grid per side.
#
#   arm    bucket        quantum     index at C-gh   vs gated
#   t9     0.25 Mpc/h    fine/256    ~69 GB   <- BREAKS THE BUDGET (as gated)
#   t9c1   0.5  Mpc/h    fine/128    17.2 GB  2x coarser
#   t9c2   1.0  Mpc/h    fine/64      2.15 GB 4x coarser  <- the likely choice
#   t9c4   2.0  Mpc/h    fine/32      0.27 GB 8x coarser
#
# The velocity codec is int16 max-range in every arm, identical to t9, so the
# position quantum is the ONLY variable and the comparison is like-for-like.
#
# PRE-REGISTERED READING (written before the job ran).
#   Statistic: max |dP/P| in the D-v2-9 gate band (k <= 0.2 k_Nyq,fine) plus the
#   RSD pair dP0/P0 and dP2/P0, each vs the SAME-K ref arm -- the identical
#   statistics D-v2-8 ratified t9 on. Bar: D-v2-9's absolute 3e-2 in band.
#   Predicted: t9c1 and t9c2 both pass with room. t9 passed by ~3 orders
#   (1.3e-4 vs a 1.3e-1 floor), and IF position error scaled linearly in the
#   quantum, 4x coarser would cost ~4x and still sit ~2 orders under.
#   BUT a CPU smoke at the `smoke` config (K=10) gave t9 6.78e-05 and t9c2
#   6.62e-04 -- about 10x for a 4x coarser quantum, STEEPER THAN LINEAR. That
#   is one tiny config at one K and is not the answer, but it is on record here
#   as the reason not to assume the linear scaling.
#   FALSIFIED IF: t9c2 misses the 3e-2 bar, or degrades faster than ~quadratic
#   in the quantum (which would put t9c4 nowhere and make even t9c1 tight).
#   Then the state tier is not 9 B/p at C-gh and the config table moves --
#   which is exactly why this runs BEFORE any decision record is drafted.
#   NOT self-ratifying: it reports numbers against an existing bar; whether the
#   tier changes is JC's call at the freeze.
#
# WRITES runs/v2/g2c_results.json, the same path the ratified G2c gate wrote.
# Per-arm state files are tagged by arm so nothing collides at the worker
# level, but that SUMMARY file ends up holding only this run's arms. The
# ratified G2c numbers live in docs/decisions.md and the worklogs, not there,
# so nothing authoritative is lost -- noted so a truncated file is not a
# mystery later.

set -e
cd ~/src/inexor

# REQUIRED: Vista's batch environment does not carry pixi on PATH. Job 896092
# omitted this line and died at the first pixi call with rc 127. (The sibling
# probe job failed the same way and, having no `set -e`, reported COMPLETED in
# 6 seconds having run nothing -- this script's `set -e` at least failed
# honestly.) The explicit check below turns a PATH problem into a one-line
# diagnosis instead of a 127.
export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null 2>&1 || {
  echo "FATAL: pixi not on PATH (looked in \$HOME/.pixi/bin)"
  exit 1
}

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=160
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache
export OMP_NUM_THREADS=1

echo "=== provenance ==="
git log --oneline -1
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
free -g | head -2
pixi run -e gpu python -c "import jax; print('jax', jax.__version__, jax.devices())"

echo
echo "=== V4f: T9 position-quantum ladder at C-dev (GH200, no cdev8 fallback) ==="
pixi run -e gpu python scripts/v2_g2c_accum_gate.py \
  --config cdev --arms t9,t9c1,t9c2,t9c4

echo
echo "=== DONE ==="
