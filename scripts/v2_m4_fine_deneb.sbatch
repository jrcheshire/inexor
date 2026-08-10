#!/bin/bash
#SBATCH -J inexor-m4-fine
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o %x-%j.log
#
# M-v2-4 leg 5: the FINE arm. Measured, NOT gated.
#
# WHY IT IS OWED. Unlike floors and the ladder, this one was never queued
# anywhere -- not in Vista 899070, not on deneb. It has no failure to explain,
# only an absence.
#
# WHAT IT MEASURES. `--leg fine` moves the FINE dtype knob and holds coarse at
# f64, the mirror of leg 1. It is deliberately not gated: the fine mesh is P^3
# and box-independent, so it carries no capacity argument at C-gh or C-hero and
# cannot fail a budget-share bar that was never about it. What it tells M-v2-6
# is whether the per-tile working set can halve too -- and the per-tile arrays
# are the term that actually dominates cdev's residency (tile_workspace 1.344
# GiB + tile_kernels 0.737 GiB against 0.187 GiB for every coarse term combined).
#
# AT cdev K=40, matching leg 1's GATED reading (E = 4.002e-05, Vista 899070) so
# the two knobs are read at the same config and step count. Reading the fine arm
# at cdev8 instead would have been ~8x cheaper and would have answered a
# different question.
#
# NO PRE-REGISTERED EXPECTATION, and stating that plainly is better than
# inventing one after the fact. Leg 1's few x 1e-6 prediction was for the COARSE
# knob and does not transfer: the fine mesh carries the short-range force, which
# is the arm carrying most of the force, so there is no argument that its f32
# cost should resemble the coarse arm's. Whatever this reads is a first
# measurement.
#
# MEMORY 32G against a MEASURED 16.0 GiB peak: cdev leg 1 was run locally to
# steady state on 2026-08-09 and plateaued at 16.0 GiB over 14 min of stepping.
# That is a CPU-backend measurement of a host-side working set, so it transfers
# -- the arrays are the same shapes. 2x headroom, and it fits deneb's 56 GB with
# room rather than needing antares.
#
# BUDGET, and this is the one wall on the milestone I am least sure of. Vista
# measured cdev K=40 paired at 65.1 min (48.8 s/step). Deneb runs cdev8 2.2x
# faster than a Vista gh node, but that ratio was measured at cdev8 and the
# record is explicit that cost does NOT scale by particle count across configs
# -- tile-to-brick decode redundancy is 8x at cdev8 and ~2x at cdev. So the
# honest bracket is 30 min (Vista's wall / 2.2) to 47 min (the record's 35.61
# s/step at cdev). 4 h covers both with margin.
#
# If it is still running past ~1 h, that is the signal to look rather than wait:
# the bracket above is wrong and the reason is worth knowing before the wall
# decides. A generous wall is free ONLY on a job that completes.

export PATH="$HOME/.pixi/bin:$PATH"

echo "=== host $(hostname)  job ${SLURM_JOB_ID}  $(date -Is) ==="

if ! command -v pixi >/dev/null 2>&1; then
    echo "FATAL: pixi not on PATH -- this job would run zero science and exit 0"
    exit 1
fi

cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
echo "commit: $(git rev-parse --short HEAD)  branch: $(git rev-parse --abbrev-ref HEAD)"

pixi run -e gpu python -c "
import jax, sys
d = jax.devices()
print('jax', jax.__version__, 'devices', d)
if d[0].platform == 'cpu':
    sys.exit('FATAL: CPU backend -- leg 1 is on the GPU and leg 5 is read beside it')
" || { echo "FATAL: jax/GPU precondition failed"; exit 1; }

STAMP="$(mktemp)"; touch "$STAMP"

pixi run -e gpu python scripts/v2_m4_f32_mesh_gate.py \
    --config cdev --leg fine --k 40 --slack 0.20 --arena-frac 0.08
rc=$?

echo ""
echo "=== CARDS WRITTEN (this job only) ==="
n=0
while IFS= read -r f; do
  echo "  $f"; n=$((n+1))
done < <(find runs/v2 -maxdepth 1 -name 'm4_gate_cdev_fine.json' -newer "$STAMP" 2>/dev/null)
[ "$n" -eq 0 ] && echo "  (none)"
rm -f "$STAMP"

echo ""
echo "fine rc=${rc};  fresh cards: ${n}"
[ "$n" -eq 0 ] && { echo "FATAL: zero cards written by THIS job"; exit 1; }
exit $rc
