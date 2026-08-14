#!/bin/bash
#SBATCH -J m6-cgh64-size
#SBATCH -p compute
#SBATCH --mem=100G
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH -o m6-cgh64-size-%j.log
# SIZING ONLY, for the GPU-marginal-value test. Not a gate, not a memory number.
#
# The question this job serves: does the accelerator earn its place in the
# engine? The test JC approved is a single Vista `gh` node running the same
# configuration four ways -- GPU on/off crossed with the sub-block gather
# compiled/eager -- so the hardware question and our own bitwise contract stop
# being confounded. That job charges (gh = 1.0 SU/node-hr) and a TIMEOUT bills
# the whole wall, so its -t must come from a measured seconds-per-step and
# nothing here has ever produced one at this configuration.
#
# Why cgh64 (512^3 in L=256) and not the production configuration: cdev, cgh64
# and C-gh all share N/tile = 2,097,152 and padded tile side P = 320, so a
# per-tile timing measured here transfers to production directly. cgh64 is the
# smallest of the three carrying production's tile COUNT regime (64 tiles
# against cdev's 8), which is what an occupancy question needs. Answering
# "is the GPU worth it" does not require 2048^3, and this is why.
#
# Second purpose, equally the point: cgh64 has never run end to end through the
# CURRENT engine. Its last outing was job 897904, a force-parity leg at commit
# 32b22f2 -- before the radix migrate, the three engine fixes, the f32 coarse
# mesh, the streamed ICs and the capacity ladder. Finding out here costs zero SU
# and one antares slot; finding out on Vista costs a burned wall.
#
# KNOBS, and a warning about reading the peak this job reports:
#   --slack 0.20 --arena-frac 0.20. The arena is deliberately GENEROUS because a
#   sizing run must not die on the D-007 arena refusal; 0.08 is the recorded
#   anchor value and has never been exercised at this configuration. That makes
#   this job's peak NOT comparable to any recorded peak (different knobs, and
#   `slack` costs state bytes). Read the WALL from this job. The peak is a
#   sanity bracket only.
#
# PRE-REGISTERED, before the run:
#   - cgh64 lands between 100 and 800 s/step on antares at -c 8. Outside that,
#     the reasoning behind the Vista request is wrong rather than imprecise and
#     the sizing needs redoing, not scaling.
#   - Peak host RSS between 11 and 40 GB. The arithmetic floor from
#     `python -m inexor.plan --preset cgh64 --cap 5284492` is 11.199 GB, and the
#     measured-over-floor ratio at cdev is 1.87x (7.461 measured / 3.982 floor).
#     Under 11 GB means the probe is not measuring the engine.
#   - Three steps complete without the arena refusal. If they do not at
#     arena-frac 0.20, the four-arm job needs a layout knob decision first and
#     is not merely unsized.
#   - NOT pre-registered and deliberately not read here: anything about the GPU.
#     antares has no accelerator; this job establishes a wall, not a verdict.
#
# Legs, cheapest first, smoke gating (job 451 burned five long legs on a defect
# its own smoke leg had already caught):
#   1 smoke               orchestrator end to end, seconds.
#   2 cgh64 gen+load      does the configuration BUILD at all -- IC generation,
#                         slab staging, loader. Separated from the step leg so a
#                         generation failure is attributable and so leg 3 reuses
#                         the staged slabs (--workdir-root is kept, not deleted).
#   3 cgh64 step K=3      the number this job exists for. K=3 not 40: this is a
#                         steady-state per-step wall, and the peak's warm-up
#                         (which saturates around step 13 at cdev) is not being
#                         measured here.
#
# Budget ~1.5 h expected, 6 h requested, zero SU, antares (--mem=100G is the
# placement knob -- antares 124 GB against deneb-the-node's 56; never -w).
# 8 of 28 cores, so the box stays usable. Nothing touches /work.
# NO set -e: independent legs, per-leg exit codes -- except smoke, which gates.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
pixi run -e default python -c "
import jax, platform, sys
d = jax.devices()
print('jax', jax.__version__, d, platform.machine(), sys.platform)
assert d[0].platform == 'cpu', 'expected the CPU backend in the default env'
" || { echo "FATAL: jax import/backend check failed"; exit 1; }

export TMPDIR="${TMPDIR:-/tmp}/m6_cgh64size_$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
echo "=== $(hostname): $(nproc) cores, commit $(git rev-parse --short HEAD) ==="
free -g | head -2

# IC staging for 512^3 writes ~20 full-grid f32 files plus the T9 slabs. At this
# configuration that is ~11 GB of scratch; refuse early rather than die mid-leg
# with a partial manifest.
avail=$(df -BG --output=avail "$TMPDIR" | tail -1 | tr -dc '0-9')
echo "scratch available under TMPDIR: ${avail} GB"
[ "${avail:-0}" -lt 40 ] && { echo "FATAL: want >=40 GB of scratch, have ${avail}"; exit 1; }

STAMP="$TMPDIR/.stamp"; touch "$STAMP"

rc_total=0
names=(); codes=()
run_leg() {
  local name="$1"; shift
  echo "=== $name ==="
  local t0=$SECONDS
  "$@"; local rc=$?
  echo "=== $name rc=$rc after $((SECONDS - t0)) s ==="
  names+=("$name"); codes+=("$rc")
  [ $rc -ne 0 ] && rc_total=1
  return 0
}

PEAK="scripts/v2_m6_engine_peak.py"
KNOBS="--slack 0.20 --arena-frac 0.20"

run_leg "smoke" pixi run -e default python $PEAK \
  --configs smoke --k 3 --legs gen load step $KNOBS \
  --workdir-root "$TMPDIR" --out-suffix _cgh64size_smoke
[ "${codes[0]}" -ne 0 ] && {
  echo "FATAL: smoke failed; a broken probe would burn both long legs (job 451)"
  exit 1
}

# ---- does the configuration build at all on the current engine
run_leg "cgh64-build" pixi run -e default python $PEAK \
  --configs cgh64 --legs gen load $KNOBS \
  --workdir-root "$TMPDIR" --out-suffix _cgh64size_build

# ---- the wall this job exists for. Reuses the slabs leg 2 staged.
run_leg "cgh64-step-k3" pixi run -e default python $PEAK \
  --configs cgh64 --k 3 --legs step $KNOBS \
  --workdir-root "$TMPDIR" --out-suffix _cgh64size_step

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "CARDS WRITTEN BY THIS JOB:"
n=0
for f in runs/v2/m6_peak_cgh64size_*.json; do
  if [ -f "$f" ] && [ "$f" -nt "$STAMP" ]; then echo "  $f"; n=$((n + 1)); fi
done
[ "$n" -eq 0 ] && { echo "  (none)"; echo "FATAL: zero fresh cards"; exit 1; }
echo "($n fresh cards)"
echo "TMPDIR left for inspection: $TMPDIR"
exit $rc_total
