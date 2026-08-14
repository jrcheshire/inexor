#!/bin/bash
#SBATCH -J m6-cgh64-confirm
#SBATCH -p compute
#SBATCH --mem=100G
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -o m6-cgh64-confirm-%j.log
# The confirmation run for TWO changes, against job 455 as the before-picture.
#
# Deliberately IDENTICAL to job 455 in config, knobs, legs, machine and core
# count. antares rather than deneb, and that costs wall time on purpose: 455 ran
# there, peak comparisons are not valid across machines, and every peak on
# record is antares. A faster box would have made the wall look better and the
# memory number unusable.
#
# BEFORE (job 455, commit f25f311): cgh64 step 2622.57 s/step, peak 14.447 GB.
#
# PRE-REGISTERED, both falsifiable, written before submission:
#
#   A. MEMORY. Per-brick velocity scales (8179cec) delete `kick_pending`, which
#      `python -m inexor.plan --preset cgh64` prices at 4.295 GB here. So the
#      peak should read ~10.1 GB against 455's 14.447. Falsified outside a
#      3-5 GB drop: less means the array is still resident somewhere, more means
#      something else moved too and the attribution is not clean.
#
#   B. WALL. Grouping the slab's rows by brick (10d1a2d) removed an N x nb^2
#      scan. Job 456/457 measured that function alone at 256^3/nb=32 going
#      39.626 -> 1.951 s with staging depth pinned at 1; scaling the keeper part
#      by this config's 8x particles puts a FLOOR of ~250 s on the saving, and
#      job 455 ran at depth 2-3, where the immigrant half of the scan is several
#      times larger again. So: a drop of at least 250 s, and I expect well over.
#      **Falsified below 150 s.**
#
# WHAT B WOULD FALSIFY, stated plainly because it is a gap in my own reasoning.
# The N x nb^2 derivation predicted the cdev -> cgh64 wall ratio to 2% (42x
# against 43x), and that agreement is what made it credible. But it was never
# checked against a phase breakdown of job 455, so "insert explains the 43x"
# assumes insert was the DOMINANT phase of that step and nothing measured it.
# The step is a whole engine step -- force, kick, drift, migrate, repack -- not
# just the migration. If the wall barely moves, the scan was real (456/457
# measured it directly) but minor at this configuration, and the 43x needs a
# different explanation. A phase breakdown is the follow-up in that case, and it
# is deliberately NOT added here: changing the instrument would break the
# like-for-like against 455, which is the whole point of this job.
#
# Slabs are REGENERATED rather than reused, and would have to be: the slab
# schema went t9-slabs-1 -> -2 with the per-brick scales, so 455's staged files
# are refused by the loader. That refusal firing is itself the schema bump
# working as intended.
#
# Legs, cheapest first, smoke gating:
#   1 smoke          orchestrator end to end, seconds
#   2 cgh64 build    gen + load; also the first end-to-end exercise of the new
#                    slab schema at this scale
#   3 cgh64 step K=3 the two numbers this job exists for
#
# ~1 h expected if B holds, 4 h requested because 455 took 2.2 h and my last two
# wall estimates were low. Zero SU. 8 of 28 cores. Nothing touches /work.
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

export TMPDIR="${TMPDIR:-/tmp}/m6_cgh64confirm_$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
echo "=== $(hostname): $(nproc) cores, commit $(git rev-parse --short HEAD) ==="
free -g | head -2

# SIZED FROM THE REQUIREMENT, not rounded up to a comfortable number. The IC
# stage writes ~20 full-grid f32 files at 512^3 (0.537 GB each = 10.7 GB) plus
# ~1.3 GB of T9 slabs, so ~12-13 GB peak; 20 leaves real headroom. The previous
# guard said 40 for no reason I can reconstruct, and job 458 refused on it with
# 33 GB free -- correctly by its own terms, and needlessly.
avail=$(df -BG --output=avail "$TMPDIR" | tail -1 | tr -dc '0-9')
echo "scratch available under TMPDIR: ${avail} GB (need ~13, guard at 20)"
[ "${avail:-0}" -lt 20 ] && { echo "FATAL: want >=20 GB of scratch, have ${avail}"; exit 1; }

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
KNOBS="--slack 0.20 --arena-frac 0.20"   # EXACTLY job 455's, or neither number compares

run_leg "smoke" pixi run -e default python $PEAK \
  --configs smoke --k 3 --legs gen load step $KNOBS \
  --workdir-root "$TMPDIR" --out-suffix _cgh64confirm_smoke
[ "${codes[0]}" -ne 0 ] && {
  echo "FATAL: smoke failed; a broken probe would burn both long legs (job 451)"
  exit 1
}

run_leg "cgh64-build" pixi run -e default python $PEAK \
  --configs cgh64 --legs gen load $KNOBS \
  --workdir-root "$TMPDIR" --out-suffix _cgh64confirm_build

run_leg "cgh64-step-k3" pixi run -e default python $PEAK \
  --configs cgh64 --k 3 --legs step $KNOBS \
  --workdir-root "$TMPDIR" --out-suffix _cgh64confirm_step

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "AGAINST JOB 455: step 2622.57 s/step, peak 14.447 GB"
echo "CARDS WRITTEN BY THIS JOB:"
n=0
for f in runs/v2/m6_peak_cgh64confirm_*.json; do
  if [ -f "$f" ] && [ "$f" -nt "$STAMP" ]; then echo "  $f"; n=$((n + 1)); fi
done
[ "$n" -eq 0 ] && { echo "  (none)"; echo "FATAL: zero fresh cards"; exit 1; }
echo "($n fresh cards)"

# CLEAN ON SUCCESS, KEEP ON FAILURE. Every M-v2-6 sbatch before this one ended
# with an unconditional "TMPDIR left for inspection", so every GREEN run leaked
# its staging: 19 directories and ~33 GB had accumulated on antares by
# 2026-08-14, and since /tmp there is on the ROOT filesystem that is a node
# problem rather than a job problem. Job 458 refused to start on its own scratch
# guard because of it.
#
# Inspection is only ever wanted when something went wrong, and that is exactly
# when this keeps it. Deletion is by `find -delete` over a path that carries the
# job id, never a recursive blast at a variable that might be empty: files
# first, then empty directories, so the blast radius is bounded by construction.
if [ $rc_total -eq 0 ] && [ -n "$SLURM_JOB_ID" ] && [ -d "$TMPDIR" ]; then
  before=$(du -sh "$TMPDIR" 2>/dev/null | cut -f1)
  find "$TMPDIR" -type f -delete
  find "$TMPDIR" -depth -type d -empty -delete
  echo "TMPDIR cleaned on success (was $before): $TMPDIR"
else
  echo "TMPDIR KEPT for inspection (rc=$rc_total): $TMPDIR"
fi
exit $rc_total
