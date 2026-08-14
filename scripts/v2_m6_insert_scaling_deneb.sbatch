#!/bin/bash
#SBATCH -J m6-insert-scaling
#SBATCH -p compute
#SBATCH -w deneb
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH -t 03:00:00
#SBATCH -o m6-insert-scaling-%j.log
# Does the migration's write-back cost N x n_bricks^2? A ONE-AXIS test.
#
# Job 455 measured cgh64 at 2622.6 s/step against cdev's 61 -- 43x the wall for
# 8x the particles, and 3.3x past the band that job pre-registered, whose own
# terms say REDO the reasoning rather than scale the number. Reading
# `SlotState._insert_slab` afterwards gives a candidate cause: every brick scans
# every row of its slab, so the comparisons come to nb slabs x nb^2 bricks x
# N/nb rows = N x nb^2, which is N^(5/3). Putting cdev and cgh64 in predicts
# 42x against the measured 43x.
#
# That agreement is why this is a measurement and not a patch. One matching
# ratio is not attribution -- this project's record is five wrong causes
# proposed and three shipped before one was measured first -- and the config
# ladder cannot settle it, because particles, bricks and coarse cells all move
# together on it (measured 2026-08-13: 64.00x / 64.00x / 61.18x smoke -> cdev8,
# degenerate by construction).
#
# PLACEMENT, and it is a deliberate exception. `-w` is normally forbidden here
# in favour of expressing the real resource, but "the faster cores" has no
# resource expression: the engine step is SERIAL (deneb 433 measured a 0.6%
# spread over 1-32 threads), so per-core speed is the only variable that matters
# and the machine has to be named. JC asked for deneb explicitly. 8 of 64 cores
# and 16 GB of 56, so the box stays usable.
#
# PRE-REGISTERED, before the run, and evaluated by the script rather than by eye:
#   - ARM A (particles FIXED at 256^3, nb 8 -> 16 -> 32): insert time QUADRUPLES
#     per doubling of nb. Acceptance 2.5-6.0x per step. Under 2.5 the term is not
#     quadratic in nb and the job-455 attribution is WRONG. Over 6.0 something
#     steeper than nb^2 is present and the model is incomplete.
#   - ARM B (control, nb FIXED at 16, n_part 128 -> 192 -> 256): insert time is
#     LINEAR in N, within 0.7-1.4x of the particle ratio. Superlinear here would
#     mean the cost is not the per-brick scan over slab rows, and arm A's reading
#     does not survive it.
#   - KNOB PROOF, and the run is VOID without it: `brick_reach` reads 1 at every
#     rung. Reach is ceil(|c| * s * INT16_MAX / (box/nb)), so at a fixed drift it
#     grows LINEARLY with nb -- an unpinned arm would measure nb^2 x reach and
#     could not separate them. The drift is SOLVED per rung from that bound, not
#     tuned, and the realized reach is on every card.
#   - `eject` timed beside `insert` as a second control: it decodes each brick
#     once, so it is linear in N and flat in nb. If it moves with nb, the harness
#     is measuring something other than what it names.
#
# Legs, cheapest first, smoke gating:
#   1 smoke        both ladders' code paths end to end, seconds
#   2 bricks       ARM A -- the claim
#   3 particles    ARM B -- the control
#
# ~20 min expected, 3 h requested. The last estimate I made was 3.3x low
# (job 455), so this one is deliberately generous; a timeout bills nothing here
# but discards every leg. Zero SU. Nothing touches /work.
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

echo "=== $(hostname): $(nproc) cores, commit $(git rev-parse --short HEAD) ==="
free -g | head -2
STAMP="/tmp/m6_insert_stamp_$SLURM_JOB_ID"; touch "$STAMP"

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

PROBE="scripts/v2_m6_insert_scaling.py"

run_leg "smoke" pixi run -e default python $PROBE --arm smoke --out-suffix _smoke
[ "${codes[0]}" -ne 0 ] && {
  echo "FATAL: smoke failed; a broken probe would burn both arms (job 451)"
  exit 1
}

run_leg "bricks" pixi run -e default python $PROBE --arm bricks --repeats 3 --out-suffix _bricks
run_leg "particles" pixi run -e default python $PROBE --arm particles --repeats 3 \
  --out-suffix _particles

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "CARDS WRITTEN BY THIS JOB:"
n=0
for f in runs/v2/m6_insert_scaling_*.json; do
  if [ -f "$f" ] && [ "$f" -nt "$STAMP" ]; then echo "  $f"; n=$((n + 1)); fi
done
[ "$n" -eq 0 ] && { echo "  (none)"; echo "FATAL: zero fresh cards"; exit 1; }
echo "($n fresh cards)"
rm -f "$STAMP"
exit $rc_total
