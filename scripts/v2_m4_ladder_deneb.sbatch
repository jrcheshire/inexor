#!/bin/bash
#SBATCH -J inexor-m4-meshladder
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o %x-%j.log
#
# M-v2-4 leg 3: the MESH LADDER -- what does the f32 coarse mesh actually buy,
# measured rather than derived?
#
# WHY IT IS OWED. Same reason as floors: queued behind `cgh64-k10` in Vista
# 899070 and killed by the wall without starting. It replaces D-v2-16 clause 3's
# "derived, not measured" residency figures -- the same clause whose 12.9/103 GB
# turned out to be the f32 numbers while the shipped code ran f64 at 25.8/206.
#
# THREE ARMS, not two, and that is what keeps the reading honest:
#   f64_whole   the pre-M-v2-4 decode at f64   (what the engine used to pay)
#   f64_slab    the shipped decode at f64      (what it pays today)
#   f32_slab    the shipped decode at f32      (what the milestone buys)
# An f64-before vs f32-after comparison would credit the DTYPE with S6's slabbed
# decode, which has nothing to do with dtype. `f64_whole_over_f64_slab` is
# expected at ~1.0 and is the control; if it is not ~1.0 the milestone's headline
# ratio is contaminated.
#
# RUNGS 128 256 512 AND NO HIGHER, which is a change from the charter's
# 128..2048 and the reason is arithmetic, not caution. Extrapolating n^3 from the
# measured n=256 arm (1.880 GiB f64, 1.089 GiB f32, on a ~0.25 GiB interpreter
# baseline):
#
#     n=512    13.3 GiB f64 /  7.0 GiB f32     fits this node
#     n=1024  104.6 GiB f64 / 53.9 GiB f32     fits NO host we have
#     n=2048  834.8 GiB f64 / 429.8 GiB f32    out of reach entirely
#
# deneb is 56 GB, antares 124 GB, a Vista gh node ~116 GB. The leg reads RATIOS
# WITHIN a rung, so it needs BOTH arms alive at a rung to yield anything at all
# -- and the f64 arm is the one that dies first. A rung where only f32 survives
# produces no ratio, so the top rungs would cost wall and return nothing.
#
# The leg's stated expectation survives the truncation: `f64_slab/f32_slab` is
# predicted to approach 2.0 FROM BELOW as the fixed baseline dilutes, and the
# n=512 projection is already ~1.90. The claim is testable at 512. What is lost
# is the confirmation that it keeps approaching, and that is a real loss to
# record rather than paper over -- an out-of-core coarse mesh at C-hero scale is
# unmeasured by this leg either way.
#
# ONE SUBPROCESS PER (rung, arm), which the leg already does: ru_maxrss is a
# high-water mark that never resets, so arms measured in one process report a
# monotone rise and the last arm always reads highest. That defect is already on
# the record from this milestone's own memory ladder; the fix is in the leg.
#
# BUDGET: the leg is MEMORY-bound, not time-bound. Measured solve walls are
# 0.048 s at n=64, 0.196 s at n=256; even at n=512 the arithmetic is seconds and
# the wall is 9 subprocess JAX imports. 1 h is generous by a wide margin.

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
    sys.exit('FATAL: CPU backend')
" || { echo "FATAL: jax/GPU precondition failed"; exit 1; }

echo "--- host memory, against which the rung ceiling was chosen ---"
free -g | head -2

STAMP="$(mktemp)"; touch "$STAMP"

# --config is NOT decorative here even though the ladder is synthetic and reads
# no config: the card path is f"m4_gate_{args.config}_{args.leg}.json" and
# --config DEFAULTS TO smoke, so omitting it would silently overwrite the
# existing runs/v2/m4_gate_smoke_ladder.json. `cdev` matches what Vista 899070's
# sbatch asked for, so the card lands where the milestone expects it.
pixi run -e gpu python scripts/v2_m4_f32_mesh_gate.py \
    --config cdev --leg ladder --rungs 128 256 512
rc=$?

echo ""
echo "=== CARDS WRITTEN (this job only) ==="
n=0
while IFS= read -r f; do
  echo "  $f"; n=$((n+1))
done < <(find runs/v2 -maxdepth 1 -name 'm4_gate_cdev_ladder.json' -newer "$STAMP" 2>/dev/null)
[ "$n" -eq 0 ] && echo "  (none)"
rm -f "$STAMP"

echo ""
echo "ladder rc=${rc};  fresh cards: ${n}"
[ "$n" -eq 0 ] && { echo "FATAL: zero cards written by THIS job"; exit 1; }
exit $rc
