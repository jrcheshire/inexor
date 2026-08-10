#!/bin/bash
#SBATCH -J inexor-m4-meshladder
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o %x-%j.log
#
# M-v2-4 leg 3: the MESH LADDER -- what does the f32 coarse mesh actually buy,
# measured rather than derived?
#
# WHY IT IS OWED. It replaces D-v2-16 clause 3's "derived, not measured"
# residency figures -- the same clause whose 12.9/103 GB turned out to be the
# f32 numbers while the shipped code ran f64 at 25.8/206. It is the milestone's
# DELIVERABLE; the accuracy legs are its safety check.
#
# THIS FILE REPLACES A GPU VERSION THAT COULD NOT MEASURE THE QUANTITY, and the
# replacement is the whole point of the re-run, so it is written down here
# rather than left in a worklog. deneb 409 returned rc=0 over 5 of 9 dead arms
# and its surviving ratios were all exactly 1.000:
#
#   - `_rss_bytes` is `ru_maxrss`, which is HOST memory. On CUDA the coarse
#     force meshes live in VRAM, so the host high-water mark is set by the
#     interpreter baseline in every arm and is blind to the dtype under test.
#     Three arms read 1.21 GiB identically. An exactly-1.0 ratio is the tell.
#   - It then OOM'd deneb's 6 GB card at n=256 f64, because `--mem` sizes the
#     HOST while VRAM was the binding resource.
#   - The old precondition aborted on `platform == 'cpu'`. It was not merely
#     unnecessary: it FORBADE the only backend on which this leg means anything.
#
# So the GPU precondition is INVERTED below, and the `--gres=gpu:1` request is
# gone. A GPU backend here does not degrade the measurement, it voids it.
#
# ENV IS `default`, NOT `gpu`. conda-forge's jax in the default env is CPU-only
# by construction, so the backend cannot silently become CUDA on a node that
# happens to have a card; and being CPU-only it carries no
# `[system-requirements] cuda = "12"`, which is enforced at ACTIVATION and would
# otherwise need `CONDA_OVERRIDE_CUDA` on a GPU-less node (the failure that
# killed Vista 865844). It is the same jax 0.10.2 as the `gpu` env, so the
# provenance `jax_version` field stays commensurable with the milestone's other
# cards; `backend` will now read `cpu`, which is the evidence the fix took.
#
# PLACEMENT BY --mem, NOT -w. Both nodes in `compute` satisfy 32G: antares
# (28 c, 124 GB, no GPU) and deneb (64 c, 56 GB, gpu:1). Either runs this
# correctly now that the backend is pinned by the env rather than by the node.
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
# RUNGS 128 256 512 by default, now with the top rung MEASURED rather than
# projected (jobs 415/416, antares, f64_slab arm):
#
#     n=128     0.65 GiB f64 /  0.65 GiB f32    baseline-dominated, ratio 1.000
#     n=256     1.48 GiB f64 /  0.93 GiB f32    ratio 1.594 / 1.615
#     n=512    11.25 GiB f64 /  6.23 GiB f32    ratio 1.805 / 1.805
#     n=1024   ~90    GiB f64 / ~50    GiB f32  extrapolated x8 from n=512
#     n=2048   ~720   GiB f64 / ~400   GiB f32  out of reach entirely
#
# The n=1024 figure is 8x the MEASURED n=512 arm, not the older 104.6 GiB
# projected from n=256, and it is the number the --mem=110G override is sized
# against. antares has 124 GB and is the only node that can host it, so the
# override pins placement by memory rather than by name, as above.
#
# The leg reads RATIOS WITHIN a rung, so it needs BOTH arms alive at a rung to
# yield anything, and the f64 arm dies first. If n=1024's f64 arm does not fit,
# the content check below fails the job on a missing top rung -- which is the
# correct signal for a job whose whole purpose is that rung, and the surviving
# lower rungs land on their own card because of SUFFIX.
#
# WHAT THE MEASURED RUNGS DO AND DO NOT SETTLE. `f64_slab/f32_slab` was
# pre-registered to approach 2.0 FROM BELOW as the fixed interpreter baseline
# dilutes, and 1.000 -> 1.594 -> 1.805 is that. The n=1024 rung is what turns
# "approaching" into a trend with a fourth point; the predicted value is ~1.93.
# An out-of-core coarse mesh at C-hero scale is unmeasured by this leg at any
# rung, and the production-scale saving is this measurement carried up by
# arithmetic rather than a reading. Both limits belong in the record.
#
# ONE SUBPROCESS PER (rung, arm), which the leg already does: ru_maxrss is a
# high-water mark that never resets, so arms measured in one process report a
# monotone rise and the last arm always reads highest. It also means an OOM at
# one rung cannot harm the rungs below it -- a failed arm is recorded as a row
# and is a datum, not a defect.
#
# BUDGET: the leg is MEMORY-bound, not time-bound. Measured solve walls are
# 0.048 s at n=64, 0.196 s at n=256; even at n=512 the arithmetic is seconds and
# the wall is 9 subprocess JAX imports. 1 h is generous by a wide margin.

export PATH="$HOME/.pixi/bin:$PATH"
export OMP_NUM_THREADS=8

echo "=== host $(hostname)  job ${SLURM_JOB_ID}  $(date -Is) ==="

if ! command -v pixi >/dev/null 2>&1; then
    echo "FATAL: pixi not on PATH -- this job would run zero science and exit 0"
    exit 1
fi

cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
echo "commit: $(git rev-parse --short HEAD)  branch: $(git rev-parse --abbrev-ref HEAD)"

# INVERTED from the GPU version on purpose: a CUDA backend makes ru_maxrss blind
# to the dtype under test, so it voids this leg rather than degrading it.
pixi run -e default python -c "
import jax, sys
d = jax.devices()
print('jax', jax.__version__, 'devices', d)
if d[0].platform != 'cpu':
    sys.exit('FATAL: backend is %s, not cpu -- ru_maxrss cannot see a mesh in VRAM' % d[0].platform)
" || { echo "FATAL: jax/CPU precondition failed"; exit 1; }

echo "--- host memory, against which the rung ceiling was chosen ---"
free -g | head -2

# Overridable so the top-rung extension does not need a second copy of this file:
#   sbatch --mem=110G --export=ALL,RUNGS="512 1024",SUFFIX="_top" <this>
# 512 is carried into that run deliberately, as an overlap rung: it is measured
# here already, so a disagreement between the two jobs is visible rather than
# assumed away. SUFFIX keeps the extension on its own card so a rung that OOMs
# cannot destroy the card that measured.
RUNGS="${RUNGS:-128 256 512}"
SUFFIX="${SUFFIX:-}"
CARD="runs/v2/m4_gate_cdev_ladder${SUFFIX}.json"
echo "rungs: ${RUNGS}   card: ${CARD}"

STAMP="$(mktemp)"; touch "$STAMP"

# --config is NOT decorative here even though the ladder is synthetic and reads
# no config: the card path is f"m4_gate_{args.config}_{args.leg}.json" and
# --config DEFAULTS TO smoke, so omitting it would silently overwrite the
# existing runs/v2/m4_gate_smoke_ladder.json. `cdev` is where the milestone
# expects the card, and it is where deneb 409's void card sits -- overwriting
# that is intended, since a card whose every ratio reads 1.000 has no value to
# preserve and leaving it invites it being read as a measurement.
pixi run -e default python scripts/v2_m4_f32_mesh_gate.py \
    --config cdev --leg ladder --rungs ${RUNGS} --out-suffix "${SUFFIX}"
rc=$?

echo ""
echo "=== CARDS WRITTEN (this job only) ==="
n=0
while IFS= read -r f; do
  echo "  $f"; n=$((n+1))
done < <(find runs/v2 -maxdepth 1 -name "$(basename "$CARD")" -newer "$STAMP" 2>/dev/null)
[ "$n" -eq 0 ] && echo "  (none)"
rm -f "$STAMP"

[ "$n" -eq 0 ] && { echo "FATAL: zero cards written by THIS job"; exit 1; }

# A card that EXISTS is not a card that MEASURED. deneb 409 wrote this exact
# file with rc=0 while 5 of 9 arms were dead and every surviving ratio was
# exactly 1.000, so the file-exists check above passed on a void run.
#
# WHAT THE VOID SIGNATURE ACTUALLY IS, corrected after job 415. The first form
# of this check failed a rung whose ratio sat at 1.000000, and it failed 415 --
# a run that measured perfectly well -- on rung 128 alone. `ru_maxrss` is a
# high-water mark over the WHOLE process, so a rung reads 1.0 whenever the peak
# is set by the jax import and XLA compile footprint rather than by the arrays:
# at n=128 the dtype-sensitive payload is ~50 MB against a 0.65 GiB floor, and
# both arms hit the identical import-driven maximum. That is baseline
# domination, and it is physics, not blindness. 415's rungs read 1.000 / 1.594 /
# 1.805, which is the predicted approach to 2.0 from below with the smallest
# rung swamped.
#
# The 409 failure is distinguishable and the discriminator is the LARGEST rung,
# where the payload is largest relative to the fixed baseline. When the host
# cannot see the mesh at all, EVERY rung reads 1.0 including the top one. So the
# rule is: the top rung must separate, and not every rung may be pinned. That
# still fails 409 exactly as before. Relaxing "no rung at 1.0" to this is a
# correction to a check I specified wrong, not a loosened tolerance -- the
# science bar for this leg is D-v2-16 clause 3's residency claim, and nothing
# about it moved.
#
# The control is now asserted too, which the first form omitted entirely: the
# header says a `f64_whole_over_f64_slab` away from ~1.0 means the headline
# ratio is contaminated by S6's slabbed decode, and a stated invariant no
# consumer checks is decoration. 5% window; 415 read 0.9960 and 1.0002.
pixi run -e default python - "$CARD" ${RUNGS} <<'PY'
import json, sys

CTL_TOL = 0.05

card, rungs = sys.argv[1], sys.argv[2:]
d = json.load(open(card))
ratios = d.get("ratios") or {}
prov = d.get("provenance") or {}
bad = []

if prov.get("backend") != "cpu":
    bad.append("backend is %r, not 'cpu'" % prov.get("backend"))

separated = []
for n in rungs:
    r = ratios.get(n)
    if r is None:
        bad.append("rung %s has NO ratio (both arms must survive to yield one)" % n)
        continue
    dt = r.get("f64_slab_over_f32_slab")
    ctl = r.get("f64_whole_over_f64_slab")
    pinned = dt is not None and abs(dt - 1.0) < 1e-6
    print("  rung %-5s f64_slab/f32_slab = %s   control f64_whole/f64_slab = %s%s"
          % (n, "None" if dt is None else "%.4f" % dt,
             "None" if ctl is None else "%.4f" % ctl,
             "   (baseline-dominated)" if pinned else ""))
    if dt is None:
        bad.append("rung %s dtype ratio is null" % n)
        continue
    if not pinned:
        separated.append((n, dt))
        # the control only means something where the arms separate at all
        if ctl is not None and abs(ctl - 1.0) > CTL_TOL:
            bad.append("rung %s control f64_whole/f64_slab = %.4f, outside 1 +/- %.2f "
                       "-- the headline ratio is contaminated by the slabbed decode"
                       % (n, ctl, CTL_TOL))

if ratios and not separated:
    bad.append("EVERY rung is pinned at 1.000000 -- the 409 signature, host memory "
               "blind to a mesh that is not on the host")
elif separated:
    top = rungs[-1]
    if top not in [n for n, _ in separated]:
        # only reachable when the top rung HAS a ratio; an absent one is already
        # reported above, and calling that "pinned" would misdescribe it
        if ratios.get(top) is not None:
            bad.append("the top rung %s is pinned at 1.000000 -- it carries the largest "
                       "payload against the fixed baseline, so it is the rung that "
                       "cannot be baseline-dominated" % top)
    else:
        print("  separation at %d of %d rungs; top rung %s reads %.4f"
              % (len(separated), len(rungs), top,
                 dict(separated)[top]))

dead = [r for r in d.get("rows", []) if r.get("failed")]
if dead:
    print("  NOTE %d of %d arms failed (an OOM is a datum): %s"
          % (len(dead), len(d.get("rows", [])),
             ", ".join("n=%s/%s" % (r.get("n_coarse"), r.get("arm")) for r in dead)))

if bad:
    print("FATAL: card written but not measured:")
    for b in bad:
        print("  - " + b)
    sys.exit(1)
print("card content OK: %d rungs with live ratios, backend cpu" % len(rungs))
PY
card_rc=$?

echo ""
echo "ladder rc=${rc};  fresh cards: ${n};  content check rc=${card_rc}"
[ "$card_rc" -ne 0 ] && exit "$card_rc"
exit $rc
