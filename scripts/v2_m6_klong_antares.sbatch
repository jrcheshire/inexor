#!/bin/bash
#SBATCH -J m6-klong
#SBATCH -p compute
#SBATCH --mem=100G
#SBATCH -c 8
#SBATCH -t 08:00:00
#SBATCH -o m6-klong-%j.log
# M-v2-6: WHERE the peak lands at the ratified cadence, measured rather than fitted.
#
# Job 450 killed the reading that made this necessary. The engine's peak does
# grow with K, but the growth is in the resident FLOOR (step-start RSS +108
# MB/step at cdev) while the within-step transient SHRINKS (-12 to -27), no
# phase's own increment trends above +17 against that, and 446's trim arm cuts
# the floor rise 3.5-4x. So it is largely glibc holding freed blocks, not the
# engine growing a working set.
#
# And at cdev it SATURATES: +146 MB/step over steps 1-8, +28 over 8-15, an 81%
# decay at 10 sigma, with the last five steps inside the 131 MB run-to-run
# sigma. The "+103 MB/step, linear, ~+4 GB at K=40" on record was a straight
# line fitted through a knee. cdev8 does the OPPOSITE and steepens (+19 -> +54),
# so one mechanism does not describe both rungs and neither ladder is long
# enough to say where either lands.
#
# The capacity question is whether a complete mock fits at D-v2-14's ratified
# K=40. Extrapolation has now been wrong twice. This job runs K=40.
#
# Legs, cheapest first, each independently useful:
#   1 smoke          the orchestrator end to end, seconds (447 and 448 both died
#                    here, in the print path, and both times it paid for itself)
#   2 cdev8 K=40     the config that STEEPENS, at the ratified cadence
#   3 cdev8 K=60     a second length, so the curve's SHAPE is measured and not
#                    assumed. Two points cannot distinguish a knee from a line;
#                    this is the whole lesson of the retraction above.
#   4 cdev8 K=40 trim  if trimming flattens the floor at long K, the growth is
#                    the allocator's and a malloc_trim at the step boundary
#                    bounds it. An INSTRUMENT arm, never an operating point.
#   5 cdev K=40      THE number the capacity claim rests on, at the anchor.
#
# Budget ~3.3 h. --mem=100G is placement: only antares (28 c, 124 GB) satisfies
# it, and every peak on record is antares. Nothing here touches /work.
# NO set -e, deliberately: independent legs, per-leg exit codes -- EXCEPT the
# smoke leg, which gates everything after it. Job 451's smoke caught the probe
# defect in leg 1 and the job spent every long leg on it anyway: the legs are
# independent of each other, not of the instrument they share.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
pixi run -e default python -c "
import jax, platform, sys
d = jax.devices()
print('jax', jax.__version__, d, platform.machine(), sys.platform)
assert d[0].platform == 'cpu', 'expected the CPU backend in the default env'
assert sys.platform == 'linux', 'the phase tracer needs procfs'
" || { echo "FATAL: jax import/backend/platform check failed"; exit 1; }

pixi run -e default python -c "
def kb(f):
    t = open('/proc/self/status').read()
    return int([l for l in t.splitlines() if l.startswith(f)][0].split()[1])
big = bytearray(200 * 1024 * 1024); big[::4096] = b'x' * (len(big) // 4096)
peak = kb('VmHWM:')
del big
open('/proc/self/clear_refs','w').write('5\n')
hwm, rss = kb('VmHWM:'), kb('VmRSS:')
print(f'VmHWM {peak/1024:.0f} MB before reset; after {hwm/1024:.0f} vs VmRSS {rss/1024:.0f} MB')
assert abs(hwm - rss) <= 4096, 'clear_refs did NOT restart VmHWM; the tracer would be blind'
assert peak > rss, 'VmHWM never rose above the resting set; the counter is not live'
" || { echo "FATAL: /proc/self/clear_refs does not reset VmHWM here"; exit 1; }

export TMPDIR="${TMPDIR:-/tmp}/m6_klong_$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
echo "=== $(hostname): $(nproc) cores, commit $(git rev-parse --short HEAD) ==="
free -g | head -2
STAMP="$TMPDIR/.stamp"; touch "$STAMP"

rc_total=0
names=(); codes=()
run_leg() {
  local name="$1"; shift
  echo "=== $name ==="
  "$@"; local rc=$?
  echo "=== $name rc=$rc ==="
  names+=("$name"); codes+=("$rc")
  [ $rc -ne 0 ] && rc_total=1
  return 0
}

TRACE="scripts/v2_m6_peak_trace.py"

run_leg "smoke" pixi run -e default python $TRACE \
  --configs smoke --k 5 --repeats 3 --arms trace control trim \
  --arena-frac 0.20 --workdir-root "$TMPDIR" --out-suffix _klong_smoke
[ "${codes[0]}" -ne 0 ] && {
  echo "FATAL: smoke failed; a broken probe would burn every long leg (job 451)"
  exit 1
}

# ---- cdev8 at the ratified cadence, and then half again as long. THREE arms at
# K=40 so gate A and gate B are evaluable at long K for the first time: every
# gate on record was read at K=5, and an instrument that is neutral over five
# steps has not been shown neutral over forty.
run_leg "cdev8-k40" pixi run -e default python $TRACE \
  --configs cdev8 --k 40 --repeats 3 --arms trace control \
  --workdir-root "$TMPDIR" --out-suffix _klong_cdev8_k40
run_leg "cdev8-k60" pixi run -e default python $TRACE \
  --configs cdev8 --k 60 --repeats 3 --arms trace \
  --workdir-root "$TMPDIR" --out-suffix _klong_cdev8_k60
run_leg "cdev8-k40-trim" pixi run -e default python $TRACE \
  --configs cdev8 --k 40 --repeats 3 --arms trim \
  --workdir-root "$TMPDIR" --out-suffix _klong_cdev8_k40_trim

# ---- the anchor at K=40, ~2.5 h and the reason for the job.
run_leg "cdev-k40" pixi run -e default python $TRACE \
  --configs cdev --k 40 --repeats 3 --arms trace \
  --workdir-root "$TMPDIR" --out-suffix _klong_cdev_k40

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "CARDS WRITTEN BY THIS JOB:"
n=0
for f in runs/v2/m6_peak_klong_*.json; do
  if [ -f "$f" ] && [ "$f" -nt "$STAMP" ]; then echo "  $f"; n=$((n + 1)); fi
done
[ "$n" -eq 0 ] && { echo "  (none)"; echo "FATAL: zero fresh cards"; exit 1; }
echo "($n fresh cards)"
echo "TMPDIR left for inspection: $TMPDIR"
exit $rc_total
