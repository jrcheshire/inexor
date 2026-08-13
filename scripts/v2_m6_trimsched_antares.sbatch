#!/bin/bash
#SBATCH -J m6-trimsched
#SBATCH -p compute
#SBATCH --mem=100G
#SBATCH -c 8
#SBATCH -t 08:00:00
#SBATCH -o m6-trimsched-%j.log
# M-v2-6: does a LIGHT-TOUCH malloc_trim recover what the aggressive one does,
# and what does the ANCHOR retain at the ratified cadence?
#
# Job 452 measured `trim` at cdev8 K=40: peak 2.147 -> 1.276 GB (-41%), floor
# slope +21.0 -> +4.2 MB/step, at +15% wall. That is the number a
# promote-to-default decision was about to be made on, and it cannot carry it,
# for two measured reasons:
#
#   1. THE ARM TRIMS 263 TIMES PER STEP at cdev8 (10,521 calls over K=40), so it
#      bounds the most aggressive schedule available and prices nothing cheaper.
#      The proposed default is ONE call per step -- a factor of 263 away.
#   2. RETENTION SHRINKS WITH CONFIG SIZE (12% cdev K=5, 32% cdev8 K=5, 41%
#      cdev8 K=40) and GREW with K at cdev8, so neither the anchor's value at
#      K=40 nor C-gh's can be inferred from what exists. cdev trim has only ever
#      run at K=5.
#
# What makes the light touch plausible is already in 452's series and is the
# pre-registration below: in late steps the trim arm's resident floor sits 1.072
# GB under the trace arm's at the step's FIRST boundary and only 0.828 GB under
# at its last, and it climbs +293 MB WITHIN a step despite all 263 calls. So the
# mid-step calls do not hold the floor down; they slow its re-growth. If that
# reading is right, one call per step gets most of the benefit for ~none of the
# cost.
#
# PRE-REGISTERED, before the run:
#   - cdev8 K=40 `trim_step` peak lands between 1.276 (trim) and 2.147 (trace),
#     and CLOSER TO trim. Nearer to trace than to the midpoint (> ~1.71 GB)
#     falsifies the floor reading above and the light touch dies with it.
#   - `trim_step` wall cost is under 2%: 40 syscalls against `trim`'s 10,521 for
#     the same +15%.
#   - cdev K=40 retention is AT LEAST its K=5 value of 12%, since retention grew
#     with K at cdev8. Lower would mean K and config size interact and neither
#     axis transfers.
#   - KNOB PROOF, refuse the card without it: `trim_calls` = 40 for every
#     trim_step arm; ~10,521 at cdev8 and ~1,561 at cdev for `trim` (263.0 and
#     39.0 boundaries/step x 40, measured in 452). An arm whose count is 0 read
#     as "trimming does not help" and must not be believed.
#
# Legs, cheapest first:
#   1 smoke            the orchestrator end to end incl. the NEW trim_step path,
#                      seconds. It gates every leg after it (job 451: five legs
#                      burned on a probe defect its own smoke leg had caught).
#   2 cdev8 K=40       trace / trim / trim_step, 3 repeats. The frequency ladder
#                      in ONE job at ONE config, so the 263x-vs-1x comparison is
#                      not cross-job. ~1.3 h.
#   3 cdev K=40        trim / trim_step, 3 repeats. THE anchor's retention at the
#                      ratified cadence, the number that was missing. Its `trace`
#                      reference is 452's 7.727 GB (sd 0.282): the trace path is
#                      behaviour-identical under this commit (`trim="off"` returns
#                      before any syscall exactly as `trim=False` did), the
#                      machine and backend match, and the effect being measured is
#                      ~0.9 GB = 3+ sigma against that scatter. ~4.1 h.
#
# Budget ~5.5 h, zero SU, antares only (--mem=100G is placement; every peak on
# record is antares and peak comparisons are not valid across machines). Nothing
# touches /work. -t is generous because a timeout bills the whole wall.
# NO set -e: independent legs, per-leg exit codes -- except smoke, which gates.

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

# malloc_trim must be REACHABLE before a job whose whole point is calling it.
pixi run -e default python -c "
import ctypes
ctypes.CDLL('libc.so.6').malloc_trim(0)
print('malloc_trim(0) callable')
" || { echo "FATAL: malloc_trim not callable; every trim arm would be a no-op"; exit 1; }

export TMPDIR="${TMPDIR:-/tmp}/m6_trimsched_$SLURM_JOB_ID"
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
  --configs smoke --k 5 --repeats 3 --arms trace trim trim_step \
  --arena-frac 0.20 --workdir-root "$TMPDIR" --out-suffix _trimsched_smoke
[ "${codes[0]}" -ne 0 ] && {
  echo "FATAL: smoke failed; a broken probe would burn every long leg (job 451)"
  exit 1
}

# ---- the frequency ladder, same job, same config: 0 vs 40 vs 10,521 calls.
run_leg "cdev8-k40-sched" pixi run -e default python $TRACE \
  --configs cdev8 --k 40 --repeats 3 --arms trace trim trim_step \
  --workdir-root "$TMPDIR" --out-suffix _trimsched_cdev8_k40

# ---- the anchor at the ratified cadence: the retention number that was missing.
run_leg "cdev-k40-sched" pixi run -e default python $TRACE \
  --configs cdev --k 40 --repeats 3 --arms trim trim_step \
  --workdir-root "$TMPDIR" --out-suffix _trimsched_cdev_k40

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "CARDS WRITTEN BY THIS JOB:"
n=0
for f in runs/v2/m6_peak_trimsched_*.json; do
  if [ -f "$f" ] && [ "$f" -nt "$STAMP" ]; then echo "  $f"; n=$((n + 1)); fi
done
[ "$n" -eq 0 ] && { echo "  (none)"; echo "FATAL: zero fresh cards"; exit 1; }
echo "($n fresh cards)"
echo "TMPDIR left for inspection: $TMPDIR"
exit $rc_total
