#!/bin/bash
#SBATCH -J m6-pad-ab
#SBATCH -p compute
#SBATCH --mem=100G
#SBATCH -c 8
#SBATCH -t 08:00:00
#SBATCH -o m6-pad-ab-%j.log
# M-v2-6 Stage 0b, the A/B: does the per-step XLA shape churn in the coarse
# chunk buffer CAUSE the engine's peak to grow with K?
#
# What is already measured (job 446, `runs/v2/m6_peak_record.md`): the growth is
# +103 MB/step at cdev K=15, linear and unsaturating over 15 steps; about 59% of
# it survives malloc_trim at cdev and 57% at cdev8, so the majority is live
# memory; and the chunk buffer took a new shape on all 10 of 10 steps at smoke
# against 1 for `cap`. The LINK between those two facts is untested. This job is
# the test, and it is the only thing here that is new.
#
# Expected signal if the churn is the cause: the ON arm's slope falls by the
# live fraction, ~60% of ~103 = ~60 MB/step at cdev. Five repeats per arm
# because the per-repeat slope scatter at K=15 is 34 MB/step, so three repeats
# would put a 60 MB/step signal at ~2 sigma and five puts it near 3.
#
# `pad_ladder` off is the A arm and NEVER an operating point. `cap` stays on its
# ladder in BOTH arms: moving `cap_rungs` would move both shape families at once
# and the reading would be unattributable. The card carries `pad_ladder`,
# `coarse_pad_distinct` and `cap_distinct` so the knob has to prove it applied
# and the control has to prove it did not move.
#
# Both arms of a config share ONE workdir, so they run on identical ICs.
#
# --mem=100G is PLACEMENT expressed as a resource: only antares (28 c, 124 GB)
# satisfies it; deneb-the-node has 56 GB. Peaks are not comparable across
# machines and every number in the record is antares, so this must be too.
# `-c 8` is honest: job 433 measured s/step FLAT across 1..32 threads.
#
# Nothing here touches /work. TMPDIR is node-local and the cards land in
# ~/src/inexor/runs/v2 on /home.
#
# Budget: ~4.9 h of compute in four legs, cheapest and most valuable first, so
# the job can be killed after any leg without losing the one before it. The
# last leg is the droppable one.
# NO set -e, deliberately: independent legs, per-leg exit codes.

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

# The high-water reset the whole instrument rests on (same check as 446).
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

# The A arm has to be REACHABLE from this checkout, or the job runs two copies
# of the same arm and reports a null. Cheap, exact, and it fails before any
# science: a shape count is integer arithmetic.
pixi run -e default python -c "
from inexor.engine import EngineConfig
assert EngineConfig(box_size=32.0, n_part=32, n_fine=64, n_coarse=16, n_tile=16,
                    b_fine=8, pad_ladder=False).pad_ladder is False, 'knob absent'
print('pad_ladder knob present')
" || { echo "FATAL: this checkout predates the pad_ladder A/B knob"; exit 1; }

export TMPDIR="${TMPDIR:-/tmp}/m6_pad_ab_$SLURM_JOB_ID"
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

# ---- SMOKE, and it is not ceremony: 446's smoke leg died on `--arena-frac`
# 0.08 (29 particles against 9 free arena slots at 32,768 particles, the D-007
# refusal working as designed). 0.20 works; 0.30/0.30 refuses from the other
# side on the repack allocation. Seconds, and it exercises every path.
run_leg "smoke" pixi run -e default python $TRACE \
  --configs smoke --k 5 --repeats 3 --arms trace control trim \
  --arena-frac 0.20 --workdir-root "$TMPDIR" --out-suffix _padab_smoke

# ---- and the SAME smoke with the A arm's knob, which exercises the one code
# path nothing else here can reach before the expensive legs: the probe refuses
# non-Linux, so `--no-pad-ladder` has never run outside a unit test. Both smoke
# cards carry `coarse_pad_distinct`; at K=5 the off arm should report 5 and the
# on arm 1, which is the knob proving it applied in the field rather than in a
# fixture. Seconds.
run_leg "smoke-off" pixi run -e default python $TRACE \
  --configs smoke --k 5 --repeats 3 --arms trace --no-pad-ladder \
  --arena-frac 0.20 --workdir-root "$TMPDIR" --out-suffix _padab_smoke_off

# ---- cdev8 A/B, ~0.6 h. The cheap config first, so a defect in the A/B
# plumbing surfaces in ten minutes rather than three hours. It is also the
# second config: the trace/trim split agreed at both (57% and 59% retained), so
# a fix that moved only one of them would be the interesting failure.
run_leg "ab-cdev8-on" pixi run -e default python $TRACE \
  --configs cdev8 --k 15 --repeats 5 --arms trace \
  --workdir-root "$TMPDIR" --out-suffix _padab_cdev8_on
run_leg "ab-cdev8-off" pixi run -e default python $TRACE \
  --configs cdev8 --k 15 --repeats 5 --arms trace --no-pad-ladder \
  --workdir-root "$TMPDIR" --out-suffix _padab_cdev8_off

# ---- cdev A/B, ~2.7 h. THE measurement. K=15 because the slope is what is
# being moved and 15 points is the trustworthy ladder; the off arm doubles as a
# reproduction of 446's +103 MB/step on the same machine and the same ICs.
run_leg "ab-cdev-on" pixi run -e default python $TRACE \
  --configs cdev --k 15 --repeats 5 --arms trace \
  --workdir-root "$TMPDIR" --out-suffix _padab_cdev_on
run_leg "ab-cdev-off" pixi run -e default python $TRACE \
  --configs cdev --k 15 --repeats 5 --arms trace --no-pad-ladder \
  --workdir-root "$TMPDIR" --out-suffix _padab_cdev_off

# ---- The corrected-probe re-run, ~1.5 h, and the DROPPABLE leg: every number
# it produces already exists, recomputed offline from 446's persisted series by
# `scripts/v2_m6_recompute_trace.py`. What it adds is a card whose own verdict
# block can be read directly instead of recomputed, on the post-fix engine.
run_leg "rerun-cdev-k5" pixi run -e default python $TRACE \
  --configs cdev --k 5 --repeats 5 --arms trace control trim \
  --workdir-root "$TMPDIR" --out-suffix _padab_cdev_k5

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "CARDS WRITTEN BY THIS JOB:"
n=0
# `m6_peak` + the out-suffix, which is what `_write` builds -- 446's cards read
# m6_peak_trace_* only because its suffixes started `_trace`. A glob written for
# those names would have found nothing here and failed a job that succeeded.
for f in runs/v2/m6_peak_padab_*.json; do
  if [ -f "$f" ] && [ "$f" -nt "$STAMP" ]; then echo "  $f"; n=$((n + 1)); fi
done
[ "$n" -eq 0 ] && { echo "  (none)"; echo "FATAL: zero fresh cards"; exit 1; }
echo "($n fresh cards)"
echo "TMPDIR left for inspection: $TMPDIR"
exit $rc_total
