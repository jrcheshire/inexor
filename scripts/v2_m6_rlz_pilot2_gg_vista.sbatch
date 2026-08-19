#!/bin/bash
#SBATCH -J m6-rlz-pilot2
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o m6-rlz-pilot2-%j.log
# PILOT, SECOND ATTEMPT. Everything the first one (920910) established is kept:
# the 2048^3 ICs are BANKED in $W from that run (rc=0, 90 min, 71.8 GB, 8.36
# B/p, manifest complete, staging cleaned), so this job does not regenerate
# them and picks up at the leg that died.
#
# WHY THE FIRST ONE DIED. `TilePool` adopts the whole SlotState plus the three
# coarse force meshes into shared memory. At this run's own knobs (--slack 0.20
# --arena-frac 0.20, alloc_margin 0.10) that is 161.4 GB, against a /dev/shm
# measured at 127.55 GB (job 922332) -- exactly half of RAM, the kernel
# default. (161.4 and not the 148.5 first quoted: the coarse meshes were
# running f64 because the driver never set the dtype, which is fixed now.) It overflowed IN CONSTRUCTION, 20.9 GB over, before step 1; the
# migrate scratch was never staged. `SharedMemory` sizes lazily, so the
# segments were created cleanly and the process took a SIGBUS on first touch,
# which is why the log has a bus error and no traceback.
#
# THE FIX, and why it costs nothing. memfd_create hands out the same anonymous
# RAM-backed shared pages as shm_open, but through the kernel's internal shm
# mount, which has NO size cap. Measured on Vista: 2.147 GB of memfd raised
# Shmem by 2.139 GB (1.00x) and moved /dev/shm used by 0.000, and a spawned
# interpreter mapped the same pages through /proc/<pid>/fd/<n> in both
# directions. No device, no writeback -- which is why this and not a file on
# /tmp (286 GB of local NVMe, and a real I/O cost to measure).
#
# WHAT THIS JOB IS FOR. The memfd path CANNOT be exercised where this was
# written: macOS has no memfd_create, so the local suite ran entirely on the
# posix fallback. Leg 1 is therefore the identity gate on the real backend,
# not a formality -- it is the first execution of that code anywhere.
#
# PRE-REGISTERED (before submission):
#   a. LEG 1 MUST PASS or nothing else runs. `test_executor.py` on this node
#      covers both backends (the env override exists so both run where both
#      exist), including the bitwise pool-vs-serial identity on memfd.
#      COVERAGE IS GUARANTEED BY THE BACKEND ASSERTION ABOVE, not by a count.
#      `BACKENDS` in the suite is derived from the same `has_memfd()` probe
#      the assertion checks, so memfd params run iff that assertion passed.
#      (A hardcoded expected count was the first version of this and it went
#      stale within the hour -- a gate that drifts every time a test is added
#      is noise, and noise is how 922557's 24/24 got waved through.)
#   b. TOTAL HOST under 200 GB of the node's 237 during stepping, where TOTAL
#      is parent peak PLUS the pool's workers. READ `total_rss_bytes`, NOT
#      `peak_rss_bytes` -- ru_maxrss cannot see other processes, and at cdev8
#      the workers held 21x the parent. Above 200 GB, LEG 2 (the 30 h run) IS
#      NOT SUBMITTED until the term is found. This is the gate, not a reading.
#
#      NOTE, and it is owed an answer either way: `inexor.plan` puts the
#      arithmetic LOWER BOUND at 203.0 GB at --arena-frac 0.20, which is
#      already above this gate (175 GB at 0.01). JC's call was to try 0.20 and
#      dial it down if it gives trouble. If (b) fails, arena_frac is the first
#      lever, not a mystery.
#   c. s_per_step against the 1971 s projection, and `projected_full_run_h`,
#      which REPLACES the ~21.9 h arithmetic. No realization wall in this
#      project has ever been measured; this is the first.
#   d. `migrate_pooled_workers` = 8 and the repack fast-path share at 2.1M
#      bricks, both of which C15 measured only at cgh64.
#   e. THE PER-PHASE HIGH-WATER, which is the number this attempt exists for
#      and which has never been taken at this scale. `--phase-instrument peak`
#      swaps the phase card from seconds to a `VmHWM` per phase, reset at each
#      boundary, so the answer to "which phase holds the peak" is measured
#      rather than modelled. `inexor.plan` prices the coarse solve at 30.1 GB
#      and 923139's 10 s sampler suggests ~79; whichever way that lands, the
#      budget stops being an argument. NOTE the card is PARENT ONLY -- VmHWM is
#      one process's -- so read it beside the workers' summed RSS, not instead.
#      Taking peaks costs wall at every boundary, so THIS RUN'S s_per_step is
#      an upper bound and criterion (c) is owed a `--phase-instrument time`
#      run before any wall figure from it is quoted.
#
# TENSION IN (b), FLAGGED NOT RESOLVED: the 200 GB gate was set when the bound
# read 164.6 GB. The corrected accounting puts this configuration at 218.7 GB
# of lower bound at these very knobs, so (b) as written would refuse a run that
# completes inside a 237 GB node. Either the gate moves or the configuration
# does, and that is JC's call, not this script's -- it is left at 200 so the
# question surfaces as a refusal rather than being quietly rewritten.
#
# arena_frac 0.10, NOT the driver default 0.20. It sets the arena's CAPACITY,
# not its occupancy -- what lands there is driven by `brick_slack` -- so this
# is ~15 GB off the resident state for no expected runtime cost, and if
# anything a cheaper O(n_arena) free-list scan. The risk is capacity, not
# correctness: a full arena REFUSES (there is a gate and a test), so too small
# costs a job and not a wrong answer. The 6% on record is CLAIMS across a
# migrate pass rather than simultaneous residency, and C15 found 88.7-96.2%
# of occupied bricks holding no residents at all, so the true peak is well
# under it -- but has never been measured. This run reports it.
#
# W=8, NOT 16, and this is a measurement not a guess. 922905 got the state
# into shared memory correctly -- Shmem flat at 110.2 GB through the load,
# 127.0 GB once the coarse meshes went in, exactly as priced -- and was then
# killed in the NINETEEN SECONDS the 16 workers took to spawn, MemAvailable
# 74.1 -> 0.0 GB. That is ~4.6 GB per worker, a term the threads file had
# already flagged as unmeasured and a possible cap on W. Eight should need
# ~37 GB of the ~74 available. The pool now prints its workers' summed RSS
# the moment they exist, so the next choice of W comes off a number.
#
# TWO THINGS CHANGED SINCE 922682, which refused rather than ran:
#   - the coarse force mesh is f32, which D-v2-22 adopted and this driver had
#     silently not been applying (25.770 -> 12.885 GB).
#   - the memfd preflight checks the peak ABOVE BASELINE, not the total.
#     Adopted fields already exist in the parent and release their private
#     copies as they are copied, so the cost is the largest single field
#     (78.3 GB) or the fresh segments (12.9 GB), whichever is bigger -- not
#     their 148.5 GB sum. 922682 was refused on the sum against 128.5 GB of
#     MemAvailable, which was the wrong comparison and my error.
#
# Cost: 6 h wall at most, 1 gg node, ~2 SU of the 9,072 (expiring 2026-11-30).
# The wall is generous on purpose -- a timeout bills the whole request, but 5
# steps at the projection is ~2.7 h and the load is unmeasured at this scale.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$SCRATCH/src/inexor" 2>/dev/null || cd "$HOME/src/inexor" || {
  echo "FATAL: no inexor checkout"; exit 1; }
# Every failure so far has been diagnosed from a partial log, and 922790
# printed NOTHING because python block-buffers stdout to a file and was
# SIGKILLed with the buffer unflushed. Unbuffered from here.
export PYTHONUNBUFFERED=1
export INEXOR_LOAD_TRACE=1
export JAX_PLATFORMS=cpu
export CONDA_OVERRIDE_CUDA=12.0

echo "=== $(hostname): $(nproc) cores, commit $(git rev-parse --short HEAD) ==="
free -g | head -2
echo "--- the budget that bound 920910, and the one that replaces it ---"
df -h /dev/shm | tail -1
df -h "$SCRATCH" | tail -1

# THE BACKEND ASSERTION. 922557 passed its identity gate 24/24 and then ran
# the 2048^3 leg on /dev/shm anyway, because `os.memfd_create` is absent from
# conda-forge's CPython (gated on the BUILD sysroot) even though the kernel
# has the syscall and the system python3 exposes it. Nothing in that job ever
# stated which backend it was on, so a green gate and a wrong backend looked
# identical. It does not run on assumption again: this asks the SAME code the
# engine will ask, in the SAME interpreter, and stops here if the answer is
# not memfd.
pixi run -e gpu python -c "
from inexor.executor import shm_backend, has_memfd
import os
b = shm_backend()
print('os.memfd_create attribute :', hasattr(os, 'memfd_create'))
print('has_memfd() probe         :', has_memfd())
print('backend in force          :', b)
assert b == 'memfd', (
    'refusing to run 2048^3 on the ' + b + ' backend: that is the capped '
    '/dev/shm mount, it holds 127.55 GB, and this configuration needs 148.5'
)
" || { echo "FATAL: not on the memfd backend; see above. Nothing else runs."; exit 1; }

W="$SCRATCH/inexor_runs/c-gh-r0"
RLZ="scripts/v2_m6_realization.py"

rc_total=0
names=(); codes=()
# BY NAME, not by position. The gates below used codes[N] and inserting a leg
# silently renumbered every one of them -- the export gate would have read the
# smoke leg's rc and published a partial state as though stepping had passed.
# A subscript that has to be re-derived every time the script changes is the
# same shape as the `$?`-capturing bug that made 922905 report COMPLETED 0:0.
declare -A RC
run_leg() {
  local name="$1"; shift
  echo "=== $name ==="
  local t0=$SECONDS
  "$@"; local rc=$?
  echo "=== $name rc=$rc after $((SECONDS - t0)) s ==="
  names+=("$name"); codes+=("$rc"); RC["$name"]=$rc
  [ $rc -ne 0 ] && rc_total=1
  return 0
}

# 1. THE GATE. First execution of the memfd path anywhere. Both backends run
#    here because both exist on this kernel; the local suite could only ever
#    see posix, so a green laptop says nothing about the arrays this job maps.
run_leg "identity-both-backends" \
  pixi run -e gpu python -m pytest tests/test_executor.py -q
if [ "${RC[identity-both-backends]}" -ne 0 ]; then
  echo "FATAL: the executor identity gate failed on this node. The pool is not"
  echo "bitwise the serial loop on the backing store it would have run on, and"
  echo "no 2048^3 result from it would mean anything. Nothing else runs."
  exit 1
fi

# 1b. THE REPORTING PATH, exercised before the expensive leg rather than after
#     it. `--phase-instrument peak` is new and its first execution anywhere is
#     this job: it needs /proc/self/clear_refs, which the laptop does not have,
#     so no local smoke could reach it. This milestone has already lost two
#     cluster jobs on a `print` (447 on another arm's key, 448 on a renamed
#     one), and a card that only runs after forty minutes of stepping is
#     untested code by construction. Two seconds here buys that.
run_leg "peak-card-smoke" pixi run -e gpu python -c "
import sys; sys.path.insert(0, 'scripts')
import v2_m6_realization as R
from v2_m6_peak_trace import PhaseTracer
R._require_linux_for_peaks()
t = PhaseTracer(trim='off')
for name in ('kernel_build', 'lead_drift', 'coarse_paint', 'coarse_solve',
             'membership', 'tile_loop_end', 'reconcile', 'migrate', 'repack'):
    t(name)
rep = t.report()
R._print_phase_card(rep, 'peak')
assert rep['run_peak'] > 0, 'PhaseTracer read a zero run peak'
assert not rep['unknown_phases'], rep['unknown_phases']
print('peak instrument OK; VmHWM readable and clear_refs writable')
"
if [ "${RC[peak-card-smoke]}" -ne 0 ]; then
  echo "FATAL: the per-phase peak instrument does not work on this node, and"
  echo "the whole point of this job is the number it produces. Nothing else runs."
  exit 1
fi

# 2. Confirm the ICs are still there and complete. They cost 90 minutes and
#    nothing in this job regenerates them; a missing manifest here is a
#    scratch purge, not a bug, and is worth saying plainly.
if [ ! -f "$W/manifest.json" ]; then
  echo "FATAL: no manifest at $W -- the banked ICs are gone (purge?)."
  echo "Regenerate with: python $RLZ ics --config c-gh --workdir $W  (~90 min)"
  exit 1
fi
echo "--- banked ICs ---"
du -sh "$W"
python3 -c "
import json; m = json.load(open('$W/realization_ics.json'))
print('  ICs from commit', m['commit'], 'at', m['when'])
print('  wall %.1f min, peak %.1f GB, %d slabs' % (
    m['wall_s']/60, m['peak_rss_bytes']/1e9, len(m['manifest']['files'])))"

# A /proc sampler for the stepping leg. Node-local, killed with the leg, and
# the only way to see WHICH term is growing when the OOM killer arrives -- it
# leaves nothing in the job log. Same idea as C11's sidecar.
#
# UNITS: /proc/meminfo is in KIBIBYTES, so GB is $2*1024/1e9 and dividing by
# 1048576 gives GiB. This printed GiB labelled GB and every reading it has ever
# produced was 7.4% low -- which is the whole of the "driver says 135.6, the
# trace says 110.9" discrepancy that stood as an open thread: 121.0 GB and
# 112.7 GiB are the same number. `MemTotal` is the tell, 237.6 GiB = 255.1 GB.
#
# 2 s, not 10: 923139's whole failure happened inside one 10 s window, so the
# shape of the spike -- which term, in which order -- was unreadable. The awk
# is free next to a step measured in minutes.
( while true; do
    awk '/^(MemTotal|MemAvailable|Shmem|Cached):/ {printf "%s=%.2fGB ", $1, $2*1024/1e9}' \
      /proc/meminfo
    echo "t=${SECONDS}s"
    sleep 2
  done ) > mem-trace-$SLURM_JOB_ID.log 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

# 3. FIVE STEPS: the leg that took the bus error, re-run on memfd. Same knobs
#    as 920910 so the comparison is clean -- the backing store is the only
#    thing that changed.
run_leg "cgh-run-5" pixi run -e gpu python $RLZ run --config c-gh --workdir "$W" \
  --tile-workers 8 --checkpoint-every 5 --stop-at 5 --arena-frac 0.10 \
  --phase-instrument peak

# 4. Export and card at step 5: PATH TESTS at scale, not science. The state is
#    five steps into a forty-step schedule and --allow-partial says so.
if [ "${RC[cgh-run-5]}" -eq 0 ]; then
  run_leg "cgh-export-partial" pixi run -e gpu python $RLZ export --config c-gh \
    --workdir "$W" --tile-workers 8 --allow-partial \
    --export-dir "$W/export_pilot"
  run_leg "cgh-card-partial" pixi run -e gpu python $RLZ card --config c-gh \
    --workdir "$W" --tile-workers 8
else
  echo "STEPPING FAILED: skipping export and card -- they would describe a state"
  echo "the engine did not finish producing."
fi

kill $SAMPLER 2>/dev/null
echo "=== MEMORY TRACE (last 20 samples) ==="
tail -20 mem-trace-$SLURM_JOB_ID.log

echo "=== LEG SUMMARY ==="
for j in "${!names[@]}"; do echo "  ${names[$j]}: rc=${codes[$j]}"; done
echo "=== DISK ==="
du -sh "$W" 2>/dev/null
df -h "$SCRATCH" | tail -1

echo
echo "=== READ THE CARDS, in this order ==="
echo "  $W/realization_run_00_05.json"
echo "     total_rss_bytes < 200e9   <- criterion (b). NOT peak_rss_bytes."
echo "     s_per_step vs 1971 projected, and projected_full_run_h, which"
echo "     REPLACES the ~21.9 h arithmetic -- the first measured wall."
echo "     migrate_pooled_workers = 16, and the repack fast share at 2.1M bricks."
echo "  $W/realization_export.json and realization_pk.json -> the paths at scale."
echo "LEG 2 (scripts/v2_m6_rlz_run_gg_vista.sbatch, 30 h, ~6.6 SU) IS NOT"
echo "AUTOMATIC: submit it only after criterion (b) is read and passed."

# THE JOB'S EXIT CODE MUST MEAN SOMETHING. `run_leg` swallows per-leg codes
# on purpose so the summary always prints, but the script then exited 0 no
# matter what -- 922905 reported COMPLETED 0:0 with its stepping leg killed,
# and that is how a dead run gets read as a live one.
exit $rc_total
