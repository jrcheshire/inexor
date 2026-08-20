#!/bin/bash
#SBATCH -J m6-c16-eject-wall
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o m6-c16-eject-wall-%j.log
# C16 = PRICE THE EJECT KERNEL ON GRACE. `--eject-kernel numpy` is load-bearing
# for the 2048^3 fit -- the planner refuses that config at 283.2 GB on the jax
# eject and passes it at 232.7 of 255.1 (0.91x) on numpy -- and the only number
# we have for what numpy costs in WALL is 1.85x per row, measured on an M4
# laptop by a throwaway probe. A 30 h leg 2 should not rest on that.
#
# THIS IS NOT AN A/B THAT PICKS. numpy is the arm that fits, so the jax arm is
# a reference point for the penalty, not a candidate. What the job produces is
# the BILL: what the fit costs in hours at 2048^3, so leg 2's wall request is
# derived rather than guessed.
#
# THE 1.85x MAY NOT SURVIVE, AND THAT IS PART OF THE QUESTION. A dry run of
# this exact readout on the laptop at cdev8 (K=3, repeats=1, W=8, receipts
# clean) read the numpy eject at 0.199 s/step against jax's 1.129 -- numpy
# FASTER by 5.7x, not slower by 1.85x. Do not treat that as the answer: it is
# one small config on an M4, the jax arm at cdev8 may still be carrying
# per-shape compilation inside the traced region, and per-call overhead
# dominates at 2M particles. But it does mean the recorded 1.85x -- a
# throwaway probe that timed the row kernels in isolation, never the pooled
# phase -- is measuring a different thing from the number that bills a run.
# Either sign is a reportable result here.
#
# THE QUANTITY IS MEASURED, NOT INFERRED. `v2_m6_phase_time.py`'s card carries
# `migrate_pool.eject_busy_s_median` separately from `insert_busy_s_median`, so
# the eject's own busy time is read directly rather than backed out of the
# migrate phase total.
#
# PRE-REGISTERED (before submission):
#   a. RECEIPTS FIRST, on every leg. `eject_kernel` on the card must read the
#      kernel the leg asked for, `migrate_pooled_workers_per_step` must be [8]
#      on every step, and `migrate_pool.eject_jax_calls_per_step` must be >0 on
#      the jax legs and [0] on the numpy ones. NOT the top-level
#      `eject_jax_calls`: that reads the PARENT counter, which is 0 by
#      construction on a pooled leg because the workers do the ejecting in
#      their own processes -- checking it marks a healthy jax leg as broken,
#      which the dry run did before this was corrected. A leg missing its
#      receipt measured nothing and its number is not reportable.
#   b. THE DELIVERABLE: eject_busy_s_median, numpy vs jax, at cgh64 W=8
#      pooled. Converted at the exact standing factor -- 1 s/step at cgh64 =
#      0.711 h at 2048^3 (= 64 * 40 / 3600) -- into hours added per 2048^3
#      realization. Reported, not gated: there is no bar to pass, because the
#      alternative arm does not fit.
#   c. The same difference on the WHOLE step (`s_per_step`), because the eject
#      being 1.85x of a small thing and 1.85x of the step are different bills.
#   d. Instrument neutrality is the probe's own rc=2, tolerated and reported.
#      A non-neutral leg means the breakdown describes the instrument; the
#      per-step total still stands.
#
# WHAT THIS DOES NOT ESTABLISH: cgh64 is 512^3. The eject is per-row work, so
# the rate should carry, but the pooled migrate's cost at 2048^3 has never been
# measured at all (nothing has ever completed a step there). This prices the
# kernel choice, not the phase at scale.
#
# BUDGET: ~55 min expected against a 2 h request (gg 0.33 SU/node-hr, so ~0.3
# SU expected, 0.66 SU worst case). Generous on purpose: a TIMEOUT bills the
# whole wall anyway, and one M-v2-5 job burned 6 h 41 min producing nothing.
#
# OPS (gg, all load-bearing): pixi on PATH; -e gpu (the default env is not
# installed on Vista); JAX_PLATFORMS=cpu (jax 0.10's CUDA plugin hard-raises on
# cuInit on a GPU-less node); CONDA_OVERRIDE_CUDA (env activation gate);
# -A JPL-SPHEREx is case-sensitive. NO set -e: per-leg exit codes, and the job
# exits $rc_total -- a wrapper that captures $? per leg and never propagates it
# reported COMPLETED 0:0 over an OOM-killed leg once (922905).

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$SCRATCH/src/inexor" 2>/dev/null || cd "$HOME/src/inexor" || {
  echo "FATAL: no inexor checkout"; exit 1; }
export JAX_PLATFORMS=cpu
export CONDA_OVERRIDE_CUDA=12.0
export PYTHONUNBUFFERED=1
pixi run -e gpu python -c "
import jax
assert jax.devices()[0].platform == 'cpu', 'expected the CPU backend'
print('jax', jax.__version__)
" || { echo "FATAL: jax import/backend check failed"; exit 1; }

echo "=== $(hostname): $(nproc) cores, commit $(git rev-parse --short HEAD) ==="
free -g | head -2

rc_total=0
names=(); codes=()
# `run_leg [-t CODES] NAME CMD...`. CODES is a comma list of NONZERO exit codes
# THIS leg pre-registers as reportable states rather than failures.
#   rc=2 = the phase probe reporting its own instrument non-neutrality
run_leg() {
  local tol=""
  [ "$1" = "-t" ] && { tol="$2"; shift 2; }
  local name="$1"; shift
  echo "=== $name ==="
  local t0=$SECONDS
  "$@"; local rc=$?
  echo "=== $name rc=$rc after $((SECONDS - t0)) s ==="
  names+=("$name"); codes+=("$rc")
  local ok=0 c
  [ $rc -eq 0 ] && ok=1
  for c in $(echo "$tol" | tr ',' ' '); do [ "$rc" = "$c" ] && ok=1; done
  [ $ok -eq 0 ] && rc_total=1
  return 0
}

PHASE="scripts/v2_m6_phase_time.py"
# Knobs held at the configuration the 2048^3 fit assumes: W=8, arena-frac 0.10,
# pooled. The two arms differ in ONE flag.
COMMON="--config cgh64 --k 5 --repeats 3 --slack 0.20 --arena-frac 0.10 \
        --tile-workers 8 --migrate-pooled"

# 1. SMOKE, at cdev8: every path the big legs use, on THIS machine and commit,
#    including the card write. The reporting path is untested code until it has
#    run -- job 923313 took the reading it was built for and left no trace of
#    it. Both kernels, because a numpy-only failure at cgh64 would cost 25 min
#    to discover.
#    --slack 0.20 is NOT decoration: the probe defaults to 0.10, and at 0.10
#    with --arena-frac 0.10 the arena refuses ("133 particles overflow ... the
#    arena has only 118 free") -- caught on the laptop while dry-running this
#    file, which is the whole point of writing the smoke first.
run_leg -t 2 "smoke-eject-jax" pixi run -e gpu python $PHASE \
  --config cdev8 --k 3 --repeats 1 --slack 0.20 --arena-frac 0.10 \
  --tile-workers 8 --migrate-pooled --eject-kernel jax \
  --out-suffix _c16_smoke_jax
run_leg -t 2 "smoke-eject-numpy" pixi run -e gpu python $PHASE \
  --config cdev8 --k 3 --repeats 1 --slack 0.20 --arena-frac 0.10 \
  --tile-workers 8 --migrate-pooled --eject-kernel numpy \
  --out-suffix _c16_smoke_numpy
for i in 0 1; do
  rc=${codes[$i]}
  if [ "$rc" -eq 2 ]; then
    echo "NOTE: ${names[$i]} non-neutral (rc=2); smoke exercises paths. Continuing."
  elif [ "$rc" -ne 0 ]; then
    echo "FATAL: smoke leg ${names[$i]} rc=$rc; not burning SU on cgh64"
    echo "=== LEG SUMMARY ==="
    for j in "${!names[@]}"; do echo "  ${names[$j]}: rc=${codes[$j]}"; done
    exit 1
  fi
done

# 2. THE MEASUREMENT, both arms at cgh64 W=8. numpy FIRST, deliberately: it is
#    the arm that fits and therefore the deliverable, so if the wall estimate
#    is wrong and the job runs out of time, the leg that gets cut is the
#    reference rather than the answer.
run_leg -t 2 "cgh64-w8-eject-numpy" pixi run -e gpu python $PHASE \
  $COMMON --eject-kernel numpy --out-suffix _c16_cgh64_numpy
run_leg -t 2 "cgh64-w8-eject-jax" pixi run -e gpu python $PHASE \
  $COMMON --eject-kernel jax --out-suffix _c16_cgh64_jax

# 3. READ IT OUT IN THE JOB, not in a later session. A card that has to be
#    fetched and interpreted by hand is a card that gets quoted wrong; the
#    receipts in (a) are checked here and the conversion in (b) is applied here.
pixi run -e gpu python - <<'PY'
import json, os, sys

def card(sfx):
    p = os.path.join("runs", "v2", f"m6_phase_time_c16_cgh64_{sfx}.json")
    if not os.path.exists(p):
        print(f"  MISSING {p}")
        return None
    with open(p) as fh:
        return json.load(fh)

n, j = card("numpy"), card("jax")
if not (n and j):
    print("VERDICT: UNREADABLE -- a card is missing")
    raise SystemExit(3)

ok = True
for name, c, want_calls in (("numpy", n, False), ("jax", j, True)):
    got = c.get("eject_kernel")
    w = c.get("migrate_pooled_workers_per_step")
    # THE POOLED RECEIPT IS THE WORKER-SUMMED COUNT, not the top-level one.
    # `eject_jax_calls` reads the PARENT counter, which is 0 by construction on
    # a pooled leg because the workers do the ejecting in their own processes;
    # checking it marks a healthy jax leg as broken. The per-step count is nb
    # (slabs), not W, so the test is >0 rather than a specific number.
    calls = c.get("migrate_pool", {}).get("eject_jax_calls_per_step", [])
    good_calls = (min(calls) > 0) if (want_calls and calls) else (calls == [0])
    print(f"  {name}: eject_kernel={got!r} workers_per_step={w} "
          f"eject_jax_calls_per_step={calls} neutral={c.get('instrument_neutral')}")
    if got != name or w != [8] or not good_calls:
        print(f"    RECEIPT FAILED for {name}: this leg measured nothing")
        ok = False
if not ok:
    print("VERDICT: RECEIPTS FAILED -- numbers are not reportable")
    raise SystemExit(4)

CONV = 64 * 40 / 3600.0   # 1 s/step at cgh64 = 0.711 h at 2048^3, exact
en = n["migrate_pool"]["eject_busy_s_median"]
ej = j["migrate_pool"]["eject_busy_s_median"]
sn, sj = n["s_per_step"], j["s_per_step"]
print()
print(f"  eject busy   numpy {en:8.3f} s/step   jax {ej:8.3f} s/step   "
      f"ratio {en / ej if ej else float('nan'):.2f}x")
print(f"  whole step   numpy {sn:8.3f} s/step   jax {sj:8.3f} s/step   "
      f"ratio {sn / sj if sj else float('nan'):.2f}x")
print()
print(f"  THE BILL at 2048^3 (x{CONV:.3f} h per s/step):")
print(f"    eject alone adds {(en - ej) * CONV:7.2f} h per realization")
print(f"    whole step  adds {(sn - sj) * CONV:7.2f} h per realization")
print(f"    numpy-arm realization from this step total: {sn * CONV:7.2f} h "
      f"(ARITHMETIC on cgh64, not a measured 2048^3 wall)")
print("VERDICT: READ")
PY
rc=$?
names+=("readout"); codes+=("$rc")
[ $rc -ne 0 ] && rc_total=1

echo "=== LEG SUMMARY ==="
for j in "${!names[@]}"; do echo "  ${names[$j]}: rc=${codes[$j]}"; done
echo "=== rc_total=$rc_total ==="
exit $rc_total
