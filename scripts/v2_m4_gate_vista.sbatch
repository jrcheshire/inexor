#!/bin/bash
#SBATCH -J inexor-m4-gate
#SBATCH -o %x-%j.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A JPL-SPHEREx
#SBATCH -t 08:00:00
#
# M-v2-4 exit gate: the f32 coarse force mesh (scripts/v2_m4_f32_mesh_gate.py).
#
# WHAT THIS ANSWERS
#   1. What does an f32 COARSE force mesh cost in evolved P(k), measured against
#      the same engine at f64 from the same ICs, in D-v2-9's band? The gated
#      reading is cdev K=40.
#   2. Does the coarse arm actually carry the dtype it was configured with? Leg
#      0 reads every intermediate and exits non-zero before anything else runs.
#      Not paranoia: THREE dtype parameters in this package turned out not to
#      apply while M-v2-4 was being built.
#   3. What does it buy, measured rather than derived? The ladder replaces
#      D-v2-16 clause 3's "derived, not measured" with three arms over
#      n_coarse.
#
# WHAT IT DOES NOT ANSWER
#   Performance. The per-tile loop is host numpy and nothing here is
#   axis-matched; the walls are budget, not measurement. Also nothing about the
#   FINE tile mesh beyond leg 5's reading, which is measured and NOT gated --
#   the fine mesh is P^3 and box-independent, so it carries no capacity
#   argument.
#
# THE BAR IS TWO-TIER (JC, 2026-08-09), and both are reported:
#   tier 1  BAR          0 < E <= 3.0e-3   budget share, 10% of D-v2-9's 3e-2
#   tier 2  EXPECTATION  E <= 1e-5         what actually discriminates
#   Above 1e-4 is a FINDING requiring investigation, not an automatic fail.
#   Stated as two because the bar alone is nearly unfailable at a 300-3000x
#   margin, which is the same objection that ruled out gating on a mesh floor.
#
# PRE-REGISTERED, before the numbers exist. cdev K=40 is expected at a few x
# 1e-6 and at most 1e-5 -- a strictly smaller perturbation than the whole-
# evolution f32 rung that read R_Q 1.1e-6, and comparable to G1's f32 paint
# scatter of 2-5e-7. The K-ladder should grow no faster than sqrt(K); linear or
# worse means a systematic bias and blocks adoption regardless of the bar.
# cgh64 vs cdev8 at matched K=10 is expected within 3x, not 16x; a 16x-class
# reading makes this a cdev-only license. The ladder is expected to approach
# 2.0x from below, never reaching it -- the int64 accumulator, the int32 paint
# and the f64 kernel BUILD do not change dtype, and the build alone is 12.0 GiB
# at C-gh.
#
# BUDGET, from M-v2-3's measured card walls on this partition (cdev8 728 s,
# cdev 2035 s at K=40, OMP=72). Leg 1 needs PAIRED runs, so each reading is 2x
# those. cdev K=40 ~1.2 h; cdev8 anchors ~0.9 h; cgh64 K=10 ~2.3 h; ladder and
# floors ~0.7 h. 8 h is deliberately generous -- billed for actual use, and the
# last several runtime estimates on this project were all low.

set -u

# NOT boilerplate. Jobs 896055 and 896092 ran ZERO science without this line,
# and 896055 reported COMPLETED 0:0 in sacct because `set -e` had been dropped.
export PATH="$HOME/.pixi/bin:$PATH"

cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
echo "commit: $(git rev-parse --short HEAD)  branch: $(git rev-parse --abbrev-ref HEAD)"
echo "host:   $(hostname)   started $(date -Is)"

command -v pixi >/dev/null 2>&1 || { echo "FATAL: pixi not on PATH"; exit 1; }

# The backend is DEMONSTRATED, not inferred. Job 897377 could only infer the GPU
# was live from a 6.83x speedup, and the log recorded no device at all.
pixi run -e gpu python -c "
import jax, sys
d = jax.devices()
print('jax', jax.__version__, 'devices:', d, flush=True)
if d[0].platform == 'cpu':
    sys.exit('FATAL: CPU backend -- this gate must exercise the CUDA path')
" || { echo "FATAL: backend check failed"; exit 1; }

export OMP_NUM_THREADS=72   # one Grace socket; the layout is host numpy

# STAMP: the CARDS WRITTEN block below counts only files newer than this.
# The m3 mould globbed `runs/v2/m3_gate_*.json` with no freshness filter, so a
# job that wrote NOTHING still passed its own "zero cards written" FATAL on a
# previous job's files -- a green row for a run that produced nothing, which is
# the exact failure 896055 already taught us once.
STAMP="$(mktemp)"; touch "$STAMP"

# No `set -e`: the legs are independent and the cheapest runs first, so a
# failure in one must not delete the evidence from the others (M1 job 11).
NAMES=(); CODES=()
run_leg () {
  local name="$1"; shift
  echo ""; echo "=== LEG ${name} : $* ==="; date -Is
  pixi run -e gpu python "$@"
  local rc=$?
  NAMES+=("$name"); CODES+=("$rc")
  echo "=== LEG ${name} exit ${rc} ==="; date -Is
}

G=scripts/v2_m4_f32_mesh_gate.py
M3=scripts/v2_m3_engine_gate.py

# --- leg 0 FIRST and at the gated config. If the coarse arm does not carry its
#     dtype, every number below reads as a spectacular pass.
run_leg ledger-cdev   "$G" --config cdev  --leg ledger --slack 0.15 --arena-frac 0.05
if [ "${CODES[0]}" -ne 0 ]; then
  echo "FATAL: the dtype ledger failed at cdev; refusing to run the readings."
  exit 1
fi

# --- leg 6: the f64 regression, BEFORE the new readings. `v2_m3_engine_gate.py`
#     verbatim; its cards must reproduce runs/v2/m3_gate_cdev8_*.json to the
#     digit (n_diff 0; max_dpp_in_band 4.666e-04). Confirmed locally at `smoke`
#     that M-v2-4's x64 refusal does not fire here -- that script sets x64 at
#     :344 before any EngineConfig is built -- but confirmed is not assumed.
run_leg m3-regress-force "$M3" --config cdev8 --leg force --out-suffix _m4regress
run_leg m3-regress-accum "$M3" --config cdev8 --leg accum --k 40 --out-suffix _m4regress

# --- leg 1 gated reading, then the anchors
run_leg accuracy-cdev  "$G" --config cdev  --leg accuracy --k 40 --slack 0.15 --arena-frac 0.05
run_leg accuracy-cdev8-k10 "$G" --config cdev8 --leg accuracy --k 10 --out-suffix _k10
run_leg accuracy-cgh64-k10 "$G" --config cgh64 --leg accuracy --k 10 --slack 0.15 \
        --arena-frac 0.05 --out-suffix _k10

# --- leg 3 ladder, reaching C-hero's coarse mesh; upper rungs may OOM and that
#     is a datum, not a defect
run_leg ladder "$G" --config cdev --leg ladder --rungs 128 256 512 1024 2048

# --- leg 4 floors, the context rungs leg 1 is read against
run_leg floors "$G" --config cdev8 --leg floors --k 40

echo ""
echo "=== LEG SUMMARY ==="
ok=0
for i in "${!NAMES[@]}"; do
  echo "  ${NAMES[$i]}: exit ${CODES[$i]}"
  [ "${CODES[$i]}" -eq 0 ] && ok=$((ok+1))
done

echo ""
echo "=== CARDS WRITTEN (this job only) ==="
n=0
while IFS= read -r f; do
  echo "  $f"; n=$((n+1))
done < <(find runs/v2 -maxdepth 1 -name 'm4_gate_*.json' -newer "$STAMP" 2>/dev/null; \
         find runs/v2 -maxdepth 1 -name 'm3_gate_*_m4regress.json' -newer "$STAMP" 2>/dev/null)
[ "$n" -eq 0 ] && echo "  (none)"
rm -f "$STAMP"

echo ""
echo "legs ok: ${ok} of ${#NAMES[@]};  fresh cards: ${n}"
[ "$ok" -gt 0 ] || { echo "FATAL: zero legs succeeded"; exit 1; }
[ "$n" -eq 0 ] && { echo "FATAL: zero cards written by THIS job"; exit 1; }
exit 0
