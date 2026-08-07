#!/bin/bash
#SBATCH -J inexor-v4f-t9-quantum
#SBATCH -w deneb
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 12:00:00
# --mem is REQUIRED on deneb (v2_g1_deneb.sbatch); 24G matches the G2c gate,
# whose orchestrator this reuses verbatim.
#SBATCH -o %x-%j.log
# V4f: is the RATIFIED 9 B/p state tier implementable at the fidelity it was
# gated at? Submit from ~/src/inexor on deneb:
#   sbatch scripts/v2_v4f_t9_quantum_deneb.sbatch
#
# WHY. D-v2-8 ratified T9 = 9 B/p (int8 cell-relative positions + int16
# velocity) on G2c's t9 arm, which passed by ~3 orders (1.3e-4 against a 1.3e-1
# floor). But v2_g2c_accum_gate.py says in its own docstring, line 35:
#
#   "The 3 B/p position claim assumes CUBE's sorted-by-cell layout; this gate
#    measures REPRESENTATION error only (storage layout is a build decision)."
#
# The caveat was never hidden -- it is in the summary printer too -- but it was
# deferred, and V4 is where that build decision is taken. The arithmetic does
# not work as assumed. The gated arm quantizes at fine_cell/256, which an int8
# can only carry if its implicit bucket is ONE FINE CELL (verified: the t9 arm's
# bucket is 0.25 Mpc/h, exactly the fine cell). At C-gh the fine mesh is
# 4096^3 = 6.9e10 cells against 8.6e9 particles -- EIGHT cells per particle --
# so a per-cell index costs ~69 GB even at one byte per entry, more than the
# 77 GB of state it indexes. CUBE avoids this by running one particle per cell;
# our fine mesh is 2x the particle grid per side, so its layout does not carry
# over.
#
# An affordable index needs a coarser bucket, which coarsens the quantum:
#
#   arm    bucket        quantum     index at C-gh   vs gated
#   t9     0.25 Mpc/h    fine/256    ~69 GB   <- BREAKS THE BUDGET (as gated)
#   t9c1   0.5  Mpc/h    fine/128    17.2 GB  2x coarser
#   t9c2   1.0  Mpc/h    fine/64      2.15 GB 4x coarser  <- the likely choice
#   t9c4   2.0  Mpc/h    fine/32      0.27 GB 8x coarser
#
# So the tier we would actually SHIP has never been measured. This job measures
# it, with the velocity codec held identical (int16 max-range) across all four
# arms so the only variable is the position quantum.
#
# PRE-REGISTERED READING (written before the job ran).
#   Statistic: max |dP/P| in the D-v2-9 gate band (k <= 0.2 k_Nyq,fine), and the
#   RSD pair dP0/P0, dP2/P0, each vs the SAME-K ref arm -- the identical
#   statistics D-v2-8 ratified t9 on. Bar: D-v2-9's absolute 3e-2 in band.
#   Predicted: t9c1 and t9c2 both pass with room (t9 passed by ~3 orders, and
#   position error should scale ~linearly in the quantum, so 4x coarser costs
#   ~4x -- still ~2 orders under). t9c4 is the one that might bite.
#   FALSIFIED IF: t9c2 fails the bar, or degrades faster than ~linearly in the
#   quantum. Then the state tier is not 9 B/p at C-gh and the config table
#   moves -- which is exactly why this runs BEFORE any ADR is drafted.
#   NOT A GATE in the ratifying sense: this reports numbers against an existing
#   bar; whether the tier changes is JC's call at the freeze.
#
# COST NOTE. This runs the full K sweep for 5 arms (ref + 4), which is more
# worker subprocesses than the original gate. deneb is free and the wall is
# generous; if the 6 GB card OOMs at cdev the pre-agreed fallback (same
# cell/spacing at 1/8 volume, so every codec quantum transfers unchanged) fires
# below, exactly as the G2c gate did.

set -e
cd ~/src/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export OMP_NUM_THREADS=1

echo "=== env ==="
git log --oneline -1
hostname
pixi run -e gpu python -c "import jax; print(jax.__version__, jax.devices())"

ARMS="t9,t9c1,t9c2,t9c4"

echo
echo "=== V4f: T9 position-quantum ladder at C-dev (arms $ARMS) ==="
pixi run -e gpu python scripts/v2_g2c_accum_gate.py --config cdev --arms "$ARMS"

# Pre-agreed fallback, identical to the G2c gate's: if any C-dev worker OOM'd
# the 6 GB card, rerun at C-dev/8. Same fine cell AND spacing, so every codec
# quantum -- which is what this job measures -- transfers unchanged.
if pixi run python - <<'EOF'
import json, sys
r = json.load(open("runs/v2/g2c_results.json"))
sys.exit(0 if any(w.get("oom") for w in r["workers"].values()) else 1)
EOF
then
  echo
  echo "=== C-dev OOM detected -> fallback: C-dev/8 ==="
  pixi run -e gpu python scripts/v2_g2c_accum_gate.py --config cdev8 --arms "$ARMS"
fi

echo
echo "=== DONE ==="
