#!/bin/bash
#SBATCH -J inexor-m4-floors
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -o %x-%j.log
#
# M-v2-4 leg 4: the FLOORS, the context rungs leg 1 is read against.
#
# WHY IT IS OWED. It was queued behind `cgh64-k10` in Vista 899070's single
# chained job. That leg blew its 2.3 h estimate, consumed 6 h 41 min without
# finishing, and took the wall with it -- so floors, budgeted at well under an
# hour, never started. One leg per job is the whole point of this file.
#
# WHAT IT MEASURES. Three single-arm f64 evolutions at cdev8, and the comparison
# is between them, not against a bar:
#
#   step floor        K vs 2K at fixed mesh      (v2_g2c_accum_gate's discipline)
#   coarse mesh floor n_coarse vs n_coarse/2     at fixed K
#
# The COARSE mesh floor does not exist anywhere on record. G2c's 1.3e-1 is the
# FINE mesh floor, and the fine mesh is not the arm this milestone changes. That
# gap is why the charter's "read against the MESH FLOOR" instruction had no
# referent, and measuring it is what gives leg 1's 3.6e-5 to 7.5e-5 readings a
# scale to be small against.
#
# NOT A BAR, and the card says so in its own `note`. A floor is context. Leg 1's
# tiers are 3.0e-3 and 1e-5 and neither is touched by whatever this reads.
#
# CAPACITY 0.20/0.08 to match every other cdev8 card in this milestone.
#
# BUDGET from deneb's measured 7.83 s/step at cdev8, single arm. The leg runs
# K=40 (40 steps) + K=80 (80 steps) + K=40 at half coarse mesh (40 steps) = 160
# steps ~= 21 min. 3 h is generous. Deneb is unbilled, but the wall is still
# sized from measurement rather than a scaling guess: job 399 came in 14x under
# an a-priori estimate, and 899070 died at a wall set the same way.

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
    sys.exit('FATAL: CPU backend -- leg 1 is on the GPU and a floor on a different backend is not its context')
" || { echo "FATAL: jax/GPU precondition failed"; exit 1; }

STAMP="$(mktemp)"; touch "$STAMP"

pixi run -e gpu python scripts/v2_m4_f32_mesh_gate.py \
    --config cdev8 --leg floors --k 40 --slack 0.20 --arena-frac 0.08
rc=$?

echo ""
echo "=== CARDS WRITTEN (this job only) ==="
n=0
while IFS= read -r f; do
  echo "  $f"; n=$((n+1))
done < <(find runs/v2 -maxdepth 1 -name 'm4_gate_cdev8_floors.json' -newer "$STAMP" 2>/dev/null)
[ "$n" -eq 0 ] && echo "  (none)"
rm -f "$STAMP"

echo ""
echo "floors rc=${rc};  fresh cards: ${n}"
[ "$n" -eq 0 ] && { echo "FATAL: zero cards written by THIS job"; exit 1; }
exit $rc
