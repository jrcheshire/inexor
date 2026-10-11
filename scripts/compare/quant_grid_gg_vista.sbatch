#!/bin/bash
#SBATCH -J quant-grid
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -o quant-grid-%j.log
# The quantization grid of scripts/compare/quant_grid.py on one gg (Grace-Grace, 144-core)
# node: every leg a separate CPU process pinned to its own 4 cores, all at once, spread over
# both sockets.
#
#   cdev8 (0.5 Mpc/h spacing): z_init 9 / 19 / 49 x 40 / 120 / 240 steps, engine and ref arms;
#     refx / refv / refxv at (9, 120) and (49, 240); ref at (49, 408), the late-time step size of
#     240 steps from z = 9 (the 2LPT start-truncation comparison)
#   cdev8-L128 (1.0 Mpc/h spacing): (9, 120), (19, 120), (49, 240), engine and ref
#
# The inexor checkout's gpu env is used through its python directly with JAX_PLATFORMS=cpu,
# never through `pixi run`, so this job cannot reinstall an env other jobs share.
#   OUT_DIR=$SCRATCH/inexor_runs/quant-grid sbatch -A <account> \
#     scripts/compare/quant_grid_gg_vista.sbatch   (from the inexor checkout, or set INEXOR_SRC)
# Expected wall ~2 h (longest leg: engine, 240 steps from z = 49).

set -u
SRC=${INEXOR_SRC:-$SLURM_SUBMIT_DIR}
[ -f "$SRC/scripts/compare/quant_grid.py" ] || { echo "FATAL: not an inexor checkout: $SRC"; exit 1; }
OUT=${OUT_DIR:?set OUT_DIR, e.g. $SCRATCH/inexor_runs/quant-grid}/job-$SLURM_JOB_ID
PY=$SRC/.pixi/envs/gpu/bin/python3.14
mkdir -p "$OUT/legs"
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
unset XLA_FLAGS CUDA_VISIBLE_DEVICES

echo "=== $(hostname) job $SLURM_JOB_ID $(date) ==="
git -C "$SRC" log --oneline -1
nproc; numactl --hardware | head -4
"$PY" -c "import jax, inexor; print('jax', jax.__version__, jax.default_backend(), inexor.__file__); assert jax.default_backend() == 'cpu'" \
  || { echo "FATAL: the env does not import on CPU"; exit 1; }

legs=()
for a in 0.1 0.05 0.02; do
  for k in 40 120 240; do
    for arm in engine ref; do legs+=("cdev8 $a $k $arm"); done
  done
done
for arm in refx refv refxv; do legs+=("cdev8 0.1 120 $arm" "cdev8 0.02 240 $arm"); done
legs+=("cdev8 0.02 408 ref")
for ak in "0.1 120" "0.05 120" "0.02 240"; do
  for arm in engine ref; do legs+=("cdev8-L128 $ak $arm"); done
done
echo "${#legs[@]} legs"

i=0
pids=()
for leg in "${legs[@]}"; do
  read -r cfg a k arm <<<"$leg"
  lo=$((4 * i)); hi=$((lo + 3))
  tag="${cfg}_a${a}_k${k}_${arm}"
  (cd "$SRC" && OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 taskset -c "$lo-$hi" \
     "$PY" scripts/compare/quant_grid.py --config "$cfg" --a-init "$a" --k "$k" --arm "$arm" \
     --out "$OUT/$tag.json" >"$OUT/legs/$tag.log" 2>&1; echo "rc=$? $tag" >>"$OUT/legs/rc.txt") &
  pids+=($!)
  i=$((i + 1))
done
wait "${pids[@]}"
echo "=== done $(date) ==="
sort "$OUT/legs/rc.txt"
grep -c "^rc=0 " "$OUT/legs/rc.txt"
ls "$OUT"/*.json | wc -l
