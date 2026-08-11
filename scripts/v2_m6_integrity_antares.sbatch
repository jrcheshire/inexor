#!/bin/bash
#SBATCH -J m6-integ
#SBATCH -p compute
#SBATCH --mem=100G
#SBATCH -c 8
#SBATCH -t 00:40:00
#SBATCH -o m6-integ-%j.log
# M-v2-6: FIND the row the cdev partition assertion is missing, rather than
# theorise about it a fourth time.
#
# cdev fails `the tiles own 16777215 rows against 16777216 particles` on x86 and
# not on Apple arm64. I have proposed three causes from reading the code and been
# wrong twice: a tile-local float ulp gap (job 431 -> 436 still failed), then a
# position-vs-storage partition mismatch (436 -> 437 still failed). A third
# reading, that the container had lost a row, is also wrong -- `SlotState.check`
# asserts reachability at `state.py:587` and `load_slot_state` calls it.
#
# So this measures instead. WORKDIR-ROOT POINTS AT JOB 437's STAGED SLABS on
# purpose: the defect is realization-dependent, so it has to be the SAME
# realization, and reusing them also skips generation entirely.
#
# --mem=100G places this on antares (124 GB) rather than deneb-the-node (56 GB),
# expressed as a resource rather than a hostname.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
WD="${WD:-/tmp/m6_cdev_437}"
[ -f "$WD/cdev/manifest.json" ] || {
  echo "FATAL: no staged cdev slabs under $WD -- the failing realization is gone;"
  echo "       regenerate with the full job rather than a different realization."
  exit 1
}
echo "=== $(hostname), commit $(git rev-parse --short HEAD), slabs $WD ==="
pixi run -e default python scripts/v2_m6_engine_peak.py \
  --configs cdev --legs integrity --workdir-root "$WD" --out-suffix _integ
rc=$?
echo "=== integrity rc=$rc ==="
exit $rc
