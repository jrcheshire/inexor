#!/usr/bin/env bash
# The laptop MPI lane: the tests that need real MPI processes, in a throwaway `pixi exec` env
# pinned to pixi.lock's python / jax / jaxlib / numpy, plus mpi4py, MPICH and pytest. The
# lock and the default env are untouched. The checkout is installed (no deps) into
# runs/mpi-lane/site, so `inexor` imports as the package it is on the cluster.
#
#   scripts/run/mpi_lane.sh                      # the MPI test files
#   scripts/run/mpi_lane.sh tests/test_comm_mpi.py -k raise
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
site="$here/runs/mpi-lane/site"
mkdir -p "$site"

pins=$(cd "$here" && pixi list --frozen --json | python3 -c '
import json, sys
v = {p["name"]: p["version"] for p in json.load(sys.stdin)}
print(" ".join(f"--spec {k}=={v[k]}" for k in ("python", "jax", "jaxlib", "numpy")))')

if [ "$#" -eq 0 ]; then
  set -- tests/test_comm_mpi.py
fi
cd "$here"
# shellcheck disable=SC2086
exec pixi exec $pins --spec mpi4py --spec mpich --spec pytest --spec pip --spec hatchling -- \
  bash -c 'pip install -q --no-deps --no-build-isolation --upgrade --target "$0" "$1" >/dev/null &&
           PYTHONPATH="$0:$1" OMP_NUM_THREADS=1 python -m pytest -q -p no:cacheprovider "${@:2}"' \
  "$site" "$here" "$@"
