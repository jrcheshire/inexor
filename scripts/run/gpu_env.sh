# shellcheck shell=bash
# Sourced by the GPU job scripts. A scratch purge deletes files from an installed pixi env (and
# from pixi's package cache) without pixi noticing, so the env is checked on the node that will
# use it, before any leg.

# ensure_gpu_env SRC: SRC's gpu env is complete (`env_census`) and imports jax, inexor and every
# inexor module, with a GPU visible; otherwise, or with REBUILD_ENV=1, it is removed (`pixi
# clean -e gpu`) and reinstalled from pixi.lock through a new, empty package cache (a damaged
# cache would be linked straight back in), then checked again. Returns 1 if it still fails.
# Jobs starting together from one checkout take turns on a lock (where `flock` exists), so a
# second job finds the env the first rebuilt; a job already running from the checkout loses
# its env during a rebuild. The cache goes under $GPU_ENV_CACHE_ROOT (default $SCRATCH).
ensure_gpu_env () {
  mkdir -p "$1/.pixi" || return 1
  if command -v flock >/dev/null 2>&1; then
    ( flock 9 && _ensure_gpu_env "$1" ) 9>"$1/.pixi/gpu_env.lock"
  else
    _ensure_gpu_env "$1"
  fi
}

_gpu_env_ok () {
  local env=$1/.pixi/envs/gpu
  env_census "$env" || return 1
  "$env/bin/python3.14" -c "
import importlib, pkgutil
import jax, inexor
for m in pkgutil.walk_packages(inexor.__path__, 'inexor.'):
    importlib.import_module(m.name)
assert jax.devices()[0].platform == 'gpu', jax.devices()
" >/dev/null 2>&1 || { echo "gpu env: jax, inexor or an inexor module does not import, or no GPU"; return 1; }
}

_ensure_gpu_env () {
  local src=$1
  if [ "${REBUILD_ENV:-0}" != 1 ] && _gpu_env_ok "$src"; then
    echo "gpu env: complete; imports jax and every inexor module and sees a GPU"
    return 0
  fi
  local cache
  cache=${GPU_ENV_CACHE_ROOT:-${SCRATCH:-$src/.pixi}}/pixi-cache-${SLURM_JOB_ID:-$$}-$(date +%s)
  echo "gpu env: rebuilding from pixi.lock through the empty cache $cache (REBUILD_ENV=${REBUILD_ENV:-0})"
  mkdir -p "$cache" || return 1
  (cd "$src" && pixi clean -e gpu && PIXI_CACHE_DIR=$cache pixi install -e gpu --locked) || {
    echo "gpu env: the rebuild failed"; return 1; }
  _gpu_env_ok "$src" || { echo "gpu env: still fails after the rebuild"; return 1; }
  echo "gpu env: rebuilt"
}

# env_census ENV: every file ENV's conda-meta manifests list exists (a symlink must resolve).
# Finds files lost from the env after its install; it cannot see a file the install never
# linked, which is why a rebuild uses an empty cache.
env_census () {
  python3 - "$1" <<'EOF'
import glob, json, os, sys
env = sys.argv[1]
total, missing = 0, []
for meta in sorted(glob.glob(os.path.join(env, "conda-meta", "*.json"))):
    with open(meta) as f:
        files = json.load(f).get("files", [])
    total += len(files)
    missing += [p for p in files if not os.path.exists(os.path.join(env, p))]
if missing or total == 0:
    print(f"env census: {len(missing)} of {total} listed files missing in {env}")
    for p in missing[:10]:
        print("  missing:", p)
    sys.exit(1)
EOF
}

# pip_target_complete DIR: DIR (a `pip install --target` tree) has every file its dist-info
# RECORDs list, and at least one RECORD.
pip_target_complete () {
  python3 - "$1" <<'EOF'
import csv, glob, os, sys
d = sys.argv[1]
records = glob.glob(os.path.join(d, "*.dist-info", "RECORD"))
missing = []
for rec in records:
    with open(rec, newline="") as f:
        missing += [row[0] for row in csv.reader(f)
                    if row and not os.path.exists(os.path.join(d, row[0]))]
if missing or not records:
    print(f"pip target {d}: {len(missing)} files missing, {len(records)} RECORD(s)")
    for p in missing[:10]:
        print("  missing:", p)
    sys.exit(1)
EOF
}
