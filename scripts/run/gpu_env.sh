# shellcheck shell=bash
# Sourced by the GPU job scripts. A scratch purge deletes files from an installed pixi env
# without pixi noticing, so the env is checked on the node that will use it, before any leg.

# ensure_gpu_env SRC: SRC's gpu env imports jax and inexor and sees a GPU; otherwise, or with
# REBUILD_ENV=1, it is removed (`pixi clean -e gpu`) and reinstalled from pixi.lock, then checked
# again. Returns 1 if it still fails. Jobs starting together from one checkout take turns on a
# lock (where `flock` exists), so a second job finds the env the first rebuilt; a job already
# running from the checkout loses its env during a rebuild.
ensure_gpu_env () {
  mkdir -p "$1/.pixi" || return 1
  if command -v flock >/dev/null 2>&1; then
    ( flock 9 && _ensure_gpu_env "$1" ) 9>"$1/.pixi/gpu_env.lock"
  else
    _ensure_gpu_env "$1"
  fi
}

_ensure_gpu_env () {
  local src=$1
  local py=$src/.pixi/envs/gpu/bin/python3.14
  local check="import jax, inexor; assert jax.devices()[0].platform == 'gpu', jax.devices()"
  if [ "${REBUILD_ENV:-0}" != 1 ] && "$py" -c "$check" >/dev/null 2>&1; then
    echo "gpu env: imports jax and inexor and sees a GPU"
    return 0
  fi
  echo "gpu env: rebuilding from pixi.lock (REBUILD_ENV=${REBUILD_ENV:-0})"
  (cd "$src" && pixi clean -e gpu && pixi install -e gpu --locked) || {
    echo "gpu env: the rebuild failed"; return 1; }
  "$py" -c "$check" || { echo "gpu env: still fails after the rebuild"; return 1; }
  echo "gpu env: rebuilt"
}
