#!/usr/bin/env bash
# One rank's entry under a process launcher (ibrun, mpiexec): starts this node's samplers and
# then becomes the command (exec), so the launcher's signals (the pre-wall USR1, the kill on a
# failed rank) reach the command itself.
#
#   rank_exec.sh [--membind NODES] [--samples PREFIX] -- CMD [ARGS...]
#
# - The rank is PMI_RANK (set by hydra-based launchers); it refuses to start without it.
# - Every `@RANK@` in PREFIX, in CMD and ARGS, and in any exported variable's value is
#   replaced by the rank, so one launch line gives each rank its own card, checkpoint dir or
#   compilation cache.
# - `--membind NODES` runs the command under `numactl --membind=NODES`.
# - `--samples PREFIX` writes, every 5 s while the command runs, PREFIX_gpu.csv (epoch,
#   nvidia-smi index, memory.used MiB, utilization %; skipped without nvidia-smi) and
#   PREFIX_mem.csv (epoch, NUMA node, has_cpus, MemTotal, MemFree, FilePages, Dirty,
#   Writeback, Mlocked, all kB; skipped without /sys NUMA nodes) -- the formats of the
#   single-node job scripts, so `device_run.py summarize` reads them. The samplers stop when
#   the command's process is gone or a zombie, and hold none of the launcher's pipes: hydra
#   waits for those to close before it reaps the rank, and an unreaped rank is still a pid.
set -euo pipefail

membind="" samples=""
while [ $# -gt 0 ]; do
  case "$1" in
    --membind) membind=$2; shift 2 ;;
    --samples) samples=$2; shift 2 ;;
    --) shift; break ;;
    *) echo "rank_exec: unknown option $1 (the command follows --)" >&2; exit 2 ;;
  esac
done
[ $# -gt 0 ] || { echo "rank_exec: no command after --" >&2; exit 2; }
rank=${PMI_RANK:-}
[ -n "$rank" ] || { echo "rank_exec: PMI_RANK is unset (not under a hydra launcher?)" >&2; exit 2; }

args=()
for a in "$@"; do args+=("${a//@RANK@/$rank}"); done
while IFS= read -r name; do
  v=${!name-}
  case "$v" in *@RANK@*) export "$name=${v//@RANK@/$rank}" ;; esac
done < <(compgen -e)
samples=${samples//@RANK@/$rank}

# the command keeps this shell's pid after the exec below
me=$$
alive () {
  local s
  s=$(ps -o stat= -p "$1" 2>/dev/null) || return 1
  case "$s" in *Z*|"") return 1 ;; esac
}
# a sampler's first act: let go of every inherited descriptor but its own stdout/stderr
detach () {
  local fd
  exec </dev/null
  for fd in /dev/fd/*; do
    fd=${fd##*/}
    if [ "$fd" -gt 2 ] 2>/dev/null; then { eval "exec $fd>&-"; } 2>/dev/null || true; fi
  done
}
if [ -n "$samples" ]; then
  mkdir -p "$(dirname "$samples")"
  if command -v nvidia-smi >/dev/null; then
    ( detach; while alive "$me"; do t=$(date +%s.%N)
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
          2>/dev/null | sed "s/^/$t, /"
        sleep 5; done ) > "${samples}_gpu.csv" 2>/dev/null &
  fi
  if compgen -G "/sys/devices/system/node/node[0-9]*" >/dev/null; then
    ( detach; while alive "$me"; do t=$(date +%s.%N)
        for d in /sys/devices/system/node/node[0-9]*; do
          n=${d##*node}; c=0; [ -n "$(tr -d '[:space:]' < "$d/cpulist")" ] && c=1
          awk -v t="$t" -v n="$n" -v c="$c" '/MemTotal:/{tot=$4} /MemFree:/{fr=$4}
            /FilePages:/{fp=$4} / Dirty:/{di=$4} / Writeback:/{wb=$4} /Mlocked:/{ml=$4}
            END{print t "," n "," c "," tot "," fr "," fp "," di "," wb "," ml}' "$d/meminfo"
        done
        sleep 5; done ) > "${samples}_mem.csv" 2>/dev/null &
  fi
fi

echo "rank_exec: rank $rank on $(hostname): ${args[*]}" >&2
if [ -n "$membind" ]; then
  exec numactl --membind="$membind" "${args[@]}"
fi
exec "${args[@]}"
