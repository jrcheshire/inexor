#!/bin/bash
#SBATCH -J m6-shm-probe
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o m6-shm-probe-%j.log
# WHY THIS EXISTS. The realization pilot (920910) generated its 2048^3 ICs
# (rc=0, 90 min) and then died 196 s into stepping with a bus error
# (SIGBUS, rc=135), immediately after `resource_tracker` reported 6 leaked
# shared_memory objects -- 6 being exactly `SHM_FIELDS`.
#
# `TilePool.__init__` puts the whole slot state plus the three coarse force
# meshes into POSIX shared memory, which on Linux is `/dev/shm`. Priced at
# c-gh from `inexor.plan`: off+w 102.05 GB, arena_bucket 0.69, the small brick
# arrays ~0.05, the three coarse meshes 12.89 -- 115.7 GB at construction, and
# 128.4 GB once `stage_migrate` adds its 12.75 GB of scratch. `SharedMemory`
# sizes lazily, so an oversubscribed tmpfs faults on TOUCH, not on create,
# which is exactly the observed failure shape and timing.
#
# The Vista LOGIN node carries /dev/shm at 119 GB against 249 GB of RAM, the
# kernel default of half. THIS JOB MEASURES THE SAME THING ON A gg COMPUTE
# NODE, because a cause inferred from a different node class is not measured.
#
# WHAT IT ESTABLISHES (and nothing else -- it runs no inexor code):
#   a. the gg node's /dev/shm SIZE and free space, and its MemTotal
#   b. that the size is REAL, by reserving it: fallocate on tmpfs commits
#      pages, so a probe that fails names the true ceiling rather than the
#      advertised one. Probes at 100 / 110 / 116 / 120 / 130 GB bracket both
#      the 115.7 GB construction figure and the 128.4 GB peak.
#   c. whether /tmp on a compute node is tmpfs (and so shares the budget) or
#      a real device, which decides whether a file-backed mmap is a fix.
# Cost: one gg node for under a minute, ~0.01 SU. No pixi, no jax, no state.

set -u
echo "=== $(hostname): $(date -Is) ==="

echo "--- (a) the budget ---"
df -h /dev/shm
grep -E '^(MemTotal|MemFree|MemAvailable|Shmem):' /proc/meminfo
echo "shm mount options:"
grep ' /dev/shm ' /proc/mounts

echo
echo "--- (b) is the advertised size real? ---"
# Each probe is independent: allocate, report, release. fallocate on tmpfs
# commits the pages, so success here means the run could have had them.
for GB in 100 110 116 120 130; do
  if fallocate -l ${GB}G /dev/shm/inexor_shm_probe 2>/dev/null; then
    echo "  ${GB} GB: RESERVED   ($(df -h --output=avail /dev/shm | tail -1) left)"
  else
    echo "  ${GB} GB: REFUSED"
  fi
  rm -f /dev/shm/inexor_shm_probe
done
echo "after the probes, /dev/shm is clean:"
ls -la /dev/shm/ | head

echo
echo "--- (c) is /tmp a separate budget? ---"
df -hT /tmp
grep -E ' /tmp ' /proc/mounts || echo "  /tmp not separately mounted"

echo
echo "=== READ IT LIKE THIS ==="
echo "  115.7 GB is what TilePool asks for at construction;"
echo "  128.4 GB is the peak once stage_migrate adds its scratch."
echo "  A /dev/shm at ~118 GB explains the SIGBUS and the 196 s."
echo "  If /dev/shm is much LARGER than 118 GB, this cause is WRONG"
echo "  and the bus error needs a different explanation."
