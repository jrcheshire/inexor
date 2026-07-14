"""M2 S4: attribute the adjoint's peak memory to NAMED buffers.

Vista job 831303 + deneb job 19 established the shape of the problem:
  - peak is FLAT in K (K=5..40, slope 0.0 B/particle) -> the residual really is
    O(1) in steps; architecture Sec. 8's premise HOLDS.
  - but peak is ~13x the analytic carry at every size 64^3..512^3 (a clean
    per-particle constant): ICs 76, forward 138, adjoint 466 B/particle, where
    Sec. 9 budgets ~68 total (36 carry + ~32 transients).
  - so the bwd adds ~330 B/particle over the forward, flat in K = ONE step's
    VJP working set. The 1024^3 flagship projects to ~466 GiB vs an 80 GB claim.
The architecture is sound; the transient budget is wrong by ~14x. This script
turns "something is fat" into a ranked list of named buffers.

Tool choice: XLA's own buffer assignment (`--xla_dump_to`), NOT nsys/ncu and not
jax.profiler.device_memory_profile.
  - nsys traces the CUDA API, but JAX preallocates ONE arena
    (XLA_PYTHON_CLIENT_MEM_FRACTION) and sub-allocates inside it with BFC, so
    nsys would report a single cudaMalloc and say nothing about what is inside.
    The buffers we need to name are invisible at that layer.
  - ncu reports kernel perf counters, not allocations -- wrong question -- and is
    blocked on Vista compute anyway (perf counters are admin-only ->
    ERR_NVGPUCTRPERM; see xphot scripts/vista_gh_e2e_bench.sbatch, which also
    documents py-spy being ptrace-blocked there).
  - device_memory_profile emits pprof BINARY, needing the Go pprof tool to read,
    and snapshots live buffers at a moment rather than the in-computation peak
    we care about.
Buffer assignment is a flag plus a text file: no new env, no new dependency, and
it names XLA temporaries inside the scan, which is exactly where the ~330
B/particle lives.

XLA_FLAGS is read at backend init, so this script sets it BEFORE importing jax
(argparse runs first, deliberately).

Compute: deneb (free). 128^3 K=5 is enough -- the quantity is per-particle and
K-independent, both measured.
    sbatch scripts/m2_mem_attrib.sbatch
Outputs a ranked table + runs/m2/mem_attrib.json. The dump itself is kept for
reading. Sets no policy; proposes no fix.
"""

import argparse
import glob
import json
import os
import re

# ---- flags BEFORE jax import (backend init reads XLA_FLAGS once) ----
_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("--n", type=int, default=128)
_ap.add_argument("--steps", type=int, default=5)
_ap.add_argument("--dump-dir", default=None, help="default: <repo>/runs/m2/xla_dump")
_ap.add_argument("--top", type=int, default=25, help="how many buffers to list")
_args = _ap.parse_args()

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUNS = os.path.join(REPO, "runs", "m2")
DUMP = _args.dump_dir or os.path.join(RUNS, "xla_dump")
os.makedirs(DUMP, exist_ok=True)
os.environ["XLA_FLAGS"] = (
    f"{os.environ.get('XLA_FLAGS', '')} --xla_dump_to={DUMP} --xla_dump_hlo_as_text"
).strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from inexor import PLANCK, BoxConfig, QuantConfig, TimeConfig  # noqa: E402
from inexor.adjoint import evolve_grad  # noqa: E402
from inexor.ic import gaussian_delta  # noqa: E402
from inexor.lpt import lpt_ics  # noqa: E402

GIB = 1024.0**3


def run_adjoint(n_mesh, K):
    box = BoxConfig(n_mesh=n_mesh, box_size=256.0)
    tcfg = TimeConfig(a_init=0.1, a_final=1.0, n_steps=K, integrator="bullfrog")
    d0 = gaussian_delta(jax.random.PRNGKey(0), n_mesh, box.box_size, PLANCK, fdtype=jnp.float32)
    x0, v0 = lpt_ics(d0, box.box_size, 0.1, PLANCK, order=2, fdtype=jnp.float32)

    def loss(a, b):
        xf, vf = evolve_grad(box, tcfg, QuantConfig(), PLANCK, "perstep", jnp.float32, a, b)
        return jnp.sum(xf**2) + 0.5 * jnp.sum(vf**2)

    g = jax.grad(loss, argnums=(0, 1))(x0, v0)
    jax.block_until_ready(g)
    return int(x0.shape[0])


# ---- buffer-assignment parsing -------------------------------------------
# Real XLA format (verified against a local dump, not assumed):
#   allocation 0: size 393216, maybe-live-out:
#    value: <4196 multiply_add_fusion.1 @0> (size=393216,offset=0): f32[32768,3]{1,0}
#    value: <4211 copy.57 @0>               (size=393216,offset=0): f32[32768,3]{1,0}
#
# CRITICAL: an `allocation` is a REUSED SLOT and many values time-share it (note
# the repeated offset=0). The peak is therefore the sum of ALLOCATION sizes;
# summing `value` sizes double-counts wildly -- five names at 12 B/particle above
# are ONE 12 B/particle buffer. So rank allocations, and name each by the values
# living in it. (The sibling *-buffer-assignment-values.txt is a DIFFERENT
# format -- a value/position listing -- and must not be globbed in.)
_ALLOC_HDR = re.compile(r"^allocation (\d+): size (\d+),\s*(.*)$")
_VALUE = re.compile(r"^\s+value: <\d+ ([^>]+?) @\d+> \(size=(\d+),offset=(\d+)\): (\S+)")


def _parse_one(path):
    """-> (allocations, sum_bytes). Each allocation: size + the values sharing it."""
    allocs = []
    cur = None
    with open(path, errors="replace") as f:
        for line in f:
            h = _ALLOC_HDR.match(line)
            if h:
                cur = {"id": int(h.group(1)), "size": int(h.group(2)),
                       "flags": h.group(3).strip().rstrip(":"), "values": []}
                allocs.append(cur)
                continue
            v = _VALUE.match(line)
            if v and cur is not None:
                cur["values"].append({"name": v.group(1), "size": int(v.group(2)),
                                      "offset": int(v.group(3)), "shape": v.group(4)})
    return allocs, sum(a["size"] for a in allocs)


def parse_dump(dump_dir):
    # Exact suffix: the -values.txt sibling is a different format.
    files = [p for p in glob.glob(os.path.join(dump_dir, "*buffer-assignment.txt"))]
    if not files:
        return None, f"no *buffer-assignment.txt in {dump_dir}"
    mods = []
    for path in files:
        allocs, total = _parse_one(path)
        if not allocs:
            continue
        mods.append({"file": os.path.basename(path), "n_allocations": len(allocs),
                     "peak_bytes": total, "allocs": allocs})
    if not mods:
        return None, f"{len(files)} file(s) found but no `allocation N: size` lines parsed"
    mods.sort(key=lambda m: m["peak_bytes"], reverse=True)
    return mods, None


def main():
    npart = run_adjoint(_args.n, _args.steps)
    dev = jax.devices()[0]
    print(f"=== m2_mem_attrib: {dev.platform} / {dev} / n={_args.n} K={_args.steps} ===")
    print(f"    n_particles={npart}   dump={DUMP}")

    mods, err = parse_dump(DUMP)
    if err:
        print(f"\nPARSE FAILED: {err}")
        print("The dump is on disk; read it directly rather than trusting an empty table.")
        return

    print(f"\n--- XLA modules by peak (sum of allocation slots); {len(mods)} dumped ---")
    for m in mods[:6]:
        print(f"  {m['peak_bytes'] / GIB:8.4f} GiB  {m['peak_bytes'] / npart:8.1f} B/particle"
              f"  {m['n_allocations']:>4} slots   {m['file'][:58]}")

    big = mods[0]
    print(f"\n--- largest module: {big['file']} ---")
    for a in sorted(big["allocs"], key=lambda x: x["size"], reverse=True)[:4]:
        print(f"  slot {a['id']:>3}: {a['size'] / npart:8.1f} B/particle  "
              f"{len(a['values']):>3} values   {a['flags'][:28]}")

    # THE ONLY VALID DECOMPOSITION available from this dump.
    #
    # Allocations are disjoint by construction, so summing ALLOCATION sizes is
    # exact and sums to the module peak. Do NOT decompose an allocation by
    # grouping its values by offset: XLA assigns OVERLAPPING address ranges to
    # values whose live ranges are disjoint in time, so distinct offsets do not
    # imply coexistence and "sum of max size per offset" OVERCOUNTS -- it can
    # exceed the allocation it claims to decompose (it did: 333 vs 240 B/particle
    # at n=128, which is how the bug was caught). Exactly decomposing an arena
    # would need a heap simulation over live ranges; this XLA build dumps no
    # peak-live set.
    #
    # So: label each allocation by its largest occupant's shape. Where an
    # allocation holds values at several offsets it is an ARENA holding many
    # shapes, and the label is a label, not a claim about its contents -- those
    # rows are marked and excluded from the shape census rather than mislabelled.
    for a in big["allocs"]:
        occ = max(a["values"], key=lambda v: v["size"], default=None)
        a["shape"] = occ["shape"].split("{")[0] if occ else "?"
        a["name"] = occ["name"] if occ else "(none)"
        a["is_arena"] = len({v["offset"] for v in a["values"]}) > 1

    allocs = sorted(big["allocs"], key=lambda a: a["size"], reverse=True)
    print(f"\n--- top {_args.top} ALLOCATIONS (disjoint; these sum to the module peak) ---")
    for a in allocs[: _args.top]:
        tag = "ARENA (mixed shapes)" if a["is_arena"] else a["shape"]
        print(f"  {a['size'] / npart:8.1f} B/p  {tag:<24} {a['name'][:30]:<30} "
              f"{len(a['values']):>3} vals  {a['flags'][:16]}")

    kinds, arena = {}, 0
    for a in big["allocs"]:
        if a["is_arena"]:
            arena += a["size"]
        else:
            kinds[a["shape"]] = kinds.get(a["shape"], 0) + a["size"]
    total = big["peak_bytes"]
    print("\n--- module peak by array shape (sums EXACTLY to the module peak) ---")
    for k, b in sorted(kinds.items(), key=lambda kv: kv[1], reverse=True)[:10]:
        print(f"  {b / npart:8.1f} B/particle  {100 * b / total:5.1f}%   {k}")
    if arena:
        print(f"  {arena / npart:8.1f} B/particle  {100 * arena / total:5.1f}%   "
              f"(arenas -- mixed shapes, not attributable from this dump)")
    print(f"  {'-' * 46}\n  {total / npart:8.1f} B/particle  100.0%   MODULE PEAK")
    print("\n  NB: this is ONE module's buffer assignment, not the process peak.")
    print("  Compare against the measured process peak (m2_mem_profile): the")
    print("  remainder lives in ICs, the carry, the loss, other modules, fragmentation.")

    os.makedirs(RUNS, exist_ok=True)
    # n in the filename: a fixed path lets a later size silently overwrite an
    # earlier one, and the census is exactly what we compare ACROSS sizes.
    path = os.path.join(RUNS, f"mem_attrib_n{_args.n}.json")
    with open(path, "w") as f:
        json.dump({
            "device": str(dev), "n_mesh": _args.n, "K": _args.steps, "n_particles": npart,
            "modules": [{"file": m["file"], "n_allocations": m["n_allocations"],
                         "peak_bytes": m["peak_bytes"],
                         "b_per_particle": m["peak_bytes"] / npart} for m in mods],
            "top_allocations": [{"size": a["size"], "b_per_particle": a["size"] / npart,
                                "shape": a["shape"], "name": a["name"],
                                "is_arena": a["is_arena"]} for a in allocs[: _args.top]],
            "peak_by_shape": {k: v for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])},
            "arena_bytes": arena, "module_peak_bytes": total,
        }, f, indent=1)
    print(f"\nwrote {path}")
    print("\n  Reference: Sec. 9 budgets ~68 B/particle total (36 carry + ~32 transients);")
    print("  measured adjoint peak is ~466. Anything here at tens of B/particle is a lead.")


if __name__ == "__main__":
    main()
