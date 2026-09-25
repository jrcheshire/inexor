"""What does the host's arena replay, and repack, cost per step: per brick or per row?

A device migrate defers every order-dependent arena mutation to one host replay
at the end of the pass, as `drift_and_migrate_pooled` does (`state.py:711-771`):
`_release_brick_arena` for every brick in slab order, then `_to_arena` for each
spilled brick's rows. At 4096^3 that is 16.8M release calls per step, and
`repack` is a whole-state host pass every step. This times both on a state with
arena residents, particles FIXED and bricks varied, so a per-brick term moves and
a per-row term does not (the design of record sec. 28).

Per rung: build at zero brick slack, one numpy migrate so residents exist, then
on copies of that state:
  releases  `_release_brick_arena(b)` over every brick, ascending.
  claims    the residents' rows handed back to `_to_arena`, one call per brick in
            ascending brick order (the replay's grouping), after the releases.
  repack    `SlotState.repack` on the post-migrate state.
Wall-clock on whatever machine runs it; no profiler.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))


def _probe():
    path = os.path.join(REPO, "scripts", "v2_d3_device_migrate.py")
    spec = importlib.util.spec_from_file_location("d3probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rung(P, n_part, nb, frac):
    from inexor import state

    st = P._build_state(n_part, nb, seed=13, brick_slack=0.0, arena_frac=0.05, with_ids=False)
    state.drift_and_migrate(st, P._c_drift(st, frac))
    p3 = int(st.buckets_per_brick)
    live = np.flatnonzero(st.arena_bucket >= 0)
    rows = st.arena_base + live
    keys = st.arena_bucket[live]
    order = np.argsort(keys // p3, kind="stable")
    rows, keys = rows[order], keys[order]
    dest, off, w = keys.copy(), st.off[rows].copy(), st.w[rows].copy()
    bricks = keys // p3
    groups = np.split(np.arange(len(keys)), np.flatnonzero(np.diff(bricks)) + 1) \
        if len(keys) else []

    a = copy.deepcopy(st)
    a._build_arena_index()
    t0 = time.perf_counter()
    for b in range(a.n_bricks):
        a._release_brick_arena(b)
    rel_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    for g in groups:
        a._to_arena(dest[g], off[g], w[g], None)
    claim_s = time.perf_counter() - t1

    r = copy.deepcopy(st)
    t2 = time.perf_counter()
    r.repack(brick_slack=0.10)
    repack_s = time.perf_counter() - t2

    rec = dict(n_part=n_part, nb=nb, n_bricks=st.n_bricks, rows=n_part**3,
               arena_residents=int(len(keys)), bricks_with_residents=len(groups),
               release_s=rel_s, release_us_per_brick=1e6 * rel_s / st.n_bricks,
               claim_s=claim_s,
               claim_us_per_group=1e6 * claim_s / max(len(groups), 1),
               claim_ns_per_row=1e9 * claim_s / max(len(keys), 1),
               repack_s=repack_s, repack_ns_per_row=1e9 * repack_s / n_part**3,
               repack_us_per_brick=1e6 * repack_s / st.n_bricks)
    print(f"[nb={nb}] {st.n_bricks:,} bricks, {len(keys):,} residents in {len(groups):,} "
          f"bricks: releases {rel_s:.3f}s ({rec['release_us_per_brick']:.2f} us/brick); "
          f"claims {claim_s:.3f}s ({rec['claim_us_per_group']:.1f} us/brick group, "
          f"{rec['claim_ns_per_row']:.1f} ns/row); repack {repack_s:.3f}s "
          f"({rec['repack_ns_per_row']:.1f} ns/row, {rec['repack_us_per_brick']:.2f} us/brick)",
          flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-part", type=int, default=256)
    ap.add_argument("--nbs", default="8,16,32")
    ap.add_argument("--frac", type=float, default=1.9,
                    help="drift of the fastest row in bricks, for arena residents")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args(argv)
    P = _probe()
    out = os.path.join(REPO, "runs", "v2", f"d3_replay_cost{args.out_suffix}.json")
    card = dict(argv=sys.argv, started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                node=os.uname().nodename,
                commit=subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
                rungs=[])
    for nb in (int(x) for x in args.nbs.split(",")):
        card["rungs"].append(rung(P, args.n_part, nb, args.frac))
        with open(out, "w") as f:
            json.dump(card, f, indent=1)
    card["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(out, "w") as f:
        json.dump(card, f, indent=1)
    print(f"card: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
