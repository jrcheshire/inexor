"""Stage 1 of the idle-half plan: PROVE `_insert_slab`'s disjoint-write premise.

C5 proved `eject` pools (its slab reads are free of write hazards); 5l and 5m
left `insert` owing the same proof before anything is built on it. The code
reading (2026-08-17) enumerates insert's write set as: state rows within the
destination brick's `brick_slot_range`, that brick's `occupancy` span, its
`vel_scale` entry -- all per-brick, and bricks partition into slabs -- plus ONE
shared surface, the arena CLAIM path (`_to_arena`), whose slot assignment is
order-dependent by construction (lowest-free-first). The pooled design that
follows is: workers write bricks, the PARENT applies arena claims serially in
brick order, so the pooled run stays bitwise the serial one.

This probe turns that reading into a measured census. During real chained
migrates (arena occupied, the engine's steady condition, production kernel):

- every `_write_brick(b)` issued while slab `bx` is inserting must satisfy
  `b in slab_bricks(bx)` -- CONTAINMENT, which implies cross-slab disjointness
  because bricks partition;
- every `_to_arena` claim is counted per slab -- the exceptional set the
  parent would serialize;
- a leg that never exercises the claim path says so (`arena_probed=false`)
  instead of reporting a pass over an unprobed surface (a gate that cannot
  fail is not a gate).

The census itself is mutation-tested in-process before any real leg runs
(`--self-test` is also a standalone mode): a forced out-of-slab write must
REFUSE, and the same write with the census off must not. An instrument that
cannot catch the violation it exists for reports nothing.

    pixi run python scripts/v2_m6_c13_insert_census.py --config nbs8 nbs16
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

import v2_m6_migrate_depth as md  # noqa: E402  (state import, _build, CONFIGS)
from inexor import state  # noqa: E402


class CensusViolation(AssertionError):
    pass


class Census:
    """Class-level wrap of `_insert_slab` / `_write_brick` / `_to_arena`.

    Context is the currently-inserting slab; a `_write_brick` outside any
    insert (build, repack) is recorded under `outside_insert`, never asserted
    -- this census owns exactly the premise it names.
    """

    def __init__(self):
        self.cur_bx = None
        self.per_slab_bricks = {}   # bx -> set of bricks written
        self.per_slab_claims = {}   # bx -> arena rows claimed
        self.outside_insert = 0
        self.violations = []

    def install(self, cls):
        census = self
        orig_insert, orig_write, orig_arena = (
            cls._insert_slab, cls._write_brick, cls._to_arena)

        def insert(self, bx, *a, **k):
            census.cur_bx = int(bx)
            census.per_slab_bricks.setdefault(census.cur_bx, set())
            census.per_slab_claims.setdefault(census.cur_bx, 0)
            try:
                return orig_insert(self, bx, *a, **k)
            finally:
                census.cur_bx = None

        def write_brick(self, b, *a, **k):
            if census.cur_bx is None:
                census.outside_insert += 1
            else:
                lo_b, hi_b = self.slab_bricks(census.cur_bx)
                if not (lo_b <= int(b) < hi_b):
                    census.violations.append((census.cur_bx, int(b)))
                    raise CensusViolation(
                        f"_write_brick({int(b)}) while slab {census.cur_bx} "
                        f"inserts: outside its bricks [{lo_b}, {hi_b})")
                census.per_slab_bricks[census.cur_bx].add(int(b))
            return orig_write(self, b, *a, **k)

        def to_arena(self, dest, *a, **k):
            if census.cur_bx is not None:
                census.per_slab_claims[census.cur_bx] += int(len(dest))
            return orig_arena(self, dest, *a, **k)

        cls._insert_slab, cls._write_brick, cls._to_arena = (
            insert, write_brick, to_arena)
        return lambda: setattr_all(cls, orig_insert, orig_write, orig_arena)

    def verdict(self):
        slabs = sorted(self.per_slab_bricks)
        # containment implies disjointness, but assert it independently anyway:
        # a wrong slab_bricks() would break disjointness while containment,
        # measured against the same wrong function, still "passed".
        seen = {}
        overlaps = []
        for bx in slabs:
            for b in self.per_slab_bricks[bx]:
                if b in seen and seen[b] != bx:
                    overlaps.append((b, seen[b], bx))
                seen[b] = bx
        claims = sum(self.per_slab_claims.values())
        return dict(
            n_slabs=len(slabs),
            n_bricks_written=len(seen),
            containment_violations=len(self.violations),
            cross_slab_overlaps=len(overlaps),
            arena_claims=int(claims),
            arena_probed=bool(claims > 0),
            claims_per_slab_max=int(max(self.per_slab_claims.values(), default=0)),
            outside_insert_writes=int(self.outside_insert),
            premise_holds=(not self.violations) and (not overlaps),
        )


def setattr_all(cls, ins, wr, ar):
    cls._insert_slab, cls._write_brick, cls._to_arena = ins, wr, ar


def self_test():
    """The census must catch a planted violation, and only then count.

    Built on the smallest depth-probe config; the planted arm calls
    `_write_brick` under a fake slab context with a brick the slab does not
    own. Both directions are asserted: refusal WITH the census, silence
    WITHOUT it (i.e. the violation is only detectable because the census
    exists, so the census is load-bearing).
    """
    cfg = md.CONFIGS["nbs8"]
    st = md._build(cfg["n_part"], cfg["nb"], cfg["box"],
                   brick_slack=0.10, arena_frac=0.20)
    cls = type(st)
    c = Census()
    restore = c.install(cls)
    try:
        c.cur_bx = 0
        lo_b, hi_b = st.slab_bricks(0)
        bad_b = hi_b  # first brick of the NEXT slab
        try:
            st._write_brick(bad_b, np.empty(0, np.int64),
                            np.empty((0, 3), np.uint8), np.empty((0, 3), np.int16))
        except CensusViolation:
            caught = True
        else:
            caught = False
        c.cur_bx = None
    finally:
        restore()
    # the same write, census off, must NOT raise (the violation is silent
    # without the instrument -- which is why the instrument exists)
    st2 = md._build(cfg["n_part"], cfg["nb"], cfg["box"],
                    brick_slack=0.10, arena_frac=0.20)
    st2._write_brick(bad_b, np.empty(0, np.int64),
                     np.empty((0, 3), np.uint8), np.empty((0, 3), np.int16))
    ok = caught
    print(f"self-test: planted out-of-slab write "
          f"{'CAUGHT' if caught else 'MISSED'} with census, silent without -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def run_config(name, calls, eject_kernel):
    cfg = md.CONFIGS[name]
    extent = cfg["box"] / cfg["nb"]
    st = md._build(cfg["n_part"], cfg["nb"], cfg["box"],
                   brick_slack=0.0, arena_frac=0.20)
    c1 = extent / (float(np.max(st.vel_scale)) * state.INT16_MAX)
    cls = type(st)
    census = Census()
    restore = census.install(cls)
    refused = None
    try:
        # the LARGEST in-schedule drift: maximal migrant volume and, at
        # slack 0, the strongest arena pressure this config can produce
        for _ in range(int(calls)):
            state.drift_and_migrate(
                st, 2.85 * c1,
                **({} if eject_kernel is None else {"kernel": eject_kernel}))
    except CensusViolation:
        raise
    except (RuntimeError, ValueError) as exc:
        # the D-007 arena refusal: at slack 0 a config can push more overflow
        # than its arena holds. The claims counted BEFORE the refusal still
        # probed the shared surface; record the leg as refused, keep them.
        refused = str(exc).splitlines()[0][:200]
    finally:
        restore()
    v = census.verdict()
    v.update(config=name, n_part=cfg["n_part"], nb=cfg["nb"], calls=int(calls),
             eject_kernel_requested=eject_kernel,
             eject_jax_calls=md._eject_jax_calls(), refused=refused)
    flag = "" if v["arena_probed"] else "  [ARENA NOT PROBED -- claim path unexercised]"
    print(f"  {name}: slabs {v['n_slabs']}, bricks {v['n_bricks_written']}, "
          f"violations {v['containment_violations']}, overlaps "
          f"{v['cross_slab_overlaps']}, arena claims {v['arena_claims']} "
          f"(max/slab {v['claims_per_slab_max']}) -> "
          f"premise {'HOLDS' if v['premise_holds'] else 'FAILS'}{flag}")
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", nargs="+", default=["nbs8", "nbs16"],
                    choices=sorted(md.CONFIGS))
    ap.add_argument("--calls", type=int, default=2,
                    help="chained migrates on one state; call 2+ runs with the "
                         "arena resident, the engine's steady condition")
    ap.add_argument("--eject-kernel", default="jax", choices=("numpy", "jax"),
                    help="production default; the census is kernel-independent "
                         "but the run should be the production path")
    ap.add_argument("--out",
                    default=os.path.join("runs", "v2", "m6_c13_insert_census.json"))
    ap.add_argument("--self-test", action="store_true",
                    help="run ONLY the mutation self-test")
    a = ap.parse_args(argv)

    if a.eject_kernel == "jax":
        import jax  # callers opt in; the compiled eject refuses without x64

        jax.config.update("jax_enable_x64", True)

    if not self_test():
        print("FATAL: the census cannot catch its own planted violation; "
              "nothing below would mean anything")
        return 4
    if a.self_test:
        return 0

    results = [run_config(n, a.calls, a.eject_kernel) for n in a.config]
    ok = all(r["premise_holds"] for r in results)
    probed = any(r["arena_probed"] for r in results)
    try:
        commit = subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    card = dict(results=results, premise_holds_all=ok, arena_probed_any=probed,
                commit=commit, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                argv=sys.argv[1:])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(card, fh, indent=1)
    print(f"card -> {a.out}")
    if not probed:
        print("VERDICT WITHHELD: no leg exercised the arena claim path; the "
              "shared surface went unprobed. Add a config or raise the drift.")
        return 3
    print(f"VERDICT: insert's per-brick writes are "
          f"{'DISJOINT across slabs' if ok else 'NOT disjoint -- premise FAILS'}; "
          f"the arena claim path is the one shared surface "
          f"(parent-side serialization is the pooled design).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
