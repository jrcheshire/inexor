"""Is the migration's write-back O(N x n_bricks^2)? A one-axis test.

## The reading under test

Job 455 measured cgh64 (512^3) at **2622.6 s/step** against cdev's (256^3)
61 s -- 43x the wall for 8x the particles, and 3.3x past the upper end of the
band that job pre-registered. Reading `SlotState._insert_slab` afterwards gives
a candidate cause with an arithmetic prediction:

    for b in range(lo_b, hi_b):            # nb^2 bricks in this slab
        sel_k = keep["dest"] // p3 == b    # FULL SCAN of the slab's keepers
        sel_i = imm["dest"] // p3 == b     # FULL SCAN of the slab's immigrants

Each brick scans every row in the slab. A slab holds ~N/nb rows and contains
nb^2 bricks, and there are nb slabs, so the comparisons come to

    nb slabs  x  nb^2 bricks  x  N/nb rows  =  N x nb^2

which, since nb grows as N^(1/3), is **N^(5/3)** rather than N. Putting the two
measured configurations in: cdev (N=1.68e7, nb=16, reach 2) against cgh64
(N=1.34e8, nb=32, reach 3) predicts 42x. Measured 43x.

**That agreement is why this script exists rather than a patch.** One ratio
matching one derivation is not a measurement -- this project's record is five
wrong causes proposed and three shipped before one was measured first -- and the
configuration ladder cannot settle it, because particles, bricks and coarse
cells all move together on it (M-v2-6, 2026-08-13: 64.00x / 64.00x / 61.18x from
smoke to cdev8, degenerate by construction). Only a one-axis arm can.

## The arms

**A (the claim): particles FIXED, brick count varied.** If the term is N x nb^2,
insert time quadruples per doubling of nb while nothing else moves.

**B (the control): brick count FIXED, particles varied.** The same term is
LINEAR in N at fixed nb. If insert time rises superlinearly here too, the
mechanism is not the one described above and the whole reading is wrong.

**The confound this design exists to kill.** `brick_reach` is
`ceil(|c_drift| * vel_scale * INT16_MAX / (box / nb))`, so at a fixed drift the
staging depth grows LINEARLY with nb -- and deeper staging means more immigrant
rows to scan. An arm that let reach float would measure nb^2 x reach and could
not separate them. So the drift is chosen per arm to pin reach at 1, and the
realized reach is reported per rung and asserted: **a rung whose reach is not 1
is void, not noisy.** With reach pinned and the drift tiny, almost nothing
migrates and `imm` is nearly empty -- which is the point. The keeper scan does
not depend on migration at all, and it is the term claimed to dominate.

`eject` is timed beside `insert` as a second control: it decodes each brick once,
so it is linear in N and independent of nb. If it moves with nb, the harness is
measuring something other than what it names.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

from inexor import state  # noqa: E402
from inexor.codec import T9Layout  # noqa: E402

import v2_m5_ic_gate as m5  # noqa: E402

SEED = 7
BOX = 128.0
BUCKET_CELLS = 2


def _build(n_part, nb, seed=SEED):
    """A uniform-random state at (n_part, nb). Positions are random rather than
    grid-like on purpose: a lattice puts every particle at a bucket centre and
    the migration then moves an unrepresentative number of them."""
    rng = np.random.default_rng(seed)
    n = int(n_part) ** 3
    x = rng.uniform(0.0, BOX, size=(n, 3))
    v = rng.normal(scale=1.0, size=(n, 3))
    t9 = T9Layout(BOX, int(n_part), BUCKET_CELLS)
    st = state.SlotState.build(x, v, t9, int(nb), arena_frac=0.05)
    return st


def _drift_for_reach_one(st, nb):
    """The largest drift that keeps `brick_reach` at 1, minus a margin.

    Derived from the bound rather than searched: reach is
    `ceil(|c| * s * INT16_MAX / extent)` with `extent = box / nb`, so any
    `|c| <= 0.9 * extent / (s * INT16_MAX)` gives reach 1. Solving it rather
    than tuning it is what makes the pin reproducible across rungs.
    """
    s = float(np.max(st.vel_scale))
    extent = BOX / float(nb)
    return 0.9 * extent / (s * state.INT16_MAX)


def _timed_migrate(st, c_drift):
    """`drift_and_migrate` with eject and insert timed separately.

    Wraps the two methods on the CLASS for the duration of one call. The timing
    lives here rather than in the package: an instrument that ships inside the
    thing it measures is one more thing that can move a number.
    """
    cls = type(st)
    orig_e, orig_i = cls._eject_slab, cls._insert_slab
    acc = {"eject_s": 0.0, "insert_s": 0.0, "n_eject": 0, "n_insert": 0}

    def eject(self, *a, **k):
        t0 = time.perf_counter()
        out = orig_e(self, *a, **k)
        acc["eject_s"] += time.perf_counter() - t0
        acc["n_eject"] += 1
        return out

    def insert(self, *a, **k):
        t0 = time.perf_counter()
        out = orig_i(self, *a, **k)
        acc["insert_s"] += time.perf_counter() - t0
        acc["n_insert"] += 1
        return out

    cls._eject_slab, cls._insert_slab = eject, insert
    try:
        t0 = time.perf_counter()
        stats = state.drift_and_migrate(st, c_drift)
        acc["total_s"] = time.perf_counter() - t0
    finally:
        cls._eject_slab, cls._insert_slab = orig_e, orig_i
    acc["brick_reach"] = int(stats["brick_reach"])
    acc["brick_reach_realized"] = int(stats["brick_reach_realized"])
    acc["arena_used"] = int(stats["arena_used"])
    return acc


def _rung(n_part, nb, repeats):
    st = _build(n_part, nb)
    c = _drift_for_reach_one(st, nb)
    runs = []
    for _ in range(int(repeats)):
        runs.append(_timed_migrate(st, c))
    ins = [r["insert_s"] for r in runs]
    ejs = [r["eject_s"] for r in runs]
    out = dict(
        n_part=int(n_part),
        n_particles=int(n_part) ** 3,
        bricks_per_side=int(nb),
        n_bricks=int(nb) ** 3,
        c_drift=float(c),
        repeats=int(repeats),
        insert_s=float(np.median(ins)),
        insert_s_min=float(np.min(ins)),
        insert_s_max=float(np.max(ins)),
        eject_s=float(np.median(ejs)),
        total_s=float(np.median([r["total_s"] for r in runs])),
        brick_reach=runs[-1]["brick_reach"],
        brick_reach_realized=runs[-1]["brick_reach_realized"],
        n_insert_calls=runs[-1]["n_insert"],
    )
    print(
        f"  n_part={n_part:4d} nb={nb:3d}  insert {out['insert_s']:8.3f} s   "
        f"eject {out['eject_s']:8.3f} s   reach {out['brick_reach']}"
        f" (realized {out['brick_reach_realized']})",
        flush=True,
    )
    return out


def _ratios(rungs, key):
    """Successive ratios of `key` along a ladder, with the axis ratio beside
    them. Reported as a pair so a reader can see what the cost is being compared
    AGAINST rather than being handed a bare number."""
    out = []
    for a, b in zip(rungs, rungs[1:]):
        out.append(
            dict(
                frm=f"n_part={a['n_part']},nb={a['bricks_per_side']}",
                to=f"n_part={b['n_part']},nb={b['bricks_per_side']}",
                cost_ratio=float(b[key] / a[key]) if a[key] > 0 else None,
                nb_ratio=b["bricks_per_side"] / a["bricks_per_side"],
                n_ratio=b["n_particles"] / a["n_particles"],
            )
        )
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", choices=["smoke", "bricks", "particles"], required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args(argv)

    if args.arm == "smoke":
        # tiny, and it exercises BOTH ladders' code paths end to end
        rungs = [_rung(32, 8, 1), _rung(32, 16, 1), _rung(64, 8, 1)]
        axis = "smoke"
    elif args.arm == "bricks":
        # ARM A: particles fixed, brick count doubling. nb must divide the
        # bucket grid (n_part / BUCKET_CELLS = 128 here), so 8/16/32 are legal
        # and 24 is not.
        rungs = [_rung(256, nb, args.repeats) for nb in (8, 16, 32)]
        axis = "bricks_per_side at fixed n_part=256"
    else:
        # ARM B (control): brick count fixed, particles varied 8x
        rungs = [_rung(n, 16, args.repeats) for n in (128, 192, 256)]
        axis = "n_part at fixed bricks_per_side=16"

    res = dict(
        arm=args.arm,
        axis=axis,
        box_size=BOX,
        bucket_cells=BUCKET_CELLS,
        rungs=rungs,
        insert_ratios=_ratios(rungs, "insert_s"),
        eject_ratios=_ratios(rungs, "eject_s"),
    )

    # KNOB PROOF. The drift is solved per rung to pin reach at 1; if it did not
    # take, the rung measured staging depth as well as brick count and the
    # comparison is confounded. Void rather than noisy.
    reaches = sorted({r["brick_reach"] for r in rungs})
    res["reach_pinned"] = reaches == [1]
    if not res["reach_pinned"]:
        print(f"VOID: brick_reach was not pinned at 1 across rungs: {reaches}", flush=True)

    # THE PRE-REGISTERED READING, evaluated here rather than by eye afterwards.
    if args.arm == "bricks":
        got = [r["cost_ratio"] for r in res["insert_ratios"]]
        res["predicted"] = "insert_s quadruples per doubling of nb (N x nb^2)"
        res["verdict_quadratic"] = all(g is not None and 2.5 <= g <= 6.0 for g in got)
        res["verdict_note"] = (
            "2.5-6.0x per doubling is the acceptance band around 4x. Below 2.5 the "
            "term is not quadratic in nb and the job-455 attribution is WRONG; above "
            "6.0 something steeper than nb^2 is present and the model is incomplete."
        )
    elif args.arm == "particles":
        got = [r["cost_ratio"] for r in res["insert_ratios"]]
        exp = [r["n_ratio"] for r in res["insert_ratios"]]
        res["predicted"] = "insert_s is LINEAR in N at fixed nb"
        res["verdict_linear"] = all(
            g is not None and 0.7 * e <= g <= 1.4 * e for g, e in zip(got, exp)
        )
        res["verdict_note"] = (
            "within 0.7-1.4x of the particle ratio. Superlinear here would mean the "
            "cost does not come from the per-brick scan over slab rows, and arm A's "
            "reading would not survive it."
        )

    res["python"] = platform.python_version()
    res["machine"] = platform.machine()
    try:
        res["commit"] = subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        res["commit"] = None
    res["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")

    class _A:
        leg = "insert_scaling"
        out_suffix = args.out_suffix
        n = rungs[-1]["n_part"]
        bricks = rungs[-1]["bricks_per_side"]
        slab = 32
        seed = SEED
        f_nl = 0.0

    res["provenance"] = m5._provenance(_A())
    path = os.path.join(REPO, "runs", "v2", f"m6_insert_scaling{args.out_suffix}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"  card -> {path}", flush=True)

    for r in res["insert_ratios"]:
        print(
            f"  insert {r['frm']} -> {r['to']}: {r['cost_ratio']:.2f}x "
            f"(nb x{r['nb_ratio']:.0f}, N x{r['n_ratio']:.2f})",
            flush=True,
        )
    if not res["reach_pinned"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
