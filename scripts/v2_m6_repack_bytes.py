"""What does a repack actually allocate? Both implementations, measured.

The M-v2-6 plan says to remove `SlotState.repack`'s O(N) scratch (93.5 GB at
C-gh, the largest single term left after the velocity array went) by porting the
in-place form from `layout.BrickPackedLayout.repack`, "which already reports
`scratch_bytes`" -- 0.13-0.52 MB, D-v2-19 clause 3, "independent of N".

Reading that function, its `scratch_peak` tracks ONLY the two chunk buffers
(`layout.py:662,670`). It also allocates three arrays sized by the particle
count that appear in no accounting: `live` (the nonzero index of occupied
slots), `parts` (the live rows gathered in key order) and `final` (each row's
destination), the last of which is built from an `arange` plus a `repeat` and so
peaks higher again. If that reading is right the reference costs >= 24 B/row
against the 9 B/row it would replace, and the port would make production WORSE.

This measures it instead of asserting it. `tracemalloc` with numpy's own domain
gives an exact allocation high-water on a laptop in seconds -- exact in the
sense that it is bit-identical across runs where an RSS peak scatters by
hundreds of MB -- and it was built for exactly this class of question after ~22
hours of cluster time went into an instrument for a term that did not bind.

**What this does NOT measure:** wall time, XLA scratch (invisible to every
in-process counter, and irrelevant here because both paths are pure numpy), and
the arena fold-in. Both repacks are measured on a freshly built state, where the
arena is empty; the fold-in adds work to both and is not what the 93.5 GB is.

    pixi run python scripts/v2_m6_repack_bytes.py
    pixi run python scripts/v2_m6_repack_bytes.py --n-part 128 --nb 16
"""

import argparse
import json
import os
import sys
import tracemalloc

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from inexor import layout, state  # noqa: E402
from inexor.codec import T9Layout  # noqa: E402

NUMPY_DOMAIN = np.lib.tracemalloc_domain
BOX = 128.0
BUCKET_CELLS = 2


def _numpy_peak(fn):
    """Numpy allocation high-water across one call, in bytes.

    The peak, not the resident delta: a transient that is freed before the call
    returns still had to fit, and on this path the transients are the question.
    """
    tracemalloc.start(1)
    try:
        tracemalloc.reset_peak()
        out = fn()
        _, peak = tracemalloc.get_traced_memory()
        snap = tracemalloc.take_snapshot().filter_traces(
            [tracemalloc.DomainFilter(True, NUMPY_DOMAIN)]
        )
        np_resident = sum(s.size for s in snap.statistics("filename"))
    finally:
        tracemalloc.stop()
    return peak, np_resident, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-part", type=int, default=128)
    ap.add_argument("--nb", type=int, default=16)
    ap.add_argument("--warmup", action="store_true", default=True,
                    help="a throwaway build+repack first; the first call in a "
                         "process allocates import-time buffers that are not "
                         "the function's own")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    n = int(args.n_part) ** 3
    t9 = T9Layout(BOX, int(args.n_part), BUCKET_CELLS)
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, BOX, size=(n, 3))
    v = rng.normal(scale=1.0, size=(n, 3))

    if args.warmup:
        w_t9 = T9Layout(BOX, 32, BUCKET_CELLS)
        wx = np.random.default_rng(0).uniform(0.0, BOX, size=(32**3, 3))
        wv = np.random.default_rng(1).normal(size=(32**3, 3))
        state.SlotState.build(wx, wv, w_t9, 8).repack()
        layout.BrickPackedLayout.build(wx, w_t9, 8).repack()

    st = state.SlotState.build(x, v, t9, int(args.nb))
    n_rows_st = int(st.off.shape[0])
    st_peak, st_res, _ = _numpy_peak(st.repack)

    bp = layout.BrickPackedLayout.build(x, t9, int(args.nb))
    n_rows_bp = int(len(bp.slot_to_particle))
    bp_peak, bp_res, bp_stats = _numpy_peak(bp.repack)

    res = dict(
        n_part=int(args.n_part),
        n_particles=n,
        bricks_per_side=int(args.nb),
        slot_state=dict(
            n_rows=n_rows_st,
            peak_bytes=int(st_peak),
            bytes_per_row=st_peak / n_rows_st,
            reports_scratch=None,
        ),
        brick_packed=dict(
            n_rows=n_rows_bp,
            peak_bytes=int(bp_peak),
            bytes_per_row=bp_peak / n_rows_bp,
            reports_scratch=int(bp_stats["scratch_bytes"]),
            reported_bytes_per_row=bp_stats["scratch_bytes"] / n_rows_bp,
        ),
    )
    res["port_would_multiply_by"] = bp_peak / st_peak

    print(f"n_part={args.n_part}^3 = {n:,} particles, nb={args.nb}")
    print(f"  SlotState.repack          peak {st_peak / 1e6:9.2f} MB   "
          f"{res['slot_state']['bytes_per_row']:6.2f} B/row")
    print(f"  BrickPackedLayout.repack  peak {bp_peak / 1e6:9.2f} MB   "
          f"{res['brick_packed']['bytes_per_row']:6.2f} B/row")
    print(f"    ...but REPORTS scratch_bytes = {bp_stats['scratch_bytes'] / 1e6:.3f} MB "
          f"({res['brick_packed']['reported_bytes_per_row']:.3f} B/row)")
    print(f"  porting would multiply the allocation by {res['port_would_multiply_by']:.2f}x")

    # THE PRE-REGISTERED READING, evaluated here rather than by eye.
    res["reference_underreports"] = bp_peak > 4 * bp_stats["scratch_bytes"]
    res["port_is_worse"] = bp_peak > st_peak
    print(f"  reference under-reports its own scratch: {res['reference_underreports']}")
    print(f"  porting it would be WORSE:               {res['port_is_worse']}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
