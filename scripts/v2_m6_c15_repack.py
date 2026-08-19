"""C15 -- where does `repack` spend its 4.74 s/step, and how far is that from the floor?

**The question.** C14 leaves `repack` as the last phase running serial on an idle
machine: 4.74 s of a 34.52 s step at cgh64 (13.7%), 3.2 busy cores of 144, ~0%
canary loss (5t). The named next move is to pool it, which at C5-like efficiency
targets ~1 s/step. **This canary exists because that target may be the wrong
order of magnitude.**

**The arithmetic that motivates it.** `repack` moves every live row twice -- once
left to compact the main runs, once right to expand them -- so at cgh64 the
payload traffic is 134.2M rows x 9 B x 4 (read+write, twice) = ~4.8 GB. C11
measured 39.5 GB/s single core on that machine in the same job, so the traffic
floor for this work on ONE core is order 0.12 s. The phase takes 4.74. **We are
order 40x above the floor on a single thread, and a 16-worker pool cannot buy
more than 16x.** If the gap is structural rather than physical, the cheaper lever
is larger and composes with pooling afterwards.

**Two candidate mechanisms, both unmeasured.**

1. *Index machinery.* Pass B builds `within` (one int64 per row, via `np.repeat`)
   and `order` (one int64 per row, via `_stable_order`) for every brick. At the
   4096 rows/brick the config table fixes at every rung that is 64 KB of int64
   bookkeeping against 37 KB of actual payload -- the index moves more memory
   than the data it permutes.
2. *The sort.* D-v2-19 already flagged that bucket order is a FIXED spatial
   ordering, so a merge should beat a comparison sort; 5l and 5n found exactly
   that shape in `migrate`'s `argsort` (~14% and ~30% of their phases). Here the
   main run is already in bucket order by construction, so the sort is a merge of
   a sorted run with a short arena tail -- and where a brick has NO arena
   residents it is the identity permutation and the whole apparatus is dead work.

A third outcome closes the lane: the phase is already bandwidth-bound and the 40x
above is my arithmetic being wrong. That is what pre-registration 1 is for.

**The census nobody has taken.** Mechanism 2's fast path depends entirely on how
many bricks carry arena residents WHEN REPACK RUNS. 5v's 8.31M spill rows (~6% of
particles) are arena CLAIMS across a whole migrate pass, not residency at the end
of it: `_release_arena_of_brick` frees a brick's slots as that brick is rewritten,
so the arena is a revolving door within the pass. The default arena is 1% of
particles, which bounds residency far below the claim count. Nobody has measured
where in that range it lands, and it decides whether the fast path is worth
building.

**Pre-registration (falsifiable, written before the first run).**

1. Measured `repack` lands **>10x** above the single-core traffic floor measured
   on the same machine in the same process. If it is within **3x**, the phase is
   already memory-bound, restructuring cannot help, and THIS LANE IS DEAD -- go
   build the pool as originally scoped.
2. **Pass B costs at least 2x pass A.** Pass A is a pure block move with no index
   work; pass B carries all of it over the same rows. Within 1.5x, the cost is
   not the index machinery and mechanism 1 is wrong.
3. **The index machinery is at least 40% of pass B** (`within` + `_stable_order`
   + `bincount`, timed apart from the payload gather). Below 20%, dropping the
   sort cannot pay and only cross-brick batching survives as a lever.
4. **At the engine's slack, more than 50% of bricks carry zero arena residents.**
   Below 20% the identity-order fast path is dead and MUST NOT be built -- the
   merge and the batching are then the only candidates.

**What is gated (exit 2), and what merely reported.**

- GATE: the timed transcription below reproduces the real `SlotState.repack`
  BITWISE -- off, w, ids, occupancy, brick_start, arena_bucket, arena_base and
  the reported slots_used. Without this the timing measures a function nothing
  calls.
- GATE: the engine's `repack` reproduces `_repack_reference` elementwise on this
  state. That oracle is kept in the tree for exactly this, and a synthetic state
  built here is a new input to it. `--skip-oracle` is for configs where its
  out-of-place allocation does not fit, and it is recorded in the card when used.
- GATE: anti-vacuity. At least one rung must carry arena residents, at least one
  brick must actually move in pass A, and at least one brick must grow and one
  shrink -- otherwise a "repack" that returned its input would pass.
- REPORTED: every timing, the arena census, and the instrumentation overhead
  (transcription total vs the real function's total on the same state).

**Exit codes.** 0 = ran, gates passed. 2 = a gate refused, no number below it is
usable. Non-zero is not "failure" here, it is the refusal doing its job.

Usage:
    pixi run python scripts/v2_m6_c15_repack.py --config smoke
    pixi run python scripts/v2_m6_c15_repack.py --config cdev --slack 0.20 0.10 0.02 0.0
    pixi run python scripts/v2_m6_c15_repack.py --config cgh64 --slack 0.20 0.0 --repeats 3
"""

import argparse
import copy
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inexor import state  # noqa: E402
from inexor.state import _stable_order, _to_index  # noqa: E402
from v2_m6_migrate_depth import CONFIGS, _build  # noqa: E402

# The drift that puts migration in the production regime (reach 3), the same
# fraction of the reach-1 threshold every W0b/C6 leg times at.
FRACTION = 2.85

# Payload bytes per row that `repack` actually moves: off (3 x uint8) + w
# (3 x int16). `ids` is added per state when present.
BYTES_OFF_W = 3 * 1 + 3 * 2


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _stream_reference(mb=512, repeats=5):
    """Single-core streaming rate, measured not assumed.

    `c = a + b` over f64 arrays far larger than any cache: 24 B of traffic per
    element (two reads, one write). Deliberately the same construction C6 used,
    so the two canaries' floors are comparable. This is the denominator that
    turns "4.74 s" into a statement about the machine rather than a number.
    """
    n = int(mb * (1 << 20) / 8)
    a = np.ones(n, dtype=np.float64)
    b = np.ones(n, dtype=np.float64)
    c = np.empty(n, dtype=np.float64)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        np.add(a, b, out=c)
        ts.append(time.perf_counter() - t0)
    best = float(np.median(ts))
    return dict(gb_s=float(n * 24 / best / 1e9), n=n, best_s=best)


# ---------------------------------------------------------------------------
# The transcription. Line for line with `SlotState.repack` (state.py), with
# section timers inserted and NOTHING else changed. The bitwise gate against the
# real function is what makes it a description of the phase rather than a
# lookalike; if you edit `repack`, this refuses until it is edited to match.
# ---------------------------------------------------------------------------


def _repack_timed(self, brick_slack=0.10, fast_path=True):
    """`SlotState.repack` with per-section timers. Returns (result, timings).

    `fast_path=False` forces every brick down the merge branch -- that is the
    BASELINE arm, the code as it stood before C15, and it is what the speedup is
    measured against. The arm must prove it applied: `T["n_fast"]` is 0 in the
    baseline and the majority in the shipped arm, and the caller gates on it.
    """
    T = dict(prologue=0.0, arena_lift=0.0, pass_a=0.0, pass_b_index=0.0,
             pass_b_move=0.0, pass_b_zero=0.0, epilogue=0.0)
    t_start = time.perf_counter()

    p3 = self.buckets_per_brick
    run_counts = np.asarray(self.occupancy, dtype=np.int64).reshape(self.n_bricks, p3)
    run_counts = run_counts.sum(axis=1)
    occ = self.occupancy.astype(np.int64)
    arena_live = np.nonzero(self.arena_bucket >= 0)[0]
    if len(arena_live):
        occ = occ + np.bincount(self.arena_bucket[arena_live], minlength=self.n_buckets)
    counts = occ.reshape(self.n_bricks, p3).sum(axis=1)
    spare = np.ceil(counts * float(brick_slack)).astype(np.int64)
    spare = np.where(counts > 0, np.maximum(spare, 1), spare)
    new_start = np.zeros(self.n_bricks + 1, dtype=np.int64)
    np.cumsum(counts + spare, out=new_start[1:])
    n_alloc = int(new_start[-1])
    if n_alloc + self.n_arena > self.off.shape[0]:
        raise ValueError(
            f"repack needs {n_alloc} slots plus a {self.n_arena}-slot arena against an "
            f"allocation of {self.off.shape[0]}. Raise alloc_margin at build."
        )
    T["prologue"] = time.perf_counter() - t_start

    scratch = 0
    t0 = time.perf_counter()
    a_rows = self.arena_base + arena_live
    a_bucket = self.arena_bucket[arena_live]
    a_ord = np.argsort(a_bucket // p3, kind="stable")
    a_bucket = a_bucket[a_ord]
    a_off = self.off[a_rows[a_ord]].copy()
    a_w = self.w[a_rows[a_ord]].copy()
    a_ids = None if self.ids is None else self.ids[a_rows[a_ord]].copy()
    scratch += a_off.nbytes + a_w.nbytes + (0 if a_ids is None else a_ids.nbytes)
    a_edge = np.searchsorted(a_bucket // p3, np.arange(self.n_bricks + 1))
    T["arena_lift"] = time.perf_counter() - t0

    # ---- pass A
    t0 = time.perf_counter()
    main_pos = np.zeros(self.n_bricks + 1, dtype=np.int64)
    np.cumsum(run_counts, out=main_pos[1:])
    n_moved_a = 0
    for b in range(self.n_bricks):
        m = int(run_counts[b])
        src, dst = int(self.brick_start[b]), int(main_pos[b])
        if m == 0 or src == dst:
            continue
        n_moved_a += 1
        buf_off = self.off[src : src + m].copy()
        buf_w = self.w[src : src + m].copy()
        scratch = max(scratch, buf_off.nbytes + buf_w.nbytes)
        self.off[dst : dst + m] = buf_off
        self.w[dst : dst + m] = buf_w
        if self.ids is not None:
            self.ids[dst : dst + m] = self.ids[src : src + m].copy()
    T["pass_a"] = time.perf_counter() - t0

    # ---- pass B
    new_occ = np.zeros(self.n_buckets, dtype=np.int64)
    bucket_ids = np.arange(p3, dtype=np.int64)
    t_idx = t_mv = t_zr = 0.0
    n_fast = n_merge = 0
    for b in range(self.n_bricks - 1, -1, -1):
        m = int(run_counts[b])
        k = int(a_edge[b + 1] - a_edge[b])
        if m + k == 0:
            continue
        mp, ns = int(main_pos[b]), int(new_start[b])

        if k == 0 and fast_path:
            n_fast += 1
            t0 = time.perf_counter()
            buf_off = self.off[mp : mp + m].copy()
            buf_w = self.w[mp : mp + m].copy()
            scratch = max(scratch, buf_off.nbytes + buf_w.nbytes)
            self.off[ns : ns + m] = buf_off
            self.w[ns : ns + m] = buf_w
            if self.ids is not None:
                self.ids[ns : ns + m] = self.ids[mp : mp + m].copy()
            t1 = time.perf_counter()
            new_occ[b * p3 : (b + 1) * p3] = self.occupancy[b * p3 : (b + 1) * p3]
            t2 = time.perf_counter()
            t_mv += t1 - t0
            t_idx += t2 - t1
        else:
            n_merge += 1
            t0 = time.perf_counter()
            within = np.repeat(bucket_ids, np.asarray(
                self.occupancy[b * p3 : (b + 1) * p3], dtype=np.int64))
            if k:
                within = np.concatenate(
                    [within, a_bucket[a_edge[b] : a_edge[b + 1]] - b * p3])
            order = _stable_order(within, p3)
            t1 = time.perf_counter()

            cat_off = self.off[mp : mp + m]
            cat_w = self.w[mp : mp + m]
            if k:
                cat_off = np.concatenate([cat_off, a_off[a_edge[b] : a_edge[b + 1]]])
                cat_w = np.concatenate([cat_w, a_w[a_edge[b] : a_edge[b + 1]]])
            else:
                cat_off, cat_w = cat_off.copy(), cat_w.copy()
            scratch = max(
                scratch, cat_off.nbytes + cat_w.nbytes + within.nbytes + order.nbytes)
            self.off[ns : ns + m + k] = cat_off[order]
            self.w[ns : ns + m + k] = cat_w[order]
            if self.ids is not None:
                cat_i = self.ids[mp : mp + m]
                if k:
                    cat_i = np.concatenate([cat_i, a_ids[a_edge[b] : a_edge[b + 1]]])
                else:
                    cat_i = cat_i.copy()
                self.ids[ns : ns + m + k] = cat_i[order]
            t2 = time.perf_counter()

            new_occ[b * p3 : (b + 1) * p3] = np.bincount(within[order], minlength=p3)
            t3 = time.perf_counter()
            t_idx += (t1 - t0) + (t3 - t2)
            t_mv += t2 - t1

        t_zr0 = time.perf_counter()
        gap_lo, gap_hi = ns + m + k, int(new_start[b + 1])
        if gap_hi > gap_lo:
            self.off[gap_lo:gap_hi] = 0
            self.w[gap_lo:gap_hi] = 0
            if self.ids is not None:
                self.ids[gap_lo:gap_hi] = -1
        t_zr += time.perf_counter() - t_zr0
    T["pass_b_index"], T["pass_b_move"], T["pass_b_zero"] = t_idx, t_mv, t_zr
    T["n_fast"], T["n_merge"] = n_fast, n_merge

    t0 = time.perf_counter()
    self.off[n_alloc:] = 0
    self.w[n_alloc:] = 0
    if self.ids is not None:
        self.ids[n_alloc:] = -1
    self.brick_start[...] = new_start
    self.occupancy[...] = _to_index(new_occ, self.index_dtype, "repacked")
    self.arena_base = n_alloc
    self.arena_bucket[:] = -1
    self._invalidate_arena_index()
    T["epilogue"] = time.perf_counter() - t0

    T["total"] = time.perf_counter() - t_start
    T["n_moved_pass_a"] = n_moved_a
    return dict(
        slots_used=n_alloc,
        slots_per_particle=n_alloc / max(self.n_particles, 1),
        scratch_bytes=int(scratch),
    ), T


# ---------------------------------------------------------------------------


def _census(st, brick_slack):
    """Everything about the state repack is about to see. Taken BEFORE it runs.

    The arena numbers are the point: pre-registration 4 lives or dies here, and
    no prior record carries them (5v's 8.31M counts CLAIMS across a migrate pass,
    not residency at its end).
    """
    p3 = st.buckets_per_brick
    occ = np.asarray(st.occupancy, dtype=np.int64)
    run_counts = occ.reshape(st.n_bricks, p3).sum(axis=1)
    arena_live = np.nonzero(st.arena_bucket >= 0)[0]
    a_brick = st.arena_bucket[arena_live] // p3
    per_brick = np.bincount(a_brick, minlength=st.n_bricks) if len(arena_live) else \
        np.zeros(st.n_bricks, dtype=np.int64)
    counts = run_counts + per_brick
    spare = np.ceil(counts * float(brick_slack)).astype(np.int64)
    spare = np.where(counts > 0, np.maximum(spare, 1), spare)
    new_start = np.zeros(st.n_bricks + 1, dtype=np.int64)
    np.cumsum(counts + spare, out=new_start[1:])
    grows = int(np.sum(new_start[:-1] > np.asarray(st.brick_start[:-1])))
    shrinks = int(np.sum(new_start[:-1] < np.asarray(st.brick_start[:-1])))
    live = int(run_counts.sum())
    occupied = int(np.sum(counts > 0))
    return dict(
        n_bricks=int(st.n_bricks),
        n_rows=int(st.off.shape[0]),
        n_particles=int(st.n_particles),
        live_rows=live,
        arena_slots=int(st.n_arena),
        arena_residents=int(len(arena_live)),
        arena_resident_frac_of_particles=float(len(arena_live) / max(st.n_particles, 1)),
        arena_fill=float(len(arena_live) / max(st.n_arena, 1)),
        bricks_occupied=occupied,
        bricks_with_no_arena=int(np.sum(per_brick == 0)),
        frac_bricks_with_no_arena=float(np.mean(per_brick == 0)),
        # among bricks that hold anything at all -- an empty brick trivially has
        # no residents and would flatter the fraction above
        frac_occupied_bricks_with_no_arena=float(
            np.mean(per_brick[counts > 0] == 0)) if occupied else 0.0,
        arena_per_brick_max=int(per_brick.max()) if st.n_bricks else 0,
        arena_per_brick_p50=float(np.percentile(per_brick, 50)) if st.n_bricks else 0.0,
        arena_per_brick_p99=float(np.percentile(per_brick, 99)) if st.n_bricks else 0.0,
        rows_per_brick_mean=float(run_counts.mean()) if st.n_bricks else 0.0,
        bricks_growing=grows,
        bricks_shrinking=shrinks,
    )


def _fields(st):
    return dict(off=st.off, w=st.w, occupancy=st.occupancy, brick_start=st.brick_start,
                arena_bucket=st.arena_bucket, ids=st.ids)


def _diff(a, b):
    """Elementwise disagreement count per field. None-vs-None counts as equal."""
    out = {}
    for k in a:
        x, y = a[k], b[k]
        if x is None and y is None:
            out[k] = 0
            continue
        if x is None or y is None or x.shape != y.shape:
            out[k] = -1
            continue
        out[k] = int(np.count_nonzero(np.asarray(x) != np.asarray(y)))
    return out


def _gate(st, brick_slack, skip_oracle, real_fn=None):
    """Three states, one input: the real function, the transcription, the oracle.

    `real_fn` is passed explicitly by the engine mode, which has `repack`
    monkeypatched at the time this runs -- calling the bound method there would
    re-enter the transcription and compare it with itself.
    """
    real_fn = real_fn if real_fn is not None else type(st).repack
    s_real = copy.deepcopy(st)
    s_fast = copy.deepcopy(st)
    s_base = copy.deepcopy(st)
    r_real = real_fn(s_real, brick_slack=brick_slack)
    r_fast, t_fast = _repack_timed(s_fast, brick_slack=brick_slack, fast_path=True)
    r_base, t_base = _repack_timed(s_base, brick_slack=brick_slack, fast_path=False)

    g = dict(
        fast_vs_real=_diff(_fields(s_real), _fields(s_fast)),
        baseline_vs_real=_diff(_fields(s_real), _fields(s_base)),
        slots_used=[int(r_real["slots_used"]), int(r_fast["slots_used"]),
                    int(r_base["slots_used"])],
        arena_base=[int(s_real.arena_base), int(s_fast.arena_base), int(s_base.arena_base)],
        # THE ARM'S RECEIPT. The baseline is the pre-C15 code only if it took the
        # fast path zero times; without this the "baseline" could quietly be the
        # shipped function and the A/B would compare a thing with itself.
        n_fast_in_fast_arm=int(t_fast["n_fast"]),
        n_fast_in_baseline_arm=int(t_base["n_fast"]),
        n_merge_in_fast_arm=int(t_fast["n_merge"]),
        oracle=None,
        oracle_skipped=bool(skip_oracle),
    )
    g["arm_separated"] = (g["n_fast_in_baseline_arm"] == 0
                          and g["n_fast_in_fast_arm"] > 0)
    g["transcription_ok"] = (
        all(v == 0 for v in g["fast_vs_real"].values())
        and all(v == 0 for v in g["baseline_vs_real"].values())
        and len(set(g["slots_used"])) == 1
        and len(set(g["arena_base"])) == 1
    )
    if skip_oracle:
        g["oracle_ok"] = None
    else:
        s_ref = copy.deepcopy(st)
        s_ref._repack_reference(brick_slack=brick_slack)
        g["oracle"] = _diff(_fields(s_real), _fields(s_ref))
        g["oracle_ok"] = all(v == 0 for v in g["oracle"].values())
    return g


def _time_repack(st, brick_slack, repeats, real=False, fast_path=True):
    """Time on a FRESH copy each repeat -- repack mutates, so a second call on
    one state would repack an already-repacked state and measure a no-op."""
    best = None
    best_T = None
    for _ in range(repeats):
        s = copy.deepcopy(st)
        if real:
            t0 = time.perf_counter()
            s.repack(brick_slack=brick_slack)
            dt = time.perf_counter() - t0
            T = dict(total=dt)
        else:
            _, T = _repack_timed(s, brick_slack=brick_slack, fast_path=fast_path)
            dt = T["total"]
        if best is None or dt < best:
            best, best_T = dt, T
        del s
    return best_T


def _floor_seconds(cen, bytes_per_row, gb_s):
    """The traffic this rearrangement cannot avoid, at the measured stream rate.

    Counted, not estimated: pass A reads and writes every live row of a brick
    that moves; pass B reads and writes every live row plus every arena resident;
    the arena lift reads and writes its residents; the spare and the tail are
    written once. Index bytes are deliberately EXCLUDED -- they are what an ideal
    implementation would not spend, so putting them in the denominator would hide
    exactly the term this canary is looking for.
    """
    live = cen["live_rows"]
    res = cen["arena_residents"]
    n_alloc_rows = cen["live_rows"] + res
    pass_a = 2 * live * bytes_per_row
    pass_b = 2 * (live + res) * bytes_per_row
    lift = 2 * res * bytes_per_row
    zeroed = max(cen["n_rows"] - n_alloc_rows, 0) * bytes_per_row
    total = pass_a + pass_b + lift + zeroed
    return dict(bytes=int(total), seconds=float(total / (gb_s * 1e9)))


def run(cfg_name, slacks, repeats, with_ids, skip_oracle, out_path):
    cfg = CONFIGS[cfg_name]
    stream = _stream_reference()
    bytes_per_row = BYTES_OFF_W + (4 if with_ids else 0)
    print(f"== C15 {cfg_name}: n_part={cfg['n_part']} nb={cfg['nb']} box={cfg['box']} "
          f"f={FRACTION} ids={with_ids} repeats={repeats}")
    print(f"  stream reference: {stream['gb_s']:.1f} GB/s single core "
          f"({bytes_per_row} payload B/row)")

    rungs = []
    for slack in slacks:
        st = _build(cfg["n_part"], cfg["nb"], cfg["box"], brick_slack=slack,
                    arena_frac=0.20)
        if with_ids:
            print("  NOTE: --with-ids requested but `_build` does not carry ids; "
                  "the state below has none and the floor uses 9 B/row.")
        scales = np.array(st.vel_scale, dtype=np.float64, copy=True)
        extent = float(st.t9.box_size) / int(st.bricks_per_side)
        s_max = float(np.max(scales))
        c_drift = FRACTION * extent / (s_max * float(np.iinfo(np.int16).max))
        # one migrate, so the arena holds what a real step leaves it holding
        state.drift_and_migrate(st, c_drift)

        cen = _census(st, slack)
        gate = _gate(st, slack, skip_oracle)
        T = _time_repack(st, slack, repeats)
        T_base = _time_repack(st, slack, repeats, fast_path=False)
        T_real = _time_repack(st, slack, repeats, real=True)
        floor = _floor_seconds(cen, bytes_per_row, stream["gb_s"])
        over = T_real["total"] / floor["seconds"] if floor["seconds"] > 0 else float("nan")
        pass_b = T["pass_b_index"] + T["pass_b_move"] + T["pass_b_zero"]

        rung = dict(
            brick_slack=float(slack), census=cen, gate=gate, timings=T,
            baseline=T_base, speedup=float(T_base["total"] / T["total"]),
            real_total_s=float(T_real["total"]), floor=floor,
            over_floor=float(over),
            pass_b_s=float(pass_b),
            b_over_a=float(pass_b / T["pass_a"]) if T["pass_a"] > 0 else float("nan"),
            index_share_of_pass_b=float(T["pass_b_index"] / pass_b) if pass_b > 0 else 0.0,
            instrument_overhead=float(T["total"] / T_real["total"] - 1.0),
        )
        rungs.append(rung)

        print(f"\n  -- brick_slack={slack}")
        print(f"     GATE both arms bitwise vs SlotState.repack: {gate['transcription_ok']} "
              f"| arms separated: {gate['arm_separated']}")
        print(f"     GATE repack elementwise vs _repack_reference: {gate['oracle_ok']}"
              + ("  (SKIPPED)" if skip_oracle else f" {gate['oracle']}"))
        print(f"     baseline {T_base['total']:.3f} s -> fast {T['total']:.3f} s "
              f"= {rung['speedup']:.2f}x")
        print(f"     arena: {cen['arena_residents']:,} residents "
              f"({100 * cen['arena_resident_frac_of_particles']:.2f}% of particles, "
              f"arena {100 * cen['arena_fill']:.1f}% full); "
              f"{100 * cen['frac_occupied_bricks_with_no_arena']:.1f}% of occupied bricks "
              f"carry none (max/brick {cen['arena_per_brick_max']})")
        print(f"     bricks: {cen['bricks_occupied']:,} occupied, "
              f"{cen['bricks_growing']:,} grow, {cen['bricks_shrinking']:,} shrink, "
              f"{T['n_moved_pass_a']:,} move in pass A")
        print(f"     real repack {T_real['total']:.3f} s | floor {floor['seconds']:.3f} s "
              f"({floor['bytes'] / 1e9:.3f} GB) | {over:.1f}x above floor")
        print(f"     prologue {T['prologue']:.3f}  arena_lift {T['arena_lift']:.3f}  "
              f"pass_A {T['pass_a']:.3f}  pass_B {pass_b:.3f}  epilogue {T['epilogue']:.3f}")
        print(f"     pass_B split: index {T['pass_b_index']:.3f} "
              f"({100 * rung['index_share_of_pass_b']:.1f}%)  "
              f"move {T['pass_b_move']:.3f}  zero {T['pass_b_zero']:.3f}")
        print(f"     B/A = {rung['b_over_a']:.2f}x | instrument overhead "
              f"{100 * rung['instrument_overhead']:+.1f}%")

    # ---- gates, over the whole sweep
    bad = [r for r in rungs if not r["gate"]["transcription_ok"]]
    if bad:
        print("\n  REFUSING: an arm is not `SlotState.repack`, so every "
              "timing above describes a function nothing calls.")
        return 2, rungs
    bad = [r for r in rungs if not r["gate"]["arm_separated"]]
    if bad:
        print("\n  REFUSING: the baseline arm took the fast path on some rung, so the "
              "A/B there compares the new code with itself.")
        return 2, rungs
    bad = [r for r in rungs if r["gate"]["oracle_ok"] is False]
    if bad:
        print("\n  REFUSING: `repack` disagrees with `_repack_reference` on this state. "
              "That is a state-container defect and it outranks the measurement.")
        return 2, rungs
    if not any(r["census"]["arena_residents"] > 0 for r in rungs):
        print("\n  REFUSING (vacuity): no rung put a single row in the arena, so the "
              "arena census is trivially 100% and pre-registration 4 is unanswered. "
              "Add a lower --slack.")
        return 2, rungs
    if not any(r["timings"]["n_moved_pass_a"] > 0 for r in rungs):
        print("\n  REFUSING (vacuity): pass A moved no brick on any rung, so its timing "
              "is a loop over `continue`.")
        return 2, rungs
    if not any(r["census"]["bricks_growing"] > 0 and r["census"]["bricks_shrinking"] > 0
               for r in rungs):
        print("\n  REFUSING (vacuity): no rung both grew and shrank a brick, so a repack "
              "that returned its input unchanged would pass.")
        return 2, rungs

    print("\n  -- pre-registrations")
    worst = max(r["over_floor"] for r in rungs)
    print(f"     1. >10x above floor:            {worst:.1f}x  -> "
          f"{'HELD' if worst > 10 else ('LANE DEAD (<3x)' if worst < 3 else 'AMBIGUOUS (3-10x)')}")
    ba = max(r["b_over_a"] for r in rungs)
    print(f"     2. pass B >= 2x pass A:         {ba:.2f}x  -> "
          f"{'HELD' if ba >= 2 else 'MISSED'}")
    ix = max(r["index_share_of_pass_b"] for r in rungs)
    print(f"     3. index >= 40% of pass B:      {100 * ix:.1f}%  -> "
          f"{'HELD' if ix >= 0.40 else ('DEAD (<20%)' if ix < 0.20 else 'MISSED')}")
    fr = [r["census"]["frac_occupied_bricks_with_no_arena"] for r in rungs
          if r["census"]["arena_residents"] > 0]
    if fr:
        f0 = min(fr)
        print(f"     4. >50% of bricks arena-free:   {100 * f0:.1f}%  -> "
              f"{'HELD' if f0 > 0.5 else ('FAST PATH DEAD (<20%)' if f0 < 0.2 else 'MISSED')}")

    card = dict(
        card="inexor-c15-repack-1",
        config=cfg_name, n_part=cfg["n_part"], nb=cfg["nb"], box=cfg["box"],
        fraction=FRACTION, repeats=repeats, bytes_per_row=bytes_per_row,
        stream=stream, commit=_git_commit(), host=platform.node(),
        machine=platform.machine(), numpy=np.__version__,
        skip_oracle=bool(skip_oracle), rungs=rungs,
    )
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(card, fh, indent=2)
        print(f"\n  card -> {out_path}")
    return 0, rungs


def run_engine(cfg_name, k, slack, arena_frac, repeats, skip_oracle, out_path,
               tile_workers=1, eject_kernel=None, migrate_pooled=False):
    """The same decomposition, on the state the ENGINE actually hands `repack`.

    Why this mode exists and the synthetic sweep above is not enough: `_build`
    lays down uniform positions, and a uniform state at the engine's slack never
    overflows a brick (job 465), so its arena is EMPTY. The engine's state is
    gravitationally clustered and spills ~6% of particles per migrate pass at the
    same slack (5v). The synthetic sweep therefore brackets arena residency with
    `brick_slack` standing in for clustering; only this mode measures where the
    real thing lands, and pre-registration 4 is about the real thing.

    The engine runs the TRANSCRIPTION, not `SlotState.repack` -- that is what
    makes the per-section split available at all -- and the first call gates it
    bitwise against the real function on a copy of that same state before any
    number is kept.

    `migrate_pooled` does not change what `repack` sees: C14 established the
    pooled pass is bitwise the serial one, so the state reaching this phase is
    identical either way. It is exposed only because pooling makes a cluster
    leg's WALL an order cheaper, and `tile_workers` for the same reason -- the
    repack numbers are indifferent to both, the job's cost is not.
    """
    import v2_m3_engine_gate as m3
    import v2_m6_phase_time as pt

    stream = _stream_reference()
    engine, ec, st, cosmo, a_grid, bfc, bft = pt._build(
        cfg_name, slack, arena_frac, tile_workers=tile_workers,
        eject_kernel=eject_kernel, migrate_pooled=migrate_pooled)
    a_steps = a_grid(m3.A_INIT, m3.A_FINAL, k, m3.SPACING)
    co = bfc(bft(a_steps, cosmo))
    print(f"== C15 engine {cfg_name}: k={k} slack={slack} arena_frac={arena_frac} "
          f"n_bricks={st.n_bricks:,} tile_workers={tile_workers} "
          f"migrate_pooled={migrate_pooled} eject={ec.eject_kernel}")
    print(f"  stream reference: {stream['gb_s']:.1f} GB/s single core")

    cls = type(st)
    real_repack = cls.repack
    steps = []

    def hooked(self, brick_slack=0.10):
        cen = _census(self, brick_slack)
        gate = None
        if not steps:
            gate = _gate(self, brick_slack, skip_oracle, real_fn=real_repack)
        # the baseline runs on a COPY, so the engine still advances down the
        # shipped path and the A/B does not change the trajectory it measures
        s_base = copy.deepcopy(self)
        _, T_base = _repack_timed(s_base, brick_slack=brick_slack, fast_path=False)
        del s_base
        res, T = _repack_timed(self, brick_slack=brick_slack, fast_path=True)
        steps.append(dict(step=len(steps), census=cen, timings=T, baseline=T_base,
                          gate=gate))
        return res

    cls.repack = hooked
    try:
        t0 = time.perf_counter()
        engine.run(st, ec, co)
        wall = time.perf_counter() - t0
    finally:
        cls.repack = real_repack

    if not steps:
        print("  REFUSING (vacuity): `repack` was never called, so nothing below "
              "describes the phase. Check repack_every.")
        return 2, steps

    g0 = steps[0]["gate"]
    print(f"  GATE both arms bitwise vs SlotState.repack: {g0['transcription_ok']}")
    print(f"       fast vs real {g0['fast_vs_real']}")
    print(f"       base vs real {g0['baseline_vs_real']}")
    print(f"  GATE the two arms are separated:            {g0['arm_separated']} "
          f"(fast path taken {g0['n_fast_in_fast_arm']:,} / {g0['n_fast_in_baseline_arm']:,} "
          f"times in fast / baseline)")
    print(f"  GATE repack elementwise vs _repack_reference: {g0['oracle_ok']}"
          + ("  (SKIPPED)" if skip_oracle else f" {g0['oracle']}"))
    if not g0["transcription_ok"]:
        print("  REFUSING: an arm is not `SlotState.repack`, so no timing below "
              "describes the shipped function.")
        return 2, steps
    if not g0["arm_separated"]:
        print("  REFUSING: the baseline arm took the fast path, so the A/B compares "
              "the new code with itself and cannot show a regression.")
        return 2, steps
    if g0["oracle_ok"] is False:
        print("  REFUSING: `repack` disagrees with `_repack_reference` on the engine's "
              "own state. That is a container defect and it outranks the measurement.")
        return 2, steps

    print(f"\n  engine wall {wall:.1f} s over {len(steps)} repack calls\n")
    hdr = ("  step   arena_res   %parts  arena-free   base_s   fast_s   speedup   "
           "floor_s   xfloor(fast)   pass_A   pass_B")
    print(hdr)
    for r in steps:
        c, T, B = r["census"], r["timings"], r["baseline"]
        pb = T["pass_b_index"] + T["pass_b_move"] + T["pass_b_zero"]
        fl = _floor_seconds(c, BYTES_OFF_W, stream["gb_s"])
        r["pass_b_s"] = float(pb)
        r["floor"] = fl
        r["over_floor"] = float(T["total"] / fl["seconds"]) if fl["seconds"] > 0 else 0.0
        r["over_floor_baseline"] = (
            float(B["total"] / fl["seconds"]) if fl["seconds"] > 0 else 0.0)
        r["index_share_of_pass_b"] = float(T["pass_b_index"] / pb) if pb > 0 else 0.0
        r["b_over_a"] = float(pb / T["pass_a"]) if T["pass_a"] > 0 else 0.0
        r["speedup"] = float(B["total"] / T["total"]) if T["total"] > 0 else 0.0
        print(f"  {r['step']:4d} {c['arena_residents']:11,} "
              f"{100 * c['arena_resident_frac_of_particles']:7.3f} "
              f"{100 * c['frac_occupied_bricks_with_no_arena']:10.1f} "
              f"{B['total']:8.3f} {T['total']:8.3f} {r['speedup']:8.2f}x "
              f"{fl['seconds']:9.4f} {r['over_floor']:13.1f} "
              f"{T['pass_a']:8.3f} {pb:8.3f}")

    last = steps[-1]
    med_speed = float(np.median([r["speedup"] for r in steps]))
    print("\n  -- pre-registrations, on the LAST step (the most clustered state seen); "
          "1-3 read on the BASELINE, which is the code they were written about")
    ob = last["over_floor_baseline"]
    print(f"     1. >10x above floor:            {ob:.1f}x  -> "
          + ("HELD" if ob > 10 else
             ("LANE DEAD (<3x)" if ob < 3 else "AMBIGUOUS (3-10x)")))
    bb = last["baseline"]
    pbb = bb["pass_b_index"] + bb["pass_b_move"] + bb["pass_b_zero"]
    ba = pbb / bb["pass_a"] if bb["pass_a"] > 0 else 0.0
    print(f"     2. pass B >= 2x pass A:         {ba:.2f}x  -> "
          + ("HELD" if ba >= 2 else "MISSED"))
    ix = bb["pass_b_index"] / pbb if pbb > 0 else 0.0
    print(f"     3. index >= 40% of pass B:      {100 * ix:.1f}%  -> "
          + ("HELD" if ix >= 0.40 else ("DEAD (<20%)" if ix < 0.20 else "MISSED")))
    f0 = last["census"]["frac_occupied_bricks_with_no_arena"]
    print(f"     4. >50% of bricks arena-free:   {100 * f0:.1f}%  -> "
          + ("HELD" if f0 > 0.5 else ("FAST PATH DEAD (<20%)" if f0 < 0.2 else "MISSED")))
    print(f"\n  MEDIAN SPEEDUP over {len(steps)} steps: {med_speed:.2f}x  "
          f"(fast path still {last['over_floor']:.1f}x above the single-core floor)")

    card = dict(
        card="inexor-c15-repack-engine-1", config=cfg_name, k=k, brick_slack=slack,
        arena_frac=arena_frac, n_bricks=int(st.n_bricks), engine_wall_s=float(wall),
        tile_workers=int(tile_workers), migrate_pooled=bool(migrate_pooled),
        eject_kernel=str(ec.eject_kernel), median_speedup=float(med_speed),
        stream=stream, bytes_per_row=BYTES_OFF_W, commit=_git_commit(),
        host=platform.node(), machine=platform.machine(), numpy=np.__version__,
        skip_oracle=bool(skip_oracle), steps=steps,
    )
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(card, fh, indent=2)
        print(f"\n  card -> {out_path}")
    return 0, steps


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # the union: the synthetic sweep keys off `v2_m6_migrate_depth.CONFIGS`,
    # engine mode off `v2_m3_engine_gate.CONFIGS`, and they are not the same set
    ap.add_argument("--config", default="smoke",
                    choices=sorted(set(CONFIGS) | {"cdev8"}))
    ap.add_argument("--slack", type=float, nargs="+", default=[0.20, 0.10, 0.02, 0.0],
                    help="brick_slack sweep -- this is the ARENA OCCUPANCY axis")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--with-ids", action="store_true")
    ap.add_argument("--skip-oracle", action="store_true",
                    help="skip the out-of-place identity oracle (it allocates O(N))")
    ap.add_argument("--out", default=None)
    ap.add_argument("--engine", type=int, default=0, metavar="K",
                    help="run the REAL engine for K steps and decompose the repack it "
                         "actually calls (clustered state). 0 = the synthetic sweep.")
    ap.add_argument("--arena-frac", type=float, default=0.02,
                    help="engine mode only; 0.02 is the phase-card convention")
    ap.add_argument("--tile-workers", type=int, default=1,
                    help="engine mode only; the repack numbers do not depend on it, "
                         "the job's wall does")
    ap.add_argument("--eject-kernel", default=None, choices=("numpy", "jax"))
    ap.add_argument("--migrate-pooled", action="store_true",
                    help="engine mode only; bitwise the serial pass (C14), so this "
                         "buys wall and changes nothing repack sees")
    args = ap.parse_args()
    if args.engine:
        rc, _ = run_engine(args.config, args.engine, args.slack[0], args.arena_frac,
                           args.repeats, args.skip_oracle, args.out,
                           tile_workers=args.tile_workers,
                           eject_kernel=args.eject_kernel,
                           migrate_pooled=args.migrate_pooled)
        return rc
    rc, _ = run(args.config, args.slack, args.repeats, args.with_ids,
                args.skip_oracle, args.out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
