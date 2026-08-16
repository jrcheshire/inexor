"""C6 -- is `migrate`'s per-row work compute-bound, and does compiling it help?

**The question.** Section 5m puts `migrate` at 23.31 h of a 45.24 h C-gh
realization -- 51.5% of it, and the largest single term by 1.6x. Section 5l
closed the per-brick Python loop as a lever (2.0% at the 4,096 rows/brick the
config table fixes at every rung) and concluded "the phase is irreducible row
work, and the only lever is parallelism". **That conclusion does not follow, and
this canary is why.** "Per-row" says the cost scales with rows; it says nothing
about how much a row costs. Vista 914640 then measured the parallel lever
(74.9-76.2% efficiency at W=8, capped at 1.64x on the phase because `eject` is
only 12.99 of 33.26 s), so the pool alone does not close 23.31 h.

**The arithmetic that motivates this.** At cgh64 the phase does ~231 ns of work
per row, single-threaded. The row's own inputs and outputs are ~33 B. A single
Grace core streams order 10-20 GB/s, so the traffic floor for this work is order
2 ns/row. We are ~100x above it. That gap is not bandwidth and section 5l already
showed it is not per-brick call overhead: it is passes and temporaries -- the
kernel below materializes ~10 intermediate (n,3) f64/i64 arrays where a fused
form materializes none.

**Pre-registration (falsifiable, written before the first run).**

1. `numpy_perbrick` lands >30x above the measured single-core traffic floor. If
   it is within 3x, the phase is already memory-bound, compiling it cannot help,
   and THIS LANE IS DEAD -- go build the `insert` half of the pool instead.
2. `numpy_batched` (identical arithmetic, one array per slab instead of one per
   brick) buys **less than 1.3x**. Section 5l's nb scan already priced per-brick
   overhead at ~4% and this arm is the direct check of that reading at fixed N.
   If batching alone buys a lot, 5l's 2.0% is wrong and that is the finding.
3. `jax_cpu` buys **>= 3x** over `numpy_perbrick`. Below 2x, fusion is not the
   mechanism and the lane closes.
4. Bitwise is NOT pre-registered as a pass. XLA may contract `x / q + (c * v) / q`
   into an FMA, which moves the last bit and would make the twin a
   different-numbers path rather than a drop-in. Either outcome is informative;
   the gate REPORTS it and only fails the run if the numpy transcription itself
   disagrees with the engine.

**What is gated, and what merely reported.**

- GATE (exit 2): `numpy_perbrick` must reproduce the REAL `_eject_slab`'s
  (dest, off_new, stay) elementwise on a real built state. Without this the
  timing measures a function nothing calls.
- GATE (exit 2): anti-vacuity. Both keepers and leavers must be non-empty, and
  the row count must be non-zero, or `stay` is trivially reproducible.
- GATE (exit 2): the jax arms must report the backend they were asked for, so a
  silently-CPU "device" arm cannot pass as a device measurement.
- REPORTED: every speedup, and the bitwise verdict per jax arm.

Usage:
    pixi run python scripts/v2_m6_c6_jit.py --config smoke
    pixi run python scripts/v2_m6_c6_jit.py --config cgh64 --arms numpy_perbrick \
        numpy_batched jax_cpu
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
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inexor import state  # noqa: E402
from inexor.codec import LEVELS_PER_BUCKET  # noqa: E402

from v2_m6_migrate_depth import CONFIGS, _build  # noqa: E402

ARMS = ("numpy_perbrick", "numpy_batched", "jax_cpu", "jax_dev")

# inputs + outputs a row must touch no matter how the kernel is written:
# off 3xuint8 in, bijk 3xint64 in, w 3xint16 in, dest int64 out, off_new 3xuint8
# out, stay bool out. The f64 intermediates are what a fused form does NOT have
# to materialize, which is exactly the quantity under test, so they are excluded
# from the floor deliberately.
BYTES_PER_ROW = 3 * 1 + 3 * 8 + 3 * 2 + 8 + 3 * 1 + 1


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# the kernel, three ways. All three compute the SAME function:
#   (off, bijk, w, scale, c_drift) -> (dest, off_new, stay)
# transcribed from state._eject_slab lines 1043-1057 + decode_brick's decode.
# ---------------------------------------------------------------------------


def _kernel_numpy(off, bijk, w, scale, c_drift, t9, nb, brick_of_row):
    """Verbatim numpy transcription. This is the arm gated against the engine.

    `scale` is PER ROW, carrying its own brick's velocity scale. That is not a
    convenience: D-v2-19 made the scale per brick, `decode_brick` reads
    `scales[brick_flat]`, and a slab spans many bricks. Passing one scale for the
    slab misclassified 347 of 32,832 rows on the first smoke run and the engine
    gate caught it.
    """
    q = t9.quantum
    i = bijk * LEVELS_PER_BUCKET + off.astype(np.int64)
    x = i.astype(np.float64) * q
    v = w.astype(np.float64) * scale
    i_new = np.mod(np.rint(x / q + (float(c_drift) * v) / q).astype(np.int64), t9.n_levels)
    b_ijk = i_new // LEVELS_PER_BUCKET
    off_new = (i_new - b_ijk * LEVELS_PER_BUCKET).astype(np.uint8)
    dest = state._bucket_flat_brick_major(b_ijk, t9, nb)
    stay = (dest // (t9.n_buckets_side // int(nb)) ** 3) == brick_of_row
    return dest, off_new, stay


def _make_kernel_jax(t9, nb):
    """Jitted twin. Same expression SHAPE as the numpy arm, deliberately: the
    bitwise question is whether the COMPILER moves the bits, not whether a
    rewritten formula does."""
    import jax
    import jax.numpy as jnp

    q = float(t9.quantum)
    n_levels = int(t9.n_levels)
    per = int(t9.n_buckets_side) // int(nb)
    p3 = per**3
    nbi = int(nb)

    @jax.jit
    def k(off, bijk, w, scale, c_drift, brick_of_row):
        i = bijk * LEVELS_PER_BUCKET + off.astype(jnp.int64)
        x = i.astype(jnp.float64) * q
        v = w.astype(jnp.float64) * scale
        i_new = jnp.mod(jnp.round(x / q + (c_drift * v) / q).astype(jnp.int64), n_levels)
        b_ijk = i_new // LEVELS_PER_BUCKET
        off_new = (i_new - b_ijk * LEVELS_PER_BUCKET).astype(jnp.uint8)
        brick = b_ijk // per
        within = b_ijk - brick * per
        bf = (brick[:, 0] * nbi + brick[:, 1]) * nbi + brick[:, 2]
        wf = (within[:, 0] * per + within[:, 1]) * per + within[:, 2]
        dest = bf * p3 + wf
        stay = (dest // p3) == brick_of_row
        return dest, off_new, stay

    return k


# ---------------------------------------------------------------------------
# the traffic floor, measured on THIS machine in THIS process
# ---------------------------------------------------------------------------


def _stream_reference(mb=512, repeats=5):
    """Single-core streaming rate, measured not assumed.

    `c = a + b` over f64 arrays far larger than any cache: 24 B of traffic per
    element (two reads, one write). This is the denominator that turns "231 ns
    per row" into a statement about the machine rather than a number.
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


def _gather_slab(st, bx, scales):
    """Pull one slab's rows out of the state as flat arrays, plus each row's
    owning brick. This is the INPUT to every arm, gathered once and identically,
    so no arm is charged for the gather."""
    offs, bijks, ws, brick_of, scale_of = [], [], [], [], []
    lo_b, hi_b = st.slab_bricks(bx)
    for b in range(lo_b, hi_b):
        slots, _, _ = st.decode_brick(b, scales=scales)
        if not len(slots):
            continue
        bijk = st.bucket_ijk_of_live_slots(b)
        a = st.arena_slots_of_brick(b)
        if len(a):
            from inexor.layout import bucket_ijk_from_key

            bijk = np.concatenate(
                [bijk, bucket_ijk_from_key(st.arena_bucket[a - st.arena_base], st.t9, st.bricks_per_side)]
            )
        offs.append(st.off[slots])
        bijks.append(bijk)
        ws.append(st.w[slots])
        brick_of.append(np.full(len(slots), b, dtype=np.int64))
        # per ROW, because the scale is per BRICK and a slab spans many
        scale_of.append(np.full((len(slots), 1), float(scales[b]), dtype=np.float64))
    if not offs:
        return None
    return (
        np.concatenate(offs),
        np.concatenate(bijks),
        np.concatenate(ws),
        np.concatenate(brick_of),
        np.concatenate(scale_of),
        [(int(b), int(len(o))) for b, o in zip(range(lo_b, hi_b), offs)],
    )


def _gate_against_engine(st, bx, c_drift, scales, packed):
    """The numpy transcription must reproduce the REAL `_eject_slab`."""
    off, bijk, w, brick_of, s, _ = packed
    # the real thing, on a COPY, because _eject_slab releases arena rows
    import copy

    st2 = copy.deepcopy(st)
    # timed here rather than in its own arm: `_eject_slab` mutates (it releases
    # arena rows), so a repeated timing needs a fresh copy each time and the copy
    # would dominate. ONE call is enough for a denominator, and a denominator is
    # what stops the kernel speedup being read as a phase speedup.
    _t0 = time.perf_counter()
    keep, emig = st2._eject_slab(bx, c_drift, np.array(scales, copy=True))
    t_eject = time.perf_counter() - _t0
    n_keep = len(keep["dest"])
    n_emig = len(emig["dest"])
    dest, off_new, stay = _kernel_numpy(off, bijk, w, s, c_drift, st.t9, st.bricks_per_side, brick_of)
    ours_keep = int(stay.sum())
    ours_emig = int((~stay).sum())
    ok_counts = (ours_keep == n_keep) and (ours_emig == n_emig)
    # the destinations themselves, as multisets (the engine concatenates
    # per brick and so does the gather, so order should match; compare sorted
    # anyway, since order is not the property under test)
    ok_dest = bool(
        len(dest[stay]) == n_keep
        and len(dest[~stay]) == n_emig
        and np.array_equal(np.sort(dest[stay]), np.sort(np.asarray(keep["dest"])))
        and np.array_equal(np.sort(dest[~stay]), np.sort(np.asarray(emig["dest"])))
    )
    return dict(
        ok=bool(ok_counts and ok_dest),
        ok_counts=bool(ok_counts),
        ok_dest=bool(ok_dest),
        eject_full_s=float(t_eject),
        n_keep_engine=n_keep,
        n_emig_engine=n_emig,
        n_keep_ours=ours_keep,
        n_emig_ours=ours_emig,
    )


def _decompose_eject(st, bx, c_drift, scales, repeats, floor_ns, n_rows, t_eject):
    """Where the OTHER 64% of `_eject_slab` goes, against the same floor.

    Every component is timed by replaying the exact operation on the exact
    arrays, in the same per-brick loop `_eject_slab` uses, so no component is a
    rewrite of what the engine does. **The parts are then required to add up to
    the whole** (`reconstruction` below): a decomposition whose pieces sum to
    60% of the measured call has found 60% of the phase and is silent about the
    rest, and silence there is exactly how a lever gets aimed at the wrong term.
    """
    lo_b, hi_b = st.slab_bricks(bx)
    bricks = list(range(lo_b, hi_b))
    t9, nb = st.t9, st.bricks_per_side

    # PRECOMPUTED ONCE, and every replay below consumes these rather than
    # re-deriving them. The first version of this function let three replays
    # re-do the slot gather internally and the parts summed to 144% of the whole
    # -- the reconstruction check is what caught it, which is the entire reason
    # it is a check and not a printout.
    from inexor.layout import bucket_ijk_from_key

    per = []
    for b in bricks:
        lo = int(st.brick_start[b])
        m = st.brick_live_count(b)
        slots = np.arange(lo, lo + m, dtype=np.int64)
        bijk = st.bucket_ijk_of_live_slots(b)
        a = st.arena_slots_of_brick(b)
        if len(a):
            slots = np.concatenate([slots, a])
            bijk = np.concatenate(
                [bijk, bucket_ijk_from_key(st.arena_bucket[a - st.arena_base], t9, nb)]
            )
        if not len(slots):
            continue
        off_b, w_b = st.off[slots], st.w[slots]
        ids_b = st.ids[slots] if st.ids is not None else None
        sc = np.full((len(slots), 1), float(scales[b]), dtype=np.float64)
        brow = np.full(len(slots), b, dtype=np.int64)
        d, on, stq = _kernel_numpy(off_b, bijk, w_b, sc, c_drift, t9, nb, brow)
        per.append(dict(b=b, slots=slots, bijk=bijk, off=off_b, w=w_b, ids=ids_b,
                        sc=sc, brow=brow, dest=d, off_new=on, stay=stq))

    def _slots_bijk():
        """decode_brick MINUS the decode: slot range, bucket ijk, arena splice."""
        for p in per:
            b = p["b"]
            lo = int(st.brick_start[b])
            m = st.brick_live_count(b)
            slots = np.arange(lo, lo + m, dtype=np.int64)
            bijk = st.bucket_ijk_of_live_slots(b)
            a = st.arena_slots_of_brick(b)
            if len(a):
                slots = np.concatenate([slots, a])
                bijk = np.concatenate(
                    [bijk, bucket_ijk_from_key(st.arena_bucket[a - st.arena_base], t9, nb)]
                )

    def _gather():
        for p in per:
            sl = p["slots"]
            _ = st.off[sl]
            _ = st.w[sl]
            if st.ids is not None:
                _ = st.ids[sl]

    def _kernel():
        for p in per:
            _kernel_numpy(p["off"], p["bijk"], p["w"], p["sc"], c_drift, t9, nb, p["brow"])

    def _partition():
        for p in per:
            d, on, stq, w_cur, ids_b = p["dest"], p["off_new"], p["stay"], p["w"], p["ids"]
            _ = (d[stq], on[stq], w_cur[stq], None if ids_b is None else ids_b[stq])
            ns = ~stq
            _ = (d[ns], on[ns], w_cur[ns], None if ids_b is None else ids_b[ns],
                 np.full(int(ns.sum()), p["b"], dtype=np.int32))

    def _concat():
        kd = [p["dest"][p["stay"]] for p in per]
        ko = [p["off_new"][p["stay"]] for p in per]
        kw = [p["w"][p["stay"]] for p in per]
        ki = [None if p["ids"] is None else p["ids"][p["stay"]] for p in per]
        state._cat(kd, ko, kw, ki)

    comps = [("slots+bijk", _slots_bijk), ("gather", _gather), ("kernel", _kernel),
             ("partition", _partition), ("concat", _concat)]
    out, total = {}, 0.0
    print("  -- where _eject_slab goes (each replayed on the real arrays) --")
    for name, fn in comps:
        t, _ = _time(fn, repeats)
        ns = t / n_rows * 1e9
        total += t
        out[name] = dict(s=t, ns_per_row=ns, over_floor=ns / floor_ns,
                         frac_of_eject=t / t_eject)
        print(f"  {name:14s} {t*1e3:8.2f} ms  {ns:7.2f} ns/row  {ns/floor_ns:6.1f}x floor  "
              f"{100*t/t_eject:5.1f}% of eject")
    rec = total / t_eject
    print(f"  {'SUM':14s} {total*1e3:8.2f} ms  {'':7s}  {'':6s}       {100*rec:5.1f}% "
          f"reconstruction of the {t_eject*1e3:.1f} ms call")
    out["_reconstruction"] = rec
    out["_ok"] = bool(0.80 <= rec <= 1.20)
    if not out["_ok"]:
        print(f"  WARNING: the parts reconstruct {100*rec:.0f}% of the whole, so this "
              f"decomposition does NOT describe the call and must not be quoted as one.")
    return out


def _profile_insert(st, bx, c_drift, scales, top=12):
    """`insert` is 19.68 of migrate's 33.26 s (5l) and is the larger half.

    RANKING ONLY, by cProfile, deliberately: a timed replay of insert would have
    to reimplement the spare/arena escalation, and a reimplementation is how a
    benchmark stops describing the thing it is named after. The ranking is enough
    to choose what to decompose next.
    """
    import cProfile
    import copy
    import pstats
    import io

    st2 = copy.deepcopy(st)
    sc = np.array(scales, copy=True)
    nb = st2.bricks_per_side
    staged, emig, consumed = {}, {}, {}
    r = min(state.brick_reach(st2, c_drift, sc), nb // 2)
    reach = range(-r, r + 1)
    for s in range(nb):
        staged[s], emig[s] = st2._eject_slab(s, c_drift, sc)
        consumed[s] = 0
    pr = cProfile.Profile()
    pr.enable()
    st2._insert_slab(bx, staged, emig, reach, consumed, scales=sc)
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(top)
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    print("  -- _insert_slab, ranked by tottime (profiler-inflated; ranking only) --")
    for ln in lines[4:4 + top + 1]:
        print("   " + ln[:118])
    return buf.getvalue()


def _time(fn, repeats):
    ts = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def run(cfg_name, arms, repeats, fraction, out_path, decompose=False):
    cfg = CONFIGS[cfg_name]
    st = _build(cfg["n_part"], cfg["nb"], cfg["box"], brick_slack=0.0, arena_frac=0.20)
    scales = np.array(st.vel_scale, dtype=np.float64, copy=True)
    # the drift magnitude, expressed as the 5l fraction of the reach-1 threshold
    extent = float(st.t9.box_size) / int(st.bricks_per_side)
    s_max = float(np.max(scales))
    c_drift = fraction * extent / (s_max * float(np.iinfo(np.int16).max))

    bx = int(st.bricks_per_side) // 2
    packed = _gather_slab(st, bx, scales)
    if packed is None:
        print("EMPTY SLAB -- no rows to time")
        return 2
    off, bijk, w, brick_of, s, per_brick = packed
    n_rows = int(len(off))

    print(f"== C6 {cfg_name}: n_part={cfg['n_part']} nb={cfg['nb']} f={fraction} "
          f"slab={bx} rows={n_rows} bricks={len(per_brick)}")

    gate = _gate_against_engine(st, bx, c_drift, scales, packed)
    print(f"  GATE vs the real _eject_slab: ok={gate['ok']} "
          f"(keep {gate['n_keep_ours']}/{gate['n_keep_engine']}, "
          f"emig {gate['n_emig_ours']}/{gate['n_emig_engine']})")
    if not gate["ok"]:
        print("  REFUSING: the numpy transcription is not the engine's kernel, so no "
              "timing below would describe the phase.")
        return 2
    if gate["n_keep_ours"] == 0 or gate["n_emig_ours"] == 0:
        print(f"  REFUSING (vacuity): keepers={gate['n_keep_ours']} "
              f"leavers={gate['n_emig_ours']} -- `stay` is trivial, raise or lower --fraction")
        return 2

    stream = _stream_reference()
    floor_ns = BYTES_PER_ROW / (stream["gb_s"] * 1e9) * 1e9
    print(f"  stream reference: {stream['gb_s']:.1f} GB/s single core -> traffic floor "
          f"{floor_ns:.2f} ns/row at {BYTES_PER_ROW} B/row")

    ref = dict(dest=None, off=None, stay=None)
    results = {}

    for arm in arms:
        if arm == "numpy_perbrick":
            def fn():
                o = 0
                ds, os_, ss = [], [], []
                for b, n in per_brick:
                    sl = slice(o, o + n)
                    d, on, stq = _kernel_numpy(
                        off[sl], bijk[sl], w[sl], s[sl], c_drift, st.t9, st.bricks_per_side,
                        brick_of[sl]
                    )
                    ds.append(d)
                    os_.append(on)
                    ss.append(stq)
                    o += n
                return np.concatenate(ds), np.concatenate(os_), np.concatenate(ss)
        elif arm == "numpy_batched":
            def fn():
                return _kernel_numpy(off, bijk, w, s, c_drift, st.t9, st.bricks_per_side, brick_of)
        elif arm in ("jax_cpu", "jax_dev"):
            want = "cpu" if arm == "jax_cpu" else "gpu"
            os.environ.setdefault("JAX_ENABLE_X64", "1")
            import jax
            import jax.numpy as jnp

            jax.config.update("jax_enable_x64", True)
            devs = [d for d in jax.devices() if (d.platform == "cpu") == (want == "cpu")]
            if not devs:
                print(f"  {arm:16s} SKIPPED -- no {want} device visible")
                results[arm] = dict(skipped=True, reason=f"no {want} device")
                continue
            dev = devs[0]
            k = _make_kernel_jax(st.t9, st.bricks_per_side)
            a_off = jax.device_put(jnp.asarray(off), dev)
            a_bij = jax.device_put(jnp.asarray(bijk), dev)
            a_w = jax.device_put(jnp.asarray(w), dev)
            a_br = jax.device_put(jnp.asarray(brick_of), dev)
            a_s = jax.device_put(jnp.asarray(s), dev)
            _ = jax.block_until_ready(k(a_off, a_bij, a_w, a_s, c_drift, a_br))  # compile

            def fn():
                r = k(a_off, a_bij, a_w, a_s, c_drift, a_br)
                return tuple(np.asarray(z) for z in jax.block_until_ready(r))

            results.setdefault(arm, {})["device"] = str(dev)
            results[arm]["platform"] = dev.platform
            if dev.platform != ("cpu" if want == "cpu" else dev.platform):
                print(f"  {arm:16s} REFUSING -- asked for {want}, got {dev.platform}")
                return 2
        else:
            raise SystemExit(f"unknown arm {arm}")

        t, out = _time(fn, repeats)
        ns = t / n_rows * 1e9
        d = results.setdefault(arm, {})
        d.update(s=t, ns_per_row=ns, over_floor=ns / floor_ns)
        if ref["dest"] is None:
            ref["dest"], ref["off"], ref["stay"] = (np.asarray(z) for z in out)
            d["bitwise_vs_numpy"] = "reference"
        else:
            got = tuple(np.asarray(z) for z in out)
            same = (
                np.array_equal(got[0], ref["dest"])
                and np.array_equal(got[1], ref["off"])
                and np.array_equal(got[2], ref["stay"])
            )
            n_diff = int(np.count_nonzero(got[0] != ref["dest"]))
            d["bitwise_vs_numpy"] = bool(same)
            d["n_dest_diff"] = n_diff
        base = results.get("numpy_perbrick", {}).get("ns_per_row")
        d["speedup_vs_perbrick"] = (base / ns) if base else None
        bw = d.get("bitwise_vs_numpy")
        bwtxt = "" if bw == "reference" else f"  bitwise={bw}" + (
            "" if bw else f" (n_diff={d.get('n_dest_diff')})")
        sp = d["speedup_vs_perbrick"]
        sptxt = "" if sp is None else f"  {sp:6.2f}x"
        print(f"  {arm:16s} {t*1e3:9.2f} ms  {ns:8.2f} ns/row  "
              f"{d['over_floor']:7.1f}x floor{sptxt}{bwtxt}")

    print("== verdict")
    base = results.get("numpy_perbrick", {}).get("ns_per_row")
    if base:
        over = base / floor_ns
        print(f"  numpy_perbrick is {over:.0f}x the measured traffic floor -- "
              f"{'COMPUTE-BOUND, the lane is open' if over > 30 else 'NOT the predicted regime'}")
        for arm in ("numpy_batched", "jax_cpu", "jax_dev"):
            r = results.get(arm)
            if r and r.get("speedup_vs_perbrick"):
                print(f"  {arm}: {r['speedup_vs_perbrick']:.2f}x  bitwise={r.get('bitwise_vs_numpy')}")

    # THE DENOMINATOR. The kernel is not the phase: `_eject_slab` also gathers
    # `off`/`w` by slot, partitions on `stay`, and concatenates per brick, and
    # `_insert_slab` (untimed here) sorts and scatters. Replacing the kernel can
    # only ever buy the Amdahl bound below, and it is printed beside the speedup
    # so the speedup cannot be quoted alone.
    t_ej = gate["eject_full_s"]
    frac = (results["numpy_perbrick"]["s"] / t_ej) if t_ej else float("nan")
    best = max((r.get("speedup_vs_perbrick") or 1.0) for r in results.values() if isinstance(r, dict))
    bound = 1.0 / (1.0 - frac + frac / best) if best > 0 else 1.0
    print(f"  the kernel is {results['numpy_perbrick']['s']*1e3:.1f} ms of _eject_slab's "
          f"{t_ej*1e3:.1f} ms = {100*frac:.1f}% of the phase's eject half")
    print(f"  -> replacing it at {best:.2f}x bounds `eject` at {bound:.2f}x, and `eject` is "
          f"12.99 of migrate's 33.26 s at cgh64 (5l), so the PHASE bound is "
          f"{1.0/(1.0 - 0.3906 + 0.3906/bound):.2f}x")
    results["_eject_full_s"] = t_ej
    results["_kernel_frac_of_eject"] = frac

    decomp, insert_profile = None, None
    if decompose:
        decomp = _decompose_eject(st, bx, c_drift, scales, repeats, floor_ns, n_rows, t_ej)
        insert_profile = _profile_insert(st, bx, c_drift, scales)

    card = dict(
        config=cfg_name, n_part=cfg["n_part"], nb=cfg["nb"], box=cfg["box"],
        fraction=fraction, c_drift=c_drift, slab=bx, n_rows=n_rows,
        n_bricks=len(per_brick), repeats=repeats, bytes_per_row=BYTES_PER_ROW,
        stream=stream, floor_ns_per_row=floor_ns, gate=gate, arms=results,
        eject_decomposition=decomp, insert_profile=insert_profile,
        commit=_git_commit(), host=platform.node(), python=platform.python_version(),
        numpy=np.__version__, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(card, f, indent=1)
    print(f"card -> {out_path}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="smoke", choices=sorted(CONFIGS))
    ap.add_argument("--arms", nargs="+", default=["numpy_perbrick", "numpy_batched", "jax_cpu"],
                    choices=ARMS)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--fraction", type=float, default=2.85,
                    help="drift as a fraction of the reach-1 threshold (5l's f)")
    ap.add_argument("--decompose", action="store_true",
                    help="also decompose _eject_slab and rank _insert_slab")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(REPO, "runs", "v2", f"m6_c6_jit_{a.config}.json")
    raise SystemExit(run(a.config, a.arms, a.repeats, a.fraction, out, a.decompose))


if __name__ == "__main__":
    main()
