"""D3 on a GB200: is the compiled migrate bitwise across backends, and what does one 4096^3 slab's migrate cost?

Every arm is a fresh subprocess, so each device high-water mark belongs to one
arm and nothing earlier in the job can set it.

  xback       A state with arena residents, migrated K times twice from the same
              build: numpy eject + numpy insert (the reference, host numpy on any
              backend) against `eject_jax` + `insert_jax` on THIS backend. Every
              state field and every stats dict must be equal after every step. The
              laptop suite proves this on CPU XLA only; on a GB200 it is the
              cross-backend claim. Anti-vacuity first: a row crossing two bricks,
              a schedule that is not all-to-all, a brick overflowing into the
              arena, an arena resident re-homed, and both compiled kernels'
              `CALLS` receipts moving.
  slab-real   A production-geometry state (4096 rows and 512 buckets per brick),
              one serial `drift_and_migrate` with both compiled kernels, the
              public `eject_rows` / `insert_rows` wrapped by a timer. Those calls
              return numpy, so each wrapped wall is upload + compute + readback,
              synchronous. The rest of the step is the host's: per-brick slot
              resolution, the census, the writes, the arena. Reported per step
              beside the numpy migrate of an identical copy, which is also checked
              equal. Two steps, because a padded row count is a fresh XLA shape:
              the kernel cache sizes are on the card.
  slab-shape  One x-slab at 4096^3 brick geometry (`--shape-nb` bricks per side,
              4096 rows per brick: 268,435,456 rows at nb=256), synthetic rows,
              through the compiled eject and then the compiled insert of its own
              output. Each kernel is timed twice: through its public call, and as
              the same program split into host prep, upload, compute and readback,
              each phase ended by `block_until_ready` or a numpy copy. The split
              program's outputs must equal the public call's (the instrument
              measures the program it names). Device peak after each kernel.

Device arms record the platform they ran on and refuse `cpu` unless
`--allow-cpu` (the laptop smoke), so a leg that silently fell back to the CPU
cannot pass as a GPU reading.

Output streams line by line and the card is rewritten after every arm, so a
killed job keeps what finished.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

INT16_MAX = 32767
FIELDS = ("off", "w", "occupancy", "brick_start", "vel_scale", "arena_bucket", "ids")


def _say(msg):
    print(msg, flush=True)


def _platform():
    import jax

    return str(jax.devices()[0].platform)


def _require_device(allow_cpu):
    p = _platform()
    if p == "cpu" and not allow_cpu:
        raise RuntimeError("device arm ran on the CPU backend; refusing to report it "
                           "as a GPU reading (pass --allow-cpu for a laptop smoke)")
    return p


def _peak():
    import jax

    stats = jax.devices()[0].memory_stats()
    return None if not stats else int(stats.get("peak_bytes_in_use", 0))


def _maxrss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**30 if sys.platform == "darwin" else r / 2**20


def _x64():
    import jax

    jax.config.update("jax_enable_x64", True)


def _nb_of(n_part):
    """Bricks per side as the engine chooses them (`plan.engine_config`); off the
    preset list, the family's 4096 rows per brick (n_part / 16)."""
    from inexor.plan import PRESETS, engine_config

    for name, g in PRESETS.items():
        if g["n_part"] == n_part:
            return g["n_fine"] // engine_config(name).n_brick
    return n_part // 16


def _build_state(n_part, nb, seed, brick_slack, arena_frac, with_ids):
    """A perturbed lattice with gaussian velocities, box = n_part / 2."""
    from inexor import state
    from inexor.codec import T9Layout

    n, L = n_part, n_part / 2.0
    sp = L / n
    ax = (np.arange(n) + 0.5) * sp
    x = np.empty((n**3, 3), dtype=np.float64)
    x[:, 0] = np.repeat(ax, n * n)
    x[:, 1] = np.tile(np.repeat(ax, n), n)
    x[:, 2] = np.tile(ax, n * n)
    rng = np.random.default_rng(seed)
    for c in range(3):
        x[:, c] += rng.normal(scale=0.25 * sp, size=n**3)
        np.mod(x[:, c], L, out=x[:, c])
    v = np.random.default_rng(seed + 1).normal(scale=0.5, size=x.shape)
    t9 = T9Layout(box_size=L, n_part=n, bucket_cells=2)
    return state.SlotState.build(x, v, t9, nb, brick_slack=brick_slack,
                                 arena_frac=arena_frac, with_ids=with_ids)


def _c_drift(st, fraction):
    """The drift that moves the fastest particle `fraction` of a brick."""
    extent = float(st.t9.box_size) / int(st.bricks_per_side)
    return fraction * extent / (float(np.max(st.vel_scale)) * INT16_MAX)


def _diff_fields(a, b):
    out = {}
    for name in FIELDS:
        x, y = getattr(a, name), getattr(b, name)
        if x is None or y is None:
            if (x is None) != (y is None):
                out[name] = "None on one side only"
            continue
        n = int(np.count_nonzero(np.asarray(x) != np.asarray(y)))
        if n:
            out[name] = n
    return out


# ------------------------------------------------------------------ the arms


def arm_xback(args):
    _x64()
    from inexor import eject_jax, insert_jax, state

    platform = _require_device(args.allow_cpu)
    n, nb = args.n_part, args.nb or _nb_of(args.n_part)
    t0 = time.perf_counter()
    st_a = _build_state(n, nb, seed=11, brick_slack=0.0, arena_frac=0.30, with_ids=True)
    st_b = copy.deepcopy(st_a)
    c = _c_drift(st_a, args.xback_frac)
    rec = dict(arm="xback", platform=platform, n_part=n, nb=nb, c_drift=c,
               drift_frac=args.xback_frac, build_s=time.perf_counter() - t0, steps=[])
    rc = 0
    overflow, rehomed, reach = 0, 0, 0
    calls0 = (eject_jax.CALLS, insert_jax.CALLS)
    for k in range(args.steps):
        rehomed += st_a.arena_used
        t1 = time.perf_counter()
        r_a = state.drift_and_migrate(st_a, c)
        t2 = time.perf_counter()
        r_b = state.drift_and_migrate(st_b, c, kernel="jax", insert_kernel="jax")
        t3 = time.perf_counter()
        overflow += r_a["n_arena_overflow"]
        reach = max(reach, r_a["brick_reach_realized"])
        diff = _diff_fields(st_a, st_b)
        stats_equal = r_a == r_b
        rec["steps"].append(dict(numpy_s=t2 - t1, jax_s=t3 - t2, stats_equal=stats_equal,
                                 field_diffs=diff, stats=r_a))
        _say(f"[xback] step {k}: numpy {t2 - t1:.1f}s, compiled {t3 - t2:.1f}s; state "
             f"BITWISE = {not diff} {diff or ''}; stats equal = {stats_equal}; "
             f"reach {r_a['brick_reach']} (realized {r_a['brick_reach_realized']}), "
             f"overflow {r_a['n_arena_overflow']}, arena_used {r_a['arena_used']}")
        if diff or not stats_equal:
            rc = 3
    calls = (eject_jax.CALLS - calls0[0], insert_jax.CALLS - calls0[1])
    vacuous = []
    if reach < 2:
        vacuous.append(f"realized reach {reach} < 2")
    if rec["steps"][-1]["stats"]["brick_reach"] >= nb // 2:
        vacuous.append("schedule all-to-all")
    if overflow == 0:
        vacuous.append("no brick overflowed")
    if rehomed == 0:
        vacuous.append("no arena resident re-homed")
    if min(calls) <= 0:
        vacuous.append(f"compiled kernels did not run (eject/insert calls {calls})")
    rec.update(eject_calls=calls[0], insert_calls=calls[1], overflow=overflow,
               residents_rehomed=rehomed, realized_reach=reach, vacuous=vacuous)
    _say(f"[xback] receipts: eject calls {calls[0]}, insert calls {calls[1]}, overflow "
         f"{overflow}, residents re-homed {rehomed}, realized reach {reach}; "
         f"VACUOUS: {vacuous or 'none'}")
    if vacuous and rc == 0:
        rc = 4
    return rec, rc


def arm_slab_real(args):
    _x64()
    from inexor import eject_jax, insert_jax, state

    platform = _require_device(args.allow_cpu)
    n, nb = args.n_part, args.nb or _nb_of(args.n_part)
    t0 = time.perf_counter()
    st = _build_state(n, nb, seed=13, brick_slack=0.10, arena_frac=0.01, with_ids=False)
    build_s = time.perf_counter() - t0
    ref = copy.deepcopy(st)
    c = _c_drift(st, args.real_frac)
    rec = dict(arm="slab-real", platform=platform, n_part=n, nb=nb,
               rows_per_slab=n**3 // nb, c_drift=c, drift_frac=args.real_frac,
               build_s=build_s, steps=[])
    _say(f"[slab-real] n_part={n} nb={nb} ({n**3 // nb:,} rows per slab) built in "
         f"{build_s:.1f}s on {platform}")

    acc = dict(eject_s=0.0, eject_n=0, insert_s=0.0, insert_n=0)
    real_eject, real_insert = eject_jax.eject_rows, insert_jax.insert_rows

    def timed_eject(*a, **k):
        t = time.perf_counter()
        out = real_eject(*a, **k)
        acc["eject_s"] += time.perf_counter() - t
        acc["eject_n"] += 1
        return out

    def timed_insert(*a, **k):
        t = time.perf_counter()
        out = real_insert(*a, **k)
        acc["insert_s"] += time.perf_counter() - t
        acc["insert_n"] += 1
        return out

    # `_eject_slab_jax` / `_insert_slab_jax` import these names at call time
    eject_jax.eject_rows, insert_jax.insert_rows = timed_eject, timed_insert
    rc = 0
    try:
        for k in range(args.steps):
            for key in acc:
                acc[key] = 0 if key.endswith("_n") else 0.0
            ce, ci = len(eject_jax._CACHE), len(insert_jax._CACHE)
            t1 = time.perf_counter()
            r_ref = state.drift_and_migrate(ref, c)
            t2 = time.perf_counter()
            r = state.drift_and_migrate(st, c, kernel="jax", insert_kernel="jax")
            t3 = time.perf_counter()
            diff = _diff_fields(ref, st)
            step = dict(numpy_s=t2 - t1, compiled_s=t3 - t2, eject_calls_s=acc["eject_s"],
                        insert_calls_s=acc["insert_s"], eject_calls=acc["eject_n"],
                        insert_calls=acc["insert_n"],
                        host_s=(t3 - t2) - acc["eject_s"] - acc["insert_s"],
                        new_eject_shapes=len(eject_jax._CACHE) - ce,
                        new_insert_shapes=len(insert_jax._CACHE) - ci,
                        field_diffs=diff, stats_equal=r == r_ref, stats=r)
            rec["steps"].append(step)
            _say(f"[slab-real] step {k}: numpy {step['numpy_s']:.2f}s | compiled "
                 f"{step['compiled_s']:.2f}s = eject calls {acc['eject_s']:.2f}s "
                 f"({acc['eject_n']}) + insert calls {acc['insert_s']:.2f}s "
                 f"({acc['insert_n']}) + host {step['host_s']:.2f}s; new shapes "
                 f"eject {step['new_eject_shapes']} insert {step['new_insert_shapes']}; "
                 f"BITWISE numpy = {not diff and step['stats_equal']}; reach "
                 f"{r['brick_reach']}, overflow {r['n_arena_overflow']}")
            if diff or not step["stats_equal"]:
                rc = 3
            if acc["eject_n"] != nb or acc["insert_n"] != nb:
                _say(f"[slab-real] RECEIPT FAILED: {acc['eject_n']} eject / "
                     f"{acc['insert_n']} insert calls for {nb} slabs")
                rc = max(rc, 4)
    finally:
        eject_jax.eject_rows, insert_jax.insert_rows = real_eject, real_insert
    rec.update(device_peak=_peak(), host_maxrss_gb=_maxrss_gb())
    return rec, rc


def _shape_rows(nb, seed, drift_frac):
    """One x-slab (bx=0) of `nb^2` bricks x 4096 rows at production brick geometry.

    n_part = 16 nb, so 8 buckets per brick side and 64 rows per bucket on average.
    Buckets, offsets, codes and per-brick scales are uniform / gaussian draws;
    the drift moves the fastest row `drift_frac` of a brick.
    """
    from inexor.codec import LEVELS_PER_BUCKET, T9Layout

    n_part = 16 * nb
    t9 = T9Layout(box_size=n_part / 2.0, n_part=n_part, bucket_cells=2)
    per = int(t9.n_buckets_side) // nb
    nb2, rows_b = nb * nb, 4096
    n = nb2 * rows_b
    rng = np.random.default_rng(seed)
    brick = np.repeat(np.arange(nb2, dtype=np.int64), rows_b)
    bijk = np.empty((n, 3), dtype=np.int64)
    bijk[:, 0] = 0
    bijk[:, 1] = (brick // nb) * per
    bijk[:, 2] = (brick % nb) * per
    bijk += rng.integers(0, per, size=(n, 3), dtype=np.int64)
    off = rng.integers(0, LEVELS_PER_BUCKET, size=(n, 3), dtype=np.uint8)
    w = np.clip(rng.normal(scale=8000.0, size=(n, 3)), -INT16_MAX, INT16_MAX).astype(np.int16)
    s_brick = rng.uniform(0.5, 1.5, size=nb2)
    scale = s_brick[brick][:, None]
    extent_levels = per * LEVELS_PER_BUCKET
    c_drift = drift_frac * extent_levels * float(t9.quantum) / (float(s_brick.max()) * INT16_MAX)
    return dict(t9=t9, nb=nb, per=per, n=n, off=off, bijk=bijk, w=w, scale=scale,
                brick=brick, s_brick=s_brick, c_drift=c_drift)


def _sync_timed(fn):
    import jax

    t = time.perf_counter()
    out = jax.block_until_ready(fn())
    return out, time.perf_counter() - t


def _same(a, b):
    return all((x is None and y is None) or
               (x is not None and y is not None and np.array_equal(np.asarray(x), np.asarray(y)))
               for x, y in zip(a, b))


def arm_slab_shape(args):
    _x64()
    import jax.numpy as jnp

    from inexor import eject_jax, insert_jax

    platform = _require_device(args.allow_cpu)
    nb = args.shape_nb
    t0 = time.perf_counter()
    R = _shape_rows(nb, seed=17, drift_frac=args.shape_frac)
    gen_s = time.perf_counter() - t0
    n, t9, nb2, p3 = R["n"], R["t9"], nb * nb, R["per"] ** 3
    rec = dict(arm="slab-shape", platform=platform, nb=nb, rows=n, bricks=nb2,
               drift_frac=args.shape_frac, c_drift=R["c_drift"], gen_s=gen_s, reps=args.reps)
    _say(f"[slab-shape] nb={nb}: one slab of {nb2:,} bricks, {n:,} rows, generated in "
         f"{gen_s:.1f}s on {platform}")

    # ---- eject: the public call
    def eject_public():
        return eject_jax.eject_rows(t9, nb, R["off"], R["bijk"], R["w"], None, R["scale"],
                                    R["c_drift"], R["brick"])

    t = time.perf_counter()
    ref_e = eject_public()
    warm_e = time.perf_counter() - t
    pub_e = []
    for _ in range(args.reps):
        t = time.perf_counter()
        out_e = eject_public()
        pub_e.append(time.perf_counter() - t)
    same_e = _same(ref_e, out_e)

    # ---- eject: the same program, split by phase
    n_pad = eject_jax._padded(n)
    fn_e = eject_jax._CACHE[(int(t9.n_buckets_side), float(t9.quantum), int(nb), n_pad, False)]
    pad = n_pad - n

    def _pad(a, fill):
        if pad == 0:
            return a
        return np.concatenate([a, np.full((pad,) + a.shape[1:], fill, dtype=a.dtype)])

    split_e = []
    for _ in range(args.reps):
        t = time.perf_counter()
        real = np.zeros(n_pad, dtype=bool)
        real[:n] = True
        host = (_pad(R["off"], 0), _pad(R["bijk"], 0), _pad(R["w"], 0),
                _pad(R["scale"], 1.0), _pad(R["brick"], -1), real)
        prep = time.perf_counter() - t
        dev, up = _sync_timed(lambda: tuple(jnp.asarray(a) for a in host))
        res, comp = _sync_timed(lambda: fn_e(dev[0], dev[1], dev[2], None, dev[3],
                                             float(R["c_drift"]), dev[4], dev[5]))
        t = time.perf_counter()
        back = (np.asarray(res[0])[:n], np.asarray(res[1])[:n], np.asarray(res[2])[:n], None,
                np.asarray(res[4])[:n], int(res[5]))
        down = time.perf_counter() - t
        dev = res = None
        split_e.append(dict(prep_s=prep, upload_s=up, compute_s=comp, readback_s=down))
    split_same_e = _same(back, ref_e)
    peak_e = _peak()
    dest, off_new, w_out, _, src, n_keep = ref_e
    rec["eject"] = dict(warm_s=warm_e, public_s=pub_e, public_median_s=float(np.median(pub_e)),
                        split=split_e, public_repeat_equal=same_e,
                        split_equal_public=split_same_e, n_keep=n_keep,
                        leaver_frac=1.0 - n_keep / n, device_peak=peak_e)
    med = {k: float(np.median([s[k] for s in split_e])) for k in split_e[0]}
    _say(f"[slab-shape] eject: warm {warm_e:.2f}s, public median {np.median(pub_e):.3f}s; "
         f"split medians prep {med['prep_s']:.3f} upload {med['upload_s']:.3f} compute "
         f"{med['compute_s']:.3f} readback {med['readback_s']:.3f}s; leavers "
         f"{1.0 - n_keep / n:.4f}; split == public {split_same_e}, repeat equal {same_e}; "
         f"device peak {peak_e / 2**30 if peak_e else 0:.2f} GB")

    # ---- insert: every eject output row, as keepers then leavers (leavers are not
    # bound for this slab and are dropped by the program, as a real slab's rows
    # bound elsewhere would never be passed; the row count is the slab's)
    s_old = R["s_brick"][src]
    cap = int(4096 * 1.10)
    starts = np.arange(nb2 + 1, dtype=np.int64) * cap

    def insert_public():
        return insert_jax.insert_rows(dest, off_new, w_out, None, s_old, 0, starts, p3)

    t = time.perf_counter()
    ref_i = insert_public()
    warm_i = time.perf_counter() - t
    pub_i = []
    for _ in range(args.reps):
        t = time.perf_counter()
        out_i = insert_public()
        pub_i.append(time.perf_counter() - t)
    keys = ("pos", "dest", "off", "w", "occupancy", "scales")
    same_i = all(np.array_equal(ref_i[k], out_i[k]) for k in keys) and \
        ref_i["n_write"] == out_i["n_write"]

    n_pad_i = insert_jax._padded(n)
    fn_i = insert_jax._CACHE[(int(p3), nb2, n_pad_i, False)]
    pad_i = n_pad_i - n

    def _pad_i(a, fill):
        a = np.asarray(a)
        if pad_i == 0:
            return a
        return np.concatenate([a, np.full((pad_i,) + a.shape[1:], fill, dtype=a.dtype)])

    split_i = []
    for _ in range(args.reps):
        t = time.perf_counter()
        real = np.zeros(n_pad_i, dtype=bool)
        real[:n] = True
        host = (_pad_i(dest, 0), _pad_i(off_new, 0), _pad_i(w_out, 0), _pad_i(s_old, 1.0),
                real, np.asarray(0, dtype=np.int64), starts,
                np.full(nb2, 32767.0, dtype=np.float64))
        prep = time.perf_counter() - t
        dev, up = _sync_timed(lambda: tuple(jnp.asarray(a) for a in host))
        res, comp = _sync_timed(lambda: fn_i(dev[0], dev[1], dev[2], None, dev[3], dev[4],
                                             dev[5], dev[6], dev[7]))
        t = time.perf_counter()
        back = {k: (None if v is None else np.asarray(v)) for k, v in res.items()}
        down = time.perf_counter() - t
        dev = res = None
        split_i.append(dict(prep_s=prep, upload_s=up, compute_s=comp, readback_s=down))
    split_same_i = all(np.array_equal(back[k], ref_i[k]) for k in keys) and \
        int(back["n_write"]) == ref_i["n_write"]
    peak_i = _peak()
    rec["insert"] = dict(warm_s=warm_i, public_s=pub_i, public_median_s=float(np.median(pub_i)),
                         split=split_i, public_repeat_equal=same_i,
                         split_equal_public=split_same_i, n_write=ref_i["n_write"],
                         n_spill=ref_i["n_spill"], device_peak_after=peak_i)
    med = {k: float(np.median([s[k] for s in split_i])) for k in split_i[0]}
    _say(f"[slab-shape] insert: warm {warm_i:.2f}s, public median {np.median(pub_i):.3f}s; "
         f"split medians prep {med['prep_s']:.3f} upload {med['upload_s']:.3f} compute "
         f"{med['compute_s']:.3f} readback {med['readback_s']:.3f}s; written "
         f"{ref_i['n_write']:,} spilled {ref_i['n_spill']:,}; split == public "
         f"{split_same_i}, repeat equal {same_i}; device peak so far "
         f"{peak_i / 2**30 if peak_i else 0:.2f} GB")
    rec["host_maxrss_gb"] = _maxrss_gb()
    rc = 0 if (same_e and split_same_e and same_i and split_same_i) else 3
    return rec, rc


def arm_peak_eject(args):
    """Device peak of the compiled eject alone at one 4096^3-geometry slab, and
    whether row counts a fraction of a percent apart share its program."""
    _x64()
    from inexor import eject_jax

    platform = _require_device(args.allow_cpu)
    nb = args.shape_nb
    R = _shape_rows(nb, seed=17, drift_frac=args.shape_frac)
    n, t9 = R["n"], R["t9"]
    calls = []
    for frac in (1.0, 0.997, 0.99):
        m = int(n * frac)
        n_shapes = len(eject_jax._CACHE)
        t = time.perf_counter()
        eject_jax.eject_rows(t9, nb, R["off"][:m], R["bijk"][:m], R["w"][:m], None,
                             R["scale"][:m], R["c_drift"], R["brick"][:m])
        calls.append(dict(rows=m, padded=eject_jax._padded(m), wall_s=time.perf_counter() - t,
                          new_program=len(eject_jax._CACHE) > n_shapes))
    peak = _peak()
    rec = dict(arm="peak-eject", platform=platform, nb=nb, rows=n, calls=calls,
               programs=len(eject_jax._CACHE), device_peak=peak,
               b_per_row=None if peak is None else peak / n,
               b_per_padded_row=None if peak is None else peak / calls[0]["padded"],
               host_maxrss_gb=_maxrss_gb())
    _say(f"[peak-eject] nb={nb} {n:,} rows (padded {calls[0]['padded']:,}): device peak "
         f"{(peak or 0) / 2**30:.2f} GiB = {(peak or 0) / n:.1f} B/row; programs "
         f"{len(eject_jax._CACHE)} over row counts "
         f"{[c['rows'] for c in calls]}; walls {[round(c['wall_s'], 2) for c in calls]} s")
    return rec, 0


def _insert_input(nb, share, reach, seed):
    """Insert rows for slab 0 at 4096^3 brick geometry: a slab's keepers, then the
    emigrants of 2 * reach + 1 source slabs, each `share` of a slab. Emigrant
    destinations are spread evenly over the slabs within reach of their source,
    so the kernel's in-slab mask sees the realistic mix of bound and not bound.
    Rows are not pre-sorted (the kernel sorts them)."""
    rng = np.random.default_rng(seed)
    nb2, p3, rows_b = nb * nb, 512, 4096
    n_slab = nb2 * rows_b
    n_keep = int(n_slab * (1.0 - share))
    n_emig = (2 * reach + 1) * int(n_slab * share)
    dest = np.empty(n_keep + n_emig, dtype=np.int64)
    dest[:n_keep] = rng.integers(0, nb2 * p3, n_keep)
    to_slab = rng.integers(-reach, reach + 1, n_emig) % nb
    dest[n_keep:] = (to_slab * nb2 + rng.integers(0, nb2, n_emig)) * p3 + \
        rng.integers(0, p3, n_emig)
    n = len(dest)
    off = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
    w = np.clip(rng.normal(scale=8000.0, size=(n, 3)), -INT16_MAX, INT16_MAX).astype(np.int16)
    s_old = rng.uniform(0.5, 1.5, size=n)
    starts = np.arange(nb2 + 1, dtype=np.int64) * int(rows_b * 1.10)
    return dest, off, w, s_old, starts, p3, n_slab


def _insert_impl(which):
    """The live kernel, or the frozen pre-cba30c3 one for the A/B.

    Separate MODULES with separate caches on purpose: both key on
    `(p3, nb2, n_pad, has_ids)`, so a shared cache would hand one arm the other's
    kernel and the comparison would be a thing against itself.
    """
    if which == "legacy":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import v2_d3_insert_legacy as mod
        return mod
    from inexor import insert_jax
    return insert_jax


def arm_insert_parity(args):
    """Frozen pre-cba30c3 insert vs the live one: elementwise, plus an HLO receipt.

    The speed and memory arms are `peak-insert --insert-impl`, run as SEPARATE
    processes because a device peak never resets and one process would conflate
    them. This arm is about agreement, so it runs both here and reports no peak.

    The HLO receipt exists because the whole rung rests on a premise a CPU-only
    jaxlib cannot check: that a stable `argsort` over a simple comparator lowers
    to a CUB radix sort, which is what makes the key's WIDTH the thing that
    mattered. Reported, not gated -- if it is absent the change is still
    permutation-identical, it just means the win has a different explanation.
    """
    _x64()
    import jax
    import jax.numpy as jnp

    platform = _require_device(args.allow_cpu)
    nb, share, reach = args.shape_nb, args.emig_share, args.reach
    dest, off, w, s_old, starts, p3, n_slab = _insert_input(nb, share, reach, seed=23)
    n = len(dest)
    outs = {}
    for which in ("legacy", "current"):
        mod = _insert_impl(which)
        outs[which] = mod.insert_rows(dest, off, w, None, s_old, 0, starts, p3)
    a, b = outs["legacy"], outs["current"]
    diffs = {}
    for k in sorted(a):
        va, vb = a[k], b[k]
        if va is None and vb is None:
            continue
        va, vb = np.asarray(va), np.asarray(vb)
        diffs[k] = int((va != vb).sum()) if va.shape == vb.shape else -1
    identical = all(v == 0 for v in diffs.values())

    # the receipt: what the sort actually lowered to on THIS backend
    hlo_hit, hlo_err = None, None
    try:
        nb2 = int(len(starts)) - 1
        n_pad = _insert_impl("current")._padded(n)
        fn = _insert_impl("current")._build(int(p3), nb2, n_pad, False)
        argv = (jnp.zeros(n_pad, jnp.int64), jnp.zeros((n_pad, 3), jnp.uint8),
                jnp.zeros((n_pad, 3), jnp.int16), None,
                jnp.ones(n_pad, jnp.float64), jnp.ones(n_pad, bool),
                jnp.asarray(0, jnp.int64),
                jnp.asarray(np.asarray(starts, dtype=np.int64)),
                jnp.full(nb2, 32767.0))
        text = jax.jit(fn).lower(*argv).compile().as_text().lower()
        hlo_hit = sorted({t for t in ("cub", "radix", "bitonic", "sort") if t in text})
    except Exception as e:  # a receipt must never be the reason a leg dies
        hlo_err = f"{e.__class__.__name__}: {e}"

    rec = dict(arm="insert-parity", platform=platform, nb=nb, emig_share=share,
               reach=reach, slab_rows=n_slab, rows=n, field_diffs=diffs,
               identical=identical, hlo_tokens=hlo_hit, hlo_error=hlo_err,
               host_maxrss_gb=_maxrss_gb())
    _say(f"[insert-parity] {n:,} rows: identical={identical} diffs={diffs}")
    _say(f"[insert-parity] HLO tokens {hlo_hit}" + (f" (receipt failed: {hlo_err})"
                                                    if hlo_err else ""))
    # rc 3 = the arms disagree, which stops adoption; a failed receipt does not
    return rec, 0 if identical else 3


def arm_peak_insert(args):
    """Device peak of the compiled insert alone at a given emigrant share and reach."""
    _x64()
    insert_jax = _insert_impl(args.insert_impl)

    platform = _require_device(args.allow_cpu)
    nb, share, reach = args.shape_nb, args.emig_share, args.reach
    t0 = time.perf_counter()
    dest, off, w, s_old, starts, p3, n_slab = _insert_input(nb, share, reach, seed=23)
    gen_s = time.perf_counter() - t0
    n = len(dest)
    walls = []
    out = None
    for _ in range(2):
        t = time.perf_counter()
        out = insert_jax.insert_rows(dest, off, w, None, s_old, 0, starts, p3)
        walls.append(time.perf_counter() - t)
    peak = _peak()
    rec = dict(arm="peak-insert", impl=args.insert_impl, platform=platform, nb=nb,
               emig_share=share, reach=reach,
               slab_rows=n_slab, rows=n, padded=insert_jax._padded(n), gen_s=gen_s,
               walls_s=walls, n_write=out["n_write"], n_spill=out["n_spill"],
               device_peak=peak, b_per_row=None if peak is None else peak / n,
               b_per_slab_row=None if peak is None else peak / n_slab,
               host_maxrss_gb=_maxrss_gb())
    _say(f"[peak-insert/{args.insert_impl}] share {share:.2f} reach {reach}: {n:,} input rows "
         f"({n / n_slab:.2f} slabs): device peak {(peak or 0) / 2**30:.2f} GiB = "
         f"{(peak or 0) / n:.1f} B/input row = {(peak or 0) / n_slab:.1f} B/slab row; "
         f"walls {[round(x, 2) for x in walls]} s; written {out['n_write']:,} spilled "
         f"{out['n_spill']:,}")
    return rec, 0


def arm_device_migrate(args):
    """`device.migrate.drift_and_migrate_device` against the serial numpy migrate of
    an identical copy: bitwise per step, walls, programs compiled, device peak.

    `device-xback`: zero slack, 30% arena, drift 1.9 bricks (reach 2, overflow);
    `device-real`: production slack 0.10, 1% arena, drift `--real-frac`."""
    _x64()
    from inexor import state
    from inexor.device import migrate

    platform = _require_device(args.allow_cpu)
    heavy = args.arm == "device-xback"
    n, nb = args.n_part, args.nb or _nb_of(args.n_part)
    t0 = time.perf_counter()
    st = _build_state(n, nb, seed=11 if heavy else 13, brick_slack=0.0 if heavy else 0.10,
                      arena_frac=0.30 if heavy else 0.01, with_ids=heavy)
    build_s = time.perf_counter() - t0
    ref = copy.deepcopy(st)
    frac = args.xback_frac if heavy else args.real_frac
    c = _c_drift(st, frac)
    rec = dict(arm=args.arm, platform=platform, n_part=n, nb=nb, rows_per_slab=n**3 // nb,
               c_drift=c, drift_frac=frac, build_s=build_s, steps=[])
    _say(f"[{args.arm}] n_part={n} nb={nb} ({n**3 // nb:,} rows per slab) built in "
         f"{build_s:.1f}s on {platform}")
    rc = 0
    for k in range(args.steps):
        t1 = time.perf_counter()
        r_ref = state.drift_and_migrate(ref, c)
        t2 = time.perf_counter()
        p0 = len(migrate._PROGRAMS)
        r = migrate.drift_and_migrate_device(st, c)
        t3 = time.perf_counter()
        receipt = r.pop("migrate_device")
        diff = _diff_fields(ref, st)
        step = dict(numpy_s=t2 - t1, device_s=t3 - t2, new_programs=len(migrate._PROGRAMS) - p0,
                    field_diffs=diff, stats_equal=r == r_ref, stats=r, receipt=receipt)
        rec["steps"].append(step)
        _say(f"[{args.arm}] step {k}: numpy {t2 - t1:.2f}s | device {t3 - t2:.2f}s; new "
             f"programs {step['new_programs']}; BITWISE numpy = {not diff and r == r_ref} "
             f"{diff or ''}; reach {r['brick_reach']} (realized {r['brick_reach_realized']}), "
             f"overflow {r['n_arena_overflow']}, arena_used {r['arena_used']}")
        if diff or r != r_ref:
            rc = 3
    # one more step with synced phase timers, AFTER the untimed ones: syncing moves
    # the wall (record sec. 21), so the untimed walls above are the step's cost and
    # this one is its breakdown
    phases = {}
    t4 = time.perf_counter()
    r_ref = state.drift_and_migrate(ref, c)
    t5 = time.perf_counter()
    r = migrate.drift_and_migrate_device(st, c, timings=phases)
    t6 = time.perf_counter()
    r.pop("migrate_device")
    diff = _diff_fields(ref, st)
    rec["timed_step"] = dict(numpy_s=t5 - t4, device_s=t6 - t5, phases=phases,
                             phases_sum_s=sum(phases.values()), field_diffs=diff,
                             stats_equal=r == r_ref)
    _say(f"[{args.arm}] timed step: numpy {t5 - t4:.2f}s | device {t6 - t5:.2f}s (synced; "
         f"phases sum {sum(phases.values()):.2f}s); BITWISE numpy = {not diff and r == r_ref}")
    for k, v in sorted(phases.items(), key=lambda kv: -kv[1]):
        _say(f"    {v:8.3f} s  {k}")
    if diff or r != r_ref:
        rc = 3
    rec.update(device_peak=_peak(), host_maxrss_gb=_maxrss_gb())
    _say(f"[{args.arm}] device peak {(rec['device_peak'] or 0) / 2**30:.2f} GiB")
    return rec, rc


def arm_device_fused(args):
    """M4: the device migrate + repack at `device-real`'s state, `--pass separate`
    (`drift_and_migrate_device` then `repack_device`) or `--pass fused`
    (`device.fused.migrate_repack_device`), each against the serial numpy migrate +
    repack of an identical copy: bitwise per step, walls, device peak, and one
    synced per-phase step. The census is the numpy migrate's per-brick membership
    (the engine takes it from the tile loop; `tests/test_device_census.py` gates
    the two equal)."""
    _x64()
    from inexor import state
    from inexor.device import fused, migrate, repack

    platform = _require_device(args.allow_cpu)
    n, nb = args.n_part, args.nb or _nb_of(args.n_part)
    slack = 0.10
    t0 = time.perf_counter()
    st = _build_state(n, nb, seed=13, brick_slack=slack, arena_frac=0.01, with_ids=False)
    build_s = time.perf_counter() - t0
    ref = copy.deepcopy(st)
    c = _c_drift(st, args.real_frac)
    rec = dict(arm=args.arm, impl=args.pass_, platform=platform, n_part=n, nb=nb,
               rows_per_slab=n**3 // nb, c_drift=c, drift_frac=args.real_frac,
               build_s=build_s, steps=[])
    _say(f"[{args.arm}/{args.pass_}] n_part={n} nb={nb} ({n**3 // nb:,} rows per slab) "
         f"built in {build_s:.1f}s on {platform}")

    def one(timings=None):
        t1 = time.perf_counter()
        r_ref = state.drift_and_migrate(ref, c)
        census = repack.repack_geometry(ref, slack)[1]
        p_ref = ref.repack(brick_slack=slack)
        t2 = time.perf_counter()
        if args.pass_ == "fused":
            m, r = fused.migrate_repack_device(st, c, census, brick_slack=slack, timings=timings)
            t3 = time.perf_counter()
            t_m, t_r = t3 - t2, 0.0
        else:
            tm = None if timings is None else {}
            tr = None if timings is None else {}
            m = migrate.drift_and_migrate_device(st, c, timings=tm)
            t_mid = time.perf_counter()
            r = repack.repack_device(st, brick_slack=slack, timings=tr)
            t3 = time.perf_counter()
            t_m, t_r = t_mid - t2, t3 - t_mid
            if timings is not None:
                timings.update({f"migrate | {k}": v for k, v in tm.items()})
                timings.update({f"repack | {k}": v for k, v in tr.items()})
        m.pop("migrate_device")
        r = {k: v for k, v in r.items() if k not in ("repack_device", "scratch_bytes")}
        p_ref = {k: v for k, v in p_ref.items() if k != "scratch_bytes"}
        diff = _diff_fields(ref, st)
        equal = m == r_ref and r == p_ref and not diff and st.arena_base == ref.arena_base
        return dict(numpy_s=t2 - t1, device_s=t3 - t2, migrate_s=t_m, repack_s=t_r,
                    field_diffs=diff, stats_equal=m == r_ref and r == p_ref, bitwise=equal)

    rc = 0
    for k in range(args.steps):
        stp = one()
        rec["steps"].append(stp)
        _say(f"[{args.arm}/{args.pass_}] step {k}: numpy {stp['numpy_s']:.2f}s | device "
             f"{stp['device_s']:.2f}s (migrate {stp['migrate_s']:.2f}, repack "
             f"{stp['repack_s']:.2f}); BITWISE numpy = {stp['bitwise']} {stp['field_diffs'] or ''}")
        rc = rc if stp["bitwise"] else 3
    phases = {}
    stp = one(phases)
    rec["timed_step"] = dict(stp, phases=phases, phases_sum_s=sum(phases.values()))
    _say(f"[{args.arm}/{args.pass_}] timed step: device {stp['device_s']:.2f}s (synced; phases "
         f"sum {sum(phases.values()):.2f}s); BITWISE numpy = {stp['bitwise']}")
    for key, v in sorted(phases.items(), key=lambda kv: -kv[1]):
        _say(f"    {v:8.3f} s  {key}")
    rc = rc if stp["bitwise"] else 3
    rec.update(device_peak=_peak(), host_maxrss_gb=_maxrss_gb())
    _say(f"[{args.arm}/{args.pass_}] device peak {(rec['device_peak'] or 0) / 2**30:.2f} GiB")
    return rec, rc


ARMS = {"xback": arm_xback, "slab-real": arm_slab_real, "slab-shape": arm_slab_shape,
        "peak-eject": arm_peak_eject, "peak-insert": arm_peak_insert,
        "insert-parity": arm_insert_parity,
        "device-xback": arm_device_migrate, "device-real": arm_device_migrate,
        "device-fused": arm_device_fused}


# ------------------------------------------------------------ the orchestrator


def _run_worker(argv, env_extra, tag):
    env = dict(os.environ, PYTHONUNBUFFERED="1", **env_extra)
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", *argv]
    _say(f"\n--- arm {tag}: {' '.join(argv)} {env_extra or ''}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=env)
    rec = None
    for line in p.stdout:
        if line.startswith("WORKER_JSON "):
            rec = json.loads(line[len("WORKER_JSON "):])
        else:
            print(line, end="", flush=True)
    rc = p.wait()
    if rec is None:
        _say(f"--- arm {tag}: rc={rc}, NO RECORD CAME BACK (not a reading)")
        return dict(arm=tag, rc=rc, record=None), max(rc, 1)
    _say(f"--- arm {tag}: rc={rc}")
    return dict(arm=tag, rc=rc, record=rec), rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--arm", choices=sorted(ARMS), help=argparse.SUPPRESS)
    ap.add_argument("--n-part", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--nb", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--arms", default="xback,slab-real,slab-shape",
                    help="comma-separated; timing arms do not run if xback fails")
    ap.add_argument("--xback-n", type=int, default=256, help="particles per side (cdev)")
    ap.add_argument("--xback-frac", type=float, default=1.9,
                    help="drift of the fastest row in bricks (>1 so rows cross two)")
    ap.add_argument("--real-n", type=int, default=512, help="particles per side (cgh64)")
    ap.add_argument("--real-frac", type=float, default=0.5)
    ap.add_argument("--shape-nb", type=int, default=256, help="bricks per side (c-hero 256)")
    ap.add_argument("--shape-frac", type=float, default=0.5)
    ap.add_argument("--emig-share", type=float, default=0.05, help=argparse.SUPPRESS)
    ap.add_argument("--reach", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--insert-impl", choices=("current", "legacy"), default="current",
                    help="which insert kernel `peak-insert` times: the live one, or "
                         "the frozen pre-cba30c3 oracle in v2_d3_insert_legacy.py")
    ap.add_argument("--pass", dest="pass_", choices=("separate", "fused"), default="fused",
                    help="device-fused: the separate device migrate + repack, or the fused")
    ap.add_argument("--peaks", default="",
                    help="'+'-separated peak arms, each its own process: 'eject' and/or "
                         "'insert:SHARE:REACH' (e.g. eject+insert:0.05:1+insert:0.2:2)")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny everything, CPU allowed: exercises the apparatus only")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args(argv)

    if args.worker:
        rec, rc = ARMS[args.arm](args)
        print("WORKER_JSON " + json.dumps(rec), flush=True)
        return rc

    xb_nb = []
    if args.smoke:
        args.xback_n, args.real_n, args.shape_nb, args.reps = 32, 64, 8, 1
        xb_nb = ["--nb", "8"]
    common = ["--steps", str(args.steps), "--reps", str(args.reps)] + \
        (["--allow-cpu"] if args.allow_cpu or args.smoke else [])
    out = os.path.join(REPO, "runs", "v2", f"d3_device_migrate{args.out_suffix}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    card = dict(argv=sys.argv, started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename,
                commit=subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
                arms=[])

    def write():
        with open(out, "w") as f:
            json.dump(card, f, indent=1)

    arms = [a for a in args.arms.split(",") if a]
    worst = 0
    if "xback" in arms:
        res, rc = _run_worker(["--arm", "xback", "--n-part", str(args.xback_n), *xb_nb,
                               "--xback-frac", str(args.xback_frac), *common], {}, "xback")
        card["arms"].append(res)
        write()
        worst = max(worst, rc)
        if rc:
            _say("\nFATAL: the cross-backend identity did not hold, was vacuous, or did "
                 "not run; no timing is read as the compiled migrate's")
            card["verdict"] = "cross-backend identity FAILED, VACUOUS or missing"
            write()
            return 1
    if "slab-real" in arms:
        res, rc = _run_worker(["--arm", "slab-real", "--n-part", str(args.real_n),
                               "--real-frac", str(args.real_frac), *common], {}, "slab-real")
        card["arms"].append(res)
        write()
        worst = max(worst, rc)
    if "slab-shape" in arms:
        res, rc = _run_worker(["--arm", "slab-shape", "--shape-nb", str(args.shape_nb),
                               "--shape-frac", str(args.shape_frac), *common], {},
                              "slab-shape")
        card["arms"].append(res)
        write()
        worst = max(worst, rc)
    for name, n_arm, frac_flag in (
            ("device-xback", args.xback_n, ["--xback-frac", str(args.xback_frac)]),
            ("device-real", args.real_n, ["--real-frac", str(args.real_frac)])):
        if name in arms:
            res, rc = _run_worker(["--arm", name, "--n-part", str(n_arm),
                                   *(xb_nb if name == "device-xback" else []), *frac_flag,
                                   *common], {}, name)
            card["arms"].append(res)
            write()
            worst = max(worst, rc)
    if "device-fused" in arms:
        res, rc = _run_worker(["--arm", "device-fused", "--n-part", str(args.real_n),
                               "--real-frac", str(args.real_frac), "--pass", args.pass_,
                               *common], {}, f"device-fused/{args.pass_}")
        card["arms"].append(res)
        write()
        worst = max(worst, rc)
    if "insert-parity" in arms:
        res, rc = _run_worker(["--arm", "insert-parity", "--shape-nb", str(args.shape_nb),
                               "--emig-share", str(args.emig_share),
                               "--reach", str(args.reach), *common],
                              {}, "insert-parity")
        card["arms"].append(res)
        write()
        worst = max(worst, rc)

    # peak arms run even if one before them fails: an out-of-memory at a large
    # share is itself the reading, and the smaller specs are still worth having
    for spec in [s for s in args.peaks.split("+") if s]:
        if spec == "eject":
            argv_p = ["--arm", "peak-eject", "--shape-nb", str(args.shape_nb),
                      "--shape-frac", str(args.shape_frac)]
        else:
            # insert:SHARE:REACH[:IMPL] -- IMPL defaults to the live kernel, so
            # every spec written before the A/B existed keeps its meaning
            parts = spec.split(":")
            _, share, reach = parts[:3]
            impl = parts[3] if len(parts) > 3 else "current"
            argv_p = ["--arm", "peak-insert", "--shape-nb", str(args.shape_nb),
                      "--emig-share", share, "--reach", reach, "--insert-impl", impl]
        res, rc = _run_worker([*argv_p, *common], {}, f"peak {spec}")
        card["arms"].append(res)
        write()
        worst = max(worst, rc)
    card["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write()
    _say(f"\ncard: {out}\nworst rc {worst}")
    return 0 if worst == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
