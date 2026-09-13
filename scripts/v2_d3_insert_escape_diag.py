"""Which operation in the compiled insert breaks the int16 bound on a GB200?

Vista 995067: `insert_jax.insert_rows` on a GB200 returned `abs_max` 49424 on
the 32^3 xback state, where CPU XLA and numpy stay within 32767. The kernel takes
each brick's scale as max(|w| * s_old) / 32767 over that brick's rows and then
rescales those same rows, so |code| <= 32767 holds for ANY inputs if every
operation returns its value. An escape therefore means an operation inside the
program returned a wrong value; this script finds which, on the captured inputs.

  capture  Build the xback state, migrate with the compiled eject + insert, and
           save the inputs of the first insert call whose `abs_max` escapes
           (or of call 0, as a control, if none does). Then migrate a fresh copy
           with the numpy eject + compiled insert and save that arm's inputs at
           the same call, so the eject's part in the inputs is read, not assumed.
  analyze  On the saved inputs: a numpy emulation of the kernel op by op; the
           production program (jit) and its eager body; a jitted program that
           returns the intermediates (sort order, brick index, row magnitude,
           per-brick max, abs_max). Each compared to numpy. Then scatter-max
           alone on random duplicate indices (the kernel's form, int64 indices
           into a zeros init, and `segment_max` with int32 indices as the GPU
           kick uses it), unsorted and sorted.

Prints the first disagreement and writes a card plus the inputs npz.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
INT16_MAX = 32767
BIG = int(np.iinfo(np.int64).max)


def _say(msg):
    print(msg, flush=True)


def _probe():
    path = os.path.join(REPO, "scripts", "v2_d3_device_migrate.py")
    spec = importlib.util.spec_from_file_location("d3probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _x64():
    import jax

    jax.config.update("jax_enable_x64", True)


# ------------------------------------------------------------------ capture


def _run_arm(P, st, c, eject_kernel, target):
    """Migrate `st`, hooking insert_rows; returns (inputs dict, call index, escaped)."""
    from inexor import insert_jax, state

    real = insert_jax.insert_rows
    box = dict(n=0, cap=None)

    def hook(dest, off, w, ids, s_old, lo_b, starts, p3):
        out = real(dest, off, w, ids, s_old, lo_b, starts, p3)
        k = box["n"]
        box["n"] += 1
        hit = out["abs_max"] > INT16_MAX if target is None else k == target
        if hit and box["cap"] is None:
            box["cap"] = dict(dest=np.array(dest), off=np.array(off), w=np.array(w),
                              s_old=np.array(s_old, dtype=np.float64), lo_b=int(lo_b),
                              starts=np.array(starts, dtype=np.int64), p3=int(p3),
                              abs_max=float(out["abs_max"]), call=k,
                              scales=np.array(out["scales"]))
        return out

    insert_jax.insert_rows = hook
    escaped = False
    try:
        for _ in range(2):
            state.drift_and_migrate(st, c, kernel=eject_kernel, insert_kernel="jax")
            if box["cap"] is not None:
                break
    except ValueError as e:
        if "escapes int16" not in str(e):
            raise
        escaped = True
    finally:
        insert_jax.insert_rows = real
    return box["cap"], escaped


def capture(args):
    _x64()
    import jax

    P = _probe()
    platform = str(jax.devices()[0].platform)
    st = P._build_state(32, 8, seed=11, brick_slack=0.0, arena_frac=0.30, with_ids=True)
    import copy

    st2 = copy.deepcopy(st)
    c = P._c_drift(st, 1.9)
    cap, escaped = _run_arm(P, st, c, "jax", None)
    if cap is None:
        _say(f"[capture] {platform}: no escape in two steps; capturing call 0 as a control")
        cap, _ = _run_arm(P, copy.deepcopy(st2), c, "jax", 0)
    target = cap["call"]
    ctl, ctl_escaped = _run_arm(P, st2, c, "numpy", target)
    same_inputs = ctl is not None and all(
        np.array_equal(cap[k], ctl[k]) for k in ("dest", "off", "w", "s_old", "starts")) \
        and cap["lo_b"] == ctl["lo_b"]
    _say(f"[capture] {platform}: jax eject -> call {target} abs_max {cap['abs_max']:.0f} "
         f"(escape raised: {escaped}); numpy eject -> call {target} abs_max "
         f"{ctl['abs_max'] if ctl else float('nan'):.0f} (escape raised: {ctl_escaped}); "
         f"inputs identical across ejects: {same_inputs}")
    np.savez(args.npz, **{k: v for k, v in cap.items() if isinstance(v, np.ndarray)},
             meta=json.dumps(dict(lo_b=cap["lo_b"], p3=cap["p3"], call=target,
                                  abs_max=cap["abs_max"], platform=platform,
                                  nb2=int(len(cap["starts"]) - 1))))
    return dict(platform=platform, call=target, abs_max_jax_eject=cap["abs_max"],
                escape_raised_jax_eject=escaped,
                abs_max_numpy_eject=None if ctl is None else ctl["abs_max"],
                escape_raised_numpy_eject=ctl_escaped, inputs_identical=same_inputs,
                rows=int(len(cap["dest"])))


# ------------------------------------------------------------------ analyze


def _np_kernel(dest, w, s_old, lo_b, p3, nb2):
    brick = dest // p3
    inslab = (brick >= lo_b) & (brick < lo_b + nb2)
    order = np.argsort(np.where(inslab, dest, BIG), kind="stable")
    dest_s, w_s, s_s, in_s = dest[order], w[order], s_old[order], inslab[order]
    bl = np.where(in_s, dest_s // p3 - lo_b, 0)
    m_row = np.abs(w_s).max(axis=1).astype(np.float64) * s_s
    vmax = np.zeros(nb2, dtype=np.float64)
    np.maximum.at(vmax, bl, np.where(in_s, m_row, 0.0))
    s_b = vmax / np.full(nb2, 32767.0)
    s_b = np.where(s_b > 0.0, s_b, 1.0)
    ratio = s_s / np.where(in_s, s_b[bl], 1.0)
    out = np.rint(w_s.astype(np.float64) * ratio[:, None])
    abs_max = float(np.max(np.where(in_s[:, None], np.abs(out), 0.0)))
    return dict(order=order, bl=bl, m_row=m_row, vmax=vmax, scales=s_b, abs_max=abs_max,
                in_s=in_s)


def _ops_program(p3, nb2, n_pad):
    import jax
    import jax.numpy as jnp

    @jax.jit
    def ops(dest, w, s_old, real, lo_b, div):
        brick = dest // p3
        inslab = real & (brick >= lo_b) & (brick < lo_b + nb2)
        order = jnp.argsort(jnp.where(inslab, dest, BIG), stable=True)
        dest_s, w_s, s_s, in_s = dest[order], w[order], s_old[order], inslab[order]
        bl = jnp.where(in_s, dest_s // p3 - lo_b, 0)
        m_row = jnp.abs(w_s).max(axis=1).astype(jnp.float64) * s_s
        vmax = jnp.zeros(nb2, dtype=jnp.float64).at[bl].max(jnp.where(in_s, m_row, 0.0))
        s_b = vmax / div
        s_b = jnp.where(s_b > 0.0, s_b, 1.0)
        ratio = s_s / jnp.where(in_s, s_b[bl], 1.0)
        out = jnp.rint(w_s.astype(jnp.float64) * ratio[:, None])
        abs_max = jnp.max(jnp.where(in_s[:, None], jnp.abs(out), 0.0))
        return dict(order=order, bl=bl, m_row=m_row, vmax=vmax, scales=s_b, abs_max=abs_max)

    return ops


def _neq(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    return int(np.count_nonzero(a != b))


def analyze(args):
    _x64()
    import jax
    import jax.numpy as jnp

    from inexor import insert_jax

    platform = str(jax.devices()[0].platform)
    z = np.load(args.npz)
    meta = json.loads(str(z["meta"]))
    dest, w, s_old, starts = z["dest"], z["w"], z["s_old"], z["starts"]
    lo_b, p3, nb2 = meta["lo_b"], meta["p3"], meta["nb2"]
    n = len(dest)
    n_pad = insert_jax._padded(n)
    pad = n_pad - n
    rec = dict(platform=platform, captured_on=meta["platform"], rows=n, n_pad=n_pad,
               nb2=nb2, codes_at_int16_min=int(np.count_nonzero(w == -32768)))

    ref = _np_kernel(dest, w, s_old, lo_b, p3, nb2)
    rec["numpy_abs_max"] = ref["abs_max"]

    def padded(a, fill):
        return a if pad == 0 else np.concatenate(
            [a, np.full((pad,) + a.shape[1:], fill, dtype=a.dtype)])

    real = np.zeros(n_pad, dtype=bool)
    real[:n] = True
    div = np.full(nb2, 32767.0)
    lo = np.asarray(lo_b, dtype=np.int64)

    # the production program, jit and eager body
    fn = insert_jax._build(p3, nb2, n_pad, False)
    args_prod = (jnp.asarray(padded(dest, 0)), jnp.asarray(padded(z["off"], 0)),
                 jnp.asarray(padded(w, 0)), None, jnp.asarray(padded(s_old, 1.0)),
                 jnp.asarray(real), jnp.asarray(lo), jnp.asarray(starts), jnp.asarray(div))
    for tag, f in (("prod_jit", fn), ("prod_eager", getattr(fn, "__wrapped__", None))):
        if f is None:
            rec[tag] = "no __wrapped__"
            continue
        out = f(*args_prod)
        sc = np.asarray(out["scales"])
        bad = np.flatnonzero(sc != ref["scales"])
        rec[tag] = dict(abs_max=float(out["abs_max"]), scales_differ=int(len(bad)),
                        worst_scale_ratio_numpy_over_this=float(
                            np.max(ref["scales"][bad] / sc[bad])) if len(bad) else 1.0)
        _say(f"[analyze {platform}] {tag}: abs_max {float(out['abs_max']):.0f} (numpy "
             f"{ref['abs_max']:.0f}); scales differing {len(bad)} of {nb2}")

    # the intermediates
    ops = _ops_program(p3, nb2, n_pad)
    o = ops(jnp.asarray(padded(dest, 0)), jnp.asarray(padded(w, 0)),
            jnp.asarray(padded(s_old, 1.0)), jnp.asarray(real), jnp.asarray(lo),
            jnp.asarray(div))
    order = np.asarray(o["order"])[:n]
    inter = dict(order_equal=_neq(order, ref["order"]),
                 order_is_permutation=bool(np.array_equal(np.sort(np.asarray(o["order"])),
                                                          np.arange(n_pad))),
                 bl_differ=_neq(np.asarray(o["bl"])[:n], ref["bl"]),
                 m_row_differ=_neq(np.asarray(o["m_row"])[:n], ref["m_row"]),
                 vmax_differ=_neq(o["vmax"], ref["vmax"]),
                 scales_differ=_neq(o["scales"], ref["scales"]),
                 abs_max=float(o["abs_max"]))
    # vmax recomputed in numpy from THIS program's own bl and m_row: separates the
    # scatter from everything upstream of it
    bl_g, m_g = np.asarray(o["bl"]), np.asarray(o["m_row"])
    in_g = np.zeros(n_pad, dtype=bool)
    in_g[:n] = ref["in_s"] if inter["order_equal"] == 0 else True
    v_from_own = np.zeros(nb2)
    np.maximum.at(v_from_own, bl_g[:n], np.where(ref["in_s"], m_g[:n], 0.0)
                  if inter["order_equal"] == 0 else m_g[:n])
    inter["vmax_vs_numpy_scatter_of_own_inputs_differ"] = _neq(o["vmax"], v_from_own)
    rec["intermediates"] = inter
    _say(f"[analyze {platform}] intermediates: {inter}")

    # scatter-max alone
    rng = np.random.default_rng(3)
    sm = []
    for n_rows, nseg in ((4096, 64), (8192, 64), (1 << 20, 65536)):
        idx = rng.integers(0, nseg, n_rows)
        vals = rng.random(n_rows)
        refv = np.zeros(nseg)
        np.maximum.at(refv, idx, vals)
        for sort in (False, True):
            ii = np.sort(idx) if sort else idx
            vv = vals[np.argsort(idx, kind="stable")] if sort else vals
            refs = np.zeros(nseg)
            np.maximum.at(refs, ii, vv)
            i64, i32, jv = jnp.asarray(ii, jnp.int64), jnp.asarray(ii, jnp.int32), jnp.asarray(vv)
            at_eager = jnp.zeros(nseg, jnp.float64).at[i64].max(jv)
            at_jit = jax.jit(lambda i, v, s=nseg: jnp.zeros(s, jnp.float64).at[i].max(v))(i64, jv)
            seg = jnp.maximum(jax.jit(lambda v, i, s=nseg: jax.ops.segment_max(
                v, i, num_segments=s))(jv, i32), 0.0)
            row = dict(rows=n_rows, segments=nseg, sorted=sort,
                       at_int64_eager=_neq(at_eager, refs), at_int64_jit=_neq(at_jit, refs),
                       segment_max_int32_jit=_neq(seg, refs))
            sm.append(row)
            _say(f"[analyze {platform}] scatter-max {row}")
    rec["scatter_max"] = sm
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("capture", "analyze", "both"))
    ap.add_argument("--npz", default=os.path.join(REPO, "runs", "v2",
                                                  "d3_insert_escape_inputs.npz"))
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args(argv)
    out = os.path.join(REPO, "runs", "v2", f"d3_insert_escape_diag{args.out_suffix}.json")
    card = dict(argv=sys.argv, started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                job=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename)
    if args.mode in ("capture", "both"):
        card["capture"] = capture(args)
    if args.mode in ("analyze", "both"):
        card["analyze"] = analyze(args)
    card["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(out, "w") as f:
        json.dump(card, f, indent=1)
    _say(f"card: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
