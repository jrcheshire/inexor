"""Device migrate and repack fused into one visit per slab, bitwise the two passes in sequence.

The insert's outputs stay on the card and the slab's new block is built from them directly, so a
slab crosses the bus once each way instead of twice. The repack can run before the migrate ends:
- its only global input, `new_start`, comes from the destination census (`device.window`, counted
  with the migrate's own eject kernel) via `repack.capacity_from_counts`;
- it folds arena residents into their bricks and zeroes everything past the new allocation, so the
  migrate's arena layout leaves no trace; the fold-in depends only on each brick's (bucket, slot)
  resident order, which is the spill order of that slab's single insert (`_to_arena` claims the
  lowest free slots ascending).

`state._replay_arena_pass` still runs as in the device migrate (consumed-emigrant census,
arena-full refusal, stats); its arena writes are erased by the repack tail, except where the new
allocation reaches past the old `arena_base`, whose rows are saved across the replay and restored.

Before a block is written, every brick's inserted membership must equal its census count, else it
refuses; earlier slabs may already be written, so a refusal leaves the state invalid (as any
mid-pass refusal does). A new block may overlap a later slab's old range: on the same card that
slab is uploaded first; across cards every such slab is uploaded before any card writes
(`repack._cross_card_slabs`).
"""

from __future__ import annotations

import numpy as np

from . import migrate as _m
from .repack import _cross_card_slabs, _repack_program, capacity_from_counts

#: Call count of fused passes.
CALLS = 0

#: Estimated device peak bytes per slab row of the repack block, charged to the per-slab budget.
REPACK_B_PER_SLAB_ROW = 70.0


def _block_inputs_program(cap_i, cap_s, w_cap, nb2, p3, has_ids):
    """Repack inputs from one slab's insert on the card: the post-insert slot window (written
    rows at their slab-relative slots, spills appended at `span`) and the per-brick index."""

    def make():
        import jax
        import jax.numpy as jnp

        @jax.jit
        def inputs(pos, off, w, ids, occupancy, g_dest, g_off, g_w, g_ids, n_write, n_spill,
                   s0, span, lo_b):
            i = jnp.arange(cap_i, dtype=jnp.int64)
            rel = jnp.where(i < n_write, pos - s0, w_cap)
            k = jnp.arange(cap_s, dtype=jnp.int64)
            spill = k < n_spill
            rk = jnp.where(spill, span + k, w_cap)
            off_win = jnp.zeros((w_cap, 3), off.dtype).at[rel].set(off, mode="drop")
            off_win = off_win.at[rk].set(g_off, mode="drop")
            w_win = jnp.zeros((w_cap, 3), w.dtype).at[rel].set(w, mode="drop")
            w_win = w_win.at[rk].set(g_w, mode="drop")
            ids_win = None
            if has_ids:
                ids_win = jnp.zeros((w_cap,), ids.dtype).at[rel].set(ids, mode="drop")
                ids_win = ids_win.at[rk].set(g_ids, mode="drop")
            occ = occupancy.reshape(nb2, p3)
            live = occ.astype(jnp.int64).sum(axis=1)
            ar_counts = jnp.zeros(nb2, jnp.int64).at[
                jnp.where(spill, g_dest // p3 - lo_b, nb2)].add(1, mode="drop")
            zero = jnp.zeros(1, jnp.int64)
            ar_offsets = jnp.concatenate([zero, jnp.cumsum(ar_counts)])
            row_offsets = jnp.concatenate([zero, jnp.cumsum(live + ar_counts)])
            ar_bucket = jnp.where(spill, g_dest - lo_b * p3, nb2 * p3)
            return (off_win, w_win, ids_win, occ, live, row_offsets, ar_offsets, ar_bucket,
                    live + ar_counts)

        return inputs

    return _m._program(("fused_inputs", cap_i, cap_s, w_cap, nb2, p3, has_ids), make)


def migrate_repack_device(st, c_drift, census_counts, brick_slack=0.10, max_staged_slabs=None,
                          timings=None, device_budget_bytes=None, devices=None):
    """`drift_and_migrate_device` followed by `repack_device`, in one pass per slab.

    `census_counts` is the per-brick post-migrate membership, int64 over
    `st.n_bricks` (`tile_loop_windowed(census=c_drift)`, summed over cards).
    `brick_slack`, `max_staged_slabs`, `timings`, `device_budget_bytes` and
    `devices` as in the two passes. Same mutations as the two passes in sequence.

    Returns `(migrate_stats, repack_stats)`: the migrate's stats dict with its
    `migrate_device` receipt (plus `fused`), and the repack's dict with a
    `repack_device` receipt naming the fused pass.
    """
    import jax.numpy as jnp

    from .. import state as _state

    global CALLS
    CALLS += 1
    nb, p3 = int(st.bricks_per_side), int(st.buckets_per_brick)
    nb2 = nb * nb
    has_ids = st.ids is not None
    counts = np.asarray(census_counts)
    if counts.dtype != np.int64 or counts.shape != (int(st.n_bricks),):
        raise ValueError(f"census_counts must be int64 over {st.n_bricks} bricks, got "
                         f"{counts.dtype} {counts.shape}")
    if int(counts.sum()) != int(st.n_particles):
        raise ValueError(f"the census counts {int(counts.sum())} rows against "
                         f"{st.n_particles} particles stored")
    new_start, n_alloc = capacity_from_counts(st, counts, brick_slack)
    old_start = np.asarray(st.brick_start, dtype=np.int64).copy()
    # the owned buckets only; `bucket_lo` is the flat ordinal of `new_occ[0]`
    new_occ = np.zeros(st.n_buckets, dtype=st.index_dtype)
    bucket_lo = int(st.bucket_lo)
    row_bytes = st.off.itemsize * 3 + st.w.itemsize * 3 + (st.ids.itemsize if has_ids else 0)
    accs = {}

    def acc(k):
        return accs.setdefault(k, dict(readahead=0, early=0, blocks=0, empty=0, scratch=0))

    def before_sweep(ctx):
        if ctx["W"] == 1:
            return
        early = _cross_card_slabs(old_start, new_start, ctx["parts"], nb2)

        def upload(k):
            card = ctx["cards"][k]
            for t in early[k]:
                if t not in card["ejected"] and t not in card["pre"]:
                    card["pre"][t] = _m._upload_slab(st, t, ctx["ar_slots"], ctx["ar_bricks"],
                                                     card["clock"], card["dev"], block=True)
                    acc(k)["early"] += 1

        ctx["run"](upload)

    def insert(st, d, reach, staged, scales_dev, clock, budget, dev, card):
        e = staged[d]
        lo_b, hi_b, s0, span = e["lo_b"], e["hi_b"], e["s0"], e["span"]
        out, nw, ns, consumed, cap_i = _m._insert_on_card(st, d, reach, staged, scales_dev,
                                                          clock, budget, dev)
        scales_h = _m._host(out["scales"], "insert: scales")
        clock.mark("insert: occupancy + scales to host")
        spills, rows = _m._spills(st, out, nw, ns, cap_i, dev)
        clock.mark("insert: spills to host")

        n_lo, n_hi = int(new_start[lo_b]), int(new_start[hi_b])
        n_rows = nw + ns
        a = acc(card["k"])
        if n_rows == 0 and n_hi == n_lo:
            if int(counts[lo_b:hi_b].sum()):
                raise AssertionError(f"slab {d} inserted no rows against a census of "
                                     f"{int(counts[lo_b:hi_b].sum())}")
            a["empty"] += 1
            st.vel_scale[lo_b:hi_b] = scales_h
            return dict(consumed=consumed, spills=spills, n_over=ns)

        a_cap = _m._ladder(ns)
        w_cap = _m._ladder(span + a_cap)
        if rows is None:
            rows = (_m._zeros(a_cap, jnp.int64, dev), _m._zeros((a_cap, 3), jnp.uint8, dev),
                    _m._zeros((a_cap, 3), jnp.int16, dev),
                    _m._zeros(a_cap, jnp.int32, dev) if has_ids else None)
        prep = _block_inputs_program(cap_i, a_cap, w_cap, nb2, p3, has_ids)
        (off_win, w_win, ids_win, occ, live, row_offsets, ar_offsets, ar_bucket,
         members) = prep(out["pos"], out["off"], out["w"], out["ids"], out["occupancy"], *rows,
                         _m._put(nw, dev, jnp.int64), _m._put(ns, dev, jnp.int64),
                         _m._put(s0, dev, jnp.int64), _m._put(span, dev, jnp.int64),
                         _m._put(lo_b, dev, jnp.int64))
        out = rows = None
        clock.mark("fused: block inputs", off_win, w_win, ids_win, members)

        got = _m._host(members, "fused: self-check")
        want = counts[lo_b:hi_b]
        bad = np.flatnonzero(got != want)
        if len(bad):
            b = int(bad[0])
            raise AssertionError(
                f"slab {d}: brick {lo_b + b} holds {int(got[b])} rows after its insert "
                f"against a census of {int(want[b])} ({len(bad)} bricks differ). Its new "
                "range was sized from the census, so the block is not written.")
        clock.mark("fused: self-check")

        cap = _m._ladder(n_rows)
        out_cap = _m._ladder(n_hi - n_lo)
        held = sum(sum(int(x.nbytes) for x in p["win"] if x is not None)
                   for p in card["pre"].values())
        budget.check(f"repacking slab {d}", held, REPACK_B_PER_SLAB_ROW, n_rows)
        block = _repack_program(cap, w_cap, a_cap, out_cap, nb2, p3, has_ids, st.off.dtype,
                                st.w.dtype, None if not has_ids else st.ids.dtype)
        out_off, out_w, out_ids, occ_s = block(
            occ, live, _m._put(old_start[lo_b:hi_b] - s0, dev), row_offsets, ar_offsets,
            ar_bucket, _m._put(new_start[lo_b:hi_b] - n_lo, dev), off_win, w_win, ids_win,
            _m._put(n_rows, dev, np.int64), _m._put(span, dev, np.int64))
        off_win = w_win = ids_win = occ = live = row_offsets = ar_offsets = ar_bucket = None
        clock.mark("fused: block program", out_off, out_w, out_ids, occ_s)

        # upload every later slab of this card whose old range this write overlaps
        t = d + 1
        while t < card["hi"] and int(old_start[t * nb2]) < n_hi:
            if t not in card["ejected"] and t not in card["pre"]:
                card["pre"][t] = _m._upload_slab(st, t, card_ar[0], card_ar[1], clock, dev,
                                                 block=True)
                a["readahead"] += 1
            t += 1
        clock.mark("fused: read-ahead")

        m = n_hi - n_lo
        st.off[n_lo:n_hi] = _m._host(out_off, "fused: block")[:m]
        st.w[n_lo:n_hi] = _m._host(out_w, "fused: block")[:m]
        if has_ids:
            st.ids[n_lo:n_hi] = _m._host(out_ids, "fused: block")[:m]
        new_occ[lo_b * p3 - bucket_lo: hi_b * p3 - bucket_lo] = _m._host(occ_s,
                                                                         "fused: occupancy")
        st.vel_scale[lo_b:hi_b] = scales_h
        a["blocks"] += 1
        a["scratch"] = max(a["scratch"], m * row_bytes)
        clock.mark("fused: write-back")
        return dict(consumed=consumed, spills=spills, n_over=ns)

    card_ar = _m.pass_arena_index(st)
    ctx = _m._device_pass(st, c_drift, timings=timings, device_budget_bytes=device_budget_bytes,
                          devices=devices, insert=insert, keep_window=False,
                          before_sweep=before_sweep)
    clock = ctx["clocks"][0]

    # the migrate's census, arena-full refusal and stats, exactly as the device
    # migrate runs them; its arena writes are erased by the tail below, except over
    # the blocks where the new allocation overlaps the old arena
    a_lo, a_hi = int(st.arena_base), min(n_alloc, int(st.arena_base) + int(st.n_arena))
    kept = [None if x is None else x[a_lo:a_hi].copy() for x in (st.off, st.w, st.ids)]
    rep = _state._replay_arena_pass(
        st, ctx["reach"], ctx["r"], ctx["r_raw"], c_drift, ctx["scales"], n_emig=ctx["n_emig"],
        rr_by_slab=ctx["rr_by_slab"], insert_res=ctx["insert_res"],
        max_staged_slabs=max_staged_slabs, census_note=" (Fused pass.)")
    for x, k in zip((st.off, st.w, st.ids), kept):
        if x is not None and a_hi > a_lo:
            x[a_lo:a_hi] = k
    clock.mark("pass: arena replay")
    n_before = ctx["n_before"]
    n_after = _state.occupancy_total(new_occ)
    if n_after != n_before:
        raise ValueError(
            f"the fused migrate + repack lost {n_before - n_after} particles ({n_before} -> "
            f"{n_after} against {st.n_particles} stored). Particles are never dropped, so this is "
            f"corruption, not imprecision. {rep['n_inserted']} of {nb} slabs inserted.")
    _m._merge_card_timings(ctx, timings)
    receipt = _m._pass_receipt(ctx, devices, device_budget_bytes, rep["spill_rows"])
    receipt["fused"] = True
    migrate_stats = _m._migrate_stats(st, ctx, rep, n_after, receipt)

    spilled = np.zeros(int(st.n_bricks), dtype=np.int64)
    for res in ctx["insert_res"].values():
        for b, dest_r, *_ in res["spills"]:
            spilled[b] += len(dest_r)
    run_counts = counts - spilled

    # the repack's tail, as `repack_device` ends
    st.off[n_alloc:] = 0
    st.w[n_alloc:] = 0
    if has_ids:
        st.ids[n_alloc:] = -1
    st.brick_start[...] = new_start
    st.occupancy[...] = new_occ
    st.arena_base = n_alloc
    st.arena_bucket[:] = -1
    st._invalidate_arena_index()
    clock.mark("pass: tail")

    a_all = [acc(k) for k in range(ctx["W"])]
    rreceipt = dict(fused=True, slabs=nb, blocks=int(sum(a["blocks"] for a in a_all)),
                    readahead_uploads=int(sum(a["readahead"] for a in a_all)),
                    empty_slabs=int(sum(a["empty"] for a in a_all)))
    if devices is not None:
        rreceipt.update(cards=ctx["W"], cross_card_early_uploads=int(sum(a["early"]
                                                                          for a in a_all)))
    repack_stats = dict(
        slots_used=n_alloc,
        slots_per_particle=n_alloc / max(st.n_particles, 1),
        scratch_bytes=int(new_occ.nbytes + max(a["scratch"] for a in a_all)),
        bricks_fast=int(np.count_nonzero((spilled == 0) & (run_counts > 0))),
        bricks_merged=int(np.count_nonzero(spilled > 0)),
        repack_device=rreceipt,
    )
    return migrate_stats, repack_stats
