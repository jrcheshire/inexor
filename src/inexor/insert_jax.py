"""Compiled twin of `_insert_slab`'s row work: group, scale, rescale, and order.

**Why a separate module, and why the arithmetic is duplicated.** The same two
reasons as `eject_jax`: `state.py` must not import jax at module level, and the
gate is elementwise equality against `SlotState._insert_slab`'s own numpy, so the
two implementations are deliberately independent (umbrella
`ab_arms_sharing_code_are_policy_blind`).

**What one call does, for one destination slab.** The rows arrive as the numpy
path concatenates them -- the slab's keepers, then the immigrants of every
reaching slab in source order -- and the program:

  1. keeps the rows bound for this slab and stable-sorts them by destination
     bucket. Per brick that is exactly the numpy order: keepers before
     immigrants, then `_stable_order` by bucket within the brick;
  2. takes each brick's velocity scale over the union of its keepers and
     immigrants, `max(|w| * s_old) / 32767`, with 1.0 for an empty brick;
  3. re-expresses every row's codes at its brick's scale, `rint(w * s_old /
     s_new)`, and reports the largest magnitude so the caller can refuse an
     int16 escape;
  4. ranks rows within their brick: a row whose rank fits the brick's allocation
     is written at `brick_start + rank`, the rest spill, in order.

**Every float operation matches numpy's by construction.** The scale's max is
exact on any backend; its division by 32767 and the rescale's division are both
by full-shape RUNTIME arrays, because CPU XLA computes a division by a scalar or
broadcast divisor as a reciprocal multiply, one ulp off numpy (umbrella
`xla_scalar_division_is_reciprocal`). The rescale is one multiply then `rint`, so
no fused multiply-add can form.

**What stays on the host** (`SlotState._insert_slab_jax`): the consumption
census, each row's old-scale gather, the writes into the state, and every arena
claim, replayed brick by brick in the numpy path's order -- claims take the
lowest free slots, so their order IS the arena layout.

**Padding.** As in `eject_jax`: `n + 1` rows padded onto the capacity ladder at
`PAD_RUNGS_PER_OCTAVE`, marked not real, and neither written nor spilled.
"""

from __future__ import annotations

import numpy as np

from .eject_jax import require_x64

PAD_RUNGS_PER_OCTAVE = 12

_CACHE: dict = {}

#: RECEIPT, not telemetry: a run that selects the compiled insert must be able
#: to prove it applied (see `eject_jax.CALLS`).
CALLS = 0


def _padded(n):
    from .forces import capacity_shape

    return int(capacity_shape(max(1, int(n)) + 1, rungs=PAD_RUNGS_PER_OCTAVE))


def narrow_key_ok(p3, nb2, n_pad):
    """May this slab's sort use the uint32 key and int32 index?

    Separated from `_build` because THE BRANCH IT GUARDS CANNOT BE RUN at any
    testable size -- tripping it needs `nb2 * p3 >= 2**32`, whose occupancy array
    alone is 34 GB -- so a test of the wide path would be a test no machine can
    fail. The decision is testable even though the branch is not, which is the
    part worth pinning.

    Production is nowhere near either bound: at 4096^3 `nb2 * p3` is 33,554,432
    (128x under) and `n_pad` is ~7e8 (3x under).
    """
    return int(nb2) * int(p3) < 2**32 and int(n_pad) < 2**31


def _build(p3, nb2, n_pad, has_ids):
    import jax
    import jax.numpy as jnp

    big = int(np.iinfo(np.int64).max)
    # THE SORT KEY IS SLAB-RELATIVE, AND NARROW. A destination inside this slab is
    # bounded by `nb2 * p3` -- 33,554,432 at 4096^3, 128x under the uint32 ceiling
    # -- and a narrow integer key is what puts a sort on a radix path instead of a
    # comparison path. The permutation is IDENTICAL, not merely equivalent:
    # subtracting a constant is order-preserving on the in-slab rows, every other
    # row takes ONE sentinel strictly above all of them, and a stable sort agrees
    # on ties. That is the same argument, and the same checked-range-with-fallback
    # policy, as `state._stable_order`, where the identical change bought migrate's
    # numpy sort 5.3x and the phase 1.67x end to end.
    #
    # Both bounds are STATIC and the fallback is the wide form, never a refusal.
    # What they prevent is measured rather than asserted: at `nb2 * p3 == 2**32`
    # the sentinel itself wraps to 0 and the out-of-slab rows sort FIRST.
    narrow = narrow_key_ok(p3, nb2, n_pad)
    # THE INDEX FAMILY IS int32 UNDER THE SAME GUARD. Every one of these counts
    # rows within one slab and is bounded by `n_pad`, so at 4096^3 they are ~2.7e8
    # and 3x under the int32 ceiling. They are also `n_pad` LONG, which is what
    # makes the width matter: `idx_all` alone is 2.15 GB per slab at int64, and the
    # prefix partition above adds four more arrays of the same length.
    #
    # `pos` is the one quantity that must stay wide, and it does so by promotion
    # rather than by luck: it is `starts[bl] + rank`, `starts` holds GLOBAL slot
    # indices (~7.5e10 at 4096^3, far past int32), and int64 + int32 -> int64 is
    # verified in `test_the_written_positions_stay_int64`. Narrowing `starts`
    # would be the silent truncation this comment exists to prevent.
    idt = jnp.int32 if narrow else jnp.int64

    @jax.jit
    def kernel(dest, off, w, ids, s_old, real, lo_b, starts, div):
        # inside the trace, not a captured constant (see `eject_jax._build`)
        idx_all = jnp.arange(n_pad, dtype=idt)
        brick = dest // p3
        inslab = real & (brick >= lo_b) & (brick < lo_b + nb2)
        if narrow:
            # the `where` runs in int64 and both of its branches already sit in
            # [0, nb2 * p3], so the cast can never see a value that wraps
            key = jnp.where(inslab, dest - lo_b * p3, nb2 * p3).astype(jnp.uint32)
            order = jnp.argsort(key, stable=True, dtype=jnp.int32)
        else:
            order = jnp.argsort(jnp.where(inslab, dest, big), stable=True)
        dest_s, off_s, w_s = dest[order], off[order], w[order]
        s_s, in_s = s_old[order], inslab[order]

        bl = jnp.where(in_s, dest_s // p3 - lo_b, 0)
        cnt = jnp.zeros(nb2, dtype=idt).at[bl].add(in_s.astype(idt))
        rank = idx_all - (jnp.cumsum(cnt) - cnt)[bl]
        fits = rank < (starts[1:] - starts[:-1])[bl]
        write = in_s & fits
        spill = in_s & jnp.logical_not(fits)

        # the scale over the union, as `_insert_slab`: |w| max per row times the
        # row's source scale. Widened to f64 BEFORE abs and max: every int16 is
        # exact in f64, so the value is numpy's, and the int16 form
        # `abs(w).max(1).astype(f64)` came back wrong in 5,293 of 15,898 rows
        # jitted on a GB200 (eager and CPU XLA exact; Vista 995228), shrinking
        # 27 of 64 brick scales until the rescale escaped int16
        m_row = jnp.abs(w_s.astype(jnp.float64)).max(axis=1) * s_s
        vmax = jnp.zeros(nb2, dtype=jnp.float64).at[bl].max(jnp.where(in_s, m_row, 0.0))
        s_b = vmax / div
        s_b = jnp.where(s_b > 0.0, s_b, 1.0)
        ratio = s_s / jnp.where(in_s, s_b[bl], 1.0)
        out = jnp.rint(w_s.astype(jnp.float64) * ratio[:, None])
        abs_max = jnp.max(jnp.where(in_s[:, None], jnp.abs(out), 0.0))
        w_new = out.astype(jnp.int16)

        occ = jnp.zeros(nb2 * p3, dtype=jnp.int64).at[
            jnp.where(write, dest_s - lo_b * p3, 0)].add(write.astype(jnp.int64))
        # Written rows first, then spills, each in brick-then-rank order. This is a
        # THREE-CLASS STABLE PARTITION, and `jnp.argsort` charged a full radix sort
        # for it -- eight passes over a 64-bit key and a 64-bit payload to separate
        # three values. The exclusive-prefix form is the SAME PERMUTATION by
        # construction, not merely an equivalent one: a stable sort lists class 0 in
        # input order, then class 1, then class 2, which is exactly what placing
        # each row at its rank within its own class produces. It is also the form
        # `eject_jax._build` already uses for its keeper/leaver partition.
        # Counts are cast explicitly -- a bool reduction promotes to float32 on
        # this stack, and these become scatter indices.
        w_i = write.astype(idt)
        s_i = spill.astype(idt)
        ex_w = jnp.cumsum(w_i) - w_i
        ex_s = jnp.cumsum(s_i) - s_i
        nw_t = ex_w[-1] + w_i[-1]
        ns_t = ex_s[-1] + s_i[-1]
        pos2 = jnp.where(write, ex_w,
                         jnp.where(spill, nw_t + ex_s,
                                   nw_t + ns_t + (idx_all - ex_w - ex_s)))
        o2 = jnp.zeros(n_pad, dtype=idt).at[pos2].set(idx_all)
        return dict(pos=(starts[bl] + rank)[o2], dest=dest_s[o2], off=off_s[o2],
                    w=w_new[o2], ids=ids[order][o2] if has_ids else None,
                    n_write=jnp.sum(write), n_spill=jnp.sum(spill), occupancy=occ,
                    scales=s_b, abs_max=abs_max)

    return kernel


def insert_rows(dest, off, w, ids, s_old, lo_b, brick_start_slab, p3):
    """Group, scale, rescale and order one destination slab's rows.

    `dest`, `off`, `w`, `ids` (or None) and `s_old` (each row's pre-migration
    source-brick scale) are the keepers then the immigrants, in `_insert_slab`'s
    concatenation order. `brick_start_slab` is `brick_start[lo_b:hi_b + 1]`.

    Returns numpy: `pos`, `dest`, `off`, `w`, `ids` with the `n_write` written
    rows first and the `n_spill` spilled rows after them; the slab's `occupancy`
    (int64, bricks x p3), each brick's new `scales`, and `abs_max`, the largest
    rescaled code magnitude (the caller refuses above int16).
    """
    import jax.numpy as jnp

    global CALLS
    CALLS += 1
    require_x64()
    n = int(len(dest))
    n_pad = _padded(n)
    nb2 = int(len(brick_start_slab)) - 1
    has_ids = ids is not None
    key = (int(p3), nb2, n_pad, has_ids)
    fn = _CACHE.get(key)
    if fn is None:
        fn = _CACHE[key] = _build(int(p3), nb2, n_pad, has_ids)
    pad = n_pad - n

    def _pad(a, fill):
        a = np.asarray(a)
        if pad == 0:
            return a
        return np.concatenate([a, np.full((pad,) + a.shape[1:], fill, dtype=a.dtype)])

    real = np.zeros(n_pad, dtype=bool)
    real[:n] = True
    out = fn(jnp.asarray(_pad(np.asarray(dest, dtype=np.int64), 0)),
             jnp.asarray(_pad(off, 0)), jnp.asarray(_pad(w, 0)),
             jnp.asarray(_pad(ids, 0)) if has_ids else None,
             jnp.asarray(_pad(np.asarray(s_old, dtype=np.float64), 1.0)),
             jnp.asarray(real), jnp.asarray(int(lo_b), dtype=jnp.int64),
             jnp.asarray(np.asarray(brick_start_slab, dtype=np.int64)),
             jnp.asarray(np.full(nb2, 32767.0, dtype=np.float64)))
    host = {k: (None if v is None else np.asarray(v)) for k, v in out.items()}
    for k in ("n_write", "n_spill"):
        host[k] = int(host[k])
    host["abs_max"] = float(host["abs_max"])
    return host
