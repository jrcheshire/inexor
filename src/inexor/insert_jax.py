"""Compiled twin of `SlotState._insert_slab`'s row work: group, scale, rescale, and order.

Separate from `state.py` and deliberately independent of its numpy arithmetic for the same
reasons as `eject_jax`. One call handles one destination slab; rows arrive as the numpy path
concatenates them (the slab's keepers, then each reaching slab's immigrants in source order):

  1. keep rows bound for this slab and stable-sort them by destination bucket (per brick:
     keepers before immigrants, then `_stable_order` by bucket);
  2. take each brick's velocity scale over its keepers and immigrants,
     `max(|w| * s_old) / 32767`, 1.0 for an empty brick;
  3. re-express codes at the brick's scale, `rint(w * s_old / s_new)`, reporting the largest
     magnitude so the caller can refuse an int16 escape;
  4. rank rows within their brick: rank < allocation is written at `brick_start + rank`, the
     rest spill, in order.

Every float op matches numpy's: both divisions are by full-shape runtime arrays because CPU XLA
turns division by a scalar/broadcast into a reciprocal multiply (one ulp off), and the rescale
is one multiply then `rint`, so no FMA can form. The host (`SlotState._insert_slab_jax`) keeps
the old-scale gather, the state writes and the arena claims, whose order is the arena layout.
Padding is as in `eject_jax`.

`device/migrate.py` reuses `_build` and `_CACHE` with the same key: a change to the signature or
key must land in both.
"""

from __future__ import annotations

import numpy as np

from .eject_jax import require_x64

PAD_RUNGS_PER_OCTAVE = 12

_CACHE: dict = {}

#: Call count (see `eject_jax.CALLS`).
CALLS = 0


def _padded(n):
    from .forces import capacity_shape

    return int(capacity_shape(max(1, int(n)) + 1, rungs=PAD_RUNGS_PER_OCTAVE))


def narrow_key_ok(p3, nb2, n_pad):
    """May this slab's sort use the uint32 key and int32 index?

    Separate from `_build` so the decision is testable: the wide branch it guards
    needs `nb2 * p3 >= 2**32`, far beyond any testable (or production) size.
    """
    return int(nb2) * int(p3) < 2**32 and int(n_pad) < 2**31


def _build(p3, nb2, n_pad, has_ids):
    import jax
    import jax.numpy as jnp

    big = int(np.iinfo(np.int64).max)
    # Slab-relative uint32 sort key (a radix sort path). The permutation is identical to
    # the wide form: subtracting a constant preserves order on in-slab rows, all other rows
    # take one sentinel above them, and the sort is stable. The bounds are static and fall
    # back to the wide form; at `nb2 * p3 == 2**32` the sentinel would wrap to 0.
    narrow = narrow_key_ok(p3, nb2, n_pad)
    # Row indices within the slab are int32 under the same guard (bounded by n_pad).
    # `pos = starts[bl] + rank` stays int64 by promotion because `starts` holds global
    # slot indices: never narrow `starts`.
    idt = jnp.int32 if narrow else jnp.int64

    @jax.jit
    def kernel(dest, off, w, ids, s_old, real, lo_b, starts, div):
        # inside the trace, not a captured constant
        idx_all = jnp.arange(n_pad, dtype=idt)
        brick = dest // p3
        inslab = real & (brick >= lo_b) & (brick < lo_b + nb2)
        if narrow:
            # both `where` branches lie in [0, nb2 * p3], so the cast cannot wrap
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

        # per-row max |w| times source scale. Widen to f64 before abs/max: the jitted
        # int16 form `abs(w).max(1)` is miscompiled on GB200 GPUs
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
        # Written rows, then spills, then the rest, each in brick-then-rank order: a
        # three-class stable partition by exclusive prefix sums (the same permutation a
        # stable argsort gives, far cheaper). Counts are cast explicitly because they
        # become scatter indices.
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
