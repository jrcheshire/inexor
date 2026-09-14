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

**Padding.** As in `eject_jax`: rows are padded to a multiple of `PAD_MULTIPLE`,
marked not real, and can be neither written nor spilled.
"""

from __future__ import annotations

import numpy as np

from .eject_jax import require_x64

PAD_MULTIPLE = 4096

_CACHE: dict = {}

#: RECEIPT, not telemetry: a run that selects the compiled insert must be able
#: to prove it applied (see `eject_jax.CALLS`).
CALLS = 0


def _padded(n):
    return int(PAD_MULTIPLE * int(np.ceil(max(1, n) / PAD_MULTIPLE)))


def _build(p3, nb2, n_pad, has_ids):
    import jax
    import jax.numpy as jnp

    big = int(np.iinfo(np.int64).max)
    idx_all = jnp.arange(n_pad, dtype=jnp.int64)

    @jax.jit
    def kernel(dest, off, w, ids, s_old, real, lo_b, starts, div):
        brick = dest // p3
        inslab = real & (brick >= lo_b) & (brick < lo_b + nb2)
        order = jnp.argsort(jnp.where(inslab, dest, big), stable=True)
        dest_s, off_s, w_s = dest[order], off[order], w[order]
        s_s, in_s = s_old[order], inslab[order]

        bl = jnp.where(in_s, dest_s // p3 - lo_b, 0)
        cnt = jnp.zeros(nb2, dtype=jnp.int64).at[bl].add(in_s.astype(jnp.int64))
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
        # written rows first, then spills, each in brick-then-rank order
        o2 = jnp.argsort(jnp.where(write, 0, jnp.where(spill, 1, 2)), stable=True)
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
