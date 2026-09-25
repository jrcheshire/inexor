"""FROZEN copy of `inexor.insert_jax` as it stood at cba30c3~1, for the A/B only.

Vendored rather than reconstructed so the speed and memory comparison runs both
arms in ONE job on ONE node. The alternative -- reading the new kernel against a
number recorded by another job on another node -- is the comparison shape that
cost record 5h a chase and that the allocator readout had to caveat.

This is an ORACLE, not a path anything runs: `inexor.insert_jax` is the live
module and `SlotState._insert_slab`'s numpy is the reference both are gated
against. Nothing here may be "kept up to date" -- the whole point is that it does
not move. Same precedent as `state._repack_reference`.
"""

from __future__ import annotations

import numpy as np

from inexor.eject_jax import require_x64

PAD_RUNGS_PER_OCTAVE = 12

#: Its OWN cache, deliberately not `insert_jax._CACHE`: the two modules key on the
#: same `(p3, nb2, n_pad, has_ids)` tuple, so a shared dict would hand one arm the
#: other arm's kernel and the A/B would compare a thing with itself.
_CACHE: dict = {}

CALLS = 0


def _padded(n):
    from inexor.forces import capacity_shape

    return int(capacity_shape(max(1, int(n)) + 1, rungs=PAD_RUNGS_PER_OCTAVE))


def _build(p3, nb2, n_pad, has_ids):
    import jax
    import jax.numpy as jnp

    big = int(np.iinfo(np.int64).max)

    @jax.jit
    def kernel(dest, off, w, ids, s_old, real, lo_b, starts, div):
        # inside the trace, not a captured constant (see `eject_jax._build`)
        idx_all = jnp.arange(n_pad, dtype=jnp.int64)
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
