"""Compiled twin of `SlotState._eject_slab`'s row work: drift, re-home, and partition.

Separate from `state.py` because state must not import jax at module level (`executor.TilePool`
sets worker CPU affinity before jax loads); everything here is imported lazily. The arithmetic
is duplicated rather than shared on purpose: the gate is elementwise equality against the numpy
path, which only means something if the two implementations are independent.

The partition is global and stable: every keeper in brick-major order, then every leaver in
brick-major order, which is the order `_eject_slab`'s concatenation produces, so both results are
contiguous slices of one buffer. Stability is required: `_insert_slab` assigns slots by walking
this order, so any reordering changes the physical layout and breaks bitwise equality.

Padding: `n + 1` rows are padded onto `forces.capacity_shape`'s ladder at `PAD_RUNGS_PER_OCTAVE`
so slabs with nearby counts share one compiled program; the `+ 1` guarantees at least one padded
row. Padded rows are neither keepers nor leavers and land at positions the host never reads.

`device/migrate.py` reuses `_build`, `_padded` and `_CACHE` with the same key: a change to the
signature or key must land in both.
"""

from __future__ import annotations

import numpy as np

from .codec import LEVELS_PER_BUCKET

PAD_RUNGS_PER_OCTAVE = 12

_CACHE: dict = {}

#: Call count: lets a run that selects `eject_kernel="jax"` prove the compiled path ran.
CALLS = 0


def _padded(n):
    from .forces import capacity_shape

    return int(capacity_shape(max(1, int(n)) + 1, rungs=PAD_RUNGS_PER_OCTAVE))


def require_x64():
    """Raise unless jax_enable_x64 is on.

    The library never toggles x64 (callers opt in), but with it off `jnp.int64`
    silently becomes int32 and the lattice index overflows, so compiled paths
    refuse rather than degrade.
    """
    import jax

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "the compiled eject path requires jax_enable_x64. With x64 off the "
            "int64 lattice index narrows to int32 and destinations are silently "
            "wrong. Enable it in the caller (this library never toggles it), or "
            "use kernel='numpy'."
        )


def _build(t9, nb, n_pad, has_ids):
    import jax
    import jax.numpy as jnp

    q = float(t9.quantum)
    n_levels = int(t9.n_levels)
    per = int(t9.n_buckets_side) // int(nb)
    p3 = per**3
    nbi = int(nb)

    @jax.jit
    def kernel(off, bijk, w, ids, scale, c_drift, brick_id, real):
        # built inside the trace; outside it would be a constant baked into the executable
        idx_all = jnp.arange(n_pad, dtype=jnp.int64)
        # the drift, in the integer domain; expression shape identical to
        # state._eject_slab's so equality holds by construction
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

        # Padded rows are excluded by two guards, `real` and the -1 brick sentinel. Each
        # alone suffices, so no test can defend either singly: do not remove one because
        # the suite stays green.
        stay = jnp.logical_and((dest // p3) == brick_id, real)
        leave = jnp.logical_and(jnp.logical_not(stay), real)

        # global stable partition: keepers, then leavers
        ex_keep = jnp.cumsum(stay) - stay
        ex_leave = jnp.cumsum(leave) - leave
        n_keep = ex_keep[-1] + stay[-1]
        pos_real = jnp.where(stay, ex_keep, n_keep + ex_leave)
        # padded rows keep their own index (>= the real row count): no collision
        pos = jnp.where(real, pos_real, idx_all)

        order = jnp.zeros(n_pad, dtype=jnp.int64).at[pos].set(idx_all)
        return (dest[order], off_new[order], w[order],
                ids[order] if has_ids else None, brick_id[order], n_keep)

    return kernel


def eject_rows(t9, nb, off, bijk, w, ids, scale, c_drift, brick_id):
    """Drift + re-home + global stable partition for one slab's rows.

    Rows must arrive in brick-major order (which is how `_eject_slab` walks
    them). Returns `(dest, off_new, w_out, ids_out, src_out, n_keep)` where
    `[:n_keep]` are the keepers and `[n_keep:]` the leavers, both in the numpy
    path's order, so a caller slices rather than gathers.
    """
    import jax.numpy as jnp

    global CALLS
    CALLS += 1
    require_x64()
    n = int(len(off))
    n_pad = _padded(n)
    has_ids = ids is not None
    key = (int(t9.n_buckets_side), float(t9.quantum), int(nb), n_pad, has_ids)
    fn = _CACHE.get(key)
    if fn is None:
        fn = _CACHE[key] = _build(t9, nb, n_pad, has_ids)

    pad = n_pad - n

    def _pad(a, fill):
        if pad == 0:
            return a
        tail = np.full((pad,) + a.shape[1:], fill, dtype=a.dtype)
        return np.concatenate([a, tail])

    real = np.zeros(n_pad, dtype=bool)
    real[:n] = True

    dest, off_new, w_out, ids_out, src_out, n_keep = fn(
        jnp.asarray(_pad(off, 0)),
        jnp.asarray(_pad(bijk, 0)),
        jnp.asarray(_pad(w, 0)),
        jnp.asarray(_pad(ids, 0)) if has_ids else None,
        jnp.asarray(_pad(scale, 1.0)),
        float(c_drift),
        # -1 matches no brick
        jnp.asarray(_pad(brick_id, -1)),
        jnp.asarray(real),
    )
    return (
        np.asarray(dest)[:n],
        np.asarray(off_new)[:n],
        np.asarray(w_out)[:n],
        None if ids_out is None else np.asarray(ids_out)[:n],
        np.asarray(src_out)[:n],
        int(n_keep),
    )
