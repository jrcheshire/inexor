"""Compiled twin of `_eject_slab`'s row work: drift, re-home, and partition.

**Why this is a separate module.** `state.py` must not import jax at module
level. `executor.TilePool` sets worker CPU affinity BEFORE jax exists in the
process (M-v2-6 Stage B), and `state` is imported by every worker, so a
module-level `import jax` there would defeat that ordering. Everything here is
imported lazily, on the first call that asks for the compiled path.

**Why it duplicates the arithmetic rather than sharing it.** The umbrella record
`ab_arms_sharing_code_are_policy_blind` is the reason: an A/B whose arms run the
same lines cannot see a change to those lines. The gate for this module is
elementwise equality against `SlotState._eject_slab`'s own numpy, so the two
implementations are deliberately independent and the duplication IS the control.

**What is compiled, and what is not.** Section 5n decomposed `_eject_slab` on two
architectures: the drift/re-home arithmetic is 27.6-35.1% of the call and the
keep/leave partition is 30.2-30.5%, both 20-80x above the machine's measured
single-core traffic floor. Those two are here. Slot resolution, the slot gather
and the arena splice (29-31%) stay in `state.py`: they are pointer-chasing and
sit at 8-30x the floor, where much less is available.

**The partition is GLOBAL and STABLE, and the "global" is what makes it cheap.**
`_eject_slab` builds per-brick keeper and leaver lists and hands them to `_cat`,
which concatenates all bricks' keepers into one array and all bricks' leavers
into another. So the target order is not per-brick at all: it is every keeper in
brick-major order, then every leaver in brick-major order. Producing exactly that
makes both results contiguous SLICES of one buffer, so the host does no gather
and no concatenate. Stability is a correctness requirement rather than a
preference: `_insert_slab` assigns slots by walking this order, so a reordering
changes the state's physical layout and breaks the bitwise chain even though no
particle is lost.

**Padding.** Row counts per slab drift as occupancy evolves, and a fresh shape is
a fresh XLA compilation (umbrella `jax_shape_recompile_cache`). Rows are padded
to a multiple of `PAD_MULTIPLE`; padded rows are marked not-real, are counted as
neither keepers nor leavers, and are parked at output positions the host slice
never reads.
"""

from __future__ import annotations

import numpy as np

from .codec import LEVELS_PER_BUCKET

PAD_MULTIPLE = 4096

_CACHE: dict = {}

#: RECEIPT, not telemetry. A knob that selects this path must be able to prove it
#: applied -- a run whose `eject_kernel="jax"` silently fell back would read as a
#: null result for the compiled path rather than as a broken instrument, and this
#: milestone has already shipped one card whose arm never ran. Probes read this
#: before and after a timed region and put the delta on the card.
CALLS = 0


def _padded(n):
    return int(PAD_MULTIPLE * int(np.ceil(max(1, n) / PAD_MULTIPLE)))


def require_x64():
    """x64 or nothing, and it must be LOUD.

    `jax_enable_x64` is the caller's choice by this repo's convention and library
    code never toggles it. But with it off, `jnp.int64` silently becomes int32,
    the global lattice index overflows, and destinations are wrong with no
    exception and a bitwise gate that fails without saying why. So the compiled
    path refuses to exist rather than degrade.
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
    idx_all = jnp.arange(n_pad, dtype=jnp.int64)

    @jax.jit
    def kernel(off, bijk, w, ids, scale, c_drift, brick_id, real):
        # --- the drift, in the integer domain (D-007). The expression SHAPE is
        # held identical to state._eject_slab's so equality is by construction
        # rather than by an exponent coincidence ---
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

        # A padded row is neither a keeper nor a leaver, so it cannot shift any
        # real row's rank in either run.
        #
        # TWO GUARDS, REDUNDANT SINGLY AND LOAD-BEARING TOGETHER, and this is
        # measured rather than asserted: mutation-testing removed `real` here
        # (7/7 still pass, the -1 brick sentinel catches it) and separately
        # changed the sentinel to 0 (7/7 still pass, `real` catches it), while
        # removing BOTH fails 5 of 7. So no test can defend either one alone.
        # Do not "simplify" one away on the strength of a green suite.
        stay = jnp.logical_and((dest // p3) == brick_id, real)
        leave = jnp.logical_and(jnp.logical_not(stay), real)

        # --- the global stable partition: keepers, then leavers ---
        ex_keep = jnp.cumsum(stay) - stay
        ex_leave = jnp.cumsum(leave) - leave
        n_keep = ex_keep[-1] + stay[-1]
        pos_real = jnp.where(stay, ex_keep, n_keep + ex_leave)
        # padded rows keep their own index, which is >= the real row count by
        # construction, so they can never collide with a real row's position
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
        # -1 can match no brick, so a padded row cannot be classified a keeper
        # even before `real` masks it
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
