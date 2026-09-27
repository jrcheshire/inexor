"""The kick and the per-brick velocity quantize, on the device (the tail of `engine.tile_task`).

T9 stores one f64 velocity scale per brick, so quantizing needs `max|v|` over each brick's owned
rows. On the device that is a segmented max over the row's brick index (`jax.ops.segment_max`),
which, unlike the host's run scan, does not depend on a brick's rows being contiguous. The result
is bitwise the host's: float max is exact and associative, and the division and round-half-even
that follow are IEEE-exact.

Padding and unowned rows are masked to zero magnitude, never compacted (a compaction would
compile a new shape per tile). A brick with no owned rows takes scale 1.0; callers use the
returned owned counts to drop such bricks.
"""

from __future__ import annotations

import numpy as np

INT16_MAX = 32767


def kick_and_quantize(v, g_short, g_long_rows, owned, brick_index, n_bricks,
                      alpha_k, bcoef, fdtype=None):
    """(w_codes, scales, owned_counts, v_new) for one tile's rows.

    Every array is at the padded row shape `cap`; nothing is compacted.

    `g_long_rows` is the coarse force aligned with rows (the host path packs it
    to the front of an owned-only buffer instead).

    Returns scales for every one of the tile's `n_bricks` bricks. A brick with
    `owned_counts == 0` gets scale 1.0 and its rows are all masked; the caller
    drops it rather than writing a scale for a brick it does not own.
    """
    import jax.numpy as jnp

    from ..eject_jax import require_x64

    require_x64()
    fdtype = jnp.float64 if fdtype is None else fdtype

    g_tot = jnp.asarray(g_short, dtype=fdtype) + jnp.asarray(g_long_rows, dtype=fdtype)
    v_new = (jnp.asarray(alpha_k, dtype=fdtype) * jnp.asarray(v, dtype=fdtype)
             + jnp.asarray(bcoef, dtype=fdtype) * g_tot)
    return quantize_per_brick(v_new, owned, brick_index, n_bricks)


def quantize_per_brick(v_new, owned, brick_index, n_bricks, scale_div=None):
    """(w_codes, scales, owned_counts, v_new, w32) from already-kicked velocities.

    For callers that form `v_new` themselves (`device.tile`, which reproduces the
    host kick's dtype promotion). Same shapes and masking as `kick_and_quantize`.

    `scale_div` is an (n_bricks,) array of 32767s in `v_new`'s dtype. Eager callers
    may omit it; a jitted caller must pass it (built inside the program it folds
    back to a scalar divisor).
    """
    import jax
    import jax.numpy as jnp

    from ..codec import rint_i
    from ..eject_jax import require_x64

    require_x64()
    own = jnp.asarray(owned).astype(bool)
    v_new = jnp.asarray(v_new)
    v_new = jnp.where(own[:, None], v_new, jnp.zeros_like(v_new))

    # max|v| per brick over owned rows and all three components; masked rows contribute 0
    mag = jnp.max(jnp.abs(v_new), axis=1)
    mag = jnp.where(own, mag, jnp.zeros_like(mag))
    seg = jnp.asarray(brick_index, dtype=jnp.int32)
    vmax = jax.ops.segment_max(mag, seg, num_segments=int(n_bricks),
                               indices_are_sorted=False)
    vmax = jnp.maximum(vmax, 0.0)  # segment_max seeds empty segments at -inf

    # Full-shape runtime divisor: CPU XLA lowers x / scalar (or a broadcast, or a
    # full_like built under jit) to x * (1/scalar), one ulp off numpy's division.
    div = jnp.full_like(vmax, INT16_MAX) if scale_div is None else jnp.asarray(scale_div)
    scales = vmax / div
    # an all-zero brick has no scale; encode to zeros and keep the decode exact
    scales = jnp.where(scales > 0.0, scales, jnp.ones_like(scales))

    owned_counts = jax.ops.segment_sum(own.astype(jnp.int32), seg,
                                       num_segments=int(n_bricks))

    # Round through int32 (`rint_i`). Divide column by column by the per-row scale, a
    # same-shape (exact) division; dividing by `scales[seg][:, None]` broadcasts and is
    # one ulp off for the same reason as above.
    s_row = scales[seg]
    w32 = rint_i(jnp.stack([v_new[:, k] / s_row for k in range(3)], axis=1))
    w_codes = jnp.where(own[:, None], w32, 0).astype(jnp.int16)
    return dict(w_codes=w_codes, scales=scales, owned_counts=owned_counts,
                v_new=v_new, w32=w32)


def assert_int16_range_device(w32, what="velocity"):
    """Host-side int16 refusal for the device kick's `w32`, for tests and debug runs.

    Not called on the step path (it would force a device sync). By construction it
    cannot fire, since the scale is max|v| / 32767 over exactly the encoded rows.
    """
    from ..codec import assert_int16_range

    return assert_int16_range(np.asarray(w32), what)
