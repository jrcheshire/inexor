"""D2b: the kick and the per-brick quantize, on the device.

WHAT THIS REPLACES. The tail of `engine.tile_task`, from `g_tot = g_short[owned]
+ g_long` to the end: a numpy f64 kick, then a Python loop over per-brick runs
computing a velocity scale and rounding to int16. Both are per-particle host
work, and with D2a's decode gone from the head of that function the kick is the
next thing standing between the tile pass and a device-resident pipeline.

THE PER-BRICK SCALE IS THE WHOLE PROBLEM. T9 stores one f64 scale per brick and
int16 codes against it, so the quantize needs `max|v|` over each brick's owned
rows before it can round anything. The host gets that from a run scan, which
works because `decode_bricks` leaves a brick's rows contiguous. On the device it
is a segmented reduction over the row's brick index -- `jax.ops.segment_max`,
one fixed shape, no dependence on the runs being contiguous at all.

That independence is worth stating: the device form would still be correct if
the row order changed, where the host form would silently produce one scale per
RUN and let a later run overwrite an earlier one's. `tile_task` asserts the
contiguity for exactly that reason. This module does not need the assumption,
and the test still pins the contiguity because the SLOT writes downstream
depend on it.

BITWISE, and it can be. `max` over floats is exact and associative, so a
segmented max cannot disagree with a run scan whatever order it reduces in. The
division and the round-half-even that follow are IEEE-exact. So this phase, like
the decode, is gated on equality rather than a tolerance.

MASKING. Rows that are padding or not owned are set to zero magnitude before the
reduction, never dropped: a compaction would key a new XLA shape per tile, which
is the fault `forces.py` records at 2,107 compilations and 74% of a step. A
brick with no owned rows therefore reduces to 0 and takes scale 1.0, matching
`encode_velocities`' all-zero-field branch -- but the caller is told the owned
count per brick so it can drop those bricks rather than writing a scale for a
brick this tile does not own.
"""

from __future__ import annotations

import numpy as np

INT16_MAX = 32767


def kick_and_quantize(v, g_short, g_long_rows, owned, brick_index, n_bricks,
                      alpha_k, bcoef, fdtype=None):
    """(w_codes, scales, owned_counts, v_new) for one tile's rows.

    Every array is at the padded row shape `cap`; nothing is compacted.

    `g_long_rows` is the coarse force ALIGNED WITH ROWS, where `tile_task`
    carries it packed to the front of an owned-only buffer. The packing is a
    transport detail of the host path (`gather_coarse_subblock` is fed a packed
    `xo`), not part of the physics, and keeping row alignment here is what lets
    the kick stay one shape.

    Returns scales for every one of the tile's `n_bricks` bricks. A brick with
    `owned_counts == 0` gets scale 1.0 and its rows are all masked; the caller
    drops it rather than writing a scale for a brick it does not own.
    """
    import jax
    import jax.numpy as jnp

    from ..codec import rint_i
    from ..eject_jax import require_x64

    # Same contract as the decode: with x64 off the f64 kick silently becomes
    # f32, and the velocity scale it derives is then wrong in the last bits for
    # every particle in the brick.
    require_x64()
    fdtype = jnp.float64 if fdtype is None else fdtype

    own = jnp.asarray(owned).astype(bool)
    g_tot = jnp.asarray(g_short, dtype=fdtype) + jnp.asarray(g_long_rows, dtype=fdtype)
    v_new = (jnp.asarray(alpha_k, dtype=fdtype) * jnp.asarray(v, dtype=fdtype)
             + jnp.asarray(bcoef, dtype=fdtype) * g_tot)
    v_new = jnp.where(own[:, None], v_new, jnp.zeros_like(v_new))

    # THE SEGMENTED MAX. `max|v|` per brick over its OWNED rows, taken across
    # all three components exactly as the host's `np.max(np.abs(vb))` does.
    # Masked rows contribute 0, which cannot raise a max of magnitudes.
    mag = jnp.max(jnp.abs(v_new), axis=1)
    mag = jnp.where(own, mag, jnp.zeros_like(mag))
    seg = jnp.asarray(brick_index, dtype=jnp.int32)
    vmax = jax.ops.segment_max(mag, seg, num_segments=int(n_bricks),
                               indices_are_sorted=False)
    vmax = jnp.maximum(vmax, 0.0)  # segment_max seeds empty segments at -inf

    scales = vmax / fdtype(INT16_MAX)
    # an all-zero brick has no scale; encode to zeros and keep the decode exact
    scales = jnp.where(scales > 0.0, scales, jnp.ones_like(scales))

    owned_counts = jax.ops.segment_sum(own.astype(jnp.int32), seg,
                                       num_segments=int(n_bricks))

    # `rint_i` (round-half-even through int32) is `codec.encode_velocities`'
    # value path and the twin of the host's `np.rint(...).astype(np.int16)`.
    # Routing through int32 is not cosmetic: float->int16 overflow is
    # backend-defined where int32->int16 narrowing is guaranteed modular.
    w32 = rint_i(v_new / scales[seg][:, None])
    w_codes = jnp.where(own[:, None], w32, 0).astype(jnp.int16)
    return dict(w_codes=w_codes, scales=scales, owned_counts=owned_counts,
                v_new=v_new, w32=w32)


def assert_int16_range_device(w32, what="velocity"):
    """D-007's refusal, moved off the step path but not dropped.

    `codec.assert_int16_range` is host-side on purpose -- calling it under jit
    forces a device sync every step. So the device kick does not call it, and
    this exists for the gates and for a debug arm to call explicitly. The
    refusal is not decoration: T9 does not clamp, so a code outside int16 would
    wrap and silently misrepresent the fastest particles in the brick, which are
    the ones that matter.

    By construction it cannot fire -- the scale is `max|v| / 32767` over exactly
    the rows being encoded, so the extremes land on +-32767 -- and checking it is
    how that stays proved rather than assumed.
    """
    from ..codec import assert_int16_range

    return assert_int16_range(np.asarray(w32), what)
