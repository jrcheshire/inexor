"""T9 particle codec: bucket-relative uint8 positions, int16 velocities, and int32 rounding.

T9 stores 9 B/p of payload: three uint8 offsets within the particle's bucket (a cube of
`bucket_cells` particle cells) at quantum bucket_size/256, and three int16 velocities against a
shared scale (one per call here; `state.SlotState` keeps one per brick). The bucket is implied by
the particle's slot in the brick-sorted layout (`layout.py`, `state.py`).

Invariants:
- wrap-never-clamp: no saturating op touches integer state; overflow raises, never corrects.
- float->int always routes through int32 (`rint_i`): float->int16 overflow is backend-defined,
  int32->int16/uint16 narrowing is modular.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


def rint_i(z):
    """Round-half-even to int32; the required route for every float->int conversion."""
    return jnp.rint(z).astype(jnp.int32)


LEVELS_PER_BUCKET = 256  # one uint8 per axis
INT16_MAX = 32767
ID_MAX_N_SIDE = 1290  # 1290^3 < 2^31-1 <= 1291^3: the int32 id refusal


@dataclass(frozen=True)
class T9Layout:
    """Geometry of the T9 position lattice.

    box_size     periodic box side, same units as positions (Mpc/h)
    n_part       particles per side (the Lagrangian grid)
    bucket_cells the bucket side in PARTICLE cells (default 2)

    Everything else is derived. n_levels must be a power of two so that
    quantum * n_levels == box_size exactly (division by 2^k only shifts the
    exponent), which keeps the periodic wrap modular rather than saturating.
    """

    box_size: float
    n_part: int
    bucket_cells: int = 2

    def __post_init__(self):
        if self.n_part < 1 or self.bucket_cells < 1:
            raise ValueError(
                f"n_part and bucket_cells must be >= 1, got {self.n_part} / {self.bucket_cells}"
            )
        if self.n_part % self.bucket_cells:
            raise ValueError(
                f"bucket_cells ({self.bucket_cells}) must divide n_part ({self.n_part}) so "
                "buckets tile the box exactly; a partial bucket at the periodic seam has no "
                "consistent origin"
            )
        n_levels = self.n_levels
        if n_levels & (n_levels - 1):
            raise ValueError(
                f"n_levels = {n_levels} is not a power of two (n_part={self.n_part}, "
                f"bucket_cells={self.bucket_cells}). quantum * n_levels would not equal "
                "box_size exactly, so the periodic wrap would saturate instead of wrapping "
                "-- forbidden by D-007 (wrap-never-clamp)."
            )
        if self.box_size <= 0.0:
            raise ValueError(f"box_size must be positive, got {self.box_size}")

    @property
    def n_buckets_side(self):
        return self.n_part // self.bucket_cells

    @property
    def n_buckets(self):
        return self.n_buckets_side**3

    @property
    def n_levels(self):
        """Position quanta across the box side."""
        return self.n_buckets_side * LEVELS_PER_BUCKET

    @property
    def quantum(self):
        """Position resolution. Exact: box_size / 2^k."""
        return self.box_size / self.n_levels

    @property
    def spacing(self):
        """Mean interparticle spacing."""
        return self.box_size / self.n_part

    @property
    def bucket_size(self):
        return self.bucket_cells * self.spacing

    def index_bytes(self, dtype=np.uint32):
        """Bytes of the per-bucket count index."""
        return self.n_buckets * np.dtype(dtype).itemsize


def lattice_index(x, layout):
    """Physical positions -> the global position lattice, int32 in [0, n_levels).

    The wrap is taken in the integer domain, where it is exactly modular.
    """
    i = rint_i(x / layout.quantum)
    return jnp.mod(i, layout.n_levels).astype(jnp.int32)


def encode_positions(x, layout):
    """Physical positions -> (uint8 bucket-relative offsets, int32 bucket indices).

    The offset is unsigned: [0, 256) quanta from the bucket's lower corner.
    """
    i = lattice_index(x, layout)
    b = i // LEVELS_PER_BUCKET
    off = i - b * LEVELS_PER_BUCKET
    return off.astype(jnp.uint8), b


def decode_positions(off, bucket_ijk, layout, fdtype=jnp.float64):
    """(offsets, bucket indices) -> physical positions.

    Reconstructs the global lattice index and multiplies once by the quantum,
    so the result is bitwise equal to `rint(x / q) * q` by construction for any
    box, particle count or dtype (the form `bucket * bucket_size + offset *
    quantum` would rely on exponent coincidences).
    """
    i = bucket_ijk.astype(jnp.int32) * LEVELS_PER_BUCKET + off.astype(jnp.int32)
    return i.astype(fdtype) * layout.quantum


def roundtrip_positions(x, layout):
    """Encode then decode, in the input's dtype."""
    off, b = encode_positions(x, layout)
    return decode_positions(off, b, layout, fdtype=x.dtype)


def encode_velocities(v):
    """D-time velocities -> (int16, scale), symmetric max-range, NO clip.

    scale = max|v| / 32767, so the extremes land exactly on +-32767. The
    asymmetric quantum 2*max|v|/65536 would send +max|v| to +32768, one code
    past int16.
    """
    vmax = jnp.max(jnp.abs(v))
    scale = vmax / INT16_MAX
    # an all-zero field has no scale; encode to zeros and keep the decode exact
    scale = jnp.where(scale > 0.0, scale, jnp.ones_like(scale))
    w32 = rint_i(v / scale)
    return w32.astype(jnp.int16), scale


def assert_int16_range(w32, what="velocity"):
    """Raise if an index lies outside int16 (never wrap or clamp).

    Host-side setup/test guard; under jit it would force a device sync.
    """
    lo, hi = int(np.min(np.asarray(w32))), int(np.max(np.asarray(w32)))
    if lo < -32768 or hi > INT16_MAX:
        raise ValueError(
            f"{what} index range [{lo}, {hi}] escapes int16 [-32768, {INT16_MAX}]. "
            "The T9 codec does not clamp (D-007, wrap-never-clamp): a saturating "
            "encode would silently misrepresent the fastest particles, which are "
            "the ones that matter. Re-derive the scale."
        )
    return lo, hi


def decode_velocities(w, scale, fdtype=jnp.float64):
    """int16 velocities -> physical, via the scale returned by encode_velocities."""
    return w.astype(fdtype) * jnp.asarray(scale, dtype=fdtype)


def refuse_ids_above_int32(n_side):
    """Refuse opt-in int32 particle ids where n_side^3 overflows int32 (n_side > 1290)."""
    if n_side > ID_MAX_N_SIDE:
        raise ValueError(
            f"particle ids are int32 and n_side = {n_side} needs {n_side**3} of them, "
            f"past 2^31-1; the tier is refused above n_side {ID_MAX_N_SIDE}. Ids are "
            "opt-in (+4 B/p) and production configs carry none."
        )


@dataclass(frozen=True)
class T9State:
    """Struct-of-arrays T9 state: 9 B/p payload, plus an opt-in id column.

    Ids are a separate column, so physics kernels never see them.
    """

    pos: jnp.ndarray  # uint8 (n, 3), bucket-relative offsets
    vel: jnp.ndarray  # int16 (n, 3)
    vel_scale: float  # the 4-byte side constant
    ids: jnp.ndarray = None  # int32 (n,) or None

    @property
    def n_particles(self):
        return self.pos.shape[0]

    @property
    def payload_bytes_per_particle(self):
        """9.00, or 13.00 with ids. Excludes the bucket index, brick CSR and
        migration slack, which layout.py accounts for."""
        return 3 * self.pos.dtype.itemsize + 3 * self.vel.dtype.itemsize + (
            self.ids.dtype.itemsize if self.ids is not None else 0
        )


def encode_state(x, v, layout, ids=None):
    """Physical (x, v) -> (T9State, bucket indices). The bucket indices are returned, not
    stored: layout.py consumes them to build the sorted order and the count index."""
    off, b = encode_positions(x, layout)
    w, scale = encode_velocities(v)
    if ids is not None:
        refuse_ids_above_int32(layout.n_part)
        ids = jnp.asarray(ids, dtype=jnp.int32)
    return T9State(pos=off, vel=w, vel_scale=scale, ids=ids), b


def decode_state(state, bucket_ijk, layout, fdtype=jnp.float64):
    """(T9State, bucket indices) -> physical (x, v)."""
    x = decode_positions(state.pos, bucket_ijk, layout, fdtype=fdtype)
    v = decode_velocities(state.vel, state.vel_scale, fdtype=fdtype)
    return x, v
