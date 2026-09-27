"""Integer lattice primitives and the straight-through autodiff boundary.

**What used to be here and is gone (2026-08-08, v1 retirement).** This module
was v1's "NOVEL CORE 1": a global uint16 position lattice spanning the box
(`encode_x`/`dequant_x`, D-004), an int16 velocity in the w-frame, and the
w-frame scale ladder (`Ladder`/`build_ladder`/`s_w0_policy`) that made
BullFrog's contracting affine kick integer-bijective so the trajectory was
bit-exactly reversible. That machinery served the v1 thesis, which measured
false on 2026-07-14 (`docs/retrospective.md`).

v2's state tier is different in kind: T9 (D-v2-14) stores positions as int8
**relative to a 1.0 Mpc/h bucket** rather than globally, at quantum
`fine_cell/64`, on a brick-sorted layout -- there is no ladder because there is
no reversibility requirement, and the velocity is a plain max-range int16. The
T9 codec lands here at M-v2-1.

What survives, and why:
- `rint_i` / `iadd` / `isub` -- the float->int and modular-arithmetic
  primitives. `painting.py` imports `rint_i`.
- `ste_round` / `ste_wrap_u` / `ste_wrap_s` -- the straight-through estimator
  boundary. Nothing calls these today. They are kept deliberately: D-v2-3 defers
  differentiability without precluding it, and this is the tested boundary a
  quantized gradient walks back through when seed VD opens. Twelve lines is a
  cheap door to leave open.

Invariants enforced here:
- wrap-never-clamp (D-007): no saturating op touches integer state; overflow
  is monitored loudly, never corrected.
- float->int ALWAYS routes through int32 (`rint_i`): float->int16 overflow is
  backend-defined, int32->int16/uint16 narrowing is guaranteed modular.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

# ============================================================================
# Integer lattice primitives
# ============================================================================


def rint_i(z):
    """Round-half-even to int32. Route float->int through int32 ALWAYS:
    float->int16 overflow is backend-defined; int32->int16/uint16 narrowing is
    guaranteed modular (architecture.md Sec. 4)."""
    return jnp.rint(z).astype(jnp.int32)


# ============================================================================
# Straight-through estimators (the autodiff boundary, architecture.md Sec. 8)
# ============================================================================


# ============================================================================
# T9: the ratified v2 state tier (D-v2-14)
# ============================================================================
#
# 9 bytes per particle of payload: three uint8 positions relative to the
# particle's BUCKET, and three int16 velocities against one global scale.
#
# The bucket is what makes the int8 affordable. D-v2-8 ratified T9 on the G2c
# `t9` arm, which quantizes at fine_cell/256 -- and an int8 carries that only if
# its bucket is ONE FINE CELL. At C-gh the fine mesh is 4096^3 = 6.9e10 cells
# against 8.6e9 particles, so a per-fine-cell index costs ~69 GB against the
# 77 GB of state it indexes. CUBE's sorted-by-cell layout works because CUBE
# runs one particle per cell; our fine mesh is 2x the particle grid per side, so
# it does not transfer. D-v2-14 therefore coarsens the bucket to 1.0 Mpc/h (two
# particle cells, `bucket_cells = 2`), quantum fine_cell/64, index 4.29 GB --
# measured at 15x margin on D-v2-9's 3e-2 bar. (The index was 2.15 GB at
# ratification, when it was uint16; M-v2-1 widened it to uint32 rather than
# establish that a bucket never holds 65535 particles -- see `layout.py`.)
#
# `bucket_cells` is chosen ON MARGIN, not on a measured ordering: the ladder is
# non-monotonic at these levels (c=1 beats the finer control on dP/P, c=4 beats
# c=2 on dP2/P0), so what job 896160 established is that all four arms PASS.
#
# There is no per-particle bucket id in the payload. The bucket is implied by
# WHERE the particle sits in a brick-sorted array plus a per-bucket count index
# -- that index is the 0.50 B/p line of the all-in figure (0.25 as ratified), and it is
# `layout.py`'s job. This module owns the transform only.

LEVELS_PER_BUCKET = 256  # one uint8 per axis; the definition of the tier
INT16_MAX = 32767
ID_MAX_N_SIDE = 1290  # 1290^3 < 2^31-1 <= 1291^3: the int32 id refusal (D-v2-14 cl.5)


@dataclass(frozen=True)
class T9Layout:
    """Geometry of the T9 position lattice.

    box_size     periodic box side, same units as positions (Mpc/h)
    n_part       particles per side (the Lagrangian grid)
    bucket_cells the bucket side in PARTICLE cells; D-v2-14 ratifies 2

    Everything else is derived. The refusals in __post_init__ are not
    defensive noise -- they are what makes the wrap modular rather than
    saturating (D-007): quantum * n_levels must equal box_size EXACTLY, which
    holds when n_levels is a power of two because dividing a float by 2^k only
    shifts its exponent.
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
        """Bytes the per-bucket count index costs (the 0.50 B/p line at C-gh)."""
        return self.n_buckets * np.dtype(dtype).itemsize


def lattice_index(x, layout):
    """Physical positions -> the global position lattice, int32 in [0, n_levels).

    The wrap is taken in the INTEGER domain, which is where it is exactly
    modular. `rint_i` routes through int32 first (float->int8 overflow is
    backend-defined).
    """
    i = rint_i(x / layout.quantum)
    return jnp.mod(i, layout.n_levels).astype(jnp.int32)


def encode_positions(x, layout):
    """Physical positions -> (uint8 bucket-relative offsets, int32 bucket indices).

    The offset is UNSIGNED: it indexes [0, 256) quanta from the bucket's lower
    corner, so a signed reading would need a bias for nothing. The stored byte
    is the same byte D-v2-14 calls an int8.
    """
    i = lattice_index(x, layout)
    b = i // LEVELS_PER_BUCKET
    off = i - b * LEVELS_PER_BUCKET
    return off.astype(jnp.uint8), b


def decode_positions(off, bucket_ijk, layout, fdtype=jnp.float64):
    """(offsets, bucket indices) -> physical positions.

    Reconstructs the global lattice index and multiplies ONCE by the quantum,
    which is the ratified probe's own expression shape (`rint(x/q) * q`), so
    bitwise equality with it holds by CONSTRUCTION.

    The obvious alternative -- `bucket * bucket_size + offset * quantum` -- was
    measured and is also bitwise identical to the probe at every config tried,
    including C-vol's L = 2580, which is not a power of two (0 of 60000
    components differ). So this is not a correctness fix and the form was not
    chosen on a measured difference: it is chosen so that equality does not rest
    on an exponent coincidence that would have to be re-verified whenever the
    box, the particle count or the dtype changes.
    """
    i = bucket_ijk.astype(jnp.int32) * LEVELS_PER_BUCKET + off.astype(jnp.int32)
    return i.astype(fdtype) * layout.quantum


def roundtrip_positions(x, layout):
    """encode then decode, in the input's dtype. The gate handle."""
    off, b = encode_positions(x, layout)
    return decode_positions(off, b, layout, fdtype=x.dtype)


def encode_velocities(v):
    """D-time velocities -> (int16, scale), symmetric max-range, NO clip.

    scale = max|v| / 32767, so the extremes land exactly on +-32767 and the
    whole range is representable. **This is deliberately NOT the probe's
    quantum.** `v2_g2c_accum_gate._rt_vel_int16_max` uses 2*max|v|/65536, which
    sends +max|v| to index +32768 -- one code past int16. That is invisible in a
    float round-trip and fatal in storage, and whether it bites depends on the
    SIGN of the extremum: if max|v| is attained on a negative component the
    indices land in [-32768, k] and fit, and if it is attained on a positive one
    they do not. Measured both ways (10k particles): index range [-10923, 32768]
    versus [-32768, 10923].

    The cost of fixing it is a quantum coarser by 65536/65534 = 1.0000305, i.e.
    0.003% -- against a tier that passes its bar with 15x margin. The velocity
    codec is identical across every position arm of the ratified gate, which is
    what makes that comparison like-for-like, so this does not touch D-v2-14's
    position decision.
    """
    vmax = jnp.max(jnp.abs(v))
    scale = vmax / INT16_MAX
    # an all-zero field has no scale; encode to zeros and keep the decode exact
    scale = jnp.where(scale > 0.0, scale, jnp.ones_like(scale))
    w32 = rint_i(v / scale)
    return w32.astype(jnp.int16), scale


def assert_int16_range(w32, what="velocity"):
    """D-007 refusal: an index outside int16 must RAISE, never wrap or clamp.

    Host-side, so it is a setup/test guard rather than something on the step
    path -- calling it under jit would force a device sync every step.
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
    """D-v2-14 clause 5: the opt-in id tier is int32 and is REFUSED where the
    particle count overflows it. 1290^3 fits; 1291^3 does not. Production C-gh
    (n_side 2048) carries no ids."""
    if n_side > ID_MAX_N_SIDE:
        raise ValueError(
            f"particle ids are int32 and n_side = {n_side} needs {n_side**3} of them, "
            f"past 2^31-1; the tier is refused above n_side {ID_MAX_N_SIDE}. Ids are "
            "opt-in (+4 B/p) and production configs carry none."
        )


@dataclass(frozen=True)
class T9State:
    """Struct-of-arrays T9 state: 9 B/p payload, plus an opt-in id column.

    Ids are a SEVENTH column rather than a field inside the position record, so
    physics kernels never see them and only layout kernels vary (D-v2-14 cl.5).
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
        """9.00, or 13.00 with ids. NOT the all-in figure: the bucket index,
        the brick CSR and the migration slack live in layout.py and bring the
        ratified total to 10.15 B/p (D-v2-14 cl.2)."""
        return 3 * self.pos.dtype.itemsize + 3 * self.vel.dtype.itemsize + (
            self.ids.dtype.itemsize if self.ids is not None else 0
        )


def encode_state(x, v, layout, ids=None):
    """Physical (x, v) -> (T9State, bucket indices). The bucket indices are
    returned rather than stored: layout.py consumes them to build the sorted
    order and the per-bucket count index, after which they are free."""
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
