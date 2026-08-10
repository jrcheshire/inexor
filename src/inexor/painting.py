"""CIC paint / read (architecture.md Secs. 5-6; jnp ports of mbody patterns).

Two paint implementations, ONE interface (density_contrast):
- paint_int: deterministic integer-accumulation paint (D-006) -- corner
  weights quantized to frac_bits fixed point, scatter-added into an int32
  mesh; integer addition is associative, so the result is bit-identical
  regardless of atomic order. PRIMAL-ONLY (no VJP rule needed). R3-verdicted:
  deterministic AND faster than f32 atomics on CUDA.
- paint_f32: the differentiable twin for the VJP path; NOT deterministic
  under f32 atomics on GPU -- never on the primal force path.

Gradients flow through the fractional CIC weights only; the integer base
cell is stop_gradient-detached (mbody painting.py:33 pattern). Migrated
verbatim from scripts/_m0_common.py (M0-verdicted bits).
"""

import jax
import jax.numpy as jnp
import numpy as np

from .codec import rint_i

_CORNERS = [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]


def field_dtype(fdtype):
    """Normalize and validate a MESH/FIELD dtype (M-v2-4).

    The one check on the jnp side, so a typo or a half-precision experiment
    fails at the call rather than producing a field nothing downstream expects.

    **This is for FIELDS, never for paint ACCUMULATORS.** The accumulators are
    fixed by D-006 and are not a knob: int32 for the deterministic primal
    paints, f64 for the differentiable twins, int64 for the engine's host
    accumulation. Narrowing an accumulator would make the primal an f32
    scatter-add, which is non-associative on CUDA and is the archived M0 R1
    counterexample. What M-v2-4 narrows is where a DECODED field is handed on.
    """
    dt = np.dtype(fdtype)
    if dt not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"field dtype must be float32 or float64, got {dt.name}")
    return dt


def _cic_pieces(positions, n_mesh, box_size):
    """Base cell (stop_gradient, mbody painting.py:33 pattern) + fractional offset."""
    d = box_size / n_mesh
    xp = positions / d
    base_f = jnp.floor(xp)
    frac = xp - base_f  # gradients flow through frac only; the cell map is p.w.-constant
    base = jax.lax.stop_gradient(base_f).astype(jnp.int32)
    return base, frac


def _corner_flat_weight(base, frac, corner, n_mesh):
    dx, dy, dz = corner
    wlo = 1.0 - frac
    wx = frac[:, 0] if dx else wlo[:, 0]
    wy = frac[:, 1] if dy else wlo[:, 1]
    wz = frac[:, 2] if dz else wlo[:, 2]
    ix = (base[:, 0] + dx) % n_mesh
    iy = (base[:, 1] + dy) % n_mesh
    iz = (base[:, 2] + dz) % n_mesh
    flat = (ix * n_mesh + iy) * n_mesh + iz
    return flat, wx * wy * wz


def paint_f32(positions, n_mesh, box_size, fdtype=jnp.float32):
    """Differentiable CIC paint (counts). The VJP-path twin; NOT deterministic
    under f32 atomics on GPU -- primal force paths use paint_int."""
    mesh = jnp.zeros((n_mesh**3,), dtype=fdtype)
    base, frac = _cic_pieces(positions, n_mesh, box_size)
    for corner in _CORNERS:
        flat, w = _corner_flat_weight(base, frac, corner, n_mesh)
        mesh = mesh.at[flat].add(w.astype(fdtype), mode="promise_in_bounds")
    return mesh.reshape(n_mesh, n_mesh, n_mesh)


def paint_int(positions, n_mesh, box_size, frac_bits=12):
    """Deterministic integer-accumulation CIC paint (architecture.md Sec. 5).

    Corner weights quantized to frac_bits fixed point, scatter-added into an
    int32 mesh: integer addition is associative, so the result is bit-identical
    regardless of atomic order. Primal-only (no VJP rule needed -- D-006).
    Returns the RAW int32 mesh; decode counts via counts_from_int.
    """
    scale = np.float32(2.0**frac_bits)  # np scalar: no device array at trace-build time
    mesh = jnp.zeros((n_mesh**3,), dtype=jnp.int32)
    base, frac = _cic_pieces(positions, n_mesh, box_size)
    for corner in _CORNERS:
        flat, w = _corner_flat_weight(base, frac, corner, n_mesh)
        mesh = mesh.at[flat].add(rint_i(w.astype(jnp.float32) * scale), mode="promise_in_bounds")
    return mesh.reshape(n_mesh, n_mesh, n_mesh)


def counts_from_int(mesh_int, frac_bits=12, fdtype=jnp.float32):
    """Decode the raw int32 paint into float counts (exact while sums < 2^24)."""
    return mesh_int.astype(fdtype) * np.float32(2.0**-frac_bits)


# The STRICT CIC stencil bound, in the same sense as TSC_CELL_WEIGHT_BOUND below:
# a cell receives from the 8 cells whose CIC stencils reach it, each with a
# per-axis weight up to 1, so the bound is 8x the occupancy rather than the 1x
# `check_int_paint_headroom` has always assumed. Derived at the M-v2-2 promotion.
CIC_CELL_WEIGHT_BOUND = 8.0


def check_int_paint_headroom(
    n_particles_total, frac_bits, max_cell_particles=1.0e4, bound=1.0
):
    """Setup-time refusal against int32 OVERFLOW in the paint accumulator.

    Architecture Sec. 5 budget: max cell mass ~1e4 particles x 2^frac_bits
    << 2^31. Loud, like the ladder guard. The SOFTER exact-decode bound
    (cell sums < 2^24 for exact int32->f32 conversion) is a measured per-run
    diagnostic (max cell occupancy, R3 pattern), not a setup-time refusal --
    exceeding it degrades count precision, exceeding 2^31 corrupts it.

    `bound` is the stencil's cell-weight bound. **The default is 1.0, which is
    the optimistic factor this guard has always carried implicitly, and it is
    kept as the default deliberately**: raising it to the strict
    `CIC_CELL_WEIGHT_BOUND` would move an existing refusal boundary and start
    rejecting configurations that every ratified v1 measurement ran under. New
    call sites pass the strict bound; `tile_paint_int` does.
    """
    worst_cell_sum = (
        float(bound) * min(float(n_particles_total), max_cell_particles) * 2.0**frac_bits
    )
    if worst_cell_sum >= 2.0**31:
        raise ValueError(
            f"int-paint headroom: worst-case cell sum ~{worst_cell_sum:.2e} >= 2^31 at "
            f"frac_bits={frac_bits} (assumed max cell occupancy "
            f"{max_cell_particles:.1e} particles, stencil bound {bound}); lower "
            "frac_bits -- the int32 accumulator would overflow (D-007-class "
            "corruption, not just imprecision)."
        )


def density_contrast(positions, n_mesh, box_size, n_particles_total, paint="int", frac_bits=12):
    """delta mesh from particle positions -- the ONE interface over both paints.

    paint="int": deterministic primal path (default; D-006).
    paint="f32": differentiable twin (VJP path only).
    """
    if paint == "int":
        check_int_paint_headroom(n_particles_total, frac_bits)
        counts = counts_from_int(paint_int(positions, n_mesh, box_size, frac_bits), frac_bits)
    elif paint == "f32":
        counts = paint_f32(positions, n_mesh, box_size)
    else:
        raise ValueError(f"paint must be 'int' or 'f32', got {paint!r}")
    mean = n_particles_total / n_mesh**3
    return counts / mean - 1.0


# ===========================================================================
# TSC: the coarse arm's assignment (promoted from scripts/v2_g5_core.py)
# ===========================================================================
#
# Triangular Shaped Cloud: a 3-point stencil per axis centred on the NEAREST
# cell, where CIC uses 2 points anchored at floor. Its window is sinc^3 rather
# than sinc^2, so it suppresses the high-k power a coarse mesh would otherwise
# alias down into the band -- which is why the long arm uses it and why
# `--assign-long` defaults to `tsc`.
#
# NOTE the f64 paint below is ORDER-DEPENDENT and therefore violates D-006 on
# the primal path: it accumulates through `.at[].add` on an f64 mesh, and f64
# atomics do not commute. `paint_tsc_int` is the required deliverable
# (D-v2-16 clause 2) and this stays as the differentiable/reference twin, in
# exactly the relationship paint_f32 has to paint_int.

_TSC_OFFSETS = (-1, 0, 1)
_TSC_CORNERS = [(dx, dy, dz) for dx in _TSC_OFFSETS for dy in _TSC_OFFSETS for dz in _TSC_OFFSETS]


def _tsc_pieces(positions, cell):
    """TSC base cell (stop_gradient) and the 3 per-axis weights (-1, 0, +1).

    The weights are the standard quadratic B-spline pieces about the nearest
    cell centre and sum to 1 identically for any offset d in [-1/2, 1/2]:
    0.5(0.5-d)^2 + (0.75-d^2) + 0.5(0.5+d)^2 == 1.
    """
    s = positions / float(cell)
    base = jnp.round(s)
    d = s - base
    base = jax.lax.stop_gradient(base).astype(jnp.int32)
    w_m = 0.5 * (0.5 - d) ** 2
    w_0 = 0.75 - d**2
    w_p = 0.5 * (0.5 + d) ** 2
    return base, (w_m, w_0, w_p)


def _tsc_corner_flat_weight(base, w, corner, n_mesh):
    dx, dy, dz = corner
    ix = (base[:, 0] + dx) % n_mesh
    iy = (base[:, 1] + dy) % n_mesh
    iz = (base[:, 2] + dz) % n_mesh
    ww = w[dx + 1][:, 0] * w[dy + 1][:, 1] * w[dz + 1][:, 2]
    return (ix * n_mesh + iy) * n_mesh + iz, ww


def paint_tsc_f64(positions, n_mesh, box_size, n_particles_total):
    """Global periodic TSC paint -> delta (n_mesh^3) f64.

    The order-dependent twin; see the section note. Bitwise-transcribed from
    `v2_g5_core.paint_tsc_f64`, whose output D-v2-10's coarse arm is measured
    against.
    """
    N = int(n_mesh)
    cell = float(box_size) / N
    base, w = _tsc_pieces(positions, cell)
    mesh = jnp.zeros((N**3,), dtype=jnp.float64)
    for corner in _TSC_CORNERS:
        flat, ww = _tsc_corner_flat_weight(base, w, corner, N)
        mesh = mesh.at[flat].add(ww, mode="promise_in_bounds")
    mean = float(n_particles_total) / float(N) ** 3
    return mesh.reshape(N, N, N) / mean - 1.0


# The worst-case per-cell weight sum for TSC, as a multiple of the maximum cell
# occupancy. DERIVED, not measured, and it is not 1.
#
# Per axis the weights are w_m = 0.5(0.5-d)^2, w_0 = 0.75-d^2, w_p = 0.5(0.5+d)^2
# for d in [-1/2, 1/2]. A given target cell can receive from any of the 27 cells
# in its neighbourhood, and the largest weight a particle in each can send is set
# by how many of its axes sit at offset 0 (max 0.75, at d=0) versus +-1 (max 0.5,
# at d=+-1/2):
#
#     1 cell,  3 axes at offset 0     0.75^3            = 0.421875
#     6 cells, 1 axis at +-1          0.5 * 0.75^2      = 0.281250  -> 1.6875
#    12 cells, 2 axes at +-1          0.5^2 * 0.75      = 0.187500  -> 2.2500
#     8 cells, 3 axes at +-1          0.5^3             = 0.125000  -> 1.0000
#                                                          total    = 5.359375
#
# Each neighbour's maximum is attained at a different d, but they are different
# PARTICLES in different cells, each free to sit at its own worst offset, so the
# sum is a genuine upper bound rather than a sum of unattainable maxima.
#
# CIC's existing bound uses an implicit factor of 1, which is optimistic: a cell
# receives from 8 cells with per-axis weights up to 1, so the strict CIC factor
# is 8. It has never bitten because the default configuration carries ~52x
# headroom, but it means this constant is not "the TSC version of a factor that
# was 1" -- it is the first one that was derived at all. That strict CIC factor
# is now `CIC_CELL_WEIGHT_BOUND` above, passed explicitly by new call sites;
# `check_int_paint_headroom`'s default stays 1.0 so no ratified configuration
# changes its refusal status underneath us.
TSC_CELL_WEIGHT_BOUND = 5.359375


def check_tsc_paint_headroom(n_particles_total, frac_bits, max_cell_particles=1.0e4):
    """Setup-time refusal against int32 overflow in the TSC accumulator.

    Same shape as `check_int_paint_headroom` but with the 27-corner stencil's own
    bound, which is 5.36x the occupancy rather than CIC's implicit 1x. At the
    default frac_bits=12 and 1e4 particles per cell the worst-case sum is
    ~2.20e8 against 2^31, so ~9.8x headroom; at frac_bits=15 it is ~1.76e9 and
    the margin is down to 1.2x, which is the regime this exists to refuse.
    """
    worst_cell_sum = (
        TSC_CELL_WEIGHT_BOUND * min(float(n_particles_total), max_cell_particles)
        * 2.0**frac_bits
    )
    if worst_cell_sum >= 2.0**31:
        raise ValueError(
            f"TSC int-paint headroom: worst-case cell sum ~{worst_cell_sum:.2e} >= 2^31 at "
            f"frac_bits={frac_bits} (assumed max cell occupancy "
            f"{max_cell_particles:.1e} particles, stencil bound "
            f"{TSC_CELL_WEIGHT_BOUND}); lower frac_bits -- the int32 accumulator would "
            "overflow (D-007-class corruption, not just imprecision)."
        )


def paint_tsc_int(positions, n_mesh, box_size, frac_bits=12, live=None):
    """Deterministic integer-accumulation TSC paint. Returns the raw int32 mesh.

    The D-v2-16 clause 2 deliverable. `paint_tsc_f64` accumulates through
    order-dependent f64 `.at[].add`, so the coarse arm ratified in D-v2-10
    violates D-006 today; integer addition is associative, so this is
    bit-identical regardless of atomic order. It is also a precondition for the
    brick-sorted layout, which reorders particles every step -- an
    order-dependent primal paint on a state whose order changes every step is
    not reproducible even on one machine.

    Primal-only, exactly as `paint_int` is: the differentiable twin is
    `paint_tsc_f64`, in the same relationship `paint_f32` has to `paint_int`.

    NB the quantization is per CORNER, so the 27 rounded weights of one particle
    do not sum to exactly 2^frac_bits the way the exact weights sum to 1. That is
    the same trade `paint_int` makes across 8 corners and it is a mass error of
    order 27 * 2^-frac_bits per particle, not a determinism problem.

    **`live` exists to keep the STREAMED paint on one XLA shape (M-v2-3).** The
    engine accumulates this over chunks of bricks, and a chunk's row count varies,
    so each chunk keys a new shape and recompiles -- measured at 2.91 s of a
    13.21 s step across 8 chunks, the same trap that cost 24.1 s in the
    long-range read. Padding needs a MASK rather than filler positions, because
    an unmasked pad row would add real mass to the mesh; masked rows contribute a
    quantized weight of exactly zero, so the accumulated mesh is bitwise what the
    unpadded chunks give.
    """
    scale = np.float32(2.0**frac_bits)  # np scalar: no device array at trace-build time
    N = int(n_mesh)
    cell = float(box_size) / N
    base, w = _tsc_pieces(positions, cell)
    mesh = jnp.zeros((N**3,), dtype=jnp.int32)
    m = None if live is None else jnp.asarray(live)
    for corner in _TSC_CORNERS:
        flat, ww = _tsc_corner_flat_weight(base, w, corner, N)
        if m is not None:
            ww = jnp.where(m, ww, 0.0)
        mesh = mesh.at[flat].add(
            rint_i(ww.astype(jnp.float32) * scale), mode="promise_in_bounds"
        )
    return mesh.reshape(N, N, N)


def density_tsc(positions, n_mesh, box_size, n_particles_total, paint="f64", frac_bits=12,
                fdtype=jnp.float64):
    """delta from a TSC assignment -- the ONE interface over both TSC paints.

    paint="int": the deterministic primal path (D-006 compliant).
    paint="f64": the order-dependent differentiable twin.

    **The default is "f64" and that is deliberate, not an oversight.** It is what
    the ratified coarse arm ran, so it is what D-v2-10, D-v2-11 and D-v2-12
    measured; flipping the default moves those numbers, which is an ADR-level
    call rather than a promotion detail. M-v2-3 is where the engine chooses, and
    until then the D-006-compliant path exists and is tested but is opt-in.

    `fdtype` (M-v2-4) is the dtype of the RETURNED FIELD. The decode and the
    mean subtraction stay f64 whatever it is.

    **HOW MUCH THAT ORDERING BUYS: almost nothing, measured.** It is kept
    because it costs nothing and is never worse, not because it rescues a case.
    Recording the measurement, because the plausible-sounding argument for it is
    wrong and would otherwise get re-derived:

      - `counts_from_int` is exact while the raw sums fit f32's significand.
        The usual statement of that is "< 2^24", which is SUFFICIENT but not
        necessary -- what matters is significand width, not magnitude, so
        `5000 * 2^12 = 625 * 2^15` decodes exactly despite being 2.05e7.
      - the coarse mean is exactly 8.0 at every config in the table (mesh:
        particle 2, coarse = fine/4), and `2**-frac_bits` is a power of two, so
        `counts / mean` is an exact rescaling that cannot move the significand.
        **Without the `- 1.0` the two orders are therefore bitwise identical,
        always** -- which is why `tile_delta_from_int`, whose field carries no
        `- 1`, gains nothing at all from the f64 decode.
      - with the `- 1.0` they can differ, but only for a cell whose raw sum
        needs more than 24 bits, i.e. >= 512x the mean. A near-mean cell has a
        raw sum around 2^15 and is exact in BOTH orders. Measured at a raw sum
        of 2^24 + 12345: f64-then-narrow is exact, direct-f32 is off by 3.05e-5
        on a delta of 511 -- 6e-8 relative, on the cells that matter least.

    So the Sterbenz-cancellation argument for this ordering is true and
    IRRELEVANT: the operands it protects were never inexact. Past the bound the
    f32 decode still rounds to nearest deterministically, so D-006 is untouched
    either way and what is at stake is exactness, not reproducibility.
    """
    fdtype = field_dtype(fdtype)
    if paint == "int":
        check_tsc_paint_headroom(n_particles_total, frac_bits)
        counts = counts_from_int(paint_tsc_int(positions, n_mesh, box_size, frac_bits),
                                 frac_bits, fdtype=jnp.float64)
        mean = float(n_particles_total) / float(n_mesh) ** 3
        return (counts / mean - 1.0).astype(fdtype)
    if paint == "f64":
        return paint_tsc_f64(positions, n_mesh, box_size, n_particles_total).astype(fdtype)
    raise ValueError(f"paint must be 'int' or 'f64', got {paint!r}")


# THE GATHER NARROWING RULE (M-v2-4), obeyed identically by all four gathers --
# the two here, `forces.tile_gather_vector` and `forces.gather_coarse_subblock`.
#
# Accumulate in the FIELD's dtype, and narrow the three-factor corner weight to
# that dtype AFTER forming the product, never before.
#
# Both halves are load-bearing.
#
#   - Following the field is what makes an f32 arm real. `dtype=gx.dtype` alone
#     does NOT: the weights come from f64 positions, and `f32_acc + f64_w *
#     f32_field` promotes the whole accumulation back to f64. Three of these
#     four already read the field's dtype and still returned f64 for exactly
#     that reason. `tests/test_force_dtypes.py` pins the trap.
#   - Narrowing the PRODUCT rather than the per-axis weights is what keeps
#     `gather_coarse_subblock` bitwise `tsc_read_vector`, which is a D-v2-16
#     clause 3 contract that has already refused a `jax.jit` at 2.220e-16.
#     f32(a)*f32(b)*f32(c) is not f32(a*b*c), so a gather that narrowed its axis
#     weights first would compute the product at a different precision than the
#     global gather it must match. MEASURED in situ, not argued: narrowing early
#     breaks the f32 contract on 113 of 186 elements at 1.19e-7, on the UNMASKED
#     path -- so it diverges everywhere, not only where padded rows are masked.
#     The two `ww` expressions are bitwise equal today (same three factors, same
#     left-to-right order, same corner sequence), and casting a bitwise-equal
#     pair leaves it bitwise equal, so the contract survives at BOTH dtypes.
#     (Masking is a second, independent reason not to touch `w_axis`: the
#     sub-block gather zeroes its axis weights for padded rows and the global
#     gather has no mask at all.)
#
# Positions, cell indices and the fractional offsets stay f64 throughout: at
# C-gh a global coordinate is O(1024) coarse cells, where an f32 ulp is 6.1e-5
# cells against a T9 quantum of 0.0039, and narrowing them would move which cell
# a particle lands in near a boundary.


def tsc_read_vector(gx, gy, gz, positions, n_mesh, box_size):
    """Read 3 mesh fields with ONE shared TSC stencil.

    The GATHER has no determinism problem at all -- it is a read followed by a
    per-particle sum in a fixed unrolled order, with no atomics -- so unlike the
    paint it needs no integer twin.

    Dtype follows the field, per the narrowing rule above.
    """
    N = int(n_mesh)
    cell = float(box_size) / N
    base, w = _tsc_pieces(positions, cell)
    fx, fy, fz = gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)
    n = positions.shape[0]
    dt = gx.dtype
    ax = jnp.zeros((n,), dtype=dt)
    ay = jnp.zeros((n,), dtype=dt)
    az = jnp.zeros((n,), dtype=dt)
    for corner in _TSC_CORNERS:
        flat, ww = _tsc_corner_flat_weight(base, w, corner, N)
        ww = ww.astype(dt)
        ax = ax + ww * fx[flat]
        ay = ay + ww * fy[flat]
        az = az + ww * fz[flat]
    return jnp.stack([ax, ay, az], axis=1)


def cic_read_vector(gx, gy, gz, positions, n_mesh, box_size):
    """Read 3 mesh fields with ONE shared CIC stencil (mbody painting.py:105;
    ~40% reverse-mode memory saving measured there).

    Dtype follows the field, per the narrowing rule above.
    """
    base, frac = _cic_pieces(positions, n_mesh, box_size)
    fx, fy, fz = gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)
    n = positions.shape[0]
    dt = gx.dtype
    ax = jnp.zeros((n,), dtype=dt)
    ay = jnp.zeros((n,), dtype=dt)
    az = jnp.zeros((n,), dtype=dt)
    for corner in _CORNERS:
        flat, w = _corner_flat_weight(base, frac, corner, n_mesh)
        w = w.astype(dt)
        ax = ax + w * fx[flat]
        ay = ay + w * fy[flat]
        az = az + w * fz[flat]
    return jnp.stack([ax, ay, az], axis=1)
