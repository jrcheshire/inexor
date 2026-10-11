"""CIC and TSC paint / read.

Each stencil has two paints behind one interface:
- integer paint (`paint_int`, `paint_tsc_int`): corner weights quantized to `frac_bits` fixed
  point and scatter-added into an int32 mesh, each particle's weights summing to exactly
  2^frac_bits (`_add_exact_mass`). Integer addition is associative, so the result is
  bit-identical regardless of atomic or particle order. Primal-only.
- float twin (`paint_f32`, `paint_tsc_f64`): differentiable, but order-dependent under float
  atomics on GPU, so never used on the deterministic primal force path.

Gradients flow through the fractional weights only; the integer base cell is stop_gradient.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .codec import rint_i

_CORNERS = [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]


def field_dtype(fdtype):
    """Normalize and validate a mesh/field dtype: float32 or float64, else ValueError.

    For decoded FIELDS only, never paint accumulators, which are fixed (int32 for the
    integer paints, f64 for the float twins, int64 for host accumulation): a float
    accumulator would make the primal paint order-dependent.
    """
    dt = np.dtype(fdtype)
    if dt not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"field dtype must be float32 or float64, got {dt.name}")
    return dt


def _cic_pieces(positions, n_mesh, box_size):
    """CIC base cell (int32, stop_gradient) and fractional offset in cells."""
    d = box_size / n_mesh
    xp = positions / d
    base_f = jnp.floor(xp)
    frac = xp - base_f
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


def _cic_is_nearest(frac, corner):
    """Bool row mask: `corner` (offsets in {0, 1} from the base cell) is the CIC corner nearest
    the particle, the largest of its 8 weights (>= 1/8). Per axis the nearer cell is base + 1
    where frac >= 1/2, so exactly one corner is nearest for every row."""
    up = frac >= 0.5
    m = [up[:, a] if d else ~up[:, a] for a, d in enumerate(corner)]
    return m[0] & m[1] & m[2]


def _add_exact_mass(mesh, corners, piece, nearest, frac_bits, live=None):
    """Scatter-add each particle's corner weights in `frac_bits` fixed point, exactly
    2^frac_bits per particle.

    `piece(corner)` -> (flat index, weight); `nearest(corner)` -> True / False when that corner
    is / is not the nearest for every particle (TSC's centre), else a bool row mask (CIC).
    Every other corner adds its weight rounded through f32; the nearest adds 2^frac_bits minus
    their sum. The nearest weight is >= 1/8 for CIC and TSC and the remainder's rounding is at
    most half a unit per other corner, so it is never negative. Rows with `live` False must
    carry zero weights and add 0 everywhere. Each corner is traced twice, for the remainder and
    for its scatter, so only the remainder outlives a corner's own arrays.
    """
    scale = np.float32(2.0**frac_bits)  # np scalars: no device array at trace-build time

    def code(w):
        return rint_i(w.astype(jnp.float32) * scale)

    rest = None
    for c in corners:
        near = nearest(c)
        if near is True:
            continue
        q = code(piece(c)[1])
        if near is not False:
            q = jnp.where(near, 0, q)
        rest = q if rest is None else rest + q
    rem = np.int32(1 << frac_bits) - rest
    if live is not None:
        rem = jnp.where(live, rem, 0)
    for c in corners:
        near = nearest(c)
        flat, w = piece(c)
        if near is True:
            q = rem
        elif near is False:
            q = code(w)
        else:
            q = jnp.where(near, rem, code(w))
        mesh = mesh.at[flat].add(q, mode="promise_in_bounds")
    return mesh


def paint_f32(positions, n_mesh, box_size, fdtype=jnp.float32):
    """Differentiable CIC paint of counts; order-dependent on GPU (use paint_int on the primal)."""
    mesh = jnp.zeros((n_mesh**3,), dtype=fdtype)
    base, frac = _cic_pieces(positions, n_mesh, box_size)
    for corner in _CORNERS:
        flat, w = _corner_flat_weight(base, frac, corner, n_mesh)
        mesh = mesh.at[flat].add(w.astype(fdtype), mode="promise_in_bounds")
    return mesh.reshape(n_mesh, n_mesh, n_mesh)


def paint_int(positions, n_mesh, box_size, frac_bits=12):
    """Deterministic integer CIC paint; returns the raw int32 mesh (decode: counts_from_int).

    Bit-identical under any scatter order; each particle paints exactly 2^frac_bits
    (`_add_exact_mass`).
    """
    mesh = jnp.zeros((n_mesh**3,), dtype=jnp.int32)
    base, frac = _cic_pieces(positions, n_mesh, box_size)
    mesh = _add_exact_mass(mesh, _CORNERS, lambda c: _corner_flat_weight(base, frac, c, n_mesh),
                           lambda c: _cic_is_nearest(frac, c), frac_bits)
    return mesh.reshape(n_mesh, n_mesh, n_mesh)


def counts_from_int(mesh_int, frac_bits=12, fdtype=jnp.float32):
    """Decode a raw int32 paint to float counts (exact while sums fit f32's 24-bit significand)."""
    return mesh_int.astype(fdtype) * np.float32(2.0**-frac_bits)


# Strict worst-case CIC cell-weight sum per unit occupancy: 8 neighbour cells, per-axis weight <= 1.
CIC_CELL_WEIGHT_BOUND = 8.0


def check_int_paint_headroom(
    n_particles_total, frac_bits, max_cell_particles=1.0e4, bound=1.0
):
    """Refuse (ValueError) a config whose worst-case int32 cell sum could reach 2^31.

    Worst case = bound x min(N, max_cell_particles) x 2^frac_bits; overflow wraps silently.
    Exceeding the softer exact-decode limit (2^24) only costs count precision and is not
    refused. The default `bound=1.0` is optimistic; it is kept so existing refusal boundaries do
    not move, and callers wanting the strict limit pass `CIC_CELL_WEIGHT_BOUND`.
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
    """CIC density contrast delta; paint="int" (deterministic, default) or "f32" (differentiable)."""
    if paint == "int":
        check_int_paint_headroom(n_particles_total, frac_bits)
        counts = counts_from_int(paint_int(positions, n_mesh, box_size, frac_bits), frac_bits)
    elif paint == "f32":
        counts = paint_f32(positions, n_mesh, box_size)
    else:
        raise ValueError(f"paint must be 'int' or 'f32', got {paint!r}")
    mean = n_particles_total / n_mesh**3
    return counts / mean - 1.0


# TSC (coarse long-range arm): 3-point stencil per axis about the NEAREST cell; its sinc^3
# window suppresses high-k aliasing on a coarse mesh better than CIC's sinc^2.

_TSC_OFFSETS = (-1, 0, 1)
_TSC_CORNERS = [(dx, dy, dz) for dx in _TSC_OFFSETS for dy in _TSC_OFFSETS for dz in _TSC_OFFSETS]


def _tsc_is_nearest(corner):
    """The TSC centre is every particle's nearest corner, its largest weight (>= 1/8)."""
    return corner == (0, 0, 0)


def _tsc_pieces(positions, cell):
    """TSC base cell (int32, stop_gradient) and per-axis weights for offsets (-1, 0, +1).

    Quadratic B-spline pieces in d in [-1/2, 1/2]; they sum to 1 identically.
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
    """Periodic TSC paint -> f64 delta (n_mesh^3); the order-dependent differentiable twin."""
    N = int(n_mesh)
    cell = float(box_size) / N
    base, w = _tsc_pieces(positions, cell)
    mesh = jnp.zeros((N**3,), dtype=jnp.float64)
    for corner in _TSC_CORNERS:
        flat, ww = _tsc_corner_flat_weight(base, w, corner, N)
        mesh = mesh.at[flat].add(ww, mode="promise_in_bounds")
    mean = float(n_particles_total) / float(N) ** 3
    return mesh.reshape(N, N, N) / mean - 1.0


# Worst-case TSC cell-weight sum per unit occupancy: over the 27 neighbours (max per-axis weight
# 0.75 at offset 0, 0.5 at +-1), 0.75^3 + 6(0.5)(0.75^2) + 12(0.5^2)(0.75) + 8(0.5^3).
TSC_CELL_WEIGHT_BOUND = 5.359375
# Fixed-point rounding on top of it, in units of 2^-frac_bits per unit occupancy: a particle in
# the cell adds its centre, at most 13 over its weight (26 corners' half-unit roundings); one in
# each of the 26 neighbours adds a rounded corner, at most half a unit over.
TSC_ROUNDING_UNITS = 26.0


def check_tsc_paint_headroom(n_particles_total, frac_bits, max_cell_particles=1.0e4):
    """`check_int_paint_headroom` with the strict TSC bound `TSC_CELL_WEIGHT_BOUND`, plus the
    fixed-point rounding `TSC_ROUNDING_UNITS`.

    At frac_bits=12 and 1e4 particles/cell the worst case is ~2.2e8 (~9.8x below 2^31).
    """
    worst_cell_sum = min(float(n_particles_total), max_cell_particles) * (
        TSC_CELL_WEIGHT_BOUND * 2.0**frac_bits + TSC_ROUNDING_UNITS
    )
    if worst_cell_sum >= 2.0**31:
        raise ValueError(
            f"TSC int-paint headroom: worst-case cell sum ~{worst_cell_sum:.2e} >= 2^31 at "
            f"frac_bits={frac_bits} (assumed max cell occupancy "
            f"{max_cell_particles:.1e} particles, stencil bound "
            f"{TSC_CELL_WEIGHT_BOUND}); lower frac_bits -- the int32 accumulator would "
            "overflow (silent wraparound, not just imprecision)."
        )


def paint_tsc_int(positions, n_mesh, box_size, frac_bits=12, live=None):
    """Deterministic integer TSC paint; returns the raw int32 mesh. Primal-only.

    Bit-identical under any scatter or particle order, which the brick-sorted layout (reordered
    every step) requires; each particle paints exactly 2^frac_bits (`_add_exact_mass`). `live`
    (bool mask) lets streamed callers pad chunks to one compiled shape: masked rows add exactly
    zero, so the mesh is bitwise the unpadded one.
    """
    N = int(n_mesh)
    cell = float(box_size) / N
    base, w = _tsc_pieces(positions, cell)
    mesh = jnp.zeros((N**3,), dtype=jnp.int32)
    m = None if live is None else jnp.asarray(live)

    def piece(corner):
        flat, ww = _tsc_corner_flat_weight(base, w, corner, N)
        return flat, (ww if m is None else jnp.where(m, ww, 0.0))

    mesh = _add_exact_mass(mesh, _TSC_CORNERS, piece, _tsc_is_nearest, frac_bits, live=m)
    return mesh.reshape(N, N, N)


def paint_tsc_int_subblock(positions, origin_cells, extent, n_mesh, box_size,
                           frac_bits=12, live=None, dead_rows="spread"):
    """Integer TSC paint into a coarse sub-block at `origin_cells` of shape `extent`.

    Weights come from the GLOBAL coordinate; only the integer index is rebased
    ((base + corner - origin) mod n_mesh), so each contribution is bitwise `paint_tsc_int`'s.
    Containment is the caller's contract, checked on the host (a live row leaving the block
    wraps silently); the stencil reaches span + 3 cells, and extent == n_mesh is the full axis.
    `extent`: Python ints; `origin_cells`: ints or an int32 array, possibly traced.
    `dead_rows`: where masked rows add their zero, "spread" (row i -> cell i mod block size) or
    "cell0"; bitwise identical, but "cell0" is a heavily duplicated index, slow on some GPUs.
    """
    N = int(n_mesh)
    cell = float(box_size) / N
    ex, ey, ez = (int(e) for e in extent)
    o = jnp.asarray(origin_cells, dtype=jnp.int32)
    ox, oy, oz = o[0], o[1], o[2]
    base, w = _tsc_pieces(positions, cell)
    mesh = jnp.zeros((ex * ey * ez,), dtype=jnp.int32)
    m = None if live is None else jnp.asarray(live)
    if dead_rows == "cell0":
        dead = 0
    elif dead_rows == "spread":
        dead = jnp.arange(positions.shape[0], dtype=jnp.int32) % (ex * ey * ez)
    else:
        raise ValueError(f"dead_rows must be 'cell0' or 'spread', got {dead_rows!r}")

    def piece(corner):
        dx, dy, dz = corner
        lx = (base[:, 0] + dx - ox) % N
        ly = (base[:, 1] + dy - oy) % N
        lz = (base[:, 2] + dz - oz) % N
        ww = w[dx + 1][:, 0] * w[dy + 1][:, 1] * w[dz + 1][:, 2]
        flat = (lx * ey + ly) * ez + lz
        if m is not None:
            ww = jnp.where(m, ww, 0.0)
            flat = jnp.where(m, flat, dead)
        return flat, ww

    mesh = _add_exact_mass(mesh, _TSC_CORNERS, piece, _tsc_is_nearest, frac_bits, live=m)
    return mesh.reshape(ex, ey, ez)


def density_tsc(positions, n_mesh, box_size, n_particles_total, paint="f64", frac_bits=12,
                fdtype=jnp.float64):
    """TSC density contrast delta; paint="int" (deterministic) or "f64" (differentiable twin).

    The default stays "f64" so existing results do not move. `fdtype` is the returned field's
    dtype; the int decode and mean subtraction are done in f64 and narrowed last.
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


# Gather dtype rule (also forces.tile_gather_vector / gather_coarse_subblock): accumulate in the
# field's dtype, narrowing the corner-weight PRODUCT (not the per-axis weights, which would break
# sub-block == global bitwise equality). Positions and offsets stay f64.


def tsc_read_vector(gx, gy, gz, positions, n_mesh, box_size):
    """Read 3 mesh fields at `positions` with one shared TSC stencil -> (n, 3), field dtype.

    Deterministic without an integer twin: no atomics, fixed summation order.
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
    """Read 3 mesh fields at `positions` with one shared CIC stencil -> (n, 3), field dtype."""
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
