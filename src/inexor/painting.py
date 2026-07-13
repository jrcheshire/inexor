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


def check_int_paint_headroom(n_particles_total, frac_bits, max_cell_particles=1.0e4):
    """Setup-time refusal against int32 OVERFLOW in the paint accumulator.

    Architecture Sec. 5 budget: max cell mass ~1e4 particles x 2^frac_bits
    << 2^31. Loud, like the ladder guard. The SOFTER exact-decode bound
    (cell sums < 2^24 for exact int32->f32 conversion) is a measured per-run
    diagnostic (max cell occupancy, R3 pattern), not a setup-time refusal --
    exceeding it degrades count precision, exceeding 2^31 corrupts it.
    """
    worst_cell_sum = min(float(n_particles_total), max_cell_particles) * 2.0**frac_bits
    if worst_cell_sum >= 2.0**31:
        raise ValueError(
            f"int-paint headroom: worst-case cell sum ~{worst_cell_sum:.2e} >= 2^31 at "
            f"frac_bits={frac_bits} (assumed max cell occupancy "
            f"{max_cell_particles:.1e} particles); lower frac_bits -- the int32 "
            "accumulator would overflow (D-007-class corruption, not just imprecision)."
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


def cic_read_vector(gx, gy, gz, positions, n_mesh, box_size):
    """Read 3 mesh fields with ONE shared CIC stencil (mbody painting.py:105;
    ~40% reverse-mode memory saving measured there)."""
    base, frac = _cic_pieces(positions, n_mesh, box_size)
    fx, fy, fz = gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)
    n = positions.shape[0]
    ax = jnp.zeros((n,), dtype=gx.dtype)
    ay = jnp.zeros((n,), dtype=gx.dtype)
    az = jnp.zeros((n,), dtype=gx.dtype)
    for corner in _CORNERS:
        flat, w = _corner_flat_weight(base, frac, corner, n_mesh)
        ax = ax + w * fx[flat]
        ay = ay + w * fy[flat]
        az = az + w * fz[flat]
    return jnp.stack([ax, ay, az], axis=1)
