"""Geometric PM force solve (architecture.md Sec. 6; mbody forces.py port).

Geometry/cosmology split: this module solves the DIMENSIONLESS Poisson
equation div g = -delta and returns g = -grad phi; ALL cosmology prefactors
live in the integrator coefficients (integrate.py). The force kernel
ik/k^2 is literally the Zel'dovich displacement kernel -- k_components is
the ONE shared implementation (lpt.py imports it), and force == ZA identity
at machine precision is a permanent test.

Sequential per-component solves: never materialize three force meshes at
once (the architecture Sec. 9 memory budget depends on it).
"""

from functools import lru_cache

import jax.numpy as jnp
import numpy as np

from .painting import cic_read_vector, density_contrast


def k_components(n_mesh, box_size, fdtype=np.float32):
    """i k_j (complex, low-rank) and 1/k^2 (full) on the rfftn half-grid, k=0 safe.

    Host-numpy build (mbody lpt.py:45 port); returns jnp arrays of the dtype
    matching fdtype (complex64 for f32, complex128 for f64).
    """
    N, L = n_mesh, box_size
    d = L / N
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=d)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=d)
    M = N // 2 + 1
    KX = kx.reshape(N, 1, 1)
    KY = kx.reshape(1, N, 1)
    KZ = kz.reshape(1, 1, M)
    k2 = KX**2 + KY**2 + KZ**2
    k2[0, 0, 0] = 1.0  # avoid 0/0; the ik_j are zero at k=0 anyway
    cdtype = np.complex128 if fdtype == np.float64 else np.complex64
    return (
        jnp.asarray((1j * KX).astype(cdtype)),
        jnp.asarray((1j * KY).astype(cdtype)),
        jnp.asarray((1j * KZ).astype(cdtype)),
        jnp.asarray((1.0 / k2).astype(fdtype)),
    )


@lru_cache(maxsize=8)
def _cached_force_fn(box, fdtype_name, paint, frac_bits):
    """Kernel-carrying force closure per (BoxConfig, dtype, paint) -- BoxConfig
    is frozen/hashable by design (config.py). lru_cache'd so repeated calls
    share ONE function object: step_fwd and step_rev MUST receive the same
    object (architecture Sec. 5 fusion mitigation)."""
    fdtype = jnp.float64 if fdtype_name == "float64" else jnp.float32
    npdt = np.float64 if fdtype_name == "float64" else np.float32
    N, L = box.n_mesh, box.box_size
    ikx, iky, ikz, inv_k2 = k_components(N, L, npdt)
    n_total = box.n_total

    def force(pos):
        delta = density_contrast(pos, N, L, n_total, paint=paint, frac_bits=frac_bits).astype(
            fdtype
        )
        dk = jnp.fft.rfftn(delta)
        # Sequential per-component solves; g_j = ik_j delta_k / k^2 (div g = -delta).
        gx = jnp.fft.irfftn(dk * ikx * inv_k2, s=(N, N, N))
        gy = jnp.fft.irfftn(dk * iky * inv_k2, s=(N, N, N))
        gz = jnp.fft.irfftn(dk * ikz * inv_k2, s=(N, N, N))
        return cic_read_vector(gx, gy, gz, pos, N, L).astype(fdtype)

    return force


def make_force_fn(box, fdtype=jnp.float32, paint="int", frac_bits=12):
    """Geometric PM force: div g = -delta; force(positions (n,3)) -> (n,3) fdtype.

    paint="int" -> deterministic integer paint (PRIMAL force path -- required
    for bit-exact replay; the f32 default failing on GPU is the archived M0
    R1 f32-paint counterexample). paint="f32" -> differentiable twin (VJP
    path). Returns a cached closure: calling twice with equal args returns
    the SAME object, which step_fwd/step_rev sharing relies on.
    """
    fdtype_name = jnp.dtype(fdtype).name
    if fdtype_name not in ("float32", "float64"):
        raise ValueError(f"fdtype must be float32 or float64, got {fdtype_name}")
    if paint not in ("int", "f32"):
        raise ValueError(f"paint must be 'int' or 'f32', got {paint!r}")
    return _cached_force_fn(box, fdtype_name, paint, frac_bits)
