"""Lagrangian perturbation theory: turn a density field into moving particles
(architecture.md Sec. 7; mbody lpt.py port, jnp + D-time conventions).

Sign convention (carried verbatim from mbody, validated by the skewness
test): Psi here is +grad lap^-1 delta, so div Psi1 = -delta0 -- the OPPOSITE
potential sign to the textbook 2LPT x = q - D1 grad phi1 + D2 grad phi2.
That flips the sign of the second-order term: positions are

    x = q + D1 Psi1 - D2 Psi2        (D2 = -(3/7) D1^2, so -D2 = +(3/7) D1^2)

and the D-time velocity v = dx/dD1 (the w-frame's native variable) is

    v_D = Psi1 - (D2 f2)/(D1 f1) Psi2 = Psi1 + (6/7) D1 Psi2

using f2 = 2 f1 and D2 = -(3/7) D1^2 (cosmology.growth_{factor,rate}_2).
A ZA mode's v_D = Psi1 is CONSTANT in D -- the design premise that makes the
quantized w-frame work (architecture.md Sec. 3).

All functions take the density mesh delta0 (z=0 normalized) as input --
generation lives in ic.py, composition in integrate.simulate. Displacement
outputs are (N^3, 3) flattened C-order, matching lagrangian_grid.
"""

import jax.numpy as jnp
import numpy as np

from .cosmology import growth_factor_2, growth_factor_a, growth_rate_2, growth_rate_a
from .forces import k_components


def _apply_ik_over_k2(field_k, n_mesh, box_size, fdtype):
    """(ik/k^2) field_k -> (N^3, 3) real displacement-like vector field."""
    N = n_mesh
    npdt = np.float64 if fdtype == jnp.float64 else np.float32
    ikx, iky, ikz, inv_k2 = k_components(N, box_size, npdt)
    px = jnp.fft.irfftn(field_k * ikx * inv_k2, s=(N, N, N))
    py = jnp.fft.irfftn(field_k * iky * inv_k2, s=(N, N, N))
    pz = jnp.fft.irfftn(field_k * ikz * inv_k2, s=(N, N, N))
    return jnp.stack([px.reshape(-1), py.reshape(-1), pz.reshape(-1)], axis=1).astype(fdtype)


def zeldovich_displacement(delta0, box_size, fdtype=jnp.float32):
    """Zel'dovich displacement Psi1 = (ik/k^2) delta0, shape (N^3, 3), z=0 norm.

    Same kernel as the force solve (forces.k_components) -- the force == ZA
    identity is a permanent test. The Nyquist plane (ill-defined spectral
    gradient of a real field) is handled by irfftn's real projection; it is
    the only residual in the div Psi1 = -delta0 identity (mbody test_lpt).
    """
    N = delta0.shape[0]
    return _apply_ik_over_k2(jnp.fft.rfftn(delta0), N, box_size, fdtype)


def _second_derivatives(delta0, box_size, fdtype=jnp.float32):
    """The six unique second derivatives phi,ij of the linear potential.

    phi_k = delta_k / k^2, phi,ij(k) = (i k_i)(i k_j) phi_k. Returns a dict
    keyed by (0,0),(1,1),(2,2),(0,1),(0,2),(1,2), each a real (N,N,N) field.
    """
    N = delta0.shape[0]
    npdt = np.float64 if fdtype == jnp.float64 else np.float32
    ikx, iky, ikz, inv_k2 = k_components(N, box_size, npdt)
    ik = (ikx, iky, ikz)
    phi_k = jnp.fft.rfftn(delta0) * inv_k2
    out = {}
    for i, j in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]:
        out[(i, j)] = jnp.fft.irfftn(ik[i] * ik[j] * phi_k, s=(N, N, N)).astype(fdtype)
    return out


def lpt2_source(delta0, box_size, fdtype=jnp.float32):
    """Second-order (2LPT) density source delta2 from the linear density.

    delta2(x) = sum_{i<j} [ phi,ii phi,jj - (phi,ij)^2 ] (Bouchet et al. 1995;
    Scoccimarro 1998). Quadratic in delta0 -- hence differentiable in f_NL and
    scaling as amplitude^2. Returns a real (N,N,N) field.
    """
    d = _second_derivatives(delta0, box_size, fdtype)
    pxx, pyy, pzz = d[(0, 0)], d[(1, 1)], d[(2, 2)]
    pxy, pxz, pyz = d[(0, 1)], d[(0, 2)], d[(1, 2)]
    return pxx * pyy + pxx * pzz + pyy * pzz - pxy**2 - pxz**2 - pyz**2


def second_order_displacement(delta0, box_size, fdtype=jnp.float32):
    """2LPT displacement Psi2 = (ik/k^2) delta2, shape (N^3, 3), z=0 normalized.

    div Psi2 = -delta2 by the same kernel identity as Psi1.
    """
    N = delta0.shape[0]
    d2k = jnp.fft.rfftn(lpt2_source(delta0, box_size, fdtype))
    return _apply_ik_over_k2(d2k, N, box_size, fdtype)


def divergence(psi, box_size):
    """FFT divergence of an (N^3, 3) vector field; returns a real (N,N,N) mesh.

    Diagnostic: div Psi1 == -delta0 away from the Nyquist plane.
    """
    n3 = psi.shape[0]
    N = round(n3 ** (1.0 / 3.0))
    npdt = np.float64 if psi.dtype == jnp.float64 else np.float32
    ikx, iky, ikz, _ = k_components(N, box_size, npdt)
    px = psi[:, 0].reshape(N, N, N)
    py = psi[:, 1].reshape(N, N, N)
    pz = psi[:, 2].reshape(N, N, N)
    div_k = jnp.fft.rfftn(px) * ikx + jnp.fft.rfftn(py) * iky + jnp.fft.rfftn(pz) * ikz
    return jnp.fft.irfftn(div_k, s=(N, N, N))


def lagrangian_grid(n_mesh, box_size, fdtype=jnp.float32):
    """Unperturbed particle positions (one per cell), shape (N^3, 3)."""
    N = n_mesh
    d = box_size / N
    coords = jnp.arange(N, dtype=fdtype) * d
    qx, qy, qz = jnp.meshgrid(coords, coords, coords, indexing="ij")
    return jnp.stack([qx.reshape(-1), qy.reshape(-1), qz.reshape(-1)], axis=1)


def za_ics(delta0, box_size, a_init, cosmo, fdtype=jnp.float32, D_of_a=None):
    """ZA state at a_init: x = wrap(q + D_i Psi1), v = dx/dD = Psi1 (D-time).

    The D-time velocity of a ZA mode is Psi1, CONSTANT in D -- the w-frame's
    design premise (architecture.md Sec. 3). Returns (x_phys, v) each (N^3, 3).
    D_of_a: optional growth override (e.g. lambda a: a for EdS pin tests).
    """
    N, L = delta0.shape[0], box_size
    D_i = D_of_a(a_init) if D_of_a is not None else growth_factor_a(a_init, cosmo)
    psi = zeldovich_displacement(delta0, L, fdtype)
    q = lagrangian_grid(N, L, fdtype)
    x = jnp.mod(q + jnp.asarray(D_i, dtype=fdtype) * psi, L)
    return x, psi


def lpt_ics(delta0, box_size, a_init, cosmo, order=2, fdtype=jnp.float32):
    """LPT state at a_init in inexor conventions: positions + D-time velocity.

    order=1: x = wrap(q + D1 Psi1),          v_D = Psi1
    order=2: x = wrap(q + D1 Psi1 - D2 Psi2), v_D = Psi1 - (D2 f2)/(D1 f1) Psi2
    (module-docstring sign convention; the v_D coefficient reduces to
    +(6/7) D1 with f2 = 2 f1 and D2 = -(3/7) D1^2). Returns (x_phys, v_D),
    each (N^3, 3) fdtype. mbody's a-time momentum is p = G_f(a_i) * v_D with
    G_f = a^3 E D' (the conversion the parity harness applies).
    """
    if order == 1:
        return za_ics(delta0, box_size, a_init, cosmo, fdtype)
    if order != 2:
        raise ValueError(f"order must be 1 or 2, got {order}")
    N, L = delta0.shape[0], box_size
    D1 = growth_factor_a(a_init, cosmo)
    f1 = growth_rate_a(a_init, cosmo)
    D2 = growth_factor_2(a_init, cosmo)
    f2 = growth_rate_2(a_init, cosmo)
    psi1 = zeldovich_displacement(delta0, L, fdtype)
    psi2 = second_order_displacement(delta0, L, fdtype)
    q = lagrangian_grid(N, L, fdtype)
    x = jnp.mod(q + jnp.asarray(D1, fdtype) * psi1 - jnp.asarray(D2, fdtype) * psi2, L)
    v_coef2 = -(D2 * f2) / (D1 * f1)  # == +(6/7) D1 for EdS-approx D2, f2
    v = psi1 + jnp.asarray(v_coef2, fdtype) * psi2
    return x, v
