"""Lagrangian perturbation theory: turn a density field into moving particles
(mbody lpt.py lineage; rebuilt at M-v2-5 on the out-of-core FFT layer -- host
numpy end to end, no six-simultaneous-fields moment, D-v2-15).

Sign convention (carried verbatim from mbody, validated by the skewness
test): Psi here is +grad lap^-1 delta, so div Psi1 = -delta0 -- the OPPOSITE
potential sign to the textbook 2LPT x = q - D1 grad phi1 + D2 grad phi2.
That flips the sign of the second-order term: positions are

    x = q + D1 Psi1 - D2 Psi2        (D2 = -(3/7) D1^2, so -D2 = +(3/7) D1^2)

and the D-time velocity v = dx/dD1 is

    v_D = Psi1 - (D2 f2)/(D1 f1) Psi2 = Psi1 + (6/7) D1 Psi2

using f2 = 2 f1 and D2 = -(3/7) D1^2 (cosmology.growth_{factor,rate}_2).

MEMORY SHAPE (the M-v2-5 rebuild). The 2LPT source used to hold six
derivative fields simultaneously and lpt_ics six (N^3, 3) arrays -- the
measured n=1024 device OOM. Now:

  - `lpt2_source_from_spec` accumulates delta2 PAIRWISE under a `resident`
    policy: "mid" (dev default) keeps at most the accumulator plus three
    derivative fields; "low" (the C-gh policy) streams each derivative to a
    disk-staged memmap (D-v2-15 clause 3: disk stages the IC stage only) and
    fuses them back one axis-0 slab at a time, so no derivative field is ever
    host-resident. THE TWO POLICIES ARE BITWISE IDENTICAL BY CONSTRUCTION --
    each element sees the same multiply/add sequence (pinned by test) -- so
    "low" is a pure memory knob, never a physics one.
  - psi is produced PER COMPONENT from a spectral copy; the (N^3, 3) stacked
    outputs survive only in the monolithic conveniences (za_ics / lpt_ics),
    which dev-scale gates and probes consume. The streamed generator (icgen)
    consumes the per-component/per-slab primitives instead and never builds
    them.

All functions take the density mesh delta0 (z=0 normalized) as input --
generation lives in ic.py. Displacement outputs are (N^3, 3) flattened
C-order, matching lagrangian_grid.
"""

import os

import numpy as np

from . import ooc_fft
from .cosmology import growth_factor_2, growth_factor_a, growth_rate_2, growth_rate_a

# The six unique second derivatives, in THE canonical order: diagonals first
# (the accumulator needs all three at once), then the squared off-diagonals
# one at a time.
_DIAG = ((0, 0), (1, 1), (2, 2))
_OFFDIAG = ((0, 1), (0, 2), (1, 2))


def _np_dtype(fdtype):
    return np.dtype(fdtype)


def _psi_from_spec(spec, n_mesh, box_size, fdtype, slab=None):
    """(ik/k^2) spec -> (N^3, 3) real displacement-like vector field.

    Monolithic convenience: one component at a time from a spectral copy, so
    the transient is one field + one spec copy, never three meshes at once.
    """
    n = int(n_mesh)
    out = np.empty((n**3, 3), dtype=_np_dtype(fdtype))
    for ax in range(3):
        comp_spec = ooc_fft.grad_invk2_spec(spec, ax, n, box_size, slab=slab or n)
        out[:, ax] = ooc_fft.irfftn_ooc(comp_spec, n).reshape(-1)
    return out


def zeldovich_displacement(delta0, box_size, fdtype=np.float32):
    """Zel'dovich displacement Psi1 = (ik/k^2) delta0, shape (N^3, 3), z=0 norm.

    Same kernel conventions as the force solve (forces.k_components; the
    force == ZA identity is a permanent test). The Nyquist plane
    (ill-defined spectral gradient of a real field) is handled by the inverse
    transform's real projection; it is the only residual in the
    div Psi1 = -delta0 identity.
    """
    d0 = np.asarray(delta0, dtype=_np_dtype(fdtype))
    return _psi_from_spec(ooc_fft.rfftn_ooc(d0), d0.shape[0], box_size, fdtype)


def lpt2_source_from_spec(delta_k, n_mesh, box_size, resident="mid", workdir=None, slab=32):
    """Second-order (2LPT) source delta2 from the linear density's SPECTRUM.

    delta2(x) = sum_{i<j} [ phi,ii phi,jj - (phi,ij)^2 ] with
    phi,ij(k) = -k_i k_j / k^2 delta_k (Bouchet et al. 1995; Scoccimarro
    1998). Quadratic in delta0, scaling as amplitude^2.

    resident="mid": accumulator + at most three derivative fields resident.
    resident="low": every derivative staged to a `.npy` memmap under workdir
    (REQUIRED then) and fused back per axis-0 slab -- no derivative field is
    ever resident. Both policies execute the identical per-element op
    sequence, so they are bitwise equal (pinned by test); delta_k is left
    intact either way.
    """
    n = int(n_mesh)
    rdt = np.float64 if delta_k.dtype == np.complex128 else np.float32

    if resident == "mid":
        diag = [
            ooc_fft.irfftn_ooc(ooc_fft.deriv2_spec(delta_k, i, j, n, box_size), n)
            for i, j in _DIAG
        ]
        xx, yy, zz = diag
        a = xx * yy
        a += xx * zz
        a += yy * zz
        del xx, yy, zz, diag
        for i, j in _OFFDIAG:
            t = ooc_fft.irfftn_ooc(ooc_fft.deriv2_spec(delta_k, i, j, n, box_size), n)
            a -= t * t
        return a

    if resident != "low":
        raise ValueError(f"resident must be 'mid' or 'low', got {resident!r}")
    if workdir is None:
        raise ValueError("resident='low' stages derivatives to disk and requires workdir")

    paths = {}
    for i, j in _DIAG + _OFFDIAG:
        path = os.path.join(workdir, f"phi_{i}{j}.npy")
        mm = np.lib.format.open_memmap(path, mode="w+", dtype=rdt, shape=(n, n, n))
        spec_ij = ooc_fft.deriv2_spec(delta_k, i, j, n, box_size, slab=slab)
        for lo, s in ooc_fft.inverse_to_slabs(spec_ij, n, slab=slab):
            mm[lo : lo + s.shape[0]] = s
        mm.flush()
        del mm, spec_ij
        paths[(i, j)] = path

    mms = {ij: np.load(paths[ij], mmap_mode="r") for ij in _DIAG + _OFFDIAG}
    out = np.empty((n, n, n), dtype=rdt)
    for lo in range(0, n, slab):
        hi = min(lo + slab, n)
        xx = np.asarray(mms[(0, 0)][lo:hi])
        yy = np.asarray(mms[(1, 1)][lo:hi])
        zz = np.asarray(mms[(2, 2)][lo:hi])
        # the SAME per-element op sequence as the "mid" branch, slab-viewed
        a = xx * yy
        a += xx * zz
        a += yy * zz
        del xx, yy, zz
        for ij in _OFFDIAG:
            t = np.asarray(mms[ij][lo:hi])
            a -= t * t
        out[lo:hi] = a
    return out


def lpt2_source(delta0, box_size, fdtype=np.float32, resident="mid", workdir=None):
    """delta2 from the linear density mesh (convenience over the spec form)."""
    d0 = np.asarray(delta0, dtype=_np_dtype(fdtype))
    delta_k = ooc_fft.rfftn_ooc(d0)
    return lpt2_source_from_spec(delta_k, d0.shape[0], box_size,
                                 resident=resident, workdir=workdir)


def second_order_displacement(delta0, box_size, fdtype=np.float32, resident="mid", workdir=None):
    """2LPT displacement Psi2 = (ik/k^2) delta2, shape (N^3, 3), z=0 normalized.

    div Psi2 = -delta2 by the same kernel identity as Psi1.
    """
    d0 = np.asarray(delta0, dtype=_np_dtype(fdtype))
    n = d0.shape[0]
    delta_k = ooc_fft.rfftn_ooc(d0)
    d2 = lpt2_source_from_spec(delta_k, n, box_size, resident=resident, workdir=workdir)
    del delta_k
    return _psi_from_spec(ooc_fft.rfftn_ooc(d2), n, box_size, fdtype)


def divergence(psi, box_size):
    """FFT divergence of an (N^3, 3) vector field; returns a real (N,N,N) mesh.

    Diagnostic (dev-scale, test oracle): div Psi1 == -delta0 away from the
    Nyquist plane. Plain numpy k arrays; not a streaming path.
    """
    n3 = psi.shape[0]
    n = round(n3 ** (1.0 / 3.0))
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=box_size / n)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=box_size / n)
    comps = (kx.reshape(n, 1, 1), kx.reshape(1, n, 1), kz.reshape(1, 1, -1))
    div_k = None
    for ax in range(3):
        term = ooc_fft.rfftn_ooc(np.ascontiguousarray(psi[:, ax].reshape(n, n, n)))
        term *= 1j * comps[ax]
        div_k = term if div_k is None else div_k + term
    return ooc_fft.irfftn_ooc(div_k, n)


def lagrangian_grid(n_mesh, box_size, fdtype=np.float32):
    """Unperturbed particle positions (one per cell), shape (N^3, 3)."""
    n = int(n_mesh)
    d = box_size / n
    coords = np.arange(n, dtype=_np_dtype(fdtype)) * _np_dtype(fdtype).type(d)
    qx, qy, qz = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.stack([qx.reshape(-1), qy.reshape(-1), qz.reshape(-1)], axis=1)


def za_ics(delta0, box_size, a_init, cosmo, fdtype=np.float32, D_of_a=None):
    """ZA state at a_init: x = wrap(q + D_i Psi1), v = dx/dD = Psi1 (D-time).

    The D-time velocity of a ZA mode is Psi1, CONSTANT in D. Returns
    (x_phys, v) each (N^3, 3). D_of_a: optional growth override (e.g.
    lambda a: a for EdS pin tests).
    """
    N, L = delta0.shape[0], box_size
    D_i = D_of_a(a_init) if D_of_a is not None else growth_factor_a(a_init, cosmo)
    psi = zeldovich_displacement(delta0, L, fdtype)
    q = lagrangian_grid(N, L, fdtype)
    dt = _np_dtype(fdtype)
    x = np.mod(q + dt.type(D_i) * psi, dt.type(L))
    return x, psi


def lpt_ics(delta0, box_size, a_init, cosmo, order=2, fdtype=np.float32,
            resident="mid", workdir=None):
    """LPT state at a_init in inexor conventions: positions + D-time velocity.

    order=1: x = wrap(q + D1 Psi1),          v_D = Psi1
    order=2: x = wrap(q + D1 Psi1 - D2 Psi2), v_D = Psi1 - (D2 f2)/(D1 f1) Psi2
    (module-docstring sign convention; the v_D coefficient reduces to
    +(6/7) D1 with f2 = 2 f1 and D2 = -(3/7) D1^2). Returns (x_phys, v_D),
    each (N^3, 3) fdtype -- the MONOLITHIC dev-scale convenience; the
    streamed generator composes the same primitives per slab and never
    builds these arrays. mbody's a-time momentum is p = G_f(a_i) * v_D with
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
    psi2 = second_order_displacement(delta0, L, fdtype, resident=resident, workdir=workdir)
    q = lagrangian_grid(N, L, fdtype)
    dt = _np_dtype(fdtype)
    # THE canonical combine sequence, shared per element with the streamed
    # generator (icgen stages U and V per slab with these exact ops): the
    # displacement is formed FIRST, then added to q -- (q + D1 psi1) - D2 psi2
    # associates differently and would break streamed == monolithic bitwise.
    u = dt.type(D1) * psi1
    u -= dt.type(D2) * psi2
    x = np.mod(q + u, dt.type(L))
    v_coef2 = -(D2 * f2) / (D1 * f1)  # == +(6/7) D1 for EdS-approx D2, f2
    v = psi1 + dt.type(v_coef2) * psi2
    return x, v
