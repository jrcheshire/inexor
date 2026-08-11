"""Initial-condition fields: Gaussian linear density + local f_NL in the
potential (mbody ic.py lineage; rebuilt in place at M-v2-5 on the plane-keyed
noise stream, the 1D |k| table and the out-of-core FFT layer -- D-v2-15
clauses 2/4/5, replace-in-place JC-ratified 2026-08-10).

Local non-Gaussianity is defined in the primordial potential:

    phi(x) = phi_G(x) + f_NL [phi_G(x)^2 - <phi_G^2>],

tied to the density by the Poisson/transfer relation delta(k) = M(k, z) phi(k)
with M = (2/3) (c/H0)^2 k^2 T(k) D_md(z) / Omega_m and D_md the growth
normalized to D = a in matter domination. delta_G is generated
sigma8-normalized, divided by M to get the physical phi_G (~1e-5 COBE scale
-- a normalization check), transformed, and multiplied back. At f_NL = 0 the
round trip recovers the Gaussian field to FFT round-off, and f_NL enters as
one multiplicative term.

Everything here is HOST NUMPY through `ooc_fft`'s canonical factorization, so
the monolithic conveniences below are the streamed generator at slab = N --
one field per (seed, N), bitwise, whichever path produced it. Two v1-era
contracts retired with that (JC, 2026-08-10): jax.grad through linear_density
(the v2 engine never differentiates ICs; `colour_white` is the seam a jnp
twin would be built behind IF differentiable ICs are ever needed, gated
then), and the pre-M-v2-5 `jax.random.normal(key, (N,N,N))` stream (a seed
now denotes a DIFFERENT realization -- IC_STREAM is the identity cards carry,
and readouts refuse to pool across it).

The IC encode (ste_round boundary) does NOT live here -- ic/lpt are pure
float producers.

M1 scope note, still standing: the transfer is eh98-sourced even when P
comes from a CAMB dump (the table backend carries no T(k);
f_NL-with-CAMB-transfer is out of scope).
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import ooc_fft
from .cosmology import growth_factor_md, ic_k_table, linear_power, transfer_eh98

# c / H0 in Mpc/h: c = 299792.458 km/s, H0 = 100 h km/s/Mpc, and the h cancels
# when lengths are measured in Mpc/h, so this is just c[km/s] / 100.
C_OVER_H0 = 299792.458 / 100.0

# The noise-stream identity (M-v2-5, D-v2-15 clause 5). A seed denotes a
# realization only relative to a stream; cards record this constant so a
# readout can refuse to pool measurements across streams. Bump it if the
# construction below ever changes in any bit-visible way.
IC_STREAM = "m5-foldin-1"

# poisson_factor materializes a full (N, N, N//2+1) float64 grid -- exactly
# the object D-v2-15 clause 2 retires from the production path. It survives
# as a small-n diagnostic helper (local_bispectrum_binned's oracle) behind
# this loud ceiling.
_POISSON_FACTOR_MAX_N = 512


# ============================================================================
# Plane-keyed white noise (M-v2-5; D-v2-15 clause 5)
#
# The canonical noise unit is ONE plane along array axis 0 (the C-order slab
# axis shared by the out-of-core FFT and the state layer's brick slabs; axis
# reading JC-ratified 2026-08-10), keyed by `jax.random.fold_in(key, i)`.
# Each plane's bits depend only on (base key, plane index, (N, N), dtype), so
# ANY slab decomposition assembles the identical field -- invariance to slab
# thickness is a property of the construction, not of the code path, and the
# M-v2-5 gate tests the theorem. Deliberately a Python loop of per-plane
# draws, never a vmap over folded keys: the per-plane stream is the one
# construction in play.
#
# NOT bit-identical to the pre-M-v2-5 monolithic `jax.random.normal(key,
# (N,N,N))` stream at any seed (clause 5: the stream is shape-dependent), and
# not resolution-independent: fixed-phase cross-resolution comparison still
# requires equal N.
# ============================================================================


def _require_stream_config(fdtype):
    """Refuse silent stream or dtype drift, never degrade.

    `fold_in`'s derived keys depend on `jax_threefry_partitionable` (True on
    the pinned jax); a run under the other setting would be a DIFFERENT stream
    carrying the same IC_STREAM tag, so it is refused rather than recorded.
    An f64 request without x64 would silently come back f32 -- the engine's
    `_refuse_f64_without_x64` logic, applied at the generator boundary.
    """
    if not jax.config.jax_threefry_partitionable:
        raise RuntimeError(
            "jax_threefry_partitionable is False; the m5-foldin stream is defined "
            "under the partitionable PRNG (jax 0.10 default) and refuses to run "
            "under any other setting"
        )
    if np.dtype(fdtype) == np.float64 and not jax.config.jax_enable_x64:
        raise RuntimeError(
            "float64 white noise requested without jax_enable_x64: jax would "
            "silently return float32, and an 'f64 reference' would not be one"
        )


def plane_key(key, i):
    """The canonical per-plane key: fold_in(base key, plane index)."""
    return jax.random.fold_in(key, i)


def white_plane(key, i, n_mesh, fdtype=np.float32):
    """One (N, N) unit-normal plane of the canonical stream, host numpy."""
    _require_stream_config(fdtype)
    jdt = jnp.dtype(np.dtype(fdtype))
    return np.asarray(jax.random.normal(plane_key(key, i), (n_mesh, n_mesh), dtype=jdt))


def white_slab(key, lo, hi, n_mesh, fdtype=np.float32):
    """Planes lo..hi-1 stacked along axis 0, (hi-lo, N, N) host numpy.

    Random access by construction: generating planes [lo, hi) never touches
    any other plane, and the result is bitwise the same rows of a full build.
    """
    if not (0 <= lo <= hi <= n_mesh):
        raise ValueError(f"plane range [{lo}, {hi}) outside [0, {n_mesh})")
    out = np.empty((hi - lo, n_mesh, n_mesh), dtype=np.dtype(fdtype))
    for i in range(lo, hi):
        out[i - lo] = white_plane(key, i, n_mesh, fdtype)
    return out


def white_noise(key, n_mesh, fdtype=np.float32):
    """The full (N, N, N) white field: white_slab over every plane."""
    return white_slab(key, 0, n_mesh, n_mesh, fdtype)


# ============================================================================
# Colour and the Poisson/transfer factor
# ============================================================================


def _colour_fn(table, n_mesh, box_size):
    """sqrt(P(|k|) * N^3 / L^3) from the 1D table -- the numpy-convention
    colour whose square recovers linear_power in the measured P(k)."""

    def f(kk):
        return np.sqrt(table.P_of_k(kk) * n_mesh**3 / box_size**3)

    return f


def _poisson_fn(cosmo, table, z=0.0, inverse=False):
    """M(k, z) (or its reciprocal) as a radial multiplier, table-interpolated."""

    def f(kk):
        M = poisson_M(kk, cosmo, z=z, table=table)
        return 1.0 / M if inverse else M

    return f


def colour_white(white, box_size, cosmo, amplitude=1.0, backend="eh98", table=None):
    """delta from a GIVEN white-noise field -- the noise-injection seam.

    This is `gaussian_delta` minus the draw, exposed so probes that need
    matched or manipulated phases (G5b's truncation ladder and friends) call
    the package instead of mirroring it -- a mirror's license dies the day the
    implementation moves, which is exactly what happened at M-v2-5.

    Convention: delta_k = rfft(white) * sqrt(P(|k|) * N^3 / L^3), DC zeroed,
    so the measured P(k) of the returned field matches cosmology.linear_power
    (permanent test). amplitude is a sigma8/A_s proxy. All host numpy through
    the canonical ooc_fft factorization; output dtype follows the white field.
    """
    n_mesh = white.shape[0]
    tab = ic_k_table(cosmo, n_mesh, box_size, backend=backend, table=table)
    spec = ooc_fft.rfftn_ooc(white)
    ooc_fft.mul_radial_inplace(spec, n_mesh, box_size, _colour_fn(tab, n_mesh, box_size),
                               dc_value=0.0)
    out = ooc_fft.irfftn_ooc(spec, n_mesh)
    if amplitude != 1.0:
        out *= np.asarray(amplitude, dtype=out.dtype)
    return out


def gaussian_delta(
    key, n_mesh, box_size, cosmo, fdtype=np.float32, amplitude=1.0, backend="eh98", table=None
):
    """Seeded z=0 linear density on the mesh: plane-keyed white noise coloured
    by P(k) from the 1D table. Monolithic convenience -- the streamed generator
    at slab = N, bitwise (the M-v2-5 invariance gate)."""
    return colour_white(
        white_noise(key, n_mesh, fdtype), box_size, cosmo,
        amplitude=amplitude, backend=backend, table=table,
    )


def poisson_M(k, cosmo, z=0.0, table=None):
    """M(k, z) = (2/3) (c/H0)^2 k^2 T(k) D_md(z) / Omega_m, for arbitrary k.

    The Poisson/transfer factor relating potential and density,
    delta_lin(k, z) = M(k, z) phi(k). Accepts scalar or array k (h/Mpc),
    preserves shape; k = 0 maps to a safe transfer placeholder (M -> 0 there
    via the k^2 anyway). Host float64 (precision island).

    table: an ICKTable (D-v2-15 clause 2). With one, T comes from the 1D
    interpolated table -- O(len(k)) with no half-grid transfer evaluation --
    and the k = 0 placeholder is the table's own smallest node (in range by
    construction; the value never matters, k^2 zeroes it). table=None keeps
    the analytic transfer for scalar/diagnostic callers.
    """
    k = np.asarray(k, dtype=np.float64)
    if table is not None:
        k_safe = np.where(k > 0, k, table.k[0])
        T = table.T_of_k(k_safe.ravel()).reshape(k.shape)
    else:
        k_safe = np.where(k > 0, k, 1.0)
        T = transfer_eh98(k_safe.ravel(), cosmo).reshape(k.shape)
    a = 1.0 / (1.0 + z)
    D_md = growth_factor_md(a, cosmo)
    return (2.0 / 3.0) * C_OVER_H0**2 * k**2 * T * D_md / cosmo.Omega_m


def poisson_factor(n_mesh, box_size, cosmo, z=0.0):
    """M(k, z) on the rfftn half-grid, float64 numpy (N, N, N//2+1).

    SMALL-N DIAGNOSTIC ONLY (the bin-averaged bispectrum oracle's grid): it
    materializes the full half-grid f64 array D-v2-15 clause 2 retired from
    the production path, so it refuses above n_mesh = 512 rather than quietly
    costing gigabytes. The k = 0 entry is 1 as a safe placeholder (callers
    keep the density DC mode at zero, so it never matters).
    """
    if n_mesh > _POISSON_FACTOR_MAX_N:
        half_gb = n_mesh * n_mesh * (n_mesh // 2 + 1) * 8 / 1e9
        raise MemoryError(
            f"poisson_factor at n_mesh={n_mesh} would materialize a {half_gb:.1f} GB "
            f"float64 half-grid; it is a small-n diagnostic (ceiling "
            f"{_POISSON_FACTOR_MAX_N}). Production paths use poisson_M(table=) "
            "through ooc_fft.mul_radial_inplace."
        )
    N, L = n_mesh, box_size
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)
    k_mag = np.sqrt(kx.reshape(N, 1, 1) ** 2 + kx.reshape(1, N, 1) ** 2 + kz.reshape(1, 1, -1) ** 2)
    M = poisson_M(k_mag, cosmo, z=z)
    return np.where(k_mag > 0, M, 1.0)


# ============================================================================
# The f_NL fields
# ============================================================================


def sq_sum_by_plane(field, tot=0.0):
    """Sum of field^2: per-plane f64 sums folded into ONE running scalar.

    THE canonical reduction for any global moment a streamed generator must
    reproduce. Each plane's sum is numpy's pairwise tree over a fixed (N, N)
    shape; the plane sums then fold LEFT, one at a time, into a single running
    f64. A streamed caller THREADS the running total through its slabs
    (`tot = sq_sum_by_plane(slab, tot)`), which replays the identical sequence
    of scalar additions whatever the slab grouping -- summing each slab
    separately and adding subtotals would re-associate and move last bits
    (measured: a 16-plane grouping differs from the monolithic fold at 1e-16
    relative), exactly the drift the invariance gate exists to catch. Divide
    once at the end; never form per-slab means.
    """
    for i in range(field.shape[0]):
        p = field[i]
        if p.dtype != np.float64:
            p = p.astype(np.float64)
        tot += float(np.sum(p * p))
    return tot


def mean_sq_by_plane(field):
    """<field^2> via the canonical plane-ordered reduction, divided once."""
    return sq_sum_by_plane(field) / field.size


def primordial_potential(key, n_mesh, box_size, cosmo, fdtype=np.float32):
    """Gaussian primordial potential phi_G(x), with delta_G = M phi_G.

    Returns a real (N,N,N) field at the ~1e-5 scale of the physical primordial
    potential -- a sanity check that M is normalized correctly.
    """
    N, L = n_mesh, box_size
    tab = ic_k_table(cosmo, N, L)
    spec = ooc_fft.rfftn_ooc(white_noise(key, N, fdtype))
    ooc_fft.mul_radial_inplace(spec, N, L, _colour_fn(tab, N, L), dc_value=0.0)
    ooc_fft.mul_radial_inplace(spec, N, L, _poisson_fn(cosmo, tab, inverse=True), dc_value=1.0)
    return ooc_fft.irfftn_ooc(spec, N)


def linear_density(key, n_mesh, box_size, cosmo, f_NL=0.0, fdtype=np.float32):
    """Linear density with optional local primordial non-Gaussianity.

    delta_G -> phi_G = delta_G/M -> phi = phi_G + f_NL (phi_G^2 - <phi_G^2>) ->
    delta = M phi. At f_NL = 0 equals gaussian_delta to FFT round-off, and
    f_NL enters as one multiplicative term (linearity is a permanent test).
    <phi_G^2> uses the plane-ordered canonical reduction, so the field is
    decomposition-invariant at every f_NL. Overall amplitude scaling belongs
    OUTSIDE this function (mbody convention: amplitude * linear_density(...)).
    """
    N, L = n_mesh, box_size
    tab = ic_k_table(cosmo, N, L)
    spec = ooc_fft.rfftn_ooc(white_noise(key, N, fdtype))
    ooc_fft.mul_radial_inplace(spec, N, L, _colour_fn(tab, N, L), dc_value=0.0)
    ooc_fft.mul_radial_inplace(spec, N, L, _poisson_fn(cosmo, tab, inverse=True), dc_value=1.0)
    phi_G = ooc_fft.irfftn_ooc(spec, N)
    mean_phi2 = mean_sq_by_plane(phi_G)
    phi_NG = phi_G + np.asarray(f_NL, dtype=phi_G.dtype) * (
        phi_G * phi_G - np.asarray(mean_phi2, dtype=phi_G.dtype)
    )
    spec = ooc_fft.rfftn_ooc(phi_NG)
    ooc_fft.mul_radial_inplace(spec, N, L, _poisson_fn(cosmo, tab), dc_value=1.0)
    return ooc_fft.irfftn_ooc(spec, N)


def local_bispectrum_template(triangles, cosmo, f_NL, z=0.0):
    """Tree-level local-f_NL density bispectrum at the given triangles.

    B(k1,k2,k3) = 2 f_NL [ M3/(M1 M2) P1 P2 + M2/(M1 M3) P1 P3
                                            + M1/(M2 M3) P2 P3 ],
    the exact tree prediction for linear_density's field (mbody ic.py port;
    the test oracle for the bispectrum estimator). triangles is a sequence of
    (k1, k2, k3) in h/Mpc; returns float64, one entry each.
    """
    tris = np.atleast_2d(np.asarray(triangles, dtype=np.float64))
    ks = np.unique(tris)
    M = dict(zip(ks, poisson_M(ks, cosmo, z=z)))
    Pk = dict(zip(ks, linear_power(ks, cosmo, z=z)))
    out = np.empty(len(tris), dtype=np.float64)
    for t, (k1, k2, k3) in enumerate(tris):
        M1, M2, M3 = M[k1], M[k2], M[k3]
        P1, P2, P3 = Pk[k1], Pk[k2], Pk[k3]
        out[t] = (
            2.0
            * f_NL
            * (M3 / (M1 * M2) * P1 * P2 + M2 / (M1 * M3) * P1 * P3 + M1 / (M2 * M3) * P2 * P3)
        )
    return out
