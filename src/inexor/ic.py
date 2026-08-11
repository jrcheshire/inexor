"""Initial-condition fields: Gaussian linear density + local f_NL in the
potential (architecture.md Sec. 7; mbody ic.py port to jnp/PRNGKey).

Local non-Gaussianity is defined in the primordial potential:

    phi(x) = phi_G(x) + f_NL [phi_G(x)^2 - <phi_G^2>],

tied to the density by the Poisson/transfer relation delta(k) = M(k, z) phi(k)
with M = (2/3) (c/H0)^2 k^2 T(k) D_md(z) / Omega_m and D_md the growth
normalized to D = a in matter domination. delta_G is generated
sigma8-normalized, divided by M to get the physical phi_G (~1e-5 COBE scale
-- a normalization check), transformed, and multiplied back. At f_NL = 0 the
round trip recovers the Gaussian field to FFT round-off, and f_NL enters as
one multiplicative term, so everything is (linearly) differentiable in f_NL.

The IC encode (ste_round boundary) does NOT live here -- ic/lpt are pure
float producers; integrate.evolve owns the quantization boundary.

M1 scope note: poisson_M uses the eh98 transfer only (the "table" P(k)
backend carries no T(k)); f_NL-with-CAMB-transfer is out of M1 scope.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .cosmology import growth_factor_md, linear_power, transfer_eh98

# c / H0 in Mpc/h: c = 299792.458 km/s, H0 = 100 h km/s/Mpc, and the h cancels
# when lengths are measured in Mpc/h, so this is just c[km/s] / 100.
C_OVER_H0 = 299792.458 / 100.0

# The noise-stream identity (M-v2-5, D-v2-15 clause 5). A seed denotes a
# realization only relative to a stream; cards record this constant so a
# readout can refuse to pool measurements across streams. Bump it if the
# construction below ever changes in any bit-visible way.
IC_STREAM = "m5-foldin-1"


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


def gaussian_delta(
    key, n_mesh, box_size, cosmo, fdtype=jnp.float32, amplitude=1.0, backend="eh98", table=None
):
    """Seeded z=0 linear density on the mesh: white noise coloured by P(k).

    Convention: delta_k = rfftn(white) * sqrt(P(|k|) * N^3 / L^3), so the
    measured P(k) of the returned field matches cosmology.linear_power
    (permanent test). amplitude is a differentiable sigma8/A_s proxy.
    """
    N, L = n_mesh, box_size
    white = jax.random.normal(key, (N, N, N), dtype=fdtype)
    dk = jnp.fft.rfftn(white)
    # |k| grid + sqrt(P) colour, host f64 then cast (precision island).
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)
    kk = np.sqrt(kx.reshape(N, 1, 1) ** 2 + kx.reshape(1, N, 1) ** 2 + kz.reshape(1, 1, -1) ** 2)
    # DC-safe |k| for the colour evaluation: the table backend refuses k
    # outside its range (incl. k = 0), and the DC colour is overwritten below.
    kk_safe = kk.copy()
    kk_safe[0, 0, 0] = kk.flat[1]
    colour = np.sqrt(
        linear_power(kk_safe.ravel(), cosmo, backend=backend, table=table).reshape(kk.shape)
        * N**3
        / L**3
    )
    colour[0, 0, 0] = 0.0  # zero the mean mode
    npdt = np.float64 if fdtype == jnp.float64 else np.float32
    dk = dk * jnp.asarray(colour.astype(npdt))
    return amplitude * jnp.fft.irfftn(dk, s=(N, N, N))


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

    The k = 0 entry is 1 as a safe placeholder (callers keep the density DC
    mode at zero, so it never matters).
    """
    N, L = n_mesh, box_size
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)
    k_mag = np.sqrt(kx.reshape(N, 1, 1) ** 2 + kx.reshape(1, N, 1) ** 2 + kz.reshape(1, 1, -1) ** 2)
    M = poisson_M(k_mag, cosmo, z=z)
    return np.where(k_mag > 0, M, 1.0)


def primordial_potential(key, n_mesh, box_size, cosmo, fdtype=jnp.float32):
    """Gaussian primordial potential phi_G(x), with delta_G = M phi_G.

    Returns a real (N,N,N) field at the ~1e-5 scale of the physical primordial
    potential -- a sanity check that M is normalized correctly.
    """
    N = n_mesh
    delta_G = gaussian_delta(key, N, box_size, cosmo, fdtype)
    npdt = np.float64 if fdtype == jnp.float64 else np.float32
    M = jnp.asarray(poisson_factor(N, box_size, cosmo).astype(npdt))
    return jnp.fft.irfftn(jnp.fft.rfftn(delta_G) / M, s=(N, N, N))


def linear_density(key, n_mesh, box_size, cosmo, f_NL=0.0, fdtype=jnp.float32):
    """Linear density with optional local primordial non-Gaussianity.

    delta_G -> phi_G = delta_G/M -> phi = phi_G + f_NL (phi_G^2 - <phi_G^2>) ->
    delta = M phi. At f_NL = 0 equals gaussian_delta to FFT round-off.
    Differentiable in f_NL (pass a jnp scalar); <phi_G^2> is constant in f_NL.
    Overall amplitude scaling belongs OUTSIDE this function (mbody convention:
    amplitude * linear_density(...)), keeping the NG transform normalization-
    independent of the amplitude proxy.
    """
    N = n_mesh
    delta_G = gaussian_delta(key, N, box_size, cosmo, fdtype)
    npdt = np.float64 if fdtype == jnp.float64 else np.float32
    M = jnp.asarray(poisson_factor(N, box_size, cosmo).astype(npdt))
    phi_G = jnp.fft.irfftn(jnp.fft.rfftn(delta_G) / M, s=(N, N, N))
    mean_phi2 = jnp.mean(phi_G**2)  # constant in f_NL (fdtype mean; phi ~ 1e-5)
    phi_NG = phi_G + f_NL * (phi_G**2 - mean_phi2)
    return jnp.fft.irfftn(jnp.fft.rfftn(phi_NG) * M, s=(N, N, N))


def local_bispectrum_template(triangles, cosmo, f_NL, z=0.0):
    """Tree-level local-f_NL density bispectrum at the given triangles.

    B(k1,k2,k3) = 2 f_NL [ M3/(M1 M2) P1 P2 + M2/(M1 M3) P1 P3
                                            + M1/(M2 M3) P2 P3 ],
    the exact tree prediction for linear_density's field (mbody ic.py port;
    the test oracle for the M4-class bispectrum estimator). triangles is a
    sequence of (k1, k2, k3) in h/Mpc; returns float64, one entry each.
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
