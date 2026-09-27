"""Initial-condition fields: Gaussian linear density plus local f_NL in the potential.

    phi(x) = phi_G(x) + f_NL [phi_G(x)^2 - <phi_G^2>],   delta(k) = M(k, z) phi(k),
    M = (2/3) (c/H0)^2 k^2 T(k) D_md(z) / Omega_m,

with D_md normalized to D = a in matter domination. delta_G is sigma8-normalized, divided by M
to get phi_G (~1e-5, a normalization check), transformed, and multiplied back; at f_NL = 0 the
round trip recovers the Gaussian field to FFT round-off.

Host numpy through `ooc_fft`'s canonical factorization, so the monolithic conveniences here are
bitwise the streamed generator at slab = N. Not differentiable (`colour_white` is the seam for a
jnp twin). The IC encode does not live here; ic/lpt are pure float producers. The transfer is
always EH98, even when P(k) comes from a CAMB table (which carries no T(k)).
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import ooc_fft
from .cosmology import growth_factor_md, ic_k_table, linear_power, transfer_eh98

# c / H0 in Mpc/h (the h cancels): c[km/s] / 100.
C_OVER_H0 = 299792.458 / 100.0

# Noise-stream identity: a seed denotes a realization only relative to a stream. Cards record it
# so readouts refuse to pool across streams; bump it on any bit-visible change below.
IC_STREAM = "m5-foldin-1"
# The same construction drawn on the GPU (`ooc_fft.noise_forward_cards`). The normal transform's
# bits are not specified across backends, so this is a different stream with its own tag.
IC_STREAM_DEVICE = "m5-foldin-1-card"

# poisson_factor materializes a full (N, N, N//2+1) float64 grid; small-n diagnostic only.
_POISSON_FACTOR_MAX_N = 512


# ============================================================================
# Plane-keyed white noise
#
# The noise unit is one plane along array axis 0 (the C-order slab axis), keyed by
# `jax.random.fold_in(key, i)`. Each plane's bits depend only on (base key, plane index, (N, N),
# dtype), so any slab decomposition assembles the identical field. A Python loop of per-plane
# draws, never a vmap over folded keys. Not resolution-independent: fixed-phase comparison across
# resolutions requires equal N.
# ============================================================================


def _require_stream_config(fdtype):
    """Refuse stream drift (`fold_in` keys depend on jax_threefry_partitionable) or f64 without x64."""
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
    """One (N, N) unit-normal plane of the canonical stream, host numpy.

    Drawn on the CPU backend explicitly, whatever jax's default device: the normal transform's
    bits are not specified across backends, so a GPU draw would be a different stream.
    Cross-machine CPU identity is checked (plane-0 fingerprint on every card), not assumed.
    """
    _require_stream_config(fdtype)
    jdt = jnp.dtype(np.dtype(fdtype))
    with jax.default_device(jax.devices("cpu")[0]):
        return np.asarray(jax.random.normal(plane_key(key, i), (n_mesh, n_mesh), dtype=jdt))


def white_slab(key, lo, hi, n_mesh, fdtype=np.float32):
    """Planes lo..hi-1 stacked along axis 0, (hi-lo, N, N) host numpy.

    Random access: bitwise the same rows as a full build, touching no other plane.
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
    """sqrt(P(|k|) N^3 / L^3) from the 1D table: numpy-convention colour recovering linear_power."""

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
    """delta from a given white-noise field: `gaussian_delta` minus the draw.

    For callers that need matched or manipulated phases. delta_k = rfft(white) *
    sqrt(P(|k|) N^3 / L^3), DC zeroed, so the measured P(k) matches cosmology.linear_power.
    amplitude is a sigma8/A_s proxy; output dtype follows the white field.
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
    """Seeded z=0 linear density: plane-keyed white noise coloured by P(k) from the 1D table.

    Monolithic convenience, bitwise the streamed generator at slab = N.
    """
    return colour_white(
        white_noise(key, n_mesh, fdtype), box_size, cosmo,
        amplitude=amplitude, backend=backend, table=table,
    )


def poisson_M(k, cosmo, z=0.0, table=None):
    """M(k, z) = (2/3) (c/H0)^2 k^2 T(k) D_md(z) / Omega_m, for arbitrary k.

    delta_lin(k, z) = M(k, z) phi(k). Scalar or array k (h/Mpc), shape preserved, host float64.
    k = 0 uses a placeholder transfer (M = 0 there via k^2). table: an ICKTable to interpolate T
    from (placeholder = its smallest node); None uses the analytic EH98 transfer.
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

    Small-n diagnostic only: refuses above n_mesh = 512 rather than materializing gigabytes.
    The k = 0 entry is a placeholder 1 (callers keep the density DC mode at zero).
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

    The canonical reduction for any global moment a streamed generator must reproduce: each
    plane is a pairwise sum over a fixed (N, N) shape, folded left into one running f64. Streamed
    callers thread the total (`tot = sq_sum_by_plane(slab, tot)`) to replay the identical
    addition sequence; adding per-slab subtotals re-associates and moves last bits. Divide once
    at the end.
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

    Returns a real (N,N,N) field at the ~1e-5 physical scale (a check that M is normalized).
    """
    N, L = n_mesh, box_size
    tab = ic_k_table(cosmo, N, L)
    spec = ooc_fft.rfftn_ooc(white_noise(key, N, fdtype))
    ooc_fft.mul_radial_inplace(spec, N, L, _colour_fn(tab, N, L), dc_value=0.0)
    ooc_fft.mul_radial_inplace(spec, N, L, _poisson_fn(cosmo, tab, inverse=True), dc_value=1.0)
    return ooc_fft.irfftn_ooc(spec, N)


def linear_density(key, n_mesh, box_size, cosmo, f_NL=0.0, fdtype=np.float32):
    """Linear density with optional local primordial non-Gaussianity.

    delta_G -> phi_G = delta_G/M -> phi = phi_G + f_NL (phi_G^2 - <phi_G^2>) -> delta = M phi.
    At f_NL = 0 equals gaussian_delta to FFT round-off. <phi_G^2> uses the plane-ordered
    reduction, so the field is decomposition-invariant. Amplitude scaling is applied by the
    caller (amplitude * linear_density(...)).
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
    the tree prediction for linear_density's field. triangles: sequence of (k1, k2, k3) in
    h/Mpc; returns float64, one entry each.
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
