"""Geometric PM force solve (architecture.md Sec. 6; mbody forces.py port).

Geometry/cosmology split: this module solves the DIMENSIONLESS Poisson
equation div g = -delta and returns g = -grad phi; ALL cosmology prefactors
live in the integrator coefficients (integrate.py). The force kernel
ik/k^2 is literally the Zel'dovich displacement kernel -- k_components is
the ONE shared implementation (lpt.py imports it), and force == ZA identity
at machine precision is a permanent test.

Sequential per-component solves: never materialize three force meshes at
once (the architecture Sec. 9 memory budget depends on it).

## The two-level split (M-v2-2, promoted from `scripts/v2_g5_core.py`)

Long force S(k)*ik/k^2 on the global COARSE mesh; short force (1-S(k))*ik/k^2
on FINE tiles, each tile+buffer FFT'd as a small periodic box.
S(k) = exp(-k^2 r_s^2). BOTH LEVELS ARE MESH-SOLVED: this is PM-PM, not P3M and
not TreePM -- the erfc form appears only as the analytic prediction used to
validate buffer scaling, never as a pair sum. The decomposition is PMFAST's;
only the matching moves from a real-space polynomial to a Fourier kernel.
**Never write "TreePM/Ewald"** for this.

Two structural properties carry the whole error argument, and both are reasons
not to "clean up" the expressions below:

  1. `short := 1.0 - S`, sharing ONE k2_safe with the long kernel, so
     `long + short == ik/k^2 == the monolithic kernel` BIT-EXACTLY, by
     construction rather than by exp/expm1 agreeing to the last ulp. Every
     nonzero number the gates report is therefore a discretization error with
     exactly one named cause, never a kernel-design error. (Floor F1.)
  2. Both kernels carry ik_j, which is EXACTLY zero at k=0, so the short force
     is invariant to the density's DC level and a tile never subtracts a mean.
     Measured, not assumed: shifting delta by c = +1/-1/+137 moves the short
     force by 8.6e-16 / 7.0e-16 / 1.1e-13.

     Read that precisely, because the stronger version is FALSE and the same
     measurement shows it. A tile still needs the global mean as a NORMALIZATION
     SCALAR: delta = counts/mean - 1 is linear in 1/mean, so a tile using its
     OWN mean does not leak a small DC error, it RESCALES the whole short force
     by mean_global/mean_tile -- an error of exactly |s-1|, O(1), which reads as
     a catastrophic tiling failure and sends you hunting buffers. It is benign
     only because the global mean is n_total/n_mesh^3, a CONFIG CONSTANT rather
     than a data-dependent reduction, so nothing is communicated.

Parameterization is dimensionless, one knob per error term, so C-dev results
transfer to C-gh/C-hero where cells do not:

    alpha = r_s / d_coarse   -> coarse representation error, exp(-pi^2 alpha^2)
    beta  = b   / r_s        -> buffer truncation error, erfc(beta/2)
    b_fine = 4 * alpha * beta                    (coarse:fine ratio is 4)

**The probe remains the oracle.** D-v2-10, D-v2-11 and D-v2-12 are measurements
OF `scripts/v2_g5_core.py`, so D-v2-16 clause 7 gates this promotion on bitwise
parity against it kept UNMODIFIED. If these expressions stop being numerically
identical, three ratified records quietly stop describing the shipped artifact.
That is why the forms here are transcriptions rather than rewrites.

The `compact` and `gauss_compact` families are deliberately NOT promoted: the
kernel study measured them at ~10x worse in the coarse arm (any real-space
window dumps the ringing tails into the long kernel where the coarse mesh
aliases them), D-v2-10 froze the gaussian family, and they stay in the probe as
the research record.
"""

from functools import lru_cache

import jax.numpy as jnp
import numpy as np

from .painting import (
    cic_read_vector,
    density_contrast,
    paint_f32,
    paint_tsc_f64,
    tsc_read_vector,
)

# Coarse:fine mesh ratio (PMFAST pattern; the config table's `coarse = fine/4`).
COARSE_RATIO = 4


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


# ===========================================================================
# the split kernel
# ===========================================================================


def kernel_grids(shape, cell, fdtype=np.float64):
    """ik_j (low-rank) + k2_true + k2_safe on the rfftn half-grid of a box of
    `shape` cells of size `cell`.

    `k_components` twin. That one takes a SCALAR n_mesh (cubic box, one kx
    reused for x and y) and cannot express a tile, whose padded box is non-cubic
    in general. Same convention otherwise: host-numpy build (precision island),
    low-rank ik_j of shape (nx,1,1)/(1,ny,1)/(1,1,nz//2+1) so they broadcast
    without materializing a full array.

    Returns (ikx, iky, ikz, k2_true, k2_safe). k2_true carries a genuine 0 at
    the DC mode; k2_safe has k2[0,0,0] = 1 to avoid 0/0. BOTH are returned
    because S must be built from k2_TRUE -- exp(-k2_safe * r_s^2) would give
    S(0) = exp(-r_s^2) != 1 and silently break the split at DC.
    """
    nx, ny, nz = (int(s) for s in shape)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=cell)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=cell)
    kz = 2.0 * np.pi * np.fft.rfftfreq(nz, d=cell)
    KX = kx.reshape(nx, 1, 1)
    KY = ky.reshape(1, ny, 1)
    KZ = kz.reshape(1, 1, nz // 2 + 1)
    k2_true = (KX**2 + KY**2 + KZ**2).astype(np.float64)
    k2_safe = k2_true.copy()
    k2_safe[0, 0, 0] = 1.0
    cdtype = np.complex128 if fdtype == np.float64 else np.complex64
    return (
        (1j * KX).astype(cdtype),
        (1j * KY).astype(cdtype),
        (1j * KZ).astype(cdtype),
        k2_true,
        k2_safe.astype(fdtype),
    )


def s_of_k(k2_true, r_s):
    """S(k) = exp(-k^2 r_s^2), the Gaussian (Ewald) split, from k2_TRUE.

    Gaussian rather than a compact polynomial because both tails are analytic:
    the coarse-representation error is exp(-pi^2 alpha^2) and the buffer
    truncation is erfc(beta/2), so each is a PREDICTABLE function of one
    dimensionless knob and a measurement checks a curve rather than scanning
    blind. The honest limit, stated once: you cannot have both -- Gaussian in k
    means infinite real-space support (buffer truncation approximates); compact
    in r means an infinite k tail (worse coarse aliasing). Heisenberg, not a
    design failure. r_s = 0 -> S == 1 exactly (the F0 degenerate limit).
    """
    return np.exp(-k2_true * float(r_s) ** 2)


def split_factor(k2_true, r_s, which):
    """S (which="long"), 1-S (which="short"), or 1 (which="mono").

    short is LITERALLY 1.0 - S sharing the same S array, so long + short == 1
    identically in floating point and the F1 identity is structural. **Do not
    "optimize" this into -expm1(-k^2 r_s^2)**: that is the cancellation-safe
    form for r_s large relative to the box (at C-dev k_min^2 r_s^2 ~ 0.048, so
    f64 loses ~1.4 of 16 digits -- a non-issue there, real at C-hero), but it
    would make the identity depend on exp and expm1 agreeing to the last ulp.
    """
    if which == "mono":
        return np.ones_like(k2_true)
    s = s_of_k(k2_true, r_s)
    if which == "long":
        return s
    if which == "short":
        return 1.0 - s
    raise ValueError(f"which must be 'mono', 'long' or 'short', got {which!r}")


def assignment_window(shape, cell, order):
    """W(k) = prod_i sinc^p(k_i cell / 2) on the rfftn half-grid; p = order.

    order=2 -> CIC (painting.py's stencil), order=3 -> TSC. `diagnostics.
    cic_window` is the p=2 special case but derives the cell from n_mesh, so it
    cannot express "W(FINE cell) on the COARSE k-grid", which is what matching
    needs.
    """
    nx, ny, nz = (int(s) for s in shape)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=cell)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=cell)
    kz = 2.0 * np.pi * np.fft.rfftfreq(nz, d=cell)

    def w(k_1d):
        return np.sinc(k_1d * float(cell) / (2.0 * np.pi)) ** int(order)

    return w(kx).reshape(nx, 1, 1) * w(ky).reshape(1, ny, 1) * w(kz).reshape(1, 1, nz // 2 + 1)


def cic_match_factor(shape, cell_solve, cell_target, clip=None, order_solve=2, order_target=2):
    """(W(k, cell_target) / W(k, cell_solve))^2 on the rfftn half-grid.

    Hockney-Eastwood kernel matching. The long arm paints and gathers on the
    COARSE cell and the short arm on the FINE cell, so their CIC windows differ
    and the sum != the monolithic force EVEN WITH A MATHEMATICALLY PERFECT
    SPLIT. Paint applies W once and gather applies W once -> W^2 per level,
    hence the square.

    MANDATORY FOR THE GAUSSIAN FAMILY, HARMFUL FOR THE WINDOWED ONES -- it is
    family-specific, and an earlier version of this docstring wrongly stated it
    as universal. Measured 2026-07-15 (n_fine=128, n_coarse=32), coarse-arm
    error with vs without matching:

        gauss   alpha=1.0        9.6e-3 -> 6.1e-3   (matching HELPS, 1.6x)
        gauss   alpha=2.0        1.3e-3 -> 9.7e-4   (helps, 1.35x)
        compact r_out=4 coarse   5.8e-2 -> 2.3e-1   (matching HURTS, 4x)
        compact r_out=2 coarse   1.2e-1 -> 3.4e-1   (hurts, 2.8x)

    Why: the factor W_f^2/W_c^2 grows without bound toward the coarse Nyquist
    and is harmless only where the long kernel is ALREADY small. S(k) guarantees
    that (S ~ 5e-5 at the coarse Nyquist at alpha=1); the windowed families'
    long kernels carry real high-k content there, so matching amplifies exactly
    the modes that should be suppressed. Clipping at 10 does not rescue it.

    `clip` bounds the returned factor, because W_f^2/W_c^2 blows up near the
    coarse Nyquist (~5x on-axis, worse at cube corners). Returns
    (factor, max_applied) so the guard is reported rather than silent.
    """
    nx, ny, nz = (int(s) for s in shape)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=cell_solve)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=cell_solve)
    kz = 2.0 * np.pi * np.fft.rfftfreq(nz, d=cell_solve)

    def w_full(cell, order):
        # np.sinc(y) = sin(pi y)/(pi y); W = prod_i sinc^order(k_i cell / 2),
        # SQUARED because paint applies it once and gather applies it again.
        def w(k_1d):
            return np.sinc(k_1d * cell / (2.0 * np.pi)) ** (2 * int(order))

        return w(kx).reshape(nx, 1, 1) * w(ky).reshape(1, ny, 1) * w(kz).reshape(1, 1, nz // 2 + 1)

    ratio = w_full(cell_target, order_target) / w_full(cell_solve, order_solve)
    max_applied = float(ratio.max())
    if clip is not None:
        ratio = np.minimum(ratio, float(clip))
    return ratio, max_applied


def split_kernels(shape, cell, which, r_s=None):
    """The (Kx, Ky, Kz) half-grid kernels of the ratified gaussian split.

    The probe's `family` argument is gone: D-v2-10 froze the gaussian family and
    the windowed ones measured ~10x worse in the coarse arm, so they stay in
    `scripts/v2_g5_core.py` as the research record rather than shipping as a
    live branch nothing selects. A caller wanting them wants the probe.
    """
    ikx, iky, ikz, k2_true, k2_safe = kernel_grids(shape, cell, np.float64)
    fac = split_factor(k2_true, 0.0 if r_s is None else r_s, which)
    return tuple((fac / k2_safe) * ik for ik in (ikx, iky, ikz))


# ===========================================================================
# the global arm (one mesh solve of a chosen split kernel)
# ===========================================================================


def density_f64(positions, n_mesh, box_size, n_particles_total):
    """delta on an n_mesh^3 grid in f64, via the PACKAGE CIC stencil.

    `paint_f32` takes an fdtype even though `density_contrast` never passes one,
    so the f64 mesh costs no new paint code and the global arm differs from
    `make_force_fn` ONLY in the kernel. That is what makes the F0 floor a test
    of the kernel rather than of the harness.
    """
    counts = paint_f32(positions, n_mesh, box_size, fdtype=jnp.float64)
    mean = float(n_particles_total) / float(n_mesh) ** 3
    return counts / mean - 1.0


def force_global(
    positions,
    n_mesh,
    box_size,
    n_particles_total,
    which,
    r_s=None,
    match=None,
    clip=None,
    assign="cic",
    pos_gather=None,
):
    """One global mesh solve of the chosen split kernel -> ((n,3) f64, max_match).

    `which="mono"` reproduces `make_force_fn`'s kernel exactly (floor F0).
    `match=(cell_solve, cell_target)` applies `cic_match_factor`.

    Sequential per-component solves: never materialize three force meshes at
    once -- this module's Sec. 9 rule, which the memory budget depends on.

    `pos_gather` splits the SOURCE of the field from the point it is READ AT.
    Default None gathers at `positions`, which is the only behaviour existing
    callers see. It exists for the frozen-background arm: the long-range field
    can be built once per step from analytic LPT positions -- known for every
    particle at every time without evolving anything -- while each tile reads
    its own particles' force at their TRUE positions. That arm is SHELVED
    (D-v2-12 killed independent tiles), and the seam is kept because reviving it
    is a named V4-architecture option, not because anything calls it today.
    """
    cell = box_size / n_mesh
    kers = split_kernels((n_mesh,) * 3, cell, which, r_s=r_s)
    max_applied = 1.0
    if match is not None:
        mf, max_applied = cic_match_factor((n_mesh,) * 3, match[0], match[1], clip=clip)
        kers = tuple(k * mf for k in kers)
    if assign == "cic":
        delta = density_f64(positions, n_mesh, box_size, n_particles_total)
    elif assign == "tsc":
        delta = paint_tsc_f64(positions, n_mesh, box_size, n_particles_total)
    else:
        raise ValueError(f"assign must be 'cic' or 'tsc', got {assign!r}")
    dk = jnp.fft.rfftn(delta)
    g = [jnp.fft.irfftn(dk * jnp.asarray(k), s=(n_mesh,) * 3) for k in kers]
    rd = positions if pos_gather is None else jnp.asarray(pos_gather)
    if assign == "cic":
        out = cic_read_vector(g[0], g[1], g[2], rd, n_mesh, box_size)
    else:
        out = tsc_read_vector(g[0], g[1], g[2], rd, n_mesh, box_size)
    return np.asarray(out, dtype=np.float64), max_applied
