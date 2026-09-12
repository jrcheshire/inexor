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

import math
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from .painting import (
    _CORNERS as _CIC_CORNERS,
)
from .codec import rint_i
from .painting import (
    _TSC_CORNERS,
    CIC_CELL_WEIGHT_BOUND,
    check_int_paint_headroom,
    cic_read_vector,
    counts_from_int,
    density_contrast,
    density_tsc,
    field_dtype,
    paint_f32,
    tsc_read_vector,
)

# Coarse:fine mesh ratio (PMFAST pattern; the config table's `coarse = fine/4`).
COARSE_RATIO = 4

# Fixed-point fraction bits for the integer paints on the tiled short arm. Same
# default as `paint_int`/`paint_tsc_int`; a knob because the strict 8x CIC bound
# makes it one (see `tile_paint_int`).
TILE_FRAC_BITS = 12


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


def split_kernels(shape, cell, which, r_s=None, fdtype=np.float64):
    """The (Kx, Ky, Kz) half-grid kernels of the ratified gaussian split.

    The probe's `family` argument is gone: D-v2-10 froze the gaussian family and
    the windowed ones measured ~10x worse in the coarse arm, so they stay in
    `scripts/v2_g5_core.py` as the research record rather than shipping as a
    live branch nothing selects. A caller wanting them wants the probe.

    `fdtype` (M-v2-4) narrows the REAL PREFACTOR and nothing else. Three things
    make that the right seam rather than an arbitrary one:

      - **The build stays f64, always.** `k2_true`, `k2_safe` and S are the
        precision island. `fac / k2_safe` is where the split's accuracy lives,
        and a pre-rounded denominator would cost accuracy for no memory: these
        are transients, and what this milestone is buying is the RESIDENT
        kernel.
      - **Narrowing the prefactor is not the obvious spelling, and the obvious
        one does not work.** `(fac / k2_safe) * ik` with a narrowed `ik` is
        `f64 * complex64`, which numpy promotes back to complex128 -- an f32 arm
        that is not one. `tests/test_force_dtypes.py` pins that trap.
      - It costs one extra rounding. The prefactor form differs from narrowing
        the finished product by at most 2 f32 ulp -- measured 1.0 to 1.5 ulp
        (max rel 1.18e-7 to 1.79e-7) over four geometries from (8,12,16) to
        64^3, both splits, so `rtol=2**-22` holds and `2**-23` does not. The
        alternative materializes a full complex128 half-grid per component
        before narrowing -- 8.6 GB each at C-gh against a kept set of 12.9 GB
        for all three -- so the 2 ulp is bought deliberately.

    At the f64 default both casts are `copy=False` no-ops on arrays that already
    carry the dtype, so the expression reduces to what it was before M-v2-4 and
    the result stays BITWISE the probe (D-v2-16 clause 7).
    """
    fdtype = np.dtype(fdtype)
    if fdtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"fdtype must be float32 or float64, got {fdtype.name}")
    ikx, iky, ikz, k2_true, k2_safe = kernel_grids(shape, cell, np.float64)
    fac = split_factor(k2_true, 0.0 if r_s is None else r_s, which)
    pref = (fac / k2_safe).astype(fdtype, copy=False)
    cdtype = np.complex128 if fdtype == np.dtype(np.float64) else np.complex64
    return tuple(pref * ik.astype(cdtype, copy=False) for ik in (ikx, iky, ikz))


# ===========================================================================
# the global arm (one mesh solve of a chosen split kernel)
# ===========================================================================


def density_f64(positions, n_mesh, box_size, n_particles_total, fdtype=jnp.float64):
    """delta on an n_mesh^3 grid via the PACKAGE CIC stencil.

    `paint_f32` takes an fdtype even though `density_contrast` never passes one,
    so the f64 mesh costs no new paint code and the global arm differs from
    `make_force_fn` ONLY in the kernel. That is what makes the F0 floor a test
    of the kernel rather than of the harness.

    `fdtype` (M-v2-4) is the RETURNED field's dtype. **The paint stays pinned at
    f64** -- the `fdtype=jnp.float64` below is not a candidate for threading.
    This is the differentiable twin's accumulator, and narrowing it makes the
    scatter-add f32, which is non-associative under CUDA atomics (M0 R3) and so
    is a D-006 violation dressed as a memory saving. Narrow the field, never the
    accumulator.
    """
    fdtype = field_dtype(fdtype)
    counts = paint_f32(positions, n_mesh, box_size, fdtype=jnp.float64)
    mean = float(n_particles_total) / float(n_mesh) ** 3
    return (counts / mean - 1.0).astype(fdtype)


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
    paint="f64",
    frac_bits=TILE_FRAC_BITS,
    fdtype=np.float64,
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

    `paint` selects the coarse assignment's accumulator on the `tsc` branch:
    "f64" is the order-dependent path D-v2-10/11/12 were measured with and stays
    the DEFAULT, so every probe-parity test compares like with like; "int" is the
    D-006-compliant path (D-v2-16 cl.2), which `paint_tsc_int` has implemented
    since M-v2-2 and which NOTHING COULD REACH, because this function called
    `paint_tsc_f64` directly rather than going through `density_tsc`. The engine
    selects "int" explicitly -- `density_tsc`'s own docstring says M-v2-3 is
    where that choice gets made.

    **The `cic` branch deliberately has no such knob.** It is order-dependent
    too, and it is not on the shipping path: TSC is the ratified coarse
    assignment, and this branch serves the mono/F0 floors and the shelved arms.
    An int option there would mean either dropping its f64 counts to
    `density_contrast`'s f32 or growing a second decode, neither worth doing for
    a path the engine never takes. Recorded so the asymmetry reads as a decision
    rather than an oversight.
    """
    fdtype = field_dtype(fdtype)
    delta, kers, max_applied = _global_delta_and_kernels(
        positions, n_mesh, box_size, n_particles_total, which,
        r_s=r_s, match=match, clip=clip, assign=assign, paint=paint, frac_bits=frac_bits,
        fdtype=fdtype,
    )
    dk = jnp.fft.rfftn(delta)
    g = [jnp.fft.irfftn(dk * jnp.asarray(k), s=(n_mesh,) * 3) for k in kers]
    rd = positions if pos_gather is None else jnp.asarray(pos_gather)
    if assign == "cic":
        out = cic_read_vector(g[0], g[1], g[2], rd, n_mesh, box_size)
    else:
        out = tsc_read_vector(g[0], g[1], g[2], rd, n_mesh, box_size)
    return np.asarray(out, dtype=fdtype), max_applied


def _global_delta_and_kernels(
    positions, n_mesh, box_size, n_particles_total, which,
    r_s=None, match=None, clip=None, assign="cic", paint="f64", frac_bits=TILE_FRAC_BITS,
    fdtype=np.float64,
):
    """The paint-and-kernel half of `force_global`, split out verbatim."""
    fdtype = field_dtype(fdtype)
    cell = box_size / n_mesh
    kers = split_kernels((n_mesh,) * 3, cell, which, r_s=r_s, fdtype=fdtype)
    max_applied = 1.0
    if match is not None:
        mf, max_applied = cic_match_factor((n_mesh,) * 3, match[0], match[1], clip=clip)
        # `mf` is a host f64 half-grid and `k` may be complex64, so an uncast
        # multiply promotes the kernel back to complex128 and silently undoes
        # the narrowing -- at the MATCHED coarse arm specifically, which is the
        # one D-v2-10 ratified. Same trap as narrowing `ik` in `split_kernels`.
        kers = tuple(k * mf.astype(fdtype, copy=False) for k in kers)
    if assign == "cic":
        if paint != "f64":
            raise ValueError(
                f"assign='cic' has no {paint!r} accumulator: the int CIC path is "
                "density_contrast's, which carries f32 counts, and the coarse arm "
                "ratified in D-v2-10 is TSC. Use assign='tsc' for the engine's "
                "D-006-compliant long arm."
            )
        delta = density_f64(positions, n_mesh, box_size, n_particles_total, fdtype=fdtype)
    elif assign == "tsc":
        delta = density_tsc(
            positions, n_mesh, box_size, n_particles_total, paint=paint,
            frac_bits=frac_bits, fdtype=fdtype,
        )
    else:
        raise ValueError(f"assign must be 'cic' or 'tsc', got {assign!r}")
    return delta, kers, max_applied


def coarse_kernel_parts(n_mesh, box_size, which, r_s=None, match=None, clip=None,
                        fdtype=np.float64):
    """The REAL half-grids the coarse solve needs, built ONCE per run.

    **Everything here is a function of geometry alone** -- shape, cell, `r_s`,
    the match pair -- so nothing in it changes between steps, and until M-v2-6
    the engine rebuilt all of it every step inside `coarse_force_meshes`. The
    fine arm already learned this: `engine.run` hoists `make_tile_force_fn` to
    once per run because the per-step rebuild re-traces the same program and
    rebuilds the same kernel triple. The coarse arm simply never got the same
    treatment, and at C-gh it is the more expensive of the two by 80x.

    What it buys, MEASURED (bytes per half-grid element, flat to 0.2% over a
    7.9x range in `half` at n_coarse 128/192/256):

        as shipped before M-v2-6   60.01 B/half EVERY step
        parts + streamed solve     16.01 B/half every step, 8.00 held

    which at C-gh (`half` = 537,919,488) is **32.28 GB per step against 8.61 GB
    per step plus 4.30 GB resident**. That mattered because the coarse solve is
    where the 2048^3 pilot died: Vista 923139 lost ~79 GB of MemAvailable inside
    one phase of step 1, against a planner that priced the whole per-step mesh
    transient at 12.9 GB by charging only its largest single term.

    **Only the REAL half-grids are kept.** The complex kernels are 24 B/half and
    are formed one component at a time inside the solve, where each is freed
    before the next exists. `pref` and the match factor are 4 B/half each at
    f32, and the `ik_j` are low-rank broadcasts ((nx,1,1) etc.) that cost
    nothing -- which is the whole reason this split is worth making.

    The match factor is kept SEPARATE from `pref` rather than folded into it,
    and that is deliberate: folding gives `(pref * mf) * ik` where the shipped
    expression is `(pref * ik) * mf`, which is one reassociation and so not
    bitwise. It would save a further 4 B/half and it is not worth re-opening
    D-v2-16 clause 7 for.

    The build peak is 28 B/half (the f64 island plus `pref`) and it is paid once
    per run, before any force mesh exists.
    """
    fdtype = field_dtype(fdtype)
    shape = (int(n_mesh),) * 3
    cell = box_size / n_mesh
    ikx, iky, ikz, k2_true, k2_safe = kernel_grids(shape, cell, np.float64)
    fac = split_factor(k2_true, 0.0 if r_s is None else r_s, which)
    pref = (fac / k2_safe).astype(fdtype, copy=False)
    # the f64 island is dead the moment `pref` exists, and at C-gh it is 12.9 GB
    # of it; holding it to the end of the function would keep the build peak at
    # the whole run's peak for no reason
    del k2_true, k2_safe, fac
    mf = None
    if match is not None:
        # cast for the same reason as in `_global_delta_and_kernels`: an f64
        # match factor would promote a complex64 kernel back to complex128
        m, _ = cic_match_factor(shape, match[0], match[1], clip=clip)
        mf = m.astype(fdtype, copy=False)
        del m
    return dict(iks=(ikx, iky, ikz), pref=pref, mf=mf,
                fdtype=fdtype, n_mesh=int(n_mesh), which=which)


def coarse_force_meshes(delta, n_mesh, box_size, which, r_s=None, match=None, clip=None,
                        fdtype=None, parts=None, out=None):
    """The three long-range force meshes from an ALREADY-PAINTED delta.

    `force_global` paints from every position AND gathers at every position,
    returning an `(n, 3)` array -- both halves O(N), and the engine needs
    neither. It streams the paint brick by brick (integer accumulation, so the
    chunked sum is bitwise the monolithic one) and reads the force per tile out
    of a staged sub-block, so what it wants from the global arm is exactly this:
    solve, and stop.

    Bitwise identical to the corresponding part of `force_global` by
    construction -- it is the same expression, called from both.

    **`fdtype=None` INFERS from `delta.dtype`, and an explicit disagreement is
    a refusal rather than a cast (M-v2-4).** This is the seam where the engine's
    coarse arm gets its precision, and the one thing that must not happen here
    is a knob that reports success without applying: if a caller narrowed its
    delta but the kernels stayed f64, the multiply would promote everything back
    and the run would look correct while costing double. Refusing a mismatched
    pair makes that unrepresentable instead of merely unlikely. Casting the
    delta to match would be the friendly alternative and is exactly wrong -- it
    would hide the caller's error.

    `parts`, if given, is a `coarse_kernel_parts` dict built once per run; when
    it is None this builds one per call, which is what every caller did before
    M-v2-6. `out`, if given, is three preallocated meshes the components are
    written into -- the pool's shared-memory views, so the solve lands where the
    workers already read and the copy `TilePool.stage_step` used to make stops
    existing. Both default to the old behaviour and neither changes a value.

    **The three force meshes never coexist as jax arrays.** They used to: the
    old form returned a list comprehension of jax arrays and the engine then
    built a numpy copy of each, and because a comprehension rebinds only after
    it completes, that is six full meshes live at once -- 25.8 GB at C-gh. One
    component at a time is 8.6.
    """
    fdtype = field_dtype(delta.dtype if fdtype is None else fdtype)
    if np.dtype(delta.dtype) != fdtype:
        raise ValueError(
            f"coarse_force_meshes: delta is {np.dtype(delta.dtype).name} but fdtype is "
            f"{fdtype.name}. These must agree -- the kernels are built at fdtype and a "
            "mismatched multiply promotes the whole solve back to the wider type, which "
            "reads as a working f32 arm that is silently costing f64 memory. Narrow the "
            "delta at its decode, or pass the dtype it already has."
        )
    if parts is None:
        parts = coarse_kernel_parts(n_mesh, box_size, which, r_s=r_s, match=match,
                                    clip=clip, fdtype=fdtype)
    elif parts["n_mesh"] != int(n_mesh) or parts["fdtype"] != fdtype:
        # a cached build outliving the geometry it was built for is silent
        # corruption, not a crash: the shapes broadcast where n_mesh matches by
        # accident and the dtype merely promotes
        raise ValueError(
            f"coarse_force_meshes: parts were built for n_mesh={parts['n_mesh']} at "
            f"{parts['fdtype'].name} but this call is n_mesh={int(n_mesh)} at "
            f"{fdtype.name}. The kernel build is geometry, so a mismatch means the "
            "cache outlived the configuration it belongs to."
        )
    cdtype = np.complex128 if fdtype == np.dtype(np.float64) else np.complex64
    if out is None:
        out = [np.empty((int(n_mesh),) * 3, dtype=fdtype) for _ in range(3)]
    dk = jnp.fft.rfftn(delta)
    # SEQUENTIAL PER-COMPONENT SOLVES, and now the comment is true of the
    # outputs as well as of the transforms. `k` is formed, used and dropped
    # inside one iteration, so the peak carries one complex kernel rather than
    # three; the expression is `(pref * ik) * mf` exactly as before, which is
    # what keeps this bitwise (`tests/test_forces.py` pins it at three
    # geometries against a build-all-three reference).
    for i in range(3):
        k = parts["pref"] * parts["iks"][i].astype(cdtype, copy=False)
        if parts["mf"] is not None:
            k = k * parts["mf"]
        g = jnp.fft.irfftn(dk * jnp.asarray(k), s=(int(n_mesh),) * 3)
        del k
        out[i][...] = np.asarray(g)
        del g
    return out


def owning_tile(positions, cell, n_tile, n_fine):
    """The tile coordinates that own each row, from the GLOBAL position alone.

    **This is the single source of the ownership partition, and it has to be a
    function of the global position or it is not a partition at all.** The old
    form tested the TILE-LOCAL coordinate, `owned = all(u >= core_lo & u <
    core_hi)` with `u = mod(x - origin, L)` computed with a DIFFERENT origin per
    tile, so two neighbouring tiles decided the same particle through two
    different subtractions and their decisions were not complementary in floating
    point. Measured (M-v2-6, antares job 431, cdev): one particle of 16,777,216
    owned by NO tile, tripping the partition assertion. At cdev the core is
    [8.0, 72.0) and for a position a few ulp below a boundary `x + 8.0` can round
    UP to exactly 72.0, so the lower tile disowns it, while `x - 56.0` rounds to
    just under 8.0, so the upper tile disowns it too -- a gap about one ulp wide
    at every core plane. It is realization-dependent, which is why it had never
    fired: cdev passes on this laptop's realization and fails on antares's,
    because the noise stream is per-machine-class (D-v2-23).

    Deciding from the global position makes the partition exact BY CONSTRUCTION
    rather than exact up to rounding: `floor(x / cell)` is one deterministic
    integer per row, tiles divide the fine mesh exactly (`n_fine % n_tile == 0`),
    so integer division sends every row to exactly one tile whatever the float
    result was. A row cannot be claimed twice or dropped, which is the property
    the assertion checks and the property D-v2-16 clause 1's tile-local sink and
    M-v2-3's velocity-scale theorem both rest on.

    Host numpy and integer on purpose: this is a layout question, not a force
    question, and it does not belong inside the jitted kernel.
    """
    n_fine = int(n_fine)
    ci = np.floor(np.asarray(positions, dtype=np.float64) / float(cell))
    ci = np.mod(ci.astype(np.int64), n_fine)  # x == box_size exactly -> cell 0
    return ci // int(n_tile)


def owned_mask(positions, tijk, cell, n_tile, n_fine, live=None):
    """Rows of `positions` owned by tile `tijk`, by POSITION; see `owning_tile`.

    Correct as a partition of SPACE, and that is not the partition the engine
    needs -- see `owned_mask_from_bricks`, which is what `engine.step` uses. This
    remains for `force_short_tiled`, whose injected `member_fn` returns particle
    indices with no brick structure to derive ownership from.
    """
    own = owning_tile(positions, cell, n_tile, n_fine)
    m = np.all(own == np.asarray(tijk, dtype=np.int64), axis=1)
    return m if live is None else (m & np.asarray(live, dtype=bool))


def owned_mask_from_bricks(brick_of_row, tijk, n_tile, n_brick, nb):
    """Rows owned by tile `tijk`, decided by the brick each row is STORED IN.

    **This is the partition the engine's assertion actually counts, and deciding
    ownership any other way is a second opinion that can disagree.** The
    position-based form (`owned_mask`) partitions SPACE exactly, which is not the
    same thing: membership comes from `SlotState.tile_bricks`, i.e. from brick
    ORDINALS, so a row whose stored brick disagrees with its position by one cell
    is handed to a tile that the position test then assigns elsewhere, and no tile
    claims it. Measured: antares job 436 at cdev, one row of 16,777,216 unowned
    with the position-based rule -- the same symptom as the tile-local float rule
    it replaced, and a different cause. The ulp-gap reading of job 431 was
    therefore incomplete: both rules fail here because both re-derive ownership
    from a coordinate instead of reading it off the layout.

    Storage-derived ownership cannot fail, and the argument is arithmetic rather
    than empirical. Bricks per tile core is `n_tile // n_brick`, the brick grid is
    `nb = n_fine // n_brick`, and `nb / (n_tile // n_brick) = n_fine // n_tile =
    tiles_side` exactly, so the brick grid partitions into tile cores with nothing
    left over: every brick belongs to exactly one core, hence every stored row to
    exactly one tile. `brick_span` pads outward with `pad >= 0`, so a tile's
    members always include its own core bricks. No float comparison is involved at
    any point.

    `brick_of_row` is the flat brick ordinal per decoded row, in the
    `(bi * nb + bj) * nb + bk` convention `tile_bricks` builds and `decode_bricks`
    concatenates in.
    """
    b = np.asarray(brick_of_row, dtype=np.int64)
    nb = int(nb)
    per = int(n_tile) // int(n_brick)
    bi, rem = np.divmod(b, nb * nb)
    bj, bk = np.divmod(rem, nb)
    t = np.asarray(tijk, dtype=np.int64)
    return (bi // per == t[0]) & (bj // per == t[1]) & (bk // per == t[2])


def tile_capacity(member_counts):
    """The per-tile row capacity: a MAX over tiles, so one jitted program serves
    every tile (a per-tile member count would key a new shape and recompile).

    The package had no source for this -- `force_short_tiled` requires `cap` as a
    required argument and only `scripts/v2_g5_core.tile_capacity` computed one,
    so nothing in the package could drive the tiled force end to end.
    """
    counts = np.asarray(list(member_counts), dtype=np.int64)
    if not len(counts):
        raise ValueError("no tiles: cap is a max over tiles and there are none")
    return int(counts.max())


# Rungs per octave for the capacity LADDER below. Three gives a worst-case pad of
# 2^(1/3) - 1 = 26% of rows and at most three distinct shapes per doubling of
# `cap`; the knob trades padded rows (which cost staging, measured LINEAR in cap)
# against retained XLA executables (which cost host memory and never come back).
CAP_RUNGS_PER_OCTAVE = 3


def capacity_shape(cap, rungs=CAP_RUNGS_PER_OCTAVE, floor_shape=0):
    """Quantize `cap` UP onto a fixed geometric ladder, monotone in both arguments.

    **This exists because `cap` moves every single step and every buffer keyed on
    it keys a new XLA shape.** Measured at cdev8 (M-v2-6 Stage 0): `cap` took ten
    distinct values over ten steps, 285,554 rising monotonically to 397,319 as
    occupancy shifts, driving ~71 XLA compilations per step and a peak host RSS
    LINEAR IN STEP COUNT at 0.331 GB/step (5.151 / 7.042 / 10.157 GB at
    K = 5 / 10 / 20, fit residuals <= 0.143 GB). Executables and their buffers are
    cached for the life of the process, so an unbounded shape family is an
    unbounded leak. At C-gh's K and P that is the term that decides whether the
    configuration runs at all.

    M-v2-3 already fixed the WITHIN-step version of this twice -- padding the
    long-range read to one shape (2,107 compilations, 24.1 s of a 32.7 s step) and
    masking the streamed coarse paint. Nothing pinned the shape ACROSS steps, and
    the general lesson was already on record: pad to a stable shape and MASK.

    The ladder is `2^(j/rungs)`, anchored globally rather than at the current
    `cap`, so the mapping depends only on `cap` and never on the history that
    reached it -- an octave-relative ladder is NOT monotone (cap 1001 -> 1250 while
    cap 1024 -> 1024) and a non-monotone shape schedule reintroduces churn.
    `floor_shape` makes it sticky across steps so an occupancy dip cannot shrink
    the shape and then regrow it, paying two compilations for no reason.

    Padding is safe because it is MASKED, not filled: the short arm passes `live`
    and the long arm `live=lv`, masked rows contribute an integer weight of exactly
    zero to the paints, per-row gathers do not mix rows, and both results are
    sliced back to the real row count. So a larger shape is BITWISE the smaller
    one -- which is the gate, not a hope.
    """
    cap = int(cap)
    floor_shape = int(floor_shape)
    if cap <= 0:
        return max(0, floor_shape)
    rungs = int(rungs)
    if rungs < 1:
        raise ValueError(f"rungs must be >= 1, got {rungs}")
    j = math.ceil(rungs * math.log2(cap))
    s = int(math.ceil(2.0 ** (j / rungs)))
    while s < cap:  # float error at a rung boundary, never more than one step
        j += 1
        s = int(math.ceil(2.0 ** (j / rungs)))
    return max(s, floor_shape)


# ===========================================================================
# tile geometry (origin-shifted; no global wrap)
# ===========================================================================

# FFT-friendly padded sizes (products of small primes). P = T + 2*b_fine is
# rounded UP to one of these: T=128, b_fine=18 -> P=164 = 4*41 has a large prime
# factor and is pathologically slow. The realized buffer is what gets reported.
#
# **Entries <= 512 are FROZEN.** They fix every already-measured tile selection
# (D-v2-10, D-v2-11, D-v2-12), and the >512 block was appended 2026-07-17 for the
# cgh64 ladder's tile512 arms, which want P = 576/640/768/1024. The list stops at
# 1024 = the largest fine mesh in the config table, and the degeneracy guard
# forbids P > n_fine anyway. Nothing may be inserted below 512: verified
# exhaustively at append time that every `want` in [1, 512] selects the same P.
FFT_FRIENDLY = (
    32, 36, 40, 48, 50, 54, 60, 64, 72, 80, 90, 96, 100, 108, 120, 128,
    144, 150, 160, 162, 180, 192, 200, 216, 240, 250, 256, 288, 300, 320,
    360, 384, 400, 432, 480, 500, 512,
    540, 576, 600, 640, 648, 720, 750, 768, 800, 810, 864, 900, 960, 972,
    1000, 1024,
)  # fmt: skip


def padded_size(n_tile, b_fine, n_fine=None):
    """FFT-friendly P >= n_tile + 2*b_fine, and the realized buffer.

    Returns (P, b_realized). The buffer GROWS to the rounded size -- the extra is
    real buffer, not padding -- so beta_realized >= beta_requested and is what
    gets reported.

    `n_fine` enables the degeneracy guard: a padded tile at least as large as the
    global mesh is not a tile at all. It does more FFT work than the monolithic
    solve it replaces, and (with the brick wrap) it is where the double-count bug
    lives.
    """
    want = int(n_tile) + 2 * int(b_fine)
    if n_fine is not None and want > int(n_fine):
        raise ValueError(
            f"padded tile {want} > fine mesh {n_fine}: the tile+buffer exceeds the box, "
            "which is degenerate (more FFT work than monolithic). Reduce beta or n_tile."
        )
    for p in FFT_FRIENDLY:
        if p >= want:
            if n_fine is not None and p > int(n_fine):
                raise ValueError(
                    f"FFT-friendly padded size {p} > fine mesh {n_fine} "
                    f"(wanted {want}); reduce beta or n_tile."
                )
            return int(p), (int(p) - int(n_tile)) // 2
    raise ValueError(f"no FFT-friendly padded size >= {want}; extend FFT_FRIENDLY")


def tile_origin_extent(tijk, n_tile, b_fine, cell):
    """(origin (3,) f64, extent f64) of a tile+buffer box in GLOBAL coords.

    origin = (t*n_tile - b_fine) * cell, and may be NEGATIVE -- deliberately, and
    it is fine: mod(x - origin, L) handles the periodic wrap exactly, so a tile
    whose buffer crosses the box boundary needs no special case.
    """
    t = np.asarray(tijk, dtype=np.int64)
    origin = (t * int(n_tile) - int(b_fine)) * float(cell)
    extent = (int(n_tile) + 2 * int(b_fine)) * float(cell)
    return origin, extent


def tile_local_coords(positions, origin, box_size):
    """u = mod(pos - origin, L): the tile-local coordinate AND the membership
    test, from ONE expression.

    A particle is in tile+buffer iff all(u < extent). Exact including buffers that
    wrap the periodic boundary -- no min-image, no branches. `painting._cic_pieces`'
    `% n_mesh` is WRONG here: it would fold a particle from the far side of the
    box into the tile.
    """
    return jnp.mod(positions - jnp.asarray(origin), float(box_size))


def _tile_cic_pieces(u, live, shape, cell):
    """Base cell, fractional offset, validity mask, and the out-of-box count.

    THE MODULO QUESTION, settled by measurement (2026-07-15). The padded tile box
    IS periodic -- that is exactly what its rfftn assumes -- so the tile paint
    wraps modulo P, the PADDED BOX's own period. What would be wrong is
    `painting.py`'s `% n_mesh`, the GLOBAL box's period, which folds far-side
    particles in. Those are different moduli, and an earlier version conflated
    "not the global modulo" with "no modulo": it required base < P-1, silently
    discarding the LAST CELL LAYER of every padded box. That produced a 4.6e-1
    error on the one-tile-equals-whole-box identity and a fake buffer-error
    plateau that looked exactly like kernel ringing. The identity check caught it.

    `ok` selects particles inside the padded box (u in [0, extent)) and the corner
    indices wrap modulo P. `n_out` counts LIVE particles outside it.

    **n_out is a CONTRACT, not a diagnostic** -- corrected 2026-08-08, when
    promotion found this docstring and `tile_paint_f64`'s asserting opposite
    things. It was a diagnostic when the brick union was merely a superset of
    tile+buffer, and healthy overhang was expected. Since `choose_brick` gained
    the `c | b_fine` condition (2026-08-07) the union is EXACTLY the padded box,
    so any nonzero value means the brick decomposition is wrong rather than
    wasteful. It stays a returned number rather than a raise because a caller may
    hand-pick a brick that violates the condition; `layout.assert_brick_divides_
    buffer` is the check that forbids that, and the gates assert n_out == 0.
    """
    nx, ny, nz = (int(s) for s in shape)
    extent = jnp.asarray([nx, ny, nz], dtype=jnp.float64) * float(cell)
    xp = u / float(cell)
    base_f = jnp.floor(xp)
    frac = xp - base_f
    base = jax.lax.stop_gradient(base_f).astype(jnp.int32)
    in_box = jnp.all((u >= 0.0) & (u < extent), axis=1)
    ok = live & in_box
    n_out = jnp.sum(live & (~in_box))
    return base, frac, ok, n_out


def _tile_corner(base, frac, wlo, corner, shape, ok):
    """One CIC corner: flat index (wrapped mod P) + weight, no-op'd on ~ok.

    Out-of-box and padding slots get a VALID index with ZERO weight, so
    mode="promise_in_bounds" is honestly safe. Out-of-range indices under that
    mode are undefined behaviour on GPU -- silent corruption, not an exception
    (G1 job 33: 63% of cells wrong from a lowering that "worked"). mode="drop" is
    REJECTED: it would silently swallow a genuinely misrouted live particle,
    exactly the bug class the gates must be able to see.

    WHICH LINE MAKES THE INDEX SAFE. It is the `% nx` below, NOT a `where` on
    `ok`. `base` comes from floor(u/cell) with u = mod(pos - origin, L) in [0, L),
    so base is bounded by n_fine and cannot overflow int32, and jnp's `%` with a
    positive modulus is non-negative. `flat` is therefore in range for EVERY row,
    live or not, and is returned unmasked. The earlier `where(ok, flat, 0)` was
    redundant for safety and expensive for a reason nobody costed: it funnelled
    every padded row -- 33% of all rows at the C-gh candidate geometry -- onto
    flat index 0, so each of the 8 unrolled scatter-adds became ~2e6 f64 atomics
    contending for ONE address. Removing it took the device phase 47.15 -> 21.26
    ms, 2.218x. Only the WEIGHT zeroing is load-bearing, and it is kept.
    """
    nx, ny, nz = shape
    dx, dy, dz = corner
    wx = frac[:, 0] if dx else wlo[:, 0]
    wy = frac[:, 1] if dy else wlo[:, 1]
    wz = frac[:, 2] if dz else wlo[:, 2]
    ix = (base[:, 0] + dx) % nx
    iy = (base[:, 1] + dy) % ny
    iz = (base[:, 2] + dz) % nz
    flat = (ix * ny + iy) * nz + iz
    return flat, jnp.where(ok, wx * wy * wz, 0.0)


def tile_paint_f64(u, live, shape, cell, mean, fdtype=jnp.float64):
    """CIC paint of tile-local coords into ONE padded tile mesh -> (mesh, n_out).

    Paints counts/mean, with NO -1. Two separate facts, both measured (module
    docstring):
      - the -1 is unnecessary, because ik(0) = 0 kills the DC term exactly, so
        the short force cannot see the offset (8.6e-16 for a shift of 1);
      - `mean` is MANDATORY and must be the GLOBAL mean n_total/n_mesh^3 -- a
        config scalar, not a reduction. A tile's own mean rescales the whole short
        force by mean_global/mean_tile, an error of exactly |s-1| (O(1)), which
        reads as a catastrophic tiling failure and sends you hunting buffers.

    Index safety is `_tile_corner`'s `% P` plus the zero weight; see there. (This
    docstring described a `flat = where(ok, flat, 0)` line until 2026-08-08 --
    `eba91ab` had removed it eight months of reading earlier, and the stale text
    was still recommending the construction that cost 2.2x.)

    `fdtype` (M-v2-4) is the RETURNED mesh's dtype. The accumulator stays f64
    and the division by `mean` happens before the narrowing: narrowing first
    would round twice, and the tile-identity residual this arm is tested to
    (1e-13) is tight enough to see that.
    """
    fdtype = field_dtype(fdtype)
    nx, ny, nz = (int(s) for s in shape)
    base, frac, ok, n_out = _tile_cic_pieces(u, live, (nx, ny, nz), cell)
    mesh = jnp.zeros((nx * ny * nz,), dtype=jnp.float64)
    wlo = 1.0 - frac
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                flat, w = _tile_corner(base, frac, wlo, (dx, dy, dz), (nx, ny, nz), ok)
                mesh = mesh.at[flat].add(w.astype(jnp.float64), mode="promise_in_bounds")
    return (mesh.reshape(nx, ny, nz) / float(mean)).astype(fdtype), n_out


def tile_paint_int(u, live, shape, cell, frac_bits=TILE_FRAC_BITS):
    """Deterministic integer CIC paint of tile-local coords -> (int32 mesh, n_out).

    The D-006 twin of `tile_paint_f64`, and the reason M-v2-3 needed one at all.
    D-v2-14 clause 4 admits the brick-sorted layout only because the paint is
    order-independent -- brick-sorting reorders particles every step, so an
    order-dependent primal paint is not reproducible even on one machine. That
    clause was discharged for the COARSE arm by `paint_tsc_int` (D-v2-16 cl.2)
    and was simply never applied to the short arm, which is where most of a
    particle's force comes from. Integer addition is associative, so this is
    bit-identical regardless of the atomic order the member sequence produces.

    Returns RAW fixed-point counts, exactly as `paint_int` does -- NOT counts/mean.
    `tile_delta_from_int` is the decode, and it is separate because the mesh is
    what a chunked accumulation adds into.

    **This is not only a determinism change, and the difference is measurable.**
    `tile_paint_f64` carries f64 corner weights; here they are quantized to
    `frac_bits` fixed point and rounded through f32, as every other int paint in
    this package does. So the short arm's ACCURACY moves too, which is why the
    engine's flip is gated on re-measuring the accumulated quantization rather
    than on this file's tests alone.

    Headroom is checked against the STRICT 8x CIC stencil bound
    (`painting.CIC_CELL_WEIGHT_BOUND`), not the 1x that `check_int_paint_headroom`
    has always assumed by default -- a tile cell inside a collapsing halo is
    exactly where the optimistic factor would be found out.
    """
    nx, ny, nz = (int(s) for s in shape)
    scale = np.float32(2.0**frac_bits)  # np scalar: no device array at trace-build time
    base, frac, ok, n_out = _tile_cic_pieces(u, live, (nx, ny, nz), cell)
    mesh = jnp.zeros((nx * ny * nz,), dtype=jnp.int32)
    wlo = 1.0 - frac
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                flat, w = _tile_corner(base, frac, wlo, (dx, dy, dz), (nx, ny, nz), ok)
                mesh = mesh.at[flat].add(
                    rint_i(w.astype(jnp.float32) * scale), mode="promise_in_bounds"
                )
    return mesh.reshape(nx, ny, nz), n_out


def tile_delta_from_int(mesh_int, mean, frac_bits=TILE_FRAC_BITS, fdtype=jnp.float64):
    """Decode `tile_paint_int`'s raw mesh to the same quantity `tile_paint_f64`
    returns: counts/mean, with NO -1 (see `tile_paint_f64` for why the -1 is
    unnecessary and why `mean` must be the GLOBAL config scalar).

    **`fdtype` CHANGED MEANING at M-v2-4**: it used to be the decode dtype,
    handed straight to `counts_from_int`; it is now the dtype of the returned
    FIELD, with the decode and the division pinned at f64. Same default, same
    result at that default, different behaviour at f32 -- which is exactly the
    shape of change that goes unnoticed. Flagged loudly because M-v2-3 lost half
    a day to the mirror image of it: `evolve_float`'s `paint` default had
    decayed, and every call site omitted the argument, so nothing showed it.
    See `density_tsc` for why the decode is pinned.
    """
    fdtype = field_dtype(fdtype)
    return (counts_from_int(mesh_int, frac_bits, fdtype=jnp.float64) / float(mean)).astype(fdtype)


def check_tile_paint_headroom(n_particles_total, frac_bits, max_cell_particles=1.0e4):
    """`check_int_paint_headroom` at the strict 8x CIC bound. See `tile_paint_int`."""
    check_int_paint_headroom(
        n_particles_total,
        frac_bits,
        max_cell_particles=max_cell_particles,
        bound=CIC_CELL_WEIGHT_BOUND,
    )


def tile_gather_vector(gx, gy, gz, u, live, shape, cell):
    """Read 3 tile fields with ONE shared CIC stencil (`cic_read_vector` twin,
    origin-shifted). Zeroed on ~live. Returns ((n,3) field-dtype, n_out).

    Dtype follows the field, per the narrowing rule in `painting.py`. This one
    hardcoded an f64 accumulator until M-v2-4, so it returned f64 even from an
    f32 tile mesh -- the only one of the four that did not even read the field's
    dtype.
    """
    nx, ny, nz = (int(s) for s in shape)
    base, frac, ok, n_out = _tile_cic_pieces(u, live, (nx, ny, nz), cell)
    fx, fy, fz = gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)
    n = u.shape[0]
    dt = gx.dtype
    ax = jnp.zeros((n,), dtype=dt)
    ay = jnp.zeros((n,), dtype=dt)
    az = jnp.zeros((n,), dtype=dt)
    wlo = 1.0 - frac
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                flat, w = _tile_corner(base, frac, wlo, (dx, dy, dz), (nx, ny, nz), ok)
                w = w.astype(dt)
                ax = ax + w * fx[flat]
                ay = ay + w * fy[flat]
                az = az + w * fz[flat]
    return jnp.stack([ax, ay, az], axis=1), n_out


# ===========================================================================
# per-tile coarse sub-block staging (D-v2-16 clause 3)
# ===========================================================================
#
# The long-range force lives on the global COARSE mesh, and holding it resident
# is affordable at C-gh and not at C-hero. Staging only the sub-block a tile can
# reach makes the whole force path O(tile) in device memory and removes that
# cliff by construction, which is why D-v2-16 calls it structural rather than an
# optimization.
#
# THE FIGURES CARRY A DTYPE, and this comment used to drop it (corrected
# M-v2-4, 2026-08-09). Three coarse force meshes are:
#
#            C-gh (1024^3)   C-hero (2048^3)
#     f32       12.9 GB          103 GB
#     f64       25.8 GB          206 GB
#
# `v4_architecture_record.md` item 6 derived the f32 row and said so; D-v2-16
# clause 3 and this comment then restated it without the qualifier. The engine
# has been running the f64 row -- `split_kernels` built at np.float64 and the
# streamed decode returns f64 -- so the resident cost has been 2x the ratified
# figure since the freeze. The CONCLUSION is untouched, and in fact stronger at
# f64: staging removes the cliff either way. M-v2-4 makes the artifact match the
# f32 row, and its gate replaces "derived, not measured" with a measured ladder.
#
# NB 103 GB also appears in `g4_record.md` as C-gh's f32 POSITIONS (2048^3 x 3
# x 4 B). Same arithmetic, different object. Do not unify them.
#
# WHY THE SUB-BLOCK IS SMALL. Only a tile's OWNED rows -- its core, not its
# buffer -- need the long force, because ownership is a partition and the kick is
# tile-local. So the block spans the core's coarse cells plus a halo for the
# assignment stencil: (T/COARSE_RATIO + 2*halo)^3 rather than anything that grows
# with the box. At C-gh's T=256 that is (64 + 4)^3 cells against the global
# 1024^3, a factor of 3,500.
#
# halo=2 rather than 1: TSC reads the NEAREST cell +-1, and its base comes from
# round() rather than floor(), so a particle at the core's edge can reach one
# cell further out than a CIC-style bound would suggest. The extra layer costs
# (68/66)^3 = 1.09x of a block that is already negligible, and the alternative is
# an off-by-one that only fires on the tiles touching a box face.

COARSE_HALO = 2


def coarse_subblock_origin_extent(tijk, n_tile, n_coarse, n_fine, halo=COARSE_HALO):
    """(origin in coarse CELLS (3,) int, extent in cells) for one tile's core.

    The origin may be negative and the block may run past the mesh; both are
    handled by wrapping at extraction, exactly as `tile_origin_extent` leans on
    `mod` rather than special-casing the boundary.
    """
    ratio = int(n_fine) // int(n_coarse)
    if int(n_tile) % ratio:
        raise ValueError(
            f"tile {n_tile} fine cells is not a whole number of coarse cells "
            f"(ratio {ratio}); the sub-block would not align with the core"
        )
    t = np.asarray(tijk, dtype=np.int64)
    per_tile = int(n_tile) // ratio
    return t * per_tile - int(halo), per_tile + 2 * int(halo)


def stage_coarse_subblock(g_coarse, origin_cells, extent):
    """Extract the (extent,)*3 periodic sub-block at `origin_cells`.

    Host-side numpy, wrapped per axis by construction, so a block straddling the
    periodic boundary needs no special case and no copy of the whole mesh.

    ONE 3-D GATHER, NOT THREE 1-D ONES. Taking axis by axis is the obvious
    spelling and it materializes the whole intermediate slab before the second
    axis narrows it: at c-hero the first take alone builds (132, 2048, 2048) =
    2.21 GB to deliver a 9.20 MB block, and it runs once per force component per
    tile -- 3 x 4096 = 12,288 times a step, i.e. **27.2 TB of host writes per
    step to produce 113 GB of output**. `np.ix_` gathers the extent^3 elements
    directly. It is the same elements in the same order, so the change is an
    identity and `test_the_subblock_is_the_wrapped_slice` pins it against the
    global mesh element by element, independent of either spelling.

    The defect was invisible until c-hero because the intermediate scales as
    extent x n_coarse^2: at cgh64 it is small enough to disappear into noise,
    and the engine has never run at c-hero. It is also ~2.21 GB of transient
    host allocation per call that no budget carries -- at W=8 workers, ~17.7 GB.
    """
    g = np.asarray(g_coarse)
    o = np.asarray(origin_cells, dtype=np.int64)
    if o.shape != (3,) or g.ndim != 3:
        raise ValueError(f"want a 3-D mesh and a (3,) origin, got {g.shape} and {o.shape}")
    span = np.arange(int(extent), dtype=np.int64)
    return g[np.ix_(*((span + int(o[axis])) % g.shape[axis] for axis in range(3)))]


def gather_coarse_subblock(
    sub_x, sub_y, sub_z, positions, origin_cells, cell_coarse, n_coarse, assign="tsc",
    live=None,
):
    """Read the long force for one tile's rows out of a staged sub-block.

    BITWISE identical to gathering from the global coarse mesh. That is a
    contract, not an aspiration: staging is a memory decision and must not move a
    number.

    **The obvious implementation does not achieve it, and measured 8.9e-16.**
    Shifting the COORDINATE into block-local space (u = mod(pos - origin, L), the
    way the tile short arm does) changes the last bits of `pos/cell` and
    therefore of the fractional offset, so the weights differ by roundoff even
    though the cell values are a verbatim slice. Roundoff would be harmless
    physically and fatal to the parity gate, which is the whole reason this path
    exists at C-hero.

    So the weights come from the GLOBAL coordinate, exactly as the global gather
    computes them, and only the integer index is shifted -- by
    `(base - origin) mod n_coarse`, which is exact. Values identical, weights
    identical, corner order identical, therefore bits identical.

    **`live` exists because of a measured compile storm (M-v2-3).** Called once
    per tile with each tile's own row count, this keys a NEW XLA shape per tile:
    profiled at 2,107 compilations and 24.1 s of a 32.7 s engine step, 74% of it,
    with 18.5 s inside `backend_compile_and_load`. That is exactly the trap
    `make_tile_force_fn` documents for the short arm -- "a per-tile member count
    would key a new shape and recompile per tile" -- and the fix is the same one:
    the caller pads rows to a fixed capacity and passes the mask, so ONE compiled
    program serves every tile. Padded rows read index 0 with zero weight, which
    is the same no-op construction `_tile_corner` uses.

    The caller must pass only rows whose stencil fits the halo. Rows outside the
    tile's core would read wrapped values from the far side of the block,
    silently and plausibly, so they are refused rather than trusted.
    """
    extent = int(np.asarray(sub_x).shape[0])
    origin = np.asarray(origin_cells, dtype=np.int64)
    xp = jnp.asarray(positions) / float(cell_coarse)
    if assign == "tsc":
        base_f = jnp.round(xp)
        d = xp - base_f
        w_axis = (0.5 * (0.5 - d) ** 2, 0.75 - d**2, 0.5 * (0.5 + d) ** 2)
        corners, first = _TSC_CORNERS, -1
    elif assign == "cic":
        base_f = jnp.floor(xp)
        frac = xp - base_f
        w_axis = (1.0 - frac, frac)
        corners, first = _CIC_CORNERS, 0
    else:
        raise ValueError(f"assign must be 'cic' or 'tsc', got {assign!r}")
    base = jax.lax.stop_gradient(base_f).astype(jnp.int32)
    # exact integer re-basing; `% n_coarse` puts a block straddling the periodic
    # boundary back in range without touching any float
    i = jnp.mod(base - jnp.asarray(origin, dtype=jnp.int32), int(n_coarse))

    i_np = np.asarray(i)
    keep = np.ones(i_np.shape[0], dtype=bool) if live is None else np.asarray(live)
    if keep.any():
        lo_needed = int(i_np[keep].min()) + first
        hi_needed = int(i_np[keep].max()) + first + len(w_axis) - 1
        if lo_needed < 0 or hi_needed >= extent:
            raise ValueError(
                f"a row's {assign} stencil reaches outside the staged sub-block: needs "
                f"[{lo_needed}, {hi_needed}] of [0, {extent - 1}]. Pass only the tile's OWNED "
                "rows, or raise the halo -- reading past the block wraps to the far side of "
                "the mesh and is silent."
            )
    if live is not None:
        # padded rows read a valid address with zero weight, exactly as
        # `_tile_corner` does; the index must stay in range for every row
        m = jnp.asarray(keep)[:, None]
        i = jnp.where(m, i, -first)
        w_axis = tuple(jnp.where(m, w, 0.0) for w in w_axis)

    fx, fy, fz = (jnp.asarray(s).reshape(-1) for s in (sub_x, sub_y, sub_z))
    n = xp.shape[0]
    dt = fx.dtype
    ax = jnp.zeros((n,), dtype=dt)
    ay = jnp.zeros((n,), dtype=dt)
    az = jnp.zeros((n,), dtype=dt)
    for dx, dy, dz in corners:
        # the product is formed in f64 and narrowed ONCE, matching
        # `_tsc_corner_flat_weight` exactly. Narrowing `w_axis` above instead
        # breaks the bitwise contract on 113 of 186 elements at 1.19e-7 --
        # measured, on the unmasked path, so it diverges everywhere and not
        # only for padded rows (see the narrowing rule in painting.py)
        ww = w_axis[dx - first][:, 0] * w_axis[dy - first][:, 1] * w_axis[dz - first][:, 2]
        ww = ww.astype(dt)
        flat = ((i[:, 0] + dx) * extent + (i[:, 1] + dy)) * extent + (i[:, 2] + dz)
        ax = ax + ww * fx[flat]
        ay = ay + ww * fy[flat]
        az = az + ww * fz[flat]
    return jnp.stack([ax, ay, az], axis=1)


# ===========================================================================
# the tiled short arm
# ===========================================================================


def make_tile_force_fn(
    n_fine,
    box_size,
    n_particles_total,
    n_tile,
    b_fine,
    r_s=None,
    paint="f64",
    frac_bits=TILE_FRAC_BITS,
    fdtype=jnp.float64,
):
    """Build the jitted per-tile short-force program -> (one_tile, geom).

    ONE jitted program is reused for every tile, which is only possible because
    `cap` is fixed: a per-tile member count would key a new shape and recompile
    per tile. Tile 0 therefore carries the whole XLA compile and the steady-state
    per-tile cost is the median over tiles 1.., which is the number any
    cross-box extrapolation must use -- tile count scales with volume, compile
    does not.

    Each tile+buffer is FFT'd as a small PERIODIC box of P^3 fine cells. The tile
    k-grid has fundamental 2pi/(P*d_f) but the SAME Nyquist pi/d_f as the global
    mesh, so the two grids periodize the identical sharp-k-truncated continuum
    kernel and differ ONLY by periodization.

    Why the wrap should be harmless: for a core target every source within the
    short kernel's range R ~ beta*r_s is present if b >= R, and images sit at
    >= P - R >= T + R where the short force is erfc-suppressed far below the
    truncation level. That was NOT taken on faith (risk R7) -- sharp-k truncation
    at the fine Nyquist gives the real-space kernel oscillatory ~1/x ringing
    tails, and had those dominated the image sums the periodization error would
    decay as a POWER LAW in P rather than erfc, making `buffer ~ 5 r_s` and every
    cost number in this engine wrong. G5's decay-law, seam-profile and
    T-independence signatures discriminated the two; D-v2-10 is the verdict.

    `one_tile(u, live) -> (g, owned, n_out)` where `owned` marks the rows this
    tile is responsible for (core, not buffer), so ownership is a partition and
    the kick can be applied tile-locally without any global force array.

    `paint="f64"` is the DEFAULT and is the order-dependent accumulator every
    D-v2-10/11/12 number was measured through, so the probe-parity tests keep
    comparing like with like. `paint="int"` is `tile_paint_int`, order-independent
    by associativity -- which is what D-v2-14 clause 4 requires of a layout that
    reorders particles every step, and what makes a bitwise gate possible when
    the engine's slot order differs from the probe's membership order. The engine
    selects "int".

    `fdtype` (M-v2-4) sets the tile MESH and its kernels: delta, the FFT, the
    three force meshes and the gather. **This arm is measured, not gated.** The
    memory argument is about the global coarse mesh, which grows with the box; a
    tile's working set is P^3 and box-independent, so narrowing it buys device
    headroom per tile rather than capacity. It is threaded here because it is
    the same plumbing, and because M-v2-6 needs to know whether the per-tile
    working set can halve too.
    """
    if paint not in ("f64", "int"):
        raise ValueError(f"paint must be 'f64' or 'int', got {paint!r}")
    if paint == "int":
        check_tile_paint_headroom(n_particles_total, frac_bits)
    fdtype = field_dtype(fdtype)
    geom = tile_geom(n_fine, box_size, n_particles_total, n_tile, b_fine,
                     paint=paint, frac_bits=frac_bits, fdtype=fdtype)
    cell, mean = geom["cell"], geom["mean"]
    P = geom["P"]
    kers = [jnp.asarray(k) for k in split_kernels((P,) * 3, cell, "short", r_s=r_s,
                                                  fdtype=fdtype)]

    def one_tile(u, live, owned):
        # `owned` is SUPPLIED, not computed here. It used to be
        # `live & all((u >= core_lo) & (u < core_hi), axis=1)`, which is not a
        # partition in floating point because each tile reaches `u` through its
        # own subtraction -- see `owning_tile` for the measured failure. Ownership
        # is a layout property of the global position and is now decided there,
        # exactly, in integers. It is passed through this function so the mask
        # cannot drift apart from the force it labels.
        if paint == "int":
            mesh_i, n_out_p = tile_paint_int(u, live, (P,) * 3, cell, frac_bits)
            # BY KEYWORD. This call dropped `fdtype` by being positional until
            # M-v2-4, so the tile arm decoded at the f64 default no matter what
            # the caller asked for -- the parameter existed and nothing could
            # reach it.
            delta = tile_delta_from_int(mesh_i, mean, frac_bits, fdtype=fdtype)
        else:
            delta, n_out_p = tile_paint_f64(u, live, (P,) * 3, cell, mean, fdtype=fdtype)
        dk = jnp.fft.rfftn(delta)
        g = [jnp.fft.irfftn(dk * k, s=(P,) * 3) for k in kers]
        out, n_out_g = tile_gather_vector(g[0], g[1], g[2], u, live, (P,) * 3, cell)
        return out, live & owned, n_out_p + n_out_g

    return jax.jit(one_tile), geom


def tile_geom(n_fine, box_size, n_particles_total, n_tile, b_fine,
              paint="f64", frac_bits=TILE_FRAC_BITS, fdtype=jnp.float64):
    """The geometry half of `make_tile_force_fn`, without building anything.

    One source of truth for the numbers `make_tile_force_fn` publishes in its
    `geom` dict, so a caller that does not need the kernels -- the pool
    executor's parent process, which never runs a tile itself -- can get
    bit-identical `cell`/`mean`/`P` without paying the kernel triple's memory
    or the jit trace. `make_tile_force_fn` reads its own locals back from this
    dict, so the two cannot drift.
    """
    fdtype = field_dtype(fdtype)
    cell = float(box_size) / int(n_fine)
    mean = float(n_particles_total) / float(n_fine) ** 3
    P, b_real = padded_size(n_tile, b_fine, n_fine=n_fine)
    core_lo = b_real * cell
    core_hi = (b_real + int(n_tile)) * cell
    return dict(P=int(P), b_realized=int(b_real), cell=cell, mean=mean,
                core_lo=core_lo, core_hi=core_hi, n_side=int(n_fine) // int(n_tile),
                paint=str(paint), frac_bits=int(frac_bits), fdtype=fdtype.name)


# A global (n,3) f64 force array is 2 x 206 GB at C-gh -- the two arrays whose
# deletion is what makes C-gh runnable at all (D-v2-16 cl.1). The accumulate sink
# below materializes one, so it is capped: it exists to let tests compare against
# the probe's host-accumulated output, not to run production.
MAX_ACCUMULATE_BYTES = 2 * 1024**3


def force_short_tiled(
    positions,
    n_fine,
    box_size,
    n_particles_total,
    n_tile,
    b_fine,
    member_fn,
    cap,
    r_s=None,
    pad_fill="cycle",
    sink=None,
    max_accumulate_bytes=MAX_ACCUMULATE_BYTES,
    paint="f64",
    frac_bits=TILE_FRAC_BITS,
    fdtype=jnp.float64,
):
    """Drive the tiled short force over every tile -> (g or None, diag).

    `member_fn(tijk) -> int64 indices` and `cap` are REQUIRED and come from the
    caller. Membership is `layout.py`'s job and the force does not own a
    bucketing; passing them in is also what lets the parity gate drive this with
    the probe's own membership, so the comparison isolates the force computation
    from the exchange.

    `paint` / `frac_bits` are passed straight to `make_tile_force_fn`; see there
    for why "f64" is the default and the engine selects "int".

    `sink=None` accumulates into a global (n,3) host array and returns it. That
    is the path D-v2-16 clause 1 keeps FOR TESTS ONLY, and it refuses above
    `max_accumulate_bytes` -- at C-gh the array it would build is 206 GB, and the
    deletion of two such arrays is the single change that makes C-gh runnable.
    Production passes a callable `sink(idx, g_owned)` invoked per tile with only
    that tile's owned rows, so nothing O(box) is ever materialized.
    """
    positions = np.asarray(positions)
    one_tile, geom = make_tile_force_fn(
        n_fine, box_size, n_particles_total, n_tile, b_fine, r_s=r_s,
        paint=paint, frac_bits=frac_bits, fdtype=fdtype,
    )
    P, b_real, cell = geom["P"], geom["b_realized"], geom["cell"]
    n_side = geom["n_side"]
    tiles = [(i, j, k) for i in range(n_side) for j in range(n_side) for k in range(n_side)]
    n = positions.shape[0]

    accumulate = sink is None
    if accumulate:
        need = n * 3 * 8
        if need > int(max_accumulate_bytes):
            raise ValueError(
                f"the global accumulate sink would allocate {need / 1024**3:.1f} GiB for "
                f"{n} particles, over the {max_accumulate_bytes / 1024**3:.1f} GiB limit. "
                "That array is what D-v2-16 clause 1 deletes (2 x 206 GB at C-gh); pass a "
                "tile-local `sink(idx, g_owned)` instead of raising the limit."
            )
        g_out = np.zeros((n, 3), dtype=geom["fdtype"])
    else:
        g_out = None

    owner_count = np.zeros((n,), dtype=np.int32)
    n_overhang_total = 0
    for t in tiles:
        idx = np.asarray(member_fn(t))
        m = len(idx)
        if m > cap:
            raise RuntimeError(f"tile {t}: {m} members > cap {cap} (host capacity is wrong)")
        # Padding fill. Pad rows are masked to zero WEIGHT either way, so both
        # arms give bitwise-identical forces; what differs is which mesh
        # addresses their (zero-weight) scatter-adds contend for. "zero" points
        # every pad row at particle 0, so all cap-m of them hit the same 8 cells
        # -- the pre-2026-08-07 behaviour, retained ONLY so the cost of that
        # contention stays measurable as an A/B (it was 2.218x on device).
        if pad_fill == "cycle" and m > 0:
            idx_pad = np.resize(idx, cap)
        elif pad_fill in ("zero", "cycle"):
            idx_pad = np.zeros((cap,), dtype=np.int64)
            idx_pad[:m] = idx
        else:
            raise ValueError(f"pad_fill must be 'cycle' or 'zero', got {pad_fill!r}")
        live_np = np.zeros((cap,), dtype=bool)
        live_np[:m] = True
        origin, _ = tile_origin_extent(t, n_tile, b_real, cell)
        xg = positions[idx_pad]
        u = jnp.mod(jnp.asarray(xg) - jnp.asarray(origin), float(box_size))
        own_np = owned_mask(xg, t, cell, n_tile, n_fine, live=live_np)
        out, owned, n_out = one_tile(u, jnp.asarray(live_np), jnp.asarray(own_np))
        out = np.asarray(out)
        owned = np.asarray(owned)
        n_overhang_total += int(n_out)
        sel = owned[:m]
        if accumulate:
            g_out[idx[sel]] = out[:m][sel]
        else:
            sink(idx[sel], out[:m][sel])
        owner_count[idx[sel]] += 1
        del out, owned

    diag = dict(
        n_tiles=len(tiles),
        n_tile=int(n_tile),
        b_requested=int(b_fine),
        b_realized=int(b_real),
        padded_P=int(P),
        cap=int(cap),
        pad_fill=str(pad_fill),
        paint=str(paint),
        frac_bits=int(frac_bits),
        fdtype=geom["fdtype"],
        fft_work_ratio=float(len(tiles) * P**3 / float(n_fine) ** 3),
        # Superset overhang: brick-union members outside the padded mesh. Since
        # choose_brick gained `c | b_fine` the union is EXACTLY the padded box, so
        # this is a CONTRACT (must be 0) and a nonzero value means the brick
        # decomposition is wrong, not merely wasteful. The pre-fix V4a card has
        # 3,044,340,012 here at T128/b96 and 0 at every other leg.
        n_overhang_total=int(n_overhang_total),
        # THE REAL CONTRACT: every particle owned by exactly one tile.
        n_owned_total=int((owner_count > 0).sum()),
        min_owner_count=int(owner_count.min()),
        max_owner_count=int(owner_count.max()),
        partition_ok=bool(
            owner_count.min() == 1 and owner_count.max() == 1
            and int((owner_count > 0).sum()) == n
        ),
    )
    return g_out, diag
