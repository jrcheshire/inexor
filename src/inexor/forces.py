"""Geometric PM force solve: dimensionless Poisson div g = -delta, returning g = -grad phi.

All cosmology prefactors live in the integrator coefficients (integrate.py). The kernel ik/k^2
is the Zel'dovich displacement kernel (force == ZA identity is tested). Per-component solves
never hold three force meshes at once.

Two-level split (PM-PM, not P3M/TreePM; the decomposition is PMFAST's with a Fourier matching
kernel): long force S(k) ik/k^2 on the global coarse mesh, short force (1 - S(k)) ik/k^2 on
fine tiles, each tile+buffer FFT'd as a small periodic box; S(k) = exp(-k^2 r_s^2). Two
structural properties carry the error argument; do not "clean up" the expressions:

  1. `short := 1.0 - S` shares one k2_safe with the long kernel, so long + short == ik/k^2
     bit-exactly, and every residual is a discretization error with one named cause.
  2. Both kernels carry ik_j, exactly zero at k = 0, so the short force ignores the density's
     DC level. A tile still needs the GLOBAL mean as a normalization scalar: a tile's own mean
     rescales its whole short force by mean_global/mean_tile. The global mean is the config
     constant n_total / n_mesh^3, so nothing is communicated.

Dimensionless knobs, one per error term (transferable across configurations):

    alpha = r_s / d_coarse   -> coarse representation error, exp(-pi^2 alpha^2)
    beta  = b   / r_s        -> buffer truncation error, erfc(beta/2)
    b_fine = 4 * alpha * beta                    (coarse:fine ratio is 4)

The kernel expressions are kept numerically identical to the reference implementation the
recorded accuracy measurements were taken with; changing their association moves bits.
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

# Coarse:fine mesh ratio.
COARSE_RATIO = 4

# Fixed-point fraction bits for the tiled short arm's integer paint (see `tile_paint_int`).
TILE_FRAC_BITS = 12


def k_components(n_mesh, box_size, fdtype=np.float32):
    """i k_j (complex, low-rank) and 1/k^2 (full) on the rfftn half-grid, k=0 safe.

    Host-numpy build; returns jnp arrays (complex64 for f32, complex128 for f64).
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
    """Force closure per (BoxConfig, dtype, paint), lru_cached so equal args share ONE object
    (step_fwd and step_rev must receive the same function).
    """
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
    """Single-level PM force: div g = -delta; force(positions (n,3)) -> (n,3) fdtype.

    paint="int": deterministic integer paint (primal path; required for bit-exact replay, since an
    f32 scatter-add is order-dependent on GPU). paint="f32": differentiable twin. Returns a cached
    closure: equal args return the same object.
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
    """ik_j (low-rank), k2_true and k2_safe on the rfftn half-grid of a `shape` box of `cell` cells.

    `k_components` twin for non-cubic boxes (padded tiles). Host numpy; ik_j have shapes
    (nx,1,1)/(1,ny,1)/(1,1,nz//2+1) so they broadcast. Returns (ikx, iky, ikz, k2_true, k2_safe):
    k2_true has a genuine 0 at DC, k2_safe has 1 there. S must be built from k2_true, or
    S(0) = exp(-r_s^2) != 1 breaks the split at DC.
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
    """S(k) = exp(-k^2 r_s^2), the Gaussian split, from k2_true.

    Gaussian because both error tails are analytic: coarse representation exp(-pi^2 alpha^2) and
    buffer truncation erfc(beta/2). r_s = 0 -> S == 1 exactly.
    """
    return np.exp(-k2_true * float(r_s) ** 2)


def split_factor(k2_true, r_s, which):
    """S (which="long"), 1-S (which="short"), or 1 (which="mono").

    short is literally 1.0 - S from the same array, so long + short == 1 in floating point. Do not
    rewrite as -expm1(-k^2 r_s^2): that would make the identity depend on exp and expm1 agreeing to
    the last ulp.
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
    """W(k) = prod_i sinc^p(k_i cell / 2) on the rfftn half-grid; order p=2 is CIC, p=3 is TSC.

    Takes the cell explicitly, so it can express W(fine cell) on the coarse k-grid.
    """
    nx, ny, nz = (int(s) for s in shape)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=cell)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=cell)
    kz = 2.0 * np.pi * np.fft.rfftfreq(nz, d=cell)

    def w(k_1d):
        return np.sinc(k_1d * float(cell) / (2.0 * np.pi)) ** int(order)

    return w(kx).reshape(nx, 1, 1) * w(ky).reshape(1, ny, 1) * w(kz).reshape(1, 1, nz // 2 + 1)


def cic_match_factor(shape, cell_solve, cell_target, clip=None, order_solve=2, order_target=2,
                     y=None):
    """(W(k, cell_target) / W(k, cell_solve))^2 on the rfftn half-grid (Hockney-Eastwood matching).

    The long arm paints and gathers on the coarse cell and the short arm on the fine cell, so
    their windows differ even with a perfect split; paint + gather apply W twice, hence the square.
    Correct for the Gaussian split only: S(k) suppresses the long kernel where the ratio grows
    toward the coarse Nyquist; kernels with real high-k content there would be amplified.
    `clip` bounds the factor. Returns (factor, max_applied) so the guard is reported.
    `y` = (lo, hi) builds only those y rows (1-D factors over the full axis, then sliced, so the
    rows are bitwise the full build's); `max_applied` is then over those rows.
    """
    nx, ny, nz = (int(s) for s in shape)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=cell_solve)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=cell_solve)
    kz = 2.0 * np.pi * np.fft.rfftfreq(nz, d=cell_solve)
    ys = slice(None) if y is None else slice(int(y[0]), int(y[1]))

    def w_full(cell, order):
        # np.sinc(y) = sin(pi y)/(pi y); squared for paint + gather
        def w(k_1d):
            return np.sinc(k_1d * cell / (2.0 * np.pi)) ** (2 * int(order))

        return (w(kx).reshape(nx, 1, 1) * w(ky)[ys].reshape(1, -1, 1)
                * w(kz).reshape(1, 1, nz // 2 + 1))

    ratio = w_full(cell_target, order_target) / w_full(cell_solve, order_solve)
    max_applied = float(ratio.max())
    if clip is not None:
        ratio = np.minimum(ratio, float(clip))
    return ratio, max_applied


def _match_orders(match):
    """Parse `match` = (cell_solve, cell_target[, order_solve, order_target]); the 2-tuple means CIC
    on both sides. Orders exist because the coarse arm uses TSC, and a CIC-order factor leaves
    sinc^2 per axis of the coarse window uncorrected.
    """
    if len(match) == 2:
        return {}
    if len(match) != 4:
        raise ValueError(f"match must be (cell_solve, cell_target[, order_solve, order_target]), "
                         f"got {match!r}")
    return dict(order_solve=int(match[2]), order_target=int(match[3]))


def split_kernels(shape, cell, which, r_s=None, fdtype=np.float64):
    """The (Kx, Ky, Kz) half-grid kernels of the Gaussian split: (split_factor / k2_safe) * ik_j.

    The build (k2, S, fac / k2_safe) is always f64; `fdtype` narrows only the real prefactor before
    the complex multiply. Narrowing `ik` instead would be f64 * complex64 -> complex128 (numpy
    promotion), silently not an f32 arm. The prefactor form is within 2 f32 ulp of narrowing the
    finished product and avoids a full complex128 half-grid per component. At the f64 default the
    casts are no-ops, so the result is bitwise the reference kernel.
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
    """delta on an n_mesh^3 grid via the package CIC stencil (f64 `paint_f32`).

    Differs from `make_force_fn` only in the kernel. `fdtype` is the returned field's dtype; the
    paint accumulator stays f64 (an f32 scatter-add is non-associative under CUDA atomics).
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
    """One global mesh solve of the chosen split kernel -> ((n,3) fdtype, max_match).

    `which="mono"` reproduces `make_force_fn`'s kernel. `match` applies `cic_match_factor`.
    `pos_gather`, if given, is where the field is read (default: `positions`), for a field built
    from different source positions. `paint` selects the TSC branch's accumulator: "f64"
    (order-dependent, the default for reference parity) or "int" (order-independent; the engine's
    choice). The `cic` branch has only the f64 accumulator; it serves the mono floors, not the
    shipping path.
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
    """The paint-and-kernel half of `force_global`."""
    fdtype = field_dtype(fdtype)
    cell = box_size / n_mesh
    kers = split_kernels((n_mesh,) * 3, cell, which, r_s=r_s, fdtype=fdtype)
    max_applied = 1.0
    if match is not None:
        mf, max_applied = cic_match_factor((n_mesh,) * 3, match[0], match[1], clip=clip,
                                           **_match_orders(match))
        # cast: an f64 `mf` would promote a complex64 kernel back to complex128
        kers = tuple(k * mf.astype(fdtype, copy=False) for k in kers)
    if assign == "cic":
        if paint != "f64":
            raise ValueError(
                f"assign='cic' has no {paint!r} accumulator: the int CIC path is "
                "density_contrast's, which carries f32 counts, and the coarse arm "
                "is TSC. Use assign='tsc' for the engine's order-independent long arm."
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


def coarse_kernel_block(n_mesh, box_size, which, r_s=None, match=None, clip=None,
                        fdtype=np.float64, y=None):
    """`pref` and `mf` (None without `match`) of the coarse kernel on y rows `y` = (lo, hi)
    (default all) of the rfftn half-grid: shape (N, hi - lo, N//2 + 1), fdtype.

    The rows are bitwise the full build's: the 1-D k and window factors are formed over the full
    axis and sliced, and the 3-D expression is the same element for element. Build peak ~28 B per
    element of the block.
    """
    fdtype = field_dtype(fdtype)
    n = int(n_mesh)
    lo, hi = (0, n) if y is None else (int(y[0]), int(y[1]))
    cell = box_size / n_mesh
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=cell)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=cell)[lo:hi]
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=cell)
    k2_true = (kx.reshape(n, 1, 1) ** 2 + ky.reshape(1, -1, 1) ** 2
               + kz.reshape(1, 1, -1) ** 2).astype(np.float64)
    k2_safe = k2_true.copy()
    if lo == 0 and hi > 0:
        k2_safe[0, 0, 0] = 1.0
    fac = split_factor(k2_true, 0.0 if r_s is None else r_s, which)
    pref = (fac / k2_safe).astype(fdtype, copy=False)
    # free the f64 island before the match factor is built
    del k2_true, k2_safe, fac
    mf = None
    if match is not None:
        # cast: an f64 match factor would promote a complex64 kernel to complex128
        m, _ = cic_match_factor((n, n, n), match[0], match[1], clip=clip, y=(lo, hi),
                                **_match_orders(match))
        mf = m.astype(fdtype, copy=False)
        del m
    return pref, mf


def _coarse_iks(n_mesh, box_size):
    """The low-rank ik_j of `kernel_grids` at f64 (complex128)."""
    n = int(n_mesh)
    cell = box_size / n_mesh
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=cell)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=cell)
    return ((1j * kx.reshape(n, 1, 1)).astype(np.complex128),
            (1j * kx.reshape(1, n, 1)).astype(np.complex128),
            (1j * kz.reshape(1, 1, n // 2 + 1)).astype(np.complex128))


def coarse_kernel_parts(n_mesh, box_size, which, r_s=None, match=None, clip=None,
                        fdtype=np.float64, cards=None):
    """The real half-grids the coarse solve needs, built once per run (they depend on geometry only).

    Returns dict(iks, pref, mf, cards, fdtype, n_mesh, which). Only real half-grids are kept: the
    complex kernels are formed one component at a time inside the solve; the ik_j are low-rank
    broadcasts. The match factor stays separate from `pref`: the solve computes `(pref * ik) * mf`,
    and folding it into `pref` would reassociate and move bits.

    `cards`, a list of `(y_lo, y_hi, device)`, keeps each card's y rows resident on that card
    instead (`pref` and `mf` are then None and `cards` holds `dict(lo, hi, device, pref, mf)`):
    each block is built on the host (`coarse_kernel_block`), placed, and freed before the next, so
    the host peak is one block's. Only the folded factorized solve on those cards can use them.
    """
    fdtype = field_dtype(fdtype)
    common = dict(iks=_coarse_iks(n_mesh, box_size), fdtype=fdtype, n_mesh=int(n_mesh),
                  which=which)
    if cards is None:
        pref, mf = coarse_kernel_block(n_mesh, box_size, which, r_s=r_s, match=match,
                                       clip=clip, fdtype=fdtype)
        return dict(pref=pref, mf=mf, cards=None, **common)
    import jax
    import jax.numpy as jnp

    placed = []
    for lo, hi, dev in cards:
        pref, mf = coarse_kernel_block(n_mesh, box_size, which, r_s=r_s, match=match,
                                       clip=clip, fdtype=fdtype, y=(lo, hi))
        put = jnp.asarray if dev is None else (lambda a, d=dev: jax.device_put(a, d))
        entry = dict(lo=int(lo), hi=int(hi), device=dev, pref=put(pref),
                     mf=None if mf is None else put(mf))
        jax.block_until_ready([v for v in (entry["pref"], entry["mf"]) if v is not None])
        placed.append(entry)
        del pref, mf
    return dict(pref=None, mf=None, cards=placed, **common)


def refuse_oversize_coarse_solve(n_mesh):
    """Refuse a monolithic device coarse solve at or above `ooc_fft.MAX_DEVICE_TRANSFORM_ELEMENTS`.

    Device FFTs at that size have returned wrong results silently (suspected 32-bit plan limits).
    CPU is exempt: the bound is empirical and was never observed there. Use the factorized solve
    (`ooc_fft.forward_from_slabs_device`) instead.
    """
    import jax

    from inexor import ooc_fft

    n_elements = int(n_mesh) ** 3
    if n_elements < ooc_fft.MAX_DEVICE_TRANSFORM_ELEMENTS:
        return
    if jax.default_backend() == "cpu":
        return
    raise ValueError(
        f"the monolithic coarse solve at n_mesh={n_mesh} is {n_elements:,} "
        f"elements, at or above the {ooc_fft.MAX_DEVICE_TRANSFORM_ELEMENTS:,} "
        f"bound where a device FFT has been MEASURED to return a wrong result "
        f"silently, on backend {jax.default_backend()!r}. Use the factorized "
        "path (`ooc_fft.forward_from_slabs_device`), which reads 6.676e-06 at "
        "2048^3; the plane is the unit."
    )


def coarse_kernel_slab(parts, axis, lo, hi, cdtype):
    """The complex kernel for one component on the x-slab [lo, hi).

    Bitwise the corresponding slice of the whole-grid kernel: only ikx slices, `pref`/`mf` slice
    along the same axis, and the expression is `(pref * ik) * mf` element for element.
    """
    ikx, iky, ikz = parts["iks"]
    ik = (ikx[lo:hi], iky, ikz)[axis]
    k = parts["pref"][lo:hi] * np.asarray(ik).astype(cdtype, copy=False)
    if parts["mf"] is not None:
        k = k * parts["mf"][lo:hi]
    return k


def _coarse_solve_factorized(delta, n_mesh, parts, cdtype, out, slab, timings=None,
                             fold_kernel=True, decomp=None, comm=None, receipt=None):
    """The coarse solve through `ooc_fft`'s canonical factorization.

    Required above the device-transform bound and when the mesh does not fit. Not bitwise the
    monolithic form (different transform order); bitwise invariant to slab thickness. Each
    component's spectrum copy is formed as the kernel multiply itself (one pass over the
    half-grid); with `fold_kernel=True` the multiply rides the inverse's axis-0 device pass instead
    (`fold_kernel=False` keeps a separate host multiply, the reference arm). `delta` may be per-card
    planes and `out` a `CardShards`, in which case no host mesh exists. `timings`, if a dict,
    accumulates seconds per part summed over components.

    Across ranks (`decomp`, `comm`, the density on the cards, the folded kernel, `out` a
    `CardShards`): each rank transforms its coarse planes, holds its y-pencils of the
    spectrum through the kernel pass, and receives the planes its cards' shards hold; the
    shards are bitwise the one-rank solve's. `receipt`, if a dict, accumulates the bytes sent.
    Every rank must call this.
    """
    import time

    from inexor import ooc_fft
    from inexor.device.coarse import CardShards

    def _add(key, t0, inner=None):
        if timings is not None:
            if inner is not None:
                for k, v in inner.items():
                    name = f"{key}: {k[:-2]}"
                    timings[name] = timings.get(name, 0.0) + v
            else:
                timings[key] = timings.get(key, 0.0) + time.perf_counter() - t0

    n = int(n_mesh)
    multi = comm is not None and comm.size > 1
    on_cards = isinstance(delta, (list, tuple))
    if multi and not (on_cards and fold_kernel and isinstance(out, CardShards)):
        raise ValueError(
            "across ranks the coarse solve needs the density on the cards, the folded kernel "
            "and card-shard output: every other form holds a whole-mesh host array")
    ft = {}
    pencils = None
    if on_cards:
        if decomp is None:
            x_parts, y_parts = [(0, n)], [(0, n)]
        else:
            cpt = int(decomp.coarse_per_tile)
            x_parts = [(lo * cpt, hi * cpt) for lo, hi in decomp.rank_planes]
            y_parts = [tuple(p) for p in decomp.rank_pencils]
        pencils = y_parts[0 if comm is None else comm.rank]
        spec = ooc_fft.forward_card_planes_to_pencils(delta, n, x_parts, y_parts, comm,
                                                      timings=ft, receipt=receipt)
    else:
        spec = ooc_fft.forward_from_slabs_device(
            lambda lo, hi: delta[lo:hi], n, slab=slab, timings=ft)
    _add("forward", None, ft)
    # `out` as `CardShards`: planes land on the cards that hold them; no host mesh
    per_card = [[] for _ in out.ranges] if isinstance(out, CardShards) else None
    # the folded kernel pass runs on those cards, split by y-pencils like the inverse's pass 2
    kdevs = None if per_card is None else [dev for _x0, _nx, dev in out.ranges]
    cards = parts.get("cards")
    if cards is not None:
        if not fold_kernel:
            raise ValueError("card-resident kernel parts need fold_kernel=True: the unfolded "
                             "multiply is a host pass over host arrays")
        have = [e["device"] for e in cards]
        if kdevs is None or have != kdevs:
            raise ValueError(
                f"the kernel parts are resident on {[str(d) for d in have]} but this solve "
                f"runs on {None if kdevs is None else [str(d) for d in kdevs]}")
    # one work buffer for all three components (avoids repeated first-touch page faults)
    work = np.empty_like(spec)
    for axis in range(3):
        it = {}
        if fold_kernel:
            # the kernel multiply rides the inverse's axis-0 device pass; hence pass2=False below
            t0 = time.perf_counter()
            ooc_fft.kspace_pass_device(
                [(1.0, spec)], n, kernel=ooc_fft.ArrayKernel.coarse(
                    parts["pref"], parts["iks"][axis], axis, parts["mf"], cdtype,
                    cards=cards),
                out=work, inverse=True, devices=kdevs, timings=it, pencils=pencils)
            _add("kernel + axis-0 pass", t0)
        else:
            t0 = time.perf_counter()
            for lo in range(0, spec.shape[0], slab):
                hi = min(lo + slab, spec.shape[0])
                work[lo:hi] = spec[lo:hi] * coarse_kernel_slab(parts, axis, lo, hi, cdtype)
            _add("multiply", t0)
        if per_card is not None and fold_kernel and on_cards:
            for k, m in enumerate(ooc_fft.inverse_pencils_to_card_shards(
                    work, n, out.ranges, y_parts, comm, timings=it, receipt=receipt)):
                per_card[k].append(m)
        elif per_card is not None:
            for k, m in enumerate(ooc_fft.inverse_to_card_shards(
                    work, n, out.ranges, timings=it, pass2=not fold_kernel)):
                per_card[k].append(m)
        else:
            for lo, block in ooc_fft.inverse_to_slabs_device(
                    work, n, slab=slab, timings=it, pass2=not fold_kernel):
                out[axis][lo:lo + block.shape[0]] = block
        _add("inverse", None, {k: v for k, v in it.items()
                               if k in ("pass1_s", "pass2_s", "transpose_s")})
    del work
    return out.assemble(per_card) if per_card is not None else out


def coarse_force_meshes(delta, n_mesh, box_size, which, r_s=None, match=None, clip=None,
                        fdtype=None, parts=None, out=None, transform="factorized",
                        slab=None, timings=None, fold_kernel=True, decomp=None, comm=None,
                        receipt=None):
    """The three long-range force meshes from an already-painted delta (solve only, no paint/gather).

    Bitwise the solve inside `force_global`. `fdtype=None` infers from `delta.dtype`; an explicit
    disagreement is refused rather than cast, because a mismatched multiply would promote the
    solve to the wider type while looking like a working narrow arm. `parts`: a
    `coarse_kernel_parts` dict (built per call if None; refused if built for another n_mesh or
    dtype). `out`: three preallocated meshes (or `CardShards`) written in place. `delta` may be the
    per-card shards of `device.paint.coarse_delta_cards` (factorized only). Components are solved
    one at a time, so the three meshes never coexist as jax arrays. `timings`, `fold_kernel`,
    `decomp`, `comm`, `receipt` (across ranks): see `_coarse_solve_factorized`.
    """
    on_cards = isinstance(delta, (list, tuple))
    ddt = np.dtype(delta[0]["delta"].dtype if on_cards else delta.dtype)
    fdtype = field_dtype(ddt if fdtype is None else fdtype)
    if ddt != fdtype:
        raise ValueError(
            f"coarse_force_meshes: delta is {ddt.name} but fdtype is "
            f"{fdtype.name}. These must agree -- the kernels are built at fdtype and a "
            "mismatched multiply promotes the whole solve back to the wider type, which "
            "reads as a working f32 arm that is silently costing f64 memory. Narrow the "
            "delta at its decode, or pass the dtype it already has."
        )
    if parts is None:
        parts = coarse_kernel_parts(n_mesh, box_size, which, r_s=r_s, match=match,
                                    clip=clip, fdtype=fdtype)
    elif parts["n_mesh"] != int(n_mesh) or parts["fdtype"] != fdtype:
        # a stale build would broadcast or promote silently rather than crash
        raise ValueError(
            f"coarse_force_meshes: parts were built for n_mesh={parts['n_mesh']} at "
            f"{parts['fdtype'].name} but this call is n_mesh={int(n_mesh)} at "
            f"{fdtype.name}. The kernel build is geometry, so a mismatch means the "
            "cache outlived the configuration it belongs to."
        )
    cdtype = np.complex128 if fdtype == np.dtype(np.float64) else np.complex64
    if out is None:
        out = [np.empty((int(n_mesh),) * 3, dtype=fdtype) for _ in range(3)]
    if transform == "factorized":
        from inexor import ooc_fft

        return _coarse_solve_factorized(
            delta, n_mesh, parts, cdtype, out,
            ooc_fft._DEF_SLAB if slab is None else int(slab), timings=timings,
            fold_kernel=fold_kernel, decomp=decomp, comm=comm, receipt=receipt)
    if on_cards:
        raise ValueError(
            "coarse_force_meshes: a density on the cards has only the factorized "
            "solve; a monolithic transform would need the whole mesh on one device")
    from inexor.device.coarse import CardShards

    if isinstance(out, CardShards):
        raise ValueError(
            "coarse_force_meshes: force meshes written onto the cards have only the "
            "factorized solve; the monolithic transform returns whole host meshes")
    if transform != "monolithic":
        raise ValueError(
            f"transform must be 'monolithic' or 'factorized', got {transform!r}")
    if parts.get("cards") is not None:
        raise ValueError("coarse_force_meshes: card-resident kernel parts have only the "
                         "factorized solve")
    refuse_oversize_coarse_solve(int(n_mesh))
    dk = jnp.fft.rfftn(delta)
    # one complex kernel alive at a time; `(pref * ik) * mf` keeps it bitwise the slab form
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
    """Tile coordinates owning each row, from the GLOBAL position: floor(x / cell) // n_tile.

    An exact integer partition: every row goes to exactly one tile. A tile-local float test is not
    complementary between neighbours (each tile subtracts a different origin), so a row can be
    disowned by both. Host numpy, outside the jitted kernel.
    """
    n_fine = int(n_fine)
    ci = np.floor(np.asarray(positions, dtype=np.float64) / float(cell))
    ci = np.mod(ci.astype(np.int64), n_fine)  # x == box_size exactly -> cell 0
    return ci // int(n_tile)


def owned_mask(positions, tijk, cell, n_tile, n_fine, live=None):
    """Rows of `positions` owned by tile `tijk`, by position (see `owning_tile`).

    A partition of space; the engine uses `owned_mask_from_bricks` instead. Used by
    `force_short_tiled`, whose `member_fn` returns indices without brick structure.
    """
    own = owning_tile(positions, cell, n_tile, n_fine)
    m = np.all(own == np.asarray(tijk, dtype=np.int64), axis=1)
    return m if live is None else (m & np.asarray(live, dtype=bool))


def owned_mask_from_bricks(brick_of_row, tijk, n_tile, n_brick, nb):
    """Rows owned by tile `tijk`, decided by the brick each row is stored in.

    The partition the engine counts: membership comes from brick ordinals, so a position-derived
    rule can disagree with the layout by one cell and leave a row unowned. Exact by arithmetic: the
    brick grid `nb = n_fine // n_brick` divides into tile cores of `n_tile // n_brick` bricks, so
    every brick (hence every stored row) belongs to exactly one tile. `brick_of_row` is the flat
    ordinal `(bi * nb + bj) * nb + bk` per decoded row.
    """
    b = np.asarray(brick_of_row, dtype=np.int64)
    nb = int(nb)
    per = int(n_tile) // int(n_brick)
    bi, rem = np.divmod(b, nb * nb)
    bj, bk = np.divmod(rem, nb)
    t = np.asarray(tijk, dtype=np.int64)
    return (bi // per == t[0]) & (bj // per == t[1]) & (bk // per == t[2])


def tile_capacity(member_counts):
    """Per-tile row capacity: the max member count over tiles, so one jitted program serves every
    tile (a per-tile count would key a new shape and recompile).
    """
    counts = np.asarray(list(member_counts), dtype=np.int64)
    if not len(counts):
        raise ValueError("no tiles: cap is a max over tiles and there are none")
    return int(counts.max())


# Rungs per octave for `capacity_shape`: worst-case pad 2^(1/3) - 1 = 26% of rows, at most
# three shapes per doubling of `cap` (padded rows vs retained XLA executables).
CAP_RUNGS_PER_OCTAVE = 3


def capacity_shape(cap, rungs=CAP_RUNGS_PER_OCTAVE, floor_shape=0):
    """Quantize `cap` up onto a fixed geometric ladder 2^(j/rungs), monotone in both arguments.

    `cap` moves every step, and each distinct shape compiles and retains a new XLA executable, so
    an unpinned shape family leaks host memory linearly in steps. The ladder is anchored globally
    (an octave-relative one is not monotone), and `floor_shape` makes it sticky across steps.
    Padding is masked (zero integer weight, per-row gathers, sliced back), so a larger shape is
    bitwise the smaller one.
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

# FFT-friendly padded sizes (products of small primes); P = T + 2*b_fine is rounded up to one
# (e.g. 164 = 4*41 would be pathologically slow). Do not insert entries <= 512: that would
# change the tile selection of existing configurations.
FFT_FRIENDLY = (
    32, 36, 40, 48, 50, 54, 60, 64, 72, 80, 90, 96, 100, 108, 120, 128,
    144, 150, 160, 162, 180, 192, 200, 216, 240, 250, 256, 288, 300, 320,
    360, 384, 400, 432, 480, 500, 512,
    540, 576, 600, 640, 648, 720, 750, 768, 800, 810, 864, 900, 960, 972,
    1000, 1024,
)  # fmt: skip


def padded_size(n_tile, b_fine, n_fine=None):
    """FFT-friendly P >= n_tile + 2*b_fine, and the realized buffer -> (P, b_realized).

    The buffer grows to the rounded size, so beta_realized >= beta_requested and is what is
    reported. `n_fine` enables the degeneracy guard: a padded tile larger than the fine mesh costs
    more than the monolithic solve and would double-count wrapped particles.
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
    """(origin (3,) f64, extent f64) of a tile+buffer box in global coords.

    origin = (t*n_tile - b_fine) * cell may be negative; mod(x - origin, L) handles the wrap.
    """
    t = np.asarray(tijk, dtype=np.int64)
    origin = (t * int(n_tile) - int(b_fine)) * float(cell)
    extent = (int(n_tile) + 2 * int(b_fine)) * float(cell)
    return origin, extent


def tile_local_coords(positions, origin, box_size):
    """u = mod(pos - origin, L): the tile-local coordinate and the membership test in one expression.

    In tile+buffer iff all(u < extent), exact across the periodic boundary. Do not use painting's
    `% n_mesh`: it would fold far-side particles into the tile.
    """
    return jnp.mod(positions - jnp.asarray(origin), float(box_size))


def _tile_cic_pieces(u, live, shape, cell):
    """Base cell, fractional offset, validity mask and out-of-box count for a padded tile.

    The padded box is periodic (its rfftn assumes so): corner indices wrap modulo P, the padded
    box's own period, never the global mesh's. `ok` = live and u in [0, extent). `n_out` counts
    live rows outside the box; it is a contract (must be 0 when bricks divide the buffer, see
    `layout.assert_brick_divides_buffer`), returned rather than raised so callers can assert it.
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
    """One CIC corner: flat index (wrapped mod P) and weight, weight zeroed on ~ok.

    The `% n` wrap keeps every index in range for every row, so mode="promise_in_bounds" is safe
    (out-of-range indices under it are silent corruption on GPU). Only the weight is masked:
    masking the index too would funnel all padded rows onto one address and serialize the
    atomics. mode="drop" is not used because it would hide a misrouted live particle.
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
    """CIC paint of tile-local coords into one padded tile mesh -> (counts/mean, n_out).

    No -1: ik(0) = 0 removes the DC term. `mean` must be the global n_total / n_mesh^3 (see
    module docstring). `fdtype` is the returned dtype; accumulation and the division stay f64.
    Order-dependent (f64 atomics); `tile_paint_int` is the deterministic twin.
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
    """Deterministic integer CIC paint of tile-local coords -> (raw int32 fixed-point mesh, n_out).

    Integer addition is associative, so the result is bit-identical under any member order or
    atomic order (required because the brick-sorted layout reorders particles). Weights are
    quantized to `frac_bits` through f32, so accuracy differs slightly from `tile_paint_f64`.
    Decode with `tile_delta_from_int`. No saturating op: overflow is refused up front by
    `check_tile_paint_headroom` at the strict 8x CIC stencil bound.
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
    """Decode `tile_paint_int`'s raw mesh to counts/mean (no -1; see `tile_paint_f64`).

    Decode and division are f64; `fdtype` is the dtype of the returned field.
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
    """Read 3 tile fields with one shared CIC stencil (origin-shifted `cic_read_vector`).

    Zeroed on ~live; the result dtype follows the field. Returns ((n,3), n_out).
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
# per-tile coarse sub-block staging
# ===========================================================================
#
# Only a tile's owned (core) rows need the long force, so each tile stages the core's coarse
# cells plus a stencil halo, (T/COARSE_RATIO + 2*halo)^3, making the force path O(tile) in
# device memory. halo=2: TSC reads the nearest cell +-1 with a round()-based base, so a core-edge
# particle can reach one cell beyond a CIC-style bound.

COARSE_HALO = 2


def coarse_subblock_origin_extent(tijk, n_tile, n_coarse, n_fine, halo=COARSE_HALO):
    """(origin in coarse cells (3,) int, extent in cells) of one tile core's coarse sub-block.

    The origin may be negative and the block may run past the mesh; extraction wraps.
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
    """Extract the (extent,)*3 periodic sub-block at `origin_cells` (host numpy, wrapped per axis).

    One `np.ix_` gather rather than per-axis takes, which would materialize an
    (extent, n, n) intermediate per call.
    """
    g = np.asarray(g_coarse)
    o = np.asarray(origin_cells, dtype=np.int64)
    if o.shape != (3,) or g.ndim != 3:
        raise ValueError(f"want a 3-D mesh and a (3,) origin, got {g.shape} and {o.shape}")
    span = np.arange(int(extent), dtype=np.int64)
    return g[np.ix_(*((span + int(o[axis])) % g.shape[axis] for axis in range(3)))]


def stencil_violation_message(assign, lo_needed, hi_needed, extent):
    return (
        f"a row's {assign} stencil reaches outside the staged sub-block: needs "
        f"[{lo_needed}, {hi_needed}] of [0, {extent - 1}]. Pass only the tile's OWNED "
        "rows, or raise the halo -- reading past the block wraps to the far side of "
        "the mesh and is silent."
    )


def check_stencil_guard(guard_out):
    """Resolve deferred stencil bounds from `gather_coarse_subblock(guard_out=...)` and refuse.

    A device->host sync: call beside the force readback, before using the result.
    """
    for lo_needed, hi_needed, extent, assign in guard_out:
        lo_i, hi_i = int(lo_needed), int(hi_needed)
        if lo_i < 0 or hi_i >= extent:
            raise ValueError(stencil_violation_message(assign, lo_i, hi_i, extent))


def _stencil_bounds(i, m, first, n_w, n_coarse, extent):
    """Lowest and highest coarse cell any live row's stencil reaches, on device.

    Dead rows get sentinels that cannot win, so an all-padded tile raises nothing.
    """
    big = jnp.int32(int(n_coarse) + int(extent) + 1)
    i_lo = i if m is None else jnp.where(m, i, big)
    i_hi = i if m is None else jnp.where(m, i, -big)
    return jnp.min(i_lo) + first, jnp.max(i_hi) + first + n_w - 1


def gather_coarse_subblock(
    sub_x, sub_y, sub_z, positions, origin_cells, cell_coarse, n_coarse, assign="tsc",
    live=None, guard_out=None,
):
    """Read the long force for one tile's rows out of a staged sub-block.

    Bitwise identical to gathering from the global coarse mesh: weights come from the GLOBAL
    coordinate (a block-local coordinate changes the last bits of the fractions) and only the
    integer index is re-based, by the exact `(base - origin) mod n_coarse`. `live` masks padded
    rows (read a valid address with zero weight) so one compiled shape serves every tile. Rows
    whose stencil leaves the block would silently read wrapped values, so they are refused:
    immediately (a sync) if `guard_out` is None, else the bounds are appended to `guard_out` as
    device scalars for `check_stencil_guard` before the result is used.
    """
    # `.shape` and a direct int32 origin so the gather can be jitted with a traced origin
    extent = int(sub_x.shape[0])
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
    # exact integer re-basing, wrapping across the periodic boundary
    i = jnp.mod(base - jnp.asarray(origin_cells, dtype=jnp.int32), int(n_coarse))

    m = None if live is None else jnp.asarray(live)[:, None]
    lo_needed, hi_needed = _stencil_bounds(i, m, first, len(w_axis), n_coarse, extent)
    if guard_out is None:
        # eager refusal (device->host sync)
        lo_i, hi_i = int(lo_needed), int(hi_needed)
        if lo_i < 0 or hi_i >= extent:
            raise ValueError(stencil_violation_message(assign, lo_i, hi_i, extent))
    else:
        # deferred: checked by `check_stencil_guard` before the result is used
        guard_out.append((lo_needed, hi_needed, extent, assign))
    if m is not None:
        # padded rows: valid address, zero weight
        i = jnp.where(m, i, -first)
        w_axis = tuple(jnp.where(m, w, 0.0) for w in w_axis)

    fx, fy, fz = (jnp.asarray(s).reshape(-1) for s in (sub_x, sub_y, sub_z))
    n = xp.shape[0]
    dt = fx.dtype
    ax = jnp.zeros((n,), dtype=dt)
    ay = jnp.zeros((n,), dtype=dt)
    az = jnp.zeros((n,), dtype=dt)
    for dx, dy, dz in corners:
        # form the product at full precision and narrow once, matching `_tsc_corner_flat_weight`
        # (narrowing `w_axis` first breaks the bitwise contract)
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

    `one_tile(u, live, owned) -> (g, live & owned, n_out)`. One program serves every tile because
    the row capacity is fixed, so tile 0 carries the compile. Each tile+buffer is FFT'd as a
    periodic P^3 box with the same Nyquist as the global fine mesh, so tile and global grids differ
    only by periodization; images sit at >= P - R where the short force is erfc-suppressed.
    `paint`: "f64" (order-dependent, the reference default) or "int" (`tile_paint_int`,
    order-independent; the engine's choice). `fdtype` sets the tile mesh, kernels and gather.
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
    # numpy, not jax arrays: the traced program embeds them as constants either way, and a jax
    # array would also stay resident on jax's default device for the life of `one_tile`
    kers = split_kernels((P,) * 3, cell, "short", r_s=r_s, fdtype=fdtype)

    def one_tile(u, live, owned):
        # `owned` is supplied by the caller (exact integer ownership; see `owning_tile`)
        if paint == "int":
            mesh_i, n_out_p = tile_paint_int(u, live, (P,) * 3, cell, frac_bits)
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
    """The geometry dict `make_tile_force_fn` publishes (P, b_realized, cell, mean, core bounds, ...),
    without building kernels or tracing; `make_tile_force_fn` reads its own values from it.
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


# Cap on the test-only global accumulate sink of `force_short_tiled` (an O(N) array).
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

    `member_fn(tijk) -> int64 indices` and `cap` come from the caller (membership belongs to the
    layout). `paint`/`frac_bits`/`fdtype`: see `make_tile_force_fn`. `pad_fill`: "cycle" repeats
    member indices into the padding, "zero" points every pad row at particle 0 (same forces, more
    atomic contention). `sink=None` accumulates a global (n,3) array (tests only; refused above
    `max_accumulate_bytes`); otherwise `sink(idx, g_owned)` is called per tile with its owned rows.
    `diag["partition_ok"]` asserts every particle is owned by exactly one tile.
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
                "That array is O(N); pass a "
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
        # pad rows have zero weight either way; "zero" makes them contend for particle 0's cells
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
        # members outside the padded box; must be 0 (else the brick decomposition is wrong)
        n_overhang_total=int(n_overhang_total),
        # contract: every particle owned by exactly one tile
        n_owned_total=int((owner_count > 0).sum()),
        min_owner_count=int(owner_count.min()),
        max_owner_count=int(owner_count.max()),
        partition_ok=bool(
            owner_count.min() == 1 and owner_count.max() == 1
            and int((owner_count > 0).sum()) == n
        ),
    )
    return g_out, diag
