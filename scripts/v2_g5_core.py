"""v2 gate G5: the two-level force split -- kernels, tiles, and the arms.

Core physics for G5 (plan: seed V2a). Imported by v2_g5_two_level_force.py (the
gate orchestrator) and by the selftest in this file. Probe code, NOT package
code: promotion is a V4/M-v2-1 decision.

The split (JC, 2026-07-15). Long force S(k)*ik/k^2 on the global COARSE mesh;
short force (1-S(k))*ik/k^2 on FINE tiles, each tile+buffer FFT'd as a small
periodic box. S(k) = exp(-k^2 r_s^2). BOTH LEVELS ARE MESH-SOLVED: this is
PM-PM, not P3M and not TreePM -- the erfc form appears only as the analytic
prediction used to validate buffer scaling, never as a pair sum. The
coarse-global + fine-tiles decomposition is PMFAST's; only the matching moves
from a real-space polynomial to a Fourier kernel.

Why Fourier rather than PMFAST's polynomial matching: it makes the error
ATTRIBUTABLE. Two structural properties, both load-bearing:

  1. short := 1.0 - S, sharing ONE k2_safe with the long kernel, so
     long + short == ik/k^2 == the monolithic kernel BIT-EXACTLY, by
     construction rather than by exp/expm1 agreeing. Every nonzero number this
     gate reports is therefore a discretization error with exactly one named
     cause -- never a kernel-design error. (Floor F1.)
  2. Both kernels carry ik_j, which is EXACTLY zero at k=0, so the short force
     is invariant to the density's DC LEVEL -- a tile never has to subtract a
     mean at all. MEASURED 2026-07-15 at 32^3, not assumed: shifting delta by
     c = +1/-1/+137 moves the short force by 8.6e-16 / 7.0e-16 / 1.1e-13
     (pure roundoff, growing with c as it must).

     But read the claim precisely, because the obvious stronger version is
     FALSE and the same measurement shows it. A tile still needs the global
     mean as a NORMALIZATION SCALAR: delta = counts/mean - 1 is linear in
     1/mean, so a tile using its OWN mean does not leak a small DC error --
     it RESCALES the whole short force by mean_global/mean_tile. Measured
     error = |s - 1| EXACTLY (0.5 at s=0.5, 1.0 at s=2.0). That is an O(1)
     smooth per-tile multiplicative offset which would read as a catastrophic
     tiling failure and send you hunting buffers.

     This is benign only because the global mean is n_total/n_mesh^3 -- a
     CONFIG CONSTANT, not a data-dependent global reduction. Nothing is
     communicated and no global pass over particles is needed.

     So the tile paints RAW COUNTS / mean, with no -1: the DC term is killed
     by ik(0) = 0 (measured above), and the config scalar carries the
     normalization. That deletes the bug class rather than guarding against
     it.

Parameterization is dimensionless, one knob per error term, so C-dev results
transfer to C-gh/C-hero where cells do not:

    alpha = r_s / d_coarse   -> coarse representation error, exp(-pi^2 alpha^2)
    beta  = b   / r_s        -> buffer truncation error, erfc(beta/2)
    b_fine = 4 * alpha * beta                    (coarse:fine ratio is 4)

Global arms reuse painting.paint_f32(fdtype=f64) + cic_read_vector, so F0
tests the KERNEL alone. Only tile arms need probe-local CIC.
"""

import numpy as np

# Coarse:fine mesh ratio (PMFAST pattern; the config table's `coarse = fine/4`).
COARSE_RATIO = 4

# FFT-friendly padded sizes (products of small primes). P = T + 2*b_fine is
# rounded UP to one of these: T=128, b_fine=18 -> P=164 = 4*41 has a large
# prime factor and is pathologically slow. The realized beta is reported.
FFT_FRIENDLY = (
    32,
    36,
    40,
    48,
    50,
    54,
    60,
    64,
    72,
    80,
    90,
    96,
    100,
    108,
    120,
    128,
    144,
    150,
    160,
    162,
    180,
    192,
    200,
    216,
    240,
    250,
    256,
    288,
    300,
    320,
    360,
    384,
    400,
    432,
    480,
    500,
    512,
    # >512: needed by the cgh64 box ladder (n_fine=1024), whose tile512 arms
    # want P = 512 + 2*b_fine = 576/640/768/1024 for b_fine = 32/64/128/256.
    # The list stops at 1024 = the largest fine mesh in the config table
    # (cdev 512, cdev8 256, cgh64 1024); the degeneracy guard forbids P > n_fine
    # anyway. Entries <=512 are frozen -- they fix every already-measured tile
    # selection, so nothing is appended below 512.
    540,
    576,
    600,
    640,
    648,
    720,
    750,
    768,
    800,
    810,
    864,
    900,
    960,
    972,
    1000,
    1024,
)


# ===========================================================================
# split kernel
# ===========================================================================


def kernel_grids(shape, cell, fdtype=np.float64):
    """ik_j (low-rank) + k2_true + k2_safe on the rfftn half-grid of a box of
    `shape` cells of size `cell`.

    forces.k_components twin. That one takes a SCALAR n_mesh (cubic box, one
    kx reused for x and y) and cannot express a tile, whose padded box is
    non-cubic in general. Same convention otherwise: host-numpy build
    (precision island), low-rank ik_j of shape (nx,1,1)/(1,ny,1)/(1,1,nz//2+1)
    so they broadcast without materializing a full array.

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
    dimensionless knob and the measurement checks a curve rather than scanning
    blind. The honest limit, stated once: you cannot have both -- Gaussian in k
    means infinite real-space support (buffer truncation approximates);
    compact in r means an infinite k tail (worse coarse aliasing). Heisenberg,
    not a design failure. r_s = 0 -> S == 1 exactly (the F0 degenerate limit).
    """
    return np.exp(-k2_true * float(r_s) ** 2)


def split_factor(k2_true, r_s, which):
    """S (which="long"), 1-S (which="short"), or 1 (which="mono").

    short is LITERALLY 1.0 - S sharing the same S array, so long + short == 1
    identically in floating point and the F1 identity is structural. Do not
    "optimize" this into -expm1(-k^2 r_s^2): that is the cancellation-safe
    form for r_s large relative to the box (at C-dev k_min^2 r_s^2 ~ 0.048, so
    f64 loses ~1.4 of 16 digits -- a non-issue here, real at C-hero), but it
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


def cic_match_factor(shape, cell_solve, cell_target, clip=None, order_solve=2, order_target=2):
    """(W(k, cell_target) / W(k, cell_solve))^2 on the rfftn half-grid.

    Hockney-Eastwood kernel matching. The long arm paints and gathers on the
    COARSE cell and the short arm on the FINE cell, so their CIC windows differ
    and the sum != the monolithic force EVEN WITH A MATHEMATICALLY PERFECT
    SPLIT. Paint applies W once and gather applies W once -> W^2 per level,
    hence the square.

    MANDATORY FOR THE GAUSSIAN FAMILY, HARMFUL FOR THE WINDOWED ONES. This is
    family-specific and an earlier version of this docstring wrongly stated it
    as universal. Measured 2026-07-15 (n_fine=128, n_coarse=32), coarse-arm
    error with vs without matching:

        gauss   alpha=1.0        9.6e-3 -> 6.1e-3   (matching HELPS, 1.6x)
        gauss   alpha=2.0        1.3e-3 -> 9.7e-4   (helps, 1.35x)
        compact r_out=4 coarse   5.8e-2 -> 2.3e-1   (matching HURTS, 4x)
        compact r_out=2 coarse   1.2e-1 -> 3.4e-1   (hurts, 2.8x)

    Why: the factor W_f^2/W_c^2 grows without bound toward the coarse Nyquist
    and is only harmless where the long kernel is ALREADY small. S(k) guarantees
    that (S ~ 5e-5 at the coarse Nyquist at alpha=1); the windowed families'
    long kernels carry real high-k content there (the real-space hole edge), so
    matching amplifies exactly the modes that should be suppressed. Clipping at
    10 does not rescue it.

    For the Gaussian family the analytic case is strong (verified numerically
    2026-07-15): the S-weighted mismatch S*(1 - Wc^2/Wf^2) is 5.4% at k=1
    against a 2.1% PM floor, and 0.9% at k=0.25 against a 0.13% floor -- the
    UNMATCHED coarse arm misses the D-v2-1 band by 2.6x at k=1 and 7.1x at
    k=0.25, i.e. WORST AT LOW k, which is where nobody would look.

    diagnostics.cic_window cannot be reused: it derives the cell from n_mesh,
    and this needs W(FINE cell) evaluated on the COARSE k-grid.

    `clip` bounds the returned factor. W_f^2/W_c^2 blows up near the coarse
    Nyquist (~5x on-axis, worse at cube corners) and is harmless ONLY because
    S ~ 5e-5 there; without a clip a corner mode with S ~ 1e-13 gets multiplied
    by ~200. Returns (factor, max_applied) so the guard is reported, not
    silent.
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


def coarse_leak_prediction(delta_k_fine, shape_fine, cell_fine, n_coarse, r_s):
    """Predicted C5 residual: the S-weighted long-force power OUTSIDE the
    coarse Nyquist cube, computed independently from the FINE delta_k.

    C5 (fine paint -> restrict -> coarse solve -> prolongate -> fine gather)
    removes every window and aliasing term, so its only surviving error is the
    long-force power the coarse mesh cannot represent at all. That residual has
    a PREDICTABLE value, and measured-vs-predicted agreement CONFIRMS the
    attribution instead of asserting it. Returns the predicted rms of the
    missing long force, in the same units as the gathered force.
    """
    ikx, iky, ikz, k2_true, k2_safe = kernel_grids(shape_fine, cell_fine, np.float64)
    s = s_of_k(k2_true, r_s)
    nx, ny, nz = (int(v) for v in shape_fine)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=cell_fine)
    kz = 2.0 * np.pi * np.fft.rfftfreq(nz, d=cell_fine)
    k_nyq_c = np.pi / (cell_fine * COARSE_RATIO)
    outside = (
        (np.abs(kx).reshape(nx, 1, 1) > k_nyq_c)
        | (np.abs(kx).reshape(1, ny, 1) > k_nyq_c)
        | (np.abs(kz).reshape(1, 1, nz // 2 + 1) > k_nyq_c)
    )
    amp2 = 0.0
    for ik in (ikx, iky, ikz):
        gk = s * ik * delta_k_fine / k2_safe
        amp2 += float(np.sum(np.abs(np.where(outside, gk, 0.0)) ** 2))
    return float(np.sqrt(amp2)) / (nx * ny * nz)


# ===========================================================================
# compact real-space split (the PMFAST-lineage arm)
# ===========================================================================
#
# WHY THIS ARM EXISTS -- measured 2026-07-15, and it is the whole reason the
# Gaussian is not the only kernel in this gate.
#
# The Gaussian split recombines EXACTLY (F1 ~ 4e-16) but TILES BADLY. Its short
# kernel (1-S)*ik/k^2 tends to ik/k^2 at high k, so it is NOT compactly
# supported in real space and is sharply truncated at the fine Nyquist; its
# real-space tails ring like ~1/r, and a tile's period-P image sum differs from
# the global period-L sum by ~1/P. Measured at n_fine=128, T=16..64, b=8..24,
# alpha=0.5..2.0:
#
#     rms|e|/rms|g_short| ~= 2.0 / P     (P = padded tile size in fine cells)
#
# confirmed by THREE independent signatures, all pointing the same way:
#   1. decay law    -- error x P = 2.00/2.01/2.33/2.12/1.95/1.96, flat; NOT the
#                      erfc(beta/2) the buffer model predicts (which spans 600x
#                      over the same scan).
#   2. spatial      -- edge/interior error ratio 1.14, i.e. UNIFORM across the
#                      tile core. Truncation would be edge-concentrated.
#   3. r_s sweep    -- error flat at 3.3-3.5e-2 while erfc(b/2r_s) spans 1.5e-8
#                      to 1.6e-1 (SEVEN orders) at fixed P.
# plus the exact limit: P = n_fine -> error 7e-15 (the periodization difference
# vanishes when the tile IS the box).
#
# So for the Gaussian the BUFFER IS NOT A KNOB; only the tile size is -- exactly
# backwards for a design whose point is that tiles are small. This is why PMFAST
# matched in real space: a compactly supported short kernel has NO images to sum,
# so tiling is EXACT rather than asymptotic. The cost moves to the coarse arm
# (1-W is less smooth than S, so the long kernel is harder to represent on the
# coarse mesh). That is the tradeoff G5 measures -- exact recombination vs exact
# tiling -- rather than assuming either.


def radius_grid(shape, cell):
    """Periodic min-image radius from the origin, on a real-space grid.

    The kernel lives at r = 0 (index 0) and the grid is periodic, so cell i
    along an axis is at min(i, n-i) -- the same min-image the FFT convolution
    implies.
    """
    nx, ny, nz = (int(s) for s in shape)
    ax = np.minimum(np.arange(nx), nx - np.arange(nx)) * float(cell)
    ay = np.minimum(np.arange(ny), ny - np.arange(ny)) * float(cell)
    az = np.minimum(np.arange(nz), nz - np.arange(nz)) * float(cell)
    return np.sqrt(ax[:, None, None] ** 2 + ay[None, :, None] ** 2 + az[None, None, :] ** 2)


def compact_window(r, r_in, r_out):
    """Quintic smoothstep W: 1 for r <= r_in, 0 for r >= r_out, C2 at both ends.

    W is the SHORT kernel's real-space taper, so the short force has support
    exactly r_out and a tile with buffer >= r_out has no images to sum. The
    long kernel carries (1 - W), which is zero inside r_in -- the "hole" whose
    edge sets how much k-space content the long kernel has, and therefore how
    well the COARSE mesh can represent it. Quintic (C2) rather than cubic
    because that hole edge is the long arm's whole difficulty: a kink there
    puts power above the coarse Nyquist, which is precisely the aliasing the
    coarse arm is trying to avoid.
    """
    x = np.clip((r - float(r_in)) / (float(r_out) - float(r_in)), 0.0, 1.0)
    return 1.0 - (6.0 * x**5 - 15.0 * x**4 + 10.0 * x**3)


def compact_stencil(n_ref, cell, r_in, r_out, r_s=None):
    """Build the compactly supported short-range force stencil ONCE -> 3 arrays
    of shape (2R+1,)^3, offset-indexed from -R..R (R = ceil(r_out/cell)).

    THE SUBTLETY THAT MAKES THIS FUNCTION EXIST (measured 2026-07-15). The
    obvious implementation -- window `irfftn(ik/k^2)` separately on each box --
    does NOT tile exactly: it gives 5e-3..2e-2, not roundoff. irfftn(ik/k^2) on
    a box of size P returns the P-PERIODIZED continuum kernel, and on the global
    box the N-periodized one. Those two already differ INSIDE the support
    radius (by the image sums ~1/P^2), so windowing cannot remove the
    difference -- each box would be convolving with a subtly different operator.

    So the stencil is built once on the reference (global fine) grid, where it
    defines the operator the monolithic reference actually uses, and then
    EMBEDDED into each tile box by embed_stencil. Same numbers, any box.
    """
    n_ref = int(n_ref)
    ikx, iky, ikz, k2_true, k2_safe = kernel_grids((n_ref,) * 3, cell, np.float64)
    w = compact_window(radius_grid((n_ref,) * 3, cell), r_in, r_out)
    # r_s=None -> window the MONO kernel (the PMFAST-lineage "compact" family).
    # r_s set  -> window the GAUSSIAN SHORT kernel (the "gauss_compact" hybrid):
    # the tails being cut are then only the band-limit ringing, because
    # (1-S)*ik/k^2 already decays like erfc, so the long kernel stays ~S*ik/k^2
    # and keeps S's anti-aliasing rolloff -- the property the pure-compact
    # family lacks and pays 10x in the coarse arm for.
    pre = 1.0 if r_s is None else split_factor(k2_true, r_s, "short")
    R = int(np.ceil(float(r_out) / float(cell)))
    if 2 * R + 1 > n_ref:
        raise ValueError(f"stencil radius {R} does not fit in the reference grid {n_ref}")
    off = np.arange(-R, R + 1)
    ii = np.mod(off, n_ref)
    out = []
    for ik in (ikx, iky, ikz):
        k_real = np.fft.irfftn(pre * ik / k2_safe, s=(n_ref,) * 3) * w
        out.append(k_real[np.ix_(ii, ii, ii)].copy())
    return tuple(out), R


def embed_stencil(stencil, R, shape, cell, r_out):
    """Place a compact stencil into a `shape` box and rfftn -> half-grid kernels.

    Exact for any box with min(shape) > 2R+1: the stencil's periodic images do
    not overlap it, so the SAME operator acts on every tile and on the global
    mesh. This is what buys exact tiling, and it is the property the Gaussian
    family structurally cannot have (its short kernel -> ik/k^2 at high k, so it
    has no compact support at all and tiles as ~2.0/P).
    """
    nx, ny, nz = (int(s) for s in shape)
    if min(nx, ny, nz) < 2 * R + 1:
        raise ValueError(
            f"box {min(nx, ny, nz)} < stencil diameter {2 * R + 1}: the compact kernel's "
            f"images would overlap (r_out={r_out}). Raise the tile/buffer or lower r_out."
        )
    off = np.arange(-R, R + 1)
    out = []
    for s in stencil:
        big = np.zeros((nx, ny, nz), dtype=np.float64)
        big[np.ix_(np.mod(off, nx), np.mod(off, ny), np.mod(off, nz))] = s
        out.append(np.fft.rfftn(big))
    return tuple(out)


def compact_kernels(shape, cell, r_in, r_out, which, n_ref, r_s=None):
    """Real-space-windowed split kernels on the rfftn half-grid of `shape`.

    K_short = rfftn(embed(stencil))    -- support exactly r_out, box-independent
    K_long  = ik_j/k^2 - K_short       -- so long + short == mono EXACTLY

    `n_ref` is the grid the stencil is defined on (the global fine mesh): the
    operator must be one object shared by every box -- see compact_stencil.
    """
    stencil, R = compact_stencil(n_ref, cell, r_in, r_out, r_s=r_s)
    k_short = embed_stencil(stencil, R, shape, cell, r_out)
    if which == "short":
        return k_short
    ikx, iky, ikz, _, k2_safe = kernel_grids(shape, cell, np.float64)
    mono = tuple(ik / k2_safe for ik in (ikx, iky, ikz))
    if which == "mono":
        return mono
    if which == "long":
        return tuple(m - s for m, s in zip(mono, k_short))
    raise ValueError(f"which must be 'mono', 'long' or 'short', got {which!r}")


# ===========================================================================
# global arms (reuse the package CIC; only the kernel varies)
# ===========================================================================


def assignment_window(shape, cell, order):
    """W(k) = prod_i sinc^p(k_i cell / 2) on the rfftn half-grid; p = order.

    order=2 -> CIC (painting.py's stencil), order=3 -> TSC. diagnostics.
    cic_window is the p=2 special case but derives the cell from n_mesh, so it
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


def _tsc_pieces(pos, n_mesh, cell):
    """TSC base cell and the 3 per-axis weights (offsets -1, 0, +1).

    Triangular Shaped Cloud: a 3-point stencil per axis centred on the NEAREST
    cell (CIC uses 2 points anchored at floor). Its window is sinc^3 rather than
    sinc^2, so it suppresses the high-k power that a coarse mesh would otherwise
    alias down into the band -- which is why it is worth a leg here.
    """
    import jax
    import jax.numpy as jnp

    s = pos / float(cell)
    base = jnp.round(s)
    d = s - base
    base = jax.lax.stop_gradient(base).astype(jnp.int32)
    w_m = 0.5 * (0.5 - d) ** 2
    w_0 = 0.75 - d**2
    w_p = 0.5 * (0.5 + d) ** 2
    return base, (w_m, w_0, w_p)


def paint_tsc_f64(pos, n_mesh, box_size, n_total):
    """Global periodic TSC paint -> delta (n_mesh^3) f64. Probe-local."""
    import jax.numpy as jnp

    N = int(n_mesh)
    cell = float(box_size) / N
    base, w = _tsc_pieces(pos, N, cell)
    mesh = jnp.zeros((N**3,), dtype=jnp.float64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                ix = (base[:, 0] + dx) % N
                iy = (base[:, 1] + dy) % N
                iz = (base[:, 2] + dz) % N
                ww = w[dx + 1][:, 0] * w[dy + 1][:, 1] * w[dz + 1][:, 2]
                flat = (ix * N + iy) * N + iz
                mesh = mesh.at[flat].add(ww, mode="promise_in_bounds")
    mean = float(n_total) / float(N) ** 3
    return mesh.reshape(N, N, N) / mean - 1.0


def tsc_read_vector(gx, gy, gz, pos, n_mesh, box_size):
    """Global periodic TSC gather of 3 fields with one shared stencil."""
    import jax.numpy as jnp

    N = int(n_mesh)
    cell = float(box_size) / N
    base, w = _tsc_pieces(pos, N, cell)
    fx, fy, fz = gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)
    n = pos.shape[0]
    ax = jnp.zeros((n,), dtype=gx.dtype)
    ay = jnp.zeros((n,), dtype=gx.dtype)
    az = jnp.zeros((n,), dtype=gx.dtype)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                ix = (base[:, 0] + dx) % N
                iy = (base[:, 1] + dy) % N
                iz = (base[:, 2] + dz) % N
                ww = w[dx + 1][:, 0] * w[dy + 1][:, 1] * w[dz + 1][:, 2]
                flat = (ix * N + iy) * N + iz
                ax = ax + ww * fx[flat]
                ay = ay + ww * fy[flat]
                az = az + ww * fz[flat]
    return jnp.stack([ax, ay, az], axis=1)


def density_f64(pos, n_mesh, box_size, n_total):
    """delta on an n_mesh^3 grid in f64, via the PACKAGE CIC stencil.

    painting.paint_f32 takes an fdtype (painting.py:49) even though
    density_contrast never passes one, so the f64 mesh costs no new paint code
    and the global arms differ from make_force_fn ONLY in the kernel. That is
    what makes floor F0 a test of the kernel rather than of the probe.
    """
    import jax.numpy as jnp

    from inexor.painting import paint_f32

    counts = paint_f32(pos, n_mesh, box_size, fdtype=jnp.float64)
    mean = float(n_total) / float(n_mesh) ** 3
    return counts / mean - 1.0


def split_kernels(shape, cell, which, family, r_s=None, r_in=None, r_out=None, n_ref=None):
    """The (Kx, Ky, Kz) half-grid kernels for either split family.

    family="gauss"   -> split_factor(S) * ik/k^2   : recombines exactly, tiles
                        as ~2.0/P (kernel ringing -- see the section header).
    family="compact" -> real-space-windowed        : tiles exactly for
                        buffer >= r_out, at a coarse-representation cost.

    ONE entry point so the two families are measured by identical code paths and
    the comparison is like-for-like rather than an artifact of two harnesses.
    """
    if family == "gauss":
        ikx, iky, ikz, k2_true, k2_safe = kernel_grids(shape, cell, np.float64)
        fac = split_factor(k2_true, 0.0 if r_s is None else r_s, which)
        return tuple((fac / k2_safe) * ik for ik in (ikx, iky, ikz))
    if family == "compact":
        if which == "mono":
            ikx, iky, ikz, _, k2_safe = kernel_grids(shape, cell, np.float64)
            return tuple(ik / k2_safe for ik in (ikx, iky, ikz))
        return compact_kernels(shape, cell, r_in, r_out, which, n_ref)
    if family == "gauss_compact":
        if which == "mono":
            ikx, iky, ikz, _, k2_safe = kernel_grids(shape, cell, np.float64)
            return tuple(ik / k2_safe for ik in (ikx, iky, ikz))
        return compact_kernels(shape, cell, r_in, r_out, which, n_ref, r_s=r_s)
    raise ValueError(f"family must be 'gauss', 'compact' or 'gauss_compact', got {family!r}")


def force_global(
    pos,
    n_mesh,
    box_size,
    n_total,
    which,
    family="gauss",
    r_s=None,
    r_in=None,
    r_out=None,
    n_ref=None,
    match=None,
    clip=None,
    assign="cic",
    pos_gather=None,
):
    """One global mesh solve of the chosen split kernel -> ((n,3) f64, max_match).

    which="mono" reproduces forces.make_force_fn's kernel exactly for either
    family (floor F0). match=(cell_solve, cell_target) applies cic_match_factor.

    Sequential per-component solves: never materialize three force meshes at
    once (forces.py's Sec. 9 rule -- the memory budget depends on it).

    pos_gather splits the SOURCE of the field from the point it is READ AT.
    Default None means gather at `pos`, which is the only behaviour any existing
    caller sees. It exists for the frozen-background test: the long-range field
    can be built once per step from the analytic LPT positions -- known for
    every particle at every time without evolving anything -- while each tile
    still reads its own particles' force at their TRUE positions. That is what
    would let tiles run independently again, since a field that depends on no
    evolved state can be precomputed for the whole schedule up front.
    """
    import jax.numpy as jnp

    from inexor.painting import cic_read_vector

    cell = box_size / n_mesh
    kers = split_kernels(
        (n_mesh,) * 3,
        cell,
        which,
        family,
        r_s=r_s,
        r_in=r_in,
        r_out=r_out,
        n_ref=(n_mesh if n_ref is None else n_ref),
    )
    max_applied = 1.0
    if match is not None:
        mf, max_applied = cic_match_factor((n_mesh,) * 3, match[0], match[1], clip=clip)
        kers = tuple(k * mf for k in kers)
    if assign == "cic":
        delta = density_f64(pos, n_mesh, box_size, n_total)
    elif assign == "tsc":
        delta = paint_tsc_f64(pos, n_mesh, box_size, n_total)
    else:
        raise ValueError(f"assign must be 'cic' or 'tsc', got {assign!r}")
    dk = jnp.fft.rfftn(delta)
    g = [jnp.fft.irfftn(dk * jnp.asarray(k), s=(n_mesh,) * 3) for k in kers]
    rd = pos if pos_gather is None else jnp.asarray(pos_gather)
    if assign == "cic":
        out = cic_read_vector(g[0], g[1], g[2], rd, n_mesh, box_size)
    else:
        out = tsc_read_vector(g[0], g[1], g[2], rd, n_mesh, box_size)
    return np.asarray(out, dtype=np.float64), max_applied


# ===========================================================================
# tile geometry + probe-local CIC (origin-shifted, no global wrap)
# ===========================================================================


def padded_size(n_tile, b_fine, n_fine=None):
    """FFT-friendly P >= n_tile + 2*b_fine, and the realized buffer.

    Returns (P, b_realized). The buffer GROWS to the rounded size (the extra is
    real buffer, not padding), so beta_realized >= beta_requested and is what
    gets reported. Rounding matters: T=128, b=18 -> want=164 = 4*41, a large
    prime factor and pathologically slow.

    `n_fine` enables the degeneracy guard: a padded tile at least as large as
    the global mesh is not a tile at all -- it does more FFT work than the
    monolithic solve it is supposed to replace, and (with the brick wrap) is
    where the double-count bug lives.
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

    origin = (t*n_tile - b_fine) * cell, and may be NEGATIVE -- that is fine
    and deliberate: mod(x - origin, L) handles the periodic wrap exactly, so a
    tile whose buffer crosses the box boundary needs no special case.
    """
    t = np.asarray(tijk, dtype=np.int64)
    origin = (t * int(n_tile) - int(b_fine)) * float(cell)
    extent = (int(n_tile) + 2 * int(b_fine)) * float(cell)
    return origin, extent


def tile_local_coords(pos, origin, box_size):
    """u = mod(pos - origin, L): the tile-local coordinate AND the membership
    test, from ONE expression.

    A particle is in tile+buffer iff all(u < extent). This is exact including
    buffers that wrap the periodic boundary -- no min-image, no branches.
    painting._cic_pieces' `% n_mesh` is WRONG here: it would fold a particle
    from the far side of the box into the tile.
    """
    import jax.numpy as jnp

    return jnp.mod(pos - jnp.asarray(origin), float(box_size))


def _tile_cic_pieces(u, live, shape, cell):
    """Base cell, fractional offset, validity mask, and the out-of-box count.

    THE MODULO QUESTION, settled by measurement (2026-07-15). The padded tile
    box IS periodic -- that is exactly what its rfftn assumes -- so the tile
    paint must wrap modulo P, the PADDED BOX's own period. What would be wrong
    is painting.py's `% n_mesh`, the GLOBAL box's period, which folds far-side
    particles into the tile. Those are different moduli, and an earlier version
    of this file conflated "not the global modulo" with "no modulo": it required
    base < P-1, silently discarding the LAST CELL LAYER of every padded box.
    That produced a 4.6e-1 error on the one-tile-equals-whole-box identity, and
    a fake buffer-error plateau that looked exactly like risk R7's ringing
    signature. The identity check is what caught it.

    So `ok` selects particles INSIDE the padded box (u in [0, extent)), and the
    corner indices wrap modulo P. n_out counts live particles outside the padded
    box: those are the brick-union superset OVERHANG, correctly excluded, an
    efficiency diagnostic and not an error.
    """
    import jax
    import jax.numpy as jnp

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
    """One CIC corner: flat index (wrapped mod P) + weight, both no-op'd on ~ok.

    Out-of-box and padding slots are sent to a VALID index with ZERO weight, so
    mode="promise_in_bounds" is honestly safe. painting.py's contract holds only
    because of its modulo; out-of-range indices under that mode are undefined
    behaviour on GPU -- silent corruption, not an exception (G1 job 33: 63% of
    cells wrong from a lowering that "worked"). mode="drop" is REJECTED: it
    would silently swallow a genuinely misrouted live particle, the exact bug
    class this gate must be able to see.
    """
    import jax.numpy as jnp

    nx, ny, nz = shape
    dx, dy, dz = corner
    wx = frac[:, 0] if dx else wlo[:, 0]
    wy = frac[:, 1] if dy else wlo[:, 1]
    wz = frac[:, 2] if dz else wlo[:, 2]
    ix = (base[:, 0] + dx) % nx
    iy = (base[:, 1] + dy) % ny
    iz = (base[:, 2] + dz) % nz
    flat = (ix * ny + iy) * nz + iz
    return jnp.where(ok, flat, 0), jnp.where(ok, wx * wy * wz, 0.0)


def tile_paint_f64(u, live, shape, cell, mean):
    """CIC paint of tile-local coords into ONE padded tile mesh -> (mesh, n_out).

    Paints counts/mean, with NO -1. Two separate facts, both measured (see the
    module docstring):
      - the -1 is unnecessary: ik(0) = 0 kills the DC term exactly, so the
        short force cannot see the offset (8.6e-16 for a shift of 1);
      - `mean` is MANDATORY and must be the GLOBAL mean n_total/n_mesh^3 -- a
        config scalar, not a reduction. A tile's own mean rescales the short
        force by mean_global/mean_tile, an error of exactly |s-1| (O(1)), which
        reads as a catastrophic tiling failure.

    THE SAFETY CONTRACT. painting.py's mode="promise_in_bounds" is safe ONLY
    because of its `% n_mesh`; out-of-range indices under that mode are
    undefined behaviour on GPU (silent corruption, NOT an exception -- G1's
    job 33 precedent: 63% of cells wrong from a lowering that "worked"). So
    every index is made in-range BY CONSTRUCTION:
        flat = where(ok, flat, 0)   -> a VALID index
        w    = where(ok, w,  0.0)   -> with zero weight: a provable no-op
    mode="drop" is REJECTED: it would silently swallow a genuinely misrouted
    live particle, which is exactly the bug class this gate must be able to
    see. n_out counts live-but-out-of-range particles and is a CONTRACT (must
    be 0), not a diagnostic.
    """
    import jax.numpy as jnp

    nx, ny, nz = (int(s) for s in shape)
    base, frac, ok, n_out = _tile_cic_pieces(u, live, (nx, ny, nz), cell)
    mesh = jnp.zeros((nx * ny * nz,), dtype=jnp.float64)
    wlo = 1.0 - frac
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                flat, w = _tile_corner(base, frac, wlo, (dx, dy, dz), (nx, ny, nz), ok)
                mesh = mesh.at[flat].add(w.astype(jnp.float64), mode="promise_in_bounds")
    return mesh.reshape(nx, ny, nz) / float(mean), n_out


def choose_brick(n_tile, b_fine, n_fine):
    """Largest brick size dividing BOTH n_tile and n_fine, with brick <= b_fine.

    Bricks are the host-side bucket grid. Bucketing on TILES and gathering the
    27 neighbours would give a 27x superset of which only ~3.4x is live -- an
    8x wasted paint. Bucketing on bricks makes the union of a tile's bricks a
    TIGHT superset of tile+buffer. b_fine = 0 -> brick = n_tile (the tile is
    its own bucket).
    """
    if int(b_fine) <= 0:
        return int(n_tile)
    best = 1
    for c in range(1, int(b_fine) + 1):
        if int(n_tile) % c == 0 and int(n_fine) % c == 0:
            best = c
    return best


def brick_buckets(pos_np, n_fine, n_brick, cell):
    """CSR-style bucket index over bricks -> (order, starts, nb).

    np.argsort(kind="stable") -- O(N log N), ONE pass, host-side and OUTSIDE
    the reported wall. The reported wall is the DEVICE force evaluation; the
    engine's real bucketing cost is an M-v2-2 measurement, not this gate's
    (the design study: "cell bucketing and per-tile fine meshes fight XLA
    static shapes", an M-v2-2 build item). Bucketing host-side is also what a
    streamed engine actually does, and it is what makes the O(tile) capacity
    claim true: the device never sees the global position array.
    """
    nb = int(n_fine) // int(n_brick)
    b = np.floor(np.asarray(pos_np, dtype=np.float64) / (float(cell) * int(n_brick)))
    b = np.mod(b.astype(np.int64), nb)
    bid = (b[:, 0] * nb + b[:, 1]) * nb + b[:, 2]
    order = np.argsort(bid, kind="stable")
    starts = np.searchsorted(bid[order], np.arange(nb**3 + 1))
    return order, starts, nb


def brick_span(n_tile, b_fine, n_brick, nb):
    """Bricks per side covering tile+buffer, with the wrap guard.

    MEASURED BUG 2026-07-15 (this is why the guard exists): tile_members walks
    bricks by MODULAR index, so once span > nb the same brick is concatenated
    more than once and its particles are painted TWICE. At n_fine=64, n_tile=32,
    b=20 that gave span=6 > nb=4 and a 3.29 RELATIVE short-force error -- silent
    density corruption that looks like a catastrophic tiling failure. Refuse it.
    """
    pad = int(np.ceil(float(b_fine) / float(n_brick)))
    span = int(n_tile) // int(n_brick) + 2 * pad
    if span > nb:
        raise ValueError(
            f"brick span {span} > brick grid {nb}: tile+buffer wraps the box and would "
            f"double-count bricks (n_tile={n_tile}, b={b_fine}, n_brick={n_brick}). "
            "The buffer is too large for this box -- reduce beta or raise n_fine."
        )
    return pad, span


def tile_members(order, starts, nb, tijk, n_tile, b_fine, n_brick):
    """Global particle indices in the brick union covering tile+buffer.

    The union is a TIGHT superset of tile+buffer: exact when n_brick divides
    b_fine, else larger by one brick per side. Periodic in brick index, guarded
    against the wrap-double-count above. Members outside the padded mesh are the
    superset OVERHANG and are correctly dropped by the in-range mask -- that is
    healthy, not an error (an earlier `n_dropped == 0` contract fired on it).
    """
    pad, span = brick_span(n_tile, b_fine, n_brick, nb)
    lo = np.asarray(tijk, dtype=np.int64) * (int(n_tile) // int(n_brick)) - pad
    idx = []
    for i in range(span):
        bi = (lo[0] + i) % nb
        for j in range(span):
            bj = (lo[1] + j) % nb
            for k in range(span):
                bk = (lo[2] + k) % nb
                bid = (bi * nb + bj) * nb + bk
                idx.append(order[starts[bid] : starts[bid + 1]])
    return np.concatenate(idx) if idx else np.empty(0, dtype=np.int64)


def tile_capacity(order, starts, nb, tiles, n_tile, b_fine, n_brick, align=1024):
    """max_t |members|, rounded up to `align`, over ALL tiles BEFORE the device
    loop -> (cap, pad_frac).

    Computed host-side so cap is EXACT: no guessed factor, no retry-on-overflow
    path, and ONE compiled program for every tile (64 recompiles at the pivot
    otherwise). pad_frac is the clustering-variance tax and belongs in the SU
    column.
    """
    ms = [len(tile_members(order, starts, nb, t, n_tile, b_fine, n_brick)) for t in tiles]
    m_max = int(max(ms))
    cap = int(np.ceil(m_max / float(align)) * align)
    return cap, float(1.0 - np.mean(ms) / cap)


def tile_gather_vector(gx, gy, gz, u, live, shape, cell):
    """Read 3 tile fields with ONE shared CIC stencil (cic_read_vector twin,
    origin-shifted). Zeroed on ~live. Returns ((n,3) f64, n_out)."""
    import jax.numpy as jnp

    nx, ny, nz = (int(s) for s in shape)
    base, frac, ok, n_out = _tile_cic_pieces(u, live, (nx, ny, nz), cell)
    fx, fy, fz = gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)
    n = u.shape[0]
    ax = jnp.zeros((n,), dtype=jnp.float64)
    ay = jnp.zeros((n,), dtype=jnp.float64)
    az = jnp.zeros((n,), dtype=jnp.float64)
    wlo = 1.0 - frac
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                flat, w = _tile_corner(base, frac, wlo, (dx, dy, dz), (nx, ny, nz), ok)
                ax = ax + w * fx[flat]
                ay = ay + w * fy[flat]
                az = az + w * fz[flat]
    return jnp.stack([ax, ay, az], axis=1), n_out


# ===========================================================================
# the tiled short arm
# ===========================================================================


def force_short_tiled(
    pos_np,
    n_fine,
    box_size,
    n_total,
    n_tile,
    b_fine,
    r_s=None,
    family="gauss",
    r_in=None,
    r_out=None,
    peak=None,
):
    """(1-S(k))*ik/k^2 on fine tiles, host-accumulated -> (g (n,3) f64, diag).

    Each tile+buffer is FFT'd as a small PERIODIC box of P^3 fine cells. The
    tile k-grid has fundamental 2pi/(P*d_f) but the SAME Nyquist pi/d_f as the
    global mesh, so the two grids periodize the identical sharp-k-truncated
    continuum kernel and differ ONLY by periodization.

    Why the wrap SHOULD be harmless: for a core target every source within the
    short kernel's range R ~ beta*r_s is present if b >= R, and images sit at
    >= P - R >= T + R where the short force is erfc-suppressed far below the
    truncation level. DO NOT BELIEVE THAT WITHOUT THE MEASUREMENT (risk R7):
    sharp-k truncation at the fine Nyquist gives the real-space kernel
    oscillatory ~1/x ringing tails, and if those dominate the image sums the
    periodization error decays as a POWER LAW in P, not erfc -- which would
    make `buffer ~ 5 r_s` and every cost number in this gate wrong. The
    orchestrator's decay-law / seam-profile / T-independence signatures
    discriminate the two.

    THE HOST ACCUMULATOR IS DELIBERATE. Each tile writes its owned rows into a
    numpy (n,3) and the device array is dropped, so the device never holds an
    O(box) output and `peak` measures exactly the tile arm. The (n,3) host
    accumulator is O(box) and that is honest -- it is the state a streamed
    engine holds anyway, and it is reported separately as host RSS.
    """
    import jax
    import jax.numpy as jnp

    cell = float(box_size) / int(n_fine)
    mean = float(n_total) / float(n_fine) ** 3
    P, b_real = padded_size(n_tile, b_fine, n_fine=n_fine)
    n_side = int(n_fine) // int(n_tile)
    tiles = [(i, j, k) for i in range(n_side) for j in range(n_side) for k in range(n_side)]

    n_brick = choose_brick(n_tile, b_real, n_fine)
    order, starts, nb = brick_buckets(pos_np, n_fine, n_brick, cell)
    cap, pad_frac = tile_capacity(order, starts, nb, tiles, n_tile, b_real, n_brick)

    kers = split_kernels(
        (P,) * 3, cell, "short", family, r_s=r_s, r_in=r_in, r_out=r_out, n_ref=n_fine
    )
    kers = [jnp.asarray(k) for k in kers]

    core_lo = b_real * cell
    core_hi = (b_real + int(n_tile)) * cell

    def one_tile(u, live):
        """ONE jitted program, reused for every tile (cap is fixed)."""
        delta, n_out_p = tile_paint_f64(u, live, (P,) * 3, cell, mean)
        dk = jnp.fft.rfftn(delta)
        g = [jnp.fft.irfftn(dk * k, s=(P,) * 3) for k in kers]
        out, n_out_g = tile_gather_vector(g[0], g[1], g[2], u, live, (P,) * 3, cell)
        owned = live & jnp.all((u >= core_lo) & (u < core_hi), axis=1)
        return out, owned, n_out_p + n_out_g

    one_tile_jit = jax.jit(one_tile)

    g_out = np.zeros((pos_np.shape[0], 3), dtype=np.float64)
    owner_count = np.zeros((pos_np.shape[0],), dtype=np.int32)
    n_overhang_total = 0
    peak_first = None

    for ti, t in enumerate(tiles):
        idx = tile_members(order, starts, nb, t, n_tile, b_real, n_brick)
        m = len(idx)
        if m > cap:
            raise RuntimeError(f"tile {t}: {m} members > cap {cap} (host capacity is wrong)")
        idx_pad = np.zeros((cap,), dtype=np.int64)
        idx_pad[:m] = idx
        live_np = np.zeros((cap,), dtype=bool)
        live_np[:m] = True
        origin, _ = tile_origin_extent(t, n_tile, b_real, cell)
        u = jnp.mod(jnp.asarray(pos_np[idx_pad]) - jnp.asarray(origin), float(box_size))
        out, owned, n_out = one_tile_jit(u, jnp.asarray(live_np))
        out = np.asarray(out)
        owned = np.asarray(owned)
        n_overhang_total += int(n_out)
        sel = owned[:m]
        g_out[idx[sel]] = out[:m][sel]
        owner_count[idx[sel]] += 1
        if ti == 0 and peak is not None:
            peak_first = peak()
        del out, owned

    diag = dict(
        family=family,
        n_tiles=len(tiles),
        n_tile=int(n_tile),
        b_requested=int(b_fine),
        b_realized=int(b_real),
        padded_P=int(P),
        # compact tiling is exact only if the buffer contains the kernel's
        # whole support; report it so the claim is checkable from the json.
        buffer_contains_support=(None if family != "compact" else bool(b_real * cell >= r_out)),
        n_brick=int(n_brick),
        cap=int(cap),
        pad_frac=float(pad_frac),
        fft_work_ratio=float(len(tiles) * P**3 / float(n_fine) ** 3),
        # Superset overhang: brick-union members outside the padded mesh,
        # correctly excluded. An efficiency diagnostic, NOT an error -- an
        # earlier `n_dropped == 0` contract fired on this healthy behaviour.
        n_overhang_total=int(n_overhang_total),
        # THE REAL CONTRACT: every particle owned by exactly one tile.
        n_owned_total=int((owner_count > 0).sum()),
        min_owner_count=int(owner_count.min()),
        max_owner_count=int(owner_count.max()),
        partition_ok=bool(
            owner_count.min() == 1
            and owner_count.max() == 1
            and int((owner_count > 0).sum()) == int(pos_np.shape[0])
        ),
        peak_after_first_tile=peak_first,
        peak_total=(peak() if peak is not None else None),
    )
    return g_out, diag
