"""Out-of-core 3D real FFTs: host-resident k-space, slab-streamed real space.

"Out of core" means out of DEVICE core (D-v2-15 clause 4: a 2048^3
`jnp.fft.rfftn` does not fit a GH200 -- workspace 7x the field). Here the
spectral array lives in host numpy and every real-space field is only ever
touched one axis-0 slab at a time, so the IC stage's peak is one spectral
array plus O(plane) buffers. Disk holds staged REAL fields and T9 slabs only
(clause 3); k-space is never spilled.

THE CANONICAL FACTORIZATION -- this, not `np.fft.rfftn`, is the layer's
definition:

  pass 1  scipy.fft.rfft2 of ONE axis-0 plane at a time;
  pass 2  scipy.fft.fft(axis=0) of ONE y-pencil-plane (N, N//2+1) at a time.

THE UNIT IS ONE PLANE BY MEASUREMENT, NOT TASTE. pocketfft's results are
batch-size dependent at the bit level: on a 32^3 f64 field, per-plane rfft2
differs from the whole-batch call in 341 elements and a 5-column axis-0
chunk differs from the whole call in 68 (2026-08-10, scipy in the locked
env; worker count moves nothing). So a slab- or chunk-sized compute unit
would make the spectrum depend on the streaming decomposition -- exactly
what D-v2-15 clause 5 defines reproducibility against. With a fixed
one-plane unit, slab thickness is an OUTER loop bound that cannot touch a
bit, and "streamed == monolithic" is a theorem the tests then merely
confirm.

Equality with `np.fft.rfftn` is NOT claimed (a different transform order
rounds differently); the tests keep a tolerance-level cross-check only.

scipy.fft, not np.fft, throughout: pocketfft preserves single precision
(np.fft silently upcasts f32 to complex128, which would both lie about the
f32 path and double the spectral residency).

Spectral multipliers reuse `forces.k_components`' conventions exactly
(fftfreq-signed ik, k = 0 mapped to 1/k^2 = 1) but are built PER SLAB in f64
-- a full (N, N, N//2+1) float64 |k| or 1/k^2 grid at 2048^3 is 34.4 GB, the
very term D-v2-15 clause 2 retires.
"""

import numpy as np
import scipy.fft

_DEF_SLAB = 32
_DEF_WORKERS = -1


def _spec_shape(n_mesh):
    return (n_mesh, n_mesh, n_mesh // 2 + 1)


def _cdtype_for(fdtype):
    return np.complex128 if np.dtype(fdtype) == np.float64 else np.complex64


def _kx(n_mesh, box_size):
    return 2.0 * np.pi * np.fft.fftfreq(n_mesh, d=box_size / n_mesh)


def _kz(n_mesh, box_size):
    return 2.0 * np.pi * np.fft.rfftfreq(n_mesh, d=box_size / n_mesh)


# ---------------------------------------------------------------------------
# the two passes, one plane at a time
# ---------------------------------------------------------------------------


def rfft2_slab(slab, workers=_DEF_WORKERS):
    """Pass 1 on a slab: per-plane 2D real FFTs, (t, N, N) -> (t, N, N//2+1).

    A loop of single-plane transforms, never one batched call: the batch size
    would otherwise leak into the bits (module docstring).
    """
    t, n = slab.shape[0], slab.shape[1]
    out = np.empty((t, n, n // 2 + 1), dtype=_cdtype_for(slab.dtype))
    for i in range(t):
        out[i] = scipy.fft.rfft2(slab[i], workers=workers)
    return out


def fft_axis0_inplace(spec, inverse=False, workers=_DEF_WORKERS):
    """Pass 2: (i)fft along axis 0, one y-pencil-plane at a time, in place.

    The unit is the contiguous copy of spec[:, y, :] -- a fixed (N, N//2+1)
    shape whatever the caller's streaming looked like, so pass 2 has no knob
    that could move a bit. O(N * M) working memory (~17 MB at 2048^3 c64).
    """
    fn = scipy.fft.ifft if inverse else scipy.fft.fft
    for y in range(spec.shape[1]):
        spec[:, y, :] = fn(np.ascontiguousarray(spec[:, y, :]), axis=0, workers=workers)
    return spec


# ---------------------------------------------------------------------------
# forward / inverse
# ---------------------------------------------------------------------------


def forward_from_slabs(slab_fn, n_mesh, slab=_DEF_SLAB, workers=_DEF_WORKERS):
    """Forward 3D rfft of a field the caller produces slab-wise.

    slab_fn(lo, hi) -> (hi-lo, N, N) real array of axis-0 planes [lo, hi).
    The real field is never materialized here -- fused generation (white
    noise -> spectrum) is the design point. Returns the (N, N, N//2+1)
    spectral array, complex64/complex128 following the slabs' dtype. slab is
    a pure memory knob: the compute unit is one plane regardless.
    """
    n = int(n_mesh)
    slab = n if slab is None else int(slab)
    if slab < 1:
        raise ValueError(f"slab must be >= 1, got {slab}")
    spec = None
    for lo in range(0, n, slab):
        hi = min(lo + slab, n)
        s = np.asarray(slab_fn(lo, hi))
        if s.shape != (hi - lo, n, n):
            raise ValueError(f"slab_fn({lo}, {hi}) returned shape {s.shape}, "
                             f"want {(hi - lo, n, n)}")
        if spec is None:
            spec = np.empty(_spec_shape(n), dtype=_cdtype_for(s.dtype))
        # per plane DIRECTLY into the target rows -- a whole-slab intermediate
        # here is a full spectrum copy at slab = n, which is how the memory
        # ladder read A = 11.74 B/p against a 2-spectrum design (Vista 902241:
        # source + intermediate + target = 3 spec-equivalents through
        # rfftn_ooc). Same per-plane transforms, so no bit moves.
        for i in range(hi - lo):
            spec[lo + i] = scipy.fft.rfft2(s[i], workers=workers)
    fft_axis0_inplace(spec, workers=workers)
    return spec


def inverse_to_slabs(spec, n_mesh, slab=_DEF_SLAB, workers=_DEF_WORKERS):
    """Inverse of `forward_from_slabs`, yielding (lo, real_slab) in axis-0 order.

    MUTATES spec (the axis-0 inverse pass runs in place) -- the caller hands
    over ownership; a spectrum needed again must be copied first, which is a
    memory decision the caller should be making explicitly anyway. The real
    slabs come back one plane-transform at a time for the same bitwise reason
    as the forward pass.
    """
    n = int(n_mesh)
    slab = n if slab is None else int(slab)
    fft_axis0_inplace(spec, inverse=True, workers=workers)
    for lo in range(0, n, slab):
        hi = min(lo + slab, n)
        out = np.empty((hi - lo, n, n), dtype=np.float64 if spec.dtype == np.complex128
                       else np.float32)
        for i in range(lo, hi):
            out[i - lo] = scipy.fft.irfft2(spec[i], s=(n, n), workers=workers)
        yield lo, out


def rfftn_ooc(field, workers=_DEF_WORKERS):
    """Monolithic convenience: the SAME factorization at slab = N."""
    n = field.shape[0]
    return forward_from_slabs(lambda lo, hi: field[lo:hi], n, slab=n, workers=workers)


def irfftn_ooc(spec, n_mesh, workers=_DEF_WORKERS):
    """Monolithic convenience; consumes spec like the generator does.

    Iterates the inverse at the default slab rather than slab = n: the
    whole-box slab buffer would be a second full field beside the assembled
    output (the 902241 double-buffer class), and slab size cannot move a bit.
    """
    n = int(n_mesh)
    out = np.empty((n, n, n), dtype=np.float64 if spec.dtype == np.complex128 else np.float32)
    for lo, s in inverse_to_slabs(spec, n, slab=_DEF_SLAB, workers=workers):
        out[lo : lo + s.shape[0]] = s
    return out


# ---------------------------------------------------------------------------
# spectral multipliers (slab-built, f64 precision island, k_components
# conventions: fftfreq-signed ik; k = 0 -> 1/k^2 = 1 with ik zero there)
# ---------------------------------------------------------------------------


def mul_radial_inplace(spec, n_mesh, box_size, f_of_k, dc_value, slab=_DEF_SLAB):
    """spec *= f(|k|), evaluated per axis-0 slab; the DC bin gets dc_value.

    f_of_k receives a float64 |k| array with the k = 0 entry replaced by the
    grid's smallest nonzero |k| (in any ICKTable's range by construction --
    the kk_safe trick); its value there is then discarded in favour of
    dc_value (0.0 zeroes the mean mode; 1.0 leaves DC untouched).
    """
    n = int(n_mesh)
    slab = n if slab is None else int(slab)
    kx = _kx(n, box_size)
    kz = _kz(n, box_size)
    k_min_nonzero = abs(kz[1])
    for lo in range(0, n, slab):
        hi = min(lo + slab, n)
        kk = np.sqrt(
            kx[lo:hi].reshape(-1, 1, 1) ** 2
            + kx.reshape(1, n, 1) ** 2
            + kz.reshape(1, 1, -1) ** 2
        )
        if lo == 0:
            kk[0, 0, 0] = k_min_nonzero
        vals = np.asarray(f_of_k(kk))
        if lo == 0:
            vals[0, 0, 0] = dc_value
        spec[lo:hi] *= vals.astype(spec.real.dtype, copy=False)
    return spec


def _ik_over_k2_slab(axis, lo, hi, n_mesh, box_size):
    """(ik_axis / k^2) for axis-0 rows [lo, hi), complex128, k = 0 entry -> 0."""
    n = int(n_mesh)
    kx = _kx(n, box_size)
    kz = _kz(n, box_size)
    kxs = kx[lo:hi].reshape(-1, 1, 1)
    kys = kx.reshape(1, n, 1)
    kzs = kz.reshape(1, 1, -1)
    k2 = kxs**2 + kys**2 + kzs**2
    if lo == 0:
        k2[0, 0, 0] = 1.0  # k_components convention; ik is zero there anyway
    k_ax = (kxs, kys, kzs)[axis]
    return (1j * k_ax) / k2


def grad_invk2_spec(spec, axis, n_mesh, box_size, slab=_DEF_SLAB):
    """COPY of spec * ik_axis / k^2 -- the displacement kernel, slab-built.

    A copy, deliberately: the multiplier has zeros (DC among them), so an
    in-place form is not invertible and the source spectrum is needed again
    for the other two components.
    """
    n = int(n_mesh)
    slab = n if slab is None else int(slab)
    out = np.empty_like(spec)
    for lo in range(0, n, slab):
        hi = min(lo + slab, n)
        m = _ik_over_k2_slab(axis, lo, hi, n, box_size)
        out[lo:hi] = spec[lo:hi] * m.astype(spec.dtype, copy=False)
    return out


def deriv2_spec(spec, i, j, n_mesh, box_size, slab=_DEF_SLAB):
    """COPY of spec * (ik_i)(ik_j) / k^2 -- the tidal/second-derivative kernel."""
    n = int(n_mesh)
    slab = n if slab is None else int(slab)
    kx = _kx(n, box_size)
    kz = _kz(n, box_size)
    out = np.empty_like(spec)
    for lo in range(0, n, slab):
        hi = min(lo + slab, n)
        comps = (
            kx[lo:hi].reshape(-1, 1, 1),
            kx.reshape(1, n, 1),
            kz.reshape(1, 1, -1),
        )
        k2 = comps[0] ** 2 + comps[1] ** 2 + comps[2] ** 2
        if lo == 0:
            k2[0, 0, 0] = 1.0
        m = -(comps[i] * comps[j]) / k2  # (ik_i)(ik_j) = -k_i k_j
        out[lo:hi] = spec[lo:hi] * m.astype(spec.real.dtype, copy=False)
    return out


# ---------------------------------------------------------------------------
# disk staging (D-v2-15 clause 3) -- EXPLICIT IO, deliberately not memmap
# ---------------------------------------------------------------------------


class StagedArray:
    """A disk-staged (n0, n, n) array written and read one axis-0 slab at a time.

    Plain `.npy` on disk (np.load can always inspect it), but accessed through
    explicit read()/write() calls rather than memmap ON PURPOSE: dirty pages
    of a written memmap are MAPPED INTO THE PROCESS and count in ru_maxrss,
    so a memmap-staged generator read ~175 B/p on its first smoke where its
    heap holds ~8 -- the instrument was measuring reclaimable page cache as if
    it were footprint (2026-08-10). Buffered file IO keeps those pages the
    kernel's, so the process's memory story stays the true one.
    """

    def __init__(self, path, dtype, shape, mode):
        self.path = path
        self.dtype = np.dtype(dtype)
        self.shape = tuple(int(s) for s in shape)
        self._row = int(np.prod(self.shape[1:])) * self.dtype.itemsize
        if mode == "w":
            with open(path, "wb") as fh:
                np.lib.format.write_array_header_1_0(
                    fh, dict(descr=np.lib.format.dtype_to_descr(self.dtype),
                             fortran_order=False, shape=self.shape)
                )
                self._data0 = fh.tell()
            # pre-extend so out-of-order slab writes are well-defined
            with open(path, "r+b") as fh:
                fh.truncate(self._data0 + self._row * self.shape[0])
        elif mode == "r":
            with open(path, "rb") as fh:
                version = np.lib.format.read_magic(fh)
                readers = {(1, 0): np.lib.format.read_array_header_1_0,
                           (2, 0): np.lib.format.read_array_header_2_0}
                hdr_shape, fortran, hdr_dtype = readers[version](fh)
                self._data0 = fh.tell()
            if hdr_shape != self.shape or hdr_dtype != self.dtype or fortran:
                raise ValueError(
                    f"{path}: header {hdr_dtype}{hdr_shape} != expected {self.dtype}{self.shape}"
                )
        else:
            raise ValueError(f"mode must be 'w' or 'r', got {mode!r}")

    @classmethod
    def create(cls, path, dtype, shape):
        return cls(path, dtype, shape, "w")

    @classmethod
    def open(cls, path, dtype, shape):
        return cls(path, dtype, shape, "r")

    def write_slab(self, lo, arr):
        arr = np.ascontiguousarray(arr, dtype=self.dtype)
        if arr.shape[1:] != self.shape[1:]:
            raise ValueError(f"slab shape {arr.shape} does not fit {self.shape}")
        with open(self.path, "r+b") as fh:
            fh.seek(self._data0 + self._row * int(lo))
            fh.write(arr.tobytes())

    def read_slab(self, lo, hi):
        lo, hi = int(lo), int(hi)
        with open(self.path, "rb") as fh:
            fh.seek(self._data0 + self._row * lo)
            buf = fh.read(self._row * (hi - lo))
        return np.frombuffer(buf, dtype=self.dtype).reshape((hi - lo,) + self.shape[1:]).copy()


# ---------------------------------------------------------------------------
# the accounting function (mesh_bytes pattern) and its refusal
# ---------------------------------------------------------------------------


def plan_bytes(n_mesh, fdtype, policy, slab=_DEF_SLAB):
    """Predicted peak host bytes per phase, labelled term by term.

    Policies:
      "forward"    one spectral array + the slab and per-plane working buffers
                   (the fused noise -> spectrum pass; no real field exists)
      "derivative" forward + ONE spectral copy (a grad/deriv2 output while the
                   source spectrum stays resident)
      "roundtrip"  forward + the assembled real field (the monolithic
                   convenience / dev-scale identity gates)
    """
    n = int(n_mesh)
    w = np.dtype(fdtype).itemsize
    cw = 2 * w
    m = n // 2 + 1
    spec = n * n * m * cw
    terms = dict(
        spec=spec,
        slab_real=slab * n * n * w,
        slab_spec=slab * n * m * cw,
        slab_kmag_f64=slab * n * m * 8,
        pencil_unit=n * m * cw,
    )
    if policy == "forward":
        pass
    elif policy == "derivative":
        terms["spec_copy"] = spec
    elif policy == "roundtrip":
        terms["field"] = n * n * n * w
    else:
        raise ValueError(f"unknown policy {policy!r}")
    terms["peak"] = sum(v for k, v in terms.items() if k != "peak")
    return terms


def require_fits(n_mesh, fdtype, policy, budget_bytes, slab=_DEF_SLAB):
    """Refuse loudly (D-007 discipline: refuse, never silently page) if the
    predicted peak exceeds the budget; returns the plan when it fits."""
    plan = plan_bytes(n_mesh, fdtype, policy, slab=slab)
    if plan["peak"] > budget_bytes:
        detail = ", ".join(
            f"{k}={v / 1e9:.2f} GB" for k, v in plan.items() if k != "peak"
        )
        raise MemoryError(
            f"ooc_fft plan '{policy}' at n={n_mesh} {np.dtype(fdtype).name} predicts "
            f"{plan['peak'] / 1e9:.2f} GB against a {budget_bytes / 1e9:.2f} GB budget "
            f"({detail}); refusing rather than paging"
        )
    return plan
