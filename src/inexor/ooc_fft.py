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

import threading
import time as _time

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


def fft_axis0_inplace(spec, inverse=False, workers=_DEF_WORKERS, progress=None):
    """Pass 2: (i)fft along axis 0, one y-pencil-plane at a time, in place.

    The unit is the contiguous copy of spec[:, y, :] -- a fixed (N, N//2+1)
    shape whatever the caller's streaming looked like, so pass 2 has no knob
    that could move a bit. O(N * M) working memory (~17 MB at 2048^3 c64).

    `progress(stage, done, total)`, if given, is called once per plane; see
    `inexor.progress.Heartbeat`.
    """
    fn = scipy.fft.ifft if inverse else scipy.fft.fft
    ny = spec.shape[1]
    for y in range(ny):
        spec[:, y, :] = fn(np.ascontiguousarray(spec[:, y, :]), axis=0, workers=workers)
        if progress is not None:
            progress("fft axis0", y + 1, ny)
    return spec


# ---------------------------------------------------------------------------
# forward / inverse
# ---------------------------------------------------------------------------


def forward_from_slabs(slab_fn, n_mesh, slab=_DEF_SLAB, workers=_DEF_WORKERS,
                       progress=None):
    """Forward 3D rfft of a field the caller produces slab-wise.

    slab_fn(lo, hi) -> (hi-lo, N, N) real array of axis-0 planes [lo, hi).
    The real field is never materialized here -- fused generation (white
    noise -> spectrum) is the design point. Returns the (N, N, N//2+1)
    spectral array, complex64/complex128 following the slabs' dtype. slab is
    a pure memory knob: the compute unit is one plane regardless.

    `progress(stage, done, total)`, if given, reports the two passes
    separately ("fft plane" then "fft axis0"): they cost differently per unit
    and one rate over both describes neither.
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
        if progress is not None:
            progress("fft plane", hi, n)
    fft_axis0_inplace(spec, workers=workers, progress=progress)
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
# the device path: the SAME factorization, planes transformed on an accelerator
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS AT ALL. The coarse solve at 4096^3 is a 2048^3 transform. The
# monolithic device form does not fit (peak/field 8.0x measured, 275 GB against
# a GB200's 185 GiB) and the HOST out-of-core form above costs 417 s/step, 39%
# of a gb node's 1080 s per-step budget on its own. Factorized per plane it is
# projected at 0.4 s. Same factorization as the host path, different executor.
#
# The spectral array stays HOST-RESIDENT here exactly as it does above: only
# planes cross to the device. That is the property that makes the design's
# memory work, not an implementation detail.

#: A device transform at or above this many elements is REFUSED, never attempted.
#:
#: MEASURED (Vista 972737, jax 0.10.2 + GB200, record 5y finding 2):
#: `jnp.fft.rfftn` of a 1536^3 f32 field -- 3.6e9 elements -- returns a WRONG
#: transform SILENTLY. Its roundtrip reads max|d|/rms 3.8e+3 where 1024^3
#: (1.07e9 elements) reads 2.9e-6, with the same peak/field ratio and a
#: plausible wall, so nothing about the call looks wrong from outside. 2^31 is
#: where those two readings bracket; the cuFFT 32-bit-plan class is the obvious
#: suspect and is NOT measured, so treat the bound as empirical.
#:
#: The factorization below never approaches it -- one 2048^2 plane is 4.2e6
#: elements, three orders under. This guard is for the batch knobs and for any
#: caller who reaches past them for a monolithic transform. It refuses rather
#: than checking a receipt afterwards because a wrong spectrum that is merely
#: reported is still a wrong spectrum, and D-007's discipline is to refuse.
MAX_DEVICE_TRANSFORM_ELEMENTS = 2**31


def refuse_oversize_device_transform(n_elements, what="transform"):
    """Refuse a device FFT big enough to hit the silent-wrong-result class."""
    n_elements = int(n_elements)
    if n_elements >= MAX_DEVICE_TRANSFORM_ELEMENTS:
        raise ValueError(
            f"{what} of {n_elements:,} elements is at or above the "
            f"{MAX_DEVICE_TRANSFORM_ELEMENTS:,} bound where a device FFT has "
            "been MEASURED to return a wrong result silently (1536^3 f32 "
            "roundtrip 3.8e+3 against 1024^3's 2.9e-6, Vista 972737). "
            "Factorize it: the plane is the unit."
        )


def _require_x64_for(dtype):
    """f64 on device needs the caller to have enabled x64, or it silently narrows.

    Same contract and same reason as `eject_jax.require_x64`: this library never
    toggles `jax_enable_x64`, and with it off a float64 field is transformed at
    single precision and handed back in a float64 container, which is a wrong
    answer wearing the right dtype.
    """
    if np.dtype(dtype) != np.float64:
        return
    import jax

    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "the device FFT path was given a float64 field with jax_enable_x64 "
            "off: jax would narrow it to f32 and the complex128 output would be "
            "single precision wearing a double dtype. Enable x64 in the caller "
            "(this library never toggles it), or pass a float32 field."
        )


def _check_spectral_dtype(got, want, what):
    """The dtype ledger, on the seam where a silent upcast would hide.

    `np.fft` upcasts f32 to complex128 and that is why the host path uses scipy;
    the device path has the mirror-image risk (a narrowing under x64-off). An
    f32 arm that came back complex128 would pass every value comparison in this
    module and double the spectral residency the whole design is sized on.
    """
    if np.dtype(got) != np.dtype(want):
        raise TypeError(
            f"{what} returned {np.dtype(got).name}, want {np.dtype(want).name}: "
            "the device FFT changed precision underneath the caller"
        )


#: How a host buffer gets to the device and back.
#:
#: "pageable" hands the driver ordinary numpy. The OS may move that memory, so
#: the GPU cannot DMA from it: the driver copies it into a staging buffer of its
#: own first, and every crossing pays that copy.
#:
#: "staged" routes through `pinned_host` -- page-locked memory the OS has
#: promised not to move, which the DMA engine reads directly. The copy is still
#: paid (host -> pinned), but EXPLICITLY, and the crossing itself is then a
#: straight DMA. D5 measured 7.8x between the two memory kinds on a synthetic
#: microbenchmark at width; whether that survives the explicit copy is the whole
#: question this policy exists to answer, and it is measured on the real
#: transform rather than on a proxy.
#:
#: The policy MUST NOT move a bit -- it changes the route, not the arithmetic --
#: and a test pins that.
TRANSFER_POLICIES = ("pageable", "staged")


def _pinned_sharding(device):
    import jax

    return jax.sharding.SingleDeviceSharding(device, memory_kind="pinned_host")


def _device_sharding(device):
    """The device target for a buffer that is currently in another memory kind.

    MUST be a sharding, not the bare `Device`. `jax.device_put(pinned, dev)`
    raises "Memory kind mismatch with xla::PjRtBuffers" -- a bare device carries
    no memory kind to switch TO, so the move out of `pinned_host` has nothing to
    target. Vista 992589 failed on exactly this: `staging_supported` probed with
    a bare device, reported False on four GB200s that stage perfectly well, and
    every staged leg refused. The refusal was right; the probe was wrong.
    """
    import jax

    return jax.sharding.SingleDeviceSharding(device)


def staging_supported(device=None):
    """Can this backend do the host -> pinned_host -> device round trip?

    Callers ASK, and refuse, rather than catching a failure and quietly
    transferring pageable: a staged arm that silently ran pageable would report
    that staging buys nothing, which is the one wrong answer this measurement
    can produce.
    """
    import jax

    try:
        dev = jax.devices()[0] if device is None else device
        probe = np.zeros((2, 2), dtype=np.float32)
        h = jax.device_put(probe, _pinned_sharding(dev))
        back = jax.device_put(h, _device_sharding(dev))
        return bool(np.array_equal(np.asarray(back), probe))
    except Exception:
        return False


def _to_device(a, device, transfer="pageable"):
    """Host -> device, under a transfer policy.

    `jnp.asarray` places on the default device; `jax.device_put` COMMITS to the
    one named. The distinction is the whole of the multi-device path -- an
    uncommitted array on a four-GPU node runs every batch on device 0 while
    looking exactly like work.
    """
    import jax
    import jax.numpy as jnp

    if transfer not in TRANSFER_POLICIES:
        raise ValueError(f"transfer must be one of {TRANSFER_POLICIES}, got {transfer!r}")
    if transfer == "staged":
        dev = jax.devices()[0] if device is None else device
        return jax.device_put(jax.device_put(a, _pinned_sharding(dev)),
                              _device_sharding(dev))
    return jnp.asarray(a) if device is None else jax.device_put(a, device)


def _from_device(d, transfer="pageable"):
    """Device -> host numpy, under the same policy."""
    import jax

    if transfer == "staged":
        dev = next(iter(d.devices()))
        return np.asarray(jax.device_put(d, _pinned_sharding(dev)))
    return np.asarray(d)


def rfft2_planes_device(planes, plane_batch=1, out=None, device=None,
                        transfer="pageable"):
    """Pass 1 on device: 2-D real FFTs of axis-0 planes, (t, N, N) -> (t, N, M).

    numpy in, numpy out -- the spectrum is host-resident by design.

    `plane_batch` IS PART OF THE TRANSFORM'S DEFINITION, not a free knob. The
    host path fixed its unit at one plane because pocketfft's results are
    batch-size dependent at the bit level (module docstring: 341 elements on a
    32^3 f64 field), and there is no reason to expect a device FFT library to be
    kinder. So the default is 1, matching the host unit, and any card or record
    that quotes a spectrum must carry the batch beside it. What IS guaranteed is
    that `slab` cannot move a bit at fixed `plane_batch` -- streaming stays an
    outer loop bound, which is the property D-v2-15 clause 5 is defined against.
    """
    import jax.numpy as jnp

    a = np.asarray(planes)
    if a.ndim != 3 or a.shape[1] != a.shape[2]:
        raise ValueError(f"want (t, N, N) planes, got {a.shape}")
    t, n = a.shape[0], a.shape[1]
    _require_x64_for(a.dtype)
    b = max(1, int(plane_batch))
    refuse_oversize_device_transform(b * n * n, "device rfft2 batch")
    cd = _cdtype_for(a.dtype)
    if out is None:
        out = np.empty((t, n, n // 2 + 1), dtype=cd)
    for lo in range(0, t, b):
        hi = min(lo + b, t)
        d = _from_device(jnp.fft.rfft2(_to_device(a[lo:hi], device, transfer),
                                       axes=(-2, -1)), transfer)
        _check_spectral_dtype(d.dtype, cd, "device rfft2")
        out[lo:hi] = d
    return out


def fft_axis0_device_inplace(spec, inverse=False, pencil_batch=1, device=None,
                             transfer="pageable"):
    """Pass 2 on device: (i)fft along axis 0, y-pencil-planes at a time, in place.

    The host twin's unit is one (N, M) pencil-plane; `pencil_batch` widens it and
    carries the same definitional status as `plane_batch` above.
    """
    import jax.numpy as jnp

    fn = jnp.fft.ifft if inverse else jnp.fft.fft
    n, ny, m = spec.shape
    b = max(1, int(pencil_batch))
    refuse_oversize_device_transform(b * n * m, "device axis-0 fft batch")
    for lo in range(0, ny, b):
        hi = min(lo + b, ny)
        blk = np.ascontiguousarray(spec[:, lo:hi, :])
        d = _from_device(fn(_to_device(blk, device, transfer), axis=0), transfer)
        _check_spectral_dtype(d.dtype, spec.dtype, "device axis-0 fft")
        spec[:, lo:hi, :] = d
    return spec


# ---------------------------------------------------------------------------
# splitting a pass across devices
# ---------------------------------------------------------------------------
#
# The factorization is embarrassingly parallel and has NO inter-device
# communication: pass 1 is independent per plane, pass 2 is independent per
# y-pencil-plane, and the spectrum stays host-resident throughout. So "use four
# GPUs" is a partition of a loop, not a distributed transform -- and the only
# thing that can stop it scaling is the shared host bus, which D1 measured this
# path to be bound by (91% of the wall, record sec. 9).


def partition_units(total, n_parts, unit):
    """Split [0, total) into `n_parts` contiguous ranges, boundaries on `unit`.

    The alignment is the transform's DEFINITION, not tidiness. This layer's
    results are batch-size dependent at the bit level (module docstring: 341
    elements on a 32^3 f64 field) and a batch restarts at the start of every
    part, so a boundary off a `unit` multiple gives a different sequence of
    batch sizes from the unpartitioned loop -- a different spectrum, silently,
    at exactly the widths nobody runs by default. With aligned boundaries the
    partition is an identity and "W devices == 1 device, bitwise" is a theorem
    the tests confirm rather than a hope.

    Refuses a width it cannot realize (fewer whole units than parts) instead of
    returning empty ranges: a part with no work is a device that quietly did
    not participate, which reads downstream as "it did not scale".
    """
    total, n_parts = int(total), int(n_parts)
    unit = max(1, int(unit))
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    if n_parts < 1:
        raise ValueError(f"n_parts must be >= 1, got {n_parts}")
    n_units = -(-total // unit)
    if n_units < n_parts:
        raise ValueError(
            f"cannot split {total} elements into {n_parts} parts at a batch "
            f"unit of {unit}: only {n_units} whole units exist, so "
            f"{n_parts - n_units} part(s) would get no work. Widen the slab, "
            "narrow the batch, or ask for fewer parts."
        )
    base, extra = divmod(n_units, n_parts)
    parts, u = [], 0
    for k in range(n_parts):
        lo = u * unit
        u += base + (1 if k < extra else 0)
        parts.append((lo, min(u * unit, total)))
    return parts


def _run_parts(fn, parts, devices):
    """Run `fn(lo, hi, device)` over `parts` concurrently, one thread per part.

    Threads, not processes: the engine is ONE process holding one host-resident
    state, and jax releases the GIL across dispatch and transfer. A single part
    runs INLINE, so the one-device path carries no pool overhead and stays the
    reference every wider width is measured against.
    """
    if len(parts) == 1:
        fn(parts[0][0], parts[0][1], devices[0])
        return
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(parts)) as ex:
        futures = [ex.submit(fn, lo, hi, dev)
                   for (lo, hi), dev in zip(parts, devices)]
        for f in futures:
            f.result()


def _devices_or_default(devices):
    if devices is None:
        return [None]
    devs = list(devices)
    if not devs:
        raise ValueError("devices= was an empty sequence; pass None for the "
                         "single-device path")
    return devs


def forward_from_slabs_device(slab_fn, n_mesh, slab=_DEF_SLAB, plane_batch=1,
                              pencil_batch=1, devices=None, timings=None,
                              transfer="pageable", kernel=None, box_size=1.0):
    """Device twin of `forward_from_slabs`; identical structure, identical contract.

    `kernel` (a `KSpaceKernel`) is applied on the card inside pass 2
    (`kspace_pass_device`); None leaves pass 2 exactly as it was.

    `devices` is a sequence of jax devices to split each pass across, or None
    for jax's own placement on one device. The split is contiguous and
    batch-aligned, so the spectrum is BITWISE identical to the devices=None one
    at the same `plane_batch` / `pencil_batch` (`partition_units`).

    Pass 1 splits the planes WITHIN a slab and the slab loop stays sequential:
    `slab_fn` is the caller's generator and is not assumed re-entrant. That puts
    a barrier at every slab boundary, which is a real property of streaming and
    is why a measured split is owed rather than assumed.

    `timings`, if a dict is passed, receives `pass1_s` and `pass2_s`. The two
    passes have different shapes -- pass 1 is per-plane device work behind a
    per-slab barrier, pass 2 is a strided HOST gather plus device work -- so a
    split that stalls is attributable rather than inferred from an Amdahl fit.
    """
    devs = _devices_or_default(devices)
    n = int(n_mesh)
    slab = n if slab is None else int(slab)
    if slab < 1:
        raise ValueError(f"slab must be >= 1, got {slab}")
    spec = None
    _p1 = 0.0
    for lo in range(0, n, slab):
        hi = min(lo + slab, n)
        s = np.asarray(slab_fn(lo, hi))
        if s.shape != (hi - lo, n, n):
            raise ValueError(f"slab_fn({lo}, {hi}) returned shape {s.shape}, "
                             f"want {(hi - lo, n, n)}")
        if spec is None:
            spec = np.empty(_spec_shape(n), dtype=_cdtype_for(s.dtype))

        def pass1(a, b, dev, s=s, lo=lo):
            rfft2_planes_device(s[a:b], plane_batch=plane_batch,
                                out=spec[lo + a:lo + b], device=dev,
                                transfer=transfer)

        _t0 = _time.perf_counter()
        _run_parts(pass1, partition_units(hi - lo, len(devs), plane_batch), devs)
        _p1 += _time.perf_counter() - _t0

    def pass2(a, b, dev):
        fft_axis0_device_inplace(spec[:, a:b, :], pencil_batch=pencil_batch,
                                 device=dev, transfer=transfer)

    _t0 = _time.perf_counter()
    if kernel is None:
        _run_parts(pass2, partition_units(n, len(devs), pencil_batch), devs)
    else:
        kspace_pass_device([(1.0, spec)], n, box_size, kernel=kernel, out=spec,
                           devices=devices, pencil_batch=pencil_batch, transfer=transfer)
    if timings is not None:
        timings["pass1_s"] = _p1
        timings["pass2_s"] = _time.perf_counter() - _t0
    return spec


def forward_from_card_planes(shards, n_mesh, plane_batch=1, pencil_batch=1,
                             timings=None, transfer="pageable"):
    """`forward_from_slabs_device` for a real-space field already on the cards.

    `shards` are dicts with `lo`, `hi`, `device` and `delta`: x-planes [lo, hi)
    of the field, shape (hi - lo, N, N), resident on that device, together
    tiling [0, N) (`device.paint.coarse_delta_cards`). Pass 1 transforms each
    card's planes where they sit, one thread per card, so no real-space plane
    crosses the bus; pass 2 is `forward_from_slabs_device`'s, split across the
    same devices. The spectrum stays host-resident.

    BITWISE the host-slab path at the same `plane_batch` / `pencil_batch`: every
    card boundary must be a `plane_batch` multiple, so the batch sequence is the
    unpartitioned one (`partition_units`). `timings` receives `pass1_s` and
    `pass2_s`.
    """
    import jax.numpy as jnp

    n = int(n_mesh)
    shards = sorted(shards, key=lambda s: int(s["lo"]))
    b = max(1, int(plane_batch))
    edge = 0
    for s in shards:
        lo, hi = int(s["lo"]), int(s["hi"])
        if lo != edge:
            raise ValueError(f"card shards do not tile [0, {n}): a shard starts at {lo} "
                             f"where the previous one ended at {edge}")
        if tuple(s["delta"].shape) != (hi - lo, n, n):
            raise ValueError(f"shard [{lo}, {hi}) has shape {tuple(s['delta'].shape)}, "
                             f"want {(hi - lo, n, n)}")
        if lo % b:
            raise ValueError(
                f"card boundary {lo} is not a multiple of plane_batch {b}: pass 1's "
                "batches would restart off the unpartitioned sequence and give a "
                "different spectrum")
        edge = hi
    if edge != n:
        raise ValueError(f"card shards do not tile [0, {n}): they end at {edge}")
    dt = np.dtype(shards[0]["delta"].dtype)
    if any(np.dtype(s["delta"].dtype) != dt for s in shards):
        raise TypeError("card shards disagree on dtype")
    _require_x64_for(dt)
    refuse_oversize_device_transform(b * n * n, "device rfft2 batch")
    cd = _cdtype_for(dt)
    spec = np.empty(_spec_shape(n), dtype=cd)
    devs = [s["device"] for s in shards]

    def pass1(k, _k1, _dev):
        s = shards[k]
        lo, w = int(s["lo"]), int(s["hi"]) - int(s["lo"])
        for a in range(0, w, b):
            z = min(a + b, w)
            d = _from_device(jnp.fft.rfft2(s["delta"][a:z], axes=(-2, -1)), transfer)
            _check_spectral_dtype(d.dtype, cd, "device rfft2")
            spec[lo + a:lo + z] = d

    _t0 = _time.perf_counter()
    _run_parts(pass1, [(k, k + 1) for k in range(len(shards))], devs)
    _p1 = _time.perf_counter() - _t0

    def pass2(a, z, dev):
        fft_axis0_device_inplace(spec[:, a:z, :], pencil_batch=pencil_batch,
                                 device=dev, transfer=transfer)

    _t0 = _time.perf_counter()
    _run_parts(pass2, partition_units(n, len(devs), pencil_batch), devs)
    if timings is not None:
        timings["pass1_s"] = _p1
        timings["pass2_s"] = _time.perf_counter() - _t0
    return spec


def inverse_to_slabs_device(spec, n_mesh, slab=_DEF_SLAB, plane_batch=1,
                            pencil_batch=1, devices=None, timings=None,
                            transfer="pageable", pass2=True):
    """Device twin of `inverse_to_slabs`. MUTATES spec, exactly as that one does.

    `devices` splits both passes as in `forward_from_slabs_device`, bitwise
    identically to the one-device path. This is the leg the coarse solve pays
    three times per step, and the only one that generates no field, so it is the
    clean thing to time: a forward's wall carries host RNG that does NOT split.

    `pass2=False` skips the axis-0 pass, for a buffer `kspace_pass_device`
    already took through it (inverse=True).
    """
    import jax.numpy as jnp

    devs = _devices_or_default(devices)
    n = int(n_mesh)
    slab = n if slab is None else int(slab)

    def _pass2(a, b, dev):
        fft_axis0_device_inplace(spec[:, a:b, :], inverse=True,
                                 pencil_batch=pencil_batch, device=dev,
                                 transfer=transfer)

    _t0 = _time.perf_counter()
    if pass2:
        _run_parts(_pass2, partition_units(spec.shape[1], len(devs), pencil_batch),
                   devs)
    _p2 = _time.perf_counter() - _t0
    _p1 = 0.0

    rdtype = np.float64 if spec.dtype == np.complex128 else np.float32
    b = max(1, int(plane_batch))
    refuse_oversize_device_transform(b * n * n, "device irfft2 batch")
    for lo in range(0, n, slab):
        hi = min(lo + slab, n)
        out = np.empty((hi - lo, n, n), dtype=rdtype)

        def pass1(a, bb, dev, lo=lo, out=out):
            for i in range(lo + a, lo + bb, b):
                j = min(i + b, lo + bb)
                d = _from_device(
                    jnp.fft.irfft2(_to_device(spec[i:j], dev, transfer),
                                   s=(n, n), axes=(-2, -1)), transfer)
                _check_spectral_dtype(d.dtype, rdtype, "device irfft2")
                out[i - lo : j - lo] = d

        _t0 = _time.perf_counter()
        _run_parts(pass1, partition_units(hi - lo, len(devs), b), devs)
        _p1 += _time.perf_counter() - _t0
        if timings is not None:
            timings["pass1_s"] = _p1
            timings["pass2_s"] = _p2
        yield lo, out


_CARD_PROGRAMS = {}
_CARD_LOCK = threading.Lock()


def _card_program(key, build):
    with _CARD_LOCK:
        fn = _CARD_PROGRAMS.get(key)
        if fn is None:
            fn = _CARD_PROGRAMS[key] = build()
    return fn


def inverse_to_card_shards(spec, n_mesh, shards, plane_batch=1, pencil_batch=1,
                           timings=None, transfer="pageable", pass2=True):
    """`inverse_to_slabs_device` with the real-space planes written onto the cards.

    `shards` is `(x0, nx, device)` per card: that card receives global x-planes
    `x0 .. x0 + nx - 1` (mod n) as one `(nx, n, n)` device array, in that order.
    Ranges may overlap and may wrap -- a halo plane is simply transformed on
    every card that holds it. Returns the arrays, one per shard. MUTATES spec,
    exactly as `inverse_to_slabs_device` does.

    Pass 2 is that function's, split across the shards' devices. Pass 1 runs one
    thread per card: each plane's `irfft2` is computed on the card that holds it
    and written into its array there by a donated plane-set program, so no
    real-space plane crosses back to the host and no host mesh is assembled.

    BITWISE `device.coarse.shard_coarse_meshes` of the host-slab path's mesh at
    the same `pencil_batch`: every plane is its own batch on both paths. That is
    also why `plane_batch` must be 1 -- a plane held by two cards would otherwise
    be transformed in two different batches.

    `pass2=False` skips the axis-0 pass, as in `inverse_to_slabs_device`.
    """
    import jax
    import jax.numpy as jnp

    if int(plane_batch) != 1:
        raise ValueError(
            f"plane_batch={plane_batch}: the card inverse transforms each plane on its "
            "own, because a plane held by two cards would otherwise sit in two "
            "different batches and come back as two different planes")
    n = int(n_mesh)
    ranges = [(int(x0), int(nx), dev) for x0, nx, dev in shards]
    if not ranges:
        raise ValueError("shards= was empty")
    for x0, nx, _dev in ranges:
        if nx < 1:
            raise ValueError(f"shard at x0={x0} holds {nx} planes")
    rdtype = np.dtype(np.float64 if spec.dtype == np.complex128 else np.float32)
    refuse_oversize_device_transform(n * n, "device irfft2 plane")
    devs = [dev for _x0, _nx, dev in ranges]

    def _pass2(a, b, dev):
        fft_axis0_device_inplace(spec[:, a:b, :], inverse=True,
                                 pencil_batch=pencil_batch, device=dev,
                                 transfer=transfer)

    _t0 = _time.perf_counter()
    if pass2:
        _run_parts(_pass2, partition_units(spec.shape[1], len(devs), pencil_batch), devs)
    _p2 = _time.perf_counter() - _t0

    put = _card_program(("plane_set",), lambda: jax.jit(
        lambda m, i, p: m.at[i].set(p), donate_argnums=0))
    out = [None] * len(ranges)

    def pass1(k, _k1, dev):
        x0, nx, _ = ranges[k]
        shape = (nx, n, n)
        zeros = _card_program(("zeros", shape, rdtype.str), lambda: jax.jit(
            lambda z: jnp.broadcast_to(z, shape)))
        z = np.zeros((), dtype=rdtype)
        m = zeros(jnp.asarray(z) if dev is None else jax.device_put(z, dev))
        for i in range(nx):
            g = (x0 + i) % n
            d = jnp.fft.irfft2(_to_device(spec[g:g + 1], dev, transfer), s=(n, n),
                               axes=(-2, -1))
            _check_spectral_dtype(d.dtype, rdtype, "device irfft2")
            idx = np.int64(i)
            m = put(m, jnp.asarray(idx) if dev is None else jax.device_put(idx, dev), d[0])
        out[k] = jax.block_until_ready(m)

    _t0 = _time.perf_counter()
    _run_parts(pass1, [(k, k + 1) for k in range(len(ranges))], devs)
    if timings is not None:
        timings["pass1_s"] = _time.perf_counter() - _t0
        timings["pass2_s"] = _p2
    return out


# ---------------------------------------------------------------------------
# k-space kernels folded into the device axis-0 pass (D6)
# ---------------------------------------------------------------------------
#
# The host IC generator multiplies each spectrum by its kernel in a
# single-threaded numpy pass, and copies the spectrum first whenever it is
# needed again: at 4096^3 that is a 275 GB host pass per kernel plus a 275 GB
# copy. Here the multiply rides the y-pencil block the axis-0 pass already
# sends to a card, so neither the host multiply nor the host copy exists, and
# the host builds nothing O(N^3).
#
# With x64 on, each single kernel is BITWISE its host twin (float64 multiplier,
# one cast to the spectrum dtype, on both sides; measured 2026-09-14). A product
# of kernels rounds ONCE here where the host chain rounds after every factor, so
# it is bitwise one host pass with the product function, not the chain (~73
# eps x rms apart at 64^3 f32). With x64 off the multiplier is built in float32
# and moves 5-65 eps x rms (colour worst: log/exp interpolation). Performance
# over bitwise is JC's call for this path (2026-09-14).


class KSpaceKernel:
    """A multiplier on the rfft half-grid, evaluated per pencil block on a card.

    Build with the classmethods and combine with `*` (factors multiply in
    order). Conventions are `_ik_over_k2_slab`, `deriv2_spec` and
    `mul_radial_inplace`'s: fftfreq-signed k, k^2 -> 1 at k = 0 for the
    derivative kernels (whose numerators vanish there), and radial kernels
    evaluated at the grid's smallest nonzero |k| at k = 0 and then overwritten
    with their `dc_value`.
    """

    __slots__ = ("factors", "consts")

    def __init__(self, factors, consts):
        self.factors = tuple(factors)
        self.consts = tuple(consts)

    def __mul__(self, other):
        return KSpaceKernel(self.factors + other.factors, self.consts + other.consts)

    @property
    def key(self):
        return self.factors

    @classmethod
    def grad_invk2(cls, axis):
        """ik_axis / k^2 -- `grad_invk2_spec`'s kernel."""
        return cls([("grad", int(axis))], [None])

    @classmethod
    def deriv2(cls, i, j):
        """(ik_i)(ik_j) / k^2 = -k_i k_j / k^2 -- `deriv2_spec`'s kernel."""
        return cls([("deriv2", int(i), int(j))], [None])

    @classmethod
    def colour(cls, table, n_mesh, box_size, dc_value=0.0):
        """sqrt(P(|k|) N^3 / L^3) from an ICKTable -- `ic._colour_fn`'s kernel."""
        n, box = int(n_mesh), float(box_size)
        k_min = _refuse_off_table(table, n, box)
        return cls([("colour", float(n**3 / box**3), k_min, float(dc_value))],
                   [(np.log(table.k), np.log(table.P))])

    @classmethod
    def poisson(cls, cosmo, table, n_mesh, box_size, inverse=False, z=0.0, dc_value=1.0):
        """M(|k|, z), or 1/M -- `ic._poisson_fn`'s kernel on the table transfer."""
        from .ic import C_OVER_H0
        from .cosmology import growth_factor_md

        n, box = int(n_mesh), float(box_size)
        k_min = _refuse_off_table(table, n, box)
        amp = (2.0 / 3.0) * C_OVER_H0**2 * growth_factor_md(1.0 / (1.0 + z), cosmo) / cosmo.Omega_m
        return cls([("poisson", float(amp), bool(inverse), k_min, float(dc_value))],
                   [(np.log(table.k), np.asarray(table.T, dtype=np.float64))])


class ArrayKernel:
    """A multiplier the CALLER already holds as arrays, applied per pencil block.

    `KSpaceKernel` evaluates analytic factors on the card from k alone; the engine's
    coarse kernel is not analytic here -- `pref` and the CIC match factor are real
    half-grids built once per run by `forces.coarse_kernel_parts`, and the `ik_j` are
    low-rank. This carries them into the same pass, so the kernel multiply happens where
    the axis-0 transform already reads the block instead of in a separate host traversal
    over the whole half-grid (85 s of a 123 s coarse solve at 4096^3, gb 1003657).

    `halfgrids` are (N, N, M) real arrays sliced on the pencil axis; `lowrank` are arrays
    broadcastable to a block, each tagged with the axis it varies along (0, 1 or 2), and
    the axis-1 one is sliced with the block. The product is formed in the order given,
    left to right, which is how the caller's host expression associates -- `(pref * ik)
    * mf`, the association `coarse_kernel_parts` refuses to change. `cdtype` is the
    complex type the product is cast to before it multiplies the spectrum.
    """

    __slots__ = ("terms", "cdtype")

    def __init__(self, terms, cdtype):
        self.terms = tuple(terms)
        self.cdtype = np.dtype(cdtype)
        for kind, obj, axis in self.terms:
            if kind not in ("half", "low"):
                raise ValueError(f"term kind {kind!r} is not 'half' or 'low'")
            if kind == "low" and axis not in (0, 1, 2):
                raise ValueError(f"low-rank term varies along axis {axis}, not 0, 1 or 2")
            if obj is None:
                raise ValueError("a kernel term is None; drop it instead")

    @classmethod
    def coarse(cls, pref, ik, axis, mf, cdtype):
        """`(pref * ik_axis) * mf` -- the engine's coarse kernel, `mf` optional.

        `ik` is cast to `cdtype` HERE, before it meets `pref`, because that is the order
        `forces.coarse_kernel_slab` casts in and the cast is not associative with the
        multiply at f32."""
        terms = [("half", pref, None), ("low", np.asarray(ik).astype(cdtype), axis)]
        if mf is not None:
            terms.append(("half", mf, None))
        return cls(terms, cdtype)

    @property
    def key(self):
        """Program cache key: structure and dtypes, never the arrays' contents."""
        return tuple((kind, None if kind == "half" else axis,
                      np.dtype(obj.dtype).str, None if kind == "half" else obj.shape)
                     for kind, obj, axis in self.terms) + (self.cdtype.str,)

    def blocks(self, lo, hi):
        """The host arrays for pencil block [lo, hi), in term order."""
        out = []
        for kind, obj, axis in self.terms:
            if kind == "half":
                out.append(np.ascontiguousarray(obj[:, lo:hi, :]))
            elif axis == 1:
                out.append(np.ascontiguousarray(np.asarray(obj).reshape(-1)[lo:hi]))
            else:
                out.append(np.ascontiguousarray(obj))
        return tuple(out)

    def on_card(self, blocks):
        """The product for one block, cast to `cdtype`, from the uploaded `blocks`."""
        import jax.numpy as jnp

        m = None
        for (kind, _obj, axis), b in zip(self.terms, blocks):
            v = b if kind == "half" else jnp.reshape(
                b, (-1, 1, 1) if axis == 0 else (1, -1, 1) if axis == 1 else (1, 1, -1))
            m = v.astype(self.cdtype) if m is None else m * v
        return m


def _refuse_off_table(table, n, box):
    """The realized |k| range must sit inside the table: the card does not refuse."""
    k_min = abs(float(_kz(n, box)[1]))
    k_max = float(np.sqrt(3.0) * np.pi * n / box)
    if k_min < table.k[0] or k_max > table.k[-1]:
        raise ValueError(
            f"grid |k| range [{k_min:.3g}, {k_max:.3g}] is outside the table's "
            f"[{table.k[0]:.3g}, {table.k[-1]:.3g}]; the device kernel would extrapolate")
    return k_min


def _kernel_on_card(factors, consts, kx, ky, kz):
    """The multiplier for one pencil block, shape broadcastable to (N, b, M)."""
    import jax.numpy as jnp

    kxs, kys, kzs = kx[:, None, None], ky[None, :, None], kz[None, None, :]
    k2 = kxs**2 + kys**2 + kzs**2
    origin = (kx == 0)[:, None, None] & (ky == 0)[None, :, None] & (kz == 0)[None, None, :]
    k2_1 = jnp.where(origin, jnp.ones_like(k2), k2)
    comps = (kxs, kys, kzs)
    m = None
    for f, c in zip(factors, consts):
        kind = f[0]
        if kind == "grad":
            v = (1j * comps[f[1]]) / k2_1
        elif kind == "deriv2":
            v = -(comps[f[1]] * comps[f[2]]) / k2_1
        else:
            k_min, dc = f[-2], f[-1]
            kk = jnp.where(origin, jnp.full_like(k2, k_min), jnp.sqrt(k2))
            lnk = jnp.log(kk)
            if kind == "colour":
                v = jnp.sqrt(jnp.exp(jnp.interp(lnk, c[0], c[1])) * f[1])
            elif kind == "poisson":
                v = f[1] * kk**2 * jnp.interp(lnk, c[0], c[1])
                if f[2]:
                    v = 1.0 / v
            else:
                raise ValueError(f"unknown kernel factor {kind!r}")
            v = jnp.where(origin, jnp.full_like(v, dc), v)
        m = v if m is None else m * v
    return m


def kspace_pass_device(sources, n_mesh, box_size=1.0, kernel=None, out=None, inverse=False,
                       transform=True, devices=None, pencil_batch=1, timings=None,
                       transfer="pageable"):
    """Combine spectra, apply a k-space kernel and run the axis-0 (i)fft, on the cards.

    `sources` is a sequence of `(coef, spec)`, every spec an (N, N, M) host array
    of one complex dtype. Each y-pencil block `sum(coef * spec[:, y0:y1, :])` is
    sent to a card, where the kernel is applied AFTER the axis-0 fft (forward:
    the array is pass 1's output) or BEFORE the axis-0 ifft (inverse), and the
    result is written into `out[:, y0:y1, :]`. `transform=False` applies the
    combination and kernel only. `kernel=None` is the identity.

    `kernel` is a `KSpaceKernel` (analytic, evaluated on the card from k) or an
    `ArrayKernel` (the caller's own half-grids and low-rank factors, uploaded per block).

    `out` defaults to a new array and MAY be one of the sources: each thread
    reads and writes only its own y range, so the pass is in place. `devices`
    and `pencil_batch` split exactly as in `forward_from_slabs_device`, and the
    result is bitwise the same at every card count.

    Replaces, for the device IC generator, `mul_radial_inplace` /
    `grad_invk2_spec` / `deriv2_spec` followed by `fft_axis0_inplace`, and the
    host spectrum copies those require.
    """
    import jax
    import jax.numpy as jnp

    srcs = [(float(c), s) for c, s in sources]
    if not srcs:
        raise ValueError("sources= was empty")
    shape, cdt = srcs[0][1].shape, np.dtype(srcs[0][1].dtype)
    n = int(n_mesh)
    if shape != _spec_shape(n):
        raise ValueError(f"source shape {shape} is not the (N, N, N//2+1) half-grid for N={n}")
    for _c, s in srcs:
        if s.shape != shape or np.dtype(s.dtype) != cdt:
            raise ValueError("sources disagree on shape or dtype")
    if cdt not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise TypeError(f"sources must be complex spectra, got {cdt.name}")
    rdt = np.dtype(np.float64 if cdt == np.complex128 else np.float32)
    _require_x64_for(rdt)
    if out is None:
        out = np.empty(shape, dtype=cdt)
    elif out.shape != shape or np.dtype(out.dtype) != cdt:
        raise ValueError("out must match the sources' shape and dtype")
    b = max(1, int(pencil_batch))
    refuse_oversize_device_transform(b * shape[0] * shape[2], "device k-space pass batch")

    kd = np.float64 if jax.config.jax_enable_x64 else np.float32
    kx = _kx(n, box_size).astype(kd)
    kz = _kz(n, box_size).astype(kd)
    arr_kernel = kernel if isinstance(kernel, ArrayKernel) else None
    factors = () if kernel is None or arr_kernel is not None else kernel.factors
    consts = () if kernel is None or arr_kernel is not None else kernel.consts
    n_src = len(srcs)

    def build():
        def fn(coefs, blocks, kxd, kyd, kzd, cst, kblocks):
            acc = coefs[0] * blocks[0]
            for c, blk in zip(coefs[1:], blocks[1:]):
                acc = acc + c * blk
            if transform and not inverse:
                acc = jnp.fft.fft(acc, axis=0)
            if factors:
                m = _kernel_on_card(factors, cst, kxd, kyd, kzd)
                acc = acc * m.astype(acc.dtype)
            if arr_kernel is not None:
                acc = acc * arr_kernel.on_card(kblocks)
            if transform and inverse:
                acc = jnp.fft.ifft(acc, axis=0)
            return acc

        return jax.jit(fn)

    prog = _card_program(("kspace", factors, n_src, bool(inverse), bool(transform),
                          None if arr_kernel is None else arr_kernel.key), build)
    devs = _devices_or_default(devices)

    def part(a, z, dev):
        put = (lambda x: jnp.asarray(x)) if dev is None else (lambda x: jax.device_put(x, dev))
        kxd, kzd = put(kx), put(kz)
        cst = tuple(None if c is None else (put(c[0].astype(kd)), put(c[1].astype(kd)))
                    for c in consts)
        coefs = tuple(put(np.asarray(c, dtype=rdt)) for c, _s in srcs)
        for lo in range(a, z, b):
            hi = min(lo + b, z)
            blocks = tuple(_to_device(np.ascontiguousarray(s[:, lo:hi, :]), dev, transfer)
                           for _c, s in srcs)
            kblocks = () if arr_kernel is None else tuple(
                _to_device(x, dev, transfer) for x in arr_kernel.blocks(lo, hi))
            d = _from_device(prog(coefs, blocks, kxd, put(kx[lo:hi]), kzd, cst, kblocks),
                             transfer)
            _check_spectral_dtype(d.dtype, cdt, "device k-space pass")
            out[:, lo:hi, :] = d

    _t0 = _time.perf_counter()
    _run_parts(part, partition_units(shape[1], len(devs), b), devs)
    if timings is not None:
        timings["kspace_s"] = timings.get("kspace_s", 0.0) + _time.perf_counter() - _t0
    return out


# ---------------------------------------------------------------------------
# pass 1 on the cards for the IC stage: noise drawn where it is transformed, and
# squares accumulated where they are inverse-transformed (D6)
# ---------------------------------------------------------------------------


def zeros_card_shards(n_mesh, devices, dtype=np.float32):
    """Card shards of zeros tiling x-planes [0, N): `[{lo, hi, device, delta}]`.

    The shard format `forward_from_card_planes` consumes, one contiguous run of
    x-planes per device (`partition_units` at unit 1).
    """
    import jax
    import jax.numpy as jnp

    n = int(n_mesh)
    dt = np.dtype(dtype)
    _require_x64_for(dt)
    shards = []
    for (lo, hi), dev in zip(partition_units(n, len(devices), 1), devices):
        shape = (hi - lo, n, n)
        zeros = _card_program(("zeros", shape, dt.str), lambda shape=shape: jax.jit(
            lambda z: jnp.broadcast_to(z, shape)))
        z = np.zeros((), dtype=dt)
        shards.append(dict(lo=lo, hi=hi, device=dev,
                           delta=zeros(jnp.asarray(z) if dev is None else jax.device_put(z, dev))))
    return shards


def noise_forward_cards(key, n_mesh, devices, fdtype=np.float32, kernel=None, box_size=1.0,
                        pencil_batch=1, timings=None, transfer="pageable"):
    """Plane-keyed white noise drawn ON the cards, forward-transformed, kernel applied.

    Plane i is `jax.random.normal(fold_in(key, i), (N, N))`, drawn and `rfft2`d on
    the card that owns x-plane i; only its 2-D spectrum crosses to the host. Then
    pass 2 with `kernel` folded in (`kspace_pass_device`, in place). Returns the
    host (N, N, N//2+1) spectrum.

    The same construction as `ic.white_plane`, but drawn on the card's backend:
    the normal transform's bits are not specified across backends, so the result
    is a DIFFERENT stream from `ic.IC_STREAM` on a GPU and carries
    `ic.IC_STREAM_DEVICE`. Bitwise the same at every card count (each plane is
    drawn and transformed on its own).
    """
    import jax
    import jax.numpy as jnp

    n = int(n_mesh)
    dt = np.dtype(fdtype)
    _require_x64_for(dt)
    refuse_oversize_device_transform(n * n, "device noise plane")
    devs = list(devices)
    cd = _cdtype_for(dt)
    spec = np.empty(_spec_shape(n), dtype=cd)
    jdt = jnp.dtype(dt)
    draw = _card_program(("noise_rfft2", n, dt.str), lambda: jax.jit(
        lambda k, i: jnp.fft.rfft2(jax.random.normal(jax.random.fold_in(k, i), (n, n),
                                                     dtype=jdt))))

    def pass1(k, _k1, dev):
        lo, hi = parts[k]
        kd = jnp.asarray(key) if dev is None else jax.device_put(key, dev)
        for i in range(lo, hi):
            idx = np.uint32(i)
            d = _from_device(draw(kd, jnp.asarray(idx) if dev is None
                                  else jax.device_put(idx, dev)), transfer)
            _check_spectral_dtype(d.dtype, cd, "device noise rfft2")
            spec[i] = d

    parts = partition_units(n, len(devs), 1)
    _t0 = _time.perf_counter()
    _run_parts(pass1, [(k, k + 1) for k in range(len(devs))], devs)
    _p1 = _time.perf_counter() - _t0
    _t0 = _time.perf_counter()
    kspace_pass_device([(1.0, spec)], n, box_size, kernel=kernel, out=spec, inverse=False,
                       devices=devs, pencil_batch=pencil_batch, transfer=transfer)
    if timings is not None:
        timings["pass1_s"] = _p1
        timings["pass2_s"] = _time.perf_counter() - _t0
    return spec


def inverse_accumulate_cards(sources, n_mesh, acc_shards, weight, kernel=None, box_size=1.0,
                             work=None, pencil_batch=1, timings=None, transfer="pageable"):
    """acc += weight * (inverse transform of kernel * sum(coef * spec))**2, on the cards.

    Pass 2 (`kspace_pass_device`, kernel folded in) writes into the host buffer
    `work`, which may not alias a source and is returned for reuse; the sources
    are left intact. Pass 1 runs one thread per card: each x-plane's `irfft2` is
    computed on the card that owns it and its weighted square is added into that
    card's shard of `acc_shards` (`zeros_card_shards` format) by a donated
    program, so no real-space plane crosses to the host. The square is taken in
    the accumulator's dtype. Returns `(acc_shards, work)` -- the shards hold new
    arrays; the old ones are donated.
    """
    import jax
    import jax.numpy as jnp

    n = int(n_mesh)
    srcs = [(c, s) for c, s in sources]
    shape, cdt = srcs[0][1].shape, np.dtype(srcs[0][1].dtype)
    if work is None:
        work = np.empty(shape, dtype=cdt)
    if any(np.shares_memory(work, s) for _c, s in srcs):
        raise ValueError("work aliases a source; the sources must survive the pass")
    acc_shards = sorted(acc_shards, key=lambda s: int(s["lo"]))
    edge = 0
    for s in acc_shards:
        if int(s["lo"]) != edge or tuple(s["delta"].shape) != (int(s["hi"]) - edge, n, n):
            raise ValueError("acc_shards must tile [0, N) with (hi - lo, N, N) arrays")
        edge = int(s["hi"])
    if edge != n:
        raise ValueError(f"acc_shards end at {edge}, not {n}")
    adt = np.dtype(acc_shards[0]["delta"].dtype)
    _require_x64_for(adt)
    refuse_oversize_device_transform(n * n, "device irfft2 plane")
    devs = [s["device"] for s in acc_shards]

    _t0 = _time.perf_counter()
    kspace_pass_device(srcs, n, box_size, kernel=kernel, out=work, inverse=True, devices=devs,
                       pencil_batch=pencil_batch, transfer=transfer)
    _p2 = _time.perf_counter() - _t0

    add = _card_program(("acc_sq", n, adt.str), lambda: jax.jit(
        lambda m, i, spec_plane, w: m.at[i].add(
            w * jnp.square(jnp.fft.irfft2(spec_plane, s=(n, n), axes=(-2, -1))[0]
                           .astype(m.dtype))),
        donate_argnums=0))

    def pass1(k, _k1, dev):
        s = acc_shards[k]
        lo, hi = int(s["lo"]), int(s["hi"])
        put = (lambda x: jnp.asarray(x)) if dev is None else (lambda x: jax.device_put(x, dev))
        m = s["delta"]
        w = put(np.asarray(weight, dtype=adt))
        for g in range(lo, hi):
            m = add(m, put(np.int64(g - lo)), _to_device(work[g:g + 1], dev, transfer), w)
        s["delta"] = jax.block_until_ready(m)

    _t0 = _time.perf_counter()
    _run_parts(pass1, [(k, k + 1) for k in range(len(acc_shards))], devs)
    if timings is not None:
        timings["pass1_s"] = timings.get("pass1_s", 0.0) + _time.perf_counter() - _t0
        timings["pass2_s"] = timings.get("pass2_s", 0.0) + _p2
    return acc_shards, work


# ---------------------------------------------------------------------------
# the roundtrip receipt
# ---------------------------------------------------------------------------


def plane_noise(n_mesh, plane, fdtype, seed=0):
    """One deterministic plane of white noise, keyed by its OWN index.

    Keyed per plane rather than per slab so the field a receipt transforms is
    independent of the streaming decomposition -- otherwise the slab-invariance
    gate would be comparing two different fields and would pass by construction.
    Same reason the IC stage keys its noise stream per plane.
    """
    rng = np.random.default_rng([int(seed), int(plane)])
    return rng.standard_normal((int(n_mesh), int(n_mesh))).astype(fdtype)


def roundtrip_residual(n_mesh, fdtype=np.float32, seed=0, slab=_DEF_SLAB,
                       plane_batch=1, pencil_batch=1, device=True):
    """max|forward-then-inverse - original| / rms(original). THE receipt.

    Streams: the field is regenerated plane by plane for the comparison rather
    than held, so this is runnable at 2048^3 where a resident field is 34.4 GB.

    This is the measurement that catches the silent-wrong-transform class. It is
    reported, never asserted here -- what a caller does with 3.8e+3 is a gate's
    decision, and `MAX_DEVICE_TRANSFORM_ELEMENTS` is what makes the wrong regime
    unreachable in the first place.
    """
    n = int(n_mesh)
    fwd = forward_from_slabs_device if device else None
    if device:
        spec = fwd(lambda lo, hi: np.stack(
            [plane_noise(n, i, fdtype, seed) for i in range(lo, hi)]),
            n, slab=slab, plane_batch=plane_batch, pencil_batch=pencil_batch)
        gen = inverse_to_slabs_device(spec, n, slab=slab,
                                      plane_batch=plane_batch,
                                      pencil_batch=pencil_batch)
    else:
        spec = forward_from_slabs(lambda lo, hi: np.stack(
            [plane_noise(n, i, fdtype, seed) for i in range(lo, hi)]),
            n, slab=slab)
        gen = inverse_to_slabs(spec, n, slab=slab)
    max_abs, sq, count = 0.0, 0.0, 0
    for lo, got in gen:
        for i in range(got.shape[0]):
            want = plane_noise(n, lo + i, fdtype, seed)
            max_abs = max(max_abs, float(np.max(np.abs(got[i] - want))))
            sq += float(np.sum(np.square(want, dtype=np.float64)))
            count += want.size
    rms = np.sqrt(sq / max(count, 1))
    return {"n_mesh": n, "dtype": np.dtype(fdtype).name, "slab": int(slab),
            "plane_batch": int(plane_batch), "pencil_batch": int(pencil_batch),
            "device": bool(device), "max_abs": max_abs, "rms": float(rms),
            "residual": float(max_abs / rms) if rms else float("inf")}


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
