"""Out-of-core 3D real FFTs: host-resident k-space, slab-streamed real space.

"Out of core" means out of DEVICE memory: the spectrum lives in host numpy and real-space
fields are touched one axis-0 slab at a time, so the peak is one spectral array plus
O(plane) buffers. k-space is never spilled to disk.

The layer is DEFINED by this factorization (not by `np.fft.rfftn`, which it does not
match bitwise):

  pass 1  scipy.fft.rfft2 of ONE axis-0 plane at a time;
  pass 2  scipy.fft.fft(axis=0) of ONE y-pencil-plane (N, N//2+1) at a time.

The unit is one plane because pocketfft results depend on batch size at the bit level;
with a fixed unit, slab thickness is an outer loop bound and streamed == monolithic
bitwise. scipy.fft, not np.fft, because np.fft upcasts f32 to complex128.

Spectral multipliers follow `forces.k_components` (fftfreq-signed ik, 1/k^2 = 1 at
k = 0) and are built per slab in float64, never as a full O(N^3) grid.
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
    """Pass 2: (i)fft along axis 0, one y-pencil-plane (N, N//2+1) at a time, in place.

    `progress(stage, done, total)`, if given, is called once per plane
    (`inexor.progress.Heartbeat`).
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

    slab_fn(lo, hi) -> (hi-lo, N, N) real array of axis-0 planes [lo, hi); the real
    field is never materialized. Returns the (N, N, N//2+1) spectrum, complex64 or
    complex128 following the slabs' dtype. `slab` is a memory knob only and cannot move
    a bit. `progress(stage, done, total)` reports the passes separately ("fft plane",
    then "fft axis0").
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
        # directly into the target rows: a whole-slab intermediate is a full spectrum
        # copy at slab = n
        for i in range(hi - lo):
            spec[lo + i] = scipy.fft.rfft2(s[i], workers=workers)
        if progress is not None:
            progress("fft plane", hi, n)
    fft_axis0_inplace(spec, workers=workers, progress=progress)
    return spec


def inverse_to_slabs(spec, n_mesh, slab=_DEF_SLAB, workers=_DEF_WORKERS):
    """Inverse of `forward_from_slabs`, yielding (lo, real_slab) in axis-0 order.

    MUTATES spec (the axis-0 inverse runs in place); copy it first if it is needed again.
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
    """Monolithic convenience; consumes (mutates) spec like `inverse_to_slabs`.

    Iterates at the default slab, not slab = n, to avoid a second full-field buffer.
    """
    n = int(n_mesh)
    out = np.empty((n, n, n), dtype=np.float64 if spec.dtype == np.complex128 else np.float32)
    for lo, s in inverse_to_slabs(spec, n, slab=_DEF_SLAB, workers=workers):
        out[lo : lo + s.shape[0]] = s
    return out


# ---------------------------------------------------------------------------
# the device path: the SAME factorization, planes transformed on an accelerator;
# the spectrum stays host-resident and only planes cross to the device
# ---------------------------------------------------------------------------

#: A device transform at or above this many elements is refused, never attempted: a
#: 1536^3 f32 `jnp.fft.rfftn` on a GB200 returns a wrong transform silently (roundtrip
#: max|d|/rms ~4e3 vs ~3e-6 at 1024^3). The bound is empirical (likely 32-bit FFT plan
#: indexing). Per-plane transforms sit far below it; the guard covers batch knobs and
#: monolithic callers.
MAX_DEVICE_TRANSFORM_ELEMENTS = 2**31


def refuse_oversize_device_transform(n_elements, what="transform"):
    """Refuse a device FFT big enough to hit the silent-wrong-result class."""
    n_elements = int(n_elements)
    if n_elements >= MAX_DEVICE_TRANSFORM_ELEMENTS:
        raise ValueError(
            f"{what} of {n_elements:,} elements is at or above the "
            f"{MAX_DEVICE_TRANSFORM_ELEMENTS:,} bound where a device FFT has "
            "been measured to return a wrong result silently (a 1536^3 f32 "
            "roundtrip error of 3.8e+3). "
            "Factorize it: the plane is the unit."
        )


def _require_x64_for(dtype):
    """Refuse a float64 device transform unless the caller enabled x64.

    The library never toggles `jax_enable_x64`; with it off, f64 would silently narrow.
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
    """Refuse a device FFT output whose dtype differs from the expected one."""
    if np.dtype(got) != np.dtype(want):
        raise TypeError(
            f"{what} returned {np.dtype(got).name}, want {np.dtype(want).name}: "
            "the device FFT changed precision underneath the caller"
        )


#: How a host buffer crosses to the device and back. "pageable": ordinary numpy, which
#: the driver copies into its own staging buffer. "staged": an explicit copy into
#: `pinned_host` (page-locked) memory, then a direct DMA. The policy changes the route,
#: never the arithmetic: results are bitwise identical.
TRANSFER_POLICIES = ("pageable", "staged")


def _pinned_sharding(device):
    import jax

    return jax.sharding.SingleDeviceSharding(device, memory_kind="pinned_host")


def _device_sharding(device):
    """Device-memory target for a buffer in another memory kind.

    Must be a sharding, not a bare `Device`: `jax.device_put(pinned, dev)` raises a
    memory-kind mismatch because a bare device names no memory kind to move to.
    """
    import jax

    return jax.sharding.SingleDeviceSharding(device)


def staging_supported(device=None):
    """Can this backend do the host -> pinned_host -> device round trip?

    Callers check this and refuse rather than silently falling back to pageable.
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
    """Host -> device under a transfer policy; `device=None` means jax's default device.

    A named device is committed with `jax.device_put` (`jnp.asarray` would place every
    part on device 0).
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

    numpy in, numpy out. `plane_batch` is part of the transform's definition (FFT bits
    may depend on batch size), so the default is 1, matching the host unit; at fixed
    `plane_batch`, `slab` cannot move a bit.
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

    `pencil_batch` widens the one-pencil-plane unit and, like `plane_batch`, is part
    of the transform's definition.
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
# splitting a pass across devices: a partition of the plane / pencil loop, with no
# inter-device communication (the spectrum stays host-resident)
# ---------------------------------------------------------------------------


def partition_units(total, n_parts, unit):
    """Split [0, total) into `n_parts` contiguous ranges, boundaries on `unit`.

    Batches restart at every part, so aligned boundaries keep the unpartitioned batch
    sequence and make W devices bitwise equal to one. Refuses when there are fewer whole
    units than parts rather than returning empty ranges.
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

    Threads share the host-resident state; jax releases the GIL across dispatch and
    transfer. A single part runs inline.
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

    `kernel` (a `KSpaceKernel` or `ArrayKernel`), if given, is applied on the card inside
    pass 2 (`kspace_pass_device`). `devices` is a sequence of jax devices to split each
    pass across, or None for one default device; the split is batch-aligned, so the
    spectrum is bitwise identical at every device count (`partition_units`). Pass 1
    splits planes within a slab; the slab loop is sequential because `slab_fn` is not
    assumed re-entrant. `timings`, if a dict, receives `pass1_s` and `pass2_s`.
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

    `shards` are dicts with `lo`, `hi`, `device`, `delta`: x-planes [lo, hi) of the
    field, shape (hi - lo, N, N), on that device, together tiling [0, N)
    (`device.paint.coarse_delta_cards`). Pass 1 transforms each card's planes in place
    (one thread per card; no real-space plane crosses the bus); pass 2 is split across
    the same devices. Bitwise equal to the host-slab path at the same batches; card
    boundaries must be `plane_batch` multiples. `timings` receives `pass1_s`, `pass2_s`.
    The one-rank case of `forward_card_planes_to_pencils`.
    """
    n = int(n_mesh)
    return forward_card_planes_to_pencils(shards, n, [(0, n)], [(0, n)], None,
                                          plane_batch=plane_batch, pencil_batch=pencil_batch,
                                          timings=timings, transfer=transfer)


def forward_card_planes_to_pencils(shards, n_mesh, x_parts, y_parts, comm=None, plane_batch=1,
                                   pencil_batch=1, timings=None, transfer="pageable",
                                   receipt=None):
    """`forward_from_card_planes` across ranks: this rank's cards hold x-planes
    `x_parts[rank]` of the field, and it returns y-pencils `y_parts[rank]` = [y0, y1) of the
    spectrum with the axis-0 FFT done, shape (N, y1 - y0, N // 2 + 1).

    `x_parts` / `y_parts` are every rank's ranges in rank order, each tiling [0, N). Pass 1
    writes each transformed plane's own y-block straight into the result and the other
    ranks' blocks into send blocks; one `Alltoallv` delivers them into x-ranges of the
    result, so nothing is copied into place. Pass 2 splits this rank's pencils over its
    cards. Bitwise the one-rank spectrum's pencils (batches restart only on batch-aligned
    boundaries; a transpose is a copy). One rank (`comm` None or of size 1) exchanges
    nothing. `receipt`, if a dict, accumulates `forward_sent_bytes`; `timings` receives
    `pass1_s`, `transpose_s`, `pass2_s`. Every rank must call this.
    """
    import jax.numpy as jnp

    n = int(n_mesh)
    r, n_ranks = (0, 1) if comm is None else (int(comm.rank), int(comm.size))
    x_parts = [(int(a), int(z)) for a, z in x_parts]
    y_parts = [(int(a), int(z)) for a, z in y_parts]
    if len(x_parts) != n_ranks or len(y_parts) != n_ranks:
        raise ValueError(f"{n_ranks} rank(s) but {len(x_parts)} x and {len(y_parts)} y ranges")
    x_lo, x_hi = x_parts[r]
    y_lo, y_hi = y_parts[r]
    m = n // 2 + 1
    shards = sorted(shards, key=lambda s: int(s["lo"]))
    b = max(1, int(plane_batch))
    edge = x_lo
    for s in shards:
        lo, hi = int(s["lo"]), int(s["hi"])
        if lo != edge:
            raise ValueError(f"card shards do not tile [{x_lo}, {x_hi}): a shard starts at {lo} "
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
    if edge != x_hi:
        raise ValueError(f"card shards do not tile [{x_lo}, {x_hi}): they end at {edge}")
    dt = np.dtype(shards[0]["delta"].dtype)
    if any(np.dtype(s["delta"].dtype) != dt for s in shards):
        raise TypeError("card shards disagree on dtype")
    _require_x64_for(dt)
    refuse_oversize_device_transform(b * n * n, "device rfft2 batch")
    cd = _cdtype_for(dt)
    spec = np.empty((n, y_hi - y_lo, m), dtype=cd)
    # the other ranks' y-blocks of this rank's planes, in x order
    sends = [None if j == r else np.empty((x_hi - x_lo, y1 - y0, m), dtype=cd)
             for j, (y0, y1) in enumerate(y_parts)]
    devs = [s["device"] for s in shards]

    def pass1(k, _k1, _dev):
        s = shards[k]
        lo, w = int(s["lo"]), int(s["hi"]) - int(s["lo"])
        for a in range(0, w, b):
            z = min(a + b, w)
            d = _from_device(jnp.fft.rfft2(s["delta"][a:z], axes=(-2, -1)), transfer)
            _check_spectral_dtype(d.dtype, cd, "device rfft2")
            spec[lo + a:lo + z] = d[:, y_lo:y_hi]
            for j, (y0, y1) in enumerate(y_parts):
                if j != r:
                    sends[j][lo - x_lo + a:lo - x_lo + z] = d[:, y0:y1]

    _t0 = _time.perf_counter()
    _run_parts(pass1, [(k, k + 1) for k in range(len(shards))], devs)
    _p1 = _time.perf_counter() - _t0

    _t0 = _time.perf_counter()
    if n_ranks > 1:
        empty = np.empty(0, dtype=np.uint8)
        comm.Alltoallv([empty if j == r else sends[j] for j in range(n_ranks)],
                       [empty if i == r else spec[a:z] for i, (a, z) in enumerate(x_parts)])
        if receipt is not None:
            receipt["forward_sent_bytes"] = receipt.get("forward_sent_bytes", 0) + sum(
                int(x.nbytes) for x in sends if x is not None)
    sends = None
    _tt = _time.perf_counter() - _t0

    def pass2(a, z, dev):
        fft_axis0_device_inplace(spec[:, a:z, :], pencil_batch=pencil_batch,
                                 device=dev, transfer=transfer)

    _t0 = _time.perf_counter()
    _run_parts(pass2, partition_units(y_hi - y_lo, len(devs), pencil_batch), devs)
    if timings is not None:
        timings["pass1_s"] = _p1
        timings["transpose_s"] = _tt
        timings["pass2_s"] = _time.perf_counter() - _t0
    return spec


def inverse_to_slabs_device(spec, n_mesh, slab=_DEF_SLAB, plane_batch=1,
                            pencil_batch=1, devices=None, timings=None,
                            transfer="pageable", pass2=True):
    """Device twin of `inverse_to_slabs`. MUTATES spec, exactly as that one does.

    `devices` splits both passes as in `forward_from_slabs_device`, bitwise identically
    to one device. `pass2=False` skips the axis-0 pass, for a buffer `kspace_pass_device`
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
                           timings=None, transfer="pageable", pass2=True, plane_pieces=None):
    """`inverse_to_slabs_device` with the real-space planes written onto the cards.

    `shards` is `(x0, nx, device)` per card: that card receives global x-planes
    `x0 .. x0 + nx - 1` (mod n) as one `(nx, n, n)` device array. Ranges may overlap and
    wrap (halo planes are transformed on every card holding them). Returns one array per
    shard; MUTATES spec. Pass 1 runs `irfft2` per plane on its card, written by a donated
    plane-set program, so no real-space plane reaches the host.

    Bitwise `device.coarse.shard_coarse_meshes` of the host-slab mesh at the same
    `pencil_batch`. `plane_batch` must be 1 so a shared plane is never transformed in two
    different batches. `pass2=False` as in `inverse_to_slabs_device`.

    `plane_pieces(g)`, if given, returns plane g as host y-blocks `(1, ny_i, M)` in y order
    (`inverse_pencils_to_card_shards`); they are uploaded and concatenated on the card, which
    is exact. None reads `spec[g]` whole.
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
            if plane_pieces is None:
                x = _to_device(spec[g:g + 1], dev, transfer)
            else:
                ps = [_to_device(q, dev, transfer) for q in plane_pieces(g)]
                x = ps[0] if len(ps) == 1 else jnp.concatenate(ps, axis=1)
            d = jnp.fft.irfft2(x, s=(n, n), axes=(-2, -1))
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


def _plane_runs(ranges, n):
    """The x planes covered by `(x0, nx)` ranges (mod n), as maximal runs [a, b) in [0, n)."""
    planes = sorted({(int(x0) + i) % n for x0, nx in ranges for i in range(int(nx))})
    runs = []
    for g in planes:
        if runs and runs[-1][1] == g:
            runs[-1][1] = g + 1
        else:
            runs.append([g, g + 1])
    return [tuple(x) for x in runs]


def inverse_pencils_to_card_shards(work, n_mesh, shards, y_parts, comm=None, timings=None,
                                   transfer="pageable", receipt=None):
    """`inverse_to_card_shards` (after a folded axis-0 pass, `pass2=False`) across ranks:
    `work` holds this rank's y-pencils `y_parts[rank]` of the spectrum, shape
    (N, y1 - y0, M), and `shards` are this rank's cards' `(x0, nx, device)` plane ranges.

    Every rank's ranges are allgathered; the planes each rank needs go to it as runs of
    consecutive planes, sent as views of `work` (one `Alltoallv` per run index). Each plane
    then reaches its card as the ranks' y-blocks in rank order, the own block read in place,
    and is concatenated there (exact) before the `irfft2`. One rank exchanges nothing and
    runs `inverse_to_card_shards` unchanged. `receipt`, if a dict, accumulates
    `inverse_sent_bytes`; `timings` gains `transpose_s`. Every rank must call this.
    """
    n = int(n_mesh)
    m = n // 2 + 1
    r, n_ranks = (0, 1) if comm is None else (int(comm.rank), int(comm.size))
    y_parts = [(int(a), int(z)) for a, z in y_parts]
    y_lo, y_hi = y_parts[r]
    if tuple(work.shape) != (n, y_hi - y_lo, m):
        raise ValueError(f"work has shape {tuple(work.shape)}, want {(n, y_hi - y_lo, m)} "
                         f"(this rank's pencils [{y_lo}, {y_hi}))")
    if n_ranks == 1:
        return inverse_to_card_shards(work, n, shards, timings=timings, transfer=transfer,
                                      pass2=False)
    _t0 = _time.perf_counter()
    mine = _plane_runs([(x0, nx) for x0, nx, _dev in shards], n)
    runs = comm.allgather(mine)
    empty = np.empty(0, dtype=np.uint8)
    recv, sent = {}, 0
    for k in range(max(len(x) for x in runs)):
        sendbufs, recvbufs = [empty] * n_ranks, [empty] * n_ranks
        for j in range(n_ranks):
            if j != r and k < len(runs[j]):
                a, z = runs[j][k]
                sendbufs[j] = work[a:z]
                sent += int(sendbufs[j].nbytes)
        if k < len(mine):
            a, z = mine[k]
            for i, (y0, y1) in enumerate(y_parts):
                if i != r:
                    recv[(i, k)] = recvbufs[i] = np.empty((z - a, y1 - y0, m), dtype=work.dtype)
        comm.Alltoallv(sendbufs, recvbufs)
    _tt = _time.perf_counter() - _t0
    where = {g: (k, g - a) for k, (a, z) in enumerate(mine) for g in range(a, z)}

    def pieces(g):
        k, o = where[g]
        return [work[g:g + 1] if i == r else recv[(i, k)][o:o + 1] for i in range(n_ranks)]

    out = inverse_to_card_shards(work, n, shards, timings=timings, transfer=transfer,
                                 pass2=False, plane_pieces=pieces)
    if timings is not None:
        timings["transpose_s"] = _tt
    if receipt is not None:
        receipt["inverse_sent_bytes"] = receipt.get("inverse_sent_bytes", 0) + sent
    return out


# ---------------------------------------------------------------------------
# k-space kernels folded into the device axis-0 pass: the multiply rides the pencil
# block already on the card, so there is no host multiply or host spectrum copy.
#
# With x64 on, a single kernel is bitwise its host twin (float64 multiplier, one cast).
# A product of kernels rounds once, so it matches one host pass with the product
# function, not the host chain of per-factor passes. With x64 off the multiplier is
# float32 and differs by a few to tens of eps x rms.
# ---------------------------------------------------------------------------


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

    For non-analytic kernels such as the engine's coarse kernel (`pref` and the CIC match
    factor are real half-grids from `forces.coarse_kernel_parts`; `ik_j` is low-rank).
    `terms` are `(kind, obj, axis)`: "half" = (N, N, M) real host array sliced on the pencil
    axis; "card" = the same half-grid already resident on the cards, as a list of
    `(y_lo, y_hi, device, array)` y-row blocks; "low" = 1-D array varying along `axis`
    (0, 1 or 2), sliced when axis is 1. The product is formed left to right, matching the host
    expression's association `(pref * ik) * mf`. `cdtype` is the complex type of the product.
    """

    __slots__ = ("terms", "cdtype")

    def __init__(self, terms, cdtype):
        self.terms = tuple(terms)
        self.cdtype = np.dtype(cdtype)
        for kind, obj, axis in self.terms:
            if kind not in ("half", "low", "card"):
                raise ValueError(f"term kind {kind!r} is not 'half', 'low' or 'card'")
            if kind == "low" and axis not in (0, 1, 2):
                raise ValueError(f"low-rank term varies along axis {axis}, not 0, 1 or 2")
            if obj is None:
                raise ValueError("a kernel term is None; drop it instead")

    @classmethod
    def coarse(cls, pref, ik, axis, mf, cdtype, cards=None):
        """`(pref * ik_axis) * mf` -- the engine's coarse kernel, `mf` optional.

        `ik` is cast to `cdtype` before it meets `pref`, matching
        `forces.coarse_kernel_slab`'s order (the cast does not commute with the f32
        multiply). `cards` (the `cards` entry of `forces.coarse_kernel_parts`) replaces the
        host `pref` / `mf` with their card-resident blocks."""
        if cards is not None:
            pref = [(e["lo"], e["hi"], e["device"], e["pref"]) for e in cards]
            mf = (None if cards[0]["mf"] is None
                  else [(e["lo"], e["hi"], e["device"], e["mf"]) for e in cards])
        kind = "half" if cards is None else "card"
        terms = [(kind, pref, None), ("low", np.asarray(ik).astype(cdtype), axis)]
        if mf is not None:
            terms.append((kind, mf, None))
        return cls(terms, cdtype)

    @property
    def key(self):
        """Program cache key: structure and dtypes, never the arrays' contents. A card term
        keys as the host half-grid it replaces: the program is the same."""
        def dt(kind, obj):
            return np.dtype((obj[0][3] if kind == "card" else obj).dtype).str

        return tuple(("low" if kind == "low" else "half", axis if kind == "low" else None,
                      dt(kind, obj), obj.shape if kind == "low" else None)
                     for kind, obj, axis in self.terms) + (self.cdtype.str,)

    def blocks(self, lo, hi):
        """The host arrays for pencil block [lo, hi), in term order (host terms only)."""
        out = []
        for kind, obj, axis in self.terms:
            if kind == "card":
                raise ValueError("a card-resident term has no host block; use device_blocks")
            if kind == "half":
                out.append(np.ascontiguousarray(obj[:, lo:hi, :]))
            elif axis == 1:
                out.append(np.ascontiguousarray(np.asarray(obj).reshape(-1)[lo:hi]))
            else:
                out.append(np.ascontiguousarray(obj))
        return tuple(out)

    def device_blocks(self, lo, hi, device, transfer="pageable"):
        """The term arrays for pencil block [lo, hi) on `device`: host terms uploaded, card
        terms sliced where they live. Refuses a block no resident range on `device` covers,
        which would otherwise multiply by another card's rows."""
        out = []
        for kind, obj, axis in self.terms:
            if kind != "card":
                if kind == "half":
                    x = np.ascontiguousarray(obj[:, lo:hi, :])
                elif axis == 1:
                    x = np.ascontiguousarray(np.asarray(obj).reshape(-1)[lo:hi])
                else:
                    x = np.ascontiguousarray(obj)
                out.append(_to_device(x, device, transfer))
                continue
            hit = [(y0, a) for y0, y1, dev, a in obj
                   if dev == device and y0 <= lo and hi <= y1]
            if not hit:
                raise ValueError(
                    f"pencil block [{lo}, {hi}) on device {device} is inside no resident "
                    f"kernel range ({[(y0, y1, str(d)) for y0, y1, d, _a in obj]}); the "
                    "pass's pencil split and the kernel's placement disagree")
            y0, a = hit[0]
            out.append(a[:, lo - y0:hi - y0, :])
        return tuple(out)

    def on_card(self, blocks):
        """The product for one block, cast to `cdtype`, from the uploaded `blocks`."""
        import jax.numpy as jnp

        m = None
        for (kind, _obj, axis), b in zip(self.terms, blocks):
            v = b if kind in ("half", "card") else jnp.reshape(
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
                       transfer="pageable", pencils=None):
    """Combine spectra, apply a k-space kernel and run the axis-0 (i)fft, on the cards.

    `sources` is a sequence of `(coef, spec)`, each spec an (N, N, M) host array of one
    complex dtype. Each y-pencil block `sum(coef * spec[:, y0:y1, :])` goes to a card; the
    kernel is applied AFTER the axis-0 fft (forward; input is pass 1's output) or BEFORE
    the axis-0 ifft (inverse), and the result is written to `out[:, y0:y1, :]`.
    `transform=False` skips the fft. `kernel` is a `KSpaceKernel`, an `ArrayKernel`, or
    None (identity). `out` may be one of the sources (each thread touches only its own y
    range). Bitwise the same at every card count. Device replacement for
    `mul_radial_inplace` / `grad_invk2_spec` / `deriv2_spec` + `fft_axis0_inplace`.

    `pencils = (y0, y1)`: the sources and `out` hold only global y-pencils [y0, y1), shape
    (N, y1 - y0, M), one rank's share (`forward_card_planes_to_pencils`); the kernel is
    evaluated at those global rows. None is the whole half-grid.
    """
    import jax
    import jax.numpy as jnp

    srcs = [(float(c), s) for c, s in sources]
    if not srcs:
        raise ValueError("sources= was empty")
    shape, cdt = srcs[0][1].shape, np.dtype(srcs[0][1].dtype)
    n = int(n_mesh)
    y0, y1 = (0, n) if pencils is None else (int(pencils[0]), int(pencils[1]))
    if not 0 <= y0 < y1 <= n:
        raise ValueError(f"pencils [{y0}, {y1}) are not inside [0, {n})")
    if shape != (n, y1 - y0, n // 2 + 1):
        raise ValueError(f"source shape {shape} is not the (N, {y1 - y0}, N//2+1) half-grid "
                         f"pencils [{y0}, {y1}) for N={n}")
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
            kblocks = () if arr_kernel is None else arr_kernel.device_blocks(
                y0 + lo, y0 + hi, dev, transfer)
            d = _from_device(prog(coefs, blocks, kxd, put(kx[y0 + lo:y0 + hi]), kzd, cst,
                                  kblocks), transfer)
            _check_spectral_dtype(d.dtype, cdt, "device k-space pass")
            out[:, lo:hi, :] = d

    _t0 = _time.perf_counter()
    _run_parts(part, partition_units(shape[1], len(devs), b), devs)
    if timings is not None:
        timings["kspace_s"] = timings.get("kspace_s", 0.0) + _time.perf_counter() - _t0
    return out


# ---------------------------------------------------------------------------
# pass 1 on the cards for the IC stage: noise drawn where it is transformed, and
# squares accumulated where they are inverse-transformed
# ---------------------------------------------------------------------------


def zeros_card_shards(n_mesh, devices, dtype=np.float32):
    """Card shards of zeros tiling x-planes [0, N): `[{lo, hi, device, delta}]`.

    The format `forward_from_card_planes` consumes; one contiguous run per device.
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

    Plane i is `jax.random.normal(fold_in(key, i), (N, N))`, drawn and `rfft2`d on the
    card owning it; only its 2-D spectrum reaches the host. Pass 2 then applies `kernel`
    in place. Returns the host (N, N, N//2+1) spectrum. Same construction as
    `ic.white_plane`, but normal-sampling bits differ across backends, so on a GPU this
    is stream `ic.IC_STREAM_DEVICE`, not `ic.IC_STREAM`. Bitwise the same at every card
    count.
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

    Pass 2 writes into host buffer `work` (must not alias a source; sources survive).
    Pass 1 computes each x-plane's `irfft2` on its owning card and adds the weighted
    square, in the accumulator's dtype, into that card's shard of `acc_shards`
    (`zeros_card_shards` format) by a donated program. Returns `(acc_shards, work)`;
    the shards hold new arrays, the old ones are donated.
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
# roundtrip check
# ---------------------------------------------------------------------------


def plane_noise(n_mesh, plane, fdtype, seed=0):
    """One deterministic plane of white noise, keyed by its OWN index.

    Keyed per plane so the field is independent of the slab decomposition.
    """
    rng = np.random.default_rng([int(seed), int(plane)])
    return rng.standard_normal((int(n_mesh), int(n_mesh))).astype(fdtype)


def roundtrip_residual(n_mesh, fdtype=np.float32, seed=0, slab=_DEF_SLAB,
                       plane_batch=1, pencil_batch=1, device=True):
    """max|forward-then-inverse - original| / rms(original), returned in a dict.

    The field is regenerated plane by plane rather than held. Reported, not asserted:
    thresholds belong to the caller.
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
# spectral multipliers (slab-built in float64; k = 0 -> 1/k^2 = 1, ik zero there)
# ---------------------------------------------------------------------------


def mul_radial_inplace(spec, n_mesh, box_size, f_of_k, dc_value, slab=_DEF_SLAB):
    """spec *= f(|k|), evaluated per axis-0 slab; the DC bin gets dc_value.

    f_of_k receives a float64 |k| array whose k = 0 entry is replaced by the grid's
    smallest nonzero |k| (keeping it inside table range); that value is then overwritten
    by dc_value (0.0 zeroes the mean mode; 1.0 leaves DC untouched).
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

    A copy because the source spectrum is needed again for the other components.
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
# disk staging: explicit IO, not memmap
# ---------------------------------------------------------------------------


class StagedArray:
    """A disk-staged (n0, n, n) array written and read one axis-0 slab at a time.

    Plain `.npy` on disk (np.load can read it), accessed through explicit IO rather than
    memmap: dirty memmap pages count in the process's ru_maxrss, so memmap would inflate
    measured peak memory with reclaimable page cache.
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
# memory accounting and refusal
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
    """Raise MemoryError if the predicted peak exceeds `budget_bytes` (refuse rather
    than page); otherwise return the `plan_bytes` plan."""
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
