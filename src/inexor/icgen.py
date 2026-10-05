"""Streamed IC generation to T9 slab files on disk, and the loader/writer for that format.

`generate_t9_slabs` runs the IC stage (plane-keyed noise, table colour, f_NL transform, 2LPT,
T9 encode) without materializing a box-sized (N^3, 3) array: real fields are staged to disk
(the IC stage is the only disk-staged stage), spectra are host-resident one at a time, and
emission walks brick-aligned Lagrangian x-slabs through a sliding window into per-destination
slab files. `load_slot_state` reassembles a host-resident SlotState; `write_t9_slabs` is its
inverse.

The host generator is bitwise `SlotState.build` on `lpt.lpt_ics` output: every elementwise op
sequence matches `ic.linear_density` -> `lpt.lpt_ics` -> `SlotState.build`, every FFT goes
through ooc_fft's one-plane compute unit, and destination slabs concatenate contributions in
ascending source-slab (= Lagrangian) order so a stable sort by bucket key reproduces `build`'s
order within every brick. The max |u| is measured and the emission refuses if it reaches the
sliding window's depth.
"""

import json
import os
import re
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import ic, ooc_fft
from .codec import INT16_MAX, T9Layout
from .cosmology import growth_factor_2, growth_factor_a, growth_rate_2, growth_rate_a, ic_k_table
from .layout import DEFAULT_INDEX_DTYPE, _stable_sort_index, _to_index
from .lpt import _DIAG, _OFFDIAG, lpt2_source_from_spec
from .state import (
    SlotState,
    _alloc_geometry,
    _bucket_flat_brick_major,
    _encode_at,
    _scales_from_sorted,
    encode_positions_host,
)

# `-2` stores one velocity scale per brick; a `-1` file (one global scale) has otherwise
# identical arrays and would decode at the wrong scale silently, so the loader refuses it.
SCHEMA = "t9-slabs-2"
MANIFEST = "manifest.json"


STAGE_DIR = "stage"


def staged_names():
    """Names of the twenty (n, n, n) intermediates `generate_t9_slabs` writes under `stage/`.

    Unused once the manifest is written.
    """
    from .lpt import _DIAG, _OFFDIAG

    names = ["phi.npy", "delta.npy"]
    names += [f"psi{o}_{ax}.npy" for o in (1, 2) for ax in range(3)]
    names += [f"{k}_{ax}.npy" for k in ("u", "v") for ax in range(3)]
    names += [f"phi_{i}{j}.npy" for i, j in _DIAG + _OFFDIAG]
    return tuple(names)


def cleanup_stage(workdir, missing_ok=True):
    """Remove the staging intermediates under `workdir/stage`, then the directory.

    Deletes only the files named by `staged_names()`, then `os.rmdir` (never a recursive
    delete), so any foreign file leaves the directory standing and is reported. Returns a
    report dict (files removed, bytes, whether the directory went) for the manifest.
    """
    stage = os.path.join(workdir, STAGE_DIR)
    report = dict(dir=stage, removed=[], bytes=0, dir_removed=False, existed=os.path.isdir(stage))
    if not report["existed"]:
        if missing_ok:
            return report
        raise FileNotFoundError(f"no staging directory at {stage}")

    for name in staged_names():
        path = os.path.join(stage, name)
        if not os.path.exists(path):
            continue
        report["bytes"] += os.path.getsize(path)
        os.remove(path)
        report["removed"].append(name)

    try:
        os.rmdir(stage)
        report["dir_removed"] = True
    except OSError as e:
        report["dir_error"] = str(e)
        report["left_behind"] = sorted(os.listdir(stage))
    return report


def _stage_spec_to(workdir, name, spec, n, slab):
    """Inverse-transform a spectrum (consuming it) into a StagedArray (explicit IO, not memmap)."""
    rdt = np.float64 if spec.dtype == np.complex128 else np.float32
    sa = ooc_fft.StagedArray.create(os.path.join(workdir, name), rdt, (n, n, n))
    for lo, s in ooc_fft.inverse_to_slabs(spec, n, slab=slab):
        sa.write_slab(lo, s)
    return sa


def generate_t9_slabs(
    workdir,
    key,
    n_part,
    box_size,
    cosmo,
    a_init,
    bricks_per_side,
    bucket_cells=2,
    f_NL=0.0,
    order=2,
    fdtype=np.float64,
    slab=32,
    backend="eh98",
    table=None,
    window=1,
    provenance=None,
    keep_stage=False,
    growth2="lcdm",
):
    """Generate T9-encoded initial-condition slabs on disk.

    Writes `t9_slab_{bx:04d}.npz` per destination brick slab, then MANIFEST last (its absence
    marks an incomplete generation, which the loader refuses). Returns the manifest dict;
    `provenance` is stored verbatim in it. order=2 only. On success the staging
    intermediates are removed (unless `keep_stage`) and the cleanup report is recorded under
    `stage_cleanup`; an interrupted run keeps them. `window` is the emission window depth in
    brick slabs; generation refuses if the max displacement reaches it.
    """
    t9 = T9Layout(box_size, n_part, bucket_cells)
    n, box = int(n_part), float(box_size)
    nb = int(bricks_per_side)
    if t9.n_buckets_side % nb:
        raise ValueError(f"bricks_per_side {nb} must divide the bucket grid {t9.n_buckets_side}")
    if n % nb:
        raise ValueError(f"bricks_per_side {nb} must divide n_part {n}")
    if order != 2:
        raise ValueError(f"the streamed generator is order=2 only, got {order}")
    dt = np.dtype(fdtype)
    os.makedirs(workdir, exist_ok=True)
    stage = os.path.join(workdir, "stage")
    os.makedirs(stage, exist_ok=True)

    # linear density: ic.linear_density's exact op sequence
    tab = ic_k_table(cosmo, n, box, backend=backend, table=table)
    spec = ooc_fft.forward_from_slabs(
        lambda lo, hi: ic.white_slab(key, lo, hi, n, dt), n, slab=slab
    )
    ooc_fft.mul_radial_inplace(spec, n, box, ic._colour_fn(tab, n, box), dc_value=0.0, slab=slab)
    ooc_fft.mul_radial_inplace(
        spec, n, box, ic._poisson_fn(cosmo, tab, inverse=True), dc_value=1.0, slab=slab
    )
    phi_sa = ooc_fft.StagedArray.create(os.path.join(stage, "phi.npy"), dt, (n, n, n))
    tot = 0.0
    for lo, s in ooc_fft.inverse_to_slabs(spec, n, slab=slab):
        phi_sa.write_slab(lo, s)
        tot = ic.sq_sum_by_plane(s, tot)
    del spec
    mean_phi2 = tot / n**3

    def _png_slab(lo, hi):
        p = phi_sa.read_slab(lo, hi)
        return p + np.asarray(f_NL, dtype=p.dtype) * (p * p - np.asarray(mean_phi2, p.dtype))

    spec = ooc_fft.forward_from_slabs(_png_slab, n, slab=slab)
    ooc_fft.mul_radial_inplace(spec, n, box, ic._poisson_fn(cosmo, tab), dc_value=1.0, slab=slab)
    delta_sa = _stage_spec_to(stage, "delta.npy", spec, n, slab)
    del spec

    # 2LPT: lpt.lpt_ics's exact op sequence
    spec = ooc_fft.forward_from_slabs(delta_sa.read_slab, n, slab=slab)
    for ax in range(3):
        _stage_spec_to(stage, f"psi1_{ax}.npy",
                       ooc_fft.grad_invk2_spec(spec, ax, n, box, slab=slab), n, slab)
    d2 = lpt2_source_from_spec(spec, n, box, resident="low", workdir=stage, slab=slab)
    del spec
    spec2 = ooc_fft.rfftn_ooc(d2)
    del d2
    for ax in range(3):
        _stage_spec_to(stage, f"psi2_{ax}.npy",
                       ooc_fft.grad_invk2_spec(spec2, ax, n, box, slab=slab), n, slab)
    del spec2

    D1 = growth_factor_a(a_init, cosmo)
    f1 = growth_rate_a(a_init, cosmo)
    D2 = growth_factor_2(a_init, cosmo, growth2)
    f2 = growth_rate_2(a_init, cosmo, growth2)
    v_coef2 = -(D2 * f2) / (D1 * f1)
    vmax = 0.0
    umax = 0.0
    for ax in range(3):
        p1 = ooc_fft.StagedArray.open(os.path.join(stage, f"psi1_{ax}.npy"), dt, (n, n, n))
        p2 = ooc_fft.StagedArray.open(os.path.join(stage, f"psi2_{ax}.npy"), dt, (n, n, n))
        u_sa = ooc_fft.StagedArray.create(os.path.join(stage, f"u_{ax}.npy"), dt, (n, n, n))
        v_sa = ooc_fft.StagedArray.create(os.path.join(stage, f"v_{ax}.npy"), dt, (n, n, n))
        for lo in range(0, n, slab):
            hi = min(lo + slab, n)
            a1 = p1.read_slab(lo, hi)
            a2 = p2.read_slab(lo, hi)
            # lpt_ics's combine, same op order
            u = dt.type(D1) * a1
            u -= dt.type(D2) * a2
            v = a1 + dt.type(v_coef2) * a2
            u_sa.write_slab(lo, u)
            v_sa.write_slab(lo, v)
            vmax = max(vmax, float(np.max(np.abs(np.asarray(v, np.float64)))))
            umax = max(umax, float(np.max(np.abs(u))))

    scale = vmax / INT16_MAX
    if scale <= 0.0:
        scale = 1.0

    brick_depth = box / nb
    if umax >= window * brick_depth:
        raise ValueError(
            f"max displacement {umax:.3f} reaches the sliding window's depth "
            f"({window} brick slab(s) = {window * brick_depth:.3f} Mpc/h); a particle "
            "could leave the window and the streamed build would misplace it. "
            "Raise `window` (and re-derive the staging cost) rather than widening silently."
        )

    u_ro = [ooc_fft.StagedArray.open(os.path.join(stage, f"u_{ax}.npy"), dt, (n, n, n))
            for ax in range(3)]
    v_ro = [ooc_fft.StagedArray.open(os.path.join(stage, f"v_{ax}.npy"), dt, (n, n, n))
            for ax in range(3)]
    written, n_total = _emit_t9_slabs(workdir, u_ro, v_ro, t9, n, box, nb, dt, slab, window)

    manifest = dict(
        schema=SCHEMA,
        files=written,
        n_particles=n_total,
        vel_scale=float(scale),
        max_displacement=float(umax),
        window=window,
        box_size=box,
        n_part=n,
        bucket_cells=int(bucket_cells),
        bricks_per_side=nb,
        a_init=float(a_init),
        order=order,
        growth2=growth2,
        f_NL=float(f_NL),
        fdtype=dt.name,
        slab=int(slab),
        ic_stream=ic.IC_STREAM,
        backend=backend,
        table_n_points=int(len(tab.k)),
        mean_phi2=float(mean_phi2),
        provenance=provenance or {},
    )
    return _write_manifest(workdir, manifest, keep_stage)


def _write_manifest(workdir, manifest, keep_stage):
    """Clean the stage (unless kept), then write the manifest carrying the cleanup report."""
    if keep_stage:
        manifest["stage_cleanup"] = dict(kept=True, reason="keep_stage=True")
    else:
        try:
            manifest["stage_cleanup"] = cleanup_stage(workdir)
        except OSError as e:
            # a cleanup failure is recorded, never allowed to cost the manifest
            manifest["stage_cleanup"] = dict(error=str(e), removed=[], bytes=0)
    with open(os.path.join(workdir, MANIFEST), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def _sort_slab_rows(keyf, lo_bucket):
    """Stable brick-major order of one destination slab's rows, keyed relative to `lo_bucket`.

    The global bucket ordinal can exceed `_stable_sort_index`'s 2^32 range; a slab's own span
    (nb^2 x per3) does not, and subtracting a constant preserves the order.
    """
    return _stable_sort_index(np.asarray(keyf, dtype=np.int64) - int(lo_bucket))


def _write_t9_slab(workdir, d, occ, off, w, scale_d, lo_bucket, lo_brick):
    """Write destination slab `d` as `t9_slab_{d:04d}.npz` (schema `SCHEMA`, crc32 per array);
    returns the file name."""
    meta = dict(
        schema=SCHEMA,
        bx=int(d),
        n_rows=int(len(off)),
        bucket_lo=int(lo_bucket),
        brick_lo=int(lo_brick),
        crc32=dict(
            occupancy=zlib.crc32(occ.tobytes()),
            off=zlib.crc32(off.tobytes()),
            w=zlib.crc32(w.tobytes()),
            scale=zlib.crc32(scale_d.tobytes()),
        ),
    )
    path = os.path.join(workdir, f"t9_slab_{int(d):04d}.npz")
    np.savez(path, meta=json.dumps(meta), occupancy=occ, off=off, w=w, scale=scale_d)
    return os.path.basename(path)


def _emit_t9_slabs(workdir, u_ro, v_ro, t9, n, box, nb, dt, slab, window):
    """Emission: Lagrangian x-slabs through a sliding window into per-destination T9 files.

    `u_ro` / `v_ro` are three readers each (`read_slab(lo, hi)` -> (hi-lo, N, N) displacement /
    velocity planes, any storage). Returns `(written, n_total)`. A destination slab is
    finalized once all source slabs within `window` of it have been emitted.
    """
    per = t9.n_buckets_side // nb
    per3 = per**3
    planes = n // nb  # particle planes per brick slab
    coords = np.arange(n, dtype=dt) * dt.type(box / n)  # lagrangian_grid's exact values

    staged = {d: {} for d in range(nb)}  # dest slab -> {src slab: contribution}
    done_src = np.zeros(nb, dtype=bool)
    written = []
    n_total = 0

    def _sources(d):
        return sorted({(d + o) % nb for o in range(-window, window + 1)})

    def _finalize(d):
        # ascending src then arrival order = ascending Lagrangian index, so the stable sort
        # reproduces SlotState.build's order within every brick
        parts = [p for s in sorted(staged[d]) for p in staged[d][s]]
        keyf = np.concatenate([p[0] for p in parts]) if parts else np.empty(0, np.int64)
        off = (np.concatenate([p[1] for p in parts]) if parts
               else np.empty((0, 3), np.uint8))
        v = (np.concatenate([p[2] for p in parts]) if parts
             else np.empty((0, 3), np.float64))
        lo_bucket = d * nb * nb * per3
        order = _sort_slab_rows(keyf, lo_bucket)
        keyf, off, v = keyf[order], off[order], v[order]
        occ = np.bincount(keyf - lo_bucket, minlength=nb * nb * per3).astype(np.int64)
        # one velocity scale per brick; a brick lies in exactly one x-slab and rows are
        # already grouped by brick, so this is SlotState.build's reduction and encoder
        lo_brick = d * nb * nb
        bcounts = occ.reshape(nb * nb, per3).sum(axis=1)
        scale_d = _scales_from_sorted(np.abs(v).max(axis=1), bcounts)
        w = _encode_at(v, scale_d[keyf // per3 - lo_brick])
        staged[d].clear()
        written.append(_write_t9_slab(workdir, d, occ, off, w, scale_d, lo_bucket, lo_brick))
        return len(keyf)

    finalized = np.zeros(nb, dtype=bool)
    chunk = max(1, min(slab, planes))  # emission transients are O(chunk), not O(brick slab)
    for src in range(nb):
        for c0 in range(src * planes, (src + 1) * planes, chunk):
            c1 = min(c0 + chunk, (src + 1) * planes)
            rows = (c1 - c0) * n * n
            x_ch = np.empty((rows, 3), dtype=np.float64)
            v_ch = np.empty((rows, 3), dtype=np.float64)
            for ax in range(3):
                u = u_ro[ax].read_slab(c0, c1)
                q = (
                    coords[c0:c1].reshape(-1, 1, 1),
                    coords.reshape(1, -1, 1),
                    coords.reshape(1, 1, -1),
                )[ax]
                x_ch[:, ax] = np.mod(q + u, dt.type(box)).reshape(-1)
                v_ch[:, ax] = v_ro[ax].read_slab(c0, c1).astype(np.float64).reshape(-1)
            off, bijk = encode_positions_host(x_ch, t9)
            keyf = _bucket_flat_brick_major(bijk, t9, nb)
            del x_ch, bijk
            bx_dest = keyf // (nb * nb * per3)
            ring = (bx_dest - src) % nb
            bad = ~((ring <= window) | (ring >= nb - window))
            if bad.any():
                raise RuntimeError(
                    f"{int(bad.sum())} particles from source slab {src} routed outside the "
                    f"+-{window} window (the displacement bound above should have caught this)"
                )
            for d in np.unique(bx_dest):
                m = bx_dest == d
                # stage float velocities: a brick's scale is known only at `_finalize`, and
                # encoding once there matches SlotState.build's single rounding
                staged[int(d)].setdefault(src, []).append((keyf[m], off[m], v_ch[m]))
            n_total += len(keyf)
        done_src[src] = True
        for d in range(nb):
            if not finalized[d] and all(done_src[s] for s in _sources(d)):
                _finalize(d)
                finalized[d] = True

    assert finalized.all()
    if n_total != n**3:
        raise RuntimeError(f"emitted {n_total} particles, expected {n**3}")
    return written, n_total


class _HostField:
    """`read_slab` over a host (N, N, N) array."""

    def __init__(self, arr):
        self.arr = arr

    def read_slab(self, lo, hi):
        return self.arr[lo:hi]


class _CardField:
    """`read_slab` over card shards tiling x-planes [0, N) (`{lo, hi, delta}` dicts)."""

    def __init__(self, shards):
        self.shards = sorted(shards, key=lambda s: int(s["lo"]))

    def read_slab(self, lo, hi):
        parts = []
        for s in self.shards:
            s_lo, s_hi = int(s["lo"]), int(s["hi"])
            a, b = max(lo, s_lo), min(hi, s_hi)
            if a < b:
                parts.append(np.asarray(s["delta"][a - s_lo:b - s_lo]))
        return parts[0] if len(parts) == 1 else np.concatenate(parts)


def generate_t9_slabs_device(
    workdir,
    key,
    n_part,
    box_size,
    cosmo,
    a_init,
    bricks_per_side,
    bucket_cells=2,
    f_NL=0.0,
    order=2,
    fdtype=np.float32,
    slab=32,
    backend="eh98",
    table=None,
    window=1,
    provenance=None,
    keep_stage=False,
    growth2="lcdm",
    devices=None,
    pencil_batch=1,
    noise="device",
    log=None,
    emission="cards",
    comm=None,
    batch_planes=ooc_fft.DEFAULT_BATCH_PLANES,
):
    """`generate_t9_slabs` with the IC stage on the devices. Same arguments and output format.

    Extra arguments: `devices` (default all jax devices), `pencil_batch`, `noise` ("device":
    stream `ic.IC_STREAM_DEVICE`; "host": the CPU stream `ic.IC_STREAM`), `log` (callable
    taking one line, called per stage), and `emission` ("cards": `device.emit`; "host":
    `_emit_t9_slabs`; bitwise-identical slabs on the CPU backend).

    Across ranks (`comm`; every rank calls this with its own `devices` and the same shared
    `workdir`): rank r owns destination brick slabs `partition_units(nb, n_ranks, 1)[r]`,
    their particle x-planes, and y-pencils `partition_units(n, n_ranks, pencil_batch)[r]`
    of every spectrum. Plane <-> pencil exchanges go in batches of `batch_planes`
    (`ooc_fft.forward_planes_to_pencils` / `inverse_pencils_to_planes`); V is staged in
    shared files, so a rank reads its neighbours' halo planes from disk; U_x is held on
    the cards and U_y / U_z on the host with `window` brick slabs of halo. Rank 0 writes
    the manifest once every rank has emitted. The files are byte for byte the one-rank
    files at any rank and card count (the manifest adds `n_ranks`); one rank runs the
    same code with no exchange. Across ranks only `noise="device"` and
    `emission="cards"` are supported.
    Differences from the host path: kernels are applied on the device inside the axis-0 pass;
    the phi round trip is skipped at f_NL = 0; the 2LPT source is
    1/2 (delta^2 - sum phi_ii^2) - sum_{i<j} phi_ij^2 (using sum phi_ii = -delta); U and V are
    inverted from k-space combinations; V is staged to disk, U_x kept on devices, U_y/U_z on
    host (peak ~3 fields). Not bitwise the host generator. Requires jax_enable_x64 (kernels
    are built in float64; fields stay `fdtype`).
    """
    import jax

    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "generate_t9_slabs_device needs jax_enable_x64: its k-space kernels are built "
            "in float64 on the card (bitwise the host kernels); in float32 they move 5-65 "
            "eps x rms. Enable x64 in the caller (the fields stay float32).")
    if noise not in ("device", "host"):
        raise ValueError(f"noise must be 'device' or 'host', got {noise!r}")
    if emission not in ("cards", "host"):
        raise ValueError(f"emission must be 'cards' or 'host', got {emission!r}")
    if comm is None:
        from .comm import SerialComm

        comm = SerialComm()
    n_ranks, rank = int(comm.size), int(comm.rank)
    if n_ranks > 1 and (noise != "device" or emission != "cards"):
        raise ValueError(
            f"across ranks the generator runs noise='device' and emission='cards' only (got "
            f"noise={noise!r}, emission={emission!r}); the host lanes are single-node parity "
            "references")
    t9 = T9Layout(box_size, n_part, bucket_cells)
    n, box = int(n_part), float(box_size)
    nb = int(bricks_per_side)
    if t9.n_buckets_side % nb:
        raise ValueError(f"bricks_per_side {nb} must divide the bucket grid {t9.n_buckets_side}")
    if n % nb:
        raise ValueError(f"bricks_per_side {nb} must divide n_part {n}")
    if order != 2:
        raise ValueError(f"the streamed generator is order=2 only, got {order}")
    dt = np.dtype(fdtype)
    ic._require_stream_config(dt)
    devs = list(jax.devices()) if devices is None else list(devices)
    if n % len(devs):
        raise ValueError(f"n_part {n} must be a multiple of the card count {len(devs)}")
    # rounded up to a multiple of the device count so every slab splits across all devices
    slab = -(-int(slab) // len(devs)) * len(devs)
    os.makedirs(workdir, exist_ok=True)
    stage = os.path.join(workdir, "stage")
    os.makedirs(stage, exist_ok=True)
    K = ooc_fft.KSpaceKernel
    kw = dict(devices=devs, pencil_batch=pencil_batch)
    timings = {}

    def _lap(name, t0):
        timings[name] = time.perf_counter() - t0
        if log is not None:
            log(f"  ic stage {name}: {timings[name]:.1f} s")

    _check_shared_workdir(workdir, comm)
    import jax.numpy as jnp

    p = n // nb
    slab_parts = ooc_fft.partition_units(nb, n_ranks, 1)
    x_parts = [(a * p, z * p) for a, z in slab_parts]
    y_parts = ooc_fft.partition_units(n, n_ranks, pencil_batch)
    x_lo, x_hi = x_parts[rank]
    y_lo, y_hi = y_parts[rank]
    # this rank's cards over its x-planes, by the one-rank rule
    cards = [(x_lo + a, x_lo + z, d) for (a, z), d in
             zip(ooc_fft.partition_units(x_hi - x_lo, len(devs), 1), devs)]
    own = [(a, z - a, d) for a, z, d in cards]
    pkw = dict(devices=devs, pencil_batch=pencil_batch, pencils=(y_lo, y_hi))
    xkw = dict(comm=comm, batch_planes=batch_planes)

    def put(x, dev):
        x = np.asarray(x)
        return jax.device_put(x, dev)

    def irfft2_plane(sp):
        return jnp.fft.irfft2(sp, s=(n, n), axes=(-2, -1))

    def axis0_inverse(spec):
        def part(a, z, dev):
            ooc_fft.fft_axis0_device_inplace(spec[:, a:z, :], inverse=True,
                                             pencil_batch=pencil_batch, device=dev)
        ooc_fft._run_parts(part, ooc_fft.partition_units(y_hi - y_lo, len(devs), pencil_batch),
                           devs)

    tab = ic_k_table(cosmo, n, box, backend=backend, table=table)
    colour = K.colour(tab, n, box)

    # white noise -> delta's spectrum (this rank's pencils)
    t0 = time.perf_counter()
    first = colour if f_NL == 0.0 else colour * K.poisson(cosmo, tab, n, box, inverse=True)
    if noise == "device":
        draw = ooc_fft.noise_plane_program(n, dt)
        key_on = {}

        def noise_plane(g, dev):
            if dev not in key_on:
                key_on[dev] = put(key, dev)
            return draw(key_on[dev], put(np.uint32(g), dev))

        spec = ooc_fft.forward_planes_to_pencils(noise_plane, n, cards, x_parts, y_parts,
                                                 dtype=dt, kernel=first, box_size=box,
                                                 pencil_batch=pencil_batch, **xkw)
    else:
        spec = ooc_fft.forward_from_slabs_device(
            lambda lo, hi: ic.white_slab(key, lo, hi, n, dt), n, slab=slab, kernel=first,
            box_size=box, **kw)
    mean_phi2 = None
    if f_NL != 0.0:
        axis0_inverse(spec)
        phi = np.empty((x_hi - x_lo, n, n), dtype=dt)
        sums = np.zeros(x_hi - x_lo)

        def to_phi(k, i, g, sp):
            pl = np.asarray(irfft2_plane(sp))[0]
            phi[g - x_lo] = pl
            sums[g - x_lo] = ic.sq_sum_by_plane(pl[None], 0.0)

        ooc_fft.inverse_pencils_to_planes(spec, n, own, y_parts, to_phi, **xkw)
        del spec
        # `ic.sq_sum_by_plane`'s fold over every plane in global order
        tot = 0.0
        for part_sums in comm.allgather(sums):
            for v in part_sums:
                tot += float(v)
        mean_phi2 = tot / n**3

        def png_plane(g, dev):
            q = phi[g - x_lo:g - x_lo + 1]
            q = q + np.asarray(f_NL, dtype=q.dtype) * (q * q - np.asarray(mean_phi2, q.dtype))
            return jnp.fft.rfft2(ooc_fft._to_device(q, dev), axes=(-2, -1))

        spec = ooc_fft.forward_planes_to_pencils(
            png_plane, n, cards, x_parts, y_parts, dtype=dt,
            kernel=K.poisson(cosmo, tab, n, box), box_size=box, pencil_batch=pencil_batch,
            **xkw)
        del phi
    _lap("delta", t0)

    # 2LPT source, accumulated on this rank's cards
    t0 = time.perf_counter()
    acc = ooc_fft.zeros_card_shards(n, devs, dt, planes=(x_lo, x_hi))
    add = ooc_fft.acc_sq_program(n, dt)
    work = None
    terms = ([(None, 0.5)] + [(K.deriv2(i, j), -0.5) for i, j in _DIAG]
             + [(K.deriv2(i, j), -1.0) for i, j in _OFFDIAG])
    for kern, weight in terms:
        work = ooc_fft.kspace_pass_device([(1.0, spec)], n, box, kernel=kern, out=work,
                                          inverse=True, **pkw)
        w_on = [put(np.asarray(weight, dtype=dt), d) for d in devs]

        def accumulate(k, i, g, sp, w_on=w_on):
            acc[k]["delta"] = add(acc[k]["delta"], put(np.int64(i), devs[k]), sp, w_on[k])

        ooc_fft.inverse_pencils_to_planes(work, n, own, y_parts, accumulate, **xkw)
    _lap("source", t0)

    t0 = time.perf_counter()

    def source_plane(g, dev):
        sh = next(a for a in acc if int(a["lo"]) <= g < int(a["hi"]))
        i = g - int(sh["lo"])
        return jnp.fft.rfft2(sh["delta"][i:i + 1], axes=(-2, -1))

    spec2 = ooc_fft.forward_planes_to_pencils(source_plane, n, cards, x_parts, y_parts,
                                              dtype=dt, pencil_batch=pencil_batch, **xkw)
    acc = None  # released before the velocities
    _lap("source_forward", t0)

    D1 = growth_factor_a(a_init, cosmo)
    f1 = growth_rate_a(a_init, cosmo)
    D2 = growth_factor_2(a_init, cosmo, growth2)
    f2 = growth_rate_2(a_init, cosmo, growth2)
    v_coef2 = -(D2 * f2) / (D1 * f1)

    # V, staged to shared files: rank 0 creates them, every rank writes its own planes
    t0 = time.perf_counter()
    v_paths = [os.path.join(stage, f"v_{ax}.npy") for ax in range(3)]
    if rank == 0:
        for path in v_paths:
            ooc_fft.StagedArray.create(path, dt, (n, n, n))
    comm.barrier()
    v_sa = [ooc_fft.StagedArray.open(path, dt, (n, n, n)) for path in v_paths]
    vmax_card = [0.0] * len(devs)
    for ax in range(3):
        work = ooc_fft.kspace_pass_device([(1.0, spec), (v_coef2, spec2)], n, box,
                                          kernel=K.grad_invk2(ax), out=work, inverse=True,
                                          **pkw)

        def to_v(k, i, g, sp, sa=v_sa[ax]):
            pl = np.asarray(irfft2_plane(sp))
            sa.write_slab(g, pl)
            vmax_card[k] = max(vmax_card[k], float(np.max(np.abs(np.asarray(pl, np.float64)))))

        ooc_fft.inverse_pencils_to_planes(work, n, own, y_parts, to_v, **xkw)
    vmax = comm.allreduce(max(vmax_card), "max")
    _lap("velocities", t0)

    # U = D1 psi1 - D2 psi2: x on the devices, y and z on the host (with halo)
    t0 = time.perf_counter()
    ooc_fft.kspace_pass_device([(D1, spec), (-D2, spec2)], n, box, out=spec, transform=False,
                               **pkw)
    del spec2
    work = ooc_fft.kspace_pass_device([(1.0, spec)], n, box, kernel=K.grad_invk2(0), out=work,
                                      inverse=True, **pkw)
    if emission == "cards":
        # each card's destination slabs plus `window` brick slabs of halo either side
        from .device import emit as demit

        ux = demit.card_slab_ranges(n, nb, devs, window, slabs=slab_parts[rank])
        set_plane = ooc_fft._card_program(("plane_set",), lambda: jax.jit(
            lambda m, i, q: m.at[i].set(q), donate_argnums=0))
        for r in ux:
            r["delta"] = jax.jit(lambda z, shape=(int(r["nx"]), n, n): jnp.broadcast_to(z, shape))(
                put(np.zeros((), dtype=dt), r["device"]))

        def to_ux(k, i, g, sp):
            ux[k]["delta"] = set_plane(ux[k]["delta"], put(np.int64(i), ux[k]["device"]),
                                       irfft2_plane(sp)[0])

        ooc_fft.inverse_pencils_to_planes(work, n, [(r["x0"], r["nx"], r["device"])
                                                    for r in ux], y_parts, to_ux, **xkw)
    else:
        ranges = ooc_fft.partition_units(n, len(devs), 1)
        arrays = ooc_fft.inverse_to_card_shards(
            work, n, [(lo, hi - lo, d) for (lo, hi), d in zip(ranges, devs)],
            pencil_batch=pencil_batch, pass2=False)
        ux = [dict(lo=lo, hi=hi, device=d, delta=a)
              for ((lo, hi), d), a in zip(zip(ranges, devs), arrays)]
        del arrays
    umax = max(float(np.asarray(jax.numpy.max(jax.numpy.abs(s["delta"])))) for s in ux)

    # U_y, U_z: this rank's planes plus `window` brick slabs either side (all planes when
    # that covers the box), read by global plane
    h = window * p
    w_lo, w_n = (0, n) if x_hi - x_lo + 2 * h >= n else ((x_lo - h) % n, x_hi - x_lo + 2 * h)
    w_cards = [(w_lo + a, z - a, d) for (a, z), d in
               zip(ooc_fft.partition_units(w_n, len(devs), 1), devs)]
    umax_card = [0.0] * len(devs)

    def host_window(field, buf=None):
        def to_host(k, i, g, sp):
            pl = np.asarray(irfft2_plane(sp))[0]
            field.arr[(g - field.lo) % n] = pl
            if x_lo <= g < x_hi:
                umax_card[k] = max(umax_card[k], float(np.max(np.abs(pl))))
        return to_host

    uy = _PlaneWindow(np.empty((w_n, n, n), dtype=dt), w_lo, n)
    work = ooc_fft.kspace_pass_device([(1.0, spec)], n, box, kernel=K.grad_invk2(1), out=work,
                                      inverse=True, **pkw)
    ooc_fft.inverse_pencils_to_planes(work, n, w_cards, y_parts, host_window(uy), **xkw)

    # U_z reuses `work`'s bytes when they hold it (n^2 (n/2+1) x 2w >= n^3 x w at one rank):
    # a device array uploaded from a slice of `work` can keep the buffer alive, so a fresh
    # allocation could hold a 4th field
    need = w_n * n * n * dt.itemsize
    flat = work.reshape(-1).view(np.uint8)
    uz_arr = (flat[:need].view(dt).reshape(w_n, n, n) if flat.size >= need
              else np.empty((w_n, n, n), dtype=dt))
    uz = _PlaneWindow(uz_arr, w_lo, n)
    del work, flat
    ooc_fft.kspace_pass_device([(1.0, spec)], n, box, kernel=K.grad_invk2(2), out=spec,
                               inverse=True, **pkw)
    ooc_fft.inverse_pencils_to_planes(spec, n, w_cards, y_parts, host_window(uz), **xkw)
    del spec
    umax = comm.allreduce(max([umax] + umax_card), "max")
    _lap("displacements", t0)

    scale = vmax / INT16_MAX
    if scale <= 0.0:
        scale = 1.0
    brick_depth = box / nb
    if umax >= window * brick_depth:
        raise ValueError(
            f"max displacement {umax:.3f} reaches the sliding window's depth "
            f"({window} brick slab(s) = {window * brick_depth:.3f} Mpc/h); a particle "
            "could leave the window and the streamed build would misplace it. "
            "Raise `window` (and re-derive the staging cost) rather than widening silently."
        )

    # every rank's V planes are on disk before any rank reads its halo
    comm.barrier()
    t0 = time.perf_counter()
    v_ro = [ooc_fft.StagedArray.open(path, dt, (n, n, n)) for path in v_paths]
    emission_s = {}
    if emission == "cards":
        written, n_local = demit.emit_t9_slabs_cards(
            workdir, ux, uy, uz, v_ro, t9, n, box, nb, dt, window, timings=emission_s,
            complete=n_ranks == 1)
    else:
        written, n_local = _emit_t9_slabs(
            workdir, [_CardField(ux), uy, uz], v_ro, t9, n, box, nb, dt, slab, window)
    ux = uy = uz = None
    n_total = int(comm.allreduce(int(n_local)))
    # each rank's files in its emitter's order, ranks in order (one rank: the emitter's list)
    written = [f for part in comm.allgather(list(written)) for f in part]
    if len(written) != nb or n_total != n**3:
        raise RuntimeError(f"the ranks wrote {len(written)} of {nb} slabs holding {n_total} of "
                           f"{n**3} particles")
    _lap("emission", t0)

    manifest = None
    # no rank still reads the staged V when rank 0 removes it
    comm.barrier()
    if rank == 0:
        manifest = dict(
            schema=SCHEMA,
            files=written,
            n_particles=n_total,
            vel_scale=float(scale),
            max_displacement=float(umax),
            window=window,
            box_size=box,
            n_part=n,
            bucket_cells=int(bucket_cells),
            bricks_per_side=nb,
            a_init=float(a_init),
            order=order,
            growth2=growth2,
            f_NL=float(f_NL),
            fdtype=dt.name,
            slab=int(slab),
            ic_stream=ic.IC_STREAM_DEVICE if noise == "device" else ic.IC_STREAM,
            backend=backend,
            table_n_points=int(len(tab.k)),
            mean_phi2=None if mean_phi2 is None else float(mean_phi2),
            generator="device",
            emission=emission,
            emission_s=emission_s,
            n_devices=len(devs),
            pencil_batch=int(pencil_batch),
            stage_s=timings,
            provenance=provenance or {},
        )
        if n_ranks > 1:
            manifest["n_ranks"] = n_ranks
        manifest = _write_manifest(workdir, manifest, keep_stage)
    return comm.bcast(manifest)


class _PlaneWindow:
    """Host planes `lo .. lo + len(arr) - 1` (mod n) of a field, sliced by global plane
    (`[g0:g1]`, a run inside the window), as the emission reads U_y and U_z."""

    def __init__(self, arr, lo, n):
        self.arr, self.lo, self.n = arr, int(lo), int(n)

    def __getitem__(self, sl):
        i0 = (int(sl.start) - self.lo) % self.n
        i1 = i0 + int(sl.stop) - int(sl.start)
        if i1 > len(self.arr):
            raise IndexError(f"planes [{sl.start}, {sl.stop}) are outside this window of "
                             f"{len(self.arr)} planes from {self.lo}")
        return self.arr[i0:i1]

    def read_slab(self, lo, hi):
        return self[lo:hi]


def _check_shared_workdir(workdir, comm):
    """Refuse unless every rank sees the same `workdir` (rank 0 writes a marker the others
    must find): the ranks write one generation together."""
    if int(comm.size) == 1:
        return
    tag = comm.bcast(f".ranks-{os.getpid()}-{time.time_ns()}" if comm.rank == 0 else None)
    path = os.path.join(workdir, tag)
    if comm.rank == 0:
        open(path, "w").close()
    comm.barrier()
    seen = comm.allgather(os.path.exists(path))
    comm.barrier()
    if comm.rank == 0:
        os.remove(path)
    if not all(seen):
        raise RuntimeError(f"ranks {[r for r, ok in enumerate(seen) if not ok]} do not see "
                           f"{workdir}: the ranks must share the IC directory")


def _shared_like(arr, alloc, tag):
    """Copy a small array into shared memory via `alloc` (identity if `alloc` is None)."""
    if alloc is None:
        return arr
    view = alloc.empty(arr.shape, arr.dtype, tag)
    view[...] = arr
    return view


def read_manifest(workdir):
    """Read and validate a T9 slab directory's manifest without loading the state.

    Refuses a missing manifest (it is written last, so absence means incomplete) and an
    unknown schema.
    """
    mpath = os.path.join(workdir, MANIFEST)
    if not os.path.exists(mpath):
        raise FileNotFoundError(
            f"no {MANIFEST} in {workdir}: the manifest is written last, so its absence "
            "marks an incomplete or interrupted generation; refusing to load"
        )
    with open(mpath) as fh:
        man = json.load(fh)
    if man.get("schema") != SCHEMA:
        raise ValueError(f"schema {man.get('schema')!r} != {SCHEMA!r}")
    return man


def slab_files(man, workdir=""):
    """The manifest's slab files indexed by brick x-slab: `[name of slab 0, ..., slab nb-1]`.

    Files are named `t9_slab_{bx:04d}.npz`; the list order is the writer's (the host IC
    emitter finalizes out of slab order). Refuses a list that does not name every slab
    exactly once.
    """
    nb = int(man["bricks_per_side"])
    by_slab = {}
    for f in man["files"]:
        m = re.fullmatch(r"t9_slab_(\d+)\.npz", os.path.basename(f))
        if m is None or int(m.group(1)) in by_slab:
            raise ValueError(f"{workdir}: manifest file {f!r} is not a distinct t9_slab_NNNN.npz")
        by_slab[int(m.group(1))] = f
    if sorted(by_slab) != list(range(nb)):
        raise ValueError(
            f"{workdir}: the manifest names slabs {sorted(by_slab)[:8]}... ({len(by_slab)} "
            f"files) but the state has {nb} brick slabs; the loader needs every slab once"
        )
    return [by_slab[d] for d in range(nb)]


def drop_file_cache(path):
    """Best-effort drop of `path`'s pages from the page cache; True if the call was made.

    For loads whose file set exceeds free memory, where cached pages only compete with the
    state. Never raises."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return True
    except (AttributeError, OSError):
        return False
    finally:
        os.close(fd)


def load_slot_state(
    workdir,
    brick_slack=0.10,
    alloc_margin=0.10,
    arena_frac=0.01,
    index_dtype=DEFAULT_INDEX_DTYPE,
    alloc=None,
    drop_cache=False,
    slabs=None,
):
    """Reassemble a host-resident SlotState from T9 slabs on disk.

    Geometry comes from `state._alloc_geometry`, as in `SlotState.build`: each brick's rows
    land at its `brick_start`, spare rows stay zero, the arena starts empty. Refuses a missing
    manifest, an unknown schema, a file list that is not one file per slab in slab order,
    and any crc mismatch. `alloc` (optional) places arrays in shared memory; `drop_cache`
    drops each slab file's page cache after each read. Reads the files twice (index, then
    payload) so only one slab's payload is live at a time.

    `slabs = (lo, hi)` loads only those brick x-slabs' files into a node-local state (see
    `SlotState`); its arena is `arena_frac` of the local particles.
    """
    man = read_manifest(workdir)
    t9 = T9Layout(man["box_size"], man["n_part"], man["bucket_cells"])
    nb = int(man["bricks_per_side"])
    per3 = (t9.n_buckets_side // nb) ** 3
    n_bricks = nb**3
    nb2 = nb * nb
    files = slab_files(man, workdir)
    s_lo, s_hi = (0, nb) if slabs is None else (int(slabs[0]), int(slabs[1]))
    if not 0 <= s_lo < s_hi <= nb:
        raise ValueError(f"slabs {slabs} are not a non-empty range inside [0, {nb})")
    whole = (s_lo, s_hi) == (0, nb)
    blo, bhi = s_lo * nb2, s_hi * nb2

    n_dropped = [0]

    def _slab(d):
        fname = files[d]
        path = os.path.join(workdir, fname)
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            occ, off, w, sc = z["occupancy"], z["off"], z["w"], z["scale"]
        if int(meta["bx"]) != d:
            raise ValueError(f"{fname} holds slab {meta['bx']}, not the slab {d} its name gives")
        for name, arr in (("occupancy", occ), ("off", off), ("w", w), ("scale", sc)):
            crc = zlib.crc32(arr.tobytes())
            if crc != meta["crc32"][name]:
                raise ValueError(
                    f"{fname}:{name} crc mismatch ({crc} != {meta['crc32'][name]}); "
                    "the slab file is corrupt, refusing to load"
                )
        if drop_cache:
            n_dropped[0] += bool(drop_file_cache(path))
        return off, w, occ, sc

    _trace = os.environ.get("INEXOR_LOAD_TRACE")

    def _say(msg):
        if not _trace:
            return
        try:
            with open("/proc/meminfo") as fh:
                mi = {k: int(v.split()[0]) * 1024 for k, v in
                      (ln.split(":", 1) for ln in fh)}
            extra = (f"  MemAvailable {mi['MemAvailable']/1e9:.1f} GB, "
                     f"Shmem {mi['Shmem']/1e9:.1f} GB")
        except (OSError, KeyError, ValueError):
            extra = ""
        print(f"  [load] {msg}{extra}", flush=True)

    n_files = s_hi - s_lo
    # with `alloc`, the index and payload are allocated directly in shared memory
    _zeros = (lambda shape, dtype, tag: np.zeros(shape, dtype=dtype)) if alloc is None \
        else alloc.zeros
    _empty = (lambda shape, dtype, tag: np.empty(shape, dtype=dtype)) if alloc is None \
        else alloc.empty

    # pass 1: occupancy and per-brick scales only, each slab narrowed to the index dtype as
    # it is read (`_to_index` refuses a count the dtype cannot hold)
    _say(f"pass 1 of 2 over {n_files} slabs (index only)")
    occupancy = _empty(((bhi - blo) * per3,), np.dtype(index_dtype), "occupancy")
    vel_scale = np.ones(n_bricks, dtype=np.float64)
    brick_counts = np.zeros(n_bricks, dtype=np.int64)
    for _i, d in enumerate(range(s_lo, s_hi)):
        _off, _w, occ, sc = _slab(d)
        j = (d - s_lo) * nb2 * per3
        occupancy[j : j + nb2 * per3] = _to_index(occ, index_dtype, "initial")
        brick_counts[d * nb2 : (d + 1) * nb2] = occ.reshape(nb2, per3).sum(axis=1)
        vel_scale[d * nb2 : (d + 1) * nb2] = sc
        del _off, _w, occ
        if _i % 32 == 31:
            _say(f"pass 1: {_i + 1}/{n_files} slabs")

    n = int(brick_counts.sum())
    if whole and n != int(man["n_particles"]):
        raise ValueError(
            f"{workdir}: the slab files hold {n} rows but the manifest records "
            f"{man['n_particles']} particles"
        )
    _, brick_start, n_alloc, n_arena = _alloc_geometry(
        brick_counts, n, brick_slack, alloc_margin, arena_frac
    )
    n_rows = n_alloc + n_arena
    _say(f"allocating off/w for {n_rows:,} rows "
         f"({n_rows * 9 / 1e9:.1f} GB, {'shared' if alloc else 'private'})")
    off_all = _zeros((n_rows, 3), np.uint8, "off")
    w_all = _zeros((n_rows, 3), np.int16, "w")
    _say("allocated; pass 2 of 2 (payload)")

    # pass 2: place the payload, one slab live at a time
    for _i, d in enumerate(range(s_lo, s_hi)):
        off, w, _occ, _sc = _slab(d)
        row = 0
        for b in range(d * nb2, (d + 1) * nb2):
            cnt = int(brick_counts[b])
            off_all[brick_start[b] : brick_start[b] + cnt] = off[row : row + cnt]
            w_all[brick_start[b] : brick_start[b] + cnt] = w[row : row + cnt]
            row += cnt
        if row != len(off):
            raise ValueError(f"slab bx={d}: placed {row} rows of {len(off)}")
        del off, w, _occ
        if _i % 32 == 31:
            _say(f"pass 2: {_i + 1}/{n_files} slabs")

    if drop_cache:
        _say(f"dropped the page cache of {n_dropped[0]} slab reads")
    _say("payload placed; building SlotState")
    st = SlotState(
        t9=t9,
        bricks_per_side=nb,
        brick_start=_shared_like(brick_start, alloc, "brick_start"),
        occupancy=occupancy,
        off=off_all,
        w=w_all,
        vel_scale=_shared_like(vel_scale, alloc, "vel_scale"),
        arena_base=n_alloc,
        arena_bucket=_shared_like(
            np.full(n_arena, -1, dtype=np.int64), alloc, "arena_bucket"),
        n_particles=n,
        ids=None,
        slabs=None if whole else (s_lo, s_hi),
        arena_frac=float(arena_frac),
    )
    st.check()
    return st


def _save_slab(path, meta, occ, off, w, scale_d):
    """Write one T9 slab on `write_t9_slabs`'s writer thread; returns its elapsed seconds.

    The arrays must not be reused by the caller after submission."""
    t0 = time.perf_counter()
    np.savez(path, meta=json.dumps(meta), occupancy=occ, off=off, w=w, scale=scale_d)
    return time.perf_counter() - t0


def write_t9_slabs(st, workdir, provenance=None, drop_ids=False, timings=None,
                   max_slabs=None, comm=None):
    """Write a `SlotState` as T9 slabs: the exact inverse of `load_slot_state`.

    Same schema `generate_t9_slabs` emits. The state is compacted: each brick's live rows plus
    its arena residents are written tight in bucket order, and `load_slot_state` re-derives
    the allocation geometry. `vel_scale` and `w` are copied verbatim, never re-encoded. The
    round trip is a fixed point, `write(load(write(st))) == write(st)` byte for byte, not
    array equality with `st` (row order within a bucket is not preserved and does not affect
    the integer paints). A state with ids refuses unless `drop_ids=True` (ids are not in the
    schema). `timings` (dict) accumulates seconds for `index`, `gather`, `crc32`, `write`
    (main thread blocked on the writer), `write thread` (writer busy time), `arena index`,
    and a `slabs` count. `max_slabs` writes only that many slabs and returns None, with no
    manifest, so the directory cannot load.

    `comm` (`comm.Comm`, default `SerialComm`): every rank calls this with its node-local
    state, and the ranks' slabs must tile the box in rank order. Rank 0 removes the old
    manifest, then (after a barrier) each rank writes its own slabs, and rank 0 writes the
    manifest once every rank has finished. A rank that raises aborts the others, so no
    manifest is written. The files are the same bytes at any rank count. Returns the manifest
    on every rank.
    """
    if st.ids is not None and not drop_ids:
        raise ValueError(
            "this state carries ids and the t9-slabs-2 schema has no room for them; "
            "pass drop_ids=True to write the state without them"
        )
    if comm is None:
        from .comm import SerialComm

        comm = SerialComm()
    nb = st.bricks_per_side
    if max_slabs is not None and comm.size > 1:
        raise ValueError("the max_slabs probe writes one rank's leading slabs; run it on one rank")
    spans = comm.allgather(st.owned_slabs)
    if [lo for lo, _ in spans] != [0] + [hi for _, hi in spans[:-1]] or spans[-1][1] != nb:
        raise ValueError(
            f"the ranks' slabs {spans} do not tile [0, {nb}) in rank order; every slab must "
            "be written by exactly one rank"
        )
    os.makedirs(workdir, exist_ok=True)
    # remove any old manifest first, so a torn overwrite refuses to load rather than mixing
    # generations (each slab's crc32 only vouches for its own file); no rank writes a slab
    # until it is gone
    mpath = os.path.join(workdir, MANIFEST)
    if comm.rank == 0 and os.path.exists(mpath):
        os.remove(mpath)
    comm.barrier()
    p3 = st.buckets_per_brick
    nbb = nb * nb                      # bricks per x-slab
    s_lo, s_hi = st.owned_slabs
    written, n_written = [], 0

    clock = time.perf_counter

    def _add(key, t0):
        if timings is not None:
            timings[key] = timings.get(key, 0.0) + clock() - t0

    # occupied arena rows sorted once by bucket, so each slab takes its residents as a slice
    t0 = clock()
    if st.n_arena:
        a_live = np.nonzero(st.arena_bucket >= 0)[0]
        a_order = np.argsort(st.arena_bucket[a_live], kind="stable")
        a_bucket_all = st.arena_bucket[a_live][a_order]
        a_slot_all = st.arena_base + a_live[a_order]
    else:
        a_bucket_all = np.empty(0, dtype=np.int64)
        a_slot_all = np.empty(0, dtype=np.int64)
    _add("arena index", t0)

    # one writer thread, depth one: slab k's write overlaps slab k+1's gather, so at most two
    # slabs' buffers are live
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="t9-writer")
    pending = None

    def _join():
        nonlocal pending
        if pending is not None:
            f, pending = pending, None
            dt = f.result()            # the only place a write error surfaces
            if timings is not None:
                timings["write thread"] = timings.get("write thread", 0.0) + dt

    try:
        for d in range(s_lo, s_hi if max_slabs is None else min(s_hi, s_lo + int(max_slabs))):
            t0 = clock()
            lo_brick = d * nbb
            lo_bucket = lo_brick * p3
            hi_bucket = lo_bucket + nbb * p3

            # the schema stores occupancy as int64
            occ = st._occ(lo_brick, lo_brick + nbb).astype(np.int64)
            counts = occ.reshape(nbb, p3).sum(axis=1)
            starts = st.brick_start[lo_brick : lo_brick + nbb].astype(np.int64)
            base = np.concatenate([[0], np.cumsum(counts)[:-1]])
            n_live = int(counts.sum())

            alo, ahi = np.searchsorted(a_bucket_all, [lo_bucket, hi_bucket])
            a_keys, a_slots = a_bucket_all[alo:ahi], a_slot_all[alo:ahi]
            if len(a_keys):
                # insertion point after all live rows in buckets <= its own
                a_pos = np.cumsum(occ)[a_keys - lo_bucket]
                u, c = np.unique(a_keys - lo_bucket, return_counts=True)
                occ[u] += c
            else:
                a_pos = np.empty(0, dtype=np.int64)
            n_rows = n_live + len(a_slots)
            _add("index", t0)

            t0 = clock()
            # each brick's live rows are a contiguous run from `brick_start`, in bucket order
            off = np.empty((n_live, st.off.shape[1]), dtype=st.off.dtype)
            w = np.empty((n_live, st.w.shape[1]), dtype=st.w.dtype)
            for b in range(nbb):
                m = int(counts[b])
                if m:
                    s0, d0 = int(starts[b]), int(base[b])
                    off[d0 : d0 + m] = st.off[s0 : s0 + m]
                    w[d0 : d0 + m] = st.w[s0 : s0 + m]
            if len(a_slots):
                off = np.insert(off, a_pos, st.off[a_slots], axis=0)
                w = np.insert(w, a_pos, st.w[a_slots], axis=0)
            scale_d = np.asarray(st.vel_scale[lo_brick : lo_brick + nbb], dtype=np.float64)
            _add("gather", t0)

            t0 = clock()
            # crc32 over the buffers directly (no .tobytes() copy)
            meta = dict(
                schema=SCHEMA,
                bx=d,
                n_rows=int(n_rows),
                bucket_lo=int(lo_bucket),
                brick_lo=int(lo_brick),
                crc32=dict(
                    occupancy=zlib.crc32(occ),
                    off=zlib.crc32(off),
                    w=zlib.crc32(w),
                    scale=zlib.crc32(scale_d),
                ),
            )
            _add("crc32", t0)
            t0 = clock()
            path = os.path.join(workdir, f"t9_slab_{d:04d}.npz")
            _join()                    # blocks only if the previous write is still going
            pending = pool.submit(_save_slab, path, meta, occ, off, w, scale_d)
            _add("write", t0)
            if timings is not None:
                timings["slabs"] = timings.get("slabs", 0) + 1
            written.append(os.path.basename(path))
            n_written += n_rows

        t0 = clock()
        _join()
        _add("write", t0)
    finally:
        pool.shutdown(wait=True)

    if max_slabs is not None:
        return None

    # a dropped arena row would be a silently deleted particle
    if n_written != st.n_live:
        raise RuntimeError(f"wrote {n_written} rows for a state holding {st.n_live} particles")
    done = comm.allgather((written, int(n_written)))

    manifest = dict(
        schema=SCHEMA,
        files=[f for files, _ in done for f in files],
        n_particles=sum(n for _, n in done),
        box_size=float(st.t9.box_size),
        n_part=int(st.t9.n_part),          # PER SIDE; `n_particles` is the total
        bucket_cells=int(st.t9.bucket_cells),
        bricks_per_side=int(nb),
        source="write_t9_slabs",
        provenance=provenance or {},
    )
    # written last: the completeness marker
    if comm.rank == 0:
        with open(mpath, "w") as fh:
            json.dump(manifest, fh, indent=1)
    comm.barrier()
    return manifest
