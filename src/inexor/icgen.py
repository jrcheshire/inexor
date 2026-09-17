"""The streamed IC generator and its loader (M-v2-5; D-v2-15 clauses 3 + 5).

`generate_t9_slabs` runs the whole IC stage -- plane-keyed noise, table
colour, the f_NL transform, 2LPT, T9 encode -- without ever materializing a
box-sized (N^3, 3) array: real fields live in disk-staged `.npy` memmaps
(clause 3: disk stages the IC stage ONLY), spectra live host-resident one at
a time, and the emission pass walks brick-aligned Lagrangian x-slabs through
a sliding window into per-destination-slab T9 files. `load_slot_state`
reassembles a host-resident SlotState at evolve start.

BITWISE THE MONOLITHIC BUILD, by construction and by gate: every elementwise
op sequence here is the same one `ic.linear_density` -> `lpt.lpt_ics` ->
`SlotState.build` executes (the canonical combine in lpt_ics, the threaded
`sq_sum_by_plane` reduction, the shared `_alloc_geometry`), every FFT goes
through ooc_fft's one-plane compute unit, and the M-v2-5 exit gate asserts
`load_slot_state(generate_t9_slabs(...)) == SlotState.build(...)` on every
array. Ordering is what makes the slot layout come out identical: destination
slabs concatenate their contributions in ascending SOURCE slab order (which
IS ascending Lagrangian order, including at the periodic seam), and a stable
sort by bucket key then reproduces `build`'s global stable sort within every
brick.

The velocity scale is exact, not estimated: the V staging pass tracks each
slab's max|v|, slabs partition the particles, so the max of per-slab maxes
is the global max -- `reconcile_velocity_scale`'s theorem at build time.
`encode_velocities_host` still asserts the int16 range per slab (D-007): the
theorem says the refusal cannot fire, and the refusal is what proves that.

The displacement bound is MEASURED, then enforced: the U staging pass tracks
max|u|, and the emission refuses if it reaches the sliding window's depth
(default one brick slab) -- an assert, not a hope; the per-slab routing check
would catch a violation anyway, and the two failing together is the design.
"""

import json
import os
import time
import zlib

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

# BUMPED at per-brick velocity scales (M-v2-6). A `-1` slab stores ONE scale for
# the whole state; a `-2` slab stores one per brick. The arrays are otherwise
# identical, so a `-1` file would load without error and decode every velocity
# at the wrong scale -- silently, and by a factor that varies brick to brick.
# Refusing it is the only way that cannot happen.
SCHEMA = "t9-slabs-2"
MANIFEST = "manifest.json"


STAGE_DIR = "stage"


def staged_names():
    """Every intermediate `generate_t9_slabs` writes under `stage/`, derived
    from the producers rather than listed by hand so the two cannot drift.

    Twenty files, each a full (n, n, n) array: about 687 GB at C-gh, where one
    of them is 34.4 GB. Nothing consumes them once the manifest is written --
    they are the streamed generator's working set, spilled to disk precisely so
    the process never holds them.
    """
    from .lpt import _DIAG, _OFFDIAG

    names = ["phi.npy", "delta.npy"]
    names += [f"psi{o}_{ax}.npy" for o in (1, 2) for ax in range(3)]
    names += [f"{k}_{ax}.npy" for k in ("u", "v") for ax in range(3)]
    names += [f"phi_{i}{j}.npy" for i, j in _DIAG + _OFFDIAG]
    return tuple(names)


def cleanup_stage(workdir, missing_ok=True):
    """Remove the staging intermediates under `workdir/stage` and the directory.

    Named files ONLY, from `staged_names()`, then `os.rmdir` -- never a
    recursive delete. The rmdir is the point rather than tidiness: it FAILS if
    anything the generator did not put there is still inside, so a stray file is
    a loud refusal instead of a silent deletion of someone's data. A run that
    stages somewhere shared is exactly where a recursive delete stops being
    recoverable.

    Called last by `generate_t9_slabs`, AFTER the manifest, so an interrupted
    generation keeps its intermediates for diagnosis: the completeness marker
    is what licenses the delete. Returns a report -- files removed, bytes
    reclaimed, and whether the directory went -- for the manifest to carry, so
    a run can say what it cleaned rather than leaving it to be inferred from an
    absence.
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
        # left standing ON PURPOSE, with what is in it, rather than forced
        report["dir_error"] = str(e)
        report["left_behind"] = sorted(os.listdir(stage))
    return report


def _stage_spec_to(workdir, name, spec, n, slab):
    """Inverse-transform a spectrum (consuming it) into a StagedArray.

    Explicit IO, never memmap: dirty mapped pages count in the process's RSS
    and would misreport the very residency the staging exists to avoid (see
    ooc_fft.StagedArray).
    """
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
):
    """Generate T9-encoded initial-condition slabs on disk.

    Writes `t9_slab_{bx:04d}.npz` per destination brick slab plus MANIFEST
    (written LAST -- its absence marks an incomplete generation and the loader
    refuses). Returns the manifest dict. `provenance` (optional dict) is
    stored verbatim in the manifest beside the generator's own fields.

    The twenty staging intermediates under `stage/` are REMOVED on success --
    687 GB per run at C-gh, and nothing reads them once the manifest exists.
    They are deleted after the manifest is written, never before: the
    completeness marker is what licenses the delete, so an interrupted
    generation keeps its working set for diagnosis. `keep_stage=True` keeps
    them regardless. What was removed is recorded in the manifest under
    `stage_cleanup`, because an absence is not evidence of a deletion.
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

    # --- linear density, streamed (ic.linear_density's exact op sequence) ---
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

    # --- LPT, streamed (lpt.lpt_ics's exact op sequence, order=2) ----------
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

    # --- U/V staging + the exact velocity scale and displacement bound -----
    D1 = growth_factor_a(a_init, cosmo)
    f1 = growth_rate_a(a_init, cosmo)
    D2 = growth_factor_2(a_init, cosmo)
    f2 = growth_rate_2(a_init, cosmo)
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
            # lpt_ics's canonical combine, per element
            u = dt.type(D1) * a1
            u -= dt.type(D2) * a2
            v = a1 + dt.type(v_coef2) * a2
            u_sa.write_slab(lo, u)
            v_sa.write_slab(lo, v)
            vmax = max(vmax, float(np.max(np.abs(np.asarray(v, np.float64)))))
            umax = max(umax, float(np.max(np.abs(u))))

    # codec.encode_velocities' scale, from the partition max (exact)
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

    # --- emission: brick-aligned x-slabs through the sliding window --------
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
    """Clean the stage (unless kept), then write the manifest carrying the report."""
    # AFTER the manifest, and the manifest is rewritten to carry the report:
    # cleaning first would delete the working set of a generation that then
    # failed to complete, and reporting nothing would leave "was it cleaned?"
    # answerable only by looking at a directory that may since have been reused.
    if keep_stage:
        manifest["stage_cleanup"] = dict(kept=True, reason="keep_stage=True")
    else:
        try:
            manifest["stage_cleanup"] = cleanup_stage(workdir)
        except OSError as e:
            # Housekeeping must never cost a completed generation its manifest.
            # At C-hero this is a multi-hour product and the intermediates are
            # a disk bill; an unwritable stage directory is the wrong reason to
            # lose the run. Recorded loudly instead.
            manifest["stage_cleanup"] = dict(error=str(e), removed=[], bytes=0)
    with open(os.path.join(workdir, MANIFEST), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def _sort_slab_rows(keyf, lo_bucket):
    """Stable brick-major order of one destination slab's rows, keyed RELATIVE to the slab.

    The global bucket ordinal reaches 2048^3 = 8.6e9 at 4096^3, past
    `_stable_sort_index`'s 2^32 (Vista 997280 died on slab 128 of 256); a slab's
    own span is nb^2 * per3. Subtracting a constant keeps the order.
    """
    return _stable_sort_index(np.asarray(keyf, dtype=np.int64) - int(lo_bucket))


def _write_t9_slab(workdir, d, occ, off, w, scale_d, lo_bucket, lo_brick):
    """Write destination slab `d` as `t9_slab_{d:04d}.npz` (schema `SCHEMA`, crc32 per
    array); returns the file name. The ONE writer both emissions use."""
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

    `u_ro` / `v_ro` are three readers each (`read_slab(lo, hi)` -> (hi-lo, N, N)
    displacement / velocity planes, any storage). Returns `(written, n_total)`.
    Shared by both generators; the op sequence is the one the bitwise gate
    against `SlotState.build` was established on.
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
        # ascending src, then chunk arrival order within a src = ascending
        # Lagrangian index, which is what licenses the stable sort below to
        # reproduce SlotState.build's global stable sort within every brick
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
        # ONE SCALE PER BRICK, over this slab's bricks only -- which is sound
        # because a brick belongs to exactly one x-slab, so no other slab can
        # contribute to it. `keyf` is ascending and buckets are brick-major, so
        # the rows are already grouped by brick and the same reduction
        # `SlotState.build` performs applies unchanged. Encoding through the
        # identical helper is what makes the bitwise gate against `build` hold by
        # construction rather than by two implementations agreeing.
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
                # the FLOAT velocity is staged, not a code. A brick's scale is a
                # max over contributors that arrive from several source slabs, so
                # it is not known until `_finalize`; encoding here and re-encoding
                # there would round twice where `SlotState.build` rounds once, and
                # the bitwise gate against it would fail for that reason alone.
                # Costs 24 B/row instead of 6 over a window of slabs.
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
    devices=None,
    pencil_batch=1,
    noise="device",
    log=None,
    emission="cards",
):
    """`generate_t9_slabs` with the IC stage on the cards (D6). Same arguments and output format.

    Extra arguments: `devices` (default every jax device), `pencil_batch`,
    `noise` ("device": drawn on the cards, stream `ic.IC_STREAM_DEVICE`; "host":
    the CPU stream `ic.IC_STREAM`, for parity against the host generator), `log`
    (a callable taking one line, called as each stage ends), and `emission`
    ("cards": `device.emit.emit_t9_slabs_cards`; "host": `_emit_t9_slabs`, the
    oracle -- the two write bitwise-identical slabs on the CPU backend).

    What differs from the host generator, all for wall and host memory:
    - every kernel is applied on the card inside the axis-0 pass
      (`ooc_fft.kspace_pass_device`), so there is no host kernel pass or copy;
    - at f_NL = 0 the phi round trip is skipped (colour x (1/M) x M = colour), and
      the delta inverse-then-forward round trip is gone at every f_NL;
    - the 2LPT source is 1/2 (delta^2 - sum phi_ii^2) - sum_{i<j} phi_ij^2
      (sum phi_ii = -delta), accumulated on the cards one field at a time;
    - U and V are inverted from k-space combinations (linear, so exact up to
      roundoff): V is staged to disk, U_x lives on the cards, U_y and U_z on the
      host -- at 4096^3 the host peak is ~3 fields (825 GB) and a card holds ~69 GB.
    Emission is the host generator's (`_emit_t9_slabs`).

    NOT bitwise the host generator; parity is gated in eps and code units.
    Requires x64: kernels are built in float64 (bitwise the host kernels);
    float32 kernels were measured 5-65 eps x rms off.
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
    # A pure memory knob, rounded up to a multiple of the card count so every slab
    # (the tail included, since n is one too) splits across all of them.
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

    tab = ic_k_table(cosmo, n, box, backend=backend, table=table)
    colour = K.colour(tab, n, box)

    # --- 1. white noise -> delta's spectrum --------------------------------
    t0 = time.perf_counter()
    first = colour if f_NL == 0.0 else colour * K.poisson(cosmo, tab, n, box, inverse=True)
    if noise == "device":
        spec = ooc_fft.noise_forward_cards(key, n, devs, dt, kernel=first, box_size=box,
                                           pencil_batch=pencil_batch)
    else:
        spec = ooc_fft.forward_from_slabs_device(
            lambda lo, hi: ic.white_slab(key, lo, hi, n, dt), n, slab=slab, kernel=first,
            box_size=box, **kw)
    mean_phi2 = None
    if f_NL != 0.0:
        phi = np.empty((n, n, n), dtype=dt)
        tot = 0.0
        for lo, s in ooc_fft.inverse_to_slabs_device(spec, n, slab=slab, **kw):
            phi[lo:lo + s.shape[0]] = s
            tot = ic.sq_sum_by_plane(s, tot)
        del spec
        mean_phi2 = tot / n**3

        def _png_slab(lo, hi):
            p = phi[lo:hi]
            return p + np.asarray(f_NL, dtype=p.dtype) * (p * p - np.asarray(mean_phi2, p.dtype))

        spec = ooc_fft.forward_from_slabs_device(
            _png_slab, n, slab=slab, kernel=K.poisson(cosmo, tab, n, box), box_size=box, **kw)
        del phi
    _lap("delta", t0)

    # --- 2. the 2LPT source, accumulated on the cards ----------------------
    t0 = time.perf_counter()
    acc = ooc_fft.zeros_card_shards(n, devs, dt)
    work = None
    terms = ([(None, 0.5)] + [(K.deriv2(i, j), -0.5) for i, j in _DIAG]
             + [(K.deriv2(i, j), -1.0) for i, j in _OFFDIAG])
    for kern, weight in terms:
        acc, work = ooc_fft.inverse_accumulate_cards([(1.0, spec)], n, acc, weight, kernel=kern,
                                                     box_size=box, work=work,
                                                     pencil_batch=pencil_batch)
    _lap("source", t0)

    t0 = time.perf_counter()
    spec2 = ooc_fft.forward_from_card_planes(acc, n, pencil_batch=pencil_batch)
    del acc
    _lap("source_forward", t0)

    D1 = growth_factor_a(a_init, cosmo)
    f1 = growth_rate_a(a_init, cosmo)
    D2 = growth_factor_2(a_init, cosmo)
    f2 = growth_rate_2(a_init, cosmo)
    v_coef2 = -(D2 * f2) / (D1 * f1)

    # --- 3. V = grad(delta + c S_2), staged to disk -------------------------
    t0 = time.perf_counter()
    vmax = 0.0
    for ax in range(3):
        ooc_fft.kspace_pass_device([(1.0, spec), (v_coef2, spec2)], n, box,
                                   kernel=K.grad_invk2(ax), out=work, inverse=True, **kw)
        v_sa = ooc_fft.StagedArray.create(os.path.join(stage, f"v_{ax}.npy"), dt, (n, n, n))
        for lo, s in ooc_fft.inverse_to_slabs_device(work, n, slab=slab, pass2=False, **kw):
            v_sa.write_slab(lo, s)
            vmax = max(vmax, float(np.max(np.abs(np.asarray(s, np.float64)))))
    _lap("velocities", t0)

    # --- 4. U = grad(D1 delta - D2 S_2): x on the cards, y and z on the host --
    t0 = time.perf_counter()
    ooc_fft.kspace_pass_device([(D1, spec), (-D2, spec2)], n, box, out=spec, transform=False,
                               **kw)
    del spec2
    ooc_fft.kspace_pass_device([(1.0, spec)], n, box, kernel=K.grad_invk2(0), out=work,
                               inverse=True, **kw)
    if emission == "cards":
        # each card's own destination slabs plus `window` brick slabs of halo either side
        from .device import emit as demit

        ux = demit.card_slab_ranges(n, nb, devs, window)
        arrays = ooc_fft.inverse_to_card_shards(
            work, n, [(r["x0"], r["nx"], r["device"]) for r in ux],
            pencil_batch=pencil_batch, pass2=False)
        for r, a in zip(ux, arrays):
            r["delta"] = a
    else:
        ranges = ooc_fft.partition_units(n, len(devs), 1)
        arrays = ooc_fft.inverse_to_card_shards(
            work, n, [(lo, hi - lo, d) for (lo, hi), d in zip(ranges, devs)],
            pencil_batch=pencil_batch, pass2=False)
        ux = [dict(lo=lo, hi=hi, device=d, delta=a)
              for ((lo, hi), d), a in zip(zip(ranges, devs), arrays)]
    del arrays
    umax = max(float(np.asarray(jax.numpy.max(jax.numpy.abs(s["delta"])))) for s in ux)

    uy = np.empty((n, n, n), dtype=dt)
    ooc_fft.kspace_pass_device([(1.0, spec)], n, box, kernel=K.grad_invk2(1), out=work,
                               inverse=True, **kw)
    for lo, s in ooc_fft.inverse_to_slabs_device(work, n, slab=slab, pass2=False, **kw):
        uy[lo:lo + s.shape[0]] = s
        umax = max(umax, float(np.max(np.abs(s))))

    # U_z takes the work buffer's own bytes (n^2 (n/2+1) x 2w >= n^3 x w) rather than a new
    # field after `del work`: a device array uploaded from a slice of `work` can keep the
    # whole buffer alive until the runtime releases it, and an allocation inside that window
    # held a 4th field on the host (measured on CPU devices, 2026-09-15: 4.17 fields where
    # the design is 3). Reuse makes the peak independent of when that happens.
    uz = work.reshape(-1).view(np.uint8)[:n**3 * dt.itemsize].view(dt).reshape(n, n, n)
    del work
    ooc_fft.kspace_pass_device([(1.0, spec)], n, box, kernel=K.grad_invk2(2), out=spec,
                               inverse=True, **kw)
    for lo, s in ooc_fft.inverse_to_slabs_device(spec, n, slab=slab, pass2=False, **kw):
        uz[lo:lo + s.shape[0]] = s
        umax = max(umax, float(np.max(np.abs(s))))
    del spec
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

    # --- 5. emission -------------------------------------------------------
    t0 = time.perf_counter()
    v_ro = [ooc_fft.StagedArray.open(os.path.join(stage, f"v_{ax}.npy"), dt, (n, n, n))
            for ax in range(3)]
    emission_s = {}
    if emission == "cards":
        written, n_total = demit.emit_t9_slabs_cards(
            workdir, ux, uy, uz, v_ro, t9, n, box, nb, dt, window, timings=emission_s)
    else:
        written, n_total = _emit_t9_slabs(
            workdir, [_CardField(ux), _HostField(uy), _HostField(uz)], v_ro, t9, n, box, nb,
            dt, slab, window)
    del ux, uy, uz
    _lap("emission", t0)

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
    return _write_manifest(workdir, manifest, keep_stage)


def _shared_like(arr, alloc, tag):
    """Move a small array into shared memory if there is an allocator.

    These are the fields that are cheap to copy but must still be SHARED, so
    they are built normally and moved. Only `off`/`w` are large enough that
    the copy itself matters, and those are written in place by the caller.
    """
    if alloc is None:
        return arr
    view = alloc.empty(arr.shape, arr.dtype, tag)
    view[...] = arr
    return view


def read_manifest(workdir):
    """The manifest for a T9 slab directory, with the same two refusals the
    loader applies: a missing manifest marks an incomplete generation (it is
    written last), and an unknown schema is not guessed at.

    Split out of `load_slot_state` so a reader that wants only the metadata --
    `python -m inexor.export` asking what epoch a checkpoint sits at -- gets the
    identical validation without materializing the state.
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


def drop_file_cache(path):
    """Drop `path`'s pages from the page cache. True if the call was made.

    A 4096^3 IC set is 641 GB of reads and the host holds ~855 GB of state on a 1026 GB
    node, so the cache cannot help the second pass and what it does instead is compete:
    gb 1003657 ran its first steps while the kernel was still draining 91 GB of it, and
    the tile loop's window staging was 924 s in that step against 55 s once the cache was
    gone. Best effort -- not every filesystem honours it, and an instrument must never be
    the failure."""
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
):
    """Reassemble a host-resident SlotState from T9 slabs on disk.

    Geometry goes through `state._alloc_geometry` -- the SAME code path as
    `SlotState.build` -- and each brick's tight rows land at its
    `brick_start`; spare rows stay zero and the arena starts empty, exactly
    as `build` leaves them. Refuses a missing manifest (incomplete
    generation), a schema it does not know, and any crc mismatch.

    `drop_cache` drops each slab file's page cache as soon as that slab has been read,
    in both passes (`drop_file_cache`); the loader's own re-read is what the second pass
    is for, and the cache is too small to serve it anyway.
    """
    man = read_manifest(workdir)
    t9 = T9Layout(man["box_size"], man["n_part"], man["bucket_cells"])
    nb = int(man["bricks_per_side"])
    per3 = (t9.n_buckets_side // nb) ** 3
    n_bricks = nb**3
    n = int(man["n_particles"])

    # TWO PASSES, and the second re-reads the files ON PURPOSE. Holding every
    # slab's payload to place it later costs the WHOLE particle set a second
    # time -- 85.9 GB at 2048^3, live at the same moment as the 117.5 GB of
    # destination arrays, for a loader peak of ~216 GB on a 255 GB node. The
    # re-read is ~81 GB off Lustre against 135 GB of resident payload, and it
    # is the cheaper side of that trade by a wide margin.
    n_dropped = [0]

    def _slab(fname):
        path = os.path.join(workdir, fname)
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            occ, off, w, sc = z["occupancy"], z["off"], z["w"], z["scale"]
        for name, arr in (("occupancy", occ), ("off", off), ("w", w), ("scale", sc)):
            crc = zlib.crc32(arr.tobytes())
            if crc != meta["crc32"][name]:
                raise ValueError(
                    f"{fname}:{name} crc mismatch ({crc} != {meta['crc32'][name]}); "
                    "the slab file is corrupt, refusing to load"
                )
        if drop_cache:
            n_dropped[0] += bool(drop_file_cache(path))
        return int(meta["bx"]), off, w, occ, sc

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

    # pass 1: the index only. off/w are dropped at the end of each iteration.
    _say(f"pass 1 of 2 over {len(man['files'])} slabs (index only)")
    occupancy = np.empty(n_bricks * per3, dtype=np.int64)
    # one scale per brick, reassembled from the slabs that own them
    vel_scale = np.ones(n_bricks, dtype=np.float64)
    for _i, fname in enumerate(man["files"]):
        d, _off, _w, occ, sc = _slab(fname)
        occupancy[d * nb * nb * per3 : (d + 1) * nb * nb * per3] = occ
        vel_scale[d * nb * nb : (d + 1) * nb * nb] = sc
        del _off, _w
        if _i % 32 == 31:
            _say(f"pass 1: {_i + 1}/{len(man['files'])} slabs")

    brick_counts = occupancy.reshape(n_bricks, per3).sum(axis=1)
    _, brick_start, n_alloc, n_arena = _alloc_geometry(
        brick_counts, n, brick_slack, alloc_margin, arena_frac
    )
    n_rows = n_alloc + n_arena
    # `alloc` puts the payload straight into shared memory, so `TilePool`
    # adopts it without a second copy. Without one this is np.zeros and the
    # behaviour is exactly what it was.
    _zeros = (lambda shape, dtype, tag: np.zeros(shape, dtype=dtype)) if alloc is None \
        else alloc.zeros
    _say(f"allocating off/w for {n_rows:,} rows "
         f"({n_rows * 9 / 1e9:.1f} GB, {'shared' if alloc else 'private'})")
    off_all = _zeros((n_rows, 3), np.uint8, "off")
    w_all = _zeros((n_rows, 3), np.int16, "w")
    _say("allocated; pass 2 of 2 (payload)")

    # pass 2: place the payload, one slab live at a time
    for _i, fname in enumerate(man["files"]):
        d, off, w, _occ, _sc = _slab(fname)
        row = 0
        for b in range(d * nb * nb, (d + 1) * nb * nb):
            cnt = int(brick_counts[b])
            off_all[brick_start[b] : brick_start[b] + cnt] = off[row : row + cnt]
            w_all[brick_start[b] : brick_start[b] + cnt] = w[row : row + cnt]
            row += cnt
        if row != len(off):
            raise ValueError(f"slab bx={d}: placed {row} rows of {len(off)}")
        del off, w, _occ
        if _i % 32 == 31:
            _say(f"pass 2: {_i + 1}/{len(man['files'])} slabs")

    if drop_cache:
        _say(f"dropped the page cache of {n_dropped[0]} slab reads")
    _say("payload placed; building SlotState")
    st = SlotState(
        t9=t9,
        bricks_per_side=nb,
        brick_start=_shared_like(brick_start, alloc, "brick_start"),
        occupancy=_shared_like(_to_index(occupancy, index_dtype, "initial"),
                               alloc, "occupancy"),
        off=off_all,
        w=w_all,
        vel_scale=_shared_like(vel_scale, alloc, "vel_scale"),
        arena_base=n_alloc,
        arena_bucket=_shared_like(
            np.full(n_arena, -1, dtype=np.int64), alloc, "arena_bucket"),
        n_particles=n,
        ids=None,
    )
    st.check()
    return st


def write_t9_slabs(st, workdir, provenance=None, drop_ids=False, timings=None,
                   max_slabs=None):
    """Write a `SlotState` as T9 slabs: the exact inverse of `load_slot_state`.

    Same `t9-slabs-2` schema `generate_t9_slabs` emits, so an evolved state and
    a freshly generated one are indistinguishable on disk and the loader needs
    no new branch. This is M-v2-6 Stage 4(a): until it existed the engine could
    run a realization and had nowhere to put the result.

    **The state is COMPACTED on the way out, and that is not bookkeeping.** A
    freshly loaded state is tight -- every brick's rows sit at `brick_start`,
    spares are zero, the arena is empty -- but an evolved one has live spares
    and arena residents, and `brick_start` reflects whatever the repack last
    chose. The slab format has no room for any of that, so the writer emits
    each brick's TRUE membership (its live run plus its arena residents, an
    arena particle still belongs to its brick) in bucket order, tight. The
    allocation geometry is deliberately not preserved: `load_slot_state`
    re-derives it from the occupancy through `_alloc_geometry`, the same call
    `SlotState.build` makes, so it is a function of the data plus the slack
    parameters rather than something the file should pin.

    **`vel_scale` is copied through, never recomputed.** `generate_t9_slabs`
    derives each brick's scale from the velocities it is about to encode, which
    is right at generation and WRONG here: `w` is already int16 against the
    existing scale, so re-deriving one from decoded velocities would re-encode
    every row and lose bits on any brick whose membership changed. The scales
    ride out verbatim and the payload is copied, not decoded.

    The round trip is therefore a FIXED POINT rather than array equality:
    `write(load(write(st)))` reproduces `write(st)` byte for byte, per-array
    crc32 included. Row order within a bucket carries no physics (D-v2-21 got a
    bitwise-identical force from a different membership order, which is what
    the integer paints buy), so array equality against `st` is the wrong
    invariant and this is the right one.

    IDs are not in the schema. A state carrying them refuses rather than
    dropping them silently; pass `drop_ids=True` to say the loss is intended.

    `timings`, if a dict, accumulates seconds per part over the slabs written
    (`index`, `gather`, `crc32`, `write`) plus `slabs`. `max_slabs` writes only
    the first that many slabs and returns None: a timing probe, with no manifest
    and no conservation check, so the directory can never load as a checkpoint.
    """
    if st.ids is not None and not drop_ids:
        raise ValueError(
            "this state carries ids and the t9-slabs-2 schema has no room for them; "
            "pass drop_ids=True to write the state without them"
        )
    os.makedirs(workdir, exist_ok=True)
    # FIRST, before a single slab moves. Writing into a directory that already
    # holds a checkpoint would otherwise leave the OLD manifest standing over a
    # half-replaced set of slabs, and that mixture loads clean: every slab's
    # crc32 lives in its own file and so agrees with whichever generation wrote
    # it. Removing the manifest up front makes a torn write refuse instead.
    mpath = os.path.join(workdir, MANIFEST)
    if os.path.exists(mpath):
        os.remove(mpath)
    nb = st.bricks_per_side
    p3 = st.buckets_per_brick
    nbb = nb * nb                      # bricks per x-slab; a brick is in exactly one
    written, n_written = [], 0

    clock = time.perf_counter

    def _add(key, t0):
        if timings is not None:
            timings[key] = timings.get(key, 0.0) + clock() - t0

    for d in range(nb if max_slabs is None else min(nb, int(max_slabs))):
        t0 = clock()
        lo_brick = d * nbb
        lo_bucket = lo_brick * p3
        hi_bucket = lo_bucket + nbb * p3
        occ_g = st.occupancy[lo_bucket:hi_bucket].astype(np.int64)

        # Every live row of the slab, in (brick, bucket) order. Buckets are
        # brick-major, so one ascending pass over the slab's occupancy IS brick
        # order -- no per-brick loop, which at C-gh would be 16,384 iterations
        # per slab and 2.1e6 per checkpoint.
        counts = occ_g.reshape(nbb, p3).sum(axis=1)
        starts = st.brick_start[lo_brick : lo_brick + nbb].astype(np.int64)
        base = np.concatenate([[0], np.cumsum(counts)[:-1]])
        slots = np.repeat(starts - base, counts) + np.arange(int(counts.sum()), dtype=np.int64)
        keys = lo_bucket + np.repeat(np.arange(nbb * p3, dtype=np.int64), occ_g)

        # Fold this slab's arena residents back into their own buckets. They are
        # few (0.57% at the operating point, D-v2-19 cl.4) so a searchsorted
        # merge beats re-sorting the slab; `side="right"` puts them after the
        # live rows of the same bucket, which is arbitrary but DETERMINISTIC,
        # and determinism is the whole of what the fixed-point gate needs.
        if st.n_arena:
            sel = np.nonzero((st.arena_bucket >= lo_bucket) & (st.arena_bucket < hi_bucket))[0]
            if len(sel):
                a_keys = st.arena_bucket[sel]
                order = np.argsort(a_keys, kind="stable")
                a_keys = a_keys[order]
                a_slots = st.arena_base + sel[order]
                pos = np.searchsorted(keys, a_keys, side="right")
                keys = np.insert(keys, pos, a_keys)
                slots = np.insert(slots, pos, a_slots)

        occ = np.bincount(keys - lo_bucket, minlength=nbb * p3).astype(np.int64)
        _add("index", t0)
        t0 = clock()
        off = st.off[slots]
        w = st.w[slots]
        scale_d = np.asarray(st.vel_scale[lo_brick : lo_brick + nbb], dtype=np.float64)
        _add("gather", t0)
        t0 = clock()
        meta = dict(
            schema=SCHEMA,
            bx=d,
            n_rows=int(len(slots)),
            bucket_lo=int(lo_bucket),
            brick_lo=int(lo_brick),
            crc32=dict(
                occupancy=zlib.crc32(occ.tobytes()),
                off=zlib.crc32(off.tobytes()),
                w=zlib.crc32(w.tobytes()),
                scale=zlib.crc32(scale_d.tobytes()),
            ),
        )
        _add("crc32", t0)
        t0 = clock()
        path = os.path.join(workdir, f"t9_slab_{d:04d}.npz")
        np.savez(path, meta=json.dumps(meta), occupancy=occ, off=off, w=w, scale=scale_d)
        _add("write", t0)
        if timings is not None:
            timings["slabs"] = timings.get("slabs", 0) + 1
        written.append(os.path.basename(path))
        n_written += len(slots)

    if max_slabs is not None:
        return None

    # Conservation, not a formality: a dropped arena row is a deleted particle
    # and nothing downstream would raise on it.
    if n_written != st.n_live:
        raise RuntimeError(f"wrote {n_written} rows for a state holding {st.n_live} particles")

    manifest = dict(
        schema=SCHEMA,
        files=written,
        n_particles=int(n_written),
        box_size=float(st.t9.box_size),
        n_part=int(st.t9.n_part),          # PER SIDE; `n_particles` is the total
        bucket_cells=int(st.t9.bucket_cells),
        bricks_per_side=int(nb),
        source="write_t9_slabs",
        provenance=provenance or {},
    )
    # LAST, and that is the completeness marker `load_slot_state` refuses on.
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest
