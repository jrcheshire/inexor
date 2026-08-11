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
import zlib

import numpy as np

from . import ic, ooc_fft
from .codec import INT16_MAX, T9Layout
from .cosmology import growth_factor_2, growth_factor_a, growth_rate_2, growth_rate_a, ic_k_table
from .layout import DEFAULT_INDEX_DTYPE, _stable_sort_index, _to_index
from .lpt import lpt2_source_from_spec
from .state import (
    SlotState,
    _alloc_geometry,
    _bucket_flat_brick_major,
    encode_positions_host,
    encode_velocities_host,
)

SCHEMA = "t9-slabs-1"
MANIFEST = "manifest.json"


def _open_stage(workdir, name, dtype, shape):
    return np.lib.format.open_memmap(
        os.path.join(workdir, name), mode="w+", dtype=dtype, shape=shape
    )


def _stage_spec_to(workdir, name, spec, n, slab):
    """Inverse-transform a spectrum (consuming it) into a staged memmap."""
    rdt = np.float64 if spec.dtype == np.complex128 else np.float32
    mm = _open_stage(workdir, name, rdt, (n, n, n))
    for lo, s in ooc_fft.inverse_to_slabs(spec, n, slab=slab):
        mm[lo : lo + s.shape[0]] = s
    mm.flush()
    return os.path.join(workdir, name)


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
):
    """Generate T9-encoded initial-condition slabs on disk.

    Writes `t9_slab_{bx:04d}.npz` per destination brick slab plus MANIFEST
    (written LAST -- its absence marks an incomplete generation and the loader
    refuses). Returns the manifest dict. `provenance` (optional dict) is
    stored verbatim in the manifest beside the generator's own fields.
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
    phi_path = os.path.join(stage, "phi.npy")
    phi_mm = _open_stage(stage, "phi.npy", dt, (n, n, n))
    tot = 0.0
    for lo, s in ooc_fft.inverse_to_slabs(spec, n, slab=slab):
        phi_mm[lo : lo + s.shape[0]] = s
        tot = ic.sq_sum_by_plane(s, tot)
    phi_mm.flush()
    del spec, phi_mm
    mean_phi2 = tot / n**3

    phi_ro = np.load(phi_path, mmap_mode="r")

    def _png_slab(lo, hi):
        p = np.asarray(phi_ro[lo:hi])
        return p + np.asarray(f_NL, dtype=p.dtype) * (p * p - np.asarray(mean_phi2, p.dtype))

    spec = ooc_fft.forward_from_slabs(_png_slab, n, slab=slab)
    ooc_fft.mul_radial_inplace(spec, n, box, ic._poisson_fn(cosmo, tab), dc_value=1.0, slab=slab)
    delta_path = _stage_spec_to(stage, "delta.npy", spec, n, slab)
    del spec

    # --- LPT, streamed (lpt.lpt_ics's exact op sequence, order=2) ----------
    delta_ro = np.load(delta_path, mmap_mode="r")
    spec = ooc_fft.forward_from_slabs(lambda lo, hi: np.asarray(delta_ro[lo:hi]), n, slab=slab)
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
        p1 = np.load(os.path.join(stage, f"psi1_{ax}.npy"), mmap_mode="r")
        p2 = np.load(os.path.join(stage, f"psi2_{ax}.npy"), mmap_mode="r")
        u_mm = _open_stage(stage, f"u_{ax}.npy", dt, (n, n, n))
        v_mm = _open_stage(stage, f"v_{ax}.npy", dt, (n, n, n))
        for lo in range(0, n, slab):
            hi = min(lo + slab, n)
            a1 = np.asarray(p1[lo:hi])
            a2 = np.asarray(p2[lo:hi])
            # lpt_ics's canonical combine, per element
            u = dt.type(D1) * a1
            u -= dt.type(D2) * a2
            v = a1 + dt.type(v_coef2) * a2
            u_mm[lo:hi] = u
            v_mm[lo:hi] = v
            vmax = max(vmax, float(np.max(np.abs(np.asarray(v, np.float64)))))
            umax = max(umax, float(np.max(np.abs(u))))
        u_mm.flush()
        v_mm.flush()
        del p1, p2, u_mm, v_mm

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
    per = t9.n_buckets_side // nb
    per3 = per**3
    planes = n // nb  # particle planes per brick slab
    coords = np.arange(n, dtype=dt) * dt.type(box / n)  # lagrangian_grid's exact values
    u_ro = [np.load(os.path.join(stage, f"u_{ax}.npy"), mmap_mode="r") for ax in range(3)]
    v_ro = [np.load(os.path.join(stage, f"v_{ax}.npy"), mmap_mode="r") for ax in range(3)]

    staged = {d: {} for d in range(nb)}  # dest slab -> {src slab: contribution}
    done_src = np.zeros(nb, dtype=bool)
    written = []
    n_total = 0

    def _sources(d):
        return sorted({(d + o) % nb for o in range(-window, window + 1)})

    def _finalize(d):
        parts = [staged[d][s] for s in sorted(staged[d])]  # ascending src = Lagrangian order
        keyf = np.concatenate([p[0] for p in parts]) if parts else np.empty(0, np.int64)
        off = (np.concatenate([p[1] for p in parts]) if parts
               else np.empty((0, 3), np.uint8))
        w = (np.concatenate([p[2] for p in parts]) if parts
             else np.empty((0, 3), np.int16))
        order = _stable_sort_index(keyf)
        keyf, off, w = keyf[order], off[order], w[order]
        lo_bucket = d * nb * nb * per3
        occ = np.bincount(keyf - lo_bucket, minlength=nb * nb * per3).astype(np.int64)
        meta = dict(
            schema=SCHEMA,
            bx=d,
            n_rows=int(len(keyf)),
            bucket_lo=int(lo_bucket),
            crc32=dict(
                occupancy=zlib.crc32(occ.tobytes()),
                off=zlib.crc32(off.tobytes()),
                w=zlib.crc32(w.tobytes()),
            ),
        )
        path = os.path.join(workdir, f"t9_slab_{d:04d}.npz")
        np.savez(path, meta=json.dumps(meta), occupancy=occ, off=off, w=w)
        staged[d].clear()
        written.append(os.path.basename(path))
        return len(keyf)

    finalized = np.zeros(nb, dtype=bool)
    for src in range(nb):
        lo = src * planes
        x_slab = np.empty((planes * n * n, 3), dtype=np.float64)
        v_slab = np.empty((planes * n * n, 3), dtype=np.float64)
        for ax in range(3):
            u = np.asarray(u_ro[ax][lo : lo + planes])
            q = (
                coords[lo : lo + planes].reshape(-1, 1, 1),
                coords.reshape(1, -1, 1),
                coords.reshape(1, 1, -1),
            )[ax]
            x_slab[:, ax] = np.mod(q + u, dt.type(box)).reshape(-1)
            v_slab[:, ax] = np.asarray(v_ro[ax][lo : lo + planes], np.float64).reshape(-1)
        off, bijk = encode_positions_host(x_slab, t9)
        keyf = _bucket_flat_brick_major(bijk, t9, nb)
        w = encode_velocities_host(v_slab, scale)
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
            staged[int(d)][src] = (keyf[m], off[m], w[m])
        n_total += len(keyf)
        done_src[src] = True
        for d in range(nb):
            if not finalized[d] and all(done_src[s] for s in _sources(d)):
                _finalize(d)
                finalized[d] = True

    assert finalized.all()
    if n_total != n**3:
        raise RuntimeError(f"emitted {n_total} particles, expected {n**3}")

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
    with open(os.path.join(workdir, MANIFEST), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def load_slot_state(
    workdir,
    brick_slack=0.10,
    alloc_margin=0.10,
    arena_frac=0.01,
    index_dtype=DEFAULT_INDEX_DTYPE,
):
    """Reassemble a host-resident SlotState from T9 slabs on disk.

    Geometry goes through `state._alloc_geometry` -- the SAME code path as
    `SlotState.build` -- and each brick's tight rows land at its
    `brick_start`; spare rows stay zero and the arena starts empty, exactly
    as `build` leaves them. Refuses a missing manifest (incomplete
    generation), a schema it does not know, and any crc mismatch.
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
    t9 = T9Layout(man["box_size"], man["n_part"], man["bucket_cells"])
    nb = int(man["bricks_per_side"])
    per3 = (t9.n_buckets_side // nb) ** 3
    n_bricks = nb**3
    n = int(man["n_particles"])

    slabs = []
    occupancy = np.empty(n_bricks * per3, dtype=np.int64)
    for fname in man["files"]:
        with np.load(os.path.join(workdir, fname)) as z:
            meta = json.loads(str(z["meta"]))
            occ, off, w = z["occupancy"], z["off"], z["w"]
        for name, arr in (("occupancy", occ), ("off", off), ("w", w)):
            crc = zlib.crc32(arr.tobytes())
            if crc != meta["crc32"][name]:
                raise ValueError(
                    f"{fname}:{name} crc mismatch ({crc} != {meta['crc32'][name]}); "
                    "the slab file is corrupt, refusing to load"
                )
        d = int(meta["bx"])
        occupancy[d * nb * nb * per3 : (d + 1) * nb * nb * per3] = occ
        slabs.append((d, off, w, occ))

    brick_counts = occupancy.reshape(n_bricks, per3).sum(axis=1)
    _, brick_start, n_alloc, n_arena = _alloc_geometry(
        brick_counts, n, brick_slack, alloc_margin, arena_frac
    )
    n_rows = n_alloc + n_arena
    off_all = np.zeros((n_rows, 3), dtype=np.uint8)
    w_all = np.zeros((n_rows, 3), dtype=np.int16)
    for d, off, w, occ in slabs:
        row = 0
        for b in range(d * nb * nb, (d + 1) * nb * nb):
            cnt = int(brick_counts[b])
            off_all[brick_start[b] : brick_start[b] + cnt] = off[row : row + cnt]
            w_all[brick_start[b] : brick_start[b] + cnt] = w[row : row + cnt]
            row += cnt
        if row != len(off):
            raise ValueError(f"slab bx={d}: placed {row} rows of {len(off)}")

    st = SlotState(
        t9=t9,
        bricks_per_side=nb,
        brick_start=brick_start,
        occupancy=_to_index(occupancy, index_dtype, "initial"),
        off=off_all,
        w=w_all,
        vel_scale=float(man["vel_scale"]),
        arena_base=n_alloc,
        arena_bucket=np.full(n_arena, -1, dtype=np.int64),
        n_particles=n,
        ids=None,
    )
    st.check()
    return st
