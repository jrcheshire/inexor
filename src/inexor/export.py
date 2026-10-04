"""Portable particle export: the evolved T9 state as plain float `.npy` arrays.

Writes `x.npy`, `v.npy` (and `ids.npy` when the state carries ids) plus an `export.json`
header with box, units, crc32 sums and provenance. Files are streamed as raw bytes after a
hand-written `.npy` header (not `open_memmap`, whose dirty pages would count against the
process), so peak memory is one brick chunk and consumers can `np.load(..., mmap_mode="r")`.

Row order is brick-major (the engine's spatial layout) and carries no Lagrangian identity;
a state built with `with_ids=True` exports `ids.npy` for cross-matching.

Across ranks (`write_particle_parts`, format `inexor-particles-2`) each rank writes its own
rows as one part, `x.r<rank>.npy` / `v.r<rank>.npy`; the parts in rank order are the
single-file arrays row for row, and the header carries each part's crc32 and the whole
array's, which is comparable with a single-file export's. `iter_particle_parts` reads either
format.

Units: positions comoving Mpc/h in [0, box_size). Velocities are the native D-time
`dx/dD` (Mpc/h per unit growth factor) unless `a` and `cosmo` convert them to peculiar km/s;
the header names the units and records the conversion factor.
"""

import dataclasses
import json
import os
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .cosmology import E_of_a, growth_factor_a, growth_rate_a

FORMAT = "inexor-particles-1"
FORMAT_PARTS = "inexor-particles-2"
HEADER = "export.json"


def peculiar_velocity_factor(a, cosmo):
    """Multiply D-time velocity by this to get peculiar velocity in km/s.

    `v_pec = a dx/dt = a f D H v_D` with `f = dlnD/dlna`; with lengths in Mpc/h the h cancels:
    `v_pec [km/s] = 100 a f(a) D(a) E(a) v_D`.
    """
    a = float(a)
    return 100.0 * a * growth_rate_a(a, cosmo) * growth_factor_a(a, cosmo) * E_of_a(a, cosmo)


class _StreamedNpy:
    """A `.npy` file of known shape, written in row chunks.

    Writes the standard header for the final shape up front, then appends raw C-order bytes
    and accumulates their crc32.
    """

    def __init__(self, path, shape, dtype):
        self.path = path
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self.rows = 0
        self.crc = 0
        self._fh = open(path, "wb")
        np.lib.format.write_array_header_1_0(
            self._fh,
            {"descr": np.lib.format.dtype_to_descr(self.dtype),
             "fortran_order": False,
             "shape": self.shape},
        )

    def append(self, block):
        b = np.ascontiguousarray(block, dtype=self.dtype)
        if b.shape[1:] != self.shape[1:]:
            raise ValueError(f"block shape {b.shape} does not match {self.shape} past axis 0")
        # A byte view, not a `tobytes()` copy; `write` and `crc32` both take a buffer.
        raw = b.reshape(-1).data.cast("B")
        self._fh.write(raw)
        self.crc = zlib.crc32(raw, self.crc)
        self.rows += b.shape[0]

    def close(self):
        """Close; raises on a short file. This is the export's conservation check.

        The declared row count is `st.n_live`, so any lost rows (a skipped brick, arena
        residents not folded in) fail here instead of producing a short file that loads clean.
        """
        self._fh.close()
        if self.rows != self.shape[0]:
            raise RuntimeError(
                f"{os.path.basename(self.path)}: wrote {self.rows} rows for a state holding "
                f"{self.shape[0]} particles. Rows were lost on the way out; the file would "
                "otherwise load clean and be short."
            )


def write_particles(
    st,
    workdir,
    dtype=np.float32,
    a=None,
    cosmo=None,
    chunk_bricks=1024,
    provenance=None,
    timings=None,
    progress=None,
):
    """Write `st` as portable `(x, v)` float arrays under `workdir`.

    Streams groups of `chunk_bricks` bricks through `SlotState.decode_bricks` (the engine's own
    decode, arena residents included), so the exported positions are those the force saw.
    Decode is f64; `dtype` is the on-disk type, and f32 resolves the T9 position quantum
    `box / n_levels` with margin (the margin shrinks as the box grows).

    `a` and `cosmo` together convert velocities to peculiar km/s; one without the other raises.
    One writer thread at depth one overlaps a chunk's write with the next chunk's decode (at
    most two chunks live). `timings`, if a dict, accumulates `decode`, `write` (time blocked on
    the writer, so parts sum to the wall), `write thread` (writer busy time, the one to take
    throughput against) and `chunks`. `progress` is called once per chunk.

    Returns the header dict. The header is removed first and written last, so its presence
    marks a complete export.
    """
    if (a is None) != (cosmo is None):
        raise ValueError(
            "peculiar-velocity output needs BOTH `a` and `cosmo`; got "
            f"a={a!r}, cosmo={'set' if cosmo is not None else None}. Omit both to write "
            "the engine's native D-time velocity."
        )
    chunk_bricks = int(chunk_bricks)
    if chunk_bricks < 1:
        raise ValueError(f"chunk_bricks must be >= 1, got {chunk_bricks}")

    os.makedirs(workdir, exist_ok=True)
    hpath = os.path.join(workdir, HEADER)
    if os.path.exists(hpath):
        os.remove(hpath)

    n = int(st.n_live)
    kms = a is not None
    vfac = peculiar_velocity_factor(a, cosmo) if kms else 1.0

    streams = {
        "x": _StreamedNpy(os.path.join(workdir, "x.npy"), (n, 3), dtype),
        "v": _StreamedNpy(os.path.join(workdir, "v.npy"), (n, 3), dtype),
    }
    if st.ids is not None:
        streams["ids"] = _StreamedNpy(os.path.join(workdir, "ids.npy"), (n,), np.int32)

    clock = time.perf_counter

    def _add(key, t0):
        if timings is not None:
            timings[key] = timings.get(key, 0.0) + clock() - t0

    def _write_chunk(x, v, ids_block):
        """Writer-thread body. Must run on one worker, one chunk at a time (row order)."""
        t0 = clock()
        streams["x"].append(x)
        streams["v"].append(v)
        if ids_block is not None:
            streams["ids"].append(ids_block)
        return clock() - t0

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="export-writer")
    pending = None
    n_chunks = 0

    def _join():
        nonlocal pending
        if pending is None:
            return
        f, pending = pending, None
        t0 = clock()
        dt = f.result()             # where a write error surfaces
        _add("write", t0)
        if timings is not None:
            timings["write thread"] = timings.get("write thread", 0.0) + dt

    try:
        bricks = range(st.n_bricks)
        n_groups = (st.n_bricks + chunk_bricks - 1) // chunk_bricks
        for gi, lo in enumerate(range(0, st.n_bricks, chunk_bricks)):
            if progress is not None:
                progress("export chunk", gi, n_groups)
            t0 = clock()
            group = list(bricks[lo : lo + chunk_bricks])
            slots, x, v = st.decode_bricks(group)
            if not len(slots):
                continue
            if kms:
                v = v * vfac
            ids_block = st.ids[slots] if "ids" in streams else None
            _add("decode", t0)
            _join()
            pending = pool.submit(_write_chunk, x, v, ids_block)
            n_chunks += 1
        _join()
        if progress is not None:
            progress("export chunk", n_groups, n_groups)
    finally:
        pool.shutdown(wait=True)

    if timings is not None:
        timings["chunks"] = n_chunks

    for s in streams.values():
        s.close()

    header = dict(
        format=FORMAT,
        n_particles=n,
        box_size=float(st.t9.box_size),
        n_part=int(st.t9.n_part),
        dtype=np.dtype(dtype).name,
        files={k: os.path.basename(s.path) for k, s in streams.items()},
        crc32={k: int(s.crc) for k, s in streams.items()},
        units=dict(
            position="Mpc/h comoving, in [0, box_size)",
            velocity="km/s peculiar" if kms else "Mpc/h per unit growth factor (dx/dD)",
        ),
        velocity_is_dtime=not kms,
        peculiar_velocity_factor=(float(vfac) if kms else None),
        a=(float(a) if kms else None),
        cosmology=(dataclasses.asdict(cosmo) if kms else None),
        row_order="brick-major (the engine's spatial layout); no Lagrangian identity",
        has_ids="ids" in streams,
        source="write_particles",
        provenance=provenance or {},
    )
    with open(hpath, "w") as fh:
        json.dump(header, fh, indent=1)
    return header


def _gf2_times(mat, vec):
    out = 0
    i = 0
    while vec:
        if vec & 1:
            out ^= mat[i]
        vec >>= 1
        i += 1
    return out


def _gf2_square(mat):
    return [_gf2_times(mat, m) for m in mat]


def _crc32_combine(crc1, crc2, len2):
    """crc32 of `A + B` from `crc1 = crc32(A)`, `crc2 = crc32(B)` and `len2 = len(B)`.

    zlib's `crc32_combine` (absent from Python's zlib): appending `len2` zero bytes to A is a
    linear map on its crc, applied by repeated squaring in O(log len2).
    """
    crc1, crc2, len2 = int(crc1), int(crc2), int(len2)
    if len2 < 0:
        raise ValueError(f"len2 must be >= 0, got {len2}")
    if len2 == 0:
        return crc1
    odd = [0xEDB88320] + [1 << i for i in range(31)]  # one zero bit
    even = _gf2_square(odd)  # two zero bits
    odd = _gf2_square(even)  # four zero bits
    while True:
        even = _gf2_square(odd)  # first pass: one zero byte
        if len2 & 1:
            crc1 = _gf2_times(even, crc1)
        len2 >>= 1
        if not len2:
            break
        odd = _gf2_square(even)
        if len2 & 1:
            crc1 = _gf2_times(odd, crc1)
        len2 >>= 1
        if not len2:
            break
    return (crc1 ^ crc2) & 0xFFFFFFFF


def _part_files(rank, with_ids):
    names = {"x": f"x.r{int(rank):04d}.npy", "v": f"v.r{int(rank):04d}.npy"}
    if with_ids:
        names["ids"] = f"ids.r{int(rank):04d}.npy"
    return names


def _row_bytes(key, dtype):
    return 4 if key == "ids" else 3 * np.dtype(dtype).itemsize


def _host_chunks(st, chunk_bricks, vfac=None):
    """(slots, x, v) per group of `chunk_bricks` of the state's own bricks, host decode;
    `v` times `vfac` when given."""
    lo, hi = st.owned_bricks
    for b0 in range(lo, hi, chunk_bricks):
        slots, x, v = st.decode_bricks(list(range(b0, min(b0 + chunk_bricks, hi))))
        if len(slots):
            yield slots, x, (v if vfac is None else v * vfac)


def write_particle_parts(
    st,
    workdir,
    comm=None,
    dtype=np.float32,
    a=None,
    cosmo=None,
    decode="host",
    devices=None,
    chunk_bricks=None,
    chunks=None,
    expect_total=None,
    provenance=None,
    timings=None,
    progress=None,
):
    """Write `st` as one part of a multi-part `(x, v)` export (format `inexor-particles-2`).

    Every rank calls this with its node-local state (one rank: the whole state). A rank's
    rows start at the exclusive scan of the ranks' particle counts, and its owned bricks are
    contiguous (x-major slabs), so the parts concatenated in rank order are
    `write_particles`'s arrays byte for byte, and the header's whole-array crc32 is theirs.

    `decode` is "host" (`decode_bricks` over groups of `chunk_bricks`, default 1024) or
    "cards" (`device.export.export_chunks` on `devices`, chunks of `chunk_bricks`, default
    the coarse paint's); the two write the same bytes. `chunks` overrides both: an iterable
    of `(slots, x, v)` in brick order over the state's own bricks, already in output units.
    `a`, `cosmo`, `dtype`, `timings` and `progress` as `write_particles`. `expect_total`
    (e.g. the checkpoint's count) is checked against the ranks' sum on rank 0.

    Rank 0 removes the header before any part is written and writes it after every part has
    closed, so its presence marks a complete export. Returns the header on every rank.
    """
    if (a is None) != (cosmo is None):
        raise ValueError(
            "peculiar-velocity output needs BOTH `a` and `cosmo`; got "
            f"a={a!r}, cosmo={'set' if cosmo is not None else None}. Omit both to write "
            "the engine's native D-time velocity."
        )
    if comm is None:
        from .comm import SerialComm

        comm = SerialComm()
    rank = int(comm.rank)
    if decode not in ("host", "cards"):
        raise ValueError(f"decode must be 'host' or 'cards', got {decode!r}")
    if chunk_bricks is not None and int(chunk_bricks) < 1:
        raise ValueError(f"chunk_bricks must be >= 1, got {chunk_bricks}")
    os.makedirs(workdir, exist_ok=True)
    hpath = os.path.join(workdir, HEADER)
    if rank == 0 and os.path.exists(hpath):
        os.remove(hpath)
    comm.barrier()

    n = int(st.n_live)
    counts = [int(c) for c in comm.allgather(n)]
    row0 = int(sum(counts[:rank]))
    kms = a is not None
    vfac = peculiar_velocity_factor(a, cosmo) if kms else 1.0
    with_ids = st.ids is not None
    names = _part_files(rank, with_ids)
    streams = {k: _StreamedNpy(os.path.join(workdir, f), (n,) if k == "ids" else (n, 3),
                               np.int32 if k == "ids" else dtype)
               for k, f in names.items()}

    clock = time.perf_counter

    def _add(key, t0):
        if timings is not None:
            timings[key] = timings.get(key, 0.0) + clock() - t0

    def _write_chunk(x, v, ids_block):
        """Writer-thread body. Must run on one worker, one chunk at a time (row order)."""
        t0 = clock()
        streams["x"].append(x)
        streams["v"].append(v)
        if ids_block is not None:
            streams["ids"].append(ids_block)
        return clock() - t0

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="export-writer")
    pending = None
    n_chunks = 0

    def _join():
        nonlocal pending
        if pending is None:
            return
        f, pending = pending, None
        t0 = clock()
        dt = f.result()             # where a write error surfaces
        _add("write", t0)
        if timings is not None:
            timings["write thread"] = timings.get("write thread", 0.0) + dt

    vf = vfac if kms else None
    if chunks is not None:
        source = iter(chunks)
    elif decode == "host":
        source = _host_chunks(st, 1024 if chunk_bricks is None else int(chunk_bricks), vf)
    else:
        from .device.export import export_chunks

        source = export_chunks(st, devices=devices, chunk_bricks=chunk_bricks, dtype=dtype,
                               vfac=vf, with_slots=with_ids, timings=timings)
    try:
        while True:
            t0 = clock()
            item = next(source, None)
            if item is None:
                break
            slots, x, v = item
            ids_block = st.ids[slots] if with_ids else None
            _add("decode", t0)
            _join()
            pending = pool.submit(_write_chunk, x, v, ids_block)
            n_chunks += 1
            if progress is not None:
                progress("export chunk", streams["x"].rows, n)
        _join()
    finally:
        pool.shutdown(wait=True)
        if hasattr(source, "close"):
            source.close()  # a card source's threads stop with it
    if timings is not None:
        timings["chunks"] = n_chunks
    for s_ in streams.values():
        s_.close()

    mine = dict(rank=rank, row0=row0, rows=n, files=dict(names),
                crc32={k: int(s_.crc) for k, s_ in streams.items()})
    parts = comm.allgather(mine)
    total = int(sum(p["rows"] for p in parts))
    header = None
    if rank == 0:
        if expect_total is not None and total != int(expect_total):
            raise RuntimeError(
                f"the parts hold {total} rows and {int(expect_total)} were expected: rows "
                "were lost or doubled across ranks; refusing to write the header")
        whole = {}
        for k in names:
            c = parts[0]["crc32"][k]
            for p in parts[1:]:
                c = _crc32_combine(c, p["crc32"][k], p["rows"] * _row_bytes(k, dtype))
            whole[k] = int(c)
        header = dict(
            format=FORMAT_PARTS,
            n_particles=total,
            box_size=float(st.t9.box_size),
            n_part=int(st.t9.n_part),
            dtype=np.dtype(dtype).name,
            parts=parts,
            crc32=whole,
            units=dict(
                position="Mpc/h comoving, in [0, box_size)",
                velocity="km/s peculiar" if kms else "Mpc/h per unit growth factor (dx/dD)",
            ),
            velocity_is_dtime=not kms,
            peculiar_velocity_factor=(float(vfac) if kms else None),
            a=(float(a) if kms else None),
            cosmology=(dataclasses.asdict(cosmo) if kms else None),
            row_order=("brick-major (the engine's spatial layout), parts in rank order; "
                       "no Lagrangian identity"),
            has_ids=with_ids,
            source="write_particle_parts",
            decode=str(decode),
            provenance=provenance or {},
        )
        with open(hpath, "w") as fh:
            json.dump(header, fh, indent=1)
    return comm.bcast(header)


def _read_header(workdir):
    hpath = os.path.join(workdir, HEADER)
    if not os.path.exists(hpath):
        raise FileNotFoundError(
            f"no {HEADER} in {workdir}: the header is written last, so its absence marks "
            "an incomplete or interrupted export; refusing to load"
        )
    with open(hpath) as fh:
        head = json.load(fh)
    if head.get("format") not in (FORMAT, FORMAT_PARTS):
        raise ValueError(f"format {head.get('format')!r} is neither {FORMAT!r} nor "
                         f"{FORMAT_PARTS!r}")
    return head


def _load_files(workdir, files, crcs, mmap):
    out = {}
    for key, fname in files.items():
        arr = np.load(os.path.join(workdir, fname), mmap_mode="r" if mmap else None)
        if not mmap:
            crc = zlib.crc32(np.ascontiguousarray(arr).tobytes())
            if crc != crcs[key]:
                raise ValueError(
                    f"{fname} crc mismatch ({crc} != {crcs[key]}); the export is "
                    "corrupt or truncated, refusing to load"
                )
        out[key] = arr
    return out


def iter_particle_parts(workdir, mmap=True):
    """Yield `(row0, x, v, ids)` per part of an export, in row order (`ids` None if absent).

    Reads both formats: a single-file export is one part at row 0. `mmap` as
    `load_particles` (False loads and verifies each part's crc32).
    """
    head = _read_header(workdir)
    if head["format"] == FORMAT:
        out = _load_files(workdir, head["files"], head["crc32"], mmap)
        yield 0, out["x"], out["v"], out.get("ids")
        return
    for p in head["parts"]:
        out = _load_files(workdir, p["files"], p["crc32"], mmap)
        if len(out["x"]) != p["rows"]:
            raise ValueError(f"part {p['rank']} holds {len(out['x'])} rows, header says "
                             f"{p['rows']}")
        yield int(p["row0"]), out["x"], out["v"], out.get("ids")


def load_particles(workdir, mmap=True):
    """Read an export back: `(header, x, v, ids)`, `ids` None when absent.

    Raises on a missing header (incomplete export). `mmap=True` returns memmaps and skips the
    crc check (it would read every byte); `mmap=False` loads and verifies crc32. A multi-part
    export is concatenated with `mmap=False`; with `mmap=True` it is refused unless it has one
    part (use `iter_particle_parts`).
    """
    head = _read_header(workdir)
    if head["format"] == FORMAT_PARTS and mmap and len(head["parts"]) > 1:
        raise ValueError(
            f"{workdir} holds {len(head['parts'])} parts, which cannot be one memmap; read "
            "them with iter_particle_parts, or pass mmap=False to concatenate in memory")
    got = list(iter_particle_parts(workdir, mmap=mmap))
    if len(got) == 1:
        _, x, v, ids = got[0]
        return head, x, v, ids
    x = np.concatenate([g[1] for g in got])
    v = np.concatenate([g[2] for g in got])
    ids = None if got[0][3] is None else np.concatenate([g[3] for g in got])
    return head, x, v, ids


def _resolve_epoch(man, args):
    """Pick the output epoch and cosmology for the CLI. Returns `(a, cosmo, why)`.

    `why` goes into the header's provenance. Precedence: `--d-time` (native dx/dD); `--a`
    (overrides a recorded epoch; needed for checkpoints that record none); the checkpoint's
    own `provenance.a` + `cosmology`; else D-time, announced. `--omega-m`/`--h` override
    single cosmology fields and are refused when no epoch is in use. An IC directory's
    top-level `a_init` is deliberately not used: it records no cosmology.
    """
    from .config import Cosmology

    rec = man.get("provenance", {}) or {}
    rec_a = rec.get("a")
    rec_cosmo = rec.get("cosmology")
    overrides = {k: v for k, v in (("Omega_m", args.omega_m), ("h", args.h)) if v is not None}

    if args.d_time:
        if args.a is not None or overrides:
            raise SystemExit(
                "--d-time writes the engine's native velocity, so --a/--omega-m/--h have "
                "nothing to act on. Drop --d-time for km/s, or drop the epoch flags."
            )
        return None, None, "--d-time"

    a = args.a if args.a is not None else rec_a
    if a is None:
        if overrides:
            raise SystemExit(
                "--omega-m/--h need an epoch: this checkpoint records none and --a was not "
                "given, so there is no scale factor to convert at. Pass --a as well, or "
                "--d-time to write the native velocity deliberately."
            )
        return None, None, "no epoch recorded; pass --a for km/s"

    base = dict(rec_cosmo) if rec_cosmo else {}
    try:
        cosmo = Cosmology(**{**base, **overrides})
    except TypeError as exc:
        raise SystemExit(
            f"the checkpoint records a cosmology this build cannot read ({exc}); "
            "pass --a with --omega-m/--h to supply one, or --d-time"
        ) from exc

    if args.a is not None and rec_a is not None and args.a != rec_a:
        why = f"--a, overriding the recorded a={rec_a:.6g}"
    elif args.a is not None:
        why = "--a"
    else:
        why = "checkpoint epoch"
    if overrides:
        why += "; cosmology overridden: " + ", ".join(
            f"{k}={v!r}" for k, v in sorted(overrides.items())
        )
    return float(a), cosmo, why


def _main(argv=None):
    """Turn a T9 checkpoint on disk into a portable export.

    `checkpoint_dir` holds a `manifest.json` (a run's `genN` checkpoint or any
    `write_t9_slabs` output). Velocities default to peculiar km/s at the epoch the
    checkpoint records (`engine.run(epoch=...)`); `--a` overrides it, `--d-time` writes the
    native dx/dD velocity, and a checkpoint with no epoch falls back to D-time and says so.
    """
    import argparse

    from .icgen import load_slot_state, read_manifest

    ap = argparse.ArgumentParser(prog="python -m inexor.export", description=_main.__doc__)
    ap.add_argument("checkpoint_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--chunk-bricks", type=int, default=1024)
    ap.add_argument(
        "--a", type=float, default=None,
        help="output scale factor, overriding the epoch the checkpoint recorded; "
             "use with --omega-m and --h",
    )
    ap.add_argument("--omega-m", type=float, default=None)
    ap.add_argument("--h", type=float, default=None)
    ap.add_argument(
        "--d-time", action="store_true",
        help="write the engine's native dx/dD velocity instead of peculiar km/s",
    )
    args = ap.parse_args(argv)

    man = read_manifest(args.checkpoint_dir)
    a, cosmo, why = _resolve_epoch(man, args)
    st = load_slot_state(args.checkpoint_dir)
    head = write_particles(
        st, args.out_dir,
        dtype=np.dtype(args.dtype),
        a=a, cosmo=cosmo,
        chunk_bricks=args.chunk_bricks,
        provenance=dict(source_checkpoint=os.path.abspath(args.checkpoint_dir),
                        epoch_source=why),
    )
    gb = head["n_particles"] * 6 * np.dtype(args.dtype).itemsize / 1e9
    if a is not None:
        print(f"  velocities: km/s peculiar at a={a:.6g}, Omega_m={cosmo.Omega_m!r}, "
              f"h={cosmo.h!r} ({why})")
    else:
        print(f"  velocities: {head['units']['velocity']} ({why})")
    print(
        f"{head['n_particles']} particles -> {args.out_dir} "
        f"({gb:.3f} GB, {head['units']['velocity']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
