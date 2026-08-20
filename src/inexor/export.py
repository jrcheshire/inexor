"""Portable particle export: the evolved state as plain float arrays.

M-v2-6 Stage 4, the SECOND writer. `icgen.write_t9_slabs` writes the state in
the engine's own T9 encoding, which is what a checkpoint and a restart need and
what nothing outside this package can read: int8 offsets into a bucket, int16
velocity codes against a per-brick scale, on a brick-sorted slot layout. A halo
finder or a mock pipeline wants six floats per particle. This module writes
those, streamed, so the file is 206 GB at C-gh and the memory to produce it is
one brick chunk.

What comes out is standard `.npy` -- `x.npy`, `v.npy`, and `ids.npy` when the
state carries ids -- plus an `export.json` header carrying the box, the units,
the integrity sums and the provenance. `.npy` because it is the most portable
thing that is also memory-mappable: a consumer opens the 103 GB position array
with `np.load(..., mmap_mode="r")` and touches only the pages it reads. The
files are written by streaming raw bytes after a hand-written header rather than
through `open_memmap`, because a 206 GB memmap fills with dirty pages that count
against the process and would land on top of a run already holding 164.6 GB.

**Row order is the engine's spatial order and carries no particle identity.**
Rows come out brick-major, which is the state's own layout, so the export is
spatially coherent and arbitrary within a bucket. Nothing recovers the
Lagrangian index from it. That costs a halo finder nothing and costs
cross-matching everything, which is why a state built `with_ids=True` gets its
ids exported alongside and a state without them says so in the header.

**Units.** Positions are comoving Mpc/h in [0, box_size). Velocities are the
engine's native D-time `v = dx/dD` (Mpc/h per unit growth factor), which is what
the state holds and is NOT what any consumer assumes. Pass `a` and `cosmo` to
convert to peculiar km/s on the way out; either way `units` in the header names
what was written, and the conversion factor rides along so a D-time file can be
converted after the fact without re-deriving it.
"""

import dataclasses
import json
import os
import zlib

import numpy as np

from .cosmology import E_of_a, growth_factor_a, growth_rate_a

FORMAT = "inexor-particles-1"
HEADER = "export.json"


def peculiar_velocity_factor(a, cosmo):
    """Multiply D-time velocity by this to get peculiar velocity in km/s.

    The engine's velocity is `v_D = dx/dD` with `x` comoving in Mpc/h. Peculiar
    velocity is `v_pec = a dx/dt`, and

        dx/dt = (dx/dD)(dD/da)(da/dt) = v_D * (f D / a) * (a H)

    using `f = dlnD/dlna`, so `v_pec = a f(a) D(a) H(a) v_D`. With H = 100 h E(a)
    km/s/Mpc and lengths in Mpc/h the `h` cancels exactly, leaving

        v_pec [km/s] = 100 * a * f(a) * D(a) * E(a) * v_D

    a pure function of the output epoch. `tests/test_export.py` pins it against
    a finite-difference `dD/da` rather than against itself.
    """
    a = float(a)
    return 100.0 * a * growth_rate_a(a, cosmo) * growth_factor_a(a, cosmo) * E_of_a(a, cosmo)


class _StreamedNpy:
    """A `.npy` file of known shape, written in row chunks.

    numpy's own writers want the whole array. This writes the standard header
    for the final shape up front and then appends raw C-order bytes, so the
    result is a file `np.load` opens normally while the peak memory is one
    chunk. crc32 accumulates over the appended bytes for free.
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
        raw = b.tobytes()
        self._fh.write(raw)
        self.crc = zlib.crc32(raw, self.crc)
        self.rows += b.shape[0]

    def close(self):
        """Refuses a short file, which is the export's CONSERVATION check.

        The declared shape comes from `st.n_live`, so a writer that lost rows --
        a brick missing from the sweep, arena residents not folded in -- lands
        here rather than producing a file that loads clean and is short. Both
        defects were planted and both are caught here, which is why there is no
        second count check downstream: it could not fire.
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
):
    """Write `st` as portable `(x, v)` float arrays under `workdir`.

    Streams in groups of `chunk_bricks` bricks through `SlotState.decode_bricks`,
    which is the engine's own decode -- the same call the streamed coarse paint
    makes -- so the exported positions are the positions the force saw, arena
    residents included, rather than a second decode path that could drift from
    it. Peak float memory is one chunk: ~4 M particles at the default and C-gh
    geometry, about 200 MB in f64 before the narrowing cast.

    `dtype` is the ON-DISK float type and defaults to f32, which resolves the
    T9 position quantum with room to spare: the stored position is a lattice
    index times `box / n_levels`, f32's spacing at magnitude `box` is
    `box * 2**-23`, and the ratio is **32x at C-gh and 16x at C-hero**
    (measured, `tests/test_export.py`). It shrinks as the box grows, so it is a
    property of the config rather than a constant, and the test carries the
    numbers. Pass `np.float64` to double the file and gain nothing the state
    holds. The DECODE is f64 either way.

    `a` and `cosmo` together convert velocities to peculiar km/s; giving one
    without the other is refused rather than silently ignored, since the failure
    would be a file whose header claims units it does not carry.

    Returns the header dict. It is written LAST and removed FIRST, so its
    presence marks a complete export -- the same contract as the T9 manifest,
    for the same reason: an interrupted export otherwise leaves full-looking
    arrays that load clean and are short.
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

    bricks = range(st.n_bricks)
    for lo in range(0, st.n_bricks, chunk_bricks):
        group = list(bricks[lo : lo + chunk_bricks])
        slots, x, v = st.decode_bricks(group)
        if not len(slots):
            continue
        streams["x"].append(x)
        # The scale rides in the f64 decode, so a km/s file is the D-time file
        # times one number and carries no extra rounding beyond the output cast.
        streams["v"].append(v * vfac if kms else v)
        if "ids" in streams:
            streams["ids"].append(st.ids[slots])

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
        # Recorded whether or not it was applied, so a D-time file can be
        # converted later without re-deriving the factor or guessing the epoch.
        peculiar_velocity_factor=(float(vfac) if kms else None),
        a=(float(a) if kms else None),
        # The epoch and the factor were here from the start; the COSMOLOGY that
        # turns one into the other was not, so a reader could see `a=0.5` and a
        # factor and still not know which Omega_m produced it. That gap widened
        # once the CLI let the epoch and the cosmology come from different
        # places, so the whole dataclass rides along.
        cosmology=(dataclasses.asdict(cosmo) if kms else None),
        row_order="brick-major (the engine's spatial layout); no Lagrangian identity",
        has_ids="ids" in streams,
        source="write_particles",
        provenance=provenance or {},
    )
    with open(hpath, "w") as fh:
        json.dump(header, fh, indent=1)
    return header


def load_particles(workdir, mmap=True):
    """Read an export back: `(header, x, v, ids)`, `ids` None when absent.

    Refuses a missing header (an incomplete export) and any crc mismatch. The
    crc check reads every byte, so it is skipped under `mmap=True` -- the whole
    point of mapping a 103 GB array is not to read it -- and the arrays come
    back as memmaps. `mmap=False` loads and verifies.
    """
    hpath = os.path.join(workdir, HEADER)
    if not os.path.exists(hpath):
        raise FileNotFoundError(
            f"no {HEADER} in {workdir}: the header is written last, so its absence marks "
            "an incomplete or interrupted export; refusing to load"
        )
    with open(hpath) as fh:
        head = json.load(fh)
    if head.get("format") != FORMAT:
        raise ValueError(f"format {head.get('format')!r} != {FORMAT!r}")

    out = {}
    for key, fname in head["files"].items():
        arr = np.load(os.path.join(workdir, fname), mmap_mode="r" if mmap else None)
        if not mmap:
            crc = zlib.crc32(np.ascontiguousarray(arr).tobytes())
            if crc != head["crc32"][key]:
                raise ValueError(
                    f"{fname} crc mismatch ({crc} != {head['crc32'][key]}); the export is "
                    "corrupt or truncated, refusing to load"
                )
        out[key] = arr
    return head, out["x"], out["v"], out.get("ids")


def _resolve_epoch(man, args):
    """Pick the output epoch and cosmology for the CLI. Returns `(a, cosmo, why)`.

    `why` is carried into the export header's provenance, so a file always says
    where its units came from rather than leaving a reader to infer it from
    whether a number looks like km/s.

    Precedence, and each rung exists for a reason:

    1. `--d-time` -- an explicit request for the engine's native `dx/dD`.
    2. `--a` -- an explicit epoch, which overrides a recorded one. Re-exporting
       a checkpoint at a different epoch is wrong, but `--a` is also how the
       banked pre-epoch artifacts get km/s at all, so this cannot refuse.
    3. the checkpoint's own `a` + `cosmology` -- the default, and the point of
       the exercise.
    4. neither -- D-time, ANNOUNCED. A silent fallback is the failure this whole
       change is about, so the caller is told which rung it landed on.

    `--omega-m` / `--h` override individual cosmology fields on top of whichever
    of 2 or 3 supplied the rest. Passing them with nothing to attach them to is
    refused rather than ignored: the request was to change a number that is not
    being used, and honouring it silently would write a header claiming a
    cosmology that did not enter the file.

    **An IC directory's `a_init` is deliberately NOT read as rung 3.** The
    generator records `a_init` at manifest top level but no cosmology, so half
    the conversion is missing; and exporting unevolved ICs to a halo finder is
    not a use for this tool. Falling back for them is correct. This note exists
    so the asymmetry reads as a choice rather than an oversight.
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
        # Named explicitly: with `--omega-m`/`--h` usable on their own, the
        # epoch and the cosmology can now come from different places, and a
        # line that reported only the epoch would leave that invisible.
        why += "; cosmology overridden: " + ", ".join(
            f"{k}={v!r}" for k, v in sorted(overrides.items())
        )
    return float(a), cosmo, why


def _main(argv=None):
    """Turn a T9 checkpoint on disk into a portable export.

    The engine writes its own encoding every step; this is how that becomes the
    thing a halo finder reads, without re-running anything. `checkpoint_dir` is
    a directory holding a `manifest.json` -- either a `genN` directory under a
    run's `checkpoint_dir`, or any `write_t9_slabs` output.

    **Peculiar km/s is the default output**, because a halo finder is the
    consumer and km/s is what it expects. That is only possible when the
    checkpoint knows its own epoch, which one written by `engine.run(epoch=...)`
    does; see `engine.epoch_record`. `--a` overrides the recorded epoch,
    `--d-time` asks for the engine's native velocity, and a checkpoint carrying
    no epoch falls back to D-time and says so on stdout.
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
    # The epoch leads, because it is the number that silently makes the file
    # wrong: a velocity converted at a neighbouring scale factor is off by tens
    # of percent and looks entirely reasonable on inspection.
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


# Guarded, and not only by convention: an unguarded module here re-imports
# recursively under `spawn` if anything downstream ever starts a pool.
if __name__ == "__main__":
    raise SystemExit(_main())
