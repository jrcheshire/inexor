"""Will this configuration fit this machine? `python -m inexor.plan`.

**Every number here already existed and nothing put them in one place.** That is
not a convenience complaint: M-v2-6 Stage 0 measured the engine's end-to-end peak
for the first time and found 8.2 GB at cdev8 where the accounted terms came to
~0.25 GB, because a 32 B/p host array in the kick was in no table. A budget that
omits a term cannot trade against it, and `EngineConfig.mesh_bytes()` -- the one
function that could have caught the analogous mesh case -- was called by nothing
outside its own test.

So this is a front end over four existing sources, not new arithmetic:
`EngineConfig.mesh_bytes` and `.step_bytes` (engine), `T9Layout.index_bytes`
(codec), `ooc_fft.plan_bytes` (the IC stage), plus the T9 payload itself.

**No site knowledge, by design.** Budgets are ARGUMENTS. The 116 GB figure that
appears throughout this project's records is a property of one Vista GH200 host,
not of inexor, and nothing in the package should know it -- pass `--host-gb 116`
if that is your machine, `--host-gb 237` for a Vista gg node, `--host-gb 1007`
for a Stampede3 h100 node. See `docs/running-elsewhere.md`.

What this does NOT do: measure anything. It is arithmetic over a config, and
where the arithmetic is known to be incomplete it says so rather than implying a
total is a peak. The engine's true peak has been measured at exactly two
configurations and both are development-scale.

Examples:
  python -m inexor.plan --preset cdev
  python -m inexor.plan --preset c-gh --host-gb 116
  python -m inexor.plan --n-part 2048 --box 1024 --n-fine 4096 --n-coarse 1024 \
      --tile 256 --buf 32 --host-gb 237 --disk-gb 2000
"""

import argparse
import sys

import numpy as np

# The ratified config table (`docs/plan-plan-v2.md`). Presets are a CONVENIENCE:
# the fine cell is held fixed across the ladder and volume grows, so these differ
# in box and particle count, not in resolution.
PRESETS = {
    "smoke": dict(n_part=32, box=32.0, n_fine=64, n_coarse=16, tile=16, buf=8),
    "cdev8": dict(n_part=128, box=64.0, n_fine=256, n_coarse=64, tile=64, buf=32),
    "cdev": dict(n_part=256, box=128.0, n_fine=512, n_coarse=128, tile=256, buf=32),
    "cgh64": dict(n_part=512, box=256.0, n_fine=1024, n_coarse=256, tile=256, buf=32),
    "c-1024": dict(n_part=1024, box=512.0, n_fine=2048, n_coarse=512, tile=256, buf=32),
    "c-gh": dict(n_part=2048, box=1024.0, n_fine=4096, n_coarse=1024, tile=256, buf=32),
    "c-hero": dict(n_part=4096, box=2048.0, n_fine=8192, n_coarse=2048, tile=512, buf=32),
}

GB = 1e9

# One pool worker's own footprint, MEASURED: Vista gg, 8 workers, 9.0 GB total
# read from /proc/<pid>/statm after a barrier task proved `_worker_init` had
# finished. Mostly the interpreter plus jax; the state itself is shared and is
# already in the tables above, so this is what each worker adds on top.
#
# It is a floor twice over -- taken at pool startup before any tile has been
# forced, and only ever at W=8 -- and it is in this module rather than in
# `executor` because it is a property of a machine, not of the code. See
# `docs/running-elsewhere.md` on why nothing in the package knows a node size.
WORKER_STARTUP_BYTES = 1.12 * GB

# The knobs the PRODUCTION path runs, in one place, because the alternative
# is what happened: `EngineConfig` defaults coarse_dtype to float64 for the
# benefit of the f64 reference arms, `v2_m6_realization.py` did not override
# it, and this planner defaulted to float32 -- so the table said FITS while
# pricing a configuration nobody was running, and the miss (12.885 GB of
# shared memory) took three separate rediscoveries and a dead job to find.
#
# Anything here is a RATIFIED choice with a decision behind it. The gate
# oracles deliberately do NOT use this: their whole job is to vary these.
RATIFIED = dict(
    coarse_dtype="float32",   # D-v2-22 / M-v2-4: 1.83x peak host, 1e-5 error
    fine_dtype="float64",     # unchanged by M-v2-4; the fine arm stays f64
    alpha=1.0,                # r_s / coarse_cell, the ratified kernel
)
BUCKET_CELLS = 2              # D-v2-20's layout


def engine_config(preset, **overrides):
    """The `EngineConfig` the production driver builds, from ONE definition.

    Both `inexor.plan` and `scripts/v2_m6_realization.py` go through this, so
    a planner verdict is about the run that will actually happen. Overrides
    are for the driver's per-invocation knobs (workers, slack, checkpointing),
    not for quietly reinstating a default this dict exists to pin.
    """
    from .engine import EngineConfig

    g = PRESETS[preset] if isinstance(preset, str) else dict(preset)
    kw = dict(
        box_size=g["box"], n_part=g["n_part"], n_fine=g["n_fine"],
        n_coarse=g["n_coarse"], n_tile=g["tile"], b_fine=g["buf"],
        **RATIFIED,
    )
    kw.update(overrides)
    return EngineConfig(**kw)


def _fmt(b):
    return f"{b / GB:10.3f} GB"


def _table(title, terms, total_label="total", reduce=sum):
    print(f"\n{title}")
    width = max(len(k) for k in terms) if terms else 1
    for k, v in sorted(terms.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<{width}}  {_fmt(v)}")
    print(f"  {'-' * width}  {'-' * 13}")
    print(f"  {total_label:<{width}}  {_fmt(reduce(terms.values()))}")


def load_stages(*, n, n_rows, n_buckets, index_itemsize, n_arena, n_bricks,
                n_slabs, shared):
    """Peak resident bytes at each stage of `icgen.load_slot_state`.

    THE LOAD PATH WAS IN NO TABLE, and it is where two jobs died. It is also
    the stage where `shared` changes the answer completely: with an allocator
    the payload is written straight into the segments the pool will use and
    the state exists ONCE; without one it is built privately and copied,
    which is 2x at the moment of copying.

    A slab is one x-slice of bricks, so its payload is n/n_slabs rows of the
    9 B T9 record plus its share of the index.
    """
    slab = n // max(n_slabs, 1) * 9 + n_buckets // max(n_slabs, 1) * 8
    occ64 = n_buckets * 8            # the int64 accumulator, both passes
    payload = n_rows * 9             # off + w
    index = n_buckets * index_itemsize
    stages = {
        "pass 1 (index, one slab live)": occ64 + slab,
        "allocate off/w": occ64 + payload,
        "pass 2 (payload, one slab live)": occ64 + payload + slab,
        # `_to_index` makes the uint32 beside the int64, and a shared build
        # copies that into a segment before the int64 goes away
        "build SlotState": occ64 + payload + index * (2 if shared else 1),
    }
    if not shared:
        # TilePool then copies every field into segments, one at a time
        stages["adopt into the pool"] = payload + index + n_arena * 8 + max(
            n_rows * 6, n_rows * 3)
    return stages


# ---------------------------------------------------------------------------
# The device design: the host is a byte store, the four GPUs do the step.
# ---------------------------------------------------------------------------
#
# WHERE EACH MESH TERM LIVES. This table is a DESIGN ASSERTION, not a reading of
# code -- the device executor does not exist yet. `MESH_PHASE` lives in `engine`
# precisely because a term's phase is a property of the code that allocates it;
# when the device lane exists this table moves into it for the same reason, and
# until then every verdict below is arithmetic over a design, not over a run.
#
#   shard   -- decomposed along x across the GPUs, so each card holds 1/n_gpus.
#              The coarse mesh is planar-decomposed for the factorized FFT, so
#              every coarse term follows the same split.
#   replica -- every card runs whole tiles, so each holds its own copy.
#   host    -- exists, but in HOST memory: no card holds it. Distinct from
#              "gone", which means the term stops existing at all. The
#              factorized coarse solve's spectrum is the reason this category
#              exists -- it is 34.4 GB at c-hero that the design deliberately
#              never puts on a card.
#   gone    -- the host pass the port deletes.
DEVICE_PLACEMENT = {
    "coarse_delta": "shard",
    "coarse_force_resident": "shard",
    "coarse_force_copy_transient": "shard",
    "coarse_kernel_pref": "shard",
    "coarse_match_factor": "shard",
    "coarse_kernel_build_f64": "shard",
    # HOST-RESIDENT: the factorized solve keeps the spectrum in numpy and sends
    # only planes across, so a card never holds one. `coarse_device_planes` is
    # what it DOES hold, and it is a replica because each card transforms its
    # own planes.
    "coarse_spectrum": "host",
    "coarse_solve_work": "host",
    "coarse_kernel_slab": "host",
    "coarse_device_planes": "replica",
    # Priced at the HOST path's int64 width, which over-charges the device form:
    # the sub-block paint accumulates int32 (`painting.paint_tsc_int_subblock`)
    # and the engine's int64 mesh is the parent-side accumulator the port
    # deletes. Left at 8 B/cell deliberately -- a budget that guesses its way
    # DOWN is the shape of a gate that cannot fail.
    "coarse_accumulator": "shard",
    "tile_kernels": "replica",
    "tile_kernel_build_f64": "replica",
    "tile_kernel_pref": "replica",
    "tile_workspace": "replica",
    # `SlotState.decode_bricks` on the host is the pass the design exists to
    # remove; it has no device counterpart because the rows are decoded inside
    # the tile kernel from the streamed slab.
    "coarse_decode_slab": "gone",
}


def shard_halo_planes():
    """x-planes a card's shard holds beyond its 1/n_gpus of the mesh, per term.

    Only for terms whose device code builds them with ghosts: the accumulator's
    1 + 2 (`device.paint.ACC_GHOST_LO/HI`) and the force meshes' `COARSE_HALO`
    on each side (`device.coarse`; 993866 held 516 planes per card at 4096^3,
    record sec. 23).
    """
    from .device.paint import ACC_GHOST_HI, ACC_GHOST_LO
    from .forces import COARSE_HALO

    return {"coarse_accumulator": ACC_GHOST_LO + ACC_GHOST_HI,
            "coarse_force_resident": 2 * COARSE_HALO}

# MEASURED, not chosen: `scripts/v2_g4_gh_memory.py` streams pinned host memory
# in CHUNK_GIB = 2.0 chunks and the device high-water sat at exactly 2 chunks
# (4.0 GiB) at every rung of the 64 -> 640 GiB ladder, on one GPU and on four
# (Vista 974476, record 5z). The per-chunk `block_until_ready` is what makes it
# 2 and not the whole set -- without it XLA keeps every staged copy alive.
STREAM_CHUNK_BYTES = 2.0 * 2**30
STREAM_CHUNKS_IN_FLIGHT = 2

# ONE COARSE-PAINT CHUNK ON A CARD (`device.paint`), MEASURED on a GB200: device
# `peak_bytes_in_use` per padded row, the `off` window upload included.
#
# JITTED, the default (Vista 993294, c672-016): 72.1 / 62.2 / 62.2 B at 16.8M /
# 84.6M / 338M padded rows. Charged at the largest reading.
PAINT_CHUNK_B_PER_ROW = 72
# EAGER (`jit=False`), same job and node: 293.0 / 269.7 / 266.0 B, reproducing
# Vista 993139 (272.0 / 264.8 / 266.2 on c672-004). For pricing the eager path.
PAINT_CHUNK_EAGER_B_PER_ROW = 266
# folded into the measured rate above; kept at 0 so the job card's reader,
# which adds the two, still resolves
PAINT_WINDOW_B_PER_ROW = 0
# THE TRACED READING: bytes alive at once under last-use freeing in the traced
# program, with no operator fusion, differenced over two padded row counts
# (125 B/row; positions + TSC weights alone are 96). It is NOT a floor for the
# compiled program: XLA fuses and reuses buffers, and measured 62 on the card.
# `tests/test_device_paint.py` re-traces the program and fails if it ever
# exceeds this, which catches program growth without a card. 121 before the
# dead rows were spread by default; the +4 is that int32 per-row index, and
# the card read 69.5-71.0 B/row with it (Vista 993600).
PAINT_CHUNK_TRACED_B_PER_ROW = 125
# a chunk's rows are padded on `forces.capacity_shape`'s ladder, whose padding
# is derived at <= 26.0%
PAINT_PAD_BOUND = 1.26

# THE DEVICE MIGRATE'S PEAK PER SLAB ROW, MEASURED on a GB200 after the R1
# retention cuts (Vista 995638, record sec. 31): 1.25 GiB at cgh64 = 320 B per
# slab row, `device.migrate.drift_and_migrate_device` on one card. Charged at
# the slab's rows x this. It scales with slab rows by ARITHMETIC (~86 GB at
# c-hero); the 4096^3 reading is owed to the R2 job.
MIGRATE_DEVICE_B_PER_SLAB_ROW = 320
# THE DEVICE REPACK'S PEAK PER SLAB ROW, MEASURED at a production-shape slab
# (nb=256, 268,439,552 rows, 4,096 residents) on a GB200: 18.75 GB device peak
# = 69.9 B per slab row, windows and program included (Vista 995813, record
# sec. 33). Charged at the slab's rows x this.
REPACK_DEVICE_B_PER_SLAB_ROW = 70
# THE FUSED DEVICE MIGRATE + REPACK'S PEAK PER SLAB ROW, MEASURED on one GB200 at
# 1024^3 (16,777,216 rows per slab): 4.02 GiB = 257 B per slab row
# (`device.fused.migrate_repack_device`, Vista 1001688, record sec. 41). INTERIM:
# not a production-shape reading; the 4096^3 smoke replaces it.
FUSED_DEVICE_B_PER_SLAB_ROW = 257


def device_window_slabs(ec):
    """x-slabs of bricks that must be resident to serve one plane of tiles.

    DERIVED, not picked. A tile draws from `brick_span` bricks per side
    (`layout.brick_span`, the same function the engine's membership uses), so
    walking tiles in x-order needs that many consecutive x-slabs live at once.
    At c-hero it is 18: 512/32 bricks across the tile plus one brick of pad on
    each side, and `choose_brick`'s `c | b_fine` contract is what makes the
    union exactly the padded box rather than a 1.7x superset.
    """
    from .layout import brick_span

    nb = max(1, ec.n_fine // ec.n_brick)
    _pad, span = brick_span(ec.n_tile, ec._b_realized, ec.n_brick, nb)
    return min(span, nb)


def _print_load_and_ic(args, ec, t9, n, rows, arena, shared):
    """The load path and the IC stage. Shared by both backends, and it is the

    same host either way: whoever does the step, the state is still built on the
    host and the IC transform is still out of core.
    """
    idx_itemsize = t9.index_bytes() // max(t9.n_buckets_side**3, 1)
    ld = load_stages(
        n=n, n_rows=rows + arena, n_buckets=t9.n_buckets_side**3,
        index_itemsize=max(idx_itemsize, 1), n_arena=arena,
        n_bricks=ec.n_brick and (args.n_fine // ec.n_brick) ** 3,
        n_slabs=args.slabs, shared=shared,
    )
    _table("LOADING THE STATE, peak resident at each stage", ld,
           total_label="PEAK (max, not sum)", reduce=max)
    print("  the total line above is a MAX: these stages do not coexist")

    try:
        from .ooc_fft import plan_bytes

        ic = plan_bytes(args.n_part, np.dtype(args.coarse_dtype), "derivative")
        print(f"\nIC STAGE (out-of-core, 'derivative' policy): peak "
              f"{_fmt(ic['peak'])}")
    except Exception as exc:  # pragma: no cover - informational only
        print(f"\nIC STAGE: not evaluable here ({exc})")
    return max(ld.values())


#: Card bytes per row of the emission programs (`device.emit`), from their array
#: inventories, NOT measured: source program per plane-chunk row (float32/float64
#: positions, int64 lattice/bucket/brick/key arrays), destination program per padded
#: row (concatenated window, gather/sort indices, float64 velocities and codes).
EMIT_SOURCE_B_PER_ROW = 150
EMIT_DEST_B_PER_ROW = 200
EMIT_KEPT_B_PER_ROW = 23  # key int64 + off uint8 x 3 + v float32 x 3, per live source row


def ic_device_stages(n, n_gpus=4, fdtype=np.float32, nb=None, slab=32, window=1, png=False,
                     emission="cards", planes_per_call=4):
    """Host, per-card and disk bytes at each stage of `icgen.generate_t9_slabs_device`.

    Returns `(host, card, disk)`: dicts of stage -> bytes. Within a stage the
    co-resident terms are summed; stages do not coexist, so a generation's peak is
    the max over stages. Mirrors the generator's allocation order (its docstring
    and stage comments). A LOWER BOUND: numpy and XLA temporaries at or below
    pencil/plane size are charged roughly and the page cache not at all.
    `png` charges the f_NL != 0 phi round trip in stage 1. `emission` prices stage 6
    for `device.emit` ("cards") or `icgen._emit_t9_slabs` ("host").
    """
    n, W = int(n), max(1, int(n_gpus))
    w = np.dtype(fdtype).itemsize
    m = n // 2 + 1
    field = n**3 * w
    spec = n * n * m * 2 * w
    pencil_host = 2 * W * n * m * 2 * w  # a block and its result, per card thread
    slab_real = int(slab) * n * n * w
    nb = int(nb) if nb else max(1, n // 16)
    rows_slab = n**3 // nb
    chunk_rows = min(int(slab), n // nb) * n * n
    emission_b = ((2 * window + 1) * rows_slab * 35  # staged keys, offsets, float64 velocities
                  + rows_slab * 80                   # a destination slab's finalize copies
                  + chunk_rows * 80)                 # one chunk's float64 positions/velocities
    host = {
        "1 noise -> delta spectrum": (spec + field if png else spec) + pencil_host,
        "2 2LPT source (accumulated on the cards)": 2 * spec + pencil_host,
        "3 source forward": 3 * spec + pencil_host,
        "4 velocities (to disk)": 3 * spec + pencil_host + slab_real,
        "5 displacements (x on the cards, y/z host)": (max(3 * spec, 2 * spec + field,
                                                           spec + 2 * field)
                                                       + pencil_host + slab_real),
        "6 emission": 2 * field + emission_b,
    }
    quarter = -(-n // W) * n * n * w
    work = 6 * n * m * 2 * w + 4 * n * n * 2 * w  # a pencil block's program + a plane's, rough
    card = {
        "1 noise -> delta spectrum": work,
        "2 2LPT source (accumulated on the cards)": quarter + work,
        "3 source forward": quarter + work,
        "4 velocities (to disk)": work,
        "5 displacements (x on the cards, y/z host)": quarter + work,
        "6 emission": quarter,
    }
    if emission == "cards":
        p = n // nb
        c = max(1, min(int(planes_per_call), p))
        halo = (-(-nb // W) + 2 * window) * p * n * n * w
        rows_src = p * n * n
        cap = int(rows_src * 1.06)  # the capacity ladder's padding, ~one rung
        per3 = (n // 2 // nb) ** 3  # bucket_cells 2: n / 2 buckets per side
        host["6 emission"] = (2 * field
                              + W * c * n * n * 5 * w                  # u_y, u_z, v uploads
                              + W * (cap * 9 + nb * nb * per3 * 8))  # D2H off/w + occupancy
        halo5 = 2 * window * p * n * n * w
        card["5 displacements (x on the cards, y/z host)"] = quarter + halo5 + work
        card["6 emission"] = (halo
                              + (2 * window + 1) * rows_src * EMIT_KEPT_B_PER_ROW
                              + max(c * n * n * EMIT_SOURCE_B_PER_ROW
                                    + rows_src * EMIT_KEPT_B_PER_ROW,  # chunk concat
                                    cap * EMIT_DEST_B_PER_ROW))
    elif emission != "host":
        raise ValueError(f"emission must be 'cards' or 'host', got {emission!r}")
    disk = {"velocity staging (3 fields)": 3 * field, "T9 slabs written": 9 * n**3}
    return host, card, disk


def device_budget(ec, *, n, n_gpus, row_bytes=9, paint_chunk_bricks=None, fused=True):
    """Per-GPU bytes for the host-state / device-step design.

    Returns `(resident, transient, phases)` in the shape the CPU column uses, so
    the same sum-within-a-phase / max-across discipline applies: glibc's arena
    behaviour is not the reason on a device, but XLA does not return a buffer to
    the pool between phases either, and understating a phase is how a run that
    does not fit gets a FITS.

    `paint_chunk_bricks` is the device coarse paint's chunk length in bricks;
    None is `device.paint.default_chunk_bricks`. `fused` prices the fused migrate +
    repack (the engine's default on this lane): its destination census runs one
    slab's eject kernel inside the tile loop, and after the loop one pass replaces
    the two.
    """
    from .engine import ONCE_PER_RUN_PHASES

    mesh = ec.mesh_bytes()
    phase_of = ec.mesh_phase()
    nc = int(ec.n_coarse)
    halo = shard_halo_planes()
    resident, transient, phases, host_mesh = {}, {}, {}, {}
    for k, v in mesh.items():
        where = DEVICE_PLACEMENT.get(k)
        if where is None:
            raise KeyError(
                f"mesh term {k!r} has no entry in DEVICE_PLACEMENT. A new term "
                "must be placed deliberately: defaulting it to either side is "
                "how an omitted term becomes a budget that cannot be traded "
                "against, which is what this module exists to prevent.")
        if where == "gone":
            continue
        if where == "host":
            # NOT skipped -- handed back so the HOST table charges it. Dropping
            # it here would put the term in neither column, which is precisely
            # the omission the KeyError above exists to prevent, arriving by a
            # different door: the device table would look smaller and nothing
            # would look bigger.
            host_mesh[k] = int(v)
            continue
        # a shard is its 1/n_gpus of the x axis plus its ghost planes; every
        # sharded term is whole x-planes, so v // nc is one plane's bytes exactly
        b = (int(v / n_gpus) + (int(v) // nc) * halo.get(k, 0) if where == "shard"
             else int(v))
        p = phase_of[k]
        if p == "resident":
            resident[k] = b
        else:
            transient[k] = b
        for one in (p,) if isinstance(p, str) else p:
            if one != "resident":
                phases[one] = phases.get(one, 0) + b

    # The two terms the CPU model has no name for, because on the CPU path the
    # state IS the working set and nothing streams.
    slabs = device_window_slabs(ec)
    nb = max(1, ec.n_fine // ec.n_brick)
    resident["slab_window (state rows a tile plane needs)"] = int(
        slabs * (n / nb) * row_bytes)
    resident["stream_chunks_in_flight"] = int(
        STREAM_CHUNKS_IN_FLIGHT * STREAM_CHUNK_BYTES)

    # The device coarse paint's own working set. The host decode it replaces is
    # "gone" in DEVICE_PLACEMENT, and until this line nothing charged the card
    # for doing that work instead.
    from .device.paint import default_chunk_bricks

    chunk_len = (default_chunk_bricks(nb) if paint_chunk_bricks is None
                 else int(paint_chunk_bricks))
    chunk_rows = n / nb**3 * chunk_len * PAINT_PAD_BOUND
    paint_b = int(chunk_rows * (PAINT_CHUNK_B_PER_ROW + PAINT_WINDOW_B_PER_ROW))
    key = f"coarse_paint_chunk ({chunk_len} bricks, decoded + painted)"
    transient[key] = paint_b
    phases["coarse_paint"] = phases.get("coarse_paint", 0) + paint_b

    slab_rows = n / nb
    if fused:
        # the census: one padded slab through the eject kernel, per card, while
        # the tile loop's window and workspace are live
        from .device.migrate import EJECT_B_PER_PADDED_ROW
        from .eject_jax import _padded

        census_b = int(EJECT_B_PER_PADDED_ROW * _padded(int(slab_rows)))
        transient["census_eject (fused pass, one padded slab)"] = census_b
        phases["tile_loop"] = phases.get("tile_loop", 0) + census_b

    in_step = sum(v for k, v in phases.items() if k not in ONCE_PER_RUN_PHASES)
    once = max((v for k, v in phases.items() if k in ONCE_PER_RUN_PHASES),
               default=0)
    # AFTER THE TILE LOOP, and deliberately NOT summed into the in-step phases:
    # the migrate and the repack run after the paint, the solve and the tile
    # loop have released their transients (JC, record sec. 29: the envelope is
    # the card less the RESIDENT terms, ~112 GB at c-hero). They are held apart
    # so the summed convention above is unchanged and the second verdict is
    # read against `resident` alone. The two do not coexist either: max.
    if fused:
        after_loop = {
            "migrate_repack_fused_pass (257 B/slab row, sec. 41, 1024^3)": int(
                FUSED_DEVICE_B_PER_SLAB_ROW * slab_rows),
        }
    else:
        after_loop = {
            "migrate_device_pass (320 B/slab row, sec. 31)": int(
                MIGRATE_DEVICE_B_PER_SLAB_ROW * slab_rows),
            "repack_device_pass (70 B/slab row, sec. 33)": int(
                REPACK_DEVICE_B_PER_SLAB_ROW * slab_rows),
        }
    return resident, transient, phases, max(in_step, once), slabs, host_mesh, after_loop


def _device_main(args, ec, t9, n, rows, arena, state):
    """The host / per-GPU split for the host-state / device-step design.

    Two columns, two budgets, two verdicts. The host column is the state and
    nothing else that scales with N; the per-GPU column is the sharded coarse
    mesh, one card's tile workspace, and the window of state a tile plane needs.
    """
    dev_args = argparse.Namespace(**vars(args))
    # ONE process per GPU. Every fine-arm term in `mesh_bytes` is multiplied by
    # `tile_workers`, so pricing this column at the CPU lane's worker count
    # would charge each card W tile workspaces it does not allocate.
    dev_args.workers = 1
    ec_dev, _ = build(dev_args)
    n_gpus = max(1, int(args.n_gpus))

    resident, transient, phases, worst_phase, slabs, host_mesh, after_loop = device_budget(
        ec_dev, n=n, n_gpus=n_gpus, paint_chunk_bricks=args.paint_chunk_bricks,
        fused=not getattr(args, "separate_passes", False))

    step = ec_dev.step_bytes(n, n_rows=rows, cap=args.cap)
    host = dict(state)
    for k, v in step.items():
        if v:
            host[f"{k} (host window of the device pass)"] = v
    # The mesh terms the design puts in HOST memory ON PURPOSE -- the factorized
    # coarse solve's spectrum and its per-component work buffer. They are the
    # design working as intended, not a port debt like the two above, so they
    # are labelled as such rather than lumped in with them.
    for k, v in host_mesh.items():
        if v:
            host[f"{k} (host by design)"] = v
    # both repack paths build the new per-bucket occupancy on the host and copy it
    # in at the end: a second bucket index, live beside the first
    host["repack new_occ (a second bucket index)"] = int(state["bucket_index"])
    # the windowed tile loop writes each core slab back from its card as one ladder
    # of `w` rows, every card at once. It downloaded the WHOLE window's `w` and staged
    # the window as numpy before gb 1002020 (~316 GB over four cards at 4096^3, the
    # overrun that killed it); the residents it still gathers are O(arena).
    from .forces import capacity_shape

    nb_dev = max(1, ec.n_fine // ec.n_brick)
    host["tile_window write-back (one slab of w per card)"] = int(
        n_gpus * int(capacity_shape(max(1, int(n / nb_dev)))) * 3 * np.dtype(np.int16).itemsize)
    _table("HOST: the state, plus the per-step terms nothing has moved yet", host)
    print(f"  state alone: {sum(state.values()) / n:6.2f} B/p")
    print("  `migrate_staging` and `repack_scratch` are the HOST windows the device "
          "migrate\n  and repack build per slab (two slab-sized numpy buffers each, "
          "from the code);\n  their device-side terms are in the AFTER-THE-LOOP "
          "table below.")

    _table(f"PER GPU (of {n_gpus}), resident through the tile loop", resident)
    if transient:
        _table(f"PER GPU (of {n_gpus}), transient (peak while that phase runs)",
               transient)
    _table(f"PER GPU (of {n_gpus}), BY PHASE", phases,
           total_label="sum of ALL phases listed")
    _table(f"PER GPU (of {n_gpus}), AFTER THE TILE LOOP (migrate, then repack; "
           "not co-resident with the phases above, JC sec. 29)", after_loop,
           total_label="max (the two do not coexist)", reduce=max)
    print(f"  the verdict below charges {_fmt(worst_phase).strip()}, which is "
          "max(the in-step\n  phases summed, the largest once-per-run phase) -- "
          "`kernel_build` runs before\n  the loop and is never co-resident with "
          "it, so adding it would overcharge.")
    print(f"  the slab window is {slabs} x-slabs, DERIVED from "
          f"`layout.brick_span`:\n  a tile draws from that many bricks per side, "
          "so walking tiles in x-order\n  needs that many consecutive slabs live. "
          "It is not a tuning knob.")

    load_peak = _print_load_and_ic(args, ec, t9, n, rows, arena, shared=True)

    # the generator runs float32 fields (the driver's GEN_FDTYPE), whatever the mesh dtypes
    # `n` here is the particle COUNT; the stage table wants particles per side
    ic_host, ic_card, ic_disk = ic_device_stages(
        args.n_part, n_gpus=n_gpus, fdtype=np.float32, nb=max(1, ec.n_fine // ec.n_brick))
    _table("IC GENERATION ON THE CARDS (its own job), HOST by stage", ic_host,
           total_label="PEAK (max, not sum)", reduce=max)
    _table(f"IC GENERATION ON THE CARDS, PER GPU (of {n_gpus}) by stage", ic_card,
           total_label="PEAK (max, not sum)", reduce=max)
    _table("IC GENERATION, DISK", ic_disk)
    for label, peak, budget in (("host", max(ic_host.values()), args.host_gb),
                                ("per GPU", max(ic_card.values()), args.device_gb)):
        if budget is not None:
            r = peak / (budget * GB)
            print(f"  IC generation {label} against {budget} GB: "
                  f"{'FITS' if r < 1.0 else 'DOES NOT FIT'} ({r:.2f}x)")

    print("\nBINDING TERMS")
    host_peak = sum(host.values())
    dev_peak = sum(resident.values()) + worst_phase
    print(f"  host, a lower bound on the run's peak: {_fmt(host_peak)}")
    print(f"  the LOAD stage peaks at:               {_fmt(load_peak)}"
          f"   {'<- BINDING' if load_peak > host_peak else ''}")
    print(f"  per GPU, resident + worst phase:       {_fmt(dev_peak)}")
    if args.host_gb is not None:
        r = host_peak / (args.host_gb * GB)
        print(f"  against --host-gb {args.host_gb}: "
              f"{'FITS' if r < 1.0 else 'DOES NOT FIT'} ({r:.2f}x)")
        rl = load_peak / (args.host_gb * GB)
        print(f"    and the LOAD stage:               "
              f"{'FITS' if rl < 1.0 else 'DOES NOT FIT'} ({rl:.2f}x)")
    else:
        print("  no --host-gb given, so no host verdict (a Vista gb node is 1026)")
    after_peak = sum(resident.values()) + max(after_loop.values())
    print(f"  per GPU, resident + after-the-loop:    {_fmt(after_peak)}")
    if args.device_gb is not None:
        r = dev_peak / (args.device_gb * GB)
        print(f"  against --device-gb {args.device_gb} per card: "
              f"{'FITS' if r < 1.0 else 'DOES NOT FIT'} ({r:.2f}x)")
        ra = after_peak / (args.device_gb * GB)
        print(f"    and after the tile loop:          "
              f"{'FITS' if ra < 1.0 else 'DOES NOT FIT'} ({ra:.2f}x)")
    else:
        print("  no --device-gb given, so no per-card verdict (a GB200 detected "
              "185 GiB = 199 GB)")
    print("\n  NB the same LOWER BOUND caveat as the CPU column, and two more "
          "that are\n  specific to this one. (1) DEVICE_PLACEMENT is still a "
          "design assertion\n  rather than a reading of the code, but the "
          "executor it asserts now EXISTS\n  and is bitwise the host engine at "
          "cgh64 on four GB200s (996685, 996857),\n  so the placements are "
          "checked against something. What is NOT checked is\n  this table at "
          "4096^3 shapes: that is the D7 smoke.\n  (2) The four-way split is "
          "charged as an exact quarter, which is right for\n  MEMORY -- each "
          "card holds its quarter however the wall splits -- and wrong\n  for "
          "WALL, where D5 measured 2.8-3.0x rather than 4x. Do not read a "
          "per-card\n  byte here as licence for a quarter of a second anywhere.")
    return 0


def build(args):
    from .codec import T9Layout
    from .engine import EngineConfig

    # `--workers` reaches the ENGINE CONFIG, not just the shm table. Every fine-arm
    # term is per worker and this was pricing one of each, which is how the tile
    # loop read 1.4 GB in the budget for the phase that killed 923313.
    ec = EngineConfig(
        box_size=args.box, n_part=args.n_part, n_fine=args.n_fine,
        n_coarse=args.n_coarse, n_tile=args.tile, b_fine=args.buf,
        coarse_dtype=args.coarse_dtype, fine_dtype=args.fine_dtype,
        tile_workers=max(int(args.workers), 1) if args.workers else 1,
        eject_kernel=args.eject_kernel,
        migrate_eject_inflight=args.eject_inflight,
        migrate_backend="device" if getattr(args, "backend", "cpu") == "device" else "host",
    )
    t9 = T9Layout(box_size=args.box, n_part=args.n_part, bucket_cells=args.bucket_cells)
    return ec, t9


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Memory anatomy of an inexor configuration, and whether it fits.",
    )
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    # The CPU column is unchanged and stays the default: the pooled host engine
    # remains a supported backend and its numbers must not move when this flag
    # is added. `device` prices the host-state / device-step design instead --
    # the host holds only the state, the GPUs hold the mesh and a slab window.
    ap.add_argument("--backend", choices=("cpu", "device"), default="cpu",
                    help="which engine to price. `device` = the host is a byte "
                         "store and the GPUs do the step (Vista gb).")
    ap.add_argument("--n-gpus", type=int, default=4,
                    help="accelerators the coarse mesh is sharded across, for "
                         "--backend device. A Vista gb node has 4.")
    ap.add_argument("--separate-passes", action="store_true",
                    help="for --backend device: price the migrate and repack as two "
                         "passes (no census) instead of the fused pass")
    ap.add_argument("--paint-chunk-bricks", type=int, default=None,
                    help="for --backend device: bricks per coarse-paint chunk on a "
                         "card. Default a quarter of an x-slab of bricks; smaller "
                         "trades card memory for more device launches.")
    ap.add_argument("--n-part", type=int, default=None, help="particles per side")
    ap.add_argument("--box", type=float, default=None, help="box size, Mpc/h")
    ap.add_argument("--n-fine", type=int, default=None)
    ap.add_argument("--n-coarse", type=int, default=None)
    ap.add_argument("--tile", type=int, default=None)
    # None, not 32: the preset fill below only writes a field that is still None,
    # so a non-None default here silently OUTRANKS the preset table. Every preset
    # but `smoke` carries buf=32, which is why the shadowing was invisible -- and
    # `smoke` (buf=8) was left unevaluable, T=16 + 2*32 = 80 against a 64 mesh
    # tripping the degeneracy guard. Same class as a decayed default at a call
    # site that omits the argument.
    ap.add_argument("--buf", type=int, default=None)
    ap.add_argument("--coarse-dtype", default="float32")
    ap.add_argument("--fine-dtype", default="float64")
    ap.add_argument("--bucket-cells", type=int, default=2)
    ap.add_argument("--slack", type=float, default=0.10, help="per-brick spare fraction")
    ap.add_argument("--alloc-margin", type=float, default=0.10)
    ap.add_argument("--arena-frac", type=float, default=0.01,
                    help="arena rows as a fraction of N. NOTE the default is "
                         "`SlotState.build`'s, but `scripts/v2_m6_realization.py` "
                         "runs 0.20, which is 28 GB of shared memory at c-gh.")
    # No default, and deliberately not derived: geometry alone UNDERSTATES it.
    # `cap` is the max over tiles of the padded per-tile row count, so it carries
    # (P/T)^3 x N/tiles, the geometric ladder's <=26.0%, AND the clustering spread
    # of per-tile occupancy -- and that last part is what geometry cannot give.
    # Measured excess over (P/T)^3 x N/tiles is 1.290x at cdev and 1.587x at cdev8,
    # shrinking as per-tile occupancy grows, so a derived cap would look like a
    # measurement and read low.
    ap.add_argument("--cap", type=int, default=None,
                    help="measured per-tile capacity, for the tile_buffers term. "
                         "cdev/cgh64/C-gh share N/tile and P, so cdev's measured "
                         "5284492 is the anchor for all three.")
    # BUDGETS ARE ARGUMENTS. No default host size: a wrong default is worse than
    # an absent one, because it silently makes a verdict up.
    ap.add_argument("--host-gb", type=float, default=None,
                    help="host RAM budget. A property of YOUR machine: Vista gh "
                         "~116 (a hard cliff), Vista gg 237, S3 h100 ~1007.")
    ap.add_argument("--device-gb", type=float, default=None, help="accelerator HBM budget")
    # A SEPARATE budget, and the reason this argument exists at all: the pool
    # holds the state in POSIX shared memory, which is a tmpfs sized by default
    # at HALF the node's RAM. This table said "FITS (0.74x)" for the c-gh run
    # that then died in pool construction (Vista 920910), because the host
    # total it checked was never the budget that bound. Read your node's with
    # `df -B1 /dev/shm`; a Vista gg node measures 127.6 GB (job 922332).
    ap.add_argument("--shm-gb", type=float, default=None,
                    help="/dev/shm budget, for the pooled (tile_workers>1) lane. "
                         "Vista gg measures 127.6. Kernel default is half of RAM.")
    ap.add_argument("--workers", type=int, default=None,
                    help="tile_workers. >1 (or unset) means the pooled lane: the "
                         "loader writes into shared memory and the state exists once")
    ap.add_argument("--eject-kernel", default="jax", choices=("numpy", "jax"),
                    help="which eject the migrate runs. MEASURED at 129 B per "
                         "slab-row for jax against 35 for numpy, and the pooled "
                         "path holds one slab PER WORKER -- so at c-gh W=8 this "
                         "is the difference between 69 GB and 19. Both kernels "
                         "are bitwise (record 5s); numpy ejects 1.6-1.9x slower")
    ap.add_argument("--eject-inflight", type=int, default=None,
                    help="cap on EJECTS running at once, separately from "
                         "--workers. An eject is 129 B per slab-row (jax) and "
                         "an insert 50, both measured, so bounding this trades "
                         "migrate memory for migrate wall while the tile loop "
                         "keeps its worker count")
    ap.add_argument("--slabs", type=int, default=128,
                    help="T9 slab files the ICs were written as (c-gh: 128)")
    ap.add_argument("--disk-gb", type=float, default=None, help="scratch budget for IC staging")
    args = ap.parse_args(argv)

    if args.preset:
        for k, v in PRESETS[args.preset].items():
            if getattr(args, k.replace("-", "_")) is None:
                setattr(args, k.replace("-", "_"), v)
    if args.buf is None:
        args.buf = 32
    missing = [k for k in ("n_part", "box", "n_fine", "n_coarse", "tile")
               if getattr(args, k) is None]
    if missing:
        ap.error(f"need {', '.join(missing)} (or --preset)")

    ec, t9 = build(args)
    n = args.n_part**3
    print(f"config: n_part={args.n_part}^3 = {n:,} particles, box={args.box} Mpc/h, "
          f"n_fine={args.n_fine}, n_coarse={args.n_coarse}, T={args.tile}, b={args.buf}")
    print(f"        fine cell {ec.fine_cell:.4f} Mpc/h, particle spacing "
          f"{args.box / args.n_part:.4f} Mpc/h, {len(ec.tiles)} tiles, "
          f"coarse {ec.coarse_dtype} / fine {ec.fine_dtype}")

    # ---- state, the thing the architecture is about
    rows = int(np.ceil(np.ceil(n * (1.0 + args.slack)) * (1.0 + args.alloc_margin)))
    arena = int(np.ceil(n * args.arena_frac))
    state = {
        "t9_payload (9 B/p)": n * 9,
        "slack + alloc_margin": (rows - n) * 9,
        "bucket_index": t9.index_bytes(),
        # The arena is n_arena EXTRA ROWS of off/w (`state.SlotState`: off is
        # (n_alloc + n_arena, 3)), and this table priced only `arena_bucket`,
        # its int64 side. At the realization's `--arena-frac 0.20` that is
        # 15.5 GB of payload the total did not carry -- the omitted-term fault
        # this module's docstring is about, in this module.
        "arena rows in off/w": arena * 9,
        "arena_bucket": arena * 8,
        "brick_start": (ec.n_brick and (args.n_fine // ec.n_brick) ** 3 + 1) * 8,
        # THE ARRAY THAT REPLACED `kick_pending`. One f64 per brick, resident,
        # against 32 B per PARTICLE held per-step: 16.8 MB against 274.9 GB at
        # C-gh. It is listed rather than folded into the noise because the term
        # it replaced was the binding one, and a reader comparing this table to
        # an older card needs to see where that went.
        "brick_scales": (ec.n_brick and (args.n_fine // ec.n_brick) ** 3) * 8,
    }
    _table("STATE (resident for the whole run)", state)
    print(f"  {'':<20}  {sum(state.values()) / n:6.2f} B/p")

    # The tables below price a host that holds the coarse mesh and runs the tile
    # loop. The device design's host does neither, so it gets its own two
    # columns rather than a footnote on these.
    if args.backend == "device":
        return _device_main(args, ec, t9, n, rows, arena, state)

    mesh = ec.mesh_bytes()
    # The split is `engine.MESH_PHASE`'s, not this module's: a term's phase is a
    # property of the code that allocates it, so the accounting and the engine
    # cannot drift apart the way they did over `coarse_dtype`.
    from .engine import STEP_PHASE

    # `ec.mesh_phase()`, not the module dict: in pool mode the tile kernel build
    # moves INTO the tile loop, because the workers do it and the parent does not
    MESH_PHASE = ec.mesh_phase()
    resident = {k: v for k, v in mesh.items() if MESH_PHASE[k] == "resident"}
    transient = {k: v for k, v in mesh.items() if k not in resident}
    _table("MESH, resident through the tile loop", resident)
    _table("MESH, transient (peak while that phase runs)", transient)

    step = ec.step_bytes(n, n_rows=rows, cap=args.cap)
    _table("PER-STEP HOST TERMS THAT SCALE WITH PARTICLES", step)

    # PHASES, which is the line the C-gh budget was missing. Summing every
    # transient overstates (the tile loop does not run during the coarse solve);
    # taking the largest single one understates, and understated is how a run
    # that does not fit gets a FITS. Terms in a phase are co-resident by
    # construction, so sum within and max across.
    phases = {}
    for src, phase_of in ((mesh, MESH_PHASE), (step, STEP_PHASE)):
        for k, v in src.items():
            p = phase_of[k]
            # a term may name several phases: it is charged to each, because
            # what a phase's line answers is "how much is live while this runs"
            for one in (p,) if isinstance(p, str) else p:
                if one != "resident":
                    phases[one] = phases.get(one, 0) + v
    _table("BY PHASE (transients summed within, because they ARE co-resident)",
           phases, total_label="sum of the in-step phases", reduce=sum)
    print("  the total is a SUM over the phases INSIDE a step and excludes "
          "kernel_build,\n  which runs once before the loop. Summing rather than "
          "maxing is deliberate:\n  glibc does not return freed arenas between "
          "phases, so a step's high-water\n  accumulates -- `malloc_trim` recovered "
          "12-41% of a run's peak here.")

    # ---- the sub-budget, for the pooled lane only
    if args.workers is None or args.workers > 1:
        from .executor import shm_terms

        shm = shm_terms(
            n_rows=rows + arena, index_bytes=t9.index_bytes(), n_arena=arena,
            n_bricks=ec.n_brick and (args.n_fine // ec.n_brick) ** 3,
            n_coarse=args.n_coarse,
            coarse_itemsize=np.dtype(args.coarse_dtype).itemsize,
        )
        _table("SHARED MEMORY (/dev/shm), the pooled lane's sub-budget", shm)
        print("  the migrate scratch is staged per step at a (K, R) this table "
              "cannot know;\n  `TilePool` holds construction to a 1.10x margin "
              "for it")
        if args.shm_gb is not None:
            sb = args.shm_gb * GB
            r = sum(shm.values()) / sb
            print(f"  against --shm-gb {args.shm_gb}: "
                  f"{'FITS' if r < 1.0 else 'DOES NOT FIT'} ({r:.2f}x the budget)")
        else:
            print("  no --shm-gb given, so no verdict. This is NOT the host "
                  "budget above: it is a\n  tmpfs, by default half the node's "
                  "RAM, and it bound the c-gh run that the host\n  line called "
                  "a fit. `df -B1 /dev/shm` on the node you will run on.")

    # ---- the load path, which is where two jobs actually died
    load_peak = _print_load_and_ic(
        args, ec, t9, n, rows, arena,
        shared=(args.workers is None or args.workers > 1))

    # ---- the verdict, with the binding term NAMED
    print("\nBINDING TERMS")
    # THE WORST PHASE, not the largest single transient. The old form was
    #     sum(state) + sum(resident) + max(transient) + sum(step)
    # which charged 12.9 GB of a 56 GB C-gh mesh transient while summing the
    # per-step host terms as though repack and migrate ran at the same instant.
    # It said FITS (0.75x) for the run that OOM-killed in its coarse solve, and
    # the phase it under-charged is the one that died: Vista 923139 lost ~79 GB
    # of MemAvailable inside it, against 12.9 charged.
    from .engine import ONCE_PER_RUN_PHASES

    in_step = sum(v for k, v in phases.items() if k not in ONCE_PER_RUN_PHASES)
    once = max((v for k, v in phases.items() if k in ONCE_PER_RUN_PHASES),
               default=0)
    worst_phase = max(in_step, once)
    # THE WORKERS ARE PROCESSES AND THIS TABLE ONLY EVER PRICED ONE. Measured
    # 1.12 GB each on a Vista gg node (8 workers, 9.0 GB total, read from
    # /proc/<pid>/statm AFTER a barrier task -- `ctx.Pool()` returns before
    # `_worker_init` has imported jax, and the first version of that probe
    # reported 0.01 GB each and then watched them grow). It is a FLOOR: the
    # reading is at pool startup, before any tile has been forced, and it is the
    # measurement this budget most needs repeating at W=16.
    n_workers = 0 if args.workers is None else max(0, int(args.workers))
    workers_b = int(n_workers * WORKER_STARTUP_BYTES) if n_workers > 1 else 0
    peak_est = (sum(state.values()) + sum(resident.values()) + worst_phase
                + workers_b)
    # TRANSIENTS ARE CANDIDATES. They were excluded here, so the line could not
    # name a transient however large -- at cdev it reported `tile_kernels` (0.791
    # GB) while `tile_workspace` (1.443) was bigger and the phase MEASURED to set
    # the peak is the tile force (job 446: `tile_short` increments 3.432 GB).
    # A "largest term" line that structurally cannot name the measured winner is
    # the shape of a gate that cannot fail.
    biggest = max(list(state.items()) + list(resident.items()) + list(step.items())
                  + [(f"{k} (transient)", v) for k, v in transient.items()],
                  key=lambda kv: kv[1])
    if workers_b:
        print(f"  {n_workers} pool workers at {_fmt(WORKER_STARTUP_BYTES)} each "
              f"(measured, gg): {_fmt(workers_b)}")
    print(f"  a lower bound on the run's peak: {_fmt(peak_est)}")
    print(f"  the LOAD stage peaks at:          {_fmt(load_peak)}"
          f"   {'<- BINDING' if load_peak > peak_est else ''}")
    print(f"  largest single term: {biggest[0]} at {_fmt(biggest[1])}")
    if "tile_buffers" not in step:
        print("  NB `tile_buffers` is NOT in the total above: it needs a measured "
              "`cap`,\n     and it is the per-tile host set that sits inside the phase "
              "measured to\n     SET the peak at cdev. Pass --cap to include it "
              "(cdev measured 5,284,492).")
    if args.host_gb is not None:
        budget = args.host_gb * GB
        ratio = peak_est / budget
        verdict = "FITS" if ratio < 1.0 else "DOES NOT FIT"
        print(f"  against --host-gb {args.host_gb}: {verdict} "
              f"({ratio:.2f}x the budget)")
    else:
        print("  no --host-gb given, so no verdict: pass your machine's host RAM")
    if args.device_gb is not None:
        dev = sum(resident.values()) + max(transient.values())
        print(f"  device-resident mesh against --device-gb {args.device_gb}: "
              f"{dev / (args.device_gb * GB):.2f}x")
    print("\n  NB this is a LOWER BOUND from arithmetic, not a measurement. It "
          "charges every\n  in-step phase, not the largest single term -- the largest-"
          "term form is what\n  called the c-gh run a fit twice. What it still cannot "
          "see is XLA's intra-jit\n  scratch, which is invisible to tracemalloc, to "
          "`live_arrays` and to\n  `memory_stats()` alike on CPU. Vista 923139 lost "
          "~79 GB inside a phase this\n  prices at 30, so treat it as a sizing floor "
          "and never as a peak.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
