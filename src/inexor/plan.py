"""Will this configuration fit this machine? `python -m inexor.plan`.

A front end over the existing byte accounting: `EngineConfig.mesh_bytes` / `.step_bytes`,
`T9Layout.index_bytes`, `ooc_fft.plan_bytes`, plus the T9 payload. Budgets (host, device,
/dev/shm, disk) are arguments; the package holds no machine sizes. This is arithmetic over a
config, not a measurement, and it reports lower bounds where the accounting is incomplete.

Examples:
  python -m inexor.plan --preset cdev
  python -m inexor.plan --preset c-gh --host-gb 116
  python -m inexor.plan --n-part 2048 --box 1024 --n-fine 4096 --n-coarse 1024 \
      --tile 256 --buf 32 --host-gb 237 --disk-gb 2000
"""

import argparse
import sys

import numpy as np

# Preset geometries. The fine cell is fixed across the ladder; presets differ in volume.
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

# One pool worker's own footprint (interpreter + jax; the state is shared and priced
# elsewhere), measured at pool startup with 8 workers. A floor: no tile had yet been forced.
WORKER_STARTUP_BYTES = 1.12 * GB

# The production knobs, in one place, so the planner and the driver price the same run
# (`EngineConfig` defaults coarse_dtype to float64 for the reference arms). Reference
# comparisons deliberately do not use this: they vary these knobs.
RATIFIED = dict(
    coarse_dtype="float32",   # f32 coarse mesh: 1.83x lower peak host, ~1e-5 error
    fine_dtype="float64",
    alpha=1.0,                # r_s / coarse_cell
)
BUCKET_CELLS = 2


def engine_config(preset, **overrides):
    """The `EngineConfig` the production driver builds: preset geometry + `RATIFIED` knobs.

    Shared by the planner and `scripts/run/realization.py`. `overrides` are for per-invocation
    knobs (workers, slack, checkpointing), not for undoing `RATIFIED`.
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

    With `shared`, the payload is written straight into the pool's segments and exists once;
    without it the state is built privately and then copied (2x while copying). A slab is one
    x-slice of bricks: n/n_slabs rows of the 9 B T9 record plus its share of the index.
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
# The device design: the host is a byte store, the GPUs do the step.
# ---------------------------------------------------------------------------
#
# Where each mesh term lives on the device lane (an assertion about the design, kept here
# rather than read from the device code):
#   shard   -- split along x across the GPUs (1/n_gpus each, plus ghost planes).
#   replica -- every card holds its own copy (each runs whole tiles).
#   host    -- held in host memory, on no card (the factorized solve's spectrum).
#   gone    -- a host pass the device lane does not have.
DEVICE_PLACEMENT = {
    "coarse_delta": "shard",
    "coarse_force_resident": "shard",
    "coarse_force_copy_transient": "shard",
    # `forces.coarse_kernel_parts` builds these in numpy and `ooc_fft.ArrayKernel.blocks`
    # slices them on the host per pencil block; only the block's product reaches a card.
    "coarse_kernel_pref": "host",
    "coarse_match_factor": "host",
    "coarse_kernel_build_f64": "host",
    # The factorized solve keeps the spectrum in numpy and ships planes; each card holds only
    # its own planes in flight.
    "coarse_spectrum": "host",
    "coarse_solve_work": "host",
    "coarse_kernel_slab": "host",
    "coarse_device_planes": "replica",
    # Priced at the host path's int64 width, an over-charge (the device sub-block paint
    # accumulates int32); deliberately not lowered.
    "coarse_accumulator": "shard",
    "tile_kernels": "replica",
    "tile_kernel_build_f64": "replica",
    "tile_kernel_pref": "replica",
    "tile_workspace": "replica",
    # Rows are decoded on the card from the streamed slab; no host decode.
    "coarse_decode_slab": "gone",
}


def shard_halo_planes():
    """x-planes a card's shard holds beyond its 1/n_gpus of the mesh, per term.

    Only for terms the device code builds with ghosts: the accumulator's
    `device.paint.ACC_GHOST_LO + ACC_GHOST_HI` and the force mesh's `COARSE_HALO` per side.
    """
    from .device.paint import ACC_GHOST_HI, ACC_GHOST_LO
    from .forces import COARSE_HALO

    return {"coarse_accumulator": ACC_GHOST_LO + ACC_GHOST_HI,
            "coarse_force_resident": 2 * COARSE_HALO}

# Streaming pinned host memory in 2 GiB chunks holds a device high-water of exactly two
# chunks, independent of the total streamed; the per-chunk `block_until_ready` is what bounds
# it (without it XLA keeps every staged copy alive).
STREAM_CHUNK_BYTES = 2.0 * 2**30
STREAM_CHUNKS_IN_FLIGHT = 2

# One coarse-paint chunk on a card (`device.paint`): measured device peak bytes per padded
# row, `off` window upload included. Jitted (default): 62-72 B, charged at the largest.
PAINT_CHUNK_B_PER_ROW = 72
# Eager (`jit=False`): 266-293 B.
PAINT_CHUNK_EAGER_B_PER_ROW = 266
# Folded into the rate above; kept at 0 because readers add the two.
PAINT_WINDOW_B_PER_ROW = 0
# Bytes alive at once in the traced (unfused) program; not a floor for the compiled one.
# `tests/test_device_paint.py` fails if a re-trace exceeds it, catching program growth
# without a card.
PAINT_CHUNK_TRACED_B_PER_ROW = 125
# `forces.capacity_shape`'s padding is <= 26.0%.
PAINT_PAD_BOUND = 1.26

# Measured device peaks per slab row, charged as slab rows x rate:
# `device.migrate.drift_and_migrate_device` (measured at cgh64; scaled by arithmetic).
MIGRATE_DEVICE_B_PER_SLAB_ROW = 320
# `device` repack, windows and program included (measured at a production-shape slab).
REPACK_DEVICE_B_PER_SLAB_ROW = 70
# `device.fused.migrate_repack_device` (measured at 1024^3, not production shape).
FUSED_DEVICE_B_PER_SLAB_ROW = 257

# Host bytes the device lane's tile loop holds above the loop's closing RSS, per card, beyond
# the priced window write-back: fixed + per slab row. MEASURED, not derived (the host side of
# card transfers, which the CPU backend does not allocate): 1.9 GB/card at cgh64 on 4 GB200
# (job 1027664), 2.0 at c-1024 on one GH200 (1029876, not in the fit), 10.5 at c-hero
# (1003657).
TILE_LOOP_HOST_FIXED_PER_CARD = 1.9 * GB
TILE_LOOP_HOST_B_PER_SLAB_ROW = 32
# The device-lane process's host floor beyond the priced state (CUDA context, jaxlib, XLA's
# host pools), keyed by cards per node. MEASURED: 4.265 / 4.281 GB on one GH200 at cgh64 /
# c-1024 (job 1029876, RSS after step 1 minus the priced resident; the smaller is taken, so
# the resident stays a lower bound); <= 3.2 GB on four GB200
# (the 32^3 smoke's host peak, job 1027664). Other card counts take the larger.
PROCESS_BASELINE_BY_CARDS = {1: 4.26 * GB, 4: 3.2 * GB}

# The lead drift's host transient above its closing RSS, per card per slab row, on several
# cards. MEASURED at c-hero on 4 GB200 only (68 GB, job 1003657); ~0 on one GH200 at cgh64
# and c-1024 (1029876). The single-card lane is charged nothing.
LEAD_DRIFT_HOST_B_PER_SLAB_ROW_PER_CARD = 63


def device_window_slabs(ec):
    """x-slabs of bricks that must be resident to serve one plane of tiles.

    Derived from `layout.brick_span` (the engine's membership function): walking tiles in x
    order needs that many consecutive x-slabs live. `choose_brick`'s `c | b_fine` contract
    makes the union exactly the padded tile box.
    """
    from .layout import brick_span

    nb = max(1, ec.n_fine // ec.n_brick)
    _pad, span = brick_span(ec.n_tile, ec._b_realized, ec.n_brick, nb)
    return min(span, nb)


def _print_load_and_ic(args, ec, t9, n, rows, arena, shared):
    """Print the load-path and IC-stage tables (host-side for both backends); return load peak."""
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


#: Card bytes per row of the `device.emit` programs, from their array inventories (not
#: measured): source per plane-chunk row, destination per padded row.
EMIT_SOURCE_B_PER_ROW = 150
EMIT_DEST_B_PER_ROW = 200
EMIT_KEPT_B_PER_ROW = 23  # key int64 + off uint8 x 3 + v float32 x 3, per live source row


def ic_device_stages(n, n_gpus=4, fdtype=np.float32, nb=None, slab=32, window=1, png=False,
                     emission="cards", planes_per_call=4):
    """Host, per-card and disk bytes at each stage of `icgen.generate_t9_slabs_device`.

    Returns `(host, card, disk)` dicts of stage -> bytes: summed within a stage, peak = max
    over stages, following the generator's allocation order. A lower bound (small temporaries
    charged roughly, page cache not at all). `png` charges the f_NL != 0 phi round trip;
    `emission` prices stage 6 for `device.emit` ("cards") or `icgen._emit_t9_slabs` ("host").
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

    Returns `(resident, transient, phases, worst_phase, slabs, host_mesh, after_loop)`, summed
    within a phase and maxed across as in the CPU column (XLA does not return buffers between
    phases). `host_mesh` holds "host"-placed terms for the host table; `after_loop` the
    post-tile-loop migrate/repack peaks, which do not coexist with the in-step phases.
    `paint_chunk_bricks` None = `device.paint.default_chunk_bricks`. `fused` prices the fused
    migrate + repack, whose census runs one slab's eject inside the tile loop.
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
            # handed back so the host table charges it
            host_mesh[k] = int(v)
            continue
        # 1/n_gpus of x plus ghost planes; sharded terms are whole x-planes, so v // nc is exact
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

    # Terms with no CPU counterpart: the streamed state window and chunks in flight.
    slabs = device_window_slabs(ec)
    nb = max(1, ec.n_fine // ec.n_brick)
    resident["slab_window (state rows a tile plane needs)"] = int(
        slabs * (n / nb) * row_bytes)
    resident["stream_chunks_in_flight"] = int(
        STREAM_CHUNKS_IN_FLIGHT * STREAM_CHUNK_BYTES)

    # The device coarse paint's working set (replaces the "gone" host decode).
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
        # census: one padded slab through the eject kernel, live during the tile loop
        from .device.migrate import EJECT_B_PER_PADDED_ROW
        from .eject_jax import _padded

        census_b = int(EJECT_B_PER_PADDED_ROW * _padded(int(slab_rows)))
        transient["census_eject (fused pass, one padded slab)"] = census_b
        phases["tile_loop"] = phases.get("tile_loop", 0) + census_b

    in_step = sum(v for k, v in phases.items() if k not in ONCE_PER_RUN_PHASES)
    once = max((v for k, v in phases.items() if k in ONCE_PER_RUN_PHASES),
               default=0)
    # Migrate and repack run after the in-step transients are released, so they are priced
    # against `resident` alone, not summed into the phases; they do not coexist (max).
    if fused:
        after_loop = {
            "migrate_repack_fused_pass (257 B/slab row, measured at 1024^3)": int(
                FUSED_DEVICE_B_PER_SLAB_ROW * slab_rows),
        }
    else:
        after_loop = {
            "migrate_device_pass (320 B/slab row)": int(
                MIGRATE_DEVICE_B_PER_SLAB_ROW * slab_rows),
            "repack_device_pass (70 B/slab row)": int(
                REPACK_DEVICE_B_PER_SLAB_ROW * slab_rows),
        }
    return resident, transient, phases, max(in_step, once), slabs, host_mesh, after_loop


def device_host_phases(ec, *, n, state, step, host_mesh, n_gpus, fused=True):
    """Host bytes of the device lane as `(resident, phases)`.

    `resident` is live for the whole run (the state and the host-held coarse kernel arrays);
    `phases` maps phase -> {term: bytes}, summed within a phase. The host peak is
    `sum(resident) + max(phase sums)`: the large numpy buffers are returned to the OS when
    freed, so phases do not accumulate (measured: RSS at phase starts differs by tens of GB).
    `state` and `step` are the planner's state table and `ec.step_bytes`; `host_mesh` the
    host-placed mesh terms from `device_budget`.
    """
    from .engine import STEP_PHASE
    from .forces import capacity_shape

    phase_of = ec.mesh_phase()
    resident, phases = dict(state), {}
    resident["process baseline (MEASURED, per card count)"] = int(PROCESS_BASELINE_BY_CARDS.get(
        int(n_gpus), max(PROCESS_BASELINE_BY_CARDS.values())))

    def add(phase, key, v):
        if v:
            phases.setdefault(phase, {})[key] = int(v)

    for k, v in host_mesh.items():
        p = phase_of[k]
        if p == "resident":
            resident[f"{k} (host-held)"] = int(v)
            continue
        for one in (p,) if isinstance(p, str) else p:
            add(one, f"{k} (host)", v)
    nb = max(1, ec.n_fine // ec.n_brick)
    slab_rows = n / nb
    for k, v in step.items():
        if k == "repack_scratch" and fused:
            continue  # the fused pass writes blocks straight back; its repack phase is empty
        add(STEP_PHASE[k], f"{k} (host window of the device pass)", v)
    # the per-bucket occupancy the repack builds beside the old one: inside the fused pass
    # (`device.fused`), or in the separate repack
    add("migrate" if fused else "repack", "repack new_occ (a second bucket index)",
        state["bucket_index"])
    # the windowed tile loop writes back one core slab of `w` per card at once
    add("tile_loop", "tile_window write-back (one slab of w per card)",
        n_gpus * int(capacity_shape(max(1, int(slab_rows)))) * 3 * np.dtype(np.int16).itemsize)
    add("tile_loop", "tile loop card transfers (MEASURED, per card)",
        n_gpus * (TILE_LOOP_HOST_FIXED_PER_CARD + TILE_LOOP_HOST_B_PER_SLAB_ROW * slab_rows))
    if n_gpus > 1:
        add("lead_drift", "lead drift card transfers (MEASURED at c-hero, per card)",
            n_gpus * LEAD_DRIFT_HOST_B_PER_SLAB_ROW_PER_CARD * slab_rows)
    # Before the first step the loader has written the n particle rows only; the slack rows
    # are first touched by migrate inserts, the lead drift's included (so crediting it the
    # whole slack keeps it a lower bound).
    for p in ("kernel_build", "lead_drift"):
        if p in phases:
            add(p, "slack rows not yet touched (credit)", -state["slack + alloc_margin"])
    return resident, phases


def _device_main(args, ec, t9, n, rows, arena, state):
    """The host / per-GPU split for the host-state / device-step design.

    The host column is the state plus host-side step windows; the per-GPU column is the
    sharded coarse mesh, one card's tile workspace and the state window a tile plane needs.
    """
    dev_args = argparse.Namespace(**vars(args))
    # One process per GPU: fine-arm terms scale with `tile_workers`, so price one.
    dev_args.workers = 1
    ec_dev, _ = build(dev_args)
    n_gpus = max(1, int(args.n_gpus))

    resident, transient, phases, worst_phase, slabs, host_mesh, after_loop = device_budget(
        ec_dev, n=n, n_gpus=n_gpus, paint_chunk_bricks=args.paint_chunk_bricks,
        fused=not getattr(args, "separate_passes", False))

    step = ec_dev.step_bytes(n, n_rows=rows, cap=args.cap)
    fused = not getattr(args, "separate_passes", False)
    host_res, host_ph = device_host_phases(ec, n=n, state=state, step=step,
                                           host_mesh=host_mesh, n_gpus=n_gpus, fused=fused)
    _table("HOST, resident for the whole run", host_res)
    print(f"  state alone: {sum(state.values()) / n:6.2f} B/p")
    _table("HOST, by phase (summed within a phase; phases do not coexist)",
           {f"{p}: {k}": v for p, terms in host_ph.items() for k, v in terms.items()},
           total_label="sum of ALL phases listed")
    host_phase_sums = {p: sum(t.values()) for p, t in host_ph.items()}
    worst_host = max(host_phase_sums, key=host_phase_sums.get)
    print(f"  worst phase: {worst_host} at {_fmt(host_phase_sums[worst_host]).strip()} "
          "above the resident")
    print("  `migrate_staging` and `repack_scratch` are the HOST windows the device "
          "migrate\n  and repack build per slab (from the code); MEASURED lines are the "
          "host side of\n  card transfers, which the CPU backend does not allocate.")

    _table(f"PER GPU (of {n_gpus}), resident through the tile loop", resident)
    if transient:
        _table(f"PER GPU (of {n_gpus}), transient (peak while that phase runs)",
               transient)
    _table(f"PER GPU (of {n_gpus}), BY PHASE", phases,
           total_label="sum of ALL phases listed")
    _table(f"PER GPU (of {n_gpus}), AFTER THE TILE LOOP (migrate, then repack; "
           "not co-resident with the phases above)", after_loop,
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

    # the generator runs float32 fields whatever the mesh dtypes; it wants particles per side
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
    host_peak = sum(host_res.values()) + host_phase_sums[worst_host]
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
          "that are\n  specific to this one. (1) DEVICE_PLACEMENT is a design "
          "assertion, checked\n  against the device executor (bitwise the host "
          "engine at cgh64 on four\n  GB200s) but not at 4096^3 shapes.\n  (2) "
          "The four-way split is charged as an exact quarter, which is right for\n"
          "  MEMORY -- each card holds its quarter however the wall splits -- and "
          "wrong\n  for WALL, where the measured speedup is 2.8-3.0x rather than "
          "4x. Do not read a\n  per-card byte here as licence for a quarter of a "
          "second anywhere.")
    return 0


def build(args):
    from .codec import T9Layout
    from .engine import EngineConfig

    # `--workers` reaches the engine config: every fine-arm term is per worker.
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
    # None, not 32: the preset fill only writes fields still None, so a default would
    # outrank the preset.
    ap.add_argument("--buf", type=int, default=None)
    ap.add_argument("--coarse-dtype", default="float32")
    ap.add_argument("--fine-dtype", default="float64")
    ap.add_argument("--bucket-cells", type=int, default=2)
    ap.add_argument("--slack", type=float, default=0.10, help="per-brick spare fraction")
    ap.add_argument("--alloc-margin", type=float, default=0.10)
    ap.add_argument("--arena-frac", type=float, default=0.01,
                    help="arena rows as a fraction of N. NOTE the default is "
                         "`SlotState.build`'s, but `scripts/run/realization.py` "
                         "runs 0.20, which is 28 GB of shared memory at c-gh.")
    # No default and not derived: `cap` (max padded per-tile rows) includes the clustering
    # spread of tile occupancy, which geometry cannot give, so a derived value would read low.
    ap.add_argument("--cap", type=int, default=None,
                    help="measured per-tile capacity, for the tile_buffers term. "
                         "cdev/cgh64/C-gh share N/tile and P, so cdev's measured "
                         "5284492 is the anchor for all three.")
    # Budgets have no defaults: a wrong default would invent a verdict.
    ap.add_argument("--host-gb", type=float, default=None,
                    help="host RAM budget. A property of YOUR machine: Vista gh "
                         "~116 (a hard cliff), Vista gg 237, S3 h100 ~1007.")
    ap.add_argument("--device-gb", type=float, default=None, help="accelerator HBM budget")
    # A separate budget: the pool holds the state in POSIX shared memory, a tmpfs sized by
    # default at half the node's RAM (`df -B1 /dev/shm`), which can bind before host RAM.
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
        # the arena is n_arena extra rows of off/w (`SlotState.off` is (n_alloc + n_arena, 3))
        "arena rows in off/w": arena * 9,
        "arena_bucket": arena * 8,
        "brick_start": (ec.n_brick and (args.n_fine // ec.n_brick) ** 3 + 1) * 8,
        # one f64 per brick (the per-brick velocity scale), not a per-particle array
        "brick_scales": (ec.n_brick and (args.n_fine // ec.n_brick) ** 3) * 8,
    }
    _table("STATE (resident for the whole run)", state)
    print(f"  {'':<20}  {sum(state.values()) / n:6.2f} B/p")

    # The tables below price a host that holds the coarse mesh and runs the tile loop.
    if args.backend == "device":
        return _device_main(args, ec, t9, n, rows, arena, state)

    mesh = ec.mesh_bytes()
    # Phases come from the engine (the code that allocates each term), not from here.
    from .engine import STEP_PHASE

    # `ec.mesh_phase()`, not the module dict: in pool mode the tile kernel build is in the loop
    MESH_PHASE = ec.mesh_phase()
    resident = {k: v for k, v in mesh.items() if MESH_PHASE[k] == "resident"}
    transient = {k: v for k, v in mesh.items() if k not in resident}
    _table("MESH, resident through the tile loop", resident)
    _table("MESH, transient (peak while that phase runs)", transient)

    step = ec.step_bytes(n, n_rows=rows, cap=args.cap)
    _table("PER-STEP HOST TERMS THAT SCALE WITH PARTICLES", step)

    # Terms in a phase are co-resident: sum within a phase, max across phases.
    phases = {}
    for src, phase_of in ((mesh, MESH_PHASE), (step, STEP_PHASE)):
        for k, v in src.items():
            p = phase_of[k]
            # a term naming several phases is charged to each
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

    # ---- the load path
    load_peak = _print_load_and_ic(
        args, ec, t9, n, rows, arena,
        shared=(args.workers is None or args.workers > 1))

    # ---- the verdict, with the binding term NAMED
    print("\nBINDING TERMS")
    # Charge the worst phase, not the largest single transient.
    from .engine import ONCE_PER_RUN_PHASES

    in_step = sum(v for k, v in phases.items() if k not in ONCE_PER_RUN_PHASES)
    once = max((v for k, v in phases.items() if k in ONCE_PER_RUN_PHASES),
               default=0)
    worst_phase = max(in_step, once)
    # pool workers are separate processes, each adding WORKER_STARTUP_BYTES
    n_workers = 0 if args.workers is None else max(0, int(args.workers))
    workers_b = int(n_workers * WORKER_STARTUP_BYTES) if n_workers > 1 else 0
    peak_est = (sum(state.values()) + sum(resident.values()) + worst_phase
                + workers_b)
    # transients are candidates for the largest term
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
          "charges every\n  in-step phase, not the largest single term. What it still "
          "cannot "
          "see is XLA's intra-jit\n  scratch, which is invisible to tracemalloc, to "
          "`live_arrays` and to\n  `memory_stats()` alike on CPU, and has measured ~2.6x this figure "
          "inside one\n  phase, so treat it as a sizing floor "
          "and never as a peak.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
