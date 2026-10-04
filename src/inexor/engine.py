"""The engine: BullFrog PM on T9 state in slot order.

Composes `integrate` (coefficients), `forces` (coarse + tiled fine arms) and `state` (the
container and the exchange) into a step loop.

BullFrog DKD's trailing half-drift of step n and leading half-drift of step n+1 share
`v_{n+1}`, so they fuse into one drift of `(dD_n + dD_{n+1})/2`. State is therefore carried
drift-synchronized (positions at step midpoints, velocities at step boundaries) and a step is

    force at the stored positions  ->  kick  ->  one fused drift + migrate

so the layout runs, and the state is quantized, once per step. Consequently the engine is
not bitwise `float_step_bullfrog` (`mod(mod(x+a)+b) != mod(x+a+b)` in float);
`float_run_bullfrog_sync` is its matched float reference.

Nothing is O(N) in floats: the long arm is read per tile from a staged coarse sub-block, the
short arm is per tile, and the kick writes velocities back per tile through the layout.
"""

import dataclasses
import hashlib
import json
import os
import time

import numpy as np

from .codec import INT16_MAX, assert_int16_range
from .forces import (
    CAP_RUNGS_PER_OCTAVE,
    COARSE_HALO,
    capacity_shape,
    coarse_force_meshes,
    coarse_kernel_parts,
    owned_mask_from_bricks,
    coarse_subblock_origin_extent,
    gather_coarse_subblock,
    make_tile_force_fn,
    check_stencil_guard,
    stage_coarse_subblock,
    tile_capacity,
    tile_geom,
    tile_origin_extent,
)
from .layout import assert_brick_divides_buffer, choose_brick
from .painting import check_tsc_paint_headroom, paint_tsc_int, paint_tsc_int_subblock
from .state import TileMembers, drift_and_migrate, drift_and_migrate_pooled

__all__ = [
    "EngineConfig", "apply_result", "checkpoint_fingerprint", "coarse_delta_streamed",
    "float_run_bullfrog_sync", "load_checkpoint", "run", "step", "tile_task",
]


def _dtype_name(x, what):
    """Normalize a mesh dtype ("float32", np.float32, jnp.float32...) to its name, numpy only.

    Raises at construction for anything but float32/float64.
    """
    dt = np.dtype(x)
    if dt not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"{what} must be float32 or float64, got {dt.name}")
    return dt.name


# The phase each budget term is charged to, so a budget sums what is live at once.
# `resident` = live for the whole run. Otherwise a phase name, or a tuple for a term spanning
# several. Terms in a phase are summed; distinct phases do not overlap (the phase hook in
# `step` marks the boundaries).
MESH_PHASE = {
    "coarse_kernel_build_f64": "kernel_build",
    "coarse_kernel_pref": "resident",
    "coarse_match_factor": "resident",
    "coarse_accumulator": "coarse_paint",
    "coarse_decode_slab": "coarse_paint",
    "coarse_spectrum": "coarse_solve",
    "coarse_solve_work": "coarse_solve",
    "coarse_kernel_slab": "coarse_solve",
    "coarse_device_planes": "coarse_solve",
    "coarse_force_copy_transient": "coarse_solve",
    # allocated in the paint, dropped by `step` once the solve's input exists
    "coarse_delta": ("coarse_paint", "coarse_solve"),
    "coarse_force_resident": "resident",
    "tile_kernels": "resident",
    "tile_kernel_build_f64": "kernel_build",
    "tile_kernel_pref": "kernel_build",
    "tile_workspace": "tile_loop",
}

STEP_PHASE = {
    "kick_pending": "tile_loop",
    "repack_scratch": "repack",
    "migrate_staging": "migrate",
    "tile_buffers": "tile_loop",
}

# Phases run once per run. The in-step phases are summed by budgets, not maxed, because glibc
# retains freed arenas between them, so a step's phases accumulate into its high-water.
ONCE_PER_RUN_PHASES = frozenset({"kernel_build"})


class EngineConfig:
    """Geometry and kernel choices for a run. Plain attributes, no jax."""

    def __init__(
        self,
        box_size,
        n_part,
        n_fine,
        n_coarse,
        n_tile,
        b_fine,
        alpha=1.0,
        paint_short="int",
        paint_long="int",
        frac_bits=12,
        chunk_bricks=64,
        brick_slack=0.10,
        repack_every=1,
        eject_kernel="jax",
        migrate_eject_inflight=None,
        coarse_dtype="float64",
        fine_dtype="float64",
        cap_rungs=CAP_RUNGS_PER_OCTAVE,
        pad_ladder=True,
        paint_subblock=True,
        tile_workers=1,
        worker_affinity=True,
        migrate_pooled=None,
        migrate_window=None,
        checkpoint_dir=None,
        checkpoint_every=1,
        migrate_backend="host",
        migrate_device_budget_bytes=None,
        coarse_backend="host",
        tile_backend="host",
        device_tile_jit=True,
        device_paint_chunk_bricks=None,
        device_tile_window=None,
        device_cards=1,
        migrate_repack_fused=None,
        coarse_fold_kernel=True,
        coarse_match_order=3,
        coarse_kernel_on_cards=None,
        device_y_blocks=1,
    ):
        self.box_size = float(box_size)
        self.n_part = int(n_part)
        self.n_fine = int(n_fine)
        self.n_coarse = int(n_coarse)
        self.n_tile = int(n_tile)
        self.b_fine = int(b_fine)
        self.alpha = float(alpha)
        # Integer (order-independent) paint on both arms: the layout reorders particles every
        # step, so an order-dependent float paint is not reproducible even on one machine.
        self.paint_short = str(paint_short)
        self.paint_long = str(paint_long)
        self.frac_bits = int(frac_bits)
        self.chunk_bricks = int(chunk_bricks)
        # The repack is required, not an optimization: frozen per-brick capacity eventually
        # overflows the arena.
        self.brick_slack = float(brick_slack)
        self.repack_every = int(repack_every)
        self.eject_kernel = str(eject_kernel)
        self.migrate_eject_inflight = (
            None if migrate_eject_inflight is None else int(migrate_eject_inflight))
        # Independent on purpose, so each moves one error term: the coarse mesh grows with the
        # box, the fine tile mesh (P^3) does not.
        self.coarse_dtype = _dtype_name(coarse_dtype, "coarse_dtype")
        self.fine_dtype = _dtype_name(fine_dtype, "fine_dtype")
        # Rungs per octave of the per-tile buffer shape ladder: `cap` moves every step, and an
        # unquantized shape would compile a new executable each time. More rungs = less
        # padding, more compilations.
        self.cap_rungs = int(cap_rungs)
        # Ladder for the coarse chunk buffer shape, separate from `cap_rungs`. False (unpadded
        # chunk shape) is a comparison arm, not an operating point.
        self.pad_ladder = bool(pad_ladder)
        # Paint each chunk into a coarse sub-block instead of a full mesh; bitwise-neutral by
        # integer associativity. False (full mesh per chunk) is a comparison arm.
        self.paint_subblock = bool(paint_subblock)
        # 1 = serial tile loop; > 1 = `executor.TilePool` running the same `tile_task`
        # (bitwise identical). `worker_affinity` pins each worker to disjoint cores before jax
        # loads (XLA-CPU sizes its spin pool by visible cores); off is not an operating point.
        self.tile_workers = int(tile_workers)
        self.worker_affinity = bool(worker_affinity)
        # Run `drift_and_migrate` on the pool (`state.drift_and_migrate_pooled`; bitwise the
        # serial pass). None = auto (pooled wherever a pool exists); True = require a pool
        # (refused at validate() without one); False = serial (serial references must say so).
        # `migrate_window` bounds scratch slots in flight; None = sized to feed the workers.
        self.migrate_pooled = None if migrate_pooled is None else bool(migrate_pooled)
        self.migrate_window = None if migrate_window is None else int(migrate_window)
        # Checkpoint every `checkpoint_every` steps into `checkpoint_dir`, alternating two
        # generations. Cadence is in steps (only step boundaries have a coherent state); 0
        # disables; inert without a directory. Per-step stats record whether it applied.
        self.checkpoint_dir = None if checkpoint_dir is None else str(checkpoint_dir)
        self.checkpoint_every = int(checkpoint_every)
        # Where migrate and repack run: "host" (serial or pooled per `migrate_pooled`) or
        # "device" (`device.migrate.drift_and_migrate_device` + `device.repack.repack_device`,
        # bitwise the serial host pass). With "device", an explicit `migrate_pooled=True` is
        # refused. `migrate_device_budget_bytes` is the per-slab device envelope the device
        # migrate refuses above; None = no limit.
        self.migrate_backend = str(migrate_backend)
        self.migrate_device_budget_bytes = (
            None if migrate_device_budget_bytes is None else int(migrate_device_budget_bytes))
        # One backend knob per phase, so phases move independently. "device" coarse =
        # `device.paint.coarse_delta_cards` + solve from the cards; "device" tile =
        # `device.tile.tile_loop_device`, one compiled program into device state.
        # `device_tile_jit=False` is the eager per-tile reference arm (bitwise `tile_task` on
        # CPU). `device_paint_chunk_bricks` None = `device.paint.default_chunk_bricks` (a
        # quarter x-slab; `chunk_bricks` would mean far too many device launches).
        self.coarse_backend = str(coarse_backend)
        self.tile_backend = str(tile_backend)
        # Fold the coarse kernel multiply into the inverse's axis-0 pass rather than a separate
        # host traversal. Bitwise either way (same association); False is a comparison arm.
        self.coarse_fold_kernel = bool(coarse_fold_kernel)
        # Keep the coarse kernel's real half-grids (`pref`, `mf`) resident on the cards, each
        # card its y-pencil rows, instead of on the host sliced per block; bitwise either way.
        # Tri-state as `device_tile_window` (applies to the folded solve into card shards);
        # read `kernel_on_cards`.
        self.coarse_kernel_on_cards = (
            None if coarse_kernel_on_cards is None else bool(coarse_kernel_on_cards))
        # Assignment order the coarse match factor divides out. The coarse arm paints and
        # gathers TSC, so 3 is correct (default); 2 (CIC order) leaves a residual sinc^2 per
        # axis on the long force and is kept for the reference parity comparisons.
        if int(coarse_match_order) not in (2, 3):
            raise ValueError(f"coarse_match_order must be 2 or 3, got {coarse_match_order}")
        self.coarse_match_order = int(coarse_match_order)
        self.device_tile_jit = bool(device_tile_jit)
        self.device_paint_chunk_bricks = (
            None if device_paint_chunk_bricks is None else int(device_paint_chunk_bricks))
        # Run the compiled device tile loop against a window of x-slabs on the card
        # (`device.window.tile_loop_windowed`) rather than the whole state; bitwise the
        # whole-state loop. None = auto (wherever the compiled device tile runs); True =
        # require (refused where it cannot apply); False = whole-state. Read `tile_window`.
        self.device_tile_window = None if device_tile_window is None else bool(device_tile_window)
        # Cards the device step splits across: coarse paint/solve by x-planes, tile loop by
        # tile planes (one thread per card), and device migrate/repack by x-slabs.
        self.device_cards = int(device_cards)
        # On a repack step, run the device migrate and repack as one visit per slab
        # (`device.fused.migrate_repack_device`, sized by the windowed tile loop's destination
        # census); bitwise the two passes. Tri-state as `device_tile_window`; read `fused_pass`.
        self.migrate_repack_fused = (
            None if migrate_repack_fused is None else bool(migrate_repack_fused))
        # Split every x-slab-sized card working set (tile window, destination census, device
        # migrate, fused repack) into this many y-blocks (`decomp.y_blocks`); bitwise any count.
        # 1 = whole slabs.
        self.device_y_blocks = int(device_y_blocks)

    @property
    def np_coarse_dtype(self):
        return np.dtype(self.coarse_dtype)

    @property
    def np_fine_dtype(self):
        return np.dtype(self.fine_dtype)

    @property
    def fused_pass(self):
        """Whether a step with a repack due runs the fused migrate + repack:
        `migrate_repack_fused` resolved (None = wherever the device migrate and the
        tile window both run)."""
        applies = self.migrate_backend == "device" and self.tile_window
        return applies and self.migrate_repack_fused is not False

    @property
    def kernel_on_cards(self):
        """Whether the coarse kernel parts live on the cards: `coarse_kernel_on_cards`
        resolved (None = wherever the folded solve writes card shards)."""
        applies = (self.tile_backend == "device" and self.device_tile_jit
                   and self.coarse_fold_kernel)
        return applies and self.coarse_kernel_on_cards is not False

    @property
    def tile_window(self):
        """Whether the compiled device tile loop runs against the x-slab window:
        `device_tile_window` resolved (None = wherever the compiled device tile runs)."""
        compiled_device = self.tile_backend == "device" and self.device_tile_jit
        return compiled_device and self.device_tile_window is not False

    @property
    def n_total(self):
        return self.n_part**3

    @property
    def coarse_cell(self):
        return self.box_size / self.n_coarse

    @property
    def fine_cell(self):
        return self.box_size / self.n_fine

    @property
    def coarse_match(self):
        """The `match` argument of the coarse solve (order 2 keeps the plain 2-tuple form)."""
        if self.coarse_match_order == 2:
            return (self.coarse_cell, self.fine_cell)
        return (self.coarse_cell, self.fine_cell, self.coarse_match_order, 2)

    @property
    def r_s(self):
        """The split scale; `alpha = r_s / coarse_cell` is the dimensionless knob."""
        return self.alpha * self.coarse_cell

    def mesh_bytes(self):
        """Per-step mesh anatomy in bytes, as a function of the two dtypes.

        Kept out of `bytes_per_particle`: mesh terms are not per-particle. Phases per term
        are `mesh_phase()`. Kernel builds stay f64 at either dtype (the precision island) and
        are reported separately. Fine-arm terms are per tile worker.
        """
        cw = self.np_coarse_dtype.itemsize
        fw = self.np_fine_dtype.itemsize
        nc = self.n_coarse
        cells = nc**3
        half = nc * nc * (nc // 2 + 1)
        from .forces import padded_size

        p = padded_size(self.n_tile, self.b_fine, n_fine=self.n_fine)[0]
        pcells = p**3
        phalf = p * p * (p // 2 + 1)
        slab = max(1, min(nc, 32))
        # In pool mode each worker builds its own tile kernels and working set (the parent
        # builds none), so fine-arm terms scale with W.
        w = max(int(self.tile_workers), 1)
        return dict(
            # --- coarse, transient
            coarse_accumulator=cells * 8,          # int64 host, dtype-independent
            coarse_decode_slab=slab * nc * nc * 8,  # one f64 slab
            # the f64 island beyond the kept pref/mf: measured 4.50 f64 half-grids at the peak
            # of `coarse_kernel_parts` (tracemalloc, nc 256 and 512), of which 1.0 is kept
            coarse_kernel_build_f64=int(3.5 * half * 8),
            # The factorized solve (one component at a time). Host-resident:
            # `_coarse_solve_factorized` keeps the spectrum in numpy and ships only planes.
            coarse_spectrum=half * 2 * cw,        # the forward's output, held
            coarse_solve_work=half * 2 * cw,      # per component: copy AS multiply
            # per-slab kernel and the `spec * k` temporary, live together; the folded solve
            # forms the kernel per pencil block on the card instead
            coarse_kernel_slab=(0 if self.coarse_fold_kernel
                                else 2 * slab * nc * (nc // 2 + 1) * 2 * cw),
            # planes the transform holds on a device; charged at 16 planes, about twice the
            # measured working set
            coarse_device_planes=16 * nc * (nc // 2 + 1) * 2 * cw,
            # --- coarse, resident for the whole run: the once-per-run build's real half-grids
            # (`forces.coarse_kernel_parts`); the complex kernels are not kept.
            coarse_kernel_pref=half * cw,
            coarse_match_factor=half * cw,
            # --- coarse, resident through the tile loop
            coarse_delta=cells * cw,
            coarse_force_resident=3 * cells * cw,
            # one component: `coarse_force_meshes` solves one at a time into the caller's buffers
            coarse_force_copy_transient=cells * cw,
            # --- fine, resident through the tile loop
            tile_kernels=w * 3 * phalf * 2 * fw,
            # the build: `split_kernels` holds k2_true, k2_safe, fac (f64) and pref (fine dtype)
            # while materializing the complex kernels; the `ik_j` are low-rank broadcasts
            tile_kernel_build_f64=w * 3 * phalf * 8,
            tile_kernel_pref=w * phalf * fw,
            # --- fine, transient per tile
            tile_workspace=w * (pcells * (fw + 4 + 3 * fw) + phalf * 2 * fw),
        )

    # Measured eject host allocations per slab row (flat in slab size). Floors: the jax
    # kernel's device buffers are not included. Both kernels are bitwise identical; numpy is
    # slower, so this is a pure memory/wall trade.
    EJECT_BYTES_PER_ROW = {"numpy": 35.0, "jax": 129.0}

    # An insert's own working set per row (its inputs are views of the shm scratch).
    INSERT_BYTES_PER_ROW = 50.25

    # The serial pass's measured whole-phase bytes per slab row (eject + insert, 2r+1 slabs).
    SERIAL_MIGRATE_BYTES_PER_ROW = 190.0

    def _migrate_b_per_row(self):
        """Bytes per slab row the migrate holds under this execution policy.

        Serial holds 2r+1 slabs in total; pooled holds one slab per worker, so it scales with
        the worker count (and the eject kernel).
        """
        if self.tile_workers <= 1 or self.migrate_pooled is False:
            return self.SERIAL_MIGRATE_BYTES_PER_ROW
        per_row = self.EJECT_BYTES_PER_ROW.get(self.eject_kernel)
        if per_row is None:
            raise ValueError(
                f"no measured eject coefficient for kernel {self.eject_kernel!r}; "
                f"known: {sorted(self.EJECT_BYTES_PER_ROW)}. A budget that guesses "
                "one would be indistinguishable from a budget that measured it."
            )
        w = self.tile_workers
        # worst case: E workers ejecting and the other W - E inserting
        e = w if self.migrate_eject_inflight is None else min(
            self.migrate_eject_inflight, w)
        return per_row * e + self.INSERT_BYTES_PER_ROW * max(w - e, 0)

    def mesh_phase(self):
        """`MESH_PHASE` for this config: in pool mode each worker builds its tile kernels on
        its first task, so the tile kernel build is charged to `tile_loop`, not `kernel_build`.
        """
        m = dict(MESH_PHASE)
        if self.tile_workers > 1:
            m["tile_kernel_build_f64"] = "tile_loop"
            m["tile_kernel_pref"] = "tile_loop"
        return m

    def step_bytes(self, n_particles, n_rows=None, cap=None):
        """Per-step host terms that scale with particles, not with the mesh (see `mesh_bytes`).

        `kick_pending` -- always 0: per-brick velocity scales (resident, `n_bricks * 8`, in
        the state table) let each brick's scale be set in its own tile, so no per-particle
        pending-kick array exists. Kept as a zero line.
        `repack_scratch` -- the in-place `SlotState.repack`'s measured working set per row.
        `migrate_staging` -- slabs in flight during the migrate (O(N^(2/3)), see below).
        `tile_buffers` -- per-worker per-tile host working set; only with a measured `cap`.
        """
        n = int(n_particles)
        rows = int(n_rows) if n_rows is not None else int(round(n * 1.21))
        # Migration staging is a few x-slabs, not the state: a slab is N / bricks_per_side, so
        # the term grows as N^(2/3). The coefficient is measured, the shape derived. The lead
        # drift (same function, before the loop) is never co-resident with a step's peak.
        nb = max(1, self.n_fine // self.n_brick)
        if self.migrate_backend == "device":
            # The device passes' host scratch: two slab-sized numpy buffers at 9 B/row (from the
            # code, not measured). Device-side terms are in `plan.device_budget`.
            host_window = int(round(2 * 9.0 * n / nb))
            out = dict(kick_pending=0, repack_scratch=host_window,
                       migrate_staging=host_window)
            if cap is not None:
                out["tile_buffers"] = (max(int(self.tile_workers), 1) * int(cap)
                                       * (8 + 1 + 24 + 24 + 1 + 8 + 24 + 24))
            return out
        out = dict(
            kick_pending=0,
            # measured B/row of the in-place repack; holds while n_buckets / n_rows is fixed,
            # as it is across the presets
            repack_scratch=int(round(rows * 0.49)),
            migrate_staging=int(round(self._migrate_b_per_row() * n / nb)),
        )
        if cap is not None:
            # per WORKER: each holds one tile's buffers at once
            out["tile_buffers"] = (max(int(self.tile_workers), 1) * int(cap)
                                   * (8 + 1 + 24 + 24 + 1 + 8 + 24 + 24))
        return out

    @property
    def n_brick(self):
        return choose_brick(self.n_tile, self._b_realized, self.n_fine)

    @property
    def _b_realized(self):
        from .forces import padded_size

        return padded_size(self.n_tile, self.b_fine, n_fine=self.n_fine)[1]

    @property
    def tiles_side(self):
        return self.n_fine // self.n_tile

    @property
    def tiles(self):
        s = self.tiles_side
        return [(i, j, k) for i in range(s) for j in range(s) for k in range(s)]

    def validate(self):
        assert_brick_divides_buffer(self.n_tile, self._b_realized, self.n_brick, self.n_fine)
        if self.paint_long == "int":
            check_tsc_paint_headroom(self.n_total, self.frac_bits)
        if self.tile_workers < 1:
            raise ValueError(f"tile_workers must be >= 1, got {self.tile_workers}")
        if self.migrate_pooled is True and self.tile_workers < 2:
            # an explicit True must not silently run serial (None = auto may)
            raise ValueError(
                f"migrate_pooled needs a pool: tile_workers is {self.tile_workers}"
            )
        if self.migrate_backend not in ("host", "device"):
            raise ValueError(
                f"migrate_backend must be 'host' or 'device', got {self.migrate_backend!r}")
        if self.migrate_backend == "device":
            if self.migrate_pooled is True:
                raise ValueError(
                    "migrate_pooled=True and migrate_backend='device' name two different "
                    "migrates; the device pass does not use the pool. Drop one.")
        for name in ("coarse_backend", "tile_backend"):
            if getattr(self, name) not in ("host", "device"):
                raise ValueError(
                    f"{name} must be 'host' or 'device', got {getattr(self, name)!r}")
        if self.coarse_backend == "device":
            if not self.paint_subblock:
                raise ValueError(
                    "paint_subblock=False asks for the full-mesh paint, which "
                    "coarse_backend='device' does not have. Drop one.")
            if self.device_paint_chunk_bricks is not None and self.device_paint_chunk_bricks < 1:
                raise ValueError(
                    f"device_paint_chunk_bricks must be >= 1, got {self.device_paint_chunk_bricks}")
        if self.tile_backend == "device" and self.tile_workers > 1:
            raise ValueError(
                f"tile_backend='device' with tile_workers={self.tile_workers}: the pool is "
                "the CPU lane and refuses a GPU parent. The device tile loop is one program "
                "per card; set tile_workers=1.")
        if not self.device_tile_jit and self.tile_backend != "device":
            raise ValueError(
                "device_tile_jit=False selects the eager DEVICE tile, but tile_backend is "
                f"{self.tile_backend!r}; the knob could not apply.")
        if self.device_tile_window is True and not (self.tile_backend == "device"
                                                    and self.device_tile_jit):
            raise ValueError(
                "device_tile_window=True windows the COMPILED device tile loop, but "
                f"tile_backend={self.tile_backend!r}, device_tile_jit={self.device_tile_jit}; "
                "the knob could not apply.")
        if self.coarse_kernel_on_cards is True and not self.kernel_on_cards:
            raise ValueError(
                "coarse_kernel_on_cards=True keeps the kernel on the cards for the folded solve "
                f"into card shards, but tile_backend={self.tile_backend!r}, device_tile_jit="
                f"{self.device_tile_jit}, coarse_fold_kernel={self.coarse_fold_kernel}; the "
                "knob could not apply.")
        if self.device_cards < 1:
            raise ValueError(f"device_cards must be >= 1, got {self.device_cards}")
        if self.device_cards > 1:
            if not (self.coarse_backend == "device" and self.tile_backend == "device"
                    and self.device_tile_jit and self.tile_window):
                raise ValueError(
                    f"device_cards={self.device_cards} splits the device step across cards: "
                    "it needs coarse_backend='device', tile_backend='device', the compiled "
                    "tile (device_tile_jit) and the window (device_tile_window).")
            if self.device_cards > self.tiles_side:
                raise ValueError(
                    f"device_cards={self.device_cards} > {self.tiles_side} tile planes: a card "
                    "would run no tiles.")
        if self.migrate_repack_fused is True and not (
                self.migrate_backend == "device" and self.tile_window):
            raise ValueError(
                "migrate_repack_fused=True fuses the DEVICE migrate with the repack and "
                "sizes it from the tile window's census, but migrate_backend="
                f"{self.migrate_backend!r}, tile_window={self.tile_window}; the knob could "
                "not apply.")
        if not 1 <= self.device_y_blocks <= self.tiles_side:
            raise ValueError(f"device_y_blocks must be in [1, {self.tiles_side}] (the tile rows "
                             f"per side), got {self.device_y_blocks}")
        if self.device_y_blocks > 1 and not (self.tile_window
                                             or self.migrate_backend == "device"):
            raise ValueError(
                f"device_y_blocks={self.device_y_blocks} splits the tile window and the device "
                f"migrate, but tile_window={self.tile_window}, migrate_backend="
                f"{self.migrate_backend!r}; the knob could not apply.")
        on_device = [n for n in ("migrate_backend", "coarse_backend", "tile_backend")
                     if getattr(self, n) == "device"]
        if on_device:
            import jax

            if not jax.config.jax_enable_x64:
                raise ValueError(
                    f"{' and '.join(on_device)}='device' needs jax_enable_x64 (the compiled "
                    "kernels' int64 lattice index is silently int32 without it). Enable it "
                    "in the driver; the library never does.")
            if self.device_cards > len(jax.devices()):
                raise ValueError(
                    f"device_cards={self.device_cards} but this process sees "
                    f"{len(jax.devices())} jax devices; more threads than cards would run "
                    "on one card while reading as a split.")
        self._refuse_f64_without_x64()
        return True

    def _refuse_f64_without_x64(self):
        """Refuse a float64 mesh dtype when jax_enable_x64 is off (it would silently be f32).

        The library never enables x64; callers opt in. Called from `validate()` (which `run`
        calls), not `step()`, so an ad-hoc single step stays possible.
        """
        import jax

        if jax.config.jax_enable_x64:
            return
        wants = [n for n in ("coarse_dtype", "fine_dtype") if getattr(self, n) == "float64"]
        if wants:
            raise ValueError(
                f"{' and '.join(wants)} ask for float64 while jax_enable_x64 is False, so "
                "the mesh would silently be float32 and an 'f64 reference' would not be "
                "one. Call jax.config.update('jax_enable_x64', True) in the driver (the "
                "library never sets it), or ask for float32 explicitly."
            )


# ===========================================================================
# the long arm, streamed
# ===========================================================================


def _chunk_cuboid(chunk_index, chunk_len, nb, n_coarse):
    """Map a brick-major run of `chunk_len` bricks to a coarse-cell cuboid.

    Returns `(cell_origin (3,), cell_span (3,))`, or None when the run is not a cuboid (the
    chunk then takes the bitwise-identical full-mesh paint). A run is a cuboid when L divides
    the brick grid cleanly: whole i-planes (nb^2 | L), whole j-rows (nb | L | nb^2) or part of a
    row (L | nb); each also gives every chunk the same shape (one XLA shape).
    """
    L = int(chunk_len)
    nb = int(nb)
    if L <= 0 or nb**3 % L or int(n_coarse) % nb:
        return None
    if L % (nb * nb) == 0:
        shape = (L // (nb * nb), nb, nb)
    elif L % nb == 0 and (nb * nb) % L == 0:
        shape = (1, L // nb, nb)
    elif nb % L == 0:
        shape = (1, 1, L)
    else:
        return None
    bpc = int(n_coarse) // nb
    s = int(chunk_index) * L
    b0 = np.array([s // (nb * nb), (s // nb) % nb, s % nb], dtype=np.int64)
    return b0 * bpc, np.array(shape, dtype=np.int64) * bpc


def _assert_stencil_contained(x, coarse_cell, origin, extent, n_coarse):
    """The host-side half of `paint_tsc_int_subblock`'s containment contract.

    Cheap and outside the jit. A violation means the cuboid derivation is wrong; the stencil
    would otherwise wrap silently to the far side of the block, so this raises.
    """
    base = np.rint(x / float(coarse_cell)).astype(np.int64)
    for ax in range(3):
        if int(extent[ax]) >= int(n_coarse):
            continue  # full axis: any index is in range by construction
        local = (base[:, ax] - int(origin[ax])) % int(n_coarse)
        if not ((local >= 1) & (local <= int(extent[ax]) - 2)).all():
            raise ValueError(
                f"sub-block paint containment violated on axis {ax}: a chunk row's "
                f"TSC base falls outside [origin+1, origin+extent-2] "
                f"(origin {int(origin[ax])}, extent {int(extent[ax])}, n {n_coarse}). "
                "The cuboid derivation is wrong; refusing rather than wrapping silently."
            )


def coarse_delta_streamed(st, cfg, stats=None, census=False, pad_shape=0, pool=None,
                          progress=None):
    """delta on the coarse mesh, accumulated brick by brick.

    Requires the integer paint: integer addition is associative, so the chunked accumulation
    is bitwise a single call over all positions (an f64 paint would change with chunking).

    `stats`, if a dict, receives the pad shapes, `coarse_peak_int` (max cell sum) and which
    path painted. `pad_shape` is the previous step's chunk shape, keeping it monotone across
    a run (`forces.capacity_shape`). `pool` runs each chunk's decode + sub-block paint on
    workers while accumulation stays here (bitwise by associativity); `census=True` and
    `paint_subblock=False` always run serial. `progress` is called once per chunk.

    `census=True` also counts cells whose integer sum does not round-trip through f32
    (`coarse_cells_inexact_f32`); two extra mesh passes. Round-trip, not `>= 2^24`, which is
    sufficient for exactness but not necessary.
    """
    if cfg.paint_long != "int":
        raise ValueError(
            "the streamed coarse paint requires the integer accumulator: an f64 "
            "accumulation is order-dependent, so chunking it changes the result"
        )
    import jax.numpy as jnp

    n = cfg.n_coarse
    mesh = np.zeros((n, n, n), dtype=np.int64)
    bricks = list(range(st.n_bricks))
    groups = [bricks[i : i + cfg.chunk_bricks] for i in range(0, len(bricks), cfg.chunk_bricks)]
    rows = [sum(st.brick_member_count(b) for b in gg) for gg in groups]
    # One padded shape for every chunk and, via the capacity ladder, across steps: each new
    # shape compiles and caches another executable for the life of the process. Pad rows
    # are masked (an unmasked pad row would add mass), so padding is bitwise neutral.
    pad_true = int(max(rows)) if rows else 0
    pad = (
        capacity_shape(pad_true, rungs=cfg.cap_rungs, floor_shape=pad_shape)
        if cfg.pad_ladder
        else pad_true
    )
    # Sub-block paint: a brick-major chunk is a spatial cuboid, so its TSC footprint is a
    # bounded block, painted locally and added through per-axis wrapped indices (bitwise the
    # full-mesh paint; avoids an n_coarse^3 mesh per chunk). An axis whose span+3 reaches
    # n_coarse runs full-axis, keeping scatter indices unique (a repeated index under
    # fancy-indexed += would drop adds).
    nb_side = cfg.n_fine // cfg.n_brick
    coarse_cell = cfg.box_size / float(n)
    n_sub = 0
    pooled_workers = 0
    if pool is not None and cfg.paint_subblock and not census:
        # chunks on the workers, integer accumulation here
        pool.stage_coarse(dict(pad=int(pad), n_coarse=n, box=cfg.box_size,
                               frac_bits=cfg.frac_bits,
                               chunk_bricks=cfg.chunk_bricks, nb_side=nb_side))
        tasks = [(gi, np.asarray(gg, dtype=np.int64))
                 for gi, (gg, m) in enumerate(zip(groups, rows)) if m]
        n_done = 0
        for res in pool.imap_coarse(tasks):
            n_done += 1
            if progress is not None:
                progress("coarse paint", n_done, len(tasks))
            if res["empty"]:
                continue
            ax = [(np.arange(int(res["extent"][a]), dtype=np.int64)
                   + int(res["origin"][a])) % n for a in range(3)]
            mesh[np.ix_(*ax)] += res["sub"]
            n_sub += 1
        pooled_workers = pool.workers
        groups = []  # the serial loop below must not run the chunks again
    for gi, (gg, m) in enumerate(zip(groups, rows)):
        if progress is not None:
            progress("coarse paint", gi, len(groups))
        if m == 0:
            continue
        _, x, _ = st.decode_bricks(gg)
        xp = np.zeros((pad, 3), dtype=np.float64)
        xp[:m] = x
        lv = np.zeros(pad, dtype=bool)
        lv[:m] = True
        cub = (
            _chunk_cuboid(gi, cfg.chunk_bricks, nb_side, n)
            if cfg.paint_subblock
            else None
        )
        if cub is None:
            mesh += np.asarray(
                paint_tsc_int(jnp.asarray(xp), n, cfg.box_size, cfg.frac_bits, live=lv),
                dtype=np.int64,
            )
            continue
        c0, span = cub
        origin = np.where(span + 3 >= n, 0, (c0 - 1) % n)
        extent = np.where(span + 3 >= n, n, span + 3)
        _assert_stencil_contained(x, coarse_cell, origin, extent, n)
        sub = np.asarray(
            paint_tsc_int_subblock(
                jnp.asarray(xp), tuple(int(o) for o in origin),
                tuple(int(e) for e in extent), n, cfg.box_size, cfg.frac_bits,
                live=lv,
            ),
            dtype=np.int64,
        )
        ax = [(np.arange(int(extent[a]), dtype=np.int64) + int(origin[a])) % n
              for a in range(3)]
        mesh[np.ix_(*ax)] += sub
        n_sub += 1
    if progress is not None and groups:
        progress("coarse paint", len(groups), len(groups))
    out, peak, inexact = _delta_from_accumulated(mesh, cfg, census=census)
    if stats is not None:
        stats["coarse_pad"] = pad
        stats["coarse_pad_true"] = pad_true
        stats["coarse_peak_int"] = peak
        # receipts that the knobs applied (pooled_workers 0 = painted serially)
        stats["coarse_subblock_chunks"] = n_sub
        stats["coarse_pooled_workers"] = pooled_workers
        if census:
            stats["coarse_cells_inexact_f32"] = inexact
            stats["coarse_exact_decode_ok"] = inexact == 0
    return out


def _delta_from_accumulated(mesh, cfg, census=False):
    """(delta, peak, inexact) from the accumulated int64 coarse paint.

    Shared by the host and device (`device.paint`) paints, so both decode identically.
    `inexact` is the f32 round-trip census count when `census`, else 0.
    """
    n = cfg.n_coarse
    peak = int(np.abs(mesh).max())
    if peak >= 2**31:
        raise ValueError(
            f"the accumulated coarse paint reached {peak}, past int32. "
            "Lower frac_bits -- int32 overflow corrupts the paint; it is not imprecision."
        )
    # Slabbed decode: the transient is one f64 slab and the result lands at the requested
    # dtype; elementwise, so bitwise the whole-array form. Decode in f64, narrow after `- 1.0`
    # (see `painting.density_tsc`).
    mean = float(cfg.n_total) / float(n) ** 3
    scale = 2.0 ** -cfg.frac_bits
    out = np.empty((n, n, n), dtype=cfg.np_coarse_dtype)
    slab = max(1, min(n, 32))
    inexact = 0
    for i0 in range(0, n, slab):
        s = mesh[i0 : i0 + slab]
        out[i0 : i0 + slab] = s.astype(np.float64) * scale / mean - 1.0
        if census:
            inexact += int(np.count_nonzero(s.astype(np.float32).astype(np.int64) != s))
    return out, peak, inexact


# ===========================================================================
# one step
# ===========================================================================


def _diagnose_partition(st, cfg):
    """Where the ownership deficit is, for the assertion's message.

    Failure path only: reports which rows are unclaimed, aliased or double-claimed, and where
    the occupancy / member-count / decode censuses disagree. On a node-local state the
    censuses cover its owned bricks and the tile claims are skipped (tiles read other ranks'
    bricks), so "unclaimed" is not reported there.
    """
    b_real = cfg._b_realized
    nb = cfg.n_fine // cfg.n_brick
    claimed = np.zeros(st.off.shape[0], dtype=np.int32)
    misaligned = []
    b_lo, b_hi = st.owned_bricks
    for t in (cfg.tiles if st.is_whole else ()):
        members = st.tile_bricks(t, cfg.n_tile, b_real, cfg.n_brick, cfg.n_fine)
        slots, _, _ = st.decode_bricks(members)
        counts = [st.brick_member_count(b) for b in members]
        if int(np.sum(counts)) != len(slots):
            misaligned.append((tuple(int(q) for q in t), int(np.sum(counts)), len(slots)))
            continue
        brick_of_row = np.repeat(np.asarray(members, dtype=np.int64), counts)
        own = owned_mask_from_bricks(brick_of_row, t, cfg.n_tile, cfg.n_brick, nb)
        # np.add.at: fancy-index += counts a repeated index once, hiding double claims
        np.add.at(claimed, np.asarray(slots)[own], 1)
    # distinct slots, since `SlotState.check` compares counts and an aliased slot passes it
    seen = np.zeros(st.off.shape[0], dtype=np.int64)
    total_decoded = 0
    for b in range(b_lo, b_hi):
        s = np.asarray(st.decode_brick(b)[0])
        total_decoded += len(s)
        np.add.at(seen, s, 1)
    live = seen > 0
    aliased = np.nonzero(seen > 1)[0]
    unclaimed = np.nonzero(live & (claimed == 0))[0] if st.is_whole else np.empty(0, np.int64)
    twice = np.nonzero(claimed > 1)[0]
    # Three per-brick censuses that must agree: occupancy vs member_count disagreeing is a
    # counting bug, member_count vs decode a span/decode bug.
    p3 = st.buckets_per_brick
    occ = st.occupancy.astype(np.int64)
    occ_per_brick = occ.reshape(b_hi - b_lo, p3).sum(axis=1)
    mc_per_brick = np.array(
        [st.brick_member_count(b) for b in range(b_lo, b_hi)], dtype=np.int64
    )
    dec_per_brick = np.array(
        [len(st.decode_brick(b)[0]) for b in range(b_lo, b_hi)], dtype=np.int64
    )
    bad = np.nonzero((occ_per_brick != mc_per_brick) | (mc_per_brick != dec_per_brick))[0]
    detail = [
        (int(b) + b_lo, int(occ_per_brick[b]), int(mc_per_brick[b]), int(dec_per_brick[b]))
        for b in bad[:4]
    ]
    lines = [
        f"  CENSUS occupancy {int(occ_per_brick.sum()) + st.arena_used} "
        f"(incl. arena), brick_member_count {int(mc_per_brick.sum())}, "
        f"decode {int(dec_per_brick.sum())}, n_particles {st.n_particles}",
        f"  bricks where the three disagree: {len(bad)} "
        f"(brick, occ, member_count, decode)={detail}",
        f"  distinct live rows {int(live.sum())}, decoded WITH duplicates "
        f"{total_decoded}, n_particles {st.n_particles}, "
        f"arena_used {st.arena_used} of {st.n_arena}",
        f"  ALIASED slots (reachable through more than one brick) {len(aliased)}"
        f"{': ' + str([int(q) for q in aliased[:5]]) if len(aliased) else ''}",
        f"  unclaimed {len(unclaimed)}, claimed-more-than-once {len(twice)}",
    ]
    for s in aliased[:5]:
        s = int(s)
        bricks_with = [b for b in range(b_lo, b_hi)
                       if s in set(int(q) for q in st.decode_brick(b)[0])]
        lines.append(
            f"  aliased slot {s}: seen {int(seen[s])}x in bricks {bricks_with[:6]}, "
            f"off={[int(q) for q in st.off[s]]}, "
            f"in_arena={s >= st.arena_base}"
        )
    if misaligned:
        lines.append(
            f"  BRICK COUNT vs DECODE LENGTH disagree on {len(misaligned)} tiles: "
            f"{misaligned[:3]}"
        )
    for s in unclaimed[:5]:
        s = int(s)
        in_arena = s >= st.arena_base
        bucket = int(st.arena_bucket[s - st.arena_base]) if in_arena else None
        brick = int(np.searchsorted(st.brick_start, s, side="right") - 1)
        lines.append(
            f"  slot {s}: in_arena={in_arena} arena_bucket={bucket} "
            f"brick_by_span={brick} off={[int(q) for q in st.off[s]]}"
        )
    return "\n".join(lines)


def _no_phase(_name):
    """The default phase hook: does nothing, allocates nothing, returns nothing."""


def tile_task(st, one_tile, C, g_coarse, t, bricks, ph=_no_phase):
    """One tile of the kick: decode -> short + long force -> quantize per brick.

    Reads `st` and writes nothing: the tile's writes are returned and applied only by
    `apply_result`. This is the executor seam: the serial loop in `step` and the pool workers
    run the same function. `C` is the small, picklable per-step header (geometry, `cap`, kick
    coefficients). `ph` is the phase hook (see `step`); `busy` walls are returned either way.
    """
    import jax.numpy as jnp

    t_dec = time.perf_counter()
    slots, x, v = st.decode_bricks(bricks)
    # the brick each decoded row is stored in (decode order); ownership is read from this
    brick_of_row = np.repeat(
        np.asarray(bricks, dtype=np.int64), [st.brick_member_count(b) for b in bricks]
    )
    m = len(slots)
    if m == 0:
        return dict(t=t, empty=True, n_owned=0, n_out=0)
    if m > C["cap"]:
        raise RuntimeError(f"tile {t}: {m} members > cap {C['cap']}")
    idx = np.resize(np.arange(m), C["cap"])
    live = np.zeros(C["cap"], dtype=bool)
    live[:m] = True
    origin, _ = tile_origin_extent(t, C["n_tile"], C["b_real"], C["cell"])
    u = jnp.mod(jnp.asarray(x[idx]) - jnp.asarray(origin), C["box"])
    # ownership from the storage brick, the same source as membership, so the two cannot
    # disagree (re-deriving it from a coordinate can lose boundary rows)
    own_rows = owned_mask_from_bricks(
        brick_of_row, t, C["n_tile"], C["n_brick"], C["n_fine"] // C["n_brick"]
    )
    own = np.zeros(C["cap"], dtype=bool)
    own[:m] = own_rows
    own &= live
    ph("tile_decode")
    t_short = time.perf_counter()
    g_short, owned, n_out = one_tile(u, jnp.asarray(live), jnp.asarray(own))
    g_short = np.asarray(g_short)[:m]
    owned = np.asarray(owned)[:m]
    ph("tile_short")
    t_long = time.perf_counter()
    if not owned.any():
        return dict(t=t, empty=True, n_owned=0, n_out=int(n_out))

    # the long force at the SAME owned rows, out of a staged sub-block
    o_cells, extent = coarse_subblock_origin_extent(
        t, C["n_tile"], C["n_coarse"], C["n_fine"], halo=COARSE_HALO
    )
    sub = [stage_coarse_subblock(g, o_cells, extent) for g in g_coarse]
    # padded to `cap` with a live mask so one XLA shape serves every tile
    n_own = int(owned.sum())
    xo = np.zeros((C["cap"], 3), dtype=np.float64)
    xo[:n_own] = x[owned]
    lv = np.zeros(C["cap"], dtype=bool)
    lv[:n_own] = True
    # `guard_out` defers the stencil-containment bounds (device scalars) until after the
    # readback, avoiding a mid-gather sync; it still fires before `g_long` is used.
    stencil_guard = []
    g_long_dev = gather_coarse_subblock(
        *sub, jnp.asarray(xo), o_cells, C["coarse_cell"], C["n_coarse"],
        assign="tsc", live=lv, guard_out=stencil_guard,
    )
    g_long = np.asarray(g_long_dev)[:n_own]
    check_stencil_guard(stencil_guard)
    ph("tile_long")
    t_quant = time.perf_counter()
    g_tot = g_short[owned] + g_long
    v_new = C["alpha_k"] * v[owned] + C["bcoef"] * g_tot
    # Quantize per brick. Owned rows are whole, contiguous brick blocks (decode order +
    # storage-brick ownership), so a run scan finds them; asserted, since a brick split over
    # two runs would silently overwrite its own scale.
    slots_o, bricks_o = slots[owned], brick_of_row[owned]
    cut = np.flatnonzero(np.diff(bricks_o)) + 1
    run_lo = np.concatenate(([0], cut))
    run_hi = np.concatenate((cut, [len(bricks_o)]))
    if len(np.unique(bricks_o)) != len(run_lo):
        raise AssertionError(
            f"tile {t}: owned rows are not grouped by brick "
            f"({len(run_lo)} runs over {len(np.unique(bricks_o))} bricks). The "
            "per-brick scale depends on a brick's rows being contiguous."
        )
    w_codes = np.empty((n_own, 3), dtype=np.int16)
    run_bricks = np.empty(len(run_lo), dtype=np.int64)
    run_scales = np.empty(len(run_lo), dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(run_lo, run_hi)):
        vb = v_new[lo:hi]
        s_b = float(np.max(np.abs(vb))) / INT16_MAX
        s_b = s_b if s_b > 0.0 else 1.0
        w_b = np.rint(vb / s_b)
        assert_int16_range(w_b)
        w_codes[lo:hi] = w_b.astype(np.int16)
        run_bricks[i] = int(bricks_o[lo])
        run_scales[i] = s_b
    t_end = time.perf_counter()
    return dict(
        t=t, empty=False, slots_o=slots_o, w_codes=w_codes, run_bricks=run_bricks,
        run_scales=run_scales, n_owned=n_own, n_out=int(n_out),
        busy=dict(decode=t_short - t_dec, short=t_long - t_short,
                  long=t_quant - t_long, quant=t_end - t_quant),
    )


def apply_result(st, res):
    """Apply one tile's writes: velocity codes at owned rows, per-brick scales.

    Writes are disjoint across tiles (ownership is a partition, asserted in `step`), so any
    application order gives the same state.
    """
    if res["empty"]:
        return
    st.write_velocities(res["slots_o"], res["w_codes"])
    st.vel_scale[res["run_bricks"]] = res["run_scales"]


def _migrate_pass(st, cfg, c_drift, pool, timings=None, comm=None, devices=None):
    """Route the step's migrate (and the lead drift) to host serial, host pooled or device.

    Returns the pass's stats dict (the device pass adds a `migrate_device` receipt). Across
    ranks (`comm`) only the device pass has a path; `devices` are the rank's cards."""
    if cfg.migrate_backend == "device":
        from .device.migrate import drift_and_migrate_device

        return drift_and_migrate_device(
            st, c_drift, device_budget_bytes=cfg.migrate_device_budget_bytes,
            devices=_cards(cfg) if devices is None else devices, timings=timings, comm=comm)
    if pool is not None and cfg.migrate_pooled is not False:
        return drift_and_migrate_pooled(
            st, c_drift, pool, kernel=cfg.eject_kernel, window=cfg.migrate_window,
            eject_inflight=cfg.migrate_eject_inflight,
        )
    return drift_and_migrate(st, c_drift, kernel=cfg.eject_kernel)


def _repack_pass(st, cfg, timings=None):
    """Route the repack like the migrate; the device pass adds a `repack_device` receipt."""
    if cfg.migrate_backend == "device":
        from .device.repack import repack_device

        return repack_device(st, brick_slack=cfg.brick_slack, devices=_cards(cfg),
                             timings=timings)
    return st.repack(brick_slack=cfg.brick_slack)


def _cards(cfg):
    """The cards the device passes split across (`cfg.device_cards`), or None for
    one card on jax's default device."""
    if cfg.device_cards <= 1:
        return None
    import jax

    return list(jax.devices()[: cfg.device_cards])


def rank_lane_refusal(cfg):
    """Why `cfg` cannot run across ranks, or None. Across ranks the step runs the production
    device lane only: device paint and folded solve into card shards, the compiled windowed
    tile loop, and the fused migrate + repack on every step."""
    if cfg is None:
        return "across ranks the step runs the production device lane only; no configuration"
    need = dict(coarse_backend=cfg.coarse_backend == "device",
                tile_backend=cfg.tile_backend == "device",
                device_tile_jit=cfg.device_tile_jit, tile_window=cfg.tile_window,
                migrate_backend=cfg.migrate_backend == "device", fused_pass=cfg.fused_pass,
                repack_every=cfg.repack_every == 1, coarse_fold_kernel=cfg.coarse_fold_kernel,
                tile_workers=cfg.tile_workers == 1)
    bad = [k for k, ok in need.items() if not ok]
    return None if not bad else (
        f"across ranks the step runs the production device lane only; this configuration "
        f"differs in {', '.join(bad)}")


def _require_rank_lane(st, cfg, comm, what):
    """Refuse a node-local state or several ranks outside the device lane."""
    multi = comm is not None and comm.size > 1
    if st.is_whole and not multi:
        return
    why = rank_lane_refusal(cfg)
    if why:
        raise NotImplementedError(f"{what}: {why}")


def _require_rank_slabs(st, comm, decomp, what):
    """Refuse a state whose slabs are not this rank's (across ranks only)."""
    if (st.is_whole and comm.size == 1) or tuple(st.owned_slabs) == tuple(decomp.slabs):
        return
    raise ValueError(f"{what}: the state holds slabs {st.owned_slabs} but rank "
                     f"{decomp.rank} of {decomp.n_ranks} owns {decomp.slabs}")


def step(st, cfg, coeff, c_drift, collect=None, census=False, cap_shape=0, pad_shape=0,
         phase=None, tile_force=None, pool=None, coarse_parts=None, device_shapes=None,
         repack_due=False, timings=None, decomp=None, comm=None, devices=None):
    """One drift-synchronized BullFrog step. Mutates `st`; returns diagnostics.

    `coeff = (alpha, beta_over_Dmid)` from `bullfrog_float_coeffs` columns 1-2; `c_drift` is
    this step's fused drift. `census=True` enables the coarse decode census.
    `tile_force` (`(one_tile, geom)` from `make_tile_force_fn`) and `coarse_parts`
    (`forces.coarse_kernel_parts`) are built once by `run`; standalone calls build their own.
    `cap_shape` / `pad_shape` / `device_shapes` carry the previous step's shapes so they stay
    monotone (and a resumed run compiles the same programs).

    `pool` (a live `executor.TilePool`, owned by `run`) runs `tile_task` on workers and applies
    results in arrival order; `tile_force` may then be `(None, geom)`. In pool mode intra-tile
    phase boundaries do not fire and stats carry worker busy/idle/concurrency figures.

    Device lane: "device" coarse paints, decodes and solves on the cards; "device" tile runs
    the loop as one compiled program (`device.tile.tile_loop_device`) or the eager reference
    under `device_tile_jit=False`, with a single `tile_loop` phase boundary. Migrate and repack
    follow `migrate_backend`. `timings`, if a dict, collects synced per-phase walls (`tile`,
    `migrate`, `solve`); syncing moves the wall, so a timed step is a breakdown, not a cost.
    `repack_due`: under `cfg.fused_pass` the migrate and repack run fused here and
    `stats["repack"]` is filled; otherwise it is None and the caller repacks.

    `phase`, if given, is called with a boundary name after each phase, so a caller can take
    a per-phase high-water mark. It takes no payload and cannot perturb the step.

    `decomp` (`decomp.Decomp`) says which tile planes this rank and each of its cards own
    (default: from `comm`'s rank and size). `comm` (`comm.Comm`, default `SerialComm`) carries
    every exchange across ranks: every rank calls `step` with its node-local state (its
    `decomp.slabs`), in the production device lane (`rank_lane_refusal`), with a repack due.
    `devices` are the rank's cards (default the first `cfg.device_cards` jax devices).
    `stats["ranks"]` records the rank and each exchange's bytes (zeros on one rank), and under
    `"comm"` the rank's exchange ledger since the previous step's record (`Comm.take_ledger`).
    """
    import jax.numpy as jnp

    from .comm import allreduce_shapes

    ph = phase if phase is not None else _no_phase
    if comm is None:
        from .comm import SerialComm

        comm = SerialComm()
    if not (st.is_whole and comm.size == 1):
        _require_rank_lane(st, cfg, comm, "engine.step")
        if not repack_due:
            raise ValueError("engine.step across ranks runs the fused migrate + repack, so "
                             "every step must have a repack due")
    if decomp is None:
        from .decomp import Decomp

        decomp = Decomp.build(cfg, n_ranks=comm.size, rank=comm.rank)
    _require_rank_slabs(st, comm, decomp, "engine.step")
    rank_rec = dict(rank=int(comm.rank), n_ranks=int(comm.size))

    alpha_k, bcoef = float(coeff[0]), float(coeff[1])

    # --- long arm: solve once, globally, on the coarse mesh
    mesh_stats = {}
    shapes_in = device_shapes or {}
    shapes_out = {}
    # cards the device step splits across; None = jax's default device
    devs = None if devices is None else list(devices)
    if devs is None and cfg.device_cards > 1:
        import jax

        devs = list(jax.devices()[: cfg.device_cards])
    if cfg.coarse_backend == "device":
        from .device.paint import coarse_delta_cards

        # density painted and decoded on the cards; the solve transforms the shards in place
        delta = coarse_delta_cards(
            st, cfg, devices=devs, stats=mesh_stats, pad_shape=pad_shape,
            chunk_bricks=cfg.device_paint_chunk_bricks, census=census,
            shape_floor=shapes_in.get("paint"), decomp=decomp, comm=comm)
        shapes_out["paint"] = {k: int(v) for k, v in mesh_stats["coarse_jit_shapes"].items()}
        # the host path's receipt keys: each device chunk is a sub-block paint, none pooled
        mesh_stats["coarse_subblock_chunks"] = int(mesh_stats["coarse_device_chunks"])
        mesh_stats["coarse_pooled_workers"] = 0
    else:
        delta = coarse_delta_streamed(st, cfg, stats=mesh_stats, census=census,
                                      pad_shape=pad_shape, pool=pool)
    ph("coarse_paint")
    # drop the host `delta` before the solve (the solve reads only `dj`)
    dj = delta if cfg.coarse_backend == "device" else jnp.asarray(delta)
    # The realized dtype, read from the painted field: the force meshes may be preallocated
    # views whose dtype merely echoes the config.
    coarse_dtype_seen = np.dtype(
        delta[0]["delta"].dtype if cfg.coarse_backend == "device" else delta.dtype).name
    del delta
    # `coarse_force_meshes` refuses a delta/fdtype mismatch. `out=` is the pool's shm views
    # (no parent-side copy), or under the compiled device tile the card shards the tiles read
    # (`device.coarse.CardShards`), in which case `g_coarse` is shard dicts.
    if cfg.tile_backend == "device" and cfg.device_tile_jit:
        from .device.coarse import CardShards

        # one shard per card: coarse x-planes under its tile planes, COARSE_HALO each side
        plane_parts = decomp.card_planes()
        solve_out = CardShards(
            [(x0, nx, None if devs is None else devs[k])
             for k, (x0, nx) in enumerate(decomp.coarse_shards(COARSE_HALO))],
            cfg.n_coarse)
    else:
        solve_out = None if pool is None else pool.g_views()
    g_coarse = coarse_force_meshes(
        dj,
        cfg.n_coarse,
        cfg.box_size,
        "long",
        r_s=cfg.r_s,
        match=cfg.coarse_match,
        fdtype=cfg.np_coarse_dtype,
        parts=coarse_parts,
        out=solve_out,
        timings=None if timings is None else timings.setdefault("solve", {}),
        fold_kernel=cfg.coarse_fold_kernel, decomp=decomp, comm=comm, receipt=rank_rec,
    )
    del dj
    ph("coarse_solve")

    # --- the neighbours' boundary slabs this rank's tile windows read (none on one rank)
    from .device.ghost import SlabView, exchange_ghosts

    ghosts, ghost_rec = exchange_ghosts(st, decomp, comm, decomp.pad)
    rank_rec.update(ghost_rec)
    view = SlabView(st, ghosts)

    # --- membership, and the capacity one jitted program needs
    b_real = cfg._b_realized
    members = TileMembers(st, cfg.n_tile, b_real, cfg.n_brick, cfg.n_fine,
                          planes=decomp.planes)
    # member count includes arena residents, which `decode_bricks` returns; the max is over
    # every rank's tiles, so all ranks compile one shape
    counts = view.tile_member_counts(cfg.n_tile, b_real, cfg.n_brick, cfg.n_fine,
                                     planes=decomp.planes)
    cap_true = comm.allreduce(tile_capacity(counts.reshape(-1)), "max")
    # Quantize the shape: `cap_true` moves every step, and each new shape grows the XLA
    # executable cache. Geometric ladder, monotone via `cap_shape`; masked, so bitwise neutral.
    cap = capacity_shape(cap_true, rungs=cfg.cap_rungs, floor_shape=cap_shape)

    if tile_force is None:
        # standalone call: build (and compile) here; `run` builds once per run
        tile_force = make_tile_force_fn(
            cfg.n_fine, cfg.box_size, cfg.n_total, cfg.n_tile, cfg.b_fine,
            r_s=cfg.r_s, paint=cfg.paint_short, frac_bits=cfg.frac_bits,
            fdtype=cfg.np_fine_dtype,
        )
    one_tile, geom = tile_force
    cell = geom["cell"]
    ph("membership")

    # the per-step header `tile_task` runs from (sent to workers with each task)
    C = dict(
        cap=int(cap), n_tile=cfg.n_tile, n_brick=cfg.n_brick, n_fine=cfg.n_fine,
        n_coarse=cfg.n_coarse, box=cfg.box_size, coarse_cell=cfg.coarse_cell,
        cell=cell, b_real=int(b_real), alpha_k=alpha_k, bcoef=bcoef,
    )

    tile_scales, n_owned, n_overhang = [], 0, 0
    # Per-brick velocity scales: a brick's rows are all kicked in one tile, so its codes are
    # written as soon as that tile is done (no per-particle pending array).
    # Executors: serial (task + apply inline, tile order); pool (same `tile_task`, arrival
    # order); compiled device (`tile_loop_device` writes into device state and returns only
    # counts, so nothing reaches the result loop below); eager device (`tile_task_device`
    # returns the `tile_task` dict).
    tasks = [(t, members[t]) for t in members]
    if cfg.tile_backend == "device" and cfg.device_tile_jit:
        from .device.tile import tile_loop_device, tile_step_shapes

        # every compiled shape is the max over ranks: each rank compiles the one-rank programs
        shapes = allreduce_shapes(comm, tile_step_shapes(st, floor=shapes_in.get("tile")))
        shapes_out["tile"] = {k: int(v) for k, v in shapes.items()}
        # the force meshes are already on the card; each tile gathers from its shard
        if cfg.tile_window:
            from .device.window import tile_loop_windowed, window_shapes

            # one tile plane's x-slabs on the card at a time; one thread per card over its own
            # tile planes (host write-backs are disjoint core slabs)
            wsh = allreduce_shapes(comm, window_shapes(
                view, cfg.n_tile, b_real, cfg.n_brick, planes=range(*decomp.planes),
                floor=shapes_in.get("window")))
            # the destination census the fused migrate + repack is sized from
            fused_now = bool(repack_due) and cfg.fused_pass
            tile_timings = [dict() for _ in plane_parts]
            shapes_out["window"] = wsh
            wshapes = dict(shapes, window=wsh)

            def run_card(k):
                a, b = plane_parts[k]
                return tile_loop_windowed(
                    view, one_tile, C, None, members, wshapes, planes=range(a, b),
                    coarse_shard=g_coarse[k], device=None if devs is None else devs[k],
                    census=float(c_drift) if fused_now else None,
                    timings=None if timings is None else tile_timings[k])

            if devs is None:
                loops = [run_card(0)]
            else:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=len(devs)) as ex:
                    loops = list(ex.map(run_card, range(len(devs))))
        else:
            loops = [tile_loop_device(st, one_tile, C, None, members, shapes,
                                      coarse_shard=g_coarse[0])]
        n_owned = sum(int(lp["n_owned"]) for lp in loops)
        n_overhang = sum(int(lp["n_out"]) for lp in loops)
        tile_scales.extend(float(lp["vel_scale_kick_max"]) for lp in loops
                           if lp["vel_scale_kick_max"] is not None)
        tile_cards = [dict(planes=int(lp.get("planes_run", cfg.tiles_side)),
                           tiles_run=int(lp["tiles_run"]),
                           **{k: lp[k] for k in ("window_live_rows_max", "residents_staged",
                                                 "window_slabs", "wrapped")
                              if k in lp})
                      for lp in loops]
        if timings is not None and cfg.tile_window:
            timings["tile"] = {f"card {k}": t for k, t in enumerate(tile_timings)}
        ph("tile_loop")
        results = ()
    elif cfg.tile_backend == "device":
        from .device.tile import tile_task_device

        results = (tile_task_device(st, one_tile, C, g_coarse, t, bricks)
                   for t, bricks in tasks)
    elif pool is not None:
        pool.stage_step(g_coarse, C)
        results = pool.imap(tasks)
    else:
        results = (tile_task(st, one_tile, C, g_coarse, t, bricks, ph=ph)
                   for t, bricks in tasks)
    t_loop = time.perf_counter()
    busy = dict(decode=0.0, short=0.0, long=0.0, quant=0.0)
    rss = {}
    for res in results:
        n_owned += res["n_owned"]
        n_overhang += res["n_out"]
        if "worker" in res:
            rss[res["worker"]] = max(rss.get(res["worker"], 0.0), res["rss_mb"])
        if res["empty"]:
            continue
        apply_result(st, res)
        tile_scales.extend(res["run_scales"].tolist())
        for key, v_b in res["busy"].items():
            busy[key] += v_b
        if pool is None:
            ph("tile_reduce")
    loop_wall = time.perf_counter() - t_loop

    # kept so the phase trace has a stable shape (nothing accumulates across the loop)
    ph("tile_loop_end")

    if n_owned != st.n_particles:
        raise AssertionError(
            f"the tiles own {n_owned} rows against {st.n_particles} particles: ownership "
            "is supposed to be a partition, so this is a geometry error.\n"
            + _diagnose_partition(st, cfg)
        )
    if n_overhang:
        raise AssertionError(
            f"{n_overhang} rows fell outside their padded tile. Since choose_brick gained "
            "its divisibility condition the brick union is EXACTLY the padded box, so a "
            "nonzero overhang means the decomposition is wrong rather than wasteful."
        )

    # nothing to reconcile (per-brick scales); boundary kept for a stable phase trace
    ph("reconcile")

    fused_now = bool(repack_due) and cfg.fused_pass
    repack_stats = None
    if fused_now:
        from .device.fused import migrate_repack_device

        census_counts = sum(lp["census_counts"] for lp in loops)
        mt = None if timings is None else timings.setdefault("migrate", {})
        stats, repack_stats = migrate_repack_device(
            st, c_drift, census_counts, brick_slack=cfg.brick_slack,
            device_budget_bytes=cfg.migrate_device_budget_bytes, devices=devs,
            timings=mt, comm=comm)
    else:
        stats = _migrate_pass(st, cfg, c_drift, pool,
                              None if timings is None else timings.setdefault("migrate", {}),
                              comm=comm, devices=devs)
    # Receipts that each knob applied, present on every step (0 / None when inactive).
    stats["migrate_repack_fused"] = fused_now
    stats["census_slabs"] = (int(sum(lp.get("census_slabs", 0) for lp in loops))
                             if fused_now else 0)
    stats["migrate_pooled_workers"] = int(stats.get("migrate_pool", {}).get("workers", 0))
    stats["migrate_backend"] = cfg.migrate_backend
    stats.setdefault("migrate_device", None)
    # arena residency after the pass (slots are freed as bricks are rewritten, so this is
    # not the claim count); informs the choice of `arena_frac`
    for _k in ("migrate", "migrate_pool"):
        _u = (stats.get(_k) or {}).get("arena_used")
        if _u is not None:
            stats["arena_used"] = int(_u)
            break
    ph("migrate")
    # `cap` is the buffer shape, `cap_true` the max it was quantized from.
    stats.update(cap=cap, cap_true=cap_true, n_tiles=len(cfg.tiles),
                 vel_scale_kick_max=float(max(tile_scales)) if tile_scales else 1.0)
    # always recorded: pooled per-phase memory and timings are not comparable to serial
    stats["tile_workers"] = int(getattr(cfg, "tile_workers", 1))
    if pool is not None:
        b_sum = float(sum(busy.values()))
        stats["pool"] = dict(
            workers=pool.workers, wall_s=loop_wall, busy_s=busy, busy_total_s=b_sum,
            concurrency=(b_sum / loop_wall) if loop_wall > 0 else 0.0,
            idle_s=pool.workers * loop_wall - b_sum,
            rss_mb=rss,
        )
    # realized dtypes, read from the arrays rather than echoed from the config
    stats["coarse_dtype"] = coarse_dtype_seen
    stats["fine_dtype"] = str(geom["fdtype"])
    # `device_shapes`: the shapes the device programs compiled at ({} on the host lane)
    stats["coarse_backend"] = cfg.coarse_backend
    stats["tile_backend"] = (cfg.tile_backend if cfg.tile_backend == "host"
                             else ("device" if cfg.device_tile_jit else "device-eager"))
    stats["device_shapes"] = shapes_out
    if cfg.tile_backend == "device" and cfg.device_tile_jit:
        stats["device_cards"] = cfg.device_cards
        stats["tile_cards"] = tile_cards
    stats.update(mesh_stats)
    md = stats.get("migrate_device") or {}
    rank_rec.update(
        paint_ghost_planes_sent=int(mesh_stats.get("coarse_ghost_planes_sent", 0)),
        emigrant_rows_sent=int(md.get("rank_emigrant_rows_sent", 0)),
        emigrant_rows_received=int(md.get("rank_emigrant_rows_received", 0)),
        hand_off_bytes_sent=int(md.get("rank_hand_off_bytes_sent", 0)),
        # the rank's particle count changes by rows_in - rows_out in the migrate
        rows_in=int(md.get("rank_rows_in", 0)), rows_out=int(md.get("rank_rows_out", 0)))
    rank_rec.setdefault("forward_sent_bytes", 0)
    rank_rec.setdefault("inverse_sent_bytes", 0)
    rank_rec["comm"] = comm.take_ledger()
    stats["ranks"] = rank_rec
    stats["repack"] = repack_stats
    stats["timings"] = timings
    if collect is not None:
        collect(stats)
    return stats


def fused_drifts(coeffs):
    """The drift-synchronized schedule from `bullfrog_float_coeffs`.

    Returns `(lead, fused)`: a leading half-drift onto the first midpoint, then one drift per
    step, `h_k + h_{k+1}`, with `h_{K-1}` alone on the last so the endpoint matches the
    boundary form.
    """
    h = np.asarray(coeffs)[:, 0]
    return float(h[0]), np.concatenate([h[:-1] + h[1:], h[-1:]])



# ------------------------------------------------------------- checkpoints

# The config fields a resume must match: those that move numbers, as physics/geometry or as a
# buffer shape (XLA reassociates by shape). Safe to change across a resume, so excluded:
#   execution policy (tile_workers, worker_affinity, migrate_pooled, migrate_window,
#     eject_kernel, migrate_backend, migrate_device_budget_bytes, migrate_repack_fused,
#     device_y_blocks) --
#     every alternative is bitwise the serial host path;
#   layout (repack_every, chunk_bricks, brick_slack) -- both paints are integer and so
#     order-independent; checkpoints store membership, not allocation;
#   checkpoint_dir, checkpoint_every.
_FINGERPRINTED = (
    "box_size", "n_part", "n_fine", "n_coarse", "n_tile", "b_fine", "alpha",
    "paint_short", "paint_long", "frac_bits", "coarse_dtype", "fine_dtype",
    "cap_rungs", "pad_ladder", "paint_subblock",
)


def checkpoint_fingerprint(cfg, coeffs):
    """Hash of what a resume must match: `_FINGERPRINTED` config fields plus `coeffs` as bytes
    (so the cosmology, a-grid and K are covered)."""
    h = hashlib.sha256()
    fp = {k: getattr(cfg, k) for k in _FINGERPRINTED}
    # only when != 2, so order-2 fingerprints are unchanged
    if getattr(cfg, "coarse_match_order", 2) != 2:
        fp["coarse_match_order"] = cfg.coarse_match_order
    h.update(json.dumps(fp, sort_keys=True).encode())
    h.update(np.ascontiguousarray(coeffs, dtype=np.float64).tobytes())
    return h.hexdigest()


def _write_checkpoint(st, cfg, coeffs, step, cap_shape, pad_shape, gen, epoch=None,
                      device_shapes=None, timings=None, comm=None):
    """Write generation `gen` of the rolling pair; returns its directory (the step's receipt).

    `epoch` is the optional `(a_steps, cosmo)` from `run` (see `epoch_record`);
    `device_shapes` are restored on resume. `comm`: every rank calls this with its node-local
    state (`icgen.write_t9_slabs`); the provenance is the same at any rank count."""
    import math

    from . import icgen

    if comm is None:
        from .comm import SerialComm

        comm = SerialComm()
    if st.is_whole:
        n_arena = int(st.n_arena)
    else:
        # the arena a whole-box load of this checkpoint gets, not the sum of the ranks'
        if st.arena_frac is None:
            raise ValueError("a node-local state without `arena_frac` cannot record the "
                             "checkpoint's arena size")
        n_total = comm.allreduce(int(st.n_live))
        n_arena = math.ceil(n_total * float(st.arena_frac))
    prov = dict(
        kind="inexor-checkpoint",
        step=int(step),
        n_steps=int(len(coeffs)),
        cap_shape=int(cap_shape),
        pad_shape=int(pad_shape),
        device_shapes={name: {k: int(v) for k, v in s.items()}
                       for name, s in (device_shapes or {}).items()},
        n_arena=n_arena,
        fingerprint=checkpoint_fingerprint(cfg, coeffs),
    )
    prov.update(epoch_record(epoch, step))
    d = os.path.join(cfg.checkpoint_dir, f"gen{gen}")
    icgen.write_t9_slabs(st, d, provenance=prov, timings=timings, comm=comm)
    return d


def epoch_record(epoch, step):
    """The `{a, cosmology}` a checkpoint carries so its units can be chosen later.

    `epoch` is `(a_steps, cosmo)` (the full `integrate.a_grid` output, n_steps + 1 entries,
    and its cosmology) or None (writes nothing). `step` is the number of completed steps, so
    the checkpoint sits at `a_steps[step]`. Provenance only: not needed to resume, and not
    in the fingerprint (which already covers it via `coeffs`).
    """
    if epoch is None:
        return {}
    a_steps, cosmo = epoch
    step = int(step)
    if not 0 <= step < len(a_steps):
        raise IndexError(
            f"checkpoint at step {step} against an a-grid of {len(a_steps)} points "
            f"({len(a_steps) - 1} steps): the schedule and the epoch grid disagree, and "
            "recording the wrong epoch would silently mis-scale every exported velocity"
        )
    return dict(a=float(a_steps[step]), cosmology=dataclasses.asdict(cosmo))


def load_checkpoint(checkpoint_dir, cfg, coeffs, brick_slack=None, alloc_margin=0.10,
                    arena_frac=None, alloc=None, comm=None, slabs=None):
    """Newest complete checkpoint under `checkpoint_dir` -> `(st, resume)`.

    Picks the highest recorded step among generations with a manifest (removed before a
    rewrite, written last, so a torn generation is skipped and the other survives). Raises
    on a fingerprint mismatch, which would otherwise splice two runs silently. Capacity is not
    stored: `brick_slack` defaults to the config's, `arena_frac` to the checkpoint's.

    Across ranks (`comm`, default `SerialComm`), rank 0 picks the generation and every rank
    loads its own brick x-slabs `slabs` (see `icgen.load_slot_state`); the ranks' particle
    counts must add up to the checkpoint's.
    """
    from . import icgen

    if comm is None:
        from .comm import SerialComm

        comm = SerialComm()
    if comm.size > 1 and slabs is None:
        raise ValueError(f"{comm.size} ranks: each must load its own `slabs`")
    best = None
    if comm.rank == 0:
        for gen in (0, 1):
            d = os.path.join(checkpoint_dir, f"gen{gen}")
            mpath = os.path.join(d, icgen.MANIFEST)
            if not os.path.exists(mpath):
                continue
            with open(mpath) as fh:
                man = json.load(fh)
            prov = man.get("provenance", {})
            if prov.get("kind") != "inexor-checkpoint":
                continue
            if best is None or int(prov["step"]) > int(best[1]["step"]):
                best = (d, dict(prov, gen=gen), man)
    # one choice for every rank, whatever each rank's view of the filesystem
    best = comm.bcast(best)
    if best is None:
        raise FileNotFoundError(
            f"no complete inexor checkpoint under {checkpoint_dir}: either nothing ran, or "
            "every generation was interrupted mid-write (the manifest is removed first and "
            "written last, so a torn generation is deliberately unloadable)"
        )
    d, prov, man = best
    want = checkpoint_fingerprint(cfg, coeffs)
    if prov.get("fingerprint") != want:
        raise ValueError(
            f"{d} was written under a different configuration or schedule "
            f"(fingerprint {prov.get('fingerprint')} != {want}); resuming would splice two "
            "different runs together and nothing downstream would notice"
        )
    if arena_frac is None:
        arena_frac = int(prov["n_arena"]) / max(1, int(man["n_particles"]))
    st = icgen.load_slot_state(
        d,
        brick_slack=cfg.brick_slack if brick_slack is None else brick_slack,
        alloc_margin=alloc_margin,
        arena_frac=arena_frac,
        alloc=alloc,
        slabs=slabs,
    )
    n_all = comm.allreduce(int(st.n_particles))
    if n_all != int(man["n_particles"]):
        raise ValueError(
            f"the ranks loaded {n_all} particles from {d}, which holds {man['n_particles']}: "
            "their slabs do not cover the box exactly once"
        )
    return st, dict(prov)

def run(st, cfg, coeffs, collect=None, census=False, phase=None, resume=None,
        stop_at=None, allocator=None, epoch=None, timed_steps=(), comm=None, decomp=None,
        devices=None):
    """Advance `st` over a whole schedule. `coeffs` from `bullfrog_float_coeffs`.

    Returns the list of per-step stats. `phase` is forwarded to `step`; `run` adds the
    `kernel_build`, `lead_drift`, `repack` and `checkpoint` boundaries.

    `resume` (from `load_checkpoint`) and `stop_at` (stop before that absolute step) run the
    schedule in segments that compose bitwise into the uninterrupted run; always pass the
    full `coeffs` (a run over `coeffs[:n]` is a different trajectory, since drifts are fused
    across steps). With checkpointing on, `stop_at` must be a checkpoint boundary.

    `epoch = (a_steps, cosmo)` is read only by checkpoints (`epoch_record`), so an export can
    convert velocities to km/s from the directory alone. `timed_steps` names absolute steps
    whose device passes, separate repack and checkpoint are timed into `stats["timings"]`.

    Across ranks (`comm`; `decomp` defaults from its rank and size), every rank runs this
    with its node-local state (`load_checkpoint(comm=, slabs=)` or `icgen.load_slot_state(
    slabs=)`), in the production device lane, and the checkpoints are written by all ranks
    together; their bytes do not depend on the rank count. `devices` are the rank's cards.
    """
    from .decomp import Decomp

    if comm is None:
        from .comm import SerialComm

        comm = SerialComm()
    _require_rank_lane(st, cfg, comm, "engine.run")
    if decomp is None:
        decomp = Decomp.build(cfg, n_ranks=comm.size, rank=comm.rank)
    _require_rank_slabs(st, comm, decomp, "engine.run")
    cfg.validate()
    timed_steps = {int(k) for k in timed_steps}
    ph = phase if phase is not None else _no_phase
    ckpt_on = bool(cfg.checkpoint_dir) and cfg.checkpoint_every > 0
    if ckpt_on and st.ids is not None:
        raise ValueError(
            "checkpointing a state that carries ids: the t9-slabs-2 schema has no room "
            "for them, so the resumed run would silently lose particle identity"
        )
    if epoch is not None:
        # checked up front; `coeffs` is the full schedule even for a `stop_at` segment
        a_steps, _ = epoch
        if len(a_steps) != len(coeffs) + 1:
            raise ValueError(
                f"epoch grid has {len(a_steps)} points for a {len(coeffs)}-step schedule; "
                f"`integrate.a_grid` emits n_steps + 1 = {len(coeffs) + 1}. A grid that is "
                "not this schedule's would record a plausible wrong epoch on every checkpoint"
            )
    pool = None
    if cfg.tile_workers > 1:
        # workers build their own kernels; the parent needs only the geometry
        from .executor import TilePool

        pool = TilePool(st, cfg, allocator=allocator)
        tile_force = (None, tile_geom(
            cfg.n_fine, cfg.box_size, cfg.n_total, cfg.n_tile, cfg.b_fine,
            paint=cfg.paint_short, frac_bits=cfg.frac_bits, fdtype=cfg.np_fine_dtype,
        ))
    else:
        # once per run: the tile program and its kernels are resident for the run
        tile_force = make_tile_force_fn(
            cfg.n_fine, cfg.box_size, cfg.n_total, cfg.n_tile, cfg.b_fine,
            r_s=cfg.r_s, paint=cfg.paint_short, frac_bits=cfg.frac_bits,
            fdtype=cfg.np_fine_dtype,
        )
    # the long arm's kernel parts, likewise once per run (geometry only); on the cards, each
    # holds the y-pencils its share of the folded solve's kernel pass multiplies
    kernel_cards = None
    if devices is not None:
        devs = list(devices)
    elif cfg.device_cards > 1:
        import jax

        devs = list(jax.devices()[: cfg.device_cards])
    else:
        devs = None
    if cfg.kernel_on_cards:
        kernel_cards = [(lo, hi, dev) for (lo, hi), dev in zip(
            decomp.card_pencils(), [None] * cfg.device_cards if devs is None else devs)]
    coarse_parts = coarse_kernel_parts(
        cfg.n_coarse, cfg.box_size, "long", r_s=cfg.r_s,
        match=cfg.coarse_match, fdtype=cfg.np_coarse_dtype, cards=kernel_cards,
    )
    ph("kernel_build")
    try:
        lead, fused = fused_drifts(coeffs)
        # Lead half-drift onto the first midpoint; skipped on resume (the checkpoint is past
        # it). The boundary fires either way.
        if resume is None:
            _migrate_pass(st, cfg, lead, pool, comm=comm, devices=devs)
        ph("lead_drift")
        out = []
        # buffer shapes carried across steps and only grown, so few shapes are compiled
        cap_shape = 0
        pad_shape = 0
        device_shapes = {}
        k0 = 0
        if resume is not None:
            # Shapes are restored rather than re-laddered: on CPU this only saves compiles
            # (padding is masked), but on GPU a program compiled at another shape may round
            # differently, so a resume must compile the same programs.
            k0 = int(resume["step"])
            cap_shape = int(resume["cap_shape"])
            pad_shape = int(resume["pad_shape"])
            device_shapes = dict(resume.get("device_shapes") or {})
        k_end = len(fused)
        if stop_at is not None:
            k_end = min(k_end, int(stop_at))
            if k_end <= k0:
                raise ValueError(
                    f"stop_at={stop_at} against a run starting at step {k0}: this segment "
                    "would advance nothing, and a no-op that returns cleanly reads as a "
                    "completed segment to whatever submits the next one."
                )
            if ckpt_on and k_end % cfg.checkpoint_every:
                raise ValueError(
                    f"stop_at={k_end} is not a multiple of checkpoint_every="
                    f"{cfg.checkpoint_every}, so the segment would stop "
                    f"{k_end % cfg.checkpoint_every} step(s) past its last checkpoint and "
                    "throw that work away. Move the stop onto a checkpoint boundary."
                )
        # A resumed run first writes the generation it did not load, so its resume point
        # survives until a newer checkpoint is complete.
        n_ckpt = 0 if resume is None else 1 - int(resume.get("gen", 1))
        for k in range(k0, k_end):
            repack_due = bool(cfg.repack_every and (k + 1) % cfg.repack_every == 0)
            timings = {} if k in timed_steps else None
            stats = step(st, cfg, (coeffs[k][1], coeffs[k][2]), float(fused[k]), collect,
                         census=census, cap_shape=cap_shape, pad_shape=pad_shape,
                         phase=phase, tile_force=tile_force, pool=pool,
                         coarse_parts=coarse_parts, device_shapes=device_shapes,
                         repack_due=repack_due, timings=timings, decomp=decomp, comm=comm,
                         devices=devs)
            cap_shape = int(stats["cap"])
            pad_shape = int(stats["coarse_pad"])
            device_shapes = stats["device_shapes"]
            if repack_due:
                # the fused pass has already repacked, inside the migrate phase
                if stats["repack"] is None:
                    stats["repack"] = _repack_pass(
                        st, cfg, None if timings is None else timings.setdefault("repack", {}))
                ph("repack")
            # after the repack (the step's settled state); None when not written
            stats["checkpoint"] = None
            if ckpt_on and (k + 1) % cfg.checkpoint_every == 0:
                stats["checkpoint"] = _write_checkpoint(
                    st, cfg, coeffs, k + 1, cap_shape, pad_shape, n_ckpt % 2, epoch=epoch,
                    device_shapes=device_shapes,
                    timings=None if timings is None else timings.setdefault("checkpoint", {}),
                    comm=comm,
                )
                n_ckpt += 1
                ph("checkpoint")
            out.append(stats)
        return out
    finally:
        if pool is not None:
            pool.close()


def float_run_bullfrog_sync(x, v, coeffs, force_fn, box_size):
    """The drift-synchronized float reference the parity gate compares against.

    Algebraically identical to looping `float_step_bullfrog` but with the engine's fused-drift
    shape, since `mod(mod(x+a)+b)` and `mod(x+a+b)` differ in float.
    """
    import jax.numpy as jnp

    lead, fused = fused_drifts(coeffs)
    x = jnp.mod(x + lead * v, box_size)
    for k in range(len(fused)):
        g = force_fn(x)
        v = coeffs[k][1] * v + coeffs[k][2] * g
        x = jnp.mod(x + float(fused[k]) * v, box_size)
    return x, v
