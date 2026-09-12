"""The v2 engine: BullFrog PM on T9 state in slot order (M-v2-3).

`integrate.py` provides per-step coefficients and a stateless float stepper;
`forces.py` provides two independent arms; `state.py` provides the container and
the exchange. Nothing composed them, and `integrate.py:14-16` says so: the v1
drivers were deleted at the retirement and "the v2 engine's own driver lands at
M-v2-3". This is that driver.

## The step, and why it is not the reference stepper's shape

BullFrog DKD carries a velocity that is SHARED across the step boundary: the
trailing half-drift of step n and the leading half-drift of step n+1 both use
`v_{n+1}`, so they fuse exactly into one drift of `(dD_n + dD_{n+1})/2`. The
engine therefore carries state DRIFT-SYNCHRONIZED -- positions at step midpoints,
velocities at step boundaries -- and one step is

    force at the stored positions  ->  kick  ->  ONE fused drift + migrate

rather than drift, force, kick, drift. Two reasons, and the second is the one
that matters:

  1. The layout runs once per step instead of twice. It is 85.5% of the force on
     the production backend, so that is ~40% of the step.
  2. The state is quantized ONCE per step, which is the cadence D-v2-14's
     ratified accumulated-error figure was measured at. The two-drift form
     quantizes twice and that number would stop describing the engine.

Measured before it was built (JC's call): moving the quantization from the step
boundary to the midpoint costs nothing. At cdev8, K=40, the ratified bucket
against its own never-quantized reference, max |dP/P| 2.311e-4 at the midpoint
against 2.821e-4 at the boundary, both ~2 orders under D-v2-9's absolute 3e-2.

The price is that the engine is NOT bitwise `float_step_bullfrog`:
`mod(mod(x+a)+b)` and `mod(x+a+b)` are equal in R and not in float (24,867 of
100,000 components, max 2.8e-14). `float_run_bullfrog_sync` below is the matched
reference the parity gate compares against; `float_step_bullfrog` is untouched,
being an mbody port under a stability contract.

## Nothing O(N) in floats, anywhere

The long arm is solved on the coarse mesh and read per tile out of a staged
sub-block (D-v2-16 cl.3); the short arm is per tile already; the kick applies
tile-locally and writes velocities back through the layout (cl.1). The two
global `(n,3)` f64 arrays that would otherwise appear are 206 GB each at C-gh,
and deleting them is what makes that configuration runnable.
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
from .state import drift_and_migrate, drift_and_migrate_pooled

__all__ = [
    "EngineConfig", "apply_result", "checkpoint_fingerprint", "coarse_delta_streamed",
    "float_run_bullfrog_sync", "load_checkpoint", "run", "step", "tile_task",
]


def _dtype_name(x, what):
    """Normalize a mesh dtype to its name at CONSTRUCTION, numpy only.

    Accepts "float32", np.float32 or jnp.float32 alike and stores the name, so
    `EngineConfig` keeps its no-jax-at-construction rule. Failing here rather
    than at first use means a typo cannot survive as far as a cluster job.
    """
    dt = np.dtype(x)
    if dt not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"{what} must be float32 or float64, got {dt.name}")
    return dt.name


# WHICH PHASE EACH BUDGET TERM IS CHARGED TO, so a caller can add up what is
# actually live at once instead of assuming one transient at a time.
#
# That assumption is why `inexor.plan` said FITS twice for a run that did not.
# It took the LARGEST SINGLE mesh transient -- 12.9 GB of a 56 GB total at C-gh
# -- and the terms it was choosing between are not alternatives: the coarse
# solve holds its kernels, its transform workspace and its output copy at the
# same moment, which is the phase Vista 923139 died in.
#
# `resident` means live for the whole run and charged unconditionally. Every
# other value names a phase, or a TUPLE of phases for a term that spans more
# than one; terms sharing a phase are summed, and the phases are maxed over,
# because they genuinely do not overlap -- the tile loop cannot run while the
# coarse solve is running, and the phase boundary hook in `step` is what makes
# that checkable rather than asserted.
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
    # BOTH, and it is not resident: the decode allocates it inside the paint and
    # `step` drops the host name as soon as the jax copy exists, so it spans the
    # paint and the solve and then goes. It used to stay bound through the whole
    # tile loop, which is what made "resident" the right label before M-v2-6.
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

# Phases that run ONCE per run rather than once per step. Everything else is
# inside the step loop, and a budget must add those up rather than take the
# largest, because glibc does not hand freed arenas back between them: this
# project measured `malloc_trim` recovering 12-41% of a run's peak, which is
# exactly the retention that makes a step's phases accumulate into its
# high-water instead of alternating. Taking the max across a step's phases
# would be the tidier model and it would have made the c-gh bound SMALLER,
# which is the direction every wrong call here has already gone.
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
    ):
        self.box_size = float(box_size)
        self.n_part = int(n_part)
        self.n_fine = int(n_fine)
        self.n_coarse = int(n_coarse)
        self.n_tile = int(n_tile)
        self.b_fine = int(b_fine)
        self.alpha = float(alpha)
        # D-006 on BOTH arms. The coarse paint's integer twin was a named
        # deliverable (D-v2-16 cl.2); the tiled short arm had the same defect and
        # no document named it, which M-v2-3's planning session found. The layout
        # reorders particles every step, so an order-dependent primal paint is
        # not reproducible even on one machine.
        self.paint_short = str(paint_short)
        self.paint_long = str(paint_long)
        self.frac_bits = int(frac_bits)
        self.chunk_bricks = int(chunk_bricks)
        # D-v2-19 clause 3: frozen capacity fails at EVERY granularity, so the
        # repack is required rather than an optimization. Built without it, the
        # engine hit the D-007 refusal with a full arena by step 10 at `smoke`.
        self.brick_slack = float(brick_slack)
        self.repack_every = int(repack_every)
        self.eject_kernel = str(eject_kernel)
        self.migrate_eject_inflight = (
            None if migrate_eject_inflight is None else int(migrate_eject_inflight))
        # M-v2-4. TWO knobs, defaulting independently, deliberately NOT coupled:
        # the coarse mesh is the gated arm (it grows with the box and is the
        # binding resident term at hero scale) and the fine tile mesh is
        # measured (P^3, box-independent). Moving one must move one error term
        # or the gate's reading is unattributable -- which is also why
        # `fine_dtype` does not follow `coarse_dtype` by default: setting one
        # knob would then silently set two.
        self.coarse_dtype = _dtype_name(coarse_dtype, "coarse_dtype")
        self.fine_dtype = _dtype_name(fine_dtype, "fine_dtype")
        # Rungs per octave for the per-tile buffer SHAPE ladder (M-v2-6 Stage 0).
        # `cap` moves every step, so an unquantized shape leaks one XLA executable
        # family per step. A knob rather than a constant because it trades padded
        # rows against retained executables and the balance is machine-dependent:
        # more rungs means less padding and more compilations.
        self.cap_rungs = int(cap_rungs)
        # The A/B knob for the SECOND shape ladder, and it exists only so the
        # coarse chunk buffer can be turned back to its pre-fix behaviour with
        # `cap` left on its ladder in both arms. Moving `cap_rungs` would move
        # both ladders at once and the reading would be unattributable, which is
        # the same reason `fine_dtype` does not follow `coarse_dtype`.
        # False is NOT an operating point: it is the arm whose slope the fix is
        # measured against.
        self.pad_ladder = bool(pad_ladder)
        # Stage 2c: paint each chunk into a coarse sub-block instead of a full
        # mesh. Bitwise-neutral by the associativity of integer addition (the
        # streamed-vs-monolithic pin is the regression); False = the A/B arm
        # that restores the full-mesh-per-chunk behaviour, the same pattern as
        # `pad_ladder`.
        self.paint_subblock = bool(paint_subblock)
        # W2: the tile-loop executor. 1 = the serial loop, exactly as before
        # (no-worse-defaults until the pooled gate passes on the target
        # machine); > 1 = the process pool in `executor.py`, which drives the
        # SAME `tile_task` from workers -- canary C2 measured 10.4x at W=8 on
        # gg with bitwise identity on three architectures (Vista 913729).
        # `worker_affinity` pins each worker to a disjoint core set BEFORE jax
        # imports there: XLA-CPU sizes its spin pool by VISIBLE cores and
        # ignores every thread env var, and the un-pinned pool measured walls
        # GROWING with W (antares 466); worth 20-49% on gg. A knob so the
        # effect stays measurable, not because off is ever an operating point.
        self.tile_workers = int(tile_workers)
        self.worker_affinity = bool(worker_affinity)
        # Idle-half Stage 2: route `drift_and_migrate` through the SAME pool,
        # workers writing brick payloads and the parent replaying the arena
        # interleave (state.drift_and_migrate_pooled). TRI-STATE, because the
        # verdict (C14, Vista 918684: bitwise on three legs, migrate 26.03 ->
        # 3.78 s/step at cgh64 W=16) licenses pooling by default but
        # `tile_workers` defaults to 1, and a bare True would make every
        # single-process EngineConfig refuse at validate():
        #   None  = AUTO, the default -- pooled wherever a pool exists, serial
        #           where one does not. No config has to name the knob to get
        #           the verdict's win.
        #   True  = REQUIRE a pool; still refuses at validate() without one, so
        #           an explicit request that cannot apply is never silently
        #           downgraded to serial.
        #   False = force serial even with a pool. This is the A/B baseline arm
        #           and every serial reference MUST name it rather than lean on
        #           the default, which no longer means serial.
        # `migrate_window` bounds the scratch slots in flight; None = sized to
        # feed the workers (see the driver's docstring for the floor).
        self.migrate_pooled = None if migrate_pooled is None else bool(migrate_pooled)
        self.migrate_window = None if migrate_window is None else int(migrate_window)
        # M-v2-6 Stage 4(b): checkpoint after every `checkpoint_every` steps into
        # `checkpoint_dir`, alternating two generations. Cadence is in STEPS and
        # never in wall-clock, because at full scale one step is 37 min at 2048^3
        # and 7.75 h at 4096^3, so a step IS the granularity -- there is no
        # coherent state between two of them to write (positions sit at
        # midpoints, velocities at boundaries). 0 disables, the `repack_every`
        # idiom. Checkpointing is inert without a directory rather than a
        # refusal, because the default config has no directory and every
        # single-process caller would otherwise start raising; the receipt in
        # the per-step stats is what proves it applied.
        self.checkpoint_dir = None if checkpoint_dir is None else str(checkpoint_dir)
        self.checkpoint_every = int(checkpoint_every)

    @property
    def np_coarse_dtype(self):
        return np.dtype(self.coarse_dtype)

    @property
    def np_fine_dtype(self):
        return np.dtype(self.fine_dtype)

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
    def r_s(self):
        """The split scale. `alpha = r_s / d_coarse` is the dimensionless knob,
        so C-dev results transfer to configurations where cells do not."""
        return self.alpha * self.coarse_cell

    def mesh_bytes(self):
        """Per-step MESH anatomy in bytes, as a function of the two dtypes.

        **Deliberately not folded into `bytes_per_particle`.** That reports a
        per-particle budget and the mesh is not per-particle; a mesh term
        divided by N would shrink as the box grows, which is the opposite of
        what the coarse mesh does. The `scaffold=0.0` precedent there is the
        same instinct -- a term that vanishes from a table is indistinguishable
        from one that was never counted.

        This exists because that is exactly what happened. D-v2-16 clause 3
        quoted 12.9 GB at C-gh and 103 GB at C-hero; those are the f32 numbers
        (`v4_architecture_record.md` item 6 says so), and the shipped code was
        f64 throughout, so the engine paid 25.8 and 206 GB from the freeze until
        M-v2-4. Nothing in the package could have shown that, because nothing
        counted the mesh at all. `test_the_coarse_meshes_match_the_ratified_
        budget` is the check that would have.

        Terms are labelled `resident` (live simultaneously during the tile loop)
        or `transient` (peak while a phase runs). The kernel BUILD stays f64 at
        either dtype -- it is the precision island -- so it does not shrink, and
        it is reported separately rather than hidden inside the kernel term.
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
        # EVERY FINE-ARM TERM IS PER WORKER, and counting them once is what let
        # 923313 die in a phase this priced at 1.4 GB. In pool mode the parent
        # builds NO tile kernels at all -- `run` hands it `(None, tile_geom(...))`
        # and each worker builds its own triple and runs its own tile -- so the
        # node holds W copies of the kernels, W builds, and W tile working sets,
        # not one. Serial runs have exactly one of each, which is what `max(W, 1)`
        # gives, so no existing measurement moves.
        #
        # This is the same fault as the pool's startup footprint being in no
        # table until it was measured, and the same one the umbrella note about
        # parent RSS missing pool workers is about: ru_maxrss sees ONE process,
        # so an instrument reading the parent cannot see this term AT ALL and
        # the budget has to carry it by construction.
        w = max(int(self.tile_workers), 1)
        return dict(
            # --- coarse, transient
            coarse_accumulator=cells * 8,          # int64 host, dtype-independent
            coarse_decode_slab=slab * nc * nc * 8,  # one f64 slab (M-v2-4)
            coarse_kernel_build_f64=3 * half * 8,   # k2_true/k2_safe/fac, the island
            # ONE component at a time, matched: `k = pref * ik` and `k * mf` are
            # live together and nothing else is. It was three components at once
            # until M-v2-6 hoisted the build out of the step; the whole group
            # then measured 60.01 B per half-grid element per step against the
            # 16.01 it costs now (flat to 0.2% over 7.9x in `half`, n_coarse
            # 128/192/256), and that 44 B/half is 23.7 GB per step at C-gh.
            # THE FACTORIZED SOLVE'S TERMS (the monolithic ones they replace are
            # in git history; `coarse_kernels` at 2 complex half-grids and
            # `coarse_fft_workspace` at 3). The record's rule was that a budget
            # must not be edited to match a measurement of code that has not yet
            # replaced the code it prices -- the factorized transform IS now the
            # default, so these price what actually runs.
            #
            # HOST-RESIDENT, all three: `_coarse_solve_factorized` keeps the
            # spectrum in numpy and only planes cross to a device. That is not an
            # implementation detail, it is the property the whole design rests
            # on, and it is why these carry the "host" placement rather than
            # being sharded across the cards.
            coarse_spectrum=half * 2 * cw,        # the forward's output, held
            coarse_solve_work=half * 2 * cw,      # per component: copy AS multiply
            # the per-slab kernel and its product. `2 *` because the kernel slab
            # and the `spec * k` temporary are live together; slab thickness is
            # an outer loop bound, so this is the whole cost of the kernel that
            # used to be a full complex half-grid.
            coarse_kernel_slab=2 * slab * nc * (nc // 2 + 1) * 2 * cw,
            # THREE complex half-grids, not one, and this is a DERIVATION rather
            # than a measurement: `dk`, the device copy `jnp.asarray(k)` makes of
            # the host kernel, and their product, all live while `irfftn` runs.
            # It is an upper bound on the group -- XLA is free to fuse the
            # multiply into the transform, and nothing in-process can see
            # whether it did (`v2_m6_host_bytes.py` is blind to jax buffers and
            # `memory_stats()` is None on CPU). Modelled high on purpose: this
            # is a budget, and the failure that costs a node is the one where a
            # term was left out.
            # What the transform actually holds ON a device, which is the term
            # the monolithic form could not make small. MEASURED at 117.5 MB at
            # 2048^3 against a 34.4 GB spectrum -- 0.0034x, the witness D1
            # existed to get. Modelled at 16 planes rather than the ~8 that
            # measurement implies, on the same "a budget is not a best case"
            # posture as the terms above.
            coarse_device_planes=16 * nc * (nc // 2 + 1) * 2 * cw,
            # --- coarse, RESIDENT for the whole run: what the once-per-run
            # build leaves behind (`forces.coarse_kernel_parts`). Both are real
            # half-grids at the coarse dtype; the complex kernels are not kept.
            # `pref = (fac / k2_safe).astype(fdtype)` was in neither arm's
            # accounting until M-v2-6, and the match factor was in NO table at
            # all -- `cic_match_factor` is called on every coarse solve and the
            # engine always passes `match`, so it was 4.30 GB of C-gh that no
            # budget had ever named.
            coarse_kernel_pref=half * cw,
            coarse_match_factor=half * cw,
            # --- coarse, resident through the tile loop
            coarse_delta=cells * cw,
            coarse_force_resident=3 * cells * cw,
            # ONE component, not three. The old form returned a list
            # comprehension of jax meshes and the engine then built a numpy copy
            # of each; a comprehension rebinds only after it completes, so all
            # six were live together -- 25.8 GB at C-gh. `coarse_force_meshes`
            # now solves one component at a time straight into the caller's
            # buffers (the pool's shm views), so the transient is one mesh and
            # the pool's own copy disappears with it. M-v2-6 measured
            # `coarse_solve` at 14.19 MB (cdev8) and 110.89 (cdev) against 6.29
            # and 50.33 for one copy; the coarse_div arm moved it 7.3x for an 8x
            # change in cells, which is what identifies this as a cells term
            # rather than a particle one.
            coarse_force_copy_transient=cells * cw,
            # --- fine, resident through the tile loop
            tile_kernels=w * 3 * phalf * 2 * fw,
            # THE BUILD, which the tile arm never counted though the coarse arm
            # always did. `split_kernels` holds `k2_true`, `k2_safe` and `fac` as
            # full f64 half-grids plus `pref` at the fine dtype while it
            # materialises the three complex kernels, so the peak of a build is
            # 80*phalf at f64 where this dict modelled 48. The `ik_j` are LOW-RANK
            # broadcasts ((nx,1,1) etc.) and cost nothing, which is why the factor
            # is 5/3 and not larger.
            #
            # Measured, and the reason this is a derivation rather than a fit:
            # `membership` (the phase that calls `make_tile_force_fn`) reads 85.60
            # MB at cdev8 against 51.12 modelled and 1319.45 at cdev against
            # 791.28 -- ratios of 1.675 and 1.667 against a derived 5/3. The
            # config ladder alone could not have found this: particles, coarse
            # cells and phalf scale together on every rung, and it took cdev's
            # 15.48x phalf against 8.00x particles to separate them.
            tile_kernel_build_f64=w * 3 * phalf * 8,
            tile_kernel_pref=w * phalf * fw,
            # --- fine, transient per tile
            tile_workspace=w * (pcells * (fw + 4 + 3 * fw) + phalf * 2 * fw),
        )

    # Eject output, MEASURED per row of the slab a worker is handed, numpy
    # domain and flat over an 8x change in slab rows (cdev8 / cdev):
    #
    #     numpy kernel   36.20 / 34.81 B/row
    #     jax kernel    142.77 / 129.35 B/row
    #
    # The jax arm's figure is its NUMPY allocations; its device buffers are
    # invisible to tracemalloc and sit on top, so both are floors and the jax
    # one is the looser of the two. Both kernels are bitwise (record 5s, and
    # re-confirmed alongside these numbers), so the 3.7x is a pure memory/wall
    # trade with no correctness content: numpy ejects measured 1.6-1.9x slower
    # on arm64 CPU.
    EJECT_BYTES_PER_ROW = {"numpy": 35.0, "jax": 129.0}

    # An insert's own buffers, MEASURED the same way and flat to 0.1% over the
    # same 8x: 50.28 / 50.25 B/row. Its INPUTS are free -- it reads slices of
    # the shm scratch, which are views -- so this is the O(brick) working set
    # alone, and it is why bounding ejects does not bound the pass on its own.
    INSERT_BYTES_PER_ROW = 50.25

    # The SERIAL pass's whole-phase coefficient, measured as `migrate`'s own
    # increment at cdev8 and cdev (181.7 and 198.9 B per slab-particle). It
    # covers eject AND insert and the 2r+1 slabs the serial schedule stages.
    SERIAL_MIGRATE_BYTES_PER_ROW = 190.0

    def _migrate_b_per_row(self):
        """Bytes per slab-row the migrate holds, for THIS execution policy.

        **The pooled path holds one slab per WORKER and the serial one holds
        2r+1 in total**, and the model had no worker count in it at all -- it
        carried the serial coefficient whatever was running. At c-gh that priced
        a pass at 12.75 GB which measured over 91: eight jax ejects of a 67M-row
        slab are 69 GB between them, and Vista 923341 died there, in the LEAD
        DRIFT, before reaching a single step.

        The insert side is NOT separately measured and is not added here. This
        is therefore a floor, and it is a floor built from the eject alone.
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
        # WORST CASE, and the worst case is the point of a ceiling: every worker
        # not held back from an eject is assumed to be running one. With the
        # bound in place the rest can only be inserts, which are 2.6x cheaper
        # but not free -- so a bound of E over W workers is E ejects PLUS W-E
        # inserts, never E alone. Modelling it as E alone is how a knob comes to
        # look like it solved a problem it only moved.
        e = w if self.migrate_eject_inflight is None else min(
            self.migrate_eject_inflight, w)
        return per_row * e + self.INSERT_BYTES_PER_ROW * max(w - e, 0)

    def mesh_phase(self):
        """`MESH_PHASE`, adjusted for how THIS config actually runs.

        The module dict describes the serial engine, where `run` hoists
        `make_tile_force_fn` and the kernel build lands at its own boundary
        before the loop. **In pool mode the parent builds no tile kernels at
        all** -- it is handed `(None, tile_geom(...))` and each worker builds
        its own triple on its first task, which happens INSIDE the tile loop.

        That distinction is not bookkeeping: `kernel_build` is a once-per-run
        phase and is held apart from the in-step sum, so leaving the tile build
        there would drop W builds out of the budget for the very step the run
        keeps dying in. A phase map that cannot express "who runs this, and
        when" is a phase map that quietly excuses the largest term.
        """
        m = dict(MESH_PHASE)
        if self.tile_workers > 1:
            m["tile_kernel_build_f64"] = "tile_loop"
            m["tile_kernel_pref"] = "tile_loop"
        return m

    def step_bytes(self, n_particles, n_rows=None, cap=None):
        """Per-step HOST terms that scale with PARTICLES, not with the mesh.

        The companion to `mesh_bytes`, and it exists for the same reason: nothing
        counted these either. M-v2-6 Stage 0 measured the engine's end-to-end peak
        for the first time and found 8.2 GB at cdev8 where state plus
        `mesh_bytes` plus the tile buffers modelled ~0.25 GB, so a planner built
        only on `mesh_bytes` would have sized C-gh and been wrong by a factor of
        thirty. A term that is not in the table cannot be traded against anything.

        Terms, each pointing at the line that allocates it:

        `brick_scales` -- one f64 per brick, the array that REPLACED
        `kick_pending`. That term held `(slots int64, v_new f64)` for every owned
        row until a global velocity scale could be reconciled -- 32 B/p, 274.9 GB
        at C-gh, the largest single term in the configuration and the one the
        module docstring wrongly claimed did not exist ("Nothing O(N) in floats,
        anywhere"). Per-brick scales delete the wait that forced it: a brick's
        rows are all kicked in one tile, so its scale is known immediately. What
        is left is `n_bricks * 8` -- 16.8 MB at C-gh, five orders down, and it is
        RESIDENT rather than per-step, so it is carried in the state table
        instead. The line stays here reading zero because a term that vanishes
        from a table is indistinguishable from one that was never counted.

        `repack_scratch` -- `SlotState.repack` allocates `zeros_like` of `off`
        and `w` while the originals stay live. The DERIVED figure was 9 B per
        row (3 for `off` plus 6 for `w`); MEASURED it is **11.1 B/row**, flat to
        2.6% across 64x in particle count (262k / 2.1M / 16.8M -> 11.35 / 11.09
        / 11.06, `scripts/v2_m6_repack_bytes.py`). The extra ~2 B/row is the two
        int64 occupancy arrays over `n_buckets` plus the sort, and it carries
        because `n_buckets` and `n_rows` keep their ratio up the config table.
        That was ~115 GB at C-gh. **Rewritten in place (M-v2-6) it measures 2.1
        B/row** -- 2.135 and 2.066 at 2.1M and 16.8M particles -- so ~22 GB, a
        5.3x reduction, and the term stops being the binding one. What remains
        is almost entirely the two int64 occupancy arrays, which are per-BUCKET;
        narrowing them to the index dtype would roughly halve it again and is
        not done here because it has not been measured.

        **D-v2-19 clause 3's in-place form does NOT fix it, and the clause's own
        number is what hid that.** `BrickPackedLayout.repack` reports
        `scratch_bytes` of 0.13-0.52 MB "independent of N", but that counts only
        its two chunk buffers (`layout.py:662,670`); it also allocates `live`,
        `parts` and `final` at one row each. Measured **39.4 B/row**, flat over
        the same 64x, so porting it would multiply this term by 3.55x. The
        reported figure is not merely low, it has the wrong SHAPE: a fixed
        buffer divided by a growing row count reads as N-independent while the
        real cost is linear. The clause's REASONING -- a repack is a monotone
        rearrangement, not a sort -- is sound and is what licenses a genuinely
        O(brick) implementation. The reference simply is not one.

        `tile_buffers` -- the per-tile host working set, `cap`-sized. Needs a
        measured `cap`; omitted when not supplied rather than guessed.
        """
        n = int(n_particles)
        rows = int(n_rows) if n_rows is not None else int(round(n * 1.21))
        # MIGRATION STAGING IS N^(2/3), NOT N, and that is the whole reason it is
        # worth a line. `state.drift_and_migrate` walks x-slabs and releases a
        # staged slab as soon as every write that could reach it has happened, so
        # the rows in flight are a handful of SLABS rather than the state: a slab
        # is N / bricks_per_side, and bricks_per_side grows as N^(1/3), so this
        # term grows as N^(2/3). Carrying it as an O(N) term would overstate it by
        # 8x at C-gh.
        #
        # The coefficient is measured, the SHAPE is derived, and two configs agree
        # on the coefficient which is what tests the shape: 181.7 B per
        # slab-particle at cdev8 and 198.9 at cdev, an 8x change in N (`migrate`
        # own-increment 47.62 and 208.58 MB, M-v2-6 host-byte instrument).
        # `lead_drift` is the SAME function called once before the loop and runs
        # 77.3 / 92.7 B on the same basis; it is not added here because it
        # completes before the step's peak and so is never co-resident with it.
        nb = max(1, self.n_fine // self.n_brick)
        out = dict(
            kick_pending=0,
            # 0.49 MEASURED (2.135/2.066 -> 0.490/0.490 at 2.1M and 16.8M
            # particles, `scripts/v2_m6_repack_bytes.py`), after the per-bucket
            # arrays came out: 11.1 out of place -> 2.1 in place -> 0.49 once
            # `repack` stopped casting `occupancy` to int64 twice, stopped
            # building an n_buckets bincount for the arena, and built its output
            # at the index dtype. This coefficient only holds while n_buckets and
            # n_rows keep their ratio, which they do up the config table.
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
            # a knob that cannot apply must refuse, not silently run serial.
            # `is True` and not truthiness: None is AUTO and falls back to
            # serial by design, an EXPLICIT True cannot.
            raise ValueError(
                f"migrate_pooled needs a pool: tile_workers is {self.tile_workers}"
            )
        self._refuse_f64_without_x64()
        return True

    def _refuse_f64_without_x64(self):
        """An f64 mesh without x64 is not an f64 mesh (M-v2-4).

        The library never enables x64 and every caller opts in, so this refuses
        nothing that works today. What it prevents is the one configuration that
        lies about itself: asking for f64, silently getting f32, and reading the
        result as an f64 reference. That is the reference arm of M-v2-4's own
        gate, where a silent degradation would not make the comparison fail --
        it would make it read ZERO, which is a pass.

        In `validate()` rather than in `step()` so an ad-hoc single step stays
        possible; `run()` calls `validate()`.
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

    Returns `(cell_origin (3,), cell_span (3,))`, or None when the run is not
    a cuboid -- and None simply routes that chunk to the full-mesh paint, so
    this is a fast path with a bitwise-identical fallback, never a
    requirement. A run IS a cuboid exactly when the chunk length divides the
    brick grid cleanly: whole i-planes (nb^2 | L), whole j-rows within one
    plane (nb | L and L | nb^2), or a fraction of one row (L | nb). All three
    also make every chunk the SAME shape (L | nb^3, so no tail chunk), which
    is what keeps the sub-block paint on one XLA shape.
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

    Cheap (one rint + compare over the chunk's real rows), OUTSIDE the jit --
    which is exactly what the fine arm's in-jit guard could not be (D-v2-21).
    A violation means the cuboid derivation is wrong, and the failure mode it
    prevents is a stencil corner wrapping to the far side of the block
    SILENTLY, so this refuses rather than falls back.
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


def coarse_delta_streamed(st, cfg, stats=None, census=False, pad_shape=0, pool=None):
    """delta on the coarse mesh, accumulated brick by brick.

    Integer addition is associative, so a chunked accumulation is **bitwise**
    what a single call over every position produces -- that is a property to
    test, not to hope for, and it is the reason the coarse paint must be the
    integer one before the paint can be streamed at all. With the f64 paint the
    chunking would change the answer.

    `stats`, if a dict, receives `coarse_peak_int` -- the max accumulated cell
    sum, which the int32 refusal below already computes, so it is free.

    `pad_shape` is the previous step's chunk shape, carried forward so the shape
    is monotone across a run for the same reason `cap` is; see the comment on
    `pad` below and `forces.capacity_shape`.

    `pool`, if given, runs each chunk's decode + sub-block paint on the tile
    pool's workers (W2 Stage C); the ACCUMULATION stays here, where integer
    associativity makes arrival order bitwise the serial order. `census=True`
    and the full-mesh A/B arm (`paint_subblock=False`) route serial regardless,
    so the gate instruments never read a pooled mesh.

    `census=True` additionally counts cells whose integer sum is NOT exactly
    representable in f32 (`coarse_cells_inexact_f32`). It is OPT-IN because it
    costs two extra passes over the mesh and is a gate instrument, not a
    production need. **It counts round-trip failures rather than cells above
    2^24**: `< 2^24` is sufficient for exactness, not necessary, and
    `5000 * 2^12 = 625 * 2^15` is exact at 2.05e7 -- a magnitude threshold
    overstates the loss, sometimes by a lot.
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
    # ONE shape for every chunk. A varying row count recompiles per chunk, which
    # profiled at 2.91 s of a 13.21 s step -- the same trap as the long-range
    # read. Padding is masked, not filled: an unmasked pad row would add mass.
    #
    # ONE shape ACROSS STEPS too, which the max alone does not give: occupancy
    # shifts every step, so `max(rows)` took ten distinct values over ten steps
    # at the smoke config (against one for `cap` since Stage 0 put that on the
    # ladder), and every buffer keyed on it keys a new XLA shape whose executable
    # is cached for the life of the process. That is the leading measured cause
    # of the run peak's +110 MB/step at cdev (M-v2-6 Stage 0b, antares 446: the
    # growth is linear over 15 steps, unsaturating, and survives malloc_trim, so
    # it is live memory rather than allocator slack). Same ladder, same rungs
    # knob, and the same masking argument makes it bitwise neutral.
    pad_true = int(max(rows)) if rows else 0
    pad = (
        capacity_shape(pad_true, rungs=cfg.cap_rungs, floor_shape=pad_shape)
        if cfg.pad_ladder
        else pad_true
    )
    # SUB-BLOCK PAINT (Stage 2c). The full-mesh form allocated and accumulated
    # an n_coarse^3 mesh PER CHUNK -- chunks ~ n_bricks ~ N and per-chunk mesh
    # ~ n_coarse^3 ~ N, a superlinear term measured at 20.4% of the cgh64 step
    # (job 464) and growing. A chunk is a brick-major run, i.e. a spatial
    # cuboid, so its TSC footprint is a bounded sub-block: paint there, then
    # add the block into the accumulator through per-axis wrapped indices.
    # Integer addition is associative and the sub-block contributions are
    # bit-identical to the full paint's (weights global, index rebased), so
    # the accumulated mesh is BITWISE unchanged -- the streamed-vs-monolithic
    # pin is the regression. An axis whose span+3 would reach n_coarse runs
    # full-axis instead, which keeps the scatter indices unique per axis (a
    # repeated index under fancy-indexed += would drop adds).
    nb_side = cfg.n_fine // cfg.n_brick
    coarse_cell = cfg.box_size / float(n)
    n_sub = 0
    pooled_workers = 0
    if pool is not None and cfg.paint_subblock and not census:
        # W2 Stage C: chunks on the workers, integer accumulation here. The
        # serial loop below is unchanged and remains the oracle the
        # streamed-vs-monolithic pin reads.
        pool.stage_coarse(dict(pad=int(pad), n_coarse=n, box=cfg.box_size,
                               frac_bits=cfg.frac_bits,
                               chunk_bricks=cfg.chunk_bricks, nb_side=nb_side))
        tasks = [(gi, np.asarray(gg, dtype=np.int64))
                 for gi, (gg, m) in enumerate(zip(groups, rows)) if m]
        for res in pool.imap_coarse(tasks):
            if res["empty"]:
                continue
            ax = [(np.arange(int(res["extent"][a]), dtype=np.int64)
                   + int(res["origin"][a])) % n for a in range(3)]
            mesh[np.ix_(*ax)] += res["sub"]
            n_sub += 1
        pooled_workers = pool.workers
        groups = []  # the serial loop below must not run the chunks again
    for gi, (gg, m) in enumerate(zip(groups, rows)):
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
    out, peak, inexact = _delta_from_accumulated(mesh, cfg, census=census)
    if stats is not None:
        # both, for the same reason `cap`/`cap_true` are both reported: one hides
        # the padding cost, the other hides the shape churn, and the churn leaked
        stats["coarse_pad"] = pad
        stats["coarse_pad_true"] = pad_true
        stats["coarse_peak_int"] = peak
        # how many chunks took the sub-block path: an A/B whose knob did not
        # apply must be readable as such (a knob must prove it applied) --
        # same rule for the pool (0 = the serial path painted this mesh)
        stats["coarse_subblock_chunks"] = n_sub
        stats["coarse_pooled_workers"] = pooled_workers
        if census:
            stats["coarse_cells_inexact_f32"] = inexact
            stats["coarse_exact_decode_ok"] = inexact == 0
    return out


def _delta_from_accumulated(mesh, cfg, census=False):
    """(delta, peak, inexact) from the accumulated int64 coarse paint.

    Shared by the host streamed paint and the device one (`device.paint`), so
    the two cannot decode differently: whatever accumulated the integers, this
    is the one place they become a density. `inexact` is the f32 round-trip
    census count when `census`, else 0.
    """
    n = cfg.n_coarse
    peak = int(np.abs(mesh).max())
    if peak >= 2**31:
        raise ValueError(
            f"the accumulated coarse paint reached {peak}, past int32. "
            "Lower frac_bits -- this is D-007-class corruption, not imprecision."
        )
    # SLABBED DECODE (M-v2-4). The old form was
    #     counts_from_int(mesh.astype(np.int32), ...) / mean - 1.0
    # which materializes an int32 copy of the whole mesh plus two full f64
    # meshes on top of the int64 accumulator -- about 30 GB at C-gh, all of it
    # transient, none of it counted anywhere. Slabbing makes the transient one
    # slab (0.27 GB at C-gh) and lets the result land straight at the requested
    # dtype. Every operation is elementwise, so the slabbed result is
    # bit-identical to the whole-array form; `tests/test_engine.py` pins the
    # streamed paint against the monolithic one and would catch it if not.
    #
    # The decode stays f64 and the narrowing happens after `- 1.0`; see
    # `painting.density_tsc` for how little that buys and why it is kept anyway.
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

    Failure path only. The bare count ("16777215 against 16777216") is
    unattributable and cost three wrong explanations before this existed: a
    tile-local float gap, a position-vs-storage mismatch, and a lost row. Each was
    consistent with the count and none was the cause. What actually distinguishes
    them is WHICH rows are unclaimed and what is unusual about them, and the state
    knows.
    """
    b_real = cfg._b_realized
    nb = cfg.n_fine // cfg.n_brick
    claimed = np.zeros(st.off.shape[0], dtype=np.int32)
    misaligned = []
    for t in cfg.tiles:
        members = st.tile_bricks(t, cfg.n_tile, b_real, cfg.n_brick, cfg.n_fine)
        slots, _, _ = st.decode_bricks(members)
        counts = [st.brick_member_count(b) for b in members]
        if int(np.sum(counts)) != len(slots):
            misaligned.append((tuple(int(q) for q in t), int(np.sum(counts)), len(slots)))
            continue
        brick_of_row = np.repeat(np.asarray(members, dtype=np.int64), counts)
        own = owned_mask_from_bricks(brick_of_row, t, cfg.n_tile, cfg.n_brick, nb)
        # np.add.at, NOT `claimed[idx] += 1`. Fancy-index += increments a repeated
        # index ONCE, so the buffered form cannot see the very duplication this is
        # looking for -- the first version of this diagnostic reported zero
        # double-claims on a state whose distinct-slot count was short by one.
        np.add.at(claimed, np.asarray(slots)[own], 1)
    # counts vs DISTINCT slots: `SlotState.check` compares counts (state.py:587),
    # so an aliased slot passes it -- occupancy still sums to n_particles while
    # one slot is reachable through two bricks and the reachable SET is short.
    seen = np.zeros(st.off.shape[0], dtype=np.int64)
    total_decoded = 0
    for b in range(st.n_bricks):
        s = np.asarray(st.decode_brick(b)[0])
        total_decoded += len(s)
        np.add.at(seen, s, 1)
    live = seen > 0
    aliased = np.nonzero(seen > 1)[0]
    unclaimed = np.nonzero(live & (claimed == 0))[0]
    twice = np.nonzero(claimed > 1)[0]
    # THREE censuses that must be identical, printed rather than inferred. Job 440
    # showed decoded == distinct == n_particles - 1 with no aliasing, so a row is
    # unreachable while `check` passes -- and `check` sums OCCUPANCY. Whichever
    # pair disagrees localizes the defect: occupancy vs member_count is a counting
    # bug, member_count vs decode is a span/decode bug.
    p3 = st.buckets_per_brick
    occ = st.occupancy.astype(np.int64)
    occ_per_brick = occ.reshape(st.n_bricks, p3).sum(axis=1)
    mc_per_brick = np.array(
        [st.brick_member_count(b) for b in range(st.n_bricks)], dtype=np.int64
    )
    dec_per_brick = np.array(
        [len(st.decode_brick(b)[0]) for b in range(st.n_bricks)], dtype=np.int64
    )
    bad = np.nonzero((occ_per_brick != mc_per_brick) | (mc_per_brick != dec_per_brick))[0]
    detail = [
        (int(b), int(occ_per_brick[b]), int(mc_per_brick[b]), int(dec_per_brick[b]))
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
        bricks_with = [b for b in range(st.n_bricks)
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

    Reads `st` and writes NOTHING: every write the tile owes comes back in the
    returned dict, and `apply_result` is the only place they land. This is the
    executor seam -- the serial loop in `step` calls task and apply inline, and
    the pool executor distributes the task over workers -- so a bitwise gate
    between the two compares two executors of ONE function rather than two
    codebases. Promoted from `scripts/v2_m6_c2_pool.py` (C2 of the wall plan),
    which keeps its own copy as the ratified canary.

    `C` is the per-step header: geometry ints, `cap`, and the kick
    coefficients. Small and picklable by construction, because in pool mode it
    rides to the workers with every task.

    `ph` is the phase hook (see `step`); the serial executor passes the real
    one so the per-phase high-water boundaries stay exactly where the
    monolithic loop had them. The `busy` wall times are returned either way --
    under overlap they are the only per-phase timing that means anything,
    since a boundary hook cannot see work that runs concurrently.
    """
    import jax.numpy as jnp

    t_dec = time.perf_counter()
    slots, x, v = st.decode_bricks(bricks)
    # the brick each decoded row came from, in the order decode_bricks
    # concatenates: this is what ownership is read off, NOT the position
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
    # ownership from the brick each row is STORED IN, which is the same thing
    # membership is built from, so the two cannot disagree. Two earlier rules
    # both re-derived it from a coordinate and both lost exactly one row of
    # 16,777,216 at cdev (antares 431 tile-local, 436 global-position). See
    # `forces.owned_mask_from_bricks` for why this one cannot.
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
    # Padded to `cap` with a live mask, for the SAME reason the short arm is:
    # a per-tile row count keys a new XLA shape, so every tile recompiles.
    # Profiled before the fix at 2,107 compilations and 24.1 s of a 32.7 s
    # step -- 74% of it, 18.5 s inside backend_compile_and_load. One shape
    # serves every tile.
    n_own = int(owned.sum())
    xo = np.zeros((C["cap"], 3), dtype=np.float64)
    xo[:n_own] = x[owned]
    lv = np.zeros(C["cap"], dtype=bool)
    lv[:n_own] = True
    # `guard_out` keeps the stencil-containment check off the critical path:
    # computing its bounds on the host meant a device->host sync in the MIDDLE
    # of the gather, before the corner loop that is the actual work, plus a host
    # min/max over every padded row. Deferred, the bounds are device scalars and
    # are resolved BELOW, after the forces have been read back and the device is
    # idle, so the refusal costs a round-trip's latency instead of a stall. It
    # still fires before `g_long` is used, which is the property that matters.
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
    # QUANTIZE HERE, per brick, instead of holding `v_new` for a global scale.
    # `decode_bricks` concatenates brick by brick and ownership is read off
    # the brick a row is STORED in, so the owned rows are whole brick blocks
    # and stay contiguous under the mask -- which is what lets a run scan
    # replace a sort. The assertion below is on that contiguity, because it
    # is load-bearing and free to check: if a brick ever appeared in two
    # runs, the second run would silently overwrite the first one's scale
    # and decode every row of it wrong.
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

    Disjoint across tiles because ownership is a partition (asserted in
    `step`), so ANY application order gives the same state -- which is what
    lets the pool executor apply results in arrival order. Batched writes are
    bitwise the per-run writes the loop used to do: same rows, same values,
    disjoint runs.
    """
    if res["empty"]:
        return
    st.write_velocities(res["slots_o"], res["w_codes"])
    st.vel_scale[res["run_bricks"]] = res["run_scales"]


def step(st, cfg, coeff, c_drift, collect=None, census=False, cap_shape=0, pad_shape=0,
         phase=None, tile_force=None, pool=None, coarse_parts=None):
    """One drift-synchronized BullFrog step. Mutates `st`; returns diagnostics.

    `coeff = (alpha, beta_over_Dmid)` from `bullfrog_float_coeffs` columns 1 and
    2; `c_drift` is the FUSED drift for this step.

    `census=True` turns on the coarse decode census; see
    `coarse_delta_streamed`. Off by default: it is a gate instrument and costs
    two extra passes over the coarse mesh.

    `tile_force`, if given, is the `(one_tile, geom)` pair from
    `make_tile_force_fn`; `run` builds it once and passes it down, because the
    per-step rebuild re-traces and re-compiles the same program every step
    (~1.5 s/step, constant -- canary C0). Standalone calls omit it and build
    their own, unchanged.

    `coarse_parts` is the same arrangement for the LONG arm and it was owed for
    longer: `forces.coarse_kernel_parts` is a function of geometry alone, `run`
    builds it once, and before M-v2-6 the whole thing was rebuilt every step
    inside the solve at 60.01 B per half-grid element -- 32.28 GB per step at
    C-gh, measured, against the 12.9 the planner charged. Standalone calls omit
    it and build their own, unchanged.

    `pool`, if given, is a live `executor.TilePool`: the tile loop dispatches
    `tile_task` over its workers and applies results in arrival order, instead
    of running task+apply inline in tile order. `run` owns the pool's
    lifecycle. In pool mode the intra-tile phase boundaries do not fire (a
    boundary hook cannot see overlapped work) and the stats carry the busy
    triple -- per-phase worker busy seconds, realized concurrency, idle --
    which is what distinguishes "not enough parallel work" from "workers
    starved" from "slower per tile under contention". `tile_force` may then be
    `(None, geom)`: the parent never runs a tile itself, so it needs the
    geometry but not the kernels.

    `phase`, if given, is called with a boundary NAME after each phase of the
    step completes. It exists because a peak is a max and a max carries no
    timestamp: M-v2-6 Stage 0 attributed the peak by differencing whole-run
    maxima between arms, and at cdev the terms it was trying to separate
    (67-179 MB) sat inside the run-to-run scatter of the maximum itself
    (sigma 45-115 MB over five repeats of one leg, antares 445), so no
    difference of maxima could be read. Naming the boundaries lets a caller
    take a high-water mark PER PHASE instead, which is a measurement of where
    the peak is rather than an inference from what it is not. The hook takes no
    payload and returns nothing on purpose -- it must not be able to perturb
    the step, and the default is a function that does nothing at all.
    """
    import jax.numpy as jnp

    ph = phase if phase is not None else _no_phase

    alpha_k, bcoef = float(coeff[0]), float(coeff[1])

    # --- long arm: solve once, globally, on the coarse mesh
    mesh_stats = {}
    delta = coarse_delta_streamed(st, cfg, stats=mesh_stats, census=census,
                                  pad_shape=pad_shape, pool=pool)
    ph("coarse_paint")
    # HAND THE FIELD OVER AND LET GO OF IT. `delta` is a full coarse mesh (4.3 GB
    # at C-gh) and the solve reads only the jax copy, so holding the numpy name
    # bound through the solve keeps a dead mesh alive across the phase where the
    # peak is. Costs nothing when the jax copy aliases the host buffer.
    dj = jnp.asarray(delta)
    # read the REALIZED dtype off the decoded field before letting go of it: the
    # receipt at the end of the step is the check that the coarse knob applied,
    # and it has to come from the array the paint produced. The force meshes
    # cannot serve -- with `out=` they are the pool's preallocated views, whose
    # dtype comes from the config, so a receipt read there would echo what it
    # was told, which is exactly the failure this milestone exists to catch.
    coarse_dtype_seen = np.dtype(delta.dtype).name
    del delta
    # `coarse_force_meshes` infers from delta.dtype and REFUSES a mismatch, so
    # the dtype cannot silently disagree with what the config asked for.
    # `out=` is the pool's own shm views: the workers read them anyway, so
    # solving into them removes the parent-side triple AND the copy stage_step
    # would otherwise make.
    g_coarse = coarse_force_meshes(
        dj,
        cfg.n_coarse,
        cfg.box_size,
        "long",
        r_s=cfg.r_s,
        match=(cfg.coarse_cell, cfg.fine_cell),
        fdtype=cfg.np_coarse_dtype,
        parts=coarse_parts,
        out=None if pool is None else pool.g_views(),
    )
    del dj
    ph("coarse_solve")

    # --- membership, and the capacity one jitted program needs
    b_real = cfg._b_realized
    members = {
        t: st.tile_bricks(t, cfg.n_tile, b_real, cfg.n_brick, cfg.n_fine) for t in cfg.tiles
    }
    # brick_member_COUNT, not brick_live_count: a brick's arena residents are
    # part of its membership and `decode_bricks` returns them, so sizing `cap`
    # off the run length alone under-counts and the force refuses.
    counts = [sum(st.brick_member_count(b) for b in members[t]) for t in cfg.tiles]
    cap_true = tile_capacity(counts)
    # QUANTIZE THE SHAPE, do not use the true max. `cap_true` moves every step as
    # occupancy shifts (measured at cdev8: ten distinct values over ten steps,
    # 285,554 -> 397,319), so every buffer keyed on it keys a new XLA shape and
    # the executable cache grows without bound -- a peak host RSS linear in K at
    # 0.331 GB/step. `capacity_shape` puts it on a geometric ladder and
    # `cap_shape` carries the previous value so it is monotone across the run.
    # Padding is masked exactly as the within-step padding already is, so this is
    # bitwise neutral; `tests/test_engine.py` pins that.
    cap = capacity_shape(cap_true, rungs=cfg.cap_rungs, floor_shape=cap_shape)

    if tile_force is None:
        # standalone calls build per step, as before; `run` hoists the build to
        # once per run -- it re-traces and re-compiles `one_tile` every call
        # (~1.5 s/step at cdev, constant: canary C0) and rebuilds the kernel
        # triple, which is the `membership` phase's transient
        tile_force = make_tile_force_fn(
            cfg.n_fine, cfg.box_size, cfg.n_total, cfg.n_tile, cfg.b_fine,
            r_s=cfg.r_s, paint=cfg.paint_short, frac_bits=cfg.frac_bits,
            fdtype=cfg.np_fine_dtype,
        )
    one_tile, geom = tile_force
    cell = geom["cell"]
    ph("membership")

    # the per-step header `tile_task` runs from: everything it needs that is
    # not an array. In pool mode this is what rides to the workers per task.
    C = dict(
        cap=int(cap), n_tile=cfg.n_tile, n_brick=cfg.n_brick, n_fine=cfg.n_fine,
        n_coarse=cfg.n_coarse, box=cfg.box_size, coarse_cell=cfg.coarse_cell,
        cell=cell, b_real=int(b_real), alpha_k=alpha_k, bcoef=bcoef,
    )

    tile_scales, n_owned, n_overhang = [], 0, 0
    # NO `pending`. It held `(slots int64, v_new f64)` for every owned row until
    # the last tile had been kicked, because a GLOBAL scale cannot be known
    # before then -- 32 B/p, 274.9 GB at C-gh, and the largest single term in the
    # whole configuration. Per-brick scales remove the wait rather than the
    # array: a brick's rows are all kicked in one tile, so its scale is known the
    # moment that tile is done and `apply_result` writes its codes immediately.
    #
    # TWO executors of ONE function. Serial: task and apply inline, in tile
    # order. Pool: the same `tile_task` from workers, applied in arrival order.
    # The bitwise executor-identity gate compares the two.
    tasks = [(t, members[t]) for t in cfg.tiles]
    if pool is not None:
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

    # The boundary stays so the trace keeps its shape and peak comparisons
    # against every card on record are like-for-like. (The `pending` array it
    # was placed to price was deleted with the per-brick scales; nothing
    # accumulates across the loop any more.)
    ph("tile_loop_end")

    if n_owned != st.n_particles:
        # SELF-DIAGNOSING, because the bare count sent me down three wrong
        # explanations. Computed only on the failure path, so the happy path pays
        # nothing. It names where the deficit is rather than leaving it to be
        # guessed from the geometry.
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

    # Nothing to reconcile: every brick's codes were written at its own scale
    # inside the loop. The phase boundary stays so the trace keeps its shape and
    # a peak comparison against every card on record is still like-for-like.
    ph("reconcile")

    if pool is not None and cfg.migrate_pooled is not False:
        stats = drift_and_migrate_pooled(
            st, c_drift, pool, kernel=cfg.eject_kernel, window=cfg.migrate_window,
            eject_inflight=cfg.migrate_eject_inflight,
        )
    else:
        stats = drift_and_migrate(st, c_drift, kernel=cfg.eject_kernel)
    # the knob's receipt, in BOTH directions: 0 on every serial card, W on
    # every pooled one (the reach fallback reports 0 through migrate_pool)
    stats["migrate_pooled_workers"] = int(stats.get("migrate_pool", {}).get("workers", 0))
    # PEAK ARENA RESIDENCY, so `arena_frac` stops being chosen by argument.
    # The 6% on record is CLAIMS across a migrate pass, not residency: the
    # arena is a revolving door (`_release_arena_of_brick` frees slots as
    # bricks are rewritten) and C15 found 88.7-96.2% of occupied bricks with
    # no residents at all. What that makes the true peak has never been
    # measured, and it is what sets how small the arena can safely be.
    for _k in ("migrate", "migrate_pool"):
        _u = (stats.get(_k) or {}).get("arena_used")
        if _u is not None:
            stats["arena_used"] = int(_u)
            break
    ph("migrate")
    # both: `cap` is the SHAPE every buffer took, `cap_true` the max over tiles it
    # was quantized from. Reporting only one of them hides either the padding cost
    # or the shape churn, and the shape churn is what leaked.
    # `vel_scale` stays a SCALAR in the stats dict -- the max over bricks, which
    # `drift_and_migrate` already put there -- so every reader of a card keeps
    # working. `vel_scale_min` is beside it: the two together are what say how
    # much the per-brick scales actually spread, and a single number cannot.
    stats.update(cap=cap, cap_true=cap_true, n_tiles=len(cfg.tiles),
                 vel_scale_kick_max=float(max(tile_scales)) if tile_scales else 1.0)
    # ALWAYS on the card, 1 included: a pooled run's per-phase memory and
    # boundary timings are void under overlap, and the comparability check
    # refuses to read them against serial cards only if the knob is recorded
    stats["tile_workers"] = int(getattr(cfg, "tile_workers", 1))
    if pool is not None:
        b_sum = float(sum(busy.values()))
        stats["pool"] = dict(
            workers=pool.workers, wall_s=loop_wall, busy_s=busy, busy_total_s=b_sum,
            concurrency=(b_sum / loop_wall) if loop_wall > 0 else 0.0,
            idle_s=pool.workers * loop_wall - b_sum,
            rss_mb=rss,
        )
    # the REALIZED dtypes, read off the arrays rather than echoed from the
    # config: a receipt that repeats what it was told cannot catch a knob that
    # did not apply, which is the whole failure mode this milestone is built
    # against
    stats["coarse_dtype"] = coarse_dtype_seen
    stats["fine_dtype"] = str(geom["fdtype"])
    stats.update(mesh_stats)
    stats["repack"] = None
    if collect is not None:
        collect(stats)
    return stats


def fused_drifts(coeffs):
    """The drift-synchronized schedule from `bullfrog_float_coeffs`.

    Returns `(lead, fused)`: a leading half-drift that puts positions on the
    first midpoint, then one drift per step -- `h_k + h_{k+1}` after each kick,
    with `h_{K-1}` alone on the last so the trajectory lands on the same endpoint
    the boundary form does.
    """
    h = np.asarray(coeffs)[:, 0]
    return float(h[0]), np.concatenate([h[:-1] + h[1:], h[-1:]])



# ------------------------------------------------------------- checkpoints

# The config fields a resumed run must match. Deliberately NOT every attribute:
# these are the ones that move numbers, either as physics/geometry or as a
# BUFFER SHAPE, and a shape belongs here because XLA reassociates by shape and
# a padded reduction can move bits (the pad-and-mask lesson). Excluded on
# purpose, with the reason each is safe to change across a resume:
#   tile_workers, worker_affinity, migrate_pooled, migrate_window -- execution
#     policy; the pooled migrate is bitwise the serial one (C14) and W is
#     exactly the thing you want to change when resuming onto another node.
#   eject_kernel -- both kernels are bitwise on arm64/x86/CUDA (record 5s).
#   repack_every, chunk_bricks, brick_slack -- move the LAYOUT, and the layout
#     carries no physics: both paints are integer and so order-independent
#     (D-v2-21), which is also why this checkpoint does not preserve
#     allocation geometry at all.
#   checkpoint_dir, checkpoint_every -- the mechanism itself.
_FINGERPRINTED = (
    "box_size", "n_part", "n_fine", "n_coarse", "n_tile", "b_fine", "alpha",
    "paint_short", "paint_long", "frac_bits", "coarse_dtype", "fine_dtype",
    "cap_rungs", "pad_ladder", "paint_subblock",
)


def checkpoint_fingerprint(cfg, coeffs):
    """What a resume must match. Covers the schedule too: `coeffs` is hashed as
    bytes, so the cosmology, the a-grid and K are all in here without the
    checkpoint having to name them or the caller having to pass a cosmology."""
    h = hashlib.sha256()
    h.update(json.dumps({k: getattr(cfg, k) for k in _FINGERPRINTED}, sort_keys=True).encode())
    h.update(np.ascontiguousarray(coeffs, dtype=np.float64).tobytes())
    return h.hexdigest()


def _write_checkpoint(st, cfg, coeffs, step, cap_shape, pad_shape, gen, epoch=None):
    """One generation of a rolling pair. Returns the directory, which is the
    receipt: a run that believed it was checkpointing and was not has `None`
    on every step.

    `epoch` is the optional `(a_steps, cosmo)` from `run`; see `epoch_record`
    for what it writes and why it is not in the fingerprint."""
    from . import icgen

    prov = dict(
        kind="inexor-checkpoint",
        step=int(step),
        n_steps=int(len(coeffs)),
        cap_shape=int(cap_shape),
        pad_shape=int(pad_shape),
        n_arena=int(st.n_arena),
        fingerprint=checkpoint_fingerprint(cfg, coeffs),
    )
    prov.update(epoch_record(epoch, step))
    d = os.path.join(cfg.checkpoint_dir, f"gen{gen}")
    icgen.write_t9_slabs(st, d, provenance=prov)
    return d


def epoch_record(epoch, step):
    """The `{a, cosmology}` a checkpoint carries so its units can be chosen later.

    `epoch` is `(a_steps, cosmo)` -- the FULL `integrate.a_grid` output and the
    cosmology it was built from -- or `None`, which writes nothing and leaves
    the manifest exactly as it was before this existed.

    **The index convention is the whole content of this function.** `a_steps`
    has `n_steps + 1` entries, `a_steps[0]` is the initial epoch, and
    `_write_checkpoint` is called with `step` = the number of COMPLETED steps.
    So a checkpoint at `step` sits at `a_steps[step]`, which is the same
    indexing the M-v2-6 driver's export leg does. It is pinned in
    `tests/test_engine.py` rather than left to a reading of two call sites.

    **Deliberately NOT in `checkpoint_fingerprint`.** The fingerprint already
    hashes `coeffs` as bytes, so the cosmology and the a-grid are inside it;
    this is the same information written down READABLY, for a consumer that has
    only the directory. Adding it to `_FINGERPRINTED` would change every
    fingerprint and invalidate every checkpoint on disk to record nothing new.

    Nothing here is required for a resume, and a manifest without it loads and
    resumes exactly as before -- which is why this is provenance rather than a
    schema field, and why the banked 2048^3 slabs still load unchanged.
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
                    arena_frac=None, alloc=None):
    """Newest complete checkpoint under `checkpoint_dir` -> `(st, resume)`.

    Picks by the recorded step rather than by mtime, and only among generations
    whose manifest is present -- the manifest is removed before a rewrite and
    written last, so a torn generation is invisible here and the older one
    survives. That is the whole reason there are two.

    Refuses a fingerprint mismatch. Resuming a state under a different geometry
    or a different schedule would not fail, it would produce a run that is half
    one thing and half another, and nothing downstream could see it.

    Capacity is the caller's to choose and is not restored from the file: this
    format stores membership, not allocation. `brick_slack` defaults to the
    config's and `arena_frac` to whatever the checkpoint recorded, which
    reproduces a comparable container without claiming to reproduce the same one.
    """
    from . import icgen

    best = None
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
            best = (d, prov, man)
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
    )
    return st, dict(prov)

def run(st, cfg, coeffs, collect=None, census=False, phase=None, resume=None,
        stop_at=None, allocator=None, epoch=None):
    """Advance `st` over a whole schedule. `coeffs` from `bullfrog_float_coeffs`.

    `phase` is forwarded to `step`; see its docstring. The boundaries `run`
    itself adds are the lead drift and the repack, so that every allocation in
    the run falls inside exactly one named phase and the phases sum to the run.

    **`stop_at` runs a SEGMENT of the schedule**, stopping before absolute step
    `stop_at`, and it is the other half of `resume`. Together they let one
    schedule cross job boundaries, which a realization at full scale needs: the
    projected wall is order a day and a queue's limit is not negotiable.

    It is deliberately NOT "run the first n steps of a shorter schedule".
    `fused_drifts` fuses each step's trailing half-drift with the next step's
    leading half, so a run over `coeffs[:n]` is a DIFFERENT trajectory rather
    than a prefix of this one. `stop_at` takes the full `coeffs` and stops, so
    the segments compose back into the uninterrupted run bitwise -- which is
    what `test_a_run_split_into_segments_is_bitwise_the_uninterrupted_one`
    asserts particle for particle.

    Stopping somewhere a checkpoint was not written throws that segment's work
    away, and doing so silently is the failure this refuses: with checkpointing
    on, `stop_at` must land on a checkpoint boundary.

    **`epoch` is `(a_steps, cosmo)` and only checkpoints read it.** The engine
    has no cosmology of its own -- `EngineConfig` is geometry and policy, and
    `coeffs` are growth-factor numbers with no scale factor left in them -- so
    a caller that wants its checkpoints to know their own epoch passes the
    `integrate.a_grid` output and the cosmology it came from. What that buys is
    `python -m inexor.export` writing peculiar km/s off a bare directory
    instead of the reader having to supply the epoch by hand and get it right.
    Omitted, everything behaves exactly as before; see `epoch_record`.
    """
    cfg.validate()
    ph = phase if phase is not None else _no_phase
    ckpt_on = bool(cfg.checkpoint_dir) and cfg.checkpoint_every > 0
    if ckpt_on and st.ids is not None:
        # fail here, not 37 minutes into the first step: the slab schema has no
        # ids and a checkpoint that dropped them would make the restart
        # non-reproducible for anything id-dependent
        raise ValueError(
            "checkpointing a state that carries ids: the t9-slabs-2 schema has no room "
            "for them, so the resumed run would silently lose particle identity"
        )
    if epoch is not None:
        # Same reason as above: check the grid against the schedule here rather
        # than at the first checkpoint, which is a step into the run. `coeffs`
        # is always the FULL schedule even for a `stop_at` segment, so this
        # invariant holds on every leg.
        a_steps, _ = epoch
        if len(a_steps) != len(coeffs) + 1:
            raise ValueError(
                f"epoch grid has {len(a_steps)} points for a {len(coeffs)}-step schedule; "
                f"`integrate.a_grid` emits n_steps + 1 = {len(coeffs) + 1}. A grid that is "
                "not this schedule's would record a plausible wrong epoch on every checkpoint"
            )
    pool = None
    if cfg.tile_workers > 1:
        # the pool's workers each build their own kernels + jitted program, so
        # the parent needs only the geometry -- bit-identical numbers from
        # `tile_geom` without the kernel triple's memory or the trace
        from .executor import TilePool

        pool = TilePool(st, cfg, allocator=allocator)
        tile_force = (None, tile_geom(
            cfg.n_fine, cfg.box_size, cfg.n_total, cfg.n_tile, cfg.b_fine,
            paint=cfg.paint_short, frac_bits=cfg.frac_bits, fdtype=cfg.np_fine_dtype,
        ))
    else:
        # ONCE per run, not per step: the rebuild re-traces and re-compiles the
        # same jitted program and rebuilds the kernel triple every step (canary
        # C0: ~1.5 s/step, constant). The triple is now resident for the whole
        # run instead of a per-step `membership` transient -- a named boundary
        # so every allocation still falls inside exactly one phase.
        tile_force = make_tile_force_fn(
            cfg.n_fine, cfg.box_size, cfg.n_total, cfg.n_tile, cfg.b_fine,
            r_s=cfg.r_s, paint=cfg.paint_short, frac_bits=cfg.frac_bits,
            fdtype=cfg.np_fine_dtype,
        )
    # The LONG arm's build, hoisted for the same reason and paid at the same
    # boundary. Geometry only, so one build serves every step; it is 4.30 GB
    # resident at C-gh against 23.7 GB of per-step transient it removes, and the
    # build's own 28 B/half peak lands here, before any force mesh exists.
    coarse_parts = coarse_kernel_parts(
        cfg.n_coarse, cfg.box_size, "long", r_s=cfg.r_s,
        match=(cfg.coarse_cell, cfg.fine_cell), fdtype=cfg.np_coarse_dtype,
    )
    ph("kernel_build")
    try:
        lead, fused = fused_drifts(coeffs)
        # onto the first midpoint; pooled under the same knob as the per-step
        # migrate (5j: this ONE call was mistaken for a per-step phase once).
        # SKIPPED on a resume -- the checkpointed state is already past it, and
        # applying it twice would drift the whole box by an extra half step.
        # The phase boundary still fires so a resumed run's phase table keeps
        # the same shape as an uninterrupted one (the `reconcile` precedent).
        if resume is None:
            if pool is not None and cfg.migrate_pooled is not False:
                drift_and_migrate_pooled(st, lead, pool, kernel=cfg.eject_kernel,
                                         window=cfg.migrate_window)
            else:
                drift_and_migrate(st, lead, kernel=cfg.eject_kernel)
        ph("lead_drift")
        out = []
        # both buffer shapes are carried ACROSS steps and only ever grow, so the
        # run visits at most a few shapes instead of one per step: `cap` from
        # Stage 0, `coarse_pad` from Stage 0b, which measured it still churning
        cap_shape = 0
        pad_shape = 0
        k0 = 0
        if resume is not None:
            # Both shapes are RESTORED rather than re-laddered from zero, so a
            # resumed run keeps the "few shapes per run" property instead of
            # paying a fresh compile. This is a COMPILE-COUNT choice and not a
            # correctness one: measured 2026-08-18 at cdev scale on arm64 CPU,
            # quadrupling both shapes on resume moves nothing (0 of ~229k rows
            # on `off` and `w`), because the padded rows are masked. The
            # split-run gate therefore cannot see this and does not claim to;
            # what it does catch is the lead drift being reapplied. Not
            # generalized to GPU, where XLA reassociates by shape.
            k0 = int(resume["step"])
            cap_shape = int(resume["cap_shape"])
            pad_shape = int(resume["pad_shape"])
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
        n_ckpt = 0
        for k in range(k0, k_end):
            stats = step(st, cfg, (coeffs[k][1], coeffs[k][2]), float(fused[k]), collect,
                         census=census, cap_shape=cap_shape, pad_shape=pad_shape,
                         phase=phase, tile_force=tile_force, pool=pool,
                         coarse_parts=coarse_parts)
            cap_shape = int(stats["cap"])
            pad_shape = int(stats["coarse_pad"])
            if cfg.repack_every and (k + 1) % cfg.repack_every == 0:
                stats["repack"] = st.repack(brick_slack=cfg.brick_slack)
                ph("repack")
            # AFTER the repack, so the checkpoint is the step's settled state.
            # The receipt goes on every step in both directions, `None` when
            # checkpointing is off: a knob must prove it applied.
            stats["checkpoint"] = None
            if ckpt_on and (k + 1) % cfg.checkpoint_every == 0:
                stats["checkpoint"] = _write_checkpoint(
                    st, cfg, coeffs, k + 1, cap_shape, pad_shape, n_ckpt % 2, epoch=epoch
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

    Algebraically identical to looping `float_step_bullfrog`, and deliberately
    NOT bitwise it: `mod(mod(x+a)+b)` and `mod(x+a+b)` differ in float. That is
    the whole reason this exists rather than the gate reusing the stepper --
    comparing the engine against a driver of a different SHAPE would report a
    roundoff difference as an engine defect.

    `float_step_bullfrog` is untouched; it is an mbody port with a stability
    contract and the probes drive it.
    """
    import jax.numpy as jnp

    lead, fused = fused_drifts(coeffs)
    x = jnp.mod(x + lead * v, box_size)
    for k in range(len(fused)):
        g = force_fn(x)
        v = coeffs[k][1] * v + coeffs[k][2] * g
        x = jnp.mod(x + float(fused[k]) * v, box_size)
    return x, v
