"""`python -m inexor.plan`: the terms it names, and the invariance the C-gh
verdict rests on.

The module had no tests. It is a front end over `mesh_bytes` / `step_bytes` /
`index_bytes` / `plan_bytes`, all of which have their own, so the arithmetic was
covered -- but the REDUCTION was not, and the reduction is what prints a verdict.
Two defects lived there: the "largest single term" line could not name a
transient however large, and `tile_buffers` vanished from the table whenever
`cap` was absent, which for a planning run is always. Both are the failure the
module's own docstring is about: a term that is not in the table cannot be traded
against anything.
"""

import numpy as np
import pytest

from inexor import engine
from inexor.engine import EngineConfig
from inexor.forces import padded_size
from inexor.plan import GB as GB_
from inexor.plan import PRESETS, main


def _ec(name):
    p = PRESETS[name]
    return EngineConfig(
        box_size=p["box"], n_part=p["n_part"], n_fine=p["n_fine"],
        n_coarse=p["n_coarse"], n_tile=p["tile"], b_fine=p["buf"],
        coarse_dtype="float32", fine_dtype="float64",
    )


# ------------------------------------------------- the invariance under the verdict


@pytest.mark.parametrize("name", ["cdev", "cgh64", "c-gh"])
def test_the_tile_transient_does_not_grow_from_the_anchor_to_c_gh(name):
    """THE fact that makes the C-gh binding term readable, and it was nowhere.

    `kick_pending` is O(N) and the tile transients scale with `cap`, which reads
    as "two terms growing at different rates". They are not: the config table
    holds the fine cell FIXED and grows volume at T=256/b=32, so cdev, cgh64 and
    C-gh have the SAME particles per tile and the SAME padded tile, hence the
    same tile transient in bytes. The phase measured to set the peak at cdev
    (job 446, `tile_short`, 3.432 GB) therefore does not grow up the ladder at
    all, while `kick_pending` grows 512x from cdev to C-gh.

    If a preset ever moves off that shared geometry this test fails, and the
    re-pricing has to be redone rather than inherited.
    """
    ec, anchor = _ec(name), _ec("cdev")
    n_per_tile = PRESETS[name]["n_part"] ** 3 / len(ec.tiles)
    anchor_per_tile = PRESETS["cdev"]["n_part"] ** 3 / len(anchor.tiles)
    assert n_per_tile == anchor_per_tile == 2097152
    assert (padded_size(PRESETS[name]["tile"], PRESETS[name]["buf"],
                        n_fine=PRESETS[name]["n_fine"])[0] == 320)
    mb, ab = ec.mesh_bytes(), anchor.mesh_bytes()
    for term in ("tile_workspace", "tile_kernels"):
        assert mb[term] == ab[term], f"{term} moved off the anchor at {name}"


def test_kick_pending_is_gone_and_its_replacement_is_five_orders_smaller():
    """`kick_pending` WAS the binding term: 32 B/p over 2048^3 = 274.9 GB against
    a ~116 GB host, enough on its own to keep C-gh off a `gh` node even if
    everything else were free. Per-brick velocity scales removed the reason it
    existed -- the engine no longer waits for a global reduction before it can
    encode -- and what replaced it is one f64 per brick.

    The line is asserted to still EXIST and read zero rather than deleted. A term
    that vanishes from a table is indistinguishable from one that was never
    counted, which is the failure mode this whole planner was built against.
    """
    n = PRESETS["c-gh"]["n_part"] ** 3
    ec = _ec("c-gh")
    assert ec.step_bytes(n)["kick_pending"] == 0
    assert "kick_pending" in ec.step_bytes(n), "the retired term must stay visible"
    n_bricks = (PRESETS["c-gh"]["n_fine"] // ec.n_brick) ** 3
    assert n_bricks * 8 < 0.02e9, "the replacement should be tens of MB, not GB"
    assert (n * 32) / (n_bricks * 8) > 1e4, "expected five orders, not a trim"


def test_the_crossover_that_made_the_anchor_the_last_tile_dominated_rung():
    """KEPT AS A RECORD, and it no longer describes the engine. It is where
    `kick_pending` overtook the tile force phase, and it is why the one measured
    peak (cdev) and the model disagreed at C-gh without contradicting each other:
    the anchor was the last rung on which the tile force won. The term is gone
    now, so nothing crosses here any more -- but the reasoning is what a future
    reader needs to interpret every card measured before the removal."""
    measured_tile_phase = 3.432e9  # job 446, cdev, `tile_short` own increment
    n_cross = measured_tile_phase / 32
    assert PRESETS["cdev"]["n_part"] ** 3 < n_cross < PRESETS["cgh64"]["n_part"] ** 3
    assert 400 < round(n_cross ** (1 / 3)) < 500


def test_the_tile_kernel_build_reproduces_the_measured_phase():
    """The M-v2-6 correction, pinned against MEASUREMENTS at two configs.

    `mesh_bytes` counted only the three complex kernels `split_kernels` returns
    and none of what it holds live to build them -- `k2_true`, `k2_safe`, `fac`
    (f64 half-grids) and `pref` (fine dtype) -- so it read 48*phalf where the
    peak is 80*phalf at f64. The coarse arm always carried the analogous term;
    the tile arm never did.

    The numbers are the `membership` phase from `v2_m6_host_bytes.py`, which is
    where `make_tile_force_fn` is called. Two configs because one cannot separate
    the kernel axis from the particle axis: they move together on every rung of
    the config table, and only cdev's 15.48x phalf against 8.00x particles
    splits them.
    """
    for name, measured_mb in (("cdev8", 85.60), ("cdev", 1319.45)):
        m = _ec(name).mesh_bytes()
        build = (m["tile_kernels"] + m["tile_kernel_build_f64"] + m["tile_kernel_pref"])
        assert build / 1e6 == pytest.approx(measured_mb, rel=0.02), (
            f"{name}: modelled {build / 1e6:.2f} MB against a measured "
            f"{measured_mb} MB for the tile kernel build"
        )
        # and the correction is 5/3 of the old term at f64, by derivation
        assert build == pytest.approx(m["tile_kernels"] * 5 / 3, rel=1e-9)


def test_the_low_rank_ik_grids_are_why_the_factor_is_five_thirds():
    """If `kernel_grids` ever returned full ik half-grids instead of low-rank
    broadcasts, the build would cost 3 more f64 grids and the factor would be
    2.17, not 5/3. The docstring promises low-rank; this fails if it stops."""
    from inexor.forces import kernel_grids

    ikx, iky, ikz, k2_true, k2_safe = kernel_grids((16, 16, 16), 1.0, np.float64)
    for a in (ikx, iky, ikz):
        assert a.size <= 16, f"ik grid is full-rank ({a.size} elements); the 5/3 is stale"
    assert k2_true.size == k2_safe.size == 16 * 16 * 9


def test_both_arms_count_the_prefactor():
    """`pref` is a full half-grid in BOTH arms and was in neither."""
    m = _ec("cdev").mesh_bytes()
    assert m["coarse_kernel_pref"] > 0 and m["tile_kernel_pref"] > 0


def test_the_coarse_force_copy_is_one_component_not_three():
    """It used to be three, and that was the bug rather than the accounting.

    `g_coarse = [np.asarray(g) for g in g_coarse]` rebinds only after the
    comprehension, so the three jax meshes survived their three numpy copies'
    creation -- six live at once, 25.8 GB at C-gh. M-v2-6 made
    `coarse_force_meshes` solve one component at a time straight into the
    caller's buffers, so the transient is ONE mesh and the pool's own copy
    stopped existing with it."""
    m = _ec("cdev").mesh_bytes()
    assert m["coarse_force_copy_transient"] * 3 == m["coarse_force_resident"]


def test_migration_staging_is_n_to_the_two_thirds_not_n():
    """The shape is derived, the coefficient is measured, and two configs
    agreeing on the coefficient is what tests the shape.

    `drift_and_migrate` walks x-slabs and releases each as soon as every write
    that could reach it is done, so rows in flight are a few SLABS -- N divided
    by bricks_per_side, which itself grows as N^(1/3). Modelling it as O(N) would
    overstate it by 8x at C-gh, which is the whole reason it earns a term.
    """
    for name, measured_mb in (("cdev8", 47.62), ("cdev", 208.58)):
        p = PRESETS[name]
        got = _ec(name).step_bytes(p["n_part"] ** 3)["migrate_staging"]
        assert got / 1e6 == pytest.approx(measured_mb, rel=0.08), (
            f"{name}: modelled {got / 1e6:.2f} MB against a measured {measured_mb}"
        )
    # and the SHAPE: 8x the particles must give 4x the term, not 8x
    a = _ec("cdev8").step_bytes(PRESETS["cdev8"]["n_part"] ** 3)["migrate_staging"]
    b = _ec("cdev").step_bytes(PRESETS["cdev"]["n_part"] ** 3)["migrate_staging"]
    assert b / a == pytest.approx(4.0, rel=0.02), "migration staging stopped being N^(2/3)"


def test_per_step_terms_do_not_depend_on_slack():
    """Measured: raising slack 0.2 -> 0.5 moves n_rows 1.217x and EVERY per-step
    phase by exactly 1.000x. Slack buys spare slots in the storage arrays, which
    the state pays for and the step never touches, because the step decodes live
    members rather than rows. So only `repack_scratch` may carry `n_rows` here.
    """
    ec = _ec("cdev8")
    n = PRESETS["cdev8"]["n_part"] ** 3
    a = ec.step_bytes(n, n_rows=int(n * 1.2))
    b = ec.step_bytes(n, n_rows=int(n * 1.5))
    for k in a:
        if k == "repack_scratch":
            assert b[k] > a[k]
        else:
            assert a[k] == b[k], f"{k} moved with n_rows; the slack arm says it must not"


# ------------------------------------------------------------------ the reduction


def test_the_largest_term_line_can_name_a_transient(capsys):
    """It could not. At cdev it reported `tile_kernels` (0.791 GB) while
    `tile_workspace` (1.443) was larger, because transients were left out of the
    candidate set -- so the line structurally could not name the tile force, the
    phase job 446 measured to SET the peak."""
    main(["--preset", "cdev", "--host-gb", "124", "--cap", "5284492"])
    out = capsys.readouterr().out
    assert "largest single term: tile_workspace (transient)" in out


def test_the_binding_term_at_c_gh_is_now_the_state_itself(capsys):
    """The end of the removals: with `kick_pending` deleted and the repack
    rewritten in place, no TRANSIENT is the largest term any more -- the state
    payload is, at 9 B/p. There is nothing left to remove that is not the
    simulation itself, so any further reduction is a codec question rather than
    an accounting one.

    C-gh still does not fit a `gh` host (1.70x, from 4.41x at the start of
    M-v2-6), and that gap is now structural: state plus resident mesh alone
    clears 116 GB. The `gg` test below is the one that changed."""
    main(["--preset", "c-gh", "--host-gb", "116", "--cap", "5284492"])
    out = capsys.readouterr().out
    assert "largest single term: t9_payload" in out
    assert "DOES NOT FIT" in out
    ratio = float(out.split("DOES NOT FIT (")[1].split("x")[0])
    assert 1.4 < ratio < 2.0, f"expected ~1.70x a gh host, got {ratio}"


def test_c_gh_now_fits_a_cpu_only_node_with_margin(capsys):
    """THE first time the production configuration fits anything.

    At the start of M-v2-6 the lower bound was 511.2 GB, 2.16x even a 237 GB
    `gg` node. Deleting `kick_pending` (274.9 GB) and rewriting the repack in
    place (115.4 -> 21.8) brought it to 164.6.

    It reads 180.5 now, and the path there ran in both directions. Charging
    every phase inside a step rather than the single largest transient PUSHED
    it up (that is the model which would have refused the run that OOM-killed);
    hoisting the coarse kernel build and rewriting the repack census pulled it
    back down by more. Neither move was for the number -- one is an accounting
    correction and the other removes real allocations.

    Pinned because the margin is what makes a capacity run proposable at all,
    and it is now thin: still a LOWER BOUND, one measured to read ~1.9x low at
    cdev, so fitting on paper is not the same as fitting."""
    main(["--preset", "c-gh", "--host-gb", "237", "--cap", "5284492"])
    out = capsys.readouterr().out
    assert "FITS" in out and "DOES NOT FIT" not in out
    est = float(out.split("a lower bound on the run's peak:")[1].split("GB")[0])
    assert est == pytest.approx(180.5, abs=1.0)
    assert est < 237.0, "the bound no longer fits the node it was sized for"


def test_removing_the_repack_scratch_too_would_still_not_reach_a_gh_host(capsys):
    """Where the remaining gap is, stated as a test so it cannot be forgotten:
    state (98.6 GB resident) plus the resident mesh (18.0) is already 116.5 GB
    against a 116 GB hard cliff, BEFORE any transient. So no per-step removal
    reaches a `gh` node -- the resident floor alone clears it -- which is what
    makes the target machine a scoping question rather than an optimization."""
    main(["--preset", "c-gh", "--host-gb", "116", "--cap", "5284492"])
    out = capsys.readouterr().out
    state_total = float(out.split("STATE")[1].split("total")[1].split("GB")[0])
    mesh_res = float(out.split("MESH, resident")[1].split("total")[1].split("GB")[0])
    assert state_total + mesh_res > 116.0, (
        "the resident floor no longer clears a gh host, so this test's premise -- "
        "and the milestone's target-machine argument -- needs re-deriving"
    )


def test_an_absent_cap_names_the_omission_instead_of_dropping_it(capsys):
    """`tile_buffers` needs a measured `cap` and is rightly not guessed -- but
    silently omitting it prints a total that excludes the per-tile host set
    sitting inside the peak-setting phase. The omission has to be visible."""
    main(["--preset", "cdev", "--host-gb", "124"])
    out = capsys.readouterr().out
    assert "`tile_buffers` is NOT in the total above" in out
    assert "tile_buffers" not in out.split("BINDING TERMS")[0], (
        "the term must be absent from the tables, not silently zero"
    )

    main(["--preset", "cdev", "--host-gb", "124", "--cap", "5284492"])
    out = capsys.readouterr().out
    assert "tile_buffers" in out.split("BINDING TERMS")[0]
    assert "NOT in the total above" not in out


def test_the_estimate_is_a_lower_bound_and_says_so(capsys):
    """Still a floor, and still known-soft: 3.363 GB against job 446's measured
    7.461 at cdev, so 2.22x low where it has been checked. The ladder is
    3.783 -> 3.305 -> 3.396 -> 3.363: the M-v2-6 terms, the phase model and the
    repack census move it in both directions and none is claimed to close the
    gap, which is why the wording stays. What the phase model DID fix is the shape of the error
    at C-gh, where the old form under-charged the coarse solve by 3x."""
    main(["--preset", "cdev", "--host-gb", "124", "--cap", "5284492"])
    out = capsys.readouterr().out
    assert "LOWER BOUND" in out and "not a measurement" in out
    est = float(out.split("a lower bound on the run's peak:")[1].split("GB")[0])
    # 3.445 (derived 9 B/row) -> 3.488 (measured 11.1, out of place) -> 3.305
    # (measured 2.1, in place) -> 3.396 (phases summed within a step) -> 3.363
    # (measured 0.49, per-brick census). scripts/v2_m6_repack_bytes.py.
    assert est == pytest.approx(3.363, abs=0.01)
    assert est < 7.461, "the bound must sit under the measured peak it bounds"


def test_no_host_budget_gives_no_verdict(capsys):
    """A wrong default host size would silently make a verdict up."""
    main(["--preset", "cdev"])
    out = capsys.readouterr().out
    assert "no --host-gb given, so no verdict" in out
    assert "FITS" not in out


def test_every_preset_is_evaluable():
    """A preset that raises is a planning tool that cannot plan the config table.

    `smoke` did raise: `--buf` defaulted to 32 rather than None, and the preset
    fill only writes fields still None, so the flag's default outranked the table
    and smoke's buf=8 never applied -- T=16 + 2*32 = 80 against a 64 fine mesh,
    straight into the degeneracy guard. Invisible for every other preset because
    they all carry buf=32.
    """
    for name in PRESETS:
        assert main(["--preset", name, "--host-gb", "116"]) == 0
        assert np.isfinite(sum(_ec(name).mesh_bytes().values()))


def test_a_preset_buffer_is_not_shadowed_by_the_flag_default(capsys):
    """The other half of the same defect, stated as the contract: the table's own
    value must reach the config."""
    main(["--preset", "smoke", "--host-gb", "116"])
    assert "b=8" in capsys.readouterr().out
    main(["--preset", "smoke", "--host-gb", "116", "--buf", "4"])
    assert "b=4" in capsys.readouterr().out, "an explicit flag must still win"


def test_the_repack_scratch_coefficient_is_the_measured_one_not_the_payload_width():
    """9 B/row is what `off` and `w` come to; 11.1 is what the function costs.

    The gap was per-bucket int64 arrays plus the sort, and it was invisible for
    as long as the term was derived from the payload width alone. Measured flat
    over 8x in particle count at every stage of the ladder, which is what says
    it is a coefficient and not a fixed cost being amortized:

        11.1 B/row   out of place
         2.1         in place (M-v2-6)
         0.49        once the per-bucket arrays came out -- `occupancy` cast to
                     int64 twice, an n_buckets bincount for the arena, and an
                     int64 output narrowed at the end

    Pinned because the derived figure is the intuitive one and would be an easy
    "simplification" to reintroduce.
    """
    n = PRESETS["c-gh"]["n_part"] ** 3
    ec = _ec("c-gh")
    rows = int(np.ceil(np.ceil(n * 1.10) * 1.10))
    scratch = ec.step_bytes(n, n_rows=rows)["repack_scratch"]
    # 9 derived -> 11.1 out of place -> 2.1 in place -> 0.49 per-brick census.
    assert scratch / rows == pytest.approx(0.49, abs=0.02)
    assert scratch < 30e9, "the in-place rewrite should put this well under a gh host"
    assert scratch / rows < 9.0, (
        "the coefficient is back at or above the payload width, so the repack is "
        "copying the payload again rather than rearranging it in place"
    )


def test_the_in_place_reference_would_make_the_repack_scratch_worse():
    """D-v2-19 clause 3 points at `BrickPackedLayout.repack` as the in-place
    form to port, on a reported `scratch_bytes` of 0.13-0.52 MB "independent of
    N". That figure counts only its two chunk buffers; the function also
    allocates `live`, `parts` and `final` at one row each, and measures 39.4
    B/row against the 11.1 it would replace -- a 3.55x REGRESSION.

    This test pins the arithmetic consequence rather than re-running the
    measurement, so the conclusion survives without the probe: any replacement
    must beat the current coefficient, and the reference does not.

    The clause's REASONING is not in dispute and is what licenses a real fix: a
    repack is a monotone rearrangement, not a sort, so an O(brick) walk exists.
    The reference is simply not that implementation.
    """
    n = PRESETS["c-gh"]["n_part"] ** 3
    rows = int(np.ceil(np.ceil(n * 1.10) * 1.10))
    current = _ec("c-gh").step_bytes(n, n_rows=rows)["repack_scratch"]
    reference_would_be = rows * 39.4
    assert reference_would_be > current, "the port is only worth doing if it wins"
    # 3.55x worse than the out-of-place form it would have replaced; against
    # the in-place form actually written it is ~19x worse.
    assert reference_would_be / current > 15.0


def test_the_load_model_matches_what_the_loader_actually_allocates(tmp_path):
    """The planner's load stages, against the real loader at a small config.

    A model nothing checks is a model that drifts, and this one now carries
    the verdict for a 2048^3 run: `load_stages` said 134.7 GB where two jobs
    had already died with nothing in any table naming that path. Held here
    against the bytes the loader really asks for.
    """
    import numpy as np

    from inexor import icgen, state
    from inexor.codec import T9Layout
    from inexor.executor import SharedAllocator
    from inexor.plan import engine_config, load_stages

    n_part, box, nb = 64, 32.0, 4
    engine_config(dict(n_part=n_part, box=box, n_fine=128, n_coarse=32,
                       tile=32, buf=8))  # the shared path must accept this geometry
    rng = np.random.default_rng(0)
    g = (np.arange(n_part) + 0.5) * (box / n_part)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    x = np.mod(q + rng.normal(scale=0.2 * box / n_part, size=q.shape), box)
    v = rng.normal(scale=0.5, size=x.shape)
    t9 = T9Layout(box_size=box, n_part=n_part, bucket_cells=2)
    st = state.SlotState.build(x, v, t9, nb, brick_slack=0.20, arena_frac=0.20)
    icgen.write_t9_slabs(st, str(tmp_path))
    n_slabs = len(icgen.load_manifest(str(tmp_path))["files"]) \
        if hasattr(icgen, "load_manifest") else nb

    alloc = SharedAllocator()
    try:
        got = icgen.load_slot_state(str(tmp_path), brick_slack=0.20,
                                    arena_frac=0.20, alloc=alloc)
        # what the loader REALLY put in shared memory
        measured = alloc.bytes_held()
        n_rows = got.off.shape[0]
        n_buckets = t9.n_buckets_side**3
        idx = np.dtype(got.occupancy.dtype).itemsize
        modelled = load_stages(
            n=n_part**3, n_rows=n_rows, n_buckets=n_buckets, index_itemsize=idx,
            n_arena=got.arena_bucket.shape[0], n_bricks=nb**3, n_slabs=n_slabs,
            shared=True)
        # the payload + index the model says the segments must hold
        want = n_rows * 9 + n_buckets * idx + got.arena_bucket.nbytes
        assert measured == pytest.approx(want, rel=0.02), (
            f"the allocator holds {measured} B where the model wants {want} B")
        # and the model's peak stage must exceed what is actually resident
        assert max(modelled.values()) >= measured
    finally:
        alloc.close()


def test_the_phase_model_would_have_refused_the_run_that_died():
    """The gate this module did not have, written against the job that needed it.

    Vista 923139 OOM-killed inside the coarse solve of step 1 while the planner
    said FITS at 0.75x. Three accounting faults, all fixed: the bound charged
    the largest SINGLE transient (12.9 GB of a 56 GB mesh total) instead of
    everything a phase holds at once; `cic_match_factor` -- called on every
    coarse solve, since the engine always passes `match` -- was in NO table; and
    the solve's transform workspace counted one complex half-grid where `dk`,
    the device copy of the kernel and their product are three.

    Reconstructing the pre-M-v2-6 solve from the terms that remain gives 68.8 GB
    against the ~79 GB the node actually lost, which is the model landing within
    13% of a measurement it had been missing by 5x. Carried through the phase
    sum it clears a 237 GB gg node, so the corrected planner refuses the run
    that died. That is the property worth pinning: not the number, the verdict.
    """
    ec = _ec("c-gh")
    m = ec.mesh_bytes()
    # the old shape: three complex kernels built per step and matched, so a
    # SECOND triple, the f64 island rebuilt every step, and three force meshes
    # copied out at once because the comprehension rebinds only at the end
    old_solve = (3 * m["coarse_kernels"]                 # six half-grids, not two
                 + m["coarse_kernel_build_f64"]
                 + m["coarse_kernel_pref"] + m["coarse_match_factor"]
                 + m["coarse_fft_workspace"]
                 + 3 * m["coarse_force_copy_transient"])
    assert old_solve / GB_ == pytest.approx(68.8, abs=1.0), (
        f"the pre-M-v2-6 solve reconstructs to {old_solve / GB_:.1f} GB; if this "
        "moved, the reconstruction is stale and the comparison is against nothing"
    )
    assert old_solve / GB_ < 79.0, (
        "the model must stay UNDER the measured 79 GB: it cannot see XLA's "
        "intra-jit scratch or glibc's retention, and a bound that claims to is "
        "no longer a bound"
    )

    mp = ec.mesh_phase()
    resident = sum(v for k, v in m.items() if mp[k] == "resident")
    step = ec.step_bytes(ec.n_total, cap=5284492)
    in_step = {"coarse_solve": old_solve}
    for src, phase_of in ((m, mp), (step, engine.STEP_PHASE)):
        for k, v in src.items():
            p = phase_of[k]
            for one in (p,) if isinstance(p, str) else p:
                if one in ("resident", "coarse_solve") or one in engine.ONCE_PER_RUN_PHASES:
                    continue
                in_step[one] = in_step.get(one, 0) + v
    # the state figure is the c-gh table's own, at the pilot's arena_frac
    old_peak = 112.476 * GB_ + resident + sum(in_step.values()) + 8 * 1.12 * GB_
    assert old_peak / GB_ > 237.0, (
        f"the run that OOM-killed reconstructs to {old_peak / GB_:.1f} GB, which "
        "fits a gg node -- so the corrected model would have passed it too"
    )

    new_solve = sum(v for k, v in m.items() if mp[k] == "coarse_solve")
    assert new_solve < old_solve / 2, (
        f"the hoisted build takes the solve from {old_solve / GB_:.1f} to "
        f"{new_solve / GB_:.1f} GB; the numpy half of that was measured at 23.7"
    )


def test_every_fine_arm_term_is_per_worker():
    """923313 died in a phase this module priced at 1.4 GB.

    In pool mode the parent builds NO tile kernels -- `run` hands it
    `(None, tile_geom(...))` -- so each of W workers builds its own triple and
    runs its own tile. The node holds W copies of every fine-arm term and the
    budget was carrying one. An instrument reading the parent cannot see this
    at all (ru_maxrss is one process's), so the accounting has to carry it by
    construction or nothing will.
    """
    fine = ("tile_kernels", "tile_workspace", "tile_kernel_build_f64",
            "tile_kernel_pref")
    one = _ec("c-gh").mesh_bytes()
    eight = EngineConfig(
        box_size=PRESETS["c-gh"]["box"], n_part=PRESETS["c-gh"]["n_part"],
        n_fine=PRESETS["c-gh"]["n_fine"], n_coarse=PRESETS["c-gh"]["n_coarse"],
        n_tile=PRESETS["c-gh"]["tile"], b_fine=PRESETS["c-gh"]["buf"],
        coarse_dtype="float32", fine_dtype="float64", tile_workers=8,
    )
    m8 = eight.mesh_bytes()
    for k in fine:
        assert m8[k] == 8 * one[k], f"{k} did not scale with tile_workers"
    # and no COARSE term moves: the long arm is solved once, in the parent
    for k, v in one.items():
        if k not in fine:
            assert m8[k] == v, f"{k} moved with the worker count and should not"
    # the per-tile buffers are per worker too
    assert (eight.step_bytes(eight.n_total, cap=1000)["tile_buffers"]
            == 8 * _ec("c-gh").step_bytes(eight.n_total, cap=1000)["tile_buffers"])


def test_the_tile_kernel_build_moves_into_the_loop_when_a_pool_runs_it():
    """`kernel_build` is held apart from the in-step sum because it runs once
    before the loop -- true of the PARENT. The workers build on their first
    task, inside the tile loop, so leaving their build in `kernel_build` would
    drop W builds out of the budget for the step the run keeps dying in."""
    serial = _ec("c-gh").mesh_phase()
    assert serial["tile_kernel_build_f64"] == "kernel_build"
    pooled = EngineConfig(
        box_size=PRESETS["c-gh"]["box"], n_part=PRESETS["c-gh"]["n_part"],
        n_fine=PRESETS["c-gh"]["n_fine"], n_coarse=PRESETS["c-gh"]["n_coarse"],
        n_tile=PRESETS["c-gh"]["tile"], b_fine=PRESETS["c-gh"]["buf"],
        coarse_dtype="float32", fine_dtype="float64", tile_workers=8,
    ).mesh_phase()
    assert pooled["tile_kernel_build_f64"] == "tile_loop"
    assert pooled["tile_kernel_pref"] == "tile_loop"
    # and the coarse arm's phases are untouched by the worker count
    assert pooled["coarse_kernels"] == serial["coarse_kernels"] == "coarse_solve"


def test_the_worker_count_reaches_the_config_the_planner_prices(capsys):
    """`--workers` fed only the shm table. It has to reach `EngineConfig`, or
    the fine-arm terms are priced for one worker while the shm table is priced
    for eight -- two halves of one report describing different runs, which is
    the fault the whole module exists to prevent."""
    main(["--preset", "c-gh", "--host-gb", "255.1", "--workers", "8",
          "--cap", "5284492"])
    eight = capsys.readouterr().out
    main(["--preset", "c-gh", "--host-gb", "255.1", "--cap", "5284492"])
    one = capsys.readouterr().out
    def bound(o):
        return float(o.split("a lower bound on the run's peak:")[1].split("GB")[0])

    assert bound(eight) > bound(one) + 20.0, (
        f"eight workers priced at {bound(eight):.1f} GB against one at "
        f"{bound(one):.1f}; the fine arm is not scaling"
    )


def test_the_pooled_migrate_holds_one_slab_per_worker():
    """The term that killed four jobs, and it was in the model as a constant.

    `drift_and_migrate_pooled` hands each worker a whole slab and
    `_eject_slab` returns keep/emig for all of it, so W slabs are live at once.
    The model carried 190 B per slab-row whatever was running -- a coefficient
    measured as the SERIAL pass's own increment, where the schedule stages
    2r+1 = 3 slabs in total. At c-gh that priced a pass at 12.75 GB which
    measured over 91, and Vista 923341 died in it, in the LEAD DRIFT, before
    reaching a single step.

    The per-row figures are measured (numpy domain, flat over an 8x change in
    slab rows at cdev8/cdev) and the two kernels are bitwise, so the 3.7x
    between them is a pure memory/wall trade.
    """
    g = PRESETS["c-gh"]
    kw = dict(box_size=g["box"], n_part=g["n_part"], n_fine=g["n_fine"],
              n_coarse=g["n_coarse"], n_tile=g["tile"], b_fine=g["buf"],
              coarse_dtype="float32", fine_dtype="float64")
    serial = EngineConfig(**kw)
    pooled = EngineConfig(**kw, tile_workers=8, eject_kernel="jax")
    lean = EngineConfig(**kw, tile_workers=8, eject_kernel="numpy")
    n = serial.n_total

    s = serial.step_bytes(n)["migrate_staging"]
    p = pooled.step_bytes(n)["migrate_staging"]
    m = lean.step_bytes(n)["migrate_staging"]

    # the serial arm keeps the coefficient it was measured with
    assert s / GB_ == pytest.approx(12.75, abs=0.2)
    # the pooled arm scales with W, and at the production kernel it is the
    # largest single per-step term at c-gh by a factor of three
    assert p / GB_ == pytest.approx(69.3, abs=1.0)
    assert p > 3 * EngineConfig(**kw, tile_workers=8).step_bytes(n)["repack_scratch"]
    # and the kernel choice is worth ~50 GB
    assert (p - m) / GB_ == pytest.approx(50.5, abs=1.5)
    # linear in W, because each worker holds one slab
    four = EngineConfig(**kw, tile_workers=4, eject_kernel="jax")
    assert 2 * four.step_bytes(n)["migrate_staging"] == p


def test_an_unmeasured_eject_kernel_is_refused_not_guessed():
    """A budget that invents a coefficient is indistinguishable from one that
    measured it, which is the fault this whole module exists against."""
    g = PRESETS["cdev"]
    ec = EngineConfig(
        box_size=g["box"], n_part=g["n_part"], n_fine=g["n_fine"],
        n_coarse=g["n_coarse"], n_tile=g["tile"], b_fine=g["buf"],
        coarse_dtype="float32", fine_dtype="float64",
        tile_workers=4, eject_kernel="numpy",
    )
    ec.eject_kernel = "something_new"
    with pytest.raises(ValueError, match="no measured eject coefficient"):
        ec.step_bytes(ec.n_total)


def test_c_gh_does_not_fit_a_gg_node_at_the_knobs_that_have_been_failing(capsys):
    """Four jobs said so; now the arithmetic does too.

    Pinned because the whole value of this module is that it refuses BEFORE a
    node is spent, and for the last four submissions it did the opposite -- it
    said FITS at 0.75x for a run that OOM-killed. If this ever flips back to
    FITS without a term being genuinely removed, something has been quietly
    dropped from the budget again.
    """
    # the pilot's own knobs: W=8, arena_frac 0.10, jax eject
    main(["--preset", "c-gh", "--host-gb", "255.1", "--workers", "8",
          "--cap", "5284492", "--eject-kernel", "jax", "--arena-frac", "0.10"])
    out = capsys.readouterr().out
    assert "DOES NOT FIT" in out
    est = float(out.split("a lower bound on the run's peak:")[1].split("GB")[0])
    assert est == pytest.approx(283.2, abs=2.0)


def test_bounding_ejects_charges_the_inserts_that_replace_them():
    """A bound that only counted what it held back would flatter itself.

    Measured, per row of the slab a task is handed: an eject is 129 B on the
    jax kernel and 35 on numpy, an insert 50. So a worker prevented from
    ejecting does not go idle -- it inserts -- and E ejects over W workers is
    E ejects PLUS W-E inserts. The consequence is worth pinning because it is
    counterintuitive: with the NUMPY eject the bound makes the pass BIGGER,
    since an insert costs more than the eject it displaces.
    """
    g = PRESETS["c-gh"]
    kw = dict(box_size=g["box"], n_part=g["n_part"], n_fine=g["n_fine"],
              n_coarse=g["n_coarse"], n_tile=g["tile"], b_fine=g["buf"],
              coarse_dtype="float32", fine_dtype="float64", tile_workers=8)
    n = EngineConfig(**kw).n_total

    def mig(**extra):
        return EngineConfig(**kw, **extra).step_bytes(n)["migrate_staging"]

    jax_free = mig(eject_kernel="jax")
    jax_cap2 = mig(eject_kernel="jax", migrate_eject_inflight=2)
    np_free = mig(eject_kernel="numpy")
    np_cap2 = mig(eject_kernel="numpy", migrate_eject_inflight=2)

    assert jax_cap2 < jax_free, "bounding the expensive side must help"
    assert np_cap2 > np_free, (
        "an insert is dearer than a numpy eject, so the bound must READ as a "
        "cost here -- a model showing a saving would be counting only what it "
        "held back"
    )
    # a bound at or above W is the unbounded case exactly
    assert mig(eject_kernel="jax", migrate_eject_inflight=8) == jax_free
    assert mig(eject_kernel="jax", migrate_eject_inflight=99) == jax_free
    # and the cheapest arrangement of the four is the plain numpy eject
    assert np_free == min(jax_free, jax_cap2, np_free, np_cap2)


# --------------------------------------------- the device column (--backend device)
#
# The CPU column stays the default and must not move when this one is added, so
# the first gate here is an invariance and not a new number.


def test_the_cpu_column_does_not_move_when_the_device_column_exists(capsys):
    """The refactor that put the load path behind a helper must be a no-op.

    Both backends build the state on the same host and transform the ICs out of
    core the same way, so `_print_load_and_ic` is shared -- and a shared helper
    is exactly where an accidental behaviour change hides. Pin the two lines the
    C-gh verdict is read off.
    """
    main(["--preset", "c-gh", "--host-gb", "237", "--arena-frac", "0.20",
          "--workers", "8"])
    out = capsys.readouterr().out
    assert "LOADING THE STATE, peak resident at each stage" in out
    assert "the total line above is a MAX: these stages do not coexist" in out
    assert "IC STAGE (out-of-core, 'derivative' policy)" in out
    # the CPU column still prices a host that holds the mesh
    assert "MESH, resident through the tile loop" in out
    assert "PER GPU" not in out


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_mesh_term_is_placed_deliberately(name):
    """A term with no side is the omitted-term fault, in the module about it.

    `DEVICE_PLACEMENT` is a design assertion and it will go stale the moment a
    new mesh term lands. It must go stale LOUDLY: defaulting an unplaced term to
    the host understates the GPU and defaulting it to the GPU understates the
    host, and either way the budget cannot be traded against.
    """
    from inexor.plan import DEVICE_PLACEMENT

    for k in _ec(name).mesh_bytes():
        assert k in DEVICE_PLACEMENT, f"{k} has no device placement"


def test_an_unplaced_mesh_term_is_refused_not_guessed(monkeypatch):
    """The anti-vacuity arm of the test above: prove the refusal actually fires."""
    from inexor import plan

    ec = _ec("cdev")
    real = ec.mesh_bytes

    def with_a_new_term():
        d = dict(real())
        d["a_term_nobody_placed"] = 1234
        return d

    monkeypatch.setattr(ec, "mesh_bytes", with_a_new_term)
    with pytest.raises(KeyError, match="a_term_nobody_placed"):
        plan.device_budget(ec, n=ec.n_total, n_gpus=4)


def test_one_gpu_charges_every_surviving_term_in_full():
    """The identity under the shard: at n_gpus=1 nothing is divided.

    This is what makes the /4 a SPLIT rather than a discount -- if the shard
    arithmetic were wrong in a way that scaled, this arm would catch it, because
    at one GPU the device column must reproduce `mesh_bytes` exactly for every
    term the design keeps.
    """
    from inexor.plan import DEVICE_PLACEMENT, device_budget

    ec = _ec("cgh64")
    mesh = ec.mesh_bytes()
    resident, transient, _phases, _worst, _slabs = device_budget(
        ec, n=ec.n_total, n_gpus=1)
    got = {**resident, **transient}
    kept = {k: v for k, v in mesh.items() if DEVICE_PLACEMENT[k] != "gone"}
    assert kept, "vacuous: no mesh term survives the placement"
    for k, v in kept.items():
        assert got[k] == v, k
    # and the deleted host pass is really gone
    assert "coarse_decode_slab" not in got


def test_the_shard_is_exactly_a_quarter_across_four_cards():
    from inexor.plan import DEVICE_PLACEMENT, device_budget

    ec = _ec("cgh64")
    mesh = ec.mesh_bytes()
    r1, t1, _p, _w, _s = device_budget(ec, n=ec.n_total, n_gpus=1)
    r4, t4, _p, _w, _s = device_budget(ec, n=ec.n_total, n_gpus=4)
    one, four = {**r1, **t1}, {**r4, **t4}
    sharded = [k for k, v in DEVICE_PLACEMENT.items()
               if v == "shard" and k in mesh]
    replicated = [k for k, v in DEVICE_PLACEMENT.items()
                  if v == "replica" and k in mesh]
    assert sharded and replicated, "vacuous: one of the two classes is empty"
    for k in sharded:
        assert four[k] == int(one[k] / 4), k
    for k in replicated:
        assert four[k] == one[k], f"{k} is replicated and must not shrink"


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_the_slab_window_is_the_brick_span_and_never_wraps_the_box(name):
    """DERIVED, not picked. A window smaller than the span would drop members.

    `brick_span` is the same function `SlotState.tile_bricks` walks, so this is
    the membership contract read as a residency requirement rather than a
    second, independent guess at it.
    """
    from inexor.layout import brick_span
    from inexor.plan import device_window_slabs

    ec = _ec(name)
    nb = max(1, ec.n_fine // ec.n_brick)
    _pad, span = brick_span(ec.n_tile, ec._b_realized, ec.n_brick, nb)
    got = device_window_slabs(ec)
    assert got == min(span, nb)
    assert 1 <= got <= nb, "a window wider than the brick grid double-counts"


def test_c_hero_fits_a_gb_node_on_the_device_backend_and_the_cpu_one_does_not(capsys):
    """The verdict the whole design turns on, both directions, one machine.

    The CPU column at 4096^3 does not fit a 1026 GB host at any worker count;
    the device column does, because the mesh moves to the cards. If this ever
    flips, the build has lost its premise and the record must say so.
    """
    main(["--preset", "c-hero", "--host-gb", "1026", "--arena-frac", "0.01",
          "--workers", "1"])
    cpu = capsys.readouterr().out
    assert "DOES NOT FIT" in cpu.split("against --host-gb")[1]

    main(["--preset", "c-hero", "--backend", "device", "--host-gb", "1026",
          "--device-gb", "199", "--arena-frac", "0.01"])
    dev = capsys.readouterr().out
    assert "against --host-gb 1026.0: FITS" in dev
    assert "against --device-gb 199.0 per card: FITS" in dev
    assert "the slab window is 18 x-slabs" in dev
