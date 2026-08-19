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

    It reads 197.2 now and that is a CORRECTION, not a regression. 164.6 came
    from charging the single largest transient and summing the per-step host
    terms; this charges every phase inside a step, which is the model that
    would have refused the run that OOM-killed. The engine change went the other
    way at the same time -- hoisting the coarse kernel build took 23.7 GB off
    the solve -- so the number would be worse still without it.

    Pinned because the margin is what makes a capacity run proposable at all,
    and it is now thin: still a LOWER BOUND, one measured to read ~1.9x low at
    cdev, so fitting on paper is not the same as fitting."""
    main(["--preset", "c-gh", "--host-gb", "237", "--cap", "5284492"])
    out = capsys.readouterr().out
    assert "FITS" in out and "DOES NOT FIT" not in out
    est = float(out.split("a lower bound on the run's peak:")[1].split("GB")[0])
    assert est == pytest.approx(197.2, abs=1.0)
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
    """Still a floor, and still known-soft: 3.396 GB against job 446's measured
    7.461 at cdev, so 2.20x low where it has been checked. The ladder is
    3.783 -> 3.305 -> 3.396: the M-v2-6 terms and then the phase model move it
    in both directions and neither is claimed to close the gap, which is why
    the wording stays. What the phase model DID fix is the shape of the error
    at C-gh, where the old form under-charged the coarse solve by 3x."""
    main(["--preset", "cdev", "--host-gb", "124", "--cap", "5284492"])
    out = capsys.readouterr().out
    assert "LOWER BOUND" in out and "not a measurement" in out
    est = float(out.split("a lower bound on the run's peak:")[1].split("GB")[0])
    # 3.445 (derived 9 B/row) -> 3.488 (measured 11.1, out of place) -> 3.305
    # (measured 2.1, in place) -> 3.396 (phases summed within a step).
    # scripts/v2_m6_repack_bytes.py.
    assert est == pytest.approx(3.396, abs=0.01)
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

    The gap is two int64 occupancy arrays over `n_buckets` plus the sort, and it
    was invisible for as long as the term was derived from the payload width
    alone. Measured flat to 2.6% over 64x in particle count -- 11.35 / 11.09 /
    11.06 B/row at 262k / 2.1M / 16.8M particles -- which is what says it is a
    coefficient and not a fixed cost being amortized.

    Pinned because the derived figure is the intuitive one and would be an easy
    "simplification" to reintroduce.
    """
    n = PRESETS["c-gh"]["n_part"] ** 3
    ec = _ec("c-gh")
    rows = int(np.ceil(np.ceil(n * 1.10) * 1.10))
    scratch = ec.step_bytes(n, n_rows=rows)["repack_scratch"]
    # 9 derived -> 11.1 measured out of place -> 2.1 measured in place.
    assert scratch / rows == pytest.approx(2.1, abs=0.05)
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

    resident = sum(v for k, v in m.items() if engine.MESH_PHASE[k] == "resident")
    step = ec.step_bytes(ec.n_total, cap=5284492)
    in_step = {"coarse_solve": old_solve}
    for src, phase_of in ((m, engine.MESH_PHASE), (step, engine.STEP_PHASE)):
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

    new_solve = sum(v for k, v in m.items()
                    if engine.MESH_PHASE[k] == "coarse_solve")
    assert new_solve < old_solve / 2, (
        f"the hoisted build takes the solve from {old_solve / GB_:.1f} to "
        f"{new_solve / GB_:.1f} GB; the numpy half of that was measured at 23.7"
    )
