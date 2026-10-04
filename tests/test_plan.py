"""`python -m inexor.plan`: the reduction that prints a verdict, the measured coefficients
the byte model is anchored to, and that every term is either in a table or named as omitted.
Per-term arithmetic (`mesh_bytes`, `step_bytes`, ...) is tested where it lives.
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


# ------------------------------------------------------ the measured anchors


@pytest.mark.parametrize("name", ["cdev", "cgh64", "c-gh"])
def test_the_tile_transient_does_not_grow_from_the_anchor_to_c_gh(name):
    """cdev, cgh64 and C-gh share particles per tile and padded tile, so the tile transients
    are identical up the ladder and the measured cdev tile phase (3.432 GB) prices C-gh.
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
    """`kick_pending` stays in the table at zero (a vanished term looks uncounted); per-brick
    velocity scales cost >1e4x less than its 32 B/particle.
    """
    n = PRESETS["c-gh"]["n_part"] ** 3
    ec = _ec("c-gh")
    assert ec.step_bytes(n)["kick_pending"] == 0
    assert "kick_pending" in ec.step_bytes(n), "the retired term must stay visible"
    n_bricks = (PRESETS["c-gh"]["n_fine"] // ec.n_brick) ** 3
    assert n_bricks * 8 < 0.02e9, "the replacement should be tens of MB, not GB"
    assert (n * 32) / (n_bricks * 8) > 1e4, "expected five orders, not a trim"


def test_the_crossover_that_made_the_anchor_the_last_tile_dominated_rung():
    """A 32 B/p term overtakes the measured cdev tile phase (3.432 GB) between cdev and
    cgh64; context for cards measured while `kick_pending` was charged.
    """
    measured_tile_phase = 3.432e9  # cdev, `tile_short` own increment
    n_cross = measured_tile_phase / 32
    assert PRESETS["cdev"]["n_part"] ** 3 < n_cross < PRESETS["cgh64"]["n_part"] ** 3
    assert 400 < round(n_cross ** (1 / 3)) < 500


def test_the_tile_kernel_build_reproduces_the_measured_phase():
    """The tile kernel build (kernels + `k2_true`, `k2_safe`, `fac`, `pref`) matches the
    measured `membership` phase within 2% at cdev8 and cdev (only cdev separates the kernel and
    particle axes), and is 5/3 of `tile_kernels` at f64.
    """
    for name, measured_mb in (("cdev8", 85.60), ("cdev", 1319.45)):
        m = _ec(name).mesh_bytes()
        build = (m["tile_kernels"] + m["tile_kernel_build_f64"] + m["tile_kernel_pref"])
        assert build / 1e6 == pytest.approx(measured_mb, rel=0.02), (
            f"{name}: modelled {build / 1e6:.2f} MB against a measured "
            f"{measured_mb} MB for the tile kernel build"
        )
        # 5/3 of the three kernels alone at f64, by derivation
        assert build == pytest.approx(m["tile_kernels"] * 5 / 3, rel=1e-9)


def test_the_low_rank_ik_grids_are_why_the_factor_is_five_thirds():
    """`kernel_grids` returns low-rank ik broadcasts; full grids would make the factor 2.17."""
    from inexor.forces import kernel_grids

    ikx, iky, ikz, k2_true, k2_safe = kernel_grids((16, 16, 16), 1.0, np.float64)
    for a in (ikx, iky, ikz):
        assert a.size <= 16, f"ik grid is full-rank ({a.size} elements); the 5/3 is stale"
    assert k2_true.size == k2_safe.size == 16 * 16 * 9


def test_both_arms_count_the_prefactor():
    """`pref` is a full half-grid in both the coarse and tile arms and is charged in both."""
    m = _ec("cdev").mesh_bytes()
    assert m["coarse_kernel_pref"] > 0 and m["tile_kernel_pref"] > 0


def test_the_coarse_force_copy_is_one_component_not_three():
    """`coarse_force_meshes` solves one component at a time into the caller's buffers, so
    the copy transient is one mesh (a third of the resident force).
    """
    m = _ec("cdev").mesh_bytes()
    assert m["coarse_force_copy_transient"] * 3 == m["coarse_force_resident"]


def test_migration_staging_is_n_to_the_two_thirds_not_n():
    """Migration staging is N^(2/3) (a few x-slabs in flight): measured coefficients at
    cdev8 and cdev agree within 8%, and 8x the particles gives 4x the term.
    """
    for name, measured_mb in (("cdev8", 47.62), ("cdev", 208.58)):
        p = PRESETS[name]
        got = _ec(name).step_bytes(p["n_part"] ** 3)["migrate_staging"]
        assert got / 1e6 == pytest.approx(measured_mb, rel=0.08), (
            f"{name}: modelled {got / 1e6:.2f} MB against a measured {measured_mb}"
        )
    a = _ec("cdev8").step_bytes(PRESETS["cdev8"]["n_part"] ** 3)["migrate_staging"]
    b = _ec("cdev").step_bytes(PRESETS["cdev"]["n_part"] ** 3)["migrate_staging"]
    assert b / a == pytest.approx(4.0, rel=0.02), "migration staging stopped being N^(2/3)"


def test_per_step_terms_do_not_depend_on_slack():
    """Only `repack_scratch` depends on `n_rows`: measured, slack 0.2 -> 0.5 moves n_rows
    1.217x and every per-step phase 1.000x.
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
    """The "largest single term" line considers transients: at cdev it names
    `tile_workspace` (1.443 GB) over `tile_kernels` (0.791).
    """
    main(["--preset", "cdev", "--host-gb", "124", "--cap", "5284492"])
    out = capsys.readouterr().out
    assert "largest single term: tile_workspace (transient)" in out


def test_the_binding_term_at_c_gh_is_now_the_state_itself(capsys):
    """At C-gh the largest term is the 9 B/p state payload, and C-gh does not fit a 116 GB
    `gh` host (~1.70x).
    """
    main(["--preset", "c-gh", "--host-gb", "116", "--cap", "5284492"])
    out = capsys.readouterr().out
    assert "largest single term: t9_payload" in out
    assert "DOES NOT FIT" in out
    ratio = float(out.split("DOES NOT FIT (")[1].split("x")[0])
    assert 1.4 < ratio < 2.0, f"expected ~1.70x a gh host, got {ratio}"


def test_c_gh_now_fits_a_cpu_only_node_with_margin(capsys):
    """C-gh fits a 237 GB `gg` node on paper at a 167.9 GB lower bound (a bound measured
    ~1.9x low at cdev).
    """
    main(["--preset", "c-gh", "--host-gb", "237", "--cap", "5284492"])
    out = capsys.readouterr().out
    assert "FITS" in out and "DOES NOT FIT" not in out
    est = float(out.split("a lower bound on the run's peak:")[1].split("GB")[0])
    assert est == pytest.approx(167.889, abs=1.0)
    assert est < 237.0, "the bound no longer fits the node it was sized for"


def test_removing_the_repack_scratch_too_would_still_not_reach_a_gh_host(capsys):
    """State (98.6 GB) plus resident mesh (18.0) exceed a 116 GB `gh` host before any
    transient, so no per-step reduction makes C-gh fit one.
    """
    main(["--preset", "c-gh", "--host-gb", "116", "--cap", "5284492"])
    out = capsys.readouterr().out
    state_total = float(out.split("STATE")[1].split("total")[1].split("GB")[0])
    mesh_res = float(out.split("MESH, resident")[1].split("total")[1].split("GB")[0])
    assert state_total + mesh_res > 116.0, (
        "the resident floor no longer clears a gh host, so this test's premise -- "
        "and the milestone's target-machine argument -- needs re-deriving"
    )


def test_an_absent_cap_names_the_omission_instead_of_dropping_it(capsys):
    """Without `--cap`, `tile_buffers` is absent from the tables and named as excluded."""
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
    """The estimate is labelled a lower bound: 3.343 GB at cdev, under the measured 7.461."""
    main(["--preset", "cdev", "--host-gb", "124", "--cap", "5284492"])
    out = capsys.readouterr().out
    assert "LOWER BOUND" in out and "not a measurement" in out
    est = float(out.split("a lower bound on the run's peak:")[1].split("GB")[0])
    assert est == pytest.approx(3.343, abs=0.01)
    assert est < 7.461, "the bound must sit under the measured peak it bounds"


def test_no_host_budget_gives_no_verdict(capsys):
    """Without `--host-gb` there is no verdict; a default host size would invent one."""
    main(["--preset", "cdev"])
    out = capsys.readouterr().out
    assert "no --host-gb given, so no verdict" in out
    assert "FITS" not in out


def test_every_preset_is_evaluable():
    """Every preset evaluates; `smoke` needs its own b=8 (b=32 pads its tile past n_fine)."""
    for name in PRESETS:
        assert main(["--preset", name, "--host-gb", "116"]) == 0
        assert np.isfinite(sum(_ec(name).mesh_bytes().values()))


def test_a_preset_buffer_is_not_shadowed_by_the_flag_default(capsys):
    """A preset's own `buf` reaches the config, and an explicit `--buf` still overrides it."""
    main(["--preset", "smoke", "--host-gb", "116"])
    assert "b=8" in capsys.readouterr().out
    main(["--preset", "smoke", "--host-gb", "116", "--buf", "4"])
    assert "b=4" in capsys.readouterr().out, "an explicit flag must still win"


def test_the_repack_scratch_coefficient_is_the_measured_one_not_the_payload_width():
    """`repack_scratch` is the measured 0.49 B/row (flat over 8x in N), not the 9 B/row
    payload width; >= 9 would mean the repack copies the payload again.
    """
    n = PRESETS["c-gh"]["n_part"] ** 3
    ec = _ec("c-gh")
    rows = int(np.ceil(np.ceil(n * 1.10) * 1.10))
    scratch = ec.step_bytes(n, n_rows=rows)["repack_scratch"]
    assert scratch / rows == pytest.approx(0.49, abs=0.02)
    assert scratch < 30e9, "the in-place rewrite should put this well under a gh host"
    assert scratch / rows < 9.0, (
        "the coefficient is back at or above the payload width, so the repack is "
        "copying the payload again rather than rearranging it in place"
    )


def test_the_in_place_reference_would_make_the_repack_scratch_worse():
    """`BrickPackedLayout.repack` measures 39.4 B/row (its reported scratch omits `live`,
    `parts`, `final`), >15x the current coefficient, so porting it would regress.
    """
    n = PRESETS["c-gh"]["n_part"] ** 3
    rows = int(np.ceil(np.ceil(n * 1.10) * 1.10))
    current = _ec("c-gh").step_bytes(n, n_rows=rows)["repack_scratch"]
    reference_would_be = rows * 39.4
    assert reference_would_be > current, "the port is only worth doing if it wins"
    # ~19x the current coefficient (and 3.55x an out-of-place copy at 11.1 B/row)
    assert reference_would_be / current > 15.0


def test_the_load_model_matches_what_the_loader_actually_allocates(tmp_path):
    """`load_stages` against the real loader's shared-memory bytes at a small config:
    payload + index + arena within 2%, and the modelled peak at or above them.
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
        measured = alloc.bytes_held()
        n_rows = got.off.shape[0]
        n_buckets = t9.n_buckets_side**3
        idx = np.dtype(got.occupancy.dtype).itemsize
        modelled = load_stages(
            n=n_part**3, n_rows=n_rows, n_buckets=n_buckets, index_itemsize=idx,
            n_arena=got.arena_bucket.shape[0], n_bricks=nb**3, n_slabs=n_slabs,
            shared=True)
        # payload + index + arena
        want = n_rows * 9 + n_buckets * idx + got.arena_bucket.nbytes
        assert measured == pytest.approx(want, rel=0.02), (
            f"the allocator holds {measured} B where the model wants {want} B")
        assert max(modelled.values()) >= measured
    finally:
        alloc.close()


def test_the_phase_model_would_have_refused_the_run_that_died():
    """The phase model refuses the monolithic-coarse-solve configuration that ran out of
    memory (~79 GB lost in the solve).

    Reconstructed from current terms that solve is 71.0 GB, under the measurement as a lower
    bound must be, and the phase sum exceeds a 237 GB `gg` node. The factorized solve is <half.
    """
    ec = _ec("c-gh")
    m = ec.mesh_bytes()
    # The monolithic solve: three kernels built and matched every step (a second
    # triple), the f64 build every step, three force meshes copied at once; in
    # complex half-grids (`coarse_spectrum` is one).
    half_grid = m["coarse_spectrum"]
    old_solve = (6 * half_grid          # three kernels, each matched: a second triple
                 + m["coarse_kernel_build_f64"]
                 + m["coarse_kernel_pref"] + m["coarse_match_factor"]
                 + 3 * half_grid        # monolithic dk, its device copy, their product
                 + 3 * m["coarse_force_copy_transient"])
    assert old_solve / GB_ == pytest.approx(71.0, abs=1.0), (
        f"the monolithic coarse solve reconstructs to {old_solve / GB_:.1f} GB; if this "
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
    # the state figure is the c-gh table's own, at arena_frac 0.10
    old_peak = 112.476 * GB_ + resident + sum(in_step.values()) + 8 * 1.12 * GB_
    assert old_peak / GB_ > 237.0, (
        f"the failing configuration reconstructs to {old_peak / GB_:.1f} GB, which "
        "fits a gg node -- so the corrected model would have passed it too"
    )

    new_solve = sum(v for k, v in m.items() if mp[k] == "coarse_solve")
    assert new_solve < old_solve / 2, (
        f"the hoisted build takes the solve from {old_solve / GB_:.1f} to "
        f"{new_solve / GB_:.1f} GB; the numpy half of that was measured at 23.7"
    )


def test_every_fine_arm_term_is_per_worker():
    """With W tile workers every fine-arm term and `tile_buffers` is charged W times and no
    coarse term moves (each worker builds its own kernels; the parent's ru_maxrss cannot see it).
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
    for k, v in one.items():
        if k not in fine:
            assert m8[k] == v, f"{k} moved with the worker count and should not"
    assert (eight.step_bytes(eight.n_total, cap=1000)["tile_buffers"]
            == 8 * _ec("c-gh").step_bytes(eight.n_total, cap=1000)["tile_buffers"])


def test_the_tile_kernel_build_moves_into_the_loop_when_a_pool_runs_it():
    """With a tile pool, the workers build their kernels on their first task inside the
    tile loop, so the tile build terms are charged to `tile_loop` rather than to the once-per-run
    `kernel_build`; coarse phases do not move.
    """
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
    assert pooled["coarse_spectrum"] == serial["coarse_spectrum"] == "coarse_solve"


def test_the_worker_count_reaches_the_config_the_planner_prices(capsys):
    """`--workers` reaches `EngineConfig`, so the fine-arm terms and the shm table price the
    same worker count (eight workers add >20 GB at C-gh).
    """
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
    """Pooled migration stages one slab per worker, linear in W: 12.75 GB serial and 69.3 GB
    pooled (W=8, jax eject) at C-gh; the numpy eject saves ~50.5 GB. Per-row coefficients are
    measured; the two eject kernels are bitwise.
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

    assert s / GB_ == pytest.approx(12.75, abs=0.2)
    # pooled, jax eject: over 3x repack_scratch
    assert p / GB_ == pytest.approx(69.3, abs=1.0)
    assert p > 3 * EngineConfig(**kw, tile_workers=8).step_bytes(n)["repack_scratch"]
    assert (p - m) / GB_ == pytest.approx(50.5, abs=1.5)
    four = EngineConfig(**kw, tile_workers=4, eject_kernel="jax")
    assert 2 * four.step_bytes(n)["migrate_staging"] == p


def test_an_unmeasured_eject_kernel_is_refused_not_guessed():
    """An eject kernel with no measured coefficient raises rather than being priced."""
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
    """At W=8, arena_frac 0.10, jax eject, C-gh does not fit a 255.1 GB node (~270.6 GB)."""
    main(["--preset", "c-gh", "--host-gb", "255.1", "--workers", "8",
          "--cap", "5284492", "--eject-kernel", "jax", "--arena-frac", "0.10"])
    out = capsys.readouterr().out
    assert "DOES NOT FIT" in out
    est = float(out.split("a lower bound on the run's peak:")[1].split("GB")[0])
    assert est == pytest.approx(270.578, abs=2.0)


def test_bounding_ejects_charges_the_inserts_that_replace_them():
    """Capping concurrent ejects at E charges E ejects plus W-E inserts (measured per slab
    row: eject 129 B jax / 35 B numpy, insert 50 B), so the cap costs more with the numpy eject.
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
    assert mig(eject_kernel="jax", migrate_eject_inflight=8) == jax_free
    assert mig(eject_kernel="jax", migrate_eject_inflight=99) == jax_free
    # the uncapped numpy eject is the cheapest of the four
    assert np_free == min(jax_free, jax_cap2, np_free, np_cap2)


# --------------------------------------------- the device column (--backend device)


def test_the_cpu_column_does_not_move_when_the_device_column_exists(capsys):
    """The shared `_print_load_and_ic` helper leaves the CPU column unchanged."""
    main(["--preset", "c-gh", "--host-gb", "237", "--arena-frac", "0.20",
          "--workers", "8"])
    out = capsys.readouterr().out
    assert "LOADING THE STATE, peak resident at each stage" in out
    assert "the total line above is a MAX: these stages do not coexist" in out
    assert "IC STAGE (out-of-core, 'derivative' policy)" in out
    assert "MESH, resident through the tile loop" in out
    assert "PER GPU" not in out


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_mesh_term_is_placed_deliberately(name):
    """Every mesh term has an entry in `DEVICE_PLACEMENT`; a defaulted side would understate
    either the GPU or the host.
    """
    from inexor.plan import DEVICE_PLACEMENT

    for k in _ec(name).mesh_bytes():
        assert k in DEVICE_PLACEMENT, f"{k} has no device placement"


def test_an_unplaced_mesh_term_is_refused_not_guessed(monkeypatch):
    """Anti-vacuity for the test above: an unplaced term raises KeyError."""
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


@pytest.mark.parametrize("kernel", ["cards", "host"])
def test_one_gpu_charges_every_surviving_term_in_full(kernel):
    """At n_gpus=1 each surviving term is charged in full plus its ghost planes (so /4 is a
    split), and lands in exactly one of the device and host columns.
    """
    from inexor.plan import device_budget, device_placement, shard_halo_planes

    placed = device_placement(kernel)
    ec = _ec("cgh64")
    mesh = ec.mesh_bytes()
    halo = shard_halo_planes()
    resident, transient, _phases, _worst, _slabs, host_mesh, _after = device_budget(
        ec, n=ec.n_total, n_gpus=1, kernel=kernel)
    got = {**resident, **transient}
    kept = {k: v for k, v in mesh.items()
            if placed[k] not in ("gone", "host", "host_block")}
    assert kept, "vacuous: no mesh term survives the placement"
    for k, v in kept.items():
        assert got[k] == v + (v // ec.n_coarse) * halo.get(k, 0), k
    # the host decode pass does not exist on the device backend
    assert "coarse_decode_slab" not in got
    # every host-placed term is charged to `host_mesh` at full value and not to the
    # device, or it would be charged nowhere
    on_host = {k: v for k, v in mesh.items() if placed[k] in ("host", "host_block")}
    assert on_host, "vacuous: no term is host-placed, so this arm proves nothing"
    for k, v in on_host.items():
        assert host_mesh[k] == v, f"{k} is host-placed but not charged to the host"
        assert k not in got, f"{k} is charged to BOTH columns"


def test_the_shard_is_a_quarter_plus_its_ghost_planes_across_four_cards():
    from inexor.plan import DEVICE_PLACEMENT, device_budget, shard_halo_planes

    ec = _ec("cgh64")
    mesh = ec.mesh_bytes()
    halo = shard_halo_planes()
    r1, t1, _p, _w, _s, _h, _a = device_budget(ec, n=ec.n_total, n_gpus=1)
    r4, t4, _p, _w, _s, _h, _a = device_budget(ec, n=ec.n_total, n_gpus=4)
    one, four = {**r1, **t1}, {**r4, **t4}
    sharded = [k for k, v in DEVICE_PLACEMENT.items()
               if v == "shard" and k in mesh]
    replicated = [k for k, v in DEVICE_PLACEMENT.items()
                  if v == "replica" and k in mesh]
    assert sharded and replicated, "vacuous: one of the two classes is empty"
    assert any(halo.get(k) for k in sharded), "vacuous: no sharded term has ghost planes"
    for k in sharded:
        plane = mesh[k] // ec.n_coarse
        assert four[k] == int(mesh[k] / 4) + plane * halo.get(k, 0), k
    for k in replicated:
        assert four[k] == one[k], f"{k} is replicated and must not shrink"


def test_the_card_force_shard_is_charged_the_bytes_a_gb200_held():
    """One card's C-hero coarse force shard (3 f32 meshes of 516 x 2048 x 2048) is charged
    the 25,971,130,368 bytes measured on a GB200.
    """
    from inexor.plan import device_budget

    ec = _ec("c-hero")
    resident, *_ = device_budget(ec, n=ec.n_total, n_gpus=4)
    assert resident["coarse_force_resident"] == 25_971_130_368


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_the_slab_window_is_the_brick_span_and_never_wraps_the_box(name):
    """The device slab window is `brick_span` (what `SlotState.tile_bricks` walks) capped at
    the brick grid.
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
    """C-hero does not fit a 1026 GB host on the CPU backend and fits it (and a 199 GB card)
    on the device backend.
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


def test_the_fused_pass_charges_its_census_in_the_tile_loop_and_one_pass_after():
    """The fused migrate + repack charges its census (one padded slab through the eject
    kernel) in the tile loop and one pass after it; `fused=False` charges neither.
    """
    from inexor.device.migrate import EJECT_B_PER_PADDED_ROW
    from inexor.eject_jax import _padded
    from inexor.plan import (
        FUSED_DEVICE_B_PER_SLAB_ROW,
        MIGRATE_DEVICE_B_PER_SLAB_ROW,
        device_budget,
    )

    ec = _ec("c-hero")
    nb = ec.n_fine // ec.n_brick
    slab_rows = ec.n_total / nb
    _r, t_f, p_f, _w, _s, _h, a_f = device_budget(ec, n=ec.n_total, n_gpus=4)
    _r, t_s, p_s, _w, _s, _h, a_s = device_budget(ec, n=ec.n_total, n_gpus=4, fused=False)
    census = int(EJECT_B_PER_PADDED_ROW * _padded(int(slab_rows)))
    assert t_f["census_eject (fused pass, one padded slab)"] == census
    assert p_f["tile_loop"] - p_s["tile_loop"] == census
    assert not any("census" in k for k in t_s)
    assert list(a_f.values()) == [int(FUSED_DEVICE_B_PER_SLAB_ROW * slab_rows)]
    assert max(a_s.values()) == int(MIGRATE_DEVICE_B_PER_SLAB_ROW * slab_rows)


def test_the_host_column_charges_the_repacks_second_bucket_index(capsys):
    from inexor.codec import T9Layout
    from inexor.plan import PRESETS, BUCKET_CELLS

    main(["--preset", "c-hero", "--backend", "device", "--host-gb", "1026",
          "--device-gb", "199", "--arena-frac", "0.01"])
    out = capsys.readouterr().out
    g = PRESETS["c-hero"]
    want = T9Layout(g["box"], g["n_part"], BUCKET_CELLS).index_bytes() / 1e9
    line = next(ln for ln in out.splitlines() if "repack new_occ" in ln)
    assert abs(float(line.split()[-2]) - want) < 1e-3
    main(["--preset", "c-hero", "--backend", "device", "--host-gb", "1026",
          "--device-gb", "199", "--arena-frac", "0.01", "--separate-passes"])
    sep = capsys.readouterr().out
    assert "census_eject" not in sep and "migrate_device_pass" in sep


def test_the_host_column_charges_one_slab_of_w_per_card_for_the_window_write_back(capsys):
    from inexor.forces import capacity_shape
    from inexor.plan import PRESETS

    main(["--preset", "c-hero", "--backend", "device", "--host-gb", "1026",
          "--device-gb", "199", "--arena-frac", "0.01", "--n-gpus", "4"])
    out = capsys.readouterr().out
    g = PRESETS["c-hero"]
    nb = 256
    want = 4 * int(capacity_shape(g["n_part"] ** 3 // nb)) * 6 / 1e9
    line = next(ln for ln in out.splitlines() if "tile_window write-back" in ln)
    assert abs(float(line.split()[-2]) - want) < 1e-3


# ------------------------------------- the device lane's host column, against measured runs


def _device_host(capsys, preset, cards, kernel="cards"):
    """(resident GB, {phase: GB above the resident, credits excluded}, peak GB) as printed."""
    main(["--preset", preset, "--backend", "device", "--n-gpus", str(cards),
          "--arena-frac", "0.01", "--coarse-kernel", kernel])
    out = capsys.readouterr().out
    res_block = out.split("HOST, resident for the whole run")[1].split("\n\n")[0]
    resident = float(next(ln for ln in res_block.splitlines()
                          if ln.strip().startswith("total")).split()[-2])
    ph_block = out.split("HOST, by phase")[1].split("\n\n")[0]
    phases = {}
    for ln in ph_block.splitlines():
        if ":" in ln and ln.rstrip().endswith("GB") and "credit" not in ln:
            p = ln.split(":")[0].strip()
            phases[p] = phases.get(p, 0.0) + float(ln.split()[-2])
    peak = float(out.split("host, a lower bound on the run's peak:")[1].split()[0])
    return resident, phases, peak


def test_the_host_phases_match_the_4096_run_they_price(capsys):
    """c-hero on four GB200s (job 1003657, `runs/v2/d7b_1003657_hero.json`, the planner's
    default knobs): each code-derived phase's host transient above that phase's closing RSS
    is within 10% of the measured one, and the resident is at most 5% under the RSS floor
    after the first step (844.6-865.4 GB), and never over it. That run held the coarse kernel
    on the host.
    """
    resident, phases, _ = _device_host(capsys, "c-hero", 4, kernel="host")
    measured = {"kernel_build": 860.6 - 733.4, "coarse_solve": 830.6 - 761.9,
                "migrate": 895.5 - 855.5}
    for p, m in measured.items():
        assert abs(phases[p] / m - 1) <= 0.10, f"{p}: priced {phases[p]:.1f} vs measured {m:.1f}"
    lo, hi = 844.6, 865.4
    assert resident <= lo, f"resident {resident:.1f} GB is over the measured floor {lo}"
    assert resident >= 0.95 * hi, f"resident {resident:.1f} GB is >5% under the floor {hi}"


def test_the_process_baseline_closes_the_single_gh200_floor(capsys):
    """One GH200 (job 1029876): the priced resident, measured baseline included, is within 5%
    of the RSS after step 1 at both sizes (5.885 GB at cgh64, 17.237 at c-1024), never over.
    That run held the coarse kernel on the host."""
    for preset, floor in (("cgh64", 5.885), ("c-1024", 17.237)):
        resident, _, _ = _device_host(capsys, preset, 1, kernel="host")
        assert 0.95 * floor <= resident <= floor, (preset, resident, floor)


def _rung_lines(out):
    """(tile-loop rung jump GB, held-from-the-rung GB) as printed."""
    jump = float(next(ln for ln in out.splitlines()
                      if "recompiled at a capacity rung" in ln).split()[-2])
    held = float(out.split("but the tile loop:")[1].split()[0])
    return jump, held


@pytest.mark.parametrize("preset,cards,jump,kept", [
    # per card, the smallest measured: c-1024 on 1 and 2 gh (job 1043437, rungs 74 and 116),
    # c-hero on 4 GB200 (1024784, rung 61); the kept rise is every phase's, 5 steps on
    ("c-1024", 1, (5.28, 5.41), (1.67, 2.26)),
    ("c-hero", 4, (16.3, 16.3), (0.51, 0.89)),
])
def test_the_rung_terms_are_the_measured_ones_per_card(capsys, preset, cards, jump, kept):
    main(["--preset", preset, "--backend", "device", "--n-gpus", str(cards)])
    got_jump, got_held = _rung_lines(capsys.readouterr().out)
    assert jump[0] - 0.01 <= got_jump / cards <= jump[1] + 0.01
    assert kept[0] - 0.01 <= got_held / cards <= kept[1] + 0.01


def test_what_the_rung_keeps_is_charged_to_every_in_step_phase_but_the_tile_loop(capsys):
    """The tile loop's rung jump is measured above the step before, so it already sits on
    the kept bytes; the kernel build runs once, before any rung. At c-gh on 2 gh the coarse
    solve binds with them in it."""
    main(["--preset", "c-gh", "--backend", "device", "--n-gpus", "1", "--n-nodes", "2"])
    out = capsys.readouterr().out
    resident = float(next(ln for ln in out.split("HOST, resident for the whole run")[1]
                          .split("\n\n")[0].splitlines()
                          if ln.strip().startswith("total")).split()[-2])
    sums = {}
    for ln in out.split("HOST, by phase")[1].split("\n\n")[0].splitlines():
        if ":" in ln and ln.rstrip().endswith("GB"):
            p = ln.split(":")[0].strip()
            sums[p] = sums.get(p, 0.0) + float(ln.split()[-2])
    _, held = _rung_lines(out)
    want = resident + max(s + (0 if p in ("tile_loop", "kernel_build") else held)
                          for p, s in sums.items())
    peak = float(out.split("host, a lower bound on the run's peak:")[1].split()[0])
    assert abs(peak - want) < 0.01, (peak, want)


@pytest.mark.parametrize("args,measured", [
    # whole-run host peaks, GB per node, of runs with the persistent compilation cache the
    # rung jump is priced with, each priced as it ran (coarse kernel on the host before
    # 6519f1e). 1043437: one gh (the smaller of the two reference nodes) and the busier of 2
    # ranks
    (["--preset", "c-1024", "--n-gpus", "1"], 25.96),
    (["--preset", "c-1024", "--n-gpus", "1", "--n-nodes", "2"], 19.26),
    # c-hero on 4 GB200 over 120 steps (1024783 / 1024784 / 1027664), at the rung
    (["--preset", "c-hero", "--n-gpus", "4", "--coarse-kernel", "host"], 943.0),
])
def test_the_host_peak_stays_under_every_measured_whole_run_peak(capsys, args, measured):
    main(args + ["--backend", "device"])
    out = capsys.readouterr().out
    peak = float(out.split("host, a lower bound on the run's peak:")[1].split()[0])
    assert peak <= measured, f"priced {peak:.2f} GB over the measured {measured} GB"


def test_the_pre_step_phases_are_credited_the_untouched_slack(capsys):
    """The kernel build and the lead drift run before any insert has touched the slack rows,
    so both are charged against the resident minus the slack; the credit is printed."""
    main(["--preset", "c-hero", "--backend", "device", "--arena-frac", "0.01"])
    out = capsys.readouterr().out
    credits = [ln for ln in out.splitlines() if "slack rows not yet touched" in ln]
    assert sorted(ln.split(":")[0].strip() for ln in credits) == ["kernel_build", "lead_drift"]
    slack = float(next(ln for ln in out.splitlines()
                       if ln.strip().startswith("slack + alloc_margin")).split()[-2])
    assert all(abs(float(ln.split()[-2]) + slack) < 1e-3 for ln in credits)


def test_the_host_peak_is_printed_once_for_the_preflight_regex(capsys):
    main(["--preset", "c-hero", "--backend", "device"])
    out = capsys.readouterr().out
    assert out.count("host, a lower bound on the run's peak:") == 1


def test_the_host_kernel_arrays_are_charged_to_the_host_not_the_cards(capsys):
    main(["--preset", "c-hero", "--backend", "device", "--coarse-kernel", "host"])
    out = capsys.readouterr().out
    res_block = out.split("HOST, resident for the whole run")[1].split("\n\n")[0]
    card_block = out.split("PER GPU (of 4), resident through the tile loop")[1].split("\n\n")[0]
    for k in ("coarse_kernel_pref", "coarse_match_factor"):
        assert k in res_block and k not in card_block, k


def test_card_kernel_arrays_move_a_quarter_each_onto_the_cards(capsys):
    """c-hero on 4 cards: the two f32 half-grids (2048 x 2048 x 1025 x 4 B = 17.2 GB each)
    leave the host resident and land 1/4 on each card; the f64 build is one card's block."""
    ec = _ec("c-hero")
    half = ec.mesh_bytes()["coarse_kernel_pref"]
    assert half == 2048 * 2048 * 1025 * 4
    host_res, *_ = _device_host(capsys, "c-hero", 4, kernel="host")
    card_res, *_ = _device_host(capsys, "c-hero", 4, kernel="cards")
    assert abs((host_res - card_res) - 2 * half / 1e9) < 0.01, (host_res, card_res)
    main(["--preset", "c-hero", "--backend", "device"])
    out = capsys.readouterr().out
    card_block = out.split("PER GPU (of 4), resident through the tile loop")[1].split("\n\n")[0]
    for k in ("coarse_kernel_pref", "coarse_match_factor"):
        ln = next(x for x in card_block.splitlines() if x.strip().startswith(k))
        assert abs(float(ln.split()[-2]) - half / 4 / 1e9) < 1e-3, ln
    build = next(x for x in out.splitlines() if "coarse_kernel_build_f64" in x)
    assert abs(float(build.split()[-2]) - ec.mesh_bytes()["coarse_kernel_build_f64"] / 4 / 1e9
               ) < 1e-3, build


# ------------------------------------------------------------- the node axis (--n-nodes)

_G8192 = ["--n-part", "8192", "--box", "4096", "--n-fine", "16384", "--n-coarse", "4096",
          "--tile", "512", "--buf", "32", "--backend", "device", "--n-gpus", "4"]


def test_one_node_is_the_output_without_the_flag(capsys):
    for extra in ([], ["--n-nodes", "1"]):
        main(["--preset", "c-hero", "--backend", "device", "--host-gb", "1026"] + extra)
        if not extra:
            without = capsys.readouterr().out
    assert capsys.readouterr().out == without


def test_the_busiest_node_holds_its_planes_share_and_the_brick_arrays_whole(capsys):
    """8192^3 on 8 nodes: 32 tile planes, 4 per node, so row and bucket terms are 1/8 of the
    global ones; brick_start stays global length."""
    main(_G8192)
    one = capsys.readouterr().out
    main(_G8192 + ["--n-nodes", "8"])
    eight = capsys.readouterr().out

    def term(out, name):
        block = out.split("STATE (resident for the whole run)")[1].split("\n\n")[0]
        return float(next(ln for ln in block.splitlines() if name in ln).split()[-2])

    for k in ("t9_payload", "bucket_index", "arena_bucket"):
        # printed to 0.001 GB, on both sides
        assert term(eight, k) == pytest.approx(term(one, k) / 8, abs=1.5e-3), k
    assert term(eight, "brick_start") == term(one, "brick_start")
    assert "[multi-node]" in eight and "[multi-node]" not in one
    assert "EXCHANGE, bytes each node SENDS per step" in eight


def test_the_spectrum_transposes_move_all_but_the_nodes_own_share(capsys):
    from inexor.plan import engine_config, multinode_terms
    from inexor.codec import T9Layout

    g = PRESETS["c-hero"]
    ec = engine_config("c-hero")
    t9 = T9Layout(g["box"], g["n_part"], 2)
    half = ec.n_coarse ** 2 * (ec.n_coarse // 2 + 1) * 2 * 4
    for N in (2, 4, 8):
        _, sent = multinode_terms(ec, n=ec.n_total, n_nodes=N, t9=t9)
        want = 4 * half / N * (N - 1) / N
        assert sent["spectrum transposes (1 forward + 3 inverse)"] == pytest.approx(want, rel=1e-9)


def test_decompositions_the_design_cannot_run_are_refused():
    with pytest.raises(SystemExit):
        main(["--preset", "c-hero", "--n-nodes", "2"])           # the CPU backend
    with pytest.raises(SystemExit):
        main(["--preset", "c-gh", "--backend", "device", "--n-gpus", "4", "--n-nodes", "8"])
    with pytest.raises(SystemExit):
        main(["--preset", "cgh64", "--backend", "device", "--n-gpus", "1", "--n-nodes", "4",
              "--reach", "4"])


def test_the_exchange_is_priced_in_seconds_only_given_a_rate(capsys):
    main(_G8192 + ["--n-nodes", "8"])
    assert "s per step" not in capsys.readouterr().out
    main(_G8192 + ["--n-nodes", "8", "--net-gbs", "10"])
    assert "at --net-gbs 10.0:" in capsys.readouterr().out


# ------------------------------------------------------------------ y-blocks


def test_y_blocks_cut_the_slab_sized_card_terms_by_the_unit_and_its_window():
    from inexor.device.migrate import EJECT_B_PER_PADDED_ROW
    from inexor.eject_jax import _padded
    from inexor.layout import brick_span
    from inexor.plan import (
        FUSED_DEVICE_B_PER_SLAB_ROW,
        FUSED_HELD_B_PER_ROW,
        device_budget,
        engine_config,
    )

    one = _ec("c-hero")
    nb = one.n_fine // one.n_brick
    slab_rows = one.n_total / nb
    pad = brick_span(one.n_tile, one._b_realized, one.n_brick, nb)[0]
    r1, t1, _p, _w, span, _h, a1 = device_budget(one, n=one.n_total, n_gpus=4)
    for n_y in (2, 4, 8):
        ec = engine_config("c-hero", migrate_backend="device", device_y_blocks=n_y)
        block = max(b - a for a, b in ec.y_block_ranges)
        r, t, _p, _w, _s, _h, a = device_budget(ec, n=ec.n_total, n_gpus=4)
        (win,) = [v for k, v in r.items() if k.startswith("slab_window")]
        assert win == int(span * slab_rows * (block + 2 * pad) / nb * 9)
        assert t["census_eject (fused pass, one padded unit)"] == int(
            EJECT_B_PER_PADDED_ROW * _padded(int(slab_rows * block / nb)))
        held = 2 * FUSED_HELD_B_PER_ROW  # reach 1, several cards
        assert list(a.values()) == [int((FUSED_DEVICE_B_PER_SLAB_ROW - held) * slab_rows
                                        * block / nb + held * slab_rows)]
        assert max(a.values()) < max(a1.values()) and win < max(r1.values())
    assert list(a1.values()) == [int(FUSED_DEVICE_B_PER_SLAB_ROW * slab_rows)]


def test_the_planner_names_the_smallest_fitting_y_block_count_at_8192(capsys):
    args = ["--n-part", "8192", "--box", "4096", "--n-fine", "16384", "--n-coarse", "4096",
            "--tile", "512", "--buf", "32", "--backend", "device", "--n-gpus", "4",
            "--n-nodes", "8", "--host-gb", "1026", "--device-gb", "199", "--arena-frac", "0.01"]

    def verdict(n_y):
        main(args + ["--y-blocks", str(n_y)])
        out = capsys.readouterr().out
        fit = int(next(ln for ln in out.splitlines() if "smallest --y-blocks" in ln).split()[-5])
        ok = [ln for ln in out.splitlines() if "per card:" in ln or "after the tile loop:" in ln]
        return fit, all("FITS (" in ln and "DOES NOT" not in ln for ln in ok)

    fit, fits_at_1 = verdict(1)
    assert not fits_at_1, "VACUOUS: whole slabs already fit"
    assert verdict(fit) == (fit, True)
    assert not verdict(fit - 1)[1]


def test_the_host_window_write_back_is_one_y_block_run(capsys):
    from inexor.forces import capacity_shape
    from inexor.plan import PRESETS, engine_config

    main(["--preset", "c-hero", "--backend", "device", "--host-gb", "1026",
          "--device-gb", "199", "--arena-frac", "0.01", "--n-gpus", "4", "--y-blocks", "4"])
    out = capsys.readouterr().out
    ec = engine_config("c-hero", migrate_backend="device", device_y_blocks=4)
    want = 4 * int(capacity_shape(int(PRESETS["c-hero"]["n_part"] ** 3 // 256
                                      * ec.y_window_fraction))) * 6 / 1e9
    line = next(ln for ln in out.splitlines() if "tile_window write-back" in ln)
    assert "y-block run" in line and abs(float(line.split()[-2]) - want) < 1e-3
