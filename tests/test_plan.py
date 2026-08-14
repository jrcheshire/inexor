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

from inexor.engine import EngineConfig
from inexor.forces import padded_size
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


def test_the_coarse_force_copy_is_counted_because_both_copies_are_live():
    """`g_coarse = [np.asarray(g) for g in g_coarse]` rebinds after the
    comprehension, so the jax originals survive their numpy copies' creation."""
    m = _ec("cdev").mesh_bytes()
    assert m["coarse_force_copy_transient"] == m["coarse_force_resident"]


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


def test_the_binding_term_at_c_gh_is_now_the_repack_scratch(capsys):
    """With `kick_pending` gone the largest single term is `repack_scratch`, and
    C-gh STILL does not fit a `gh` host -- 2.04x rather than 4.41x. Worth pinning
    both halves: removing the largest term was necessary and is not sufficient,
    and a reading that stopped at "the binding term was removed" would have
    concluded the opposite."""
    main(["--preset", "c-gh", "--host-gb", "116", "--cap", "5284492"])
    out = capsys.readouterr().out
    assert "largest single term: repack_scratch" in out
    assert "DOES NOT FIT" in out
    ratio = float(out.split("DOES NOT FIT (")[1].split("x")[0])
    assert 1.9 < ratio < 2.2, f"expected ~2.04x a gh host after the removal, got {ratio}"


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
    """Still a floor, and still known-soft: 3.982 GB against job 446's measured
    7.461 at cdev, so 1.87x low where it has been checked. It was 3.783 (1.97x)
    before the M-v2-6 terms went in -- the corrections close some of the gap and
    are not claimed to close all of it, which is why the wording stays."""
    main(["--preset", "cdev", "--host-gb", "124", "--cap", "5284492"])
    out = capsys.readouterr().out
    assert "LOWER BOUND" in out and "not a measurement" in out
    est = float(out.split("a lower bound on the run's peak:")[1].split("GB")[0])
    assert est == pytest.approx(3.445, abs=0.01)
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
