"""M-v2-6 Stage 0: the engine's END-TO-END peak host memory, which nothing measures.

Every residency figure on record prices a component in ISOLATION. M-v2-4's ladder
measured the coarse working set with the paint accumulator held live by the harness
(`runs/v2/m4_f32_mesh_record.md`); D-v2-16 clause 3's figures were derived; the IC
term was priced on its own (M-v2-5 leg V). `EngineConfig.mesh_bytes()` exists and
is called by nothing outside its own test, so no instrument has ever compared a
model of the engine's memory against the engine running.

THE ESTIMAND: net-of-baseline peak host RSS (`ru_maxrss`) around
`load_slot_state` -> `engine.run(K)`, one subprocess per (config, leg) because a
high-water mark never resets -- the v4d/M-v2-5 method verbatim. CPU BACKEND ONLY:
on CUDA the meshes live in VRAM and host RSS sees only the JAX baseline, which is
deneb 409's defect (three arms read 1.21 GiB identically and every ratio came out
exactly 1.0). This probe REFUSES a non-CPU backend rather than measuring the wrong
pool.

**ICs are generated in their OWN process and reach the engine through disk.** The
first version of this probe built the state from `v2_m3_engine_gate.make_ics`, in
process, and that is an instrument defect of the exact class this milestone is
about: `make_ics` runs the monolithic `lpt_ics`, whose peak is 70-90 B/p
(M-v2-5 leg V), so a high-water mark would have carried the GENERATOR's peak into
every engine reading and swamped the term being looked for. Using
`icgen.generate_t9_slabs` + `icgen.load_slot_state` is both the production path and
the only way this measurement is honest.

## What it is looking for, and why the arms are shaped this way

Reading this session, `engine.step` holds an O(N) FLOAT array that three docstrings
say does not exist. `pending` (`engine.py:405`) accumulates `(slots[owned], v_new)`
per tile and is not consumed until the scale reconciliation at `engine.py:467-476`;
ownership is ASSERTED to be a partition (`:455-459`), so at the end of the tile
loop it holds exactly `n_particles` rows of one int64 plus three float64 =
**32 B/p**, which is 275 GB at C-gh. `engine.py:39-45` says "Nothing O(N) in
floats, anywhere" and names deleting exactly this array as "what makes that
configuration runnable".

An instrument that cannot see that term is not an instrument, so the legs ISOLATE
it rather than reporting one blob:

  gen            `generate_t9_slabs`. Not an engine number: it is here as a
                 cross-check against M-v2-5's fitted 8.79 B/p, and because the
                 slabs have to exist before anything else runs.
  load           `load_slot_state` only. Stage 2d's baseline (it holds an int64
                 `occupancy` and its narrowed copy simultaneously, `icgen.py:342-364`).
  repack_only    load, then ONE `SlotState.repack`. It allocates `zeros_like` of
                 `off` and `w` (`state.py:936-938`) while the originals stay live,
                 so it carries its own ~10 B/p transient and Stage 2b needs a
                 baseline for it. MEASURED DIRECTLY rather than by differencing
                 against `repack_every=0`: the first version of this probe did the
                 latter and the layout hit the D-007 arena refusal inside 10 steps
                 at cdev8, which is D-v2-19 clause 3 doing exactly what it says.
                 The repack cannot be switched off to difference against, so it is
                 measured on its own and reported with whether its peak could set
                 the step's at all (a peak is a max, not a sum).
  step           K steps, production knobs. The number M-v2-6 actually needs.
  step_f32       coarse mesh at f32, production knobs. A CONTROL on the mesh model
                 rather than a measurement of the engine: the f64-minus-f32 delta
                 is predicted exactly by `mesh_bytes()`, so if the measured delta
                 does not match it, the model does not describe this machine and
                 the residual below is not readable. M-v2-4 measured the same
                 decomposition reproducing two rungs to four digits.
  step_coarse_lo `n_coarse // 4` at fixed `n_part`. THE ISOLATING ARM.
                 The state and `pending` are untouched while every mesh term falls
                 64x, which is the only way to separate an O(N) term from an
                 O(n_coarse^3) one on this config table -- `n_coarse` is
                 `n_part / 2` at every ratified config, so N and the mesh are
                 EXACTLY confounded across it and no laddering over configs
                 separates them. `coarse_subblock_origin_extent` derives its ratio
                 (`forces.py:870`) rather than assuming 4, so this is a supported
                 configuration; it changes the split scale `r_s = alpha *
                 coarse_cell`, so it is an INSTRUMENT ARM and never an operating
                 point (the `cap_mult` precedent).

## PRE-REGISTERED (before any leg ran)

The 32 B/p is DERIVED from the code, not fitted: `slots` is int64 (8 B) and `v_new`
is (m, 3) float64 (24 B), over a partition of the particles. Likewise the repack
transient is `9 B * n_rows / N` with `n_rows ~ 1.1-1.21 N`, i.e. ~10-11 B/p.
Neither number is a threshold anyone picked.

  GATE 1 (the mesh model describes the machine): measured
  `net(step) - net(step_f32)` within 25% of `mesh_bytes()`'s predicted f64-f32
  delta. A failure VOIDS gate 2 rather than failing the milestone -- it means the
  model is wrong, not that the engine is.

  GATE 2 (the instrument sees the O(N) float term): on `step_coarse_lo`, the
  residual after subtracting the MEASURED state arrays, the modelled mesh and the
  `cap`-derived tile buffers is at least 0.8 x 32 B/p. ONE-SIDED deliberately:
  unmodelled transients can only ADD to a peak, so a lower bound is the honest
  shape and a tight bracket would claim precision the mesh model does not have.

  GATE 3 (it is the same term at both configs): the gate-2 residual per particle
  agrees across configs within a factor of 1.5. An O(N) term must; an
  n_coarse^3 term misattributed as one would not.

  REPORTED, NOT GATED: the repack transient (`step` minus `step_norepack`), which
  is Stage 2b's baseline; the load-phase peak, which is Stage 2d's; and
  `bytes_per_particle()`'s claim beside the measured array bytes, because that
  function omits the `alloc_margin` allocation entirely (`n_rows = n_alloc +
  n_arena` at `state.py:135` against the `n_slots` it divides by at `:987`) and the
  gap is ~1 B/p at the default margin -- a term that vanishes from a table is
  indistinguishable from one that was never counted, which is what that function's
  own docstring says.

Usage:
  pixi run python scripts/v2_m6_engine_peak.py --configs smoke cdev8 --k 3
  pixi run python scripts/v2_m6_engine_peak.py --worker cdev8:step:3:/tmp/wd  # internal
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO, "runs", "v2")

sys.path.insert(0, HERE)
import v2_m3_engine_gate as m3  # noqa: E402
import v2_m5_ic_gate as m5  # noqa: E402

CONFIGS = m3.CONFIGS
SEED = m3.SEED
GEN_FDTYPE = np.float32  # the production generator dtype (M-v2-5 leg V)

# steps: 0 = no engine.run, None = --k. `repack` calls SlotState.repack directly.
LEGS = {
    "gen": dict(steps=0, coarse="float64", coarse_div=1),
    "load": dict(steps=0, coarse="float64", coarse_div=1),
    "integrity": dict(steps=0, coarse="float64", coarse_div=1, integrity=True),
    "repack_only": dict(steps=0, coarse="float64", coarse_div=1, repack=True),
    "step": dict(steps=None, coarse="float64", coarse_div=1),
    "step_f32": dict(steps=None, coarse="float32", coarse_div=1),
    "step_coarse_lo": dict(steps=None, coarse="float64", coarse_div=4),
}
STEP_LEGS = tuple(k for k, v in LEGS.items() if v["steps"] is None)

# `pending` at engine.py:405,453: one int64 slot ordinal plus three float64
# velocity components per particle, over a partition. Derived, not fitted.
PENDING_BPP = 8 + 3 * 8
GATE2_FLOOR = 0.8 * PENDING_BPP
GATE1_TOL = 0.25
GATE3_FACTOR = 1.5


def icgen_manifest():
    """The manifest filename, read from the package rather than duplicated here."""
    from inexor import icgen

    return icgen.MANIFEST


def _maxrss_bytes():
    """Peak RSS of this process. KB on Linux, BYTES on macOS (the v4d note)."""
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


def _state_array_bytes(st):
    """Measured, not modelled: what the container actually allocated."""
    terms = dict(
        off=int(st.off.nbytes),
        w=int(st.w.nbytes),
        occupancy=int(st.occupancy.nbytes),
        brick_start=int(st.brick_start.nbytes),
        arena_bucket=int(st.arena_bucket.nbytes),
        ids=0 if st.ids is None else int(st.ids.nbytes),
    )
    terms["total"] = int(sum(terms.values()))
    return terms


def _tile_buffer_bytes(cap):
    """Host-side per-tile buffers, from the cap the engine reports.

    Per tile in `engine.step`: `idx` cap*8 (:413), `live` cap*1 (:414), the `u`
    mirror cap*3*8 (:417), `xo` cap*3*8 (:436), `lv` cap*1 (:438), and
    `decode_bricks` returning slots+x+v at cap*(8+24+24) (:407,
    state.py:688-708). APPROXIMATE by construction -- it is a model of a
    transient, reported so the residual is read against something rather than
    against zero.
    """
    if cap is None:
        return None
    return int(cap) * (8 + 1 + 24 + 24 + 1 + 8 + 24 + 24)


def _require_cpu():
    import jax

    jax.config.update("jax_enable_x64", True)
    if jax.devices()[0].platform != "cpu":
        raise SystemExit(
            "FATAL: non-CPU backend. ru_maxrss is HOST memory; on CUDA the meshes "
            "live in VRAM and every ratio reads 1.0 (deneb 409). Run the CPU env."
        )
    return jax


def _engine_config(g, coarse_dtype, repack_every, coarse_div, slack):
    from inexor import engine

    ec = engine.EngineConfig(
        box_size=g["L"], n_part=g["n_part"], n_fine=g["n_fine"],
        n_coarse=g["n_coarse"] // coarse_div,
        n_tile=g["tile"], b_fine=g["buf"], alpha=m3.ALPHA,
        brick_slack=slack, repack_every=repack_every, coarse_dtype=coarse_dtype,
    )
    # a peak card from a pooled run would not be comparable to any card on
    # record (worker RSS lives in other processes; overlap voids per-phase
    # attribution), so this instrument is serial-only by construction
    assert ec.tile_workers == 1, "peak cards are only readable on the serial executor"
    return ec


def _worker(cfg, leg, k_steps, workdir, slack, arena_frac, alloc_margin):
    jax = _require_cpu()
    g = m3._geom(cfg)

    from inexor.config import Cosmology

    # backend init in EVERY leg, so `baseline` nets it out of the others
    import inexor.ic as ic

    key = jax.random.PRNGKey(SEED)
    ic.white_plane(key, 0, 8, GEN_FDTYPE)

    if leg == "baseline":
        print(json.dumps(dict(maxrss=_maxrss_bytes())), flush=True)
        return

    from inexor import icgen

    cosmo = Cosmology()
    spec = LEGS[leg]
    ec = _engine_config(g, spec["coarse"], 1, spec["coarse_div"], slack)
    ec.validate()
    nb = g["n_fine"] // ec.n_brick
    out = dict(n_coarse=int(ec.n_coarse), bricks_per_side=int(nb),
               mesh_bytes=ec.mesh_bytes(), n_tiles=len(ec.tiles))

    if leg == "gen":
        icgen.generate_t9_slabs(
            workdir, key, g["n_part"], g["L"], cosmo, m3.A_INIT, nb,
            fdtype=GEN_FDTYPE, slab=32,
        )
        out["maxrss"] = _maxrss_bytes()
        out["bpp"] = out["maxrss"] / g["n_part"] ** 3
        print(json.dumps(out), flush=True)
        return

    st = icgen.load_slot_state(
        workdir, brick_slack=slack, alloc_margin=alloc_margin, arena_frac=arena_frac
    )
    out.update(
        state_bytes=_state_array_bytes(st),
        bpp_claimed=st.bytes_per_particle(),
        n_particles=int(st.n_particles),
        n_rows=int(st.off.shape[0]),
        n_slots=int(st.n_slots),
        n_arena=int(st.n_arena),
        n_bricks=int(st.n_bricks),
        peak_after_load=_maxrss_bytes(),
    )
    if spec.get("integrity"):
        # WHY THIS LEG EXISTS. cdev fails the engine's partition assertion at
        # `16777215 rows against 16777216 particles` on x86 and not on Apple
        # arm64, and I have now proposed three explanations from reading the code
        # and been wrong twice (a tile-local float gap, then a position-vs-storage
        # mismatch). This finds the missing row instead of arguing about it: every
        # census the assertion could be comparing, plus the identity of whatever
        # slot no tile claims.
        import collections

        from inexor import forces as F

        b_real = ec._b_realized
        nb = g["n_fine"] // ec.n_brick
        per_brick_count = np.array(
            [st.brick_member_count(b) for b in range(st.n_bricks)], dtype=np.int64
        )
        decoded_len = np.array(
            [len(st.decode_brick(b)[0]) for b in range(st.n_bricks)], dtype=np.int64
        )
        claims = collections.Counter()
        rows_decoded = 0
        for t in ec.tiles:
            members = st.tile_bricks(t, ec.n_tile, b_real, ec.n_brick, g["n_fine"])
            slots, _, _ = st.decode_bricks(members)
            rows_decoded += len(slots)
            brick_of_row = np.repeat(
                np.asarray(members, dtype=np.int64),
                [st.brick_member_count(b) for b in members],
            )
            if len(brick_of_row) != len(slots):
                out["MISALIGNED"] = dict(tile=list(map(int, t)),
                                         counts=int(len(brick_of_row)),
                                         decoded=int(len(slots)))
                break
            own = F.owned_mask_from_bricks(brick_of_row, t, ec.n_tile, ec.n_brick, nb)
            for s in np.asarray(slots)[own]:
                claims[int(s)] += 1
        owned_total = sum(claims.values())
        out["integrity"] = dict(
            n_particles=int(st.n_particles),
            n_live=int(st.n_live),
            arena_used=int(st.arena_used),
            sum_brick_member_count=int(per_brick_count.sum()),
            sum_decoded_per_brick=int(decoded_len.sum()),
            count_vs_decode_mismatched_bricks=int((per_brick_count != decoded_len).sum()),
            rows_decoded_over_tiles=int(rows_decoded),
            slots_claimed_once=int(sum(1 for v in claims.values() if v == 1)),
            slots_claimed_twice_or_more=int(sum(1 for v in claims.values() if v > 1)),
            owned_total=owned_total,
            deficit=int(st.n_particles) - owned_total,
            n_bricks=int(st.n_bricks),
            bricks_per_side=int(nb),
            bricks_per_core=int(ec.n_tile // ec.n_brick),
            check_ok=bool(st.check()),
        )
        # name the missing slots, with everything needed to attribute them
        live_slots = set()
        for b in range(st.n_bricks):
            live_slots.update(int(s) for s in st.decode_brick(b)[0])
        missing = sorted(live_slots - set(claims))
        out["integrity"]["n_missing_slots"] = len(missing)
        det = []
        for s in missing[:10]:
            b = int(np.searchsorted(st.brick_start, s, side="right") - 1)
            det.append(dict(
                slot=int(s), brick=b,
                in_arena=bool(s >= st.arena_base),
                arena_bucket=(int(st.arena_bucket[s - st.arena_base])
                              if s >= st.arena_base else None),
                off=[int(q) for q in st.off[s]],
            ))
        out["integrity"]["missing_detail"] = det
        print(json.dumps(dict(maxrss=_maxrss_bytes(), **out)), flush=True)
        return

    if spec.get("repack"):
        # Isolate the repack transient DIRECTLY rather than by disabling it. The
        # first version of this leg ran the engine with repack_every=0 and the
        # layout hit the D-007 arena refusal inside 10 steps at cdev8 -- which is
        # D-v2-19 clause 3 doing exactly what it says ("frozen capacity fails at
        # EVERY granularity"), so the repack cannot be switched off to difference
        # against. Calling it once on a freshly loaded state measures the same
        # transient without needing the engine to survive without it.
        out["repack_stats"] = st.repack(brick_slack=slack)
        st.check()
    if spec["steps"] is None:
        import time

        from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

        n_steps = int(k_steps)
        a_steps = a_grid(m3.A_INIT, m3.A_FINAL, n_steps, m3.SPACING)
        co = bullfrog_float_coeffs(bullfrog_table(a_steps, cosmo))
        from inexor import engine

        seen = []
        t0 = time.perf_counter()
        engine.run(st, ec, co, collect=seen.append)
        out["wall_s"] = time.perf_counter() - t0
        out["s_per_step"] = out["wall_s"] / n_steps
        out["k_steps"] = n_steps
        out["cap"] = int(seen[-1]["cap"]) if seen else None
        # cap PER STEP, not just the last. If it moves, every eager jnp op keyed
        # on it takes a new XLA shape each step and the executable cache grows
        # without bound -- which is the leading explanation for a peak that
        # scales with K. M-v2-3 padded to one shape WITHIN a step; nothing pins
        # it ACROSS steps.
        out["cap_per_step"] = [int(s["cap"]) for s in seen]
        out["cap_distinct"] = len({int(s["cap"]) for s in seen})
        # `cap_true` is the unquantized max over tiles: what the shape WOULD have
        # been, so the two distinct-counts are the before/after of the leak in one
        # run rather than across two
        if seen and "cap_true" in seen[0]:
            out["cap_true_per_step"] = [int(s["cap_true"]) for s in seen]
            out["cap_true_distinct"] = len({int(s["cap_true"]) for s in seen})
            out["pad_frac_max"] = max(
                s["cap"] / s["cap_true"] - 1.0 for s in seen if s["cap_true"]
            )
        out["arena_used_per_step"] = [int(s.get("arena_used", -1)) for s in seen]
        # the migration's reach, bound and REALIZED, per step. The realized
        # value is the field evidence for the missing-particle mechanism (a
        # 2-brick x-mover at cdev, antares 442/445): job 445 confirmed the fix
        # by completing, but its card could not say WHY, because these fields
        # were not persisted. A claim like "at most one brick per axis" must be
        # readable off the card, not reconstructed from a rerun.
        out["brick_reach_per_step"] = [int(s.get("brick_reach", -1)) for s in seen]
        out["brick_reach_realized_per_step"] = [
            int(s.get("brick_reach_realized", -1)) for s in seen
        ]
        st.check()
    out["maxrss"] = _maxrss_bytes()
    print(json.dumps(out), flush=True)


def _run(cfg, leg, k_steps, workdir, knobs):
    spec = f"{cfg}:{leg}:{k_steps}:{workdir}"
    cmd = [
        sys.executable, os.path.abspath(__file__), "--worker", spec,
        "--slack", str(knobs["slack"]), "--arena-frac", str(knobs["arena_frac"]),
        "--alloc-margin", str(knobs["alloc_margin"]),
    ]
    out = subprocess.check_output(cmd, text=True, cwd=REPO)
    return json.loads(out.strip().splitlines()[-1])


def _analyse(legs, n):
    """Attribute the peak. Every subtrahend is measured or modelled, never fitted."""
    base = legs["baseline"]["maxrss"]
    res = dict(baseline_bytes=base, n_particles=n)
    for name, d in legs.items():
        if name == "baseline":
            continue
        res[name] = dict(
            peak=d["maxrss"], net=d["maxrss"] - base, net_bpp=(d["maxrss"] - base) / n,
            wall_s=d.get("wall_s"), s_per_step=d.get("s_per_step"), cap=d.get("cap"),
        )

    st_meas = legs["load"]["state_bytes"]["total"]
    claimed = legs["load"]["bpp_claimed"]["total"] * n
    res["state_measured_bytes"] = st_meas
    res["state_bpp_measured"] = st_meas / n
    res["state_bpp_claimed"] = legs["load"]["bpp_claimed"]["total"]
    res["bpp_unreported_gap"] = (st_meas - claimed) / n

    # GATE 1 -- does mesh_bytes() describe this machine?
    pred = sum(legs["step"]["mesh_bytes"].values()) - sum(legs["step_f32"]["mesh_bytes"].values())
    meas = legs["step"]["maxrss"] - legs["step_f32"]["maxrss"]
    res["mesh_model"] = dict(
        predicted_f64_minus_f32=pred, measured_f64_minus_f32=meas,
        ratio=(meas / pred) if pred else None,
    )
    res["gate1_mesh_model_ok"] = bool(pred and abs(meas / pred - 1.0) <= GATE1_TOL)

    # GATE 2 -- the isolating arm
    lo = legs["step_coarse_lo"]
    mesh_lo = sum(lo["mesh_bytes"].values())
    tiles = _tile_buffer_bytes(lo.get("cap"))
    resid = lo["maxrss"] - base - st_meas - mesh_lo - (tiles or 0)
    res["isolating_arm"] = dict(
        n_coarse=lo["n_coarse"], net=lo["maxrss"] - base, state=st_meas,
        mesh_modelled=mesh_lo, tile_buffers_modelled=tiles,
        residual=resid, residual_bpp=resid / n,
        pending_predicted_bpp=PENDING_BPP,
        residual_over_predicted=(resid / n) / PENDING_BPP,
    )
    res["gate2_sees_on_bpp_term"] = bool(resid / n >= GATE2_FLOOR)

    # REPORTED -- Stage 2b's and Stage 2d's baselines
    rp = legs["repack_only"]["maxrss"] - legs["load"]["maxrss"]
    res["repack_transient"] = dict(
        bytes=rp, bpp=rp / n, predicted_bpp=9.0 * legs["load"]["n_rows"] / n,
        # a peak is a max, not a sum: the repack only contaminates the step
        # residual if its own peak EXCEEDS the step's, which this reports rather
        # than assumes
        repack_only_net=legs["repack_only"]["maxrss"] - base,
        step_net=legs["step"]["maxrss"] - base,
        could_set_the_step_peak=bool(
            legs["repack_only"]["maxrss"] >= legs["step"]["maxrss"]
        ),
    )
    res["load_phase"] = dict(
        net=legs["load"]["maxrss"] - base, net_bpp=(legs["load"]["maxrss"] - base) / n,
        state_bpp_measured=st_meas / n,
    )
    return res


def _write(res, suffix, knobs):
    """m5's card shape with an m6 name, reusing its provenance (one source)."""

    class _A:
        leg = "peak"
        out_suffix = suffix
        n = knobs.get("n_part")
        bricks = knobs.get("bricks_per_side")
        slab = 32
        seed = SEED
        f_nl = 0.0

    try:
        res["commit"] = subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        res["commit"] = None
    res["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    prov = m5._provenance(_A())
    prov["knobs"] = knobs
    # ARCHITECTURE IS A COMPARABILITY AXIS, not a footnote (JC, 2026-08-11).
    # M-v2-5 measured the IC noise stream differing between Apple arm64 and Linux
    # aarch64, and M-v2-6 found a partition defect that fires on an x86-64
    # realization of cdev and not on this laptop's -- so a seed does not name a
    # realization across architectures, and neither peaks nor pass/fail transfer.
    # NB Apple arm64 and Grace aarch64 are BOTH arm64 by `machine` and are still
    # known to differ, so equal `arch` is necessary and not sufficient; `platform`
    # separates them and is recorded beside it.
    prov["arch"] = platform.machine()
    prov["platform"] = sys.platform
    prov["libc"] = "-".join(platform.libc_ver()) or None
    res["provenance"] = prov
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"m6_peak{suffix}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"  card -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="+", default=["cdev8"], choices=list(CONFIGS))
    ap.add_argument("--k", type=int, default=3, help="engine steps per stepping leg")
    ap.add_argument("--legs", nargs="+", default=None, choices=list(LEGS))
    ap.add_argument("--workdir-root", default=None,
                    help="where T9 slabs are staged; defaults under TMPDIR. Kept, "
                         "not deleted, so reruns skip generation.")
    ap.add_argument("--out-suffix", default="")
    # Capacity knobs are first-class: they set n_rows, hence the state bytes AND
    # the repack transient, so they are part of the measurement rather than
    # defaults. They are also `check_comparability` refusal axes. The values are
    # M-v2-4's cdev anchor (deneb 408/411); the package defaults 0.10/0.01
    # exhaust the arena at small K, which is the recorded trap -- a SMALLER K
    # drifts further per step and migrates more.
    ap.add_argument("--k-ladder", type=int, nargs="+", default=None,
                    help="run the `step` leg at each K and fit net peak vs K. A "
                         "peak that GROWS with step count is an accumulation, not "
                         "a working set -- the decisive test for a per-step leak.")
    ap.add_argument("--cap-rungs", type=int, default=None,
                    help="rungs per octave for the buffer-shape ladder. Exposed so "
                         "the padding regime can be MOVED: a defect that survives "
                         "rungs=1 and rungs=64 unchanged is not caused by padding.")
    ap.add_argument("--slack", type=float, default=0.20)
    ap.add_argument("--arena-frac", type=float, default=0.08)
    ap.add_argument("--alloc-margin", type=float, default=0.10)
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        cfg, leg, k, wd = args.worker.split(":", 3)
        _worker(cfg, leg, int(k), wd, args.slack, args.arena_frac, args.alloc_margin)
        return 0

    want = args.legs or list(LEGS)
    root = args.workdir_root or os.path.join(
        os.environ.get("TMPDIR", tempfile.gettempdir()), "m6_peak"
    )
    res = dict(
        k_steps=args.k, legs=want, workdir_root=root,
        method="one subprocess per (config, leg); ICs reach the engine through disk",
        gen_fdtype=np.dtype(GEN_FDTYPE).name,
        pending_predicted_bpp=PENDING_BPP,
    )
    knobs = dict(
        k=args.k, legs=want, gen_fdtype=np.dtype(GEN_FDTYPE).name,
        slack=args.slack, arena_frac=args.arena_frac, alloc_margin=args.alloc_margin,
    )
    res["knobs"] = knobs

    if args.k_ladder:
        cfg = args.configs[0]
        wd = os.path.join(root, cfg)
        os.makedirs(wd, exist_ok=True)
        base = _run(cfg, "baseline", 0, wd, knobs)["maxrss"]
        if not os.path.exists(os.path.join(wd, icgen_manifest())):
            _run(cfg, "gen", 0, wd, knobs)
        rungs = {}
        for k in sorted(args.k_ladder):
            d = _run(cfg, "step", k, wd, knobs)
            # the shape FAMILY, not just the last shape: the leak is one
            # executable family per distinct shape, so `cap_distinct` is the
            # quantity the ladder is supposed to collapse and `cap_true_distinct`
            # is what it would have been without it
            rungs[str(k)] = dict(
                peak=d["maxrss"], net=d["maxrss"] - base,
                s_per_step=d.get("s_per_step"), cap=d.get("cap"),
                cap_per_step=d.get("cap_per_step"),
                cap_distinct=d.get("cap_distinct"),
                cap_true_per_step=d.get("cap_true_per_step"),
                cap_true_distinct=d.get("cap_true_distinct"),
                pad_frac_max=d.get("pad_frac_max"),
            )
            print(f"[{cfg}] K={k}: peak {d['maxrss'] / 1e9:.3f} GB, "
                  f"net {(d['maxrss'] - base) / 1e9:.3f} GB, "
                  f"{d['s_per_step']:.2f} s/step", flush=True)
        ks = np.array(sorted(args.k_ladder), dtype=np.float64)
        nets = np.array([rungs[str(int(k))]["net"] for k in ks])
        res["k_ladder"] = dict(config=cfg, baseline_bytes=base, rungs=rungs)
        if len(ks) >= 2:
            # a two-parameter fit needs two rungs. One rung would return a
            # least-norm solution rather than an error, i.e. a slope that looks
            # like a measurement and is not one.
            slope, icpt = np.linalg.lstsq(
                np.stack([ks, np.ones_like(ks)], axis=1), nets, rcond=None
            )[0]
            resid = nets - (slope * ks + icpt)
            res["k_ladder"].update(
                per_step_bytes=float(slope), fixed_bytes=float(icpt),
                resid_gb=[float(r / 1e9) for r in resid],
                # a working set is K-INDEPENDENT; a slope means it accumulates
                accumulates_per_step=bool(slope > 0.05 * nets.max() / max(ks)),
            )
            print(f"[{cfg}] fit: {slope / 1e9:.3f} GB PER STEP + "
                  f"{icpt / 1e9:.3f} GB fixed; "
                  f"resid {['%.3f' % (r / 1e9) for r in resid]} GB", flush=True)
        else:
            res["k_ladder"].update(per_step_bytes=None, fixed_bytes=None)
            print(f"[{cfg}] one rung: no fit (a slope needs two)", flush=True)
        _write(res, args.out_suffix or "_kladder", knobs)
        return 0

    for cfg in args.configs:
        wd = os.path.join(root, cfg)
        os.makedirs(wd, exist_ok=True)
        legs = {"baseline": _run(cfg, "baseline", 0, wd, knobs)}
        base = legs["baseline"]["maxrss"]
        print(f"[{cfg}] baseline {base / 1e9:.3f} GB  (workdir {wd})", flush=True)
        # cheapest first, and EVERY requested leg runs: an earlier version built
        # this list from two membership tests that between them did not cover
        # `repack_only`, so the leg was silently dropped and its absence read as
        # a clean run -- the "a gate that cannot fail" class, in the orchestrator
        # derive the order from LEGS rather than from hand-written membership
        # tests. Two versions of this dropped a leg silently by listing the
        # non-stepping legs by name and forgetting one (`repack_only`, then
        # `integrity`) -- the same "gate that cannot fail" class, twice, in the
        # orchestrator. Now the schedule is a permutation of `want` by
        # construction and the check below cannot be reached.
        pre = [ln for ln in LEGS if ln in want and ln not in STEP_LEGS]
        order = pre + [ln for ln in LEGS if ln in want and ln in STEP_LEGS]
        missing = [ln for ln in want if ln not in order]
        if missing:
            raise SystemExit(f"FATAL: legs requested but not scheduled: {missing}")
        if "gen" not in order and not os.path.exists(os.path.join(wd, icgen_manifest())):
            order.insert(0, "gen")
        for leg in order:
            d = _run(cfg, leg, args.k, wd, knobs)
            legs[leg] = d
            sps = d.get("s_per_step")
            tail = f", {sps:.2f} s/step" if sps else ""
            print(f"[{cfg}] {leg}: peak {d['maxrss'] / 1e9:.3f} GB, "
                  f"net {(d['maxrss'] - base) / 1e9:.3f} GB{tail}", flush=True)

        if set(LEGS) <= set(legs):
            n = legs["load"]["n_particles"]
            a = _analyse(legs, n)
            res[cfg] = dict(legs=legs, analysis=a)
            iso, mm = a["isolating_arm"], a["mesh_model"]
            print(f"[{cfg}] state {a['state_bpp_measured']:.2f} B/p measured vs "
                  f"{a['state_bpp_claimed']:.2f} claimed "
                  f"(unreported {a['bpp_unreported_gap']:.2f})", flush=True)
            print(f"[{cfg}] mesh model f64-f32 measured/predicted = "
                  f"{mm['ratio']:.3f}" if mm["ratio"] else
                  f"[{cfg}] mesh model delta not evaluable", flush=True)
            print(f"[{cfg}] isolating residual {iso['residual_bpp']:.1f} B/p = "
                  f"{iso['residual_over_predicted']:.2f}x the predicted "
                  f"{PENDING_BPP} B/p", flush=True)
            print(f"[{cfg}] repack transient {a['repack_transient']['bpp']:.1f} B/p "
                  f"(predicted {a['repack_transient']['predicted_bpp']:.1f}); "
                  f"load {a['load_phase']['net_bpp']:.1f} B/p", flush=True)
            knobs.setdefault("n_part", CONFIGS[cfg]["n_part"])
            knobs.setdefault("bricks_per_side", legs["load"]["bricks_per_side"])
        else:
            res[cfg] = dict(legs=legs, analysis=None)

    done = [c for c in args.configs if res[c]["analysis"]]
    g1 = all(res[c]["analysis"]["gate1_mesh_model_ok"] for c in done)
    g2 = all(res[c]["analysis"]["gate2_sees_on_bpp_term"] for c in done)
    if len(done) >= 2:
        bpps = [res[c]["analysis"]["isolating_arm"]["residual_bpp"] for c in done]
        spread = (max(bpps) / min(bpps)) if min(bpps) > 0 else float("inf")
        g3 = bool(spread <= GATE3_FACTOR)
    else:
        spread, g3 = None, None
        print("NOTE: gate 3 needs two configs; not evaluated", flush=True)
    res["verdict"] = dict(
        gate1_mesh_model_ok=g1, gate2_sees_on_bpp_term=g2,
        gate3_consistent_across_configs=g3, gate3_spread=spread,
        configs_analysed=done,
    )
    if set(LEGS) <= set(want):
        ok = bool(done) and g1 and g2 and (g3 is not False)
    else:
        # a subset of legs was requested, so the gates were never evaluable. Say
        # that rather than reporting FAIL, which would read as a finding.
        ok = True
        res["verdict"]["note"] = (
            f"legs {sorted(set(LEGS) - set(want))} not requested, so gates 1-3 were "
            "not evaluated; this run reports measurements only"
        )
    res["ok"] = ok
    _write(res, args.out_suffix, knobs)
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
