"""M-v2-6's deliverable: a COMPLETE mock, generated, stepped, exported and scored.

Every other script in this milestone measures a phase. This one produces the
thing the milestone is for, and it exists because the pieces Stage 4 built have
never been run together: `export.write_particles` and `summary.pk_summary_card`
have no caller outside their own tests, and a full 2048^3 IC generation end to
end is listed under "what this does NOT establish" in the M-v2-5 record.

**Subcommands are separate PROCESSES on purpose**, not stages of one run:

    ics     generate the T9 slabs (the streamed M-v2-5 path) and clean up staging
    run     step the schedule, or a SEGMENT of it, checkpointing as it goes
    export  write plain .npy (x, v) anything can read
    card    the run's own P(k) against theory, as a z profile

A high-water mark never resets, so one process per phase is the only way each
phase's peak host RSS is readable -- the same reason `v2_m6_engine_peak.py`
generates its ICs in a subprocess, and it matters more here: the monolithic
generator's peak is 70-90 B/p and would swamp every engine reading it preceded.
It also means a queue limit costs one phase, not the run.

**Segments, and why they are the unit.** The projected wall at 2048^3 is order a
day and the gg QOS caps a job at two. `run --stop-at` advances the schedule to an
absolute step and stops on a checkpoint; the next job resumes from disk. Segments
compose back into the uninterrupted run BITWISE
(`test_a_run_split_into_segments_is_bitwise_the_uninterrupted_one`), because each
one is handed the full coefficient list and told where to stop rather than a
truncated schedule -- `fused_drifts` makes `coeffs[:n]` a different trajectory.

**What is gated, and what is merely reported.**

- GATE: CPU backend. The pool refuses a non-CPU parent and every phase card on
  record is CPU; a run that silently used a device would be comparable to
  nothing.
- GATE: the geometry this drives must equal `v2_m3_engine_gate`'s for any config
  both define. Two config tables that drift are how a "2048^3 result" stops
  describing the ladder it is supposed to cap.
- GATE: `run` refuses to start from ICs when a checkpoint exists, and says which
  it used. Silently re-running from step 0 in a resume job would burn the wall
  and look like success.
- GATE (in the library, named here so it is not re-derived): `write_particles`
  refuses a short file, `pk_summary_card` refuses an empty card, and `stop_at`
  refuses to stop off a checkpoint boundary.
- REPORTED: every wall, every peak, the phase card, the repack fast/merge split,
  the checkpoint receipts, and the z profile. No verdict is emitted anywhere --
  `band_verdict` takes a band the caller names, and naming it is not this
  script's business.

Usage (the 2048^3 realization, one phase per job step):

    python scripts/v2_m6_realization.py ics    --config c-gh --workdir $W
    python scripts/v2_m6_realization.py run    --config c-gh --workdir $W --stop-at 5
    python scripts/v2_m6_realization.py run    --config c-gh --workdir $W
    python scripts/v2_m6_realization.py export --config c-gh --workdir $W
    python scripts/v2_m6_realization.py card   --config c-gh --workdir $W
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import v2_m3_engine_gate as m3  # noqa: E402
from inexor.plan import PRESETS  # noqa: E402
from v2_m6_engine_peak import _maxrss_bytes, _require_cpu  # noqa: E402
from v2_m6_phase_time import PhaseTimer  # noqa: E402

# The generator dtype the M-v2-5 record measured the production path at.
GEN_FDTYPE = np.float32
K_STEPS = 40


def _geom(cfg_name):
    """Geometry from the ratified preset table, CHECKED against the engine gate's.

    `plan.PRESETS` is the only table carrying 2048^3; `m3.CONFIGS` is the one
    every engine card was measured through. They agree today, and this asserts it
    rather than trusting it, because a silent divergence would make this script's
    output incomparable to the ladder it caps.
    """
    p = PRESETS[cfg_name]
    g = dict(n_part=p["n_part"], L=p["box"], n_fine=p["n_fine"],
             n_coarse=p["n_coarse"], tile=p["tile"], buf=p["buf"])
    if cfg_name in m3.CONFIGS:
        ref = m3._geom(cfg_name)
        for k in ("n_part", "n_fine", "n_coarse", "tile", "buf"):
            if int(g[k]) != int(ref[k]):
                raise ValueError(
                    f"geometry tables disagree on {cfg_name}.{k}: plan.PRESETS says "
                    f"{g[k]}, v2_m3_engine_gate says {ref[k]}. Every engine card on "
                    "record was measured through the second one."
                )
        if float(g["L"]) != float(ref["L"]):
            raise ValueError(f"geometry tables disagree on {cfg_name}.L")
    return g


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _cosmo():
    from inexor.config import Cosmology

    return Cosmology()


def _engine_config(g, args, checkpoint_dir):
    """The production config, from `inexor.plan.engine_config`.

    ONE definition, shared with the planner. It used to be built here from
    scratch, which is how the driver ended up running an f64 coarse mesh that
    M-v2-4 had rejected while the planner priced f32 and said FITS.
    """
    from inexor.plan import engine_config

    ec = engine_config(
        dict(n_part=g["n_part"], box=g["L"], n_fine=g["n_fine"],
             n_coarse=g["n_coarse"], tile=g["tile"], buf=g["buf"]),
        brick_slack=args.slack, tile_workers=args.tile_workers,
        checkpoint_dir=checkpoint_dir, checkpoint_every=args.checkpoint_every,
        # AUTO by default, never True: C14 made the library default a tri-state
        # precisely because a hard True refuses when no pool exists, and this
        # driver must not turn a serial smoke run into a refusal.
        **({} if args.migrate_pooled is None else
           {"migrate_pooled": args.migrate_pooled}),
        **({} if args.eject_kernel is None else {"eject_kernel": args.eject_kernel}),
    )
    ec.validate()
    return ec


def _coeffs(cosmo):
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

    a_steps = a_grid(m3.A_INIT, m3.A_FINAL, K_STEPS, m3.SPACING)
    return bullfrog_float_coeffs(bullfrog_table(a_steps, cosmo)), a_steps


def _card(kind, args, body, tag=""):
    card = dict(card=f"inexor-realization-{kind}-1", config=args.config,
                workdir=args.workdir, commit=_git_commit(), host=platform.node(),
                machine=platform.machine(), numpy=np.__version__,
                k_steps=K_STEPS, when=time.strftime("%Y-%m-%dT%H:%M:%S"), **body)
    # the tag keeps a segmented run's cards: without it each segment's card
    # overwrote the last and a multi-day run would end holding only its final leg
    path = os.path.join(args.workdir, f"realization_{kind}{tag}.json")
    with open(path, "w") as fh:
        json.dump(card, fh, indent=2, default=str)
    print(f"  card -> {path}")
    return card


def _ckpt_dir(args):
    return os.path.join(args.workdir, "ckpt")


def _newest_checkpoint_step(args):
    """The newest COMPLETE checkpoint's step, or None. Reads manifests only."""
    d = _ckpt_dir(args)
    best = None
    for gen in ("gen0", "gen1"):
        man = os.path.join(d, gen, "manifest.json")
        if not os.path.exists(man):
            continue
        try:
            with open(man) as fh:
                step = int(json.load(fh).get("step", -1))
        except (ValueError, OSError):
            continue
        if best is None or step > best:
            best = step
    return best


# ---------------------------------------------------------------------------


def cmd_ics(args):
    jax = _require_cpu()
    g = _geom(args.config)
    from inexor import icgen

    os.makedirs(args.workdir, exist_ok=True)
    key = jax.random.PRNGKey(args.seed)
    ec_nb = _engine_config(g, args, None)
    nb = g["n_fine"] // ec_nb.n_brick
    print(f"== ICs {args.config}: n_part={g['n_part']} L={g['L']} "
          f"bricks_per_side={nb} ({nb ** 3:,} bricks) -> {args.workdir}")

    t0 = time.perf_counter()
    man = icgen.generate_t9_slabs(
        args.workdir, key, g["n_part"], g["L"], _cosmo(), m3.A_INIT, nb,
        fdtype=GEN_FDTYPE, slab=args.slab, keep_stage=args.keep_stage,
    )
    wall = time.perf_counter() - t0
    peak = _maxrss_bytes()
    clean = man.get("stage_cleanup")
    print(f"  wall {wall / 60:.1f} min | peak host {peak / 1e9:.1f} GB "
          f"({peak / g['n_part'] ** 3:.1f} B/p)")
    print(f"  staging cleanup: {clean}")
    if not args.keep_stage:
        stage = os.path.join(args.workdir, icgen.STAGE_DIR)
        if os.path.isdir(stage):
            print(f"  NOTE: {stage} still stands -- a stray file blocked the rmdir. "
                  "The named intermediates are gone; this is reported, not fatal.")
    return _card("ics", args, dict(
        wall_s=wall, peak_rss_bytes=peak,
        bytes_per_particle=peak / g["n_part"] ** 3,
        bricks_per_side=int(nb), manifest=man, seed=args.seed, slab=args.slab,
    )) and 0


def cmd_run(args):
    _require_cpu()
    g = _geom(args.config)
    from inexor import engine, icgen

    cosmo = _cosmo()
    co, a_steps = _coeffs(cosmo)
    d = _ckpt_dir(args)
    os.makedirs(d, exist_ok=True)
    ec = _engine_config(g, args, d)

    # THE STATE IS BUILT STRAIGHT INTO SHARED MEMORY, so it exists once
    # rather than twice. Before this the loader made ~135 GB of private
    # arrays at c-gh and TilePool copied them into another ~135 GB; job
    # 922723 was OOM-killed doing exactly that. Serial runs get no allocator
    # and no pool, and behave as they always did.
    from inexor.executor import SharedAllocator, malloc_trim

    allocator = SharedAllocator() if ec.tile_workers > 1 else None

    have = _newest_checkpoint_step(args)
    resume = None
    t_load = time.perf_counter()
    if have is not None:
        st, resume = engine.load_checkpoint(d, ec, co, arena_frac=args.arena_frac,
                                            alloc=allocator)
        src = f"checkpoint at step {int(resume['step'])}"
        if int(resume["step"]) >= K_STEPS:
            print(f"  NOTHING TO DO: the checkpoint is already at step "
                  f"{int(resume['step'])} of {K_STEPS}.")
            return 0
    else:
        st = icgen.load_slot_state(
            args.workdir, brick_slack=args.slack, alloc_margin=args.alloc_margin,
            arena_frac=args.arena_frac, alloc=allocator,
        )
        src = "the ICs (step 0)"
    t_load = time.perf_counter() - t_load
    # glibc keeps freed arenas, and the pool's segments are fresh kernel pages
    # that cannot be served from them, so the loader's transients and the
    # pool's demand STACK unless this is called. Reported, not assumed.
    trimmed = malloc_trim()

    k0 = 0 if resume is None else int(resume["step"])
    stop = args.stop_at if args.stop_at else K_STEPS
    print(f"== RUN {args.config}: from {src} -> step {stop} of {K_STEPS}")
    print(f"  state {st.n_particles:,} particles, {st.n_bricks:,} bricks, "
          f"{st.off.shape[0]:,} rows; load {t_load:.1f} s")
    print(f"  W={ec.tile_workers} pooled_migrate={ec.migrate_pooled} "
          f"eject={ec.eject_kernel} slack={args.slack} ckpt_every={args.checkpoint_every}")
    # the dtypes were in no log line, which is most of why the f64 coarse mesh
    # kept being re-found rather than read
    print(f"  coarse={ec.coarse_dtype} fine={ec.fine_dtype} "
          f"arena_frac={args.arena_frac} alloc_margin={args.alloc_margin}")
    print(f"  state in shared memory: "
          f"{'yes, %.1f GB' % (allocator.bytes_held() / 1e9) if allocator else 'no (serial)'}"
          f"; malloc_trim={trimmed}")

    ph = PhaseTimer()
    stats = []
    t0 = time.perf_counter()
    out = engine.run(st, ec, co, phase=ph, resume=resume, stop_at=stop,
                     collect=stats.append, allocator=allocator)
    wall = time.perf_counter() - t0
    peak = _maxrss_bytes()

    n = len(out)
    per_step = wall / max(n, 1)
    ck = [i for i, o in enumerate(out) if o["checkpoint"] is not None]
    rp = [o["repack"] for o in out if o.get("repack")]
    print(f"\n  {n} steps in {wall / 3600:.3f} h = {per_step:.2f} s/step "
          f"| peak host {peak / 1e9:.1f} GB")
    print(f"  checkpoints written at segment steps {ck} "
          f"(absolute {[k0 + 1 + i for i in ck]})")
    if rp:
        f0 = rp[-1].get("bricks_fast", 0)
        m0 = rp[-1].get("bricks_merged", 0)
        print(f"  repack last step: {f0:,} fast / {m0:,} merged bricks "
              f"({100 * f0 / max(f0 + m0, 1):.1f}% fast)")
    # THE POOL'S WORKERS ARE OTHER PROCESSES, and `ru_maxrss` cannot see them.
    # `v2_m6_engine_peak` is serial-only for exactly this reason. A peak read
    # from the parent alone at W=16 would report a fraction of what the node is
    # actually holding, which is the one number deciding whether the full run
    # fits -- so the workers' own RSS is summed in and the total is what the
    # memory criterion is read against.
    arena_peak = max((d.get("arena_used", 0) for d in stats), default=0)
    if arena_peak:
        print(f"  arena peak residency: {arena_peak:,} rows "
              f"({100 * arena_peak / max(st.n_arena, 1):.1f}% of the arena, "
              f"{100 * arena_peak / max(st.n_particles, 1):.2f}% of particles)")
    w_rss = 0.0
    last = stats[-1] if stats else {}
    pool_rss = (last.get("pool") or {}).get("rss_mb") or {}
    if pool_rss:
        w_rss = sum(float(v) for v in pool_rss.values()) * 1e6
    print(f"  memory: parent peak {peak / 1e9:.1f} GB + {len(pool_rss)} workers "
          f"{w_rss / 1e9:.1f} GB = {(peak + w_rss) / 1e9:.1f} GB of the node")
    print(f"  RECEIPTS: migrate_pooled_workers={last.get('migrate_pooled_workers')} "
          f"tile_workers={last.get('tile_workers')} "
          f"coarse_pooled_workers={last.get('coarse_pooled_workers')}")
    rep = ph.report()
    print("  phase card (s over this segment):")
    for k, v in list(rep["per_phase"].items()):
        if v > 0:
            print(f"     {k:<16s} {v:9.2f}  {100 * rep['per_phase_frac'][k]:5.1f}%")

    if n == 0:
        print("  REFUSING: the segment advanced no steps.")
        return 2
    _card("run", args, dict(
        from_step=k0, to_step=k0 + n, n_steps=n, source=src,
        wall_s=wall, s_per_step=per_step, load_s=t_load, peak_rss_bytes=peak,
        tile_workers=ec.tile_workers, migrate_pooled=bool(ec.migrate_pooled),
        eject_kernel=str(ec.eject_kernel), brick_slack=args.slack,
        arena_frac=args.arena_frac, checkpoint_every=args.checkpoint_every,
        phase=rep, per_step_stats=stats, a_steps=list(map(float, a_steps)),
        projected_full_run_h=per_step * K_STEPS / 3600.0,
        worker_rss_bytes=w_rss, total_rss_bytes=peak + w_rss,
        arena_peak_rows=arena_peak, n_arena=int(st.n_arena),
        migrate_pooled_workers=last.get("migrate_pooled_workers"),
    ), tag=f"_{k0:02d}_{k0 + n:02d}")
    return 0


def _state_at_head(args, ec, co):
    from inexor import engine

    have = _newest_checkpoint_step(args)
    if have is None:
        raise SystemExit("no checkpoint to read; run `run` first")
    st, resume = engine.load_checkpoint(_ckpt_dir(args), ec, co,
                                        arena_frac=args.arena_frac)
    return st, int(resume["step"])


def cmd_export(args):
    _require_cpu()
    g = _geom(args.config)
    from inexor import export

    cosmo = _cosmo()
    co, a_steps = _coeffs(cosmo)
    ec = _engine_config(g, args, _ckpt_dir(args))
    st, step = _state_at_head(args, ec, co)
    if step < K_STEPS and not args.allow_partial:
        raise SystemExit(
            f"the checkpoint is at step {step} of {K_STEPS}; exporting now would "
            "produce a mock at the wrong epoch. Pass --allow-partial if that is "
            "deliberate."
        )
    out_dir = args.export_dir or os.path.join(args.workdir, "export")
    a_out = float(a_steps[-1]) if step >= K_STEPS else float(a_steps[step])
    print(f"== EXPORT {args.config} at step {step}, a={a_out:.4f} -> {out_dir}")

    t0 = time.perf_counter()
    man = export.write_particles(
        st, out_dir, dtype=np.float32, a=a_out, cosmo=cosmo,
        chunk_bricks=args.chunk_bricks,
        provenance=dict(config=args.config, step=step, commit=_git_commit(),
                        workdir=args.workdir),
    )
    wall = time.perf_counter() - t0
    peak = _maxrss_bytes()
    # SIZES FROM THE FILESYSTEM, not from the manifest: the manifest's `files`
    # maps a role to a NAME, and the first version of this line assumed it mapped
    # to a dict of stats and crashed after a clean export. Reporting code is
    # untested code until it has run.
    tot = 0
    for name in man.get("files", {}).values():
        try:
            tot += os.path.getsize(os.path.join(out_dir, str(name)))
        except OSError:
            pass
    print(f"  wall {wall / 60:.1f} min | peak host {peak / 1e9:.1f} GB"
          + (f" | {tot / 1e9:.1f} GB written" if tot else ""))
    print("  NOTE: velocities are the engine's D-time dx/dD unless the header "
          "says otherwise; the epoch and cosmology above set the km/s factor.")
    _card("export", args, dict(step=step, a_out=a_out, out_dir=out_dir,
                               wall_s=wall, peak_rss_bytes=peak, manifest=man))
    return 0


def cmd_card(args):
    _require_cpu()
    g = _geom(args.config)
    from inexor import summary

    cosmo = _cosmo()
    co, a_steps = _coeffs(cosmo)
    ec = _engine_config(g, args, _ckpt_dir(args))
    st, step = _state_at_head(args, ec, co)
    a_out = float(a_steps[-1]) if step >= K_STEPS else float(a_steps[step])
    print(f"== P(k) CARD {args.config} at step {step}, a={a_out:.4f}")

    t0 = time.perf_counter()
    card = summary.pk_summary_card(st, ec, cosmo, a_out, slab=args.slab,
                                   min_weight=args.min_weight)
    wall = time.perf_counter() - t0
    z = np.asarray(card["z_profile"], dtype=float)
    k = np.asarray(card["k_mean"], dtype=float)
    k_nl = float(card["k_nonlinear"])
    lin = k < k_nl
    print(f"  wall {wall / 60:.1f} min | {card['n_bins']} bins over "
          f"k = {k.min():.4f} to {k.max():.4f}, k_nonlinear = {k_nl:.4f}")
    # THE WHOLE-RANGE MEDIAN IS NOT THE NUMBER TO READ, and printing it alone
    # invites the wrong conclusion. The oracle is LINEAR theory, so a bin above
    # k_nonlinear is being compared against a prediction that does not apply
    # there; a large |z| in that band is the simulation being nonlinear, not the
    # engine being wrong. At a small box every bin can land above k_nl -- cdev8
    # does -- and then the card cannot speak to accuracy at all, which is worth
    # saying out loud rather than leaving a frightening median on the page.
    if lin.any():
        print(f"  BELOW k_nonlinear ({int(lin.sum())} bins): |z| median "
              f"{np.median(np.abs(z[lin])):.2f} max {np.max(np.abs(z[lin])):.2f}")
    else:
        print("  NO BIN LIES BELOW k_nonlinear at this geometry, so this card "
              "says nothing about accuracy against linear theory. Every bin is "
              "in the nonlinear regime the oracle does not model. Expected at a "
              "small box; at 2048^3 the fundamental mode is 0.006 h/Mpc and most "
              "of the range is linear.")
    print(f"  whole range, for completeness: |z| median {np.median(np.abs(z)):.2f} "
          f"max {np.max(np.abs(z)):.2f}")
    print("  NO VERDICT is emitted: `band_verdict` takes a band the caller names, "
          "and the k range this run is trusted over is not this script's call.")
    _card("pk", args, dict(step=step, a_out=a_out, wall_s=wall,
                           n_bins_below_k_nl=int(lin.sum()),
                           peak_rss_bytes=_maxrss_bytes(), summary=card))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("phase", choices=("ics", "run", "export", "card"))
    ap.add_argument("--config", default="cdev8", choices=sorted(PRESETS))
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--seed", type=int, default=m3.SEED)
    ap.add_argument("--slab", type=int, default=32)
    ap.add_argument("--keep-stage", action="store_true",
                    help="keep the IC intermediates (~687 GB at 2048^3)")
    ap.add_argument("--slack", type=float, default=0.20)
    ap.add_argument("--arena-frac", type=float, default=0.20)
    ap.add_argument("--alloc-margin", type=float, default=0.10)
    ap.add_argument("--tile-workers", type=int, default=16)
    ap.add_argument("--migrate-pooled", action="store_true", default=None)
    ap.add_argument("--serial-migrate", dest="migrate_pooled", action="store_false")
    ap.add_argument("--eject-kernel", default="jax", choices=("numpy", "jax"))
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--stop-at", type=int, default=None,
                    help="absolute step to stop before; must be a checkpoint boundary")
    ap.add_argument("--export-dir", default=None)
    ap.add_argument("--chunk-bricks", type=int, default=1024)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--min-weight", type=float, default=100.0)
    args = ap.parse_args()
    return {"ics": cmd_ics, "run": cmd_run, "export": cmd_export,
            "card": cmd_card}[args.phase](args)


if __name__ == "__main__":
    sys.exit(main())
