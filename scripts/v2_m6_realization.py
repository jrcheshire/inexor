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
import math
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
from inexor.plan import PRESETS, RATIFIED  # noqa: E402
from v2_m6_engine_peak import _maxrss_bytes, _require_cpu  # noqa: E402
from v2_m6_peak_trace import PhaseTracer  # noqa: E402
from v2_m6_phase_time import PhaseTimer  # noqa: E402

# The generator dtype the M-v2-5 record measured the production path at.
GEN_FDTYPE = np.float32
# The DEFAULT step count. `--k-steps` overrides it; the module constant stays
# because `v2_d7_hero_smoke.py` imports `_coeffs` and because every card and
# checkpoint on record was written at 40.
K_STEPS = 40


class _StreamingTracer(PhaseTracer):
    """`PhaseTracer`, but every boundary is EMITTED when it happens.

    923313 exists because nothing had taken a per-phase high-water at c-gh. It
    took one and I never saw it, because the card is accumulated and printed
    when the run finishes and the run was SIGKILLed in step 1 -- so the job
    measured exactly the thing it was built to measure and left no record of it.
    Both prior attempts had died mid-run; a report that only exists at the end
    was never going to survive one.

    So each boundary prints as it is crossed. A killed run leaves the phases it
    reached, in order, with the reading that was live when it died -- which is
    the line the next diagnosis starts from. `PYTHONUNBUFFERED` is set by the
    sbatch, so a SIGKILL cannot strand these in a buffer either (922790 printed
    nothing at all for exactly that reason).

    `MemAvailable` rides along because `VmHWM` is the PARENT'S, and at C-gh the
    parent is not where the pool's memory is. A phase whose parent peak is flat
    while the node's available memory collapses is the workers, and those two
    columns side by side are what distinguishes that from the parent growing.

    Subclassed rather than edited in: `v2_m6_peak_trace.py` is a ratified probe
    and D-v2-16 clause 7 gates promotion on those being unmodified.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._step_no = 0

    def __call__(self, name):
        # the parent's reading FIRST, so the print cannot perturb what it reports
        super().__call__(name)
        if name == "coarse_paint":
            self._step_no += 1
        avail = ""
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        avail = " avail %7.2f" % (int(line.split()[1]) * 1024 / 1e9)
                        break
        except OSError:
            pass
        if self.series:
            _, hwm, own = self.series[-1]
            print("  [phase] step %2d %-16s peak %7.2f  own %7.2f%s"
                  % (self._step_no, name, hwm / 1e9, own / 1e9, avail), flush=True)


def _require_linux_for_peaks():
    """Per-phase high-water needs procfs; there is no macOS equivalent.

    Checked by READING the files rather than by testing `sys.platform`, for the
    same reason `executor.has_memfd()` probes instead of consulting an
    attribute: the platform name is a proxy for the capability and this
    milestone has already shipped one gate that trusted the proxy and was wrong
    (`hasattr(os, "memfd_create")` is False on the conda-forge interpreter that
    can memfd perfectly well). Here the capability is the file.
    """
    for path in ("/proc/self/status", "/proc/self/clear_refs"):
        if not os.path.exists(path):
            raise SystemExit(
                f"FATAL: --phase-instrument peak needs {path}, which this "
                f"system ({sys.platform}) does not have. Per-phase high-water "
                "marks are Linux-only; use --phase-instrument time here and "
                "take the peaks on the cluster."
            )


def _print_phase_card(rep, instrument):
    """The phase card, in the units the instrument that produced it reports.

    A FUNCTION rather than inline in `cmd_run` so the node can exercise it in a
    second before spending forty minutes reaching it. This milestone has lost
    two cluster jobs on a `print` -- 447 on a key belonging to another arm's
    worker, 448 on one renamed out from under it -- and a reporting path that
    only ever runs after the expensive part is untested code by construction.

    **The peak numbers are the PARENT ONLY.** `VmHWM` is one process's, and the
    pool's workers are others; their RSS is summed separately in the `memory:`
    line above. A phase peak here is not a node total and must not be read as
    one.
    """
    if instrument != "peak":
        print("  phase card (s over this segment):")
        for k, v in list(rep["per_phase"].items()):
            if v > 0:
                print(f"     {k:<16s} {v:9.2f}  {100 * rep['per_phase_frac'][k]:5.1f}%")
        return
    # PEAK and OWN both, because they answer different questions and this
    # milestone has confused them before: `peak` is the absolute RSS reached
    # while the phase ran, which is what a host ceiling cares against; `own` is
    # that minus the RSS the phase started from, i.e. what the phase itself
    # allocated. A large peak with a near-zero own is a phase running inside
    # someone else's residency, and differencing two maxima cannot tell those
    # apart -- which is precisely how M-v2-6 Stage 0's gates became unreadable.
    print("  phase card (GB high-water over this segment, PARENT PROCESS ONLY):")
    rows = sorted(rep["phases"].items(), key=lambda kv: -kv[1]["peak"])
    for k, v in rows:
        if v["peak"] > 0:
            print(f"     {k:<16s} peak {v['peak'] / 1e9:8.2f}  "
                  f"own {v['delta'] / 1e9:8.2f}  x{v['visits']}")
    print(f"     {'RUN PEAK':<16s} peak {rep['run_peak'] / 1e9:8.2f}"
          "   accumulated across boundaries, NOT ru_maxrss")
    if rep.get("unknown_phases"):
        print(f"     unknown boundaries: {rep['unknown_phases']}")


def _split_terms(g):
    """(alpha, beta, coarse-representation error, buffer-truncation error).

    `forces.py` models both terms of the two-level split analytically:
    alpha = r_s/d_coarse gives exp(-pi^2 alpha^2) and beta = b/r_s gives
    erfc(beta/2). RATIFIED alpha is 1.0, so r_s is one coarse cell.
    """
    d_coarse = float(g["L"]) / int(g["n_coarse"])
    d_fine = float(g["L"]) / int(g["n_fine"])
    alpha = float(g.get("alpha", RATIFIED["alpha"]))
    r_s = alpha * d_coarse
    beta = (int(g["buf"]) * d_fine) / r_s
    return alpha, beta, math.exp(-math.pi**2 * alpha**2), math.erfc(beta / 2.0)


def _geom(cfg_name, n_fine=None, buf=None, n_coarse=None, n_part=None):
    """Geometry from the ratified preset table, CHECKED against the engine gate's.

    `n_part` overrides the particles per side, for a mass-resolution ladder. The
    box and both meshes are held, so the force (fine cell, coarse cell, r_s,
    beta) is identical across arms in physical units and only the interparticle
    spacing varies. Every preset sits at the same 0.5 Mpc/h spacing, so this is
    the one resolution axis the table cannot reach. Refused unless a power of
    two (D-007) that the brick grid divides.

    `n_coarse` overrides the coarse mesh. THE SPLIT SCALE IS THEN HELD: r_s is
    `alpha * coarse_cell` and alpha is ratified at 1.0, so refining the coarse
    mesh at fixed alpha would SHRINK r_s and change the force decomposition
    between arms -- that is a different experiment (is the ratified split
    right?) from the convergence one (is the coarse solve resolved?). Deriving
    alpha to hold r_s physically fixed keeps the decomposition identical, and
    only ever RAISES alpha, which drives the coarse-representation term
    exp(-pi^2 alpha^2) further down. Coarsening at fixed r_s would lower alpha
    instead: at alpha 0.5 that term is 0.085, so it is refused.

    `n_fine` overrides the preset's fine mesh, for a force-resolution ladder.
    THE BUFFER IS THEN DERIVED, not left alone: `buf` is counted in FINE CELLS,
    so holding it fixed while refining the mesh shrinks the PHYSICAL buffer and
    blows up the split's truncation error -- at cgh64, erfc(beta/2) runs
    1.5e-8 -> 4.7e-3 -> 1.6e-1 over a 512..4096 ladder. The finest arm would be
    16% wrong from truncation alone and would read as convergence going the
    wrong way. Deriving buf holds beta, so `n_fine` varies force resolution and
    nothing else. An explicit `buf` still wins, and says what it did to beta.

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

    # the cross-check above is the POINT of this function and must see the
    # ratified preset, so any override lands after it
    if n_fine is not None and int(n_fine) != int(g["n_fine"]):
        _, beta0, _, _ = _split_terms(g)
        ratio = int(n_fine) // int(g["n_coarse"])
        g["n_fine"] = int(n_fine)
        g["buf"] = int(buf) if buf is not None else int(round(ratio * RATIFIED["alpha"] * beta0))
        if g["tile"] + 2 * g["buf"] > g["n_fine"]:
            raise SystemExit(
                f"tile {g['tile']} + 2*buf {g['buf']} exceeds n_fine {g['n_fine']}"
            )
    elif buf is not None:
        g["buf"] = int(buf)

    if n_coarse is not None and int(n_coarse) != int(g["n_coarse"]):
        r_s = RATIFIED["alpha"] * float(g["L"]) / int(g["n_coarse"])
        g["n_coarse"] = int(n_coarse)
        g["alpha"] = r_s / (float(g["L"]) / int(n_coarse))
        if g["alpha"] < RATIFIED["alpha"]:
            raise SystemExit(
                f"holding r_s at {r_s:g} Mpc/h needs alpha={g['alpha']:.3f}, below "
                f"the ratified {RATIFIED['alpha']:g}. The coarse-representation "
                f"error exp(-pi^2 alpha^2) would be {math.exp(-math.pi ** 2 * g['alpha'] ** 2):.2e} "
                "against 5.17e-05, which is larger than anything this is measuring."
            )

    if n_part is not None and int(n_part) != int(g["n_part"]):
        # brick-grid divisibility is refused by the IC generator and the loader
        # (`bricks_per_side must divide n_part`), which own that layout
        n = int(n_part)
        if n < 2 or n & (n - 1):
            raise SystemExit(f"n_part {n} is not a power of two (D-007)")
        g["n_part"] = n

    alpha, beta, e_rep, e_trunc = _split_terms(g)
    print(f"  geometry {cfg_name}: n_part={g['n_part']} n_fine={g['n_fine']} "
          f"n_coarse={g['n_coarse']} T={g['tile']} b={g['buf']} | spacing "
          f"{float(g['L']) / g['n_part']:.4f}, fine cell "
          f"{float(g['L']) / g['n_fine']:.4f} Mpc/h")
    print(f"  split: alpha={alpha:g} beta={beta:g} r_s={alpha * float(g['L']) / g['n_coarse']:g} "
          f"Mpc/h | coarse repr {e_rep:.2e}, buffer truncation {e_trunc:.2e}")
    return g


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _ic_provenance(generator):
    """What produced these ICs, stored IN THE MANIFEST beside the slabs.

    The run card carries commit/host/machine already, but the card is a separate
    file and the slabs outlive it -- 998798's 4096^3 manifest went to disk with
    `provenance: {}` and nothing in c-hero-r0 says which backend, jax or allocator
    wrote it. Every 4096^3 run reads these slabs, and D-v2-23's `ic_stream` only
    separates the noise streams, not the rest. Also the allocator receipt for an
    A/B: the manifest, not a job label, says which arm an arm was.
    """
    prov = dict(generator=generator, commit=_git_commit(), host=platform.node(),
                machine=platform.machine(), numpy=np.__version__,
                when=time.strftime("%Y-%m-%dT%H:%M:%S"))
    try:
        import jax

        prov["jax"] = jax.__version__
        prov["x64"] = bool(jax.config.read("jax_enable_x64"))
        prov["devices"] = sorted({d.device_kind for d in jax.devices()})
        prov["n_devices"] = len(jax.devices())
    except Exception as e:  # a CPU generator on a node with no jax device
        prov["jax"] = f"unread ({e.__class__.__name__})"
    # the two knobs 997814 -> 998798 turned, verbatim, so a wall comparison across
    # generations can refuse rather than guess
    prov["allocator"] = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "bfc") or "bfc"
    for var in ("XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB", "XLA_FLAGS"):
        prov[var] = os.environ.get(var, "(unset)")
    return prov


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
        # a derived alpha is the whole point of --n-coarse; without this the
        # split scale would silently revert to the ratified default
        **({} if "alpha" not in g else {"alpha": g["alpha"]}),
        # getattr: tests build a bare Namespace, and an absent flag must mean
        # the library default
        coarse_match_order=getattr(args, "coarse_match_order", 3),
    )
    ec.validate()
    return ec


def _coeffs(cosmo, k_steps=K_STEPS, a_init=None, growth2="lcdm"):
    """The BullFrog coefficients and the scale-factor grid for `k_steps` steps.

    `a_init` defaults to the ratified start, a = 0.1 (z = 9). It enters the
    a-grid and so the coefficients, which the checkpoint fingerprint hashes:
    arms at different starts cannot cross-resume.

    Every arm ends at A_FINAL whatever `k_steps` is, so cards from different
    step counts share their epoch and their k bins and difference directly.
    The checkpoint fingerprint hashes these coefficients, so a run at one step
    count cannot resume another's checkpoint -- it is refused rather than
    silently continued onto a different trajectory. `growth2` selects the
    BullFrog weights' second-order growth ("lcdm", or the legacy "eds"), and
    also enters the coefficients and so the fingerprint.
    """
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

    a0 = m3.A_INIT if a_init is None else float(a_init)
    if not 0.0 < a0 < m3.A_FINAL:
        raise SystemExit(f"a_init {a0} must lie in (0, {m3.A_FINAL})")
    a_steps = a_grid(a0, m3.A_FINAL, int(k_steps), m3.SPACING)
    return bullfrog_float_coeffs(bullfrog_table(a_steps, cosmo, growth2=growth2)), a_steps


def _require_ic_epoch(ic_dir, a_init):
    """Refuse ICs generated at a different epoch from the run's `a_init`.

    The generator bakes a_init into the displacements (D1, D2) and velocities
    (f1, f2); the run takes it from the a-grid. Evolving one under the other
    starts the right field at the wrong time and nothing downstream can see it.
    A manifest from before `a_init` was recorded is accepted only at the
    ratified start, which is the only one that existed then.
    """
    from inexor import icgen

    with open(os.path.join(ic_dir, icgen.MANIFEST)) as fh:
        have = json.load(fh).get("a_init")
    want = float(a_init)
    if have is None:
        if want != m3.A_INIT:
            raise SystemExit(
                f"the ICs in {ic_dir} record no a_init, so they predate the flag and "
                f"were made at a = {m3.A_INIT}; this run asks for a_init = {want}")
        return
    if abs(float(have) - want) > 1e-12 * want:
        raise SystemExit(
            f"the ICs in {ic_dir} were generated at a_init = {float(have)!r}, and "
            f"this run asks for {want!r}. Regenerate them, or pass the matching --a-init.")


def _require_ic_growth2(ic_dir, growth2):
    """Refuse ICs whose 2LPT second-order growth differs from the run's.

    The generator bakes D2 and f2 into the ICs; the run's BullFrog weights take
    the same choice. A manifest that records none predates the flag and was made
    with the EdS approximation.
    """
    from inexor import icgen

    with open(os.path.join(ic_dir, icgen.MANIFEST)) as fh:
        have = json.load(fh).get("growth2", "eds")
    if have != growth2:
        raise SystemExit(
            f"the ICs in {ic_dir} were generated with growth2 = {have!r} and this run "
            f"asks for {growth2!r}. Regenerate them, or pass the matching --growth2.")


def _card(kind, args, body, tag=""):
    card = dict(card=f"inexor-realization-{kind}-1", config=args.config,
                workdir=args.workdir, commit=_git_commit(), host=platform.node(),
                machine=platform.machine(), numpy=np.__version__,
                k_steps=int(args.k_steps), growth2=getattr(args, "growth2", "lcdm"),
                when=time.strftime("%Y-%m-%dT%H:%M:%S"), **body)
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
    if args.generator == "device":
        # the device generator's host peak is still ru_maxrss; its card peak is read
        # by the job's nvidia-smi sampler, not here
        import jax

        jax.config.update("jax_enable_x64", True)
    else:
        jax = _require_cpu()
    g = _geom(args.config, args.n_fine, args.buf, args.n_coarse, args.n_part)
    from inexor import icgen

    os.makedirs(args.workdir, exist_ok=True)
    key = jax.random.PRNGKey(args.seed)
    ec_nb = _engine_config(g, args, None)
    nb = g["n_fine"] // ec_nb.n_brick
    print(f"== ICs {args.config} ({args.generator} generator): n_part={g['n_part']} "
          f"L={g['L']} bricks_per_side={nb} ({nb ** 3:,} bricks) -> {args.workdir}",
          flush=True)

    prov = _ic_provenance(args.generator)
    print(f"  provenance: allocator={prov['allocator']} "
          f"fraction={prov['XLA_CLIENT_MEM_FRACTION']} jax={prov.get('jax')}", flush=True)
    t0 = time.perf_counter()
    if args.generator == "device":
        man = icgen.generate_t9_slabs_device(
            args.workdir, key, g["n_part"], g["L"], _cosmo(), args.a_init, nb,
            fdtype=GEN_FDTYPE, slab=args.slab, keep_stage=args.keep_stage,
            pencil_batch=args.pencil_batch, noise=args.noise, provenance=prov,
            bucket_cells=args.bucket_cells, log=lambda line: print(line, flush=True),
            growth2=args.growth2,
        )
    else:
        man = icgen.generate_t9_slabs(
            args.workdir, key, g["n_part"], g["L"], _cosmo(), args.a_init, nb,
            fdtype=GEN_FDTYPE, slab=args.slab, keep_stage=args.keep_stage,
            provenance=prov, bucket_cells=args.bucket_cells, growth2=args.growth2,
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
        generator=args.generator, n_part=g["n_part"],
    )) and 0


def cmd_run(args):
    _require_cpu()
    g = _geom(args.config, args.n_fine, args.buf, args.n_coarse, args.n_part)
    from inexor import engine, icgen

    cosmo = _cosmo()
    co, a_steps = _coeffs(cosmo, args.k_steps, args.a_init, args.growth2)
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
        if int(resume["step"]) >= args.k_steps:
            print(f"  NOTHING TO DO: the checkpoint is already at step "
                  f"{int(resume['step'])} of {args.k_steps}.")
            return 0
    else:
        _require_ic_epoch(args.workdir, args.a_init)
        _require_ic_growth2(args.workdir, args.growth2)
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
    stop = args.stop_at if args.stop_at else args.k_steps
    print(f"== RUN {args.config}: from {src} -> step {stop} of {args.k_steps}")
    print(f"  state {st.n_particles:,} particles, {st.n_bricks:,} bricks, "
          f"{st.off.shape[0]:,} rows; load {t_load:.1f} s")
    print(f"  W={ec.tile_workers} pooled_migrate={ec.migrate_pooled} "
          f"eject={ec.eject_kernel} slack={args.slack} ckpt_every={args.checkpoint_every}")
    # the dtypes were in no log line, which is most of why the f64 coarse mesh
    # kept being re-found rather than read
    print(f"  coarse={ec.coarse_dtype} fine={ec.fine_dtype} "
          f"arena_frac={args.arena_frac} alloc_margin={args.alloc_margin} "
          f"coarse_match={ec.coarse_match}")
    print(f"  state in shared memory: "
          f"{'yes, %.1f GB' % (allocator.bytes_held() / 1e9) if allocator else 'no (serial)'}"
          f"; malloc_trim={trimmed}")

    # ONE phase callback, so the two instruments are exclusive rather than
    # composed, and that is deliberate: `PhaseTracer` writes /proc/self/clear_refs
    # at every boundary and the reset costs wall, which is the whole reason
    # `v2_m6_phase_time.py` exists as a separate probe. Running both would give a
    # phase card whose seconds describe the instrument.
    #
    # `--phase-instrument peak` is what answers the question this milestone is
    # stuck on. Nothing has ever taken a per-phase high-water at c-gh: 923139
    # died inside the coarse solve and all we have is a 10 s system sampler,
    # against which `inexor.plan` under-charged that phase by 5x. In pool mode
    # the intra-tile boundaries do not fire, so this is ~7 clear_refs per step
    # against a step measured in minutes.
    if args.phase_instrument == "peak":
        # REFUSE NOW, not at the first boundary. `PhaseTracer` needs
        # /proc/self/clear_refs, which macOS does not have, and the first
        # boundary is on the far side of a load measured in minutes -- so
        # without this the failure mode is "the job died after the expensive
        # part, on the instrument".
        _require_linux_for_peaks()
        ph = _StreamingTracer(trim="off")
    else:
        ph = PhaseTimer()
    stats = []
    t0 = time.perf_counter()
    # `epoch` costs nothing at run time and is what lets `python -m inexor.export`
    # write km/s off a bare checkpoint directory, without a reader having to
    # know this driver's a-grid and reproduce it by hand.
    out = engine.run(st, ec, co, phase=ph, resume=resume, stop_at=stop,
                     collect=stats.append, allocator=allocator,
                     epoch=(a_steps, cosmo))
    wall = time.perf_counter() - t0
    # `clear_refs` RESETS ru_maxrss ALONG WITH VmHWM -- both read the kernel's
    # one `mm->hiwater_rss` -- so after a traced run `_maxrss_bytes()` reports
    # the peak since the last boundary, not the run's. `PhaseTracer` accumulates
    # `run_peak` across boundaries for exactly this reason; taking it from there
    # is not a preference, it is the only correct source under tracing.
    peak = ph.run_peak if args.phase_instrument == "peak" else _maxrss_bytes()

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
    _print_phase_card(rep, args.phase_instrument)

    if n == 0:
        print("  REFUSING: the segment advanced no steps.")
        return 2
    _card("run", args, dict(
        from_step=k0, to_step=k0 + n, n_steps=n, source=src,
        wall_s=wall, s_per_step=per_step, load_s=t_load, peak_rss_bytes=peak,
        tile_workers=ec.tile_workers, migrate_pooled=bool(ec.migrate_pooled),
        eject_kernel=str(ec.eject_kernel), brick_slack=args.slack,
        arena_frac=args.arena_frac, checkpoint_every=args.checkpoint_every,
        # WHICH instrument produced `phase`, on the card rather than inferable
        # from its shape: the two reports carry different keys and different
        # units, and a reader that guesses wrong reads seconds as gigabytes.
        # It also records that `peak_rss_bytes` came from `PhaseTracer.run_peak`
        # rather than ru_maxrss, which clear_refs would have made meaningless.
        phase_instrument=args.phase_instrument,
        phase=rep, per_step_stats=stats, a_steps=list(map(float, a_steps)),
        projected_full_run_h=per_step * args.k_steps / 3600.0,
        worker_rss_bytes=w_rss, total_rss_bytes=peak + w_rss,
        arena_peak_rows=arena_peak, n_arena=int(st.n_arena),
        migrate_pooled_workers=last.get("migrate_pooled_workers"),
    ), tag=f"_{k0:02d}_{k0 + n:02d}")
    return 0


def _state_at_head(args, ec, co, alloc=None):
    from inexor import engine

    have = _newest_checkpoint_step(args)
    if have is None:
        raise SystemExit("no checkpoint to read; run `run` first")
    st, resume = engine.load_checkpoint(_ckpt_dir(args), ec, co,
                                        arena_frac=args.arena_frac, alloc=alloc)
    return st, int(resume["step"])


def _ic_state(args, alloc=None):
    """The IC slot state in `--ic-dir`, for a card at step 0.

    The ICs are a different artifact from a checkpoint -- `load_checkpoint`
    refuses them, by their manifest's `provenance.kind` -- so carding them
    needs `cmd_run`'s own IC branch rather than `_state_at_head`.

    The refusal below is the point of the function. A checkpoint generation
    loads perfectly well through `load_slot_state`, and `cmd_card` would then
    take its epoch from the `step = 0` this returns and compare a step-40 state
    against the a = 0.1 oracle -- a card that is wrong by D(a)^2 and says so
    nowhere.
    """
    from inexor import icgen

    man = os.path.join(args.ic_dir, icgen.MANIFEST)
    if not os.path.exists(man):
        raise SystemExit(f"no {icgen.MANIFEST} in --ic-dir {args.ic_dir}")
    with open(man) as fh:
        kind = json.load(fh).get("provenance", {}).get("kind")
    if kind == "inexor-checkpoint":
        raise SystemExit(
            f"--ic-dir {args.ic_dir} is a CHECKPOINT, not an IC generation. "
            "Carding it here would place it at step 0 and score it against the "
            "a = %.4f oracle. Drop --ic-dir to card the newest checkpoint."
            % float(args.a_init)
        )
    _require_ic_epoch(args.ic_dir, args.a_init)
    _require_ic_growth2(args.ic_dir, args.growth2)
    return icgen.load_slot_state(
        args.ic_dir, brick_slack=args.slack, alloc_margin=args.alloc_margin,
        arena_frac=args.arena_frac, alloc=alloc,
    )


def cmd_export(args):
    _require_cpu()
    g = _geom(args.config, args.n_fine, args.buf, args.n_coarse, args.n_part)
    from inexor import export

    cosmo = _cosmo()
    co, a_steps = _coeffs(cosmo, args.k_steps, args.a_init, args.growth2)
    ec = _engine_config(g, args, _ckpt_dir(args))
    st, step = _state_at_head(args, ec, co)
    if step < args.k_steps and not args.allow_partial:
        raise SystemExit(
            f"the checkpoint is at step {step} of {args.k_steps}; exporting now would "
            "produce a mock at the wrong epoch. Pass --allow-partial if that is "
            "deliberate."
        )
    out_dir = args.export_dir or os.path.join(args.workdir, "export")
    a_out = float(a_steps[-1]) if step >= args.k_steps else float(a_steps[step])
    print(f"== EXPORT {args.config} at step {step}, a={a_out:.4f} -> {out_dir}")

    t0 = time.perf_counter()
    man = export.write_particles(
        st, out_dir, dtype=np.float32, a=a_out, cosmo=cosmo,
        chunk_bricks=args.chunk_bricks,
        provenance=dict(config=args.config, step=step, commit=_git_commit(),
                        workdir=args.workdir),
        progress=_heartbeat(args),
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
    # Report the units this export ACTUALLY carries, off the returned header.
    # The line here used to say "D-time unless the header says otherwise",
    # which was true and useless: this leg always passes `a` and `cosmo`, so it
    # always writes km/s, and a reader had to go open the header to learn it.
    print(f"  velocities: {man['units']['velocity']} at a={a_out:.6g}, "
          f"Omega_m={cosmo.Omega_m!r}, h={cosmo.h!r}"
          + (f" (x{man['peculiar_velocity_factor']:.6g} on the engine's dx/dD)"
             if not man["velocity_is_dtime"] else ""))
    _card("export", args, dict(step=step, a_out=a_out, out_dir=out_dir,
                               wall_s=wall, peak_rss_bytes=peak, manifest=man))
    return 0


def _heartbeat(args):
    """The progress callback for the two hour-long product legs, or None.

    Off by `--heartbeat 0`, which is how an A/B keeps the legs comparable; the
    call itself is a clock read per chunk against a chunk costing ~0.1 s.
    """
    if not args.heartbeat:
        return None
    from inexor.progress import Heartbeat

    return Heartbeat(every=float(args.heartbeat))


def _linear_band(k, k_nl, scan_hi):
    """Which bins linear theory applies to, and how to print the nonlinear scale.

    `summary.nonlinear_scale` returns None when the linear Delta^2 never reaches
    1 anywhere it scanned -- the ordinary state of an early output, and what
    `float(card["k_nonlinear"])` died on in gb 1010730's cgh64 card leg at
    a = 0.1189. With no crossing, every bin under the scan ceiling is linear;
    bins above it were never examined and are left out rather than assumed.
    Returns (mask, text).
    """
    k = np.asarray(k, dtype=float)
    if k_nl is not None:
        return k < float(k_nl), f"{float(k_nl):.4f}"
    if scan_hi is None:
        return np.zeros(k.shape, dtype=bool), "unknown (the card carries no scan range)"
    return k <= float(scan_hi), f"none below {float(scan_hi):g} h/Mpc"


def cmd_card(args):
    _require_cpu()
    g = _geom(args.config, args.n_fine, args.buf, args.n_coarse, args.n_part)
    from inexor import summary
    from inexor.executor import SharedAllocator, TilePool, malloc_trim

    cosmo = _cosmo()
    co, a_steps = _coeffs(cosmo, args.k_steps, args.a_init, args.growth2)
    ec = _engine_config(g, args, _ckpt_dir(args))

    # THE PAINT IS THE CARD'S LARGEST STAGE AND IT IS THE ONE THAT POOLS.
    # `pk_summary_card` is one call: a streamed paint over `n_bricks /
    # chunk_bricks` chunks, then a transform, then the binning. Only the first
    # is embarrassingly parallel, and it is the only one of the three that the
    # engine's own runs have ever parallelised -- the card was the consumer
    # nobody pooled. Same machinery, same guarantee: workers return bounded
    # sub-blocks and the parent accumulates, so integer associativity makes the
    # pooled mesh BITWISE the serial one.
    #
    # The pool is `paint_only`: a full TilePool allocates three coarse force
    # meshes (103.1 GB at c-hero) and builds a tile kernel per worker, and the
    # card reads neither.
    #
    # THE STATE IS LOADED STRAIGHT INTO SHARED MEMORY so it exists once rather
    # than twice -- without the allocator `TilePool` copies every field and the
    # 754.6 GB state becomes 1509 GB against a 1026 GB node. Same fault that
    # OOM-killed job 922723.
    pooled = args.card_pool
    if pooled is None:
        pooled = ec.tile_workers > 1
    allocator = SharedAllocator() if pooled else None

    t_load = time.perf_counter()
    if args.ic_dir:
        st, step = _ic_state(args, alloc=allocator), 0
    else:
        st, step = _state_at_head(args, ec, co, alloc=allocator)
    t_load = time.perf_counter() - t_load
    # glibc keeps freed arenas and the pool's segments are fresh kernel pages
    # that cannot be served from them, so the loader's transients and the
    # pool's demand STACK without this.
    trimmed = malloc_trim() if pooled else None

    a_out = float(a_steps[-1]) if step >= args.k_steps else float(a_steps[step])
    print(f"== P(k) CARD {args.config} at step {step}, a={a_out:.4f}")
    print(f"  state {st.n_particles:,} particles, {st.n_bricks:,} bricks; "
          f"load {t_load:.1f} s")

    pool = None
    t0 = time.perf_counter()
    try:
        if pooled:
            t_pool = time.perf_counter()
            pool = TilePool(st, ec, allocator=allocator, paint_only=True)
            t_pool = time.perf_counter() - t_pool
            # A knob must prove it applied: W and the shm actually held, not
            # the fact that `--card-pool` was passed. Spawn is reported apart
            # from `wall` because it is FIXED -- 16 interpreters importing jax
            # -- so it is a large share of a cgh64 card and a rounding error on
            # a hero one, and a serial-vs-pooled ratio that leaves it inside is
            # read at the wrong scale.
            print(f"  paint pool: W={pool.workers}, spawn {t_pool:.1f} s, "
                  f"state in shared memory {allocator.bytes_held() / 1e9:.1f} GB, "
                  f"malloc_trim={trimmed}")
        else:
            print("  paint pool: none (serial)")
        # EVERY ARM MUST REPORT ON THE SAME BINS. The default band is
        # [0, half Nyquist] of the CARD'S OWN coarse mesh, so two arms at
        # different coarse meshes would silently measure different k and the
        # difference between them would be a resampling, not a result.
        edges = None
        if args.k_max:
            edges = np.linspace(0.0, float(args.k_max), int(args.n_bins) + 1)
            print(f"  bins PINNED: {int(args.n_bins)} over k = 0 to {float(args.k_max):g}"
                  f" (the card's own half-Nyquist is "
                  f"{0.5 * np.pi * ec.n_coarse / ec.box_size:.4f})")
        card = summary.pk_summary_card(st, ec, cosmo, a_out, slab=args.slab,
                                       min_weight=args.min_weight, edges=edges,
                                       progress=_heartbeat(args), pool=pool)
    finally:
        if pool is not None:
            pool.close()
    wall = time.perf_counter() - t0
    z = np.asarray(card["z_profile"], dtype=float)
    k = np.asarray(card["k_mean"], dtype=float)
    k_nl = card["k_nonlinear"]
    lin, k_nl_txt = _linear_band(k, k_nl, card.get("k_nonlinear_scan", [None, None])[1])
    print(f"  wall {wall / 60:.1f} min | {card['n_bins']} bins over "
          f"k = {k.min():.4f} to {k.max():.4f}, k_nonlinear = {k_nl_txt}")
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
    elif k_nl is None:
        print("  The linear Delta^2 never reaches 1 over the range scanned, so "
              "there is no nonlinear scale to place these bins against, and the "
              "band runs past the range that was looked at.")
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
                           load_s=t_load,
                           card_pool_workers=(0 if pool is None else pool.workers),
                           pool_spawn_s=(None if pool is None else t_pool),
                           n_bins_below_k_nl=int(lin.sum()),
                           ic_dir=args.ic_dir,
                           peak_rss_bytes=_maxrss_bytes(), summary=card),
          # an IC card and a checkpoint card of the same run are two different
          # epochs of one realization and belong side by side; sharing
          # `realization_pk.json` would mean the second silently replaced the
          # first, which is the comparison both exist for
          tag=("_ics" if args.ic_dir else ""))
    return 0


def build_parser():
    """The CLI, separately so another driver can build the same configuration."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("phase", choices=("ics", "run", "export", "card"))
    ap.add_argument("--config", default="cdev8", choices=sorted(PRESETS))
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--seed", type=int, default=m3.SEED)
    ap.add_argument("--slab", type=int, default=32)
    ap.add_argument("--n-fine", type=int, default=None,
                    help="override the preset's fine mesh, for a force-resolution "
                         "ladder. The buffer is DERIVED to hold the split's beta "
                         "unless --buf is also given")
    ap.add_argument("--n-coarse", type=int, default=None,
                    help="override the coarse mesh. alpha is DERIVED to hold the "
                         "split scale r_s physically fixed, so the force "
                         "decomposition is identical across arms and only the "
                         "coarse solve's resolution varies")
    ap.add_argument("--n-part", type=int, default=None,
                    help="override the particles per side, for a mass-resolution "
                         "ladder. Box and both meshes are held, so only the "
                         "interparticle spacing varies. A different n_part at the "
                         "same seed is an UNRELATED realization")
    ap.add_argument("--a-init", type=float, default=m3.A_INIT,
                    help="starting scale factor, for ICs and the step grid alike "
                         "(default the ratified 0.1, z = 9). Loading ICs made at "
                         "another epoch is refused")
    ap.add_argument("--coarse-match-order", type=int, default=3, choices=(2, 3),
                    help="assignment order the coarse match factor divides out. "
                         "The coarse arm paints TSC, so 3 (default) is correct; 2 "
                         "(CIC) is the legacy arm the probe parities were ratified "
                         "on. In the checkpoint fingerprint when not 2, so arms "
                         "cannot cross-resume")
    ap.add_argument("--growth2", default="lcdm", choices=("lcdm", "eds"),
                    help="second-order growth for the 2LPT ICs and the BullFrog "
                         "weights: the LCDM ODE solution (default) or the legacy EdS "
                         "-(3/7) D^2, which converges to an EdS-coupled solution. "
                         "Recorded in the IC manifest; run and card refuse ICs made "
                         "with the other")
    ap.add_argument("--buf", type=int, default=None,
                    help="override the buffer in FINE CELLS. Changes beta and the "
                         "split's truncation error, both of which get printed")
    ap.add_argument("--bucket-cells", type=int, default=2,
                    help="ics: position bucket side in particle cells (D-v2-14 "
                         "ratifies 2). The quantum is bucket/256, so 1 halves it. "
                         "Recorded in the manifest; run and card inherit it")
    ap.add_argument("--keep-stage", action="store_true",
                    help="keep the IC intermediates (~687 GB at 2048^3)")
    ap.add_argument("--generator", default="host", choices=("host", "device"),
                    help="ics: the host generator, or the IC stage on the cards (D6)")
    ap.add_argument("--noise", default="device", choices=("device", "host"),
                    help="ics --generator device: draw the noise on the cards "
                         "(IC_STREAM_DEVICE) or on the CPU (IC_STREAM)")
    ap.add_argument("--pencil-batch", type=int, default=1,
                    help="ics --generator device: y-pencil planes per card program")
    ap.add_argument("--slack", type=float, default=0.20)
    ap.add_argument("--arena-frac", type=float, default=0.20)
    ap.add_argument("--alloc-margin", type=float, default=0.10)
    ap.add_argument("--tile-workers", type=int, default=16)
    ap.add_argument("--migrate-pooled", action="store_true", default=None)
    ap.add_argument("--serial-migrate", dest="migrate_pooled", action="store_false")
    ap.add_argument("--eject-kernel", default="jax", choices=("numpy", "jax"))
    ap.add_argument("--phase-instrument", default="time", choices=("time", "peak"),
                    help="what the phase hook measures. 'time' is the phase card "
                         "(PhaseTimer); 'peak' takes a per-phase HIGH-WATER "
                         "(PhaseTracer, Linux only) and costs wall at every "
                         "boundary, so the two are exclusive and a run reports "
                         "one or the other, never both")
    ap.add_argument("--k-steps", type=int, default=K_STEPS,
                    help="number of BullFrog steps from a_init to a_final. Every "
                         "count ends at the same epoch, so cards from different "
                         "counts share their k bins. Changing it changes the "
                         "checkpoint fingerprint, so arms cannot cross-resume")
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--stop-at", type=int, default=None,
                    help="absolute step to stop before; must be a checkpoint boundary")
    ap.add_argument("--export-dir", default=None)
    ap.add_argument("--chunk-bricks", type=int, default=1024)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--min-weight", type=float, default=100.0)
    ap.add_argument("--k-max", type=float, default=None,
                    help="card: pin the top of the binning instead of taking "
                         "half the card's own Nyquist. Required to compare arms "
                         "whose coarse meshes differ, since otherwise each "
                         "reports on its own band")
    ap.add_argument("--n-bins", type=int, default=64,
                    help="card: bins over [0, --k-max]")
    ap.add_argument("--ic-dir", default=None,
                    help="card: read the ICs in this directory at step 0 "
                         "(a = a_init) instead of the newest checkpoint under "
                         "--workdir. The card still lands in --workdir, so the "
                         "IC generation is never written to. Refuses a "
                         "checkpoint directory, which would be scored against "
                         "the wrong epoch")
    ap.add_argument("--card-pool", action="store_true", default=None,
                    help="card: run the streamed paint on a paint-only TilePool "
                         "of --tile-workers workers (bitwise the serial mesh). "
                         "Default: on whenever --tile-workers > 1")
    ap.add_argument("--serial-card", dest="card_pool", action="store_false",
                    help="card: force the serial paint (the A/B's other arm)")
    ap.add_argument("--heartbeat", type=float, default=60.0,
                    help="seconds between progress lines in the card and export "
                         "loops; 0 turns them off. gb 1010938 ran 3 h 23 min "
                         "inside the card with no way to read its progress")
    return ap


def main():
    args = build_parser().parse_args()
    return {"ics": cmd_ics, "run": cmd_run, "export": cmd_export,
            "card": cmd_card}[args.phase](args)


if __name__ == "__main__":
    sys.exit(main())
