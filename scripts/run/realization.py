"""Generate, step, export and score one complete mock, one phase per process.

Subcommands:

    ics     generate the T9 IC slabs (host, or `--generator device` on the cards)
    run     step the schedule, or a segment of it, checkpointing as it goes
    export  write plain .npy (x, v) from the newest checkpoint
    card    the run's P(k) against linear theory, as a z profile, to a JSON card

Each subcommand is its own PROCESS, since a high-water mark never resets: that is
the only way each phase's peak host RSS is readable, and a queue limit costs one
phase, not the run.

Segments: `run --stop-at N` advances to absolute step N, which must be a checkpoint
boundary, and stops; the next `run` resumes from the newest checkpoint. Segments
compose BITWISE into the uninterrupted run
(`test_a_run_split_into_segments_is_bitwise_the_uninterrupted_one`) because each
is handed the full coefficient list and a stop step rather than a truncated
schedule -- `fused_drifts` makes `coeffs[:n]` a different trajectory.

Conventions and refusals:

- `run`, `export` and `card` require the CPU backend; so does `ics` unless
  `--generator device`.
- The preset geometry must equal `_instruments.CONFIGS` for any config both define.
- `run` resumes from a checkpoint whenever one exists and prints which source it
  used; starting from ICs, it refuses ICs made at another `--a-init`/`--growth2`.
- `export` refuses a checkpoint short of `--k-steps` unless `--allow-partial`.
- In the library: `write_particles` refuses a short file, `pk_summary_card` an
  empty card, and `stop_at` a stop off a checkpoint boundary.
- Walls, peaks, the phase card and the z profile go to
  `realization_{ics,run,export,pk}*.json` in `--workdir`. No verdict is emitted.

Usage (one phase per job step):

    python scripts/run/realization.py ics    --config c-gh --workdir $W
    python scripts/run/realization.py run    --config c-gh --workdir $W --stop-at 5
    python scripts/run/realization.py run    --config c-gh --workdir $W
    python scripts/run/realization.py export --config c-gh --workdir $W
    python scripts/run/realization.py card   --config c-gh --workdir $W
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

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _instruments import (  # noqa: E402
    A_FINAL,
    A_INIT,
    CONFIGS,
    SEED,
    SPACING,
    PhaseTimer,
    PhaseTracer,
    _maxrss_bytes,
    _require_cpu,
)
from _instruments import _geom as _base_geom  # noqa: E402
from inexor.plan import PRESETS, RATIFIED  # noqa: E402

# The IC generator's float dtype on the production path.
GEN_FDTYPE = np.float32
# The default step count; `--k-steps` overrides it. `device_run.py` imports
# `_coeffs`, whose default this is.
K_STEPS = 40


class _StreamingTracer(PhaseTracer):
    """`PhaseTracer` that also PRINTS each boundary as it is crossed.

    A run killed mid-step still leaves the phases it reached, in order, with the
    live reading (the end-of-run card would be lost). Relies on unbuffered stdout
    (`PYTHONUNBUFFERED`, set by the sbatch) so a SIGKILL cannot strand lines.

    `MemAvailable` is printed beside the parent's `VmHWM` because the pool's
    memory is in the workers: a flat parent peak with collapsing available memory
    is the workers growing, not the parent.
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
    """Refuse `--phase-instrument peak` without procfs (no macOS equivalent).

    Probes the files themselves rather than `sys.platform`: the capability is the
    file, and a platform name is only a proxy for it.
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
    """Print the phase card in the units of `instrument` ("time": s; "peak": GB).

    A separate function so the reporting path can be exercised cheaply before a
    long run reaches it.

    Peak numbers are the PARENT PROCESS ONLY; the pool workers' RSS is summed
    separately in `cmd_run`'s `memory:` line. A phase peak is not a node total.
    """
    if instrument != "peak":
        print("  phase card (s over this segment):")
        for k, v in list(rep["per_phase"].items()):
            if v > 0:
                print(f"     {k:<16s} {v:9.2f}  {100 * rep['per_phase_frac'][k]:5.1f}%")
        return
    # `peak` is the absolute RSS reached while the phase ran (what a host ceiling
    # cares about); `own` is that minus the RSS the phase started from (what the
    # phase itself allocated). A large peak with near-zero own is a phase running
    # inside someone else's residency.
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
    erfc(beta/2). The default alpha is `RATIFIED["alpha"]` = 1.0, so r_s is one
    coarse cell.
    """
    d_coarse = float(g["L"]) / int(g["n_coarse"])
    d_fine = float(g["L"]) / int(g["n_fine"])
    alpha = float(g.get("alpha", RATIFIED["alpha"]))
    r_s = alpha * d_coarse
    beta = (int(g["buf"]) * d_fine) / r_s
    return alpha, beta, math.exp(-math.pi**2 * alpha**2), math.erfc(beta / 2.0)


def _geom(cfg_name, n_fine=None, buf=None, n_coarse=None, n_part=None):
    """Geometry dict for preset `cfg_name`, with optional resolution overrides.

    `plan.PRESETS` is the source; for configs also in `_instruments.CONFIGS` the
    two tables must agree (ValueError otherwise), so small-config and preset
    results stay comparable. Overrides apply after that check:

    - `n_part`: particles per side (mass-resolution ladder). Box and meshes are
      held, so only the interparticle spacing varies. Must be a power of two;
      brick divisibility is checked by the IC generator and loader.
    - `n_coarse`: coarse mesh, with the split scale r_s = alpha * coarse_cell HELD
      physically fixed by deriving alpha. Refining raises alpha, which only lowers
      the coarse-representation error exp(-pi^2 alpha^2); coarsening (alpha below
      the default 1.0; 0.085 at alpha 0.5) is refused.
    - `n_fine`: fine mesh, with `buf` (counted in FINE cells) derived to hold beta.
      A fixed buf would shrink the physical buffer: at cgh64, erfc(beta/2) runs
      1.5e-8 -> 4.7e-3 -> 1.6e-1 over a 512..4096 ladder. An explicit `buf` wins.

    Prints the geometry and both split error terms.
    """
    p = PRESETS[cfg_name]
    g = dict(n_part=p["n_part"], L=p["box"], n_fine=p["n_fine"],
             n_coarse=p["n_coarse"], tile=p["tile"], buf=p["buf"])
    if cfg_name in CONFIGS:
        ref = _base_geom(cfg_name)
        for k in ("n_part", "n_fine", "n_coarse", "tile", "buf"):
            if int(g[k]) != int(ref[k]):
                raise ValueError(
                    f"geometry tables disagree on {cfg_name}.{k}: plan.PRESETS says "
                    f"{g[k]}, _instruments.CONFIGS says {ref[k]}."
                )
        if float(g["L"]) != float(ref["L"]):
            raise ValueError(f"geometry tables disagree on {cfg_name}.L")

    # the cross-check above must see the unmodified preset, so overrides land after it
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
                f"the default {RATIFIED['alpha']:g}. The coarse-representation "
                f"error exp(-pi^2 alpha^2) would be {math.exp(-math.pi ** 2 * g['alpha'] ** 2):.2e} "
                "against 5.17e-05, which is larger than anything this is measuring."
            )

    if n_part is not None and int(n_part) != int(g["n_part"]):
        n = int(n_part)
        if n < 2 or n & (n - 1):
            raise SystemExit(f"n_part {n} is not a power of two")
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
    """Provenance dict for the IC manifest: generator, commit, host, jax, XLA env.

    Stored in the manifest because the slabs outlive the run card; it is also the
    allocator receipt for an A/B between generator configurations.
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
    # the allocator knobs, verbatim, so cross-run wall comparisons can check them
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

    One definition shared with the planner, so the driver runs exactly the
    configuration the planner prices.
    """
    from inexor.plan import engine_config

    ec = engine_config(
        dict(n_part=g["n_part"], box=g["L"], n_fine=g["n_fine"],
             n_coarse=g["n_coarse"], tile=g["tile"], buf=g["buf"]),
        brick_slack=args.slack, tile_workers=args.tile_workers,
        checkpoint_dir=checkpoint_dir, checkpoint_every=args.checkpoint_every,
        # unset = the library's AUTO default, never True: a hard True refuses when
        # no pool exists, which would turn a serial smoke run into a refusal
        **({} if args.migrate_pooled is None else
           {"migrate_pooled": args.migrate_pooled}),
        **({} if args.eject_kernel is None else {"eject_kernel": args.eject_kernel}),
        # the alpha derived by --n-coarse; without it the split scale would
        # silently revert to the default
        **({} if "alpha" not in g else {"alpha": g["alpha"]}),
        # getattr: tests build a bare Namespace, and an absent flag must mean
        # the library default
        coarse_match_order=getattr(args, "coarse_match_order", 3),
    )
    ec.validate()
    return ec


def _coeffs(cosmo, k_steps=K_STEPS, a_init=None, growth2="lcdm"):
    """The BullFrog coefficients and the scale-factor grid for `k_steps` steps.

    Returns (coeffs, a_steps). `a_init` defaults to A_INIT = 0.1 (z = 9); every
    grid ends at A_FINAL, so cards at different step counts share their epoch and
    k bins. `growth2` selects the BullFrog weights' second-order growth ("lcdm",
    or the legacy "eds"). All three enter the coefficients, which the checkpoint
    fingerprint hashes, so arms differing in any of them cannot cross-resume.
    """
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

    a0 = A_INIT if a_init is None else float(a_init)
    if not 0.0 < a0 < A_FINAL:
        raise SystemExit(f"a_init {a0} must lie in (0, {A_FINAL})")
    a_steps = a_grid(a0, A_FINAL, int(k_steps), SPACING)
    return bullfrog_float_coeffs(bullfrog_table(a_steps, cosmo, growth2=growth2)), a_steps


def _require_ic_epoch(ic_dir, a_init):
    """Refuse ICs generated at a different epoch from the run's `a_init`.

    The generator bakes a_init into the displacements (D1, D2) and velocities
    (f1, f2); the run takes it from the a-grid. Evolving one under the other
    starts the right field at the wrong time and nothing downstream can see it.
    A manifest that records no `a_init` is accepted only at A_INIT = 0.1.
    """
    from inexor import icgen

    with open(os.path.join(ic_dir, icgen.MANIFEST)) as fh:
        have = json.load(fh).get("a_init")
    want = float(a_init)
    if have is None:
        if want != A_INIT:
            raise SystemExit(
                f"the ICs in {ic_dir} record no a_init, so they predate the flag and "
                f"were made at a = {A_INIT}; this run asks for a_init = {want}")
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
    # the tag keeps one card per segment of a segmented run
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
        # the device generator's host peak is still ru_maxrss; its card (GPU) peak
        # is read by the job's nvidia-smi sampler, not here
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

    # With a pool, the state is loaded straight into shared memory so it exists
    # once, not twice (private arrays plus TilePool's copy: ~135 GB each at
    # c-gh). Serial runs get no allocator and no pool.
    from inexor.executor import SharedAllocator, malloc_trim

    allocator = SharedAllocator() if ec.tile_workers > 1 else None

    have = _newest_checkpoint_step(args)
    resume = None
    t_load = time.perf_counter()
    if have is not None:
        st, resume = engine.load_checkpoint(d, ec, co, arena_frac=args.arena_frac,
                                            alloc_margin=args.alloc_margin, alloc=allocator)
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
    # pool's demand STACK unless this is called. The result is printed.
    trimmed = malloc_trim()

    k0 = 0 if resume is None else int(resume["step"])
    stop = args.stop_at if args.stop_at else args.k_steps
    print(f"== RUN {args.config}: from {src} -> step {stop} of {args.k_steps}")
    print(f"  state {st.n_particles:,} particles, {st.n_bricks:,} bricks, "
          f"{st.off.shape[0]:,} rows; load {t_load:.1f} s")
    print(f"  W={ec.tile_workers} pooled_migrate={ec.migrate_pooled} "
          f"eject={ec.eject_kernel} slack={args.slack} ckpt_every={args.checkpoint_every}")
    print(f"  coarse={ec.coarse_dtype} fine={ec.fine_dtype} "
          f"arena_frac={args.arena_frac} alloc_margin={args.alloc_margin} "
          f"coarse_match={ec.coarse_match}")
    print(f"  state in shared memory: "
          f"{'yes, %.1f GB' % (allocator.bytes_held() / 1e9) if allocator else 'no (serial)'}"
          f"; malloc_trim={trimmed}")

    # ONE phase callback, so the two instruments are exclusive: `PhaseTracer`
    # writes /proc/self/clear_refs at every boundary and the reset costs wall, so
    # a combined card's seconds would partly describe the instrument. In pool
    # mode the intra-tile boundaries do not fire, so `peak` costs ~7 clear_refs
    # per step.
    if args.phase_instrument == "peak":
        # refuse now, not at the first boundary, which comes after a load
        # measured in minutes
        _require_linux_for_peaks()
        ph = _StreamingTracer(trim="off")
    else:
        ph = PhaseTimer()
    stats = []
    t0 = time.perf_counter()
    # `epoch` is stored with the checkpoints so `python -m inexor.export` can
    # write km/s from a bare checkpoint directory without this driver's a-grid
    out = engine.run(st, ec, co, phase=ph, resume=resume, stop_at=stop,
                     collect=stats.append, allocator=allocator,
                     epoch=(a_steps, cosmo))
    wall = time.perf_counter() - t0
    # `clear_refs` resets ru_maxrss along with VmHWM (both read `mm->hiwater_rss`),
    # so under tracing the run's peak is only available as `PhaseTracer.run_peak`
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
    # The pool's workers are other processes, invisible to `ru_maxrss`, so their
    # own RSS (from the last step's stats) is summed in for the node total.
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
        # which instrument produced `phase` (seconds vs bytes), and so whether
        # `peak_rss_bytes` came from `PhaseTracer.run_peak` or ru_maxrss
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
    st, resume = engine.load_checkpoint(_ckpt_dir(args), ec, co, arena_frac=args.arena_frac,
                                        alloc_margin=args.alloc_margin, alloc=alloc)
    return st, int(resume["step"])


def _ic_state(args, alloc=None):
    """The IC slot state in `--ic-dir`, for a card at step 0.

    The ICs are a different artifact from a checkpoint -- `load_checkpoint`
    refuses them, by their manifest's `provenance.kind` -- so carding them
    needs `cmd_run`'s own IC branch rather than `_state_at_head`.

    Refuses a checkpoint directory: it would load through `load_slot_state`, and
    `cmd_card` would then score an evolved state against the a_init oracle,
    wrong by D(a)^2 with no warning. Also refuses an `--a-init`/`--growth2`
    mismatch.
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
    # sizes from the filesystem: the manifest's `files` maps a role to a file NAME
    tot = 0
    for name in man.get("files", {}).values():
        try:
            tot += os.path.getsize(os.path.join(out_dir, str(name)))
        except OSError:
            pass
    print(f"  wall {wall / 60:.1f} min | peak host {peak / 1e9:.1f} GB"
          + (f" | {tot / 1e9:.1f} GB written" if tot else ""))
    # the velocity units this export actually carries, from the returned header
    # (km/s, since this leg always passes `a` and `cosmo`)
    print(f"  velocities: {man['units']['velocity']} at a={a_out:.6g}, "
          f"Omega_m={cosmo.Omega_m!r}, h={cosmo.h!r}"
          + (f" (x{man['peculiar_velocity_factor']:.6g} on the engine's dx/dD)"
             if not man["velocity_is_dtime"] else ""))
    _card("export", args, dict(step=step, a_out=a_out, out_dir=out_dir,
                               wall_s=wall, peak_rss_bytes=peak, manifest=man))
    return 0


def _heartbeat(args):
    """The progress callback for the export and card loops, or None if `--heartbeat 0`.

    Cost is one clock read per chunk against a chunk costing ~0.1 s.
    """
    if not args.heartbeat:
        return None
    from inexor.progress import Heartbeat

    return Heartbeat(every=float(args.heartbeat))


def _linear_band(k, k_nl, scan_hi):
    """Which bins linear theory applies to, and how to print the nonlinear scale.

    `summary.nonlinear_scale` returns None when the linear Delta^2 never reaches
    1 anywhere it scanned (ordinary at early epochs). Then every bin at or below
    the scan ceiling `scan_hi` is linear; bins above it were never examined and
    are left out. Returns (mask, text).
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

    # The card's streamed paint (its largest stage; transform and binning follow)
    # can run on a pool: workers return bounded sub-blocks and the parent
    # accumulates integers, so the pooled mesh is BITWISE the serial one. The
    # pool is `paint_only`: a full TilePool allocates three coarse force meshes
    # (103.1 GB at c-hero) and a tile kernel per worker, which the card never
    # reads. The state goes straight into shared memory so it exists once:
    # otherwise TilePool copies every field (754.6 GB -> 1509 GB at c-hero,
    # against a 1026 GB node).
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
    # see `cmd_run`: without the trim the loader's transients and the pool stack
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
            # print the W and shm actually held, not that `--card-pool` was
            # passed. Spawn is a fixed cost (W interpreters importing jax),
            # reported apart from `wall` so serial-vs-pooled ratios exclude it.
            print(f"  paint pool: W={pool.workers}, spawn {t_pool:.1f} s, "
                  f"state in shared memory {allocator.bytes_held() / 1e9:.1f} GB, "
                  f"malloc_trim={trimmed}")
        else:
            print("  paint pool: none (serial)")
        # the default band is [0, half Nyquist] of the card's own coarse mesh;
        # `--k-max` pins the bins so arms at different coarse meshes share them
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
    # The oracle is LINEAR theory, so |z| is only meaningful below k_nonlinear;
    # above it a large |z| is the simulation being nonlinear. At a small box
    # (cdev8) every bin can lie above k_nl, and the card says so.
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
          # IC and checkpoint cards of one run are kept side by side, not overwritten
          tag=("_ics" if args.ic_dir else ""))
    return 0


def build_parser():
    """The CLI, separately so another driver can build the same configuration."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("phase", choices=("ics", "run", "export", "card"))
    ap.add_argument("--config", default="cdev8", choices=sorted(PRESETS))
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
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
    ap.add_argument("--a-init", type=float, default=A_INIT,
                    help="starting scale factor, for ICs and the step grid alike "
                         "(default 0.1, z = 9). Loading ICs made at "
                         "another epoch is refused")
    ap.add_argument("--coarse-match-order", type=int, default=3, choices=(2, 3),
                    help="assignment order the coarse match factor divides out. "
                         "The coarse arm paints TSC, so 3 (default) is correct; 2 "
                         "(CIC) is the legacy arm. In the checkpoint fingerprint when not 2, so arms "
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
                    help="ics: position bucket side in particle cells (default "
                         "2). The quantum is bucket/256, so 1 halves it. "
                         "Recorded in the manifest; run and card inherit it")
    ap.add_argument("--keep-stage", action="store_true",
                    help="keep the IC intermediates (~687 GB at 2048^3)")
    ap.add_argument("--generator", default="host", choices=("host", "device"),
                    help="ics: the host generator, or the IC stage on the cards")
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
                         "loops; 0 turns them off")
    return ap


def main():
    args = build_parser().parse_args()
    return {"ics": cmd_ics, "run": cmd_run, "export": cmd_export,
            "card": cmd_card}[args.phase](args)


if __name__ == "__main__":
    sys.exit(main())
