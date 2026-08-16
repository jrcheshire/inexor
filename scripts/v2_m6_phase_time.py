"""Where does a step's WALL go? Per-phase timing, with a neutrality control.

Every phase instrument in this milestone so far measures MEMORY:
`v2_m6_peak_trace.py` resets `VmHWM` at each boundary, which is the right tool
for a peak and the wrong one for a wall, because the reset itself costs time.
Nothing times the phases, so where a step's seconds go has never been measured
on the current code -- and the milestone's binding constraint moved from memory
to wall today, so that is now the question.

**Two things this is built to avoid**, both paid for earlier in M-v2-6.

An instrument that moves what it measures. `engine.step` takes a `phase`
callback defaulting to a no-op; this one does a `perf_counter` per boundary and
nothing else, and the `control` arm runs the identical configuration with
`phase=None`. If the two disagree on total wall by more than the run-to-run
scatter, the breakdown is describing the instrument. That is the same gate
shape `v2_m6_peak_trace.py` uses for the peak, and it is reported rather than
assumed.

A reading taken from one run. Phases differ hugely in cost and the cheap ones
are where per-call overhead hides, so `--repeats` runs the whole thing several
times and the report is a median with the spread beside it.

**What it cannot see:** anything inside a jitted region, which is one phase
boundary as far as this is concerned; and the first step, which pays
compilation and is reported separately rather than folded in -- a mean over K
that includes step 1 is a compile measurement wearing a step's clothes.

    pixi run python scripts/v2_m6_phase_time.py --config cdev8 --k 5
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

import v2_m3_engine_gate as m3  # noqa: E402

PHASES = (
    "kernel_build", "lead_drift", "coarse_paint", "coarse_solve", "membership",
    "tile_decode", "tile_short", "tile_long", "tile_reduce",
    "tile_loop_end", "reconcile", "migrate", "repack",
)


class PhaseTimer:
    """Wall between consecutive boundaries, accumulated per phase name.

    Per STEP as well as per phase, so step 1's compilation can be separated
    instead of averaged into the answer.
    """

    def __init__(self):
        self.total = {p: 0.0 for p in PHASES}
        self.per_step = []
        self._cur = {}
        self._t = time.perf_counter()
        self.unknown = []

    def __call__(self, name):
        now = time.perf_counter()
        dt = now - self._t
        self._t = now
        if name not in self.total:
            self.unknown.append(name)
            self.total[name] = 0.0
        self.total[name] += dt
        self._cur[name] = self._cur.get(name, 0.0) + dt
        # `repack` is the last boundary of a step when it runs; `migrate` when
        # it does not. Either way the step ends after the migrate, so close the
        # bucket there and let a repack land in the next one rather than
        # guessing the cadence.
        if name == "migrate":
            self.per_step.append(self._cur)
            self._cur = {}

    def report(self):
        tot = sum(self.total.values())
        return dict(
            total_s=tot,
            per_phase={k: v for k, v in sorted(self.total.items(), key=lambda kv: -kv[1])},
            per_phase_frac={k: (v / tot if tot else 0.0) for k, v in self.total.items()},
            n_steps_seen=len(self.per_step),
            unknown_phases=sorted(set(self.unknown)),
        )


_IC_CACHE = {}


def _enable_x64():
    """Callers opt in; library code never toggles it (repo convention).

    `v2_m3_engine_gate.py:346` does the same, and this script builds its ICs
    through that module's `make_ics`, which refuses float64 white noise without
    it rather than silently returning float32 -- so this is load-bearing, not
    hygiene.
    """
    import jax

    jax.config.update("jax_enable_x64", True)


def _build(cfg_name, slack, arena_frac, tile=None, buf=32, tile_workers=1,
           paint_subblock=True):
    _enable_x64()
    import jax.numpy as jnp  # noqa: F401  (engine import order)

    from inexor import engine, state
    from inexor.codec import T9Layout
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

    # `_geom` supplies tile/buf, which CONFIGS does not carry -- the same
    # defaults the M-v2-3 gate runs at, so a breakdown here describes the
    # geometry every ratified engine number was measured at.
    g = m3._geom(cfg_name, tile=tile, buf=buf)
    # ICs CACHED across runs. Seven engine runs (warmup + three arms twice) at
    # cdev would otherwise pay seven full linear-density + LPT generations for
    # one set of numbers. The state is rebuilt fresh from the cached (x, v)
    # every time, so each run still starts from an identical container -- the
    # ICs are an input, not part of what is being timed.
    key = (cfg_name, g["tile"], g["buf"])
    if key not in _IC_CACHE:
        _IC_CACHE[key] = m3.make_ics(g)
    x, v, cosmo = _IC_CACHE[key]
    t9 = T9Layout(g["L"], g["n_part"], 2)
    ec = engine.EngineConfig(
        box_size=g["L"], n_part=g["n_part"], n_fine=g["n_fine"], n_coarse=g["n_coarse"],
        n_tile=g["tile"], b_fine=g["buf"], alpha=m3.ALPHA, brick_slack=slack,
        tile_workers=tile_workers, paint_subblock=paint_subblock,
    )
    ec.validate()
    st = state.SlotState.build(x, v, t9, ec.n_brick and (g["n_fine"] // ec.n_brick),
                               brick_slack=slack, arena_frac=arena_frac)
    return engine, ec, st, cosmo, a_grid, bullfrog_float_coeffs, bullfrog_table


def _one(cfg_name, k, slack, arena_frac, timed, tile=None, buf=32, tile_workers=1,
         paint_subblock=True):
    engine, ec, st, cosmo, a_grid, bfc, bft = _build(
        cfg_name, slack, arena_frac, tile, buf, tile_workers, paint_subblock
    )
    a_steps = a_grid(m3.A_INIT, m3.A_FINAL, k, m3.SPACING)
    co = bfc(bft(a_steps, cosmo))
    ph = PhaseTimer() if timed else None
    seen = []
    t0 = time.perf_counter()
    if timed:
        engine.run(st, ec, co, phase=ph, collect=seen.append)
    else:
        engine.run(st, ec, co, collect=seen.append)
    wall = time.perf_counter() - t0
    return wall, (ph.report() if ph else None), seen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="cdev8", choices=sorted(m3.CONFIGS))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--slack", type=float, default=0.10)
    ap.add_argument("--arena-frac", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=1,
                    help="throwaway runs before the arms are compared; the "
                         "first run in a process pays compilation and the "
                         "second arm would inherit it")
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--buf", type=int, default=32)
    ap.add_argument("--paint-subblock", type=int, default=1, choices=(0, 1),
                    help="Stage 2c A/B arm: 0 restores the full-mesh-per-chunk "
                         "coarse paint (bitwise neutral; NOT an operating "
                         "point). The card carries coarse_subblock_chunks so "
                         "the knob proves it applied")
    ap.add_argument("--tile-workers", type=int, default=1,
                    help="run the POOL executor with this many workers (1 = "
                         "serial, unchanged). Under overlap the tile_* boundary "
                         "rows read ~0 by construction; the pooled reading is "
                         "the busy triple reported under 'pool' on the card")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args(argv)

    # WARM UP BEFORE COMPARING ARMS. Without this the traced arm always runs
    # first in each repeat and pays the JAX compilation the control then reuses,
    # so the neutrality gate reads a compile cost as instrument overhead --
    # measured at +0.66 s on a 4.98 s control at `smoke`, i.e. it FAILED for the
    # wrong reason. The host-byte probe carries the same trap and the same
    # default: the first run in a process is not a measurement of the run.
    for _ in range(int(args.warmup)):
        # the FULL K, not a shortened one: a different step count keys
        # different `cap` shapes, so a cheap warmup would prime the wrong
        # executables. It also drifts differently -- a SMALLER K drifts FURTHER
        # per step and can exhaust the arena where the real run does not, which
        # is how the first version of this line died.
        _one(args.config, args.k, args.slack, args.arena_frac, False,
             args.tile, args.buf, args.tile_workers, bool(args.paint_subblock))

    traced, control = [], []
    reports, pool_steps, sub_chunks = [], [], []
    for _ in range(int(args.repeats)):
        w, rep, seen = _one(args.config, args.k, args.slack, args.arena_frac, True,
                            args.tile, args.buf, args.tile_workers,
                            bool(args.paint_subblock))
        traced.append(w)
        reports.append(rep)
        pool_steps.extend(s["pool"] for s in seen if "pool" in s)
        sub_chunks.extend(int(s.get("coarse_subblock_chunks", -1)) for s in seen)
        w2, _, _ = _one(args.config, args.k, args.slack, args.arena_frac, False,
                        args.tile, args.buf, args.tile_workers,
                        bool(args.paint_subblock))
        control.append(w2)

    t_med, c_med = float(np.median(traced)), float(np.median(control))
    c_sd = float(np.std(control, ddof=1)) if len(control) > 1 else float("nan")

    # merge the per-phase totals across repeats by median
    names = sorted({k for r in reports for k in r["per_phase"]})
    merged = {n: float(np.median([r["per_phase"].get(n, 0.0) for r in reports])) for n in names}
    tot = sum(merged.values())

    res = dict(
        config=args.config, k=args.k, repeats=args.repeats,
        slack=args.slack, arena_frac=args.arena_frac,
        traced_wall_s=traced, control_wall_s=control,
        traced_median_s=t_med, control_median_s=c_med, control_sd_s=c_sd,
        phase_s=merged,
        phase_frac={n: (merged[n] / tot if tot else 0.0) for n in names},
        s_per_step=t_med / max(int(args.k), 1),
        unknown_phases=sorted({p for r in reports for p in r["unknown_phases"]}),
        tile_workers=int(args.tile_workers),
        paint_subblock=bool(args.paint_subblock),
        # the knob's own receipt: >0 sub-block chunks per step when on, 0 when
        # the full-mesh arm ran (a knob must prove it applied)
        coarse_subblock_chunks_per_step=sorted(set(sub_chunks)),
    )
    if pool_steps:
        # the pooled reading: per-step busy triple, medians over every traced
        # step. Boundary rows above stay on the card but read ~0 for the tile
        # phases -- overlapped work is invisible to a boundary hook.
        res["pool"] = dict(
            wall_s_median=float(np.median([p["wall_s"] for p in pool_steps])),
            busy_s_median={n: float(np.median([p["busy_s"][n] for p in pool_steps]))
                           for n in ("decode", "short", "long", "quant")},
            busy_total_s_median=float(np.median([p["busy_total_s"] for p in pool_steps])),
            concurrency_median=float(np.median([p["concurrency"] for p in pool_steps])),
            idle_s_median=float(np.median([p["idle_s"] for p in pool_steps])),
            rss_mb_max=float(max((max(p["rss_mb"].values()) for p in pool_steps
                                  if p["rss_mb"]), default=-1.0)),
            workers=int(pool_steps[0]["workers"]),
        )
    # GATE: the instrument must not MATERIALLY move the wall it reports.
    #
    # Two sigma alone does not work here, and finding that out is why the bound
    # has two terms. Caching the ICs made the arms so reproducible that sigma
    # fell to ~10 ms, so a 2-sigma band is ~20 ms -- below timer noise, and a
    # band that tight fails on sign alone (it fired at -0.02 s, with the TRACED
    # arm faster, which is not an overhead). A gate that cannot pass on a clean
    # run is not measuring the instrument.
    #
    # So: two sigma OR 2% of the wall, whichever is larger, and both terms are
    # reported so a reader can see which one bound. The 2% is a materiality
    # floor rather than a tolerance on a measurement -- a phase breakdown that
    # attributes to a few percent is not invalidated by a 2% shift in the total,
    # and anything big enough to matter is far above it.
    res["instrument_overhead_s"] = t_med - c_med
    sigma_bound = 2.0 * c_sd if c_sd == c_sd else float("nan")
    material_bound = 0.02 * c_med
    res["gate_sigma_bound_s"] = sigma_bound
    res["gate_material_bound_s"] = material_bound
    bound = max(sigma_bound, material_bound) if sigma_bound == sigma_bound else material_bound
    res["gate_bound_used_s"] = bound
    res["gate_bound_that_bound"] = (
        "sigma" if sigma_bound == sigma_bound and sigma_bound >= material_bound else "material"
    )
    res["instrument_neutral"] = abs(res["instrument_overhead_s"]) <= bound

    print(f"{args.config} K={args.k}: traced {t_med:.2f} s, control {c_med:.2f} s "
          f"(sd {c_sd:.2f}), {res['s_per_step']:.2f} s/step")
    if pool_steps:
        p = res["pool"]
        print(f"  pool W={p['workers']}: tile-loop wall {p['wall_s_median']:.2f} s/step, "
              f"concurrency {p['concurrency_median']:.2f}, idle {p['idle_s_median']:.2f} s, "
              f"max worker RSS {p['rss_mb_max']:.0f} MB")
    print(f"  instrument neutral: {res['instrument_neutral']} "
          f"(overhead {res['instrument_overhead_s']:+.3f} s against a "
          f"{bound:.3f} s bound, set by {res['gate_bound_that_bound']})")
    for n in sorted(names, key=lambda k: -merged[k]):
        if merged[n] <= 0:
            continue
        print(f"    {n:<16} {merged[n]:8.2f} s  {100 * merged[n] / tot:5.1f}%")
    if res["unknown_phases"]:
        print(f"  UNKNOWN PHASES (engine emitted a name this script does not "
              f"know): {res['unknown_phases']}")

    try:
        res["commit"] = subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        res["commit"] = None
    res["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    path = os.path.join(REPO, "runs", "v2", f"m6_phase_time{args.out_suffix}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"  card -> {path}")
    return 0 if res["instrument_neutral"] is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
