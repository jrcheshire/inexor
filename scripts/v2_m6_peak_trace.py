"""M-v2-6 Stage 0b: WHERE the engine's peak is, by PHASE, against a measured noise floor.

Stage 0 (`v2_m6_engine_peak.py`) attributed the peak by DIFFERENCING whole-run
maxima between arms -- f64 against f32, `n_coarse` against `n_coarse // 4`. That
method cannot work at cdev and job 445 contains its own proof:

  five independent measurements of the SAME leg (cdev, K=5, f64 coarse) --
  the `step` leg 7.503 GB, the K-ladder's K=5 rung 7.280, and repeats 1/2/3 at
  7.538/7.575/7.471 -- give sigma = 115 MB over all five, 45 MB over the four
  tightest.

Against that scatter, gate 1's PREDICTED f64-minus-f32 signal is 67.6 MB (0.6
sigma) and its +-25% tolerance is +-16.9 MB, a quarter of the noise; the
isolating arm removes 178.7 MB of modelled coarse mesh (1.6 sigma) and measured
132 MB HIGHER. So neither gate could be read, in either direction, and the
2.034 ratio on record is one draw of a noisy difference rather than a finding.

The defect is structural, not a matter of tolerance: **a maximum carries no
timestamp**, so differencing two maxima assumes both were set at the same
moment by the same phase, and nothing checked that. The one number in Stage 0
that IS far outside the scatter is the unattributed residual, 4.27 GB = 254.5
B/p at 37 sigma -- and gate 2 "passing" does not identify it, because a
ONE-SIDED floor at 0.8 x 32 B/p is cleared eightfold by an unmodelled term of
any origin whatsoever.

THIS probe measures the same peak a different way. It takes a high-water mark
PER PHASE, by resetting the kernel's own `VmHWM` counter at every phase
boundary (`/proc/self/clear_refs`, value 5, Linux >= 4.0), so each phase reports
the true maximum reached while it ran rather than a sample of it. That turns
"which of my model's terms is wrong" into "which phase owns the 4.3 GB", which
is a question about the engine rather than about the model.

## Arms, and what each one separates

  trace       the phase-resolved run. The measurement.
  control     the SAME run with `phase=None` -- no boundaries, no clear_refs
              writes, `ru_maxrss` only. Its job is to prove the instrument did
              not move what it measures; if `control` and `trace` disagree on
              the run peak by more than the repeat scatter, the trace is
              measuring an instrumented engine and nothing else it says counts.
  trim        the traced run with `malloc_trim(0)` at every boundary. glibc
              does not return freed heap to the OS by default and the step
              churns ~1 GB of numpy per tile at eight tiles, so an unknown part
              of the peak is RETAINED-FREE rather than live. Trimming at each
              boundary is what separates the two. An INSTRUMENT ARM, never an
              operating point (the `cap_mult` precedent): a production run
              would pay the syscall.

## What is REPORTED and what is GATED

Reported: per phase, the peak RSS reached during it and its increment over the
RSS at its own start, each as a median over repeats with the spread beside it.
Every phase number is quoted against the run-peak scatter measured in the SAME
job, so a difference smaller than the noise is visibly smaller than the noise.

GATED, and both gates are about the INSTRUMENT rather than the engine:

  GATE A (the instrument is neutral): |median(trace peak) - median(control
  peak)| <= 2 x sigma(control peak). Not a picked tolerance -- sigma is measured
  here, from the repeats, and 2 sigma is the weakest statement that the two arms
  are the same measurement.

  GATE B (the phases account for the run): max over phases of the phase peak is
  within 2 sigma of the CONTROL arm's run peak -- an independent, uninstrumented
  measurement. Never against the traced run peak, which is the max over those
  same phase peaks by construction and would make this gate return zero for any
  engine and any defect. A phase decomposition whose largest phase
  falls short of the whole run has a peak living in unnamed code, and the
  attribution below it would be an attribution of the wrong thing. In job 446
  it failed NEGATIVE -- the top phase EXCEEDED the run peak, which is physically
  impossible -- and that is what exposed `ru_maxrss` being reset by clear_refs
  along with `VmHWM`. A gate on an impossibility is worth having.

Neither gate can pass vacuously: A fails if the instrument perturbs, B fails if
the boundaries miss the peak, and both are two-sided against a sigma measured
in the same job rather than against a number anyone chose.

Usage:
  pixi run python scripts/v2_m6_peak_trace.py --configs cdev8 --k 5 --repeats 3
  pixi run python scripts/v2_m6_peak_trace.py --worker cdev8:trace:5:/tmp/wd  # internal
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO, "runs", "v2")

sys.path.insert(0, HERE)
import v2_m3_engine_gate as m3  # noqa: E402
import v2_m6_engine_peak as p0  # noqa: E402

CONFIGS = m3.CONFIGS
ARMS = ("trace", "control", "trim")
GATE_SIGMA = 2.0

# The phase names `engine.step` and `engine.run` emit, in the order a step
# visits them. Listed here so the probe REFUSES a name it does not know rather
# than silently reporting a partial decomposition if the engine gains a phase.
PHASES = (
    "lead_drift", "coarse_paint", "coarse_solve", "membership",
    "tile_decode", "tile_short", "tile_long", "tile_reduce",
    "tile_loop_end", "reconcile", "migrate", "repack",
)


def _require_linux():
    """`VmHWM` and `clear_refs` are procfs. There is no macOS equivalent.

    This is not a portability gap worth papering over: the umbrella record
    already has macOS peaks reading ~3x low and one laptop point moving 6.484
    -> 9.855 GB minutes apart, so a Darwin fallback would produce numbers that
    look like measurements and are not.
    """
    if sys.platform != "linux":
        raise SystemExit(
            f"FATAL: {sys.platform} has no /proc/self/clear_refs, so a per-phase "
            "high-water mark cannot be taken. Run this on antares/deneb."
        )


def _status_kb(field):
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith(field):
                return int(line.split()[1])
    raise RuntimeError(f"{field} absent from /proc/self/status")


def _hwm():
    return _status_kb("VmHWM:") * 1024


def _rss():
    return _status_kb("VmRSS:") * 1024


def _reset_hwm():
    """Reset VmHWM to the current VmRSS (Linux >= 4.0, clear_refs value 5)."""
    with open("/proc/self/clear_refs", "w") as fh:
        fh.write("5\n")


def _malloc_trim():
    import ctypes

    ctypes.CDLL("libc.so.6").malloc_trim(0)


def phase_growth(series):
    """Per phase, the trend in its OWN increment: last visit's delta minus first.

    **This was wrong in job 446 and the correction is the useful part.** It read
    the trend in each phase's ABSOLUTE peak, and every phase's absolute peak
    rises simply because the process's resident set ratchets upward through the
    run -- so it reported +1233 to +1308 MB for `membership`, `coarse_solve`,
    `coarse_paint` and `tile_decode` alike, which is one process-wide climb
    restated twelve times, not an attribution. A phase that is itself
    accumulating allocates MORE each visit, so the trend has to be read on the
    increment, which is invariant to what everyone else has left resident.

    `step_ladder` carries the process-wide climb, once, where it belongs.
    """
    if not series:
        return {}
    first, last, visits = {}, {}, {}
    for name, _peak, delta in series:
        if name not in first:
            first[name] = delta
        last[name] = delta
        visits[name] = visits.get(name, 0) + 1
    return {
        n: dict(first_visit_delta=first[n], last_visit_delta=last[n],
                growth=last[n] - first[n], visits=visits[n])
        for n in first
    }


def step_ladder(series, split="coarse_paint"):
    """Per STEP: the resident set it started from, and the maximum it reached.

    The process-wide climb, reported once. A working set is K-independent, so a
    ladder that keeps rising means something accumulates; one that flattens
    means the run was warming an allocator up. That distinction decides whether
    job 445's cdev fit (0.157 GB PER STEP + 6.200 fixed, from K=5 and K=10
    alone) may be extrapolated to a production K at all, and nothing measured
    it -- two rungs cannot tell a slope from the start of a curve.
    """
    steps, cur = [], None
    for rec in series or []:
        if rec[0] == split:
            if cur is not None:
                steps.append(cur)
            cur = []
        if cur is not None:
            cur.append(rec)
    if cur:
        steps.append(cur)
    return [dict(start=s[0][1] - s[0][2], peak=max(r[1] for r in s), boundaries=len(s))
            for s in steps]


class PhaseTracer:
    """Per-phase high-water marks. One instance per run; call it at boundaries.

    Each call closes the phase named by its argument: it reads `VmHWM` (the
    true maximum since the last reset, not a sample), attributes it to that
    phase, then resets the counter and records the RSS the next phase starts
    from. A phase visited many times per run -- every `tile_*` name is visited
    once per tile per step -- keeps the MAX over visits, because the question
    is what the peak was, not what it usually was.

    Two numbers per phase, and they answer different questions. `peak` is the
    absolute RSS reached while the phase ran, which is what a host ceiling
    cares about. `delta` is that minus the RSS the phase started from, which is
    what the phase itself allocated. A phase can have a large `peak` and a zero
    `delta` -- that is a phase running while someone ELSE's memory is resident,
    and it is exactly the distinction differencing two maxima cannot make.

    A per-visit SERIES is kept beside the maxima, because a max over visits
    cannot show a trend and the question that needs one is open: job 445's cdev
    K-ladder fit 0.157 GB PER STEP + 6.200 GB fixed, so something in the step
    accumulates, and `coarse_delta_streamed` takes a new `pad` -- hence a new
    XLA shape -- on every step (measured at the smoke config: ten distinct
    shapes over ten steps, against one for `cap` since the Stage 0 ladder).
    Whether that churn is what grows the peak is what the series answers: if it
    is, `coarse_paint`'s per-visit peak rises step over step and no other
    phase's does.
    """

    def __init__(self, trim=False, series=True):
        self.trim = trim
        self.phases = {}
        self.order = []
        self.unknown = []
        self.series = [] if series else None
        # THE RUN PEAK, and it has to be accumulated here rather than read at the
        # end. `clear_refs` resets `mm->hiwater_rss`, and BOTH `VmHWM` and
        # getrusage's `ru_maxrss` read that same field -- so after a traced run
        # `ru_maxrss` reports the peak since the LAST boundary, not the run's.
        # Job 446 shipped that mistake: it read cdev8's traced peak as 1.824 GB
        # against 2.010 actual, i.e. 0.19 GB BELOW an untraced control, which
        # looked like the instrument lowering the peak and was the instrument
        # mismeasuring it. Each boundary's reading is the max since the previous
        # reset, so the max over boundaries is exactly the run's high-water and
        # costs no extra syscall.
        self.run_peak = 0
        self._trim_if_asked()
        _reset_hwm()
        self._start = _rss()

    def _trim_if_asked(self):
        if self.trim:
            _malloc_trim()

    def __call__(self, name):
        hwm, rss = _hwm(), _rss()
        if name not in PHASES:
            self.unknown.append(name)
        e = self.phases.get(name)
        if e is None:
            e = self.phases[name] = dict(peak=0, delta=0, visits=0, rss_end=0)
            self.order.append(name)
        e["peak"] = max(e["peak"], hwm)
        e["delta"] = max(e["delta"], hwm - self._start)
        e["rss_end"] = max(e["rss_end"], rss)
        e["visits"] += 1
        self.run_peak = max(self.run_peak, hwm)
        if self.series is not None:
            self.series.append((name, hwm, hwm - self._start))
        self._trim_if_asked()
        _reset_hwm()
        self._start = _rss()

    def report(self):
        return dict(
            phases={k: dict(v) for k, v in self.phases.items()},
            order=list(self.order),
            unknown_phases=sorted(set(self.unknown)),
            series=self.series,
            growth=phase_growth(self.series),
            step_ladder=step_ladder(self.series),
            run_peak=self.run_peak,
        )


def _worker(cfg, arm, k_steps, workdir, slack, arena_frac, alloc_margin, pad_ladder=True):
    _require_linux()
    jax = p0._require_cpu()
    g = m3._geom(cfg)

    from inexor.config import Cosmology

    import inexor.ic as ic

    key = jax.random.PRNGKey(p0.SEED)
    ic.white_plane(key, 0, 8, p0.GEN_FDTYPE)  # backend init, as in Stage 0

    from inexor import icgen
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

    cosmo = Cosmology()
    ec = p0._engine_config(g, "float64", 1, 1, slack)
    # the A/B knob (M-v2-6 Stage 0b). OFF restores the pre-fix behaviour of the
    # coarse chunk buffer -- a new XLA shape every step -- with `cap` left on
    # its ladder in both arms, so the slope difference is attributable to one
    # shape family. `coarse_pad_distinct` on the card is what proves it applied.
    ec.pad_ladder = bool(pad_ladder)
    ec.validate()
    st = icgen.load_slot_state(
        workdir, brick_slack=slack, alloc_margin=alloc_margin, arena_frac=arena_frac
    )
    out = dict(
        arm=arm, n_coarse=int(ec.n_coarse), mesh_bytes=ec.mesh_bytes(),
        n_tiles=len(ec.tiles), state_bytes=p0._state_array_bytes(st),
        n_particles=int(st.n_particles), n_rows=int(st.off.shape[0]),
        rss_after_load=_rss(), maxrss_after_load=_hwm(),
    )

    n_steps = int(k_steps)
    a_steps = a_grid(m3.A_INIT, m3.A_FINAL, n_steps, m3.SPACING)
    co = bullfrog_float_coeffs(bullfrog_table(a_steps, cosmo))

    from inexor import engine

    import time

    seen = []
    tracer = PhaseTracer(trim=(arm == "trim")) if arm != "control" else None
    t0 = time.perf_counter()
    engine.run(st, ec, co, collect=seen.append, phase=tracer)
    out["wall_s"] = time.perf_counter() - t0
    out["s_per_step"] = out["wall_s"] / n_steps
    out["k_steps"] = n_steps
    # `ru_maxrss` for the WHOLE process, so the traced and untraced arms are
    # compared on the identical statistic Stage 0 used. VmHWM is useless for
    # this at the end of a traced run: the tracer has been resetting it.
    out["maxrss_raw"] = p0._maxrss_bytes()
    out["rss_end"] = _rss()
    # the CONTROL arm never resets, so its ru_maxrss IS the run peak; a traced
    # arm's is not (see `PhaseTracer.run_peak`). One field, `run_peak`, is what
    # the arms are compared on, and the raw reading stays beside it so the
    # reset is visible on the card rather than argued from a docstring.
    out["run_peak"] = out["maxrss_raw"] if tracer is None else tracer.run_peak
    out["hiwater_was_reset"] = bool(tracer is not None
                                    and out["maxrss_raw"] < tracer.run_peak)
    out["cap"] = int(seen[-1]["cap"]) if seen else None
    out["cap_distinct"] = len({int(s["cap"]) for s in seen})
    # BOTH shape families, because an A/B that cannot show its knob moved is
    # not an A/B: `pad_ladder` off must give one distinct pad per step and
    # `cap_distinct` must be unchanged between the arms.
    out["pad_ladder"] = bool(ec.pad_ladder)
    out["coarse_pad"] = int(seen[-1]["coarse_pad"]) if seen else None
    out["coarse_pad_distinct"] = len({int(s["coarse_pad"]) for s in seen})
    out["coarse_pad_true_distinct"] = len({int(s["coarse_pad_true"]) for s in seen})
    if tracer is not None:
        out.update(tracer.report())
    st.check()
    print(json.dumps(out), flush=True)


def _spawn(cfg, arm, k_steps, workdir, knobs):
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--worker", f"{cfg}:{arm}:{k_steps}:{workdir}",
        "--slack", str(knobs["slack"]), "--arena-frac", str(knobs["arena_frac"]),
        "--alloc-margin", str(knobs["alloc_margin"]),
    ]
    if not knobs.get("pad_ladder", True):
        cmd.append("--no-pad-ladder")
    out = subprocess.check_output(cmd, text=True, cwd=REPO)
    return json.loads(out.strip().splitlines()[-1])


def _stats(xs):
    """Median, sigma and spread. sigma over fewer than 3 points is not reported.

    A sigma from two points is a number, not an estimate, and quoting one would
    reintroduce exactly the false precision this probe exists to remove.
    """
    xs = sorted(xs)
    return dict(
        n=len(xs), median=statistics.median(xs), min=xs[0], max=xs[-1],
        spread=xs[-1] - xs[0],
        sigma=(statistics.stdev(xs) if len(xs) >= 3 else None),
    )


def _aggregate(runs):
    """Per-arm run peaks and per-phase peaks, each with its own scatter."""
    agg = {}
    for arm, rs in runs.items():
        if not rs:
            continue
        a = dict(
            run_peak=_stats([r["run_peak"] for r in rs]),
            run_peak_raw=_stats([r["maxrss_raw"] for r in rs]),
            wall_s=_stats([r["wall_s"] for r in rs]),
            repeats=len(rs),
        )
        if "phases" in rs[0]:
            names = [n for n in PHASES if n in rs[0]["phases"]]
            a["phases"] = {
                n: dict(
                    peak=_stats([r["phases"][n]["peak"] for r in rs]),
                    delta=_stats([r["phases"][n]["delta"] for r in rs]),
                    visits=rs[0]["phases"][n]["visits"],
                )
                for n in names
            }
            a["growth"] = {
                n: _stats([r["growth"][n]["growth"] for r in rs])
                for n in names
                if all(n in r.get("growth", {}) for r in rs)
            }
            nl = min(len(r.get("step_ladder", [])) for r in rs)
            a["step_ladder"] = [
                dict(start=_stats([r["step_ladder"][i]["start"] for r in rs]),
                     peak=_stats([r["step_ladder"][i]["peak"] for r in rs]))
                for i in range(nl)
            ]
            a["unknown_phases"] = sorted({p for r in rs for p in r["unknown_phases"]})
        agg[arm] = a
    return agg


def _verdict(agg):
    """Gates A and B, both against a sigma measured in this job."""
    v = dict(gate_a_instrument_neutral=None, gate_b_phases_reach_the_peak=None)
    tr, ct = agg.get("trace"), agg.get("control")
    if tr and ct and ct["run_peak"]["sigma"]:
        d = abs(tr["run_peak"]["median"] - ct["run_peak"]["median"])
        v["instrument_delta_bytes"] = d
        v["instrument_delta_sigma"] = d / ct["run_peak"]["sigma"]
        v["gate_a_instrument_neutral"] = bool(d <= GATE_SIGMA * ct["run_peak"]["sigma"])
    if tr and "phases" in tr:
        top = max(tr["phases"].items(), key=lambda kv: kv[1]["peak"]["median"])
        v["top_phase"] = top[0]
        v["top_phase_peak"] = top[1]["peak"]["median"]
        # AGAINST THE CONTROL ARM, and that is the whole point. The traced run
        # peak is now the max over boundary readings, i.e. the max over phase
        # peaks BY CONSTRUCTION -- so comparing the top phase against it would
        # compare a number with itself and yield exactly zero for any engine, any
        # config, any defect. Fixing the run-peak statistic turned this gate
        # vacuous, and a gate that cannot fail is worse than no gate. The
        # independent reference is `control`: an uninstrumented process whose
        # `ru_maxrss` nothing reset.
        if ct and ct["run_peak"]["sigma"]:
            short = ct["run_peak"]["median"] - top[1]["peak"]["median"]
            v["control_peak_minus_top_phase"] = short
            v["control_peak_minus_top_phase_sigma"] = short / ct["run_peak"]["sigma"]
            v["gate_b_phases_reach_the_peak"] = bool(
                abs(short) <= GATE_SIGMA * ct["run_peak"]["sigma"]
            )
        else:
            v["gate_b_note"] = (
                "no control arm with a sigma, so the phase decomposition has no "
                "independent reference and gate B is not evaluable"
            )
    return v


def _print(cfg, agg, verdict, unknown):
    gb = 1e9
    print(f"\n=== {cfg} ===", flush=True)
    for arm in ARMS:
        a = agg.get(arm)
        if not a:
            continue
        rp = a["run_peak"]
        sig = f", sigma {rp['sigma'] / gb:.3f}" if rp["sigma"] else ""
        print(f"[{cfg}] {arm:8s} run peak median {rp['median'] / gb:.3f} GB "
              f"(spread {rp['spread'] / gb:.3f}{sig}, n={rp['n']})", flush=True)
    tr = agg.get("trace")
    if tr and "phases" in tr:
        s = tr["run_peak"]["sigma"] or float("nan")
        print(f"[{cfg}] phases, peak and own increment, in units of the "
              f"run-peak sigma ({s / 1e6:.0f} MB):", flush=True)
        rows = sorted(tr["phases"].items(), key=lambda kv: -kv[1]["peak"]["median"])
        grow = tr.get("growth", {})
        for name, p in rows:
            g = grow.get(name, {}).get("median")
            gtxt = f"   growth {g / 1e6:+8.0f} MB" if g is not None else ""
            print(f"    {name:14s} peak {p['peak']['median'] / gb:7.3f} GB   "
                  f"delta {p['delta']['median'] / gb:7.3f} GB "
                  f"= {p['delta']['median'] / s:6.1f} sigma   "
                  f"visits {p['visits']}{gtxt}", flush=True)
        top = max(grow.items(), key=lambda kv: kv[1]["median"], default=None)
        if top and top[1]["median"] > 0:
            print(f"[{cfg}] the peak GROWS most in `{top[0]}`: "
                  f"{top[1]['median'] / 1e6:+.0f} MB first visit to last "
                  f"(a working set is K-independent; a slope accumulates)", flush=True)
    lad = agg.get("trace", {}).get("step_ladder")
    if lad:
        print(f"[{cfg}] per-step ladder (start -> peak, GB), median over repeats:",
              flush=True)
        print("    " + "  ".join(f"{d['start']['median'] / gb:.2f}->"
                                 f"{d['peak']['median'] / gb:.2f}" for d in lad),
              flush=True)
        rise = lad[-1]["peak"]["median"] - lad[0]["peak"]["median"]
        print(f"    step peak rises {rise / 1e6:+.0f} MB over {len(lad)} steps "
              f"({rise / 1e6 / max(1, len(lad) - 1):+.0f} MB/step); a working set "
              f"is K-independent", flush=True)
    for k in ("gate_a_instrument_neutral", "gate_b_phases_reach_the_peak"):
        print(f"[{cfg}] {k}: {verdict.get(k)}", flush=True)
    if verdict.get("top_phase"):
        print(f"[{cfg}] the peak is set in `{verdict['top_phase']}`; the run peak "
              f"exceeds it by {verdict['run_peak_minus_top_phase'] / 1e6:.0f} MB "
              f"({verdict['run_peak_minus_top_phase_sigma']:.1f} sigma)", flush=True)
    if unknown:
        print(f"[{cfg}] UNKNOWN phase names emitted: {unknown}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["cdev8"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3,
                    help="per arm; 3 is the minimum for a sigma, 5 is better")
    ap.add_argument("--arms", nargs="+", default=["trace", "control"], choices=ARMS)
    ap.add_argument("--workdir-root", default=None)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--slack", type=float, default=0.20)
    ap.add_argument("--arena-frac", type=float, default=0.08)
    ap.add_argument("--alloc-margin", type=float, default=0.10)
    ap.add_argument("--no-pad-ladder", action="store_true",
                    help="restore the pre-fix per-step chunk shape (the A arm of "
                         "the M-v2-6 Stage 0b A/B); never an operating point")
    ap.add_argument("--worker", default=None)
    a = ap.parse_args()

    knobs = dict(slack=a.slack, arena_frac=a.arena_frac, alloc_margin=a.alloc_margin,
                 pad_ladder=not a.no_pad_ladder)
    if a.worker:
        cfg, arm, k, wd = a.worker.split(":", 3)
        _worker(cfg, arm, int(k), wd, **knobs)
        return

    _require_linux()
    root = a.workdir_root or tempfile.mkdtemp(prefix="m6_trace_")
    os.makedirs(OUT_DIR, exist_ok=True)
    res = dict(k_steps=a.k, repeats=a.repeats, arms=a.arms, knobs=knobs,
               workdir_root=root)
    ok = True
    for cfg in a.configs:
        wd = os.path.join(root, cfg)
        os.makedirs(wd, exist_ok=True)
        # the ICs are generated ONCE, in their own process, and reach every arm
        # through disk -- the generator's 70-90 B/p peak must not land in any
        # arm's high-water mark (Stage 0's first instrument defect)
        if not os.path.exists(os.path.join(wd, p0.icgen_manifest())):
            print(f"[{cfg}] generating ICs -> {wd}", flush=True)
            p0._run(cfg, "gen", 0, wd, knobs)
        # the same baseline leg Stage 0 nets against, so absolute numbers here
        # can be read beside its nets rather than only against each other
        base = p0._run(cfg, "baseline", 0, wd, knobs)["maxrss"]
        print(f"[{cfg}] baseline {base / 1e9:.3f} GB  (workdir {wd})", flush=True)
        runs = {}
        for arm in a.arms:
            # printed AS THEY LAND, not aggregated at the end. Job 446 ran the
            # cdev8 leg for 17 minutes emitting nothing, so a hung run and a slow
            # one look identical from the log -- and the log is all a cluster job
            # gives you. The cost is one line per run.
            rs = []
            for i in range(a.repeats):
                d = _spawn(cfg, arm, a.k, wd, knobs)
                rs.append(d)
                print(f"[{cfg}] {arm} {i + 1}/{a.repeats}: peak "
                      f"{d['maxrss'] / 1e9:.3f} GB, {d['s_per_step']:.2f} s/step",
                      flush=True)
            runs[arm] = rs
        agg = _aggregate(runs)
        verdict = _verdict(agg)
        unknown = sorted({p for r in runs.get("trace", []) for p in r["unknown_phases"]})
        res[cfg] = dict(aggregate=agg, verdict=verdict, runs=runs,
                        baseline_bytes=base, unknown_phases=unknown)
        knobs.setdefault("n_part", CONFIGS[cfg]["n_part"])
        _print(cfg, agg, verdict, unknown)
        ok = ok and verdict.get("gate_a_instrument_neutral") is not False
        ok = ok and verdict.get("gate_b_phases_reach_the_peak") is not False
        ok = ok and not unknown
    res["ok"] = ok
    p0._write(res, a.out_suffix or "_trace", knobs)
    print("OK" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
