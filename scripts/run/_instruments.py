"""Host-memory and wall instruments for the run drivers: per-phase high-water
marks (`PhaseTracer`), per-phase wall (`PhaseTimer`), peak RSS, and the base
geometry table for the small configs."""

import sys
import time

CONFIGS = {
    "smoke": dict(n_part=32, L=32.0, n_fine=64, n_coarse=16),
    "cdev8": dict(n_part=128, L=64.0, n_fine=256, n_coarse=64),
    "cdev": dict(n_part=256, L=128.0, n_fine=512, n_coarse=128),
    "cgh64": dict(n_part=512, L=256.0, n_fine=1024, n_coarse=256),
}
A_INIT, A_FINAL, SPACING, SEED = 0.1, 1.0, "log", 0
ALPHA = 1.0


def _geom(cfg, tile=None, buf=32):
    g = dict(CONFIGS[cfg])
    g["tile"] = tile or (256 if g["n_fine"] >= 512 else g["n_fine"] // 4)
    g["buf"] = min(buf, g["tile"] // 2)
    return g



TRIM_MODES = ("off", "all", "step")



# `engine.step`'s first phase, emitted unconditionally once per step
STEP_BOUNDARY_PHASE = "coarse_paint"



# The phase names `engine.step` and `engine.run` emit, in the order a step
# visits them. Listed here so the probe REFUSES a name it does not know rather
# than silently reporting a partial decomposition if the engine gains a phase.
PHASES = (
    "kernel_build", "lead_drift", "coarse_paint", "coarse_solve", "membership",
    # the device lane's compiled tile loop has ONE boundary where the host lane's
    # per-tile pipeline has four (caa2533); both lanes appear here
    "tile_loop",
    "tile_decode", "tile_short", "tile_long", "tile_reduce",
    "tile_loop_end", "reconcile", "migrate", "repack", "checkpoint",
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

    def __init__(self, trim="off", series=True):
        if trim is True:
            trim = "all"
        elif trim is False:
            trim = "off"
        if trim not in TRIM_MODES:
            raise ValueError(f"trim must be one of {TRIM_MODES}, got {trim!r}")
        self.trim = trim
        self.trim_calls = 0
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

    def _trim_if_asked(self, name=None):
        """`all` = every boundary (263 per step at cdev8); `step` = ONCE per step.

        The two are a factor of 263 apart in call count, which is why `trim` alone
        could not price a step-boundary default: job 452 measured -41% peak at
        +15% wall for the 263x schedule and that bounds the most aggressive
        possible version of the intervention, not the proposed one.

        `step` keys on `coarse_paint` because it is `engine.step`'s FIRST phase
        and is unconditional. `repack` would be the natural other end and is
        wrong: it fires only when `cfg.repack_every` divides the step index
        (`engine.py:781`), so a `repack`-keyed trim would silently become
        every-Nth-step, or never.
        """
        if self.trim == "off":
            return
        if self.trim == "step" and name != STEP_BOUNDARY_PHASE:
            return
        _malloc_trim()
        self.trim_calls += 1

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
        self._trim_if_asked(name)
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
            trim_mode=self.trim,
            trim_calls=self.trim_calls,
        )



TIMER_PHASES = (
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
        self.total = {p: 0.0 for p in TIMER_PHASES}
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



def _maxrss_bytes():
    """Peak RSS of this process. KB on Linux, BYTES on macOS (the v4d note)."""
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024



def _require_cpu():
    import jax

    jax.config.update("jax_enable_x64", True)
    if jax.devices()[0].platform != "cpu":
        raise SystemExit(
            "FATAL: non-CPU backend. ru_maxrss is HOST memory; on CUDA the meshes "
            "live in VRAM and every ratio reads 1.0 (deneb 409). Run the CPU env."
        )
    return jax

