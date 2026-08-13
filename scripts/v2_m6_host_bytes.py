"""The engine's HOST allocations, measured exactly and deterministically, on any machine.

**Why this exists.** Every memory number in M-v2-6 so far is a peak RSS, which
costs a cluster job (procfs; macOS reads ~3x low), needs three repeats for a
sigma, and still leaves 12-41% of itself as glibc retention that has to be A/B'd
out with `malloc_trim`. That instrument answered "how many bytes did this process
hold", when the question the capacity claim actually needs is "how many bytes did
the engine ALLOCATE" -- and the second one is exact.

`tracemalloc` gives it, with one property the RSS probe could never have: the
high-water is tracked CONTINUOUSLY by the runtime, so a transient allocated and
freed between two readings is still counted. Stage 0's structural defect -- "a
maximum carries no timestamp", so differencing two sampled maxima assumes both
were set at the same moment -- cannot occur here.

**What it sees and what it does not**, measured before this was written:

  numpy array data      EXACT. numpy routes data allocations through
                        PyTraceMalloc_Track under `np.lib.tracemalloc_domain`.
                        Verified: a 64 MiB f64 array reads 64.00 MiB, a view
                        adds nothing, a copy adds exactly one more.
  jax Arrays            invisible here (a 64 MiB CPU jax array moves this by
                        3.2 MiB of Python wrapper). `jax.live_arrays()` sees them
                        but only as a SNAPSHOT.
  XLA intra-jit scratch INVISIBLE TO EVERYTHING in-process. A jit whose temporary
                        is 256 MiB moves tracemalloc by 3.3 MiB and
                        `live_arrays` by 0; `memory_stats()` is None on CPU.

That blindness is affordable for exactly one reason, and it is measured rather
than assumed: **the terms that bind at C-gh are all numpy.** `kick_pending`
(274.9 GB) is built by `v_new = alpha_k * v[owned] + bcoef * g_tot` after
`np.asarray` at the jax boundary; `SlotState` and the repack scratch (93.5 GB)
are `np.zeros`/`np.empty`. The invisible half is the tile force, whose transient
is config-INVARIANT and K-INVARIANT (cdev, cgh64 and C-gh share N/tile = 2,097,152
and P = 320; job 452 showed no K dependence), so it is the one term that does not
need re-measuring at scale.

**The gate this instrument can pass and the RSS probe could not: an EXACT peak.**
The numpy high-water is bit-identical across runs -- 142,654,851 bytes at cdev8
K=3, four runs, cycle collector on and off -- because an array's size is fixed by
its shape. A peak RSS never could: its run-to-run sigma is 44 MB at cdev8 and 282
MB at cdev. Two things are deliberately NOT gated, each for a measured reason:
individual boundary readings, which differ by exactly 4 bytes at `tile_long` (one
scalar straddling the read, unchanged by disabling gc); and the all-domain peak,
which Python bookkeeping scatters ~0.1%. Both are reported with their spread
attached, so nothing needs a picked tolerance.

Usage (laptop, seconds; no cluster, no procfs, no sigma):
  pixi run python scripts/v2_m6_host_bytes.py --config cdev8 --k 5
  pixi run python scripts/v2_m6_host_bytes.py --config cdev8 --k 5 --vs-model
"""

import argparse
import json
import os
import sys
import tracemalloc

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import v2_m3_engine_gate as m3  # noqa: E402
import v2_m6_engine_peak as p0  # noqa: E402
from v2_m6_peak_trace import PHASES  # noqa: E402

MB = 1e6
NUMPY_DOMAIN = np.lib.tracemalloc_domain


class HostByteTracer:
    """Per-phase host allocation high-water, from tracemalloc's own counters.

    Mirrors `v2_m6_peak_trace.PhaseTracer`'s shape so the two are read the same
    way, but the readings come from the allocator's own counters rather than
    from procfs, so there is no syscall and no sampling of a maximum.

    TWO counters, because neither alone does the job and the pair is what makes
    the instrument honest:

      all-domain `get_traced_memory()` -- gives a CONTINUOUS high-water, so a
        transient between boundaries is still counted. But it also counts Python
        bookkeeping, which is NOT reproducible: warm runs of the smoke config
        scatter ~12 kB (0.3%) in this counter.
      numpy-domain snapshot -- EXACT and reproducible, because a numpy array's
        size is fixed by its shape. But it is a snapshot at the moment asked, so
        it misses anything allocated and freed between two boundaries.

    The gate runs on the numpy-domain PEAK, which is exact. It deliberately does
    NOT run on the individual boundary readings: those differ by exactly 4 bytes
    at `tile_long` between otherwise identical runs (one scalar array straddling
    the read; disabling the cycle collector doubles the count of affected
    boundaries and changes no magnitude), so gating them would fail forever on a
    non-term. Filtering costs 0.058 ms per boundary, measured -- 0.6 s over a full
    cdev8 run, which is why it can be afforded at every boundary rather than
    sampled.
    """

    def __init__(self):
        self.phases = {}
        self.order = []
        self.unknown = []
        self.series = []
        self.np_series = []
        self.run_peak = 0
        self.np_peak = 0
        tracemalloc.reset_peak()
        self._start = tracemalloc.get_traced_memory()[0]

    @staticmethod
    def _numpy_bytes():
        snap = tracemalloc.take_snapshot().filter_traces(
            [tracemalloc.DomainFilter(True, NUMPY_DOMAIN)]
        )
        return sum(s.size for s in snap.statistics("filename"))

    def __call__(self, name):
        cur, peak = tracemalloc.get_traced_memory()
        npb = self._numpy_bytes()
        if name not in PHASES:
            self.unknown.append(name)
        e = self.phases.get(name)
        if e is None:
            e = self.phases[name] = dict(peak=0, delta=0, visits=0, np_resident=0)
            self.order.append(name)
        e["peak"] = max(e["peak"], peak)
        e["delta"] = max(e["delta"], peak - self._start)
        e["np_resident"] = max(e["np_resident"], npb)
        e["visits"] += 1
        self.run_peak = max(self.run_peak, peak)
        self.np_peak = max(self.np_peak, npb)
        self.series.append((name, peak, peak - self._start))
        self.np_series.append(npb)
        tracemalloc.reset_peak()
        self._start = cur

    def report(self):
        return dict(
            phases={k: dict(v) for k, v in self.phases.items()},
            order=list(self.order),
            unknown_phases=sorted(set(self.unknown)),
            run_peak=self.run_peak,
            np_peak=self.np_peak,
            series=self.series,
            np_series=self.np_series,
        )


def _ensure_ics(cfg, wd, slack, arena_frac, alloc_margin):
    """Generate the T9 slabs if the workdir is empty, mirroring the `gen` leg."""
    jax = p0._require_cpu()
    g = m3._geom(cfg)
    from inexor import icgen
    from inexor.config import Cosmology

    ec = p0._engine_config(g, "float64", 1, 1, slack)
    key = jax.random.PRNGKey(p0.SEED)
    icgen.generate_t9_slabs(
        wd, key, g["n_part"], g["L"], Cosmology(), m3.A_INIT,
        g["n_fine"] // ec.n_brick, fdtype=p0.GEN_FDTYPE, slab=32,
    )


def run_once(cfg, k_steps, workdir, slack=0.2, arena_frac=0.08, alloc_margin=0.1):
    """One run. Returns the tracer report plus the exact state term."""
    jax = p0._require_cpu()
    g = m3._geom(cfg)

    from inexor import icgen
    from inexor.config import Cosmology
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

    import inexor.ic as ic

    key = jax.random.PRNGKey(p0.SEED)
    ic.white_plane(key, 0, 8, p0.GEN_FDTYPE)  # backend init, as the RSS probe does

    cosmo = Cosmology()
    ec = p0._engine_config(g, "float64", 1, 1, slack)
    ec.validate()
    st = icgen.load_slot_state(
        workdir, brick_slack=slack, alloc_margin=alloc_margin, arena_frac=arena_frac
    )
    a_steps = a_grid(m3.A_INIT, m3.A_FINAL, int(k_steps), m3.SPACING)
    co = bullfrog_float_coeffs(bullfrog_table(a_steps, cosmo))

    from inexor import engine

    # START TRACING HERE, after the state is loaded: the state is measured
    # exactly by `state_bytes` and tracing it would double-count it into the
    # per-step figure, which is what the planner compares against.
    tracemalloc.start(1)  # nframe=1: we want counters, not tracebacks
    try:
        tracer = HostByteTracer()
        seen = []
        engine.run(st, ec, co, collect=seen.append, phase=tracer)
        rep = tracer.report()
    finally:
        tracemalloc.stop()
    rep["state_bytes"] = p0._state_array_bytes(st)
    rep["n_particles"] = int(st.n_particles)
    rep["n_rows"] = int(st.off.shape[0])
    rep["cap"] = int(seen[-1]["cap"]) if seen else None
    rep["model"] = ec.step_bytes(int(st.n_particles), n_rows=int(st.off.shape[0]))
    return rep


def compare_repeats(reps):
    """THE GATE: the numpy PEAK must be bit-identical across runs.

    Measured, at cdev8 K=3 over four runs and with the cycle collector both on
    and off: `np_peak` is identical to the byte (142,654,851) every time, while
    79-184 of 790 individual boundary readings differ -- by exactly **4 bytes**,
    at `tile_long`. That is one scalar array whose construction straddles the
    boundary read, not a term. Disabling gc doubles the count and changes no
    magnitude, which is what rules out collector timing as anything that matters.

    So the gate is on the peak, which is the number quoted and is exact, and the
    series deviation is REPORTED rather than gated. That ordering matters: gating
    the series would fail on a 4-byte scalar forever, and the fix would have been
    a picked tolerance -- which is the move this project has had to retract more
    than once. The all-domain peak is not gated at all: Python bookkeeping
    scatters it ~0.1%, and it is quoted with that spread attached.
    """
    if len(reps) < 2:
        return None, "one run: determinism not evaluable"
    a = reps[0]
    peaks = [r["run_peak"] for r in reps]
    spread = (max(peaks) - min(peaks)) / max(1, min(peaks))
    worst, ndiff = 0, 0
    for i, b in enumerate(reps[1:], start=2):
        if a["np_peak"] != b["np_peak"]:
            return False, (f"runs 1 and {i} disagree on the numpy PEAK: "
                           f"{a['np_peak']:,} vs {b['np_peak']:,} bytes "
                           f"({abs(a['np_peak'] - b['np_peak']) / MB:.3f} MB) -- the "
                           "engine allocates differently run to run")
        for x, y in zip(a["np_series"], b["np_series"]):
            if x != y:
                ndiff += 1
                worst = max(worst, abs(x - y))
    return True, (f"{len(reps)} runs agree on the numpy peak to the BYTE "
                  f"({a['np_peak']:,}); {ndiff} boundary readings differ by at most "
                  f"{worst} B (reported, not gated); all-domain peak spread "
                  f"{spread * 100:.2f}% = Python bookkeeping")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="cdev8", choices=sorted(m3.CONFIGS))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2,
                    help="2 is the determinism gate; more only re-proves it")
    ap.add_argument("--warmup", type=int, default=1,
                    help="runs to DISCARD first. The first run in a process "
                         "compiles, and compilation allocates: measured 2.09x "
                         "the warm peak at smoke (8.79 vs 4.22 MB). Never 0 "
                         "unless you want the compiler in your budget.")
    ap.add_argument("--slack", type=float, default=0.2)
    ap.add_argument("--arena-frac", type=float, default=0.08)
    ap.add_argument("--alloc-margin", type=float, default=0.1)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--vs-model", action="store_true",
                    help="compare against EngineConfig.step_bytes term by term")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    import tempfile

    wd = a.workdir or os.path.join(tempfile.gettempdir(), f"m6_hostbytes_{a.config}")
    if not os.path.isdir(wd):
        os.makedirs(wd, exist_ok=True)
        _ensure_ics(a.config, wd, a.slack, a.arena_frac, a.alloc_margin)

    for _ in range(a.warmup):  # compile, then throw the numbers away
        run_once(a.config, a.k, wd, a.slack, a.arena_frac, a.alloc_margin)
    reps = [run_once(a.config, a.k, wd, a.slack, a.arena_frac, a.alloc_margin)
            for _ in range(a.repeats)]
    r = reps[0]
    ok, msg = compare_repeats(reps)

    print(f"\n=== {a.config}, K={a.k}, {a.repeats} runs "
          f"({r['n_particles']:,} particles, {r['n_rows']:,} rows) ===")
    print(f"  DETERMINISM: {msg}")
    if r["unknown_phases"]:
        print(f"  UNKNOWN PHASES (decomposition incomplete): {r['unknown_phases']}")
    sb = r["state_bytes"]["total"]
    print(f"\n  state (exact, from array nbytes)   {sb / MB:10.2f} MB"
          f"   {sb / r['n_particles']:6.2f} B/p")
    print(f"  per-step host high-water            {r['run_peak'] / MB:10.2f} MB"
          f"   {r['run_peak'] / r['n_particles']:6.2f} B/p")
    print(f"  of which numpy arrays (exact)       {r['np_peak'] / MB:10.2f} MB"
          f"   {r['np_peak'] / r['n_particles']:6.2f} B/p")
    print("\n  by phase (max over visits of the phase's own increment):")
    ph = r["phases"]
    for name in sorted(ph, key=lambda n: -ph[n]["delta"]):
        e = ph[name]
        print(f"    {name:16} {e['delta'] / MB:10.2f} MB   peak {e['peak'] / MB:10.2f} MB"
              f"   x{e['visits']}")

    if a.vs_model:
        print("\n  against EngineConfig.step_bytes (the planner's O(N) terms):")
        tot = 0
        for k, v in sorted(r["model"].items(), key=lambda kv: -kv[1]):
            print(f"    {k:16} modelled {v / MB:10.2f} MB")
            tot += v
        print(f"    {'sum':16}          {tot / MB:10.2f} MB   against a measured "
              f"per-step high-water of {r['run_peak'] / MB:.2f} MB "
              f"({r['run_peak'] / tot:.2f}x)")
        print("\n  NB the measured figure is the WHOLE host high-water of a step, so it "
              "\n  includes terms step_bytes does not model. A ratio near 1.0 would mean "
              "\n  the planner's O(N) family is the whole host story; well above 1.0 "
              "\n  names how much is missing, which is the number the C-gh budget needs.")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump(dict(config=a.config, k=a.k, repeats=a.repeats,
                           determinism_ok=ok, determinism=msg, runs=reps), fh)
        print(f"\n  card -> {a.out}")
    return 0 if ok is not False else 1


if __name__ == "__main__":
    sys.exit(main())
