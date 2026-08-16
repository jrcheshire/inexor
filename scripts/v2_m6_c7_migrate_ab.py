"""C7 -- do 5o's and 5p's cdev speedups survive at cgh64?

**The question.** Two `migrate` changes landed on cdev legs measured on one
machine: the compiled eject path (5o, 1.30x) and the radix-path sort (5p,
1.13x), stacking to 1.52x. Neither has run at 64x the particles, and this
milestone has already been bitten twice by a small-config number that did not
transfer -- Stage 2c measured ~2% at cdev and 20.2% at cgh64 (5k), and
`numpy_batched` inverted between arm64 and x86 (5n). A cdev-only speedup is a
hypothesis about C-gh, not a result.

**The design.** All arms run in ONE process against a state rebuilt from the
same seed, so the starting configuration is identical by construction and no
cross-job machine drift enters (5k: pairing arms in one job is what settled the
+8.6%/+6.0% drift that looked like a real effect).

    baseline   wide sort  + numpy eject   <- what main did before either change
    radix      radix sort + numpy eject   <- 5p alone
    compiled   radix sort + jax eject     <- 5p + 5o, the candidate

**Bitwise is carried by HASH, not by comparison.** Holding three cgh64 states at
once to diff them costs ~4.5 GB on top of a ~25 GB build peak for no extra
information: a sha256 over each state array is exact, and an inequality names
which array moved just as well as a count would. What it does not give is HOW
MANY elements differ, which is why a mismatch here means re-running the pair at
cdev under the elementwise gate rather than debugging from the hash.

**Pre-registered, before submission:**

1. The stacked speedup at cgh64 is **1.3-1.8x** on the phase. The cdev figure is
   1.52x and the mechanism is per-row work in both arms, so a large move in
   either direction means the mechanism is not what the cdev legs measured.
2. **Every arm is bitwise identical.** This is not a prediction, it is a
   requirement -- a mismatch voids the change rather than the run, and both
   changes are already gated elementwise at cdev in the suite.
3. `radix` alone lands **1.05-1.25x**. It is the narrower claim and the one whose
   key range is structurally fixed (`buckets_per_brick` = 512 at every rung).

A miss on (1) or (3) is reportable and does not void anything. A miss on (2)
stops the promotion.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inexor import state  # noqa: E402

from v2_m6_migrate_depth import CONFIGS, _build  # noqa: E402

ARMS = ("baseline", "radix", "compiled")
STATE_ARRAYS = ("off", "w", "occupancy", "brick_start", "vel_scale", "arena_bucket")

_RADIX = state._stable_order


def _wide(key, n_values):
    """The comparison sort `_stable_order` replaced, for the control arm."""
    return np.argsort(np.asarray(key), kind="stable")


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _hashes(st):
    out = {}
    for name in STATE_ARRAYS:
        a = getattr(st, name, None)
        if a is None:
            continue
        out[name] = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
    return out


def _c_drift(st, fraction):
    extent = float(st.t9.box_size) / int(st.bricks_per_side)
    return fraction * extent / (float(np.max(st.vel_scale)) * float(np.iinfo(np.int16).max))


def run_config(name, arms, repeats, fraction):
    cfg = CONFIGS[name]
    print(f"== C7 {name}: n_part={cfg['n_part']} nb={cfg['nb']} f={fraction} "
          f"repeats={repeats}")
    res = {}
    for arm in arms:
        state._stable_order = _RADIX if arm != "baseline" else _wide
        kernel = "jax" if arm == "compiled" else "numpy"
        times, h, stats = [], None, None
        for r in range(repeats):
            st = _build(cfg["n_part"], cfg["nb"], cfg["box"],
                        brick_slack=0.0, arena_frac=0.20)
            c = _c_drift(st, fraction)
            t0 = time.perf_counter()
            s = state.drift_and_migrate(st, c, kernel=kernel)
            times.append(time.perf_counter() - t0)
            if h is None:
                h, stats = _hashes(st), s
            del st
        res[arm] = dict(times=times, best=float(min(times)), median=float(np.median(times)),
                        hashes=h, stats={k: (int(v) if isinstance(v, (int, np.integer))
                                             else float(v)) for k, v in stats.items()},
                        sort="wide" if arm == "baseline" else "radix", eject=kernel)
        print(f"  {arm:10s} best {res[arm]['best']:8.3f} s   "
              f"median {res[arm]['median']:8.3f} s   "
              f"(sort={res[arm]['sort']}, eject={kernel})")
    state._stable_order = _RADIX

    base = res.get("baseline", {}).get("best")
    for arm in arms:
        if arm != "baseline" and base:
            res[arm]["speedup_vs_baseline"] = base / res[arm]["best"]

    # --- the requirement, checked before any speedup is believed ---
    ref = res[arms[0]]["hashes"]
    bitwise = True
    for arm in arms[1:]:
        for k, v in res[arm]["hashes"].items():
            if ref.get(k) != v:
                bitwise = False
                print(f"  BITWISE FAILURE: state.{k} differs between {arms[0]} and {arm} "
                      f"({ref.get(k)} vs {v})")
    stats_equal = all(res[a]["stats"] == res[arms[0]]["stats"] for a in arms)
    if not stats_equal:
        bitwise = False
        print("  BITWISE FAILURE: migration stats differ across arms")
    print(f"  bitwise across all arms: {'PASS' if bitwise else 'FAIL'}")
    for arm in arms:
        if "speedup_vs_baseline" in res[arm]:
            print(f"  {arm} is {res[arm]['speedup_vs_baseline']:.2f}x the baseline")
    return res, bitwise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=["cdev"], choices=sorted(CONFIGS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--fraction", type=float, default=2.85)
    ap.add_argument("--out", default=os.path.join(REPO, "runs", "v2", "m6_c7_migrate_ab.json"))
    a = ap.parse_args()

    if "compiled" in a.arms:
        import jax

        jax.config.update("jax_enable_x64", True)
        print(f"jax {jax.__version__} on {jax.devices()[0].platform}, x64 on")

    card = dict(configs={}, arms=a.arms, repeats=a.repeats, fraction=a.fraction,
                commit=_git_commit(), host=platform.node(), numpy=np.__version__,
                slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    rc = 0
    for name in a.config:
        res, bitwise = run_config(name, a.arms, a.repeats, a.fraction)
        card["configs"][name] = dict(result=res, bitwise=bitwise)
        if not bitwise:
            rc = 2
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(card, f, indent=1)
    print(f"card -> {a.out}")
    if rc:
        print("EXIT 2: a bitwise requirement failed. The speedups above are not "
              "adoptable and the pair needs re-running at cdev under the elementwise gate.")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
