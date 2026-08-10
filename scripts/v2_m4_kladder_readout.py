"""Read the M-v2-4 K-ladder cards: does the f32 coarse-mesh error scale with K?

REPLACES an endpoint fit that was actively misleading. The first version of this
lived inline in `v2_m4_kladder_deneb.sbatch` and computed

    slope = log(E[-1] / E[0]) / log(K[-1] / K[0])

over the sorted rungs. On deneb job 399 that reported **-0.13** for readings of
3.365e-5 / 8.457e-5 / 2.800e-5 at K = 20/40/80 -- a tidy near-zero slope through
data whose MIDDLE point is 2.5x both ends on the max and 10x both ends on the
median. A two-point fit cannot see non-monotonicity, and the number it prints
looks exactly like a clean result. Do not reintroduce it.

What this prints instead:

  - every card, grouped by K, with n and the spread;
  - per-K mean and sd of E_max and E_median when n > 1, and an explicit refusal
    to quote a spread when n == 1;
  - a least-squares fit in log E vs log K, over the per-K MEANS, with the
    residuals shown so a bad fit is visible rather than averaged away;
  - a monotonicity check, stated separately from the fit, because a slope is
    only meaningful if the data is monotone in the first place.

Reading the slope, per M-v2-4's pre-registration:
  ~0.5  roundoff accumulating as a random walk     benign
  ~1.0  a systematic bias                          blocks adoption
  ~0.0  a one-off; the absolute level needs its own explanation
"""

import argparse
import glob
import json
import math
import os


def load(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        with open(p) as fh:
            d = json.load(fh)
        if "E_max" not in d or "k" not in d:
            continue
        prov = d.get("provenance") or {}
        knobs = prov.get("knobs") or {}
        rows.append(dict(
            k=int(d["k"]), seed=int(d.get("seed", 0)),
            e_max=float(d["E_max"]), e_med=float(d["E_median"]),
            bins=int(d.get("n_band_bins", 0)),
            vacuous=bool(d.get("vacuous", False)),
            card=os.path.basename(p),
            backend=prov.get("backend"),
            host_cores=prov.get("host_cores"),
            slack=knobs.get("slack"),
            arena_frac=knobs.get("arena_frac"),
        ))
    return rows


def check_comparability(rows):
    """Refuse to pool cards that are not measurements of the same thing.

    A glob is not an experiment. These rungs are pooled into per-K means and a
    slope, which is only meaningful if every card ran the same estimator on the
    same machinery -- and two axes can break that silently:

      BACKEND. XLA-CPU and CUDA are different computations, not the same
      computation at different speeds, and the CPU reduction order additionally
      follows the host core count. A cdev8 K=10 anchor measured on a laptop CPU
      cannot baseline a cgh64 reading measured on a cluster GPU, which is the
      whole reason the anchor exists.

      CAPACITY. `slack` and `arena_frac` set the brick/arena headroom. The
      K-ladder and seed-replicate jobs both ran 0.20/0.08 and NOTHING on their
      cards recorded it, so for a while the only evidence they matched lived in
      two sbatch files.

    Cards written before provenance existed report None. That is reported as
    UNKNOWN rather than assumed compatible: an absent field is not a pass.
    """
    def spread(field):
        return sorted({r[field] for r in rows}, key=lambda v: (v is None, v))

    problems = []
    for field, label in (("backend", "backend"), ("host_cores", "host cores"),
                         ("slack", "slack"), ("arena_frac", "arena_frac")):
        vals = spread(field)
        if len(vals) > 1:
            detail = ", ".join(
                f"{'UNKNOWN' if v is None else v}"
                f" ({sum(1 for r in rows if r[field] == v)} cards)" for v in vals
            )
            problems.append(f"{label}: {detail}")
    unknown = [r["card"] for r in rows if r["backend"] is None]
    return problems, unknown


def _stats(vals):
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, None
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--glob", default="runs/v2/m4_gate_cdev8_accuracy_k*.json")
    args = ap.parse_args()

    rows = [r for r in load(args.glob) if not r["vacuous"]]
    if not rows:
        print("no usable cards matched", args.glob)
        return 1

    ks = sorted({r["k"] for r in rows})
    print(f"{len(rows)} cards over K = {ks}\n")

    # BEFORE any pooling: are these cards measurements of the same thing?
    problems, unknown = check_comparability(rows)
    if unknown:
        print(f"WARNING: {len(unknown)} card(s) predate provenance and record no backend "
              "or capacity knobs. They cannot be shown to be comparable, and an absent "
              "field is not a pass:")
        for c in unknown:
            print(f"    {c}")
        print()
    if problems:
        print("REFUSING to pool these cards -- they disagree on an axis that changes the "
              "measurement:")
        for p in problems:
            print(f"    {p}")
        print("\n  Per-K means and a slope over a mixed set would be arithmetic on "
              "incommensurable numbers. Re-glob one homogeneous set, or re-run the odd "
              "cards where the rest were measured.")
        return 2

    means = {}
    for k in ks:
        got = [r for r in rows if r["k"] == k]
        mx, sx = _stats([r["e_max"] for r in got])
        md, sd = _stats([r["e_med"] for r in got])
        means[k] = mx
        print(f"K={k:3d}  n={len(got)}")
        for r in sorted(got, key=lambda r: r["seed"]):
            print(f"    seed {r['seed']:2d}  E_max {r['e_max']:.3e}  "
                  f"E_med {r['e_med']:.3e}  bins {r['bins']}  {r['card']}")
        if sx is None:
            print("    mean E_max %.3e   sd: n=1, NO spread is measurable -- this rung "
                  "licenses no statement about scatter" % mx)
        else:
            print(f"    mean E_max {mx:.3e}  sd {sx:.3e}  ({100 * sx / mx:.0f}%)")
            print(f"    mean E_med {md:.3e}  sd {sd:.3e}")
        print()

    # monotonicity, stated BEFORE the fit and separately from it
    seq = [means[k] for k in ks]
    mono = all(b >= a for a, b in zip(seq, seq[1:])) or all(b <= a for a, b in zip(seq, seq[1:]))
    print(f"monotone in K: {mono}")
    if not mono:
        worst = max(range(1, len(seq) - 1),
                    key=lambda i: max(seq[i] / seq[i - 1], seq[i] / seq[i + 1]),
                    default=None)
        if worst is not None:
            print(f"  NOT monotone: K={ks[worst]} sits {seq[worst] / seq[worst - 1]:.1f}x "
                  f"above K={ks[worst - 1]} and {seq[worst] / seq[worst + 1]:.1f}x above "
                  f"K={ks[worst + 1]}. A power-law slope through this is not meaningful; "
                  "either a rung is anomalous or the per-K scatter is comparable to the "
                  "trend, and only replicates can tell those apart.")

    if len(ks) < 2:
        print("\nfewer than two K values; no fit")
        return 0

    # least squares in log-log over the per-K means, residuals shown
    xs = [math.log(k) for k in ks]
    ys = [math.log(means[k]) for k in ks]
    xb, yb = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - xb) ** 2 for x in xs)
    slope = sum((x - xb) * (y - yb) for x, y in zip(xs, ys)) / sxx
    inter = yb - slope * xb
    print(f"\nleast-squares slope d(log E)/d(log K) = {slope:+.2f}  "
          f"(over {len(ks)} K values, on per-K means)")
    print("  residuals in log E:")
    for k, x, y in zip(ks, xs, ys):
        print(f"    K={k:3d}  {y - (slope * x + inter):+.3f}")
    print("\n  ~0.5 random walk (benign); ~1.0 systematic bias (blocks adoption per the")
    print("  pre-registration); ~0.0 one-off, and the absolute level needs its own")
    print("  explanation. A slope is only worth reading if `monotone in K` is True and")
    print("  the per-K sd is small against the spread across K.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
