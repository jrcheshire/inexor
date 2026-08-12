"""M-v2-6 Stage 0b, recomputed from the persisted per-visit series (antares 446).

This is the reduction `runs/v2/m6_peak_record.md` reports, kept as code because
the cards' own `aggregate` and `verdict` blocks predate two probe fixes and must
not be quoted. Run it from the directory holding the cards:

    python scripts/v2_m6_recompute_trace.py

The raw series is what made the correction free: a wrong reduction survives if
the per-visit readings are persisted, and that is worth designing for.


Reductions, stated once and used everywhere:
  run peak      = max over the boundary readings in a run, then MEDIAN over repeats
  phase increment = MAX over that phase's visits in a run (a peak is set by ONE
                  allocation moment, not by an average one), then MEDIAN over repeats
  step ladder   = max over the boundary readings inside a step, split at each
                  `coarse_paint`; median over repeats per step; OLS slope
The control arm has no boundaries, so its peak is its own ru_maxrss, which nothing
reset. `series` entries are (phase, high-water at the boundary, own increment).
"""
import json
import numpy as np

GB = 1e9


def load(tag):
    d = json.load(open(f"m6_peak_trace_{tag}.json"))
    k = [x for x in d if isinstance(d[x], dict) and "runs" in d[x]][0]
    return d, d[k]


def peak(r):
    return max(s[1] for s in r["series"]) if "series" in r else r["maxrss"]


def ladder(r):
    out, cur = [], []
    for name, pk, _ in r["series"]:
        if name == "coarse_paint" and cur:
            out.append(max(cur))
            cur = []
        cur.append(pk)
    out.append(max(cur))
    return np.array(out[1:], dtype=float) / GB  # drop the lead_drift prologue


def slope(y):
    return np.polyfit(np.arange(len(y)), y, 1)[0] * 1000  # MB/step


for tag in ("cdev8", "cdev", "cdev_k15"):
    top, c = load(tag)
    tr = c["runs"]["trace"]
    print("=" * 78)
    print(f"### {tag}  (job {top['slurm_job_id']}, {top['commit'][:7]}, K={top['k_steps']}, "
          f"{len(tr)} repeats)")
    for arm in top["arms"]:
        runs = c["runs"].get(arm) or []
        if not runs:
            continue
        p = np.array([peak(r) for r in runs]) / GB
        print(f"  {arm:8s} peak median {np.median(p):.3f} GB  sd {np.std(p, ddof=1):.3f}  "
              f"n={len(p)}  s/step {np.mean([r['s_per_step'] for r in runs]):.1f}")
    sd = np.std([peak(r) for r in tr], ddof=1) / GB
    print(f"  --- phase increment (max over visits, median over repeats); "
          f"sigma = corrected trace peak sd = {sd * 1000:.0f} MB")
    rows = []
    for n in tr[0]["order"]:
        per = [max(s[2] for s in r["series"] if s[0] == n) for r in tr
               if any(s[0] == n for s in r["series"])]
        if per:
            rows.append((n, np.median(per) / GB, len([s for s in tr[0]["series"] if s[0] == n])))
    for n, m, v in sorted(rows, key=lambda x: -x[1]):
        print(f"    {n:14s} {m:7.3f} GB   {m / sd:6.1f} sigma   visits/run {v:4d}")
    L = np.median(np.stack([ladder(r) for r in tr]), axis=0)
    sl = [slope(ladder(r)) for r in tr]
    print(f"  --- step ladder (median over repeats): {np.round(L, 2).tolist()}")
    print(f"      slope of the median ladder {slope(L):.0f} MB/step; "
          f"per repeat {np.round(sl, 0).tolist()} (mean {np.mean(sl):.0f})")
    for arm in ("trim", "control"):
        runs = c["runs"].get(arm) or []
        if runs and "series" in runs[0]:
            Lm = np.median(np.stack([ladder(r) for r in runs]), axis=0)
            print(f"      {arm} ladder slope {slope(Lm):.0f} MB/step "
                  f"(per repeat mean {np.mean([slope(ladder(r)) for r in runs]):.0f})")
    mb = tr[0]["mesh_bytes"]
    tile = mb["tile_kernels"] + mb["tile_workspace"]
    ts = np.median([max(s[2] for s in r["series"] if s[0] == "tile_short") for r in tr]) / GB
    print(f"  --- mesh_bytes tile terms {tile / GB:.3f} GB modelled vs tile_short "
          f"{ts:.3f} measured -> short by {ts - tile / GB:.3f} GB")
    print(f"      state {tr[0]['state_bytes']['total'] / GB:.3f} GB, n_rows {tr[0]['n_rows']}, "
          f"tiles {tr[0]['n_tiles']}, cap {tr[0]['cap']} ({tr[0]['cap_distinct']} distinct)")
