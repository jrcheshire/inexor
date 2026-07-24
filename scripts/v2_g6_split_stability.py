"""v2 G6: is the two-level split's low-k error a CORRECTABLE transfer?

D-v2-10 accepted a low-k architectural floor with eyes open: below k ~ 1.5 the
split is the sim's dominant P(k) error (a coherent suppression, ~1% at k = 0.5,
plateauing ~2.5e-2 above k ~ 1.4), it shrinks as 1/P, and "a measured-transfer
correction cannot CURRENTLY be claimed to remove it". That clause is about
EVIDENCE, not mechanism. Three separable properties decide it, and this script
measures all three from an ensemble of per-seed v2_g5_two_level_force runs:

  1. DETERMINISTIC OR STOCHASTIC, within one realization. A correctable error is
     a multiplicative window: T(k) != 1 with r(k) = 1. Any decorrelation
     1 - r > 0 is irreducible -- no per-k factor removes it. (pk_cross, per arm.)
  2. REALIZATION-STABLE, across seeds. Per-bin scatter sigma_R(k) against the
     ensemble mean Rbar(k), plus the LEAVE-ONE-OUT test that is the actual
     decision number: build Tbar from N-1 seeds, apply it to the held-out seed,
     and report what the residual error is against what it was uncorrected.
  3. LATTICE-LOCKED OR NOT. "Stable across seeds" and "locked to the tile grid"
     are different properties -- the tile lattice is fixed in space while
     structure moves between realizations -- so the shift arms move the lattice
     at fixed ICs and ask whether R(k) cares.

Shape invariance across (T, b) rides along: the 1/P law says the AMPLITUDE
scales, and says nothing about whether the shape of T(k) is P-independent. If
the P-rescaled curves lie on top of each other, one measured transfer
generalizes across the config table instead of needing re-measurement per point.

NOTHING HERE SELF-RATIFIES. It reports floors and margins; the claim threshold
is JC's call once the numbers exist (repo convention: measure the floor first).

Usage:
  pixi run python scripts/v2_g6_split_stability.py [--config cdev] [--dir runs/v2]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from v2_g5_core import padded_size  # noqa: E402

# The identity arms must sit this far below the SMALLEST measured split error
# before any origin number may be read. Relative, not absolute, so it cannot rot
# as the split error moves with P: what matters is that the plumbing is
# negligible against the signal it is used to interpret.
IDENTITY_MARGIN = 0.01
LOW_K = 0.5  # h/Mpc -- where D-v2-10 quotes the coherent ~1% suppression


def load_ensemble(d, cfg):
    """Per-seed result JSONs -> [(seed, doc)], sorted, with the plumbing guards.

    A missing guard is a FAILED guard here (retrospective Sec 5): every check
    below hard-fails rather than warning, because each one has a failure mode
    that looks exactly like the answer we want.
    """
    docs = []
    for path in sorted(glob.glob(os.path.join(d, f"g5_results_{cfg}_seed*.json"))):
        with open(path) as fh:
            doc = json.load(fh)
        docs.append((int(doc.get("seed", -1)), doc, path))
    if len(docs) < 2:
        die(f"need >= 2 seeds, found {len(docs)} in {d}")

    # Distinct ICs. Two seeds that produced the same initial conditions would
    # report flawless realization stability -- the exact wrong answer this
    # ensemble could produce silently.
    seen = {}
    for seed, doc, path in docs:
        fp = json.dumps(doc.get("ic_fingerprint") or {}, sort_keys=True)
        if fp in ("{}", "null"):
            die(f"seed {seed}: no ic_fingerprint in {path} (stale probe -- rerun)")
        if fp in seen:
            die(f"seeds {seen[fp]} and {seed} have IDENTICAL ICs ({fp}) -- seed plumbing is broken")
        seen[fp] = seed

    # One k grid, or the stack is meaningless.
    k0 = None
    for seed, doc, path in docs:
        arm = first_plain_arm(doc)
        k = np.asarray(doc["evolve"][arm]["k"])
        if k0 is None:
            k0 = k
        elif k.shape != k0.shape or not np.allclose(k, k0):
            die(f"seed {seed}: k grid differs from the first seed's")
    return docs, k0


def die(msg):
    print(f"!! G6 FAIL: {msg}")
    sys.exit(1)


def plain_arms(doc):
    """two_level arms with the lattice at its default origin, in run order."""
    return [
        a
        for a in doc["evolve"]
        if a.startswith("two_level_T") and "_shift" not in a and isinstance(doc["evolve"][a], dict)
    ]


def shift_arms(doc):
    return [a for a in doc["evolve"] if a.startswith("two_level_T") and "_shift" in a]


def first_plain_arm(doc):
    arms = plain_arms(doc)
    if not arms:
        die("no plain two_level evolve arm in a result file")
    return arms[0]


def arm_tb(arm):
    """'two_level_T128_b32[_shift64]' -> (128, 32, shift or None)."""
    parts = arm.split("_")
    T = int(parts[2][1:])
    b = int(parts[3][1:])
    sh = int(parts[4][5:]) if len(parts) > 4 and parts[4].startswith("shift") else None
    return T, b, sh


def stack(docs, arm, key="dP_over_P"):
    return np.array([doc["evolve"][arm][key] for _, doc, _ in docs], dtype=float)


def check_identities(docs, plain_min):
    """The degenerate limits. Returns the record; hard-fails if absent or loose.

    A full-tile shift maps the lattice onto itself, and a mono arm is exactly
    equivariant under a lattice translation -- so both must reproduce their
    unshifted twin. If they do not, the shift PLUMBING moved, and every origin
    number downstream would read that as physics.
    """
    found = {}
    for seed, doc, _ in docs:
        for arm in doc["evolve"]:
            if not isinstance(doc["evolve"][arm], dict):
                continue
            if arm.startswith("mono_shift"):
                # Measured against mono, so the identity IS this number: a
                # translated mono arm must reproduce mono itself.
                found.setdefault("mono_shift", (seed, arm, doc["evolve"][arm]["max_abs_gate"]))
            elif arm.startswith("two_level_T") and "_shift" in arm:
                T, b, sh = arm_tb(arm)
                if sh != T:
                    continue  # a real origin arm, not the identity
                # NOT the same test: every arm's dP/P is measured against MONO,
                # so a full-tile-shifted tiled arm carries the whole split error
                # and its max_abs_gate is ~the signal, not ~zero. The identity is
                # that it reproduces its UNSHIFTED TWIN curve for curve.
                base = f"two_level_T{T}_b{b}"
                if base not in doc["evolve"]:
                    continue
                d = np.abs(
                    np.asarray(doc["evolve"][arm]["dP_over_P"])
                    - np.asarray(doc["evolve"][base]["dP_over_P"])
                )
                found.setdefault("tile_period", (seed, arm, float(d.max())))
    for name in ("mono_shift", "tile_period"):
        if name not in found:
            die(
                f"identity arm '{name}' never ran -- run one seed with --shift-identity. "
                "A check that did not run reads exactly like one that passed."
            )
    tol = IDENTITY_MARGIN * plain_min
    rec = {}
    what = dict(
        mono_shift="translated mono vs mono",
        tile_period="full-period shift vs its unshifted twin, per k bin",
    )
    for name, (seed, arm, val) in found.items():
        ok = bool(val is not None and val < tol)
        rec[name] = dict(seed=seed, arm=arm, metric=val, what=what[name], tol=tol, ok=ok)
        print(
            f"  identity {name:12s} (seed {seed}) max|dP/P| = {val:.3e}  tol {tol:.3e}  "
            f"{'OK' if ok else 'FAIL'}   [{what[name]}]"
        )
    if not all(v["ok"] for v in rec.values()):
        die("a degenerate-limit identity failed -- the shift plumbing is not clean")
    return rec


def summarize_arm(docs, arm, k, k_gate):
    """Ensemble statistics + the leave-one-out correction test for one arm."""
    R = stack(docs, arm)  # (n_seed, n_k)
    n = R.shape[0]
    band = k <= k_gate
    low = k <= LOW_K
    mean = R.mean(axis=0)
    sd = R.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_scatter = np.where(np.abs(mean) > 0, sd / np.abs(mean), np.nan)

    # LEAVE-ONE-OUT: the transfer a user could actually build (from the OTHER
    # seeds) applied to a realization it has never seen. Correcting in-sample
    # would flatter the result by construction.
    resid = np.empty_like(R)
    for i in range(n):
        others = np.delete(R, i, axis=0).mean(axis=0)
        resid[i] = R[i] - others
    raw_band = np.abs(R[:, band]).max(axis=1)
    res_band = np.abs(resid[:, band]).max(axis=1)
    raw_low = np.abs(R[:, low]).max(axis=1)
    res_low = np.abs(resid[:, low]).max(axis=1)

    cross = [doc["evolve"][arm].get("cross") for _, doc, _ in docs]
    if any(c is None for c in cross):
        die(f"arm {arm}: no cross-spectrum (stale probe) -- 1-r is the correctability test")
    if any(len(c["k"]) != len(k) for c in cross):
        die(f"arm {arm}: cross-spectrum k grid differs from the dP/P grid")
    one_minus_r = np.array([c["one_minus_r"] for c in cross], dtype=float)
    T_k = np.array([c["T"] for c in cross], dtype=float)

    T, b, _ = arm_tb(arm)
    P, _ = padded_size(T, b)
    return dict(
        arm=arm,
        tile=T,
        buf=b,
        padded_P=P,
        n_seeds=n,
        k=k.tolist(),
        mean=mean.tolist(),
        sd=sd.tolist(),
        rel_scatter=rel_scatter.tolist(),
        resid_mean=np.abs(resid).mean(axis=0).tolist(),
        one_minus_r_mean=one_minus_r.mean(axis=0).tolist(),
        T_mean=T_k.mean(axis=0).tolist(),
        band=dict(
            uncorrected_median=float(np.median(raw_band)),
            loo_residual_median=float(np.median(res_band)),
            gain=float(np.median(raw_band) / np.median(res_band)) if res_band.all() else None,
            max_one_minus_r=float(one_minus_r[:, band].max()),
        ),
        low_k=dict(
            k_max=LOW_K,
            uncorrected_median=float(np.median(raw_low)),
            loo_residual_median=float(np.median(res_low)),
            gain=float(np.median(raw_low) / np.median(res_low)) if res_low.all() else None,
            max_one_minus_r=float(one_minus_r[:, low].max()),
            max_rel_scatter=float(np.nanmax(rel_scatter[low])),
        ),
    )


def origin_dependence(docs, k, k_gate):
    """Per seed: shifted-lattice arm minus fixed-lattice arm, same ICs.

    Read against the seed-to-seed scatter of the fixed arm. If moving the
    lattice costs much less than changing the realization, the error is a
    periodization property of the tile PERIOD (correctable in principle);
    if it is comparable, the error is tied to where the tile walls sit.
    """
    pairs = []
    for seed, doc, _ in docs:
        for arm in shift_arms(doc):
            T, b, sh = arm_tb(arm)
            if sh == T:
                continue  # the identity, not an arm
            base = f"two_level_T{T}_b{b}"
            if base not in doc["evolve"]:
                continue
            pairs.append(
                (
                    seed,
                    sh,
                    np.asarray(doc["evolve"][arm]["dP_over_P"]),
                    np.asarray(doc["evolve"][base]["dP_over_P"]),
                )
            )
    if not pairs:
        return None
    band = k <= k_gate
    d = np.array([p[2] - p[3] for p in pairs])
    fixed = np.array([p[3] for p in pairs])
    return dict(
        n_pairs=len(pairs),
        shifts=[int(p[1]) for p in pairs],
        max_abs_diff_band_median=float(np.median(np.abs(d[:, band]).max(axis=1))),
        rms_diff_band=float(np.sqrt((d[:, band] ** 2).mean())),
        seed_scatter_band=float(np.sqrt((fixed[:, band].std(axis=0, ddof=1) ** 2).mean())),
        mean_shifted=(fixed + d).mean(axis=0).tolist(),
        mean_fixed=fixed.mean(axis=0).tolist(),
    )


def shape_invariance(summaries, k, k_gate):
    """Do the P-rescaled mean curves coincide? (Amplitude ~ 1/P is known.)"""
    if len(summaries) < 2:
        return None
    band = (k <= k_gate) & (k > 0)
    a, b = summaries[0], summaries[1]
    ra = np.asarray(a["mean"])[band] * a["padded_P"]
    rb = np.asarray(b["mean"])[band] * b["padded_P"]
    denom = 0.5 * (np.abs(ra) + np.abs(rb))
    frac = np.abs(ra - rb) / np.where(denom > 0, denom, np.nan)
    return dict(
        arms=[a["arm"], b["arm"]],
        padded_P=[a["padded_P"], b["padded_P"]],
        max_frac_diff_band=float(np.nanmax(frac)),
        median_frac_diff_band=float(np.nanmedian(frac)),
    )


def make_figure(summaries, docs, k, k_gate, origin, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.ticker import NullFormatter

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax in axes.ravel():
        ax.set_xscale("log")
    s0 = summaries[0]
    R = stack(docs, s0["arm"])
    ax = axes[0, 0]
    for i in range(R.shape[0]):
        ax.plot(k, R[i], color="0.75", lw=0.7)
    mean = np.asarray(s0["mean"])
    sd = np.asarray(s0["sd"])
    ax.plot(k, mean, color="C0", lw=1.8, label=f"mean, N={s0['n_seeds']}")
    ax.fill_between(k, mean - sd, mean + sd, color="C0", alpha=0.25, lw=0, label=r"$\pm\sigma$")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axvline(k_gate, color="k", ls=":", lw=0.8)
    ax.set_ylabel(r"$\Delta P/P$  (tiled $-$ mono)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for s in summaries:
        ax.plot(k, np.asarray(s["rel_scatter"]), label=f"T{s['tile']}/b{s['buf']}")
    ax.axhline(1.0, color="k", lw=0.6)
    ax.axvline(k_gate, color="k", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\sigma_R(k)\,/\,|\bar R(k)|$")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    for i, s in enumerate(summaries):
        ax.plot(k, np.abs(np.asarray(s["mean"])), color=f"C{i}", label=f"|mean| T{s['tile']}")
        ax.plot(k, np.asarray(s["resid_mean"]), color=f"C{i}", ls="--",
                label=f"leave-one-out resid T{s['tile']}")
    ax.axvline(k_gate, color="k", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel(r"$k$  [$h\,\mathrm{Mpc}^{-1}$]")
    ax.set_ylabel(r"$|\Delta P/P|$")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    for i, s in enumerate(summaries):
        ax.plot(k, np.asarray(s["one_minus_r_mean"]), color=f"C{i}", label=f"T{s['tile']}/b{s['buf']}")
    if origin:
        ax.plot(k, np.abs(np.asarray(origin["mean_shifted"]) - np.asarray(origin["mean_fixed"])),
                color="C3", ls="-.", label="|shifted - fixed| lattice")
    ax.axvline(k_gate, color="k", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel(r"$k$  [$h\,\mathrm{Mpc}^{-1}$]")
    ax.set_ylabel(r"$1-r(k)$")
    ax.legend(frameon=False, fontsize=8)

    for ax in axes.ravel():
        # set_xscale re-instantiates the locator/formatter, so this must come
        # AFTER every panel is drawn: with under a decade between decade ticks
        # matplotlib labels the minor ticks too and they smear together.
        ax.xaxis.set_minor_formatter(NullFormatter())

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"\nwrote {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="cdev")
    ap.add_argument("--dir", default=os.path.join(REPO, "runs", "v2"))
    ap.add_argument("--out-prefix", default="g6_split_stability")
    args = ap.parse_args()

    docs, k = load_ensemble(args.dir, args.config)
    k = np.asarray(k)
    k_gate = float(docs[0][1]["geometry"]["k_gate"])
    seeds = [s for s, _, _ in docs]
    print(f"=== G6 split stability: {args.config}, {len(docs)} seeds {seeds} ===")
    print(f"band k <= {k_gate:.2f}; low-k statistic at k <= {LOW_K}\n")

    arms = plain_arms(docs[0][1])
    # Every seed must carry every arm, or the stack silently mixes ensembles.
    for seed, doc, path in docs:
        missing = [a for a in arms if a not in doc["evolve"]]
        if missing:
            die(f"seed {seed} is missing arm(s) {missing} ({path})")

    summaries = [summarize_arm(docs, a, k, k_gate) for a in arms]
    plain_min = min(s["band"]["uncorrected_median"] for s in summaries)

    print("--- degenerate limits (read these before anything else) ---")
    identities = check_identities(docs, plain_min)

    print("\n--- per arm ---")
    for s in summaries:
        lo, bd = s["low_k"], s["band"]
        print(f"  {s['arm']}  (P = {s['padded_P']})")
        print(f"    in-band : uncorrected {bd['uncorrected_median']:.3e} -> leave-one-out "
              f"{bd['loo_residual_median']:.3e}  (x{bd['gain'] or float('nan'):.1f})   "
              f"max 1-r {bd['max_one_minus_r']:.2e}")
        print(f"    k<={LOW_K}: uncorrected {lo['uncorrected_median']:.3e} -> leave-one-out "
              f"{lo['loo_residual_median']:.3e}  (x{lo['gain'] or float('nan'):.1f})   "
              f"max sigma/|mean| {lo['max_rel_scatter']:.2f}")

    shape = shape_invariance(summaries, k, k_gate)
    if shape:
        print(f"\n--- shape invariance ({shape['arms'][0]} vs {shape['arms'][1]}, "
              f"P = {shape['padded_P'][0]} vs {shape['padded_P'][1]}) ---")
        print(f"  P-rescaled curves agree to {shape['median_frac_diff_band']:.1%} median, "
              f"{shape['max_frac_diff_band']:.1%} max, in-band")

    origin = origin_dependence(docs, k, k_gate)
    if origin:
        print(f"\n--- tile-origin dependence ({origin['n_pairs']} shifted pairs) ---")
        print(f"  moving the lattice : rms |shift - fixed| = {origin['rms_diff_band']:.3e}")
        print(f"  changing the seed  : rms seed scatter    = {origin['seed_scatter_band']:.3e}")

    out = dict(
        config=args.config,
        seeds=seeds,
        k_gate=k_gate,
        low_k=LOW_K,
        identities=identities,
        arms=summaries,
        shape_invariance=shape,
        origin_dependence=origin,
        note="Reports floors and margins only. The claim threshold for 'the split error is "
        "correctable as a measured transfer' is JC's call (D-v2-10 amendment), not this "
        "script's.",
    )
    out_json = os.path.join(args.dir, f"{args.out_prefix}.json")
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {out_json}")
    make_figure(summaries, docs, k, k_gate, origin, os.path.join(args.dir, f"{args.out_prefix}.png"))


if __name__ == "__main__":
    main()
