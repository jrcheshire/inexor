"""M0 R2 probe: w-frame ladder range budget (docs/roadmap.md M0 table, row R2).

Question: does the alpha ladder's dynamic-range cost leave acceptable velocity
resolution at int16, over K = 5..15 BullFrog steps?

Method (M0 plan): f64 reference BullFrog vs the SAME run with the TRUE integer
w-frame velocity update -- w held in int32 (so an int16 overflow is *measured*
as lost headroom rather than corrupting the trajectory; below the bound this is
bit-for-bit the int16 pipeline), positions/forces in f64 so every deviation
from the pure-f64 arm is attributable to velocity quantization alone.

Arms per config: F (f64 reference), F-2K (PM stepping floor), F32 (float
floor, fiducial only), Vq(s_w0) (the subject). All arms share IC bits, so
P(k) ratios are deterministic comparisons (no sample variance).

Kill/pivot trigger: unacceptable noise -> FastPM-additive promoted, or
remainder ledger built. Range-fail with noise-pass -> remainder ledger.

Run:  pixi run python scripts/m0_r2_ladder.py [--quick] [--outdir runs/m0/r2]
CPU, x64 (this script is the caller that enables it; _m0_common never does).
"""

# ruff: noqa: E402  (jax.config must precede any array-creating import)
import argparse
import json
import time
from pathlib import Path

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _m0_common as mc

# Okabe-Ito CVD-safe categorical order (fixed assignment, never cycled).
OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]

F64 = jnp.float64
F32 = jnp.float32


# ----------------------------------------------------------------------------
# drivers
# ----------------------------------------------------------------------------


def make_float_step(force, L):
    @jax.jit
    def stepf(x, v, dD2, alpha, bcoef):
        x = jnp.mod(x + dD2 * v, L)
        g = force(x)
        v = alpha * v + bcoef * g
        x = jnp.mod(x + dD2 * v, L)
        return x, v

    return stepf


def run_float(x0, v0, ladder, stepf):
    x, v = x0, v0
    for k in range(ladder.n_steps):
        x, v = stepf(x, v, 0.5 * ladder.dD[k], ladder.alphas[k], ladder.betas[k] / ladder.D_mid[k])
    return np.asarray(x), np.asarray(v)


def make_vq_step(force, L):
    @jax.jit
    def stepq(x, w, drift_pre, drift_post, kappa):
        # drift_pre/post = 0.5 dD_k s_w0 P_{k,k+1}: decode w -> physical velocity.
        x = jnp.mod(x + drift_pre * w.astype(F64), L)
        g = force(x)
        w = w + mc.rint_i(kappa * g)  # the TRUE integer kick (exact, additive)
        x = jnp.mod(x + drift_post * w.astype(F64), L)
        return x, w, jnp.max(jnp.abs(w))

    return stepq


def run_vq(x0, v0, ladder, stepq):
    s_w0 = ladder.s_w0
    w = mc.rint_i(v0 / s_w0)  # int32 storage; int16 bound MONITORED, not wrapped
    x = x0
    maxw = [int(jnp.max(jnp.abs(w)))]
    for k in range(ladder.n_steps):
        pre = 0.5 * ladder.dD[k] * s_w0 * ladder.P[k]
        post = 0.5 * ladder.dD[k] * s_w0 * ladder.P[k + 1]
        x, w, mw = stepq(x, w, pre, post, ladder.kappa[k])
        maxw.append(int(mw))
    v_final = np.asarray(w, dtype=np.float64) * (s_w0 * ladder.P[-1])
    return np.asarray(x), v_final, np.array(maxw)


def measure_pk(x, N, L, fdtype=F64, n_bins=24):
    mesh = mc.paint_f32(jnp.asarray(x, dtype=fdtype), N, L, fdtype)
    delta = np.asarray(mesh, dtype=np.float64) - 1.0  # n_particles == n_mesh
    k_nyq = np.pi * N / L
    return mc.pk_estimator(delta, L, n_bins=n_bins, k_max=k_nyq)


def rw_prediction(ladder):
    """Random-walk position-noise prediction: each kick's rounding error
    (uniform, std s_w0 |P_{k+1}| / sqrt(12) in v units) drifts for the
    remaining growth time. Per-axis rms."""
    D_K = ladder.D_steps[-1]
    terms = (ladder.s_w0 * np.abs(ladder.P[1:]) / np.sqrt(12.0)) * (D_K - ladder.D_mid)
    return float(np.sqrt(np.sum(terms**2)))


# ----------------------------------------------------------------------------
# the matrix
# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="64^3 smoke matrix")
    ap.add_argument("--outdir", default="runs/m0/r2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    N = 64 if args.quick else 128
    L = 500.0
    cosmo = mc.PLANCK
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    k_nyq = np.pi * N / L

    schedules = {
        "log0.1": dict(a_init=0.1, spacing="log"),
        "log0.04": dict(a_init=0.04, spacing="log"),
        "lin0.04": dict(a_init=0.04, spacing="linear"),
    }
    K_fid = [5, 8, 10, 12, 15] if not args.quick else [5, 10]
    K_alt = [5, 10, 15] if not args.quick else [10]
    K_lin = [5, 8, 15] if not args.quick else [8]
    K_guard = [10, 12]  # expect the |alpha| floor to fire here (session finding)
    B_OFFSETS = [-3, -2, -1, 0, 1, 2] if not args.quick else [-1, 0, 1]
    K_BSWEEP = 10 if not args.quick else 10

    t0 = time.time()
    force64 = mc.make_force_fn(N, L, N**3, fdtype=F64)
    step64 = make_float_step(force64, L)
    stepq = make_vq_step(force64, L)

    # ICs per a_init (same delta0 seed; arms within a config share bits).
    key = jax.random.PRNGKey(args.seed)
    delta0 = mc.linear_delta0(key, N, L, cosmo, fdtype=F64)
    ics = {}
    for name, sch in schedules.items():
        if sch["a_init"] not in ics:
            ics[sch["a_init"]] = mc.za_ics(delta0, N, L, sch["a_init"], cosmo, F64)

    results = {"config": dict(N=N, L=L, seed=args.seed, k_nyq=k_nyq), "runs": [], "guard": []}
    f_cache = {}  # (schedule, K, n_steps_mult) -> (x, v, pk)

    def get_float_arm(sname, K, mult=1):
        keyc = (sname, K, mult)
        if keyc not in f_cache:
            sch = schedules[sname]
            steps = mc.a_grid(sch["a_init"], 1.0, K * mult, sch["spacing"])
            lad = mc.ladder_constants(steps, cosmo, 1.0, 1.0)
            x0, v0 = ics[sch["a_init"]]
            tt = time.time()
            x, v = run_float(x0, v0, lad, step64)
            kk, pk, _ = measure_pk(x, N, L)
            print(f"  [F  ] {sname} K={K * mult}: {time.time() - tt:.1f}s")
            f_cache[keyc] = (x, v, kk, pk)
        return f_cache[keyc]

    def vq_arm(sname, K, b_off):
        sch = schedules[sname]
        steps = mc.a_grid(sch["a_init"], 1.0, K, sch["spacing"])
        x0, v0 = ics[sch["a_init"]]
        lad_unit = mc.ladder_constants(steps, cosmo, 1.0, 1.0)
        vmax0 = float(jnp.max(jnp.abs(v0)))
        s_pol = mc.s_w0_policy(vmax0, lad_unit.P[-1])
        s_w0 = s_pol * 2.0**-b_off  # +b = finer resolution = less headroom
        lad = mc.ladder_constants(steps, cosmo, s_w0, 1.0)
        tt = time.time()
        xq, vq, maxw = run_vq(x0, v0, lad, stepq)
        xF, vF, kk, pkF = get_float_arm(sname, K)
        kkq, pkq, _ = measure_pk(xq, N, L)
        headroom = np.log2(32767.0 / np.maximum(maxw, 1))
        sel = kk <= 0.5 * k_nyq
        rec = dict(
            schedule=sname,
            K=K,
            b_off=b_off,
            s_w0=s_w0,
            bits_ladder=lad.bits_consumed,
            maxw=maxw.tolist(),
            headroom_min=float(headroom.min()),
            overflow=bool(maxw.max() > 32767),
            k=kk.tolist(),
            pk_ratio=(pkq / pkF - 1.0).tolist(),
            err_k02=float(np.interp(0.2, kk, np.abs(pkq / pkF - 1.0))),
            err_band_max=float(np.max(np.abs(pkq / pkF - 1.0)[sel])),
            pos_rms=mc.min_image_rms(xq, xF, L),
            pos_rms_pred=rw_prediction(lad),
            vel_rms_rel=float(np.sqrt(np.mean((vq - vF) ** 2)) / np.std(vF)),
            c_growth=float(np.max(np.abs(vF)) / vmax0),
            wall_s=time.time() - tt,
        )
        results["runs"].append(rec)
        print(
            f"  [Vq ] {sname} K={K} b={b_off:+d}: headroom_min={rec['headroom_min']:+.2f} "
            f"err@0.2={rec['err_k02']:.2e} band_max={rec['err_band_max']:.2e} "
            f"pos_rms={rec['pos_rms']:.3e} (pred {rec['pos_rms_pred']:.3e}) "
            f"{rec['wall_s']:.1f}s"
        )
        return rec

    # --- guard-fire demonstration (no runs; the assertion itself is the result)
    for K in K_guard:
        steps = mc.a_grid(0.04, 1.0, K, "linear")
        try:
            mc.ladder_constants(steps, cosmo, 1.0, 1.0)
            results["guard"].append(dict(schedule="lin0.04", K=K, fired=False))
            print(f"  [GRD] lin0.04 K={K}: guard DID NOT fire (unexpected)")
        except ValueError as e:
            results["guard"].append(dict(schedule="lin0.04", K=K, fired=True, msg=str(e)))
            print(f"  [GRD] lin0.04 K={K}: guard fired as predicted")

    # --- fiducial schedule: K sweep at policy s_w0, with stepping floors
    floors = {}
    for K in K_fid:
        vq_arm("log0.1", K, 0)
        _, _, kk, pkF = get_float_arm("log0.1", K)
        _, _, _, pkF2 = get_float_arm("log0.1", K, mult=2)
        floors[K] = np.abs(pkF / pkF2 - 1.0)
    results["floors"] = {str(K): f.tolist() for K, f in floors.items()}

    # --- float32 floor at fiducial
    Kf = K_BSWEEP
    sch = schedules["log0.1"]
    steps = mc.a_grid(sch["a_init"], 1.0, Kf, sch["spacing"])
    lad = mc.ladder_constants(steps, cosmo, 1.0, 1.0)
    x0, v0 = ics[sch["a_init"]]
    force32 = mc.make_force_fn(N, L, N**3, fdtype=F32)
    step32 = make_float_step(force32, L)
    x32, _ = run_float(x0.astype(F32), v0.astype(F32), lad, step32)
    _, _, kk, pkF = get_float_arm("log0.1", Kf)
    kk32, pk32, _ = measure_pk(x32, N, L)
    results["f32_floor"] = dict(k=kk.tolist(), ratio=(pk32 / pkF - 1.0).tolist())
    print("  [F32] fiducial floor measured")

    # --- b sweep at fiducial K
    for b in B_OFFSETS:
        if not (
            b == 0 and any(r["schedule"] == "log0.1" and r["K"] == Kf for r in results["runs"])
        ):
            vq_arm("log0.1", Kf, b)

    # --- alternate schedules at policy s_w0
    for K in K_alt:
        vq_arm("log0.04", K, 0)
    for K in K_lin:
        vq_arm("lin0.04", K, 0)

    # ------------------------------------------------------------------
    # verdict
    # ------------------------------------------------------------------
    hd_gate = np.log2(1.0 / 0.9)  # the 0.9*32767 monitor line
    window = []
    for b in B_OFFSETS:
        rr = [
            r
            for r in results["runs"]
            if r["schedule"] == "log0.1" and r["K"] == Kf and r["b_off"] == b
        ]
        if not rr:
            continue
        r = rr[0]
        kkr = np.array(r["k"])
        sel = kkr <= 0.5 * k_nyq
        floor = np.interp(kkr[sel], kk, floors[Kf])
        ratio = np.abs(np.array(r["pk_ratio"]))[sel]
        ok_range = bool(r["headroom_min"] >= hd_gate)
        # Two noise readings, reported separately (no silent tolerance):
        # strict = below the K-vs-2K stepping floor at EVERY k <= 0.5 k_Nyq
        # (the floor is ~1e-6 at low k where BullFrog is near-exact -- a very
        # hard bar); abs = below 1e-4 relative everywhere in the band.
        ok_strict = bool((ratio <= floor).all())
        ok_abs = bool((ratio <= 1e-4).all())
        window.append(
            dict(
                b=b,
                ok_range=ok_range,
                ok_noise_strict=ok_strict,
                ok_noise_1e4=ok_abs,
                ok=ok_range and ok_abs,
            )
        )
    n_ok = sum(1 for wdw in window if wdw["ok"])
    n_strict = sum(1 for wdw in window if wdw["ok_range"] and wdw["ok_noise_strict"])
    verdict = "PASS" if n_ok >= 2 else "FAIL"
    results["verdict"] = dict(window=window, n_ok=n_ok, n_strict=n_strict, verdict=verdict)

    with open(out / "r2_results.json", "w") as fh:
        json.dump(
            results,
            fh,
            indent=1,
            default=lambda o: o.item() if isinstance(o, np.generic) else str(o),
        )

    make_figures(results, floors, out, k_nyq, Kf, K_fid, B_OFFSETS)

    cg = [r["c_growth"] for r in results["runs"] if r["schedule"] == "log0.1"]
    print(
        f"\nR2 verdict: {verdict} (b window: {n_ok}/{len(window)} pass "
        f"[range + noise<1e-4]; {n_strict}/{len(window)} also below the strict "
        f"stepping floor at ALL k<=0.5k_Nyq -- gate bar to be set with JC)"
    )
    print(f"  measured c_growth (log0.1) = {np.mean(cg):.2f} (policy placeholder was 4.0)")
    print(f"  guard fired: {[g['K'] for g in results['guard'] if g['fired']]} (lin0.04)")
    print(f"  total wall: {(time.time() - t0) / 60:.1f} min")
    print(f"  outputs: {out.resolve()}")


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------


def make_figures(results, floors, out, k_nyq, Kf, K_fid, B_OFFSETS):
    runs = results["runs"]
    cmap = plt.cm.viridis

    # Fig 1: headroom vs step (K sweep at b=0, both log schedules)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, K in enumerate(K_fid):
        for sname, ls in [("log0.1", "-"), ("log0.04", "--")]:
            rr = [r for r in runs if r["schedule"] == sname and r["K"] == K and r["b_off"] == 0]
            if rr:
                hw = np.log2(32767.0 / np.maximum(np.array(rr[0]["maxw"]), 1))
                ax.plot(
                    np.arange(len(hw)),
                    hw,
                    ls,
                    color=cmap(i / max(len(K_fid) - 1, 1)),
                    label=f"{sname} K={K}" if ls == "-" or K in (5, 10, 15) else None,
                )
    ax.axhline(0.0, color="k", lw=1)
    ax.axhline(np.log2(1 / 0.9), color="k", lw=0.8, ls=":")
    ax.set_xlabel("step")
    ax.set_ylabel("headroom  log2(32767 / max|w|)  [bits]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "r2_fig1_headroom.png", dpi=150)
    plt.close(fig)

    # Fig 2: P(k) ratio vs k -- (a) b sweep at K=Kf, (b) K sweep at b=0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    kk_f32 = np.array(results["f32_floor"]["k"])
    r_f32 = np.abs(np.array(results["f32_floor"]["ratio"]))
    for ax, (title, sel_fn, color_of) in zip(
        axes,
        [
            (
                f"b sweep, log0.1 K={Kf}",
                lambda r: r["schedule"] == "log0.1" and r["K"] == Kf,
                lambda r: cmap((r["b_off"] - min(B_OFFSETS)) / (max(B_OFFSETS) - min(B_OFFSETS))),
            ),
            (
                "K sweep, log0.1 b=0",
                lambda r: r["schedule"] == "log0.1" and r["b_off"] == 0,
                lambda r: cmap(K_fid.index(r["K"]) / max(len(K_fid) - 1, 1)),
            ),
        ],
    ):
        for r in runs:
            if sel_fn(r):
                lab = f"b={r['b_off']:+d}" if "b sweep" in title else f"K={r['K']}"
                ax.loglog(r["k"], np.abs(r["pk_ratio"]), color=color_of(r), label=lab, lw=1.4)
        fl = np.interp(kk_f32, np.array(runs[0]["k"]), floors[Kf])
        ax.loglog(kk_f32, fl, "k:", lw=1.2, label="PM stepping floor (F vs F-2K)")
        ax.loglog(kk_f32, r_f32, color="gray", ls="--", lw=1.2, label="f32 floor")
        ax.axvline(0.5 * k_nyq, color="k", lw=0.8, alpha=0.5)
        ax.set_xlabel("k [h/Mpc]")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("|P_Vq / P_F - 1|")
    fig.tight_layout()
    fig.savefig(out / "r2_fig2_pk_ratio.png", dpi=150)
    plt.close(fig)

    # Fig 3: the trade-off at K=Kf -- two shared-x panels (no dual axis)
    bs, hmins, errs = [], [], []
    for b in B_OFFSETS:
        rr = [r for r in runs if r["schedule"] == "log0.1" and r["K"] == Kf and r["b_off"] == b]
        if rr:
            bs.append(b)
            hmins.append(rr[0]["headroom_min"])
            errs.append(rr[0]["err_k02"])
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    axes[0].plot(bs, hmins, "o-", color=OI[0])
    axes[0].axhline(np.log2(1 / 0.9), color="k", ls=":", lw=0.8)
    axes[0].axhline(0, color="k", lw=1)
    axes[0].set_ylabel("min headroom [bits]")
    axes[0].grid(alpha=0.25)
    axes[1].semilogy(bs, errs, "o-", color=OI[1])
    fl02 = np.interp(0.2, np.array(runs[0]["k"]), floors[Kf])
    axes[1].axhline(fl02, color="k", ls=":", lw=0.8)
    axes[1].set_ylabel("|P ratio - 1| at k=0.2")
    axes[1].set_xlabel("b offset from s_w0 policy  (+ = finer velocity resolution)")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "r2_fig3_tradeoff.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
