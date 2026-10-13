"""Initial-condition validation on the streamed generator's staged fields (paper App. A.3).

Reads the `stage/` intermediates that `realization.py ics --keep-stage` keeps (the z = 0 linear
density `delta`, the 2LPT displacement fields `psi1_*`, `psi2_*` and the emitted displacement
`u_*`, all float32 on the particle grid) and the IC directory's `manifest.json` (n_part, box,
a_init, growth2). Gaussian EH98 ICs only (the manifest's backend and f_NL are checked); the
cosmology is the default `Cosmology()`, the one the driver uses.

Subcommands, each writing one JSON (default `runs/validation/icval/<name>.json`):

    pk-lin      P(k) of the staged linear density over the generator's own linear P, averaged
                over each bin's modes: 48 log bins from k_f to the Nyquist, chi^2, and the
                mode-level mean of P_hat / P_lin.
    lpt         psi1, psi2 and u = D1 psi1 - D2 psi2 re-derived in core in float64 from the
                staged delta, against the staged float32 out-of-core fields; div psi1 + delta.
    div         the Zel'dovich identity div psi1 = -delta in Fourier space: residual power
                fraction on and off the Nyquist planes.
    s3          reduced skewness S3 at R = 4, 6, 8, 12 Mpc/h of the a_init density: Zel'dovich
                arm (x = q + D1 psi1), 2LPT arm (x = q + u, the staged displacement), CIC on a
                256^3 mesh, deconvolved, Fourier top hat; the linear field as a Gaussian null;
                leave-one-out jackknife over nsub^3 sub-cubes; tree-level predictions.
    ds3-tree    the tree-level Delta S3 = 3 <d1^2 s> / <d1^2>^2 on this realization's Lagrangian
                fields (d1 = D1 delta, s = D2 div psi2), same jackknife.
    ds3-paired  Delta S3 = S3(2LPT) - S3(ZA) minus its tree-level value on this realization, at
                displacement amplitudes s (D1 x s, D2 x s^2), one paired jackknife for all.

    pixi run --frozen python scripts/run/realization.py ics --config cgh64 \\
        --workdir runs/validation/cgh64 --keep-stage
    pixi run --frozen python scripts/validation/ic_validation.py pk-lin
    pixi run --frozen python scripts/validation/ic_validation.py s3 --nsub 4
"""

import argparse
import json
import os
import time

import numpy as np
import scipy.fft as sfft
from _common import jackknife_err, kgrid, maxrss, tophat, write_json

RS = [4.0, 6.0, 8.0, 12.0]   # top-hat radii, Mpc/h
S3_MESH = 256                # Eulerian CIC mesh for the skewness


class Stage:
    """The IC directory's geometry (from its manifest) and its staged arrays."""

    def __init__(self, ic_dir):
        with open(os.path.join(ic_dir, "manifest.json")) as fh:
            man = json.load(fh)
        if man.get("backend") != "eh98" or float(man.get("f_NL", 0.0)) != 0.0:
            raise SystemExit(f"{ic_dir}: Gaussian EH98 ICs only (backend "
                             f"{man.get('backend')!r}, f_NL {man.get('f_NL')})")
        self.dir = os.path.join(ic_dir, "stage")
        if not os.path.isdir(self.dir):
            raise SystemExit(f"{self.dir} missing: make the ICs with --keep-stage")
        self.n = int(man["n_part"])
        self.box = float(man["box_size"])
        self.a_init = float(man["a_init"])
        self.growth2 = man["growth2"]

    def load(self, name, mmap=False):
        return np.load(os.path.join(self.dir, name), mmap_mode="r" if mmap else None)

    def growth(self, cos):
        from inexor.cosmology import growth_factor_2, growth_factor_a

        return (growth_factor_a(self.a_init, cos),
                growth_factor_2(self.a_init, cos, self.growth2))


def _cosmo():
    from inexor.config import Cosmology

    return Cosmology()


# ---------------------------------------------------------------------------
# pk-lin
# ---------------------------------------------------------------------------

def run_pk_lin(st, args):
    from inexor.cosmology import ic_k_table, linear_power

    N, L, W = st.n, st.box, args.workers
    cos = _cosmo()
    tab = ic_k_table(cos, N, L, backend="eh98")          # the generator's colour table
    delta = st.load("delta.npy").astype(np.float64)
    print("delta mean", delta.mean(), "var", delta.var(), flush=True)
    dk = sfft.rfftn(delta, workers=W)
    del delta
    kx = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    kz = 2 * np.pi * np.fft.rfftfreq(N, d=L / N)
    wz = np.full(kz.size, 2.0)                           # full-grid multiplicity (card convention)
    wz[0] = 1.0
    wz[-1] = 1.0
    kf, knyq = 2 * np.pi / L, np.pi * N / L
    edges = np.geomspace(kf * 0.999, knyq, 49)           # 48 log bins, kf .. Nyquist
    nb = len(edges) - 1
    S = {n: np.zeros(nb) for n in ("w", "k", "p", "ptab", "plin", "r")}
    norm = L**3 / float(N) ** 6
    for i in range(N):
        kk = np.sqrt(kx[i] ** 2 + kx[:, None] ** 2 + kz[None, :] ** 2)
        p = (dk[i].real ** 2 + dk[i].imag ** 2) * norm
        w = np.broadcast_to(wz, kk.shape)
        sel = kk > 0
        idx = np.digitize(kk[sel], edges) - 1
        ok = (idx >= 0) & (idx < nb)
        idx, kv, pv, wv = idx[ok], kk[sel][ok], p[sel][ok], w[sel][ok]
        pt = tab.P_of_k(kv)
        pl = linear_power(kv, cos)
        for name, v in (("w", wv), ("k", kv * wv), ("p", pv * wv), ("ptab", pt * wv),
                        ("plin", pl * wv), ("r", pv / pt * wv)):
            S[name] += np.bincount(idx, weights=v, minlength=nb)
    good = S["w"] > 0
    w = S["w"][good]
    k = S["k"][good] / w
    ratio = S["p"][good] / S["ptab"][good]          # bin-averaged P / bin-averaged P_lin
    rmode = S["r"][good] / w                        # mode-mean of P_hat / P_lin
    tab_vs_eh = S["ptab"][good] / S["plin"][good]
    sig = np.sqrt(2.0 / w)
    chi2 = float(np.sum(((ratio - 1) / sig) ** 2))

    def mean_err(mask):
        # mode-level mean of P_hat/P_lin; each independent complex mode |d|^2/P ~ Exp(1)
        ww = w[mask]
        m = float(np.sum(rmode[mask] * ww) / ww.sum())
        return m, float(1.0 / np.sqrt(ww.sum() / 2.0))

    out = dict(
        binning="48 log bins from k_f to the mesh Nyquist; full-grid mode weights; P_lin = the "
                "generator's ic_k_table (EH98, 32768 points) averaged over each bin's modes",
        k=k.tolist(), ratio=ratio.tolist(), sigma=sig.tolist(), n_modes=w.tolist(),
        table_over_analytic_EH98=tab_vs_eh.tolist(),
        chi2=chi2, dof=int(good.sum()),
        mean_k_lt_0p5=mean_err(k < 0.5), mean_all=mean_err(k > 0),
        max_abs_table_over_analytic_minus_1=float(np.max(np.abs(tab_vs_eh - 1))),
    )
    print(f"{'k':>8} {'P/Plin':>8} {'sigma':>8} {'dev/sig':>8} {'Nmodes':>10}")
    for a, b, c, d in zip(k, ratio, sig, w):
        print(f"{a:8.4f} {b:8.4f} {c:8.4f} {(b - 1) / c:+8.2f} {d:10.0f}")
    print("chi2/dof", chi2, out["dof"], chi2 / out["dof"])
    print("mean P_hat/P_lin, k<0.5:", out["mean_k_lt_0p5"], " all k:", out["mean_all"])
    print("max |table/analytic EH98 - 1| (bin-averaged):",
          out["max_abs_table_over_analytic_minus_1"])
    return out


# ---------------------------------------------------------------------------
# lpt, div
# ---------------------------------------------------------------------------

def _rms_compare(mine, staged):
    staged = staged.astype(np.float64)
    d = mine - staged
    rms = float(np.sqrt(np.mean(staged ** 2)))
    return dict(rms_staged=rms, rms_diff_over_rms=float(np.sqrt(np.mean(d ** 2))) / rms,
                max_abs_diff_over_rms=float(np.max(np.abs(d))) / rms)


def run_lpt(st, args):
    N, W = st.n, args.workers
    D1, D2 = st.growth(_cosmo())
    K = kgrid(N, st.box)
    k2 = K[0] ** 2 + K[1] ** 2 + K[2] ** 2
    k2[0, 0, 0] = 1.0

    def irfft(x):
        return sfft.irfftn(x, s=(N, N, N), workers=W)

    res = {}
    dk = sfft.rfftn(st.load("delta.npy").astype(np.float64), workers=W)
    dk[0, 0, 0] = 0.0
    psi1 = []
    for ax in range(3):
        p = irfft(dk * (1j * K[ax] / k2))
        res[f"psi1_{ax}"] = _rms_compare(p, st.load(f"psi1_{ax}.npy"))
        print(f"psi1_{ax}", res[f"psi1_{ax}"], flush=True)
        psi1.append(p)

    # second-order source: sum_{i<j} phi_ii phi_jj - phi_ij^2, phi_ij(k) = -k_i k_j / k^2 delta_k
    def phi(i, j):
        return irfft(dk * (-(K[i] * K[j]) / k2))

    xx, yy = phi(0, 0), phi(1, 1)
    src = xx * yy
    zz = phi(2, 2)
    src += xx * zz
    src += yy * zz
    del xx, yy, zz
    for i, j in ((0, 1), (0, 2), (1, 2)):
        t = phi(i, j)
        src -= t * t
    del t
    print("delta2 mean", src.mean(), "rms", src.std(), flush=True)
    s2k = sfft.rfftn(src, workers=W)
    del src
    for ax in range(3):
        p2 = irfft(s2k * (1j * K[ax] / k2))
        res[f"psi2_{ax}"] = _rms_compare(p2, st.load(f"psi2_{ax}.npy"))
        print(f"psi2_{ax}", res[f"psi2_{ax}"], flush=True)
        u = D1 * psi1[ax] - D2 * p2
        res[f"u_{ax}"] = _rms_compare(u, st.load(f"u_{ax}.npy"))
        print(f"u_{ax}", res[f"u_{ax}"], flush=True)
        del p2, u
    del psi1
    # Zel'dovich identity on the staged field: div psi1 = -delta
    div = np.zeros_like(dk)
    for ax in range(3):
        div += 1j * K[ax] * sfft.rfftn(st.load(f"psi1_{ax}.npy").astype(np.float64), workers=W)
    divx = irfft(div)
    dl = st.load("delta.npy").astype(np.float64)
    res["div_psi1_plus_delta"] = dict(
        rms_over_rms_delta=float(np.sqrt(np.mean((divx + dl) ** 2)) / dl.std()))
    print("div psi1 + delta", res["div_psi1_plus_delta"])
    res["D1"], res["D2"], res["D2_over_D1sq"] = D1, D2, D2 / D1 ** 2
    return res


def run_div(st, args):
    N, W = st.n, args.workers
    K = kgrid(N, st.box)
    dk = sfft.rfftn(st.load("delta.npy").astype(np.float64), workers=W)
    r = dk.copy()
    for ax in range(3):
        r += 1j * K[ax] * sfft.rfftn(st.load(f"psi1_{ax}.npy").astype(np.float64), workers=W)
    nyq = np.zeros(r.shape, bool)
    nyq[N // 2, :, :] = True
    nyq[:, N // 2, :] = True
    nyq[:, :, -1] = True
    a2 = np.abs(dk) ** 2
    r2 = np.abs(r) ** 2
    out = dict(resid_power_frac_all=float(r2.sum() / a2.sum()),
               resid_power_frac_off_nyquist=float(r2[~nyq].sum() / a2[~nyq].sum()),
               resid_power_frac_on_nyquist=float(r2[nyq].sum() / a2[nyq].sum()),
               delta_power_frac_on_nyquist=float(a2[nyq].sum() / a2.sum()))
    print(out)
    return out


# ---------------------------------------------------------------------------
# skewness: Eulerian CIC density of displaced particles, smoothing, sub-cube moments
# ---------------------------------------------------------------------------

def paint_displaced(st, disp, m):
    """CIC density contrast on an m^3 mesh of the particles at q + disp(lo, hi), where `disp`
    returns three float64 displacement arrays for Lagrangian x-planes lo:hi."""
    N, L = st.n, st.box
    h = L / m
    rho = np.zeros(m ** 3)
    q1 = np.arange(N) * (L / N)
    for lo in range(0, N, 16):
        hi = lo + 16
        d = disp(lo, hi)
        g = []
        for ax in range(3):
            q = (q1[lo:hi, None, None] if ax == 0 else q1[None, :, None] if ax == 1
                 else q1[None, None, :])
            g.append(np.mod((q + d[ax]) / h, m).ravel())
        i0 = [np.floor(x).astype(np.int64) for x in g]
        f = [x - i for x, i in zip(g, i0)]
        idx, wts = [], []
        for cx in (0, 1):
            for cy in (0, 1):
                for cz in (0, 1):
                    idx.append(((i0[0] + cx) % m) * m * m + ((i0[1] + cy) % m) * m
                               + (i0[2] + cz) % m)
                    wts.append((f[0] if cx else 1 - f[0]) * (f[1] if cy else 1 - f[1])
                               * (f[2] if cz else 1 - f[2]))
        rho += np.bincount(np.concatenate(idx), weights=np.concatenate(wts), minlength=m ** 3)
    rho = rho.reshape(m, m, m)
    return rho / rho.mean() - 1.0


def smoothed(field, box, workers, cic):
    """{R: top-hat smoothed field} for R in RS; `cic` divides the CIC window out first."""
    n = field.shape[0]
    fk = sfft.rfftn(field, workers=workers)
    K = kgrid(n, box)
    kk = np.sqrt(K[0] ** 2 + K[1] ** 2 + K[2] ** 2)
    if cic:
        hh = box / n
        win = np.ones_like(kk)
        for c in K:
            win = win * np.sinc(c * hh / (2 * np.pi)) ** 2
        fk = fk / win
    fk[0, 0, 0] = 0.0
    return {R: sfft.irfftn(fk * tophat(kk * R), s=(n, n, n), workers=workers) for R in RS}


def sub_means(nsub, fn, *fields):
    """Means of fn(*sub-blocks of fields) over the nsub^3 sub-cubes, in C order. The block is
    sliced before `fn` is applied; slicing a precomputed product instead sums in another order
    (last-bit differences)."""
    b = fields[0].shape[0] // nsub
    st = range(0, fields[0].shape[0], b)
    return np.array([np.mean(fn(*(f[i:i + b, j:j + b, k:k + b] for f in fields)))
                     for i in st for j in st for k in st])


def _ident(x):
    return x


def _sq(x):
    return x ** 2


def _cube(x):
    return x ** 3


def _sq_times(a, b):
    return a ** 2 * b


def s3_jack(m2, m3):
    """S3 = <d^3> / <d^2>^2 from sub-cube moments, its jackknife error and leave-one-outs."""
    s3 = m3.mean() / m2.mean() ** 2
    n = m2.size
    loo = np.array([np.delete(m3, i).mean() / np.delete(m2, i).mean() ** 2 for i in range(n)])
    return float(s3), float(jackknife_err(loo)), loo


def run_s3(st, args):
    from scipy.integrate import quad

    from inexor.cosmology import linear_power

    N, L, W, M, nsub = st.n, st.box, args.workers, S3_MESH, args.nsub
    cos = _cosmo()
    D1, D2 = st.growth(cos)

    def moments(fields):
        return {R: (sub_means(nsub, _sq, d), sub_means(nsub, _cube, d))
                for R, d in fields.items()}

    # gamma1 = dln sigma^2 / dln R from linear theory (continuous) and from the box's modes
    def sig2_cont(R):
        def f(lnk):
            k = np.exp(lnk)
            return k ** 3 * linear_power(k, cos)[0] * tophat(k * R) ** 2
        return quad(f, np.log(1e-4), np.log(50.0), limit=400)[0] / (2 * np.pi ** 2)

    def dlog(fn, R, e=0.01):
        return (np.log(fn(R * (1 + e))) - np.log(fn(R * (1 - e)))) / (np.log(1 + e)
                                                                       - np.log(1 - e))

    lin = st.load("delta.npy").astype(np.float64) * D1
    mlin = moments(smoothed(lin, L, W, cic=False))
    lk = sfft.rfftn(lin, workers=W)
    K = kgrid(N, L)
    kk = np.sqrt(K[0] ** 2 + K[1] ** 2 + K[2] ** 2)
    wz = np.full(N // 2 + 1, 2.0)
    wz[0] = 1
    wz[-1] = 1
    pw = (np.abs(lk) ** 2) * wz[None, None, :]
    pw[0, 0, 0] = 0

    def sig2_real(R, pw=pw, kk=kk):
        return float(np.sum(pw * tophat(kk * R) ** 2)) / N ** 6

    g1c = {R: dlog(sig2_cont, R) for R in RS}
    g1r = {R: dlog(sig2_real, R) for R in RS}
    del sig2_real
    s2c = {R: sig2_cont(R) * D1 ** 2 for R in RS}
    del lk, pw, kk, K, lin

    def za(lo, hi):
        return [D1 * np.asarray(st.load(f"psi1_{ax}.npy", mmap=True)[lo:hi], np.float64)
                for ax in range(3)]

    def lpt2(lo, hi):
        return [np.asarray(st.load(f"u_{ax}.npy", mmap=True)[lo:hi], np.float64)
                for ax in range(3)]

    print("painting ZA", flush=True)
    mza = moments(smoothed(paint_displaced(st, za, M), L, W, cic=True))
    print("painting 2LPT", flush=True)
    m2l = moments(smoothed(paint_displaced(st, lpt2, M), L, W, cic=True))

    rows = []
    dS3_pred = -2 * D2 / D1 ** 2
    for R in RS:
        sl, el, _ = s3_jack(*mlin[R])
        sz, ez, lz = s3_jack(*mza[R])
        s2, e2, l2 = s3_jack(*m2l[R])
        dS = s2 - sz
        edS = float(jackknife_err(l2 - lz))
        row = dict(R=R, sigma2_lin_meas=float(mlin[R][0].mean()), sigma2_lin_theory=s2c[R],
                   sigma2_za=float(mza[R][0].mean()), sigma2_2lpt=float(m2l[R][0].mean()),
                   gamma1_theory=g1c[R], gamma1_realization=g1r[R],
                   S3_lin=sl, S3_lin_err=el,
                   S3_za=sz, S3_za_err=ez, S3_za_pred=4 + g1c[R], S3_za_pred_real=4 + g1r[R],
                   S3_2lpt=s2, S3_2lpt_err=e2, S3_2lpt_pred=4 + g1c[R] + dS3_pred,
                   S3_2lpt_pred_real=4 + g1r[R] + dS3_pred,
                   dS3=dS, dS3_err=edS, dS3_pred=dS3_pred)
        rows.append(row)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                          for k, v in row.items()}), flush=True)
    return dict(D1=D1, D2=D2, D2_over_D1sq=D2 / D1 ** 2, mesh=M, cic_deconvolved=True,
                jackknife=f"{nsub**3} sub-cubes, leave-one-out", rows=rows)


def tree_moments(st, D1, D2, nsub, workers, product_first=False):
    """{R: (sub-cube <d1_R^2>, <d1_R^2 s_R>)} on the Lagrangian mesh, d1 = D1 delta,
    s = D2 div psi2 (the 2LPT term's leading contribution to the Eulerian density).
    `product_first` forms the products on the whole mesh before the sub-cube means."""
    N, L = st.n, st.box
    K = kgrid(N, L)
    kk = np.sqrt(K[0] ** 2 + K[1] ** 2 + K[2] ** 2)
    d1k = D1 * sfft.rfftn(st.load("delta.npy").astype(np.float64), workers=workers)
    sk = np.zeros_like(d1k)
    for ax in range(3):
        sk += D2 * 1j * K[ax] * sfft.rfftn(st.load(f"psi2_{ax}.npy").astype(np.float64),
                                           workers=workers)
    d1k[0, 0, 0] = 0
    sk[0, 0, 0] = 0
    out = {}
    for R in RS:
        w = tophat(kk * R)
        a = sfft.irfftn(d1k * w, s=(N,) * 3, workers=workers)
        b = sfft.irfftn(sk * w, s=(N,) * 3, workers=workers)
        if product_first:
            out[R] = (sub_means(nsub, _ident, a ** 2), sub_means(nsub, _ident, a ** 2 * b))
        else:
            out[R] = (sub_means(nsub, _sq, a), sub_means(nsub, _sq_times, a, b))
    return out


def run_ds3_tree(st, args):
    D1, D2 = st.growth(_cosmo())
    pred = -2 * D2 / D1 ** 2
    rows = []
    for R, (m2, m12) in tree_moments(st, D1, D2, args.nsub, args.workers).items():
        val = 3 * m12.mean() / m2.mean() ** 2
        n = m2.size
        loo = np.array([3 * np.delete(m12, i).mean() / np.delete(m2, i).mean() ** 2
                        for i in range(n)])
        rows.append(dict(R=R, dS3_tree_realization=float(val), err=float(jackknife_err(loo)),
                         dS3_tree_expectation=pred))
        print(rows[-1], flush=True)
    return dict(NSUB=args.nsub, rows=rows)


def run_ds3_paired(st, args):
    W, M, nsub = args.workers, S3_MESH, args.nsub
    D1_0, D2_0 = st.growth(_cosmo())
    psi1 = [st.load(f"psi1_{ax}.npy", mmap=True) for ax in range(3)]
    psi2 = [st.load(f"psi2_{ax}.npy", mmap=True) for ax in range(3)]
    # the tree term is amplitude independent: unit amplitude
    tree = tree_moments(st, D1_0, D2_0, nsub, W, product_first=True)

    def displaced(c1, c2):
        def disp(lo, hi):
            out = []
            for ax in range(3):
                d = c1 * np.asarray(psi1[ax][lo:hi], np.float64)
                if c2:
                    d = d + c2 * np.asarray(psi2[ax][lo:hi], np.float64)
                out.append(d)
            return out
        return disp

    def stats(m):
        """(dS3, tree, remainder) from per-sub-volume means m."""
        def s3(m2, m3):
            return m3.mean() / m2.mean() ** 2
        d = s3(m["z2_2"], m["z3_2"]) - s3(m["z2_z"], m["z3_z"])
        t = 3 * m["t12"].mean() / m["t2"].mean() ** 2
        return np.array([d, t, d - t])

    out = []
    for amp in args.amps:
        D1, D2 = D1_0 * amp, D2_0 * amp ** 2
        za = smoothed(paint_displaced(st, displaced(D1, 0.0), M), st.box, W, cic=True)
        lpt = smoothed(paint_displaced(st, displaced(D1, -D2), M), st.box, W, cic=True)
        for R in RS:
            m = dict(z2_z=sub_means(nsub, _ident, za[R] ** 2),
                     z3_z=sub_means(nsub, _ident, za[R] ** 3),
                     z2_2=sub_means(nsub, _ident, lpt[R] ** 2),
                     z3_2=sub_means(nsub, _ident, lpt[R] ** 3),
                     t2=tree[R][0], t12=tree[R][1])
            val = stats(m)
            n = m["t2"].size
            loo = np.array([stats({k: np.delete(v, i) for k, v in m.items()})
                            for i in range(n)])
            err = jackknife_err(loo)
            row = dict(amp=amp, R=R, sigma2_za=float(m["z2_z"].mean()), dS3=float(val[0]),
                       dS3_err=float(err[0]), tree_real=float(val[1]), remainder=float(val[2]),
                       remainder_err=float(err[2]),
                       remainder_over_sigma2=float(val[2] / m["z2_z"].mean()))
            out.append(row)
            print(json.dumps({k: round(v, 5) for k, v in row.items()}), flush=True)
    return dict(NSUB=nsub, D1=D1_0, D2=D2_0, rows=out)


RUNNERS = {"pk-lin": run_pk_lin, "lpt": run_lpt, "div": run_div, "s3": run_s3,
           "ds3-tree": run_ds3_tree, "ds3-paired": run_ds3_paired}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=sorted(RUNNERS))
    ap.add_argument("--ic-dir", default="runs/validation/cgh64",
                    help="IC directory made with --keep-stage (default %(default)s)")
    ap.add_argument("--out", default=None,
                    help="output JSON (default runs/validation/icval/<what>[_nsubN].json)")
    ap.add_argument("--workers", type=int, default=4, help="scipy.fft workers")
    ap.add_argument("--nsub", type=int, default=4,
                    help="s3, ds3-tree, ds3-paired: jackknife sub-cubes per side")
    ap.add_argument("--amps", type=float, nargs="+", default=[1.0, 0.5, 0.25],
                    help="ds3-paired: displacement amplitudes")
    args = ap.parse_args()
    st = Stage(args.ic_dir)
    out = args.out
    if out is None:
        tag = args.what.replace("-", "_")
        if args.what in ("s3", "ds3-tree", "ds3-paired"):
            tag += f"_nsub{args.nsub}"
        out = os.path.join("runs", "validation", "icval", f"{tag}.json")
    t0 = time.perf_counter()
    res = RUNNERS[args.what](st, args)
    write_json(out, res)
    print(f"{args.what}: {time.perf_counter() - t0:.0f} s, peak RSS {maxrss() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
