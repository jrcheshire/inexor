"""M-v2-5 leg V: the memory ladder -- does the streamed generator kill the 90 B/p term?

THE ESTIMAND: net-of-baseline peak host RSS per particle (ru_maxrss, one
subprocess per (arm, rung, phase) because a high-water mark never resets --
the v4d method verbatim), on the CPU BACKEND ONLY (on CUDA the arrays live in
VRAM and every host ratio reads 1.0 -- deneb 409's defect; this probe REFUSES
a non-CPU backend rather than measuring the wrong pool).

Arms, all at fdtype float32 (the production dtype, and the instrument class
job 896159's record is stated in):

  old_mirror    the licensed pre-M-v2-5 colour path (v2_m5_ic_gate's mirror).
                Its own control: it must REPRODUCE D-v2-15 clause 1's ~90 B/p
                flat term (126.5 / 95.4 / 89.6 recorded at 256/512/1024, with
                the caveat that the record was host-isolated-on-GPU and these
                are CPU-backend totals -- the within-instrument comparison
                against the new arms is what is honest here).
  new_mono      the monolithic conveniences (gaussian_delta, linear_density,
                zeldovich, lpt_ics). NOT the product: the (N^3, 3) arrays
                dominate at ~70-90 B/p and that is expected and reported.
  new_streamed  generate_t9_slabs end to end. THE DELIVERABLE.

PRE-REGISTERED (2026-08-10, before any antares rung ran; stated honestly:
the FIT FORM below was chosen after laptop smokes at n = 128/256/512 showed
a sub-cubic overhead term -- net 0.39 / 0.95 / 2.09 GB against a pure-payload
prediction of 0.03 / 0.35 / 1.3 -- and the m4 ladder's decomposition pattern
was adopted so that term is measured instead of polluting a raw per-particle
reading; the BAR VALUES were fixed before the fit ever ran on cluster data):

  The verdict fits net(n) = A * n^3 + C over the rungs (residuals reported;
  a bad fit is a finding, not a shrug).

  Tier 1 -- the BAR: the cubic coefficient A <= 9.0 B/p AND the extrapolated
  2048^3 total (A * 2048^3 + C) < 77.3 GB = 116 GB / 1.5, the margin the plan
  requires against the GH200 host cliff.

  Tier 2 -- the EXPECTATION: A ~= 8.0 B/p. Derivation: the peak phase holds
  TWO complex64 spectra (the source and one deriv2/grad copy) at ~n^3/2
  elements x 8 B = 4 B/p each; every other design term is O(slab) or
  O(n^2)-class and lands in C. A > 12 is a FINDING (an uncounted CUBIC
  term -- the project's recurring defect class); C > 3 GB is a separate
  finding (the churn term outgrowing its laptop measurement); the old arm
  failing to reproduce ~90 B/p impeaches the instrument, not the code under
  test. NB glibc arena retention (umbrella memory) may move C between macOS
  and Linux; A is the transferable number.

  new_mono is expected at 70-90 B/p (dominated by psi1/psi2/q/u/x at 12 B/p
  each) -- deliberately not gated; it exists so nobody mistakes the monolithic
  convenience for the memory product.

  LAPTOP SMOKE CHARACTERIZED (macOS, 2026-08-10, in the card _smoke): the
  fit read A = 12.45 there, and the excess was PINNED to the allocator, not
  an array -- tracemalloc peak 0.213 GB at n=256 against an RSS net of 0.95,
  with each phase individually on design (forward+grad measured 0.24-0.26 vs
  0.27 predicted, numpy-noise and jax-noise arms alike). macOS retains
  freed spec-sized buffers phase over phase (5 spec-equivalents at n=256,
  1.7 at n=512 -- sub-cubic); glibc munmaps large frees, so antares is
  expected to read A ~= 8. A > 9 ON ANTARES is therefore a real finding
  about the code, not the laptop's allocator, and gets chased.

CONTENT GUARD (the 409 -> 415 lesson pair): the TOP rung must separate the
arms (new_streamed < 0.5x old_mirror there); a probe where every arm reads
alike at the top is measuring its own baseline, not the code.

Usage:
  pixi run python scripts/v2_m5_memladder.py --rungs 256 512 1024
  pixi run python scripts/v2_m5_memladder.py --worker old_mirror:colour:256   # internal
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO, "runs", "v2")
SEED = 0

ARMS = ("old_mirror", "new_mono", "new_streamed")
PHASES = {
    "old_mirror": ("baseline", "colour", "linear_density"),
    "new_mono": ("baseline", "colour", "linear_density", "psi1", "full"),
    "new_streamed": ("baseline", "full"),
}


def _maxrss_bytes():
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


def _worker(arm, phase, n):
    import jax

    jax.config.update("jax_enable_x64", True)
    if jax.devices()[0].platform != "cpu":
        raise SystemExit(
            "FATAL: non-CPU backend. ru_maxrss is HOST memory; on CUDA the arrays "
            "live in VRAM and every ratio reads exactly 1.0 (deneb 409). Run this "
            "probe in the CPU env."
        )
    import jax.numpy as jnp

    from inexor.config import Cosmology

    cosmo = Cosmology()
    key = jax.random.PRNGKey(SEED)
    # backend init in EVERY phase, so 'baseline' nets it out of the others
    import inexor.ic as ic

    ic.white_plane(key, 0, 8, np.float32)

    if phase == "baseline":
        pass
    elif arm == "old_mirror":
        sys.path.insert(0, HERE)
        import v2_m5_ic_gate as m5

        if phase == "colour":
            m5._old_gaussian_delta(key, n, float(n), cosmo, jnp.float32)
        elif phase == "linear_density":
            m5._old_linear_density(key, n, float(n), cosmo, f_NL=0.0, fdtype=jnp.float32)
    elif arm == "new_mono":
        from inexor import lpt

        if phase == "colour":
            ic.gaussian_delta(key, n, float(n), cosmo, fdtype=np.float32)
        elif phase == "linear_density":
            ic.linear_density(key, n, float(n), cosmo, fdtype=np.float32)
        elif phase == "psi1":
            d0 = ic.linear_density(key, n, float(n), cosmo, fdtype=np.float32)
            lpt.zeldovich_displacement(d0, float(n), np.float32)
        elif phase == "full":
            d0 = ic.linear_density(key, n, float(n), cosmo, fdtype=np.float32)
            with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as wd:
                lpt.lpt_ics(d0, float(n), 0.1, cosmo, order=2, fdtype=np.float32,
                            resident="low", workdir=wd)
    elif arm == "new_streamed":
        from inexor import icgen

        nb = max(4, n // 8)  # production-shaped brick count (brick slab ~8 planes)
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as wd:
            icgen.generate_t9_slabs(wd, key, n, float(n), cosmo, 0.1, nb,
                                    fdtype=np.float32, slab=32)
    print(json.dumps(dict(maxrss=_maxrss_bytes())), flush=True)


def _run(arm, phase, n):
    out = subprocess.check_output(
        [sys.executable, os.path.abspath(__file__), "--worker", f"{arm}:{phase}:{n}"],
        text=True, cwd=REPO,
    )
    return json.loads(out.strip().splitlines()[-1])["maxrss"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rungs", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        arm, phase, n = args.worker.split(":")
        _worker(arm, phase, int(n))
        return 0

    res = dict(rungs={}, fdtype="float32", method="one subprocess per (arm, rung, phase)")
    for n in sorted(args.rungs):  # cheapest first: a top-rung OOM still yields the rest
        rung = {}
        for arm in ARMS:
            base = None
            for phase in PHASES[arm]:
                peak = _run(arm, phase, n)
                if phase == "baseline":
                    base = peak
                    rung[f"{arm}_baseline"] = peak
                else:
                    net = peak - base
                    bpp = net / n**3
                    rung[f"{arm}_{phase}"] = dict(peak=peak, net=net, bpp=bpp)
                    print(f"  n={n} {arm}:{phase}: peak {peak / 1e9:.2f} GB, "
                          f"net {net / 1e9:.2f} GB = {bpp:.1f} B/p", flush=True)
        res["rungs"][str(n)] = rung

    top = str(max(args.rungs))
    t = res["rungs"][top]
    streamed_top = t["new_streamed_full"]["bpp"]
    old = t["old_mirror_linear_density"]["bpp"]

    # the m4 decomposition: net(n) = A n^3 + C, gated on the CUBIC coefficient
    ns = np.array(sorted(args.rungs), dtype=np.float64)
    nets = np.array([res["rungs"][str(int(n))]["new_streamed_full"]["net"] for n in ns])
    X = np.stack([ns**3, np.ones_like(ns)], axis=1)
    (A, C), *_ = np.linalg.lstsq(X, nets, rcond=None)
    resid = nets - X @ np.array([A, C])
    scaled = A * 2048**3 + C
    res["verdict"] = dict(
        top_rung=int(top),
        streamed_bpp_top=streamed_top,
        old_bpp_top=old,
        fit_A_bpp=float(A),
        fit_C_gb=float(C / 1e9),
        fit_resid_gb=[float(r / 1e9) for r in resid],
        scaled_2048_gb=float(scaled / 1e9),
        tier1_pass=bool(A <= 9.0 and scaled < 77.3e9),
        tier2_finding=bool(A > 12.0 or C > 3e9),
        content_guard_arms_separate=bool(streamed_top < 0.5 * old),
    )
    v = res["verdict"]
    print(f"FIT over rungs: A = {A:.2f} B/p (bar 9.0), C = {C / 1e9:.2f} GB, "
          f"resid {['%.2f' % (r / 1e9) for r in resid]} GB", flush=True)
    print(f"extrapolated 2048^3 total {scaled / 1e9:.1f} GB (bar 77.3); "
          f"top rung raw {streamed_top:.1f} B/p vs old {old:.1f}", flush=True)
    ok = v["tier1_pass"] and v["content_guard_arms_separate"]
    res["ok"] = bool(ok)

    sys.path.insert(0, HERE)
    import v2_m5_ic_gate as m5

    class _A:
        leg = "memladder"
        out_suffix = args.out_suffix
        n = int(top)
        bricks = None
        slab = 32
        seed = SEED
        f_nl = 0.0

    m5._write(res, _A())
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
