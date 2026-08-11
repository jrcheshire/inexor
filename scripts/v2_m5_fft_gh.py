"""M-v2-5 leg VI: the out-of-core FFT layer at 2048^3 on one GH200 node.

THE CLAIM UNDER TEST (D-v2-15 clause 4): a 2048^3 real-to-complex transform
that `jnp.fft.rfftn` cannot fit on this device (measured ceiling between
1024^3 and 1536^3, workspace 7x the field) runs through ooc_fft's
host-resident factorization inside the ~116 GB host budget, correctly.

No monolithic reference exists at this scale BY CONSTRUCTION, so correctness
is asserted by checks that need none:

  roundtrip   forward(white) then inverse, compared PER SLAB against the
              re-generated white noise (the construction is random-access,
              so the reference is re-derived, never stored):
              max|delta| / rms(white) <= 1e-5 at f32 (the eps*sqrt(log N)
              class; the f64 floor measured at dev scale is ~1e-15 and the
              f32 unit-test floor ~1e-7 -- 1e-5 is 100x that, a bar not a fit).
  Parseval    sum(w^2) (threaded plane-ordered reduction) vs sum|W|^2 / N^3
              to 1e-6 relative at f32.
  P(k)        the coloured spectrum binned hermitian-weighted against the
              table's own P: per-bin |P/P_lin - 1| < 5 sqrt(2/N_modes) over
              k <= 0.5 k_Nyq. At 2048^3 the bins carry 1e4-1e7 modes, so this
              is a 0.1-1%-level test of the colour at scale -- a swapped axis
              or a mis-binned table reads at many sigma.
  refusal     require_fits(2048, f64, "derivative", 116 GB) must RAISE (the
              f64 derivative plan is over this host by design).

Plus, before any 2048^3 work:

  invariance  the white field and the coloured spectrum are bitwise invariant
              to slab thickness AT 256^3 ON THIS MACHINE (the gate's leg I
              re-asserted where the job runs; noise is pinned to the CPU
              backend by ic.white_plane, so CUDA cannot fork the stream);
              the plane-0 fingerprint is RECORDED for cross-machine
              comparison (reported, not gated: XLA-CPU erfinv bits across
              x86/aarch64/arm64 are unverified, pre-registered as a finding
              either way).
  io-probe    one 8 GB slab written and re-read on the working filesystem,
              timed, BEFORE the big phase commits (the staging rate here is
              unmeasured; the plan budgets ~1.25 TB of traffic at a full
              C-gh generation, which this leg does NOT run).

MEMORY ACCOUNTING: ru_maxrss (host peak) reported per phase against
plan_bytes' prediction; the job FAILS if measured exceeds predicted by more
than 1.3x (the accounting function exists to be checked, not trusted).

Run on a Vista gh node (the FFTs are host-CPU; the GH200's role here is
being the production node whose host budget the claim is about):
  pixi run -e gpu python scripts/v2_m5_fft_gh.py --n 2048 --workdir $SCRATCH/m5_fft
"""

import argparse
import hashlib
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(REPO, "runs", "v2")
SEED = 0


def _maxrss():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


def phase_invariance_256(res):
    import jax

    from inexor import ic, ooc_fft
    from inexor.config import Cosmology
    from inexor.cosmology import ic_k_table

    n = 256
    key = jax.random.PRNGKey(SEED)
    cosmo = Cosmology()
    ref = ic.white_noise(key, n, np.float64)
    ok = True
    for t in (1, 7, n):
        parts = [ic.white_slab(key, lo, min(lo + t, n), n, np.float64) for lo in range(0, n, t)]
        ok &= bool(np.array_equal(np.concatenate(parts), ref))
    spec_a = ooc_fft.forward_from_slabs(lambda lo, hi: ref[lo:hi], n, slab=7)
    spec_b = ooc_fft.rfftn_ooc(ref)
    tab = ic_k_table(cosmo, n, float(n))
    for s in (spec_a, spec_b):
        ooc_fft.mul_radial_inplace(s, n, float(n), ic._colour_fn(tab, n, float(n)),
                                   dc_value=0.0, slab=13)
    ok &= bool(np.array_equal(spec_a, spec_b))
    res["invariance_256"] = dict(ok=bool(ok))
    res["plane0_sha256_n64_f64"] = hashlib.sha256(
        ic.white_plane(key, 0, 64, np.float64).tobytes()
    ).hexdigest()
    print(f"  invariance at 256^3 on this machine: {'ok' if ok else 'FAIL'}", flush=True)
    print(f"  plane-0 fingerprint (cross-machine, reported): "
          f"{res['plane0_sha256_n64_f64'][:16]}...", flush=True)
    return ok


def phase_io_probe(res, workdir):
    from inexor import ooc_fft

    os.makedirs(workdir, exist_ok=True)
    n_planes, n = 512, 2048  # 512 x 2048^2 f32 = 8.6 GB
    sa = ooc_fft.StagedArray.create(os.path.join(workdir, "ioprobe.npy"),
                                    np.float32, (n_planes, n, n))
    blk = np.ones((64, n, n), dtype=np.float32)
    t0 = time.perf_counter()
    for lo in range(0, n_planes, 64):
        sa.write_slab(lo, blk)
    t1 = time.perf_counter()
    for lo in range(0, n_planes, 64):
        sa.read_slab(lo, lo + 64)
    t2 = time.perf_counter()
    gb = n_planes * n * n * 4 / 1e9
    res["io_probe"] = dict(gb=gb, write_s=t1 - t0, read_s=t2 - t1,
                           write_gbs=gb / (t1 - t0), read_gbs=gb / (t2 - t1))
    os.remove(os.path.join(workdir, "ioprobe.npy"))
    print(f"  io probe: {gb:.1f} GB write {gb / (t1 - t0):.2f} GB/s, "
          f"read {gb / (t2 - t1):.2f} GB/s", flush=True)
    return True


def phase_roundtrip(res, n):
    import jax

    from inexor import ic, ooc_fft

    key = jax.random.PRNGKey(SEED)
    plan = ooc_fft.plan_bytes(n, np.float32, "forward")
    print(f"  plan: forward peak {plan['peak'] / 1e9:.1f} GB predicted", flush=True)

    tot = [0.0]

    def slab_fn(lo, hi):
        s = ic.white_slab(key, lo, hi, n, np.float32)
        tot[0] = ic.sq_sum_by_plane(s, tot[0])
        return s

    t0 = time.perf_counter()
    spec = ooc_fft.forward_from_slabs(slab_fn, n, slab=32)
    t_fwd = time.perf_counter() - t0

    # Parseval on the spectrum, hermitian-weighted, slab-streamed in f64
    m = n // 2 + 1
    acc = 0.0
    for lo in range(0, n, 32):
        hi = min(lo + 32, n)
        p = (spec[lo:hi].real.astype(np.float64) ** 2
             + spec[lo:hi].imag.astype(np.float64) ** 2)
        wgt = np.full(m, 2.0)
        wgt[0] = 1.0
        if n % 2 == 0:
            wgt[-1] = 1.0
        acc += float(np.sum(p * wgt))
    parseval_rel = abs(acc / n**3 - tot[0]) / tot[0]

    t0 = time.perf_counter()
    max_d = 0.0
    for lo, s in ooc_fft.inverse_to_slabs(spec, n, slab=32):
        w = ic.white_slab(key, lo, lo + s.shape[0], n, np.float32)
        max_d = max(max_d, float(np.max(np.abs(s - w))))
    t_inv = time.perf_counter() - t0
    rms = float(np.sqrt(tot[0] / n**3))
    rel = max_d / rms

    peak = _maxrss()
    res["roundtrip"] = dict(
        n=n, fwd_s=t_fwd, inv_s=t_inv, max_abs_delta=max_d, rms=rms, rel=rel,
        parseval_rel=parseval_rel, peak_rss=peak,
        plan_peak=plan["peak"],
        rss_vs_plan=peak / plan["peak"],
    )
    ok = rel <= 1e-5 and parseval_rel < 1e-6 and peak < 1.3 * plan["peak"]
    print(f"  roundtrip: fwd {t_fwd:.1f} s, inv {t_inv:.1f} s, max|d|/rms {rel:.2e} "
          f"(bar 1e-5), Parseval {parseval_rel:.2e} (bar 1e-6)", flush=True)
    print(f"  peak rss {peak / 1e9:.1f} GB vs plan {plan['peak'] / 1e9:.1f} GB "
          f"(x{peak / plan['peak']:.2f}, bar 1.3)", flush=True)
    return ok


def phase_coloured_pk(res, n):
    import jax

    from inexor import ic, ooc_fft
    from inexor.config import Cosmology
    from inexor.cosmology import ic_k_table

    key = jax.random.PRNGKey(SEED)
    cosmo = Cosmology()
    box = float(n) / 2.0  # spacing 0.5 Mpc/h, the config-table convention
    tab = ic_k_table(cosmo, n, box)
    t0 = time.perf_counter()
    spec = ooc_fft.forward_from_slabs(
        lambda lo, hi: ic.white_slab(key, lo, hi, n, np.float32), n, slab=32
    )
    ooc_fft.mul_radial_inplace(spec, n, box, ic._colour_fn(tab, n, box),
                               dc_value=0.0, slab=32)
    t_gen = time.perf_counter() - t0

    # hermitian-weighted radial P(k), slab-streamed, f64 accumulation
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=box / n)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=box / n)
    k_nyq = np.pi * n / box
    edges = np.linspace(0.0, 0.5 * k_nyq, 65)
    m = n // 2 + 1
    wgt_z = np.full(m, 2.0)
    wgt_z[0] = 1.0
    if n % 2 == 0:
        wgt_z[-1] = 1.0
    psum = np.zeros(len(edges) - 1)
    ksum = np.zeros(len(edges) - 1)
    wsum = np.zeros(len(edges) - 1)
    for lo in range(0, n, 32):
        hi = min(lo + 32, n)
        kk = np.sqrt(kx[lo:hi].reshape(-1, 1, 1) ** 2 + kx.reshape(1, n, 1) ** 2
                     + kz.reshape(1, 1, -1) ** 2)
        p = (spec[lo:hi].real.astype(np.float64) ** 2
             + spec[lo:hi].imag.astype(np.float64) ** 2)
        w3 = np.broadcast_to(wgt_z, kk.shape)
        idx = np.digitize(kk.ravel(), edges) - 1
        sel = (idx >= 0) & (idx < len(psum)) & (kk.ravel() > 0)
        np.add.at(psum, idx[sel], (p * w3).ravel()[sel])
        np.add.at(ksum, idx[sel], (kk * w3).ravel()[sel])
        np.add.at(wsum, idx[sel], w3.ravel()[sel])
    del spec
    good = wsum > 100
    pk = psum[good] / wsum[good] * box**3 / n**6
    kmean = ksum[good] / wsum[good]
    p_lin = tab.P_of_k(kmean)
    z = (pk / p_lin - 1.0) / np.sqrt(2.0 / wsum[good])
    peak = _maxrss()
    res["coloured_pk"] = dict(
        n=n, box=box, gen_s=t_gen, bins=int(good.sum()),
        max_abs_z=float(np.abs(z).max()), peak_rss=peak,
    )
    ok = bool(np.all(np.abs(z) < 5.0))
    print(f"  coloured P(k): {int(good.sum())} bins to 0.5 k_Nyq, "
          f"max|z| {np.abs(z).max():.2f} (bar 5)", flush=True)
    return ok


def phase_f64_refusal(res):
    from inexor import ooc_fft

    try:
        ooc_fft.require_fits(2048, np.float64, "derivative", budget_bytes=116e9)
    except MemoryError as e:
        res["f64_refusal"] = dict(ok=True, message=str(e)[:200])
        print("  f64 derivative plan at 2048^3 REFUSES a 116 GB budget: ok", flush=True)
        return True
    res["f64_refusal"] = dict(ok=False)
    print("  f64 refusal DID NOT FIRE", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--workdir", default=os.environ.get("SCRATCH", "/tmp"))
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)

    res = {}
    ok = True
    for name, fn in (
        ("invariance_256", lambda: phase_invariance_256(res)),
        ("io_probe", lambda: phase_io_probe(res, os.path.join(args.workdir, "m5_ioprobe"))),
        ("f64_refusal", lambda: phase_f64_refusal(res)),
        ("roundtrip", lambda: phase_roundtrip(res, args.n)),
        ("coloured_pk", lambda: phase_coloured_pk(res, args.n)),
    ):
        print(f"phase {name}", flush=True)
        try:
            got = fn()
        except Exception as e:  # a dead phase must not silently pass the job
            print(f"  phase {name} DIED: {e!r}", flush=True)
            got = False
            res[name] = dict(ok=False, error=repr(e))
        ok &= bool(got)
    res["ok"] = bool(ok)

    sys.path.insert(0, HERE)
    import v2_m5_ic_gate as m5

    class _A:
        leg = "fft-gh"
        out_suffix = args.out_suffix
        n = args.n
        bricks = None
        slab = 32
        seed = SEED
        f_nl = 0.0

    m5._write(res, _A())
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
