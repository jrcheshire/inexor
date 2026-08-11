"""M-v2-5 exit gate: streamed ICs + the out-of-core FFT. Probe code, NOT package code.

Built in `v2_m3_engine_gate.py`'s mould (its `CONFIGS` imported for the engine
smoke). Exit gate B (the transfer-table bar) is `v2_m5_table_bar.py`'s card;
everything else lands here.

WHY NOT THE INSTRUMENT THE CHARTER NAMES. D-v2-18's ladder row says "tile-IC
identity vs monolithic at f64". It cannot be read literally: D-v2-15 clause 5
(ratified at the V4 freeze) establishes that `jax.random.normal`'s stream is
shape-dependent, so the new per-plane `fold_in` construction is NOT
bit-identical to the old monolithic stream for any seed -- there is no
old-monolithic field the new tiles could be identical TO. Third milestone
running whose written exit criterion could not be read literally (M-v2-3,
M-v2-4 precedent). The re-scope, JC-ratified 2026-08-10:

  (i)  DECOMPOSITION INVARIANCE of the new construction -- the same field,
       bitwise at f64, whatever slab thickness produced it (leg `invariance`,
       asserted END TO END: two full generations at different slab knobs must
       write byte-identical T9 files);
  (ii) END-TO-END IDENTITY -- the streamed generator through T9 slabs and the
       loader is bitwise the monolithic SlotState build of the same (new)
       field (leg `e2e`);
  (iii) old-vs-new agreement is STATISTICAL only (leg `stats`: identical
       ensembles by construction, so the comparison is plumbing, not physics).

THE ESTIMANDS, stated so they can be checked mechanically:

  ledger      colour touches no 3D |k| array: subprocess ru_maxrss of one
              gaussian_delta at n=256 f64, net of an import-only baseline,
              < 3x the field bytes (the old path paid ~30x: ~15 half-grid f64
              temporaries, D-v2-15 clause 1). Plus realized dtypes.
  invariance  n_diff == 0 between every pair of white-field assemblies at
              thickness {1, 7, N/2, N}, AND byte-equality of every T9 slab
              file + vel_scale + mean_phi2 across two full generations at
              slab 7 vs N (f_NL = 10, so the mean_phi2 reduction is in play).
              A different seed must break it (anti-vacuity).
  e2e         load_slot_state(generate_t9_slabs(...)) == SlotState.build(x, v)
              fed the monolithic chain: n_diff == 0 on off / w / occupancy /
              brick_start, vel_scale EXACTLY equal, arena empty, check()
              green, occupancy non-degenerate.
  stats       seed-averaged <P_new>/<P_old> over S=32 seeds at n=128, per-bin
              |ratio - 1| < 4 sigma_bin with sigma_bin = sqrt(4/(S N_modes)),
              chi^2/dof in [0.5, 1.6]; the old arm is this file's licensed
              mirror. Plumbing: all plane hashes distinct; per-plane moments
              within 5 sigma; new != old bitwise at the same seed (clause 5
              says they MUST differ). Tier 2: indistinguishable from the
              split-half control within the new stream -- any 4 sigma bin is
              plumbing, not physics.
  engine-smoke the engine consumes a LOADED state: K=3 BullFrog steps at the
              `smoke` config from load_slot_state(...) land bitwise where the
              same steps from the monolithic build land.

Legs, and where each runs:

  mirror-license  laptop      era-aware (see leg docstring); S0 card = the license
  ledger          laptop      precondition, exits non-zero
  invariance      laptop      n=64 default; deneb anchor at n=512 (own sbatch)
  e2e             laptop      n=64 default; deneb anchor at n=512 (own sbatch)
  stats           laptop      S=32, n=128
  engine-smoke    laptop      smoke engine config, K=3
  (table bar      laptop      v2_m5_table_bar.py, own card)
  (memory ladder  antares     v2_m5_memladder: 3 arms x rungs, own sbatch)
  (ooc 2048^3     Vista gh    v2_m5_fft_gh.py, own sbatch)
"""

import argparse
import hashlib
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


# ---------------------------------------------------------------------------
# The old-stream mirror: verbatim reimplementation of the PRE-M-v2-5 ic.py
# colour path. `cosmology.linear_power(backend="eh98")`, `transfer_eh98` and
# `growth_factor_md` are NOT being replaced (they are the reference the new
# table is validated against), so importing them here is legitimate; what is
# mirrored is exactly what M-v2-5 replaces -- the monolithic
# `jax.random.normal` stream and the full-3D-half-grid colour evaluation.
# Licensed bitwise against inexor.ic at the S0 commit (the frozen card
# m5_gate_mirror-license.json); after the replacement it is the OLD arm of
# the stats leg and the memory ladder.
# ---------------------------------------------------------------------------

C_OVER_H0 = 299792.458 / 100.0


def _old_gaussian_delta(key, n_mesh, box_size, cosmo, fdtype=None):
    """Verbatim mirror of pre-M-v2-5 ic.gaussian_delta (eh98 backend, amplitude 1)."""
    import jax
    import jax.numpy as jnp

    from inexor.cosmology import linear_power

    if fdtype is None:
        fdtype = jnp.float32
    N, L = n_mesh, box_size
    white = jax.random.normal(key, (N, N, N), dtype=fdtype)
    dk = jnp.fft.rfftn(white)
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)
    kk = np.sqrt(kx.reshape(N, 1, 1) ** 2 + kx.reshape(1, N, 1) ** 2 + kz.reshape(1, 1, -1) ** 2)
    kk_safe = kk.copy()
    kk_safe[0, 0, 0] = kk.flat[1]
    colour = np.sqrt(linear_power(kk_safe.ravel(), cosmo).reshape(kk.shape) * N**3 / L**3)
    colour[0, 0, 0] = 0.0
    npdt = np.float64 if fdtype == jnp.float64 else np.float32
    dk = dk * jnp.asarray(colour.astype(npdt))
    return jnp.fft.irfftn(dk, s=(N, N, N))


def _old_poisson_factor(n_mesh, box_size, cosmo, z=0.0):
    """Verbatim mirror of pre-M-v2-5 ic.poisson_factor (full 3D half-grid M(k))."""
    from inexor.cosmology import growth_factor_md, transfer_eh98

    N, L = n_mesh, box_size
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)
    k_mag = np.sqrt(
        kx.reshape(N, 1, 1) ** 2 + kx.reshape(1, N, 1) ** 2 + kz.reshape(1, 1, -1) ** 2
    )
    k_safe = np.where(k_mag > 0, k_mag, 1.0)
    T = transfer_eh98(k_safe.ravel(), cosmo).reshape(k_mag.shape)
    a = 1.0 / (1.0 + z)
    D_md = growth_factor_md(a, cosmo)
    M = (2.0 / 3.0) * C_OVER_H0**2 * k_mag**2 * T * D_md / cosmo.Omega_m
    return np.where(k_mag > 0, M, 1.0)


def _old_linear_density(key, n_mesh, box_size, cosmo, f_NL=0.0, fdtype=None):
    """Verbatim mirror of pre-M-v2-5 ic.linear_density."""
    import jax.numpy as jnp

    N = n_mesh
    if fdtype is None:
        fdtype = jnp.float32
    delta_G = _old_gaussian_delta(key, N, box_size, cosmo, fdtype)
    npdt = np.float64 if fdtype == jnp.float64 else np.float32
    M = jnp.asarray(_old_poisson_factor(N, box_size, cosmo).astype(npdt))
    phi_G = jnp.fft.irfftn(jnp.fft.rfftn(delta_G) / M, s=(N, N, N))
    mean_phi2 = jnp.mean(phi_G**2)
    phi_NG = phi_G + f_NL * (phi_G**2 - mean_phi2)
    return jnp.fft.irfftn(jnp.fft.rfftn(phi_NG) * M, s=(N, N, N))


# ---------------------------------------------------------------------------
# leg: mirror-license
# ---------------------------------------------------------------------------


def leg_mirror_license():
    """The mirror against inexor.ic, with expectations set by the tree's era.

    PRE-REPLACEMENT (ic.IC_STREAM absent): bitwise identity on every arm --
    the license, banked as the frozen card m5_gate_mirror-license.json at the
    S0 commit. POST-REPLACEMENT (IC_STREAM present): the field arms MUST
    DIFFER at every seed (D-v2-15 clause 5 -- identity here would mean the
    fold_in stream was never wired), while poisson_factor, whose analytic body
    did not move, MUST still match. Same instrument, two eras, both fail-able.

    Anti-vacuity either way: the fields have real dynamic range (rms > 0.1),
    and the different-seed control does NOT match.
    """
    import jax
    import jax.numpy as jnp

    from inexor import ic
    from inexor.config import Cosmology

    new_stream = hasattr(ic, "IC_STREAM")
    fields_equal = not new_stream
    print(f"  era: {'post-replacement (fields must differ)' if new_stream else 'pre-replacement (license: fields must match)'}",
          flush=True)

    cosmo = Cosmology()
    arms = {}
    ok = True

    def _cmp(name, a, b, expect_equal=True):
        nonlocal ok
        a = np.asarray(a)
        b = np.asarray(b)
        n_diff = int(np.sum(a != b))
        rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
        passed = (n_diff == 0) == expect_equal
        ok = ok and passed
        arms[name] = dict(n_diff=n_diff, rms=rms, expect_equal=expect_equal, ok=passed)
        tag = "ok" if passed else "FAIL"
        print(f"  {name}: n_diff={n_diff} rms={rms:.3e} [{tag}]", flush=True)

    for N in (64, 128):
        for fdt, fname in ((jnp.float64, "f64"), (jnp.float32, "f32")):
            key = jax.random.PRNGKey(SEED)
            mine = _old_gaussian_delta(key, N, 128.0, cosmo, fdt)
            theirs = ic.gaussian_delta(key, N, 128.0, cosmo, fdtype=fdt)
            _cmp(f"gaussian_delta_n{N}_{fname}", mine, theirs, expect_equal=fields_equal)

    key = jax.random.PRNGKey(SEED)
    for f_NL in (0.0, 10.0):
        mine = _old_linear_density(key, 64, 128.0, cosmo, f_NL=f_NL, fdtype=jnp.float64)
        theirs = ic.linear_density(key, 64, 128.0, cosmo, f_NL=f_NL, fdtype=jnp.float64)
        _cmp(f"linear_density_n64_f64_fnl{int(f_NL)}", mine, theirs, expect_equal=fields_equal)

    _cmp(
        "poisson_factor_n64",
        _old_poisson_factor(64, 128.0, cosmo),
        ic.poisson_factor(64, 128.0, cosmo),
    )

    # anti-vacuity: rms must be real, and a different seed must NOT match
    rms_all = [a["rms"] for a in arms.values()]
    if min(rms_all) < 0.1:
        ok = False
        print(f"  ANTI-VACUITY FAIL: min rms {min(rms_all):.3e} < 0.1", flush=True)
    other = _old_gaussian_delta(jax.random.PRNGKey(SEED + 1), 64, 128.0, cosmo, jnp.float64)
    same = ic.gaussian_delta(jax.random.PRNGKey(SEED), 64, 128.0, cosmo, fdtype=jnp.float64)
    _cmp("different_seed_differs", other, same, expect_equal=False)

    fp = hashlib.sha256(
        np.asarray(
            _old_gaussian_delta(jax.random.PRNGKey(SEED), 64, 128.0, cosmo, jnp.float64)
        ).tobytes()
    ).hexdigest()
    return dict(ok=ok, arms=arms, old_stream_fingerprint_n64_f64_seed0=fp)


# ---------------------------------------------------------------------------
# leg: ledger (leg 0)
# ---------------------------------------------------------------------------


def _maxrss_bytes():
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


def _ledger_worker(mode):
    """Subprocess body: import baseline vs one n=256 f64 gaussian_delta.

    One subprocess per measurement because ru_maxrss is a monotone high-water
    mark (the v4d method); the DIFFERENCE is the colour path's footprint.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    import inexor.ic as ic
    from inexor.config import Cosmology

    # BOTH modes initialize the jax backend (one tiny draw): without this the
    # difference conflates the colour path with XLA's own startup footprint.
    ic.white_plane(jax.random.PRNGKey(SEED), 0, 8, np.float64)
    if mode == "colour":
        ic.gaussian_delta(jax.random.PRNGKey(SEED), 256, 128.0, Cosmology(), fdtype=np.float64)
    print(json.dumps(dict(maxrss=_maxrss_bytes())), flush=True)


def _run_ledger_worker(mode):
    out = subprocess.check_output(
        [sys.executable, os.path.abspath(__file__), "--leg", "ledger",
         "--worker-ledger", mode],
        text=True, cwd=REPO,
    )
    return json.loads(out.strip().splitlines()[-1])["maxrss"]


def leg_ledger(args):
    import jax

    from inexor import ic, ooc_fft
    from inexor.config import Cosmology

    cosmo = Cosmology()
    res = {"ok": True}

    # realized dtypes, f64 and f32
    for dt in (np.float64, np.float32):
        w = ic.white_slab(jax.random.PRNGKey(SEED), 3, 9, 32, dt)
        d = ic.gaussian_delta(jax.random.PRNGKey(SEED), 32, 32.0, cosmo, fdtype=dt)
        spec = ooc_fft.rfftn_ooc(ic.white_noise(jax.random.PRNGKey(SEED), 32, dt))
        ok = (w.dtype == np.dtype(dt) and d.dtype == np.dtype(dt)
              and spec.dtype == (np.complex128 if np.dtype(dt) == np.float64 else np.complex64))
        res[f"dtypes_{np.dtype(dt).name}"] = bool(ok)
        res["ok"] &= ok
        print(f"  realized dtypes at {np.dtype(dt).name}: {'ok' if ok else 'FAIL'}", flush=True)

    # random access (the streamability anti-vacuity)
    ref = ic.white_noise(jax.random.PRNGKey(SEED), 64, np.float64)
    ra = np.array_equal(ic.white_slab(jax.random.PRNGKey(SEED), 13, 20, 64, np.float64),
                        ref[13:20])
    res["random_access"] = bool(ra)
    res["ok"] &= ra
    print(f"  random access mid-field: {'ok' if ra else 'FAIL'}", flush=True)

    # The memory shape of the colour path, by subprocess high-water marks.
    # THE BAR IS DERIVED, NOT PICKED: the monolithic convenience legitimately
    # holds white (1x field) + the complex spectrum (2x) + the assembling
    # output (1x) = 4x, plus O(plane) working buffers and allocator slop
    # budgeted at 1x -> 5x field. What the leg exists to exclude is the OLD
    # colour evaluation's ~15 half-grid f64 temporaries (~16x field: 2.19 GB
    # measured at n=256, job 896159), and 5x vs ~16x is the separation.
    base = _run_ledger_worker("baseline")
    colour = _run_ledger_worker("colour")
    field_bytes = 256**3 * 8
    net = colour - base
    bar = 5 * field_bytes
    ok = net < bar
    res["colour_footprint"] = dict(
        baseline=base, colour=colour, net=net, field_bytes=field_bytes,
        bar=bar, bar_derivation="white 1x + spec 2x + out 1x + buffers 1x",
        old_path_class="~16x field (measured 2.19 GB at n=256, job 896159)",
        ok=bool(ok),
    )
    res["ok"] &= ok
    print(f"  colour net footprint {net / 1e6:.0f} MB vs derived bar {bar / 1e6:.0f} MB "
          f"(field {field_bytes / 1e6:.0f} MB; old path ~2190 MB): {'ok' if ok else 'FAIL'}",
          flush=True)
    return res


# ---------------------------------------------------------------------------
# leg: invariance (leg I)
# ---------------------------------------------------------------------------


def leg_invariance(args):
    import jax

    from inexor import ic, icgen
    from inexor.config import Cosmology

    n = args.n
    cosmo = Cosmology()
    key = jax.random.PRNGKey(args.seed)
    res = {"ok": True, "n": n}

    # white-field assembly across thicknesses, f64
    ref = ic.white_noise(key, n, np.float64)
    for t in (1, 7, max(n // 2, 1), n):
        parts = [ic.white_slab(key, lo, min(lo + t, n), n, np.float64) for lo in range(0, n, t)]
        n_diff = int(np.sum(np.concatenate(parts) != ref))
        res[f"white_t{t}_n_diff"] = n_diff
        res["ok"] &= n_diff == 0
        print(f"  white thickness {t}: n_diff={n_diff}", flush=True)
    other = int(np.sum(ic.white_noise(jax.random.PRNGKey(args.seed + 1), n, np.float64) != ref))
    res["white_other_seed_n_diff"] = other
    res["ok"] &= other > 0
    print(f"  white other seed differs: n_diff={other} (must be > 0)", flush=True)

    # END-TO-END: two full generations, slab 7 vs slab n, byte-identical files
    nb = args.bricks
    with tempfile.TemporaryDirectory() as wa, tempfile.TemporaryDirectory() as wb:
        ma = icgen.generate_t9_slabs(wa, key, n, float(n), cosmo, 0.1, nb,
                                     f_NL=10.0, fdtype=np.float64, slab=7)
        mb = icgen.generate_t9_slabs(wb, key, n, float(n), cosmo, 0.1, nb,
                                     f_NL=10.0, fdtype=np.float64, slab=n)
        same_scale = ma["vel_scale"] == mb["vel_scale"]
        same_mean = ma["mean_phi2"] == mb["mean_phi2"]
        n_diff_files = 0
        for fname in ma["files"]:
            with np.load(os.path.join(wa, fname)) as za, \
                 np.load(os.path.join(wb, fname)) as zb:
                for name in ("occupancy", "off", "w"):
                    n_diff_files += int(np.sum(za[name] != zb[name]))
        res["generation_slab7_vs_slabN"] = dict(
            vel_scale_equal=bool(same_scale), mean_phi2_equal=bool(same_mean),
            n_diff_payload=n_diff_files,
        )
        ok = same_scale and same_mean and n_diff_files == 0
        res["ok"] &= ok
        print(f"  full generation slab 7 vs {n}: payload n_diff={n_diff_files}, "
              f"vel_scale equal={same_scale}, mean_phi2 equal={same_mean}", flush=True)
    return res


# ---------------------------------------------------------------------------
# leg: e2e (leg III)
# ---------------------------------------------------------------------------


def leg_e2e(args):
    import jax

    from inexor import ic, icgen, lpt, state
    from inexor.codec import T9Layout
    from inexor.config import Cosmology

    n = args.n
    box = float(n)
    nb = args.bricks
    cosmo = Cosmology()
    key = jax.random.PRNGKey(args.seed)
    res = {"ok": True, "n": n, "bricks": nb, "f_NL": args.f_nl}

    with tempfile.TemporaryDirectory() as wd:
        icgen.generate_t9_slabs(wd, key, n, box, cosmo, 0.1, nb,
                                f_NL=args.f_nl, fdtype=np.float64, slab=args.slab)
        st = icgen.load_slot_state(wd)
    d0 = ic.linear_density(key, n, box, cosmo, f_NL=args.f_nl, fdtype=np.float64)
    x, v = lpt.lpt_ics(d0, box, 0.1, cosmo, order=2, fdtype=np.float64)
    ref = state.SlotState.build(x, v, T9Layout(box, n, 2), nb)

    checks = dict(
        off=int(np.sum(st.off != ref.off)),
        w=int(np.sum(st.w != ref.w)),
        occupancy=int(np.sum(np.asarray(st.occupancy) != np.asarray(ref.occupancy))),
        brick_start=int(np.sum(st.brick_start != ref.brick_start)),
    )
    res["n_diff"] = checks
    res["vel_scale_equal"] = bool(st.vel_scale == ref.vel_scale)
    res["arena_empty"] = bool(st.arena_used == 0 == ref.arena_used)
    st.check()
    occ = np.asarray(st.occupancy, np.int64)
    res["occupancy_peak_over_mean"] = float(occ.max() / max(occ[occ > 0].mean(), 1e-9))
    res["ok"] = (all(v == 0 for v in checks.values()) and res["vel_scale_equal"]
                 and res["arena_empty"] and res["occupancy_peak_over_mean"] > 2.0)
    for k, nd in checks.items():
        print(f"  {k}: n_diff={nd}", flush=True)
    print(f"  vel_scale equal: {res['vel_scale_equal']}; occupancy peak/mean "
          f"{res['occupancy_peak_over_mean']:.1f}", flush=True)
    return res


# ---------------------------------------------------------------------------
# leg: stats (leg IV)
# ---------------------------------------------------------------------------


def leg_stats(args):
    import jax
    import jax.numpy as jnp

    from inexor import ic
    from inexor.config import Cosmology
    from inexor.diagnostics import pk_estimator

    n, box, S = 128, 128.0, 32
    cosmo = Cosmology()
    res = {"ok": True, "n": n, "seeds": S}

    # plumbing: distinct planes, sane per-plane moments, adjacent correlation
    w = ic.white_noise(jax.random.PRNGKey(SEED), n, np.float64)
    hashes = {hashlib.sha256(w[i].tobytes()).hexdigest() for i in range(n)}
    res["plane_hashes_distinct"] = len(hashes) == n
    mom_ok = True
    sig_m = 5.0 / np.sqrt(n * n)
    sig_v = 5.0 * np.sqrt(2.0 / (n * n))
    for i in range(n):
        mom_ok &= abs(float(w[i].mean())) < sig_m and abs(float(w[i].var()) - 1.0) < sig_v
    res["plane_moments_ok"] = bool(mom_ok)
    corr = np.array([np.corrcoef(w[i].ravel(), w[i + 1].ravel())[0, 1] for i in range(n - 1)])
    res["adjacent_corr_max"] = float(np.abs(corr).max())
    corr_ok = res["adjacent_corr_max"] < 5.0 / np.sqrt(n * n)
    old = np.asarray(_old_gaussian_delta(jax.random.PRNGKey(SEED), n, box, cosmo, jnp.float64))
    new = ic.gaussian_delta(jax.random.PRNGKey(SEED), n, box, cosmo, fdtype=np.float64)
    res["new_differs_from_old_bitwise"] = bool(np.any(old != new))
    res["ok"] &= (res["plane_hashes_distinct"] and mom_ok and corr_ok
                  and res["new_differs_from_old_bitwise"])
    print(f"  plumbing: hashes distinct={res['plane_hashes_distinct']}, moments={mom_ok}, "
          f"adj corr max={res['adjacent_corr_max']:.2e}, new!=old={res['new_differs_from_old_bitwise']}",
          flush=True)

    # the gated bar: seed-averaged <P_new>/<P_old>
    kmax = 0.6 * np.pi * n / box
    pk_new, pk_old = [], []
    for s in range(S):
        dn = ic.gaussian_delta(jax.random.PRNGKey(s), n, box, cosmo, fdtype=np.float64)
        do = np.asarray(_old_gaussian_delta(jax.random.PRNGKey(s), n, box, cosmo, jnp.float64))
        kc, pn, nm = pk_estimator(dn, box, kmax=kmax)
        _, po, _ = pk_estimator(do, box, kmax=kmax)
        pk_new.append(pn)
        pk_old.append(po)
    sel = nm > 50
    ratio = np.mean(pk_new, axis=0)[sel] / np.mean(pk_old, axis=0)[sel]
    sig = np.sqrt(4.0 / (S * nm[sel]))
    z = (ratio - 1.0) / sig
    chi2_dof = float(np.mean(z**2))
    res["bins"] = int(sel.sum())
    res["max_abs_z"] = float(np.abs(z).max())
    res["chi2_dof"] = chi2_dof
    bar_ok = bool(np.all(np.abs(z) < 4.0) and 0.5 < chi2_dof < 1.6)
    res["ok"] &= bar_ok
    print(f"  <P_new>/<P_old>: {res['bins']} bins, max|z|={res['max_abs_z']:.2f} (bar 4), "
          f"chi2/dof={chi2_dof:.2f} (band 0.5-1.6)", flush=True)

    # tier-2 context: the split-half floor WITHIN the new stream
    half = S // 2
    r_split = (np.mean(pk_new[:half], axis=0)[sel] / np.mean(pk_new[half:], axis=0)[sel])
    z_split = (r_split - 1.0) / np.sqrt(4.0 / (half * nm[sel]))
    res["split_half_max_abs_z"] = float(np.abs(z_split).max())
    res["split_half_chi2_dof"] = float(np.mean(z_split**2))
    print(f"  split-half control: max|z|={res['split_half_max_abs_z']:.2f}, "
          f"chi2/dof={res['split_half_chi2_dof']:.2f}", flush=True)
    return res


# ---------------------------------------------------------------------------
# leg: engine-smoke
# ---------------------------------------------------------------------------


def leg_engine_smoke(args):
    import jax

    from inexor import engine, ic, icgen, lpt, state
    from inexor.codec import T9Layout
    from inexor.config import Cosmology
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table

    sys.path.insert(0, HERE)
    import v2_m3_engine_gate as m3

    g = m3._geom("smoke")
    cosmo = Cosmology()
    key = jax.random.PRNGKey(args.seed)
    ec = engine.EngineConfig(
        box_size=g["L"], n_part=g["n_part"], n_fine=g["n_fine"], n_coarse=g["n_coarse"],
        n_tile=g["tile"], b_fine=g["buf"], alpha=m3.ALPHA, brick_slack=0.25,
    )
    ec.validate()
    nb = g["n_fine"] // ec.n_brick
    co = bullfrog_float_coeffs(bullfrog_table(a_grid(0.1, 0.3, 3, "log"), cosmo))

    def _run_from(st):
        engine.run(st, ec, co)
        st.check()
        return np.concatenate([st.decode_brick(b)[1] for b in range(st.n_bricks)])

    with tempfile.TemporaryDirectory() as wd:
        icgen.generate_t9_slabs(wd, key, g["n_part"], g["L"], cosmo, 0.1, nb,
                                fdtype=np.float64)
        st_loaded = icgen.load_slot_state(wd, brick_slack=0.25, arena_frac=0.10)
    x, v = lpt.lpt_ics(
        ic.linear_density(key, g["n_part"], g["L"], cosmo, fdtype=np.float64),
        g["L"], 0.1, cosmo, order=2, fdtype=np.float64,
    )
    st_built = state.SlotState.build(x, v, T9Layout(g["L"], g["n_part"], 2), nb,
                                     brick_slack=0.25, arena_frac=0.10)

    x_loaded = _run_from(st_loaded)
    x_built = _run_from(st_built)
    n_diff = int(np.sum(x_loaded != x_built))
    rms = float(np.sqrt(np.mean(x_loaded**2)))
    ok = n_diff == 0 and rms > 0.1
    print(f"  K=3 engine steps, loaded vs built state: n_diff={n_diff}, "
          f"decoded rms={rms:.3f}", flush=True)
    return dict(ok=bool(ok), n_diff=n_diff, decoded_rms=rms, k_steps=3, config="smoke")


# ---------------------------------------------------------------------------
# provenance + card
# ---------------------------------------------------------------------------


def _provenance(args):
    """Where this card was produced and what was asked of it (m4 pattern).

    New load-bearing field this milestone: `ic_stream`. After M-v2-5 a seed
    under `inexor.ic` denotes a different realization than the same seed before
    it; cards carry the stream identity so a readout can REFUSE to pool across
    it. Cards written before the field exists read as UNKNOWN, never backfilled.
    `threefry_partitionable` is recorded because `fold_in`'s derived keys depend
    on it (True on the locked jax 0.10.2; the generator asserts rather than
    sets it).
    """
    import jax

    from inexor import ic

    dev = jax.devices()[0]
    try:
        x64 = bool(jax.config.jax_enable_x64)
    except Exception:
        x64 = None
    return dict(
        backend=dev.platform,
        device_kind=dev.device_kind,
        n_devices=jax.device_count(),
        host_cores=os.cpu_count(),
        x64=x64,
        jax_version=jax.__version__,
        threefry_partitionable=bool(jax.config.jax_threefry_partitionable),
        ic_stream=getattr(ic, "IC_STREAM", "pre-m5-normal"),
        xla_flags=os.environ.get("XLA_FLAGS"),
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        argv=sys.argv[1:],
        knobs=dict(n=args.n, bricks=args.bricks, slab=args.slab,
                   seed=args.seed, f_nl=args.f_nl),
    )


def _write(res, args):
    try:
        res["commit"] = subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        res["commit"] = None
    res["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    res["provenance"] = _provenance(args)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"m5_gate_{args.leg}{args.out_suffix}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"  card -> {path}", flush=True)


LEGS = {
    "mirror-license": lambda args: leg_mirror_license(),
    "ledger": leg_ledger,
    "invariance": leg_invariance,
    "e2e": leg_e2e,
    "stats": leg_stats,
    "engine-smoke": leg_engine_smoke,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leg", default="mirror-license", choices=sorted(LEGS))
    ap.add_argument("--n", type=int, default=64, help="grid for invariance/e2e legs")
    ap.add_argument("--bricks", type=int, default=4)
    ap.add_argument("--slab", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--f-nl", type=float, default=0.0)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--worker-ledger", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker_ledger is not None:
        _ledger_worker(args.worker_ledger)
        return 0

    import jax

    jax.config.update("jax_enable_x64", True)

    print(f"leg {args.leg}", flush=True)
    res = LEGS[args.leg](args)
    _write(res, args)
    if not res["ok"]:
        print("LEG FAILED", flush=True)
        return 1
    print("leg passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
