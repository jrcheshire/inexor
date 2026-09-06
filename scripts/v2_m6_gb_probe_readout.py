"""Readout for the gb probe: sum the 4096^3 per-step projection against gb's wall.

Reads the cards the gb job writes (see scripts/v2_m6_gb_probe_vista.sbatch)
and prints ONE table: each per-step term at the target preset, its source
card, and the sum against the bar the machine sets -- gb's MaxWall is 12 h,
so K=40 fits one job only if a step costs <= 1080 s. The bar is derived,
not picked.

Every term is arithmetic on a measurement at a smaller size, and the table
says how each was scaled:
  tile loop     tiles(P) x per-tile DEVICE median / n_gpus, from the g5 force
                leg on a B200 (the host plumbing phases are shown beside it
                and are what the design deletes)
  decode+quant  engine phases on a B200 at cdev, scaled by particle count
  coarse paint  nb slabs x per-slab sub-block paint / n_gpus (full-mesh row
                shown as the pessimistic bound)
  migrate eject nb slabs x per-slab kernel / n_gpus (end-to-end row includes
                the H2D/D2H a host-resident state pays)
  coarse solve  min(host out-of-core: 4 x measured 2048^3 transform,
                device: 4 x the largest fitting size scaled by n^3 log n)
  streaming     2 x state bytes / aggregate host->device rate (four GPUs
                pulling at once), single-GPU rate if the 4x cards are absent

The four-GPU split is ASSUMED perfect (n_gpus = 4.0). The insert half of the
migrate, repack, the kick and host bookkeeping are NOT measured and are
listed as such under the sum. A missing load-bearing card WITHHOLDS the
verdict rather than passing by omission.

Run (after the job, from the checkout that holds the cards):
    pixi run python scripts/v2_m6_gb_probe_readout.py --suffix _gb
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
OUT_DIR = os.path.join(REPO, "runs", "v2")
sys.path.insert(0, HERE)

WALL_S = 12 * 3600.0  # gb MaxWall, qlimits 2026-08-20
STATE_BYTES_PER_PARTICLE = 10.54  # D-v2-20 all-in T9 (payload + index + slack)
CDEV_N_PART = 256


def _load(name):
    path = os.path.join(OUT_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _arm(card, name, **match):
    if not card:
        return None
    for r in card.get("arms", []):
        if r.get("arm") == name and all(r.get(k) == v for k, v in match.items()):
            return r
    return None


def _force_cards(suffix):
    out = {}
    for tile in (256, 512):
        for fd in ("f64", "f32"):
            c = _load(f"m6_gb_force_T{tile}_{fd}{suffix}.json")
            if c:
                out[(tile, fd)] = c
    return out


def _streaming_rate(card, arm="staged"):
    """Best completed rung's GB/s for one arm of a g4 card, and which rung."""
    if not card:
        return None, None
    best = None
    for r in card.get("configs", []):
        if r.get("arm") != arm or not r.get("completed") or r.get("real"):
            continue
        gbs = r.get("gbytes_per_s")
        if gbs is None:
            continue
        gib = float(r.get("working_set_gib", 0))
        if best is None or gib > best[1]:
            best = (float(gbs), gib)
    return (best[0], best[1]) if best else (None, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--suffix", default="_gb", help="card suffix the job used")
    ap.add_argument("--preset", default="c-hero", help="target preset to project")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--n-gpus", type=float, default=4.0)
    ap.add_argument("--probe-suffix", default=None,
                    help="suffix of the m6_gb_probe card if it differs from --suffix")
    args = ap.parse_args()

    from v2_m6_gb_probe import geometry

    g = geometry(args.preset)
    n_gpus = float(args.n_gpus)
    bar = WALL_S / args.k
    missing = []
    unmeasured = ["migrate insert (only the eject half is measured)",
                  "repack", "kick", "host-side bookkeeping a built pipeline keeps",
                  f"the {n_gpus:.0f}-GPU split efficiency (assumed 1.0)"]

    probe = _load(f"m6_gb_probe{args.probe_suffix or args.suffix}.json")
    forces = _force_cards(args.suffix)
    engine = _load(f"m6_phase_time{args.suffix}_b200_cdev.json")
    hostfft = _load(f"m5_gate_fft-gh{args.suffix}.json")
    stream_a = _load(f"g4_gh_memory{args.suffix}_a.json")
    stream_b = _load(f"g4_gh_memory{args.suffix}_b.json")
    stream_4x = [_load(f"g4_gh_memory{args.suffix}4x_gpu{i}.json") for i in range(4)]

    rows = []  # (label, s_per_step, in_sum, source)

    # --- tile loop -------------------------------------------------------
    tile_choice = None
    for tile in (512, 256):
        for fd in ("f64", "f32"):
            c = forces.get((tile, fd))
            if not c:
                continue
            an = c.get("anatomy") or {}
            dev_med = (an.get("device") or {}).get("median_s")
            steady = an.get("steady_per_tile_s")
            tiles = (g["n_fine"] // tile) ** 3
            src = f"m6_gb_force_T{tile}_{fd}{args.suffix}.json"
            if dev_med is not None:
                s = tiles * dev_med / n_gpus
                use = tile_choice is None and fd == "f64"
                rows.append((f"tile loop, device only, P={tile + 2 * g['buf']} {fd} "
                             f"({tiles} tiles x {dev_med * 1e3:.1f} ms)", s, use, src))
                if use:
                    tile_choice = (tile, fd, s)
            if steady is not None:
                rows.append((f"tile loop, as measured incl. host plumbing, "
                             f"P={tile + 2 * g['buf']} {fd}", tiles * steady / n_gpus, False, src))
    if tile_choice is None:
        missing.append("force card (m6_gb_force_T*_f64)")

    # --- decode + quantize ---------------------------------------------
    if engine:
        k_eng = int(engine.get("k", 1))
        ph = engine.get("phase_s", {})
        dq = {n: v / k_eng for n, v in ph.items() if "decode" in n or "quant" in n}
        ratio = g["n_total"] / CDEV_N_PART**3
        if dq:
            s = sum(dq.values()) * ratio / n_gpus
            rows.append(("decode + quantize, engine cdev on B200 x "
                         f"{ratio:.0f} ({', '.join(sorted(dq))})", s, True,
                         f"m6_phase_time{args.suffix}_b200_cdev.json"))
        else:
            unmeasured.append("decode + quantize (no decode/quant phase on the engine card)")
        for n in sorted(ph):
            rows.append((f"  [engine cdev on B200, per step, unscaled] {n}", ph[n] / k_eng,
                         False, "same card"))
    else:
        unmeasured.append("decode + quantize (engine card absent)")

    # --- coarse paint ----------------------------------------------------
    probe_preset = (probe or {}).get("geometry", {}).get("preset")
    if probe and probe_preset != g["preset"]:
        rows.append((f"  !! probe card is preset {probe_preset}, not {g['preset']}: slab terms "
                     "are scaled PER ROW and carry that preset's fixed overhead", None, False,
                     "m6_gb_probe"))

    def per_slab(rec, median_s):
        # measured rows -> the target's rows per slab, linearly in rows
        return float(median_s) * g["rows_per_slab"] / float(rec["rows"])

    paint = _arm(probe, "paint")
    if paint and paint.get("completed"):
        ps = per_slab(paint, paint["subblock"]["median_s"])
        s = g["nb"] * ps / n_gpus
        rows.append((f"coarse paint, sub-block ({g['nb']} slabs x {ps:.2f} s)", s, True,
                     "m6_gb_probe"))
        if paint.get("full") is not None:
            pf = per_slab(paint, paint["full"]["median_s"])
            rows.append((f"coarse paint, full {g['n_coarse']}^3 mesh (pessimistic, "
                         f"{pf:.2f} s/slab)", g["nb"] * pf / n_gpus, False, "m6_gb_probe"))
        else:
            rows.append(("coarse paint, full mesh: " + ("OOM" if paint.get("full_oom") else
                         str(paint.get("full_error"))[:50]), None, False, "m6_gb_probe"))
        if not paint.get("mass_equal", True):
            rows.append(("  !! sub-block mass != full-mesh mass: containment broken", None,
                         False, "m6_gb_probe"))
        if not paint.get("headroom", {}).get("ok", True):
            rows.append(("  !! TSC int32 headroom refuses at this N: " +
                         str(paint["headroom"].get("message"))[:60], None, False, "m6_gb_probe"))
    else:
        missing.append("paint arm (m6_gb_probe)")

    # --- migrate eject ---------------------------------------------------
    ej = _arm(probe, "eject")
    if ej and ej.get("completed"):
        pd = per_slab(ej, ej["device_only"]["median_s"])
        pe = per_slab(ej, ej["end_to_end"]["median_s"])
        s_dev = g["nb"] * pd / n_gpus
        s_e2e = g["nb"] * pe / n_gpus
        rows.append((f"migrate eject, kernel on device ({g['nb']} slabs x {pd:.2f} s; "
                     f"leave {ej['leave_fraction']:.1%})", s_dev, True, "m6_gb_probe"))
        rows.append((f"migrate eject, end to end incl. H2D/D2H ({pe:.2f} s/slab)", s_e2e, False,
                     "m6_gb_probe"))
    else:
        missing.append("eject arm (m6_gb_probe)")

    # --- coarse solve ----------------------------------------------------
    solve_host = None
    if hostfft and hostfft.get("roundtrip"):
        rt = hostfft["roundtrip"]
        if rt.get("fwd_s") is not None and rt.get("inv_s") is not None:
            solve_host = 4.0 * 0.5 * (float(rt["fwd_s"]) + float(rt["inv_s"]))
            rows.append((f"coarse solve, host out-of-core {rt.get('n')}^3 "
                         f"(fwd {rt['fwd_s']:.1f} / inv {rt['inv_s']:.1f} s, x4)",
                         solve_host, False, f"m5_gate_fft-gh{args.suffix}.json"))
    solve_dev = None
    fits_one_gpu = None
    if probe:
        done = [r for r in probe.get("arms", []) if r.get("arm") == "fft" and r.get("completed")]
        target = g["n_coarse"]
        exact = [r for r in done if int(r["n"]) == target]
        fits_one_gpu = bool(exact)
        if done:
            r = exact[0] if exact else max(done, key=lambda r: int(r["n"]))
            n = int(r["n"])
            t = 0.5 * (r["fwd"]["median_s"] + r["inv"]["median_s"])
            scale = (target / n) ** 3 * (math.log2(target) / math.log2(n))
            solve_dev = 4.0 * t * scale
            rows.append((f"coarse solve, device, from {n}^3 "
                         f"({t:.2f} s/transform, peak/field {r.get('peak_over_field') or 0:.1f}x"
                         f"{', direct' if exact else ', scaled n^3 log n'}, x4)",
                         solve_dev, False, "m6_gb_probe"))
        if not fits_one_gpu:
            rows.append((f"  {target}^3 f32 rfftn on one GPU: does NOT fit as a monolithic "
                         "transform (needs the plane-factorized form or a multi-GPU split)",
                         None, False, "m6_gb_probe"))
    cands = [v for v in (solve_host, solve_dev) if v is not None]
    if cands:
        rows.append(("coarse solve, min of the two forms", min(cands), True, "above"))
    else:
        missing.append("coarse solve (neither the host FFT card nor an fft arm)")

    # --- streaming -------------------------------------------------------
    state_bytes = g["n_total"] * STATE_BYTES_PER_PARTICLE
    agg = 0.0
    n_agg = 0
    for c in stream_4x:
        gbs, _ = _streaming_rate(c)
        if gbs:
            agg += gbs
            n_agg += 1
    single, single_gib = _streaming_rate(stream_b) if stream_b else (None, None)
    if single is None:
        single, single_gib = _streaming_rate(stream_a)
    if n_agg == 4:
        s = 2 * state_bytes / (agg * 1e9)
        rows.append((f"streaming, 2 x {state_bytes / 1e9:.0f} GB state at {agg:.0f} GB/s "
                     "aggregate over 4 GPUs", s, True, f"g4_gh_memory{args.suffix}4x_gpu*.json"))
    elif single:
        s = 2 * state_bytes / (single * 1e9)
        rows.append((f"streaming, 2 x {state_bytes / 1e9:.0f} GB state at {single:.0f} GB/s, "
                     f"ONE GPU at the {single_gib:.0f} GiB rung ({n_agg}/4 4x cards)", s, True,
                     f"g4_gh_memory{args.suffix}_[ab].json"))
    else:
        missing.append("streaming rate (no g4 card completed a rung)")
    if single:
        rows.append((f"  single-GPU staged rate at the largest completed rung: {single:.0f} GB/s "
                     f"at {single_gib:.0f} GiB", None, False, "g4"))
    if stream_b:
        top = max((float(r.get("working_set_gib", 0)) for r in stream_b.get("configs", [])
                   if r.get("completed") and r.get("arm") in ("staged", "coherent")), default=0)
        rows.append((f"  largest host working set streamed by one GPU: {top:.0f} GiB "
                     f"(state at {args.preset} is {state_bytes / 1024**3:.0f} GiB)", None, False,
                     "g4 _b"))

    # --- print -----------------------------------------------------------
    print(f"=== gb probe readout: {args.preset} = {g['n_part']}^3, K={args.k}, "
          f"{n_gpus:.0f} GPUs assumed, bar {bar:.0f} s/step (12 h wall) ===")
    print(f"{'term':96s} {'s/step':>9s}  {'in sum':>6s}  source")
    for label, s, use, src in rows:
        s_txt = "-" if s is None else f"{s:9.1f}"
        print(f"{label[:96]:96s} {s_txt:>9s}  {'yes' if use else '':>6s}  {src}")
    total = sum(s for _, s, use, _ in rows if use and s is not None)
    hours = total * args.k / 3600.0
    print(f"\n{'SUM of the terms marked yes':96s} {total:9.1f}  -> {hours:.1f} h at K={args.k}")
    if missing:
        print("\nVERDICT WITHHELD -- load-bearing cards missing:")
        for m in missing:
            print(f"  - {m}")
    else:
        if total <= bar:
            print(f"\nFITS ONE 12 h gb JOB: {total:.0f} s/step against {bar:.0f} "
                  f"({total / bar:.2f}x of the bar), before the unmeasured terms below")
        else:
            seg = math.ceil(hours / 12.0)
            print(f"\nDOES NOT FIT ONE JOB: {total:.0f} s/step is {total / bar:.2f}x the bar; "
                  f"{seg} twelve-hour segments at two jobs per user, before the unmeasured terms")
    print("\nNOT measured here, and the sum is a FLOOR without them:")
    for u in unmeasured:
        print(f"  - {u}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
