"""D6: the legs around a device IC generation on a gb node that are not the generation.

    preflight  jax devices, scratch free space and quota, the planner's IC stage table
               at 2048^3 and 4096^3 against a gb node. rc 2 on a refusal.
    smoke      the device generator on the cards against the host generator at 32^3:
               4 cards == 1 card bitwise; occupancy exact, codes within 1 (the
               PROPOSED bars in tests/test_icgen_device.py); scales reported. rc 3 on a
               miss.
    project    the 2048^3 card -> a 4096^3 projection (every stage x8 log-scaled wall,
               host peak x8). rc 3 past the pre-registered stop: wall > 9 h or host peak
               > 0.9 x 1026 GB.
    allocator-ab
               two IC cards whose manifests record DIFFERENT allocators and the same
               everything else -> per-stage seconds ratio. Reported, not gated; rc 3
               if an arm cannot identify itself or an axis other than the allocator
               moved.
    cleanup    remove a generation by NAME (manifest files, manifest, card), then rmdir,
               which refuses if anything else is inside.

The generations themselves run through `scripts/run/realization.py ics --generator device`.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

GB = 1e9
GB_HOST, GB_CARD = 1026.0, 199.0
STOP_WALL_H, STOP_HOST_FRAC, STOP_CARD_FRAC = 6.0, 0.9, 0.9


def _quota_remaining_gb(path):
    """Remaining Lustre quota in GB, or None when `lfs quota` is absent or unparseable."""
    try:
        out = subprocess.run(["lfs", "quota", "-q", "-u", os.environ.get("USER", ""), path],
                             capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None, ""
    for line in out.splitlines():
        f = line.split()
        if len(f) >= 4 and f[1].rstrip("*").isdigit() and f[3].isdigit():
            used_kb, limit_kb = int(f[1].rstrip("*")), int(f[3])
            return (None if limit_kb == 0 else (limit_kb - used_kb) * 1024 / GB), out
    return None, out


def cmd_preflight(args):
    import jax

    from inexor import plan

    rc = 0
    print(f"jax devices: {jax.devices()}")
    # The allocator must show WHICH one is running, not merely that a knob was set.
    # 998798 printed `bytes_limit 0.0 GiB` on all four cards and that read as a
    # failure to apply the fraction; it is how cuda_async reports -- BFC sets
    # bytes_limit to fraction x card, CUDA's pool reports none and keeps
    # peak_bytes_in_use (compute_stats=true). So derive the allocator from the stats
    # and REFUSE when it disagrees with the one the environment asked for.
    asked = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "bfc").strip().lower() or "bfc"
    frac = os.environ.get("XLA_CLIENT_MEM_FRACTION", "(unset, compiled default 0.75)")
    print(f"  allocator asked: {asked}; XLA_CLIENT_MEM_FRACTION={frac}")
    limit_gb = None
    gpus = [d for d in jax.devices() if d.platform in ("gpu", "cuda", "rocm")]
    if not gpus:
        # a CPU device reports no allocator stats at all, so every card would read
        # "cuda_async" and an unset variable would refuse against itself
        print("  no GPU device: the allocator discrimination does not apply; "
              "the per-card ceiling falls back to the card size")
    for dev in gpus:
        stats = dev.memory_stats() or {}
        lim = stats.get("bytes_limit") or 0
        seen = "bfc" if lim else "cuda_async"
        if lim:
            limit_gb = min(lim / GB, limit_gb if limit_gb is not None else lim / GB)
        print(f"  {dev}: reports {seen}; bytes_limit "
              f"{'none' if not lim else f'{lim / 2**30:,.1f} GiB ({lim / GB:,.1f} GB)'}"
              f"; keys {sorted(stats)[:6]}")
        if seen != asked:
            print(f"REFUSE: asked for {asked}, the runtime reports {seen} on {dev}")
            rc = 2
    # Only ever tightens: the card's own size stays the bound when no limit is reported.
    ceiling = min(GB_CARD, limit_gb) if limit_gb else GB_CARD
    print(f"  per-card ceiling used for the planner verdict: {ceiling:,.1f} GB")
    os.makedirs(args.root, exist_ok=True)
    free = shutil.disk_usage(args.root).free / GB
    quota, raw = _quota_remaining_gb(args.root)
    print(f"{args.root}: filesystem free {free:,.0f} GB; quota remaining "
          f"{'unparsed (not a verdict)' if quota is None else f'{quota:,.0f} GB'}")
    if raw:
        print(raw.rstrip())
    for label, avail in (("filesystem free", free), ("quota remaining", quota)):
        if avail is not None and avail < args.need_gb:
            print(f"REFUSE: {label} {avail:,.0f} GB < {args.need_gb:,.0f} GB needed")
            rc = 2
    tmp = shutil.disk_usage("/tmp")
    print(f"/tmp: total {tmp.total / GB:,.0f} GB, free {tmp.free / GB:,.0f} GB (not used)")
    for n, nb in ((2048, 128), (4096, 256)):
        host, card, disk = plan.ic_device_stages(n, n_gpus=len(jax.devices()), nb=nb)
        hp, cp = max(host.values()), max(card.values())
        print(f"planner {n}^3: host peak {hp / GB:,.1f} GB ({hp / GB / GB_HOST:.2f}x), "
              f"per card {cp / GB:,.1f} GB ({cp / GB / ceiling:.2f}x of {ceiling:,.0f}), "
              f"disk {sum(disk.values()) / GB:,.0f} GB")
        if hp / GB > GB_HOST or cp / GB > ceiling:
            print(f"REFUSE: the planner says {n}^3 does not fit")
            rc = 2
    return rc


def cmd_smoke(args):
    import jax

    jax.config.update("jax_enable_x64", True)
    from inexor import icgen
    from inexor.config import Cosmology

    n, box, nb, a_init, slab = 32, 32.0, 4, 0.1, 5
    devs = jax.devices()
    print(f"smoke on {len(devs)} x {devs[0].platform}")

    def gen(fn, **kw):
        d = tempfile.mkdtemp(dir=args.tmp)
        man = fn(d, jax.random.PRNGKey(0), n, box, Cosmology(), a_init, nb, slab=slab, **kw)
        st = icgen.load_slot_state(d)
        st.check()
        return man, st

    # kernel parity against numpy on THIS backend, reported (CPU XLA reads bitwise)
    from inexor import ic, ooc_fft
    from inexor.cosmology import ic_k_table

    K = ooc_fft.KSpaceKernel
    tab = ic_k_table(Cosmology(), n, box)
    white = np.random.default_rng(3).standard_normal((n, n, n)).astype(np.float32)
    spec = ooc_fft.forward_from_slabs(lambda lo, hi: white[lo:hi], n, slab=8)
    col, ipo = ic._colour_fn(tab, n, box), ic._poisson_fn(Cosmology(), tab, inverse=True)
    kernels = dict(
        deriv2_01=(K.deriv2(0, 1), ooc_fft.deriv2_spec(spec, 0, 1, n, box)),
        grad_2=(K.grad_invk2(2), ooc_fft.grad_invk2_spec(spec, 2, n, box)),
        colour_x_invpoisson=(K.colour(tab, n, box) * K.poisson(Cosmology(), tab, n, box,
                                                              inverse=True),
                             ooc_fft.mul_radial_inplace(spec.copy(), n, box,
                                                        lambda kk: col(kk) * ipo(kk), 0.0)),
    )
    card = dict(kernels_eps_x_rms={})
    for name, (kern, ref) in kernels.items():
        got = ooc_fft.kspace_pass_device([(1.0, spec)], n, box, kernel=kern, transform=False,
                                         devices=devs[:1])
        d = np.max(np.abs(got.astype(np.complex128) - ref.astype(np.complex128)))
        rms = np.sqrt(np.mean(np.abs(ref.astype(np.complex128)) ** 2))
        card["kernels_eps_x_rms"][name] = float(d / (np.finfo(np.float32).eps * rms))
    print(f"kernel parity vs numpy (eps x rms): {card['kernels_eps_x_rms']}")

    rc = 0
    for f_NL in (0.0, 10.0):
        _mh, host = gen(icgen.generate_t9_slabs, fdtype=np.float32, f_NL=f_NL)
        _m1, one = gen(icgen.generate_t9_slabs_device, fdtype=np.float32, f_NL=f_NL,
                       noise="host", devices=devs[:1])
        _m4, four = gen(icgen.generate_t9_slabs_device, fdtype=np.float32, f_NL=f_NL,
                        noise="host", devices=devs[:4])
        same = all(np.array_equal(getattr(one, k), getattr(four, k))
                   for k in ("occupancy", "off", "w", "vel_scale"))
        occ = np.array_equal(four.occupancy, host.occupancy)
        row = dict(four_eq_one=same, occupancy_equal=occ)
        if occ:
            doff = np.abs(four.off.astype(np.int64) - host.off.astype(np.int64))
            dw = np.abs(four.w.astype(np.int64) - host.w.astype(np.int64))
            row.update(off_max=int(doff.max()), off_n=int((doff > 0).sum()),
                       w_max=int(dw.max()), w_n=int((dw > 0).sum()), rows=int(dw.size),
                       scale_rel=float(np.max(np.abs(four.vel_scale / host.vel_scale - 1))))
        print(f"f_NL={f_NL}: {row}")
        if not (same and occ and row["off_max"] <= 1 and row["w_max"] <= 1):
            print("MISS: a pre-registered smoke gate failed")
            rc = 3
        card[str(f_NL)] = row
    with open(args.out, "w") as fh:
        json.dump(card, fh, indent=2)
    return rc


class _Window:
    """A host field that holds only planes [g0, g0 + len) of a larger box, indexed globally."""

    def __init__(self, arr, g0):
        self.arr, self.g0 = arr, g0

    def __getitem__(self, sl):
        return self.arr[sl.start - self.g0:sl.stop - self.g0]

    def read_slab(self, lo, hi):
        return self.arr[lo - self.g0:hi - self.g0]


def cmd_emit_shape(args):
    """One destination slab of the card emission at production shapes, on ONE card.

    Synthetic fields over its three source slabs only (u within the window, v ~ N(0,1)).
    Reports the card's peak (`memory_stats`, never reset -- hence its own process), the
    per-source and per-destination seconds, and projects the whole emission on four
    cards: the peak with this process's 3-slab u_x shard swapped for the real halo shard,
    and nb destinations over the cards. rc 3 if the projected card peak exceeds
    STOP_CARD_FRAC x GB_CARD.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    from inexor.codec import T9Layout
    from inexor.device import emit
    from inexor.plan import ic_device_stages

    n, nb, window, W = args.n, args.nb, 1, 4
    box, dt = n / 2, np.dtype(np.float32)
    p = n // nb
    d = nb // 2
    g0, g1 = (d - window) * p, (d + window + 1) * p
    t9 = T9Layout(box, n, 2)
    rng = np.random.default_rng(0)
    depth = box / nb
    shape = (g1 - g0, n, n)
    t0 = time.perf_counter()
    u = [(rng.random(shape, dtype=np.float32) * 1.8 - 0.9) * np.float32(depth) for _ in range(3)]
    v = [rng.standard_normal(shape, dtype=np.float32) for _ in range(3)]
    print(f"fields for {g1 - g0} planes of {n}^2: {time.perf_counter() - t0:.0f} s", flush=True)
    dev = jax.devices()[0]
    shard = dict(lo=d, hi=d + 1, x0=g0, nx=g1 - g0, device=dev,
                 delta=jax.device_put(u[0], dev))
    t = {}
    out = tempfile.mkdtemp(dir=args.tmp)
    t0 = time.perf_counter()
    names, rows = emit.emit_t9_slabs_cards(out, [shard], _Window(u[1], g0), _Window(u[2], g0),
                                           [_Window(a, g0) for a in v], t9, n, box, nb, dt,
                                           window, timings=t, complete=False)
    wall = time.perf_counter() - t0
    peak = (dev.memory_stats() or {}).get("peak_bytes_in_use")
    shard_b = (g1 - g0) * n * n * dt.itemsize
    halo_b = (nb // W + 2 * window) * p * n * n * dt.itemsize
    proj_peak = None if peak is None else peak - shard_b + halo_b
    _h, card, _d = ic_device_stages(n, n_gpus=W, nb=nb)
    # per card: nb/W destinations and nb/W + 2*window sources; the first call compiled
    dest_each = t["dest_s"] + t["write_s"]
    src_each = (t["source_s"] + t["upload_s"]) / (2 * window + 1)
    proj_s = (nb // W) * dest_each + (nb // W + 2 * window) * src_each
    res = dict(n=n, nb=nb, slab=d, rows=rows, wall_s=wall, split_s=t,
               card_peak_gb=None if peak is None else peak / GB,
               card_peak_projected_gb=None if proj_peak is None else proj_peak / GB,
               planner_card_emission_gb=card["6 emission"] / GB,
               emission_projected_s_per_card=proj_s, cold_includes_compile=True)
    print(json.dumps(res, indent=2))
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    for f in names:
        os.remove(os.path.join(out, f))
    if proj_peak is not None and proj_peak / GB > STOP_CARD_FRAC * GB_CARD:
        print(f"STOP: projected card peak {proj_peak / GB:.0f} GB > "
              f"{STOP_CARD_FRAC} x {GB_CARD} GB")
        return 3
    return 0


def cmd_project(args):
    with open(args.card) as fh:
        c = json.load(fh)
    n0 = int(c["n_part"])
    stages = c["manifest"]["stage_s"]
    growth = 8.0 * math.log(2 * n0) / math.log(n0)  # n^3 log n per side-doubling
    proj = {k: v * growth for k, v in stages.items()}
    wall_h = c["wall_s"] * growth / 3600
    host_gb = c["peak_rss_bytes"] * 8 / GB
    gpu_peak_gb = None
    if args.gpu_log and os.path.exists(args.gpu_log):
        vals = []
        with open(args.gpu_log) as fh:
            for line in fh:
                f = [x.strip() for x in line.split(",")]
                if len(f) >= 2 and f[1].split()[0].isdigit():
                    vals.append(int(f[1].split()[0]))  # MiB
        if vals:
            gpu_peak_gb = max(vals) * 2**20 / GB
    out = dict(from_card=args.card, n_from=n0, n_to=2 * n0, growth=growth,
               stage_s_from=stages, stage_s_projected=proj, wall_h_projected=wall_h,
               host_peak_gb_from=c["peak_rss_bytes"] / GB, host_peak_gb_projected=host_gb,
               gpu_peak_gb_from=gpu_peak_gb,
               gpu_peak_gb_projected=None if gpu_peak_gb is None else gpu_peak_gb * 8)
    print(json.dumps(out, indent=2))
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    rc = 0
    if wall_h > STOP_WALL_H:
        print(f"STOP: projected {2 * n0}^3 wall {wall_h:.1f} h > {STOP_WALL_H} h")
        rc = 3
    if host_gb > STOP_HOST_FRAC * GB_HOST:
        print(f"STOP: projected host peak {host_gb:.0f} GB > {STOP_HOST_FRAC} x {GB_HOST} GB")
        rc = 3
    return rc


def cmd_allocator_ab(args):
    """Two IC generations, one per allocator, SAME job and same node -> stage ratios.

    998798's 1.15x against 997814 crossed two jobs and two node allocations, which is
    the shape that produced the 8.6% cross-job drift record 5h had to chase down. Both
    arms here run back to back in one job, so the ratio is the allocator.

    The arms must differ in the allocator and NOTHING else: `manifest.provenance`
    carries the allocator, so the arm is identified by what the generator recorded
    rather than by the label the sbatch passed. rc 3 on a mismatched axis or on two
    arms that turn out to be the same allocator -- a vacuous A/B must not read as a
    null result.
    """
    arms = []
    for path in (args.a, args.b):
        with open(path) as fh:
            c = json.load(fh)
        prov = (c.get("manifest", {}) or {}).get("provenance") or {}
        arms.append(dict(path=path, card=c, prov=prov,
                         allocator=prov.get("allocator", "unrecorded")))
    a, b = arms
    rc = 0
    if a["allocator"] == "unrecorded" or b["allocator"] == "unrecorded":
        print("REFUSE: an arm's manifest carries no provenance.allocator -- it was "
              "generated before the provenance fix and cannot identify itself")
        rc = 3
    elif a["allocator"] == b["allocator"]:
        print(f"REFUSE: both arms report allocator {a['allocator']} -- not an A/B")
        rc = 3
    # `allocator` is the environment verbatim, which is the right thing to store and
    # the wrong thing to trust alone: a CPU-backend generation records the variable it
    # was handed while no GPU allocator ever ran, so the label would be live and the
    # quantity vacuous. Caught on the laptop, where exactly that happened.
    for arm in arms:
        devs = arm["prov"].get("devices")
        if devs is not None and all(d == "cpu" for d in devs):
            print(f"REFUSE: {arm['path']} ran on {devs} -- no GPU allocator was "
                  f"exercised, so its {arm['allocator']} label means nothing")
            rc = 3
    # everything that is not the allocator must match, or the ratio is not about it
    for key, get in (("n_part", lambda c: c["n_part"]),
                     ("generator", lambda c: c["generator"]),
                     ("commit", lambda c: c.get("commit")),
                     ("host", lambda c: c.get("host")),
                     ("emission", lambda c: c["manifest"].get("emission")),
                     ("ic_stream", lambda c: c["manifest"].get("ic_stream")),
                     ("n_devices", lambda c: c["manifest"].get("n_devices"))):
        va, vb = get(a["card"]), get(b["card"])
        if va != vb:
            print(f"REFUSE: arms differ on {key}: {va!r} vs {vb!r}")
            rc = 3
    sa = a["card"]["manifest"]["stage_s"]
    sb = b["card"]["manifest"]["stage_s"]
    ratios = {stage: sb[stage] / sa[stage] for stage in sa
              if stage in sb and sa[stage]}
    wall = b["card"]["wall_s"] / a["card"]["wall_s"]
    host = b["card"]["peak_rss_bytes"] / a["card"]["peak_rss_bytes"]
    # A refused comparison keeps its numbers on the card for the audit trail and OFF
    # the screen: a ratio printed under a REFUSE line is still the thing a reader
    # carries away, and these ratios are exactly the shape of a real result.
    if rc:
        print(f"\nWITHHELD: {len(ratios)} stage ratios computed and written to "
              f"{args.out}, not printed -- the comparison above is refused, so they "
              f"are not attributable to the allocator.")
    else:
        print(f"\n{'stage':<16}{a['allocator']:>14}{b['allocator']:>14}{'b/a':>9}")
        for stage, r in ratios.items():
            print(f"{stage:<16}{sa[stage]:>14.1f}{sb[stage]:>14.1f}{r:>9.2f}")
        print(f"{'wall':<16}{a['card']['wall_s']:>14.1f}"
              f"{b['card']['wall_s']:>14.1f}{wall:>9.2f}")
        print(f"{'host peak GB':<16}{a['card']['peak_rss_bytes'] / GB:>14.1f}"
              f"{b['card']['peak_rss_bytes'] / GB:>14.1f}{host:>9.2f}")
    out = dict(a=dict(path=a["path"], allocator=a["allocator"], provenance=a["prov"],
                      stage_s=sa, wall_s=a["card"]["wall_s"],
                      peak_rss_bytes=a["card"]["peak_rss_bytes"]),
               b=dict(path=b["path"], allocator=b["allocator"], provenance=b["prov"],
                      stage_s=sb, wall_s=b["card"]["wall_s"],
                      peak_rss_bytes=b["card"]["peak_rss_bytes"]),
               stage_ratio_b_over_a=ratios, wall_ratio_b_over_a=wall,
               host_peak_ratio_b_over_a=host, n_part=a["card"]["n_part"], rc=rc)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    # reported, not gated: this measures a wall, and which allocator to run is a
    # capacity question the 4096^3 peak decides, not this ratio
    print(f"\ncard -> {args.out}")
    return rc


def cmd_cleanup(args):
    man_path = os.path.join(args.workdir, "manifest.json")
    with open(man_path) as fh:
        man = json.load(fh)
    removed = 0
    for name in list(man["files"]) + ["realization_ics.json", "manifest.json"]:
        p = os.path.join(args.workdir, name)
        if os.path.exists(p):
            removed += os.path.getsize(p)
            os.remove(p)
    try:
        os.rmdir(args.workdir)
        print(f"removed {removed / GB:,.1f} GB and {args.workdir}")
    except OSError as e:
        print(f"removed {removed / GB:,.1f} GB; {args.workdir} left standing: {e} "
              f"({sorted(os.listdir(args.workdir))})")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--root", required=True)
    p.add_argument("--need-gb", type=float, default=1500.0)
    p = sub.add_parser("smoke")
    p.add_argument("--out", required=True)
    p.add_argument("--tmp", default=None)
    p = sub.add_parser("emit-shape")
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--nb", type=int, default=256)
    p.add_argument("--out", required=True)
    p.add_argument("--tmp", default=None)
    p = sub.add_parser("project")
    p.add_argument("--card", required=True)
    p.add_argument("--gpu-log", default=None)
    p.add_argument("--out", required=True)
    p = sub.add_parser("allocator-ab")
    p.add_argument("--a", required=True, help="IC card for arm A")
    p.add_argument("--b", required=True, help="IC card for arm B")
    p.add_argument("--out", required=True)
    p = sub.add_parser("cleanup")
    p.add_argument("--workdir", required=True)
    args = ap.parse_args()
    return {"preflight": cmd_preflight, "smoke": cmd_smoke, "emit-shape": cmd_emit_shape,
            "project": cmd_project, "allocator-ab": cmd_allocator_ab,
            "cleanup": cmd_cleanup}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
