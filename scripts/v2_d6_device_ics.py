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
    cleanup    remove a generation by NAME (manifest files, manifest, card), then rmdir,
               which refuses if anything else is inside.

The generations themselves run through `scripts/v2_m6_realization.py ics --generator device`.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

GB = 1e9
GB_HOST, GB_CARD = 1026.0, 199.0
STOP_WALL_H, STOP_HOST_FRAC = 9.0, 0.9


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
              f"per card {cp / GB:,.1f} GB ({cp / GB / GB_CARD:.2f}x), disk "
              f"{sum(disk.values()) / GB:,.0f} GB")
        if hp / GB > GB_HOST or cp / GB > GB_CARD:
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
    p = sub.add_parser("project")
    p.add_argument("--card", required=True)
    p.add_argument("--gpu-log", default=None)
    p.add_argument("--out", required=True)
    p = sub.add_parser("cleanup")
    p.add_argument("--workdir", required=True)
    args = ap.parse_args()
    return dict(preflight=cmd_preflight, smoke=cmd_smoke, project=cmd_project,
                cleanup=cmd_cleanup)[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
