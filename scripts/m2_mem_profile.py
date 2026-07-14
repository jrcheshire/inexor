"""M2 S4 (R6-class): peak device-memory profile of the exact-replay adjoint.

Architecture Sec. 9 budgets the 1024^3 flagship at a 36 GiB persistent scan
carry + per-reverse-step transients -> "realistic peak ~65-75 GiB, fits 80 GB
with no slack to waste", and tags that budget
**[M0: R6-class profiling at 512^3 before believing the extrapolation]**.
This script IS that profiling. It measures rather than assumes: device totals,
peaks, and the carry all come from the runtime or from dtypes, never from the
Sec. 9 prose.

Per (n_mesh, checkpoint) config it reports:
  - peak after ICs (the LPT/IC baseline),
  - forward-only peak,
  - full-adjoint peak (forward + reverse sweep),
  - the EXACT analytic carry (from dtypes, not the doc's rounded GiB),
and extrapolates the measured peak to the 1024^3 flagship linearly in n_part.

Each config runs in a FRESH SUBPROCESS. XLA's peak_bytes_in_use is monotonic
within a process and has no reset API, so a second config in the same process
inherits the first's high-water mark. One process per config is the only way to
get an uncontaminated peak.

CEILING-1 (Sec. 9): does XLA alias the 36 GiB scan carry in place, or
double-buffer it (-> 72 GiB, busting the budget)? The reverse sweep is
`jax.lax.scan` (adjoint.py `_evolve_grad_bwd`) REGARDLESS of the forward
`driver`, so this is a live question. Discriminator: carry and transients both
scale as n_part, so no n-sweep separates them -- instead compare the measured
peak against the two competing predictions at the SAME n:
    aliased      peak ~= 1 x carry + transients
    double-buff  peak ~= 2 x carry + transients
At 512^3 that is ~9 GiB vs ~13.5 GiB using Sec. 9's own transient estimate, a
~50% split that a measurement resolves. This script PRINTS both predictions and
the measured value; it deliberately ratifies NO verdict -- the driver decision
(keep scan vs promote a per-step-jit donated bwd, mirroring integrate.run_perstep)
is JC's at the S4 gate, per the tolerances convention.

The checkpoint A/B tests Sec. 9's other empirical claim: the within-step
`jax.checkpoint` around the paint/read (adjoint.py rev_body) is credited with
"~18% peak-VJP saving" -- a number inherited from mbody, never measured here.

Compute placement: Vista GH200 via scripts/m2_vista.sbatch (the 512^3 leg needs
~9 GiB predicted; it OOMs deneb's 6 GB 3050 -- that is why M1's 512^3 smoke was
deferred here). Smaller n run anywhere. On a CPU backend JAX exposes no
memory_stats, so peaks report null and the run is a plumbing smoke only.

Run:
    pixi run -e gpu python scripts/m2_mem_profile.py --n 64,128,256,512
    pixi run python scripts/m2_mem_profile.py --n 32,64          # CPU smoke
Outputs runs/m2/mem_profile.json + a printed table.
"""

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUNS = os.path.join(REPO, "runs", "m2")

# Persistent scan-carry bytes per particle, EXACT from the dtypes that
# _evolve_grad_bwd actually carries: (x_int uint16 x3, w_int int16 x3,
# xbar f32 x3, wbar f32 x3) -- architecture Sec. 9's table, recomputed not copied.
CARRY_BYTES_PER_PARTICLE = 3 * (2 + 2 + 4 + 4)  # = 36 B -> 36.0 GiB at 1024^3
GIB = 1024.0**3
FLAGSHIP_N = 1024

# The claim is "fits an 80 GB GPU". MIND THE UNITS: an 80 GB card is 80e9 bytes
# = 74.5 GiB, so Sec. 9's budget of "~65-75 GiB -- fits 80 GB with no slack to
# waste" is optimistic at its own upper end: 75 GiB = 80.5 GB, which does NOT
# fit. The real headroom is thinner than the prose reads. Compare in bytes.
FLAGSHIP_LIMIT_BYTES = 80e9

# Sec. 9's transient estimate, used ONLY to place the two ceiling-1 predictions
# on the same axis as the measurement. Peak 65-75 GiB minus the 36 GiB carry
# => 29-39 GiB of transients at 1024^3, i.e. ~27-36 B/particle. Midpoint here;
# the printed predictions are a reading aid, NOT a gate.
TRANSIENT_BYTES_PER_PARTICLE_EST = 32


def _fmt_gib(b):
    return "     -" if b is None else f"{b / GIB:7.3f}"


# ---------------------------------------------------------------------------
# single-config measurement (runs in its own process; --single)
# ---------------------------------------------------------------------------


def _run_single(n_mesh, K, driver, checkpoint):
    import jax
    import jax.numpy as jnp

    # A/B for Sec. 9's "~18% peak-VJP saving": neutralize the within-step
    # rematerialization. adjoint.rev_body looks `jax.checkpoint` up at call
    # time, so patching the module attribute before the sweep runs is enough.
    if not checkpoint:
        jax.checkpoint = lambda f, *a, **k: f

    from inexor import PLANCK, BoxConfig, QuantConfig, TimeConfig
    from inexor.adjoint import evolve_grad
    from inexor.ic import linear_density
    from inexor.losses import band_power_loss, fundamental_k_edges
    from inexor.lpt import lpt_ics

    fdtype = jnp.float32  # production dtype; x64 is a gate/FD concern, not a memory one
    box = BoxConfig(n_mesh=n_mesh, box_size=256.0)
    time = TimeConfig(a_init=0.1, a_final=1.0, n_steps=K, integrator="bullfrog")
    quant, cosmo = QuantConfig(), PLANCK

    dev = jax.devices()[0]

    def peak():
        try:
            return (dev.memory_stats() or {}).get("peak_bytes_in_use")
        except Exception:  # CPU backend / no stats support
            return None

    def limit():
        try:
            return (dev.memory_stats() or {}).get("bytes_limit")
        except Exception:
            return None

    # ---- ICs ----
    key = jax.random.PRNGKey(0)
    d0 = linear_density(key, n_mesh, box.box_size, cosmo, f_NL=5.0, fdtype=fdtype)
    x0, v0 = lpt_ics(d0, box.box_size, time.a_init, cosmo, order=2, fdtype=fdtype)
    x0, v0 = jax.block_until_ready(x0), jax.block_until_ready(v0)
    del d0
    peak_ic = peak()

    # Band-power loss with a zeros target: the cheapest realistic scalar. A
    # field-L2 target would hold an extra n^3 reference field on device and
    # contaminate the very number we are measuring.
    k_edges = fundamental_k_edges(n_mesh, box.box_size, n_bins=8)
    zeros_pk = jnp.zeros(len(k_edges) - 1, dtype=fdtype)

    def loss(x0_, v0_):
        x_f, _v_f = evolve_grad(box, time, quant, cosmo, driver, fdtype, x0_, v0_)
        return band_power_loss(x_f, box, k_edges, zeros_pk)

    # ---- forward-only peak ----
    jax.block_until_ready(loss(x0, v0))
    peak_fwd = peak()

    # ---- full adjoint peak (fwd + reverse sweep) ----
    g = jax.grad(loss, argnums=(0, 1))(x0, v0)
    jax.block_until_ready(g)
    peak_adj = peak()

    n_part = int(x0.shape[0])
    return {
        "n_mesh": n_mesh,
        "n_particles": n_part,
        "K": K,
        "driver": driver,
        "checkpoint": checkpoint,
        "platform": dev.platform,
        "device": str(dev),
        "bytes_limit": limit(),
        "carry_bytes_analytic": n_part * CARRY_BYTES_PER_PARTICLE,
        "peak_after_ic": peak_ic,
        "peak_forward": peak_fwd,
        "peak_adjoint": peak_adj,
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def _spawn(n_mesh, K, driver, checkpoint):
    """Run one config in a fresh process; return its JSON record (or an error)."""
    cmd = [
        sys.executable, os.path.abspath(__file__), "--single",
        "--n", str(n_mesh), "--steps", str(K), "--driver", driver,
    ]
    if not checkpoint:
        cmd.append("--no-checkpoint")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        tail = (p.stderr or "").strip().splitlines()
        # An OOM is a RESULT (it maps the ceiling), not a crash to hide.
        return {
            "n_mesh": n_mesh, "K": K, "driver": driver, "checkpoint": checkpoint,
            "error": tail[-1] if tail else f"exit {p.returncode}",
            "oom": any("RESOURCE_EXHAUSTED" in ln or "Out of memory" in ln for ln in tail),
        }
    return json.loads(p.stdout.strip().splitlines()[-1])


def _report(recs):
    print("\n===== R6 peak device memory: exact-replay adjoint =====")
    lim = next((r.get("bytes_limit") for r in recs if r.get("bytes_limit")), None)
    dev = next((r.get("device") for r in recs if r.get("device")), "?")
    print(f"  device: {dev}   HBM limit: {_fmt_gib(lim)} GiB"
          if lim else f"  device: {dev}   HBM limit: unreported (CPU backend?)")
    print("\n  n_mesh  ckpt   carry     ICs     fwd  adjoint   adj/carry   note")
    for r in recs:
        if r.get("error"):
            print(f"  {r['n_mesh']:>6}  {str(r['checkpoint'])[:1]}      "
                  f"{'OOM' if r.get('oom') else 'ERR':>6}   {r['error'][:44]}")
            continue
        carry, adj = r["carry_bytes_analytic"], r["peak_adjoint"]
        ratio = f"{adj / carry:8.2f}x" if (adj and carry) else "       -"
        print(f"  {r['n_mesh']:>6}  {str(r['checkpoint'])[:1]}   {_fmt_gib(carry)} "
              f"{_fmt_gib(r['peak_after_ic'])} {_fmt_gib(r['peak_forward'])} "
              f"{_fmt_gib(adj)}  {ratio}")

    # ---- checkpoint A/B (Sec. 9's inherited "~18% peak-VJP saving") ----
    print("\n  --- within-step jax.checkpoint A/B (Sec. 9 claims ~18% peak-VJP saving) ---")
    by = {(r["n_mesh"], r["checkpoint"]): r for r in recs if not r.get("error")}
    for n in sorted({n for n, _ in by}):
        on, off = by.get((n, True)), by.get((n, False))
        if on and off and on["peak_adjoint"] and off["peak_adjoint"]:
            saving = 1.0 - on["peak_adjoint"] / off["peak_adjoint"]
            print(f"  n={n:<5} on {_fmt_gib(on['peak_adjoint'])}  off {_fmt_gib(off['peak_adjoint'])}"
                  f"  -> saving {saving * 100:5.1f}%")
        elif on and not off:
            print(f"  n={n:<5} checkpoint-off leg not run (--ab to enable)")

    # ---- ceiling-1 reading aid + flagship extrapolation ----
    print("\n  --- ceiling-1: does XLA alias the scan carry, or double-buffer it? ---")
    print("  (predictions use Sec. 9's OWN transient estimate ~"
          f"{TRANSIENT_BYTES_PER_PARTICLE_EST} B/particle; they are a reading aid, NOT a gate)")
    for r in recs:
        if r.get("error") or not r.get("peak_adjoint"):
            continue
        npart, adj = r["n_particles"], r["peak_adjoint"]
        aliased = npart * (CARRY_BYTES_PER_PARTICLE + TRANSIENT_BYTES_PER_PARTICLE_EST)
        double = npart * (2 * CARRY_BYTES_PER_PARTICLE + TRANSIENT_BYTES_PER_PARTICLE_EST)
        near = "aliased" if abs(adj - aliased) < abs(adj - double) else "DOUBLE-BUFFERED"
        print(f"  n={r['n_mesh']:<5} ckpt={str(r['checkpoint'])[:1]}  measured {_fmt_gib(adj)}"
              f" | aliased {_fmt_gib(aliased)} | double {_fmt_gib(double)}  -> nearer: {near}")

    print("\n  --- flagship extrapolation (linear in n_particles) ---")
    print(f"  target: the 80 GB claim = {FLAGSHIP_LIMIT_BYTES / GIB:.1f} GiB"
          f" (80e9 B). Sec. 9 predicts 65-75 GiB -- note its OWN upper end,"
          f" 75 GiB = {75 * GIB / 1e9:.1f} GB, does NOT fit.")
    for r in recs:
        if r.get("error") or not r.get("peak_adjoint") or r["n_mesh"] >= FLAGSHIP_N:
            continue
        scale = (FLAGSHIP_N / r["n_mesh"]) ** 3
        proj = r["peak_adjoint"] * scale
        verdict = "fits" if proj < FLAGSHIP_LIMIT_BYTES else "BUSTS"
        print(f"  from n={r['n_mesh']:<5} ckpt={str(r['checkpoint'])[:1]}"
              f" -> 1024^3 projected {_fmt_gib(proj)} GiB = {proj / 1e9:5.1f} GB"
              f"   -> {verdict} the 80 GB claim")
    print("\n  Verdict + driver decision are JC's at the S4 gate; this script ratifies nothing.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", default="64,128,256", help="comma-separated n_mesh sweep")
    ap.add_argument("--steps", type=int, default=10, help="K (n_steps); carry is K-independent")
    ap.add_argument("--driver", default="perstep", choices=("scan", "perstep"),
                    help="FORWARD driver; the reverse sweep is lax.scan either way")
    ap.add_argument("--ab", action="store_true", help="also run the checkpoint-off leg")
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-checkpoint", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.single:
        rec = _run_single(int(args.n), args.steps, args.driver, not args.no_checkpoint)
        print(json.dumps(rec))
        return

    ns = [int(x) for x in args.n.split(",") if x.strip()]
    ckpts = [True, False] if args.ab else [True]
    recs = []
    for n in ns:
        for ck in ckpts:
            print(f"[m2_mem_profile] n_mesh={n} K={args.steps} driver={args.driver} "
                  f"checkpoint={ck} ...", flush=True)
            recs.append(_spawn(n, args.steps, args.driver, ck))

    _report(recs)
    os.makedirs(RUNS, exist_ok=True)
    path = os.path.join(RUNS, "mem_profile.json")
    with open(path, "w") as f:
        json.dump({
            "carry_bytes_per_particle": CARRY_BYTES_PER_PARTICLE,
            "transient_bytes_per_particle_est": TRANSIENT_BYTES_PER_PARTICLE_EST,
            "configs": recs,
        }, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
