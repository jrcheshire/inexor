"""M-v2-2: is the pinned host gather an ffi build item, or can jax express it?

WHAT THIS DECIDES. D-v2-16 clause 6 measured the streaming constraint as the HOST
GATHER (12.2 GB/s at a 37 KB brick span) rather than the transfer (pinned H2D
176-221 GB/s, flat in run length), and priced the fix at 6.9x on `stage` at f64
and 18x at T9 -- **contingent on gathering directly into cudaHostAlloc-backed
memory, "which jax cannot express"**. That clause is why the plan carries an
ffi-level build item.

But V3/G4 already ran `staged` mode against a `pinned_host` memory kind on this
same stack, so jax can plainly ALLOCATE pinned buffers. What is unproven is the
narrower claim: whether a scattered host gather can land its output IN one
without an extra full-size copy. If it can, the 18x is reachable with no ffi
extension at all and the build item disappears; if it cannot, this measures the
copy that the extension exists to delete, which is the number that justifies
writing it.

Either answer is worth having before anyone writes CUDA, which is the whole
point of running it first.

WHAT IT REPORTS, per path:
  - the scattered gather alone (numpy fancy-index over brick-span runs)
  - the host->device transfer alone
  - the total, in GB/s of PAYLOAD moved, at both f64 positions (what staging
    moves today) and the 9 B T9 record (what it will move)

Deliberately NOT measured: anything about the force, or about whether the tiled
arm gets faster. This is a memory-path microbenchmark and its numbers describe
the harness, not the engine -- read it against
`reference-benchmark-measures-the-harness`.

Run:
    python scripts/v2_m2_pinned_gather.py --n-rows 40000000 --runs 512
"""

import argparse
import json
import os
import time

import numpy as np


def _time(fn, reps=5, warmup=2):
    """Min of `reps`, after warmup. Min rather than mean: we are after the
    achievable rate of one path, and a laptop/shared-node scheduler only ever
    adds time."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def build_indices(n_rows, n_runs, run_len, seed=0):
    """Scattered runs of `run_len` contiguous rows -- the brick-span access
    pattern, not uniform random scatter.

    This matters: `choose_brick` forces brick <= buffer, which fixes the run at
    4096 particles = 37 KB at T9, and D-v2-16 clause 6's 12.2 GB/s was measured
    at exactly that span. Uniform random indices would measure a different and
    much worse access pattern that the layout never produces.
    """
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_rows - run_len, n_runs, dtype=np.int64)
    return (starts[:, None] + np.arange(run_len, dtype=np.int64)[None, :]).reshape(-1)


def probe_memory_kinds():
    """What this stack actually exposes. The G4 record warns that only
    ['device', 'pinned_host'] appear here and that ATS/'coherent' is untested,
    so the vocabulary is reported rather than assumed."""
    import jax

    dev = jax.devices()[0]
    try:
        kinds = sorted(m.kind for m in dev.addressable_memories())
    except Exception as exc:  # pragma: no cover - platform dependent
        kinds = [f"<unavailable: {exc}>"]
    return dict(platform=dev.platform, device=str(dev), memory_kinds=kinds)


def pinned_sharding():
    """A SingleDeviceSharding on pinned_host, or None if the stack lacks it."""
    import jax

    try:
        return jax.sharding.SingleDeviceSharding(jax.devices()[0], memory_kind="pinned_host")
    except Exception:
        return None


def can_write_into_pinned(shape, dtype):
    """THE question clause 6 turns on: is a pinned_host buffer writable in place?

    If numpy can take a writable view of one, a gather can target it directly and
    the ffi extension is unnecessary. jax arrays are immutable by contract, so
    the expected answer is no -- but the cost of being wrong about this is
    writing a CUDA extension we did not need, so it is checked rather than
    reasoned about.
    """
    import jax
    import jax.numpy as jnp

    sh = pinned_sharding()
    if sh is None:
        return dict(supported=False, reason="no pinned_host memory kind on this stack")
    buf = jax.device_put(jnp.zeros(shape, dtype), sh)
    try:
        view = np.asarray(buf)
        writable = bool(view.flags.writeable)
        if writable:  # would be a genuine in-place target; verify it sticks
            view[0] = 1
            writable = bool(np.asarray(buf)[0] == 1)
        return dict(supported=True, writable_view=writable)
    except Exception as exc:
        return dict(supported=True, writable_view=False, reason=str(exc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rows", type=int, default=40_000_000, help="source rows on the host")
    ap.add_argument("--runs", type=int, default=512, help="brick-span runs per gather")
    ap.add_argument("--run-len", type=int, default=4096, help="rows per run (a brick)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    info = probe_memory_kinds()
    print(f"backend: {info['platform']}  {info['device']}")
    print(f"memory kinds: {info['memory_kinds']}")
    if info["platform"] == "cpu":
        print("!! CPU backend: the pinned-vs-pageable distinction does not exist here.")

    idx = build_indices(args.n_rows, args.runs, args.run_len)
    n_gather = len(idx)
    print(f"source {args.n_rows:,} rows;  gather {n_gather:,} rows "
          f"in {args.runs} runs of {args.run_len}")

    rng = np.random.default_rng(1)
    # f64 positions = what staging moves today; the T9 record = what it will move
    payloads = {
        "f64_xyz": np.ascontiguousarray(rng.normal(size=(args.n_rows, 3)).astype(np.float64)),
        "t9_9byte": np.ascontiguousarray(
            rng.integers(0, 255, size=(args.n_rows, 9), dtype=np.uint8)
        ),
    }

    sh = pinned_sharding()
    writable = can_write_into_pinned((1024,), jnp.float64)
    print(f"pinned_host buffer writable in place: {writable}")

    results = dict(info=info, args=vars(args), pinned_writable=writable, paths={})
    for name, src in payloads.items():
        bytes_moved = n_gather * src.shape[1] * src.dtype.itemsize
        gb = bytes_moved / 1e9

        def gather():
            return src[idx]

        staged = src[idx]

        def to_device_pageable():
            jax.block_until_ready(jnp.asarray(staged))

        t_gather = _time(gather, reps=args.reps)
        t_h2d = _time(to_device_pageable, reps=args.reps)

        row = dict(
            bytes=int(bytes_moved),
            gather_s=t_gather,
            gather_gbps=gb / t_gather,
            h2d_pageable_s=t_h2d,
            h2d_pageable_gbps=gb / t_h2d,
        )

        if sh is not None and info["platform"] != "cpu":
            def to_pinned():
                jax.block_until_ready(jax.device_put(staged, sh))

            pinned = jax.device_put(staged, sh)
            # A bare device is REJECTED as a destination when the source carries a
            # memory kind ("Memory kind mismatch with xla::PjRtBuffers", job 397);
            # the destination has to be a sharding that names `device` explicitly.
            dev_sh = jax.sharding.SingleDeviceSharding(
                jax.devices()[0], memory_kind="device"
            )

            def pinned_to_device():
                jax.block_until_ready(jax.device_put(pinned, dev_sh))

            t_pin = _time(to_pinned, reps=args.reps)
            t_p2d = _time(pinned_to_device, reps=args.reps)
            row.update(
                host_to_pinned_s=t_pin,
                host_to_pinned_gbps=gb / t_pin,
                pinned_to_device_s=t_p2d,
                pinned_to_device_gbps=gb / t_p2d,
                # what the ffi would delete: the extra full-size copy from the
                # gather's pageable output into the pinned buffer
                ffi_would_save_s=t_pin,
                total_via_pinned_s=t_gather + t_pin + t_p2d,
                total_via_pageable_s=t_gather + t_h2d,
            )
        results["paths"][name] = row

        print(f"\n[{name}] {gb:.3f} GB of payload")
        for k, v in row.items():
            if k.endswith("_gbps"):
                print(f"    {k:<26} {v:8.2f} GB/s")
            elif k.endswith("_s"):
                print(f"    {k:<26} {v * 1e3:8.1f} ms")

    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "runs", "v2", "m2_pinned_gather.json",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
