# Is the pinned host gather an ffi build item? Partly, and the 18x needs re-deriving

**Measurement record, not a verdict.** deneb job 398 (RTX 3050, jax 0.10.2,
branch `jc/v2-force-promote` @ `e462d94`), two legs, ~2 minutes. Cards:
`m2_pinned_gather_brickspan.json`, `m2_pinned_gather_longrun.json`.
Nothing here amends D-v2-16.

## What was asked

Clause 6 prices the streaming fix at 6.9x on `stage` at f64 and **18x at T9**,
contingent on gathering into `cudaHostAlloc`-backed memory, "which jax cannot
express" -- which is why the plan carries an ffi build item. But V3/G4 ran
`staged` mode against a `pinned_host` memory kind on this same stack, so jax can
plainly *allocate* pinned buffers. Only the narrower claim was unproven: can a
scattered gather land its output in one without an extra full-size copy?

## The answer to that question: no, and the copy is measured

**`pinned_host` exists and is NOT writable in place.** The stack exposes
`['device', 'pinned_host']`, `jax.device_put(..., memory_kind="pinned_host")`
works, and `np.asarray()` of the result is a read-only view. So a numpy gather
cannot target one, and **clause 6's premise stands: a zero-copy
gather-into-pinned needs the ffi.**

What the extension would delete is the host->pinned copy, measured at **4.0 ms of
a 20.6 ms path at f64 (19%) and 1.0 ms of 13.6 ms at T9 (7%)**.

| leg | payload | gather | H2D pageable | host->pinned | pinned->device | total pageable | total pinned |
|---|---|---|---|---|---|---|---|
| 512 x 4096 | f64 50.3 MB | 13.1 ms | 6.3 ms | 4.0 ms | 3.9 ms | **19.4** | **20.5** |
| 512 x 4096 | T9 18.9 MB | 11.1 ms | 2.2 ms | 1.0 ms | 1.5 ms | **13.4** | **13.6** |
| 32 x 65536 | f64 50.3 MB | 12.7 ms | 6.3 ms | 4.0 ms | 3.9 ms | **19.0** | **20.6** |
| 32 x 65536 | T9 18.9 MB | 11.0 ms | 2.2 ms | 1.0 ms | 1.5 ms | **13.2** | **13.5** |

## THE FINDING: the gather is per-ROW, not per-byte

Both payloads gather the **same 2,097,152 rows**. The T9 record is 9 B against
f64's 24 B -- **2.66x fewer bytes -- and takes only 1.16x less time** (11.0 vs
12.7 ms), so its apparent bandwidth is *worse* (1.71 vs 3.95 GB/s). A scattered
fancy-index is bound by per-row index and cache-miss work, not by the bytes it
moves.

**This puts clause 6's ordering in question.** 18x at T9 against 6.9x at f64 is a
*larger* win for the narrower record, which is what you would expect if the
bottleneck scaled with bytes. Measured, the bottleneck does not. That does not
make 18x wrong -- job 896408's derivation is not in front of me and it measured a
different machine -- but the two statements need reconciling before the 18x is
quoted again, and the reconciliation is owed to whoever writes the extension.

**Run length is irrelevant here, which the control leg exists to say.** 512 runs
of 4096 (the 37 KB brick span) and 32 runs of 65536 give the same gather time to
within 3%. Clause 6 records pinned H2D as flat in run length while the gather was
not; on this CPU both are flat, so the two cannot be separated by run length at
these sizes.

## What does NOT transfer, and it is the headline caveat

**deneb is an RTX 3050 over PCIe at ~13 GB/s. C-gh is a GH200 with NVLink-C2C at
the 176-221 GB/s clause 6 measured -- roughly 15x.** So:

- the finding that **the pinned path is net SLOWER here** (20.5 vs 19.4 ms) is a
  statement about PCIe, **not** about C-gh, and must not be read as "the ffi is
  not worth it". The extra copy costs 4.0 ms while the faster transfer saves only
  2.4 ms; at a 15x fabric the second term dominates and the sign flips.
- the transfer columns are therefore uninformative about production.
- the **gather** columns are a host-CPU property and are the part worth carrying.

## What this changes about the plan

1. **The ffi build item survives** -- pinned is not writable in place, so the
   copy is real and only an extension removes it.
2. **But the gather is 67-83% of the path and the ffi does not make it faster.**
   It fuses gather+copy, deleting the copy; the gather work remains. Any
   projection that attributes the whole 18x to the extension is attributing to it
   a cost it does not touch.
3. **The 18x wants re-deriving at T9** against the per-row result above.
4. A C-gh-relevant number needs a Vista `gh` leg; deneb cannot produce one,
   and the sbatch here is reusable as-is.

## Cost

Two deneb jobs, ~4 minutes total, zero SU. Job 397 died on my own bug
(`jax.device_put` to a bare device is rejected when the source carries a memory
kind) and its sbatch guard did the right thing: zero cards written -> `FATAL` ->
exit 1, rather than a green row over nothing.
