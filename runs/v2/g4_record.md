# G4 -- the GH200 memory-path reality check (v2 seed V3)

**Status: PRE-REGISTRATION. Written 2026-08-06 while job 894010 was running,
before any ladder number existed.** Sections 1-4 are the reading rules and
predictions; section 5 is empty until the card lands. Nothing here
self-ratifies: the V3 exit is JC's call.

Provenance: `main @7a5ad00`. Probe `scripts/v2_g4_gh_memory.py`, job
`scripts/v2_g4_gh_vista.sbatch` (Vista `gh`, 8 h). Prior run: job 894005
(the reachability smoke, `gh-dev`, superseded -- see sec. 2).

## 1. The question, and why it is not the seed's question

C-gh is 2048^3 particles. At the ratified T9 state tier (D-v2-8, 9 B/p) that
is **77.3 GB of compressed state alone = 81% of the GH200's 96 GB HBM**
before a single mesh exists. So C-gh is real only if host-resident state is
reachable at useful bandwidth.

The seed (plan-plan V3) framed this as coherent (ATS) vs explicit
`pinned_host` staging vs infeasible. **That trichotomy is not expressible on
this stack.** Job 894005 measured the available memory kinds as
`['device', 'pinned_host']` -- there is no `unpinned_host` on jax 0.10.2 +
CUDA aarch64, though CPU JAX on the laptop reports all three. Pageable LPDDR
is therefore not addressable through JAX memory kinds at all, and **no
result from this job licenses the phrase "the coherent path works"**.

What the three arms actually contrast:

| arm | working set lives in | who moves the bytes |
|---|---|---|
| `hbm` | device HBM | nobody (control / ceiling) |
| `staged` | `pinned_host` | us, explicit `device_put` per chunk |
| `coherent` | `pinned_host` | XLA, via a jitted op with `out_shardings` on device |

So `coherent` vs `staged` is **implicit vs explicit staging out of page-locked
host memory**. That is a narrower question than the seed asked, and it is the
one the hardware and the framework will answer. Whether ATS is reachable by
some route outside JAX memory kinds is untested and out of scope here.

## 2. What is already established, and must not be re-derived

From job 894005 (`gh-dev`, 2026-08-06, ran to completion):

- **Memory kinds:** `['device', 'pinned_host']`. Node 212 GB total / 190 free;
  HBM reports 97871 MiB.
- **The instrument reproduces both known ceilings**, which is what makes the
  unknown arm's number readable at all (measure the reference's own floor
  first, or a number below it is a bound and not a measurement):
  - `hbm` 3021-3359 GB/s against the E2 card's **3400 GB/s** HBM;
  - `staged` 364-367 GB/s against E2's **375 GB/s** C2C read.
  These were not fitted to anything. They are the parity check.
- **Two defects found and fixed** (`7a5ad00`), each of which would have
  emptied the verdict column without failing loudly:
  1. `coherent` did not compile -- `E1200
     CompileTimeHostOffloadOutputLocationMismatch`. An op on host-resident
     data defaults its output to host memory space. It measured nothing.
  2. `staged` held **every** staged chunk on device at once (`pk/set` exactly
     1.000 at both smoke sizes): async dispatch outlives the `del`. It was
     not a slow streamer, it was not streaming. Above the cliff it would have
     OOM'd for a reason unrelated to the memory path.

Leg 1 of 894010 re-runs the small ladder and aborts the job if `coherent`
still errors. A fix is a claim until it runs on the hardware.

## 3. Reading rules (fixed before the numbers)

**The trap this is shaped around:** `coherent` can silently BE `staged`. If
XLA inserts a wholesale copy to HBM, a rate alone cannot tell the two apart,
and the arm would report a plausible number while having measured the thing
it was supposed to be contrasted against. Three independent witnesses, and
the verdict needs them to agree:

1. **Capacity (primary; binary; needs no bandwidth model).** The ladder
   straddles the cliff at 64/88/104/128 GiB. An arm that COMPLETES a 128 GiB
   working set cannot have held it all in 96 GB of HBM. This is the witness
   that is hardest to fake.
2. **HBM residency.** `peak_bytes_in_use / working_set`. ~1.0 means the set
   was copied wholesale; `chunk/set` (0.031 at 64 GiB, 0.016 at 128 GiB with
   2 GiB chunks) means it streamed. Anything in between is partial buffering
   and must be reported as such, not rounded to a story.
3. **Bandwidth against the E2 ceilings** (HBM 3400, LPDDR 486, C2C r/w
   375/297 GB/s). A rate materially above C2C did not cross C2C per byte; a
   rate at HBM speed means the data was resident and the arm is lying.

**Preconditions on reading anything at all:**
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` must be in force. With a preallocated
  pool the cliff moves to wherever the pool was sized and
  `peak_bytes_in_use` reports the POOL, so all three arms would read
  HBM-resident wherever the data lived. Check the sbatch env echo.
- The `hbm` arm ABOVE the cliff is EXPECTED to OOM. That is a designed output
  bracketing where the cliff sits, not a failed job. An `hbm` arm that
  somehow completes 128 GiB invalidates the whole ladder (it would mean the
  allocator is not doing what we think) and must be chased before anything
  else is read.
- `host_kind_used` must be read off every row before any row is described.

## 4. Pre-registered expectations

Recorded so that a surprise is visible as a surprise. These are expectations,
not gates.

- **P1.** `hbm` completes 64 and 88 GiB, OOMs at 104 and 128. If the cliff
  lands somewhere else, the effective HBM budget is not 96 GB and every
  capacity number in the config table needs revisiting.
- **P2.** `staged` completes all four rungs with residency ~`chunk/set`, at
  roughly the C2C rate already measured (364-367 GB/s), flat in working-set
  size. Flatness is the real content: it is what makes >96 GB usable at all.
- **P3.** `coherent` completes all four rungs. Its rate is the open question.
  Three outcomes and what each means:
  - **at or near `staged`'s rate, streaming residency** -> XLA is doing the
    same transfer we hand-rolled. The arm is redundant but the path is
    confirmed and simpler to write.
  - **materially FASTER than `staged`** -> XLA is overlapping transfer with
    compute in a way the explicit loop does not. Worth pursuing.
  - **residency ~1.0 and HBM-class rate at the rungs that fit, OOM at 104+**
    -> it is a wholesale copy wearing the coherent label. The arm dies and
    `staged` is the only route.
- **P4.** The C-gh paint point: `hbm` OOMs (103 GB of f32 positions vs 96 GB
  HBM, by construction), `staged` and `coherent` complete. The paint rate
  will be well below the ladder's reduce rate because paint is not
  bandwidth-bound -- **do not compare those two numbers**; the ladder
  measures the memory path, the paint point measures whether C-gh runs.

**What this job cannot settle, stated in advance:**
- ATS / pageable-LPDDR access (not expressible; see sec. 1).
- A T9-specific throughput. The probe streams f32 positions as the state
  stand-in; T9 is 9 B/p with a different access pattern, so this gives the
  memory PATH and its bandwidth, not a T9 rate. If V4's node-ladder call
  wants that, it is a second measurement.
- Anything about multi-GPU or the C-hero node ladder.

## 5. Results

*(empty -- job 894010 in flight)*

## 6. Verdict

*(empty -- V3 exit is JC's call, on the record in sec. 5)*
