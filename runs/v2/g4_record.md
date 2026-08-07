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
2. **HBM residency.** ~~`peak_bytes_in_use / working_set`. ~1.0 means the set
   was copied wholesale; `chunk/set` (0.031 at 64 GiB, 0.016 at 128 GiB with
   2 GiB chunks) means it streamed.~~ Anything in between is partial buffering
   and must be reported as such, not rounded to a story.

   **CORRECTED 2026-08-06, after leg 1 of 894010 and BEFORE any ladder rung
   existed. The struck-through rule is wrong and would have misread a healthy
   arm.** The ratio's DENOMINATOR moves with the rung, so a perfectly
   streaming arm at a small working set still reads ~1.0: leg 1's `staged`
   arm held a constant 4.00 GiB (two 2 GiB chunks, double-buffered) at both
   the 4 and 8 GiB rungs, giving pk/set = 1.0000 and 0.5000 for identical
   behaviour. Same defect class as G3's withdrawn `rho`, which divided by the
   cross transfer and so measured 1/r^2 -- a statistic whose denominator
   carries the thing being tested cannot see the thing being tested.

   **The statistic is ABSOLUTE peak, equivalently `peak_bytes_in_use / chunk`
   (recorded as `hbm_peak_over_chunk`).** Read it as:
   - **constant in working-set size** -> streaming. The count is how deep the
     buffering is (leg 1: 2.0 chunks for both `staged` and `coherent`).
   - **growing with working-set size** -> wholesale copy. This is what the
     pre-fix `staged` arm did: 4.0 GiB at a 4 GiB set, 8.0 GiB at an 8 GiB
     set.
   The ratio is still reported, but it is descriptive, not diagnostic.
   NB this correction makes the witness MORE discriminating, and it is forced
   by arithmetic visible without any verdict-bearing rung -- the ladder above
   the cliff had not run when it was written.
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

### 5.1 Leg 1 (gate, below the cliff) -- 894010, `runs/v2/g4_gh_memory_smoke.json`

GATE PASS: the `coherent` arm compiled and ran at both rungs, so the
expensive legs were allowed to proceed. `host_kind_used = pinned_host` on
both host arms, as sec. 1 requires be stated.

| arm | set GiB | GB/s | peak GiB | peak/chunk | peak/set |
|---|---|---|---|---|---|
| hbm | 4.0 | 3004.5 | 6.00 | 3.0 | 1.5000 |
| staged | 4.0 | 352.8 | 4.00 | 2.0 | 1.0000 |
| coherent | 4.0 | 407.5 | 4.00 | 2.0 | 1.0000 |
| hbm | 8.0 | 3359.4 | 10.00 | 5.0 | 1.2500 |
| staged | 8.0 | 358.7 | 4.00 | 2.0 | 0.5000 |
| coherent | 8.0 | 413.3 | 4.00 | 2.0 | 0.5000 |

Three readings, all provisional -- **every rung here is BELOW the cliff, so
none of them bears on the capacity witness**, which is the primary one:

1. **Both fixes took.** `staged` and `coherent` hold a constant 4.00 GiB
   (2.0 chunks) independent of working-set size = streaming, double-buffered.
   The pre-fix arm scaled its peak with the set. This is what licenses
   expecting the ladder to survive above the cliff at all.
2. **The instrument still reproduces the HBM ceiling** (3004-3359 GB/s vs
   E2's 3400), so nothing about the rewrite disturbed the control.
3. **`coherent` is ~15% FASTER than `staged`** (407.5 vs 352.8; 413.3 vs
   358.7), and sits ABOVE E2's 375 GB/s C2C-read ceiling while `staged` sits
   just below it. That is pre-registered outcome P3-b (XLA overlapping
   transfer with compute in a way the explicit loop does not). **Not to be
   quoted as a result yet**: two rungs, both below the cliff, and a rate
   above a published ceiling is exactly the kind of number that turns out to
   be a harness artifact. The ladder is what tests it.

### 5.2 Legs 2-3 -- **VOID. The ladder measured JAX's defaults, not the GH200.**

Job 894010 ran to completion and every rung above 64 GiB OOM'd. **None of
those OOMs is a statement about the hardware**, and no capacity claim may be
read off this card.

| arm | set GiB | GB/s | peak GiB | status |
|---|---|---|---|---|
| hbm | 64 | 3744.4 | 66.00 | ok |
| staged | 64 | 366.4 | 4.00 | ok |
| coherent | 64 | - | 4.00 | OOM (host, needed 128.5 **KiB** more) |
| hbm | 88 / 104 / 128 | - | 68.00 | OOM (device) |
| staged | 88 / 104 / 128 | - | 4.00 | OOM (host) |
| coherent | 88 / 104 / 128 | - | 4.00 | OOM (host) |
| all | REAL (C-gh) | - | - | OOM |

**Both ceilings are software defaults.** The error text named one of them
outright:

- **Host arms: `XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB`, default 64 GB**, on a node
  with 192 GB free. This fits every observation exactly: `staged` at the
  64 GiB rung just squeezed in, and `coherent` at the SAME rung failed
  needing a further **128.5 KiB** for a compile-time allocation -- it had
  consumed the entire default budget. The 88/104/128 rungs never had a
  chance.
- **Device arm: OOM at a 68.00 GiB peak against 95.6 GiB of HBM**, i.e. ~0.71
  of the card, consistent with a default memory fraction rather than the
  card's capacity. Not independently confirmed on this run, which is itself
  the defect -- `bytes_limit` was never recorded.

**What this cost, and the lesson.** The primary witness -- capacity -- was
never exercised against the hardware at all; the ladder answered a question
about environment-variable defaults. Sec. 3's preconditions guarded the wrong
direction: they said an `hbm` arm that COMPLETED 128 GiB would invalidate the
ladder, but said nothing about one that FAILS EARLY, which voids it just as
completely. A benchmark that OOMs is not self-evidently measuring memory
(umbrella reference-benchmark-measures-the-harness,
reference-knob-must-prove-it-applied).

**Fixed, in the code rather than in a checklist:** every worker now records
`device_bytes_limit` read back off the device plus the three env knobs, and
the summary prints a preconditions block that declares the capacity witness
**VOID** when the host cap is below the ladder top or the device cap is below
0.9 of physical HBM. The two caps are deliberately checked against different
references, because the `hbm` arm is *designed* to die above the cliff and
comparing its cap to the ladder top would void every possible run. The job
now exports `XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=160` and
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` -- but setting them is a claim, and the
readback is the evidence.

**What survives from 894010, and it is not nothing:**
- The instrument's ceiling parity holds at the rungs that ran (hbm 3744 GB/s
  at 64 GiB; staged 366 GB/s, flat against leg 1's 353-359).
- **Streaming is confirmed at scale for `staged`**: peak pinned at 4.00 GiB
  (2.0 chunks) at a 64 GiB working set, i.e. 16x the set with no growth in
  residency. That is the behaviour the whole architecture needs, and it is
  the one real result of the run.
- The `coherent` arm's failure mode is now understood and is not a property
  of the arm.

**Not yet measured, and still the whole point:** whether ANY arm holds a
working set larger than HBM. Every rung that would have tested it was capped.

### 5.3 Re-run -- job 894036, `main @2aa1152`. THE LADDER IS READABLE.

Provenance: Vista `gh`, COMPLETED exit 0:0, elapsed **34m05s** against an 8 h
request (generous by design; TACC bills actual use). Cards
`runs/v2/g4_gh_memory.json` + `_smoke.json`, both committed.

**Cost of G4, all three jobs:** 894005 (gh-dev) 0:00:28 + 894010 (gh) 0:11:20
+ 894036 (gh) 0:34:05 = **45m53s = 0.765 node-hours**, hence **~0.8 SU**, or
~1.1 SU if each job takes the ~15-minute minimum charge. The node-hours are
measured; the SU figure is derived from an assumed 1 SU/node-hour rate.

**Correction:** this line first read "7 SUs", taken from the JPL-SPHEREx
balance delta across the session. That is a PROJECT-WIDE balance shared by
multiple users, so it attributes nothing to these jobs -- during the same
window two `xphot-inject-corpus` jobs ran 1:50 each, and other people draw on
the same allocation. `sacct` without `-a` shows only one's own jobs, which
makes the contamination invisible if you look there to reconcile. Cost is
attributed from the JOBS' OWN elapsed node-time, never from a shared balance.

**Preconditions passed, by readback and not by assertion:** `device
bytes_limit` **90.2 GiB** (0.95 of the card, from
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`), host limit **149.0 GiB** (from
`XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=160`), ladder top 128 GiB. Summary printed
`caps clear the ladder top: capacity witness is readable`. This also confirms
894010's diagnosis after the fact: the default fraction is 0.75, i.e. 71.7
GiB, against the 68 GiB peak that run actually died at.

| arm | set GiB | GB/s | peak GiB | peak/chunk | > HBM | status |
|---|---|---|---|---|---|---|
| hbm | 64 | 3742.8 | 66.00 | 33.0 | no | ok |
| staged | 64 | 367.0 | 4.00 | 2.0 | no | ok |
| coherent | 64 | 418.0 | 4.00 | 2.0 | no | ok |
| hbm | 88 | - | 88.00 | - | no | OOM |
| staged | 88 | 360.0 | 4.00 | 2.0 | no | ok |
| coherent | 88 | 418.5 | 4.00 | 2.0 | no | ok |
| hbm | 104 | - | 88.00 | - | no | OOM |
| staged | 104 | 359.1 | 4.00 | 2.0 | **yes** | ok |
| coherent | 104 | 429.2 | 4.00 | 2.0 | **yes** | ok |
| hbm | 128 | - | 88.00 | - | no | OOM |
| staged | 128 | **2.4** | 4.00 | 2.0 | **yes** | ok |
| coherent | 128 | **1.6** | 4.00 | 2.0 | **yes** | ok |
| hbm | REAL C-gh | - | 90.00 | - | - | OOM |
| staged | REAL C-gh | 7.8 | 31.33 | - | - | ok |
| coherent | REAL C-gh | 2.8 | 16.00 | - | - | ok |

**1. The capacity witness FIRES.** `staged` and `coherent` both completed the
104 and 128 GiB rungs -- working sets larger than the 95.6 GiB of HBM -- with
device peak pinned at **4.00 GiB (2.0 chunks) at every rung including 128
GiB**. Constant peak across a 2x span of working set is the streaming
signature under the corrected witness (sec. 3), and a set that exceeds HBM
while only 4 GiB is resident cannot have been copied wholesale. Both host
arms genuinely stream state larger than the card.

**2. The rate is flat to 104 GiB and then COLLAPSES at 128.** staged
367.0 / 360.0 / 359.1 GB/s at 64/88/104, then **2.4**; coherent 418.0 / 418.5
/ 429.2, then **1.6**. This is a ~150-270x fall, and it is not a blip: it
shows in all 5 reps, with wall times of 42-100 s against 0.26-0.31 s one rung
below, and with large scatter (staged 128: 99.9/42.3/81.8/57.3/51.5 s) where
every other rung is stable to the third digit. **Mechanism NOT established.**
The leading candidate is host memory pressure -- 128 GiB is 137.4 GB of
page-locked memory on a node with ~192 GB available, and pinned pages cannot
be reclaimed -- and the scatter is consistent with contention rather than a
clean bandwidth ceiling. It is deliberately not asserted here; it needs its
own measurement.

**3. C-gh's actual state sits in the flat regime, with margin.** T9 at
2048^3 is 77.3 GB = **72.0 GiB**, comfortably below the 104 GiB rung that ran
at full rate and well clear of the 128 GiB collapse. So the collapse, whatever
it is, does not sit between C-gh and the hardware.

**4. The C-gh paint point RAN.** 2048^3 particles painted from host-resident
positions on one GH200: `staged` 7.8 GB/s, `coherent` 2.8 GB/s, while `hbm`
OOM'd as designed (96.0 GiB of f32 positions against a 90.2 GiB cap). Per P4
these rates must NOT be compared with the ladder's -- paint is not
bandwidth-bound, and the ladder measures the memory path while this point
measures whether C-gh runs at all. It does.

**5. `coherent` beats `staged` by 14-20% at every rung that ran cleanly**
(418.0/418.5/429.2 vs 367.0/360.0/359.1) -- pre-registered outcome P3-b, XLA
overlapping transfer with compute in a way the explicit loop does not.
**One thing unresolved and flagged rather than banked:** 429.2 GB/s is ABOVE
the E2 card's 375 GB/s C2C-read figure. Either that figure is not the binding
limit for this access pattern, or the arm is not doing what the label says.
The reduction touches each byte once, so cache cannot explain it. Until that
is closed, the 14-20% is a solid RELATIVE result and the absolute number
should not be quoted against the E2 ceilings.

**Against the pre-registration (sec. 4):**
- **P1 partially WRONG.** Predicted `hbm` completes 64 and 88 and dies at
  104/128; measured, it dies at **88** (peak 88.00 GiB, i.e. all chunks
  allocated, then the reduction's temporaries hit the 90.2 GiB cap). The
  usable device cliff is between 64 and 88 GiB, tighter than predicted.
- **P2 HELD to 104 GiB, FAILED at 128** (item 2). The flatness that makes
  >96 GB usable is real over 64-104 GiB.
- **P3 -> outcome (b)**, with the caveat in item 5.
- **P4 HELD** exactly.

## 5.4 Stampede3 h100 -- the C-hero fabric, MATCHED (job 3380722)

Same script, same rungs, same 2 GiB chunk, same caps, one GPU pinned. The
two cards are within 2% on HBM (S3 93.6 GiB detected vs GH200 95.6 GiB), so
the cliff sits in the same place on both and any ladder difference is the
memory FABRIC, not capacity. Node: 4x H100 (95830 MiB each), 96 cores,
**1006.9 GiB host** -- confirming the config table's C-hero row from the
machine rather than from the table. Memory kinds `['device', 'pinned_host']`,
identical to the GH200; `unpinned_host` exists on neither.

| arm | GiB | Vista GH200 | S3 H100 | ratio |
|---|---|---|---|---|
| hbm | 64 | 3742.8 | 2287.7 | 1.64x |
| hbm | 88/104/128 | OOM | OOM | -- |
| staged | 64 | 367.0 | 53.4 | **6.87x** |
| staged | 88 | 360.0 | 53.4 | **6.74x** |
| staged | 104 | 359.1 | 53.5 | **6.71x** |
| staged | 128 | 2.4 | 53.4 | 0.04x |
| coherent | 64 | 418.0 | 51.2 | 8.16x |
| coherent | 88 | 418.5 | 51.2 | 8.17x |
| coherent | 104 | 429.2 | 51.2 | 8.38x |
| coherent | 128 | 1.6 | 51.2 | 0.03x |

**1. The PCIe prediction is CONFIRMED and quantified: 6.71-6.87x** on the
staged arm, against a predicted "~6-7x". S3 sits at 51-53 GB/s, squarely in
the PCIe 5 x16 range; the GH200's 359-429 GB/s is C2C. Device peak is 4.00
GiB on both machines at every rung, so both stream identically -- the
mechanism transfers, only the rate does not.

**2. The 128 GiB collapse is GH200-SPECIFIC, and this is independent
evidence for the LPDDR-capacity hypothesis.** S3 runs 128 GiB at 53.4 GB/s,
perfectly flat, no collapse at all. Same rung, same code, same chunk -- on a
node with 1007 GiB of host memory, where 128 GiB is 13% of host rather than
118% of the GH200's 116 GB LPDDR. So the collapse is not structural to the
ladder or to that working-set size; it is that machine running out of the
memory the set has to live in. Job 894118's bracket still locates the knee,
but the mechanism now has a cross-machine control it did not have before.

**3. The coherent advantage does NOT transfer.** On the GH200 `coherent`
beat `staged` by 14-20%; on S3 it is 4% SLOWER (51.2 vs 53.4) at every rung.
XLA overlapping transfer with compute buys something on a C2C link with
headroom and nothing on a saturated PCIe one. Anything V4 concludes about
`coherent` is a Grace-Hopper statement, not a general one.

### fp64 on H100 (leg 1) -- the baseline the gb run will be read against

| op | f32 | f64 | statistic |
|---|---|---|---|
| gemm n=4096 | 315.7 TFLOP/s | 60.6 TFLOP/s | ratio **0.192** |
| gemm n=8192 | 407.4 TFLOP/s | 63.7 TFLOP/s | ratio **0.156** |
| fft3d n=256 | 0.0005 s | 0.0009 s | wall **1.72x** |
| fft3d n=512 | 0.0034 s | 0.0065 s | wall **1.92x** |

H100 f64 GEMM at 60.6-63.7 TFLOP/s is ~90-95% of the part's fp64 peak, so
fp64 here is genuinely full-rate and is the right control.

**METRIC CORRECTION, and it must be applied to the gb card too.** The
script's printed `f64:f32` ratio for `fft3d` reads 1.04-1.16, which invites
reading f64 as FASTER. It is not: `complex128` moves exactly 2.00x the bytes
of `complex64`, so a GB/s ratio carries the dtype in its numerator and cannot
be an fp64-health statistic. The clean read is WALL: 2.00x = pure bandwidth
scaling with no fp64 penalty, >2.00x = a real penalty. H100 lands at
1.72-1.92x, i.e. slightly better than bandwidth scaling and no penalty. The
GEMM ratio needs no correction (same FLOP count both dtypes).
**The script was deliberately NOT changed**: job 894167 is queued against the
committed tree and mutating it under a pending job would mean the gb run
executes something other than what its control ran. Both walls are in the
JSON, so the correction is applied at readout instead.

**What this does NOT say.** The 6.9x is a BANDWIDTH-BOUND figure. The one
real-workload point we have -- the C-gh paint on the GH200 -- ran at 7.8 GB/s
staged, 46x BELOW that machine's own ladder rate, because paint is not
bandwidth-bound. So the fabric gap is an upper bound on the end-to-end
penalty for real work, and possibly a very loose one. Sizing C-hero from
6.9x would be reading a bandwidth ratio as a wall ratio. The measurement that
would settle it is the paint point on S3, which this job did not run.

## 5.5 The knee is LPDDR capacity -- CLOSED (job 894118)

| rung | set | staged GB/s | coherent GB/s | vs 116 GB LPDDR |
|---|---|---|---|---|
| 104 GiB | 111.7 GB | 363.9 | 434.1 | fits |
| **108 GiB** | **116.0 GB** | **358.9** | **443.9** | fits, AT the boundary |
| **112 GiB** | **120.3 GB** | **2.9** | **7.3** | exceeds by 4.3 GB |
| 116 GiB | 124.6 GB | 3.4 | 3.4 | exceeds |
| 120 GiB | 128.8 GB | 2.6 | 3.7 | exceeds |
| 128 GiB | 137.4 GB | 2.6 | 3.6 | exceeds |

**The mechanism is established.** Full rate at 116.0 GB, a ~100x collapse at
120.3 GB: the last rung that fits physical LPDDR runs at full speed and the
first rung that exceeds it falls off a cliff. The config table's 116 GB sits
inside a 4.3 GB bracket. Sec. 5.3 recorded this as "mechanism NOT
established"; it is now, and by two independent routes -- this bracket, plus
S3 h100 running the identical 128 GiB rung at full rate on a 1007 GiB host
(sec. 5.4), which rules out anything structural to the code or the rung.

**The usable number for V4: a Vista GH200 streams a host working set up to
~116 GB (108 GiB), and falls off a cliff immediately past it.** This is a
hard capacity edge, not a soft degradation -- there is no graceful region to
operate in, so a production config must sit clear of it rather than near it.
C-gh's T9 state is 72.0 GiB = 77.3 GB, which leaves ~1.5x headroom.

Do not read the collapsed rungs' relative values (staged 2.9 vs coherent 7.3
at 112 GiB): scatter in that regime is large and the ordering is not stable.
The only content there is "collapsed".

## 5.6 The Hopper fp64 baseline, reproduced on two parts (job 894166 + 3380722 leg 1)

| | GH200 (894166) | S3 H100 (3380722) |
|---|---|---|
| gemm n=4096 f64 | 59.8 TFLOP/s | 60.6 TFLOP/s |
| gemm n=8192 f64 | 64.6 TFLOP/s | 63.7 TFLOP/s |
| **gemm f64:f32** | **0.193 / 0.157** | **0.192 / 0.156** |
| fft3d f64 | 481.2 / 562.4 GB/s | 307.7 / 332.7 GB/s |
| fft3d WALL f64/f32 | 1.61 / 1.79 | 1.72 / 1.92 |

Same Hopper die in two packagings, on two machines, with two independent pixi
installs of the same jax 0.10.2. Absolute f64 GEMM agrees to ~1.5% and the
f64:f32 ratio to 0.001. **This is the control the gb card gets read against,
and it is a reproduced baseline rather than a single point** -- which matters,
because a one-machine baseline could not distinguish "GB200 fp64 is throttled"
from "our harness measures fp64 badly".

FFT absolute rates differ (HBM3e vs HBM3) while the wall ratios stay under
2.00 on both, i.e. neither part penalises fp64 beyond its byte count. Read
the gb card the same way: GEMM ratio directly, FFT by WALL (sec. 5.4's metric
correction), never the printed fft GB/s ratio.

GH200 node from the dump: 1 device, 72 cores, 212.7 GiB host -- consistent
with the LPDDR knee at 116 GB found in sec. 5.5.

## 6. Verdict

*(empty -- V3 exit is JC's call, on the record in sec. 5)*
