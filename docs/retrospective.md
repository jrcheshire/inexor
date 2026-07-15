# Retrospective — what M2 measured, and what a fresh attempt should know

Written 2026-07-14, at the point where M2 S4's measurements undercut the design's
motivating premise. This is the distillation for a **future attempt**, not a
session log; the blow-by-blow is in `m1-results.md` / `m2-results.md` and the
commit history. Read this BEFORE re-reading `architecture.md`, because parts of
that document's motivation are now known to be wrong and are flagged below.

Nothing here is a bug report. The code does what it was designed to do. The
finding is that what it was designed to do is worth less than the design assumed,
and that this was knowable earlier and more cheaply than we found out.

---

## 1. The motivating premise was never verified against the comparison target

This is the big one, and it is a **process** failure rather than a technical one.

`architecture.md` Sec. 1 motivates the whole design:

> the standard escape — adjoint with reverse-time replay — replays in floating
> point, **drifting by up to ~5e-2 of the field std in single precision**
> (pmwd Table 2 [PMWD])

That number is wrong for the purpose it is used for, in three separate ways, and
each is enough on its own:

1. **Different code.** It is pmwd's. The live comparison target is DISCO-DJ, whose
   replay is a different implementation. Nothing entitles us to assume it
   inherits pmwd's error.
2. **Different quantity.** 5e-2 is a *field* RMSD — the drift of the reconstructed
   *state*. The claim the paper needs is about *gradient* error. These are not the
   same number and need not even be the same order.
3. **Never measured.** It sat in the prior-art table from the founding research
   session (2026-07-10) to M2 S4 (2026-07-14) without anyone computing the
   equivalent quantity for the code we actually benchmark against.

**Measured instead** (deneb, `scripts/m2_p2_disco_drift.py` + `m2_grad_gate.py
--k-sweep`; each code against ITS OWN f64 gradient, which is the comparison that
does not fold in convention differences):

| | gradient error vs own f64 |
|---|---|
| DISCO-DJ f32 float replay | see `runs/m2/p2_disco_drift.json` (CPU smoke at n=16: **~1e-5**, growing ~1.13x from K=3 to 6) |
| inexor int16 adjoint (D-015) | **2e-4 .. 3.3e-3** (quantization + STE) |

If that ordering holds at scale, the design's position is: **inexor's own
quantization error is larger than the float-replay error it exists to eliminate.**
The thing being fixed was not broken at the level the design assumed, and the fix
costs accuracy of its own.

Note the subtlety that makes this easy to get wrong, and that we got wrong:
"bit-exact" describes the **replay being reproducible**, NOT the gradient being
accurate. inexor's gradient is the exact transpose of the *quantized* map it ran;
against the smooth reference it carries the quantization error above. Claiming
"better-than-f64 fidelity" (as was briefly claimed in this session) conflates the
two and is false.

**Lesson.** Before designing against a competitor's weakness, measure that
weakness yourself, in the exact quantity your claim is about, on the exact code
you will be compared to. A number from a different code's paper is a hypothesis.

---

## 2. The memory thesis is dead for a structural reason, not a fixable one

Peak device memory of the full adjoint (deneb RTX 3050; scale-invariant per
particle across 64^3..512^3, so these extrapolate):

| | B/particle |
|---|---|
| inexor (int16) | 452.6 (64^3) / 488.2 (128^3) |
| **DISCO-DJ f32** | **447.3 / 436.6** |
| DISCO-DJ f64 | 817.4 / 815.2 |

Read those numbers carefully, because they are the end of claim 2:

- **The transient is inherent.** Two independent codes, ~440 B/particle. That is
  what a CIC+FFT force VJP costs in JAX. It is not our sloppiness; there is no
  optimization pass that reclaims it.
- **The carry is ~8% of peak.** `architecture.md` Sec. 9 budgets 36 B/particle of
  carry + ~32 of transients = ~68 total. The carry number is right. **The
  transient estimate is wrong by ~14x**, and it is the term that dominates.
- **So the entire state-width thesis fights for ~3%.** int state saves 12
  B/particle against a 466 B/particle peak. Even a perfect implementation wins a
  few percent, not a category.
- **inexor is currently slightly WORSE than DISCO-DJ f32** (488 vs 437 at 128^3):
  our 12 B/particle carry advantage is more than cancelled by ~50 B/particle of
  extra transient.
- **1024^3 on 80 GB needs <= 74 B/particle.** Nobody is within 6x. Max N per 80 GB
  card is ~512^3 for both codes. (And "80 GB" = 74.5 GiB — Sec. 9's own stated
  upper end of 75 GiB does not fit; the budget never had the slack its prose
  claimed.)

**Lesson.** Measure the *framework's* adjoint transient floor before designing
around state width. In JAX PM adjoints the force VJP's scratch outweighs the
phase-space state by ~10x, which caps what any state-compression idea can win.
A one-day measurement of DISCO-DJ's peak at M0 would have priced the entire
thesis before a line of `codec.py` was written.

---

## 3. What actually works, and is worth keeping

None of this is invalidated. If someone revisits the mechanism, start here.

- **Exact reversibility at scale.** 512^3, K=10, `n_diff=0`, 6.4 s on one GH200
  (Vista job 831303). 134M particles, bit-exact round trip. The CUBE x JANUS x
  autodiff intersection is real and it runs.
- **O(1)-in-steps, measured.** Peak flat across K=5..40 (slope 0.0 B/particle,
  both 64^3 and 128^3). Reversibility-as-checkpointing delivers exactly what
  Sec. 8 claims. **But this is table stakes**: pmwd and DISCO-DJ are also O(1) in
  steps (DISCO-DJ's live adjoint is a hand-rolled `custom_vjp` backsolve keeping
  only the final state — its Diffrax path is dead code). Passing this wins
  nothing against them; failing it would merely have been fatal.
- **Deterministic integer paint on CUDA** where the f32 path is not (R3, and
  re-measured in M2 S4 layer by layer).
- **The custom_vjp + STE twin composes** with outer `jax.grad`, including nested —
  the upgrade over mbody's eager-only adjoints. D-015 gate passed.

---

## 4. The GPU nondeterminism finding (durable, non-obvious, transferable)

Measured bottom-up (`scripts/m2_nondet_source.py`, deneb job 15), because the
first, well-motivated hypothesis was wrong:

- The **f32 CIC scatter-add is nondeterministic on CUDA** — atomic accumulation
  order. paint_f32 differs in 156/4096 elements across 8 identical calls; the full
  bwd in 18384/24576.
- **The primal int path is bit-stable** (0/12288), so reversibility is untouched.
  D-006 requires the STE twin to paint in f32, so the gradient inherits the
  nondeterminism *by design*.
- Magnitude is f32 roundoff (median-rel ~1e-7, corr 1.000000000): **no ratified
  gate is threatened.** What is dead is the assumption that a GPU gradient is
  bit-reproducible.
- `XLA_FLAGS=--xla_gpu_deterministic_ops=true` removes it at every layer, but
  costs **3.2-3.8x on the bwd** and **more memory** (OOM'd 256^3 where the default
  arm completed). Adopted for the bit-equality TESTS only (`detflag` marker +
  `test-det` task); production stays on the fast path.

---

## 5. Measurement methodology that worked (reuse verbatim)

- **One config per fresh subprocess.** XLA's `peak_bytes_in_use` is monotonic
  within a process with no reset API; sequential in-process configs silently
  report the running maximum.
- **Sweep the axis that discriminates.** R6's n-sweep produced a beautifully
  consistent 13x that said *nothing* about mechanism, because carry and transient
  both scale as n_part. Only the K-sweep could test O(1)-in-steps — and the
  profiler originally fixed K=10, i.e. it swept the one axis that could not
  answer its own question.
- **Compare each code against its own f64.** Cross-code gradient comparison folds
  in LPT/kernel/unit conventions; "how far is your cheap gradient from your own
  exact-arithmetic gradient" is the drift question and is directly comparable.
- **Guard against no-op adjoints** (`grad_norm`, `adjoint_ran`). A silently zero
  gradient produces a small, flattering peak that looks like a *great* result and
  nothing else in a memory harness notices.
- **Smoke foreign-env arms locally on CPU first.** CPU reports no `memory_stats`
  so it measures nothing, but it exercises the whole API path for free. This
  caught two invented DISCO-DJ API calls in seconds that had already cost a GPU
  job.
- **Global metrics, not per-component.** S3 established the per-particle gradient
  is noise-dominated where the reference is ~0.

---

## 6. Traps that cost real time (do not re-learn these)

- **CPU XLA dumps are not evidence about GPU peaks.** Fusion differs materially
  (meshes 21 B/particle on CPU vs 73 on GPU for the same computation).
- **XLA allocations are reused slots.** Ranking `value:` lines double-counts.
  Worse, grouping an arena's values **by offset also overcounts** — XLA assigns
  overlapping address ranges to values whose live ranges are disjoint in time, so
  distinct offsets do NOT imply coexistence. Only summing *allocations* is sound,
  and a `preallocated-temp` arena (~81% of the scan module's peak) is **not
  decomposable** from the dump: this XLA build emits no peak-live set.
- **`jax.checkpoint` does not reduce peak when the recompute IS the peak.**
  Measured -2.5% to -7.2%, against Sec. 9's inherited "~18% saving" (from mbody,
  never tested here).
- **`pytest -n auto` on a single-GPU node is destructive.** `auto` = CPU count;
  under `--exclusive` that was 64 xdist workers each preallocating
  `XLA_PYTHON_CLIENT_MEM_FRACTION` of the same 6 GB card. The resulting failures
  look exactly like genuine test failures. Every GPU sbatch carries
  `PYTEST_XDIST_AUTO_NUM_WORKERS=1`.
- **pixi `solve-group` binds only environments that DECLARE it.** Putting it on one
  env leaves that env in a group of one, free to drift.
- **`deno_task_shell` (pixi tasks) has no `if/then/fi` and no `{ ...; }`** — both
  are parse errors. And `test ... || (echo ERR; exit 1)` prints the error and
  **runs the body anyway**. Only `test ... && <body>` aborts.
- **pixi lock versions are PLATFORM-DEPENDENT.** Reading "the" numpy version off
  one platform and reporting it as the whole picture is wrong (it bit this
  session twice).
- **`dynamic = ["version"]` blocks foreign-platform solves** — pixi must execute
  the build backend to read metadata and cannot for a platform it can't run.
  Static `[project] version` is why xcat's aarch64 lock always worked.
- **"80 GB" is 74.5 GiB** (80e9 B). Compare in bytes.

---

## 7. If you take a fresh crack at this

In rough order of what would have saved the most time:

1. **Price the thesis before building it.** One measurement — the comparison
   target's adjoint peak, and its gradient error vs its own f64 — would have
   priced both claims at M0. Both are ~1 day on a free GPU. Both were deferred to
   M2 S5, by which point the design was built.
2. **Do not inherit numbers across codes or across quantities.** Every published
   figure in the prior-art table is a hypothesis about *that* code, in *that*
   quantity, at *that* config.
3. **The transient sets the budget.** If your idea saves state width, compute
   `state_saved / measured_peak` first. Below ~10%, you need a different axis: the
   transient itself, or a framework whose adjoint scratch is small, or a regime
   (very large K? very many rollouts?) where step-independence stops being table
   stakes.
4. **The reversibility mechanism is sound and reusable.** int16 fixed-point phase
   space, the w-frame ladder, the deterministic int paint, the STE twin adjoint —
   all measured working at 512^3 on CUDA. If a use case appears where *exactness*
   or *reproducibility* is the product (rather than memory), the machinery exists
   and is validated.
5. **The honest surviving pitch, if any**: not memory, and not accuracy-vs-f32
   (we are worse). It would have to be something exactness buys that neither f32
   nor f64 replay gives — e.g. bitwise-reproducible gradients across runs/machines
   for a workflow that needs it. That is a much narrower claim than the outline's,
   and nobody has yet shown a use case that demands it.
