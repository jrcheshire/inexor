# Paper outline (working)

**Working title:** Exactly reversible, compressed-state differentiable N-body
simulation
**Venue target:** The Open Journal of Astrophysics (OJAp). Methods paper.
**Author:** James Cheshire (Caltech).

## The three claims

1. **Exact discrete adjoint at O(1) memory in steps.** Integer phase space +
   rounded-increment updates (JANUS pattern) make the PM time evolution
   bit-exactly reversible, so the reverse-mode adjoint replays the *identical*
   trajectory — no float replay drift (pmwd f32: RMSD to 5.2e-2 of field std),
   no f64 tax, no checkpoint hierarchy.
2. **12 -> 6 bytes/particle differentiable state** (CUBE's compression brought
   to autodiff): the 1024^3-particle full adjoint fits a single 80 GB GPU
   (~65-75 GiB peak) where a float-state design does not.
3. **f32-speed gradients with f64-grade replay fidelity** — decisive on
   consumer GPUs (f64 at 1/64 rate), demonstrated on commodity hardware.

Secondary contributions: the w-frame scale ladder (making BullFrog's
contracting affine kick integer-bijective); determinism-by-associativity
integer CIC paint; the observation that fixed-point storage and bit-exact
reversibility are the same discretization decision (CUBE and JANUS literatures
do not cite each other).

## Money plots (produced at M2/M3)

- P1: max differentiable N per 80 GB GPU — inexor vs pmwd vs DISCO-DJ (OOM
  curve). [M2]
- P2: gradient fidelity vs float-replay adjoint at f32; replay drift vs steps
  (bit-exact flatline vs growing float drift). [M2]
- P3: adjoint wall-clock overhead vs naive AD and float replay. [M2]
- P4: quantization error ladder (int8/int16/f32) vs the PM error floor,
  CUBE-style, under few-step BullFrog. [M1/M3]
- P5: consumer-GPU demo table (f32 exact vs f64 float-replay throughput). [M3]
- P6: science demo — field-level f_NL/IC recovery on the inexor forward
  model. [M4]

## Section skeleton

1. Introduction: the memory wall in differentiable simulation; gradient
   memory as the binding constraint (pmwd OOM evidence; BORG/FlowPM
   trade-offs); the CUBE x JANUS gap.
2. Method: fixed-point phase space; the reversible integer BullFrog step and
   the w-frame ladder; deterministic integer paint; the custom_vjp
   exact-replay adjoint and the STE treatment of rounding.
3. Validation: reversibility (exact); forward physics vs mbody/DISCO-DJ;
   gradient gates vs float twin + FD; quantization error vs the PM floor.
4. Performance: memory budget and the money plots.
5. Science demo: field-level f_NL toy.
6. Limitations (honest): PM-only forces; K <~ 12 int16+BullFrog range budget
   (FastPM beyond); schedule feasibility — every |alpha_k| >= 0.05 or the
   ladder refuses: int16+BullFrog needs K >= 3 from a_i = 0.1 (alpha_1 =
   0.018 at K = 2, on the zero-crossing) and lin-0.04 spacing dies near
   K = 11, a structural constraint float BullFrog does not have (measured,
   R2; D-012); STE gradient bias at coarse quantization (measured, R4);
   bit-exactness scoped to a process/device/XLA version; single-node (mesh
   memory dominance; two-level mesh as future work).

## Citation seed list (verified primary sources only — session 2026-07-10)

- Yu, Pen & Wang 2018, ApJS 237:24 (CUBE) — arXiv:1712.06121
- CUBE2 — arXiv:2512.12629
- Cheng et al. 2020 (Cosmo-pi) — arXiv:2003.03931
- Rein & Tamayo 2017, MNRAS 473:3351 (JANUS) — arXiv:1704.07715
- Li et al., ApJS (pmwd adjoint) — arXiv:2211.09815
- DISCO-DJ II — arXiv:2510.05206
- Rampf, List & Hahn 2024 (BullFrog) — arXiv:2409.19049
- Potter, Stadel & Teyssier 2017 (PKDGRAV3) — arXiv:1609.08621
- Garrison et al. 2021 (Abacus), MNRAS 508:575 — arXiv:2110.11392
- Maksimova et al. 2021 (AbacusSummit) — arXiv:2110.11398
- Habib et al. (HACC), CACM 2017
- Springel et al. 2021 (GADGET-4) — arXiv:2010.03567
- Springel 2005 (Gadget-2, the 80 B/p baseline) — astro-ph/0505010
- Tassev, Zaldarriaga & Eisenstein 2013 (COLA) — arXiv:1301.0322
- Leclercq et al. 2020 (sCOLA tiling) — arXiv:2003.04925
- Greener, PNAS 122(22) 2025 (reversible MD) — arXiv:2412.04374
- Matsubara et al. 2021 (symplectic adjoint) — arXiv:2102.09750
- Zhang & Constantinescu (CAMS/Revolve) — arXiv:2106.13879
- Angulo & Hahn 2022 (Living Reviews) — arXiv:2112.05165

**Citation guard** (search-layer errors caught by verification — never cite):
"HACC 36 B/particle" (that figure is PKDGRAV3's 2LPT IC footnote);
"CUBE2 20 B/p vs Gadget-2 80 B/p" (conflation; not in CUBE2).
