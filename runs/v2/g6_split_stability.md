# G6 -- is the split's low-k error a correctable transfer, and does the correction transport?

Running record for the low-k architectural floor that D-v2-10 deferred. Companion
to `g5b_abs_transfer.md` (C-dev's own discretization error), `g5_kernel_findings.md`
(the kernel study) and `cost_of_memory.md` (the V1-V3 capacity instrument).

Runs: deneb job 188 (G6, cdev, 16 seeds), deneb job 192 (G6b, cdev8, 16 seeds
plus the 8x transport), Vista job 866415 (G6c, cgh64, 6 seeds plus both transport
legs). Scripts `v2_g6_split_stability.py` and `v2_g6b_calib_transport.py`.
Aggregates `g6_split_stability_{cdev8,cdev,cgh64}.json`, transport cards
`g6b_transport_{cdev8_cdev,cdev_cgh64,cdev8_cgh64}.{json,png}`.

**The verdict is JC's.** Nothing here sets a threshold; D-v2-9's bar (absolute
|dP/P| <= 3.0e-02 over k <= k_gate = 2.51) is the only number with standing.

## The question

D-v2-10 recorded that a measured-transfer correction "cannot currently be
claimed", and left the low-k floor for its own assessment. Two things had to be
true for the correction to be worth anything:

  (a) the split's low-k error is a coherent, removable WINDOW, not decorrelation;
  (b) the transfer T-bar can be calibrated ONCE on a small box and applied to
      production boxes.

(b) is the compute question. The only memory-expensive object in the scheme is
the MONOLITHIC reference, and it is needed only during calibration. If T-bar
converges at a small box, calibration is cheap even for a cosmology grid. If it
drifts with volume, calibration inherits the production box's cost and the whole
saving is notional.

## Degenerate limits (read before any number below)

Every config's identity arms passed, per k bin, against a tolerance set from its
own uncorrected band median:

    config   tile_period (full-period shift vs unshifted twin)   mono_shift    tol
    cdev8    4.30e-13                                            5.33e-15      2.35e-04
    cdev     1.63e-12                                            1.71e-14      1.97e-04
    cgh64    4.99e-12                                            6.31e-14      1.91e-04

A full-period tile-origin shift must reproduce its unshifted twin exactly, and a
translated monolithic run must reproduce the untranslated one. Both are ~8 orders
under tolerance everywhere, so the ratio machinery is not manufacturing the
signal it is being used to measure.

## (a) The error is a near-pure window, and it is removable

Measured at C-dev (deneb 188, 16 seeds), leave-one-out: T-bar is built from the
other 15 seeds and applied to the held-out one.

  - **Removable.** k <= 0.5 median 8.92e-03 -> 9.90e-04, a factor 9.0. Over the
    gate band 1.97e-02 -> 4.25e-03, a factor 4.6.
  - **A window, not decorrelation.** max(1 - r) at k <= 0.5 is 3e-05, which is
    phase-perfect. Over the full gate band it is 3.4e-03, and that residual lives
    at the band edge where the high-k ringing sits, not at low k.
  - **Not tile-lattice-locked.** Randomizing the tile origin over 16 pairs costs
    rms 2.6e-05 in band against a seed-to-seed scatter of 2.4e-03, two orders
    smaller. Origin randomization buys essentially nothing; the error lives in the
    structure, consistent with the uniform seam profile.
  - **P-independent at low k.** T128 (P=192) and T256 (P=384) are identical below
    the band edge at cdev, and at cgh64 the evolved dP/P is 1.896e-02 for
    T128 = T256 = T512 alike (job 861849). Bigger tiles do NOT shrink the low-k
    floor. The 1/P law D-v2-10 measured is the band-EDGE plateau, a different
    feature. Only a transfer touches the low-k end.
  - **The scatter is physical, not sampling noise.** sigma/|mean| at k <= 0.5 is
    0.20 / 0.18 / 0.16 at cdev8 / cdev / cgh64, and the measured scatter sits
    100-500x BELOW the sqrt(2/N_modes) cosmic-variance floor, because the ratio is
    taken between two runs sharing initial conditions and cosmic variance cancels
    (r ~ 1). The mean is therefore cheap to pin: at cdev the median sigma/sqrt(N)
    over k <= 0.5 is 1.2e-04 against a median |mean| of 3.9e-03, so the ensemble
    mean is determined to ~3%. The per-realization spread is what is irreducible
    by more calibration seeds, and it is the floor on any single-box correction.
    (Note the spread also falls with VOLUME, not just with N: cgh64's median sd
    over k <= 0.5 is 2.1e-04 against cdev's 4.9e-04, which is why its
    leave-one-out residual is absolutely smaller despite running fewer seeds.)

The low-k transfer is a clean power law, R ~ k^n, which is what licenses
extrapolating T-bar below a small box's fundamental.

## (b) The correction transports across volume

The config table is built for this: cdev8 / cdev / cgh64 hold the fine cell
(0.25), the coarse mesh and the gate band FIXED and vary only VOLUME, 1x / 8x /
64x. Comparing their ensemble-mean transfers is therefore pure box transport at
fixed resolution.

Each box's own leave-one-out card, for reference:

    config   N    arm            P     k<=0.5 uncorr -> LOO     gate band uncorr -> LOO
    cdev8    16   T64/b16        96    1.000e-02 -> 1.693e-03   2.357e-02 -> 8.634e-03
    cdev     16   T128/b32       192   8.923e-03 -> 9.900e-04   1.965e-02 -> 4.249e-03
    cgh64     6   T512/b32       576   8.312e-03 -> 4.468e-04   1.913e-02 -> 2.231e-03

The transport legs, in the ABSOLUTE units D-v2-9's bar is written in. "gate band"
is k <= 2.51; the bar is 3.0e-02.

    leg                       gate band                k <= 0.5                 sigma_calib
                              uncorr -> transported    uncorr -> transported    at k<=0.5
    cdev8 -> cdev   (1x->8x)  1.602e-02 -> 1.889e-03   4.450e-03 -> 1.017e-03   3.5
    cdev  -> cgh64  (8x->64x) 1.554e-02 -> 4.208e-04   3.743e-03 -> 1.641e-04   1.4
    cdev8 -> cgh64  (1x->64x) 1.560e-02 -> 2.227e-03   4.173e-03 -> 1.294e-03   4.7

Every leg lands far under the bar on the gate band: 1/16, 1/71 and 1/13 of
3.0e-02 respectively. The worst case, calibrating on a 1x box and applying at
64x, still leaves 2.23e-03.

Low-k power-law indices, fit over the COMMON window of each pair (see the trap
below):

    leg               big vs small          delta
    cdev8 -> cdev     1.528 vs 1.435        0.093
    cdev  -> cgh64    1.623 vs 1.542        0.081
    cdev8 -> cgh64    1.552 vs 1.435        0.117

All three are inside the 0.15 the script uses to call a shape MATCH.

## Three statistics, three different questions

They disagree in tone and it is worth being explicit about why, because reading
any one of them alone gives the wrong answer.

  1. **Absolute residual against the 3.0e-02 bar.** Does the transported error
     matter for the gate? No, anywhere: the worst leg is 1/13 of the bar.
  2. **sigma_calib, the residual in units of the combined error on the two
     ensemble means.** Is a transport bias RESOLVED at all? At 8x, 1.4 sigma,
     which is consistent with no detectable bias. At 64x from a 1x box, 4.7
     sigma, which is a real, measured drift. So the transfer is not perfectly
     box-invariant; the drift is simply far too small to matter here.
  3. **The ratio to the big box's own leave-one-out residual.** This is the one
     that misleads. The leave-one-out residual is a PER-REALIZATION prediction
     error, of order sigma, while the transport residual is a MEAN-to-MEAN bias,
     of order sigma/sqrt(N). They are different estimands and the ratio between
     them is not a "cost". Read it only as a comparison of the transport bias
     against the irreducible single-box floor: at 8x the bias is 0.37 of that
     floor, so it adds nothing to what a local correction achieves anyway; at 64x
     from a 1x box it is 2.9x the floor, so it becomes the dominant error term.
     Still 1/13 of the bar either way.

## Two traps this card fell into, both now fixed in the script

Recorded because both inverted a verdict, and both survived a first read.

  - **The transport script's "in-band" was the whole k OVERLAP**, out to the big
    box's Nyquist (12.6 for cgh64), while D-v2-9's bar is written on k <= 2.51.
    Over the full overlap the 1x -> 64x correction reads as a REGRESSION,
    4.97e-03 transported against 4.32e-03 uncorrected. On the gate band the same
    leg is a factor 7.0 improvement. The excess is entirely high-k ringing that a
    low-k T-bar has no business correcting and was never asked to. `gate_band` is
    now reported alongside, and is what to gate on. (Note the per-config
    AGGREGATE cards were always gated correctly at k <= k_gate; only the transport
    script was not.)
  - **The low-k index was fit from each box's OWN fundamental**, so a bigger box
    was fit over a wider window and any departure from a pure power law showed up
    as an index difference with no box dependence behind it. Own-window the three
    read 1.435 / 1.542 / 1.634, an apparent monotone drift with volume; on a
    common window they are 1.435 / 1.528 / 1.552, and the 1x -> 64x delta falls
    from 0.198 to 0.117, crossing from DIFFER to MATCH. Both fits are now
    reported, and the MATCH/DIFFER call is made on the common window.

Both changes are additive. `band` and the own-window index keys keep their
original names and values, so cards recorded before 2026-07-27 stay comparable;
verified against the as-run cards for all three legs (preserved keys reproduce to
<= 3.5e-15, which is `polyfit` platform noise between Vista aarch64 and macOS
arm64; the cdev8 -> cdev leg, originally computed on the laptop, reproduces
exactly).

## What this does and does not license

Licensed by measurement:

  - T-bar is calibrated ONCE on a small ensemble, stored, and applied to every
    single-box production mock. No ensemble and no monolithic reference per mock.
    The memory-expensive object need not live at the production box.
  - The correction removes ~89% of the low-k suppression within a config, and
    transporting it across 64x of volume leaves the residual at ~1/13 of the bar
    or better.
  - Tile size and tile origin are measured stable, so neither is a
    re-calibration trigger.

NOT licensed, and deliberately so:

  - **The RESOLUTION axis is untested.** Everything here holds the fine cell at
    0.25. Whether changing the production cell forces re-calibration is a
    different question and was not asked.
  - **Cosmology and redshift are unmeasured** as re-calibration triggers. Redshift
    is bounded (calibrate at the output z); cosmology is argued smooth, not shown.
  - **One config family.** Correctability was measured within a single kernel and
    split configuration (gauss + TSC + matching).
  - **A ~12 to 20% per-realization scatter is irreducible.** Even a perfect T-bar
    leaves ~1e-03 at k <= 0.5 on a single box, and no number of calibration seeds
    removes it.
  - **cgh64 ran 6 seeds against 16 at the other two rungs**, so its own
    leave-one-out is the least well determined number on this page.
  - **D-v2-10's "cannot currently claim a transfer correction" clause is answered
    but not amended.** That is an ADR edit and JC's call.
