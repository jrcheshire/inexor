# G5 kernel findings -- the split-family tradeoff (seed V2a)

Running record of the G5 kernel study, 2026-07-15. Committed by exception from
the gitignored `runs/` (`git add -f`), same rule as `cost_of_memory.md`: this is
a RECORD, not run output, and it must survive machines.

**VERDICT (2026-07-16): G5 PASSED — D-v2-10** (gauss + TSC + matching,
2.61e-2 vs the 3e-2 D-v2-9 bar at tile 128 / buf 32). The "inverts M3's
memory story" tension below was resolved by scale framing, not by a new
knob: error ∝ 1/P but the tile working set is an ABSOLUTE ~O(P^3) cost,
box-independent, so at the config-table homes (GH200/H100) big tiles are
cheap and the buffer stays minimal. See D-v2-10 for the operating-point
rule (V4) and the G5c follow-up measurement. The kernel-mechanism findings
below stand unchanged.

All numbers are CPU f64, `scripts/v2_g5_core.py`.

## Configuration of these measurements (READ THIS BEFORE QUOTING ANY NUMBER)

n_fine = 128, n_coarse = 32, L = 128, n_part = 64^3, d_f = 1.0, d_c = 4.0;
particles = a perturbed lattice (sigma = 1.2 cells), NOT an evolved snapshot;
metric = rms|dg| / rms|g_mono| over all particles, unbanded.

C-dev is 4x larger (n_fine 512, n_coarse 128) and the gate band is
k <= 0.2 k_Nyq,fine = 2.51 h/Mpc. **These are FORCE errors, not evolved dP/P**,
and the D-v2-1 bar is stated in dP/P of an evolved field -- so nothing here can
be read against the bar without the evolution arm. The PM mesh floor for scale:
~1.3e-1 dP/P at the gate band (G2c, cost_of_memory.md).

## Floors that pass (nothing below is readable without these)

| floor | check | result |
|---|---|---|
| F0a | `kernel_grids` vs `forces.k_components` (cubic) | **bit-exact** |
| F0b | probe mono vs `make_force_fn(f64, paint="f32")` | 1.15e-7 = the f32 PAINT, attributed |
| F1 | long + short vs mono, gauss / compact / gauss_compact | 3.1e-16 / 3.3e-16 / 3.6e-16 |
| tile identity | one tile = whole box, b=0, P = n_fine | 4.5e-16 |
| DC shift | short force under delta -> delta + c, c = 1 / -1 / 137 | 8.6e-16 / 7.0e-16 / 1.1e-13 |

## The three families

`gauss`: S(k) = exp(-k^2 r_s^2), short := 1 - S. Recombines exactly; short
kernel -> ik/k^2 at high k, so it has NO compact real-space support.

`compact`: window the MONO kernel in real space (quintic smoothstep, support
r_out) -> PMFAST lineage. Stencil built ONCE on the fine grid and embedded per
box (see below).

`gauss_compact`: window the GAUSSIAN SHORT kernel in real space. The hybrid --
intended to keep S(k)'s rolloff AND get compact support.

## Finding 1 -- the Gaussian tiles as 2.0/P, and the buffer is NOT a knob

MEASURED, three independent signatures, all agreeing:

1. **decay law**: rms|e|/rms|g_short| x P = 2.00, 2.01, 2.33, 2.12, 1.95, 1.96
   across T = 16..64, b = 8..24. Flat. The buffer model predicts erfc(beta/2),
   which spans 600x over the same scan -- measured/erfc runs 0.19 -> 1101.
2. **spatial profile**: edge/interior error ratio **1.14**, i.e. UNIFORM across
   the tile core. erfc truncation would be edge-concentrated.
3. **r_s sweep at fixed P**: error flat at 3.3-3.5e-2 while erfc(b/2r_s) spans
   1.5e-8 -> 1.6e-1 (SEVEN orders).

plus the exact limit: P = n_fine -> **7e-15** (the periodization difference
vanishes when the tile IS the box).

Mechanism: the short kernel is sharply truncated at the fine Nyquist, so its
real-space tails ring like ~1/r; a tile's period-P image sum differs from the
global period-L sum by ~1/P. **The only knob is P (tile size), not the buffer**
-- exactly backwards for a design whose premise is that tiles are small. The
`beta = b/r_s` parameterization in the V2a plan measures nothing for this family,
and sCOLA's "buffer >= 25 Mpc/h, r ~ 3.2" does not describe this scheme.

## Finding 2 -- the compact families tile EXACTLY, and cost ~10x in the coarse arm

With buffer >= support: **1.7e-15 .. 5.2e-15**. With buffer < support (b=8 vs
support 12): 9.6e-3. Exactly the right failure mode.

But the coarse arm pays for it (reference = that family's own `long_fine`;
tiling excluded entirely):

| family | param | CIC bare | CIC match | TSC bare | TSC match |
|---|---|---|---|---|---|
| gauss | alpha=1.0 | 9.56e-3 | 6.13e-3 | 1.16e-2 | **4.66e-3** |
| compact | r_out=3 coarse | 7.01e-2 | 2.67e-1 | 6.31e-2 | 9.56e-2 |
| compact | r_out=4 coarse | 5.83e-2 | 2.34e-1 | **4.99e-2** | 8.54e-2 |
| gauss_compact | beta=4 | 5.33e-2 | 2.23e-1 | **4.38e-2** | 8.04e-2 |

## Finding 3 -- THE RINGING IS CONSERVED (the hybrid falsified its own premise)

`gauss_compact` was built to get both: S(k)'s anti-aliasing AND compact support.
It gets exact tiling (2.7e-15) and its coarse arm is **5.3e-2 -- no better than
pure compact**. The hypothesis that S(k)'s rolloff would survive real-space
windowing is FALSE.

The mechanism is sharper than "Gaussian vs compact": **the band-limit ringing
must be carried by one arm or the other, and neither can absorb it.** Any
real-space window moves the ringing tails out of the short kernel and into the
long kernel -- and those tails ARE high-k content, which the coarse mesh aliases.
Keep the ringing in the short arm -> tile at 2.0/P. Move it to the long arm ->
pay ~5e-2 in coarse aliasing. TSC does not rescue it (1.2x where ~10x is needed),
so this is structural, not an artifact of the assignment order.

## Finding 4 -- kernel matching is FAMILY-SPECIFIC (a correction)

The V2a plan asserted matching is mandatory, full stop. It is mandatory for
`gauss` (helps 1.35-1.6x) and **harmful for the windowed families** (hurts 2.8-4x
with CIC; still hurts 1.7x with TSC). The factor W_f^2/W_c^2 grows without bound
toward the coarse Nyquist and is harmless only where the long kernel is ALREADY
small; S(k) guarantees that (~5e-5 at the coarse Nyquist at alpha=1), the
windowed long kernels do not. Clipping at 10 does not rescue it.

Analytic case for the gauss family, verified numerically: the S-weighted
mismatch S*(1 - Wc^2/Wf^2) is 5.4% at k=1 vs a 2.1% PM floor and 0.9% at k=0.25
vs 0.13% -- i.e. unmatched, it misses the D-v2-1 band by 2.6x at k=1 and **7.1x
at k=0.25. It fails WORST AT LOW k**, which is where nobody would look.

## Where this points (NOT a verdict)

`gauss` + TSC + matching on the coarse mesh, with the tile size as the knob:

| | coarse floor | tiling | total at C-dev P=192 | asymptote |
|---|---|---|---|---|
| gauss (TSC+match) | 4.66e-3 | 2.0/P | ~1.1e-2 | **4.7e-3** |
| compact / hybrid | ~4.4e-2 | exact | ~4.4e-2 | ~4.4e-2 |

Gaussian wins ~4x at C-dev-scale P and ~9x asymptotically. But it inverts M3's
memory story -- error falls only with bigger tiles, and big tiles are what M3
exists to avoid. That tension is the real G5 result and it needs the evolution
arm before anyone ratifies anything.

## Traps found by measurement (each cost a round; do not re-learn)

1. **Brick double-count**: `tile_members` walks bricks by modular index, so
   span > nb concatenates the same brick twice and paints its particles twice.
   Gave a 3.29 RELATIVE error. Guarded in `brick_span`.
2. **The last cell layer**: requiring `base < P-1` silently discards the padded
   box's last layer -> 4.6e-1 on the one-tile identity, AND a fake buffer
   plateau that mimicked R7's ringing signature. The padded box IS periodic
   (its rfftn says so); wrap modulo **P**, not modulo n_fine. "Not the global
   modulo" is not "no modulo".
3. **Per-box kernel rebuild**: windowing `irfftn(ik/k^2)` separately on each box
   does NOT tile exactly (5e-3..2e-2) -- irfftn returns the P-periodized kernel,
   which already differs inside the support. Build the stencil ONCE, embed it.
4. **The tile mean**: a tile using its own mean does not leak a DC error, it
   RESCALES the short force by mean_global/mean_tile, error exactly |s-1| (O(1)).
   Deleted by painting counts/mean with no -1; the global mean is a CONFIG
   SCALAR (n_total/n_mesh^3), never a reduction.
5. **`n_dropped == 0` was a wrong contract**: brick-union members outside the
   padded mesh are the intended superset overhang. The real contract is the
   ownership partition (every particle owned by exactly one tile).

## Open

- Everything above is n_fine=128, one clustering level, force error. Needs
  C-dev scale + an evolved snapshot + the evolution arm (dP/P vs the mesh floor).
- The compact coarse arm is not exhausted: window shape (r_in/r_out was fixed at
  0.5-0.6 arbitrarily) and higher-order assignment beyond TSC are unexplored.
  TSC bought only 1.2x, so the prior is that it stays ~10x behind.
- The 2.0/P law's crossover as P -> L is unmapped (the scan jumps 96 -> 128).
