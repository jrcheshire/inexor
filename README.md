# inexor

Exactly reversible, compressed-state differentiable N-body simulation in JAX.

A particle-mesh code whose phase space lives on a fixed-point integer lattice
(int16 default = 12 bytes/particle; int8 opt-in = 6 B/p, following CUBE), which
makes time evolution bit-exactly reversible (following JANUS) — so reverse-mode
gradients come from an **exact replay** adjoint with memory independent of the
number of time steps, at f32 compute speed. Existing differentiable PM codes
(pmwd, DISCO-DJ) replay in floating point, which drifts in single precision;
inexor's replay returns the *same bits*.

One design decision (integer phase space) buys three properties:
compression, exact reversibility, and periodic boundaries by integer wrap.

**Status: pre-M0.** Design documents only; no implementation yet.
See `docs/architecture.md` (design), `docs/roadmap.md` (milestones M0–M4;
M0 is a hard go/no-go gate), `docs/decisions.md` (ADR log), and
`paper/outline.md`.

## Layout (planned)

- `src/inexor/` — the package; novel core = `codec.py` (fixed-point state) and
  `adjoint.py` (custom_vjp exact-replay adjoint); the rest ports mbody's PM
  physics with dtype discipline.
- `tests/` — pytest; `pixi run test` / `test-fast` / `lint` / `format`.

## Environments

Pixi: `default` (CPU JAX, macOS-arm64 dev) and `gpu` (linux-64 CUDA 12).
