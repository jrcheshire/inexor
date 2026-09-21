"""A heartbeat for the long opaque loops.

The card and the export are each ONE call that runs for hours at hero scale and
prints nothing until it returns. gb 1010938 spent 3 h 23 min inside
`pk_summary_card` against a 56-minute estimate and was cancelled with no way to
say how far through it was, or even which of its three stages it was in -- the
overrun was only visible once the wall was in sight. The same shape as the
`PhaseTracer` card that never printed: the work was measured, and nothing came
out.

`Heartbeat` is what the library's loops report to. Throttled by WALL TIME rather
than iteration count, so the line rate does not depend on how big the loop is;
flushed on every line, because a SIGKILL strands a buffered one; and the rate it
quotes is the CURRENT stage's, never a running average over stages that count
different things.
"""

import sys
import time


def _hms(s):
    s = int(max(0.0, s))
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


class Heartbeat:
    """`hb(stage, done, total)` -- a progress line at most every `every` seconds.

    Pass one as `progress=` to `coarse_delta_streamed`, `forward_from_slabs`,
    `binned_power` or `write_particles`; they call it once per unit of work and
    are unchanged when it is None.

    The first and last call of a stage always print, so a stage that fits inside
    one interval still leaves its cost on the page rather than vanishing. Stages
    are delimited by the name changing: each gets its own clock, and `elapsed`
    on the closing line is that stage's whole wall.
    """

    def __init__(self, every=60.0, out=None, clock=time.monotonic, prefix="  [hb]"):
        self.every = float(every)
        self.out = out if out is not None else sys.stdout
        self.clock = clock
        self.prefix = prefix
        self._stage = None
        self._t0 = 0.0
        self._last = 0.0

    def __call__(self, stage, done, total):
        now = self.clock()
        first = stage != self._stage
        if first:
            self._stage, self._t0, self._last = stage, now, now
        done, total = int(done), int(total)
        last = total > 0 and done >= total
        if not (first or last or now - self._last >= self.every):
            return
        self._last = now
        el = now - self._t0
        pct = f"{100.0 * done / total:5.1f}%" if total > 0 else "  ? %"
        line = f"{self.prefix} {stage}: {done:,}/{total:,} {pct} in {_hms(el)}"
        # A rate needs at least one completed unit and a clock that has moved;
        # printing "inf/s" on the opening line reads as a measurement.
        if done > 0 and el > 0.0:
            rate = done / el
            line += f", {rate:.3g}/s"
            if not last and total > done:
                line += f", ETA {_hms((total - done) / rate)}"
        if last:
            line += " DONE"
        print(line, file=self.out, flush=True)
