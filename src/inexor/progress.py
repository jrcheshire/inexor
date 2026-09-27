"""Wall-time-throttled progress lines for long single-call loops (card, export).

Lines are throttled by wall time, not iteration count, flushed on every line so a SIGKILL
does not strand a buffered one, and the quoted rate is the current stage's only.
"""

import sys
import time


def _hms(s):
    s = int(max(0.0, s))
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


class Heartbeat:
    """`hb(stage, done, total)` -- a progress line at most every `every` seconds.

    Passed as `progress=` to `coarse_delta_streamed`, `forward_from_slabs`, `binned_power`
    and `write_particles`, which call it once per unit of work. A change of `stage` name
    starts a new clock; the first and last call of each stage always print.
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
        # No rate until a unit is done and the clock has moved (never "inf/s").
        if done > 0 and el > 0.0:
            rate = done / el
            line += f", {rate:.3g}/s"
            if not last and total > done:
                line += f", ETA {_hms((total - done) / rate)}"
        if last:
            line += " DONE"
        print(line, file=self.out, flush=True)
