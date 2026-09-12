"""The device execution lane: the host is a byte store, the GPUs do the step.

SEPARATE FROM `executor` ON PURPOSE. `executor.TilePool` refuses a non-CPU
parent backend and spawns its workers with `JAX_PLATFORMS=cpu`; its own comment
names a device lane as a different executor, and this is it. The CPU lane stays
a supported backend and its suite stays green -- nothing here changes it.

WHAT EVERY PHASE BUILT HERE INHERITS (record `runs/v2/device_design_record.md`
secs. 7-11):

1. **Do NOT pin per crossing.** Pinned memory is 6.8x faster on the bus only
   for a buffer pinned once and reused. JAX arrays are immutable, so staging
   each crossing through `pinned_host` is 1.2-1.4x SLOWER than plain numpy, and
   `cudaHostRegister` on numpy memory is ignored by XLA (sec. 11). For this
   engine's numpy-resident state the lever is moving LESS host traffic.
2. **Move in the largest unit the algorithm allows.** 44.4 GB/s at a 16.8 MB
   plane against 201 GB/s at a 2 GiB chunk (pinned), and every device launch
   has a fixed cost regardless of size.
3. **Name the node on any timing.** 1.28-1.36x of node-to-node spread was
   measured on identical work.
"""
