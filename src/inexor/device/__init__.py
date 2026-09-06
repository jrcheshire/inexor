"""The device execution lane: the host is a byte store, the GPUs do the step.

SEPARATE FROM `executor` ON PURPOSE. `executor.TilePool` refuses a non-CPU
parent backend and spawns its workers with `JAX_PLATFORMS=cpu`; its own comment
names a device lane as a different executor, and this is it. The CPU lane stays
a supported backend and its suite stays green -- nothing here changes it.

WHAT D1 ESTABLISHED, and what every phase built here inherits (record
`runs/v2/device_design_record.md` secs. 7-9):

1. **Pin every host buffer that crosses to the device.** Pageable moved 137.5 GB
   at 6.5 GB/s where pinned moved it at 44.4 -- 6.8x, and 91% of the coarse
   solve's wall was that traffic.
2. **Move in the largest unit the algorithm allows.** 44.4 GB/s at a 16.8 MB
   plane against 201 GB/s at a 2 GiB chunk; the unit is a second constraint and
   it is worth another 4.5x.
3. **Name the node on any timing.** 1.36x of node-to-node spread was measured on
   identical work at 2048^3.
"""
