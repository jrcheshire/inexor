"""The device execution lane: the host is a byte store, the GPUs do the step.

Separate from `executor`, whose `TilePool` refuses a non-CPU parent backend and runs its workers
with `JAX_PLATFORMS=cpu`; the CPU lane remains a supported backend. Rules for code here:

1. Do not pin memory per crossing: JAX arrays are immutable, so staging each crossing through
   `pinned_host` is slower than plain numpy, and `cudaHostRegister` on numpy memory is ignored
   by XLA. The lever is moving less host traffic.
2. Move in the largest unit the algorithm allows: bandwidth grows strongly with transfer size
   and every device launch has a fixed cost.
3. Report the node with any timing: identical work varies by >1.2x across nodes.
"""
