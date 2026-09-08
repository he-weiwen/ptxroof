// Fixture wrapper (PLAN.md, fixture policy): includes the kernel-ladder
// kernel unchanged. hgemm_coalesced is a plain __global__ function, so
// the include alone emits its PTX body.
#include "2_coalesced.cuh"
