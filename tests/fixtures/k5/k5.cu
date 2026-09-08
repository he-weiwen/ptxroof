// Fixture wrapper (PLAN.md, fixture policy): the ladder kernel is a
// C++ template; one configuration is instantiated explicitly here
// (including the bare header alone yields PTX with no kernel body at
// all — verified). BM=64 BN=64 BK=8 TM=8 TN=8 is the S1 design point:
// per outer-loop iteration each thread does 1024 flops against 32
// global bytes, AI(global) = 32 flop/B.
#include "5_2d_blocktiling.cuh"

template __global__ void hgemm_2d_blocktiling<64, 64, 8, 8, 8>(
    int M, int N, int K, float alpha, const half *A, const half *B,
    float beta, half *C);
