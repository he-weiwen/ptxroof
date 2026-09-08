// Fixture wrapper (PLAN.md, fixture policy): the ladder kernel is a
// C++ template; one configuration is instantiated explicitly here.
// Same tile as k11 (BM=BN=128, BK=16, 64x64 warp tiles, 128 threads)
// with the tile loads issued as cp.async into a double buffer.
#include "12_double_buffer.cuh"

template __global__ void hgemm_double_buffer<128, 128, 16, 64, 64, 128>(
    int M, int N, int K, float alpha, const half *A, const half *B,
    float beta, half *C);
