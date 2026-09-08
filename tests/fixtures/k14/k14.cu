// Fixture wrapper (PLAN.md, fixture policy): the ladder kernel is a
// C++ template; one configuration is instantiated explicitly here.
// Same tile as k12 (BM=BN=128, BK=16, 64x64 warp tiles, 128 threads,
// cp.async double buffer) with the fragments loaded by ldmatrix and
// the products issued as inline mma.sync.m16n8k16.
#include "14_ldmatrix_mma.cuh"

template __global__ void hgemm_ldmatrix_mma<128, 128, 16, 64, 64, 128>(
    int M, int N, int K, float alpha, const half *A, const half *B,
    float beta, half *C);
