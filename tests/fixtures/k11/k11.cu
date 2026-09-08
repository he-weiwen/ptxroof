// Fixture wrapper (PLAN.md, fixture policy): the ladder kernel is a
// C++ template; one configuration is instantiated explicitly here.
// BM=128 BN=128 BK=16 WM=64 WN=64, 128 threads: each warp owns a 64x64
// tile as 4x4 wmma m16n16k16 fragments, so one BK step is 4 A-fragment
// loads, 16 B-fragment loads and 16 wmma.mma per warp.
#include "11_wmma.cuh"

template __global__ void hgemm_wmma<128, 128, 16, 64, 64, 128>(
    int M, int N, int K, float alpha, const half *A, const half *B,
    float beta, half *C);
