// Fixture source (PLAN.md, fixture policy): two straight-line
// tensor-core kernels with no loops, so every count in the report is
// a plain constant. One 16x16x16 product each way the ISA offers it:
// the wmma API (load a, load b, mma, store d) and a hand-written
// mma.sync.m16n8k16 with the fragments assembled from plain loads.
#include <cuda_fp16.h>
#include <mma.h>

__global__ void gemm_wmma_16x16x16(const __half *A, const __half *B, float *C) {
  using namespace nvcuda::wmma;

  fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
  fragment<accumulator, 16, 16, 16, float> c_frag;

  fill_fragment(c_frag, 0.0f);
  load_matrix_sync(a_frag, A, 16);
  load_matrix_sync(b_frag, B, 16);
  mma_sync(c_frag, a_frag, b_frag, c_frag);
  store_matrix_sync(C, c_frag, 16, mem_row_major);
}

__global__ void gemm_mma_16x8x16(const __half *A, const __half *B, float *C) {
  unsigned a[4], b[2];
  float c[4] = {0.f, 0.f, 0.f, 0.f};

  int lane = threadIdx.x & 31;
  int group = lane >> 2;
  int tid = lane & 3;
  const __half2 *A2 = reinterpret_cast<const __half2 *>(A);
  a[0] = *reinterpret_cast<const unsigned *>(&A2[(group + 0) * 8 + tid]);
  a[1] = *reinterpret_cast<const unsigned *>(&A2[(group + 8) * 8 + tid]);
  a[2] = *reinterpret_cast<const unsigned *>(&A2[(group + 0) * 8 + tid + 4]);
  a[3] = *reinterpret_cast<const unsigned *>(&A2[(group + 8) * 8 + tid + 4]);
  const __half2 *B2 = reinterpret_cast<const __half2 *>(B);
  b[0] = *reinterpret_cast<const unsigned *>(&B2[group * 8 + tid]);
  b[1] = *reinterpret_cast<const unsigned *>(&B2[group * 8 + tid + 4]);

  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));

  int row = lane >> 2;
  int col = (lane & 3) * 2;
  C[(row + 0) * 16 + col + 0] = c[0];
  C[(row + 0) * 16 + col + 1] = c[1];
  C[(row + 8) * 16 + col + 0] = c[2];
  C[(row + 8) * 16 + col + 1] = c[3];
}
