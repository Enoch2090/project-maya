// Quick probe: do fp8 (E4M3) / int8 tensor-core GEMMs work through muBLAS on mp_22, and how fast?
#include <stdio.h>
#include <stdlib.h>
#include <musa_runtime.h>
#include <mublas.h>

#define E 288
#define N1 2048
#define K1 4096
#define R 227
#define BLK 256

static musaEvent_t ev0, ev1;
static mublasHandle_t H;

template <class F>
static double bench(const char* name, int warm, int iters, F fn, double flops) {
  for (int i = 0; i < warm; i++) fn();
  musaDeviceSynchronize();
  musaEventRecord(ev0, 0);
  for (int i = 0; i < iters; i++) fn();
  musaEventRecord(ev1, 0);
  musaDeviceSynchronize();
  float ms = 0;
  musaEventElapsedTime(&ms, ev0, ev1);
  double per = ms / iters;
  printf("  %-40s %9.3f ms  %8.2f TFLOP/s\n", name, per, flops / (per * 1e9));
  fflush(stdout);
  return per;
}

int main() {
  musaEventCreate(&ev0);
  musaEventCreate(&ev1);
  mublasCreate(&H);
  mublasSetStream(H, 0);
  float alpha = 1.0f, beta = 0.0f;
  double fl = 2.0 * E * R * (double)N1 * K1;

  // big single GEMM for peak
  {
    const int P = 4096;
    void *a, *b;
    float* c;
    musaMalloc(&a, (size_t)P * P);
    musaMalloc(&b, (size_t)P * P);
    musaMalloc((void**)&c, (size_t)P * P * 4);
    musaMemset(a, 0, (size_t)P * P);
    musaMemset(b, 0, (size_t)P * P);
    double flp = 2.0 * P * P * P;
    mublasStatus_t st = mublasGemmEx(H, MUBLAS_OP_N, MUBLAS_OP_N, P, P, P, &alpha, a, MUSA_R_8F_E4M3,
                                     P, b, MUSA_R_8F_E4M3, P, &beta, c, MUSA_R_32F, P,
                                     MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
    printf("fp8 e4m3 %d^3: status=%d\n", P, (int)st);
    fflush(stdout);
    if (st == MUBLAS_STATUS_SUCCESS)
      bench("fp8 4096^3", 2, 5, [&]() {
        mublasGemmEx(H, MUBLAS_OP_N, MUBLAS_OP_N, P, P, P, &alpha, a, MUSA_R_8F_E4M3, P, b,
                     MUSA_R_8F_E4M3, P, &beta, c, MUSA_R_32F, P, MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
      }, flp);
    musaFree(a); musaFree(b); musaFree(c);
  }
  // int8 big single GEMM
  {
    const int P = 4096;
    void *a, *b;
    int* c;
    musaMalloc(&a, (size_t)P * P);
    musaMalloc(&b, (size_t)P * P);
    musaMalloc((void**)&c, (size_t)P * P * 4);
    musaMemset(a, 0, (size_t)P * P);
    musaMemset(b, 0, (size_t)P * P);
    double flp = 2.0 * P * P * P;
    int ialpha = 1, ibeta = 0;
    mublasStatus_t st = mublasGemmEx(H, MUBLAS_OP_N, MUBLAS_OP_N, P, P, P, &ialpha, a, MUSA_R_8I, P, b,
                                     MUSA_R_8I, P, &ibeta, c, MUSA_R_32I, P, MUBLAS_COMPUTE_32I,
                                     MUBLAS_GEMM_DEFAULT);
    printf("int8 %d^3: status=%d\n", P, (int)st);
    fflush(stdout);
    if (st == MUBLAS_STATUS_SUCCESS)
      bench("int8 4096^3", 2, 5, [&]() {
        mublasGemmEx(H, MUBLAS_OP_N, MUBLAS_OP_N, P, P, P, &ialpha, a, MUSA_R_8I, P, b, MUSA_R_8I, P,
                     &ibeta, c, MUSA_R_32I, P, MUBLAS_COMPUTE_32I, MUBLAS_GEMM_DEFAULT);
      }, flp);
    musaFree(a); musaFree(b); musaFree(c);
  }
  // fp8 strided batched at MoE shape
  {
    void *W, *A;
    float* C;
    musaMalloc(&W, (size_t)E * N1 * K1);
    musaMalloc(&A, (size_t)E * BLK * K1);
    musaMalloc((void**)&C, (size_t)E * BLK * N1 * 4);
    musaMemset(W, 0, (size_t)E * N1 * K1);
    musaMemset(A, 0, (size_t)E * BLK * K1);
    musaMemset(C, 0, (size_t)E * BLK * N1 * 4);
    mublasStatus_t st = mublasGemmStridedBatchedEx(
        H, MUBLAS_OP_T, MUBLAS_OP_N, N1, R, K1, &alpha, W, MUSA_R_8F_E4M3, K1, (long long)N1 * K1, A,
        MUSA_R_8F_E4M3, K1, (long long)BLK * K1, &beta, C, MUSA_R_32F, N1, (long long)BLK * N1, E,
        MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
    printf("fp8 strided batched gate: status=%d\n", (int)st);
    fflush(stdout);
    if (st == MUBLAS_STATUS_SUCCESS)
      bench("fp8 gate StridedBatchedEx 288x(227,2048,4096)", 2, 5, [&]() {
        mublasGemmStridedBatchedEx(H, MUBLAS_OP_T, MUBLAS_OP_N, N1, R, K1, &alpha, W, MUSA_R_8F_E4M3,
                                   K1, (long long)N1 * K1, A, MUSA_R_8F_E4M3, K1,
                                   (long long)BLK * K1, &beta, C, MUSA_R_32F, N1,
                                   (long long)BLK * N1, E, MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
      }, fl);
    musaFree(W); musaFree(A); musaFree(C);
  }
  printf("done\n");
  return 0;
}
