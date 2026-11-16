// Verify mublasGemmStridedBatchedEx semantics against per-expert mublasGemmEx.
// Tiny sizes; uses all-ones so results must be exactly K.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <musa_runtime.h>
#include <mublas.h>
#include <musa_fp16.h>

#define E 5
#define R 7
#define BLK 8
#define K 64
#define N 32

int main() {
  __half *dW, *dA;
  float *dCs, *dCr;
  musaMalloc((void**)&dW, (size_t)E * N * K * 2);
  musaMalloc((void**)&dA, (size_t)E * BLK * K * 2);
  musaMalloc((void**)&dCs, (size_t)E * BLK * N * 4);
  musaMalloc((void**)&dCr, (size_t)E * BLK * N * 4);
  // all ones on host
  __half* hW = (__half*)malloc((size_t)E * N * K * 2);
  __half* hA = (__half*)malloc((size_t)E * BLK * K * 2);
  for (size_t i = 0; i < (size_t)E * N * K; i++) hW[i] = __float2half_rn(1.0f);
  for (size_t i = 0; i < (size_t)E * BLK * K; i++) hA[i] = __float2half_rn(1.0f);
  musaMemcpy(dW, hW, (size_t)E * N * K * 2, musaMemcpyHostToDevice);
  musaMemcpy(dA, hA, (size_t)E * BLK * K * 2, musaMemcpyHostToDevice);
  musaMemset(dCs, 0, (size_t)E * BLK * N * 4);
  musaMemset(dCr, 0, (size_t)E * BLK * N * 4);

  mublasHandle_t h;
  mublasCreate(&h);
  mublasSetStream(h, 0);
  float alpha = 1.0f, beta = 0.0f;

  // reference: one mublasGemmEx per expert
  for (int e = 0; e < E; e++) {
    mublasStatus_t st = mublasGemmEx(h, MUBLAS_OP_T, MUBLAS_OP_N, N, R, K, &alpha,
                                     dW + (size_t)e * N * K, MUSA_R_16F, K,
                                     dA + (size_t)e * BLK * K, MUSA_R_16F, K, &beta,
                                     dCr + (size_t)e * BLK * N, MUSA_R_32F, N,
                                     MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
    if (st) printf("single e=%d status=%d\n", e, (int)st);
  }
  musaDeviceSynchronize();

  mublasStatus_t st = mublasGemmStridedBatchedEx(
      h, MUBLAS_OP_T, MUBLAS_OP_N, N, R, K, &alpha, dW, MUSA_R_16F, K, (long long)N * K, dA,
      MUSA_R_16F, K, (long long)BLK * K, &beta, dCs, MUSA_R_32F, N, (long long)BLK * N, E,
      MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
  musaDeviceSynchronize();
  printf("strided status=%d\n", (int)st);

  float* cs = (float*)malloc((size_t)E * BLK * N * 4);
  float* cr = (float*)malloc((size_t)E * BLK * N * 4);
  musaMemcpy(cs, dCs, (size_t)E * BLK * N * 4, musaMemcpyDeviceToHost);
  musaMemcpy(cr, dCr, (size_t)E * BLK * N * 4, musaMemcpyDeviceToHost);
  double md = 0;
  int bad = 0;
  for (size_t i = 0; i < (size_t)E * R * N; i++) {
    // only compare rows < R of each block (BLK>R padding rows are never written)
    size_t e = i / ((size_t)R * N), r = (i / N) % R, c = i % N;
    size_t idx = (e * BLK + r) * N + c;
    double d = fabs(cs[idx] - cr[idx]);
    if (d > 1e-3) { if (bad < 5) printf("  mismatch e=%zu r=%zu c=%zu strided=%f ref=%f\n", e, r, c, cs[idx], cr[idx]); bad++; }
    if (d > md) md = d;
  }
  printf("strided vs single: max_abs_err=%.6g  bad=%d  C0=%.1f (expect %d)  %s\n", md, bad, cs[0], K,
         bad == 0 ? "PASS" : "FAIL");
  return bad != 0;
}
