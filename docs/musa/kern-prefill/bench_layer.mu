// Layer-level benchmark: memory-vs-compute bound, real per-layer time, fp16 TC peak, stream bandwidth.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <musa_runtime.h>
#include <mublas.h>
#include <musa_fp16.h>

#define E  288
#define N1 2048
#define K1 4096
#define N2 4096
#define K2 2048
#define BLKMAX 512

static musaEvent_t ev0, ev1;
static mublasHandle_t H;

template <class F>
static double bench(const char* name, int warm, int iters, F fn, double flops_per_call, double bytes_per_call) {
  for (int i = 0; i < warm; i++) fn();
  musaDeviceSynchronize();
  musaEventRecord(ev0, 0);
  for (int i = 0; i < iters; i++) fn();
  musaEventRecord(ev1, 0);
  musaDeviceSynchronize();
  float ms = 0;
  musaEventElapsedTime(&ms, ev0, ev1);
  double per = ms / iters;
  printf("  %-44s %9.3f ms  %8.2f TFLOP/s  %7.1f GB/s\n", name, per, flops_per_call / (per * 1e9),
         bytes_per_call / (per * 1e6));
  fflush(stdout);
  return per;
}

__global__ void cpyk(const __half* in, __half* out, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (; i < n; i += stride) out[i] = in[i];
}

int main(int argc, char** argv) {
  int dev = argc > 1 ? atoi(argv[1]) : 0;
  musaSetDevice(dev);
  musaEventCreate(&ev0);
  musaEventCreate(&ev1);
  mublasCreate(&H);
  mublasSetStream(H, 0);

  size_t szWgu = (size_t)2 * E * N1 * K1 * 2;
  size_t szWd  = (size_t)E * N2 * K2 * 2;
  size_t szAg  = (size_t)E * BLKMAX * K1 * 2;
  size_t szAd  = (size_t)E * BLKMAX * K2 * 2;
  size_t szCgu = (size_t)2 * E * BLKMAX * N1 * 4;
  size_t szCd  = (size_t)E * BLKMAX * N2 * 4;
  printf("alloc %.2f GB\n", (szWgu + szWd + szAg + szAd + szCgu + szCd) / 1e9);
  fflush(stdout);
  __half *dWgu, *dWd, *dAg, *dAd;
  float *dCgu, *dCd;
  musaMalloc((void**)&dWgu, szWgu);
  musaMalloc((void**)&dWd, szWd);
  musaMalloc((void**)&dAg, szAg);
  musaMalloc((void**)&dAd, szAd);
  musaMalloc((void**)&dCgu, szCgu);
  musaMalloc((void**)&dCd, szCd);
  musaMemset(dWgu, 0, szWgu);
  musaMemset(dWd, 0, szWd);
  musaMemset(dAg, 0, szAg);
  musaMemset(dAd, 0, szAd);
  musaMemset(dCgu, 0, szCgu);
  musaMemset(dCd, 0, szCd);
  printf("alloc+memset done\n");
  fflush(stdout);
  float alpha = 1.0f, beta = 0.0f;

  // ---- fp16 tensor-core peak: 4096^3 ----
  printf("\n=== fp16 TC peak: 4096^3 GemmEx ===\n");
  fflush(stdout);
  {
    const int P = 4096;
    __half *pA, *pB;
    float* pC;
    musaMalloc((void**)&pA, (size_t)P * P * 2);
    musaMalloc((void**)&pB, (size_t)P * P * 2);
    musaMalloc((void**)&pC, (size_t)P * P * 4);
    musaMemset(pA, 0, (size_t)P * P * 2);
    musaMemset(pB, 0, (size_t)P * P * 2);
    double fl = 2.0 * P * P * P;
    bench("fp16 4096^3 NN", 2, 5, [&]() {
      mublasGemmEx(H, MUBLAS_OP_N, MUBLAS_OP_N, P, P, P, &alpha, pA, MUSA_R_16F, P, pB, MUSA_R_16F,
                   P, &beta, pC, MUSA_R_32F, P, MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
    }, fl, 6.0 * P * P * 2 + (double)P * P * 4);
    musaFree(pA); musaFree(pB); musaFree(pC);
  }

  // ---- stream bandwidth (read+write copy kernel, full grid) ----
  printf("\n=== streaming bandwidth (copy kernel) ===\n");
  fflush(stdout);
  {
    long n = 256L * 1024 * 1024 / 2;  // 256 MB of fp16
    __half *s, *d;
    musaMalloc((void**)&s, n * 2);
    musaMalloc((void**)&d, n * 2);
    musaMemset(s, 0, n * 2);
    bench("256MB fp16 copy kernel (grid=56*8)", 2, 5, [&]() { cpyk<<<56 * 8, 256>>>(s, d, n); },
          0.0, 2.0 * n * 2);
    musaFree(s); musaFree(d);
  }

  // ---- M scaling: is strided batched compute or memory bound? ----
  printf("\n=== M scaling, gate StridedBatchedEx batch=288 [N=2048 K=4096] ===\n");
  fflush(stdout);
  for (int M : {128, 227, 256, 384, 512}) {
    int blk = 256;
    if (M > 256) blk = 512;
    char nm[128];
    double fl = 2.0 * E * M * (double)N1 * K1;
    double bytes = (double)E * N1 * K1 * 2 + (double)E * M * K1 * 2 + (double)E * M * N1 * 4;
    snprintf(nm, sizeof nm, "M=%d (blk=%d)", M, blk);
    bench(nm, 2, 5, [&]() {
      mublasGemmStridedBatchedEx(H, MUBLAS_OP_T, MUBLAS_OP_N, N1, M, K1, &alpha, dWgu, MUSA_R_16F,
                                 K1, (long long)N1 * K1, dAg, MUSA_R_16F, K1, (long long)blk * K1,
                                 &beta, dCgu, MUSA_R_32F, N1, (long long)blk * N1, E,
                                 MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
    }, fl, bytes);
  }

  // ---- real layer timing ----
  printf("\n=== full MoE layer: gate + up + down (3 strided-batched calls) ===\n");
  fflush(stdout);
  for (int pad : {0, 1}) {
    int M = pad ? 256 : 227;
    int blk = 256;
    double fgu = 4.0 * E * M * (double)N1 * K1;
    double fd  = 2.0 * E * M * (double)N2 * K2;
    double bgu = 2.0 * ((double)E * N1 * K1 * 2) + 2 * (double)E * M * K1 * 2 + 2 * (double)E * M * N1 * 4;
    double bd  = (double)E * N2 * K2 * 2 + (double)E * M * K2 * 2 + (double)E * M * N2 * 4;
    char nm[160];
    double per = bench(pad ? "layer pad M=256 (gate+up+down)" : "layer M=227 (gate+up+down)", 2, 5, [&]() {
      // gate (first half of dWgu)
      mublasGemmStridedBatchedEx(H, MUBLAS_OP_T, MUBLAS_OP_N, N1, M, K1, &alpha, dWgu, MUSA_R_16F,
                                 K1, (long long)N1 * K1, dAg, MUSA_R_16F, K1, (long long)blk * K1,
                                 &beta, dCgu, MUSA_R_32F, N1, (long long)blk * N1, E,
                                 MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
      // up (second half)
      mublasGemmStridedBatchedEx(H, MUBLAS_OP_T, MUBLAS_OP_N, N1, M, K1, &alpha,
                                 dWgu + (size_t)E * N1 * K1, MUSA_R_16F, K1, (long long)N1 * K1, dAg,
                                 MUSA_R_16F, K1, (long long)blk * K1, &beta, dCgu + (size_t)E * blk * N1,
                                 MUSA_R_32F, N1, (long long)blk * N1, E, MUBLAS_COMPUTE_32F,
                                 MUBLAS_GEMM_DEFAULT);
      // down
      mublasGemmStridedBatchedEx(H, MUBLAS_OP_T, MUBLAS_OP_N, N2, M, K2, &alpha, dWd, MUSA_R_16F,
                                 K2, (long long)N2 * K2, dAd, MUSA_R_16F, K2, (long long)blk * K2,
                                 &beta, dCd, MUSA_R_32F, N2, (long long)blk * N2, E,
                                 MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
    }, fgu + fd, bgu + bd);
    snprintf(nm, sizeof nm, "  -> 45 layers = %.3f s ; 8192 tok / that = %.0f tok/s (1 GPU, MoE GEMM only)",
             per * 45 / 1e3, 8192.0 / (per * 45 / 1e3));
    printf("%s\n", nm);
    fflush(stdout);
  }
  printf("\ndone\n");
  return 0;
}
