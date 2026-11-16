// MoE expert GEMM benchmark on MUSA (MTT S4000, mp_22)
// Compares: per-expert mublasGemmEx loop / mublasGemmStridedBatchedEx / mublasGemmGroupedBatchedEx
// Shapes: gate/up [N=2048, K=4096], down [N=4096, K=2048], E=288 experts, R rows/expert.
// fp16 A/B, fp32 C, compute 32F.  Synthetic data, perf-oriented.
//
// build: mcc -x musa --offload-arch=mp_22 -O3 -std=c++17 moe_gemm_bench.mu -o moe_gemm_bench -lmublas
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <musa_runtime.h>
#include <mublas.h>
#include <musa_fp16.h>

#define E       288
#define K1      4096
#define N1      2048
#define K2      2048
#define N2      4096
#define R       227
#define BLK     256          // row-block stride per expert (>= R)
#define TFLOP1  (2.0*E*R*(double)N1*K1)   // one gate matrix, all experts
#define TFLOPGU (2.0*TFLOP1)              // gate + up
#define TFLOPD  (2.0*E*R*(double)N2*K2)   // down
#define TFLAYER (TFLOPGU + TFLOPD)

__global__ void fillh(__half* p, long n, float s) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = __float2half_rn((float)(((i * 37 + (long)s * 13) % 17) - 8) * 0.0031f);
}
__global__ void fillf(float* p, long n, float s) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = (float)(((i * 53 + (long)s * 7) % 19) - 9) * 0.0017f;
}

static musaEvent_t ev0, ev1;
static mublasHandle_t H;

template <class F>
static double bench(const char* name, int warm, int iters, F fn, double flops_per_call) {
  for (int i = 0; i < warm; i++) fn();
  musaDeviceSynchronize();
  musaEventRecord(ev0, 0);
  for (int i = 0; i < iters; i++) fn();
  musaEventRecord(ev1, 0);
  musaDeviceSynchronize();
  float ms = 0;
  musaEventElapsedTime(&ms, ev0, ev1);
  double per = ms / iters;
  printf("  %-46s %9.3f ms  %8.2f TFLOP/s\n", name, per, flops_per_call / (per * 1e9));
  fflush(stdout);
  return per;
}

int main(int argc, char** argv) {
  int dev = argc > 1 ? atoi(argv[1]) : 0;
  musaSetDevice(dev);
  musaDeviceProp prop;
  musaGetDeviceProperties(&prop, dev);
  printf("device %d: %s  MP=%d  warpsz=%d  smem/blk=%zu\n", dev, prop.name, prop.multiProcessorCount,
         prop.warpSize, prop.sharedMemPerBlock);
  printf("shapes: E=%d R=%d  gate/up[N=%d,K=%d] down[N=%d,K=%d]\n", E, R, N1, K1, N2, K2);
  printf("one layer = %.3f TFLOP (gate+up %.3f + down %.3f); x45 = %.1f TFLOP\n",
         TFLAYER / 1e12, TFLOPGU / 1e12, TFLOPD / 1e12, 45 * TFLAYER / 1e12);
  fflush(stdout);

  musaEventCreate(&ev0);
  musaEventCreate(&ev1);
  mublasCreate(&H);
  mublasSetStream(H, 0);
  int ver = 0;
  mublasGetVersion(H, &ver);
  printf("mublas version %d\n", ver);

  // ---------------- allocations ----------------
  __half *dWgu, *dWd, *dWgT, *dAg, *dAd;
  float *dCgu, *dCd;
  size_t szWgu = (size_t)2 * E * N1 * K1 * 2;   // gate + up
  size_t szWd  = (size_t)E * N2 * K2 * 2;       // down
  size_t szWgT = (size_t)E * K1 * N1 * 2;       // gate in [K,N] layout (OP_N variant)
  size_t szAg  = (size_t)E * BLK * K1 * 2;
  size_t szAd  = (size_t)E * BLK * K2 * 2;
  size_t szCgu = (size_t)2 * E * BLK * N1 * 4;
  size_t szCd  = (size_t)E * BLK * N2 * 4;
  printf("alloc: Wgu=%.2f Wd=%.2f WgT=%.2f act/C=%.2f total=%.2f GB\n", szWgu / 1e9, szWd / 1e9,
         szWgT / 1e9, (szAg + szAd + szCgu + szCd) / 1e9,
         (szWgu + szWd + szWgT + szAg + szAd + szCgu + szCd) / 1e9);
  fflush(stdout);
  musaMalloc((void**)&dWgu, szWgu);
  musaMalloc((void**)&dWd, szWd);
  musaMalloc((void**)&dWgT, szWgT);
  musaMalloc((void**)&dAg, szAg);
  musaMalloc((void**)&dAd, szAd);
  musaMalloc((void**)&dCgu, szCgu);
  musaMalloc((void**)&dCd, szCd);
  {
    int th = 256;
    fillh<<<(szWgu / 2 + th - 1) / th, th>>>(dWgu, szWgu / 2, 1.0f);
    fillh<<<(szWd / 2 + th - 1) / th, th>>>(dWd, szWd / 2, 2.0f);
    fillh<<<(szWgT / 2 + th - 1) / th, th>>>(dWgT, szWgT / 2, 7.0f);
    fillh<<<(szAg / 2 + th - 1) / th, th>>>(dAg, szAg / 2, 3.0f);
    fillh<<<(szAd / 2 + th - 1) / th, th>>>(dAd, szAd / 2, 4.0f);
    fillf<<<(szCgu / 4 + th - 1) / th, th>>>(dCgu, szCgu / 4, 5.0f);
    fillf<<<(szCd / 4 + th - 1) / th, th>>>(dCd, szCd / 4, 6.0f);
    musaDeviceSynchronize();
    printf("fill done\n");
    fflush(stdout);
  }

  float alpha = 1.0f, beta = 0.0f;

  // ---------------- phase 1: fp16 tensor-core peak ----------------
  printf("\n=== phase1: fp16 GEMM peak (muBLAS GemmEx, fp16 in / fp32 out) ===\n");
  fflush(stdout);
  {
    const int P = 2048;
    __half *pA, *pB;
    float* pC;
    musaMalloc((void**)&pA, (size_t)P * P * 2);
    musaMalloc((void**)&pB, (size_t)P * P * 2);
    musaMalloc((void**)&pC, (size_t)P * P * 4);
    musaMemset(pA, 0, (size_t)P * P * 2);
    musaMemset(pB, 0, (size_t)P * P * 2);
    double fl = 2.0 * P * P * P;
    bench("fp16 2048^3 NN", 3, 10, [&]() {
      mublasGemmEx(H, MUBLAS_OP_N, MUBLAS_OP_N, P, P, P, &alpha, pA, MUSA_R_16F, P, pB, MUSA_R_16F,
                   P, &beta, pC, MUSA_R_32F, P, MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
    }, fl);
    bench("fp16 2048^3 TN", 3, 10, [&]() {
      mublasGemmEx(H, MUBLAS_OP_T, MUBLAS_OP_N, P, P, P, &alpha, pA, MUSA_R_16F, P, pB, MUSA_R_16F,
                   P, &beta, pC, MUSA_R_32F, P, MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
    }, fl);
    bench("fp16 2048^3 NN compute=16F (C fp16)", 3, 10, [&]() {
      mublasGemmEx(H, MUBLAS_OP_N, MUBLAS_OP_N, P, P, P, &alpha, pA, MUSA_R_16F, P, pB, MUSA_R_16F,
                   P, &beta, pC, MUSA_R_16F, P, MUBLAS_COMPUTE_16F, MUBLAS_GEMM_DEFAULT);
    }, fl);
    musaFree(pA); musaFree(pB); musaFree(pC);
  }

  // ---------------- phase 2: device memory bandwidth ----------------
  printf("\n=== phase2: device memory bandwidth (DtoD memcpy) ===\n");
  fflush(stdout);
  {
    size_t nb = (size_t)512 << 20;
    void *s, *d;
    musaMalloc(&s, nb);
    musaMalloc(&d, nb);
    musaMemset(s, 0, nb);
    musaMemset(d, 0, nb);
    double per = bench("512MB DtoD memcpy", 2, 5, [&]() { musaMemcpy(d, s, nb, musaMemcpyDeviceToDevice); }, 0.0);
    printf("  -> copy bandwidth %.1f GB/s\n", nb / (per * 1e6));
    fflush(stdout);
    musaFree(s);
    musaFree(d);
  }

  // ------------- pointer helpers -------------
  auto wg_ptr = [&](int e) { return (const void*)(dWgu + (size_t)e * N1 * K1); };
  auto wu_ptr = [&](int e) { return (const void*)(dWgu + (size_t)(E + e) * N1 * K1); };
  auto wgt_ptr = [&](int e) { return (const void*)(dWgT + (size_t)e * K1 * N1); };
  auto wd_ptr = [&](int e) { return (const void*)(dWd + (size_t)e * N2 * K2); };
  auto ag_ptr = [&](int e, int blk) { return (const void*)(dAg + (size_t)e * blk * K1); };
  auto ad_ptr = [&](int e, int blk) { return (const void*)(dAd + (size_t)e * blk * K2); };
  auto cg_ptr = [&](int half, int e, int blk) { return (void*)(dCgu + (size_t)(half * E + e) * blk * N1); };
  auto cd_ptr = [&](int e, int blk) { return (void*)(dCd + (size_t)e * blk * N2); };

  // ---------------- phase 3: per-expert mublasGemmEx loop ----------------
  printf("\n=== phase3: 288 x mublasGemmEx per layer (launch overhead) ===\n");
  fflush(stdout);
  for (int M : {227, 256}) {
    char nm[128];
    snprintf(nm, sizeof nm, "gate single GemmEx M=%d", M);
    double one = bench(nm, 10, 30, [&]() {
      mublasGemmEx(H, MUBLAS_OP_T, MUBLAS_OP_N, N1, M, K1, &alpha, wg_ptr(0), MUSA_R_16F, K1,
                   ag_ptr(0, BLK), MUSA_R_16F, K1, &beta, cg_ptr(0, 0, BLK), MUSA_R_32F, N1,
                   MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
    }, 2.0 * M * N1 * K1);
    snprintf(nm, sizeof nm, "gate 288 GemmEx loop M=%d", M);
    double tot = bench(nm, 2, 3, [&]() {
      for (int e = 0; e < E; e++)
        mublasGemmEx(H, MUBLAS_OP_T, MUBLAS_OP_N, N1, M, K1, &alpha, wg_ptr(e), MUSA_R_16F, K1,
                     ag_ptr(e, BLK), MUSA_R_16F, K1, &beta, cg_ptr(0, e, BLK), MUSA_R_32F, N1,
                     MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
    }, TFLOP1);
    printf("     per-call extra vs single = %.1f us ; useful(TFLOP over 227 rows) = %.2f TFLOP/s\n",
           (tot - one) * 1000.0, TFLOP1 / (tot * 1e9));
    fflush(stdout);
  }

  // ---------------- phase 4: strided batched ----------------
  printf("\n=== phase4: mublasGemmStridedBatchedEx (uniform M) ===\n");
  fflush(stdout);
  for (int M : {227, 256}) {
    char nm[128];
    mublasStatus_t st = MUBLAS_STATUS_SUCCESS;
    snprintf(nm, sizeof nm, "gate StridedBatchedEx batch=288 M=%d", M);
    double per = bench(nm, 2, 5, [&]() {
      st = mublasGemmStridedBatchedEx(
          H, MUBLAS_OP_T, MUBLAS_OP_N, N1, M, K1, &alpha, dWgu, MUSA_R_16F, K1, (long long)N1 * K1,
          dAg, MUSA_R_16F, K1, (long long)BLK * K1, &beta, dCgu, MUSA_R_32F, N1,
          (long long)BLK * N1, E, MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
    }, TFLOP1);
    printf("     status=%d useful = %.2f TFLOP/s\n", (int)st, TFLOP1 / (per * 1e9));
    snprintf(nm, sizeof nm, "gate+up StridedBatchedEx batch=576 M=%d", M);
    double per2 = bench(nm, 2, 5, [&]() {
      st = mublasGemmStridedBatchedEx(
          H, MUBLAS_OP_T, MUBLAS_OP_N, N1, M, K1, &alpha, dWgu, MUSA_R_16F, K1, (long long)N1 * K1,
          dAg, MUSA_R_16F, K1, (long long)BLK * K1, &beta, dCgu, MUSA_R_32F, N1,
          (long long)BLK * N1, 2 * E, MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
    }, TFLOPGU);
    printf("     status=%d useful = %.2f TFLOP/s\n", (int)st, TFLOPGU / (per2 * 1e9));
    fflush(stdout);
  }

  // ---------------- phase 5: grouped batched (variable M) ----------------
  printf("\n=== phase5: mublasGemmGroupedBatchedEx (per-call M) ===\n");
  fflush(stdout);
  {
    int gsz1[1] = {E};
    int gsz2[1] = {2 * E};
    mublasOperation_t ta[1] = {MUBLAS_OP_T}, tb[1] = {MUBLAS_OP_N};
    float al[1] = {1.0f}, be[1] = {0.0f};
    const void* A1[E];
    const void* B1[E];
    void* C1[E];
    int m1[E], n1[E], k1[E], lda1[E], ldb1[E], ldc1[E];
    for (int e = 0; e < E; e++) {
      A1[e] = wg_ptr(e); B1[e] = ag_ptr(e, BLK); C1[e] = cg_ptr(0, e, BLK);
      m1[e] = R; n1[e] = N1; k1[e] = K1; lda1[e] = K1; ldb1[e] = K1; ldc1[e] = N1;
    }
    mublasStatus_t st = mublasGemmGroupedBatchedEx(H, ta, tb, m1, n1, k1, al, A1, MUSA_R_16F, lda1,
                                                   B1, MUSA_R_16F, ldb1, be, C1, MUSA_R_32F, ldc1, 1,
                                                   gsz1, MUBLAS_COMPUTE_32F);
    printf("  grouped gate status=%d\n", (int)st);
    fflush(stdout);
    if (st == MUBLAS_STATUS_SUCCESS) {
      // correctness spot-check vs single GemmEx on expert 0
      mublasGemmEx(H, MUBLAS_OP_T, MUBLAS_OP_N, N1, R, K1, &alpha, wg_ptr(0), MUSA_R_16F, K1,
                   ag_ptr(0, BLK), MUSA_R_16F, K1, &beta, cg_ptr(1, 0, BLK), MUSA_R_32F, N1,
                   MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
      musaDeviceSynchronize();
      float a[8], b[8];
      musaMemcpy(a, cg_ptr(0, 0, BLK), 8 * 4, musaMemcpyDeviceToHost);
      musaMemcpy(b, cg_ptr(1, 0, BLK), 8 * 4, musaMemcpyDeviceToHost);
      printf("  spot-check grouped vs single: C[0]=%.6f ref=%.6f\n", a[0], b[0]);
      double best = bench("gate GroupedBatchedEx gsz=288 M=227", 2, 5, [&]() {
        mublasGemmGroupedBatchedEx(H, ta, tb, m1, n1, k1, al, A1, MUSA_R_16F, lda1, B1, MUSA_R_16F,
                                   ldb1, be, C1, MUSA_R_32F, ldc1, 1, gsz1, MUBLAS_COMPUTE_32F);
      }, TFLOP1);
      printf("     useful = %.2f TFLOP/s\n", TFLOP1 / (best * 1e9));

      // gate+up in ONE grouped call (576 entries) -- activations shared
      const void* A2[2 * E];
      const void* B2[2 * E];
      void* C2[2 * E];
      int m2[2 * E], n2[2 * E], k2[2 * E], lda2[2 * E], ldb2[2 * E], ldc2[2 * E];
      for (int e = 0; e < E; e++) {
        A2[e] = wg_ptr(e);     B2[e] = ag_ptr(e, BLK); C2[e] = cg_ptr(0, e, BLK);
        A2[E + e] = wu_ptr(e); B2[E + e] = ag_ptr(e, BLK); C2[E + e] = cg_ptr(1, e, BLK);
        m2[e] = m2[E + e] = R;
        n2[e] = n2[E + e] = N1;
        k2[e] = k2[E + e] = K1;
        lda2[e] = lda2[E + e] = K1;
        ldb2[e] = ldb2[E + e] = K1;
        ldc2[e] = ldc2[E + e] = N1;
      }
      st = mublasGemmGroupedBatchedEx(H, ta, tb, m2, n2, k2, al, A2, MUSA_R_16F, lda2, B2, MUSA_R_16F,
                                      ldb2, be, C2, MUSA_R_32F, ldc2, 1, gsz2, MUBLAS_COMPUTE_32F);
      printf("  grouped gate+up status=%d\n", (int)st);
      fflush(stdout);
      if (st == MUBLAS_STATUS_SUCCESS)
        bench("gate+up GroupedBatchedEx batch=576 M=227", 2, 5, [&]() {
          mublasGemmGroupedBatchedEx(H, ta, tb, m2, n2, k2, al, A2, MUSA_R_16F, lda2, B2, MUSA_R_16F,
                                     ldb2, be, C2, MUSA_R_32F, ldc2, 1, gsz2, MUBLAS_COMPUTE_32F);
        }, TFLOPGU);

      // variable M (true 8192x8 routing: 64 experts get 228 rows)
      int vt = 0;
      for (int e = 0; e < E; e++) { m1[e] = R + (e < 64 ? 1 : 0); vt += m1[e]; }
      printf("  variable-M total rows=%d (8192*8=%d)\n", vt, 8192 * 8);
      double fv = 2.0 * (double)vt * N1 * K1;
      double r = bench("gate GroupedBatchedEx variable M", 2, 5, [&]() {
        mublasGemmGroupedBatchedEx(H, ta, tb, m1, n1, k1, al, A1, MUSA_R_16F, lda1, B1, MUSA_R_16F,
                                   ldb1, be, C1, MUSA_R_32F, ldc1, 1, gsz1, MUBLAS_COMPUTE_32F);
      }, fv);
      printf("     useful = %.2f TFLOP/s\n", fv / (r * 1e9));
      for (int e = 0; e < E; e++) m1[e] = R;
    }

    // OP_N variant: gate weights pre-transposed to [K,N] row-major
    {
      const void* A3[E];
      void* C3[E];
      int m3[E], n3[E], k3[E], lda3[E], ldb3[E], ldc3[E];
      for (int e = 0; e < E; e++) {
        A3[e] = wgt_ptr(e); C3[e] = cg_ptr(1, e, BLK);
        m3[e] = R; n3[e] = N1; k3[e] = K1;
        lda3[e] = N1; ldb3[e] = K1; ldc3[e] = N1;
      }
      mublasOperation_t taN[1] = {MUBLAS_OP_N};
      st = mublasGemmGroupedBatchedEx(H, taN, tb, n3, m3, k3, al, A3, MUSA_R_16F, lda3, B1, MUSA_R_16F,
                                      ldb1, be, C3, MUSA_R_32F, ldc3, 1, gsz1, MUBLAS_COMPUTE_32F);
      printf("  grouped gate OP_N-layout status=%d\n", (int)st);
      fflush(stdout);
      if (st == MUBLAS_STATUS_SUCCESS) {
        // here m/n roles swapped: op(A)=[n3 x k3] -> m=n3? keep simple: use transA=N with (m=N1,n=R)
        bench("gate GroupedBatchedEx OP_N weights", 2, 5, [&]() {
          mublasGemmGroupedBatchedEx(H, taN, tb, n3, m3, k3, al, A3, MUSA_R_16F, lda3, B1,
                                     MUSA_R_16F, ldb1, be, C3, MUSA_R_32F, ldc3, 1, gsz1,
                                     MUBLAS_COMPUTE_32F);
        }, TFLOP1);
      }
    }
  }

  // ---------------- phase 6: down projection ----------------
  printf("\n=== phase6: down projection [K=2048 -> N=4096] ===\n");
  fflush(stdout);
  {
    bench("down single GemmEx M=227", 10, 30, [&]() {
      mublasGemmEx(H, MUBLAS_OP_T, MUBLAS_OP_N, N2, R, K2, &alpha, wd_ptr(0), MUSA_R_16F, K2,
                   ad_ptr(0, BLK), MUSA_R_16F, K2, &beta, cd_ptr(0, BLK), MUSA_R_32F, N2,
                   MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
    }, 2.0 * R * N2 * K2);
    bench("down 288 GemmEx loop", 2, 3, [&]() {
      for (int e = 0; e < E; e++)
        mublasGemmEx(H, MUBLAS_OP_T, MUBLAS_OP_N, N2, R, K2, &alpha, wd_ptr(e), MUSA_R_16F, K2,
                     ad_ptr(e, BLK), MUSA_R_16F, K2, &beta, cd_ptr(e, BLK), MUSA_R_32F, N2,
                     MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
    }, TFLOPD);
    mublasStatus_t st = MUBLAS_STATUS_SUCCESS;
    bench("down StridedBatchedEx batch=288", 2, 5, [&]() {
      st = mublasGemmStridedBatchedEx(H, MUBLAS_OP_T, MUBLAS_OP_N, N2, R, K2, &alpha, dWd, MUSA_R_16F,
                                      K2, (long long)N2 * K2, dAd, MUSA_R_16F, K2,
                                      (long long)BLK * K2, &beta, dCd, MUSA_R_32F, N2,
                                      (long long)BLK * N2, E, MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
    }, TFLOPD);
    printf("     status=%d\n", (int)st);
    int gsz[1] = {E};
    mublasOperation_t ta[1] = {MUBLAS_OP_T}, tb[1] = {MUBLAS_OP_N};
    float al[1] = {1.0f}, be[1] = {0.0f};
    const void* A1[E];
    const void* B1[E];
    void* C1[E];
    int m1[E], n1[E], k1[E], lda1[E], ldb1[E], ldc1[E];
    for (int e = 0; e < E; e++) {
      A1[e] = wd_ptr(e); B1[e] = ad_ptr(e, BLK); C1[e] = cd_ptr(e, BLK);
      m1[e] = R; n1[e] = N2; k1[e] = K2; lda1[e] = K2; ldb1[e] = K2; ldc1[e] = N2;
    }
    st = mublasGemmGroupedBatchedEx(H, ta, tb, m1, n1, k1, al, A1, MUSA_R_16F, lda1, B1, MUSA_R_16F,
                                    ldb1, be, C1, MUSA_R_32F, ldc1, 1, gsz, MUBLAS_COMPUTE_32F);
    printf("  grouped down status=%d\n", (int)st);
    if (st == MUBLAS_STATUS_SUCCESS)
      bench("down GroupedBatchedEx batch=288 M=227", 2, 5, [&]() {
        mublasGemmGroupedBatchedEx(H, ta, tb, m1, n1, k1, al, A1, MUSA_R_16F, lda1, B1, MUSA_R_16F,
                                   ldb1, be, C1, MUSA_R_32F, ldc1, 1, gsz, MUBLAS_COMPUTE_32F);
      }, TFLOPD);

    // ---------------- phase 7: full layer ----------------
    printf("\n=== phase7: full MoE layer (gate+up then down) ===\n");
    fflush(stdout);
    int gsz2[1] = {2 * E};
    const void* A2[2 * E];
    const void* B2[2 * E];
    void* C2[2 * E];
    int m2[2 * E], n2[2 * E], k2[2 * E], lda2[2 * E], ldb2[2 * E], ldc2[2 * E];
    for (int e = 0; e < E; e++) {
      A2[e] = wg_ptr(e);     B2[e] = ag_ptr(e, BLK);     C2[e] = cg_ptr(0, e, BLK);
      A2[E + e] = wu_ptr(e); B2[E + e] = ag_ptr(e, BLK); C2[E + e] = cg_ptr(1, e, BLK);
      m2[e] = m2[E + e] = R;
      n2[e] = n2[E + e] = N1;
      k2[e] = k2[E + e] = K1;
      lda2[e] = lda2[E + e] = K1;
      ldb2[e] = ldb2[E + e] = K1;
      ldc2[e] = ldc2[E + e] = N1;
    }
    int gd[1] = {E};
    const void* Ad[E];
    const void* Bd[E];
    void* Cd[E];
    int md[E], nd[E], kd[E], lad[E], lbd[E], lcd[E];
    for (int e = 0; e < E; e++) {
      Ad[e] = wd_ptr(e); Bd[e] = ad_ptr(e, BLK); Cd[e] = cd_ptr(e, BLK);
      md[e] = R; nd[e] = N2; kd[e] = K2; lad[e] = K2; lbd[e] = K2; lcd[e] = N2;
    }
    double per = bench("FULL LAYER grouped (576 + 288 calls)", 2, 5, [&]() {
      mublasGemmGroupedBatchedEx(H, ta, tb, m2, n2, k2, al, A2, MUSA_R_16F, lda2, B2, MUSA_R_16F,
                                 ldb2, be, C2, MUSA_R_32F, ldc2, 1, gsz2, MUBLAS_COMPUTE_32F);
      mublasGemmGroupedBatchedEx(H, ta, tb, md, nd, kd, al, Ad, MUSA_R_16F, lad, Bd, MUSA_R_16F,
                                 lbd, be, Cd, MUSA_R_32F, lcd, 1, gd, MUBLAS_COMPUTE_32F);
    }, TFLAYER);
    printf("  -> one layer @8K prompt = %.2f ms ; 45 layers = %.2f s ; 8192 tok prefill = %.0f tok/s (GEMM-only, 1 GPU)\n",
           per, per * 45 / 1e3, 8192.0 / (per * 45 / 1e3));
    fflush(stdout);
  }

  printf("\nfreed, exit\n");
  musaFree(dWgu); musaFree(dWd); musaFree(dWgT); musaFree(dAg); musaFree(dAd); musaFree(dCgu); musaFree(dCd);
  musaEventDestroy(ev0); musaEventDestroy(ev1);
  return 0;
}
