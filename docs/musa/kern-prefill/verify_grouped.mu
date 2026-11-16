// Probe mublasGemmGroupedBatchedEx parameter-array semantics on tiny shapes.
// usage: ./verify_grouped <mode>
//   mode 0: all parameter arrays on HOST
//   mode 1: pointer arrays (A/B/C) on DEVICE, scalars/ints on host
//   mode 2: ALL parameter arrays on DEVICE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <musa_runtime.h>
#include <mublas.h>
#include <musa_fp16.h>

#define E 5
#define R 7
#define BLK 8
#define K 64
#define N 32

int main(int argc, char** argv) {
  int mode = argc > 1 ? atoi(argv[1]) : 2;
  __half *dW, *dA;
  float *dC;
  musaMalloc((void**)&dW, (size_t)E * N * K * 2);
  musaMalloc((void**)&dA, (size_t)E * BLK * K * 2);
  musaMalloc((void**)&dC, (size_t)E * BLK * N * 4);
    __half* hW = (__half*)malloc((size_t)E * N * K * 2);
  __half* hA = (__half*)malloc((size_t)E * BLK * K * 2);
  for (size_t i = 0; i < (size_t)E * N * K; i++) hW[i] = __float2half_rn((float)(i / ((size_t)N * K) + 1));
  for (size_t i = 0; i < (size_t)E * BLK * K; i++) hA[i] = __float2half_rn(1.0f);
  musaMemcpy(dW, hW, (size_t)E * N * K * 2, musaMemcpyHostToDevice);
  musaMemcpy(dA, hA, (size_t)E * BLK * K * 2, musaMemcpyHostToDevice);
  musaMemset(dC, 0, (size_t)E * BLK * N * 4);
  
  mublasHandle_t h;
  mublasCreate(&h);
  mublasSetStream(h, 0);
  float alpha = 1.0f, beta = 0.0f;
  (void)alpha; (void)beta;

  // per-call arrays (host copies)
  mublasOperation_t htransa[E], htransb[E];
  int hm[E], hn[E], hk[E], hlda[E], hldb[E], hldc[E], hgs[1] = {E};
  float hal[E], hbe[E];
  const void* hAarr[E];
  const void* hBarr[E];
  void* hCarr[E];
  for (int e = 0; e < E; e++) {
    htransa[e] = MUBLAS_OP_T;
    htransb[e] = MUBLAS_OP_N;
    hm[e] = R; hn[e] = N; hk[e] = K; hlda[e] = K; hldb[e] = K; hldc[e] = N;
    hal[e] = 1.0f; hbe[e] = 0.0f;
    hAarr[e] = dW + (size_t)e * N * K;
    hBarr[e] = dA + (size_t)e * BLK * K;
    hCarr[e] = dC + (size_t)e * BLK * N;
  }

  if (mode == 0) {
    printf("mode0 host arrays\n");
    fflush(stdout);
    mublasStatus_t st = mublasGemmGroupedBatchedEx(h, htransa, htransb, hm, hn, hk, hal, hAarr,
                                                   MUSA_R_16F, hlda, hBarr, MUSA_R_16F, hldb, hbe,
                                                   hCarr, MUSA_R_32F, hldc, 1, hgs, MUBLAS_COMPUTE_32F);
    printf("status=%d\n", (int)st);
  } else {
    // device copies of everything
    mublasOperation_t *dta, *dtb;
    int *dm, *dn, *dk, *dlda, *dldb, *dldc, *dgs;
    float *dal, *dbe;
    const void **dAarr, **dBarr;
    void** dCarr;
    musaMalloc((void**)&dta, E * sizeof(mublasOperation_t));
    musaMalloc((void**)&dtb, E * sizeof(mublasOperation_t));
    musaMalloc((void**)&dm, E * 4);
    musaMalloc((void**)&dn, E * 4);
    musaMalloc((void**)&dk, E * 4);
    musaMalloc((void**)&dlda, E * 4);
    musaMalloc((void**)&dldb, E * 4);
    musaMalloc((void**)&dldc, E * 4);
    musaMalloc((void**)&dgs, 4);
    musaMalloc((void**)&dal, E * 4);
    musaMalloc((void**)&dbe, E * 4);
    musaMalloc((void**)&dAarr, E * sizeof(void*));
    musaMalloc((void**)&dBarr, E * sizeof(void*));
    musaMalloc((void**)&dCarr, E * sizeof(void*));
    musaMemcpy(dta, htransa, E * sizeof(mublasOperation_t), musaMemcpyHostToDevice);
    musaMemcpy(dtb, htransb, E * sizeof(mublasOperation_t), musaMemcpyHostToDevice);
    musaMemcpy(dm, hm, E * 4, musaMemcpyHostToDevice);
    musaMemcpy(dn, hn, E * 4, musaMemcpyHostToDevice);
    musaMemcpy(dk, hk, E * 4, musaMemcpyHostToDevice);
    musaMemcpy(dlda, hlda, E * 4, musaMemcpyHostToDevice);
    musaMemcpy(dldb, hldb, E * 4, musaMemcpyHostToDevice);
    musaMemcpy(dldc, hldc, E * 4, musaMemcpyHostToDevice);
    musaMemcpy(dgs, hgs, 4, musaMemcpyHostToDevice);
    musaMemcpy(dal, hal, E * 4, musaMemcpyHostToDevice);
    musaMemcpy(dbe, hbe, E * 4, musaMemcpyHostToDevice);
    musaMemcpy(dAarr, hAarr, E * sizeof(void*), musaMemcpyHostToDevice);
    musaMemcpy(dBarr, hBarr, E * sizeof(void*), musaMemcpyHostToDevice);
    musaMemcpy(dCarr, hCarr, E * sizeof(void*), musaMemcpyHostToDevice);
    musaDeviceSynchronize();
    printf("mode%d device arrays\n", mode);
    fflush(stdout);
    mublasStatus_t st;
    if (mode == 1) {
      st = mublasGemmGroupedBatchedEx(h, htransa, htransb, hm, hn, hk, hal, (const void* const*)dAarr,
                                      MUSA_R_16F, hlda, (const void* const*)dBarr, MUSA_R_16F, hldb,
                                      hbe, (void* const*)dCarr, MUSA_R_32F, hldc, 1, hgs,
                                      MUBLAS_COMPUTE_32F);
    } else {
      st = mublasGemmGroupedBatchedEx(h, dta, dtb, dm, dn, dk, dal, (const void* const*)dAarr,
                                      MUSA_R_16F, dlda, (const void* const*)dBarr, MUSA_R_16F, dldb,
                                      dbe, (void* const*)dCarr, MUSA_R_32F, dldc, 1, dgs,
                                      MUBLAS_COMPUTE_32F);
    }
    printf("status=%d\n", (int)st);
  }
  musaDeviceSynchronize();
  printf("sync done\n");
  fflush(stdout);

  float* cs = (float*)malloc((size_t)E * BLK * N * 4);
  
  musaMemcpy(cs, dC, (size_t)E * BLK * N * 4, musaMemcpyDeviceToHost);
  int bad = 0;
  for (int e = 0; e < E; e++) {
    float expect = (float)(e + 1) * K;
    for (int r = 0; r < R; r++)
      for (int c = 0; c < N; c++) {
        float a = cs[((size_t)e * BLK + r) * N + c];
        if (fabs(a - expect) > 1e-2) { if (bad < 3) printf("  mismatch e=%d r=%d c=%d got %f want %f\n", e, r, c, a, expect); bad++; }
      }
  }
  printf("correctness: bad=%d/%d  %s\n", bad, E * R * N, bad ? "FAIL" : "PASS");
  return bad != 0;
}
