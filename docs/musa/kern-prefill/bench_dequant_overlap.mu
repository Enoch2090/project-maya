// Dequant-cost proxy + stream-overlap test.
//  (1) dequant-like kernel: read ~2 bit/weight, write fp16  -> one layer of expert weights (14.5 GB out)
//  (2) the same layer's gate+up+down strided-batched GEMM reading that fp16 buffer
//  (3) both at once on two streams: does overlap help, or do they fight over HBM?
#include <stdio.h>
#include <stdlib.h>
#include <musa_runtime.h>
#include <mublas.h>
#include <musa_fp16.h>

#define E  288
#define N1 2048
#define K1 4096
#define N2 4096
#define K2 2048
#define R  227
#define BLK 256

static size_t NW;  // fp16 weights per layer = E*(N1*K1*2 + N2*K2) elements? compute below

static musaEvent_t e0, e1;
static mublasHandle_t H;

// dequant proxy: 4 output halves per input byte (roughly 2.06 bpw), LUT-ish math
__global__ void deqk(const unsigned char* __restrict__ q, __half* __restrict__ out, long nhalf) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (; i < nhalf; i += stride) {
    long byte = i >> 2;
    int sh = (int)((i & 3) * 4);
    unsigned int nib = (q[byte] >> sh) & 0xF;
    out[i] = __float2half_rn((float)nib * 0.0625f - 0.5f);
  }
}

__global__ void wrk(__half* __restrict__ out, long nhalf) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long stride = (long)gridDim.x * blockDim.x;
  for (; i < nhalf; i += stride) out[i] = __float2half_rn(0.25f);
}

int main() {
  musaEventCreate(&e0);
  musaEventCreate(&e1);
  mublasCreate(&H);
  mublasSetStream(H, 0);
  float alpha = 1.0f, beta = 0.0f;

  size_t wgu = (size_t)2 * E * N1 * K1;   // elements, gate+up
  size_t wd  = (size_t)E * N2 * K2;   // down
  size_t nW = wgu + wd;               // fp16 weights in one layer = 14.5 GB
  size_t nq = nW / 4;                 // "2 bit" packed quantized
  printf("one layer fp16 weights = %.2f GB, quantized %zu bytes = %.2f GB\n", nW * 2 / 1e9, nq,
         nq / 1e9);
  fflush(stdout);

  unsigned char* dq;
  __half* dW;
  __half* dA;
  float* dC;
  musaMalloc((void**)&dq, nq);
  musaMalloc((void**)&dW, nW * 2);
  musaMalloc((void**)&dA, (size_t)E * BLK * K1 * 2);
  musaMalloc((void**)&dC, (size_t)2 * E * BLK * N1 * 4);
  musaMemset(dq, 0x21, nq);
  musaMemset(dW, 0, nW * 2);
  musaMemset(dA, 0, (size_t)E * BLK * K1 * 2);
  musaMemset(dC, 0, (size_t)2 * E * BLK * N1 * 4);
  printf("alloc done\n");
  fflush(stdout);

  int grid = 56 * 16;
  auto deq = [&]() { deqk<<<grid, 256>>>(dq, dW, (long)nW); };
  auto gemm = [&]() {
    mublasGemmStridedBatchedEx(H, MUBLAS_OP_T, MUBLAS_OP_N, N1, R, K1, &alpha, dW, MUSA_R_16F, K1,
                               (long long)N1 * K1, dA, MUSA_R_16F, K1, (long long)BLK * K1, &beta,
                               dC, MUSA_R_32F, N1, (long long)BLK * N1, E, MUBLAS_COMPUTE_32F,
                               MUBLAS_GEMM_DEFAULT);
    mublasGemmStridedBatchedEx(H, MUBLAS_OP_T, MUBLAS_OP_N, N1, R, K1, &alpha,
                               dW + (size_t)wgu, MUSA_R_16F, K1, (long long)N1 * K1, dA, MUSA_R_16F,
                               K1, (long long)BLK * K1, &beta, dC + (size_t)E * BLK * N1, MUSA_R_32F,
                               N1, (long long)BLK * N1, E, MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
    mublasGemmStridedBatchedEx(H, MUBLAS_OP_T, MUBLAS_OP_N, N2, R, K2, &alpha,
                               dW + (size_t)wgu, MUSA_R_16F, K2, (long long)N2 * K2,
                               dA, MUSA_R_16F, K2, (long long)BLK * K2, &beta,
                               dC, MUSA_R_32F, N2, (long long)BLK * N2, E, MUBLAS_COMPUTE_32F,
                               MUBLAS_GEMM_DEFAULT);
  };

  double tdeq = 0, tgemm = 0, tboth = 0;
  // (1) dequant alone
  for (int i = 0; i < 3; i++) deq();
  musaDeviceSynchronize();
  musaEventRecord(e0, 0);
  for (int i = 0; i < 3; i++) deq();
  musaEventRecord(e1, 0);
  musaDeviceSynchronize();
  { float ms; musaEventElapsedTime(&ms, e0, e1); tdeq = ms / 3; }
  // (2) gemm alone
  for (int i = 0; i < 3; i++) gemm();
  musaDeviceSynchronize();
  musaEventRecord(e0, 0);
  for (int i = 0; i < 3; i++) gemm();
  musaEventRecord(e1, 0);
  musaDeviceSynchronize();
  { float ms; musaEventElapsedTime(&ms, e0, e1); tgemm = ms / 3; }
  // (2b) pure fp16 write of the same 14.5 GB (upper bound for dequant)
  double twr = 0;
  for (int i = 0; i < 3; i++) wrk<<<grid, 256>>>(dW, (long)nW);
  musaDeviceSynchronize();
  musaEventRecord(e0, 0);
  for (int i = 0; i < 3; i++) wrk<<<grid, 256>>>(dW, (long)nW);
  musaEventRecord(e1, 0);
  musaDeviceSynchronize();
  { float ms; musaEventElapsedTime(&ms, e0, e1); twr = ms / 3; }

  // (3) both on two streams
  musaStream_t s2;
  musaStreamCreate(&s2);
  mublasSetStream(H, s2);
  for (int i = 0; i < 2; i++) { deq(); gemm(); }
  musaDeviceSynchronize();
  musaEventRecord(e0, 0);
  for (int i = 0; i < 3; i++) {
    deq();            // default stream
    gemm();           // stream s2 (mublas handle)
  }
  musaEventRecord(e1, 0);
  musaDeviceSynchronize();
  { float ms; musaEventElapsedTime(&ms, e0, e1); tboth = ms / 3; }

  printf("dequant-proxy (read %.2f GB + write %.2f GB) : %8.3f ms/layer  (%.0f GB/s traffic, x45 = %.2f s)\n",
         nq / 1e9, nW * 2 / 1e9, tdeq, (nq + nW * 2) / (tdeq * 1e6), tdeq * 45 / 1e3);
  printf("pure fp16 write 14.5 GB (no dequant math)    : %8.3f ms/layer  (%.0f GB/s, x45 = %.2f s)\n", twr,
         nW * 2 / (twr * 1e6), twr * 45 / 1e3);
  printf("GEMM layer (gate+up+down)                    : %8.3f ms/layer  (x45 = %.2f s)\n", tgemm,
         tgemm * 45 / 1e3);
  printf("both on 2 streams                            : %8.3f ms  (sum=%.3f, max=%.3f) -> overlap gain %.2fx\n",
         tboth, tdeq + tgemm, tdeq > tgemm ? tdeq : tgemm, (tdeq + tgemm) / tboth);
  printf("=> serial 4-card estimates: dequant+gemm no-overlap %.2f s (%.0f tok/s), with overlap %.2f s (%.0f tok/s)\n",
         (tdeq + tgemm) * 45 / 1e3, 8192.0 / ((tdeq + tgemm) * 45 / 1e3), tboth * 45 / 1e3,
         8192.0 / (tboth * 45 / 1e3));
  printf("done\n");
  return 0;
}
