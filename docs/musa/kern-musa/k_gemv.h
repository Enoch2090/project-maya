#pragma once
#include "k_decl.h"

// Optimized quantized GEMV.
// one warp = one row: the 32 lanes split a k-chunk, lane handles 32*NSUB elements
// block = WPB warps; CPB = WPB/R warps per row; grid.x = n_experts*ceil(n_out/R),
// grid.y = number of cross-block k splits (needs ATOMIC).
template<int TY, int R, int WPB, int CPB, int NSUB, int M, bool ATOMIC>
__global__ void __launch_bounds__(WPB*32)
gemv_k(const uint8_t* __restrict__ w,
       const int8_t* __restrict__ xq,   // [M][ng][36]
       const float* __restrict__ xs,    // [M][ng]
       float* __restrict__ y,           // [M][n_out]
       int rowtiles, int n_out, int ng, int row_bytes)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int e  = blockIdx.x / rowtiles;
    const int rt = blockIdx.x - e * rowtiles;
    const int row = rt * R + (warp % R);
    const int cpb = warp / R;
    const int chunk = blockIdx.y * CPB + cpb;
    const size_t e_stride = (size_t)n_out * row_bytes;
    const uint8_t* wrow = w + (size_t)e * e_stride + (size_t)row * row_bytes;

    const int sbase = chunk * (32 * NSUB) + lane;
    uint64_t qv[NSUB], pv[NSUB];
    uint32_t av[NSUB];
    uint16_t dv[NSUB];

#pragma unroll
    for (int i = 0; i < NSUB; ++i) {
        const int s = sbase + 32 * i;
        pv[i] = 0; av[i] = 0;
        if (TY == T_IQ2XXS) {
            const uint8_t* p = wrow + (size_t)(s >> 3) * 66;
            qv[i] = *(const uint64_t*)(p + (s & 7) * 8);
            dv[i] = *(const uint16_t*)p;
        } else if (TY == T_IQ2S) {
            const uint8_t* p = wrow + (size_t)(s >> 3) * 82;
            qv[i] = *(const uint64_t*)(p + 2 + (s & 7) * 4);
            av[i] = *(const uint32_t*)(p + 34 + (s & 7) * 4);
            dv[i] = *(const uint16_t*)p;
            pv[i] = (uint64_t)p[66 + (s & 7)] | ((uint64_t)p[74 + (s & 7)] << 8);
        } else {
            const uint8_t* p = wrow + (size_t)(s >> 3) * 98;
            qv[i] = *(const uint64_t*)(p + 2 + (s & 7) * 8);
            av[i] = *(const uint32_t*)(p + 66 + (s & 7) * 4);
            dv[i] = *(const uint16_t*)p;
        }
    }

    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;

#pragma unroll
    for (int i = 0; i < NSUB; ++i) {
        const int s = sbase + 32 * i;
        const float d = __half2float(*(const __half*)&dv[i]);
        if (TY == T_IQ2XXS) {
            uint32_t a0 = (uint32_t)qv[i], a1 = (uint32_t)(qv[i] >> 32);
            float db = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
            int w0s[4], w1s[4];
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                uint64_t g = D_IQ2XXS[(a0 >> (8*l)) & 255];
                uint64_t mm = D_KSIGNS64[(a1 >> (7*l)) & 127];
                uint64_t sg = (g ^ mm) + (mm & 0x0101010101010101ull);
                w0s[l] = (int)sg; w1s[l] = (int)(sg >> 32);
            }
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const int* xp = (const int*)(xq + ((size_t)m * ng + s) * 36);
                int sA = 0, sB = 0;
#pragma unroll
                for (int l = 0; l < 4; ++l) { sA = __dp4a(w0s[l], xp[2*l], sA); sB = __dp4a(w1s[l], xp[2*l+1], sB); }
                acc[m] = fmaf((float)(sA + sB), db * xs[(size_t)m * ng + s], acc[m]);
            }
        } else if (TY == T_IQ2S) {
            uint32_t qs32 = (uint32_t)qv[i];
            uint32_t sgn = av[i];
            int qh = (int)(pv[i] & 0xff), sc = (int)(pv[i] >> 8);
            float db0 = d * (0.5f + (float)(sc & 0xf)) * 0.25f;
            float db1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                int gi = (int)((qs32 >> (8*l)) & 0xff) | ((qh << (8 - 2*l)) & 0x300);
                uint64_t g = D_IQ2S[gi];
                uint64_t mm = D_SGB[(sgn >> (8*l)) & 0xff];
                uint64_t sg = (g ^ mm) + (mm & 0x0101010101010101ull);
                int w0 = (int)sg, w1 = (int)(sg >> 32);
                float f = (l < 2) ? db0 : db1;
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    const int* xp = (const int*)(xq + ((size_t)m * ng + s) * 36) + 2*l;
                    int sm = __dp4a(w0, xp[0], __dp4a(w1, xp[1], 0));
                    acc[m] = fmaf((float)sm, f * xs[(size_t)m * ng + s], acc[m]);
                }
            }
        } else {
            uint64_t q3 = qv[i];
            uint32_t aux = av[i];
            float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                uint64_t mm = D_KSIGNS64[(aux >> (7*l)) & 127];
                uint32_t mlo = (uint32_t)mm, mhi = (uint32_t)(mm >> 32);
                uint32_t g1 = D_IQ3XXS[(q3 >> (8*(2*l))) & 255];
                uint32_t g2 = D_IQ3XXS[(q3 >> (8*(2*l+1))) & 255];
                int w0 = (int)((g1 ^ mlo) + (mlo & 0x01010101u));
                int w1 = (int)((g2 ^ mhi) + (mhi & 0x01010101u));
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    const int* xp = (const int*)(xq + ((size_t)m * ng + s) * 36) + 2*l;
                    int sm = __dp4a(w0, xp[0], __dp4a(w1, xp[1], 0));
                    acc[m] = fmaf((float)sm, db * xs[(size_t)m * ng + s], acc[m]);
                }
            }
        }
    }

#pragma unroll
    for (int m = 0; m < M; ++m) {
        float v = acc[m];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off, 32);
        acc[m] = v;
    }

    if (CPB > 1) {
        __shared__ float smem[CPB][R][M];
        if (lane == 0) {
#pragma unroll
            for (int m = 0; m < M; ++m) smem[cpb][warp % R][m] = acc[m];
        }
        __syncthreads();
        const int t = threadIdx.x;
        if (t < R * M) {
            const int rr = t / M, m = t % M;
            float v = 0.f;
#pragma unroll
            for (int c = 0; c < CPB; ++c) v += smem[c][rr][m];
            const size_t idx = (size_t)m * n_out + (rt * R + rr);
            if (ATOMIC) atomicAdd(&y[idx], v); else y[idx] = v;
        }
    } else if (lane == 0) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const size_t j = (size_t)m * n_out + row;
            if (ATOMIC) atomicAdd(&y[j], acc[m]); else y[j] = acc[m];
        }
    }
}

// read-only ceiling: identical address pattern, no dequant work
template<int TY, int R, int WPB, int CPB, int NSUB>
__global__ void __launch_bounds__(WPB*32)
read_k(const uint8_t* __restrict__ w, float* __restrict__ y,
       int rowtiles, int n_out, int row_bytes)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int e  = blockIdx.x / rowtiles;
    const int rt = blockIdx.x - e * rowtiles;
    const int row = rt * R + (warp % R);
    const int cpb = warp / R;
    const int chunk = blockIdx.y * CPB + cpb;
    const uint8_t* wrow = w + (size_t)e * n_out * row_bytes + (size_t)row * row_bytes;
    const int sbase = chunk * (32 * NSUB) + lane;
    uint64_t a = 0;
#pragma unroll
    for (int i = 0; i < NSUB; ++i) {
        const int s = sbase + 32 * i;
        uint64_t v;
        if (TY == T_IQ2XXS) v = *(const uint64_t*)(wrow + (size_t)(s>>3)*66 + (s&7)*8);
        else if (TY == T_IQ2S) v = *(const uint64_t*)(wrow + (size_t)(s>>3)*82 + 2 + (s&7)*4);
        else v = *(const uint64_t*)(wrow + (size_t)(s>>3)*98 + 2 + (s&7)*8);
        a += v;
    }
    if (a == 0xdeadbeefull) y[row] = 1.f;
}

// naive ggml-style baseline: block 256 = 8 rows, warp = row, lane = 8 consecutive elems
template<int TY>
__global__ void __launch_bounds__(256)
naive_k(const uint8_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,
        int n_out, int k, int row_bytes)
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= n_out) return;
    const uint8_t* wrow = w + (size_t)blockIdx.y * n_out * row_bytes + (size_t)row * row_bytes;
    const int nb = k / 256;
    float acc = 0.f;
    for (int sb = 0; sb < nb; ++sb) {
        const float* xb = x + sb * 256 + lane * 8;
        const int sbl = lane >> 2, grp = lane & 3;
        if (TY == T_IQ2XXS) {
            const uint8_t* p = wrow + (size_t)sb * 66;
            uint64_t q = *(const uint64_t*)(p + sbl * 8);
            uint32_t a0 = (uint32_t)q, a1 = (uint32_t)(q >> 32);
            float d = __half2float(*(const __half*)p);
            float db = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
            uint64_t g = D_IQ2XXS[(a0 >> (8*grp)) & 255];
            uint8_t sgn = D_KSB[(a1 >> (7*grp)) & 127];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float wv = db * (float)((g >> (8*j)) & 0xff);
                if (sgn & (1u << j)) wv = -wv;
                acc = fmaf(wv, xb[j], acc);
            }
        } else if (TY == T_IQ2S) {
            const uint8_t* p = wrow + (size_t)sb * 82;
            int qh = p[66 + sbl], sc = p[74 + sbl];
            uint8_t sgn = p[34 + sbl * 4 + grp];
            float d = __half2float(*(const __half*)p);
            float db = d * (0.5f + (float)((grp < 2) ? (sc & 0xf) : (sc >> 4))) * 0.25f;
            uint64_t g = D_IQ2S[(int)p[2 + sbl*4 + grp] | ((qh << (8 - 2*grp)) & 0x300)];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float wv = db * (float)((g >> (8*j)) & 0xff);
                if (sgn & (1u << j)) wv = -wv;
                acc = fmaf(wv, xb[j], acc);
            }
        } else {
            const uint8_t* p = wrow + (size_t)sb * 98;
            uint32_t aux = *(const uint32_t*)(p + 66 + sbl * 4);
            uint8_t sgn = D_KSB[(aux >> (7*grp)) & 127];
            float d = __half2float(*(const __half*)p);
            float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
            uint32_t g1 = D_IQ3XXS[p[2 + sbl * 8 + 2*grp]];
            uint32_t g2 = D_IQ3XXS[p[2 + sbl * 8 + 2*grp + 1]];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float wv = db * (float)((g1 >> (8*j)) & 0xff);
                if (sgn & (1u << j)) wv = -wv;
                acc = fmaf(wv, xb[j], acc);
                float wv2 = db * (float)((g2 >> (8*j)) & 0xff);
                if (sgn & (1u << (j+4))) wv2 = -wv2;
                acc = fmaf(wv2, xb[j+4], acc);
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off, 32);
    if (lane == 0) y[row] = acc;
}

// ---- shared-memory variant: tables + xq staged in __shared__ ----
template<int TY, int R, int WPB, int CPB, int NSUB, int M>
__global__ void __launch_bounds__(WPB*32)
gemv_ks(const uint8_t* __restrict__ w,
        const int8_t* __restrict__ xq,   // [M][ng][36]
        const float* __restrict__ xs,    // [M][ng]
        float* __restrict__ y,
        int rowtiles, int n_out, int ng, int row_bytes)
{
    const int th = WPB * 32, tid = threadIdx.x;
    __shared__ uint64_t sg[TY == T_IQ2S ? 1024 : 256];
    __shared__ uint64_t ssig[TY == T_IQ2S ? 256 : 128];
    __shared__ int8_t   sxq[4608];
    __shared__ float    sxs[8 * 128];
    const int ntab = (TY == T_IQ2S) ? 1024 : 256;
    for (int i = tid; i < ntab; i += th)
        sg[i] = (TY == T_IQ3XXS) ? (uint64_t)D_IQ3XXS[i] : (TY == T_IQ2S ? D_IQ2S[i] : D_IQ2XXS[i]);
    const int nsig = (TY == T_IQ2S) ? 256 : 128;
    for (int i = tid; i < nsig; i += th)
        ssig[i] = (TY == T_IQ2S) ? D_SGB[i] : D_KSIGNS64[i];
    for (int i = tid; i < M * ng * 36; i += th) sxq[i] = xq[i];
    for (int i = tid; i < M * ng; i += th) sxs[i] = xs[i];
    __syncthreads();

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e  = blockIdx.x / rowtiles;
    const int rt = blockIdx.x - e * rowtiles;
    const int row = rt * R + (warp % R);
    const int cpb = warp / R;
    const int chunk = blockIdx.y * CPB + cpb;
    const uint8_t* wrow = w + (size_t)e * n_out * row_bytes + (size_t)row * row_bytes;
    const int sbase = chunk * (32 * NSUB) + lane;

    uint64_t qv[NSUB], pv[NSUB];
    uint32_t av[NSUB];
    uint16_t dv[NSUB];
#pragma unroll
    for (int i = 0; i < NSUB; ++i) {
        const int s = sbase + 32 * i;
        pv[i] = 0; av[i] = 0;
        if (TY == T_IQ2XXS) {
            const uint8_t* p = wrow + (size_t)(s >> 3) * 66;
            qv[i] = *(const uint64_t*)(p + (s & 7) * 8);
            dv[i] = *(const uint16_t*)p;
        } else if (TY == T_IQ2S) {
            const uint8_t* p = wrow + (size_t)(s >> 3) * 82;
            qv[i] = *(const uint64_t*)(p + 2 + (s & 7) * 4);
            av[i] = *(const uint32_t*)(p + 34 + (s & 7) * 4);
            dv[i] = *(const uint16_t*)p;
            pv[i] = (uint64_t)p[66 + (s & 7)] | ((uint64_t)p[74 + (s & 7)] << 8);
        } else {
            const uint8_t* p = wrow + (size_t)(s >> 3) * 98;
            qv[i] = *(const uint64_t*)(p + 2 + (s & 7) * 8);
            av[i] = *(const uint32_t*)(p + 66 + (s & 7) * 4);
            dv[i] = *(const uint16_t*)p;
        }
    }
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
#pragma unroll
    for (int i = 0; i < NSUB; ++i) {
        const int s = sbase + 32 * i;
        const float d = __half2float(*(const __half*)&dv[i]);
        if (TY == T_IQ2XXS) {
            uint32_t a0 = (uint32_t)qv[i], a1 = (uint32_t)(qv[i] >> 32);
            float db = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                uint64_t g = sg[(a0 >> (8*l)) & 255];
                uint64_t mm = ssig[(a1 >> (7*l)) & 127];
                uint64_t wq = (g ^ mm) + (mm & 0x0101010101010101ull);
                int w0 = (int)wq, w1 = (int)(wq >> 32);
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    const int* xp = (const int*)(sxq + ((size_t)m * ng + s) * 36) + 2*l;
                    int sm = __dp4a(w0, xp[0], __dp4a(w1, xp[1], 0));
                    acc[m] = fmaf((float)sm, db * sxs[(size_t)m * ng + s], acc[m]);
                }
            }
        } else if (TY == T_IQ2S) {
            uint32_t qs32 = (uint32_t)qv[i];
            uint32_t sgn = av[i];
            int qh = (int)(pv[i] & 0xff), sc = (int)(pv[i] >> 8);
            float db0 = d * (0.5f + (float)(sc & 0xf)) * 0.25f;
            float db1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                int gi = (int)((qs32 >> (8*l)) & 0xff) | ((qh << (8 - 2*l)) & 0x300);
                uint64_t g = sg[gi];
                uint64_t mm = ssig[(sgn >> (8*l)) & 0xff];
                uint64_t wq = (g ^ mm) + (mm & 0x0101010101010101ull);
                int w0 = (int)wq, w1 = (int)(wq >> 32);
                float f = (l < 2) ? db0 : db1;
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    const int* xp = (const int*)(sxq + ((size_t)m * ng + s) * 36) + 2*l;
                    int sm = __dp4a(w0, xp[0], __dp4a(w1, xp[1], 0));
                    acc[m] = fmaf((float)sm, f * sxs[(size_t)m * ng + s], acc[m]);
                }
            }
        } else {
            uint64_t q3 = qv[i];
            uint32_t aux = av[i];
            float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                uint64_t mm = ssig[(aux >> (7*l)) & 127];
                uint32_t mlo = (uint32_t)mm, mhi = (uint32_t)(mm >> 32);
                uint32_t g1 = (uint32_t)sg[(q3 >> (8*(2*l))) & 255];
                uint32_t g2 = (uint32_t)sg[(q3 >> (8*(2*l+1))) & 255];
                int w0 = (int)((g1 ^ mlo) + (mlo & 0x01010101u));
                int w1 = (int)((g2 ^ mhi) + (mhi & 0x01010101u));
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    const int* xp = (const int*)(sxq + ((size_t)m * ng + s) * 36) + 2*l;
                    int sm = __dp4a(w0, xp[0], __dp4a(w1, xp[1], 0));
                    acc[m] = fmaf((float)sm, db * sxs[(size_t)m * ng + s], acc[m]);
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        float v = acc[m];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off, 32);
        acc[m] = v;
    }
    if (CPB > 1) {
        __shared__ float red[CPB][R][M];
        if (lane == 0) {
#pragma unroll
            for (int m = 0; m < M; ++m) red[cpb][warp % R][m] = acc[m];
        }
        __syncthreads();
        const int t = threadIdx.x;
        if (t < R * M) {
            const int rr = t / M, m = t % M;
            float v = 0.f;
#pragma unroll
            for (int c = 0; c < CPB; ++c) v += red[c][rr][m];
            y[(size_t)m * n_out + (rt * R + rr)] = v;
        }
    } else if (lane == 0) {
#pragma unroll
        for (int m = 0; m < M; ++m) y[(size_t)m * n_out + row] = acc[m];
    }
}

// ---- isolation probes: same access pattern, partial math ----
// mode 0: qs loads + grid/sign table lookups only (no dp4a / no xq)
// mode 1: qs loads + dp4a with synthetic grid (no table lookups, no xq)
// mode 2: qs loads + table lookups + dp4a (no xq: xq word fixed)
template<int TY, int R, int WPB, int CPB, int NSUB, int MODE>
__global__ void __launch_bounds__(WPB*32)
probe_k(const uint8_t* __restrict__ w, const int8_t* __restrict__ xq, float* __restrict__ y,
        int rowtiles, int n_out, int row_bytes)
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e  = blockIdx.x / rowtiles;
    const int rt = blockIdx.x - e * rowtiles;
    const int row = rt * R + (warp % R);
    const int cpb = warp / R;
    const int chunk = blockIdx.y * CPB + cpb;
    const uint8_t* wrow = w + (size_t)e * n_out * row_bytes + (size_t)row * row_bytes;
    const int sbase = chunk * (32 * NSUB) + lane;
    uint64_t qv[NSUB], pv[NSUB]; uint32_t av[NSUB]; uint16_t dv[NSUB];
#pragma unroll
    for (int i = 0; i < NSUB; ++i) {
        const int s = sbase + 32 * i;
        pv[i] = 0; av[i] = 0;
        if (TY == T_IQ2XXS) {
            const uint8_t* p = wrow + (size_t)(s >> 3) * 66;
            qv[i] = *(const uint64_t*)(p + (s & 7) * 8);
            dv[i] = *(const uint16_t*)p;
        } else if (TY == T_IQ2S) {
            const uint8_t* p = wrow + (size_t)(s >> 3) * 82;
            qv[i] = *(const uint64_t*)(p + 2 + (s & 7) * 4);
            av[i] = *(const uint32_t*)(p + 34 + (s & 7) * 4);
            dv[i] = *(const uint16_t*)p;
            pv[i] = (uint64_t)p[66 + (s & 7)] | ((uint64_t)p[74 + (s & 7)] << 8);
        } else {
            const uint8_t* p = wrow + (size_t)(s >> 3) * 98;
            qv[i] = *(const uint64_t*)(p + 2 + (s & 7) * 8);
            av[i] = *(const uint32_t*)(p + 66 + (s & 7) * 4);
            dv[i] = *(const uint16_t*)p;
        }
    }
    float acc = 0.f;
    int isum = 0;
#pragma unroll
    for (int i = 0; i < NSUB; ++i) {
        const int s = sbase + 32 * i;
        if (TY == T_IQ2XXS) {
            uint32_t a0 = (uint32_t)qv[i], a1 = (uint32_t)(qv[i] >> 32);
            float db = __half2float(*(const __half*)&dv[i]) * (0.5f + (float)(a1 >> 28)) * 0.25f;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                uint64_t sg = D_IQ2XXS[(a0 >> (8*l)) & 255];
                uint64_t mm = D_KSIGNS64[(a1 >> (7*l)) & 127];
                uint64_t wq = (sg ^ mm) + (mm & 0x0101010101010101ull);
                if (MODE == 0) {
                    acc += db * (float)(int)((wq & 0xff) + ((wq >> 8) & 0xff));
                } else if (MODE == 1) {
                    int w0 = (int)qv[i], w1 = (int)(qv[i] >> 32);
                    const int* xp = (const int*)(xq + ((size_t)0 + s) * 36) + 2*l;
                    isum += __dp4a(w0, xp[0], __dp4a(w1, xp[1], 0));
                } else {
                    int w0 = (int)wq, w1 = (int)(wq >> 32);
                    const int* xp = (const int*)(xq + ((size_t)0 + s) * 36) + 2*l;
                    isum += __dp4a(w0, xp[0], __dp4a(w1, xp[1], 0));
                }
            }
        } else if (TY == T_IQ2S) {
            uint32_t qs32 = (uint32_t)qv[i]; uint32_t sgn = av[i];
            int qh = (int)(pv[i] & 0xff), sc = (int)(pv[i] >> 8);
            float db0 = __half2float(*(const __half*)&dv[i]) * (0.5f + (float)(sc & 0xf)) * 0.25f;
            float db1 = __half2float(*(const __half*)&dv[i]) * (0.5f + (float)(sc >> 4)) * 0.25f;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                int gi = (int)((qs32 >> (8*l)) & 0xff) | ((qh << (8 - 2*l)) & 0x300);
                uint64_t sg = D_IQ2S[gi];
                uint64_t mm = D_SGB[(sgn >> (8*l)) & 0xff];
                uint64_t wq = (sg ^ mm) + (mm & 0x0101010101010101ull);
                if (MODE == 0) {
                    acc += ((l < 2) ? db0 : db1) * (float)(int)((wq & 0xff) + ((wq >> 8) & 0xff));
                } else if (MODE == 1) {
                    int w0 = (int)qv[i], w1 = (int)(qv[i] >> 32);
                    const int* xp = (const int*)(xq + ((size_t)0 + s) * 36) + 2*l;
                    isum += __dp4a(w0, xp[0], __dp4a(w1, xp[1], 0));
                } else {
                    int w0 = (int)wq, w1 = (int)(wq >> 32);
                    const int* xp = (const int*)(xq + ((size_t)0 + s) * 36) + 2*l;
                    isum += __dp4a(w0, xp[0], __dp4a(w1, xp[1], 0));
                }
            }
        } else {
            uint64_t q3 = qv[i]; uint32_t aux = av[i];
            float db = __half2float(*(const __half*)&dv[i]) * (0.5f + (float)(aux >> 28)) * 0.5f;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                uint64_t mm = D_KSIGNS64[(aux >> (7*l)) & 127];
                uint32_t mlo = (uint32_t)mm, mhi = (uint32_t)(mm >> 32);
                uint32_t g1 = D_IQ3XXS[(q3 >> (8*(2*l))) & 255];
                uint32_t g2 = D_IQ3XXS[(q3 >> (8*(2*l+1))) & 255];
                if (MODE == 0) {
                    acc += db * (float)(int)((g1 & 0xff) + (g2 & 0xff));
                } else {
                    int w0 = (int)((g1 ^ mlo) + (mlo & 0x01010101u));
                    int w1 = (int)((g2 ^ mhi) + (mhi & 0x01010101u));
                    const int* xp = (const int*)(xq + ((size_t)0 + s) * 36) + 2*l;
                    isum += __dp4a(w0, xp[0], __dp4a(w1, xp[1], 0));
                }
            }
        }
    }
    float v = acc + (float)isum;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off, 32);
    if (lane == 0 && v == 1.2345e30f) y[row] = v;
}

// ---- layout B: lane = one row (32 rows per warp), warps split k.
//      x vector accesses become warp-uniform (broadcast); grid/sign tables in shared.
template<int TY, int WPB, int NSUB, int U>
__global__ void __launch_bounds__(WPB*32)
gemv_lb(const uint8_t* __restrict__ w,
        const int8_t* __restrict__ xq,   // [ng][36], ng = k/32
        const float* __restrict__ xs,    // [ng]
        float* __restrict__ y,
        int rowtiles, int n_out, int ng, int row_bytes)
{
    const int th = WPB * 32, tid = threadIdx.x;
    __shared__ uint64_t sg[TY == T_IQ2S ? 1024 : 256];
    __shared__ uint64_t ssig[TY == T_IQ2S ? 256 : 128];
    __shared__ float red[WPB][32];
    const int ntab = (TY == T_IQ2S) ? 1024 : 256;
    for (int i = tid; i < ntab; i += th)
        sg[i] = (TY == T_IQ3XXS) ? (uint64_t)D_IQ3XXS[i] : (TY == T_IQ2S ? D_IQ2S[i] : D_IQ2XXS[i]);
    const int nsig = (TY == T_IQ2S) ? 256 : 128;
    for (int i = tid; i < nsig; i += th)
        ssig[i] = (TY == T_IQ2S) ? D_SGB[i] : D_KSIGNS64[i];
    __syncthreads();

    const int warp = tid >> 5, lane = tid & 31;
    const int e  = blockIdx.x / rowtiles;
    const int rt = blockIdx.x - e * rowtiles;
    const int row = rt * 32 + lane;
    const uint8_t* wrow = w + (size_t)e * n_out * row_bytes + (size_t)row * row_bytes;
    const int s0 = warp * NSUB;

    float acc = 0.f;
    for (int base = 0; base < NSUB; base += U) {
        uint64_t qv[U], pv[U];
        uint32_t av[U];
        uint16_t dv[U];
#pragma unroll
        for (int i = 0; i < U; ++i) {
            const int s = s0 + base + i;
            pv[i] = 0; av[i] = 0;
            if (TY == T_IQ2XXS) {
                const uint8_t* p = wrow + (size_t)(s >> 3) * 66;
                qv[i] = *(const uint64_t*)(p + (s & 7) * 8);
                dv[i] = *(const uint16_t*)p;
            } else if (TY == T_IQ2S) {
                const uint8_t* p = wrow + (size_t)(s >> 3) * 82;
                qv[i] = *(const uint64_t*)(p + 2 + (s & 7) * 4);
                av[i] = *(const uint32_t*)(p + 34 + (s & 7) * 4);
                dv[i] = *(const uint16_t*)p;
                pv[i] = (uint64_t)p[66 + (s & 7)] | ((uint64_t)p[74 + (s & 7)] << 8);
            } else {
                const uint8_t* p = wrow + (size_t)(s >> 3) * 98;
                qv[i] = *(const uint64_t*)(p + 2 + (s & 7) * 8);
                av[i] = *(const uint32_t*)(p + 66 + (s & 7) * 4);
                dv[i] = *(const uint16_t*)p;
            }
        }
#pragma unroll
        for (int i = 0; i < U; ++i) {
            const int s = s0 + base + i;
            const float d = __half2float(*(const __half*)&dv[i]);
            const float xsc = xs[s];
            if (TY == T_IQ2XXS) {
                uint32_t a0 = (uint32_t)qv[i], a1 = (uint32_t)(qv[i] >> 32);
                float db = d * (0.5f + (float)(a1 >> 28)) * 0.25f * xsc;
                int sA = 0, sB = 0;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    uint64_t mm = ssig[(a1 >> (7*l)) & 127];
                    uint64_t sq = (sg[(a0 >> (8*l)) & 255] ^ mm) + (mm & 0x0101010101010101ull);
                    const int* xp = (const int*)(xq + (size_t)s * 36) + 2*l;
                    sA = __dp4a((int)sq, xp[0], sA);
                    sB = __dp4a((int)(sq >> 32), xp[1], sB);
                }
                acc = fmaf((float)(sA + sB), db, acc);
            } else if (TY == T_IQ2S) {
                uint32_t qs32 = (uint32_t)qv[i], sgn = av[i];
                int qh = (int)(pv[i] & 0xff), sc = (int)(pv[i] >> 8);
                float db0 = d * (0.5f + (float)(sc & 0xf)) * 0.25f * xsc;
                float db1 = d * (0.5f + (float)(sc >> 4)) * 0.25f * xsc;
                int s0v = 0, s1v = 0;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    int gi = (int)((qs32 >> (8*l)) & 0xff) | ((qh << (8 - 2*l)) & 0x300);
                    uint64_t mm = ssig[(sgn >> (8*l)) & 0xff];
                    uint64_t sq = (sg[gi] ^ mm) + (mm & 0x0101010101010101ull);
                    const int* xp = (const int*)(xq + (size_t)s * 36) + 2*l;
                    int t = __dp4a((int)sq, xp[0], __dp4a((int)(sq >> 32), xp[1], 0));
                    if (l < 2) s0v += t; else s1v += t;
                }
                acc = fmaf((float)s0v, db0, fmaf((float)s1v, db1, acc));
            } else {
                uint64_t q3 = qv[i]; uint32_t aux = av[i];
                float db = d * (0.5f + (float)(aux >> 28)) * 0.5f * xsc;
                int sA = 0, sB = 0;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    uint64_t mm = ssig[(aux >> (7*l)) & 127];
                    uint32_t mlo = (uint32_t)mm, mhi = (uint32_t)(mm >> 32);
                    uint32_t g1 = (uint32_t)sg[(q3 >> (8*(2*l))) & 255];
                    uint32_t g2 = (uint32_t)sg[(q3 >> (8*(2*l+1))) & 255];
                    int w0 = (int)((g1 ^ mlo) + (mlo & 0x01010101u));
                    int w1 = (int)((g2 ^ mhi) + (mhi & 0x01010101u));
                    const int* xp = (const int*)(xq + (size_t)s * 36) + 2*l;
                    sA = __dp4a(w0, xp[0], sA);
                    sB = __dp4a(w1, xp[1], sB);
                }
                acc = fmaf((float)(sA + sB), db, acc);
            }
        }
    }
    red[warp][lane] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = 0.f;
#pragma unroll
        for (int w2 = 0; w2 < WPB; ++w2) v += red[w2][tid];
        y[(size_t)rt * 32 + tid] = v;
    }
}
