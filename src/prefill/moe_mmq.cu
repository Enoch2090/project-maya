// src/prefill/moe_mmq.cu - see include/strata/prefill/moe_mmq.hpp.  llama.cpp's MMQ (ggml-cuda, MIT) is compiled
// from the pinned llama.cpp checkout the build already takes ggml from; src/prefill/ggml_cuda_host.cu supplies the
// few host symbols of ggml-cuda.cu it references.
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/failure.hpp"

#include "common.cuh"
#include "convert.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

#include <atomic>   // glm-dsa B15.2: the refused products' count
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

namespace strata::prefill::mmq {
namespace {

// A failed launch (an allocation, most of the time) ends this request, not the process: the reason is thrown and
// the caller turns it into an error for the request (include/strata/prefill/failure.hpp).
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill mmq: %s: %s\n", what, cudaGetErrorString(e));
        strata::prefill::fail(std::string("prefill mmq: ") + what + ": " + cudaGetErrorString(e));
    }
}

int64_t pad512(int64_t n) { return (n + 511) / 512 * 512; }

__global__ void copy16_kernel(const uint4* __restrict__ a, int64_t na, const uint4* __restrict__ b, int64_t nb,
                              uint4* __restrict__ ab_dst, const uint4* __restrict__ c, int64_t nc, uint4* __restrict__ c_dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < na) ab_dst[i] = a[i];
    else if (i < na + nb) ab_dst[i] = b[i - na];
    else if (i < na + nb + nc) c_dst[i - na - nb] = c[i - na - nb];
}
__global__ void copy1_kernel(const uint8_t* __restrict__ a, int64_t na, const uint8_t* __restrict__ b, int64_t nb,
                             uint8_t* __restrict__ ab_dst, const uint8_t* __restrict__ c, int64_t nc, uint8_t* __restrict__ c_dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < na) ab_dst[i] = a[i];
    else if (i < na + nb) ab_dst[i] = b[i - na];
    else if (i < na + nb + nc) c_dst[i - na - nb] = c[i - na - nb];
}

// Strata blob: gate/up codes [1280][640 B], down codes [2560][160 B], gate/up scales [1280][40] f16, down scales
// [2560][10] f16 (the layout of prefill/kernels.cu's blob_dequant_kernel).  A GGUF Q2_0 block is {f16 d; 16 code
// bytes} with the same 2-bit codes in the same order, so a block is a scale and a 16-byte run of codes.
__global__ void strata_q2_kernel(const uint8_t* __restrict__ blob, uint16_t* __restrict__ gu, uint16_t* __restrict__ dn) {
    constexpr size_t O_D_CODES = (size_t) 1280 * 640, O_GU_SC = O_D_CODES + (size_t) 2560 * 160,
                     O_D_SC = O_GU_SC + (size_t) 1280 * 40 * 2;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one GGUF block
    const int64_t n_gu = 1280LL * 40, n_d = 2560LL * 10;
    const uint8_t* codes;
    const uint16_t* scale;
    uint16_t* out;
    if (i < n_gu) {
        const int64_t row = i / 40, b = i % 40;
        codes = blob + row * 640 + b * 16;
        scale = (const uint16_t*) (blob + O_GU_SC) + row * 40 + b;
        out = gu + i * 9;
    } else if (i < n_gu + n_d) {
        const int64_t j = i - n_gu, row = j / 10, b = j % 10;
        codes = blob + O_D_CODES + row * 160 + b * 16;
        scale = (const uint16_t*) (blob + O_D_SC) + row * 10 + b;
        out = dn + j * 9;
    } else {
        return;
    }
    const uint4 q = *(const uint4*) codes;
    const uint16_t* qh = (const uint16_t*) &q;
    out[0] = *scale;
#pragma unroll
    for (int k = 0; k < 8; ++k) out[1 + k] = qh[k];
}

__global__ void swiglu_kernel(const float* __restrict__ gu, float* __restrict__ h, int64_t rows, int64_t n_ff,
                              bool interleaved) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff, k = i % n_ff;
    const float* row = gu + r * 2 * n_ff;
    const float g = interleaved ? row[2 * k] : row[k], u = interleaved ? row[2 * k + 1] : row[n_ff + k];
    h[i] = g / (1.0f + __expf(-g)) * u;
}

__global__ void iota_kernel(int32_t* dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (int32_t) i;
}

unsigned blocks(int64_t n) { return (unsigned) ((n + 255) / 256); }

}  // namespace

bool built() { return true; }

bool supported(int t) {
    switch ((ggml_type) t) {
        case GGML_TYPE_Q2_0: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q8_0:   // the draft layer's dense matrices (E-9)
        case GGML_TYPE_IQ1_S:  // GLM-5.3-Flash's IQ1_S experts
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
            return true;   // (the Maya quants' candidates, docs/QUANT-PLAN.md; IQ1_M has no MMQ)
        default:
            return false;
    }
}

size_t matrix_bytes(int t, int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) (cols / ggml_blck_size((ggml_type) t)) * ggml_type_size((ggml_type) t);
}

size_t q8_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) pad512(cols) * sizeof(block_q8_1_mmq) / (4 * QK8_1) + 128 * sizeof(block_q8_1_mmq);
}

size_t q8_block_bytes() { return sizeof(block_q8_1_mmq); }

void quantize(const float* x, const int32_t* ids, void* xq, int t, int64_t cols, int64_t ld, int64_t rows, void* stream) {
    if (rows <= 0) return;
    quantize_mmq_q8_1_cuda(x, ids, xq, (ggml_type) t, cols, ld, rows * ld, rows * ld, pad512(cols), rows, 1, 1,
                           (cudaStream_t) stream);
    ck(cudaGetLastError(), "quantize");
}

Context::Context() {
    int dev = 0;
    cudaGetDevice(&dev);
    ctx_ = new ggml_backend_cuda_context(dev);
}
Context::~Context() { delete (ggml_backend_cuda_context*) ctx_; }

namespace {

void dispatch(ggml_backend_cuda_context& ctx, const mmq_args& a, cudaStream_t s) {
    switch (a.type_x) {
        case GGML_TYPE_Q2_0: mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, a, s); break;
        case GGML_TYPE_IQ2_XXS: mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, a, s); break;
        case GGML_TYPE_IQ2_XS: mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, a, s); break;
        case GGML_TYPE_IQ2_S: mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, a, s); break;
        case GGML_TYPE_IQ3_XXS: mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, a, s); break;
        case GGML_TYPE_IQ3_S: mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, a, s); break;
        case GGML_TYPE_IQ4_NL: mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, a, s); break;
        case GGML_TYPE_IQ4_XS: mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, a, s); break;
        case GGML_TYPE_Q8_0: mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, a, s); break;
        case GGML_TYPE_IQ1_S: mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, a, s); break;
        case GGML_TYPE_Q5_0: mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, a, s); break;   // (the Qwen path only: prefill.cpp)
        case GGML_TYPE_Q2_K: mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, a, s); break;
        case GGML_TYPE_Q3_K: mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, a, s); break;
        case GGML_TYPE_Q4_K: mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, a, s); break;
        case GGML_TYPE_Q5_K: mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, a, s); break;
        case GGML_TYPE_Q6_K: mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, a, s); break;
        default:
            std::fprintf(stderr, "prefill mmq: type %d is not covered\n", (int) a.type_x);
            std::exit(1);
    }
    ck(cudaGetLastError(), "mul_mat_q");
}

}  // namespace

// glm-dsa B15.2: MMQ finds expert e at w + e * (expert_bytes / type size) WHOLE blocks (stride_channel_x), so a
// stride that is not a whole number of blocks would be truncated and every expert after the first parsed off its
// block boundaries.  stride_refused() refuses such a product without a launch and counts it (launch_errors(): the
// GLM prompt path fails on it).  Every Maya caller's stride is whole blocks, so this never fires on Flash.
static std::atomic<int> g_refused{0};
int launch_errors() { return g_refused.load(std::memory_order_relaxed); }
static bool stride_refused(const Product& p) {
    const ggml_type t = (ggml_type) p.type;
    if (!supported(p.type) || p.expert_bytes % ggml_type_size(t) == 0) return false;
    if (g_refused.fetch_add(1, std::memory_order_relaxed) < 8)
        std::fprintf(stderr, "prefill mmq: type %d: expert stride %zu B is not whole %zu-B blocks (refused)\n", (int) t,
                     p.expert_bytes, ggml_type_size(t));
    return true;
}

void Context::run(const Product& p, void* stream) {
    if (p.n <= 0 || p.max_rows <= 0) return;
    if (stride_refused(p)) return;   // glm-dsa B15.2
    const ggml_type t = (ggml_type) p.type;
    const int64_t qk = ggml_blck_size(t), bpr = p.w_cols / qk;
    const mmq_args a = {(const char*) p.w, t, (const int*) p.xq, p.ids, p.bounds, p.dst, nullptr,
                        p.w_cols, p.w_rows, p.total_rows, bpr, p.total_rows, p.ld_dst,
                        p.n, p.n, (int64_t) (p.expert_bytes / ggml_type_size(t)), 0, 0,
                        1, 1, 0, 0, 0,
                        p.max_rows, p.max_rows};
    dispatch(*(ggml_backend_cuda_context*) ctx_, a, (cudaStream_t) stream);
}

// STRATA_MMQ_GLU: the build compiles MMQ from a copy of ggml-cuda with third_party/ggml/mmq-fused-swiglu.patch
bool glu_supported(int t, int64_t n_ff) {
#if defined(STRATA_MMQ_GLU)
    if (!ggml_cuda_mmq_fusion_supported((ggml_type) t) || n_ff <= 0 || n_ff % 128 != 0) return false;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    return turing_mma_available(ggml_cuda_info().devices[dev].cc);
#else
    (void) t;
    (void) n_ff;
    return false;
#endif
}

void Context::run_glu(const Product& p, float limit, void* stream) {
#if defined(STRATA_MMQ_GLU)
    if (p.n <= 0 || p.max_rows <= 0) return;
    if (stride_refused(p)) return;   // glm-dsa B15.2
    const ggml_type t = (ggml_type) p.type;
    const int64_t qk = ggml_blck_size(t), bpr = p.w_cols / qk, n_ff = p.w_rows / 2;
    // x the up rows, x_gate the gate rows of each expert's matrix; the output has n_ff rows
    mmq_args a = {(const char*) p.w + (size_t) n_ff * (size_t) bpr * ggml_type_size(t), t, (const int*) p.xq, p.ids,
                  p.bounds, p.dst, nullptr,
                  p.w_cols, n_ff, p.total_rows, bpr, p.total_rows, p.ld_dst,
                  p.n, p.n, (int64_t) (p.expert_bytes / ggml_type_size(t)), 0, 0,
                  1, 1, 0, 0, 0,
                  p.max_rows, p.max_rows};
    a.x_gate = (const char*) p.w;
    a.glu_limit = limit;
    dispatch(*(ggml_backend_cuda_context*) ctx_, a, (cudaStream_t) stream);
#else
    (void) p;
    (void) limit;
    (void) stream;
    std::fprintf(stderr, "prefill mmq: this build has no fused gate/up product\n");
    std::exit(1);
#endif
}

void gather_native(const void* gate, const void* up, size_t gu_half_bytes, const void* down, size_t d_bytes,
                   void* gu_dst, void* d_dst, void* stream) {
    const cudaStream_t s = (cudaStream_t) stream;
    const bool a16 = ((uintptr_t) gate | (uintptr_t) up | (uintptr_t) down | (uintptr_t) gu_dst | (uintptr_t) d_dst |
                      gu_half_bytes | d_bytes) % 16 == 0;
    if (a16) {
        const int64_t na = (int64_t) gu_half_bytes / 16, nc = (int64_t) d_bytes / 16;
        copy16_kernel<<<blocks(2 * na + nc), 256, 0, s>>>((const uint4*) gate, na, (const uint4*) up, na, (uint4*) gu_dst,
                                                          (const uint4*) down, nc, (uint4*) d_dst);
    } else {
        const int64_t na = (int64_t) gu_half_bytes, nc = (int64_t) d_bytes;
        copy1_kernel<<<blocks(2 * na + nc), 256, 0, s>>>((const uint8_t*) gate, na, (const uint8_t*) up, na,
                                                         (uint8_t*) gu_dst, (const uint8_t*) down, nc, (uint8_t*) d_dst);
    }
    ck(cudaGetLastError(), "gather_native");
}

void gather_strata_q2(const uint8_t* blob, void* gu_dst, void* d_dst, void* stream) {
    strata_q2_kernel<<<blocks(1280LL * 40 + 2560LL * 10), 256, 0, (cudaStream_t) stream>>>(blob, (uint16_t*) gu_dst,
                                                                                         (uint16_t*) d_dst);
    ck(cudaGetLastError(), "gather_strata_q2");
}

void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream) {
    if (rows <= 0) return;
    swiglu_kernel<<<blocks(rows * n_ff), 256, 0, (cudaStream_t) stream>>>(gu, h, rows, n_ff, interleaved);
    ck(cudaGetLastError(), "swiglu");
}

void iota(int32_t* dst, int64_t n, void* stream) {
    if (n <= 0) return;
    iota_kernel<<<blocks(n), 256, 0, (cudaStream_t) stream>>>(dst, n);
    ck(cudaGetLastError(), "iota");
}

bool to_f16_supported(int t) { return ggml_get_to_fp16_cuda((ggml_type) t) != nullptr; }

// ---- fast IQ2_XXS / IQ2_S dequantization --------------------------------------------------------------
// ggml's dequantize_row_iq2_xxs launches one 32-thread block a 256-value superblock: 512 bytes of output a block,
// each thread's 8 halves going out as four 32-bit stores.  Measured on an MTT S4000 (research/r4-attn/bench_deq2.mu)
// a 32 MB expert dequantizes at 118 GB/s written, against 366 GB/s the same card reaches on a read+write mix and
// 556 GB/s on stores alone; a superblock a warp with one 128-bit store a thread reaches 211 GB/s (1.79x).
// The arithmetic is ggml's, unchanged - only which thread writes what, and how wide it writes, differs.
// MAYA_GLM_DEQ=0 restores ggml's kernel (A/B); MAYA_GLM_DEQ_CHECK=1 compares the two element for element.
namespace {

// R24: the 2 KB grid and the 128 B sign table are read at DATA-DEPENDENT indices (aux8[il] picks one of 256
// entries), so a 32-lane warp issues up to 32 unrelated 8-byte requests at a table that is L1-resident but
// never coalesced.  At the engine's own geometry (one expert's gate/up, random blob so the indices diverge)
// the body writes at 199 GB/s where the store ceiling at that geometry is 509; dropping ONLY the grid
// gather recovers 1.24x and dropping only the signs 1.07x (research/r24/deq_why.mu).  Staging both tables
// in shared memory - one coalesced load a block - makes the same divergent access hit smem instead:
// 0.168 -> 0.103 ms an expert = 1.63x, output byte for byte identical (research/r24/deq_smem2.mu).
// The tables are unchanged; only where they live differs.  MAYA_GLM_DEQ_SMEM=0 keeps the global-table body.
// IQ2_XXS: a warp a superblock, each thread's 8 halves stored as one 128-bit write
// (grid_t/sign_t are raw byte tables: entry i starts at grid_t + 8 * i, as (iq2xxs_grid + i) did)
__device__ __forceinline__ void deq_iq2xxs_t(const void* vx, int64_t ibs, __half* yy,
                                             const uint8_t* __restrict__ grid_t, const uint8_t* __restrict__ sign_t) {
    const block_iq2_xxs* x = (const block_iq2_xxs*) vx;
    const int tid = threadIdx.x & 31;
    const int il = tid / 8, ib = tid % 8;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*) q2;
    const uint8_t* grid = grid_t + (size_t) aux8[il] * 8;
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = sign_t[(aux32 >> 7 * il) & 127];
    __half2 h[4];
#pragma unroll
    for (int j = 0; j < 8; j += 2)
        h[j / 2] = __floats2half2_rn(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f),
                                     d * grid[j + 1] * (signs & kmask_iq2xs[j + 1] ? -1.f : 1.f));
    *(uint4*) (yy + (size_t) ibs * QK_K + 32 * ib + 8 * il) = *(const uint4*) h;
}
__device__ __forceinline__ void deq_iq2xxs_w(const void* vx, int64_t ibs, __half* yy) {
    deq_iq2xxs_t(vx, ibs, yy, (const uint8_t*) iq2xxs_grid, ksigns_iq2xs);
}

// R36(w1): the sign application in deq_iq2xxs_t is eight `signs & kmask_iq2xs[j] ? -1.f : 1.f` selects
// sitting in the middle of the float multiply chain.  A select on a float then a multiply is a dependency;
// IEEE negation, by contrast, commutes with rounding - so flipping the sign bit of the PACKED halves after
// the conversion is the same number, bit for bit.  This table holds what those eight selects would have
// produced, pre-XORed into four uint32 (one a __half2): 128 entries x 16 B = 2 KB in shared memory, staged
// next to the grid and the sign table the block already builds.  It is a new helper, so deq_iq2xxs_t is
// untouched and DEQ_CHECK still compares the two element for element.
__device__ __forceinline__ void deq_iq2xxs_xor_t(const void* vx, int64_t ibs, __half* yy,
                                                 const uint8_t* __restrict__ grid_t,
                                                 const uint4* __restrict__ mask_t) {
    const block_iq2_xxs* x = (const block_iq2_xxs*) vx;
    const int tid = threadIdx.x & 31;
    const int il = tid / 8, ib = tid % 8;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*) q2;
    const uint8_t* grid = grid_t + (size_t) aux8[il] * 8;
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint4 m = mask_t[(aux32 >> 7 * il) & 127];
    const uint32_t* mm = (const uint32_t*) &m;
    __half2 h[4];
#pragma unroll
    for (int j = 0; j < 8; j += 2)
        h[j / 2] = __floats2half2_rn(d * grid[j], d * grid[j + 1]);
#pragma unroll
    for (int j = 0; j < 4; ++j) ((uint32_t*) h)[j] ^= mm[j];
    *(uint4*) (yy + (size_t) ibs * QK_K + 32 * ib + 8 * il) = *(const uint4*) h;
}
// the table the block stages: entry i is ksigns_iq2xs[i]'s eight bits as four half2 sign masks
// (__floats2half2_rn(a, b) puts a in the low half - bit 15 - and b in the high half - bit 31)
__device__ __forceinline__ uint4 iq2xxs_sign_mask(int i) {
    const uint8_t s = ksigns_iq2xs[i];
    uint32_t m[4] = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int j = 0; j < 8; ++j)
        if (s & kmask_iq2xs[j]) m[j / 2] |= (j & 1) ? 0x80000000u : 0x00008000u;
    return make_uint4(m[0], m[1], m[2], m[3]);
}

// IQ2_S: the same shape (the down projections of the iq2_s layers), same shared-table treatment
__device__ __forceinline__ void deq_iq2s_t(const void* vx, int64_t ibs, __half* yy,
                                           const uint8_t* __restrict__ grid_t) {
    const block_iq2_s* x = (const block_iq2_s*) vx;
    const int tid = threadIdx.x & 31;
    const int il = tid / 8, ib = tid % 8;
    const uint8_t* grid = grid_t + (size_t) (x[ibs].qs[4 * ib + il] |
                                             ((x[ibs].qh[ib] << (8 - 2 * il)) & 0x300)) * 8;
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K / 8 + 4 * ib + il];
    __half2 h[4];
#pragma unroll
    for (int j = 0; j < 8; j += 2)
        h[j / 2] = __floats2half2_rn(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f),
                                     d * grid[j + 1] * (signs & kmask_iq2xs[j + 1] ? -1.f : 1.f));
    *(uint4*) (yy + (size_t) ibs * QK_K + 32 * ib + 8 * il) = *(const uint4*) h;
}
__device__ __forceinline__ void deq_iq2s_w(const void* vx, int64_t ibs, __half* yy) {
    deq_iq2s_t(vx, ibs, yy, (const uint8_t*) iq2s_grid);
}

bool deq_smem_on() {
    static const bool on = [] {
        const char* v = getenv("MAYA_GLM_DEQ_SMEM");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

// eight superblocks a block, one warp each (the pre-R24 body: the tables read from global)
__global__ void deq_fast_kernel(int type, const void* vx, __half* yy, int64_t nsb) {
    const int64_t sb = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    if (sb >= nsb) return;
    if (type == GGML_TYPE_IQ2_XXS) deq_iq2xxs_w(vx, sb, yy);
    else deq_iq2s_w(vx, sb, yy);
}

// R27: the down matrix (IQ2_S) wants a different superblocks-per-block than gate/up.  Its table is 4x the
// gate/up one (8 KB against 2 KB) and each block re-copies it, so at the engine's 8 a block pays 33.6 MB of
// table traffic to write 33.6 MB of output (gate/up: 17.8 MB for 67 MB); per superblock it costs 2.32 ns
// against gate/up's 1.51.  Measured in research/r27/deq_dn.mu at the engine's own geometry (4 experts a
// batch, R24's smem bodies verbatim): down 8 -> 0.076, 16 -> 0.064, 32 -> 0.064 ms an expert; gate/up is
// flat 8 -> 16 (0.099 -> 0.097) and worse at 32 (0.103).  32 halves the table copies and is where down's
// curve bottoms.  MAYA_GLM_DEQ_SPB_DN overrides it (8 restores the pre-R27 behaviour, for A/B).
__global__ void deq_fast_smem_kernel(int type, const void* vx, __half* yy, int64_t nsb, int spb_dn) {
    if (type == GGML_TYPE_IQ2_XXS) {
        __shared__ __align__(8) uint64_t sg[256];    // (the table's own element type: an 8-byte load a thread)
        __shared__ uint8_t ss[128];
        for (int i = (int) threadIdx.x; i < 256; i += (int) blockDim.x) sg[i] = iq2xxs_grid[i];
        if (threadIdx.x < 128) ss[threadIdx.x] = ksigns_iq2xs[threadIdx.x];
        __syncthreads();
        const int64_t sb = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
        if (sb >= nsb) return;
        deq_iq2xxs_t(vx, sb, yy, (const uint8_t*) sg, (const uint8_t*) ss);
    } else {
        __shared__ __align__(8) uint64_t sg[1024];
        for (int i = (int) threadIdx.x; i < 1024; i += (int) blockDim.x) sg[i] = iq2s_grid[i];
        __syncthreads();
        const int64_t sb = (int64_t) blockIdx.x * spb_dn + (threadIdx.x >> 5);
        if (sb >= nsb) return;
        deq_iq2s_t(vx, sb, yy, (const uint8_t*) sg);
    }
}

// R27: superblocks a block for the down matrix (IQ2_S).  8 is the pre-R27 value; 32 measured best.
int deq_spb_dn() {
    static const int n = [] {
        const char* v = getenv("MAYA_GLM_DEQ_SPB_DN");
        const int x = v != nullptr ? std::atoi(v) : 0;
        return (x == 8 || x == 16 || x == 32) ? x : 32;
    }();
    return n;
}

bool deq_fast_on() {
    static const bool on = [] {
        const char* v = getenv("MAYA_GLM_DEQ");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

// R36(w1): the one-expert launch is a forward declaration; its body lives after deq_multi_ssrc_kernel.
// R36(w1): where do the dequant's bytes actually go?  A null result has to be attributable, so the four
// arms (multi/single x select/xor) each accumulate the output bytes they hand back, and MAYA_GLM_DEQ_ARM_STATS=1
// prints the totals at exit.  Without this, "the kernel got 1.16x faster and nothing moved" is unfalsifiable.
static long long g_arm_bytes[4] = {0, 0, 0, 0};   // index = (multi ? 2 : 0) | (xor ? 1 : 0)
static std::atomic<bool> g_arm_hooked{false};
inline void arm_bytes(int multi, int xs, int64_t n) {
    const int i = (multi ? 2 : 0) | (xs ? 1 : 0);
    __atomic_fetch_add(&g_arm_bytes[i], (long long) n * 2, __ATOMIC_RELAXED);
    static const bool on = getenv("MAYA_GLM_DEQ_ARM_STATS") != nullptr;
    if (on && !g_arm_hooked.exchange(true))
        std::atexit([] {
            const char* nm[4] = {"single/select", "single/xor", "multi/select", "multi/xor"};
            for (int i = 0; i < 4; ++i)
                if (g_arm_bytes[i])
                    std::fprintf(stderr, "prefill mmq ARM_STATS %-14s %.1f GB\n", nm[i], (double) g_arm_bytes[i] / 1e9);
        });
}

bool deq_one_ssrc(int t, const void* w, uint16_t* dst, int64_t nsb, cudaStream_t s);

bool to_f16_fast(int t, const void* w, uint16_t* dst, int64_t n, cudaStream_t s) {
    if (t != GGML_TYPE_IQ2_XXS && t != GGML_TYPE_IQ2_S) return false;
    if (!deq_fast_on()) return false;
    const int64_t nsb = n / QK_K;   // (the packs' matrices are whole superblocks)
    if (nsb == 0 || nsb * QK_K != n) return false;
    // R36(w1): the ONE-expert launch takes R35's shared-source body too.  R33 put the batched launch in and
    // R35 made its source shared, but the engine's batches are mostly nb = 1-3 (MAYA_GLM_PREFILL_BATCH_STATS=1
    // on the delivery config: layer 3 alone had nb = 3,2,1,1,2,1,2,1,1,1,1,..., 41 experts in 13 batches), so
    // most expert matrices never reached to_f16_multi at all and kept the un-staged source - which is why
    // R36's sign-XOR arm moved its kernel 1.16x and res_deq only 0.3%: it was optimising a minority path.
    if (deq_one_ssrc(t, w, dst, nsb, s)) return true;
    const int spb_dn = deq_spb_dn();
    if (deq_smem_on()) {
        const int spb = t == GGML_TYPE_IQ2_S ? spb_dn : 8;
        deq_fast_smem_kernel<<<(unsigned) ((nsb + spb - 1) / spb), (unsigned) (spb * 32), 0, s>>>(t, w, (half*) dst, nsb,
                                                                                                 spb_dn);
    } else {
        deq_fast_kernel<<<(unsigned) ((nsb + 7) / 8), 256, 0, s>>>(t, w, (half*) dst, nsb);
    }
    ck(cudaGetLastError(), "to_f16 fast");
    return true;
}

// MAYA_GLM_DEQ_CHECK=1: both kernels over one blob, compared element for element (the fast path's only licence)
// R24: the licence is per TYPE, not per process.  With a single call only the gate/up matrix (IQ2_XXS, whichever
// layer ran first) was licensed and the DOWN matrix (IQ2_S - ggml's enum value 22, the other half of the traffic)
// never was, so a change to the shared table staging could pass the check while breaking every down product.
// Measured with this: IQ2_XXS 0 bad of 16777216, IQ2_S 0 bad of 8388608.
void deq_check(int t, const void* w, int64_t n, cudaStream_t s) {
    static const bool on = getenv("MAYA_GLM_DEQ_CHECK") != nullptr;
    // keyed by the TYPE, not by a family: the fast path covers IQ2_XXS and IQ2_S and nothing else, so those two
    // are the licences that matter (an IQ3 type reaching here is ggml's own kernel and proves nothing about ours)
    static bool licensed_gu = false, licensed_dn = false;   // (IQ2_XXS gate/up, IQ2_S down)
    bool& lic = (t == GGML_TYPE_IQ2_XXS) ? licensed_gu : licensed_dn;
    if (t != GGML_TYPE_IQ2_XXS && t != GGML_TYPE_IQ2_S) return;   // (not our kernel)
    if (!on || lic) return;
    lic = true;
    const to_fp16_cuda_t f = ggml_get_to_fp16_cuda((ggml_type) t);
    if (f == nullptr || n % QK_K != 0) return;
    half* a = nullptr;
    half* b = nullptr;
    if (cudaMalloc(&a, (size_t) n * 2) != cudaSuccess || cudaMalloc(&b, (size_t) n * 2) != cudaSuccess) {
        cudaGetLastError();
        if (a) cudaFree(a);
        if (b) cudaFree(b);
        return;
    }
    f(w, a, n, s);
    if (deq_smem_on()) {
        const int spb = t == GGML_TYPE_IQ2_S ? deq_spb_dn() : 8;
        deq_fast_smem_kernel<<<(unsigned) (((n / QK_K) + spb - 1) / spb), (unsigned) (spb * 32), 0, s>>>(
            t, w, (half*) b, n / QK_K, deq_spb_dn());
    } else {
        deq_fast_kernel<<<(unsigned) (((n / QK_K) + 7) / 8), 256, 0, s>>>(t, w, (half*) b, n / QK_K);
    }
    std::vector<half> A((size_t) n), B((size_t) n);
    cudaStreamSynchronize(s);
    cudaMemcpy(A.data(), a, (size_t) n * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(B.data(), b, (size_t) n * 2, cudaMemcpyDeviceToHost);
    int64_t bad = 0;
    float worst = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        const float x = __half2float(A[(size_t) i]), y = __half2float(B[(size_t) i]);
        if (x != y) {
            ++bad;
            worst = std::max(worst, std::fabs(x - y));
        }
    }
    std::fprintf(stderr, "prefill mmq DEQ_CHECK type %d n %lld: %lld bad of %lld (worst %.6g)\n", t, (long long) n,
                 (long long) bad, (long long) n, worst);
    cudaFree(a);
    cudaFree(b);
}

}  // namespace

// R33: ONE launch for a whole batch's experts.  The engine dequantizes a batch of `nb` experts with TWO
// launches EACH (glm_prefill.cu: mmq::to_f16(gu) then mmq::to_f16(dn) inside `for k < nb`), so a 16-expert
// batch is 32 launches.  Measured on this card with the real IQ2_XXS body and a 1.5 GB non-resident source
// (research/r33/w1_launch.mu): 1 expert a launch 279.2 GB/s, 16 a launch 320.2 GB/s - the launch count is
// worth 1.15x on the dequant, which is 13.5% of prefill's work.
//
// The batch's sources are not contiguous (the batch holds whatever experts the chunk routed), but its
// DESTINATIONS are (b.wgu + k * gu_elems), so only the source pointers travel: nb <= 32 of them, by value.
// Every expert is nsb superblocks and nsb % spb == 0 for both matrices (65536 / 8, 32768 / 32), so one block
// never straddles two experts and each block still stages the tables in shared memory exactly as before.
struct DeqSrcs { const void* p[32]; };


// R35(w1): the source bytes staged in shared memory too, not just the tables.
//
// R24 put the 2 KB grid and the 128 B sign table in shared memory and measured 1.63x, because their indices
// are data-dependent and diverge.  The SOURCE granules have the same problem and were never treated: a warp
// dequantizing superblock sb has each of 32 lanes read x[sb].qs + 4*(lane%8), i.e. 8 distinct 8-byte granules
// requested 4x each, and the whole 66-byte superblock sits in a different granule of the source.  Measured on
// this card over the engine's own per-batch output (16 experts, 805.3 MB of FP16, research/r35/w1_deq4.mu,
// 7 reps, the same body and grid both arms):
//     reference (engine body, tables in shared)   318.8 GB/s written
//     + the source staged in shared, SBS=8        413.5
//     + SBS=16, 256 threads                       419.6
//     + SBS=16, 512 threads                       426.1   = 1.337x
// and the ceilings from w1_wbw/w1_deq2: a pure store of the same bytes is 682-697 GB/s, and the same store
// with only the block header read is 439 - so this change recovers most of the read cost, not the tail.
//
// One warp owns SBS contiguous superblocks; the block fetches them with 16-byte coalesced loads and the
// decode then reads shared memory where the 4x redundancy costs nothing.  The arithmetic is byte-identical
// (deq_iq2xxs_t/deq_iq2s_t are untouched; only the pointer they read from differs), so MAYA_GLM_DEQ_SMEM_SRC=0
// restores the global-source body for an A/B, and DEQ_CHECK still compares element for element.
template <int SBS, bool XORSIGN = false>
__global__ void deq_multi_ssrc_kernel(int type, DeqSrcs srcs, __half* dst, int64_t nsb, int spb) {
    const int k = (int) blockIdx.y;
    const int warps = (int) (blockDim.x >> 5);
    const int w = (int) (threadIdx.x >> 5);
    const int tid = (int) (threadIdx.x & 31);
    // warp w owns the run [sb0, sb0 + SBS)
    const int64_t sb0 = (int64_t) blockIdx.x * (int64_t) (warps * SBS) + (int64_t) w * SBS;
    __half* out = dst + (size_t) k * (size_t) nsb * QK_K;
    const size_t bytes = (size_t) SBS * (type == GGML_TYPE_IQ2_XXS ? sizeof(block_iq2_xxs) : sizeof(block_iq2_s));
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* stg = smem;                                    // warps * bytes
    uint64_t* sg = (uint64_t*) (smem + (size_t) warps * bytes);    // the tables after it
    uint8_t* ss = (uint8_t*) (sg + (type == GGML_TYPE_IQ2_XXS ? 256 : 1024));
    // R36(w1): the sign XOR's mask table rides after the sign bytes (2 KB, only when the arm is on)
    uint4* smask = XORSIGN ? (uint4*) (ss + 128) : nullptr;
    const unsigned char* xb = (const unsigned char*) srcs.p[k];
    // coalesced 16-byte fetch of this warp's run (the last granule may be partial; SBS is small so it is exact)
    const size_t words = (bytes + 15) / 16;
    const size_t blk = bytes / (size_t) SBS;                 // sizeof(block_iq2_*)
    const unsigned char* srun = xb + (size_t) sb0 * blk;
    unsigned char* drun = stg + (size_t) w * bytes;
    for (size_t i = (size_t) tid; i < words; i += 32) {
        const size_t off = i * 16;
        const size_t n = bytes - off < 16 ? bytes - off : 16;
        if (n == 16) *(uint4*) (drun + off) = *(const uint4*) (srun + off);
        else for (size_t b = 0; b < n; ++b) drun[off + b] = srun[off + b];
    }
    if (type == GGML_TYPE_IQ2_XXS) {
        for (int i = (int) threadIdx.x; i < 256; i += (int) blockDim.x) sg[i] = iq2xxs_grid[i];
        if (threadIdx.x < 128) ss[threadIdx.x] = ksigns_iq2xs[threadIdx.x];
        if (XORSIGN)
            for (int i = (int) threadIdx.x; i < 128; i += (int) blockDim.x) smask[i] = iq2xxs_sign_mask(i);
    } else {
        for (int i = (int) threadIdx.x; i < 1024; i += (int) blockDim.x) sg[i] = iq2s_grid[i];
    }
    __syncthreads();
    const size_t szb = blk;
    for (int s = 0; s < SBS; ++s) {
        const int64_t sb = sb0 + s;
        if (sb >= nsb) break;
        const void* sr = drun + (size_t) s * (size_t) szb;
        // The helpers use `ibs` for BOTH the source index (x[ibs]) and the destination offset (ibs*QK_K).
        // Here the source is the staged block s and the destination is absolute superblock sb, so the helpers
        // are called with ibs = 0 and an absolute `out` - one block each, so 0 addresses exactly block s of sr.
        if (type == GGML_TYPE_IQ2_XXS && XORSIGN)
            deq_iq2xxs_xor_t(sr, 0, out + (size_t) sb * QK_K, (const uint8_t*) sg, smask);
        else if (type == GGML_TYPE_IQ2_XXS) deq_iq2xxs_t(sr, 0, out + (size_t) sb * QK_K, (const uint8_t*) sg, ss);
        else deq_iq2s_t(sr, 0, out + (size_t) sb * QK_K, (const uint8_t*) sg);
    }
}

// R36(w1): one expert through the shared-source body (R35's deq_multi_ssrc_kernel at count = 1).
// MAYA_GLM_DEQ_SMEM_SRC1=0 restores the old single path for an A/B.  Same grid geometry as the batched
// launch (the 2-D shape R33 measured 9.8x on, here with count = 1), same SBS and block size, so the two
// paths stay one body.
namespace {
bool deq_one_ssrc(int t, const void* w, uint16_t* dst, int64_t nsb, cudaStream_t s) {
    // R36(w1): default OFF (0 = the pre-R36 single path).  Routing the ONE-expert launch through R35's
    // shared-source body measured dead neutral end to end (A/B/A/B: off 845.8 / 845.0, on 844.7 / 845.2)
    // AND dead neutral on res_deq itself (4173.5->4164.2, 4691.4->4694.4, 4452.0->4452.5, 3262.9->3295.4 ms).
    // Same reason as the sign XOR above: the engine's ARM_STATS says the fast bodies carry 404.9 GB a card,
    // 24-29% of what res_deq bills, so moving bytes around inside them is worth <1% of the stream.
    // MAYA_GLM_DEQ_SMEM_SRC1=1 turns it on.
    static const int sbs = [] {
        const char* v = getenv("MAYA_GLM_DEQ_SMEM_SRC1");
        if (v == nullptr || std::atoi(v) == 0) return 0;
        const char* n = getenv("MAYA_GLM_DEQ_SRC_SBS1");
        const int k = n != nullptr ? std::atoi(n) : 16;
        return (k == 8 || k == 16 || k == 32) ? k : 16;
    }();
    if (sbs == 0) return false;
    static const bool xs = [] {
        const char* v = getenv("MAYA_GLM_DEQ_XOR_SIGN");
        return v == nullptr || v[0] == 0 || std::atoi(v) != 0;
    }();
    const bool xor_arm = xs && t == GGML_TYPE_IQ2_XXS;
    const int bs = 512, warps = bs / 32;
    const size_t blk = t == GGML_TYPE_IQ2_XXS ? sizeof(block_iq2_xxs) : sizeof(block_iq2_s);
    size_t smem = (size_t) warps * (size_t) sbs * blk + (size_t) (t == GGML_TYPE_IQ2_XXS ? 256 : 1024) * sizeof(uint64_t) +
                  (t == GGML_TYPE_IQ2_XXS ? 128 : 0) + (xor_arm ? 128 * sizeof(uint4) : 0);
    if (smem > (48u << 10)) return false;
    DeqSrcs srcs{};
    srcs.p[0] = w;
    const int64_t gsb = (int64_t) warps * (int64_t) sbs;
    const dim3 grid((unsigned) ((nsb + gsb - 1) / gsb), 1u);
    const auto launch = [&](auto kern) { kern<<<grid, (unsigned) bs, smem, s>>>(t, srcs, (half*) dst, nsb, 8); };
    // R36(w1) diagnostic: MAYA_GLM_DEQ_ONE_STATS=1 says once per type whether this path is taken and what
    // shape it launches with, so a null result can never be blamed on the path not being reached.
    static const bool one_stats = getenv("MAYA_GLM_DEQ_ONE_STATS") != nullptr;
    if (one_stats) {
        static bool said_gu = false, said_dn = false;
        bool& said = (t == GGML_TYPE_IQ2_XXS) ? said_gu : said_dn;
        if (!said) {
            said = true;
            std::fprintf(stderr, "prefill mmq ONE_SSRC type %d nsb %lld sbs %d xor %d smem %zu -> %lld blocks\n", t,
                         (long long) nsb, sbs, (int) xor_arm, smem,
                         (long long) ((nsb + (int64_t) warps * sbs - 1) / ((int64_t) warps * sbs)));
        }
    }
    if (t == GGML_TYPE_IQ2_XXS) {
        if (xor_arm) {
            if (sbs == 8) launch(deq_multi_ssrc_kernel<8, true>);
            else if (sbs == 32) launch(deq_multi_ssrc_kernel<32, true>);
            else launch(deq_multi_ssrc_kernel<16, true>);
        } else if (sbs == 8) launch(deq_multi_ssrc_kernel<8>);
        else if (sbs == 32) launch(deq_multi_ssrc_kernel<32>);
        else launch(deq_multi_ssrc_kernel<16>);
        ck(cudaGetLastError(), "to_f16 one ssrc");
        arm_bytes(0, (int) xor_arm, nsb * QK_K);
        return true;
    }
    if (nsb % (int64_t) sbs != 0) return false;
    if (sbs == 8) launch(deq_multi_ssrc_kernel<8>);
    else if (sbs == 32) launch(deq_multi_ssrc_kernel<32>);
    else launch(deq_multi_ssrc_kernel<16>);
    ck(cudaGetLastError(), "to_f16 one ssrc");
    return true;
}
}  // namespace

__global__ void deq_multi_smem_kernel(int type, DeqSrcs srcs, __half* dst, int64_t nsb, int spb) {
    const int64_t groups = nsb / (int64_t) spb;
    // The grid is 2-D (groups x count) and nsb % spb == 0, so k < 32 and sb < nsb hold for every thread; the
    // guards are NOT early returns because a return before the table staging below would skip a
    // __syncthreads() the block's other warps still wait on.  (to_f16_multi checks both conditions.)
    //
    // The shape is not cosmetic: launched as ONE flat 1-D grid of groups*count blocks the same code runs at
    // 74 GB/s, and as a 2-D (groups, count) grid at 723 GB/s - 9.8x, from the launch geometry alone
    // (research/r33/w1_srcs.mu, identical blocks, identical bytes; 8192-block grids are also fine, so it is
    // the very long 1-D grid this card dislikes, not the batch).
    const int k = (int) blockIdx.y;
    const int64_t g = (int64_t) blockIdx.x;
    const int64_t sb = g * (int64_t) spb + (threadIdx.x >> 5);
    __half* out = dst + (size_t) k * (size_t) nsb * QK_K;
    (void) groups;
    if (type == GGML_TYPE_IQ2_XXS) {
        __shared__ __align__(8) uint64_t sg[256];
        __shared__ uint8_t ss[128];
        for (int i = (int) threadIdx.x; i < 256; i += (int) blockDim.x) sg[i] = iq2xxs_grid[i];
        if (threadIdx.x < 128) ss[threadIdx.x] = ksigns_iq2xs[threadIdx.x];
        __syncthreads();
        deq_iq2xxs_t(srcs.p[k], sb, out, (const uint8_t*) sg, (const uint8_t*) ss);
    } else {
        __shared__ __align__(8) uint64_t sg[1024];
        for (int i = (int) threadIdx.x; i < 1024; i += (int) blockDim.x) sg[i] = iq2s_grid[i];
        __syncthreads();
        deq_iq2s_t(srcs.p[k], sb, out, (const uint8_t*) sg);
    }
}

// false: the caller keeps the per-expert loop (some type/size the fast path does not cover).
// MAYA_GLM_DEQ_BATCH=0: the per-expert loop (A/B for this change, and the fallback if a shape slips through).
// R36(w1): the sign-XOR arm's licence.  MAYA_GLM_DEQ_CHECK=1 already compares the SINGLE-expert fast path to
// ggml's kernel, and the single path never takes the XOR arm.  This compares the multi (batched) launch's XOR
// output to the same helper's select form over the same bytes, so a change to the mask table cannot pass the
// check while producing different numbers.
void deq_check_multi_xor(int t, const void* const* w, int count, int64_t n, cudaStream_t s) {
    static const bool on = getenv("MAYA_GLM_DEQ_CHECK") != nullptr;
    static bool done = false;
    if (!on || done || t != GGML_TYPE_IQ2_XXS || count < 1) return;
    done = true;
    const int64_t nsb = n / QK_K;
    if (nsb == 0 || nsb * QK_K != n) return;
    half* a = nullptr;
    half* b = nullptr;
    if (cudaMalloc(&a, (size_t) n * 2) != cudaSuccess || cudaMalloc(&b, (size_t) n * 2) != cudaSuccess) {
        cudaGetLastError();
        if (a) cudaFree(a);
        if (b) cudaFree(b);
        return;
    }
    // both arms, one launch each, over the same source and the same layout the engine uses
    const int bs = 512, warps = bs / 32, SBS = 16;
    const int64_t gsb = (int64_t) warps * SBS;
    const dim3 grid2((unsigned) ((nsb + gsb - 1) / gsb), 1u);
    const size_t smem_no = (size_t) warps * SBS * sizeof(block_iq2_xxs) + 256 * sizeof(uint64_t) + 128;
    const size_t smem_xs = smem_no + 128 * sizeof(uint4);
    DeqSrcs srcs{};
    srcs.p[0] = w[0];
    deq_multi_ssrc_kernel<16, false><<<grid2, bs, smem_no, s>>>(t, srcs, (half*) a, nsb, 8);
    deq_multi_ssrc_kernel<16, true><<<grid2, bs, smem_xs, s>>>(t, srcs, (half*) b, nsb, 8);
    std::vector<half> A((size_t) n), B((size_t) n);
    cudaStreamSynchronize(s);
    cudaMemcpy(A.data(), a, (size_t) n * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(B.data(), b, (size_t) n * 2, cudaMemcpyDeviceToHost);
    int64_t bad = 0;
    for (int64_t i = 0; i < n; ++i)
        if (((const uint16_t*) A.data())[(size_t) i] != ((const uint16_t*) B.data())[(size_t) i]) ++bad;
    std::fprintf(stderr, "prefill mmq DEQ_XOR_CHECK n %lld: %lld bad of %lld\n", (long long) n, (long long) bad,
                 (long long) n);
    cudaFree(a);
    cudaFree(b);
}

bool to_f16_multi_on() {
    static const bool on = [] {
        const char* v = getenv("MAYA_GLM_DEQ_BATCH");
        return v == nullptr || v[0] == 0 || std::atoi(v) != 0;
    }();
    return on;
}

// The predicate, separately from the launch: the caller has TWO matrices per expert (gate/up and down) and
// they can have different types - the packs use IQ2_XXS for gate/up and IQ2_S for down, but some layers carry a
// third type for down that only ggml's own kernel covers.  Asking first is what keeps the batch from
// dequantizing one matrix batched and then re-doing BOTH in the fallback loop (measured: 31% slower).
bool to_f16_multi_ok(int t, int64_t n, int count) {
    if (!to_f16_multi_on()) return false;
    if (!deq_fast_on() || !deq_smem_on()) return false;
    if (count < 2 || count > 32) return false;
    if (t != GGML_TYPE_IQ2_XXS && t != GGML_TYPE_IQ2_S) return false;
    const int64_t nsb = n / QK_K;
    if (nsb == 0 || nsb * QK_K != n) return false;
    const int spb = t == GGML_TYPE_IQ2_S ? deq_spb_dn() : 8;
    return nsb % (int64_t) spb == 0;
}

bool to_f16_multi(int t, const void* const* w, uint16_t* dst, int64_t n, int count, void* stream) {
    if (!to_f16_multi_ok(t, n, count)) return false;
    cudaStream_t s = (cudaStream_t) stream;
    const int64_t nsb = n / QK_K;
    const int spb = t == GGML_TYPE_IQ2_S ? deq_spb_dn() : 8;
    DeqSrcs srcs{};
    for (int k = 0; k < count; ++k) {
        if (w[k] == nullptr) return false;
        srcs.p[k] = w[k];
    }
    // R35(w1): MAYA_GLM_DEQ_SMEM_SRC=0 keeps the global-source body (the A/B arm); default on is the
    // shared-source one, worth 1.337x on the dequant's write bandwidth (see deq_multi_ssrc_kernel).
    static const int ssrc_sbs = [] {
        const char* v = getenv("MAYA_GLM_DEQ_SMEM_SRC");
        if (v != nullptr && std::atoi(v) == 0) return 0;
        const char* n = getenv("MAYA_GLM_DEQ_SRC_SBS");
        const int k = n != nullptr ? std::atoi(n) : 16;
        return (k == 8 || k == 16 || k == 32) ? k : 16;
    }();
    // R36(w1): MAYA_GLM_DEQ_XOR_SIGN=0 turns off the sign-XOR arm (the A/B), which applies to IQ2_XXS only
    // (its eight sign selects are the ones the mask table replaces; IQ2_S reads its signs from the block's
    // own qs bytes and its body is unchanged).
    // R36(w1): default OFF.  The arm is bit-exact (DEQ_XOR_CHECK: 0 bad of 16777216) and 1.161x on the
    // dequant body, and it stays 1.14x even at the engine's real nb = 1-3 geometry (research/r36/w1_deq8.mu),
    // but it measures dead neutral in the engine: A/B/A/B on 32K gives select 844.9 / 845.2 against XOR
    // 844.6 / 845.4 tok/s, and res_deq itself moves 0.3%.  The reason is not this kernel's speed: the fast
    // dequant bodies are only 404.9 GB a card = 1.11-1.35 s of the 3.3-4.7 s that res_deq bills (24-29%,
    // MAYA_GLM_DEQ_ARM_STATS=1), so a 1.16x kernel is ~0.7% of the stream.  MAYA_GLM_DEQ_XOR_SIGN=1 turns it on.
    static const bool xor_sign = [] {
        const char* v = getenv("MAYA_GLM_DEQ_XOR_SIGN");
        return v != nullptr && std::atoi(v) != 0;
    }();
    if (ssrc_sbs > 0) {
        const int bs = 512;                       // the sweep's best (426.1 GB/s against 419.6 at 256)
        const int warps = bs / 32;
        const size_t blk = t == GGML_TYPE_IQ2_XXS ? sizeof(block_iq2_xxs) : sizeof(block_iq2_s);
        const bool xs = xor_sign && t == GGML_TYPE_IQ2_XXS;
        const size_t smem = (size_t) warps * (size_t) ssrc_sbs * blk +
                            (size_t) (t == GGML_TYPE_IQ2_XXS ? 256 : 1024) * sizeof(uint64_t) +
                            (t == GGML_TYPE_IQ2_XXS ? 128 : 0) +
                            (xs ? 128 * sizeof(uint4) : 0);   // the sign mask table (R36)
        if (smem <= 48u << 10) {
            const int64_t gsb = (int64_t) warps * (int64_t) ssrc_sbs;
            const dim3 grid2((unsigned) ((nsb + gsb - 1) / gsb), (unsigned) count);
            const auto launch = [&](auto kern) {
                kern<<<grid2, (unsigned) bs, smem, s>>>(t, srcs, (half*) dst, nsb, spb);
            };
            const auto launch_x = [&](auto kern) {
                kern<<<grid2, (unsigned) bs, smem, s>>>(t, srcs, (half*) dst, nsb, spb);
            };
            if (t == GGML_TYPE_IQ2_XXS) {
                if (xs) {
                    if (ssrc_sbs == 8) launch_x(deq_multi_ssrc_kernel<8, true>);
                    else if (ssrc_sbs == 32) launch_x(deq_multi_ssrc_kernel<32, true>);
                    else launch_x(deq_multi_ssrc_kernel<16, true>);
                } else if (ssrc_sbs == 8) launch(deq_multi_ssrc_kernel<8>);
                else if (ssrc_sbs == 32) launch(deq_multi_ssrc_kernel<32>);
                else launch(deq_multi_ssrc_kernel<16>);
                ck(cudaGetLastError(), "to_f16 multi ssrc");
                arm_bytes(1, (int) xs, (int64_t) n * (int64_t) count);
                if (xs) deq_check_multi_xor(t, w, count, n, s);
                return true;
            }
            // IQ2_S: the down matrix, whose spb comes from MAYA_GLM_DEQ_SPB_DN (32 by default); only take the
            // shared-source path where nsb is a multiple of the block's superblocks, else fall through.
            if (t == GGML_TYPE_IQ2_S && nsb % (int64_t) spb == 0) {
                if (ssrc_sbs == 8) launch(deq_multi_ssrc_kernel<8>);
                else if (ssrc_sbs == 32) launch(deq_multi_ssrc_kernel<32>);
                else launch(deq_multi_ssrc_kernel<16>);
                ck(cudaGetLastError(), "to_f16 multi ssrc");
                return true;
            }
        }
    }
    const dim3 grid((unsigned) (nsb / (int64_t) spb), (unsigned) count);
    deq_multi_smem_kernel<<<grid, (unsigned) (spb * 32), 0, s>>>(t, srcs, (half*) dst, nsb, spb);
    ck(cudaGetLastError(), "to_f16 multi");
    return true;
}


bool to_f16(int t, const void* w, uint16_t* dst, int64_t n, void* stream) {
    if (to_f16_fast(t, w, dst, n, (cudaStream_t) stream)) {
        deq_check(t, w, n, (cudaStream_t) stream);
        return true;
    }
    const to_fp16_cuda_t f = ggml_get_to_fp16_cuda((ggml_type) t);
    if (f == nullptr) return false;
    f(w, (half*) dst, n, (cudaStream_t) stream);
    ck(cudaGetLastError(), "to_f16");
    return true;
}


extern "C" void strata_mmq_pool_slab(int dev, void* base, size_t bytes);   // src/prefill/ggml_cuda_host.cu's pool

void pool_slab(int dev, void* base, size_t bytes) { strata_mmq_pool_slab(dev, base, bytes); }

}  // namespace strata::prefill::mmq
