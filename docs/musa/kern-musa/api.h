#pragma once
// ---------------- minimal integration interface ----------------
// one function per quant type; x must be quantized to int8 (per 32 elements) first
// via musa_gemv_quant_x(). All kernels are M=1 GEMV; w points at ONE expert matrix,
// row-major [n_out][k] in the native ggml block layout.
//   IQ2_XXS : gate/up   n_out=2048 k=4096   (66 B / 256 elem)
//   IQ2_S   : down      n_out=4096 k=2048   (82 B / 256 elem)
//   IQ3_XXS : down      n_out=4096 k=2048   (98 B / 256 elem)
// returns 0 on success; launches on stream s (pass 0 for default stream).


// per-32-group int8 quantization of x into the padded layout used by the kernels
__global__ void quant_x_k(const float* __restrict__ x, int8_t* __restrict__ xq,
                          float* __restrict__ xs, int ng)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ng) return;
    const float* xg = x + (size_t)g * 32;
    float amax = 0.f;
    for (int i = 0; i < 32; ++i) { float v = fabsf(xg[i]); if (v > amax) amax = v; }
    float d = amax / 127.f;
    if (d == 0.f) d = 1.f;
    xs[g] = d;
    int8_t* o = xq + (size_t)g * 36;
    for (int i = 0; i < 32; ++i) {
        int v = (int)lrintf(xg[i] / d);
        if (v > 127) v = 127; if (v < -127) v = -127;
        o[i] = (int8_t)v;
    }
    for (int i = 32; i < 36; ++i) o[i] = 0;
}

static int8_t* g_xq = nullptr;
static float*  g_xs = nullptr;

extern "C" int musa_gemv_quant_x(const float* x, int k, musaStream_t s)
{
    const int ng = k / 32;
    if (!g_xq) {
        if (musaMalloc((void**)&g_xq, 128 * 36) != musaSuccess) return -1;
        if (musaMalloc((void**)&g_xs, 128 * sizeof(float)) != musaSuccess) return -1;
    }
    quant_x_k<<<(ng + 127) / 128, 128, 0, s>>>(x, g_xq, g_xs, ng);
    return 0;
}

extern "C" int musa_mv_iq2xxs(const void* w, float* y, int n_out, int k, musaStream_t s)
{
    const int rb = (k / 256) * 66;
    const uint8_t* wp = (const uint8_t*)w;
    // layout A (warp=row, 4 warps per row => 1024 blocks/expert): best all-round granularity
    if (k == 4096) {
        gemv_k<T_IQ2XXS, 2, 8, 4, 1, 1, false><<<dim3(n_out / 2), 256, 0, s>>>(wp, g_xq, g_xs, y, n_out / 2, n_out, k / 32, rb);
    } else if (k == 2048) {
        gemv_k<T_IQ2XXS, 4, 8, 2, 1, 1, false><<<dim3(n_out / 4), 256, 0, s>>>(wp, g_xq, g_xs, y, n_out / 4, n_out, k / 32, rb);
    } else return -1;
    return 0;
}

extern "C" int musa_mv_iq2s(const void* w, float* y, int n_out, int k, musaStream_t s)
{
    const int rb = (k / 256) * 82;
    const uint8_t* wp = (const uint8_t*)w;
    if (k == 2048) {
        gemv_k<T_IQ2S, 4, 8, 2, 1, 1, false><<<dim3(n_out / 4), 256, 0, s>>>(wp, g_xq, g_xs, y, n_out / 4, n_out, k / 32, rb);
    } else return -1;
    return 0;
}

extern "C" int musa_mv_iq3xxs(const void* w, float* y, int n_out, int k, musaStream_t s)
{
    const int rb = (k / 256) * 98;
    const uint8_t* wp = (const uint8_t*)w;
    if (k == 2048) {
        gemv_k<T_IQ3XXS, 4, 8, 2, 1, 1, false><<<dim3(n_out / 4), 256, 0, s>>>(wp, g_xq, g_xs, y, n_out / 4, n_out, k / 32, rb);
    } else return -1;
    return 0;
}
