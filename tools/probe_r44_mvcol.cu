// R44 step 1 microbenchmark: does mv()'s ncol=2 read the weight rows ONCE for both columns?
//
// The shape is the model's own output head: Q6_K, n_in=4096, n_out=154880, 520.4 MB of weight rows.
// One column = what the tail's head GEMV reads per token today; two columns = what the merged launch would
// read if the trunk's head and the draft block's head shared one pass over output.weight.
//
//   ncol=1 twice  -> 2 x 520.4 MB
//   ncol=2 once   -> 520.4 MB if the rows are shared, 1040.8 MB if they are not
//
// Also prints whether column 0 of the two-column launch is bit for bit the one-column launch's (the header's
// promise: "Each column's sum is the single-column kernel's, bit for bit") - on a MUSA build the one-column job
// goes through mv_rdna and the two-column one through the tiled kernels, so this is a real question.
#include <strata/kernels/glm_fast.hpp>

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace strata::kernels::glmf;

#define CK(x)                                                                                       \
    do {                                                                                            \
        cudaError_t e_ = (x);                                                                       \
        if (e_ != cudaSuccess) {                                                                    \
            std::printf("FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));   \
            std::exit(1);                                                                           \
        }                                                                                           \
    } while (0)

static constexpr int kNIn = 4096, kNOut = 154880;
static constexpr int kRowBytes = kNIn / 256 * 210;   // Q6_K: 210 B / 256 values
static constexpr int kXqStride = kNIn / 32 * 36;     // q8_1: 36 B / 32 values

// a Q6_K block that is not degenerate: d = 1.0, every scale 4..7, ql/qh a fixed pattern
static void fill_weights(uint8_t* p, size_t nblocks) {
    uint8_t* b = p;
    for (size_t i = 0; i < nblocks; ++i, b += 210) {
        b[0] = 0x00; b[1] = 0x3C;                          // fp16 1.0
        for (int k = 0; k < 16; ++k) b[2 + k] = (uint8_t) (4 + (k & 3));
        for (int k = 0; k < 128; ++k) b[18 + k] = (uint8_t) ((k * 37 + i * 11) & 0xFF);
        for (int k = 0; k < 64; ++k) b[146 + k] = (uint8_t) ((k * 91 + i * 7) & 0xFF);
    }
}

static void fill_xq(uint8_t* b, int seed) {
    for (int blk = 0; blk < kNIn / 32; ++blk, b += 36) {
        b[0] = 0x00; b[1] = 0x3C;                          // d = 1.0
        b[2] = 0x00; b[3] = 0x00;                          // s = 0
        for (int k = 0; k < 32; ++k) b[4 + k] = (uint8_t) (int8_t) (((k * 13 + blk * 5 + seed * 29) & 63) - 32);
    }
}

static float ms_of(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.f;
    CK(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 20;
    const bool use_rdna_cols = getenv("MAYA_GLM_MV_TILED") != nullptr && std::atoi(getenv("MAYA_GLM_MV_TILED")) == 0;
    CK(cudaSetDevice(0));
    cudaDeviceProp pr{};
    CK(cudaGetDeviceProperties(&pr, 0));
    std::printf("device %s, %d CU, %.0f MHz | head shape: Q6_K %d -> %d, %.1f MB a pass%s\n", pr.name,
                pr.multiProcessorCount, pr.clockRate / 1000.0, kNIn, kNOut, (double) kRowBytes * kNOut / 1e6,
                use_rdna_cols ? "  [MAYA_GLM_MV_TILED=0: the column kernels]" : "");

    const size_t wbytes = (size_t) kRowBytes * kNOut;
    std::vector<uint8_t> hw(wbytes);
    fill_weights(hw.data(), (size_t) kNOut * (kNIn / 256));
    void* w = nullptr;
    CK(cudaMalloc(&w, wbytes));
    CK(cudaMemcpy(w, hw.data(), wbytes, cudaMemcpyHostToDevice));
    std::vector<uint8_t> hx(2 * kXqStride);
    fill_xq(hx.data(), 1);
    fill_xq(hx.data() + kXqStride, 2);

    void *x1 = nullptr, *x2 = nullptr;
    CK(cudaMalloc(&x1, 2 * kXqStride));
    CK(cudaMemcpy(x1, hx.data(), 2 * kXqStride, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&x2, 2 * kXqStride));
    CK(cudaMemcpy(x2, hx.data(), 2 * kXqStride, cudaMemcpyHostToDevice));

    float *y1a = nullptr, *y1b = nullptr, *y2 = nullptr;
    CK(cudaMalloc(&y1a, (size_t) kNOut * sizeof(float)));
    CK(cudaMalloc(&y1b, (size_t) kNOut * sizeof(float)));
    CK(cudaMalloc(&y2, 2 * (size_t) kNOut * sizeof(float)));

    MvJob one = {};
    one.w = w; one.xq = x1; one.y = y1a; one.type = 14; one.n_in = kNIn; one.n_out = kNOut; one.alpha = 1.0f;

    MvJob two = one;
    two.xq = x2; two.y = y2; two.ncol = 2; two.xq_cstride = kXqStride; two.y_cstride = kNOut;

    // ---- correctness first: column 0 and column 1 of the 2-column launch against the 1-column launches
    MvJob oa = one;
    if (!mv(&oa, 1, 0)) { std::printf("FAIL mv ncol=1 (a)\n"); return 1; }
    MvJob ob = one;
    ob.xq = (char*) x1 + kXqStride;
    ob.y = y1b;
    if (!mv(&ob, 1, 0)) { std::printf("FAIL mv ncol=1 (b)\n"); return 1; }
    if (!mv(&two, 1, 0)) { std::printf("FAIL mv ncol=2\n"); return 1; }
    CK(cudaDeviceSynchronize());

    std::vector<float> ha(kNOut), hb(kNOut), hc(kNOut), hd(kNOut);
    CK(cudaMemcpy(ha.data(), y1a, (size_t) kNOut * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hb.data(), y1b, (size_t) kNOut * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hc.data(), y2, (size_t) kNOut * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hd.data(), (char*) y2 + (size_t) kNOut * 4, (size_t) kNOut * 4, cudaMemcpyDeviceToHost));
    long bad0 = 0, bad1 = 0, nz = 0;
    double maxrel = 0.0;
    for (int r = 0; r < kNOut; ++r) {
        if (std::memcmp(&ha[r], &hc[r], 4) != 0) {
            ++bad0;
            const double d = std::abs((double) ha[r] - (double) hc[r]);
            const double m = std::abs((double) ha[r]) > 1e-30 ? d / std::abs((double) ha[r]) : d;
            if (m > maxrel) maxrel = m;
        }
        if (std::memcmp(&hb[r], &hd[r], 4) != 0) ++bad1;
        if (ha[r] != 0.0f) ++nz;
    }
    std::printf("bit-exactness over %d rows: column0 differs in %ld (max rel %.2e), column1 in %ld "
                "(rows with a nonzero sum: %ld)\n", kNOut, bad0, maxrel, bad1, nz);

    // ---- timing: alternate the two arms so a clock change hits both
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    double t1[2] = {0, 0}, t2[2] = {0, 0};
    for (int round = 0; round < 2; ++round) {
        CK(cudaEventRecord(e0, 0));
        for (int i = 0; i < iters; ++i) {
            MvJob j = one;
            if (!mv(&j, 1, 0)) { std::printf("FAIL\n"); return 1; }
        }
        CK(cudaEventRecord(e1, 0));
        CK(cudaEventSynchronize(e1));
        t1[round] = ms_of(e0, e1) / iters;

        CK(cudaEventRecord(e0, 0));
        for (int i = 0; i < iters; ++i) {
            MvJob j = two;
            if (!mv(&j, 1, 0)) { std::printf("FAIL\n"); return 1; }
        }
        CK(cudaEventRecord(e1, 0));
        CK(cudaEventSynchronize(e1));
        t2[round] = ms_of(e0, e1) / iters;
    }
    const double a1 = (t1[0] + t1[1]) / 2, a2 = (t2[0] + t2[1]) / 2;
    const double mb = (double) kRowBytes * kNOut / 1e6;
    std::printf("\nncol=1  %7.3f ms for one column    %6.1f GB/s  (%.1f MB of rows)\n", a1, mb / a1, mb);
    std::printf("ncol=2  %7.3f ms for two columns   %6.1f GB/s if the rows are read once (%.1f MB either way)\n", a2,
                2 * mb / a2, mb);
    std::printf("ratio 2-col / 1-col = %.3f  (1.0 = rows shared, 2.0 = not shared)\n", a2 / a1);
    std::printf("two 1-column launches %.3f ms vs one 2-column %.3f ms -> %.3f ms saved per token, %.1f MB not read\n",
                2 * a1, a2, 2 * a1 - a2, mb);
    CK(cudaFree(w));
    return 0;
}
