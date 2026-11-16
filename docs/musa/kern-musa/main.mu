#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <vector>
#include <string>
#include <algorithm>
#include <musa_runtime.h>
#include <musa_fp16.h>
#include "tables.h"
#include "k_decl.h"
#include "host_ref.h"
#include "k_gemv.h"
#include "api.h"

#define CK(x) do{ musaError_t _e=(x); if(_e!=musaSuccess){ printf("MUSA ERR %s line %d\n", musaGetErrorName(_e), __LINE__); exit(1);} }while(0)

struct Ctx {
    Shape sh; int E; int M;
    uint8_t* dw; int8_t* dxq; float* dxs; float* dy;
    size_t bank_bytes;
    std::vector<float> yf, yi;
};

template<int TY,int R,int WPB,int CPB,int NSUB,int M>
static void launch(dim3 g, Ctx& c, int rowtiles, bool use_shared)
{
    const Shape& sh = c.sh;
    if (use_shared)
        gemv_ks<TY,R,WPB,CPB,NSUB,M><<<g, WPB*32>>>(c.dw, c.dxq, c.dxs, c.dy, rowtiles, sh.n_out, sh.ng, sh.row_bytes);
    else
        gemv_k<TY,R,WPB,CPB,NSUB,M,false><<<g, WPB*32>>>(c.dw, c.dxq, c.dxs, c.dy, rowtiles, sh.n_out, sh.ng, sh.row_bytes);
}

template<int TY,int R,int WPB,int CPB,int NSUB,int M>
static void run_one(const char* tag, Ctx& c, int iters, bool use_shared)
{
    const Shape& sh = c.sh;
    const int rowtiles = sh.n_out / R;
    const int chunks = sh.k / (1024 * NSUB);
    if (chunks % CPB || chunks / CPB < 1) { printf("%-30s cfg invalid (chunks=%d CPB=%d)\n", tag, chunks, CPB); return; }
    const int S = chunks / CPB;
    const size_t nbytes = (size_t)M * sh.n_out * sizeof(float);

    CK(musaMemset(c.dy, 0, nbytes));
    {
        dim3 g(rowtiles, S);
        launch<TY,R,WPB,CPB,NSUB,M>(g, c, rowtiles, use_shared);
        CK(musaDeviceSynchronize());
    }
    std::vector<float> yk((size_t)M * sh.n_out);
    CK(musaMemcpy(yk.data(), c.dy, nbytes, musaMemcpyDeviceToHost));
    double maxrel_i = 0, maxabs_f = 0, nrm = 0;
    for (int r = 0; r < sh.n_out; ++r) {
        double ref = c.yi[r], got = yk[r];
        maxrel_i = std::max(maxrel_i, fabs(got - ref) / std::max(1e-6, fabs(ref)));
        maxabs_f = std::max(maxabs_f, fabs((double)c.yf[r] - got));
        nrm = std::max(nrm, fabs((double)c.yf[r]));
    }

    dim3 g((int)((size_t)c.E * rowtiles), S);
    const int th = WPB * 32;
    for (int i = 0; i < 3; ++i) launch<TY,R,WPB,CPB,NSUB,M>(g, c, rowtiles, use_shared);
    CK(musaDeviceSynchronize());
    musaEvent_t e0, e1; CK(musaEventCreate(&e0)); CK(musaEventCreate(&e1));
    float ms = 1e30f;
    for (int rep = 0; rep < 5; ++rep) {
        CK(musaEventRecord(e0, 0));
        for (int i = 0; i < iters; ++i) launch<TY,R,WPB,CPB,NSUB,M>(g, c, rowtiles, use_shared);
        CK(musaEventRecord(e1, 0));
        CK(musaEventSynchronize(e1));
        float t = 0; CK(musaEventElapsedTime(&t, e0, e1)); t /= iters;
        if (t < ms) ms = t;
    }
    CK(musaEventDestroy(e0)); CK(musaEventDestroy(e1));
    // single-expert launch (what a per-expert decode launch looks like)
    dim3 g1(rowtiles, S);
    musaEvent_t f0, f1; CK(musaEventCreate(&f0)); CK(musaEventCreate(&f1));
    float ms1 = 1e30f;
    for (int rep = 0; rep < 3; ++rep) {
        CK(musaEventRecord(f0, 0));
        for (int i = 0; i < 50; ++i) launch<TY,R,WPB,CPB,NSUB,M>(g1, c, rowtiles, use_shared);
        CK(musaEventRecord(f1, 0)); CK(musaEventSynchronize(f1));
        float t = 0; CK(musaEventElapsedTime(&t, f0, f1)); t /= 50;
        if (t < ms1) ms1 = t;
    }
    CK(musaEventDestroy(f0)); CK(musaEventDestroy(f1));
    // 8 experts in one launch (one layer's top-k)
    dim3 g8(8 * rowtiles, S);
    musaEvent_t h0, h1; CK(musaEventCreate(&h0)); CK(musaEventCreate(&h1));
    float ms8 = 1e30f;
    for (int rep = 0; rep < 3; ++rep) {
        CK(musaEventRecord(h0, 0));
        for (int i = 0; i < 20; ++i) launch<TY,R,WPB,CPB,NSUB,M>(g8, c, rowtiles, use_shared);
        CK(musaEventRecord(h1, 0)); CK(musaEventSynchronize(h1));
        float t = 0; CK(musaEventElapsedTime(&t, h0, h1)); t /= 20;
        if (t < ms8) ms8 = t;
    }
    CK(musaEventDestroy(h0)); CK(musaEventDestroy(h1));
    const double one_mb = (double)sh.n_out * sh.row_bytes / 1e6;
    printf("%-14s R=%d W=%3d CPB=%d NSUB=%d M=%d S=%d %s | blk=%6d th=%3d | bank %7.3f ms %7.1f GB/s | 1e(%4d blk) %7.3f ms %6.1f GB/s | 8e(%5d blk) %7.3f ms %6.1f GB/s | err_i8=%.1e\n",
           tag, R, WPB, CPB, NSUB, M, S, use_shared ? "SHR" : "GLB",
           (int)g.x, th, ms, (double)c.bank_bytes / (ms * 1e6), (int)g1.x, ms1, one_mb / ms1,
           (int)g8.x, ms8, 8.0 * one_mb / ms8, maxrel_i);
    fflush(stdout);
}

template<int TY,int R,int WPB,int CPB,int NSUB>
static void run_read(const char* tag, Ctx& c, int iters)
{
    const Shape& sh = c.sh;
    const int rowtiles = sh.n_out / R;
    const int chunks = sh.k / (1024 * NSUB);
    const int S = chunks / CPB;
    const int th = WPB * 32;
    dim3 g((int)((size_t)c.E * rowtiles), S);
    for (int i = 0; i < 3; ++i)
        read_k<TY,R,WPB,CPB,NSUB><<<g, th>>>(c.dw, c.dy, rowtiles, sh.n_out, sh.row_bytes);
    CK(musaDeviceSynchronize());
    musaEvent_t e0, e1; CK(musaEventCreate(&e0)); CK(musaEventCreate(&e1));
    float ms = 1e30f;
    for (int rep = 0; rep < 5; ++rep) {
        CK(musaEventRecord(e0, 0));
        for (int i = 0; i < iters; ++i)
            read_k<TY,R,WPB,CPB,NSUB><<<g, th>>>(c.dw, c.dy, rowtiles, sh.n_out, sh.row_bytes);
        CK(musaEventRecord(e1, 0));
        CK(musaEventSynchronize(e1));
        float t = 0; CK(musaEventElapsedTime(&t, e0, e1)); t /= iters;
        if (t < ms) ms = t;
    }
    CK(musaEventDestroy(e0)); CK(musaEventDestroy(e1));
    printf("%-14s R=%d W=%3d CPB=%d NSUB=%d READ-ONLY | blk=%6d | %7.3f ms %7.1f GB/s\n",
           tag, R, WPB, CPB, NSUB, (int)g.x, ms, (double)c.bank_bytes / (ms * 1e6));
    fflush(stdout);
}


template<int TY,int R,int WPB,int CPB,int NSUB,int MODE>
static void run_probe(const char* tag, Ctx& c, int iters)
{
    const Shape& sh = c.sh;
    const int rowtiles = sh.n_out / R;
    const int S = (sh.k / (1024 * NSUB)) / CPB;
    const int th = WPB * 32;
    dim3 g((int)((size_t)c.E * rowtiles), S);
    for (int i = 0; i < 3; ++i) probe_k<TY,R,WPB,CPB,NSUB,MODE><<<g, th>>>(c.dw, c.dxq, c.dy, rowtiles, sh.n_out, sh.row_bytes);
    CK(musaDeviceSynchronize());
    musaEvent_t e0, e1; CK(musaEventCreate(&e0)); CK(musaEventCreate(&e1));
    float ms = 1e30f;
    for (int rep = 0; rep < 5; ++rep) {
        CK(musaEventRecord(e0, 0));
        for (int i = 0; i < iters; ++i) probe_k<TY,R,WPB,CPB,NSUB,MODE><<<g, th>>>(c.dw, c.dxq, c.dy, rowtiles, sh.n_out, sh.row_bytes);
        CK(musaEventRecord(e1, 0)); CK(musaEventSynchronize(e1));
        float t = 0; CK(musaEventElapsedTime(&t, e0, e1)); t /= iters;
        if (t < ms) ms = t;
    }
    CK(musaEventDestroy(e0)); CK(musaEventDestroy(e1));
    const char* mn = MODE == 0 ? "tabonly" : MODE == 1 ? "dot-notab" : "tab+dot";
    printf("%-14s R=%d W=%3d CPB=%d NSUB=%d PROBE %-9s | blk=%6d | %7.3f ms %7.1f GB/s\n",
           tag, R, WPB, CPB, NSUB, mn, (int)g.x, ms, (double)c.bank_bytes / (ms * 1e6));
    fflush(stdout);
}


template<int TY,int WPB,int NSUB,int U>
static void run_lb(const char* tag, Ctx& c, int iters)
{
    const Shape& sh = c.sh;
    const int rowtiles = sh.n_out / 32;
    const int th = WPB * 32;
    CK(musaMemset(c.dy, 0, (size_t)sh.n_out * sizeof(float)));
    gemv_lb<TY,WPB,NSUB,U><<<dim3(rowtiles), th>>>(c.dw, c.dxq, c.dxs, c.dy, rowtiles, sh.n_out, sh.ng, sh.row_bytes);
    CK(musaDeviceSynchronize());
    std::vector<float> yk(sh.n_out);
    CK(musaMemcpy(yk.data(), c.dy, (size_t)sh.n_out * sizeof(float), musaMemcpyDeviceToHost));
    double mrel = 0, mabs = 0;
    for (int r = 0; r < sh.n_out; ++r) {
        mrel = std::max(mrel, fabs((double)yk[r] - c.yi[r]) / std::max(1e-6, fabs((double)c.yi[r])));
        mabs = std::max(mabs, fabs((double)yk[r] - c.yf[r]));
    }
    dim3 g((int)((size_t)c.E * rowtiles));
    for (int i = 0; i < 3; ++i) gemv_lb<TY,WPB,NSUB,U><<<g, th>>>(c.dw, c.dxq, c.dxs, c.dy, rowtiles, sh.n_out, sh.ng, sh.row_bytes);
    CK(musaDeviceSynchronize());
    musaEvent_t e0, e1; CK(musaEventCreate(&e0)); CK(musaEventCreate(&e1));
    float ms = 1e30f;
    for (int rep = 0; rep < 5; ++rep) {
        CK(musaEventRecord(e0, 0));
        for (int i = 0; i < iters; ++i) gemv_lb<TY,WPB,NSUB,U><<<g, th>>>(c.dw, c.dxq, c.dxs, c.dy, rowtiles, sh.n_out, sh.ng, sh.row_bytes);
        CK(musaEventRecord(e1, 0)); CK(musaEventSynchronize(e1));
        float t = 0; CK(musaEventElapsedTime(&t, e0, e1)); t /= iters;
        if (t < ms) ms = t;
    }
    CK(musaEventDestroy(e0)); CK(musaEventDestroy(e1));
    dim3 g1(rowtiles);
    musaEvent_t f0, f1; CK(musaEventCreate(&f0)); CK(musaEventCreate(&f1));
    float ms1 = 1e30f;
    for (int rep = 0; rep < 3; ++rep) {
        CK(musaEventRecord(f0, 0));
        for (int i = 0; i < 50; ++i) gemv_lb<TY,WPB,NSUB,U><<<g1, th>>>(c.dw, c.dxq, c.dxs, c.dy, rowtiles, sh.n_out, sh.ng, sh.row_bytes);
        CK(musaEventRecord(f1, 0)); CK(musaEventSynchronize(f1));
        float t = 0; CK(musaEventElapsedTime(&t, f0, f1)); t /= 50;
        if (t < ms1) ms1 = t;
    }
    CK(musaEventDestroy(f0)); CK(musaEventDestroy(f1));
    const double one_mb = (double)sh.n_out * sh.row_bytes / 1e6;
    printf("%-14s LANE-ROW W=%d NSUB=%d U=%d | blk=%6d th=%3d | bank %7.3f ms %7.1f GB/s | 1 expert(%d blk) %7.3f ms %6.1f GB/s err_i8=%.1e err_fx=%.1e\n",
           tag, WPB, NSUB, U, (int)g.x, th, ms, (double)c.bank_bytes / (ms * 1e6), (int)g1.x, ms1, one_mb / ms1, mrel, mabs);
    fflush(stdout);
}


static void test_api(const char* tag, Ctx& c)
{
    const Shape& sh = c.sh;
    static float* dx = nullptr;
    static float* dy = nullptr;
    if (!dx) { CK(musaMalloc((void**)&dx, 4096 * sizeof(float))); CK(musaMalloc((void**)&dy, 8192 * sizeof(float))); }
    // x row 0 is at c.dw? no: keep a host copy through c.dy? use caller-provided via globals
    extern const float* g_hx_row0;
    CK(musaMemcpy(dx, g_hx_row0, (size_t)sh.k * sizeof(float), musaMemcpyHostToDevice));
    CK(musaMemset(dy, 0, (size_t)sh.n_out * sizeof(float)));
    if (musa_gemv_quant_x(dx, sh.k, 0)) { printf("  API %s: quant failed\n", tag); return; }
    int rc = -1;
    if (sh.ty == T_IQ2XXS) rc = musa_mv_iq2xxs(c.dw, dy, sh.n_out, sh.k, 0);
    else if (sh.ty == T_IQ2S) rc = musa_mv_iq2s(c.dw, dy, sh.n_out, sh.k, 0);
    else rc = musa_mv_iq3xxs(c.dw, dy, sh.n_out, sh.k, 0);
    if (rc) { printf("  API %s: launch failed rc=%d\n", tag, rc); return; }
    CK(musaDeviceSynchronize());
    std::vector<float> yk(sh.n_out);
    CK(musaMemcpy(yk.data(), dy, (size_t)sh.n_out * sizeof(float), musaMemcpyDeviceToHost));
    double mx = 0, nrm = 0;
    for (int r = 0; r < sh.n_out; ++r) { mx = std::max(mx, fabs((double)yk[r] - c.yf[r])); nrm = std::max(nrm, fabs((double)c.yf[r])); }
    // API timing: quant_x + gemv, per expert (E copies), including the quantize launch
    musaEvent_t e0, e1; CK(musaEventCreate(&e0)); CK(musaEventCreate(&e1));
    const int N = 20;
    float best = 1e30f;
    for (int rep = 0; rep < 5; ++rep) {
        CK(musaEventRecord(e0, 0));
        for (int i = 0; i < N; ++i) {
            musa_gemv_quant_x(dx, sh.k, 0);
            if (sh.ty == T_IQ2XXS) musa_mv_iq2xxs(c.dw, dy, sh.n_out, sh.k, 0);
            else if (sh.ty == T_IQ2S) musa_mv_iq2s(c.dw, dy, sh.n_out, sh.k, 0);
            else musa_mv_iq3xxs(c.dw, dy, sh.n_out, sh.k, 0);
        }
        CK(musaEventRecord(e1, 0)); CK(musaEventSynchronize(e1));
        float t = 0; CK(musaEventElapsedTime(&t, e0, e1)); t /= N;
        if (t < best) best = t;
    }
    CK(musaEventDestroy(e0)); CK(musaEventDestroy(e1));
    printf("  API %-8s musa_mv_%s(x) -> ok, max_abs_err=%.2e (|y|=%.2f); 1 expert + quant: %.3f ms = %.1f GB/s (%.1f us/expert)\n",
           tag, sh.ty == T_IQ2XXS ? "iq2xxs" : sh.ty == T_IQ2S ? "iq2s" : "iq3xxs", mx, nrm,
           best, (double)sh.n_out * sh.row_bytes / (best * 1e6), best * 1000.0);
    fflush(stdout);
}

const float* g_hx_row0 = nullptr;
static void quantize_x(const float* x, int k, int ng, int8_t* xq, float* xs)
{
    for (int g = 0; g < ng; ++g) {
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, fabsf(x[g*32+i]));
        float d = amax / 127.f;
        if (d == 0.f) d = 1.f;
        xs[g] = d;
        for (int i = 0; i < 32; ++i) {
            int v = (int)lrintf(x[g*32+i] / d);
            if (v > 127) v = 127; if (v < -127) v = -127;
            xq[g*36 + i] = (int8_t)v;
        }
        for (int i = 32; i < 36; ++i) xq[g*36 + i] = 0;
    }
}

int main(int argc, char** argv)
{
    int iters = argc > 1 ? atoi(argv[1]) : 20;
    CK(musaMemcpyToSymbol(D_IQ2XXS, H_IQ2XXS_GRID, sizeof(H_IQ2XXS_GRID)));
    CK(musaMemcpyToSymbol(D_IQ2S, H_IQ2S_GRID, sizeof(H_IQ2S_GRID)));
    CK(musaMemcpyToSymbol(D_IQ3XXS, H_IQ3XXS_GRID, sizeof(H_IQ3XXS_GRID)));
    CK(musaMemcpyToSymbol(D_KSIGNS64, H_KSIGNS64, sizeof(H_KSIGNS64)));
    CK(musaMemcpyToSymbol(D_KSB, H_KSIGNS_IQ2XS, sizeof(H_KSIGNS_IQ2XS)));
    {
        std::vector<uint64_t> sgb(256);
        for (int i = 0; i < 256; ++i) {
            uint64_t v = 0;
            for (int j = 0; j < 8; ++j) if (i & (1 << j)) v |= (uint64_t)0xff << (8*j);
            sgb[i] = v;
        }
        CK(musaMemcpyToSymbol(D_SGB, sgb.data(), 256 * 8));
    }

    printf("reading blobs...\n"); fflush(stdout);
    auto gate = pread_blob(SHARD1, 11894060640LL, 2048ull * 4096 / 256 * 66);
    auto dn22 = pread_blob(SHARD1, 11113330272LL, 4096ull * 2048 / 256 * 82);
    auto dn18 = pread_blob(SHARD1, 1870220128LL, 4096ull * 2048 / 256 * 98);

    struct Case { Shape sh; std::vector<uint8_t> blob; int E; const char* tag; };
    std::vector<Case> cases;
    cases.push_back({Shape{T_IQ2XXS, 2048, 4096, 1056, 128, "gate"}, gate, 64, "IQ2XXS"});
    cases.push_back({Shape{T_IQ2S,   4096, 2048, 656,   64, "down"}, dn22, 48, "IQ2S"});
    cases.push_back({Shape{T_IQ3XXS, 4096, 2048, 784,   64, "down"}, dn18, 40, "IQ3XXS"});

    for (auto& cs : cases) {
        const Shape& sh = cs.sh;
        Ctx c; c.sh = sh; c.E = cs.E; c.M = 1;
        size_t bn = (size_t)cs.E * cs.blob.size();
        c.bank_bytes = bn;
        std::vector<uint8_t> hbank(bn);
        for (int e = 0; e < cs.E; ++e) memcpy(hbank.data() + (size_t)e * cs.blob.size(), cs.blob.data(), cs.blob.size());
        CK(musaMalloc((void**)&c.dw, bn));
        CK(musaMemcpy(c.dw, hbank.data(), bn, musaMemcpyHostToDevice));
        const int ng = sh.ng;
        CK(musaMalloc((void**)&c.dxq, (size_t)8 * ng * 36));
        CK(musaMalloc((void**)&c.dxs, (size_t)8 * ng * sizeof(float)));
        CK(musaMalloc((void**)&c.dy, (size_t)8 * sh.n_out * sizeof(float)));

        std::vector<float> hx((size_t)8 * sh.k);
        unsigned seed = 12345;
        for (size_t i = 0; i < hx.size(); ++i) { seed = seed * 1664525u + 1013904223u; hx[i] = ((seed >> 8) & 0xffff) / 32768.f - 1.f; }
        std::vector<int8_t> hxq((size_t)8 * ng * 36);
        std::vector<float> hxs((size_t)8 * ng);
        for (int m = 0; m < 8; ++m) quantize_x(hx.data() + (size_t)m * sh.k, sh.k, ng, hxq.data() + (size_t)m * ng * 36, hxs.data() + (size_t)m * ng);
        CK(musaMemcpy(c.dxq, hxq.data(), (size_t)8 * ng * 36, musaMemcpyHostToDevice));
        CK(musaMemcpy(c.dxs, hxs.data(), (size_t)8 * ng * sizeof(float), musaMemcpyHostToDevice));

        printf("=== %s (%s n_out=%d k=%d row_bytes=%d E=%d bank=%.1f MB) ===\n", cs.tag, sh.name, sh.n_out, sh.k, sh.row_bytes, cs.E, bn / 1e6);
        fflush(stdout);
        printf("  cpu reference...\n"); fflush(stdout);
        g_hx_row0 = hx.data();
        cpu_ref(sh, cs.blob.data(), hx.data(), hxq.data(), hxs.data(), c.yf, c.yi);
        test_api(cs.tag, c);

        // naive baseline
        {
            float* dxf = nullptr;
            CK(musaMalloc((void**)&dxf, (size_t)sh.k * sizeof(float)));
            CK(musaMemcpy(dxf, hx.data(), (size_t)sh.k * sizeof(float), musaMemcpyHostToDevice));
            dim3 g(sh.n_out / 8, c.E);
            CK(musaMemset(c.dy, 0, (size_t)sh.n_out * sizeof(float)));
            for (int i = 0; i < 2; ++i) {
                if (sh.ty == T_IQ2XXS) naive_k<T_IQ2XXS><<<g, 256>>>(c.dw, dxf, c.dy, sh.n_out, sh.k, sh.row_bytes);
                else if (sh.ty == T_IQ2S) naive_k<T_IQ2S><<<g, 256>>>(c.dw, dxf, c.dy, sh.n_out, sh.k, sh.row_bytes);
                else naive_k<T_IQ3XXS><<<g, 256>>>(c.dw, dxf, c.dy, sh.n_out, sh.k, sh.row_bytes);
            }
            CK(musaDeviceSynchronize());
            std::vector<float> yt(sh.n_out);
            CK(musaMemcpy(yt.data(), c.dy, (size_t)sh.n_out * sizeof(float), musaMemcpyDeviceToHost));
            double mx = 0, nrm = 0;
            for (int r = 0; r < sh.n_out; ++r) { mx = std::max(mx, fabs((double)c.yf[r] - yt[r])); nrm = std::max(nrm, fabs((double)c.yf[r])); }
            musaEvent_t e0, e1; CK(musaEventCreate(&e0)); CK(musaEventCreate(&e1));
            float best = 1e30f;
            for (int rep = 0; rep < 5; ++rep) {
            CK(musaEventRecord(e0, 0));
            for (int i = 0; i < iters; ++i) {
                if (sh.ty == T_IQ2XXS) naive_k<T_IQ2XXS><<<g, 256>>>(c.dw, dxf, c.dy, sh.n_out, sh.k, sh.row_bytes);
                else if (sh.ty == T_IQ2S) naive_k<T_IQ2S><<<g, 256>>>(c.dw, dxf, c.dy, sh.n_out, sh.k, sh.row_bytes);
                else naive_k<T_IQ3XXS><<<g, 256>>>(c.dw, dxf, c.dy, sh.n_out, sh.k, sh.row_bytes);
            }
            CK(musaEventRecord(e1, 0)); CK(musaEventSynchronize(e1));
            float tt = 0; CK(musaEventElapsedTime(&tt, e0, e1)); tt /= iters;
            if (tt < best) best = tt;
            }
            float ms = best;
            CK(musaEventDestroy(e0)); CK(musaEventDestroy(e1));
            printf("  %-12s NAIVE f32-dequant MV_ROWS=8 | blk=%6d th=256 | %7.3f ms %7.1f GB/s err_fx=%.1e (|y|=%.2f)\n",
                   cs.tag, (int)g.x, ms, (double)bn / (ms * 1e6), mx, nrm);
            fflush(stdout);
            CK(musaFree(dxf));
        }

        if (sh.ty == T_IQ2XXS) {
            run_read<T_IQ2XXS,8,8,1,4>(cs.tag, c, iters);
            run_read<T_IQ2XXS,4,8,2,2>(cs.tag, c, iters);
            run_read<T_IQ2XXS,2,8,4,1>(cs.tag, c, iters);
            run_lb<T_IQ2XXS,8,16,16>(cs.tag, c, iters);
            run_lb<T_IQ2XXS,8,16,4>(cs.tag, c, iters);
            run_lb<T_IQ2XXS,8,16,8>(cs.tag, c, iters);
            run_lb<T_IQ2XXS,4,32,4>(cs.tag, c, iters);
            run_lb<T_IQ2XXS,8,8,8>(cs.tag, c, iters);
            run_probe<T_IQ2XXS,8,8,1,4,0>(cs.tag, c, iters);
            run_probe<T_IQ2XXS,8,8,1,4,1>(cs.tag, c, iters);
            run_probe<T_IQ2XXS,8,8,1,4,2>(cs.tag, c, iters);
            run_one<T_IQ2XXS,8,8,1,4,1>(cs.tag, c, iters, false);
            run_one<T_IQ2XXS,4,8,2,2,1>(cs.tag, c, iters, false);
            run_one<T_IQ2XXS,2,8,4,1,1>(cs.tag, c, iters, false);
            run_one<T_IQ2XXS,8,8,1,4,1>(cs.tag, c, iters, true);
            run_one<T_IQ2XXS,4,8,2,2,1>(cs.tag, c, iters, true);
            run_one<T_IQ2XXS,2,8,4,1,1>(cs.tag, c, iters, true);
            run_one<T_IQ2XXS,4,8,2,2,4>(cs.tag, c, iters, false);
            run_one<T_IQ2XXS,4,8,2,2,8>(cs.tag, c, iters, false);
            run_one<T_IQ2XXS,4,8,2,2,8>(cs.tag, c, iters, true);
        } else if (sh.ty == T_IQ2S) {
            run_read<T_IQ2S,8,8,1,2>(cs.tag, c, iters);
            run_read<T_IQ2S,4,8,2,1>(cs.tag, c, iters);
            run_lb<T_IQ2S,8,8,4>(cs.tag, c, iters);
            run_lb<T_IQ2S,8,8,8>(cs.tag, c, iters);
            run_lb<T_IQ2S,4,16,4>(cs.tag, c, iters);
            run_probe<T_IQ2S,8,8,1,2,0>(cs.tag, c, iters);
            run_probe<T_IQ2S,8,8,1,2,1>(cs.tag, c, iters);
            run_probe<T_IQ2S,8,8,1,2,2>(cs.tag, c, iters);
            run_one<T_IQ2S,8,8,1,2,1>(cs.tag, c, iters, false);
            run_one<T_IQ2S,4,8,2,1,1>(cs.tag, c, iters, false);
            run_one<T_IQ2S,8,8,1,2,1>(cs.tag, c, iters, true);
            run_one<T_IQ2S,4,8,2,1,1>(cs.tag, c, iters, true);
            run_one<T_IQ2S,8,8,1,2,8>(cs.tag, c, iters, false);
            run_one<T_IQ2S,8,8,1,2,8>(cs.tag, c, iters, true);
        } else {
            run_read<T_IQ3XXS,8,8,1,2>(cs.tag, c, iters);
            run_read<T_IQ3XXS,4,8,2,1>(cs.tag, c, iters);
            run_lb<T_IQ3XXS,8,8,4>(cs.tag, c, iters);
            run_lb<T_IQ3XXS,8,8,8>(cs.tag, c, iters);
            run_lb<T_IQ3XXS,4,16,4>(cs.tag, c, iters);
            run_probe<T_IQ3XXS,8,8,1,2,0>(cs.tag, c, iters);
            run_probe<T_IQ3XXS,8,8,1,2,1>(cs.tag, c, iters);
            run_probe<T_IQ3XXS,8,8,1,2,2>(cs.tag, c, iters);
            run_one<T_IQ3XXS,8,8,1,2,1>(cs.tag, c, iters, false);
            run_one<T_IQ3XXS,4,8,2,1,1>(cs.tag, c, iters, false);
            run_one<T_IQ3XXS,8,8,1,2,1>(cs.tag, c, iters, true);
            run_one<T_IQ3XXS,4,8,2,1,1>(cs.tag, c, iters, true);
            run_one<T_IQ3XXS,8,8,1,2,8>(cs.tag, c, iters, false);
        }
        musaFree(c.dw); musaFree(c.dxq); musaFree(c.dxs); musaFree(c.dy);
    }
    printf("done\n");
    return 0;
}
