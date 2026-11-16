// src/kernels/cuda/exl3_stub_musa.cpp - the EXL3 (exllamav3) kernels, not built in the MUSA engine.
//
// WHY THIS EXISTS.  mcc's device pass does not finish on src/kernels/cuda/exl3.cu: 30 minutes at 100% CPU
// with no output, on the same file that the HIP and CUDA compilers build in under a minute.  The kernels it
// holds decode EXL3 quantized weights - a format none of the GLM-5.3-Flash GGUFs use (their experts are
// IQ2_XXS / IQ2_S / IQ3_XXS, through ggml's own CPU/MMQ paths).  Every caller reaches these functions only
// after asking supported()/k2_supported() about the weight type, which answers false here, so nothing calls
// them at run time; they exist to satisfy the linker.
//
// The EXL3 legacy decode (src/kernels/cuda/exl3_legacy.cu) is built as on every backend: it shares no code
// with this file.
//
// What this costs on MUSA: an EXL3-quantized model cannot be loaded.  Everything else is unaffected.

#include "strata/kernels/exl3.hpp"

#include <cstdio>

namespace strata::kernels::exl3 {

namespace {
// Said once per process, so a caller that does reach one of these learns why it got nothing instead of
// silently reading zeros.
void not_built(const char* what) {
    static bool said = false;
    if (!said) {
        said = true;
        std::fprintf(stderr,
                     "exl3 %s: EXL3 weights are not supported by this engine (MUSA build: its kernels are "
                     "not compiled in)\n",
                     what);
    }
}
}  // namespace

Mat view(const Mat& m, int c0, int nc) {
    (void) c0;
    (void) nc;
    not_built("view");
    return m;
}
bool k2_supported(int k2, int cb) {
    (void) k2;
    (void) cb;
    return false;
}
bool supported(const Mat& m) {
    (void) m;
    return false;
}
bool mv(const MvJob* jobs, int n, cudaStream_t s) {
    (void) jobs;
    (void) n;
    (void) s;
    not_built("mv");
    return false;
}
void set_skip(const int* flag) { (void) flag; }
int mv_splits(const Mat& m) {
    (void) m;
    return 0;
}
int mv_cols(const Mat& m) {
    (void) m;
    return 0;
}
bool prepare(cudaStream_t s) {
    (void) s;
    return true;   // no scratch to allocate: there are no EXL3 kernels to run
}
bool mixed_supported(const ExpertLayout& lay) {
    (void) lay;
    return false;
}
void moe_gate_up(const MoeArgs& a, const float* x, float* h, cudaStream_t s) {
    (void) a;
    (void) x;
    (void) h;
    (void) s;
    not_built("moe_gate_up");
}
void moe_down(const MoeArgs& a, const float* h, const float* sh_out, float* out, cudaStream_t s) {
    (void) a;
    (void) h;
    (void) sh_out;
    (void) out;
    (void) s;
    not_built("moe_down");
}
bool win_experts(const WinArgs& a, cudaStream_t s) {
    (void) a;
    (void) s;
    not_built("win_experts");
    return false;
}
bool several_ok() { return false; }
bool mma_supported() { return false; }
bool prompt_tc() { return false; }
bool prompt_wmma() { return false; }
void set_gate_up(const SetArgs& a, const float* x, int ldx, const int* row_idx, uint16_t* h16, int ldh,
                 cudaStream_t s) {
    (void) a;
    (void) x;
    (void) ldx;
    (void) row_idx;
    (void) h16;
    (void) ldh;
    (void) s;
    not_built("set_gate_up");
}
void set_down(const SetArgs& a, const uint16_t* h16, int ldh, float* y, int ldy, cudaStream_t s) {
    (void) a;
    (void) h16;
    (void) ldh;
    (void) y;
    (void) ldy;
    (void) s;
    not_built("set_down");
}
void gemm_rows(const Mat& m, const uint16_t* x16, int ldx, int rows, float* y, int ldy, float beta, cudaStream_t s) {
    (void) m;
    (void) x16;
    (void) ldx;
    (void) rows;
    (void) y;
    (void) ldy;
    (void) beta;
    (void) s;
    not_built("gemm_rows");
}
void reconstruct_rows(const Mat& m, uint16_t* dst, bool bf16, cudaStream_t s) {
    (void) m;
    (void) dst;
    (void) bf16;
    (void) s;
    not_built("reconstruct_rows");
}
void reconstruct_inner_rows(const Mat& m, uint16_t* dst, cudaStream_t s) {
    (void) m;
    (void) dst;
    (void) s;
    not_built("reconstruct_inner_rows");
}
void rows_rot_f16(const float* x, int ldx, const int* idx, int nrows, int ncols, const uint16_t* suh, uint16_t* out,
                  cudaStream_t s) {
    (void) x;
    (void) ldx;
    (void) idx;
    (void) nrows;
    (void) ncols;
    (void) suh;
    (void) out;
    (void) s;
    not_built("rows_rot_f16");
}
void rows_rot_out(float* y, int ldy, int nrows, int ncols, const uint16_t* svh, cudaStream_t s) {
    (void) y;
    (void) ldy;
    (void) nrows;
    (void) ncols;
    (void) svh;
    (void) s;
    not_built("rows_rot_out");
}
void reconstruct_inner(const Mat& m, uint16_t* dst, cudaStream_t s) {
    (void) m;
    (void) dst;
    (void) s;
    not_built("reconstruct_inner");
}
void rows_f16(const float* x, int ldx, const int* idx, int nrows, int ncols, uint16_t* out, cudaStream_t s) {
    (void) x;
    (void) ldx;
    (void) idx;
    (void) nrows;
    (void) ncols;
    (void) out;
    (void) s;
    not_built("rows_f16");
}
bool legacy_f32() { return false; }
bool prefer_legacy_f32(int dev) {
    (void) dev;
    return false;
}
void legacy_decode_table(int cb, bool f32, float* d_out, cudaStream_t s) {
    (void) cb;
    (void) f32;
    (void) d_out;
    (void) s;
    not_built("legacy_decode_table");
}
int launch_errors() { return 0; }

}  // namespace strata::kernels::exl3
