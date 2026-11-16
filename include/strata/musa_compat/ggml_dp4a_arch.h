#pragma once
// Included (via -include) by the ggml MMQ translation units of the MUSA build only.
//
// ggml's device code picks its quantized matmul implementation by tier with `#if __CUDA_ARCH__ >= 610`:
// below it the pure-FMA "pascal older" vec_dot, at or above it the __dp4a byte-dot path.  mcc defines no
// __CUDA_ARCH__ (MUSA has __MUSA_ARCH__ instead), so both the device and the host config chose the slow
// tier, and the host choice has to match the device one or the shared-memory layout diverges.
//
// MUSA has __dp4a (measured on this hardware), so the whole MMQ compile is pinned to that tier: this
// header is included only by those translation units, and only in the device pass (__MUSA_ARCH__ is
// defined there alone, exactly as __CUDA_ARCH__ is on nvcc).  cmake/..'s MMQ branch sets the host side to
// the same number (ggml_cuda_host.cu's d.cc).
#if defined(__MUSA_ARCH__) && !defined(__CUDA_ARCH__)
#define __CUDA_ARCH__ 610
#endif
