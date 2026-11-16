#pragma once
// Device functions the ported kernels use that MUSA does not spell itself.
// (__shfl_*_sync, __ballot_sync, __ldg, __ldcg, __syncwarp, __dp4a and the fp16/bf16 builtins
// all exist in MUSA's headers with CUDA's names - nothing to add for them.)
#ifndef wall_clock64
// CUDA 12 nanosecond wall-clock read (HIP spells it for gfx11). MUSA has clock64 (cycles); the sources use this
// only for a profiler timestamp, where a monotone counter is what matters.
#define wall_clock64() ((unsigned long long) clock64())
#endif
#ifndef __nanosleep
// CUDA's back-off hint has no MUSA equivalent; a no-op keeps the surrounding spin-wait loop correct
// (it just spins without the pause hint).
#define __nanosleep(ns) ((void)0)
#endif

#if defined(__MUSACC__)
// MUSA's __ldcg covers the 64/16/8-bit types and the vectors, but not unsigned int - which the glm-dsa
// kernels use for their cross-block counters (the HIP build supplies the same overload from its own
// hip_compat layer).  A volatile load is the same "read L2, not a stale L1 line" access.
__device__ __forceinline__ unsigned int __ldcg(const unsigned int* p) {
    return *(volatile const unsigned int*) p;
}
#endif

#if defined(__MUSACC__)
// ... and not float, which the GLM GEMVs read their published partials through.
__device__ __forceinline__ float __ldcg(const float* p) {
    return *(volatile const float*) p;
}
#endif
