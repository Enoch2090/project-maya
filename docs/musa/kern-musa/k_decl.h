#pragma once
#include <stdint.h>
// device tables (filled by host at startup) + shared enums
enum { T_IQ2XXS = 0, T_IQ2S = 1, T_IQ3XXS = 2 };

__device__ uint64_t D_IQ2XXS[256];   // iq2xxs_grid
__device__ uint64_t D_IQ2S[1024];    // iq2s_grid
__device__ uint32_t D_IQ3XXS[256];   // iq3xxs_grid
__device__ uint64_t D_KSIGNS64[128]; // 7-bit sign idx -> 8 bytes 0x00/0xFF
__device__ uint64_t D_SGB[256];      // 8-bit sign mask -> 8 bytes 0x00/0xFF
__device__ uint8_t D_KSB[128];       // 7-bit sign idx -> 8-bit sign mask

static inline int block_bytes_of(int ty) { return ty == T_IQ2XXS ? 66 : ty == T_IQ2S ? 82 : 98; }
