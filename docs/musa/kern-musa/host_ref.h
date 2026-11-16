#include <stdio.h>
#include "k_decl.h"
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <vector>
#include <string>
#include <musa_runtime.h>
#include <musa_fp16.h>
#include "tables.h"
#include "k_gemv.h"

#define CK(x) do{ musaError_t _e=(x); if(_e!=musaSuccess){ printf("MUSA ERR %s line %d\n", musaGetErrorName(_e), __LINE__); exit(1);} }while(0)

static const char* SHARD1 = "/root/autodl-tmp/models/Maya-S-v2-IQ2_XXS/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf";

struct Shape { int ty; int n_out; int k; int row_bytes; int ng; const char* name; };

static float h2f(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float v;
    if (e == 0) v = ldexpf((float)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = ldexpf((float)(1024 + m), (int)e - 25);
    return s ? -v : v;
}

static std::vector<uint8_t> pread_blob(const char* path, long long off, size_t n) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { printf("open failed %s\n", path); exit(1); }
    std::vector<uint8_t> v(n);
    size_t got = 0;
    while (got < n) {
        ssize_t r = pread(fd, v.data() + got, n - got, off + (long long)got);
        if (r <= 0) { printf("pread failed at %lld\n", off + (long long)got); exit(1); }
        got += (size_t)r;
    }
    close(fd);
    return v;
}

// CPU reference: float-x dequant dot (yf) and int8-x emulation (yi, matches kernel algorithm)
static void cpu_ref(const Shape& sh, const uint8_t* blob, const float* x,
                    const int8_t* xq, const float* xscale,
                    std::vector<float>& yf, std::vector<float>& yi)
{
    yf.assign(sh.n_out, 0.f);
    yi.assign(sh.n_out, 0.f);
    const int nb = sh.k / 256;
    uint64_t q64; (void)q64;
    for (int r = 0; r < sh.n_out; ++r) {
        const uint8_t* wrow = blob + (size_t)r * sh.row_bytes;
        double af = 0.0, ai = 0.0;
        for (int sb = 0; sb < nb; ++sb) {
            const float* xb = x + (size_t)sb * 256;
            const int8_t* xqb = xq + (size_t)(sb * 8) * 36;  // padded layout, base of superblock
            const float* xsc = xscale + (size_t)sb * 8;
            if (sh.ty == T_IQ2XXS) {
                const uint8_t* p = wrow + (size_t)sb * 66;
                float d = h2f(*(const uint16_t*)p);
                for (int sbl = 0; sbl < 8; ++sbl) {
                    uint64_t q; memcpy(&q, p + sbl * 8, 8);
                    uint32_t a0 = (uint32_t)q, a1 = (uint32_t)(q >> 32);
                    float db = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
                    int isum = 0;
                    for (int grp = 0; grp < 4; ++grp) {
                        uint64_t g = H_IQ2XXS_GRID[(a0 >> (8*grp)) & 255];
                        uint8_t sgn = H_KSIGNS_IQ2XS[(a1 >> (7*grp)) & 127];
                        for (int j = 0; j < 8; ++j) {
                            int gv = (int)((g >> (8*j)) & 255);
                            int neg = (sgn >> j) & 1;
                            int pos = sbl * 32 + grp * 8 + j;
                            float wv = db * (float)gv * (neg ? -1.f : 1.f);
                            af += (double)wv * xb[pos];
                            isum += (neg ? -gv : gv) * (int)xqb[sbl * 36 + grp * 8 + j];
                        }
                    }
                    ai += (double)isum * (double)(db * xsc[sbl]);
                }
            } else if (sh.ty == T_IQ2S) {
                const uint8_t* p = wrow + (size_t)sb * 82;
                float d = h2f(*(const uint16_t*)p);
                for (int sbl = 0; sbl < 8; ++sbl) {
                    int qh = p[66 + sbl], sc = p[74 + sbl];
                    float db0 = d * (0.5f + (float)(sc & 0xf)) * 0.25f;
                    float db1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
                    for (int grp = 0; grp < 4; ++grp) {
                        int gi = (int)p[2 + sbl*4 + grp] | ((qh << (8 - 2*grp)) & 0x300);
                        uint64_t g = H_IQ2S_GRID[gi];
                        uint8_t sgn = p[34 + sbl*4 + grp];
                        float db = (grp < 2) ? db0 : db1;
                        int isum = 0;
                        for (int j = 0; j < 8; ++j) {
                            int gv = (int)((g >> (8*j)) & 255);
                            int neg = (sgn >> j) & 1;
                            int pos = sbl * 32 + grp * 8 + j;
                            float wv = db * (float)gv * (neg ? -1.f : 1.f);
                            af += (double)wv * xb[pos];
                            isum += (neg ? -gv : gv) * (int)xqb[sbl * 36 + grp * 8 + j];
                        }
                        ai += (double)isum * (double)(db * xsc[sbl]);
                    }
                }
            } else {
                const uint8_t* p = wrow + (size_t)sb * 98;
                float d = h2f(*(const uint16_t*)p);
                for (int sbl = 0; sbl < 8; ++sbl) {
                    uint32_t aux; memcpy(&aux, p + 66 + sbl * 4, 4);
                    float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
                    int isum = 0;
                    for (int grp = 0; grp < 4; ++grp) {
                        uint8_t sgn = H_KSIGNS_IQ2XS[(aux >> (7*grp)) & 127];
                        uint32_t g1 = H_IQ3XXS_GRID[p[2 + sbl*8 + 2*grp]];
                        uint32_t g2 = H_IQ3XXS_GRID[p[2 + sbl*8 + 2*grp + 1]];
                        for (int j = 0; j < 4; ++j) {
                            int gva = (int)((g1 >> (8*j)) & 255);
                            int gvb = (int)((g2 >> (8*j)) & 255);
                            int na = (sgn >> j) & 1, nb2 = (sgn >> (j+4)) & 1;
                            int pos = sbl * 32 + grp * 8 + j;
                            float wva = db * (float)gva * (na ? -1.f : 1.f);
                            float wvb = db * (float)gvb * (nb2 ? -1.f : 1.f);
                            af += (double)wva * xb[pos] + (double)wvb * xb[pos+4];
                            isum += (na ? -gva : gva) * (int)xqb[sbl * 36 + grp * 8 + j] + (nb2 ? -gvb : gvb) * (int)xqb[sbl * 36 + grp * 8 + j + 4];
                        }
                    }
                    ai += (double)isum * (double)(db * xsc[sbl]);
                }
            }
        }
        yf[r] = (float)af;
        yi[r] = (float)ai;
    }
}
