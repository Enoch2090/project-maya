#pragma once
// MUSA's bfloat16 type is __mt_bfloat16/__mt_bfloat162; the sources use CUDA's __nv_* spelling.
#include <musa_bf16.h>
#define __nv_bfloat16 __mt_bfloat16
#define __nv_bfloat162 __mt_bfloat162
