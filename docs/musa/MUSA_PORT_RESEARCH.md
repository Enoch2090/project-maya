# MUSA 后端移植调研：给 Maya 照抄的做法

调研日期 2026-11-15。目标机 = MTT S4000 + MUSA SDK 5.2.0，本机可编译可跑 kernel，标"[实测]"的结论来自这台卡上的实际编译/运行，其余来自公开源码与文档。查不到的写在 §9。

## 0. 结论先行

1. **有现成官方后端可抄，结构与我们的计划一致。** llama.cpp 的 MUSA 后端 = `ggml/src/ggml-cuda/*.cu` 原样复用 + 162 行宏表 `vendors/musa.h` + 160 行 `ggml-musa/CMakeLists.txt`。kernel 源码一行没改，只换 runtime 符号和编译 flag。→ [ggml-musa](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-musa)、[vendors/musa.h](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vendors/musa.h)、[CMakeLists.txt](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-musa/CMakeLists.txt)
2. **移植面比预想小。** GitHub 全仓代码搜索 `GGML_USE_MUSA` 只命中 **24 个文件**；主文件 `ggml-cuda.cu`（6970 行）里只有 6 处条件编译。对我们的 14 万行来说，`musa_compat` 的接口工作量大致就是抄一份这个头文件。
3. **最危险的坑不是 API 而是 `__CUDA_ARCH__`。** mcc 在 device pass 只定义 `__MUSA_ARCH__`、不定义 `__CUDA_ARCH__`，于是所有 `#if __CUDA_ARCH__ >= X` 恒为假：**kernel 被编成空函数却照样 launch**。llama.cpp 因此在 MUSA 上 q8_0 反量化全出 NaN，到 2026-09 才修（[PR #29508](https://github.com/ggml-org/llama.cpp/pull/29508)）。Maya 里任何按 arch 分档的 kernel 都会中这一枪。
4. 解法：device pass 里把 `__CUDA_ARCH__` 定义成最高档 1300，再逐个关掉会因此被误开的 NVIDIA 专属特性（§4）。
5. **warp**：MUSA SDK 5.2 提供 CUDA 同名的 `__shfl_*_sync / __ballot_sync / __all_sync / __any_sync / __activemask / __syncwarp`（[官方 Warp 函数页](https://docs.mthreads.com/musa-sdk/version-5.2.0/programming_guide/musa_cpp_syntax/warp_functions)），llama.cpp 一行未改。但 S4000 的**硬件 warp 宽度 128**（[实测] `prop.warpSize=128`），CUDA 风格内建仍按 32 lane 逻辑 warp 工作（[实测] 见 §5）。→ 规则：kernel 里永远按 32 lane 写，绝不用 `warpSize` 或 `prop.warpSize` 做适配。
6. **S4000 关键参数 [实测]**：56 个 MP、sharedMemPerBlock **72 KiB**、47.9 GB 显存、D2D 拷贝 **705 GB/s**（峰值 768）、`canMapHostMemory=1`、L2 24 MB。网上说的"共享内存只有 28 KiB"是 S80/S3000（QY1）的事，**S4000 放得下 48 KiB 的 MMQ tile**。
7. **可能省掉整个 `musa_compat` 的另一条路**：MUSA SDK 自带编译期兼容层 **MUSA Mapping**（`libMusaMapping.so` + ~3 万条映射表 + CUDA 头文件映射），加 `-fplugin` 就能直接编未修改的 CUDA 源码（[文档](https://docs.mthreads.com/musa-sdk/version-5.2.0/toolkits/musa_mapping/)）。Maya 体量大，建议先花半天用它做可行性验证，再决定要不要自写兼容头。
8. **未找到**：Maya 仓库无任何 MUSA 相关代码/issue/PR；S4000(mp_22) 上 llama.cpp 的官方 CI 覆盖（CI 只跑 S5000/mp_31）。

## 1. llama.cpp MUSA 后端：确实存在，状态与时间线

- 代码位置：`ggml/src/ggml-musa/`（只有 `CMakeLists.txt`、`mudnn.cu`、`mudnn.cuh`），宏表在 `ggml/src/ggml-cuda/vendors/musa.h`，与 `vendors/cuda.h`、`vendors/hip.h` 平级——**这就是 llama.cpp 版的 `hip_compat/`**。
- 同一份 kernel 编两种后端：CMake 用 `file(GLOB ../ggml-cuda/*.cu)` 把整个 CUDA 后端拉进 MUSA 目标（含 fattn / mmq / mmvq 的 template instances）。
- 构建：`cmake -B build -DGGML_MUSA=ON`；默认 `MUSA_ARCHITECTURES="22;31"`（S4000 + S5000），`21`（S70/S80/S3000）已弃用（[PR #30203](https://github.com/ggml-org/llama.cpp/pull/30203)）。见 [docs/build.md#musa](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#musa)。
- 时间线：2025-03 首个 WIP [#12383](https://github.com/ggml-org/llama.cpp/pull/12383) → 2025-10 编译 flag 调优 [#16265](https://github.com/ggml-org/llama.cpp/pull/16265) → 2026-06 补 muBLAS batched 映射 [#25033](https://github.com/ggml-org/llama.cpp/pull/25033) → 2026-07 小显存设备关 MMQ [#26141](https://github.com/ggml-org/llama.cpp/pull/26141) → 2026-09 修 PH1 算子 [#29193](https://github.com/ggml-org/llama.cpp/pull/29193) → 2026-09 定义 `__CUDA_ARCH__` [#29508](https://github.com/ggml-org/llama.cpp/pull/29508) → 2026-10 默认归档 22/31。
- 成熟度证据：PR #29508 在 MTT S5000 + SDK 5.2.0 上 `test-backend-ops` **12507/12507** 全过。但**这些 PR 与 CI 都只在 mp_31（S5000）上跑**，mp_22（S4000）只有编译覆盖。
- **需要反驳的一条社区说法**：有博客称"上游 llama.cpp 的 MUSA 后端是个空壳（只有 memcpy，跑不了真推理）"（[来源](https://blog.meowhead.cn/posts/qwen3-next-80b-deployment/)）。这与源码不符（CMake 把全部 kernel 编进去，且 #29193/#29508 在真卡上跑全量算子测试），更像 2025 年早期状态或该机器"宿主驱动 2.7 装不上 5.x SDK"的误判。该文同时给出一条真坑：**宿主驱动版本是硬约束**，SDK 5.x 需要驱动 5.x。
- 另一套东西别混淆：摩尔线程官方另有 **Llama.cpp-MUSA-M1000**（M1000 端侧卡，[文档](https://docs.mthreads.com/llama.cpp-musa-m1000/Llama.cpp-MUSA-M1000/)），与 S4000 无关；官方给 S4000 的推理栈是 vLLM-MUSA / SGLang-MUSA，没有 S4000 版 llama.cpp 发行包。

## 2. CMake 集成：llama.cpp 原文（节选，行的顺序略有整理）

```cmake
set(MUSA_PATH $ENV{MUSA_PATH})            # 默认 /usr/local/musa
set(CMAKE_CXX_COMPILER "${MUSA_PATH}/bin/clang++")
list(APPEND CMAKE_MODULE_PATH "${MUSA_PATH}/cmake")
find_package(MUSAToolkit)

file(GLOB GGML_SOURCES_MUSA "../ggml-cuda/*.cu")   # 整个 CUDA 后端
set_source_files_properties(${GGML_SOURCES_MUSA} PROPERTIES LANGUAGE CXX)
foreach(SOURCE ${GGML_SOURCES_MUSA})
  foreach(ARCH ${MUSA_ARCHITECTURES})               # 默认 22;31
    set(COMPILE_FLAGS "${COMPILE_FLAGS} --cuda-gpu-arch=mp_${ARCH}")
  endforeach()
  set(COMPILE_FLAGS "-Od3 -fno-strict-aliasing -ffast-math -fsigned-char -x musa -mtgpu -fmusa-flush-denormals-to-zero ${COMPILE_FLAGS}")
  set_property(SOURCE ${SOURCE} PROPERTY COMPILE_FLAGS ${COMPILE_FLAGS})
endforeach()

ggml_add_backend_library(ggml-musa ${GGML_HEADERS_MUSA} ${GGML_SOURCES_MUSA})
target_link_libraries(ggml-musa PRIVATE MUSA::musart MUSA::mublas MUSA::musa_driver)
```

要点：
- **不用 `enable_language()`**：MUSA 没有 CMake 语言。靠 `find_package(MUSAToolkit)` 拿 `MUSA::musart / MUSA::mublas / MUSA::musa_driver`（静态版 `musart_static`/`mublas_static`）。
- `.cu` 声明成 `LANGUAGE CXX`，再挂 `-x musa -mtgpu`，等于"让 clang++ 当驱动按 MUSA 语言编"。VMM/api 是否可用决定要不要链 `musa_driver`（MUSA 支持 VMM，llama.cpp 设备初始化会打印 `VMM: yes`）。
- 编译器在**子目录里** `set(CMAKE_CXX_COMPILER ...)`；CMake 在 `project()` 之后改 CXX 编译器通常不生效，Maya 更稳的是 configure 时直接传 `-DCMAKE_CXX_COMPILER=/usr/local/musa/bin/mcc`（官方文档也推荐用 `mcc_wrapper` 替代 nvcc 的角色）。
- flag 含义：`-Od3` 是 mcc 的调试档（llama.cpp 用它提速）；`-ffast-math` + `-fmusa-flush-denormals-to-zero` ≈ nvcc `--use_fast_math`（[实测] mp_22 会警告"硬件不支持 FTZ，无法完全保证"）；`--cuda-gpu-arch` 可重复传做多架构 fatbin（[实测] 通过）。整串 flag [实测] 被 mcc 5.2.0 接受。
- 官网对 CMake 的说法（[MUSA Mapping 文档](https://docs.mthreads.com/musa-sdk/version-5.2.0/toolkits/musa_mapping/)）：**不要启用系统 CMake 的 CUDA 语言**（会注入 `-x cu`，`--offload-arch=mp_XX` 会被当 CUDA arch 报错）；把 `CMAKE_CXX_COMPILER` 设成 `mcc_wrapper` 时 `CMAKE_C_COMPILER` 必须保持 gcc（mcc 会把 `.c` 按 C++ 处理，C 编译器探测失败）；`set(CMAKE_CUDA_STANDARD)` 不生效，用 `CMAKE_CXX_STANDARD`。
- Maya 落法（对应 `cmake/hip_backend.cmake`）：把 `enable_language(HIP)` 换成"mcc 作为 CXX 编译器 + find_package(MUSAToolkit)"；源文件属性从 `LANGUAGE HIP` 换成 `LANGUAGE CXX` + 上面的 `-x musa` flag 串；`-include include/strata/musa_compat/cuda_runtime.h` 保留。

## 3. 兼容头 / 宏映射表

`vendors/musa.h` 的骨架（节选，整份 162 行，[原文](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vendors/musa.h)）：

```c
#include <musa_runtime.h>   // + musa.h / mublas.h / musa_bf16.h / musa_fp16.h
#ifdef __MUSA_ARCH__
#define __CUDA_ARCH__ 1300  // GGML_CUDA_CC_RUBIN，见 §4
#endif

#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define CUBLAS_COMPUTE_32F_FAST_16F MUBLAS_COMPUTE_32F_FAST_16F
#define CUBLAS_TF32_TENSOR_OP_MATH  MUBLAS_TENSOR_OP_MATH
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP MUBLAS_GEMM_DEFAULT
#define CUDA_R_16F  MUSA_R_16F
#define CUDA_R_16BF MUSA_R_16BF
#define cublasComputeType_t cudaDataType_t
#define cublasGemmEx mublasGemmEx
#define cublasSgemmStridedBatched mublasSgemmStridedBatched
#define cudaDataType_t musaDataType_t
#define cudaMalloc musaMalloc
#define cudaMemcpyAsync musaMemcpyAsync
#define cudaStream_t musaStream_t
#define cudaError_t musaError_t
#define cudaGetLastError musaGetLastError
#define cudaHostRegister musaHostRegister
#define cudaFuncSetAttribute musaFuncSetAttribute
#define cudaGraphExecUpdate musaGraphExecUpdate  // 以及整组 Graph/VMM 的 CU*→MU*、cu*→mu*

typedef __mt_bfloat16 nv_bfloat16;
typedef __mt_bfloat162 nv_bfloat162;
```

非显而易见的差异：
- **muBLAS 没有独立的 computeType**：`cublasComputeType_t` 被 `#define` 成 `cudaDataType_t`，`CUBLAS_COMPUTE_16F` → `CUDA_R_16F`；只有 `MUBLAS_COMPUTE_32F_FAST_16F` 这类存在。Maya 里所有 `cublasGemmEx(..., CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F, ...)` 的写法要么照上面宏化，要么逐处改。
- **bf16 类型名**：`__mt_bfloat16` / `__mt_bfloat162`；[实测] `__hfma2` + `__mt_bfloat16` + `__bfloat162float` 在 mp_22 上编译通过。
- **VMM / Driver API / Graph**：`CUdeviceptr`、`cuMemCreate`、`cuMemMap`、`cudaGraphInstantiate` 等都有对应映射，说明 MUSA 这些面基本齐。
- `cub` 直接可用：llama.cpp 用 `GGML_CUDA_USE_CUB` 在 MUSA 上照样启用。

给 Maya 的对应关系：`include/strata/hip_compat/` 现在是"伪 CUDA 头 + `#define cudaXxx hipXxx`"，并在 CMake 里 `-include .../cuda_runtime.h` 强制注入（见 [hip_backend.cmake](https://github.com/mw00/project-maya/blob/main/cmake/hip_backend.cmake)）。MUSA 版 = 同一批文件复制成 `musa_compat/`、映射目标换成 musa、`-include` 换路径。**但** `hip_compat/intrinsics.hpp` 里那套 wave64→逻辑 32 lane 映射在 MUSA 上不需要（[实测] MUSA 内建本来就是 32 lane 语义），可以直接直通。

## 4. 编译期最危险的一枪：`__CUDA_ARCH__`

- 事实：mcc 的 device pass 只定义 `__MUSA_ARCH__`（[实测] 值 = mp_21/22/31 → 210/220/310），`__CUDA_ARCH__` 不存在。于是共享源码里的 `#if __CUDA_ARCH__ >= 800` 全部取假 → kernel body 变空、但 launch 照常成功，症状是**数值全错而不是编译/运行报错**。
- llama.cpp 的修法：device pass 里把它定成 1300（最高档），再把因"假装是新架构"而被打开的 NVIDIA 专属特性逐个关掉：

```c
#ifdef __MUSA_ARCH__
#define __CUDA_ARCH__ 1300
#endif
```

- 必须显式关掉的特性和开关（[源码 common.cuh](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/common.cuh)）：`TURING_MMA_AVAILABLE`、`AMPERE_MMA_AVAILABLE`、`CP_ASYNC_AVAILABLE`、`__reduce_add_sync`、Ampere/Pascal-DP4A 的 MMQ/MMVQ 配置、`__bfloat1622float2`。llama.cpp 一律写成 `!defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= ...`。
- host 端没有 `__MUSA_ARCH__`，只能用运行时 cc。llama.cpp 把 MUSA 设备编码成 `0x100000 + major*0x100 + minor*0x10`：QY1=0x100210（S80/S3000）、QY2=0x100220（**S4000**）、PH1=0x100310（S5000），再按档位 gate 特性。
- S4000 上的实际 gate 结果（读自 common.cuh）：fast fp16 ✅（≥QY2）、fp16 MMA ✅、fast bf16 ❌、bf16 MMA ❌（只对 PH1 开）、FlashAttention ✅（`__MUSA_ARCH__ >= 220`）、CUDA graph 可选（默认关）。

## 5. warp 内建与"逻辑 32 lane"（[实测] 本机 S4000）

| 探针 | 结果 | 含义 |
|---|---|---|
| `prop.warpSize` / `musaInfo warpSize` | **128** | 硬件 SIMD 宽度，别拿它当"warp" |
| `__shfl_sync(0xffffffff, t, 0)`（block=128） | 0 / 32 / 64 / 96 | 按 32 lane 分组取值，与 CUDA 语义一致 |
| `__shfl_xor_sync(..., width=32)` 归约 | 496 / 1520 / 2544 / 3568 | 每组 32 个线程求和，正确 |
| `__ballot_sync(0xffffffff, ...)` | `0x55555555`（32 位） | mask 只覆盖 32 lane |
| 官方文档 | MP21/MP22 硬件 warp = **128**、MP31 = 32，并建议用 `warpSize` | 与实测的"内建仍是 32 lane"不同层，别混用 |

llama.cpp 的写法（简化后，[源码](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/common.cuh)）：

```cpp
#define WARP_SIZE 32                 // 硬编码 32，不用 warpSize
template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
    for (int offset = width/2; offset > 0; offset >>= 1)
        x += __shfl_xor_sync(0xffffffff, x, offset, width);   // 显式传 width
    return x;
}
```

已知会挂的点：`TOPK_MOE` 曾在 `__syncthreads()` 里死锁（[PR #29193](https://github.com/ggml-org/llama.cpp/pull/29193)）；`half2 → uint32 mask` 比较要 `MUSART_VERSION >= 40300`（SDK 4.3+，我们 5.2 满足）；mask 参数只接受 32 位（0xffffffff），别传 128 位。Maya 侧动作：`grep -rn warpSize` 确认没把硬件宽度当逻辑 warp；`hip_compat/intrinsics.hpp` 的 wave64 逻辑在 MUSA 不要照搬。

## 6. 库差异 / graph / pinned memory / muDNN

| 面 | CUDA | MUSA | 注意 |
|---|---|---|---|
| BLAS | cuBLAS/cuBLASLt | muBLAS 1.13.0 / mublasLt（[实测] 本机 lib 版本） | 无独立 computeType，见 §3；batched 映射是后补的（#25033） |
| DNN | cuDNN | muDNN | llama.cpp 注释："mudnn 还没有静态库"，只动态链；可用 `mudnn::Unary::IDENTITY` 加速 D2D copy（可选，不阻塞首版） |
| Graph | cudaGraph* | musaGraph*（[文档](https://docs.mthreads.com/musa-sdk/version-5.2.0/programming_guide/features/musa_graphs)） | llama.cpp `GGML_MUSA_GRAPHS` 默认 **OFF**，注释写 "experimental, unstable"；#29193 才让它能编。**第一版别开** |
| pinned/mapped host mem | cudaHostRegister / cudaHostGetDevicePointer | 同名映射（[实测] S4000 `canMapHostMemory=1`） | Maya 的 pinned staging/doorbell 逻辑可沿用；映射指针 API 需实测 |
| warp 宽度 | 32 | 逻辑 32 / 硬件 128 | §5 |
| CUB / CUTLASS | cub + CUTLASS/CuTe | cub 有（[实测] SDK 内含 /usr/local/musa/include/cub，llama.cpp 的 CUB 路径在 MUSA 上启用）；mutlass/mute 本机未装 | 要 tensor core 模板层得另装 mute，或走 muDNN/内建；llama.cpp 在 MUSA 上直接关 MMA |

## 7. MTT S4000 参数与速度量级

官方规格（[产品规格书](https://docs.mthreads.com/s4000/s4000-doc-online/product_specifications/)）：芯片 QY102AA-800、1.5 GHz、48 GB GDDR6 384-bit、**768 GB/s**、向量 FP32/FP16 25 TFLOPS、TF32 50 TFLOPS、**FP16/BF16 张量 100 TFLOPS**、INT8 200 TOPS、PCIe 5.0 x16、450 W。

[实测] 本机 `musaInfo`：compute capability **2.2**（= mp_22）、`multiProcessorCount` **56**、`maxThreadsPerMultiProcessor` 6144、`sharedMemPerBlock` **72 KiB**、`sharedMemPerMultiprocessor` 72 KiB、`warpSize` 128、L2 **24 MB**、`regsPerBlock` 131072、显存 47.90 GB、`canMapHostMemory` 1。D2D 拷贝（512 MiB × 20 次）**705 GB/s**，约为峰值的 92%。

速度量级推算：decode 受显存带宽限制。7B 的 Q4_K 约 4 GB 权重/步 → 上限 ≈ 705/4 ≈ **176 tok/s**，工程实现系数 0.3–0.5 → 现实 **50–90 tok/s** 量级。可对照的第三方数据：MTT-S4000 上 RWKV-6-7B fp16 **30.4 tok/s**、14B fp16 16.2 tok/s（[RWKV 官方跑分表](https://www.rwkv.cn/docs/RWKV-Performance-Data/others)）；官方 M1000（不同芯片）Qwen3.5-9B Q4_K decode 8.9 tps（[性能页](https://docs.mthreads.com/llama.cpp-musa-m1000/Llama.cpp-MUSA-M1000/AIBook/AIBook1.5.0/performance/)）。选档建议：**先 Q4_K/Q5_K**（官方 llama.cpp-musa 也只针对 Q4_K 做优化），别把 bf16 当快路径（S4000 上 llama.cpp 不开 bf16 MMA）。

## 8. 给 Maya 的施工顺序

1. `cmake/musa_backend.cmake`：照 `hip_backend.cmake` 骨架，`enable_language(HIP)` → "mcc 作 CXX 编译器 + `find_package(MUSAToolkit)`"；源文件 `LANGUAGE CXX` + `-x musa -mtgpu --cuda-gpu-arch=mp_22`；保留 `-include include/strata/musa_compat/cuda_runtime.h`。
2. `include/strata/musa_compat/`：复制 `hip_compat` 的 4 个头，映射目标换 musa；宏表先照抄 §3，再按 Maya 实际调用点补。
3. `grep -rn '__CUDA_ARCH__'`：逐个确认门限；按 §4 定义 + 关特性。这一步不做，后面所有数值错误都会看起来像"kernel 写错了"。
4. `grep -rn 'cublas\|cudnn\|cudaGraph\|cuMem\|cudaHost'`：对照 §3/§6 补映射表空洞。
5. `grep -rn '__device__'`：**跨 TU 的 `__device__` 函数要么挪进头文件/`__forceinline__`，要么开 `-fgpu-rdc` 走 5 步设备链接**（[实测] 不开 rdc 直接 `lld: error: undefined protected symbol`；mcc 文档那套 `--musa-device-only` + `lld -shared` + `-fcuda-include-gpubinary` 流程在本机跑通，输出 `v=42`）。llama.cpp 不用 rdc 是因为它的 device 函数基本都在头里。
6. 先跑通三件事再谈性能：一个 GEMM/GEMV、Q4_K 的 dequant+dot、一次 attention（可拿 ggml 的 `test-backend-ops` 当现成核对器）。
7. 已知雷区自查：空格子 kernel（§4）、warpSize=128（§5）、FTZ 无硬件支持（§7）、graph 不稳（§6）、muBLAS computeType（§3）、28 KiB 共享内存只属于 QY1 不属于 S4000（§7）。

## 9. 未找到 / 未验证

- Maya 仓库的 MUSA 相关 issue/PR/代码：**0 条**（issues、PR、code search 均为 0）→ 全新工作，无内部先例。可类推的只有 [docs/AMD_MAYA.md](https://github.com/mw00/project-maya/blob/main/docs/AMD_MAYA.md)：非 NVIDIA 移植的主要代价集中在"架构档位判断、矩阵核心有无、warp 宽度、pinned 内存能否复用"四类问题。
- llama.cpp 的 MUSA CI 是否覆盖 mp_22（S4000）：**未找到**（CI 与相关 PR 都跑 mp_31）。
- S4000 上 llama.cpp 的官方跑分：**未找到**（官方只发布 M1000 的性能表）。网上"S4000 7B Q4_K_M 46–50 tok/s"出自内容农场，未标注版本，仅作量级参考。
- mcc 的 clang 基线：**[实测] 确认 clang 14.0.0 / mcc 5.2.0**（`mcc --version`），与任务说明一致。C++20 可用，C++23 特性需自行试。
- tensor core 模板库 mute/mutlass：SDK 默认不含（本机实测），官方 Mapping 文档里有 cute/ → mute/ 的映射条目，说明是独立发行包。llama.cpp 没走这条路，MUSA 上一律关 MMA。
- musify（文本级 CUDA→MUSA）在 14 万行工程上的效果：**无公开数据**（[文档](https://docs.mthreads.com/musa-sdk/version-5.2.0/toolkits/musify/)：纯文本匹配、~3 万条映射、支持 `MUSIFY_EXCL_LINE/START/STOP` 排除标记、可自定义映射表）。
- MUSA Mapping 在 ggml 风格代码上的成功率：**无公开案例**（llama.cpp 没用它，而是自己写 `vendors/musa.h`）。

## 10. 链接汇总

| 用途 | 链接 |
|---|---|
| llama.cpp MUSA 后端目录 | https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-musa |
| MUSA 宏表（抄这个） | https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vendors/musa.h |
| MUSA CMake（抄这个） | https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-musa/CMakeLists.txt |
| 特性 gate / warp 实现 | https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/common.cuh |
| `__CUDA_ARCH__` 那个坑 | https://github.com/ggml-org/llama.cpp/pull/29508 |
| MUSA SDK 5.2.0 编程指南 | https://docs.mthreads.com/musa-sdk/version-5.2.0/programming_guide/ |
| mcc 编译器手册 | https://docs.mthreads.com/musa-sdk/version-5.2.0/toolkits/mcc_compiler/ |
| MUSA Mapping（零改源码方案） | https://docs.mthreads.com/musa-sdk/version-5.2.0/toolkits/musa_mapping/ |
| musify | https://docs.mthreads.com/musa-sdk/version-5.2.0/toolkits/musify/ |
| S4000 规格书 | https://docs.mthreads.com/s4000/s4000-doc-online/product_specifications/ |
| 官方 llama.cpp-MUSA-M1000 | https://docs.mthreads.com/llama.cpp-musa-m1000/Llama.cpp-MUSA-M1000/ |
| vLLM-MUSA（S4000 官方栈参考） | https://github.com/MooreThreads/vllm-musa |
| torch_musa（PyTorch 移植范例） | https://github.com/MooreThreads/torch_musa |
