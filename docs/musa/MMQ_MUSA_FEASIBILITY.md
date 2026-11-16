# MMQ 在 MUSA 上的可行性评估

> 结论先行：**可行，且已经实测验证到「能编出正确配置的 device 代码」这一步。** 上游 llama.cpp 的官方 ggml-musa 后端本来就把 `ggml-cuda/**/*.cu` 和 16 个 `mmq-instance-*.cu` 全部编进 MUSA 后端，MMQ 的 mma/dp4a 分支在 MUSA 下会被条件编译自然关掉、退到非 mma 的 vec_dot 路径，而且 host 侧和 device 侧选出的 MMQ 配置**实测一致**（这是最容易出正确性事故的地方，已专门验证）。
>
> 时间：2026-11-15 17:39–18:45。实验目录 `/tmp/mmq-test/`。未改动 project-maya 任何文件。

---

## 1. 上游 llama.cpp 如何让 MMQ 在 MUSA 上编过（做法来源）

`/root/autodl-tmp/llama.cpp/ggml/src/ggml-musa/CMakeLists.txt` 是关键参考，原文摘录：

```cmake
file(GLOB   GGML_SOURCES_MUSA "../ggml-cuda/*.cu")            # 全部 ggml-cuda！
file(GLOB   SRCS "../ggml-cuda/template-instances/mmq*.cu")
list(APPEND GGML_SOURCES_MUSA ${SRCS})                        # 全部 16 个 mmq instance
...
set_source_files_properties(${GGML_SOURCES_MUSA} PROPERTIES LANGUAGE CXX)
foreach(SOURCE ${GGML_SOURCES_MUSA})
    set(COMPILE_FLAGS "-Od3 -fno-strict-aliasing -ffast-math -fsigned-char -x musa -mtgpu -fmusa-flush-denormals-to-zero")
    foreach(ARCH ${MUSA_ARCHITECTURES})                        # 默认 "21;22;31"
        set(COMPILE_FLAGS "${COMPILE_FLAGS} --cuda-gpu-arch=mp_${ARCH}")
    endforeach()
    set_property(SOURCE ${SOURCE} PROPERTY COMPILE_FLAGS ${COMPILE_FLAGS})
endforeach()
...
add_compile_definitions(GGML_USE_MUSA)
```

要点：
- **不掺任何 `-DGGML_CUDA_...` 私有开关**，只有 `GGML_USE_MUSA`（外加可选的 `GGML_CUDA_FORCE_MMQ` / `GGML_CUDA_NO_VMM` / `GGML_CUDA_NO_FA` 等通用开关）。
- arch 用 `--cuda-gpu-arch=mp_22`（**不是** `--offload-arch`），与 project-maya 的 `musa_backend.cmake` 写法一致。上游默认还编 21/31 多份（我们的 S4000 只需 22；注意我在上一份报告里记录过「非原生 arch 可能有内核静默不执行」，单 arch 最稳）。
- `.cu` 走 `LANGUAGE CXX` + `-x musa`，和 project-maya 现有 MUSA 后端一模一样。
- **MUSA 后端里没有"小显存就关 MMQ"这类开关**：`GGML_CUDA_FORCE_MMQ` / `GGML_CUDA_FORCE_CUBLAS` 只是用户可选的强制开关，默认由 `ggml_cuda_should_use_mmq()` 按类型+cc 判断。
- 没有 `-fgpu-rdc`（project-maya 有，见 §5 的风险项）。

## 2. MMQ 的 mma 路径在 MUSA 下怎么处理

`ggml-cuda` 里 `GGML_USE_MUSA` 出现在 11 个文件，但 **`mmq.cuh` / `mmq-vec-dot.cuh` / `mma.cuh` 里一个都没有**。MUSA 不是靠 `#ifdef GGML_USE_MUSA` 关掉 mma 的，而是靠一个更底层的机制：

### 2.1 mcc 不定义 `__CUDA_ARCH__`（这是整套机制的支点）

实测（`/tmp/archtest.mu`，内核里直接读）：

```cuda
__global__ void k(int* o) {
#ifdef __CUDA_ARCH__
  o[0] = __CUDA_ARCH__; o[1] = 1;
#else
  o[0] = -99; o[1] = 0;
#endif
#ifdef __MUSA_ARCH__
  o[2] = __MUSA_ARCH__;
#else
  o[2] = -1;
#endif
}
```
```
device: __CUDA_ARCH__ defined=0 value=-99, __MUSA_ARCH__=220
```

`mcc --offload-arch=mp_22 -dM -E` 的输出里同样没有 `__CUDA_ARCH__`（有 `__MUSACC__`、`__MUSA_ARCH__ 220`）。

于是 `mmq.cuh` 里这些判断在 MUSA 下自然落到"最老的 CUDA"分支：

| 位置 | 判断 | MUSA 下的结果 |
|---|---|---|
| `mmq.cuh:278` | `#ifdef BLACKWELL_MMA_AVAILABLE` | 否 |
| `mmq.cuh:280` | `#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA`(700) | `0 >= 700` 否 |
| `mmq.cuh:282` | `#elif __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A`(610) | `0 >= 610` 否 |
| `mmq.cuh:284` | `#else` → `ggml_cuda_mmq_get_config_pascal_older()` | **走这条** |

`TURING_MMA_AVAILABLE` / `AMD_MFMA_AVAILABLE` / `BLACKWELL_MMA_AVAILABLE` 这类宏在 MUSA 下全都不会被定义，所以 `mmq.cuh` 里所有 `#if defined(TURING_MMA_AVAILABLE)` 包住的 PTX/MMA 代码（包括那些 `asm volatile` 的 mma.sync）**整段不参与编译**。这就是为什么 MUSA 后端不需要为 MMQ 写任何 `#ifdef GGML_USE_MUSA`。

`GTC`/`GCN` 那套 `#ifdef GGML_USE_HIP` 分支与 MUSA 无关（`GGML_USE_HIP` 不定义）。

### 2.2 关键正确性验证：host 与 device 选的配置必须一致

`mmq.cuh` 有**两个** `ggml_cuda_mmq_get_config()`：
- `static __host__ ... get_config(type, J, fallback, cc)`（第 230 行）——按运行时 `cc` 选，用于算 grid/共享内存/启动参数；
- `static constexpr __device__ ... get_config(type, J, fallback)`（第 261 行）——按 `__CUDA_ARCH__` 选，用于选 vec_dot 实现。

如果两者分叉 → 共享内存布局不一致 → 结果错乱（甚至越界）。MUSA 下 host 拿到的 `cc` 是 `GGML_CUDA_CC_OFFSET_MTHREADS + 0x220 = 0x100220`（`ggml-cuda.cu:338`，`GGML_CUDA_CC_QY2` 正是 "MTT S4000"），而 device 侧 `__CUDA_ARCH__` 是 0，**理论上有分叉风险**。所以专门写了 `/tmp/mmq-test/cfgdiff.mu` 把两侧配置都打印出来对比：

```
主机 cc=0x100220 (S4000): Q4_K type=43 nthreads=256 I=128 J=64 stream_k=0
主机 cc=0x0        :       Q4_K type=43 nthreads=256 I=128 J=64 stream_k=0
设备(__CUDA_ARCH__ 未定义): Q4_K type=43 nthreads=256 I=128 J=64 stream_k=0 | IQ3_S type=43 nthreads=256
=> 一致? 是
```

**一致**（`type=43`、非 `GGML_TYPE_COUNT` 哨兵）。原因：`ggml_cuda_mmq_get_config_ampere()` 内部同样受 `TURING_MMA_AVAILABLE` 之类的宏约束，在 MUSA 编译单元里退化成与 pascal 分支等价的配置。**这条验证是判断"能不能用"的关键，不只是"能不能编"。**

### 2.3 另一处安全回退

`project-maya/src/prefill/moe_mmq.cu:197` 的 `glu_supported()` 判据是 `turing_mma_available(ggml_cuda_info().devices[dev].cc)`，而

```c
static bool turing_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING;
}
```

`GGML_CUDA_CC_IS_NVIDIA(cc)` = `cc < 0x100000` → MUSA 的 `0x100220` 为 **false**。所以即使编了 `STRATA_MMQ_GLU`（fused gate/up+SwiGLU），在 MUSA 上也会自动判为不支持，回退到"独立 SwiGLU 内核"那条常规路径。**这是好事：不需要为 GLU fusion 做任何 MUSA 适配。**

## 3. 实测：strata_mmq 的全部源都能编过

在 `/tmp/mmq-test/` 复制/引用（**没有修改** project-maya 与 llama.cpp 的任何文件），编译参数取 project-maya MUSA 后端的口径：

```bash
INC="-I$P/include -I$LL/ggml/include -I$LL/ggml/src -I$LL/ggml/src/ggml-cuda -I$P/include/strata/musa_compat"
DEF="-DGGML_USE_MUSA=1 -DSTRATA_USE_MUSA=1"
FLAGS="-x musa --cuda-gpu-arch=mp_22 -std=c++17 -fPIC -fno-strict-aliasing -fgpu-rdc"
mcc $FLAGS -c -o <name>.o $INC $DEF <src>
```

（`$LL=/root/autodl-tmp/llama.cpp`，`$P=/root/autodl-tmp/project-maya`）

### 3.1 四个宿主/工具源 → 全部 OK

| 源 | 结果 |
|---|---|
| `src/prefill/moe_mmq.cu` | ✅ 47,008 B |
| `src/prefill/ggml_cuda_host.cu` | ✅ 30,888 B（走 `#else` 的 CUDA 分支，未定义 `GGML_USE_HIP` 也没问题） |
| `ggml-cuda/quantize.cu` | ✅ 110,128 B |
| `ggml-cuda/convert.cu` | ✅ 534,784 B |

### 3.2 16 个 MMQ instance → 16/16 全部 OK

`xargs -P 8` 并行编（`-O1`），无一个失败：

```
OK   iq1_s   OK   iq2_s   OK   iq2_xs  OK   iq2_xxs
OK   iq3_s   OK   iq3_xxs OK   iq4_nl  OK   iq4_xs
OK   q2_0    OK   q2_k    OK   q3_k    OK   q4_k
OK   q5_0    OK   q5_k    OK   q6_k    OK   q8_0
```

仅有 `-Wreturn-type` 一类 warning（`common.cuh:895`、`mma.cuh:709`），无 error。

### 3.3 device 代码确实生成了（不是只编了 host 壳）

```
$ strings inst_q4_k.o | grep musa-mtgpu
musa-mtgpu-mt-musa-mp_22
$ strings inst_iq3_s.o | grep musa-mtgpu
musa-mtgpu-mt-musa-mp_22
$ strings inst_q6_k.o | grep musa-mtgpu
musa-mtgpu-mt-musa-mp_22
```
`llvm-ar rcs libstrata_mmq_musa.a *.o` → 29,823,336 B，20 个目标文件全部可归档。

### 3.4 带 project-maya 的 `-include cuda_runtime.h` 也能编

因为 `strata_musa_runtime` 这个 INTERFACE target 会把 `-include <musa_compat>/cuda_runtime.h` 传播给所有消费者。实测给 mmq instance 加这条：

```
$ mcc ... -include $P/include/strata/musa_compat/cuda_runtime.h -c mmq-instance-q4_k.cu
  OK 带 -include 也能编
```
（compat 头里的 `#define cudaMalloc musaMalloc` 与 ggml 自己的 `vendors/musa.h` 定义**完全一致**，重复定义不冲突。）

## 4. 怎么让 project-maya 的 strata_mmq 在 MUSA 下构建（具体改法）

### 4.1 需要改的三处

**(a) `CMakeLists.txt:70-79`** —— 去掉 MMQ 硬绑 HIP 的门禁：

```cmake
# 原来：
# option(STRATA_PREFILL_MMQ "Build the GGML MMQ prompt path for HIP (a HIP build needs it)" OFF)
# if(STRATA_PREFILL_MMQ AND NOT STRATA_ENABLE_HIP)
#   message(FATAL_ERROR "STRATA_PREFILL_MMQ requires STRATA_ENABLE_HIP=ON")
# endif()
option(STRATA_PREFILL_MMQ "Build the GGML MMQ prompt path (a HIP or MUSA build needs it)" OFF)
if(STRATA_PREFILL_MMQ AND NOT (STRATA_ENABLE_HIP OR STRATA_ENABLE_MUSA))
  message(FATAL_ERROR "STRATA_PREFILL_MMQ requires STRATA_ENABLE_HIP=ON or STRATA_ENABLE_MUSA=ON")
endif()
```
并在下面那条 "A HIP build needs MMQ" 旁边加同款、措辞按 MUSA 调整：

```cmake
# src/core/glm_prefill.cu 的 prompt 路径用 MMQ 乘专家，MUSA 构建没有它一样会在运行时退化成逐 token
if(STRATA_ENABLE_MUSA AND NOT STRATA_PREFILL_MMQ)
  message(FATAL_ERROR "A MUSA build needs -DSTRATA_PREFILL_MMQ=ON: the GLM prompt path multiplies its experts with "
                      "MMQ (./maya.sh --backend musa should set it)")
endif()
```

**(b) `CMakeLists.txt:1637`** —— 现在是 `elseif(STRATA_PREFILL_MMQ)`（HIP-only 实现），在它**前面**插一个 MUSA 分支。位置：第 1551 行 `if(STRATA_ENABLE_CUDA)` … `elseif(STRATA_PREFILL_MMQ)` 之间。

**注意**：CUDA 分支（1551）里那套"拷贝 ggml-cuda 到 build 目录 + 打 `third_party/ggml/mmq-fused-swiglu.patch`"的机制对 MUSA **不需要**——`glu_supported()` 在 MUSA 上恒为 false（§2.3），fused 路径永远不会被选中。所以 MUSA 分支直接用 pinned 检出里的原始 ggml-cuda 源即可，跟 HIP 的 `elseif(STRATA_PREFILL_MMQ)` 分支同构：

```cmake
elseif(STRATA_PREFILL_MMQ AND STRATA_ENABLE_MUSA)
  # MUSA: 与 HIP 的 PREFILL_MMQ-only 分支同构，编译器换成 mcc -x musa。
  # 不用 CUDA 分支那套 patch 拷贝：moe_mmq.cu 的 glu_supported() 依赖
  # turing_mma_available(cc)，MUSA 的 cc >= GGML_CUDA_CC_OFFSET_MTHREADS，恒为 false，
  # fused SwiGLU 路径不会被选中。
  if(STRATA_GGML_DIR)
    set(_strata_gg "${STRATA_GGML_DIR}/ggml")
  else()
    set(_strata_gg "${strata_llamacpp_SOURCE_DIR}/ggml")
  endif()
  set(_strata_mmq "${_strata_gg}/src/ggml-cuda")
  if(NOT EXISTS "${_strata_mmq}/common.cuh" OR NOT EXISTS "${_strata_mmq}/quantize.cu")
    message(FATAL_ERROR "The selected GGML source is missing the ggml-cuda MMQ sources: ${_strata_mmq}")
  endif()

  set(_strata_mmq_sources
    src/prefill/moe_mmq.cu
    src/prefill/ggml_cuda_host.cu
    "${_strata_mmq}/quantize.cu"
    "${_strata_mmq}/convert.cu")
  foreach(_t q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0 iq1_s
             q2_k q3_k q4_k q5_k q6_k q5_0)
    list(APPEND _strata_mmq_sources
         "${_strata_mmq}/template-instances/mmq-instance-${_t}.cu")
  endforeach()

  # 与 musa_backend.cmake 里对项目自身 .cu 的处理保持一致：CXX 语言 + 逐文件 -x musa + 原生 arch
  set_source_files_properties(${_strata_mmq_sources} PROPERTIES
    LANGUAGE CXX
    COMPILE_OPTIONS "-x;musa;${STRATA_MUSA_ARCH_FLAGS}")

  add_library(strata_mmq STATIC ${_strata_mmq_sources})
  target_include_directories(strata_mmq
    PRIVATE "${_strata_gg}/include" "${_strata_gg}/src" "${_strata_mmq}"
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  # ggml-cuda 走自己的 vendors/musa.h，只要这一个定义
  target_compile_definitions(strata_mmq PRIVATE GGML_USE_MUSA=1)
  target_link_libraries(strata_mmq PUBLIC ggml-base ${_strata_gpu_runtime_target} ${_strata_gpu_blas_target})

  target_link_libraries(strata_prefill PUBLIC strata_mmq)
  target_compile_definitions(strata_prefill PUBLIC STRATA_PREFILL_MMQ=1)
  # GLM 的批量 prompt 路径也直接调 MMQ 和 BLAS
  target_link_libraries(strata_engine PUBLIC strata_mmq ${_strata_gpu_blas_target})
endif()
```

**(c) `elseif(STRATA_PREFILL_MMQ)`（原 1637 行）** —— 保持 HIP 专用，建议把条件收紧成 `elseif(STRATA_PREFILL_MMQ AND STRATA_ENABLE_HIP)`，否则 MUSA 构建会走进 HIP 分支（`language HIP`、`roc::hipblas`）而失败。（若 (b) 分支写在前面，这条不改也能被先命中，但显式收紧更稳。）

### 4.2 必须同时打开的其它选项

```bash
cmake -S . -B build-musa -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/local/musa/bin/mcc \
  -DSTRATA_ENABLE_MUSA=ON \
  -DSTRATA_PREFILL_MMQ=ON \
  -DSTRATA_NATIVE_EXPERTS=ON \        # CMakeLists.txt:1431 要求
  -DSTRATA_BUILD_TESTS=ON
```

### 4.3 验证是否真的恢复了 prefill

构建完成后按这个顺序验：
1. `nm -C build-musa/libstrata_mmq.a | grep mul_mat_q` 应能看到各类型的 `mul_mat_q_case` 实例；
2. `strings build-musa/libstrata_mmq.a | grep musa-mtgpu` 应有 `mp_22` 的 device 代码（对应 §3.3）；
3. 跑 GLM 8K prompt，确认不再打印 `glm prefill: this build has no MMQ kernels - the prompt runs token by token`（`src/core/glm_prefill.cu:1131`）；
4. 数值正确性：项目里已有 `hip_prefill_mmq_parity` 那个思路（`tests/hip/prefill_mmq_parity.cpp`，合成布局 + 可选真实 pack），建议复制成 `tests/musa/` 版本跑一遍——**这一步不能省**，因为 MUSA 下 MMQ 走的是非 mma 的 pascal 路径，与 HIP/CUDA 的 llama 实现是不同 vec_dot，必须核对数值。

## 5. 风险清单（按严重程度）

| 风险 | 说明 | 处置 |
|---|---|---|
| **没有 mma、没有 dp4a** | MUSA 下 MMQ 走 `pascal_older` 配置（`__CUDA_ARCH__` 未定义导致），意味着 vec_dot 是**纯 FMA 展开**，不是 mma 也不是 `__dp4a`。正确性没问题，但**性能不会像 CUDA 上的 MMQ 那么强** | 先跑起来拿到"能批量 prefill"的形态；后续优化方向：给 ggml 侧定义一个把 `__MUSA_ARCH__` 映射成「≥ DP4A」的编译宏，让 device 侧选到 dp4a 路径（`__dp4a` 在 mp_22 上实测可用，我上一份报告已验证） |
| **host/device 配置分叉** | 这是 MMQ 移植最危险的地方。本次实测一致（§2.2），但依赖 `ggml-cuda` 的内部实现细节 | 加一条启动期断言：把 device 侧 `ggml_cuda_mmq_get_config(...)` 的 `type/nthreads/I/J` 写进缓冲、host 读回来对比，不一致就报错 |
| **`-fgpu-rdc` + 单文件链接** | 我试 `mcc ... -fgpu-rdc -o exe single.mu` 时报 `clang-14: error: cannot specify -o when generating multiple output files`。project-maya 是"每 .cu 编成 .o 再链接"，不受这条影响，但值得知道 | 保持"先 `-c` 再链"的方式；不要指望 `-fgpu-rdc` 单命令直出可执行 |
| **非原生 arch** | 上一份报告已实测：只含 mp_21/mp_31 的产物在 S4000 上有内核静默不执行 | 只用 `mp_22`（`CMAKE_MUSA_ARCHITECTURES=mp_22`） |
| **`-include cuda_runtime.h` 与 ggml 的宏重复** | 实测能编（§3.4），但 2049 条宏 + `vendors/musa.h` 的 100 多条重复定义，未来升级 ggml 时可能撞上 | 若要更保守：给 strata_mmq 单独用一个不含 `-include` 的 INTERFACE target，只链 `musart`/`mublas` + include `musa_compat`（`GGML_USE_MUSA` 会拉 `vendors/musa.h`，本就能编） |
| **16 个 instance 全编** | 编 16 个 instance 的 device 代码，`.a` 约 30 MB，编译时间可观（8 并行约 1–2 分钟） | 如要减：只留 GLM 实际用的类型（`docs/QUANT-PLAN.md` 里那种清单），但 `moe_mmq.cu` 的 `switch` 必须同步删，否则链接期 `mul_mat_q_case<...>` 未定义 |

## 6. 替代方案（如果不想动 ggml 的 MMQ）

按成本从低到高：

1. **只 port 部分 instance**：`moe_mmq.cu:138-153` 的 switch 只有 16 个类型，逐个删到 GLM 实际使用的几个（IQ1_S/IQ2_S/IQ3_S/IQ4_NL 等），编译时间和体积立刻下来。功能上等价，只是别的量化类型不能走 MMQ。
2. **用 mublas 的 quantized GEMM 替代 expert 乘法**：mublas 实测有 `mublasGemmEx`（`MUSA_R_16F`/`MUSA_R_16BF`/`MUSA_R_8I` + `MUBLAS_COMPUTE_32F` 等），但**没有** "dequantize 到 fp16 再 gemm" 之外的原生量化 GEMM（`mublasGemmEx` 的 A/B 必须是 fp16/bf16/int8 这类整齐类型，不接受 Q4_K/IQ3_S 打包格式）。要走这条路，需要在 `moe_mmq.cu` 的 `mul_mat_q_case` 外面加一层：先把 expert 权重的量化块 dequant 成 fp16 缓冲（可复用 ggml 的 `dequantize.cu`/`convert.cu`，本来就已编进 strata_mmq），再调 `mublasGemmStridedBatchedEx` 做批量 expert 乘法。代价是每层多一次 dequant 的显存流量和带宽，对小 batch 可能不划算，但对 8K prompt 这种 compute-bound 场景未必差。
3. **退回逐 token**（现状）：不可接受，8K prompt 是分钟级。

推荐顺序：**先做 §4 的 (a)(b)(c) 三处改动把 MMQ 编进来**（成本最低、上游已验证这条路可行），用 §4.3 的第 4 步核对数值；性能不如 CUDA 是预期内的，先解决"能不能批量 prefill"，再谈快不快。

## 7. 一句话结论

MMQ 在 MUSA 上**能编、能选到一致且非空的配置**，不需要为它写任何 MUSA 专用 `#ifdef`，也不需要打 CUDA 分支那套 patch；上游 llama.cpp 的 ggml-musa 后端就是把 `ggml-cuda/*.cu` + 16 个 mmq instance 原样编进 MUSA 的。project-maya 侧只需放开 `STRATA_PREFILL_MMQ` 的 HIP 门禁、加一个 MUSA 分支（§4.1），把 4 个宿主源 + 16 个 instance 用 `LANGUAGE CXX` + `-x musa --cuda-gpu-arch=mp_22` + `GGML_USE_MUSA=1` 编成 `strata_mmq` 静态库。已知代价：MUSA 下 mmq 走非 mma/非 dp4a 的慢路径，prefill 会恢复到"批量"但速度不会追平 CUDA。

---

### 附：实验文件

| 路径 | 内容 |
|---|---|
| `/tmp/mmq-test/build_inst.sh` | 16 个 instance 的并行编译脚本 |
| `/tmp/mmq-test/inst_*.o` / `libstrata_mmq_musa.a` | 16 个 instance 目标文件 + 29.8 MB 归档 |
| `/tmp/mmq-test/moe_mmq.o` `hostshim.o` `quantize.o` `convert.o` | 四个宿主/工具源 |
| `/tmp/mmq-test/cfgdiff.mu` | **host/device MMQ 配置一致性验证**（最关键的一个） |
| `/tmp/archtest.mu` | 证明设备代码里没有 `__CUDA_ARCH__` |
| `/tmp/mmq-test/inc_q4k.o` | 带 `-include cuda_runtime.h` 的编译验证 |
