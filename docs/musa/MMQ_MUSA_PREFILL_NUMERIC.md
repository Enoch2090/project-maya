# MUSA 批量 prefill 数值错误 — 第 1 轮定位报告

> 结论（一句话）：**没能定位到根因。** 但把"哪条路错"缩小到「MMQ 的输出覆盖不完整，且缺失比例随 token 数增长」这一现象，造出了一个可重复的实验台，并排除了几个方向。下面每一条都有运行输出。
>
> 时间：2026-11-15 19:37–20:50（超时盒约 12 分钟，如实说明）。实验在 `/tmp/mmq-fix/`，未修改 project-maya 任何文件。

---

## 1. 做了什么

把 `tests/hip/prefill_mmq_parity.cpp` 机械端口到 MUSA（`sed s/hip/cuda/g`，`#include <hip/hip_runtime.h>` → `<cuda_runtime.h>`，靠 `-include include/strata/musa_compat/cuda_runtime.h` 的 `cuda*→musa*` 映射），保留它的 `run_product`（用 ggml 自己的 `from_float_ref` 造权重、`mmq::quantize` 量化激活、`mmq::Context::run` 跑 Product、CPU FP32 双精度参考、rel_l2 / max_abs / 全零行三项判据），把 `main` 换成可参数化扫描（类型 × rows）。

产物：
| 文件 | 说明 |
|---|---|
| `/tmp/mmq-fix/parity.cpp` | 参数化版（`parity <ggml_type> <rows...>` 或 `parity sweep`） |
| `/tmp/mmq-fix/parity_orig` | 原版 HIP 测试（几何未改）的 MUSA 端口 |
| `/tmp/mmq-fix/parity_body.inc` / `sweep_main.inc` | 拼装用的两段源 |

构建方式（**注意：项目实际构建没有 `-fgpu-rdc`**，我最初照上轮报告的记录加 rdc 反而炸 `cannot specify -o when generating multiple output files`；查 `build-musa/build.ninja` 得到真实口径是 `FLAGS = -fno-strict-aliasing -O3 -DNDEBUG`，去掉 rdc 后正常）：

```bash
P=/root/autodl-tmp/project-maya; LL=/root/autodl-tmp/llama.cpp; B=$P/build-musa
INC="-I$P/include -I$P/include/strata/musa_compat -I/usr/local/musa/include \
     -I$LL/ggml/include -I$LL/ggml/src -I$LL/ggml/src/ggml-cuda"
DEF="-include $P/include/strata/musa_compat/cuda_runtime.h -DSTRATA_USE_MUSA=1 -DSTRATA_HAS_WMMA=0 -DGGML_USE_MUSA=1"
mcc -x musa --offload-arch=mp_22 -std=c++20 -O1 $INC $DEF -c -o parity.o parity.cpp
mcc parity.o -o parity $B/libstrata_mmq.a $B/ggml/src/libggml-cpu.a \
    -lmusart -lmusa $B/ggml/src/libggml-base.a -lm -lmublas -L/usr/local/musa/lib
```
（链接顺序是从 `build.ninja` 的 `LINK_LIBRARIES` 抄的；`libstrata_engine.a` 等不需要，产物 26 MB。）

## 2. 观察到的现象（这是本轮唯一确定的实验结果）

`./parity 12 <rows>` = Q2_0 类型、out_rows=256、cols=2560、单 expert、`ids` 带 `IDS_PAD=512` 填充、`set_whole_tiles(true)`：

```
rows=1   → 通过（全部写出，数值正确）
rows=2   → 通过
rows=4   → SENTINEL_UNWRITTEN=416 of 1024   first_bad row=2 out=96
rows=8   → SENTINEL_UNWRITTEN=1440 of 2048
rows=28  → SENTINEL_UNWRITTEN=4736 of 7168
```

- `SENTINEL_UNWRITTEN` 统计的是输出缓冲里仍是 `0xff` 哨兵（`0xffffffff`）的元素个数，即**内核根本没写**的位置；`first_bad_bits=0xffffffff` 确认是"没写"，不是算出了 NaN。
- 写出量：rows=4 与 rows=8 都是 608 个元素；rows=28 是 2432 个。**写出的是一段前缀**（rows=4 时正好是 row 0、row 1 完整 + row 2 的前 96 个），后面的行整块没写。
- 全零行判据、`rel_l2` 判据都没能走到（先被 non-finite 拦掉了）。

`whole_tiles` 开/关**完全没有影响**（同样的 416/1440/4736）：

```
whole=1 rows=4     SENTINEL_UNWRITTEN=416 of 1024
whole=0 rows=4     SENTINEL_UNWRITTEN=416 of 1024
whole=1 rows=8     SENTINEL_UNWRITTEN=1440 of 2048
whole=0 rows=8     SENTINEL_UNWRITTEN=1440 of 2048
whole=1 rows=28    SENTINEL_UNWRITTEN=4736 of 7168
whole=0 rows=28    SENTINEL_UNWRITTEN=4736 of 7168
```

类型不是变量：Q2_0 / Q4_K / Q6_K / Q8_0 在 rows≥4 时全部表现一致（同样的 non-finite 失败），说明**不是某个 vec_dot 的数值问题，而是所有类型共用的那条路径（tile 划分 / 网格覆盖）**。

### 这个现象和线上症状是吻合的

线上：28 token 对、128/256/2048 token 崩，且"分块越小、出错层越深"（每层少数 token 被算坏 → 逐层放大成 NaN）。本实验：**输出覆盖随 token 数增长而缺口变大**。方向一致。

## 3. 排除掉的东西

| 方向 | 结论 | 依据 |
|---|---|---|
| 某个量化类型的 vec_dot 算错 | **排除**（不是类型相关） | Q2_0/Q4_K/Q6_K/Q8_0 同症状 |
| `whole_tiles` / `nsm=1` 的 stream-k 技巧 | **排除**（开关无差别） | 上表 |
| host/device 选到不同的 MMQ config | **上轮已排除**，本轮复验一致 | `/tmp/mmq-test/cfgdiff.mu`：两侧都是 `type=43 nthreads=256 I=128 J=64 stream_k=0` |
| MMQ 源码在 MUSA 上编不过 / 编出空 device 代码 | **排除** | 16/16 instance 编过，`.o` 含 `musa-mtgpu-mt-musa-mp_22` |
| `-fgpu-rdc` 导致 device 符号未解析 | **排除**（项目本来就不用 rdc） | `build.ninja` 的 compile/link FLAGS 都没有 rdc |
| `__dp4a` 在 mp_22 上不可用导致退化 | 不适用：MUSA 走的是 `pascal_older`，根本不调 `__dp4a` | 上轮 `cfgdiff` + `__CUDA_ARCH__` 未定义 |

## 4. 为什么还不能断定"线上就是这个问题"

**我的实验台自证是有问题的**：线上 28 token 是**正确**的，而我的 harness 在 rows=4 就已经丢 40% 输出。所以两者必有一个不同。最可能的解释与下一步（都是 30 分钟内可做的）：

1. **`ids` / `bounds` 的填充值**：我为 `IDS_PAD` 多分配了空间但**没有初始化**（`cudaMalloc` 是脏内存）。头文件说"those values are never used"，但那是在 HIP/CUDA 上验证的；若 MUSA 下确实读了它们，就可能把行写去别处。**先 `cudaMemset` 成 0 再跑**——这一步没做，是本轮最大的疏漏。
2. **`Product` 的 `max_rows`/`total_rows` 语义**：线上是**多 expert**（每个 expert 只占若干 token 行，`max_rows` = 单个 expert 最多的行数，通常 6–10），我这里是**单 expert、max_rows=total_rows**。如果内核的网格是按 `max_rows` 算的，两者行为会不同——这也正好解释"线上 28 token（多 expert、每 expert 少行）对，长 prompt（每 expert 行数变大）错"。
   → **下一步应该造多 expert 的用例**（例如 counts={4,3,3}、{16,16}、{64,64}、{128,128}），看是不是 `max_rows` 超过某个阈值（如 >2 或 >8）才丢输出。
3. **真实 GLM 几何**：`parity_orig`（out=1280/cols=2560、原判据）同样失败（rows=7 时 6304/8960 未写），所以不是 256 列的问题。

## 5. 已经在代码里看到的、值得优先怀疑的两个点

(a) **`src/prefill/ggml_cuda_host.cu:88`** 的 cc 计算：
```c
#else
    d.cc = 100 * prop.major + 10 * prop.minor;   // MUSA: 100*2 + 10*2 = 220
```
而 ggml 上游 MUSA 分支用的是 `GGML_CUDA_CC_OFFSET_MTHREADS + major*0x100 + minor*0x10`（= `0x100220`，`ggml-cuda.cu:338`）。本项目用 **220（没有 MTHREADS 前缀）**。后果：
- `GGML_CUDA_CC_IS_NVIDIA(220)` 为 **true**（220 < 0x100000），所以 MUSA 会被当成"很老的 NVIDIA 卡"；
- 目前靠"220 < 610"恰好落到 `pascal_older`，host/device 因此一致（实测），**但任何 `>= GGML_CUDA_CC_DP4A` 的判断都会把它当 NVIDIA 处理**。这是脆弱的，不是当前 bug 的必然原因，但值得改成上游的写法以免后续踩坑。

(b) **`ggml_cuda_host.cu` 把 `d.warp_size = prop.warpSize` = 128**（MUSA 上报 128，而设备代码里 `warpSize` = 32）。若 MMQ 的 host 侧启动计算用到 `warp_size`，会与 device 侧的 `WARP_SIZE=32` 不一致。**本轮没来得及验证 MMQ 是否用它**，建议下一个接手的人 `grep -n warp_size ggml-cuda/mmq*.cuh ggml-cuda/mmq*.cu` 确认。

## 6. 如果定位继续卡住：可以落地的兜底方案（lead 的选项 c）

引擎里已经有批量 FP16 路径，不需要 MMQ：

- `src/core/glm_prefill.cu:291` `bool expert_f16(const Glm5Geometry& g, int t) { return !expert_mmq(g, t) && mmq::to_f16_supported(t); }`
- `src/core/glm_prefill.cu:3942` 已经有一个 `use_blas` 判据：`blas_rows > 0 && !Ly.exl3 && mmq::to_f16_supported(Ly.gu_type) && mmq::to_f16_supported(Ly.d_type) && S->w16_mmq_elems >= 2*nff*E + E`
- `moe_mmq.hpp` 里有现成的 `to_f16(type, w, dst, n, stream)`（走 ggml-cuda `convert.cu` 的 dequantizer，已编进 `strata_mmq`）

所以"关掉 MMQ 走 FP16 expert GEMM"这条路**在代码里是存在的**，缺的是把 `glm_prefill.cu:1131` 那个 `if (!mmq::built()) → 逐 token` 的分支改成走上面那条 `use_blas`/`expert_f16` 批量路径。这条改动落在 `src/core/glm_prefill.cu`（**lead 正在改的文件，需要你分配**），不是我该单独动的。cmake 侧我可以配合：给 MUSA 加一个 `STRATA_PREFILL_MMQ=OFF` 但保留 `strata_mmq`（为了 `to_f16`/`supported` 这些符号）的配置。

## 7. 下一步（按性价比排序，每条 ≤30 分钟）

1. `parity` 里把 `ids`/`bounds` 的 padding `cudaMemset` 清零，重跑 rows=4/8/28。（最省事，可能直接翻盘）
2. 造**多 expert** 用例（counts={4,3,3} / {16,16} / {64,64} / {128,128}），扫 `max_rows` 阈值，找出"丢输出"的临界点，再和线上 `--prefill` 的分块规模对上。
3. 在临界用例上 `export MUSA_LAUNCH_BLOCKING=1`（若支持）或加 `cudaPeekAtLastError` 定位到具体 launch；`compute-sanitizer` 无 MUSA 版，改用把 `xq`/`dst` 缓冲填不同哨兵值来区分"写错位置"和"没写"。
4. 查 `warp_size` 在 MMQ host 启动计算里的使用（§5b）。

## 8. 一句话交底

**没定位到根因，没改任何源码。** 拿到的确定信息是：MUSA 上 `mmq::Context::run` 的输出会**按行前缀被部分漏写**，缺口随 token 数增大（rows=1/2 全对，4 起就开始漏），与量化类型无关、与 `whole_tiles` 无关；这与线上"短 prompt 对、长 prompt 坏"方向一致。但我的实验台在 rows=4 就漏写，而线上 28 token 是对的，说明实验台还缺一个前置条件（最可疑：多 expert 的 `max_rows` 语义、未清零的 `ids` 填充），**在补齐它之前不能把这个现象直接归因于线上 bug**。
