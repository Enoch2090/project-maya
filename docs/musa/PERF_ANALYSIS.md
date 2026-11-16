# MUSA 性能：逐项修复对两个指标的影响（截至本轮）

机器 4× MTT S4000 48 GB，MUSA 5.2.0，模型 Maya-S-v2-IQ2_XXS（96.5 GB 专家全驻显存），16K 上下文。
下面每行都是同一台机器实测；decode 口径 = 3 条贪心 256-token 答案的均值，prefill = 冷 prompt（`--prompt-cache 0`）。

| # | 修复 | prefill 2K | prefill 8K | decode |
|---|---|---|---|---|
| 0 | 起点：MUSA 移植完成，专家走 MMQ，256-token sub-batch | 报错 ✗ | 报错 ✗ | 14.6（suffix 投机，深度 1） |
| 1 | MMQ host/device warp 不一致（128 vs 32）+ cc 钉 610 档 → 数值正确 | 45 | 59 | 14.6 |
| 2 | DFlash2 drafter（decode，`--spec 5`） | 45 | 59 | **16.7** |
| 3 | 专家乘法走 FP16 产品（`MAYA_GLM_MMQ=0`）+ `SUB=8192` | 133 | 267 | 16.7 |
| 4 | 把 3 内置为默认（FP16 专家 + 整 chunk sub-batch） | **129** | **250** | 16.7 |

- 第 0 行的"报错"：修 warp 之前，2048/8192 token 的 prompt 会报 `glm router: prompt layer 4 invalid expert ID or
  non-finite score`，引擎退回逐 token（当时 8K ≈ 14 tok/s）。
- 第 3 与第 4 行差 6%（267 vs 250），是同一配置两次引擎运行之间的波动。
- **decode 本轮没有再涨**：14.6（DFlash 之前）→ 16.7（DFlash 之后）就停在这里；kernel 侧的 8× GEMV 与
  batched GEMM 目前只是**独立验证过**（`research/KERN_MUSA_GEMV.md`、`research/MUSA_MOE_GEMM.md`），还没接进引擎。
- 参考：关掉投机是 8.3 tok/s，所以 DFlash 已经拿了 2.0×；再往上要靠 kernel + 图捕获。

## 第九轮：三条路都没有净收益（如实记录）

| 尝试 | 结果 | 对两个指标的影响 |
|---|---|---|
| batched 专家 GEMM（`MAYA_GLM_PREFILL_BATCH`，三层模式都试了） | **数值不对**：与引擎自身产品逐元素比对 `max|diff|=0.08–0.13`，几乎每个元素都差（151/128/133 行的三个专家全部如此）。strided-batched / 逐专家 GEMM / 共享反量化三种模式**同样错**，所以问题在我这套 staging+拷回，不在 GEMM 调用。已把探针移除、路径默认关闭。 | prefill 129 / 250、decode 16.7，**均无变化** |
| decode 层切分（MUSA 上静态代价模型可能切错） | `auto` 16.57/15.74、`8,20,32` 15.67/14.97、`11,23,35` 16.58/15.75、`15,26,37` 15.65/15.00 → **auto 已是最好**（与 11,23,35 并列） | 无变化 |
| 投机深度 4/5/6/7 | 4→16.11/15.56、**5→16.57/15.75（最好）**、6→16.63/15.44、7→16.69/15.33 | 无变化（5 已是默认） |

**结论**：这三条都不是当前瓶颈，继续在原处微调是负收益。已确认的瓶颈与必要条件不变：
- prefill：MoE 占 53%（6.5 s/卡 @8K），GEMM 换一次 batched 调用在独立 harness 里是 7.2×（74.6 TFLOP/s），
  但要落地必须先把 OUTP 滚动窗口的交互写对；四卡串行→并行是 4× 的必要条件。
- decode：120 ms/token 里约 25 ms 是 launch 入队（每 token ~1080 次 × 23 µs），kernel 侧独立 harness 已验证
  优化 GEMV 快 2.2–3.0×；要吃到必须做「按 top-k 批量 launch + 图捕获」这两件引擎内部改动。

## 一、当前默认配置的实测## 一、当前默认配置的实测（不需要任何环境变量）

| 项目 | 本轮结束 | 本轮开始 |
|---|---|---|
| prefill 2K | 15.6 s → **131 tok/s** | 45 tok/s |
| prefill 8K | 32.6 s → **251 tok/s** | 59 tok/s（MMQ）/ 174（FP16+env） |
| decode | 16.7 tok/s（DFlash spec 5） | 16.7 tok/s（没动） |
| 目标 | 2000+ / 50+ | |

prefill 8K 的分卡耗时：`CUDA0 6570 + CUDA1 8844 + CUDA2 8868 + CUDA3 8151 = 32.4 s`（**仍是四卡串行相加**）。

## 二、本轮修的三件事（都已验证）

1. **MMQ 只写前缀行**：host 用 `prop.warpSize`（MUSA 报硬件宽度 128）算 `nwarps=256/128=2`、`block_dims=(128,2)`，
   而 device 代码的 `ggml_cuda_get_physical_warp_size()` 是编译期 32（期望 `(32,8)`）。修：MUSA 下
   `src/prefill/ggml_cuda_host.cu` 的 `d.warp_size=32`，并让 host `d.cc` 与 device `__CUDA_ARCH__` 都取 610
   （dp4a 档），两侧共享内存布局才一致。修完 2K/8K prompt 不再因 NaN 掉进逐 token 路径。
2. **专家乘法默认走 FP16 产品**：MUSA 上 `mmq_disabled()` 默认 true（`MAYA_GLM_MMQ=1` 可强制回 MMQ）。
   8K prefill：MMQ 175 → FP16 产品 **267 tok/s**。
3. **sub-batch 提到整 chunk（≤8192）**：原来 MUSA 固定 256-token sub-batch（NVIDIA 的自动放大分支被 `#if HIP||MUSA`
   挡掉），8K prompt 会把 mixers/shared-expert 的 FP16 反量化重复 32 次。改后 8K 再快一档。

## 三、两个 kernel 任务的实测结论（决定下一步）

### prefill：专家 GEMM 换一次 batched 调用 = 7.2×

| 写法（一层 288 expert，M≈227 行/专家，K=4096，N=2048） | 时间 | 有效算力 |
|---|---|---|
| 现状：每 expert 一次 `mublasGemmEx` | 107.7 ms | 10.2 TFLOP/s |
| **一层一次 `mublasGemmStridedBatchedEx`（M pad 到 256）** | **14.7 ms** | **74.6 TFLOP/s（fp16 峰值的 96%）** |

- 整层 gate+up+down 实测 44.6 ms → 45 层单卡 2.0 s（等价 4080 tok/s 的 MoE-GEMM 吞吐）。
- `mublasGemmGroupedBatchedEx` 是坏的（只算对 7×7 个 tile，真实规模卡死 GPU）→ 用 strided batched + pad。
- fp8/int8 张量核 API 全部 `NOT_SUPPORTED(8)`；iq 的 8-bit 点积只能自己写或走 fp16。

**关键约束**：fp16 权重 45 层共 652 GB，装不下；每次 prefill 都要把 86 GB 量化权重反量化成 fp16（1.4–2.0 s，
与 GEMM 重叠实测无效，都吃 HBM）→ 必须**融合**进 GEMM。

**2000 tok/s @8K 的账**（预算 4.1 s，MoE GEMM 共 148 TFLOP）：
- 四卡串行：需 36–59 TFLOP/s/卡（有 74，够），但反量化必须融合掉；
- 四卡并行：只需 9–15 TFLOP/s/卡（5× 余量）；
- 还有约 30 s 的非 MoE 开销（attention 28%、投影等）也要压到 2–3 s。
→ **最大杠杆是把"四卡串行"变成"四卡并行"（4×）**，其次是融合反量化，第三是 batched GEMM。

### decode：kernel 能到 8.8×，但被 launch 风暴吃掉

| 形状（真实 GGUF expert，M=1，L2 冷） | 引擎风格基线 | 优化版 |
|---|---|---|
| gate/up IQ2_XXS 2048×4096 | 33.9 GB/s | **100.2 GB/s（3.0×）** |
| down IQ2_S 4096×2048 | 41.6 GB/s | **91.5 GB/s（2.2×）** |
| down IQ3_XXS 4096×2048 | 93.7 GB/s | **114.8 GB/s（1.2×）** |
| M=8（spec verify） | — | 8 行只花 3.46× 单行时间（等效 2.3× 吞吐） |

- 混合 96.7 GB/s → 每卡 0.33 GB/token = 3.4 ms → 四卡串行 13.6 ms 的权重读取。
- 但每 token 有 **900–2000 次 kernel launch**，实测入队 20–45 ms（空 kernel 23 µs/次）→ 这是现在 120 ms/token 的主项。
- **musaGraph 收掉这批 launch → 15–18 ms/token ≈ 55–65 tok/s**；再叠跨卡流水还有余量。
- 剩余瓶颈（离纯读上限 712 GB/s 差 7×）：IQ codebook 查表（IQ2_XXS 等效 164 GB/s）+ int8 点积（162 GB/s）。
  再往上要换数据布局（载入时把 IQ2_XXS 重排成 int8 四元组+scale，或按 k 位置打包所有行）。

## 四、下一步（按此顺序落地）

1. **decode：把优化 GEMV 接进 `glm_fast.cu` 的 mv 路径 + 给 decode 循环接 musaGraph**（引擎的 GLM 路径目前没有图
   捕获；musaGraph 已在本机验证可用）。预期 16.7 → 50+。
2. **prefill：把 `run_blas_q` 里逐 expert 的 `cublasGemmEx` 换成一层一次 `cublasGemmStridedBatchedEx`**
   （compat 头已把该符号映射到 mublas；引擎 KV 段已在用这条链）。预期 MoE 53% 的那部分 6.5 s → 0.9 s，
   prefill 251 → 约 450–500 tok/s。
3. **prefill：四卡并行**（现在是串行相加）。这是 4× 的杠杆，也是 2000 tok/s 的必要条件。
4. **prefill：融合反量化进 GEMM**，消掉 1.4–2 s。
5. 反复测，逐项 A/B。

## 五、交付物

- 构建/运行：`docs/MUSA_MAYA.md`；本轮默认配置已内置（无需环境变量）。
- 两个 kernel 任务：`research/MUSA_MOE_GEMM.md` + `research/kern-prefill/`（5 个可复现程序）、
  `research/KERN_MUSA_GEMV.md` + `research/kern-musa/`（含 `api.h` 与自检 `test_api`）。
- 分阶段实测：`/tmp/maya_p2_f16.log`（相位剖析）、`/tmp/maya_def.log`（默认配置）等。
