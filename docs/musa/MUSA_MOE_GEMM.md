# MUSA MoE prefill 专家矩阵乘：最快路径实测

> 机器：4× MTT S4000 48GB，MUSA 5.2.0 / 驱动 5.2.0-server，`mcc` 5.2.0（clang 14），`mp_22`，mublas 1.13.0。
> 全部数字来自真实编译 + 真实运行；代码 `/tmp/kern-prefill/`（副本 `/root/autodl-tmp/research/kern-prefill/`），日志 `run2.log` / `layer.log`。
> shape：E=288 expert，8K prompt 时每 expert 227 行；gate/up `[K=4096 → N=2048]`、down `[K=2048 → N=4096]`；fp16 权重 + fp16 激活 + fp32 输出（`MUBLAS_COMPUTE_32F`），权重 `[N,K]` 行主序（`OP_T`）。
> 时间 2026-11-15 22:05–22:40；基准跑在 GPU0（当时引擎占 GPU1–3，GPU0 独享）。

---

## 0. 结论（先行）

1. **把"每 expert 一次 `mublasGemmEx`"（一层 288 次）换成"一次 `mublasGemmStridedBatchedEx`（batch=288）"，MoE 专家 GEMM 从 10.2 → 74.6 TFLOP/s（7.2×）**，单卡 45 层从 14.5 s → **2.0 s**。这是本次最硬的结论，且已用 `verify_stride.mu` 验证数值正确（max_err=0）。
2. **行数 pad 到 256 是免费的**（M=227 14.707 ms vs M=256 14.693 ms）：这一步是权重带宽受限，多算的 12% 本来闲着。所以不需要 `mublasGemmGroupedBatchedEx` 的可变 M。
3. **`mublasGemmGroupedBatchedEx` 不能用**：符号在（`mublas_functions.h:15180`），调用返回 0，但结果只算对了 7×7 个 tile，真实规模（288 expert / 22 GB）直接把 GPU 卡死。**别用**。
4. **fp8/int8 张量核在 mublas 上不支持**：`MUSA_R_8F_E4M3` / `MUSA_R_8I` 的 GEMM 全部返回 `NOT_SUPPORTED(8)`。
5. **真瓶颈不是 GEMM，是权重字节**：fp16 专家权重 14.5 GB/层、45 层 652 GB，4 卡 192 GB 装不下（量化后 92 GB 才装得下）。所以 F16 路线必须"量化权重常驻 + 每层反量化"，而反量化 = 读 92 GB + 写 652 GB = 744 GB 流量，实测**纯写 14.5 GB/层只要 31 ms，但反量化写一遍 79 ms/层（笨 kernel）、按 400 GB/s 估约 41 ms/层**，即 **1.4–3.6 s/次 prefill**，正好吃掉 2000 tok/s 预算（4.1 s）的一大半。
6. **双流重叠反量化与 GEMM 没用**：实测 1.00×（两个都吃 HBM，互相抢带宽）。唯一能消掉这 1.4–2 s 的办法是**把反量化融合进 GEMM**（量化权重常驻，kernel 里解成 fp16 喂张量核）。
7. **2000 tok/s 的算力需求**：8K prompt = 4.1 s 预算，MoE 专家 GEMM 共 148 TFLOP → 四卡串行需要 **36–59 TFLOP/s/卡**（视给 MoE 留多少时间），四卡并行只要 **9–15 TFLOP/s/卡**。现有 74 TFLOP/s 说明**算力不是问题，四卡串行才是问题**（引擎现在把四卡耗时相加 = 串行）。

---

## 1. 实测数据

### 1.1 硬件上限（这次测的）

| 项 | 实测 | 方法 |
|---|---|---|
| fp16 张量核峰值 | **77.5 TFLOP/s** | `mublasGemmEx` fp16 4096³ fp16→fp32（2048³ 只有 45.5，尺寸压不满） |
| fp16 张量核（fp16 输出/累加） | 51.1 TFLOP/s | 2048³ `MUBLAS_COMPUTE_16F` |
| fp32 SGEMM | 12.9 TFLOP/s | 2048³（前次已测） |
| HBM 读+写混合 | **366 GB/s** | 满 grid fp16 copy kernel（56×8 block × 256 线程） |
| HBM 纯写 | **465 GB/s** | 同上 grid，只写 14.5 GB |
| HBM memcpy（DtoD） | 349 GB/s | 512 MB `musaMemcpy` |

### 1.2 一层 288 个 expert GEMM 的三种写法（gate 矩阵，M=227 N=2048 K=4096）

| 写法 | 一层 gate 耗时 | 有效算力 | 说明 |
|---|---|---|---|
| (a) 288 次 `mublasGemmEx` | **107.7 ms** | **10.2 TFLOP/s** | 单次 0.368 ms；launch 开销不是瓶颈（单次 ≈ 循环内一次） |
| (b) 1 次 `mublasGemmStridedBatchedEx` batch=288 | **14.7 ms** | **74.6 TFLOP/s** | = 96% 的 fp16 峰值；`verify_stride` 数值 PASS |
| (c) `mublasGemmGroupedBatchedEx`（可变 M） | — | — | **坏，见 §1.4** |

为什么 (a) 这么慢：单 expert 单 GEMM 的输出是 227×2048，按 128×128 tile 只有 ~32 个 CTA，56 个 SM 装不满 → 只有 10 TFLOP/s；循环 288 次也没法让它们同时在飞。(b) 把 288 个 GEMM 装进一个 kernel = 9216 个 tile，SM 全满。

M 维度扫描（gate strided batched，288 expert）：

| M | 耗时 | 有效算力 | 有效带宽 |
|---|---|---|---|
| 128 | 14.535 ms | 42.6 TFLOP/s | 374 GB/s |
| **227** | **14.707 ms** | **74.6 TFLOP/s** | 401 GB/s |
| 256 | 14.693 ms | 84.2 TFLOP/s | 411 GB/s |
| 512 | 28.901 ms | 85.6 TFLOP/s | 251 GB/s |

→ M ≤ 256 时**权重带宽受限**（读 4.83 GB 权重 / 14.7 ms = 329 GB/s 只是权重的，加激活/输出 401 GB/s）；M=512 才转成算力受限（85.6 ≈ 峰值）。**8K prefill 正落在带宽受限区**。

### 1.3 整层 × 45 层

| 项 | 实测 |
|---|---|
| 一层 = gate + up + down（3 次 strided batched） | **44.64 ms**（73.7 TFLOP/s，402.7 GB/s） |
| 45 层（单卡，fp16 权重常驻） | **2.01 s** |
| 等价 MoE-GEMM 吞吐、单卡 | **≈4080 tok/s** |

### 1.4 `mublasGemmGroupedBatchedEx` 是坏的

- 签名在 `/usr/local/musa/include/internal/mublas_functions.h:15180`，与 cuBLAS 12 的 `cublasGemmGroupedBatchedEx` 一致。
- 小规模（E=5, M=7, N=32, K=64，答案可解析验证）：`status=0`，但**只写对了前 7 列**（7×7 tile），其余为零。三种参数传法（全 host / 指针数组 device / 全 device）结果一样错。
- 真实规模（288 expert，22 GB 权重）：调用返回 0 后 `musaDeviceSynchronize()` **永不返回**，GPU 卡死，只能 kill。
- 反正也不需要它：可变 M 的价值只是省 pad，而 pad 实测零成本（§1.2）。

### 1.5 fp8 / int8：mublas 不支持

`MUSA_R_8F_E4M3`、`MUSA_R_8I` 的 `mublasGemmEx`（4096³）和 `mublasGemmStridedBatchedEx`（MoE 形状）全部 `MUBLAS_STATUS_NOT_SUPPORTED (8)`。想用 8-bit 张量核只能自己写 kernel（`mtmusa::wmma` 的 int8 8x16x16 fragment 已验证可编可跑）。

### 1.6 反量化成本 + 双流重叠（这次重点问的）

用一个"2 bit 打包读、fp16 写"的代理 kernel（读 1.81 GB → 写 14.50 GB，即一层专家权重）：

| 项 | 实测 | ×45 层 |
|---|---|---|
| 纯写 14.5 GB（不含反量化数学） | 31.2 ms/层（465 GB/s） | **1.40 s** |
| 反量化代理 kernel | 78.9 ms/层（207 GB/s 流量） | 3.55 s |
| 按实测 400 GB/s、744 GB 总量估 | ≈41 ms/层 | **1.84 s** |
| 引擎侧此前的记录 | — | 0.8–1 s |

- 我写的代理 kernel 是"一个 kernel 干完读+写"的朴素版，比理想带宽差一倍；**上界是纯写 1.40 s，理论 1.84 s，引擎报 0.8–1 s**。结论取 **1–2 s/次 prefill**，量级确定。
- **双流重叠实测 1.00×（无收益）**：反量化 78.9 ms + GEMM 44.7 ms，两流同时跑 123.6 ms ≈ 相加。因为 GEMM 本身已经在 400 GB/s，两个 kernel 抢同一条 HBM。**所以"用双流把反量化藏起来"这条路不通**。
- 整模型一次性反量化成 fp16 **物理上做不到**：652 GB > 4×48 GB。

---

## 2. 每层 FLOPs：3.29 TFLOP（任务书的 1.4 少乘了 2）

```
一层 = 2 · E · rows · N · K · 3 = 2 × 288 × 227 × 2048 × 4096 × 3 = 3.29 TFLOP
     （gate+up 2.19 + down 1.10）
45 层 = 148 TFLOP
```
行数没问题（8K×8 expert = 65536 行 ÷ 288 = 227.5）；是 2·M·N·K 的 2 漏了。任务书按 295 TFLOP@8K 估，实际 148 TFLOP@8K。

---

## 3. 路线账（8K prompt，只算 MoE 专家 GEMM + 反量化，不含 attention）

| 方案 | 四卡串行（引擎现状：耗时相加） | 四卡并行（专家切分） |
|---|---|---|
| A 现状：288×`GemmEx` + 逐层反量化 | 14.5 + 1.8 = **16.3 s → 503 tok/s** | 4.1 s → 2000 tok/s |
| B strided batched + 逐层反量化 | 2.0 + 1.8 = **3.8 s → 2156 tok/s** | 0.95 s → 8600 tok/s |
| C strided batched + 想用双流重叠反量化 | **无效**（实测 1.00× 重叠收益） | 无效 |
| D strided batched + 反量化融合进 GEMM | **2.0 s → 4100 tok/s** | 0.50 s → 16400 tok/s |

- A 行的 503 tok/s 是"只算 MoE GEMM"的上限；引擎 fp16 路径实测 174 tok/s（8K = 47 s），意味着除专家 GEMM 外还有 **~30 s 其它开销**（attention、其它 GEMM、gather/scatter、内存格式转换等）。**只换 GEMM 一项，174 → 大概 220–240 tok/s**。
- B 行是"最省事改动"能到的位置：2000 tok/s 线上下，且没给 attention 留时间。
- D 行（融合）才是真正能把 MoE 部分压到 2.0 s 以内的方案。

---

## 4. 要 2000 tok/s 还缺什么

2000 tok/s @ 8192 token = **4.1 s / prefill**；MoE 专家 GEMM = 148 TFLOP。

| 场景 | 需要每卡算力 | 现有 | 判断 |
|---|---|---|---|
| 四卡串行，MoE GEMM 给 2.5 s | **59 TFLOP/s** | 74 | 算力刚好够，但**必须**反量化融合/为零，否则差 2 s |
| 四卡串行，MoE GEMM 给 4.1 s | 36 TFLOP/s | 74 | 够，但 attention 没时间 |
| 四卡并行，MoE GEMM 给 2.5 s | **15 TFLOP/s** | 74 | 5× 余量 |

**缺的三样，按价值排序：**

1. **四卡真并行（4×，最大项）**。现在 `CUDA0 27558 + CUDA1 39034 + CUDA2 38775 + CUDA3 32930 = 138501 ms ≈ 总时间`，说明四卡是排队的。按 expert 切卡（每卡 72 expert、全 45 层）→ 每卡工作量 1/4，wall time 从 sum 变 max，直接 4×。**这一项比 GEMM 优化更值钱。**
2. **消除反量化的 1.4–2 s**。双流重叠已实测无效（抢带宽），只能融合：量化权重常驻显存（92 GB，23 GB/卡），GEMM kernel 里查表解成 fp16 喂张量核。融合后每层只读 ~1.9 GB 量化权重（4.4 GB 流量 → 12 ms），层时间回到 44 ms 的算力受限，2.0 s/45 层。
3. **把 non-MoE 的 ~30 s 开销压掉**。这是 2000 tok/s 真正的硬骨头：4.1 s 预算里，MoE GEMM 最优也要 1.0–2.0 s，只剩 2–3 s 给 attention + 其它。**建议引擎侧先按层打点**（attention / 其它 GEMM / 反量化 / gather 各多少），否则不知道往哪里使劲。

---

## 5. 建议的落地顺序

1. **先把 288 次 `cublasGemmEx` 换成 1 次 `cublasGemmStridedBatchedEx`（7.2×，改动最小）**
   - `src/core/glm_prefill.cu` 里 expert GEMM 现在是逐 expert 的 `cublasGemmEx`；compat 头 `include/strata/musa_compat/cublas_v2.h:19` 直接 `#define cublasGemmStridedBatchedEx mublasGemmStridedBatchedEx`，**改动是纯替换、不用加链接配置**（引擎里 KV 那段 `glm_prefill.cu:3000` 已经在用 `cublasGemmStridedBatchedEx`，说明这条调用链通）。
   - 每层 3 次调用：gate(288)、up(288)、down(288)（或把 gate/up 权重拼成连续 576 个 matrix 一次提交，注意激活要按 576 份 stride 排或复用同一段）。
   - 要求：每 expert 的 fp16 权重在显存里连续（`L.w16` 现在就是按 expert 分段连续）；行数 pad 到统一 256（实测零成本）；激活和输出按同样 stride 排（`stride_b = 256·K`、`stride_c = 256·N`）；`lda=K`、`ldb=K`、`ldc=N`、`OP_T/OP_N`、C 用 fp32。
2. **四卡按 expert 切分，把"串行相加"改成并行**（4×，比 1 更值钱）。
3. **反量化融合进 batched GEMM**（或退一步：把反量化结果缓存复用，但 652 GB 只能逐层用一次，缓存没意义）。
4. 若要再往上（>8000 tok/s）：`mtmusa::wmma` 手写 tiled GEMM + 融合反量化。依据：fp16 峰值 77.5 TFLOP/s 已实测，strided batched 已能跑 74（96% 峰值），说明 mublas 路径已到顶，想再快只能改数学/数据量（int8/fp8 在 mublas 不支持，只能自己写 wmma int8 8x16x16）。
5. 不推荐的路：`mublasGemmGroupedBatchedEx`（坏）、fp8/int8 GEMM（不支持）、双流重叠反量化（实测 1.00×）。

---

## 6. 复现

```bash
cd /tmp/kern-prefill        # 副本 /root/autodl-tmp/research/kern-prefill/
mcc -x musa --offload-arch=mp_22 -O3 -std=c++17 bench_layer.mu -o bench_layer -lmublas
./bench_layer 0             # 1.8 s 出 §1.1–1.3 全部数字
mcc -x musa --offload-arch=mp_22 -O3 -std=c++17 bench_dequant_overlap.mu -o bench_dequant_overlap -lmublas
./bench_dequant_overlap     # §1.6 反量化 + 双流重叠
mcc -x musa --offload-arch=mp_22 -O3 verify_stride.mu -o verify_stride -lmublas
./verify_stride             # PASS, max_abs_err=0
mcc -x musa --offload-arch=mp_22 -O3 verify_grouped.mu -o verify_grouped -lmublas
./verify_grouped 1          # grouped 语义探针：返回 0 但结果错
mcc -x musa --offload-arch=mp_22 -O3 probe_fp8.mu -o probe_fp8 -lmublas
./probe_fp8                 # fp8/int8 = status 8 (NOT_SUPPORTED)
# 注意：moe_gemm_bench.mu 的 phase5（grouped）会把 GPU 卡死，跑的话务必 timeout
```

| 文件 | 内容 |
|---|---|
| `bench_layer.mu` | fp16 峰值、流带宽、M 扫描、整层×45 层（结论主力） |
| `bench_dequant_overlap.mu` | 反量化代理成本 + 双流重叠收益 |
| `verify_stride.mu` | strided batched vs 逐 expert GEMM 正确性 |
| `verify_grouped.mu` | grouped batched 参数语义探针 |
| `probe_fp8.mu` | fp8 E4M3 / int8 GEMM 可用性 |
| `moe_gemm_bench.mu` | 三种 API 全对比（phase5 会卡 GPU） |
| `layer.log` / `run2.log` | 原始输出 |
