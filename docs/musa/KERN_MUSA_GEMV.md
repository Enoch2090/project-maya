# MUSA 量化专家 GEMV（decode M=1..8）实测报告

机器：4× MTT S4000（mp_22, 56 MP/卡, L2 24 MB, MUSA 5.2.0 / mcc 5.2.0）。
数据：真实 GGUF 专家权重（`/root/autodl-tmp/models/Maya-S-v2-IQ2_XXS/*-00001-of-00003.gguf`；按 `pack/native_experts.txt` 的绝对偏移 pread 一个 expert：gate IQ2_XXS 2,162,688 B、down IQ2_S 2,686,976 B、down IQ3_XXS 3,211,264 B）。
源码：`/root/autodl-tmp/research/kern-musa/`（同样在 `/tmp/kern-musa/`）。2026-11-15。

## 0. 结论

单卡、M=1、L2 冷（把 40–64 个 expert 铺成 128–138 MB 的「专家银行」一次性扫完）：

| 形状 | 基线（引擎风格 f32 反量化 + FMA, MV_ROWS=8） | 优化版（64 expert 一把跑） | 倍率 | 同访问模式纯读上限 |
|---|---:|---:|---:|---:|
| gate/up IQ2_XXS 2048×4096 | 33.9 GB/s | **101.7 GB/s** | **3.0×** | 712 GB/s |
| down IQ2_S 4096×2048 | 41.6 GB/s | **91.6 GB/s** | **2.2×** | 717 GB/s |
| down IQ3_XXS 4096×2048 | 93.7 GB/s | **120.1 GB/s** | **1.28×** | 716 GB/s |

**但启动粒度决定实际能拿到多少**（这是本轮最重要的发现）：

| 启动方式 | gate IQ2_XXS | down IQ2_S | down IQ3_XXS |
|---|---:|---:|---:|
| 一个 expert 一次 launch（当前引擎的结构，grid 512–1024 块） | 59.0 GB/s | 65.2 GB/s | 82.5 GB/s |
| 一层 top-k=8 个 expert 合成一次 launch（grid 4096–8192 块） | **76.9 GB/s** | **90.9 GB/s** | **100.9 GB/s** |
| 64 个 expert 一次 launch（grid 65536 块） | 101.7 GB/s | 91.6 GB/s | 120 GB/s |

- 现状：decode 120 ms/token（16.7 tok/s），每卡每 token 读 ~0.33 GB → 有效带宽约 11 GB/s。
- 优化版混合带宽（2×gate + 1×down 每 expert 7.01 MB）：**按 expert 单独 launch 61 GB/s；top-8 合成 launch 81.7 GB/s；大 bank 97.5 GB/s**。
- 折算：0.33 GB/卡 ÷ 61–82 GB/s = 4.0–5.4 ms/卡，四卡串行 **16–22 ms → 46–62 tok/s（只算 GEMV）**。
- 但 120 ms 里还有 900–2000 次 launch（每 expert × 3 个矩阵 × 45 层 ≈ 1080 次）× 23 µs ≈ 25 ms 入队。**按现在的「每 expert 一次 launch」结构，GEMV 21.6 ms + launch 25 ms ≈ 47 ms/token ≈ 21 tok/s**；把 top-k 合成一次 launch（launch 降到 ~135 次）+ musaGraph 收掉之后 **≈ 20–25 ms → 40–50 tok/s**；再叠 M=8 verify（×2.2）才稳过 50。
- 距纯读上限还差 7×：瓶颈是 IQ 反量化的 codebook 查表 + int8 点积，不再是显存。

## 1. 做法

- 独立 harness（`main.mu`）：从 GGUF pread 真实 expert blob → 显存里复制成 40–64 份（>L2）→ 一次 kernel 扫完整个 bank；musaEvent 计时，每配置 5 轮 ×20 次**取最小值**（这台机器单次测量会偏 2–3×，必须取 min）。
- 数值：CPU float 反量化参考（`host_ref.h`，表来自 `tables.h`，由 `gen_tables.py` 从 ggml-common.h 抽出）+ CPU int8 仿真。
- 两种 lane 映射都实现了：
  - **A: warp=行**（`gemv_k` 全局表 / `gemv_ks` 表进 `__shared__`）。`R` 行/块、`CPB`=warp/行、`NSUB`=每 lane 的 sub-block 数；grid = (n_expert × n_out/R) × S。每 lane 4–16 个独立 8 B 载入先发射再算。
  - **B: lane=行**（`gemv_lb`，一个 warp 覆盖 32 行、8 个 warp 在 K 上切）。x 的所有访问变成 warp 内广播、表进 `__shared__`。大 bank 下最快（101.7），但一个 expert 只有 64 块，单 expert launch 时反而最慢（25 GB/s）。
- int8：x 按 32 元素一组量化（`quant_x_k`，scale=amax/127，等价 ggml q8_1），点积 `__dp4a`。
- 符号处理（MUSA 没有 `__vsub4/__vcmpne4`）：`sg = (grid ^ ksigns64[idx]) + (mask & 0x0101010101010101)`。因为 IQ2/IQ3 的 codebook 字节值域是 8..43 / 4..62（恒非零、<128），逐字节「异或 0xFF 再 +1」正好等于取负且不会跨字节进位，一条 64-bit 异或 + 一条加法就得到 8 个带符号 int8。
- split-K：warp 内/块内用 `__shfl_down_sync`（设备侧 warp=32）+ 共享内存归约。**没走 global atomic**：实测 atomicAdd 归约把 1.5 ms 拉到 13 ms，不能用于 split-K。

## 2. 数值

- 对 CPU float 参考：max_abs_err 7.3–9.3e-3（|y|≈1.8，相对 ~5e-3），全部来自 x 的 int8 量化，与 ggml 一致。
- 对 CPU int8 仿真：相对误差 1e-4–5e-3 → kernel 就是该算法，无索引/符号错误。
- `main.mu` 里 `test_api` 每轮自动核对（三个 API 的误差都在上面区间）。

## 3. 瓶颈定位（隔离探针，138 MB bank，GB/s）

| 探针 | IQ2_XXS | IQ2_S | IQ3_XXS |
|---|---:|---:|---:|
| 只读 qs（上限） | 712 | 717 | 716 |
| 只查 codebook/符号（不算点积） | 164 | 134 | 255 |
| 只做 int8 点积（不查表） | 162 | 197 | 107 |
| 查表 + 点积（完整核） | 64.5 | 87.5 | 107 |

查表与点积各自把时间拉到读权重的 3–5 倍，叠加后 7–11 倍。IQ3_XXS 反而最快：codebook 是 256×4 B（查表字节只有 IQ2 的一半），说明 per-lookup 开销比字节数更敏感。

## 4. 要达到 50 tok/s 还差什么（按收益排序）

1. **把一层 top-k 的 expert 合成一次 launch**（grid = n_expert_in_launch × n_out/R）。实测 59 → 77 GB/s（gate），launch 次数从 ~1080 降到 ~135，入队开销从 25 ms 降到 ~3 ms。这是最省事、收益最大的一项。
2. **musaGraph 收掉剩余 launch**：128–135 次 launch × 23 µs ≈ 3 ms 也要收掉，否则 decode 到不了 20 ms 量级。
3. **跨卡流水**：现在四卡串行（wall = 四卡之和），叠起来理论 ×3.5–4（decode 能吃一部分）。
4. **M=6/8 verify**：同一份权重喂 8 行只多 3.2×（不是 8×）时间 → 每条被验证 token 的吞吐 2.2–2.5×，投机解码直接放大。
5. **想再往 200+ GB/s 走要换反量化路径**（当前离纯读上限 7×）：载入时把 IQ2_XXS 离线重排成「每 4 元素一个 int8 四元组 + 每 32 元素一个 fp16 scale」（每卡 48 GB 放不下全量第二份，只能按层/expert 增量重排、hot expert 常驻），或把权重按「同一 K 位置的所有行打包」转置，换取 codebook 在 warp 内广播。

## 5. 复现 / 接口

```bash
cd /tmp/kern-musa
mcc -x musa --offload-arch=mp_22 -O3 -std=c++17 main.mu -o gemv
./gemv 20     # 基线 / 纯读上限 / 各配置（bank + 单 expert + 8 expert 三种粒度）/ 探针 / API 自检
```

可集成接口（`api.h`，M=1，每类型一个函数）：

```c
int musa_gemv_quant_x(const float* x, int k, musaStream_t s);   // 每 32 元素量化成 int8（内部缓存 4.6 KB 暂存）
int musa_mv_iq2xxs (const void* w, float* y, int n_out, int k, musaStream_t s);  // gate/up，k=4096
int musa_mv_iq2s   (const void* w, float* y, int n_out, int k, musaStream_t s);  // down，k=2048（IQ2_S 层）
int musa_mv_iq3xxs (const void* w, float* y, int n_out, int k, musaStream_t s);  // down，k=2048（IQ3_XXS 层）
```

`w` 指向一个 expert 的权重起点（native ggml block 布局，行主序 [n_out][k]）。这三个函数用的是「各种启动粒度都不差」的配置（layout A：gate R=2/CPB=4、down R=4/CPB=2，每 expert 512–1024 块）：单 expert 59/65/82 GB/s，top-8 合成 launch 77/91/101 GB/s，大 bank 77/92/104 GB/s。

**集成注意（实测踩到的）**：这个最小接口是「一个 expert 一次调用 + 一次 quant launch」，实测一个 expert 走完整 API 要 **110 µs**（其中一大半是两次 launch 的入队/固定开销：量化 1 次 + GEMV 1 次），一层 8 个 expert × 3 个矩阵 = 24 次调用 ≈ 2.6 ms，完全不可接受。正确用法是：
1. **每层一次** `musa_gemv_quant_x`（x 在该层 gate/up/down 之间是共用的）；
2. GEMV 扩成 `(const void* const* wlist, int n_experts, ...)`：一次 launch 覆盖一层 top-k 个 expert。`gemv_k` 的 grid.x 本来就是 `n_expert × n_out/R`，只是把 `e` 从 `blockIdx.x / rowtiles` 改成读 `wlist[e]`；这样一层 3 次 launch，全 token ~135 次，才可能配 musaGraph。

文件：`k_gemv.h`（`gemv_k` / `gemv_ks` / `gemv_lb` / `read_k` / `naive_k` / `probe_k`）、`api.h`（`quant_x_k` + 三个对外函数）、`main.mu`（harness）、`host_ref.h`、`tables.h`、`gen_tables.py`。

## 6. MUSA 上的坑（本轮实测）

- `__dp4a` 可用（mp_22，必须 `-O2/-O3`，`-O1` 不支持）。
- 没有 `__vsub4/__vcmpne4/__byte_perm`（CUDA IQ dp4a 参考实现依赖它们），上面用「异或+加 0x01」等价替代。
- 设备代码里 warp 宽度是 32（主机 API 报 128）。
- global atomicAdd 归约极慢（1.5 ms → 13 ms），split-K 不要用 atomic，用 partial 或块内归约。
- 表放 `__shared__` 不一定更快：块很小时每块复制表的开销会吃掉收益（gate 上 SHR 91 vs LANE-ROW 101）。
