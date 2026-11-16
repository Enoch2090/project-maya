# GLM-5.3-Flash 在 MUSA（摩尔线程 MTT S4000）上的部署与测速报告

日期：2026-11-15 · 机器：AutoDL 容器，4× MTT S4000 48 GB，MUSA 5.2.0 / mcc 5.2.0（clang 14）· 模型：
Project Maya `Maya-S-v2-IQ2_XXS`（GLM-5.3-Flash 321B，96.5 GB，GGUF IQ2/IQ3 专家）

## 一、结论：速度

| 指标 | 实测值 | 测量方法 |
|---|---|---|
| **decode（写答案）** | **14.6 tok/s**（三次均值；单次 14.2–15.2；72–77 ms/token） | 3 条固定题目，贪心 `temperature=0`，各生成 256 token，先热身一次；取引擎 `DONE` 行的 `generated/decode_ms` |
| **prefill（读 prompt）** | 批量路径实测 **230 tok/s**（28 token prompt）；**长 prompt 稳定运行值 ≈14 tok/s** | 见"已知问题 1"：批量 MMQ 路径超过约 100 token 数值出错，引擎退化为逐 token 读 prompt，此时 prefill 速率 = decode 速率 |

规格摊开看：4 卡共 192 GB 显存，模型 96.5 GB 全部驻留显存（专家池 93.5 GB、12096 个专家一个不缺），
CPU/RAM 通道完全空闲，所以这是个干净的四卡 decode 数字，不是 SSD/内存换来的。

## 二、做了什么（MUSA 适配）

原仓库只支持 CUDA / HIP / Metal / Intel Arc。这次新增一个 MUSA 后端，做法与仓库既有的 AMD 后端同构：

1. **`include/strata/musa_compat/`**（新增，8 个头文件）：把 CUDA 写法映射到 MUSA——`cudaXxx`→`musaXxx`
   共 147 个符号、cuBLAS→muBLAS、`CUDA_R_*`→`MUSA_R_*`、`__nv_bfloat16`→`__mt_bfloat16`、fp16/fp8/
   profiler/driver 头，以及 `__nanosleep`、`__ldcg(float/unsigned)` 这两个 MUSA 缺的重载。
   关键点：MUSA 只在设备编译期定义 `__MUSA_ARCH__`、两种编译都定义 `__MUSACC__`，于是把
   `__HIP_DEVICE_COMPILE__` / `__HIPCC__` 接到它们上——源码里已经为 AMD 写好并验证过的"非 NVIDIA"分支
   就原样生效（无 mma.sync、无 cp.async、F32 attention、非 WMMA 路径）。
2. **`cmake/musa_backend.cmake`**（新增）：CMake 没有 MUSA 语言，所以 `.cu` 以 CXX 语言走 mcc
   `-x musa --offload-arch=mp_22 -O3` 编译，所有编译单元强制 `-include cuda_runtime.h`（与 hip_backend 一致）。
   C 编译器用 MUSA 自带 clang（mcc 拒收 C 输入）。
3. **源码条件编译扩展**：73 个文件、+556/−439 行，把 `STRATA_USE_HIP` 条件扩成 `HIP 或 MUSA`，
   并逐个修掉被带进来的 AMD 专属代码（amdgcn 内建、`__hip_atomic_load`、`v_*` 内联汇编、gfx 架构探测、
   `gcnArchName`、hipBLAS 拼写等）。`cmake/glm_dsa.cmake` 与 `cmake/glm_dsa.d/*` 的 GPU 源注册也加了 MUSA。
4. **`src/kernels/cuda/exl3_stub_musa.cpp`**（新增）：EXL3 量化格式的 kernel 在 MUSA 上编不动
   （mcc 设备端 codegen 在 `exl3.cu` 上 30 分钟不结束），而本模型不含 EXL3 权重，因此给链接器一个桩，
   运行期不会被选中（`supported()` 返回 false）。真要用 EXL3 可加 `-DSTRATA_MUSA_EXL3_KERNELS=ON` 等它编。
5. **MMQ 放开**：GLM 的 prompt 批量路径依赖 ggml 的 MMQ kernel。原 CMake 把它硬绑 HIP，现在 MUSA 也能编
   （19 个 ggml-cuda 源 + 16 个 instance 全部编过；host/device 两侧 MMQ 配置实测一致）。这条路能编能跑，
   但数值上出了问题，见下。

完整说明写在仓库里：`project-maya/docs/MUSA_MAYA.md`。

## 三、已知问题（按影响排序）

1. **批量 prefill 在 MUSA 上数值错误**（这是 prefill 数字上不去的唯一原因）。8192 token 分块时，
   2048 token prompt 在第 4 层报 `glm router: prompt layer 4 invalid expert ID or non-finite score
   (0 of 16376 routes)`；把分块降到 128，同样的损坏出现在第 8 层；降到 256 时第 4 层有 160/2048 条路由无效。
   28 token 的 prompt 正常且很快（230 tok/s）。引擎的 NaN 守卫会把该对话状态丢弃并改为逐 token 重读
   prompt，所以长 prompt 仍然能答，只是按 decode 速度走（≈14 tok/s）。
   首要怀疑对象：ggml 在 MUSA 上选中的 `vec_dot` 是 `pascal_older` 档（无 mma、无 `__dp4a`）的纯 FMA 实现，
   这是这块硬件上从没被跑过的路径；下一步应把 `tests/hip/prefill_mmq_parity.cpp` 移植成 MUSA 版做逐位核对。
2. **decode 受带宽约束**：单 token 72–77 ms，比仓库里同量化的 NVIDIA 行慢（那是 MUSA 上反量化+点积路径的
   问题，不是移植层）。
3. **EXL3 量化模型无法加载**（见上）；GGUF IQ 量化正常。
4. **无视觉编码器**：本 build 是纯文本（`maya-vision` 是独立的 llama.cpp 程序，未移植）。
5. 只验证了引擎（`--serve` 的 GEN/DONE 通道）；`serve/` 的网页面板和 OpenAI 兼容 API 没跑。

## 四、复现

```bash
# 模型（已就位）：/root/autodl-tmp/models/Maya-S-v2-IQ2_XXS/
#   pack 由 tools/iq_pack.py --compat-bf16 生成，908 MiB，与 3 个 shard 同级
export MUSA_VISIBLE_DEVICES=0,1,2,3          # MUSA 只认这个选卡变量
export LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH
cd /root/autodl-tmp/project-maya
./build-musa/maya --serve --glm-pack /root/autodl-tmp/models/Maya-S-v2-IQ2_XXS/pack \
  --max-context 16384 --expert-cache auto --prefill auto --kv-streaming off --layer-split auto
# stdin: GEN 256 temperature=0 <逗号分隔的 token id>
# 启动完成后 stderr 打印 READY 16384 stop pause
```

构建命令在 `docs/MUSA_MAYA.md`；测量脚本：`/root/autodl-tmp/research/bench_musa.py`。

## 五、当前进程状态

- 引擎在跑（`--serve`，4 卡，16384 context，专家全驻显存），日志 `/tmp/maya_serve.log`，
  命令管道 `/tmp/maya_in.fifo`。
- 换卡数/上下文重启用 `/tmp/serve.sh`；测速批处理用 `/tmp/bench_run.sh`（stdin 读 `/tmp/cmds.txt`）。
- 关键日志：`/tmp/maya_run.log`（首轮测速）、`/tmp/maya_batch.log`（3×256 decode 复测）。
