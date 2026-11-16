# Maya GLM 引擎：直接命令行跑法与 bench 测量口径

调研 2026-11-15。只做了静态读码 + 本机实测 tokenizer/env（MUSA SDK 5.2.0 + 4×MTT S4000）。**build-musa/maya 仍在编译中（glm_fast.cu 编译耗时很长），`--help` 实测未完成**，见 §10。没跑任何模型。

## 0. 结论（先看这 8 条）

1. **入口就是 `./build-musa/maya --glm-pack <模型目录>/pack`**：没有 `--model`；模型 3 个 shard 由引擎自己在 pack 的**上级目录**找（`maya.py:3429` 注释 "the engine finds the GGUF files at `<pack>/..`"，实现 `src/core/glm_model.cu` `load_pack`）。`--glm-pack` 会覆盖默认的 `--pack pack/full`。
2. **跑之前必须先建 pack**（引擎的索引 + ~1 GB 小浮点权重 + tokenizer），命令在 §7 第一段；在 3 个 shard 全部下完之前建不了。我们那台机器 17:10 时 shard2 还在下。
3. **4 卡用 `--layer-split auto`**（≥2 张可见卡时 split 本来就是默认，`MAYA_GLM_SPLIT=0` 强制单卡）。切法是**按层切**，不是张量并行、也不是数据并行：每个 part 有自己的专家池和 CPU lane；≥3 卡 + 多请求时是流水线（`src/core/glm_model.cu:2077` 起）。
4. **MUSA 上选卡只认 `MUSA_VISIBLE_DEVICES`**（本机实测：`CUDA_VISIBLE_DEVICES`、`MTHREADS_VISIBLE_DEVICES` 都被忽略，仍然是 4 张卡）。maya.py 只会设 `CUDA_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES`（`serve/server.py:2036`），所以绕过 maya.py、直接 `export MUSA_VISIBLE_DEVICES=0,1,2,3` 最省事。
5. **maya 没有 `--bench`/`--timing` 开关**。测速就是起 `--serve`，从 stdin 发 `GEN <max_new> [k=v] <ids>`，解析 stdout 的 DONE 行。`./maya.sh --bench` 也只是这样驱动引擎（`maya.py:5661-5790`）。
6. **bench 的测量口径**：3 个固定题目、贪心（`temperature=0`）、每答案最多 **256** token、先跑 1 个 64-token 热身；prefill 用本仓文档拼的 **2048** 与 **8192** token 提示词各测一次（先各热一次）。decode tok/s = `generated/decode_ms*1000`（只统计 ≥64 token 的答案），prefill tok/s = `提示词tokens/prompt_ms*1000`。
7. **我们没编 MMQ —— 这里的后果比"走 FP16 兜底"严重**：GLM 引擎的 prompt 路径（`src/core/glm_prefill.cu`）在 `mmq::built()` 为假时直接打印 "this build has no MMQ kernels - the prompt runs token by token" 并把 `prefill_chunk()` 返回 0，于是**整个 prompt 逐 token 过 decode 路径**；不存在"FP16 expert GEMM 兜底"。8K prompt 预计是**分钟级**（约 8191 × 单 token decode 时间）。详见 §5。
8. **MUSA 上的 4 卡速度没有公开参考**。最接近的同类数据：3×V100 + RTX3060（PCIe 3.0，专家多数在 RAM）单请求 45-46 tok/s、4 并发 20-21 tok/s（`bench/results/2026-11-02-4gpu-4x128k/README.md`）。S4000 单卡 768 GB/s、4 卡 192 GB 显存 > 模型 96.5 GB，专家可全部驻留 VRAM，decode 有希望更好，但必须先跑出数。

## 1. 二进制与代码位置

- CMake target：`add_executable(maya src/program/generate.cpp)` + 链 `strata_engine strata_prefill strata_spec`（`CMakeLists.txt:706`）。所有子命令（GLM / Strata pack / ds4 / mm3）是不同二进制，GLM 引擎只有 `maya` 一个。
- `--glm-pack` 走 `glm_pack_generate()`（`src/program/generate.cpp:1093` 定义，`:5047` 分派）；不是 `--glm-pack` 时走老的 Strata `--pack` 路径（那条路要 `--tokens` 且有自己的 usage）。
- 命令行解析在 `generate.cpp:4730-4990`，`usage()` 在 `generate.cpp:373`（打到 stderr，`--help` 后 return 0）。注意这份 usage 是 Strata 视角写的，`--glm-pack` 的语义要看 §2 的代码位置。

## 2. 关键参数（GLM 引擎）

| 参数 | 默认 | 作用 / 代码 |
|---|---|---|
| `--glm-pack DIR` | 无（缺省用 `--pack pack/full`） | 模型 pack 目录；shard 在 `DIR/..`。`generate.cpp:4739`、`:2093` |
| `--serve` | off | 用 stdin/stdout 协议跑（READY / GEN / DONE），见 §6 |
| `--tokens "1,2,3"` / `--tokens-file` | 无 | 非 serve 模式必填（引擎内**没有 tokenizer**，只收 token id，`generate.cpp:4099`） |
| `--max-new N` | 16 | 生成多少 token |
| `--max-context N` | 4096 | KV/状态容量；bench 里决定 8K prefill 能不能做（要 ≥8192+64） |
| `--kv fp16\|int8\|q4_0\|k8v4` | fp16 | KV 存储格式；`--serve` 的 `--kv-resident`/`--kv-streaming` 是它的流式版本 |
| `--expert-cache auto\|N` | 0/auto | VRAM 专家池大小；`auto` = 空闲显存减去 reserve，写满后再校验 |
| `--expert-cache-per-layer` | off | 每层各自分槽（否则按到达顺序，命中率会塌） |
| `--vram-reserve-mib N` | 700 | 专家池旁边留的显存 |
| `--prefill auto\|N\|0` | auto | GLM 引擎把它转成 `MAYA_GLM_PREFILL`（`generate.cpp:1114`）。**无 MMQ 时 auto/N 都退化成逐 token** |
| `--layer-split auto\|K1,K2,..` | 空 = auto | 多卡按层切；`auto` 按各卡空闲显存搜索边界（2-4 卡），也可写死 `10,21,33` 这样每卡第一层 |
| `--split-device d1,d2..` | 空 = 可见卡依次 | 显式指定参与切分的卡 |
| `--parallel N` | 1 | 同时解码的请求数（1-8）；单卡共享权重读出，多卡是流水线 |
| `--greedy` / `--seed` / `--top-k` / `--top-p` / `--temperature` | 贪心 | 采样；bench 用 `GEN` 行里的 `temperature=0` 覆盖 |
| `--expert-profile P` | 空 | 启动时按 profile 预热 VRAM 专家池（省启动时间；GLM 包可以不放） |
| `--vision` | off | 允许 GENI（图片）请求；文本不需要 |

常用环境变量（`docs/SETTINGS.md` 有全表）：`MAYA_GLM_SPLIT`（auto/0/边界）、`MAYA_GLM_DEVS`（参与切分的卡号）、`MAYA_GLM_RAM_GB`（pin 的 RAM 层大小，默认按空闲内存算）、`MAYA_GLM_RAM_HEADROOM_GB`（默认 6）、`MAYA_GLM_RESERVE_MB`、`MAYA_GLM_EXPERT_CACHE`、`MAYA_GLM_PREFILL`、`MAYA_GLM_NO_PREFILL=1`（强制逐 token）、`MAYA_GLM_NO_MTP` / `MAYA_GLM_NO_SPEC`（关 NextN 草稿）、`MAYA_GLM_PROF`（打印各卡耗时占比）。

## 3. 多卡怎么切（问题 1 的补充）

- 默认就切：`load_pack_place()` 注释写"the layer split is the DEFAULT on multi-GPU hosts"（`src/core/glm_model.cu:2074`）。`MAYA_GLM_SPLIT=0` 才单卡。
- `--layer-split auto` → `glm_search_bounds()`，按每卡放完自己层后的空闲显存 + 专家覆盖曲线挑最快的布局（2-4 卡），不是均分。
- 显式边界示例（4 卡 48 GB）：`MAYA_GLM_SPLIT=auto` 或 `--layer-split 12,24,36`（每张后卡的起始层）。
- 每张卡有自己的专家池；**一次请求的 token 是串过所有 part 的**，所以单请求速度取决于最慢那张卡；多请求（`--parallel`）才并行。
- 也可以只暴露部分卡：`MUSA_VISIBLE_DEVICES=0,1` 让引擎只看到 2 张（此时设备号重排为 0,1）。

## 4. 模型加载需要哪些文件（问题 4）

```
/root/autodl-tmp/models/Maya-S-v2-IQ2_XXS/
├── GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-0000{1,2,3}-of-00003.gguf   # 必需（引擎在 pack/.. 找）
├── pack/                                                        # 必需（工具生成，见 §7）
│   ├── index.txt            # 张量索引（每个 tensor 在哪个 shard 的偏移）
│   ├── native_experts.txt   # 每层专家的位置/类型
│   ├── dense.bin            # ~1 GB 小浮点权重（router、共享专家等）
│   ├── tokenizer/{vocab.json,merges.txt,token_type.json,chat_template.jinja}   # bench/服务用
│   └── expert-profile.bin   # 可选（预热用，跑过会自己写）
└── vision/
    ├── GLM-5.3-Flash-vocab.gguf        # 9.4 MB
    └── mmproj-GLM-5.3-Flash-F16.gguf   # 1.1 GB，vision encoder
```

- **vision 不是必需的**：文本请求不需要 `vision/`；只有 `--vision` + GENI 图片请求才用（`generate.cpp:7890` 检查 `--vision`）。`mmproj` 是图片编码器。
- **`vision/GLM-5.3-Flash-vocab.gguf` 的用途未在引擎代码里找到引用**（grep 未见）；它大概率给 `maya-vision`/服务端做视觉侧的 token 表用。文本跑法的结论：**不需要它**。
- **tokenizer 必需（对 bench/服务而言）**：引擎本身不需要（只收 id），但 `maya.py --bench` 要用 `<pack>/tokenizer` 编码提示词；缺了它 bench 直接 fail。文本 tokenizer 由 pack 构建写入。
- 缺 pack / 缺 shard：`load_pack` 失败，起不来（不是回退）。

## 5. prefill 路径核查（问题 5）——和"FP16 expert 兜底"的假设不同

**结论：没编 MMQ 时，prefill 能跑，但整个 prompt 是逐 token 过 decode 路径，不是 FP16 GEMM 的批量 prefill。**

- GLM 引擎的 prompt 路径是 `src/core/glm_prefill.cu`（`Glm5Model::prefill_setup`），**不是** `src/prefill/prefill.cpp`（那是 Strata `--pack` 路径和测试用的库；`grep -rln "strata/prefill/prefill.hpp" src include` 只命中 prefill.cpp 自己和 generate.cpp）。
- `glm_prefill.cu:1131`：
  ```cpp
  if (!mmq::built()) {
      std::fprintf(stderr, "glm prefill: this build has no MMQ kernels - the prompt runs token by token\n");
      return true;   // 不是错误：起得来，只是没有批量 prefill
  }
  ```
  `prefill_chunk()`（`glm_prefill.cu:1048-1051`）在 `pf_ == nullptr` 时返回 0；`generate.cpp:1995` 据此走 else 分支，prompt 全部用 `model.forward()` 一个 token 一个 token 读。
- `expert_f16()`（`glm_prefill.cu:291`）那条"把专家反量化成 FP16 再用 BLAS 乘"的路，只在 **MMQ 已启用**、且个别专家类型 MMQ 不支持时才生效（`!expert_mmq(g,t) && mmq::to_f16_supported(t)`）。MMQ 完全没编时 `mmq::supported()` 恒假，`prefill_setup` 在更早的地方就 return 了，所以 FP16 那条根本不会跑。
- 影响：8K prompt ≈ 8191 次单 token 前向。假设 decode 25 ms/token，就是 **~3.4 分钟**；`bench` 的 8K prefill 数字会非常难看（但数值正确）。启动日志里能看到上面那行 `glm prefill:`，可用来确认。
- 想恢复批量 prefill（后续大杠杆）：CMake 现在把 `STRATA_PREFILL_MMQ` 硬绑 `STRATA_ENABLE_HIP`（`CMakeLists.txt:70-77` 对 HIP 是 FATAL_ERROR 要求打开），MUSA 分支需要放开这个限制，然后把 `src/prefill/moe_mmq.cu`、`ggml_cuda_host.cu`、ggml-cuda 的 `quantize.cu`/`convert.cu` 和 15 个 `mmq-instance-*.cu` 用 mcc + musa_compat 编成 `strata_mmq`（照 `CMakeLists.txt:1600` 的 HIP 版本改）。这是新工作量，不在本任务范围。

## 6. bench 到底怎么测（问题 2，逐条对齐 maya.py）

`maya.py` 的 `bench()`（`:5661`）：起 `cfg["exe"] --serve <engine_args>`，环境 `child_env(cfg)`，等 `READY`，然后用 stdin 发 `GEN` 行，从 stdout 读 `DONE`。

| 项 | 值 |
|---|---|
| 题目 | 固定 3 条（写 Python CSV 解析 / 讲光合作用 / 写 canvas 弹球 HTML），`maya.py:5602` |
| 对话格式 | GLM 的标记 + thinking off：`"[gMASK]<sop><|user|>\n{}<|assistant|>\n</think>"`（`maya.py:5624`） |
| 采样 | 贪心：`GEN` 行带 `temperature=0`（`maya.py:5640`） |
| 每答案上限 | **256** token（`maya.py:5724` `gen(chat(q), 256)`） |
| 热身 | 先 `GEN 64` 回答 "Say hello in five languages."（首答偏慢，不计入） |
| decode 统计 | `generated / decode_ms * 1000`；只把 **≥64 token** 的答案算进均值（`BENCH_MIN_ANSWER`，`maya.py:5618`、`:5737`） |
| prefill | 提示词 = 仓库 README + `docs/*.md` + `bench/results/*.md` 拼起来再 tokenize；测 **2048** 和 **8192** token 两档（要求 `n+64 <= --max-context` 且 `n <= doc` 长度）；每档先发一个 1-token 生成的读热身，再测；`prefill tok/s = n / prompt_ms * 1000` |
| 输出 | `maya-bench.txt`（3 行头 + 每条 decode + 每条 prefill + 引擎日志尾巴） |

**DONE 行字段**（`generate.cpp:8729`，注释在 `:7045`）：

```
DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <finish> <drafts accepted> <drafts offered> <reused> [hits] [lookups]
```

**两个口径细节**（值得写进报告）：
- `prompt_ms` 从请求开始到 prompt 读完，只含 **n-1** 个 prompt token；最后一个 prompt token 的 forward 在 decode 窗口里（`generate.cpp:8377` "the verify windows: the first holds the last prompt token alone"）。所以 prefill 的 tok/s 用 `n/prompt_ms` 会**略高**、decode 的 tok/s 会**略低**（首个输出 token 的那次前向算在 decode 里）。
- `decode_ms` 是墙钟，包含采样、Python 侧无关；对比 HTTP 的 `/v1/chat/completions` 数字会低一点（后者从"首个 token 到达客户端"起算）。

**协议行**（`generate.cpp:4305`、`:7840-7870`）：`READY <ctx> stop pause [multi=N]`；`GEN <max_new> [temperature= top_p= top_k= seed= rid= ...] <id,id,...>`；`DONE ...` / `ERR ...`；`QUIT`。还有要忽略的行：`REUSED <n>`、`PP ...`（批量 prefill 进度，我们没 MMQ 不会出现）、`NOTE ...`、`T <rid> <id>`（serve 的流式 token 行，bench 不读）**——解析时只认 DONE/ERR**。

## 7. 最小可跑命令（4 卡 / 8K）（a）

```bash
# 0) 前提：3 个 shard 下完（缺失会 load_pack 失败）。建 pack（几分钟，CPU 即可）：
export MAYA_GGUF_PY=/root/autodl-tmp/llama.cpp/gguf-py
cd /root/autodl-tmp/project-maya
python3 tools/iq_pack.py \
  --gguf /root/autodl-tmp/models/Maya-S-v2-IQ2_XXS/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf \
  --out  /root/autodl-tmp/models/Maya-S-v2-IQ2_XXS/pack \
  --compat-bf16
# 产物校验：pack/{index.txt,native_experts.txt,dense.bin,tokenizer/vocab.json,...}

# 1) 起引擎（4 卡整机、8K 上下文；--serve 的 stdin 是命令通道，日志在 stderr）
export MUSA_VISIBLE_DEVICES=0,1,2,3
export LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH
cd /root/autodl-tmp/project-maya
./build-musa/maya --serve \
  --glm-pack /root/autodl-tmp/models/Maya-S-v2-IQ2_XXS/pack \
  --max-context 8192 \
  --expert-cache auto \
  --prefill auto \
  --kv-streaming off \
  --layer-split auto
```

逐行作用与坑：
- `MUSA_VISIBLE_DEVICES`：**唯一有效**的选卡变量（实测 `CUDA_VISIBLE_DEVICES`/`MTHREADS_VISIBLE_DEVICES` 无效果）。写 `0,1` 就只用两张，设备号在引擎里重排成 0,1。
- `LD_LIBRARY_PATH=/usr/local/musa/lib`：用 `musart`/`mublas` 动态库；直接跑二进制（不经 maya.py）时要自己加。
- `--glm-pack`：给 **pack 目录**，不是 shard 路径；shard 放在它的上一层。
- `--max-context 8192`：8K；bench 的 8K prefill 需要 `8192+64 <= max-context`，所以想测 8K 得给 **≥8256**（maya.py 的 config 一般给 32768 或更大）。**用 `--max-context 16384` 更省事**。
- `--expert-cache auto`：把空闲显存给专家池（`--vram-reserve-mib` 默认 700）。4×48 GB 装 96.5 GB 模型，按层切后每卡约 24 GB 权重，**很可能全部驻留 VRAM**（`MAYA_GLM_COMPLETE_POOL` 默认 on），启动会一次性搬，首次启动慢。
- `--prefill auto`：无 MMQ 时只打印 `glm prefill: this build has no MMQ kernels - the prompt runs token by token`，实际逐 token；保留这个参数是为了将来换回 MMQ.
- `--kv-streaming off`：maya.py 对 glm-dsa 包写 off（`G53.KV_STREAMING` 为 False），其它包写 `on`（`maya.py:125`、`:3910`）；8K 上下文 KV 不大，off 最简单。
- `--layer-split auto`：让引擎自己搜层边界；也可以写死 `--layer-split 12,24,36`（后 3 张卡的起始层）。
- 起不来时看 stderr：`READY <ctx> stop pause` 是加载完成的标志；`glm prefill:`、`glm fast: CUDA0 ...`、`glm split:` 这些行说明它选了哪条路。

发一条请求（token id 由 tokenizer 生成；本机实测这条 prompt 的 id 是 13 个）：

```bash
python3 -c "import sys; sys.path.insert(0,'/root/autodl-tmp/project-maya/tools'); import strata_tokenizer as ST; tk=ST.Tokenizer.from_dir('/root/autodl-tmp/models/Maya-S-v2-IQ2_XXS/pack/tokenizer'); print(','.join(map(str, tk.encode('[gMASK]<sop><|user|>\n你好<|assistant|>\n</think>', parse_special=True))))"
# 然后在引擎的 stdin 里发（示例用实测的 13 个 id）：
# GEN 8 temperature=0 154822,154824,154827,198,45494,23745,304,4236,15447,13,154828,198,154842
```

## 8. 测 prefill / decode 速度（b）

现成脚本：`/root/autodl-tmp/research/bench_glm.py`（复刻 maya.py `--bench` 的口径，语法已过 `py_compile`，但**没有引擎/pack 可跑，未实测**）：

```bash
python3 /root/autodl-tmp/research/bench_glm.py \
  --exe /root/autodl-tmp/project-maya/build-musa/maya \
  --pack /root/autodl-tmp/models/Maya-S-v2-IQ2_XXS/pack \
  --gpus 0,1,2,3 --ctx 16384 --max-new 256
# 快速冒烟：--topics 1 --no-prefill
```

它做的事：`MUSA_VISIBLE_DEVICES` + `--serve --glm-pack ... --layer-split auto` 起引擎 → 等 READY → 热身 `GEN 64` → 3 题各 `GEN 256 temperature=0` → 2048/8192 两档 prefill 各 `GEN 1` → 打印每行 tok/s 与均值，stderr 落到 `research/bench_glm.stderr.log`。

不想用脚本时，最小的手工口径（等价于 maya.py 的单条 decode）：

```bash
# 起引擎（同 §7），另开一个管道喂 3 行：热身 + 1 条测试 + 退出
{ echo "GEN 64 temperature=0 <warmup_ids>"; sleep 1;
  echo "GEN 256 temperature=0 <ids>"; sleep 600;
  echo "QUIT"; } | ./build-musa/maya --serve ... 2>engine.log | tee out.log
# out.log 里找 DONE <gen> <prompt> <prompt_ms> <decode_ms> ...
#   decode tok/s  = gen / decode_ms * 1000
#   prefill tok/s = prompt / prompt_ms * 1000
```

prefill 的坑：**8K prompt 在无 MMQ 下是逐 token 读**，可能几分钟，别以为卡死；`PP` 进度行不会出现（那是批量 prefill 才有），只能等 `DONE`。

## 9. 绕过 maya.py / 让 maya.py 用我们的二进制（问题 3）

- **完全不需要 maya.py**：§7 两条命令就是全部。maya.py 只是"下载 → 建 pack → 写 config → 起进程"的封装。
- maya.py 怎么定位引擎：config JSON 的 `"exe"` 直接 exec（`maya.py:102` `EXE = BUILD/"maya"`，`serve/server.py` 用它起进程）；**没有**任何环境变量能让它指向 `build-musa/maya`，只能改 config 或 `--config` 指一个新的。
- 我们手工编译的二进制**不需要**进 `build/`：直接跑即可。若想让 `./maya.sh --bench` 驱动它，写一个最小 config（`backend` 别写 hip/cuda，`gpu` 写卡号）：

```json
{
  "exe": "/root/autodl-tmp/project-maya/build-musa/maya",
  "args": ["--glm-pack", "/root/autodl-tmp/models/Maya-S-v2-IQ2_XXS/pack",
           "--max-context", "16384", "--expert-cache", "auto",
           "--prefill", "auto", "--kv-streaming", "off"],
  "cwd": "/root/autodl-tmp/project-maya",
  "tokenizer": "/root/autodl-tmp/models/Maya-S-v2-IQ2_XXS/pack/tokenizer",
  "gpu": [0, 1, 2, 3],
  "port": 8080
}
```
  ```bash
  export MUSA_VISIBLE_DEVICES=0,1,2,3          # maya.py 的 child_env 只会设 CUDA/HIP_VISIBLE_DEVICES
  python3 maya.py --config /path/to/maya-musa.json --bench
  ```
  坑：maya.py 的 `--bench` 会先查 `http://127.0.0.1:<port>/health`，有服务在跑就直接 fail（"the benchmark needs the GPU memory it holds"），所以先停掉 maya 服务；`gpu` >1 张时 `engine_args()` 会自动补 `--layer-split auto`（`serve/server.py:2027`），别重复写。

## 10. 未验证 / 待补

- **`--help` 实测未完成**：`build-musa` 仍在编译（`glm_fast.cu` 单文件 mcc 编译要十几分钟以上，本机 128 核也没用，mcc 单线程前端）。编译完先跑 `./build-musa/maya --help`（打到 stderr）核对 §2 的参数表，以及 `./build-musa/maya --version`（**generate.cpp 里没找到 --version**，属于未确认）。
- **pack 还没建**（模型 17:10 时 shard2 仍在下载）：pack 构建耗时与产物大小未实测（代码注释说 "a few minutes"、dense.bin ~1 GB）。
- **没跑过真实请求**，所以 4×S4000 的 load 时间、VRAM 是否真能全驻留、decode tok/s 全是估计，不是实测。
- 逐 token prefill 的实际耗时倍数：依赖 decode 单 token 时间，未测。
- `vision/GLM-5.3-Flash-vocab.gguf` 的消费方（服务端/vision helper）未在引擎代码里定位到。

## 11. 速查清单

| 想做的事 | 命令/开关 |
|---|---|
| 只跑 2 张卡 | `MUSA_VISIBLE_DEVICES=0,1` + `--layer-split auto` |
| 单卡跑 | `MUSA_VISIBLE_DEVICES=0` + `MAYA_GLM_SPLIT=0` |
| 更长的上下文 | `--max-context 32768`（8K prefill 测试要求 ≥8256） |
| 关掉 NextN 草稿（A/B） | `MAYA_GLM_NO_MTP=1` 或 `MAYA_GLM_NO_SPEC=1` |
| 关掉批量 prefill（对照） | `MAYA_GLM_NO_PREFILL=1`（我们这版本来就是逐 token） |
| 看各卡耗时占比 | `MAYA_GLM_PROF=1`（每卡一行） |
| 确认自己有没有 MMQ | 启动日志里没有 `glm prefill: this build has no MMQ kernels` 就是有 |
| 确认选卡生效 | 启动日志的 `Device 0..3: MTT S4000` 行数（或 `MUSA_VISIBLE_DEVICES` 前后对比） |

源码位置（GitHub 同一份：github.com/mw00/project-maya）：`src/program/generate.cpp:373`（usage）、`:1093`（glm_pack_generate）、`:1995`（prompt 分块）、`:4730-4990`（CLI）、`:7045/:8729`（DONE）、`src/core/glm_prefill.cu:1048/1131`（prefill 开关）、`src/core/glm_model.cu:2049`（split/env）、`maya.py:5602-5790`（bench）、`serve/server.py:2027-2060`（engine_args/child_env）。
