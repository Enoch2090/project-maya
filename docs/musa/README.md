# MUSA (Moore Threads) — the port, its numbers, and the work behind them

The entry point is [`../MUSA_MAYA.md`](../MUSA_MAYA.md): how to build the MUSA backend, how to run it, and what
it limits. This folder holds the measurements and the experiments those two files are based on.

Measured on 4× MTT S4000 48 GB, MUSA 5.2.0 / mcc 5.2.0 (clang 14), model Maya-S-v2-IQ2_XXS (96.5 GB, every
expert resident in VRAM), 16K context, one request at a time.

| | |
|---|---|
| prefill, 8K prompt | **251 tok/s** (32.6 s), 2K: 132 tok/s |
| decode, single stream | **17.1 tok/s** (`--dflash … --spec 5`) |
| start to `READY` | ~40 s |

## The documents

| file | what it is |
|---|---|
| [`PERF_ANALYSIS.md`](PERF_ANALYSIS.md) | the running record: what each fix did to the two numbers, what failed and why, what is left |
| [`DEPLOY_REPORT.md`](DEPLOY_REPORT.md) | the deployment as delivered: model layout, pack, the exact commands |
| [`MUSA_PORT_RESEARCH.md`](MUSA_PORT_RESEARCH.md) | prior art: llama.cpp's own MUSA backend, the macro maps, the S4000's architecture |
| [`MMQ_MUSA_FEASIBILITY.md`](MMQ_MUSA_FEASIBILITY.md) | whether ggml's MMQ kernels can build and run here (they can), and the CMake needed |
| [`MMQ_MUSA_PREFILL_NUMERIC.md`](MMQ_MUSA_PREFILL_NUMERIC.md) | the hunt for the under-written rows — led to the host/device warp-width bug |
| [`MUSA_MOE_GEMM.md`](MUSA_MOE_GEMM.md) | the MoE GEMM study: one `mublasGemmStridedBatchedEx` per layer is 7.2× a per-expert GEMM |
| [`KERN_MUSA_GEMV.md`](KERN_MUSA_GEMV.md) | the quantized GEMV study: 2.2–3.0× over the engine's own shape, and where the ceiling is |
| [`MAYA_RUN_BENCH.md`](MAYA_RUN_BENCH.md) | how to run the engine by hand and how `--bench` measures (prompt lengths, token counts) |
| `kern-musa/`, `kern-prefill/` | the benchmark programmes behind the two studies, buildable with `mcc` |

## What the port changed

- `include/strata/musa_compat/` — the mapping layer, MUSA's counterpart of `hip_compat/`.
- `cmake/musa_backend.cmake` — `-DSTRATA_ENABLE_MUSA=ON`; `.cu` files compile as CXX through `mcc` with
  `-x musa`, force-included with `cuda_runtime.h`, exactly as the AMD backend does it.
- 74 sources gained `STRATA_USE_MUSA` in their `STRATA_USE_HIP` conditionals, so MUSA takes the same portable
  branches AMD does.
- `src/kernels/cuda/exl3_stub_musa.cpp` — EXL3 weights are not built here (mcc's device codegen does not
  finish on `exl3.cu`, and the GLM-5.3-Flash GGUFs carry no EXL3 weights).

## Three bugs worth remembering

1. **MMQ wrote only a prefix of its output rows.** The host launch took `prop.warpSize` — 128 on these cards,
   the hardware width — while the device code is compiled for a logical 32-lane warp, giving
   `block_dims=(128,2)` where the kernel expects `(32,8)`. Fixed in `src/prefill/ggml_cuda_host.cu`.
2. **`gcnArchName` is a HIP string and MUSA has none.** The compat layer maps it onto the card's name, so every
   `strncmp(prop.gcnArchName, "gfx11…")` gate answered false and quietly disabled engine paths written for
   AMD: the tuned expert kernels (`rdna3_expert_device`), the latency-hiding dense GEMV (`mv_rdna_device`), the
   MTP decode gate. Each needed its own MUSA branch.
3. **`__CUDA_ARCH__` is never defined under `mcc`.** Every `#if __CUDA_ARCH__ >= 610` picked the FMA tier.
   The compat layer now answers it for the device pass only (610, the `__dp4a` tier), which is what the
   hardware supports.

## Where the remaining gap is

`decode` at 17.1 tok/s needs 20 ms/token for 50. Of the current 66 ms, only 31–33 ms is GPU stream time;
the rest is kernel-launch queueing (roughly a thousand launches a token at ~23 µs) and the three cross-card
hops of the layer split. The MoE (43% of the GPU time) runs at ~92 GB/s against a 368 GB/s ceiling and is
close to what its current kernel shape gives — the larger wins left are folding the launch storm (graph
capture) and making the four cards overlap instead of running one after another. `PERF_ANALYSIS.md` has the
numbers.
