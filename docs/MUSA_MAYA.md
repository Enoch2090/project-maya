# GLM-5.3-Flash on MUSA (Moore Threads)

The MUSA backend runs Maya's GLM engine on Moore Threads cards through `mcc`. It is built the way the AMD
backend is built: the sources stay CUDA-shaped, and a header layer maps `cudaXxx` onto `musaXxx`.

Measured on 4× MTT S4000 48 GB (MUSA 5.2.0, driver 5.2.0, mcc 5.2.0 = clang 14), Maya-S-v2-IQ2_XXS
(96.5 GB, all experts in VRAM):

| | |
|---|---|
| **Decode** | **17.1 tok/s** single stream (mean of three greedy 256-token answers: 17.00 / 15.95 / 18.39; 59–65 ms/token) with the DFlash2 drafter (`--dflash … --spec 5`) |
| **Prefill** | **251 tok/s** at an 8K prompt (32.6 s), 131 tok/s at 2K - the FP16 expert products over RAI/muBLAS with whole-chunk sub-batches (see Known limits for what still bounds it) |
| Start | ~40 s to `READY` (97.6 GB of weights read from NVMe at 5.5–6 GB/s per card) |

## What the port adds

- `include/strata/musa_compat/` — the mapping layer, the MUSA counterpart of `hip_compat/`:
  `cuda_runtime.h` (147 `cudaXxx`→`musaXxx` names, plus the compiler questions: MUSA defines `__MUSACC__`
  in both passes and `__MUSA_ARCH__` only in the device pass, so the port answers `__HIPCC__` and
  `__HIP_DEVICE_COMPILE__` with them and the sources' already-vetted non-NVIDIA branches compile as they
  do on AMD), `cublas_v2.h` (cuBLAS → muBLAS, `CUDA_R_*` → `MUSA_R_*`), `cuda_fp16.h`, `cuda_bf16.h`
  (`__nv_bfloat16` → `__mt_bfloat16`), `cuda_fp8.h`, `cuda_profiler_api.h`, `cuda.h`, `intrinsics.hpp`
  (`__nanosleep`, and the `__ldcg(float)` / `__ldcg(unsigned)` overloads MUSA's headers lack).
- `cmake/musa_backend.cmake` — `-DSTRATA_ENABLE_MUSA=ON`. CMake has no MUSA language, so the `.cu` files
  compile as CXX through `mcc` with `-x musa;--offload-arch=<arch>;-O3`, and every translation unit is
  force-included with `cuda_runtime.h`, exactly as `hip_backend.cmake` does it. The C compiler is MUSA's
  own `clang` (mcc refuses C input), which keeps one compiler family.
- The sources' `STRATA_USE_HIP` conditionals became `(STRATA_USE_HIP || STRATA_USE_MUSA)`, so MUSA takes
  the same portable branches AMD does (no `mma.sync`, no `cp.async`, the F32 attention, the non-WMMA
  prompt path). 73 files, 556 insertions, 439 deletions.
- `src/kernels/cuda/exl3_stub_musa.cpp` — EXL3 (exllamav3) weights are not built for MUSA: mcc's device
  pass does not finish on `exl3.cu` (30+ minutes, no output), and the GLM-5.3-Flash GGUFs carry no EXL3
  weights. `-DSTRATA_MUSA_EXL3_KERNELS=ON` builds the real file if you want to wait.
- `STRATA_PREFILL_MMQ` is now allowed for MUSA and builds ggml's MMQ instances through mcc (they compile;
  the host and device MMQ configs agree, which is the correctness precondition). See Known limits for the
  numerical problem that surfaced.

## Build

```bash
export MAYA_GGUF_PY=<llama.cpp>/gguf-py
cmake -S . -B build-musa -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/local/musa/bin/mcc \
  -DCMAKE_C_COMPILER=/usr/local/musa/bin/clang \
  -DSTRATA_ENABLE_MUSA=ON -DSTRATA_PREFILL_MMQ=ON \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_MUSA_ARCHITECTURES=mp_22 \
  -DSTRATA_GGML_DIR=<llama.cpp checkout at third_party/ggml/VERSION.txt>
cmake --build build-musa --target maya -j 24
```

`-DCMAKE_MUSA_ARCHITECTURES` matters: MTT S4000 is `mp_22` (`__MUSA_ARCH__` 220, compute capability 2.2);
a binary built without pinning it carries several architectures and picks one that may not be the card's.
`mp_21` is the S80/S3000 family.

## Run

```bash
export MUSA_VISIBLE_DEVICES=0,1,2,3      # the only device-selection variable MUSA honours
export LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH
./build-musa/maya --serve \
  --glm-pack <model dir>/pack \
  --max-context 16384 --expert-cache auto --prefill auto --kv-streaming off --layer-split auto
```

The pack is built with `python3 tools/iq_pack.py --gguf <shard 1> --out <model dir>/pack --compat-bf16`;
the shards live one directory above the pack. `--layer-split auto` gives 4 cards 11/12/12/10 layers at a
16384 context, with all 12096 experts in VRAM (93.5 GB of expert pool) and the CPU lane off.

`READY <ctx> stop pause` on stderr means loaded; `GEN <n> temperature=0 <ids>` on stdin is one request and
`DONE <generated> <prompt> <prompt_ms> <decode_ms> ...` the answer.

## Known limits

1. **The MMQ prompt path was numerically wrong on MUSA past a small prompt** (fixed in this port: the host launch geometry took the driver's hardware warp width, 128, where the kernel is compiled for 32; `src/prefill/ggml_cuda_host.cu` now reports 32 and pins the tier to dp4a).  What remains **slow** is MMQ's throughput on MUSA, so the FP16 expert products are the default here (`MAYA_GLM_MMQ=1` forces MMQ back).  Old description: With `--prefill auto` (8192-token
   chunks) a 2048-token prompt fails at layer 4 with `glm router: prompt layer 4 invalid expert ID or
   non-finite score (0 of 16376 routes)`; at `--prefill 128` the same corruption surfaces at layer 8
   (`0 of 1024 routes`), and at `--prefill 256` at layer 4 with 160 of 2048 routes invalid. Prompting with
   28 tokens works and is fast (230 tok/s). The engine drops the conversation's state when its NaN guard
   fires and re-reads the prompt token by token, so a long prompt still gets answered, at decode speed.
   The prime suspect is the portable `vec_dot` ggml selects for MUSA (its MMQ config resolves to
   `pascal_older`: no mma, no `__dp4a`), i.e. a kernel that was never exercised on this hardware —
   `tests/hip/prefill_mmq_parity.cpp` is the harness to port for a bit-level check.
2. **Decode is slower than the NVIDIA rows** for the same quant: 59–65 ms/token for 18 B active parameters
   across four cards. Profiling (`MAYA_GLM_PROF=1`) puts 31–33 ms of that in GPU stream time and the rest in
   kernel-launch queueing (~1000 launches a token at ~23 us here) plus the layer split's cross-card hops;
   the MoE, 43% of the GPU time, runs at ~92 GB/s against this card's 368 GB/s. `docs/musa/PERF_ANALYSIS.md`
   has the breakdown and what each attempt measured.
3. **EXL3 quants cannot be loaded** (see above); GGUF IQ quants are what this build serves.
4. **No vision encoder** in this build (the engine runs text only; `maya-vision` is a separate llama.cpp
   build and is not ported here).
5. The `--serve` dashboard/API front end (`serve/`) was not exercised; this port validated the engine.
