#pragma once
// Included only by STRATA_ENABLE_MUSA builds (through -include, as hip_compat is for HIP).
// It maps the CUDA-shaped sources onto MUSA: the same pure macro layer the AMD build uses, with
// cudaXxx -> musaXxx and cuBLAS -> muBLAS. Everything below the includes is a name mapping; the
// sources themselves stay unchanged, exactly as they do on HIP.
#include <musa_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>
// CUDA's warp intrinsics live in MUSA's device extension headers (mp_ext_30_intrinsics.h: __shfl_sync,
// __ballot_sync, __activemask, ...). They are device-only, so host translation units skip them.
#ifdef __MUSACC__
#include <mp_ext_30_intrinsics.h>
#endif
// The sources ask two questions that the CUDA compiler answers with __CUDACC__/__CUDA_ARCH__ and
// HIP with __HIPCC__/__HIP_DEVICE_COMPILE__.  MUSA's names are __MUSACC__ (defined in both passes)
// and __MUSA_ARCH__ (the device pass only, like __CUDA_ARCH__).  Both answers must be the non-NVIDIA
// ones: MUSA has neither mma.sync nor cp.async, and the sources' HIP branches are where their portable
// kernels live.  So "a GPU compiler" is __HIPCC__ and "the device pass" is __HIP_DEVICE_COMPILE__.
#if defined(__MUSACC__) && !defined(__HIPCC__)
#define __HIPCC__ 1
#endif
#if defined(__MUSA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
#define __HIP_DEVICE_COMPILE__ 1
#endif
// The same question, third form: the kernels pick their inner loop by tier with `#if __CUDA_ARCH__ >= 610`
// (below it a plain FMA dequantize-and-dot, at or above it __dp4a, four int8 products an instruction).  mcc
// defines no __CUDA_ARCH__, so every one of them took the FMA tier - on hardware that has __dp4a (measured).
// 610 is the dp4a tier and nothing above it (no mma, no cp.async, no PTX); 800-and-up asm paths stay out.
// Device pass only: __MUSA_ARCH__ is defined there alone, exactly where nvcc defines __CUDA_ARCH__.
#if defined(__MUSA_ARCH__) && !defined(__CUDA_ARCH__)
#define __CUDA_ARCH__ 610
#endif

// CUDA's annotation avoids per-thread copies of const kernel parameters; MUSA passes the same consts
// without this CUDA-only optimization annotation.
#ifndef __grid_constant__
#define __grid_constant__
#endif
// Do not let MUSA's legacy macro corrupt libstdc++ attribute names.
#ifdef __noinline__
#undef __noinline__
#endif

// ---- the name map (cudaXxx -> musaXxx), one line per symbol the sources use ----
#define CUDA_ERROR_OUT_OF_MEMORY MUSA_ERROR_OUT_OF_MEMORY
#define CUDA_MODULE_LOADING MUSA_MODULE_LOADING
#define CUDA_R_16BF MUSA_R_16BF
#define CUDA_R_16F MUSA_R_16F
#define CUDA_R_32F MUSA_R_32F
#define CUDA_SUCCESS MUSA_SUCCESS
#define cudaDevAttrCanUseHostPointerForRegisteredMem musaDevAttrCanUseHostPointerForRegisteredMem
#define cudaDevAttrClockRate musaDevAttrClockRate
#define cudaDevAttrComputeCapabilityMajor musaDevAttrComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor musaDevAttrComputeCapabilityMinor
#define cudaDevAttrGlobalMemoryBusWidth musaDevAttrGlobalMemoryBusWidth
#define cudaDevAttrHostNumaId musaDevAttrHostNumaId
#define cudaDevAttrIntegrated musaDevAttrIntegrated
#define cudaDevAttrMaxSharedMemoryPerBlockOptin musaDevAttrMaxSharedMemoryPerBlockOptin
#define cudaDevAttrMemoryClockRate musaDevAttrMemoryClockRate
#define cudaDevAttrMultiProcessorCount musaDevAttrMultiProcessorCount
#define cudaDeviceAttr musaDeviceAttr
#define cudaDeviceCanAccessPeer musaDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess musaDeviceEnablePeerAccess
#define cudaDeviceGetAttribute musaDeviceGetAttribute
#define cudaDeviceGetPCIBusId musaDeviceGetPCIBusId
#define cudaDeviceGetStreamPriorityRange musaDeviceGetStreamPriorityRange
#define cudaDeviceMapHost musaDeviceMapHost
#define cudaDeviceProp musaDeviceProp
#define cudaDeviceScheduleSpin musaDeviceScheduleSpin
#define cudaDeviceSynchronize musaDeviceSynchronize
#define cudaDriverEntryPointQueryResult musaDriverEntryPointQueryResult
#define cudaDriverEntryPointSuccess musaDriverEntryPointSuccess
#define cudaDriverGetVersion musaDriverGetVersion
#define cudaEnableDefault musaEnableDefault
#define cudaErrorInvalidConfiguration musaErrorInvalidConfiguration
#define cudaErrorMemoryAllocation musaErrorMemoryAllocation
#define cudaErrorNotReady musaErrorNotReady
#define cudaErrorPeerAccessAlreadyEnabled musaErrorPeerAccessAlreadyEnabled
#define cudaErrorStreamCaptureUnsupported musaErrorStreamCaptureUnsupported
#define cudaErrorUnknown musaErrorUnknown
#define cudaError_t musaError_t
#define cudaEventCreate musaEventCreate
#define cudaEventCreateWithFlags musaEventCreateWithFlags
#define cudaEventDestroy musaEventDestroy
#define cudaEventDisableTiming musaEventDisableTiming
#define cudaEventElapsedTime musaEventElapsedTime
#define cudaEventQuery musaEventQuery
#define cudaEventRecord musaEventRecord
#define cudaEventSynchronize musaEventSynchronize
#define cudaEvent_t musaEvent_t
#define cudaFree musaFree
#define cudaFreeHost musaFreeHost
#define cudaFuncAttributeMaxDynamicSharedMemorySize musaFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributes musaFuncAttributes
#define cudaFuncGetAttributes musaFuncGetAttributes
#define cudaFuncGetName musaFuncGetName
#define cudaFuncSetAttribute musaFuncSetAttribute
#define cudaGetDevice musaGetDevice
#define cudaGetDeviceCount musaGetDeviceCount
#define cudaGetDeviceProperties musaGetDeviceProperties
#define cudaGetDriverEntryPoint musaGetDriverEntryPoint
#define cudaGetDriverEntryPointByVersion musaGetDriverEntryPointByVersion
#define cudaGetErrorString musaGetErrorString
#define cudaGetLastError musaGetLastError
#define cudaGraphDestroy musaGraphDestroy
#define cudaGraphExecDestroy musaGraphExecDestroy
#define cudaGraphExec_t musaGraphExec_t
#define cudaGraphGetNodes musaGraphGetNodes
#define cudaGraphInstantiate musaGraphInstantiate
#define cudaGraphKernelNodeGetParams musaGraphKernelNodeGetParams
#define cudaGraphLaunch musaGraphLaunch
#define cudaGraphNodeGetType musaGraphNodeGetType
#define cudaGraphNodeType musaGraphNodeType
#define cudaGraphNodeTypeKernel musaGraphNodeTypeKernel
#define cudaGraphNodeTypeMemcpy musaGraphNodeTypeMemcpy
#define cudaGraphNodeTypeMemset musaGraphNodeTypeMemset
#define cudaGraphNode_t musaGraphNode_t
#define cudaGraphUpload musaGraphUpload
#define cudaGraph_t musaGraph_t
#define cudaHostAlloc musaHostAlloc
#define cudaHostAllocDefault musaHostAllocDefault
#define cudaHostAllocMapped musaHostAllocMapped
#define cudaHostAllocPortable musaHostAllocPortable
#define cudaHostGetDevicePointer musaHostGetDevicePointer
#define cudaHostRegister musaHostRegister
#define cudaHostRegisterMapped musaHostRegisterMapped
#define cudaHostRegisterPortable musaHostRegisterPortable
#define cudaHostUnregister musaHostUnregister
#define cudaInitDevice musaInitDevice
#define cudaKernelNodeParams musaKernelNodeParams
#define cudaLaunchHostFunc musaLaunchHostFunc
#define cudaMalloc musaMalloc
#define cudaMallocHost musaMallocHost
#define cudaMemGetInfo musaMemGetInfo
#define cudaMemcpy musaMemcpy
#define cudaMemcpy2D musaMemcpy2D
#define cudaMemcpy2DAsync musaMemcpy2DAsync
#define cudaMemcpyAsync musaMemcpyAsync
#define cudaMemcpyDefault musaMemcpyDefault
#define cudaMemcpyDeviceToDevice musaMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost musaMemcpyDeviceToHost
#define cudaMemcpyFromSymbol musaMemcpyFromSymbol
#define cudaMemcpyHostToDevice musaMemcpyHostToDevice
#define cudaMemcpyPeer musaMemcpyPeer
#define cudaMemcpyPeerAsync musaMemcpyPeerAsync
#define cudaMemcpyToSymbol musaMemcpyToSymbol
#define cudaMemset musaMemset
#define cudaMemsetAsync musaMemsetAsync
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor musaOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaPeekAtLastError musaPeekAtLastError
#define cudaProfilerStart musaProfilerStart
#define cudaProfilerStop musaProfilerStop
#define cudaRuntimeGetVersion musaRuntimeGetVersion
#define cudaSetDevice musaSetDevice
#define cudaStreamBeginCapture musaStreamBeginCapture
#define cudaStreamCaptureModeThreadLocal musaStreamCaptureModeThreadLocal
#define cudaStreamCaptureStatus musaStreamCaptureStatus
#define cudaStreamCaptureStatusNone musaStreamCaptureStatusNone
#define cudaStreamCreate musaStreamCreate
#define cudaStreamCreateWithFlags musaStreamCreateWithFlags
#define cudaStreamCreateWithPriority musaStreamCreateWithPriority
#define cudaStreamDestroy musaStreamDestroy
#define cudaStreamEndCapture musaStreamEndCapture
#define cudaStreamIsCapturing musaStreamIsCapturing
#define cudaStreamLegacy musaStreamLegacy
#define cudaStreamNonBlocking musaStreamNonBlocking
#define cudaStreamQuery musaStreamQuery
#define cudaStreamSynchronize musaStreamSynchronize
#define cudaStreamWaitEvent musaStreamWaitEvent
#define cudaStream_t musaStream_t
#define cudaSuccess musaSuccess

// The sources read CUDA/HIP's gcnArchName (a "gfx1201" string) to classify AMD cards for their tuning
// probes.  MUSA cards have no gfx name; the field maps onto the device name ("MTT S4000"), which matches
// none of those prefixes, so every AMD tuning probe takes its "unsupported" default - the portable kernel
// Maya must run here anyway.
#define gcnArchName name

// The sources spell a few device APIs with hip names inside their non-NVIDIA branches
// (device.cu's device flags, expert_cache.cpp's device query): the same MUSA calls answer them.
#define hipSuccess musaSuccess
#define hipGetDevice musaGetDevice
#define hipSetDevice musaSetDevice
#define hipGetDeviceProperties musaGetDeviceProperties
#define hipSetDeviceFlags musaSetDeviceFlags
#define hipDeviceProp_t musaDeviceProp

#include "intrinsics.hpp"
