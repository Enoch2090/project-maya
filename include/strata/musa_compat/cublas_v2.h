#pragma once
// cuBLAS-shaped sources on muBLAS: same entry points, same argument order (mublas follows the cublas
// naming), MUSA_R_* datatypes in place of CUDA_R_*.
#include <mublas_v2.h>
#include <library_types.h>

#define CUBLAS_COMPUTE_32F MUBLAS_COMPUTE_32F
#define CUBLAS_DEFAULT_MATH MUBLAS_DEFAULT_MATH
#define CUBLAS_GEMM_DEFAULT MUBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP MUBLAS_GEMM_DEFAULT_TENSOR_OP
#define CUBLAS_OP_N MUBLAS_OP_N
#define CUBLAS_OP_T MUBLAS_OP_T
#define CUBLAS_PEDANTIC_MATH MUBLAS_PEDANTIC_MATH
#define CUBLAS_STATUS_EXECUTION_FAILED MUBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_SUCCESS MUBLAS_STATUS_SUCCESS
#define cublasCreate mublasCreate
#define cublasDestroy mublasDestroy
#define cublasGemmEx mublasGemmEx
#define cublasGemmStridedBatchedEx mublasGemmStridedBatchedEx
#define cublasHandle_t mublasHandle_t
#define cublasSetMathMode mublasSetMathMode
#define cublasSetStream mublasSetStream
#define cublasSetWorkspace mublasSetWorkspace
#define cublasSgemm mublasSgemm
#define cublasSgemmStridedBatched mublasSgemmStridedBatched
#define cublasStatus_t mublasStatus_t
#define cublas_v2 mublas_v2
#define cudaDataType musaDataType
