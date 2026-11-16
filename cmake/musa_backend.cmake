# Opt-in MUSA (Moore Threads) configuration.  MUSA is a CUDA-shaped toolchain with a clang driver:
# mcc compiles `-x musa` sources and links the device image.  CMake has no MUSA language (its CUDA
# language drives nvcc, HIP drives hipcc), so the CUDA-shaped .cu files are compiled as CXX with
# `-x musa`, the way llama.cpp's ggml-musa backend does it.  mcc is the CXX compiler as well, so the
# device code, the host code and the link step share one clang and one libstdc++.
#
# Pass -DCMAKE_CXX_COMPILER=/usr/local/musa/bin/mcc when configuring (see docs/MUSA_MAYA.md).

if(NOT DEFINED MUSA_HOME OR MUSA_HOME STREQUAL "")
  if(DEFINED ENV{MUSA_HOME})
    set(MUSA_HOME "$ENV{MUSA_HOME}")
  else()
    set(MUSA_HOME "/usr/local/musa")
  endif()
endif()
if(NOT EXISTS "${MUSA_HOME}/include/musa_runtime.h")
  message(FATAL_ERROR "Maya MUSA: no MUSA toolkit at ${MUSA_HOME} (set -DMUSA_HOME=... or $MUSA_HOME)")
endif()

# MTT S4000 reports mp_22 (compute capability 2.2, __MUSA_ARCH__ 220); mp_21 is the S80/S3000 family.
if(NOT DEFINED CMAKE_MUSA_ARCHITECTURES OR CMAKE_MUSA_ARCHITECTURES STREQUAL "")
  set(CMAKE_MUSA_ARCHITECTURES mp_22 CACHE STRING "Maya MUSA target architecture(s), e.g. mp_22")
endif()
# A -DCMAKE_MUSA_ARCHITECTURES typed by hand may use spaces; normalise both spellings to a CMake list.
string(REPLACE " " ";" _strata_musa_norm "${CMAKE_MUSA_ARCHITECTURES}")
list(FILTER _strata_musa_norm EXCLUDE REGEX "^$")
list(REMOVE_DUPLICATES _strata_musa_norm)
if(NOT _strata_musa_norm)
  message(FATAL_ERROR "Maya MUSA: CMAKE_MUSA_ARCHITECTURES is empty")
endif()
set(STRATA_MUSA_ARCH_FLAGS "")
foreach(_arch IN LISTS _strata_musa_norm)
  list(APPEND STRATA_MUSA_ARCH_FLAGS "--cuda-gpu-arch=${_arch}")
endforeach()
# The compiled architectures reach the runtime device check (src/core/device.cu) as "mp_22".
string(REPLACE ";" "," STRATA_MUSA_ARCHS "${_strata_musa_norm}")
set(CMAKE_MUSA_ARCHITECTURES "${_strata_musa_norm}" CACHE STRING "Maya MUSA target architecture(s)" FORCE)

set(STRATA_MUSA_COMPAT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/strata/musa_compat")

# No -fgpu-rdc: with relocatable device code mcc defers a translation unit's device image to a device-link
# step that its driver only performs for a whole-program invocation, and its objects then leave __mt_fatbin
# undefined (the MUSA runtime does not export it).  Without the flag each object carries its own device image
# and the ordinary link is enough.  -fno-strict-aliasing: the kernels interpret weight blocks through
# uint4/byte pointers, exactly the aliasing CUDA compilers tolerate.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing")

set(_strata_gpu_runtime_target strata_musa_runtime)
set(_strata_gpu_blas_target strata_mublas)
set(_strata_gpu_include_directories "${MUSA_HOME}/include" "${STRATA_MUSA_COMPAT_INCLUDE_DIR}")

add_library(strata_musa_runtime INTERFACE)
target_include_directories(strata_musa_runtime BEFORE INTERFACE
  "${STRATA_MUSA_COMPAT_INCLUDE_DIR}" "${MUSA_HOME}/include")
# STRATA_USE_MUSA: the sources' non-NVIDIA branches.  STRATA_HAS_WMMA=0: MUSA has no rocWMMA/wmma
# equivalent the kernels could take, so the portable kernels are the ones that must be selected.
target_compile_definitions(strata_musa_runtime INTERFACE
  STRATA_USE_MUSA=1 STRATA_HAS_WMMA=0 "STRATA_MUSA_ARCHS=\"${STRATA_MUSA_ARCHS}\"")
target_link_directories(strata_musa_runtime INTERFACE "${MUSA_HOME}/lib")
# The driver library (libmusa.so.1, in the system path) carries the profiler entry points the engine calls.
target_link_libraries(strata_musa_runtime INTERFACE musart musa)
# Every translation unit sees the cudaXxx -> musaXxx map first, host-only .cpp files included (they call
# cudaMalloc/cudaGraphLaunch directly), the same way hip_backend.cmake force-includes hip_compat's.
target_compile_options(strata_musa_runtime INTERFACE
  "-include" "${STRATA_MUSA_COMPAT_INCLUDE_DIR}/cuda_runtime.h")

add_library(strata_mublas INTERFACE)
target_link_directories(strata_mublas INTERFACE "${MUSA_HOME}/lib")
target_link_libraries(strata_mublas INTERFACE mublas)

# CMake infers nothing from the .cu suffix for a compiler it does not know as a GPU compiler.
file(GLOB_RECURSE _strata_musa_sources CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/bench/*.cu")
if(STRATA_BUILD_TESTS)
  file(GLOB_RECURSE _strata_musa_test_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu")
  list(APPEND _strata_musa_sources ${_strata_musa_test_sources})
endif()
foreach(_src IN LISTS _strata_musa_sources)
  set_source_files_properties("${_src}" PROPERTIES
    LANGUAGE CXX
    COMPILE_OPTIONS "-x;musa;${STRATA_MUSA_ARCH_FLAGS};-O3")
endforeach()

# EXL3 (exllamav3) quants: mcc's device pass does not finish on src/kernels/cuda/exl3.cu (30+ minutes, no
# output), and the GLM-5.3-Flash GGUFs carry no EXL3 weights.  The stub satisfies the linker; nothing selects
# its functions at run time (their supported() answers false).  Pass -DSTRATA_MUSA_EXL3_KERNELS=ON to try the
# real file (and expect a very long compile).
option(STRATA_MUSA_EXL3_KERNELS "Build EXL3 kernels for MUSA (mcc codegen is extremely slow on them)" OFF)
if(NOT STRATA_MUSA_EXL3_KERNELS)
  set(STRATA_EXL3_IMPL "src/kernels/cuda/exl3_stub_musa.cpp")
endif()

message(STATUS "Strata: MUSA enabled, arch ${STRATA_MUSA_ARCHS}, toolkit ${MUSA_HOME}")
