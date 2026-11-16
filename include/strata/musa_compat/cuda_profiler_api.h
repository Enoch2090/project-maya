#pragma once
// MUSA's profiler entry points spell themselves with the driver's mu prefix.
#include <musaProfiler.h>
#define cudaProfilerStart muProfilerStart
#define cudaProfilerStop muProfilerStop
