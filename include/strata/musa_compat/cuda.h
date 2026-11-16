#pragma once
// The CUDA driver API's header. The sources only take its virtual-memory types here; the driver
// entry points themselves are looked up at run time (expert_cache.cpp), and that path is disabled
// for MUSA the way it is for HIP.
#include <musa.h>
