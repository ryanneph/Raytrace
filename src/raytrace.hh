#ifndef __RAYTRACE_HH__
#define __RAYTRACE_HH__

#include <iostream>
#include "vector_types.h"

void raytrace2d_c(
    float* rpl,
    float* dests,
    float3 source,
    uint   npts,
    float* dens,
    float3 densStart,
    uint3  densSize,
    float3 densSpacing,
    std::ostream& cout,
    float  stop_early=-1.f,
    uint   ssfactor=3
);

#endif //__RAYTRACE_HH__

