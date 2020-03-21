#ifndef __RAYTRACE_HH__
#define __RAYTRACE_HH__

#include <iostream>
#include "vector_types.h"

void raytrace_c(
    float*        rpl,
    const float*  sources,
    const float*  dests,
    unsigned int  npts,
    float*        dens,
    float3        densStart,
    uint3         densSize,
    float3        densSpacing,
    float         stop_early=-1.f
    );

void beamtrace_c(
  float*        rpl,
  const float   sad,
  const uint2   detDims,
  const float3  detCenter,
  const float2  detSpacing,
  const float2  detPixelSize,
  const float   detAzi,
  const float   detZen,
  const float   detAng,
  const float*  dens,
  const float3  densStart,
  const uint3   densSize,
  const float3  densSpacing,
  const float   stop_early
);

#endif //__RAYTRACE_HH__

