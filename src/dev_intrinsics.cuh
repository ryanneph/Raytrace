#ifndef __CUDA_UTILITES_DEV_INTRINSICS_H__
#define __CUDA_UTILITES_DEV_INTRINSICS_H__

#include "helper_math.h"

// use device intrinsics for device execution (faster but less accurate hardware solutions)
// TODO: -x given to intrinsic cos/sin gives NaN (investigate if true)
#if defined(__CUDA_ARCH__)
    // use device intrinsics for device execution (faster but less accurate hardware solutions)
    #define fast_cosf(x)                __cosf(x)
    #define fast_sinf(x)                __sinf(x)
    #define fast_sincosf(x, sptr, cptr) __sincosf(x, sptr, cptr)
    #define fast_powf(x, n)             __powf(x, n)
    #define fast_sq(x)                  __fmul_rn(x,x)
#else
    // fallback to helper_math.h definitions (which auto-fallback to host functions for gcc compiled code)
    #define fast_cosf(x)                cosf(x)
    #define fast_sinf(x)                sinf(x)
    #define fast_sincosf(x, sptr, cptr) sincosf(x, sptr, cptr)
    #define fast_powf(x, n)             powf(x, n)
    #define fast_sq(x)                  x*x
#endif


// extensions for vector_types
inline __device__ float3 __fmaf_rn(const float& a, const float3& b, const float3& c) {
    return float3{
        __fmaf_rn(a, b.x, c.x),
        __fmaf_rn(a, b.y, c.y),
        __fmaf_rn(a, b.z, c.z)
        };
}
inline __device__ float3 __fmaf_rn(const float3& a, const float& b, const float3& c) {
    return float3{
        __fmaf_rn(a.x, b, c.x),
        __fmaf_rn(a.y, b, c.y),
        __fmaf_rn(a.z, b, c.z)
        };
}
inline __device__ float3 __fmaf_rn(const float3& a, const float3& b, const float& c) {
    return float3{
        __fmaf_rn(a.x, b.x, c),
        __fmaf_rn(a.y, b.y, c),
        __fmaf_rn(a.z, b.z, c)
        };
}
inline __device__ float3 __fmaf_rn(const float3& a, const float3& b, const float3& c) {
    return float3{
        __fmaf_rn(a.x, b.x, c.x),
        __fmaf_rn(a.y, b.y, c.y),
        __fmaf_rn(a.z, b.z, c.z)
        };
}
inline __device__ float3 fdividef(const float3& a, const float3& b) {
    return float3{
        fdividef(a.x, b.x),
        fdividef(a.y, b.y),
        fdividef(a.z, b.z)
        };
}
inline __device__ float3 __fadd_rn(const float3& a, const float3& b) {
    return float3{
        __fadd_rn(a.x, b.x),
        __fadd_rn(a.y, b.y),
        __fadd_rn(a.z, b.z)
    };
}
inline __device__ float3 __fsub_rn(const float3& a, const float3& b) {
    return float3{
        __fsub_rn(a.x, b.x),
        __fsub_rn(a.y, b.y),
        __fsub_rn(a.z, b.z)
    };
}
inline __device__ float2 __fmul_rn(const float2& a, const float2& b) {
    return float2{
        __fmul_rn(a.x, b.x),
        __fmul_rn(a.y, b.y)
    };
}
inline __device__ float3 __fmul_rn(const float3& a, const float3& b) {
    return float3{
        __fmul_rn(a.x, b.x),
        __fmul_rn(a.y, b.y),
        __fmul_rn(a.z, b.z)
    };
}
inline __device__ float3 __fmul_rn(const float3& a, const float& b) {
    return float3{
        __fmul_rn(a.x, b),
        __fmul_rn(a.y, b),
        __fmul_rn(a.z, b)
    };
}
inline __device__ float3 __fdiv_rn(const float3& a, const float3& b) {
    return float3{
        __fdiv_rn(a.x, b.x),
        __fdiv_rn(a.y, b.y),
        __fdiv_rn(a.z, b.z)
    };
}
inline __device__ float3 __frcp_rn(const float3& a) {
    return float3{
        __frcp_rn(a.x),
        __frcp_rn(a.y),
        __frcp_rn(a.z)
    };
}


#endif //__CUDA_UTILITES_DEV_INTRINSICS_H__
