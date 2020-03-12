#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include "helper_math.h" // cuda toolkit vector types

// CUDA device function qualifier for use in host/device compilable library code
#define CUDEV_FXN __host__ __device__


// Rotate vec around center using arbitrary axis
CUDEV_FXN float3 rotateAroundAxisRHS( const float3& vec, const float3& center, const float3& rotation_axis, const float& theta);
CUDEV_FXN float3 rotateAroundAxisAtOriginRHS(const float3& p, const float3& r, const float& t);

// rotation of point "vec" around "center" by angles "theta" and "phi
// composed of Rz(theta) -> Ry(phi)
/* ARGS
 *   theta: azimuth  - kernel weight fan angle (0->2pi) [radians]
 *   phi:   zenith   - kernel rotational symmetry angle (0->2pi) [radians]
 */
CUDEV_FXN float3 rotateBeamRHS( const float3& vec, const float3& center, const float& theta, const float& phi, const float& coll );
CUDEV_FXN float3 rotateBeamAtOriginRHS( const float3& vec, const float& theta, const float& phi, const float& coll );
// inversion: composed of Ry(-phi) -> Rz(-theta)
CUDEV_FXN float3 inverseRotateBeamRHS( const float3& vec, const float3& center, const float& theta, const float& phi, const float& coll );
CUDEV_FXN float3 inverseRotateBeamAtOriginRHS( const float3& vec, const float& theta, const float& phi, const float& coll );


// correct for beamlet divergence rotations
CUDEV_FXN float3 rotateBeamletRHS( const float3& vec, const float3& center, const float& theta, const float& phi );
CUDEV_FXN float3 rotateBeamletAtOriginRHS( const float3& vec, const float& theta, const float& phi );
CUDEV_FXN float3 inverseRotateBeamletRHS( const float3& vec, const float3& center, const float& theta, const float& phi );
CUDEV_FXN float3 inverseRotateBeamletAtOriginRHS( const float3& vec, const float& theta, const float& phi );

// Defines rotations to re-orient ray-traced volume in-line with the convolutional ray
CUDEV_FXN float3 rotateKernelRHS( const float3& vec, const float3& center, const float& theta, const float& phi );
CUDEV_FXN float3 rotateKernelAtOriginRHS( const float3& vec, const float& theta, const float& phi );
CUDEV_FXN float3 inverseRotateKernelRHS( const float3& vec, const float3& center, const float& theta, const float& phi );
CUDEV_FXN float3 inverseRotateKernelAtOriginRHS( const float3& vec, const float& theta, const float& phi );

#include "geometry.cu"

#endif // __GEOMETRY_H__
