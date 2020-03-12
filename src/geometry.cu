#include "geometry.cuh"

#include <cstdio>
#include "dev_intrinsics.cuh"


/* COORDINATE SYSTEM DEFINITION
 * Uses VarianIEC motion scale:
 *   azimuth - Gantry/Z-axis      rotation - (0 defined as entering patient anterior, gantry above)
 *   zenith  - Couch/non-coplanar rotation - (0 defined as coplanar beam, couch perp. to linac body)
 *
 * EXAMPLES FOR PATIENT ORIENTATION HFS:
 *   azimuth = 0           : entering from patient anterior
 *   azimuth = 90          : entering from patient left
 *   azimuth = 180         : entering from patient posterior
 *   azimuth = 270         : entering from patient right
 *
 *   0    < zenith  < 180  : CW couch-kick (top view)
 *   -180 < zenith  <   0  : CCW couch-kick (top view)
 **/


// Rotate vec around center using arbitrary axis
CUDEV_FXN float3 rotateAroundAxisRHS(const float3& p, const float3& q, const float3& r, const float& t) {
    // ASSUMES r IS NORMALIZED ALREADY
    // p - vector to rotate
    // q - center point
    // r - rotation axis
    // t - rotation angle
    // non-vectorized version
    //    x,y,z = p.(x,y,z)
    //    a,b,c = q.(x,y,z)
    //    u,v,w = r.(x,y,z)
    //
    //    /* (a*(fast_sq(v)+fast_sq(w)) - u*(b*v + c*w - u*x - v*y - w*z))*(1-fast_cosf(t)) + x*fast_cosf(t) + (-c*v + b*w - w*y + v*z)*fast_sinf(t), */
    //    /* (b*(fast_sq(u)+fast_sq(w)) - v*(a*u + c*w - u*x - v*y - w*z))*(1-fast_cosf(t)) + y*fast_cosf(t) + ( c*u - a*w + w*x - u*z)*fast_sinf(t), */
    //    /* (c*(fast_sq(u)+fast_sq(v)) - w*(a*u + b*v - u*x - v*y - w*z))*(1-fast_cosf(t)) + z*fast_cosf(t) + (-b*u + a*v - v*x + u*y)*fast_sinf(t) */

    float sptr, cptr;
    fast_sincosf(t, &sptr, &cptr);
    return make_float3(
            (q.x*(fast_sq(r.y)+fast_sq(r.z)) - r.x*(q.y*r.y + q.z*r.z - r.x*p.x - r.y*p.y - r.z*p.z))*(1-cptr) + p.x*cptr + (-q.z*r.y + q.y*r.z - r.z*p.y + r.y*p.z)*sptr,
            (q.y*(fast_sq(r.x)+fast_sq(r.z)) - r.y*(q.x*r.x + q.z*r.z - r.x*p.x - r.y*p.y - r.z*p.z))*(1-cptr) + p.y*cptr + ( q.z*r.x - q.x*r.z + r.z*p.x - r.x*p.z)*sptr,
            (q.z*(fast_sq(r.x)+fast_sq(r.y)) - r.z*(q.x*r.x + q.y*r.y - r.x*p.x - r.y*p.y - r.z*p.z))*(1-cptr) + p.z*cptr + (-q.y*r.x + q.x*r.y - r.y*p.x + r.x*p.y)*sptr
            );
}
CUDEV_FXN float3 rotateAroundAxisAtOriginRHS(const float3& p, const float3& r, const float& t) {
    // ASSUMES r IS NORMALIZED ALREADY and center is (0, 0, 0)
    // p - vector to rotate
    // r - rotation axis
    // t - rotation angle
    float sptr, cptr;
    fast_sincosf(t, &sptr, &cptr);
    return make_float3(
            (-r.x*(-r.x*p.x - r.y*p.y - r.z*p.z))*(1-cptr) + p.x*cptr + (-r.z*p.y + r.y*p.z)*sptr,
            (-r.y*(-r.x*p.x - r.y*p.y - r.z*p.z))*(1-cptr) + p.y*cptr + (+r.z*p.x - r.x*p.z)*sptr,
            (-r.z*(-r.x*p.x - r.y*p.y - r.z*p.z))*(1-cptr) + p.z*cptr + (-r.y*p.x + r.x*p.y)*sptr
            );
}

// convert RCS coords to BEV coords
CUDEV_FXN float3 rotateBeamRHS( const float3& vec, const float3& center, const float& theta, const float& phi, const float& coll ) {
    // first rotate around y-axis by phi+coll then rotate point around z'-axis at center by -theta
    float sptr, cptr;
    fast_sincosf(-phi, &sptr, &cptr);
    float3 rotation_axis = make_float3(sptr, 0.0f, cptr);                          // couch rotation
    float3 tmp = rotateAroundAxisRHS(vec, center, rotation_axis, -theta);          // gantry rotation
    return rotateAroundAxisRHS(tmp, center, make_float3(0.f, 1.f, 0.f), phi+coll); // coll rotation + correction
}
CUDEV_FXN float3 rotateBeamAtOriginRHS( const float3& vec, const float& theta, const float& phi, const float& coll ) {
    float sptr, cptr;
    fast_sincosf(-phi, &sptr, &cptr);
    float3 rotation_axis = make_float3(sptr, 0.0f, cptr);                          // couch rotation
    float3 tmp = rotateAroundAxisAtOriginRHS(vec, rotation_axis, -theta);          // gantry rotation
    return rotateAroundAxisAtOriginRHS(tmp, make_float3(0.f, 1.f, 0.f), phi+coll); // coll rotation + correction
}
// convert BEV coords to RCS coords
CUDEV_FXN float3 inverseRotateBeamRHS( const float3& vec, const float3& center, const float& theta, const float& phi, const float& coll ) {
    // invert what was done in forward rotation
    float3 tmp = rotateAroundAxisRHS(vec, center, make_float3(0.f, 1.f, 0.f), -(phi+coll)); // coll rotation + correction
    float sptr, cptr;
    fast_sincosf(-phi, &sptr, &cptr);
    float3 rotation_axis = make_float3(sptr, 0.0f, cptr);                                   // couch rotation
    return rotateAroundAxisRHS(tmp, center, rotation_axis, theta);                          // gantry rotation
}
CUDEV_FXN float3 inverseRotateBeamAtOriginRHS( const float3& vec, const float& theta, const float& phi, const float& coll ) {
    // invert what was done in forward rotation
    float3 tmp = rotateAroundAxisAtOriginRHS(vec, make_float3(0.f, 1.f, 0.f), -(phi+coll)); // coll rotation + correction
    float sptr, cptr;
    fast_sincosf(-phi, &sptr, &cptr);
    float3 rotation_axis = make_float3(sptr, 0.0f, cptr);                                   // couch rotation
    return rotateAroundAxisAtOriginRHS(tmp, rotation_axis, theta);                          // gantry rotation
}


// convert bev (beamlet) coords to BEV (beam) coords
CUDEV_FXN float3 rotateBeamletRHS( const float3& vec, const float3& center, const float& theta, const float& phi) {
    return inverseRotateBeamletRHS(vec, center, -theta, phi);
}
CUDEV_FXN float3 rotateBeamletAtOriginRHS( const float3& vec, const float& theta, const float& phi) {
    return inverseRotateBeamletAtOriginRHS(vec, -theta, phi);
}
// convert BEV (beam) coords to bev (beamlet) coords
CUDEV_FXN float3 inverseRotateBeamletRHS( const float3& vec, const float3& center, const float& theta, const float& phi) {
    // first rotate around y-axis by -phi then rotate point around z'-axis at center by theta
    float sptr, cptr;
    fast_sincosf(-phi, &sptr, &cptr);
    float3 rotation_axis = make_float3(sptr, 0.0f, cptr); // couch rotation
    return rotateAroundAxisRHS(vec, center, rotation_axis, -theta);     // gantry rotation
}
CUDEV_FXN float3 inverseRotateBeamletAtOriginRHS( const float3& vec, const float& theta, const float& phi) {
    float sptr, cptr;
    fast_sincosf(-phi, &sptr, &cptr);
    float3 rotation_axis = make_float3(sptr, 0.0f, cptr); // couch rotation
    return rotateAroundAxisAtOriginRHS(vec, rotation_axis, -theta);     // gantry rotation
}


// convert BEV coords to REV coords
CUDEV_FXN float3 rotateKernelRHS( const float3& vec, const float3& center, const float& theta, const float& phi ) {
    // similar to beam rotation but by first rotating around y-axis by phi then x'-axis at center by theta
    float sptr, cptr;
    fast_sincosf(phi, &sptr, &cptr);
    float3 untilt = rotateAroundAxisRHS(vec, center, make_float3(cptr, 0.f, -sptr), -theta);
    return rotateAroundAxisRHS(untilt, center, make_float3(0.f, 1.f, 0.f), -phi);
}
CUDEV_FXN float3 rotateKernelAtOriginRHS( const float3& vec, const float& theta, const float& phi ) {
    float sptr, cptr;
    fast_sincosf(phi, &sptr, &cptr);
    float3 untilt = rotateAroundAxisAtOriginRHS(vec, make_float3(cptr, 0.f, -sptr), -theta);
    return rotateAroundAxisAtOriginRHS(untilt, make_float3(0.f, 1.f, 0.f), -phi);
}
// convert REV coords to BEV coords
CUDEV_FXN float3 inverseRotateKernelRHS( const float3& vec, const float3& center, const float& theta, const float& phi ) {
    // undo what was done by rotateKernelRHS
    float3 roll = rotateAroundAxisRHS(vec, center, make_float3(0.f, 1.f, 0.f), phi);               // kernel roll
    float sptr, cptr;
    fast_sincosf(phi, &sptr, &cptr);
    return rotateAroundAxisRHS(roll, center, make_float3(cptr, 0.f, -sptr), theta);  // kernel tilt
}
CUDEV_FXN float3 inverseRotateKernelAtOriginRHS( const float3& vec, const float& theta, const float& phi ) {
    float3 roll = rotateAroundAxisAtOriginRHS(vec, make_float3(0.f, 1.f, 0.f), phi);               // kernel roll
    float sptr, cptr;
    fast_sincosf(phi, &sptr, &cptr);
    return rotateAroundAxisAtOriginRHS(roll, make_float3(cptr, 0.f, -sptr), theta);  // kernel tilt
}
