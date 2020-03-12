ctypedef unsigned int uint

import numpy as np
cimport numpy as np

cdef extern from "vector_types.h":
    cdef struct float2:
        float x
        float y
    cdef struct float3:
        float x
        float y
        float z
    cdef struct uint2:
        uint x
        uint y
    cdef struct uint3:
        uint x
        uint y
        uint z

cdef extern from "src/raytrace.h":
    cdef void raytrace_c(
        float*        rpl,
        const float*  sources,
        const float*  dests,
        unsigned int  npts,
        float*        dens,
        float3        densStart,
        uint3         densSize,
        float3        densSpacing,
        float         stop_early
    )
    cdef void beamtrace_c(
        float*        rpl,
        const float   sad,
        const uint2   detSize,
        const float3  detCenter,
        const float2  detSpacing,
        const float   detAzi,
        const float   detZen,
        const float   detAng,
        const float*  dens,
        const float3  densStart,
        const uint3   densSize,
        const float3  densSpacing,
        const float   stop_early
    )

def raytrace(sources, dests, vol, vol_start, vol_spacing, stop_early=-1):
    """Measures the pathlength along each ray specified by (source, dest) coordinate pairs through a voxelized
    volume.

    Args:
        sources ([(x,y,z), ...]: list of 3d coordinates indicating ray startpoints
        dests   ([(x,y,z), ...]: list of 3d coordinates indicating ray endpoints
    """
    for v in [sources, dests, vol]:
        assert isinstance(v, np.ndarray)
    assert sources.ndim == 2
    assert dests.ndim == 2
    assert vol.ndim == 3

    cdef uint  npts_ = dests.shape[0]
    cdef float[::1] rpl_ = np.zeros((npts_), dtype=np.float32)
    cdef float[:,::1] sources_ = np.ascontiguousarray(sources.astype(np.float32))
    cdef float[:,::1] dests_ = np.ascontiguousarray(dests.astype(np.float32))
    cdef float[:,:,::1] dens_ = np.ascontiguousarray(vol.astype(np.float32))
    cdef float3 dens_start_
    cdef uint3  dens_size_
    cdef float3 dens_spacing_

    # init structs/vects
    dens_start_.x, dens_start_.y, dens_start_.z = vol_start
    dens_size_.x, dens_size_.y, dens_size_.z = vol.shape[::-1]
    dens_spacing_.x, dens_spacing_.y, dens_spacing_.z = vol_spacing

    raytrace_c(
        rpl=&rpl_[0],
        sources=&sources_[0,0],
        dests=&dests_[0,0],
        npts=npts_,
        dens=&dens_[0,0,0],
        densStart=dens_start_,
        densSize=dens_size_,
        densSpacing=dens_spacing_,
        stop_early=stop_early
    )
    return np.asarray(rpl_)

def beamtrace(sad, det_size, det_center, det_spacing, det_azi, det_zen, det_ang, vol, vol_start, vol_spacing, stop_early=-1):
    """Measures the pathlength average for 9 points at the corner/mid-edges/center of each diverging square beam
    originating from a common source coordinate and passing through the respective destination coordinates.
    Unlike the more general raytrace(), this assumes geometry of a diverging square beam, such that the common "source" is determined implicitly by the "sad" (source-to-isocenter-distance), the beam angles (azi, zen, ang) and the isocenter coordinates (det_center).

    This is similar to the simpler raytrace function but can additionally be used to detect collisions of each
    beam volume with non-zero voxels in the voxelized volume. This is particularly useful in Radiation
    Treatment Planning for detecting which sub-beams in an x-ray beam intesect with a tumor volume
    """

    for v in [vol]:
        assert isinstance(v, np.ndarray)
    assert vol.ndim == 3

    cdef float[:,::1] rpl_ = np.zeros(det_size, dtype=np.float32)
    cdef float[:,:,::1] dens_ = np.ascontiguousarray(vol.astype(np.float32))
    cdef uint2 det_size_
    cdef float3 det_center_
    cdef float2 det_spacing_
    cdef float3 dens_start_
    cdef uint3  dens_size_
    cdef float3 dens_spacing_

    # init structs/vects
    det_size_.x, det_size_.y = det_size
    det_center_.x, det_center_.y, det_center_.z = det_center
    det_spacing_.x, det_spacing_.y = det_spacing
    dens_start_.x, dens_start_.y, dens_start_.z = vol_start
    dens_size_.x, dens_size_.y, dens_size_.z = vol.shape[::-1]
    dens_spacing_.x, dens_spacing_.y, dens_spacing_.z = vol_spacing

    beamtrace_c(
        rpl=&rpl_[0,0],
        sad=sad,
        detSize=det_size_,
        detCenter=det_center_,
        detSpacing=det_spacing_,
        detAzi=det_azi,
        detZen=det_zen,
        detAng=det_ang,
        dens=&dens_[0,0,0],
        densStart=dens_start_,
        densSize=dens_size_,
        densSpacing=dens_spacing_,
        stop_early=stop_early
    )
    return np.asarray(rpl_)
