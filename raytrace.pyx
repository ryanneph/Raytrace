cdef extern from "<iostream>" namespace "std":
    cdef cppclass ostream
    ostream cout
ctypedef unsigned int uint

import numpy as np
cimport numpy as np

cdef extern from "vector_types.h":
    cdef struct float3:
        float x
        float y
        float z
    cdef struct uint3:
        uint x
        uint y
        uint z


cdef extern from "src/raytrace.hh":
    cdef void raytrace_c(
        float*       rpl,
        const float* dests,
        float3       source,
        unsigned int npts,
        float*       dens,
        float3       densStart,
        uint3        densSize,
        float3       densSpacing,
        ostream      cout,
        float        stop_early,
        uint         ssfactor
    )

def raytrace(dests, source, vol, vol_start, vol_spacing, stop_early=-1, ssfactor=1):
    """Wrapper around raytracing C-function that handle data conversion to/from C-types
    Args:
        dests ([(x,y,z), ...]: list of 3d coordinates indicating ray endpoints
        source((x,y,z)): 3d coordinate of shared starting point
    """

    for v in [dests, vol]:
        assert isinstance(v, np.ndarray)
    assert dests.ndim == 2
    assert vol.ndim == 3

    cdef uint   npts_ = dests.shape[0]
    cdef float[::1] rpl_ = np.zeros((npts_), dtype=np.float32)
    cdef float[:,::1] dests_ = dests.astype(np.float32)
    cdef float[:,:,::1] dens_ = vol.astype(np.float32)
    cdef float3 src_
    cdef float3 dens_start_
    cdef uint3  dens_size_
    cdef float3 dens_spacing_

    # init structs/vects
    src_.x, src_.y, src_.z = source
    dens_start_.x, dens_start_.y, dens_start_.z = vol_start
    dens_size_.x, dens_size_.y, dens_size_.z = vol.shape[::-1]
    dens_spacing_.x, dens_spacing_.y, dens_spacing_.z = vol_spacing

    raytrace_c(&rpl_[0], &dests_[0, 0], src_, npts_, &dens_[0, 0, 0], dens_start_, dens_size_, dens_spacing_, cout, stop_early, ssfactor)
    return np.asarray(rpl_)
