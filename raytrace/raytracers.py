import logging
import math
import warnings
from collections.abc import Iterable

import numpy as np

logger = logging.getLogger(__name__)

try:
    from .raytrace_ext import inverseRotateBeamAtOrigin
except ImportError:
    from .geometry import inverseRotateBeamAtOriginRHS as inverseRotateBeamAtOrigin

class NonIntersectingRayError(Exception):
    pass

def siddonraytracer(vols, source, target, start, spacing):
    """Siddon raytracer for calculating intersection positions and lengths for rays coincident with
    a subdivided cube (of voxels)

    Args:
        vols: asdf
        src: asdf
        dst: asdf
        start: asdf
        spacing: asdf
    """
    if isinstance(vols, np.ndarray):
        vols = [vols]
    size = np.array(vols[0].shape[::-1])
    Nplanes = size+1
    start = np.array(start)
    spacing = np.array(spacing)
    source = np.array(source)
    target = np.array(target)
    assert len(start) == 3
    assert len(spacing) == 3
    assert source.shape == target.shape

    # number of intersection planes
    Nx, Ny, Nz = Nplanes
    diff = target - source
    raylength = np.linalg.norm(diff)

    empty_result = ([], [], [list() for _ in range(len(vols))], raylength, [])

    # boundary plane positions (eq 3)
    first_plane = start - spacing/2 # indices are defined at voxel center
    last_plane  = first_plane + (size)*spacing

    # parametric boundary values (eq 4)
    with np.errstate(divide='ignore'):
        alpha_first = (first_plane-source)/diff
        alpha_last = (last_plane-source)/diff
        # resolve errors so alpha_min and alpha_max are still valid
        alpha_first[~np.isfinite(alpha_first)] = -float('inf')
        alpha_last[~np.isfinite(alpha_last)] = float('inf')

    # ray entrance/exit parameter values (eq 5)
    alpha_min = np.amax(np.maximum(0, np.minimum(alpha_first, alpha_last)))
    alpha_max = np.amin(np.minimum(1, np.maximum(alpha_first, alpha_last)))

    logger.debug('source:      {}'.format(source))
    logger.debug('target:      {}'.format(target))
    logger.debug('start:       {}'.format(start))
    logger.debug('diff:        {}'.format(diff))
    logger.debug('raylength:   {}'.format(raylength))
    logger.debug('first_plane: {}'.format(first_plane))
    logger.debug('last_plane:  {}'.format(last_plane))
    logger.debug('alpha_first: {}'.format(alpha_first))
    logger.debug('alpha_last:  {}'.format(alpha_last))
    logger.debug('alpha_min:   {}'.format(alpha_min))
    logger.debug('alpha_max:   {}'.format(alpha_max))

    if alpha_max <= alpha_min:
        raise NonIntersectingRayError()

    # calcualate intersection plane indices (eq 6)
    imin = np.zeros((3,))
    imax = np.zeros((3,))
    for ax in range(3):
        if diff[ax] >=0:
            imin[ax] = math.ceil( size[ax] - (last_plane[ax] - alpha_min*diff[ax] - source[ax])/spacing[ax] )
            imax[ax] = math.floor( (source[ax] + alpha_max*diff[ax] - first_plane[ax])/spacing[ax] )
        else:
            imin[ax] = math.ceil( size[ax] - (last_plane[ax] - alpha_max*diff[ax] - source[ax])/spacing[ax] )
            imax[ax] = math.floor( (source[ax] + alpha_min*diff[ax] - first_plane[ax])/spacing[ax] )

    logger.debug('imin: {}'.format(imin))
    logger.debug('imax: {}'.format(imax))

    # generate per-axis param lists (eq 7)
    alphas = set([alpha_min, alpha_max])
    for ax in range(3):
        if diff[ax] != 0:
            alphas.update( ((first_plane[ax] + np.arange(imin[ax], imax[ax]+1, 1)*spacing[ax]) - source[ax])/diff[ax] )
        else:
            pass
    alphas = list(alphas)
    alphas.sort()
    logger.debug('alphas: {}'.format(alphas))

    # calculate voxel intersection lengths (eq 10)
    lengths = raylength*np.diff(alphas)
    logger.debug('lengths: {}'.format(lengths))

    # calculate intersection midpoints (eq 13)
    alpha_arr = np.array(alphas)
    alpha_mids = (alpha_arr[:-1]+alpha_arr[1:])/2
    logger.debug('alpha_mids: {}'.format(alpha_mids))

    # calculate voxel indices (eq 12)
    indices = np.empty((len(alpha_mids), 3), dtype=int)
    for ax in range(3):
        indices[:, ax] = np.floor( (source[ax] + alpha_mids*diff[ax] - first_plane[ax])/spacing[ax] )
    logger.debug('indices: {}'.format(indices))

    # read densities corresponding to indices
    try:
        densities = []
        for ii, vol in enumerate(vols):
            d = vol[indices[:, 2], indices[:, 1], indices[:, 0]]
            densities.append(d)
            logger.debug('densities ({}): {}'.format(ii, d))
    except IndexError:
        raise NonIntersectingRayError()
    logger.debug('')

    return (alphas, lengths, densities, raylength, indices)

def spottrace(sad, det_dims, det_center, det_spacing, det_pixelsize, det_azi, det_zen, det_ang, vols, vol_start, vol_spacing):
    """Emulates the diverging square beam geometry of proton beams. Unlike beamtrace(), this function
    produces the minimum and maximum radiological path length for each closed segment of the target
    mask intersecting with each ray. From this information, a set of proton pencil beam energies can
    be derived for constructing a scanning-spot-style of proton beam delivery.

    Args:
        sad (float):                             distance from source to isocenter (equivalently: focal point to center of detector plane)
        det_dims (int_x, int_y):                 number of detector elements
        det_center (float_x, float_y, float_z):  coordinates of detector plane center
        det_spacing (float_x, float_y):          spacing between adjacent detector elements (not the same as element size)
        det_pixelsize (float_x, float_y):        element size
        det_azi:                                 angle (radians) of azimuth (linac gantry angle)
        det_zen:                                 angle (radians) of zenith (linac couch angle)
        det_ang:                                 angle (radians) of detector plane rotation around its normal vector (linac collimator rotation)
        density_vol:                             density volume containing voxels through which to raytrace
        mask_vol:                                binary volume containing mask definition
        vol_start (float_x, float_y, float_z):   coordinates of center of first voxel in vol
        vol_spacing (float_x, float_y, float_z): spacing between adjacent voxels in volume (assumes direct adjacency)

    returns [[]]
    """

    # calculate ray sources/targets
    Nx, Nz = det_dims
    bx, bz = np.meshgrid(np.arange(det_dims[0]), np.arange(det_dims[1]))
    det_size = [(det_dims[ii]-1)*det_spacing[ii] for ii in range(2)]
    det_center = np.array(det_center)

    source = np.array(inverseRotateBeamAtOrigin((0,-sad, 0), det_azi, det_zen, det_ang)) + det_center
    targets = np.zeros((Nz*Nx, 3))
    targets[:, 0] = -0.5*det_size[0] + np.ravel(bx)*det_spacing[0]
    targets[:, 1] = 0
    targets[:, 2] = -0.5*det_size[1] + np.ravel(bz)*det_spacing[1]
    for ii in range(len(targets)):
        targets[ii] = np.array(inverseRotateBeamAtOrigin(targets[ii], det_azi, det_zen, det_ang)) + det_center
        # extend ray through volume
        targets[ii] = source + 10*(targets[ii]-source)
    targets = targets.reshape((Nz, Nx, 3))

    depths = np.empty((Nz, Nx), dtype=object)
    for rr in range(det_dims[1]):
        for cc in range(det_dims[0]):
            try:
                (_, ray_lengths, ray_vals, _, _) = siddonraytracer(
                    vols=vols,
                    source=source,
                    target=targets[rr, cc],
                    start=vol_start,
                    spacing=vol_spacing
                )
            except NonIntersectingRayError:
                depths[rr, cc] = []
                continue
            ray_dens, ray_mask = ray_vals
            if not np.any(ray_mask>0):
                # no intersection
                depths[rr, cc] = []
                continue

            # find closed intersection segments
            edges = np.diff(np.concatenate([[0], ray_mask]))
            ray_depths = np.cumsum(ray_lengths*ray_dens)
            ray_entries = ray_depths[edges==1]
            ray_exits   = ray_depths[edges==-1]
            assert len(ray_entries) == len(ray_exits)
            depths[rr, cc] = list(zip(ray_entries, ray_exits))
    return depths

def beamtrace(sad, det_dims, det_center, det_spacing, det_pixelsize, det_azi, det_zen, det_ang, vol, vol_start, vol_spacing, stop_early=-1):
    source = np.add(inverseRotateBeamAtOrigin((0, -sad, 0), det_azi, det_zen, det_ang), det_center)

    rpl = np.zeros(det_dims[::-1])

    detsize = np.multiply(np.subtract(det_dims, 1), det_spacing)
    for bixel_idx_z in range(det_dims[1]):
        for bixel_idx_x in range(det_dims[0]):
            bixel_ctr_fcs = np.array((
                -0.5*detsize[0] + bixel_idx_x*det_spacing[0],
                0,
                -0.5*detsize[1] + bixel_idx_z*det_spacing[1],
            ))
            bixel_ctr = inverseRotateBeamAtOrigin(bixel_ctr_fcs, det_azi, det_zen, det_ang)
            bixel_ctr = np.add(bixel_ctr, det_center)

            shortdiff = np.subtract(bixel_ctr, source)
            sink = source + 10*shortdiff

            try:
                (_, ray_lengths, ray_vals, _, _) = siddonraytracer(vol, source, sink, vol_start, vol_spacing)
                rpl[bixel_idx_z, bixel_idx_x] = np.dot(ray_lengths, ray_vals[0])
            except NonIntersectingRayError:
                rpl[bixel_idx_z, bixel_idx_x] = 0

    return rpl

def raytrace(sources, dests, vol, vol_start, vol_spacing, stop_early=None):
    if not isinstance(sources, Iterable):
        sources = [sources]
    sources = np.array(sources)
    if not isinstance(dests, Iterable):
        dests = [dests]
    dests = np.array(dests)

    rpl = np.zeros((len(sources),))

    for idx in range(len(sources)):
        src = sources[idx]
        dest = dests[idx]
        try:
            (_, ray_lengths, ray_vals, _, _) = siddonraytracer(vol, src, dest, vol_start, vol_spacing)
            rpl[idx] = np.dot(ray_lengths, ray_vals[0])
        except NonIntersectingRayError:
            rpl[idx] = 0
    return rpl
