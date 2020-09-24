import sys, os
from os.path import join as pjoin
#  sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.path.pardir))

curdir = os.path.dirname(os.path.abspath(__file__))

import numpy as np
from raytrace import raytrace, beamtrace, enableDebugOutput


vol = np.load(pjoin(curdir, 'test_data', 'PTV.npy'))
print('volume shape: ', vol.shape)

sad = 1000.0
det_dims = (80, 80)
det_spacing = (5.0, 5.0)
det_pixelsize = (5.0, 5.0)
spacing = np.array((2.5,2.5,2.5))
isocenter = np.divide(vol.shape[::-1],2.0)*spacing

common_source = isocenter.copy()
common_source[1] -= sad
sources = np.ones((np.product(det_dims), 3))*common_source
xx, zz = np.meshgrid(
    np.arange(-det_dims[0]*det_spacing[0]//2, det_dims[0]*det_spacing[0]//2, det_spacing[0]) + isocenter[0] + det_spacing[0]/2,
    np.arange(-det_dims[1]*det_spacing[1]//2, det_dims[1]*det_spacing[1]//2, det_spacing[1]) + isocenter[2] + det_spacing[1]/2,
)
yy = np.ones_like(xx) * isocenter[1]
dests = np.stack([xx, yy, zz]).reshape((3, -1)).T


# raytrace
raytrace_args = {
    'sources':     sources,
    'dests':       dests,
    'vol':         vol,
    'vol_start':   (0,0,0),
    'vol_spacing': spacing,
    'stop_early':  -1,
}
#  enableDebugOutput()
rpl_ray = raytrace(**raytrace_args).reshape(det_dims)


# beamtrace
beamtrace_args = {
    'sad':           sad,
    'det_dims':      det_dims,
    'det_center':    isocenter,
    'det_spacing':   det_spacing,
    'det_pixelsize': det_pixelsize,
    'det_azi':       0,
    'det_zen':       0,
    'det_ang':       0,
    'vol':           vol,
    'vol_start':     (0,0,0),
    'vol_spacing':   spacing,
    'stop_early':    -1,
}
rpl_beam =  beamtrace(**beamtrace_args).reshape(det_dims)


# reporting / comparison
import matplotlib.pyplot as plt
fig = plt.figure()
axes = fig.subplots(2,3)

ax = axes[0,0]
ax.imshow(rpl_ray, origin='lower', interpolation='none')
ax.set_title('RPL (ray)')

ax = axes[1,0]
ax.imshow(rpl_beam, origin='lower', interpolation='none')
ax.set_title('RPL (beam)')

ax = axes[0,1]
ax.imshow(np.where(rpl_ray>0, 1, 0), origin='lower', interpolation='none')
ax.set_title('MIP (ray)')

ax = axes[1,1]
ax.imshow(np.where(rpl_beam>0, 1, 0), origin='lower', interpolation='none')
ax.set_title('MIP (beam)')

ax = axes[1,2]
ax.imshow(np.where(rpl_beam>0, 1, 0)-np.where(rpl_ray>0, 1, 0), origin='lower', interpolation='none')
ax.set_title('MIP(beam)-MIP(ray)')

plt.show()
