import numpy as np
import matplotlib.pyplot as plt

from raytrace import beamtrace
import visualize

# setup volume with "hole"
#   convention for beamtrace is along +y axis when angles=(0,0,0)
vol = np.ones((11,5,11))
vol[2, 2:, 2] = 0
vol[5, :, 5] = 0
vol[8, 4:, 8] = 0

# define constrained "fluence plane" geometry
sad = 1000.0
plane_dims = (11, 11)
plane_center = (5.5, 0, 5.5)
plane_spacing = (0.5, 0.5)
plane_pixelsize = (0.5, 0.5)
angles = (0, 0, 0) # adjustable "beam angles"

# run raytrace
rpl = beamtrace(sad, plane_dims, plane_center, plane_spacing, plane_pixelsize, *angles, vol, vol_start=(0,0,0), vol_spacing=(1,1,1))
rpl = rpl.reshape((11,11))

# show raytracing output
fig = visualize.spaced_detector(rpl, plane_spacing, plane_pixelsize)
plt.show()
