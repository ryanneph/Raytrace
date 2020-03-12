import numpy as np
import matplotlib.pyplot as plt

from raytrace import beamtrace

# setup volume with "hole"
#   convention for beamtrace is along +y axis when angles=(0,0,0)
vol = np.ones((11,5,11))
vol[2, 2:, 2] = 0
vol[5, :, 5] = 0
vol[8, 4:, 8] = 0

# define constrained "fluence plane" geometry
sad = 10.0
plane_size = (11, 11)
plane_center = (5.5, 0, 5.5)
plane_spacing = (1,1)
angles = (0, 0, 0) # adjustable "beam angles"

# run raytrace
rpl = beamtrace(sad, plane_size, plane_center, plane_spacing, *angles, vol, vol_start=(0,0,0), vol_spacing=(1,1,1))
rpl = rpl.reshape((11,11))

# show raytracing output
import matplotlib.pyplot as plt
plt.imshow(rpl)
plt.colorbar()
plt.show()
