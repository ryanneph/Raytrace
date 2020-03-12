import numpy as np
import matplotlib.pyplot as plt

from raytrace import raytrace

# setup volume with "hole"
vol = np.ones((5,11,11))
vol[2:, 2, 2] = 0
vol[:, 5, 5] = 0
vol[4:, 8, 8] = 0

# create 3D source-destination pairs
sources = np.stack([
    *np.meshgrid(
        np.arange(0.5, 11.5, 1),
        np.arange(0.5, 11.5, 1),
    ),
    -10.0*np.ones((11,11)),
]).reshape((3,-1)).T
dests = sources.copy()
dests[:, 2] = 10.0

# run raytrace
rpl = raytrace(dests, sources, vol, vol_start=(0,0,0), vol_spacing=(1,1,1))
rpl = rpl.reshape((11,11))

# show raytracing output
import matplotlib.pyplot as plt
plt.imshow(rpl)
plt.colorbar()
plt.show()
