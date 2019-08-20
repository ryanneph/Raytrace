# Raytrace
General purpose python ray tracing library implemented in on GPU in CUDA as a Cython extension

This code is based on Siddon's algorithm, which has been quite extensively used in medical physics for CT projection and radiation treatment planning.

## Usage
This library allows one to calculate the _radiologic path-length (rpl)_ along a defined ray through a 3D voxel volume. This functionality can be used to obtain the forward radon transform of a volume for any given source-detector geometry, or even detect collisions with solid objects (represented as binary volumes) -- for example.
Below is an example of how to perform raytracing to get the rpl for a set of rays, each defined by a pair of source and destination coordinates, relative to the volume start coordinates and voxelsize:
```python
import numpy as np
from raytrace import raytrace

vol = np.ones((5,11,11))
vol[:, 5, 5] = 0
source = (6, 6, -10)
dests = np.array([
    ( 0,  0, 10),
    ( 6,  6, 10),
    (11, 11, 10),
])
rpl = raytrace(dests, source, vol, vol_start=(0,0,0), vol_spacing=(1,1,1))
print(rpl)

```
