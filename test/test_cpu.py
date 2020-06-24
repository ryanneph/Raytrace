import sys, os
from os.path import join as pjoin
#  sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.path.pardir))
import logging
import math

curdir = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import raytrace
from raytrace import siddonraytracer, NonIntersectingRayError, spottrace
import matplotlib as mpl
import matplotlib.pyplot as plt

def interactive_viewer(vols, titles=[]):
    if titles:
        assert len(titles)==len(vols)
    else:
        titles = [None]*len(vols)
    for vol in vols[1:]:
        assert isinstance(vol, np.ndarray)
        assert vol.shape == vols[0].shape

    fig, axes = plt.subplots(1,len(vols))
    ims = []
    for ax, vol, title in zip(axes, vols, titles):
        ims.append(ax.imshow(vol[0]))
        ax.set_title(title)

    class PassByRef():
        def __init__(self, value):
            self.value = value
    zzref = PassByRef(0)

    def press(event, zzref):
        zz = zzref.value
        if event.key in ['q', 'esc']:
            plt.close(fig)
        elif event.key == 'left':
            if zz<= 0: return
            zz -= 1
        elif event.key == 'right':
            if zz>=(vols[0].shape[0]-1): return
            zz += 1
        for im, vol in zip(ims, vols):
            axes[0].set_ylabel('z-slice {} of {}'.format(zz+1, vols[0].shape[0]))
            im.set_data(vol[zz])
            fig.canvas.draw()
        zzref.value = zz
    fig.canvas.mpl_connect('key_press_event', lambda event: press(event, zzref))
    plt.show()

class SiddonRayTraceTestCase():
    def __init__(self, vols, source, target, start, spacing, alphas, lengths, densities, raylength, indices):
        self.vols = vols
        self.source = source
        self.target = target
        self.start = start
        self.spacing = spacing

        self.alphas = alphas
        self.lengths = lengths
        self.densities = densities
        self.raylength = raylength
        self.indices = indices

    def runtest(self):
        try:
            alphas, lengths, densities, raylength, indices = siddonraytracer(
                vols=self.vols,
                source=self.source,
                target=self.target,
                start=self.start,
                spacing=self.spacing,
            )
        except NonIntersectingRayError:
            print('non-intersecting ray')
            return
        #  np.testing.assert_array_equal(indices, self.indices)
        print('alphas:', alphas)
        print('lengths:', lengths)
        print('densities:', densities)
        print('raylength:', raylength)
        print('indices:', indices)

if __name__ == "__main__":
    # define test geometry
    vol = np.zeros((8,8,8))
    for yy in range(vol.shape[1]):
        vol[:, yy, : ] = yy+1
    #  for zz in range(vol.shape[0]):
    #      vol[zz, :, :] *= zz+1

    mask = np.zeros((8,8,8))
    mask[:, 2:6, 2] = 1.0
    mask[:, 2, 2:6] = 1.0
    mask[:, 5, 2:5] = 1.0

    #  interactive_viewer([vol, mask], titles=('vol', 'mask'))

    # setup module logging
    #  raytrace.enableDebugOutput()

    depths = spottrace(
        sad=100000,
        det_dims=(10,10),
        det_center=(3,0,3),
        det_spacing=(1,1),
        det_pixelsize=(1,1),
        det_azi=0,
        det_zen=0,
        det_ang=0,
        vols=[vol, mask],
        vol_start=(0,0,0),
        vol_spacing=(1,1,1)
    )
    #  sources = [(3,-10,3),
    #             (3,10,3),
    #             (2,-10,2),
    #             ]
    #  dests = [(3,10,3),
    #           (3,-10,3),
    #           (2,10,2),
    #           ]
    #  testcases = [
    #      SiddonRayTraceTestCase(
    #          vols=None,
    #          source=source,
    #          target=target,
    #          start=None,
    #          spacing=None,
    #          alphas=None,
    #          lengths=[1]*8,
    #          densities=[None, None],
    #          raylength=None,
    #          indices=None
    #      )
    #      for source, target in zip(sources, dests)
    #  ]

    #  for testcase in testcases:
    #      testcase.vols = [vol, mask]
    #      testcase.start = (0, 0, 0)
    #      testcase.spacing = (1,1,1)
    #      testcase.runtest()


