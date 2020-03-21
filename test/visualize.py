import numpy as np
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

def spaced_detector(arr, pixelspacing, pixelsize, cmap='viridis'):
    dims = arr.shape[::-1]
    size = (np.array(dims)-1)*pixelspacing
    norm = Normalize(vmax=np.amax(arr), vmin=np.amin(arr))
    cmap = plt.get_cmap(cmap)

    def draw_rectangle(ax, ix, iy, val):
        x = ix*pixelspacing[0]
        y = iy*pixelspacing[1]
        ax.add_patch(
            plt.Rectangle(
                xy=(x-0.5*pixelsize[0],
                    y-0.5*pixelsize[1]),
                width=pixelsize[0], height=pixelsize[1],
                facecolor=cmap(norm(val)),
                edgecolor='black',
            )
        )
        ax.text(
            x, y, '{:0.1f}'.format(val),
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=8,
        )

    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, width_ratios=(1,0.05),
                           left=0.05, bottom=0.05, top=0.95, right=0.95,
                           wspace=0)
    ax = fig.add_subplot(gs[0,0])
    ax_cbar = fig.add_subplot(gs[0,1])
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.set_xlim((-pixelsize[0]*0.5, size[0]+pixelsize[0]*0.5))
    ax.set_ylim((-pixelsize[1]*0.5, size[1]+pixelsize[1]*0.5))
    for yy in range(dims[1]):
        for xx in range(dims[0]):
            draw_rectangle(ax, xx, yy, arr[yy, xx])
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax_cbar,
    )
    return fig
