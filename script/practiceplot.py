import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr

import geocat.viz.util as gvutil

# define plot
def add_axes(fig, grid_space):
    ax = fig.add_subplot(grid_space, projection=ccrs.PlateCarree())

    # Add land to the subplot
    ax.add_feature(cfeature.LAND,
                   facecolor="none",
                   edgecolor='black',
                   linewidths=0.5,
                   zorder=2)

    # Usa geocat.viz.util convenience function to set axes parameters
    gvutil.set_axes_limits_and_ticks(ax,
                                     ylim=(0,25),
                                     xlim=(77, 100),
                                     xticks=np.arange(77,101 , 5),
                                     yticks=np.arange(0, 26, 5))

    # Use geocat.viz.util convenience function to add minor and major tick lines
    gvutil.add_major_minor_ticks(ax, labelsize=12)

    # Use geocat.viz.util convenience function to make plots look like NCL
    # plots by using latitude, longitude tick labels
    gvutil.add_lat_lon_ticklabels(ax)

    gvutil.set_titles_and_labels(ax,
                                 ylabel='Latitude',
                                 xlabel='Longitude',
                                 labelfontsize=16)

    # Remove the degree symbol from tick labels
    ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol=''))
    ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol=''))

    return ax


# Open a netCDF data file using xarray default engine and load the data into xarrays
ds = xr.open_dataset("data/rectilinear_grid_2D.nc")

# Read variables
tsurf = ds.tsurf  # surface temperature in K
date = tsurf.time

# Generate figure (set its size (width, height) in inches)
fig = plt.figure(figsize=(13, 9.4), constrained_layout=True)

# Create gridspec to hold six subplots
grid = fig.add_gridspec(ncols=3, nrows=2)

# Add the axes
ax1 = add_axes(fig, grid[0, 0])
ax2 = add_axes(fig, grid[0, 1])
ax3 = add_axes(fig, grid[0, 2])
ax4 = add_axes(fig, grid[1, 0])
ax5 = add_axes(fig, grid[1, 1])
ax6 = add_axes(fig, grid[1, 2])

# Set plot index list
plot_idxs = [0, 6, 18, 24, 30, 36]

# Set contour levels
levels = np.arange(220, 316, 1)
# Set colormap
cmap = plt.get_cmap('magma')

for i, axes in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
    dataset = tsurf[plot_idxs[i], :, :]
    # Contourf plot data
    contour = axes.contourf(dataset.lon,
                            dataset.lat,
                            dataset.data,
                            vmin=250,
                            vmax=310,
                            cmap=cmap,
                            levels=levels)


# Set colorbounds of norm
colorbounds = np.arange(249, 311, 1)
# Use cmap to create a norm and mappable for colorbar to be correctly plotted
norm = mcolors.BoundaryNorm(colorbounds, cmap.N)
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

# Add colorbar for all six plots
fig.colorbar(mappable,
                 ax=[ax1, ax2, ax3, ax4, ax5, ax6],
                 ticks=colorbounds[1:-1:3],
                 drawedges=True,
                 orientation='horizontal',
                 shrink=0.82,
                 pad=0.01,
                 aspect=35,
                 extendfrac='auto',
                 extendrect=True)
    # Add figure titles
fig.suptitle("rectilinear_grid_2D.nc", fontsize=22, fontweight='bold')
ax1.set_title("surface temperature", loc="left", fontsize=16, y=1.05)
ax2.set_title("degK", loc="right", fontsize=15, y=1.05)


plt.savefig('multiexample.png',dpi = 300)
    # Show plot
plt.show()