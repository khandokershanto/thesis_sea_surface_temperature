# Import libraries
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import Grid
import numpy as np
import xarray as xr

import geocat.viz.util as gvutil
from geocat.viz import cmaps as gvcmaps

# define plot
def add_axes(fig, grid_space):
    ax = fig.add_subplot(grid_space, projection=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)

    # Add land to the subplot
    ax.add_feature(cfeature.LAND,
                   facecolor="lightgray",
                   edgecolor='black',
                   linewidths=0.5,
                   zorder=2)

    # Usa geocat.viz.util convenience function to set axes parameters
    gvutil.set_axes_limits_and_ticks(ax,
                                     ylim=(5,25),
                                     xlim=(80, 100),
                                     xticks=np.arange(80,101 , 5),
                                     yticks=np.arange(5, 26, 5))

    # Use geocat.viz.util convenience function to add minor and major tick lines
    gvutil.add_major_minor_ticks(ax, labelsize=12)

    # Use geocat.viz.util convenience function to make plots look like NCL
    # plots by using latitude, longitude tick labels
    gvutil.add_lat_lon_ticklabels(ax)


    # Remove the degree symbol from tick labels
    ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol=''))
    ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol=''))

    return ax
###################

# define month
def add_text(month):
    months = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct',
        'Nov', 'Dec'
    ]
    month = months[month]
    return month


# Open a netCDF data file using xarray default engine and load the data into xarrays
ds = xr.open_dataset("data/noaa_oisst_v2_merged_1982_2020.nc")

# Read variables
sst = ds.sst # sea surface temperature

# monthly mean climatology
sst_mclm = sst.groupby('time.month').mean(dim='time')

# Generate figure (set its size (width, height) in inches)
fig = plt.figure(figsize=(12, 8.2), constrained_layout=True)
# Create gridspec to hold six subplots
grid = fig.add_gridspec(ncols=4, nrows=3)
#grid = Grid(fig, rect=111, nrows_ncols=(3, 4),axes_pad=0.25, label_mode='L')



# Add the axes
ax1 = add_axes(fig, grid[0, 0])
ax2 = add_axes(fig, grid[0, 1])
ax3 = add_axes(fig, grid[0, 2])
ax4 = add_axes(fig, grid[0, 3])
ax5 = add_axes(fig, grid[1, 0])
ax6 = add_axes(fig, grid[1, 1])
ax7 = add_axes(fig, grid[1, 2])
ax8 = add_axes(fig, grid[1, 3])
ax9 = add_axes(fig, grid[2, 0])
ax10 = add_axes(fig, grid[2, 1])
ax11 = add_axes(fig, grid[2, 2])
ax12 = add_axes(fig, grid[2, 3])


# Set plot index list
plot_idxs = [0,1,2,3,4,5,6,7,8,9,10,11]

# Set contour levels
levels = np.arange(21, 31, 0.5)

# Set colormap
cmap = gvcmaps.BlueYellowRed

for i, axes in enumerate([ax1, ax2, ax3, ax4, ax5, ax6,ax7,ax8,ax9,ax10,ax11,ax12]):
    dataset = sst_mclm[plot_idxs[i], :, :]
    # Contourf plot data
    contour = axes.contourf(dataset.lon,
                            dataset.lat,
                            dataset.data,
                            vmin=21,
                            vmax=31,
                            cmap=cmap,
                            levels=levels)
    # Add month name as text
    axes.text(0.05,
              0.95,
              add_text(i),
              ha='left',
              va='top',
              transform=axes.transAxes,
              fontsize=12,
              zorder=5)

# Set colorbounds of norm
colorbounds = np.arange(21, 31, 0.5)
# Use cmap to create a norm and mappable for colorbar to be correctly plotted
norm = mcolors.BoundaryNorm(colorbounds, cmap.N)
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

# Add colorbar for all six plots
ax=[ax1, ax2, ax3, ax4, ax5, ax6,ax7,ax8,ax9,ax10,ax11,ax12]
cbar = fig.colorbar(mappable,
                 ax=[ax1, ax2, ax3, ax4, ax5, ax6,ax7,ax8,ax9,ax10,ax11,ax12],
                 ticks=colorbounds[1:-1:2],
                 drawedges=True,
                 orientation='vertical',
                 shrink=0.90,
                 pad=0.03,
                 aspect=35,
                 extendfrac='auto',
                 extendrect=True)
cbar.set_label('SST (°C)',fontsize = 14)
cbar.ax.tick_params(labelsize = 12)
plt.savefig('multiexample.png',dpi = 300)

