import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import geocat.viz.util as gvutil
from geocat.viz import cmaps as gvcmaps

# load datasets
winter = xr.open_dataset(r'data\win_sen.nc')
spring = xr.open_dataset(r'data\spr_sen.nc')
summer = xr.open_dataset(r'data\sum_sen.nc')
fall = xr.open_dataset(r'data\fall_sen.nc')


mkt_win = winter.layer*10
mkt_spr = spring.layer*10
mkt_sum = summer.layer*10
mkt_fall = fall.layer*10
mkt_fall.min(),mkt_fall.max()

# concating
mkt_season = xr.concat([mkt_win,mkt_spr,mkt_sum,mkt_fall],dim='layer')

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
def add_text(season):
    seasons = [
        'Winter','Spring','Summer','Fall'
    ]
    season = seasons[season]
    return season


# Generate figure (set its size (width, height) in inches)
fig = plt.figure(figsize=(8.5, 7.5), constrained_layout=True)
# Create gridspec to hold six subplots
grid = fig.add_gridspec(ncols=2, nrows=2)
#grid = Grid(fig, rect=111, nrows_ncols=(3, 4),axes_pad=0.25, label_mode='L')

# Add the axes
ax1 = add_axes(fig, grid[0, 0])
ax2 = add_axes(fig, grid[0, 1])
ax3 = add_axes(fig, grid[1, 0])
ax4 = add_axes(fig, grid[1, 1])
# index
plot_idxs = [0,1,2,3]

# Set contour levels
#levels = np.arange(21, 31, 0.5)
levels = np.linspace(-0.37,0.37,11)

# Set colormap
cmap = gvcmaps.BlueYellowRed

for i, axes in enumerate([ax1, ax2, ax3, ax4]):
    dataset = mkt_season[plot_idxs[i], :, :]
    # Contourf plot data
    contour = axes.contourf(dataset.longitude,
                            dataset.latitude,
                            dataset.data,
                            vmin=-0.37,
                            vmax=0.37,
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
colorbounds = np.linspace(-0.37, 0.37, 11)
# Use cmap to create a norm and mappable for colorbar to be correctly plotted
norm = mcolors.BoundaryNorm(colorbounds, cmap.N)
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

# Add colorbar for all six plots
ax=[ax1, ax2, ax3, ax4]
cbar = fig.colorbar(mappable,
                 ax=[ax1, ax2, ax3, ax4],
                 ticks=colorbounds[1:-1:2],
                 drawedges=True,
                 orientation='vertical',
                 shrink=0.90,
                 pad=0.03,
                 aspect=35,
                 extendfrac='auto',
                 extendrect=True)
cbar.set_label('°C/decade)',fontsize = 14)
cbar.ax.tick_params(labelsize = 12)
plt.savefig('mk_test_seasonal.png',dpi = 300)




