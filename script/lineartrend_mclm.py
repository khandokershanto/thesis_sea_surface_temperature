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


# create an example dataset
da = xr.open_dataset(r'data\noaa_oisst_v2_merged_1982_2020.nc')
sst = da.sst
date = sst.time
ndate = len(date)
ndate = np.arange(len(date))

# define a function to compute a linear trend of a timeseries
def linear_trend(x):
    pf = np.polyfit(ndate, x, 1)
    # we need to return a dataarray or else xarray's groupby won't be happy
    return xr.DataArray(pf[0])

def linear_trend(x):
    date = x.time
    ndate = np.arange(len(date))
    pf = np.polyfit(ndate, x, 1)
    # we need to return a dataarray or else xarray's groupby won't be happy
    return xr.DataArray(pf[0]*120)


# stack lat and lon into a single dimension called allpoints
stacked = sst.stack(allpoints=['lat', 'lon'])
# apply the function over allpoints to calculate the trend at each point
trend = stacked.groupby('allpoints').apply(linear_trend)
# unstack back to lat lon coordinates
trend_unstacked = trend.unstack('allpoints')

sst_clim = trend_unstacked
sst_clim.lon

##              PLOT Figure
# Now plot mean SST climatology
############################################
# Generate figure (set its size (width, height) in inches)
fig = plt.figure(figsize=(7.6, 6.5))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.LAND,
                   facecolor="darkgray",
                   edgecolor='black',
                   linewidths=1,
                   zorder=2)

# Usa geocat.viz.util convenience function to set axes parameters
gvutil.set_axes_limits_and_ticks(ax,
                                    ylim=(5,25),
                                    xlim=(80, 100),
                                    xticks=np.arange(80,101 , 5),
                                    yticks=np.arange(5, 26, 5))

    # Use geocat.viz.util convenience function to add minor and major tick lines
gvutil.add_major_minor_ticks(ax, labelsize=14)
gvutil.add_lat_lon_ticklabels(ax)

gvutil.set_titles_and_labels(ax,
                                 maintitle= 'SST Trend (1982-2020)',
                                 maintitlefontsize= 18,
                                 ylabel='Latitude',
                                 xlabel='Longitude',
                                 labelfontsize=16)

# Remove the degree symbol from tick labels
ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol=''))
ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol=''))

# Set contour levels
#levels = np.arange(0.04, 0.25, 0.0)
levels = np.linspace(0.04,0.25,11)
# Set colormap
cmap = gvcmaps.BlueYellowRed

p = ax.contourf(sst_clim.lon,
                        sst_clim.lat,
                        sst_clim.data,
                        vmin=0.04,
                        vmax=0.25,
                        cmap=cmap,
                        levels=levels)

# Set colorbounds of norm
colorbounds = np.linspace(0.04,0.25,11)
# Use cmap to create a norm and mappable for colorbar to be correctly plotted
norm = mcolors.BoundaryNorm(colorbounds, cmap.N)
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)


# cax for adjusting colorbar width
cax = fig.add_axes([ax.get_position().x1+0.02,ax.get_position().y0,0.03,ax.get_position().height])
cbar = plt.colorbar(mappable=mappable,ax=ax, cax=cax,
                 ticks=colorbounds[1:-1:],
                 drawedges=True,
                 orientation='vertical',
                 shrink=1,
                 pad=0.04,
                 aspect=55,
                 extendfrac='auto',
                 extendrect=True)
cbar.set_label('°C/decade',fontsize = 12)
cbar.ax.tick_params(labelsize = 12)
plt.savefig('SST_Trend_1982-2019.png',dpi = 300)