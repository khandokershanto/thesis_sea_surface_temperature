import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import os
from datetime import datetime

import geocat.viz.util as gvutil
from geocat.viz import cmaps as gvcmaps

# open dataset
ds = xr.open_dataset('data/monmean1981-2019.nc')
nds = xr.open_dataset(r'G:\oisst\noaa_oisst_monthly.nc')


# anomaly
sst = ds.analysed_sst.sel(time=slice('1982-01-01','2019-12-01')) - 273.14
nsst = nds.sst.sel(time=slice('1982-01-01','2019-12-01')) #nino
nclm = nsst.groupby('time.month').mean(dim='time')
clm = sst.groupby('time.month').mean(dim='time')

nanm = (nsst.groupby('time.month') - nclm) # nino anomaly
anm = (sst.groupby('time.month') - clm) # anomaly

# defining a function
def areaave(indat, latS, latN, lonW, lonE):
    lat = indat.lat
    lon = indat.lon

    if (((lonW < 0) or (lonE < 0)) and (lon.values.min() > -1)):
        anm = indat.assign_coords(lon=((lon + 180) % 360 - 180))
        lon = ((lon + 180) % 360 - 180)
    else:
        anm = indat

    iplat = lat.where((lat >= latS) & (lat <= latN), drop=True)
    iplon = lon.where((lon >= lonW) & (lon <= lonE), drop=True)

    #  print(iplat)
    #  print(iplon)
    odat = anm.sel(lat=iplat, lon=iplon).mean(("lon", "lat"), skipna=True)
    return (odat)

# -- Calculate nino3.4 index
nino = areaave(nanm,-5,5,-170,-120)
ninoSD=nino/nino.std(dim='time')
rninoSD = nino.rolling(time=7, center=True).mean('time')

# -- Calculate bob area averaged sst
bob = areaave(anm,5,25,80,100)

# -- Detrending
def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit


# -- Running mean
ranm = bob.rolling(time=7, center=True).mean('time')
rdanm = detrend_dim(ranm,'time',1)


# simultaneous
cor0 = xr.corr(rninoSD, rdanm, dim="time")
reg0 = xr.cov(rninoSD, rdanm, dim="time")/rninoSD.var(dim='time',skipna=True).values

def make_fig(cor,reg, grid_space):
    ax = fig.add_subplot(grid_space,
                         projection=ccrs.PlateCarree())
    ax.set_extent([80, 100, 5, 25], crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5, alpha=0.6)

    # Usa geocat.viz.util convenience function to set axes parameters
    gvutil.set_axes_limits_and_ticks(ax,
                                     xlim= (80,100),
                                     ylim=(5, 25),
                                     xticks=np.arange(80, 100, 5),
                                     yticks=np.arange(5, 25, 5))

    # Use geocat.viz.util convenience function to make plots look like NCL
    # plots by using latitude, longitude tick labels
    gvutil.add_lat_lon_ticklabels(ax)
    # Remove the degree symbol from tick labels
    ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol=''))
    ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol=''))

    # Use geocat.viz.util convenience function to add minor and major tick lines
    gvutil.add_major_minor_ticks(ax, labelsize=10)

    # Make sure that tick marks are only on the left and bottom sides of subplot
    ax.tick_params('both', which='both', top=False, right=False)

    # Import the default color map
    newcmp = gvcmaps.BlueYellowRed
    index = [5, 20, 35, 50, 65, 85, 95, 110, 125, 0, 0, 135, 150, 165, 180, 200, 210, 220, 235, 250]
    color_list = [newcmp[i].colors for i in index]
    # -- Change to white
    color_list[9] = [1., 1., 1.]
    color_list[10] = [1., 1., 1.]

    # Define dictionary for kwargs
    kwargs = dict(
        vmin=-1.0,
        vmax=1.0,
        levels=21,
        colors=color_list,
        add_colorbar=False,  # allow for colorbar specification later
        transform=ccrs.PlateCarree(),  # ds projection
    )
    # Contouf-plot U data (for filled contours)
    fillplot = cor.plot.contourf(ax=ax, **kwargs)



    # Add land to the subplot
    ax.add_feature(cfeature.LAND,
                   facecolor='lightgray',
                   edgecolor='black',
                   linewidths=0.5,
                   zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=0.5, zorder=1)

    # Set subplot titles
    gvutil.set_titles_and_labels(ax,
                                 lefttitle='degC',
                                 lefttitlefontsize=10,
                                 righttitle='$(W m s^{-2})$',
                                 righttitlefontsize=10)




    return ax,fillplot

fig = plt.figure(figsize=(10, 12))
grid = fig.add_gridspec(ncols=1,nrows=1)

ax1, fill1 = make_fig(cor0,reg0, grid[0,0])

fig.colorbar(fill1,
                 ax=[ax1],
#                 ticks=np.linspace(-5, 5, 11),
                 drawedges=True,
                 orientation='horizontal',
                 shrink=0.5,
                 pad=0.05,
                 extendfrac='auto',
                 extendrect=True)
