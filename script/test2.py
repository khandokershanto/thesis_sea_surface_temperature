import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as PathEffects

import os
from datetime import datetime

import geocat.viz.util as gvutil
from geocat.viz import cmaps as gvcmaps


# == netcdf file name and location"
# open dataset
ds = xr.open_dataset('data/noaa_oisst_v2_merged_1982_2020.nc')
nds = xr.open_dataset('data/noaa_oisst_monthly.nc')


# anomaly
sst = ds.sst.sel(time=slice('1982-01-01','2019-12-01')) - 273.14
nsst = nds.sst.sel(time=slice('1982-01-01','2019-12-01')) #nino
nclm = nsst.groupby('time.month').mean(dim='time')
clm = sst.groupby('time.month').mean(dim='time')

nanm = (nsst.groupby('time.month') - nclm) # nino anomaly
anm = (sst.groupby('time.month') - clm) # anomaly


# -- regional average
def wgt_areaave(indat, latS, latN, lonW, lonE):
  lat=indat.lat
  lon=indat.lon

  if ( ((lonW < 0) or (lonE < 0 )) and (lon.values.min() > -1) ):
     anm=indat.assign_coords(lon=( (lon + 180) % 360 - 180) )
     lon=( (lon + 180) % 360 - 180)
  else:
     anm=indat

  iplat = lat.where( (lat >= latS ) & (lat <= latN), drop=True)
  iplon = lon.where( (lon >= lonW ) & (lon <= lonE), drop=True)

#  print(iplat)
#  print(iplon)
  wgt = np.cos(np.deg2rad(lat))
  odat=anm.sel(lat=iplat,lon=iplon).weighted(wgt).mean(("lon", "lat"), skipna=True)
  return(odat)

# -- Calculate nino3.4 index
nino=wgt_areaave(nanm,-5,5,-170,-120)
ninoSD=nino/nino.std(dim='time')
rninoSD=ninoSD.rolling(time=7, center=True).mean('time')


# -- Detorending
def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

ranm = anm.rolling(time=7, center=True).mean('time')
rdanm = detrend_dim(ranm,'time',1)


##############################################################################
def makefig(data, grid_space):
    # Fix the artifact of not-shown-data around 0 and 360-degree longitudes
    data = gvutil.xr_add_cyclic_longitudes(data, 'lon')
    # Generate axes using Cartopy to draw coastlines
    ax = fig.add_subplot(grid_space,
                         projection=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5, alpha=0.6)

    # Use geocat.viz.util convenience function to set axes limits & tick values
    gvutil.set_axes_limits_and_ticks(ax,
                                     xlim=(80, 100),
                                     ylim=(5, 25),
                                     xticks=np.arange(80, 101, 5),
                                     yticks=np.arange(5, 26, 5))

    # Use geocat.viz.util convenience function to add minor and major tick lines
    gvutil.add_major_minor_ticks(ax, labelsize=10)

    # Use geocat.viz.util convenience function to make latitude, longitude tick labels
    gvutil.add_lat_lon_ticklabels(ax)

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
    fillplot = data.plot.contourf(ax=ax, **kwargs)

    # Draw map features on top of filled contour
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=0.5, zorder=1)


    # Use geocat.viz.util convenience function to add titles to left and right of the plot axis.
    gvutil.set_titles_and_labels(ax,
                                 lefttitle='',
                                 lefttitlefontsize=16,
                                 righttitle='',
                                 righttitlefontsize=16,
                                 xlabel="Longitude",
                                 labelfontsize= 10,
                                 ylabel="Latitude")

    return ax, fillplot



#
data1 = xr.corr(rninoSD, rdanm, dim="time")


# Show the plot
fig = plt.figure(figsize=(8,7))
grid = fig.add_gridspec(ncols=1, nrows=1)
#grid = fig.add_gridspec(ncols=2, nrows=3, hspace=-0.20)

# add subplots
ax1, fill1 = makefig(data1, grid[0,0])

# add colorbar
fig.colorbar(fill1,
                 ax=[ax1],
#                 ticks=np.linspace(-5, 5, 11),
                 drawedges=True,
                 orientation='horizontal',
                 shrink=0.5,
                 pad=0.15,
                 extendfrac='auto',
                 extendrect=True,
                 aspect = 35)

plt.draw()
plt.savefig('ninocor.png',dpi = 300)