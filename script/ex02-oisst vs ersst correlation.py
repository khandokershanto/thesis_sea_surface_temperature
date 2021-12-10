import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import os
from datetime import datetime
# geocat for more aesthetic plot
import geocat.viz.util as gvutil
#import geocat.datafiles as gdf
from geocat.viz import cmaps as gvcmaps

#load data
ersst = xr.open_dataset('data/sst.mnmean.nc')
oisst = xr.open_dataset('data/monmean1981-2019.nc')



#time slicing
oisst_sst = oisst['analysed_sst'].sel(time=slice('1982-01-01','2019-12-31'))
oisst_sst = oisst_sst - 273.14
ersst_sst = ersst['sst'].sel(time=slice('1982-01-01','2019-12-31'))

time = oisst_sst.time

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

oisst_ts = areaave(oisst_sst,5,25,80,100)
ersst_ts = areaave(ersst_sst,5,25,80,100)


np.corrcoef([ersst_ts],[oisst_ts])

# plot
fig = plt.figure(figsize=[10,6])
ax1 = fig.add_subplot(111)
ax1.plot(time, ersst_ts, '-',  linewidth=1)
ax1.plot(time, oisst_ts, '-',  linewidth=1.5)
ax1.set_ylabel('SST ($^\circ$C)',size = 14)
ax1.set_xlabel('Year',size = 14)
leg = ax1.legend(['ERSST','OISST'],frameon = False,fontsize = 'large',ncol = 2)

#legend line width increase
for line in leg.get_lines():
    line.set_linewidth(3.0)
#



