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


# ----- Parameter setting ------
# == Figure name ==

# == netcdf file name and location"
fnc = 'data/sst.mnmean.nc'
ds = xr.open_dataset(fnc)
print(ds)

#demo plot
ds['sst'][0].plot()

sst = ds['sst'].sel(time=slice('1920-01-01','2020-12-01'))
#anomaly
clm = sst.sel(time=slice('1920-01-01','2020-12-01')).groupby('time.month').mean(dim='time')
anm = (sst.groupby('time.month') - clm)

time = anm.time
# area average time series
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


# bob sst
ts_bob = areaave(sst,5,25,80,100)
ts_bob_1yr = ts_bob.rolling(time=12, center=True).mean('time')


fig = plt.figure(figsize=[10,6])
ax1 = fig.add_subplot(111)
ax1.plot(time, ts_bob, '-',  linewidth=1,alpha = 0.5,color = 'black')
ax1.plot(time, ts_bob_1yr, '-',  linewidth=1.5,color = 'red')
ax1.set_ylabel('SST ($^\circ$C)',size = 14)
ax1.set_xlabel('Year',size = 14)
ax1.legend(['Monthly SST','1 year running mean'])

# Use geocat.viz.util convenience function to set axes parameters
ystr = 1920
yend = 2020
dyr = 20
itime = np.arange(time.size)
ist, = np.where(time == pd.Timestamp(year=ystr, month=1, day=1) )
iet, = np.where(time == pd.Timestamp(year=yend, month=1, day=1) )

gvutil.set_axes_limits_and_ticks(ax1,
                                 ylim=(25, 31),
                                 yticks=np.linspace(25, 31, 7),
                                 yticklabels=np.linspace(25, 31, 7),
                                 xlim=(itime[0], itime[-1])


###
# Use geocat.viz.util convenience function to add minor and major tick lines
gvutil.add_major_minor_ticks(ax1,
                             x_minor_per_major=4,
                             y_minor_per_major=2,
                             labelsize=15)

plt.draw()
plt.tight_layout()
plt.savefig("moving_avg.png",dpi=300)



## test
x = xr.open_dataset(r'G:\oisst\noaa_oisst_monthly.nc')
