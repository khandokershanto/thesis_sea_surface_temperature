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
# open dataset
da = xr.open_dataset(r'data\noaa_oisst_v2_merged_1982_2020.nc')
oisst_sst = da.sst # sea surface temperature
oisst_mclim = oisst_sst.groupby('time.month').mean(dim='time')

# anomaly data
oisst_anm = (oisst_sst.groupby('time.month') - oisst_mclim)


#define custom seasons <- winter = NDJF-11,12,1,2 ; spring = MAM-3,4,5 ; Summer = JJA-6,7,8 ; Fall = SO-9,10
winter = oisst_sst.time.dt.month.isin([1,2,11,12])
winter_clim = oisst_sst.sel(time = winter)

spring = oisst_sst.time.dt.month.isin([3,4,5])
spring_clim = oisst_sst.sel(time = spring)

summer = oisst_sst.time.dt.month.isin([6,7,8])
summer_clim = oisst_sst.sel(time = summer)

fall = oisst_sst.time.dt.month.isin([9,10])
fall_clim = oisst_sst.sel(time = fall)


# linear trend function
def linear_trend(x):
    date = x.time
    ndate = np.arange(len(date))
    pf = np.polyfit(ndate, x, 1)
    # we need to return a dataarray or else xarray's groupby won't be happy
    return xr.DataArray(pf[0]*120)

# stack lat and lon into a single dimension called allpoints
winter_st = winter_clim.stack(allpoints=['lat', 'lon'])
winter_trnd = winter_st.groupby('allpoints').apply(linear_trend)
winter_slope = winter_trnd.unstack('allpoints')

spring_st = spring_clim.stack(allpoints=['lat', 'lon'])
spring_trnd = spring_st.groupby('allpoints').apply(linear_trend)
spring_slope = spring_trnd.unstack('allpoints')

summer_st = summer_clim.stack(allpoints=['lat', 'lon'])
summer_trnd = summer_st.groupby('allpoints').apply(linear_trend)
summer_slope = summer_trnd.unstack('allpoints')

fall_st = fall_clim.stack(allpoints=['lat', 'lon'])
fall_trnd = fall_st.groupby('allpoints').apply(linear_trend)
fall_slope = fall_trnd.unstack('allpoints')

winter_slope.plot()
spring_slope.plot()
summer_slope.plot()
fall_slope.plot()