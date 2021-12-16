import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import geocat.viz.util as gvutil
from geocat.viz import cmaps as gvcmaps

# open datasets
nino = xr.open_dataset(r'data\noaa_oisst_monthly.nc')
oisst = xr.open_dataset(r'data\noaa_oisst_v2_merged_1982_2020.nc')


# Read variables and make monthly average climatology
oisst_sst = oisst.sst # sea surface temperature
nino_sst = nino.sst
nino_mclm = nino_sst.groupby('time.month').mean(dim='time')
oisst_mclim = oisst_sst.groupby('time.month').mean(dim='time')

# anomaly data
oisst_anm = (oisst_sst.groupby('time.month') - oisst_mclim)
nino_anm = (nino_sst.groupby('time.month') - nino_mclm)

#define custom seasons <- winter = NDJF-11,12,1,2 ; spring = MAM-3,4,5 ; Summer = JJA-6,7,8 ; Fall = SO-9,10
#----------------------------------------------------------------------
winter = oisst_anm.time.dt.month.isin([1,2,11,12])
winter_clim = oisst_anm.sel(time = winter)

spring = oisst_anm.time.dt.month.isin([3,4,5])
spring_clim = oisst_anm.sel(time = spring)

summer = oisst_anm.time.dt.month.isin([6,7,8])
summer_clim = oisst_anm.sel(time = summer)

fall = oisst_anm.time.dt.month.isin([9,10])
fall_clim = oisst_anm.sel(time = fall)

# 5 month rolling average for all season
rwinter = winter_clim.rolling(time=5, center=True).mean('time')
rsummer = summer_clim.rolling(time=5, center=True).mean('time')
rspring = spring_clim.rolling(time=5, center=True).mean('time')
rfall = fall_clim.rolling(time=5, center=True).mean('time')

# -- Detrending
def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

drwinter = detrend_dim(rwinter,'time',1)
drsummer = detrend_dim(rsummer,'time',1)
drspring = detrend_dim(rspring,'time',1)
drfall = detrend_dim(rfall,'time',1)

# nino index and rolling mean
nino_idx = nino_anm.mean(('lat','lon'),skipna=True)
ninoSD=nino_idx/nino_idx.std(dim='time')
# 5 month running mean
rninoSD = ninoSD.rolling(time=5, center=True).mean('time')

# correlation
cor_winter = xr.corr(rninoSD, drwinter, dim="time")
cor_spring = xr.corr(rninoSD, drspring, dim="time")
cor_summer = xr.corr(rninoSD, drsummer, dim="time")
cor_fall = xr.corr(rninoSD, drfall, dim="time")


##      Figure plot (multi)
def makefig(cor,grid_space):
    # Fix the artifact of not-shown-data around 0 and 360-degree longitudes
    cor = gvutil.xr_add_cyclic_longitudes(cor, 'lon')
    # Generate axes using Cartopy to draw coastlines
    ax = fig.add_subplot(grid_space,
                         projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_xlabel('Longitude')
    ax.add_feature(cfeature.LAND,
                   facecolor="lightgray",
                   edgecolor='black',
                   linewidths=1,
                   zorder=2)

    # Use geocat.viz.util convenience function to set axes limits & tick values
    gvutil.set_axes_limits_and_ticks(ax,
                                     xlim=(80, 100),
                                     ylim=(5, 25),
                                     xticks=np.arange(80, 101, 5),
                                     yticks=np.arange(5,26,5))

    # Use geocat.viz.util convenience function to add minor and major tick lines
    gvutil.add_major_minor_ticks(ax, labelsize=12)

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
        vmin=-0.64,
        vmax=0.64,
        levels=21,
        colors=color_list,
        add_colorbar=False,  # allow for colorbar specification later
        transform=ccrs.PlateCarree(),  # ds projection
    )

    # Contouf-plot U data (for filled contours)
    fillplot = cor.plot.contourf(ax=ax, **kwargs)


    return ax, fillplot

fig = plt.figure(figsize=(8.5, 7.5), constrained_layout=True)
grid = fig.add_gridspec(ncols=2, nrows=2)


ax1, fill1 = makefig(cor_winter, grid[0,0])
ax2, fill2 = makefig(cor_spring, grid[0,1])
ax3, fill3 = makefig(cor_summer, grid[1,0])
ax4, fill4 = makefig(cor_fall, grid[1,1])


ax1.set_xlabel(""),ax1.set_ylabel("")
ax2.set_xlabel(""),ax2.set_ylabel("")
ax3.set_xlabel(""),ax3.set_ylabel("")
ax4.set_xlabel(""),ax4.set_ylabel("")
#ax1.set_xlabel(""),ax1.set_ylabel("Latitude",fontsize=14)
#ax3.set_xlabel("Longitude",fontsize=14),ax3.set_ylabel("Latitude",fontsize=14)
#ax4.set_xlabel("Longitude",fontsize=14),ax4.set_ylabel("")

ax=[ax1, ax2, ax3, ax4]
cbar = fig.colorbar(fill1,
                 ax=[ax1, ax2, ax3, ax4],
                 drawedges=True,
                 orientation='vertical',
                 shrink=0.85,
                 pad=0.03,
                 extendfrac='auto',
                 extendrect=True)
cbar.ax.tick_params(labelsize = 12)
plt.savefig('seasonalCorwithnino3.4.png',dpi = 300)