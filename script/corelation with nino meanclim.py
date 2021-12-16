
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


# monthly climatology in DataArray format
oisst_sst = oisst.sst.sel(time=slice('1982-01-16','2019-12-16'))
nino_sst = nino.sst
oisst_mclim = oisst_sst.groupby('time.month').mean(dim='time')
nino_mclm = nino_sst.groupby('time.month').mean(dim='time')

# anomaly data
oisst_anm = (oisst_sst.groupby('time.month') - oisst_mclim)
nino_anm = (nino_sst.groupby('time.month') - nino_mclm)

# 5 month rolling average
roisst = oisst_anm.rolling(time=5, center=True).mean('time')
# -- Detorending
def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit
droisst = detrend_dim(roisst,'time',1)

nino_idx = nino_anm.mean(('lat','lon'),skipna=True)
ninoSD=nino_idx/nino_idx.std(dim='time')
# 5 month running mean
rninoSD = ninoSD.rolling(time=5, center=True).mean('time')

# correlation
cor0 = xr.corr(nino_idx, oisst_anm, dim="time")
cor1 = xr.corr(rninoSD,droisst,dim='time')



# -- figure plot

def makefig(cor,grid_space):
    # Fix the artifact of not-shown-data around 0 and 360-degree longitudes
    cor = gvutil.xr_add_cyclic_longitudes(cor, 'lon')
    # Generate axes using Cartopy to draw coastlines
    ax = fig.add_subplot(grid_space,
                         projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_xlabel('Longitude')
    ax.add_feature(cfeature.LAND,
                   facecolor="darkgray",
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
    gvutil.add_major_minor_ticks(ax, labelsize=14)

    # Use geocat.viz.util convenience function to make latitude, longitude tick labels
    gvutil.add_lat_lon_ticklabels(ax)
    gvutil.set_titles_and_labels(ax,
                                 maintitle='Climatology 1982-2019',
                                 maintitlefontsize=18,
                                 ylabel='Latitude',
                                 xlabel='Longitude',
                                 labelfontsize=16)

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


fig = plt.figure(figsize=(7.6, 6.5))
grid = fig.add_gridspec(ncols=1, nrows=1)

ax1, fill1 = makefig(cor1, grid[0,0])
ax1.set_xlabel('longitude')
ax1.set_ylabel('Latitude')

fig.colorbar(fill1,
                 ax=ax1,
                 drawedges=True,
                 orientation='vertical',
                 shrink=0.90,
                 pad=0.05,
                 extendfrac='auto',
                 extendrect=True)

fig.suptitle('SST correlation with Nino3.4', fontsize=18, y=0.94,x = 0.43)
plt.savefig('correlationwithnino3.4.png',dpi = 300)





