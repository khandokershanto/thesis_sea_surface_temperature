#            Mann-Kendall Trend Test from R
import xarray as xr
import numpy as np

# create an example dataset
da = xr.open_dataset(r'data\senslopeall.nc')
winter = xr.open_dataset(r'data\win_sen.nc')
spring = xr.open_dataset(r'data\spr_sen.nc')
summer = xr.open_dataset(r'data\sum_sen.nc')
fall = xr.open_dataset(r'data\fall_sen.nc')

mkt_all = da.layer
mkt_win = winter.layer
mkt_spr = spring.layer
mkt_sum = summer.layer
mkt_fall = fall.layer

mkt_fall.mean()








