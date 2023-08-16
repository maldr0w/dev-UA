
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy as cp


## Extracting dataset variables
FILE_NAME = 'ice_thickness_2021.nc'
GRID_RESOLUTION = 25000 # m

def extract_data(file):
    '''
    input: netCDF datafile
    output: lat/lon and sea ice thickness data-array    
        '''

    ds = xr.open_dataset(file)  # opens dataset file

    # extracting variables by index
    latitude, longitude = ds['lat'].values, ds['lon'].values
    sea_ice_thickness_with_nan = ds['sea_ice_thickness'][0,:].values  # nan values will be useful in plotting
    sea_ice_thickness = np.nan_to_num(sea_ice_thickness_with_nan)  # replace nan values with 0

    ds.close()  # closes dataset to save resources

    return latitude, longitude, sea_ice_thickness, sea_ice_thickness_with_nan


latitude, longitude, sea_ice_thickness, sea_ice_thickness_with_nan = extract_data(FILE_NAME)
lat_min, lon_min, lat_max, lon_max = latitude.min(), longitude.min(), latitude.max(), longitude.max()

def scatter_plot():
    fig, ax = plt.subplots(1, 1, figsize=(5,5), 
                               subplot_kw = {'projection':cp.crs.NorthPolarStereo()})
    
    # add ocean and land outlines and colors
    ax.add_feature(cp.feature.OCEAN)
    ax.add_feature(cp.feature.LAND, edgecolor='black', zorder=1)
    ax.set_facecolor((1.0,1.0,1.0))
    ax.set_extent([-180,180,90,66], cp.crs.PlateCarree()) # set coordinate extent
    # changing extent will increase or decrease coverage
    
    # add gridlines displaying latitude and longitude angles
    ax.gridlines(draw_labels=True, alpha=0.5, color='gray', 
                 linestyle='-', linewidth=0.5, 
                 xlocs = np.arange(-180, 181, 30),
                 ylocs = np.arange(60, 91, 5))
    
    # visualize sea ice thickness as scatterpoints with corresponding longitude and latitude
    sc = ax.scatter(longitude, latitude, c=sea_ice_thickness_with_nan,
                    cmap='jet', marker='o', s= 1,
                    transform=cp.crs.PlateCarree())
    
    # adding colorbar 
    plt.colorbar(sc, ax=ax, orientation='vertical')
    plt.show()
    
# scatterplot()

import geopandas as gpd
from shapely.geometry import Point


land_geometries = [geom for geom in cfeature.LAND.geometries()]

def is_on_land(lon,lat):
    point = Point(lon,lat)
    for geom in land_geometries:
        if geom.contains(point):
            return True
    return False

on_land_flags = [is_on_land(lon,lat) for lon,lat in zip(longitude,latitude)]
print(on_land_flags)