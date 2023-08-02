
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import pyproj
# g_star.py - UIT - Martin Stave - mvrtinstave@gmail.com
# currently creating spaced grid using ice thickness dataset


'''
plan:
create an algorithm for pathing in the artic
get array of coordinates, figure out which is the best path A*
- use ice index to navigate
- globle land mask to navigate
   
    '''

# access dataset within nc file 
file_name = 'ice_thickness_2021.nc'

# loading datafile into xarray dataset
ds = xr.open_dataset(file_name)
print(ds.attrs)  # get files attributes, 

# dataset has EPSG:6931 projection which usually has units in meters
# but the coordinate units are in degrees, which is weird

# converting the dataarrays to pandas dataframe
lat = ds['lat'].to_pandas()
lon = ds['lon'].to_pandas()
ice_thickness = ds['sea_ice_thickness'].isel(time=0).to_pandas()  # with specified dimensions 


lon = np.ravel(lon)
lat = np.ravel(lat)
ice_thickness = np.ravel(ice_thickness)  # flattening ice thickness to 1D array
ice_thickness = np.nan_to_num(ice_thickness)  # turn nan values to 0


wgs84 = pyproj.CRS('EPSG:4326')  # WGS84 is world geodetic system, in degrees
arctic_meter_proj = pyproj.CRS('EPSG:6931')  # EPSG:6931 is a meter-based system suitable for Arctic regions

# defining transformer
transformer = pyproj.Transformer.from_crs(wgs84, arctic_meter_proj)


# Transforming a single point
lon_sample, lat_sample = lon[0], lat[0]
lon_transformed_sample, lat_transformed_sample = transformer.transform(lon_sample, lat_sample)
# print("Transformed coordinates sample:", lon_transformed_sample, lat_transformed_sample)

    
# transforming coordinates
lon_transformed, lat_transformed = transformer.transform(lon, lat)

# defining grid spacing, original grid is 25km
lon_span = lon_transformed.max() - lon_transformed.min()  # computing spans (total range of lon and lat values)
lat_span = lat_transformed.max() - lat_transformed.min()  # in meters (as stereographic projection is in meters)

# print(lon_span)
# print(lat_span)


grid_spacing = 25000  # 25km, could be changed...


# print("lon_span:", lon_span)
# print("lat_span:", lat_span)
# print("grid_spacing:", grid_spacing)
# print("lon_span / grid_spacing:", lon_span / grid_spacing)
# print("lat_span / grid_spacing:", lat_span / grid_spacing)
# print("Transformed coordinates sample:", lon_transformed[:10], lat_transformed[:10])


# computing equivalent bins for each dimension
num_bins_lon = np.ceil(lon_span / grid_spacing).astype(int)  # lon_span/grid_space = how many 25km bins can fit in total lon range
num_bins_lat = np.ceil(lat_span / grid_spacing).astype(int)  # np.ceil rounds up to nearest whole number, as integer. adjust max or min for more accurate bin


# print("num_bins_lon:", num_bins_lon)
# print("num_bins_lat:", num_bins_lat)


# print("num_bins_lon:", num_bins_lon)
# print("num_bins_lat:", num_bins_lat)


# combining variables into one datafram
df = pd.DataFrame({
    'lon': lon_transformed,
    'lat': lat_transformed,
    'ice_thickness': ice_thickness
})


# defining bin edges
lon_bins = np.linspace(df['lon'].min(), df['lon'].max(), num_bins_lon + 1)
lat_bins = np.linspace(df['lat'].min(), df['lat'].max(), num_bins_lat + 1)

# assigning each point to a bin (grid cell)
df['lon_bin'] = pd.cut(df['lon'], bins = lon_bins, labels = False, include_lowest=True)
df['lat_bin'] = pd.cut(df['lat'], bins = lat_bins, labels = False, include_lowest=True)

# creating a 2D array for ice thickness based on bins
ice_thickness_grid = df.groupby(['lon_bin','lat_bin'])['ice_thickness'].mean().unstack().values


# testing the grid
# plt.figure(figsize=(10,10))
# plt.imshow(ice_thickness_grid, origin='lower')
# plt.colorbar(label='Ice Thickness')
# plt.title('Ice Thickness grid')
# plt.show()


# print(ice_thickness_grid)
































