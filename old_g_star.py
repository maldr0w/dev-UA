
import numpy as np
import pyproj
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
# g_star.py - UIT - Martin Stave - mvrtinstave@gmail.com
# inititates and runs the A* algorithm on sea ice thickness dataset

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

# converting the dataarrays to pandas dataframe
lat = ds['lat'].to_pandas()
lon = ds['lon'].to_pandas()
ice_thickness = ds['sea_ice_thickness'].isel(time=0).to_pandas()  # with specified dimensions 

ice_thickness = np.nan_to_num(ice_thickness)  # turn nan values to 0
ice_thickness = np.ravel(ice_thickness)  # flattening ice thickness to 1D array

# print(ds.attrs)
# # defining coordinate system
# wgs84 = pyproj.CRS('EPSG:4326')  # 4326 is world geodetic system
# polar_projection = pyproj.CRS('EPSG:3995')  # Arctic polar stereographic projection

# # defining tranformer 
# transformer = pyproj.Transformer.from_crs(wgs84, polar_projection)

# flattening lon and lat to 1D arrays
lon = np.ravel(lon)
lat = np.ravel(lat)

# limiting values in array to account for infinity errors
# lat = np.clip(latitude, -89.9, 89.9)  
# lon = np.clip(longitude, -179.9, 179.9)  

# # transforming coordinates
# lon, lat = transformer.transform(longitude, latitude)

# print('Transformed Lon:', lon)

# defining grid spacing, original grid is 25km
lon_span = lon.max() - lon.min()  # computing spans (total range of lon and lat values)
lat_span = lat.max() - lat.min()  # in meters (as stereographic projection is in meters)

grid_spacing = 25000  # 25km, could be changed...

# computing equivalent bins for each dimension
num_bins_lon = np.ceil(lon_span / grid_spacing).astype(int)  # lon_span/grid_space = how many 25km bins can fit in total lon range
num_bins_lat = np.ceil(lat_span / grid_spacing).astype(int)  # np.ceil rounds up to nearest whole number, as integer

'''
this ensures that each bin is approx 25km.
for more accuracy of bin size, adjust max or min coordinates
    '''

# combining variables into one datafram
df = pd.DataFrame({
    'lon': lon,
    'lat': lat,
    'ice_thickness': ice_thickness
})

# df['lon'] = df['lon'].round(5)

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


print(ice_thickness_grid)




# print("num_bins_lon:", num_bins_lon)
# print("num_bins_lat:", num_bins_lat)
# print("lon_span:", lon_span)
# print("lat_span:", lat_span)
# print("Unique lon:", np.unique(longitude))
# print("Unique lat:", np.unique(latitude))













