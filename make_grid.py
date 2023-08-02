
import matplotlib.pyplot as plt
import netCDF4 as nc
from pyproj import Proj, transform
import numpy as np
import rasterio

# 


# reading the netCDF file
filename = 'ice_thickness_2021.nc'
dataset = nc.Dataset(filename)

# extracting latitude, longitude, and ice thickness data
lat = dataset.variables['lat'][:]  # copy full content of lat and lon variables
lon = dataset.variables['lon'][:]
ice_thickness = dataset.variables['sea_ice_thickness'][0,:]

ice_thickness = np.nan_to_num(ice_thickness)  # turn nan values to 0

# converting lat/lon (degrees) to Polar Stereographic (meters)
proj_latlon = Proj(proj='latlong', datum='WGS84')  # lan/lat projection with WGS84 refrence
proj_polar_stereo = Proj(init='epsg:32661')  # North Polar Stereographic projection which has units in meters
lon_m, lat_m = transform(proj_latlon, proj_polar_stereo, lon.flatten(), lat.flatten())  # transforming lat/lon coordinates from one projection to the other

lon_m = lon_m.reshape(lon.shape)  # reshaping back to 2D arrays as this allows for each point in a 
lat_m = lat_m.reshape(lat.shape)  # geospatial grid to be associated with a unique (lat,lon) coordinate pair

# creating a grid and assign ice thickness to each corresponding point

# defining resolution (given) and extent
resolution = 25000  # 25 km
xmin, xmax = np.min(lon_m), np.max(lon_m)
ymin, ymax = np.min(lat_m), np.max(lat_m)

# defining number of cells/bins with size (resolution) fitting the range min to max
cols = int(np.ceil((xmax - xmin) / resolution))  # ceil rounds to the nearest integer
rows = int(np.ceil((ymax - ymin) / resolution))

# initializing spatial transformation for the grid of pixels (raster image)
transform = rasterio.transform.from_origin(xmin, ymin, resolution, resolution) 

# defining the destination data array
ice_thickness_grid = np.empty(shape=(rows, cols))

# using reverse indexing to fill the ice thickness grid array 
for i in range(len(lat)):
    for j in range(len(lon)):
        x = lon_m[i, j]
        y = lat_m[i, j]
        col, row = ~transform * (x, y)
        col, row = int(col), int(row)
        ice_thickness_grid[row, col] = ice_thickness[i, j]

# now we have a grid in meters with corresponding ice thickness

# graphing the grid
plt.figure(figsize=(10,10))
plt.imshow(ice_thickness_grid, origin='lower')
plt.grid(color='white', linestyle='-', linewidth=0.5)
plt.colorbar(label='Ice Thickness')
plt.title('Ice Thickness grid')
plt.show()

# print(ice_thickness_grid)




