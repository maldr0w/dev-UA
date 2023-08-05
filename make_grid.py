
import matplotlib.pyplot as plt
import netCDF4 as nc
from pyproj import Proj, transform as proj_transform
import numpy as np
import rasterio

# make_grid.py - UIT - Martin Stave - Mvrtinstave@gmail.com
# collects data and transforms lat/lon to meters in a grid 

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
lon_m, lat_m = proj_transform(proj_latlon, proj_polar_stereo, lon.flatten(), lat.flatten())  # transforming lat/lon coordinates from one projection to the other

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
raster_transform = rasterio.transform.from_origin(xmin, ymin, resolution, resolution) 

# defining the destination data array
ice_thickness_grid = np.empty(shape=(rows, cols))

# using reverse indexing to fill the ice thickness grid array 
for i in range(len(lat)):
    for j in range(len(lon)):
        x = lon_m[i, j]
        y = lat_m[i, j]
        col, row = ~raster_transform * (x, y)  # ~ inverts transform, transforming from pixel coordinate to geographical coords
        col, row = int(col), int(row)
        ice_thickness_grid[row, col] = ice_thickness[i, j]

# now we have a grid in meters with corresponding ice thickness

zoomed_grid = ice_thickness_grid[100:-100, 100:-100]

# print(ice_thickness_grid)
# print(ice_thickness_grid.shape)



# determining what the closest point is in the data array
# lat_point = 78.522
# lon_point = 63.946

lat_point = 69.204
lon_point = 170.727
diff_array = np.sqrt((lat-lat_point)**2 + (lon-lon_point)**2)
index = np.unravel_index(np.argmin(diff_array, axis=None), diff_array.shape)

nearest_lat = lat[index]
nearest_lon = lon[index]

ice_thickness_at_point = ice_thickness_grid[index]

print(nearest_lat)
print(nearest_lon)
print(ice_thickness_at_point)

# Transform nearest latitude and longitude into meters
nearest_lon_m, nearest_lat_m = proj_transform(proj_latlon, proj_polar_stereo, nearest_lon, nearest_lat)
print(nearest_lon_m, nearest_lat_m)

# Transform back:
lon_d, lat_d = proj_transform(proj_polar_stereo, proj_latlon, nearest_lon_m, nearest_lat_m)
print(f"lon_d:{lon_d},lat_d:{ lat_d}")

# step_size = np.diff(zoomed_grid)
# average_step_size = np.mean(step_size)
# print(average_step_size)
col = (nearest_lon_m - xmin) / resolution
row = (nearest_lat_m - ymin) / resolution

col = int(col)
row = int(row)

# col,row = ~raster_transform * (col,row)

print(f"Grid coordinates: {col}, {row}")
# cols, rows = ice_thickness_grid.shape

print(f"xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}, resolution: {resolution}")
print(f"nearest_lon_m: {nearest_lon_m}, nearest_lat_m: {nearest_lat_m}")


# # graphing the grid
# plt.figure(figsize=(10,10))
# plt.imshow(ice_thickness_grid)#, origin='lower')
# plt.plot(col,row, 'ro')
# # plt.grid(color='white', linestyle='-', linewidth=0.5)
# plt.colorbar(label='Ice Thickness')
# plt.title('Ice Thickness grid')
# plt.show()


plt.figure(figsize=(10,10))
# plot the grid with the origin at the lower left corner
plt.imshow(ice_thickness_grid)
# plot the point, adjusting the row index so it also starts at the bottom
plt.plot(col, row, 'ro')  
plt.colorbar(label='Ice Thickness')
plt.title('Ice Thickness grid')
plt.show()
