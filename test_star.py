import matplotlib.pyplot as plt
import netCDF4 as nc
from pyproj import Proj, transform
import numpy as np
import rasterio

# Step 1: Read the netCDF file
filename = 'ice_thickness_2021.nc'
dataset = nc.Dataset(filename)

# Step 2: Extract latitude, longitude, and ice thickness data
lat = dataset.variables['lat'][:]
lon = dataset.variables['lon'][:]
ice_thickness = dataset.variables['sea_ice_thickness'][0,:] # Replace 'ice_thickness' with the correct variable name
ice_thickness = np.nan_to_num(ice_thickness)  # turn nan values to 0

# Step 3: Convert lat/lon (degrees) to Polar Stereographic (meters)
proj_latlon = Proj(proj='latlong',datum='WGS84')
proj_ps = Proj(init='epsg:32661')  # North Polar Stereographic
lon_m, lat_m = transform(proj_latlon, proj_ps, lon.flatten(), lat.flatten())
lon_m = lon_m.reshape(lon.shape)
lat_m = lat_m.reshape(lat.shape)

# Step 4 & 5: Create a grid and assign ice thickness to each point
# Here, I'm assuming that the resolution of your grid is 25 km (as mentioned in the attributes you shared)
resolution = 25000  # 25 km in meters
xmin, ymin, xmax, ymax = np.min(lon_m), np.min(lat_m), np.max(lon_m), np.max(lat_m)

# Create a grid with the above extents and resolution
cols = int(np.ceil((xmax - xmin) / resolution))
rows = int(np.ceil((ymax - ymin) / resolution))

# Create the transform
transform = rasterio.transform.from_origin(xmin, ymin, resolution, resolution)

# Create the destination data array
ice_thickness_grid = np.empty(shape=(rows, cols))

# Fill the destination data array with ice thickness values using reverse indexing
for i in range(len(lat)):
    for j in range(len(lon)):
        x = lon_m[i, j]
        y = lat_m[i, j]
        col, row = ~transform * (x, y)
        col, row = int(col), int(row)
        ice_thickness_grid[row, col] = ice_thickness[i, j]

# Now you have a grid (ice_thickness_grid) in meters with corresponding ice thickness
# This grid can be used as input to the A* algorithm


# testing the grid
plt.figure(figsize=(10,10))
plt.imshow(ice_thickness_grid, origin='lower')
plt.grid(color='white', linestyle='-', linewidth=0.5)
plt.colorbar(label='Ice Thickness')
plt.title('Ice Thickness grid')
plt.show()

# print(ice_thickness_grid)