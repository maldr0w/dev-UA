
import rasterio
import numpy as np
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer
from rasterio.transform import from_origin

## projection

file_name = 'ice_thickness_2021.nc'
ds = xr.open_dataset(file_name)
# print(ds.variables)

ice_thickness = ds['sea_ice_thickness'][0,:].values
ice_thickness = np.nan_to_num(ice_thickness)


# defining coordinate refrence systems
latlon_crs = CRS(proj='latlong', datum='WGS84')  # Lat/Lon projection (degrees)
stereo_crs = CRS('EPSG:32661')  # North Polar Stereographic projection (meters)

# defining transformers for meters and degrees
transformer_m = Transformer.from_crs(latlon_crs, stereo_crs)
transformer_d = Transformer.from_crs(stereo_crs, latlon_crs)

# transforming
lon_m, lat_m = transformer_m.transform(ds['lon'].values, ds['lat'].values)
# lon_d, lat_d = transformer_d.transform(x,y)

# get extent of coordinates
xmin, xmax, ymin, ymax = lon_m.min(), lon_m.max() + 1, lat_m.min(), lat_m.max() + 1
# print(f'xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}')

# defining grid parameters
resolution = 25000  # 25km

n_cols = int(np.ceil((xmax - xmin) / resolution))  # getting the grid dimensions
n_rows = int(np.ceil((ymax - ymin) / resolution))  # while rounding up to nearest integer
# print(n_cols, n_rows) # 534 x 534

# defining transformation that maps pixel coordinates to geographic coordinates
raster_transform = from_origin(xmin, ymax, resolution, resolution)
# print(raster_transform)

# initializing grid with zeros
ice_thickness_grid = np.zeros((n_rows, n_cols))

# populating the grid
# iterating over all the points in the 2D ice_thickness grid
for i in range(ice_thickness.shape[0]):
    for j in range(ice_thickness.shape[1]):

        # extracting corresponding geographic coordinates (in meters) for each point
        x, y = lon_m[i, j], lat_m[i, j]
        
        # applying inverse raster transform, transforming geographic coordinates to pixel coordinates
        row, col = ~raster_transform * (x,y)
        row, col = int(row), int(col)  # converting from float back to integer
        
        try:
            # assigning current ice thickness to correct cell in the grid
            ice_thickness_grid[row, col] = ice_thickness[i, j]
        
        except IndexError:
        
            print(f'Error with i={i}, j={j}, row={row}, col={col}')
            break

# print(ice_thickness_grid)

