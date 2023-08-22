
# package imports
import numpy as np
import xarray as xr

# function imports
from make_grid_2 import get_coordinates, convert_to_grid, transform_to_meters, transform_to_pixels, reverse_array, transform_point

from make_plot_2 import scatter_plot, pixel_plot

FILE_NAME = 'ice_thickness_2021.nc'
GRID_RESOLUTION = 25_000  # meters


# collecting variables
ice_thickness_nan, ice_thickness, latitude, longitude = get_coordinates(FILE_NAME)

ice_thickness_grid, longitude_meter, latitude_meter, x_min, y_max = convert_to_grid(latitude,longitude,ice_thickness,GRID_RESOLUTION)

# print(f'ice thickness shape: {ice_thickness.shape}')
# print(f'latitude shape = {latitude.shape}, longitude shape = {longitude.shape}')


# start_lat, start_lon = 72.849, 30.409
start_lat, start_lon = 68.310, 18.489
end_lat, end_lon = 64.549, 169.574


# def transform_point(latitude_point,longitude_point):

#     # transforming point from degrees to meters
#     x_meter, y_meter = transform_to_meters(latitude_point,longitude_point)

#     # computing difference array between grid points and transformed point
#     diff_array = np.sqrt((latitude_meter - y_meter) ** 2 + (longitude_meter - x_meter) ** 2)

#     index = np.unravel_index(np.argmin(diff_array, axis=None), diff_array.shape)

#     ds = xr.open_dataset(FILE_NAME)

#     nearest_latitude = ds['lat'].values[index]
#     nearest_longitude = ds['lon'].values[index]
    
#     ds.close()
    
#     longitude_point_meter, latitude_point_meter = transform_to_meters(nearest_latitude, nearest_longitude)
#     longitude_point_pixel, latitude_point_pixel = tranform_to_pixels(longitude_point_meter, latitude_point_meter, x_min, y_max, GRID_RESOLUTION)

#     return longitud_point_pixel, latitude_point_pixel

start_x_pixel, start_y_pixel = transform_point(FILE_NAME,GRID_RESOLUTION,start_lat, start_lon,latitude_meter, longitude_meter,x_min, y_max)

# print(start_x_pixel, start_y_pixel)
start_point = (start_x_pixel, start_y_pixel)

# def transform_point(lat,lon):
#     x_meter, y_meter = transform_to_meters(lat,lon)
#     x_pixel, y_pixel = transform_to_pixels(x_meter, y_meter, x_min, y_max, GRID_RESOLUTION)
# # start_pixel_x,start_pixel_y = reverse_array(start_pixel_x, start_pixel_y)
#     return x_pixel, y_pixel

# start_x_pixel, start_y_pixel = transform_point(start_lat,start_lon)
# end_x_pixel, end_y_pixel = transform_point(end_lat, end_lon)

# def closest_grid_pixel(pixel, grid):
#     x_indices, y_indices = np.indices(grid.shape)

#     x_diff = (x_indices - pixel[0]) ** 2
#     y_diff = (y_indices - pixel[1]) ** 2

#     total_diff = x_diff * y_diff

#     min_index = np.unravel_index(total_diff.argmin(),total_diff.shape)

#     return min_index

# start_pixel = (start_x_pixel, start_y_pixel)
# start_point = closest_grid_pixel(start_pixel, ice_thickness_grid)
# print(x)
# start_point = (flip_point(start_pixel_x,start_pixel_y))


# grid_height, grid_width = ice_thickness_grid.shape[0], ice_thickness_grid.shape[1]

# def flip_point(x,y):
    # flipped_x = grid_width - x - 1
    # flipped_y = grid_height - y - 1
    # return flipped_x, flipped_y

# start_point = (flip_point(start_x_pixel,start_y_pixel))
# end_point = (flip_point(end_x_pixel,end_y_pixel))

# plotting:
# scatter_plot(latitude,longitude, ice_thickness_nan)
# pixel_plot(ice_thickness_grid)
pixel_plot(ice_thickness_grid, start_point)
# pixel_plot(ice_thickness_grid, end_point)







# latitude_meter = convert_to_grid(longitude,latitude,ice_thickness,GRID_RESOLUTION)
