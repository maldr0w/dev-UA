
import xarray as xr
import numpy as np
from pyproj import CRS, Transformer
from rasterio.transform import from_origin

# extract coordinates
def get_coordinates(nc_file):

    ds = xr.open_dataset(nc_file)

    # extracting variables
    ice_thickness_nan = ds['sea_ice_thickness'][0,:].values
    ice_thickness = np.nan_to_num(ice_thickness_nan)  # set nan values to 0

    latitude, longitude = ds['lat'].values, ds['lon'].values

    ds.close()  # closing dataset

    return ice_thickness_nan, ice_thickness, latitude, longitude


def transform_to_meters(latitude, longitude):
    ''' Transforms latitude and longitude from degrees to pixel coordinates'''
    
    # defining coordinate refrence systems
    geo_crs = CRS(proj='latlong', datum='WGS84')  # geographical
    stereo_crs = CRS('EPSG:32661')  # North Polar Stereographic

    # initializing degree and meter conversion
    degrees_to_meters = Transformer.from_crs(geo_crs, stereo_crs)
    meters_to_degrees = Transformer.from_crs(stereo_crs, geo_crs)

    # transforming latitude and longitude from degrees to meters
    longitude_meter, latitude_meter = degrees_to_meters.transform(longitude, latitude)

    return longitude_meter, latitude_meter    

def transform_to_degrees(longitude_meter, latitude_meter):
    ''' Transforms longitude and latitude back to degrees'''

    pass

def reverse_array(longitude,latitude):
    # reversing the arrays, so that the mapping is correctly orientated
    flip_up_to_down = np.flipud(longitude)
    flip_left_to_right = np.fliplr(latitude)
    
    return flip_up_to_down, flip_left_to_right

def transform_to_pixels(x, y, x_min, y_max, grid_resolution):    
    # initializing raster transform which transform geographical coordinates to pixel coordinates
    raster_transform = from_origin(x_min, y_max, grid_resolution, grid_resolution)

    # inverse raster transforming 
    col, row = ~raster_transform * (x,y)  # maps geographic coordinates to pixel coordinates
    col, row = int(col), int(row)
    
    return col, row



def transform_point(FILE_NAME,GRID_RESOLUTION,latitude_point,longitude_point, latitude_meter, longitude_meter, x_min, y_max):

    # transforming point from degrees to meters
    x_meter, y_meter = transform_to_meters(latitude_point,longitude_point)

    # computing difference array between grid points and transformed point
    diff_array = np.sqrt((latitude_meter - y_meter) ** 2 + (longitude_meter - x_meter) ** 2)

    index = np.unravel_index(np.argmin(diff_array, axis=None), diff_array.shape)

    ds = xr.open_dataset(FILE_NAME)

    nearest_latitude = ds['lat'].values[index]
    nearest_longitude = ds['lon'].values[index]
    
    ds.close()
    
    longitude_point_meter, latitude_point_meter = transform_to_meters(nearest_latitude, nearest_longitude)
    longitude_point_pixel, latitude_point_pixel = transform_to_pixels(longitude_point_meter, latitude_point_meter, x_min, y_max, GRID_RESOLUTION)

    return longitude_point_pixel, latitude_point_pixel


def convert_to_grid(latitude, longitude, ice_thickness,  grid_resolution):
    ''' converts ice thickness coordinates into a pixel grid '''

    # defining longitude and latitude in meters
    longitude_meter, latitude_meter = transform_to_meters(latitude, longitude)

    # reversing order of values in arrays
    longitude_meter, latitude_meter = reverse_array(longitude_meter, latitude_meter)

    # defining grid parameters
    x_min, x_max = longitude_meter.min(), longitude_meter.max() + 1
    y_min, y_max = latitude_meter.min(), latitude_meter.max() + 1

    # number of cols and rows
    n_cols = int(np.ceil((x_max - x_min) / grid_resolution))  # 534
    n_rows = int(np.ceil((y_max - y_min) / grid_resolution))  # 534

    # creating ice thickness pixel grid
    ice_thickness_grid = np.zeros((n_rows, n_cols))  # array uses row,col convention

    # iterating ove all points in 2D ice thickness grid
    # for idx in range(len(latitude)):
    #     x, y = longitude_meter[idx], latitude_meter[idx]
    #     col, row = ~raster_transform * (x,y)
    #     col, row = int(col), int(row)

    #     ice_thickness_grid[row,col] = ice_thickness[idx]
    
    # iterating over all points in the 2D ice thickness grid
    for i in range(ice_thickness.shape[0]):
        for j in range(ice_thickness.shape[1]):

            x, y = longitude_meter[i,j], latitude_meter[i,j]  # extracting corresp. geographic coordinate (in meters) for each point
            col, row = transform_to_pixels(x,y,x_min, y_max, grid_resolution)
            
            # assigning current ice thickness to corresponding cell in the grid
            ice_thickness_grid[col,row] = ice_thickness[i,j]  # np.array, (row,col) convention
            # ice_thickness_grid = np.flipud(ice_thickness_grid)
            # ice_thickness_grid = np.rot90(ice_thickness_grid,-1)

            
    return ice_thickness_grid, longitude_meter, latitude_meter, x_min, y_max


# deccf transform_point()
