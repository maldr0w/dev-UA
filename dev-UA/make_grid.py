
import rasterio
import numpy as np
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer
from rasterio.transform import from_origin



def convert_to_grid(nc_file, resolution):
    '''
    inputs a netCDF datafile
    converts the lat/lon data points from degrees to meters in a pixel grid
        '''
    
    ds = xr.open_dataset(nc_file)  # opening dataset

    # defining variables
    ice_thickness = ds['sea_ice_thickness'][0,:].values
    ice_thickness = np.nan_to_num(ice_thickness)  # set nan values to zero
    
    lon, lat = ds['lon'].values, ds['lat'].values

    # defining coordinate reference system
    latlon_crs = CRS(proj='latlong', datum='WGS84')  # lat/lon projection in degrees
    stereographic_crs = CRS('EPSG:32661')  # North Polar Stereographic projection in meters

    # initializing tranformers for both degrees and meter convertion
    transformer_m = Transformer.from_crs(latlon_crs, stereographic_crs)  # degrees to meters
    transformer_d = Transformer.from_crs(stereographic_crs, latlon_crs)  # meters to degrees

    # transforming
    lon_m, lat_m = transformer_m.transform(lon, lat)

    # get extent of coordinates and defining grid parameters
    xmin, ymin, xmax, ymax = lon_m.min(), lat_m.min(), lon_m.max() + 1, lat_m.max() + 1
    n_cols, n_rows = int(np.ceil((xmax - xmin) / resolution)), int(np.ceil((ymax - ymin) / resolution))
        
    # initialize raster transform which maps pixel coordinates to geographic coordinates
    raster_transform = from_origin(xmin, ymax, resolution, resolution)

    # initializing grid with zeros
    ice_thickness_grid = np.zeros((n_rows, n_cols))  # np.array uses (row,col) convention

    # iterating over all points in the 2D ice thickness grid
    for i in range(ice_thickness.shape[0]):
        for j in range(ice_thickness.shape[1]):

            x, y = lon_m[i,j], lat_m[i,j]  # extracting corresp. geographic coordinate (in meters) for each point
            col, row = ~raster_transform * (x,y)  # inverse raster transform: transforming geographic coordinates to pixel coordinates
            col, row = int(col), int(row)

            # assigning current ice thickness to corresponding cell in the grid
            ice_thickness_grid[row,col] = ice_thickness[i,j]  # np.array, (row,col) convention

    return ice_thickness_grid, transformer_m, transformer_d, lon_m, lat_m, raster_transform, ds
    

def transform_point(lat_point, lon_point):
    '''
    transforms lat,lon coordinates to pixel coordinates
    and extracts ice thickness at give coordinate    
        '''
    
    # transforming point from degree to meters
    lon_point_m, lat_point_m = transformer_m.transform(lon_point,lat_point)

    # computing difference array between grid points and transformed point
    diff_array_m = np.sqrt((lat_m - lat_point_m)**2 + (lon_m - lon_point_m)**2)  # heuristic distance
    # extracting index of grid point with lowest difference / closest gridpoint to the transformed point
    index_m = np.unravel_index(np.argmin(diff_array_m,axis=None), diff_array_m.shape)

    # finding the equivalent coordinates in the dataset with this index 
    nearest_lon = ds['lon'].values[index_m]
    nearest_lat = ds['lat'].values[index_m]

    # transforming the nearest coordinates in the dataset to meters
    lon_point_m, lat_point_m = transformer_m.transform(nearest_lon, nearest_lat)

    # tranforming again into pixel coordinates
    lon_point_p, lat_point_p = ~raster_transform * (lon_point_m, lat_point_m)
    lon_point_p, lat_point_p = int(lon_point_p), int(lat_point_p)  # set as closest integer

    # extracting ice thickness at this point in grid
    ice_thickness_at_point = ice_thickness_grid[lat_point_p, lon_point_p]  # np.array follow row,col index order
    
    return lon_point_p, lat_point_p


def revert_point(lon_pixel,lat_pixel):
    '''
    transform pixel coordinates back to lat/lon coordinates    
        '''
    # convert pixel coordinates back to meters
    lon_m, lat_m = raster_transform * (lon_pixel, lat_pixel)

    # convert meters to degrees
    lon, lat = transformer_d.transform(lon_m, lat_m)

    return lat, lon



file_name = 'ice_thickness_2021.nc'
resolution = 25000  # 25km

# extracting parameters
ice_thickness_grid, transformer_m, transformer_d, lon_m, lat_m, raster_transform, ds = convert_to_grid(file_name, resolution)

# testing point mapping
# lat_point = 74.877
# lon_point = 9.359
# lon_point_p, lat_point_p = transform_point(lat_point, lon_point)
# print(lon_point_p, lat_point_p)

# lon, lat = revert_point(lon_point_p, lat_point_p)
# print(lon,lat)


# start_point = (72.305, 27.676)
# lat_end_p = 67.259, 
# lon_end_p = 168.511

# lon_end_p, lat_end_p = transform_point(lat_end_p, lon_end_p)
# plotting the grid

# def plot_grid(grid, point_x, point_y):
#     plt.figure(figsize=(10,10))    
#     plt.imshow(grid,cmap='jet',origin='lower')
#     plt.colorbar(label='Ice Thickness')
#     plt.title('Ice Thickness grid')

#     plt.xlim(100,450)
#     plt.ylim(100,450)
    
#     plt.plot(point_x,point_y, 'ro')
#     plt.plot(lon_end_p, lat_end_p, 'ro')
      
#     plt.show()

# plot_grid(ice_thickness_grid, lon_point_p, lat_point_p)



