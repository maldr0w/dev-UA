
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy as cp


## Extracting dataset variables
FILE_NAME = 'ice_thickness_2021.nc'
RESOLUTION = 25000 # m

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

lat,lon = latitude, longitude


from shapely.geometry import Point
from global_land_mask import globe

## Finding which coordinates are on land

# combining latitude and longitude coordinates
# points = [Point(lat,lon) for lat, lon in zip(np.ravel(lat),np.ravel(lon))]

points = np.array(list(zip(latitude.flatten(), longitude.flatten())))
# print(points[0][0])


# add points from the dataset that are on land to a list
# land_mask = [point for point in points if globe.is_land(point.x,point.y)]
# print(f'lat={point.x}, lon={point.y} is on land!')


## Projection
from pyproj import CRS, Transformer
from rasterio.transform import from_origin
# defining coordinate refrence systems
latlon_crs = CRS(proj='latlong', datum='WGS84')  # lat/lon projection in degrees
stereographic_crs = CRS('EPSG:32661')  # North Polar Stereographic projection in meters

# initializing transformers for degrees and meter convertion
transformer_m = Transformer.from_crs(latlon_crs, stereographic_crs)  # degrees to meters
transformer_d = Transformer.from_crs(stereographic_crs, latlon_crs)  # meters to degrees

# transforming
lon_m, lat_m = transformer_m.transform(lon, lat)

land_mask_m = []
for point in land_mask:
    transformer_m.transform(point.x,point.y)
    land_mask_m.append(point)
    
print(land_mask_m[0]) 


# get extent of coordinates and defining grid parameters
xmin, ymin, xmax, ymax = lon_m.min(), lat_m.min(), lon_m.max() + 1, lat_m.max() + 1
n_cols, n_rows = int(np.ceil((xmax - xmin) / RESOLUTION)), int(np.ceil((ymax - ymin) / RESOLUTION))
    
# initialize raster transform which maps pixel coordinates to geographic coordinates
raster_transform = from_origin(xmin, ymax, RESOLUTION, RESOLUTION)

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
        land_mask_grid[row,col] = land_mask[i,j]

