
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy as cp


## Extracting dataset variables
FILE_NAME = 'ice_thickness_2021.nc'
GRID_RESOLUTION = 25_000 # m

'''
input: netCDF datafile
output: lat/lon and sea ice thickness data-array    
    '''
ds = xr.open_dataset(FILE_NAME)  # opens dataset file
# extracting variables by index
latitude, longitude = ds['lat'].values, ds['lon'].values
sea_ice_thickness_with_nan = ds['sea_ice_thickness'][0,:].values  # nan values will be useful in plotting
sea_ice_thickness = np.nan_to_num(sea_ice_thickness_with_nan)  # replace nan values with 0
# ds.close()  # closes dataset to save resources

latitude, longitude, sea_ice_thickness, sea_ice_thickness_with_nan

# latitude, longitude, sea_ice_thickness, sea_ice_thickness_with_nan = extract_data(FILE_NAME)
lat_min, lon_min, lat_max, lon_max = latitude.min(), longitude.min(), latitude.max(), longitude.max()

def scatter_plot():
    fig, ax = plt.subplots(1, 1, figsize=(5,5),subplot_kw = {'projection':cp.crs.NorthPolarStereo()})
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


from shapely.geometry import Point
from global_land_mask import globe

## Finding which coordinates are on land

# combining latitude and longitude coordinates
points = np.array(list(zip(latitude.flatten(), longitude.flatten())))

# add points from the dataset that are on land to a list with lat/lon order
land_mask = [point for point in points if globe.is_land(point[0],point[1])]
# print(f'lat={point.x}, lon={point.y} is on land!')
# print(land_mask[0])


## Projection
from pyproj import CRS, Transformer
from rasterio.transform import from_origin

# defining coordinate refrence systems
latlon_crs = CRS(proj='latlong', datum='WGS84')  # lat/lon projection in degrees
stereographic_crs = CRS('EPSG:32661')  # North Polar Stereographic projection in meters

# initializing transformers for degrees and meter convertion
transformer_m = Transformer.from_crs(latlon_crs, stereographic_crs)  # degrees to meters
transformer_d = Transformer.from_crs(stereographic_crs, latlon_crs)  # meters to degrees

# transforming longitude and latitude coordinates into meters
longitude_m, latitude_m = transformer_m.transform(longitude, latitude)

# transforming land_mask coordinates to meters
land_mask_m = [transformer_m.transform(coord[1],coord[0]) for coord in land_mask]
# land_x, land_y = zip(*land_mask_m)

# get extent of coordinates and defining grid parameters
xmin, ymin, xmax, ymax = longitude_m.min(), latitude_m.min(), longitude_m.max() + 1, latitude_m.max() + 1
n_cols, n_rows = int(np.ceil((xmax - xmin) / GRID_RESOLUTION)), int(np.ceil((ymax - ymin) / GRID_RESOLUTION))
    
# initialize raster transform which maps pixel coordinates to geographic coordinates
raster_transform = from_origin(xmin, ymax, GRID_RESOLUTION, GRID_RESOLUTION)

# initializing grids with zeros
sea_ice_thickness_grid = np.zeros((n_rows, n_cols))  # np.array uses (row,col) convention
land_grid = np.zeros((n_rows,n_cols))

# iterating over all points in the 2D ice thickness grid
for i in range(sea_ice_thickness.shape[0]):
    for j in range(sea_ice_thickness.shape[1]):
        
        x, y = longitude_m[i,j], latitude_m[i,j]  # extracting corresp. geographic coordinate (in meters) for each point
        col, row = ~raster_transform * (x,y)  # inverse raster transform: transforming geographic coordinates to pixel coordinates
        col, row = int(col), int(row)

        # assigning current ice thickness to corresponding cell in the grid
        sea_ice_thickness_grid[row,col] = sea_ice_thickness[i,j]  # np.array, (row,col) convention

# raster transforming land mask
# for x, y in land_mask_m:
    # col, row = ~raster_transform * (x, y)
    # col, row = int(col), int(row)

    # land_grid[row, col] = 1
    # sea_ice_thickness_grid[row,col] = 4



plt.figure(figsize=(8,8))
plt.imshow(sea_ice_thickness_grid, cmap='jet', origin='lower')
# plt.imshow(land_grid)
# plt.imshow(land_grid, origin='lower')
plt.colorbar(label='Ice Thickness')
plt.title('Sea Ice Thickness')


def transform_point(lat_point, lon_point):
    '''
    transforms lat,lon coordinates to pixel coordinates
    and extracts ice thickness at give coordinate    
        '''
    
    # transforming point from degree to meters
    lon_point_m, lat_point_m = transformer_m.transform(lon_point,lat_point)

    # tranforming again into pixel coordinates
    lon_point_p, lat_point_p = ~raster_transform * (lon_point_m, lat_point_m)
    lon_point_p, lat_point_p = int(lon_point_p), int(lat_point_p)  # set as closest integer

    # Now, snap these coordinates to the nearest point in the sea_ice_thickness_grid.
    # Assuming sea_ice_thickness_grid has the shape (num_rows, num_cols)
    num_rows, num_cols = sea_ice_thickness_grid.shape
    row_coords, col_coords = np.mgrid[0:num_rows, 0:num_cols]

    diff_array_p = np.sqrt((row_coords - lat_point_p)**2 + (col_coords - lon_point_p)**2)
    nearest_pixel_index = np.unravel_index(np.argmin(diff_array_p, axis=None), diff_array_p.shape)

    point_row, point_col = nearest_pixel_index
    
    return point_row, point_col

def revert_point(lon_pixel,lat_pixel):
    '''
    transform pixel coordinates back to lat/lon coordinates    
        '''
    # convert pixel coordinates back to meters
    lon_m, lat_m = raster_transform * (lon_pixel, lat_pixel)

    # convert meters to degrees
    lon, lat = transformer_d.transform(lon_m, lat_m)

    return lon, lat





# Initializing A* search algorithm

def heuristic(node,goal):
    '''
    heuristic estimate (Manhatten distance)
    estimate of distance between specified node and goal node
        '''
    x1, y1 = node
    x2, y2 = goal
    estimated_distance = abs(x2 - x1) + abs(y2 - y1)    
    return estimated_distance


def reconstruct_path(current_node,came_from): 
    '''
    returns reconstructed path as a list of nodes from start node to goal node
    by iterating over visited nodes from the came_from set
        '''
    path = [current_node]  # initializing with current_node
    while current_node in came_from:  # iterating through came from set
        current_node = came_from[current_node]  # assign current node to the node it came from
        path.insert(0,current_node)  # insert current node at the front of the path list
    return path


def get_neighbors(node,max_x,max_y):
    '''
    returns a list of adjacent nodes to the arg node
        '''
    x,y = node
    moves = [(0,1), (0,-1), (1,0), (-1,0)]
    neighbors = [(x + dx, y + dy) for dx, dy in moves if 0 <= x + dx < max_x and 0 <= y + dy < max_y]

    return neighbors


def cost_between(node1,node2):
    '''
    returns more accurate cost estimate between nodes 
    based on the euclidean distance between nodes
        '''
    cost = 0
    x1, y1 = node1
    x2, y2 = node2
    cost = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5  # euclidean distance
    
    return cost


def A_star_search_algorithm(start_coordinate, end_coordinate, grid):
    '''
    returns most optimal path between star_coordinate and end_coordinate
    path is chosen based on cost
        '''
    # extracting lat, lon coordinates
    lat_start, lon_start = start_coordinate
    lat_end, lon_end = end_coordinate

    # transforming lat/lon coordinates to pixel coordinates
    lon_start_point, lat_start_point = transform_point(lat_start, lon_start)  # considering lon as x and lat as y
    lon_end_point, lat_end_point = transform_point(lat_end, lon_end)
    
    start = (lon_start_point, lat_start_point)
    goal = (lon_end_point, lat_end_point)

    # initializing sets for exploring and disgarding nodes
    open_set = {start}  # will hold explorable nodes in a unordered list (set)
    closed_set = set()  # will hold previously explored nodes, empty on init
    came_from = {}  # will hold parent nodes, empty on init
    
    # defining scoreing
    g_score = {start:0}  # cost from start node to each node
    f_score = {start:heuristic(start, goal)}  # estimated total cost from start node to goal node via each node

    # iterating over available nodes
    while open_set:

        # set current node to the node in the open set with the smallest f score
        current_node = min(open_set, key=lambda node: f_score[node])  # smalles f score = lowest cost to reach goal node

        if current_node == goal:
            return reconstruct_path(current_node, came_from)

        # lowest f score has been found
        open_set.remove(current_node)  # remove current node from explorable nodes
        closed_set.add(current_node)  # add current node to explored nodes

        # iterating through current node's neighbors
        for neighbor in get_neighbors(current_node, grid.shape[0], grid.shape[1]):

            # extracting sea ice thickness at current neighbor
            neighbor_ice_thickness = grid[neighbor[0]][neighbor[1]]
        
            
            # extracting land mask at current neigbor
            # neighbor_land = land_grid[neighbor[0]][neighbor[1]]
            neighbor_land = grid[neighbor[0]][neighbor[1]]
            
            # continuing if neighbor has already been explored
            if neighbor in closed_set:
                continue

            # continuing if sea ice thickness of current neighbor is too thick
            if neighbor_ice_thickness > 2:
                continue

            # continuing if current neighbor is on land
            # if neighbor_land == 4:
                # print('on land!')
                # continue

            # adding g score as cost from start node to current node and cost between current node and neighbor
            tentative_g_score = g_score[current_node] + cost_between(current_node, neighbor)  # changes if a smaller cost path is found

            # if neighbor has been explored or added g score is more optimal than current most optimal path
            if neighbor not in open_set or tentative_g_score < g_score[neighbor]:

                # set current node as parent node
                came_from[neighbor] = current_node

                # updating g score
                g_score[neighbor] = tentative_g_score 

                # total cost from start node to goal node via this neighbor
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)

                # checking if neighbor has been explored
                if neighbor not in open_set:
                    open_set.add(neighbor)  # add neighbor to set of explorable nodes

    # if no path has been found 
    return None

# defining start and end coordinates (lat,lon)
start_coordinate = (68.914,39.027)
end_coordinate = (63.687, 147.118)

start_lon_p,start_lat_p = transform_point(start_coordinate[0], start_coordinate[1])
end_lon_p, end_lat_p = transform_point(end_coordinate[0], end_coordinate[1])

start_lon, start_lat = revert_point(start_lon_p, start_lat_p)

# print(start_lon, start_lat)




path = A_star_search_algorithm(start_coordinate, end_coordinate, sea_ice_thickness_grid)
zoom_amount = (150,400)

# plt.title(f'start:{start_point_1}, end:{end_point_1}, dist: {path_1_length}', fontsize = 8)
# plt.title(f'start:{start_point_1}, end:{end_point_1}', fontsize = 8)
# plt.plot(*zip(*path), color='red', label = f'{start_coordinate}') # zip fixes this line somehow   plt.title('E = 2')
plt.imshow(sea_ice_thickness_grid, cmap='jet',origin='lower', interpolation='nearest')
# plt.imshow(sea_ice_thickness_grid, cmap='jet', interpolation='nearest')

plt.plot(start_lon_p, start_lat_p,'ro')
plt.plot(end_lon_p, end_lat_p,'ro')

plt.xlim(zoom_amount)
plt.ylim(zoom_amount)
plt.show()






