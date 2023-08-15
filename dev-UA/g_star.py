
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import pyproj

from make_grid import ice_thickness_grid, transformer_m, transformer_d, raster_transform, ds, transform_point, revert_point, lon_m, lat_m
from global_land_mask import globe

# main.py - UIT - Martin Stave - mvrtinstave@gmail.com
# applying A* algorithm on a grid of the sea ice thickness @ North Pole


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
    path = [current_node]  # initializing with the current_node
    while current_node in came_from:  # set iteration
        current_node = came_from[current_node]  # assign current node to the node it came from
        path.insert(0,current_node)  # insert current node at the front of the path list
    return path
    
def get_neighbors(node, max_x, max_y):
    '''
    returns a list of adjacent nodes to the arg node
        '''
    # neighbors = []  # create empty list
    # x, y = node  # get node coordinates

    # # add node neighbors adding and removing 1 to each parameter
    # neighbors.append((x,y+1))  # up 
    # neighbors.append((x,y-1))  # down 
    # neighbors.append((x+1,y))  # right 
    # neighbors.append((x-1,y))  # left 

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


def find_nearest_water(lat,lon, radius=5):
    for i in range(radius):
        for j in range(radius):
            new_lat = lat + i * LATITUDE_INCREMENT
            new_lon = lon +j * LONGITUDE_INCREMENT
            if not globe.is_land(new_lat, new_lon):
                return new_lat, new_lon


def reduce_path(path):
    reduced_path = [path[0]]
    last_point = None
    next_point = None
    
    for i, point in enumerate(path):
        if point in reduced_path:
            continue
        
        if i < len(path) - 1:
            next_point = path[i+1]
            last_point = path[i-1]
        
        x,y = point
        x_next, y_next = next_point
        x_last, y_last = last_point

        if x_last != x_next and y_last != y_next:
            reduced_path.append(point)
  
    return reduced_path

def draw_path(path, map): # drawing path on a given map
    # map = np.rot90(map,-1)
    # path = np.rot90(path,-1)
    plt.imshow(map, cmap='jet',origin='lower', interpolation='nearest')
    plt.plot(*zip(*path), color='red') # zip fixes this line somehow       
    plt.show()

# input lat lon, coordinates 
# map = metered grid
# convert lat lon to closest node point in lat lon dataset node to meter node
# proceed
def g_star_search(start_coordinate,goal_coordinate,grid):
    ''' main function
    returns the most optimal path from the predefined start node to the goal node
    nodes within the path are chosen based on cost balance
        '''
    
    # defining lat and lon start and end points 
    lat_start, lon_start = start_coordinate
    lat_end, lon_end = goal_coordinate

    
    # check if input coordinates are on land
    if globe.is_land(lat_start, lon_start):
        print('starting point is on land!')

    if globe.is_land(lat_end,lon_end):
        print('ending point is on land!')
    
    # transforming start and end points into pixel coordinates
    lon_start_point, lat_start_point = transform_point(lat_start,lon_start)
    lon_end_point, lat_end_point = transform_point(lat_end,lon_end)   
    
    start = (lon_start_point, lat_start_point)
    goal = (lon_end_point, lat_end_point)

    # print(start)
    # print(goal)
    
    # initializing sets
    open_set = {start}  # open set will hold explorable nodes, set = unordered list
    closed_set = set()  # empty set, for already explored nodes
    came_from = {}  # empty dict, to store parent nodes  
    
    gscore = {start: 0}  # gscore is the cost from start node to each node, start is keyed 0
    fscore = {start: heuristic(start, goal)}  # estimated total cost from start to goal via each node

    # print(fscore)
    
    while open_set:  # while there are available nodes

        # set current node to the node with smalles fscore in open_set
        # the node with the smallest fscore will have the lowest total cost to reach the goal node
        current_node = min(open_set, key=lambda node: fscore[node])

        # print(current_node)
        
        if current_node == goal:  # check wether we are at goal node
            return reconstruct_path(current_node, came_from)

        # when lowest f-score has been found
        open_set.remove(current_node)  # remove the node from set of possible nodes
        closed_set.add(current_node)  # add the node to set of already explored nodes 

        for neighbor in get_neighbors(current_node, grid.shape[0], grid.shape[1]):  # loop over current nodes neighbors
            current_ice = grid[neighbor[0]][neighbor[1]]
            
            if neighbor in closed_set:  # if a neighbor has already been explored we move on to the next neighbor
                continue

            # sea ice thickness specification
            if current_ice > 2:  # ignore nodes with values higher than 2
                # print(current_ice)
                continue
            
            # land specification (based on globe module)
            # lon,lat = revert_point(neighbor[0],neighbor[1])
            # print(lat,lon)
            # if globe.is_land(lat,lon): # check wether a neighbor is on land
                # print('found land!')
                # continue

            # adding gscore (cost) from start to current_node and cost between current node and neighbor
            # this score is tentative as it will change if a better path to the neighbor node is found later in the search
            tentative_gscore = gscore[current_node] + cost_between(current_node, neighbor)

            # check if neighbor has been explored
            # if the tentative score is less than real gscore, new score is more optimal 
            # smaller gscore = closer to start along current path, larger = path cost is larger
            if neighbor not in open_set or tentative_gscore < gscore[neighbor]:
                
                came_from[neighbor] = current_node  # set the current node as parent node
                
                gscore[neighbor] = tentative_gscore  # update gscore
                fscore[neighbor] = tentative_gscore + heuristic(neighbor, goal)  # total cost from start to goal via this path (via neighbor)

                if neighbor not in open_set:  # if this neighbor has not been explored 
                    open_set.add(neighbor)  # add to set of explorable nodes


    return None  # if no path has been found return None     


def pixel_vector_magnitude(dx,dy,resolution):
    '''
    computes distance between two points in a grid
    '''
    if dx == 0:  # if change only in y direction
        return abs(dy) * resolution
    elif dy == 0:  # if change only in x direction
        return abs(dx) * resolution
    else:  # for any other path, diagonal or non-straight line
        return np.sqrt((dx ** 2)+(dy ** 2)) * resolution  # euclidean distance, scaled by resolution

def pixel_length(path):
    '''
    compute total path length based on distance per pixel (resolution)
    '''
    previous = path[0]
    length = 0.0
    # iterate over path, segment by segment
    for segment in path[1:]:
        # changes in x and y direction
        dx = segment[0] - previous[0]
        dy = segment[1] - previous[1]
        # sum length of current segment to total length
        length += pixel_vector_magnitude(dx, dy, resolution)
        previous = segment  # set current point as previous for next iteration
    return length


resolution = 25000

# Defining start and end points
start_point_1 = (72.305, 27.676)
end_point_1 = (67.259, 168.511)

start_point_2 = (71.711,60.062)
end_point_2 = (67.259, 168.511)

start_point_3 = (62.210,57.162)
end_point_3 = (57.349,173.091)

# initializing g_search
path_1 = g_star_search(start_point_1,end_point_1,ice_thickness_grid)
path_2 = g_star_search(start_point_2,end_point_2,ice_thickness_grid)
path_3 = g_star_search(start_point_3,end_point_3,ice_thickness_grid)

# draw_path(path, ice_thickness_grid)

# if path_1 is None:
#     print('path 1 is none')
# else:
#     path_1_length = pixel_length(path_1)
    
# if path_2 is None:
#     print('path 2 is none')
# else: 
#     path_2_length = pixel_length(path_2)

# if path_3 is None:
#     print('path 3 is none')
# else:
#     path_3_length = pixel_length(path_3)


# print(path_1_length)
# print(path_2_length)
# print(path_3_length)

# plotting
plt.figure().set_figwidth(15)
plt.title("")

zoom_amount = (150,400)

plt.subplot(1, 3, 1)
# plt.title(f'start:{start_point_1}, end:{end_point_1}, dist: {path_1_length}', fontsize = 8)
plt.title(f'start:{start_point_1}, end:{end_point_1}', fontsize = 8)
plt.imshow(ice_thickness_grid, cmap='jet',origin='lower', interpolation='nearest')
plt.plot(*zip(*path_1), color='red', label = f'{start_point_1}') # zip fixes this line somehow   plt.title('E = 2')
plt.xlim(zoom_amount)
plt.ylim(zoom_amount)

plt.subplot(1, 3, 2)
plt.title(f'start:{start_point_2}, end:{end_point_2}', fontsize = 8)
plt.imshow(ice_thickness_grid, cmap='jet',origin='lower', interpolation='nearest')
plt.plot(*zip(*path_2), color='red') # zip fixes this line somehow   plt.title('E = 2')
plt.xlim(zoom_amount)
plt.ylim(zoom_amount)

plt.subplot(1, 3, 3)
plt.title(f'start:{start_point_3}, end:{end_point_3}', fontsize = 8)
plt.imshow(ice_thickness_grid, cmap='jet',origin='lower', interpolation='nearest')
plt.plot(*zip(*path_3), color='red') # zip fixes this line somehow   plt.title('E = 2')
plt.xlim(zoom_amount)
plt.ylim(zoom_amount)

# plt.show()



for values in ice_thickness_grid:
    print(values)






