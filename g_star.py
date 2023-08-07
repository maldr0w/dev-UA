
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import pyproj

from make_grid import ice_thickness_grid, transformer_m, transformer_d, raster_transform, ds, transform_point
from global_land_mask import globe

# g_star.py - UIT - Martin Stave - mvrtinstave@gmail.com
# applying A* algorithm on grid of the Arctic


'''
plan:
create an algorithm for pathing in the arctic
get array of coordinates, figure out which is the best path A*
- use ice index to navigate
- globle land mask to navigate
   
    '''


def heuristic(node,goal):
    '''
    heuristic estimate (Manhatten distance)
    estimate of distance between specified node and goal node
        '''
    x1, y1 = node
    x2, y2 = goal
    estimated_distance = abs(x2 - x1) + abs(y2 - y1)    
    return estimated_distance


def reconstruct_path(current_node, came_from): 
    '''
    returns reconstructed path as a list of nodes from start node to goal node
    by iterating over visited nodes from the came_from set
        '''
    path = [current_node]  # initializing with the current_node
    while current_node in came_from:  # set iteration
        current_node = came_from[current_node]  # assign current node to the node it came from
        path.insert(0,current_node)  # insert current node at the front of the path list
    return path
    
    
def get_neighbors(node):
    '''
    returns a list of adjacent nodes to the arg node
        '''
    neighbors = []  # create empty list
    x, y = node  # get node coordinates

    # add node neighbors adding and removing 1 to each parameter
    neighbors.append((x,y+1))  # up 
    neighbors.append((x,y-1))  # down 
    neighbors.append((x+1,y))  # right 
    neighbors.append((x-1,y))  # left 
    
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


# input lat lon, coordinates 
# map = metered grid
# convert lat lon to closest node point in lat lon dataset node to meter node
# proceed
def g_star_search(start, goal, map):
    ''' main function
    returns the most optimal path from the predefined start node to the goal node
    nodes within the path are chosen based on cost balance
        '''
    
    



    # initializing sets
    open_set = {start}  # open set will hold explorable nodes, set = unordered list
    closed_set = set()  # empty set, for already explored nodes
    came_from = {}  # empty dict, to store parent nodes  
    
    gscore = {start: 0}  # gscore is the cost from start node to each node, start is keyed 0
    fscore = {start: heuristic(start, goal)} # estimated total cost from start to goal via each node

    while open_set:  # while there are available nodes

        # set current node to the node with smalles fscore in open_set
        # the node with the smallest fscore will have the lowest total cost to reach the goal node
        current_node = min(open_set, key=lambda node: fscore[node])
        
        if current_node == goal:  # check wether we are at goal node
            return reconstruct_path(current_node, came_from)

        # when lowest f-score has been found
        open_set.remove(current_node)  # remove the node from set of possible nodes
        closed_set.add(current_node)  # add the node to set of already explored nodes 

        for neighbor in get_neighbors(current_node):  # loop over current nodes neighbors
            if neighbor in closed_set:  # if a neighbor has already been explored we move on to the next neighbor
                continue

            # sea ice thickness specification
            if map[neighbor[0]][neighbor[1]] > 2:  # ignore nodes with values higher than 2
                continue

            # land specification (based on globe module)
            # if globe.is_land(neighbor): # check wether a neighbor is on land
            #     continue

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

























