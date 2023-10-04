import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import utils



utils.print_entrypoint(__name__, __file__)
import graph_creation
import ship_class
import fuel_class
import data_extraction

coordinate_data = data_extraction.dataset['sea_ice_thickness'].coords
latitude_data, longitude_data = coordinate_data['lat'].to_numpy(), coordinate_data['lon'].to_numpy()
mean_ice_thickness = np.nanmean(data_extraction.mapdata) # (1 * np.nanstd(only_values_ice_thickness))
HEURISTIC_FUEL = fuel_class.fuel_list[2]
HEURISTIC_THICKNESS = 0.0

def find_closest_index(coordinate):
    '''
    Finds index in the dataset corresponding to the
    coordinate closest to the provided coordinate 
       '''
    latitude_difference = np.apply_along_axis(lambda x: abs(coordinate[0] - x), 0, latitude_data)
    longitude_difference = np.apply_along_axis(lambda x: abs(coordinate[1] - x), 0, longitude_data)
    arr_difference = np.apply_along_axis(lambda x: x[0] + x[1], 2, np.dstack((latitude_difference, longitude_difference)))

    y_idx, x_idx = np.unravel_index(indices=np.argmin(arr_difference), shape=np.shape(arr_difference))
    return y_idx, x_idx

def find_closest_coordinate(coordinate):
    idx = find_closest_index(coordinate)
    return latitude_data[idx], longitude_data[idx]
# Initializing A* search algorithm


# mean_ice_thickness = np.mean(dist_corr.mapdata,where=np.invert(np.isnan(dist_corr.mapdata)))
from scipy import stats
# mean_ice_thickness = np.std(np.extract(np.invert(np.logical_or(np.isnan(dist_corr.mapdata),np.equal(dist_corr.mapdata, 0.))), dist_corr.mapdata))
ICE_THRESHOLD = 0.6
# only_values_ice_thickness = np.extract(np.invert(np.logical_or(np.isnan(dist_corr.mapdata), np.less(dist_corr.mapdata, ICE_THRESHOLD))), dist_corr.mapdata)

def great_circle(lat1, lon1, lat2, lon2):
    # a = np.deg2rad(lat1)
    # b = np.deg2rad(lat2)
    # x = np.deg2rad(lon1)
    # y = np.deg2rad(lon2)
    radius = 6_371_000
    the1, phi1 = np.deg2rad(lat1), np.deg2rad(lon1)
    the2, phi2 = np.deg2rad(lat2), np.deg2rad(lon2)
    a = np.sin((the2 - the1) / 2.0) ** 2.0
    b = np.cos(the1) * np.cos(the2) * (np.sin((phi2 - phi1) / 2.0) ** 2.0) 
    return (2 * radius) * np.arcsin(np.sqrt(a + b))
    # c = abs(x - y)
    # return 112320 * np.rad2deg(
    #     np.arccos(
    #         (np.cos(a) * np.cos(b) * np.cos(c))
    #         + (np.sin(a) * np.sin(b))
    #     )
    # )
# from enum import Enum
# class Optimization(Enum):
#     DISTANCE = 1
#     CONSUMPTION = 2

def cost(node, neighbor, ship, trip_fuel, unit_rate):
    '''
    cost function [g(n)]

    Using the same calculation as the heuristic, 
    this function returns the actual cost, using the average of the
    node thickness and the neighbor thickness, as well as a specified fuel
    '''
    x1, y1 = node
    x2, y2 = neighbor

    lat1 = latitude_data[y1, x1]
    lat2 = latitude_data[y2, x2]

    lon1 = longitude_data[y1, x1]
    lon2 = longitude_data[y2, x2]

    # estimated_distance = great_circle(lat1, lon1, lat2, lon2)
    estimated_thickness = 0.5 * (data_extraction.mapdata[y1, x1] + data_extraction.mapdata[y2, x2])
    estimated_distance = utils.unit_distance * np.sqrt((abs(x1 - x2) ** 2) + (abs(y1 - y2) ** 2)) 

    return ship.get_costs(trip_fuel, ship.v_limit(estimated_thickness), estimated_distance, estimated_thickness)
    # return (1 + estimated_thickness) * estimated_distance
    # return estimated_distance
    
# Methanol as comparison, gives the cheapest overall cost
def get_heuristic_unit_rate(ship, fuel_type, heuristic_thickness=0.0):
    # return ship.get_trip_consumption(fuel_type, ship.v_limit(HEURISTIC_THICKNESS), thickness=HEURISTIC_THICKNESS)
    return ship.get_costs(fuel_type, ship.v_limit(heuristic_thickness) / 2, utils.unit_distance, heuristic_thickness)

def heuristic(node, goal, ship, fuel_type, unit_rate):
    '''
    heuristic estimate (Great circle distance) [h(n)]
    estimate of distance between specified node and goal node
    
    Using methanol as a heuristic fuel due to it being found to be the cheapest,
    and assuming no ice during the journey, an optimistic estimate is given    
        '''
            
    x1, y1 = node
    x2, y2 = goal

    # lat1 = latitude_data[y1, x1]
    # lat2 = latitude_data[y2, x2]

    # lon1 = longitude_data[y1, x1]
    # lon2 = longitude_data[y2, x2]
    
    estimated_units_covered = max(abs(x1 - x2), abs(y1 - y2))
    # estimated_distance = great_circle(lat1, lon1, lat2, lon2)
    estimated_distance = utils.unit_distance * max(abs(x1 - x2), abs(y1 - y2))
    # estimated_distance = np.sqrt((abs(x1 - x2) ** 2) + (abs(y1 - y2) ** 2))
    # estimated_distance = np.sqrt((abs(lat1 - lat2) ** 2) + (abs(lon1 - lon2) ** 2))
    # return estimated_distance * 25000.
    # weight = heuristic_unit_rate * (estimated_distance / 25000.0)
    # weight = heuristic_unit_rate * estimated_distance
    # return heuristic_unit_rate * (estimated_distance / 25000.0)
    # return estimated_distance * (1.0 / (np.e ** mean_ice_thickness)) 
    # return estimated_distance
    return ship_class.HEURISTIC_BASAL_RATE * estimated_units_covered
    return ship.get_costs(fuel_type, ship.v_limit(0.0), estimated_distance, 0.0)
    # return 25000.0 * estimated_distance

    # return ship.get_costs(fuel_type, ship.v_limit(HEURISTIC_THICKNESS), estimated_distance, HEURISTIC_THICKNESS) 

def reconstruct_path(current_node,came_from): 
    '''
    returns reconstructed path as a list of nodes from start node to goal node
    by iterating over visited nodes from the came_from set
        '''
    path = [current_node]  # initializing with current_node
    while current_node in came_from:  # iterating through came from set
        current_node = came_from[current_node]  # assign current node to the node it came from
        path.insert(0,current_node)  # insert current node at the front of the path list
    print('\tFinished.')
    return path

from collections import defaultdict

from heapq import heapify, heappush, heappop, nsmallest
max_cons_weight = 1.0
max_emis_weight = 25.0
def A_star_search_algorithm(start_coordinate, end_coordinate, trip_fuel, ship):
    '''
    returns most optimal path between start_coordinate and end_coordinate
    path is chosen based on cost
        '''
    print('\tStarting A*... (this may in some cases take a while)')

    unit_rate = get_heuristic_unit_rate(ship, trip_fuel)

    start_y, start_x = find_closest_index(start_coordinate) 
    goal_y, goal_x = find_closest_index(end_coordinate)

    y_res, x_res = 1, 1
    start = (x_res * start_x, y_res * start_y)
    goal = (x_res * goal_x, y_res * goal_y)


    # defining scoreing
    g_score = {}  # cost from start node to each node
    f_score = {}
    # Init empty map with infinity g_score
    y_bound = y_res * np.shape(data_extraction.mapdata)[0]
    x_bound = y_res * np.shape(data_extraction.mapdata)[1]
    for y in np.arange(0, y_bound):
        for x in np.arange(0, x_bound):
            g_score[x, y] = float('inf')
            f_score[x, y] = float('inf')

    # Estimate of cost to reach the current node
    g_score[start] = 0.
    # Estimate of cost from start node to goal node for each node
    f_score[start] = g_score[start] + heuristic(start, goal, ship, trip_fuel, unit_rate)
    # initializing sets for exploring and disregarding nodes
    came_from = {}  # will hold parent nodes, empty on init

    # The closed_set will hold previously explored nodes
    closed_set = []

    # The open set (heap) will hold currently explorable nodes
    # Using binheap, since the smallest element will always
    # be the first, drastically speeding up the search
    open_heap = []
    heapify(open_heap)
    heappush(open_heap, (f_score[start], start))

    neighbors = [(0, 1), (0, -1), (1, 0), (1, 1), (1, -1), (-1, 0), (-1, -1), (-1, 1)]
    # iterating over available nodes
    while open_heap:
        # Get lowest f_score element
        current_f_score, current_node = heappop(open_heap)
        closed_set.append(current_node)

        # If at goal, return the path used
        if current_node == goal:
            # f_final = current_f_score
            print('\tFinished.\n\tReconstructing path...')
            return reconstruct_path(current_node, came_from), current_f_score

        # iterating through current node's neighbors
        for dx, dy in neighbors:
            # x, y = current_node
            neighbor = (current_node[0] + dx, current_node[1] + dy)
            # If neighbor is inside chosen bounds
            if 0 <= neighbor[0] < x_bound and 0 <= neighbor[1] < y_bound:
                # Proper control flow requires this statement to fail,
                # but the previous to pass
                if np.isnan(data_extraction.mapdata[neighbor[1], neighbor[0]]):
                    # Indeterminate data, proceed to next iteration
                    continue
                # if neighbor is in closed set, due to our heuristic,
                # we can guarantee a more optimal path there has been found,
                # so we can skip it
                if neighbor in closed_set:
                    continue
                # At this point the following else-statement will be skipped,
                # and the node will be considered OK
            else:
                # Current node is OOB, proceed to next iteration
                continue

            # Potential g_score of neighbor
            tentative_g_score = g_score[current_node] + cost(current_node, neighbor, ship, trip_fuel, unit_rate)  # changes if a smaller cost path is found

            # checks if the current path to neighbor is better than previous (or none)
            if tentative_g_score < g_score[neighbor]:
                # set current node as parent node
                came_from[neighbor] = current_node

                # updating g score
                g_score[neighbor] = tentative_g_score 

                # total cost from start node to goal node via this neighbor
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal, ship, trip_fuel, unit_rate)

                # checking if neighbor has been explored
                # if neighbor not in [i[1] for i in open_heap]:
                #     heappush(open_heap, (f_score[neighbor], neighbor))  # add neighbor to set of explorable nodes
                heappush(open_heap, (f_score[neighbor], neighbor))

    # if no path has been found 
    print("Search failed!")
    return reconstruct_path(current_node, came_from), current_f_score

def plot_path(path):
    map = data_extraction.init_map()
    print('Plotting path...')
    [data_extraction.plot_coord(lon, lat, map=map)
        for lon, lat in
        [(longitude_data[y_p, x_p], latitude_data[y_p, x_p]) for x_p, y_p in path]
    ]
    print('Finished.')


def run_search(start, end):
    path, score = A_star_search_algorithm(start, end, fuel_class.fuel_list[0], ship=ship_class.ship_list[0])
    if path != None:
        plot_path(path)
        data_extraction.save_coord_map(str(start) + '_' + str(end) + '_' + str(score) + '€')
        print ('New path available in images directory!')


# TESTING SECTION

# defining start and end coordinates (lat,lon)
start_coordinate = (66.898,-162.596)
end_coordinate = (68.958, 33.082)

# import distance_correction as dist_corr

MONGSTAD = ('Mongstad', (60.810, 5.032))
MIZUSHIMA = ('Mizushima', (34.504, 133.714))
KOTZEBUE = ('Kotzebue', (66.898, -162.596))
MURMANSK = ('Murmansk', (68.958, 33.082))
REYKJAVIK = ('Reykjavik', (64.963, 19.103))
TASIILAP = ('Tasiilap', (65.604, -37.707))
    
coordinates = [
    [KOTZEBUE, MURMANSK],
    [MURMANSK, REYKJAVIK],
    [REYKJAVIK, TASIILAP],
    [TASIILAP, MURMANSK]
]

def plot_between(start_place, end_place, fuel, ship):
    path, score = A_star_search_algorithm(start_place[1], end_place[1], fuel, ship)
    plot_path(path)
    data_extraction.save_coord_map(start_place[0] + ' to ' + end_place[0] + '(' + str(score) + ', ' + fuel.name + ')')
def test():
    for fuel in fuel_class.fuel_list:
        plot_between(MURMANSK, KOTZEBUE, fuel, ship_class.ship_list[0])
def profile():
    import cProfile, pstats
    profiler = cProfile.Profile()
    print("Starting profiling...")
    profiler.enable()
    plot_between(MURMANSK, KOTZEBUE, fuel_class.fuel_list[0], ship_class.ship_list[0])
    profiler.disable()
    print("Dumping stats...")
    stats = pstats.Stats(profiler).sort_stats('ncalls')
    stats.strip_dirs()
    stats.dump_stats('new_star_profiling')
    print("Finished.")

# plot_between(MONGSTAD, MIZUSHIMA, HEURISTIC_FUEL, vessel.ship_list[0]) 
# plot_between(MONGSTAD, KOTZEBUE, HEURISTIC_FUEL, vessel.ship_list[0]) 
# plot_between(KOTZEBUE, MURMANSK, HEURISTIC_FUEL, vessel.ship_list[0]) 
# plot_between(KOTZEBUE, REYKJAVIK, HEURISTIC_FUEL, vessel.ship_list[0]) 
# plot_between(KOTZEBUE, TASIILAP, HEURISTIC_FUEL, vessel.ship_list[0]) 
# 61.092643338340345, -20.37209268308615
# 59.55292944551115, 172.49462664899926

# plot_between(start_coordinate, end_coordinate, HEURISTIC_FUEL)
# for fuel_type in fuel.fuel_list:
#     path, score = A_star_search_algorithm(start_coordinate, end_coordinate, fuel_type)
#     if path != None:
#         plot_path(path)
#         dist_corr.save_coord_map(start_coordinate.__str__() + ' - ' + end_coordinate.__str__() + '(' + fuel_type.name + ', ' + str(score) + ')')
    
# path = A_star_search_algorithm((61.093, 1.372), (55.552, 172.495), fuel_type=fuel.fuel_list[1])
# plot_path(path)
# dist_corr.save_coord_map('PxtoPyfuel1')
