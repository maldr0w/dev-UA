import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import utils

utils.print_entrypoint(__name__, __file__)


import vessel_graphing as vess_data
import vessel
import fuel
import distance_correction as dist_corr

coordinate_data = dist_corr.dataset['sea_ice_thickness'].coords
latitude_data, longitude_data = coordinate_data['lat'].to_numpy(), coordinate_data['lon'].to_numpy()

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

mean_ice_thickness = np.nanmean(dist_corr.mapdata) # (1 * np.nanstd(only_values_ice_thickness))
print("Mean ice thickness " + str(mean_ice_thickness))
def great_circle(lat1, lon1, lat2, lon2):
    a = np.deg2rad(lat1)
    b = np.deg2rad(lat2)
    x = np.deg2rad(lon1)
    y = np.deg2rad(lon2)
    c = abs(x - y)
    return 112320 * np.rad2deg(
        np.arccos(
            (np.cos(a) * np.cos(b) * np.cos(c))
            + (np.sin(a) * np.sin(b))
        )
    )
from enum import Enum
class Optimization(Enum):
    DISTANCE = 1
    CONSUMPTION = 2
    
def heuristic(node,goal,trip_fuel,cons_weight, emis_weight, estimate_thickness=False,ship=vessel.ship_list[0], mean_th=0.0):
    '''
    heuristic estimate (Great circle distance)
    estimate of distance between specified node and goal node
    
    estimate_thickness (default: False) will determine whether
    value returned is based on just distance, or
    ice thickness as well 
        '''
            
    x1, y1 = node
    x2, y2 = goal

    lat1 = latitude_data[y1, x1]
    lat2 = latitude_data[y2, x2]

    lon1 = longitude_data[y1, x1]
    lon2 = longitude_data[y2, x2]

    estimated_distance = great_circle(lat1, lon1, lat2, lon2)

    if estimate_thickness:
        estimated_thickness = 0.5 * (dist_corr.mapdata[y1, x1] + dist_corr.mapdata[y2, x2])
        kg_consumed = ship.fuel_for_trip(trip_fuel,t=estimated_thickness, v=ship.v_limit(t=estimated_thickness), d=estimated_distance)
        cons_price = cons_weight * trip_fuel.get_price(kg_consumed)
        
        emis_price = np.nan_to_num(trip_fuel.get_emission_price(kg_consumed), posinf=0.0, neginf=0.0)
        if emis_price != None:
            if np.isnan(emis_price):
                print('isnan')
            return cons_price + (emis_weight * emis_price)
        else:
            return cons_price
        # return cons_price + emis_price
    # In this case, we will use the mean ice thickness, since we are guessing (using a heuristic fuel)
    else:
        # mean_ice_thickness = np.nanmean(dist_corr.mapdata[y1:y2 + 1, x1:x2 + 1])
        # print(mean_ice_thickness)
        velocity = ship.v_limit(t=mean_ice_thickness)
        # kg_consumed = ship.fuel_for_trip(trip_fuel,v=velocity,d=estimated_distance, t=mean_ice_thickness)
        worst_case_consumed = ship.fuel_for_trip(fuel.fuel_list[4], v=velocity, d=estimated_distance, t=mean_ice_thickness)
        cons_price = cons_weight * fuel.fuel_list[4].get_price(worst_case_consumed)
        worst_case_emitted = ship.fuel_for_trip(fuel.fuel_list[1], v=velocity, d=estimated_distance, t=mean_ice_thickness)
        emis_price = emis_weight * fuel.fuel_list[1].get_emission_price(worst_case_emitted)
        return cons_price + emis_price
        # return (cons_weight * fuel.fuel_list[4].get_price(worst_case_consumed)) + (emis_weight * fuel.fuel_list[1].get_emission_price(worst_case_emitted))
    # kg_consumed = ship.fuel_for_trip(fuel_type,v=ship.v_limit(t=estimated_distance), d=estimated_distance)
    # return (cons_weight * trip_fuel.get_price(kg_consumed)) + (emis_weight * trip_fuel.get_emission_price(kg_consumed))

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
def A_star_search_algorithm(start_coordinate, end_coordinate, trip_fuel, heuristic_fuel=None, ship=vessel.ship_list[0], cons_weight=max_cons_weight, emis_weight=max_emis_weight):
    '''
    returns most optimal path between start_coordinate and end_coordinate
    path is chosen based on cost
        '''
    print('\tStarting A*...')

    start_y, start_x = find_closest_index(start_coordinate) 
    goal_y, goal_x = find_closest_index(end_coordinate)

    neighbors = [(0, 1), (0, -1), (1, 0), (1, 1), (1, -1), (-1, 0), (-1, -1), (-1, 1)]
    y_res, x_res = 1, 1
    start = (x_res * start_x, y_res * start_y)
    goal = (x_res * goal_x, y_res * goal_y)

    # initializing sets for exploring and disregarding nodes

    # Using binheap, since the smallest element will always
    # be the first, drastically speeding up the search
    open_heap = []
    heapify(open_heap)

    open_set = {start}  # will hold explorable nodes in a unordered list (set)
    came_from = {}  # will hold parent nodes, empty on init
    goal_thickness = dist_corr.mapdata[goal_y, goal_x]
    # defining scoreing
    g_score = {}  # cost from start node to each node
    f_score = {}
    # Init empty map with infinity g_score
    y_bound = y_res * np.shape(dist_corr.mapdata)[0]
    x_bound = y_res * np.shape(dist_corr.mapdata)[1]
    for y in np.arange(0, y_bound):
        for x in np.arange(0, x_bound):
            g_score[x, y] = float('inf')
            f_score[x, y] = float('inf')

    # Estimate of cost to reach the current node
    g_score[start] = 0.
    # Estimate of cost from start node to goal node for each node
    f_score[start] = heuristic(start, goal, trip_fuel, cons_weight, emis_weight)

    heappush(open_heap, (f_score[start], start))

    # iterating over available nodes
    while open_heap:
        # Get lowest f_score element
        current_node = heappop(open_heap)[1]

        # curr_x, curr_y = current_node
        # slice = dist_corr.mapdata[curr_y:goal_y, curr_x:goal_x]
        # if np.shape(slice)[0] == 0 or np.shape(slice)[1] == 0:
        #     mean_thickness = 0.0
        # else:
        #     mean_thickness = np.nanmean(slice)
        #     if np.isnan(mean_thickness):
        #         mean_thickness = float('inf')

        # If at goal, return the path used
        if current_node == goal:
            print('\tFinished.\n\tReconstructing path...')
            return reconstruct_path(current_node, came_from)

        # iterating through current node's neighbors
        for dx, dy in neighbors:
            x, y = current_node
            neighbor = (x + dx, y + dy)
            # If neighbor is inside chosen indices
            if 0 <= neighbor[0] < x_bound and 0 <= neighbor[1] < y_bound:
                if np.isnan(dist_corr.mapdata[neighbor[1], neighbor[0]]):
                    # Either land, or otherwise meaningless data
                    continue
            else:
                # New pos OOB
                continue

            current_cost = heuristic(current_node, neighbor, trip_fuel, cons_weight, emis_weight, estimate_thickness=True)
            tentative_g_score = g_score[current_node] + current_cost  # changes if a smaller cost path is found

            if tentative_g_score < g_score[neighbor]:
                # set current node as parent node
                came_from[neighbor] = current_node

                # updating g score
                g_score[neighbor] = tentative_g_score 

                # total cost from start node to goal node via this neighbor
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal, trip_fuel, cons_weight, emis_weight)

                # checking if neighbor has been explored
                if neighbor not in [i[1] for i in open_heap]:
                    heappush(open_heap, (f_score[neighbor], neighbor))  # add neighbor to set of explorable nodes

    # if no path has been found 
    print("Search failed!")
    return reconstruct_path(current_node, came_from)

# defining start and end coordinates (lat,lon)
start_coordinate = (66.898,-162.596)
end_coordinate = (68.958, 33.082)

# import distance_correction as dist_corr

KOTZEBUE = (66.898, -162.596)
MURMANSK = (68.958, 33.082)
REYKJAVIK = (64.963, 19.103)
TASIILAP = (65.604, -37.707)
    
coordinates = [
    [KOTZEBUE, MURMANSK],
    [MURMANSK, REYKJAVIK],
    [REYKJAVIK, TASIILAP],
    [TASIILAP, MURMANSK]
]

def plot_path(path):
    map = dist_corr.init_map()
    print('Plotting path...')
    [dist_corr.plot_coord(lon, lat, map=map)
        for lon, lat in
        [(longitude_data[y_p, x_p], latitude_data[y_p, x_p]) for x_p, y_p in path]
    ]
    print('Finished.')

if __name__ == '__main__':
    # If file is ran as main, do one profiled run
    import cProfile, pstats
    profiler = cProfile.Profile()
    print("Starting profiling...")
    profiler.enable()
    # for start_coordinate, end_coordinate in coordinates:
    path = A_star_search_algorithm(start_coordinate, end_coordinate, fuel.fuel_list[0], fuel.fuel_list[4])
    profiler.disable()
    print("Dumping stats...")
    stats = pstats.Stats(profiler).sort_stats('ncalls')
    stats.strip_dirs()
    stats.dump_stats('new_star_profiling')
    print("Finished.")
    plot_path(path)
    dist_corr.save_coord_map('Murmansk_to_alaska')

# 61.092643338340345, -20.37209268308615
# 59.55292944551115, 172.49462664899926

path = A_star_search_algorithm((61.093, -20.372), (55.552, 172.495), fuel.fuel_list[0], fuel.fuel_list[4])
plot_path(path)
dist_corr.save_coord_map('Tasiilap_to_reykjavik')

# for fuel_type in fuel.fuel_list:
#     path = A_star_search_algorithm((61.093, 1.372), (55.552, 172.495), fuel_type=fuel_type)
#     plot_path(path)
#     dist_corr.save_coord_map('PxtoPy' + fuel_type.name)

# Cheapest overall case
#  Methanol is cheapest, Hydrogen is most expensive

# path = A_star_search_algorithm((61.093, -20.372), (55.552, 172.495), fuel.fuel_list[2], fuel.fuel_list[4])
# plot_path(path)
# dist_corr.save_coord_map('Comparing Methanol to Hydrogen-Heuristic')

# path = A_star_search_algorithm((61.093, -20.372), (55.552, 172.495), fuel.fuel_list[2], fuel.fuel_list[2])
# plot_path(path)
# dist_corr.save_coord_map('Comparing Methanol to Methanol-Heuristic')

# path = A_star_search_algorithm((61.093, -20.372), (55.552, 172.495), fuel.fuel_list[4], fuel.fuel_list[4])
# plot_path(path)
# dist_corr.save_coord_map('Comparing Hydrogen to Hydrogen-Heuristic')

print('Checking all fuel cases against most expensive')

print('Total price: Heuristic - Hydrogen')
print('Price of consumption: Heuristic - Hydrogen')
print('Price of emission: Heuristic - Natural Gas')
for fuel_type in fuel.fuel_list:
    if fuel_type.name != 'hydrogen' and fuel_type.name != 'dme':
        path = A_star_search_algorithm(start_coordinate, end_coordinate, fuel_type, cons_weight=0.0)
        if path != None:
            plot_path(path)
            dist_corr.save_coord_map(fuel_type.name + ': (100% emission price)')

    path = A_star_search_algorithm(start_coordinate, end_coordinate, fuel_type, emis_weight=0.75 * max_emis_weight, cons_weight= 0.25 * max_cons_weight)
    if path != None:
        plot_path(path)
        dist_corr.save_coord_map(fuel_type.name + ': (25% consumption price-75% emission price, default)')

    path = A_star_search_algorithm(start_coordinate, end_coordinate, fuel_type, emis_weight=0.66 * max_emis_weight, cons_weight= 0.33 * max_cons_weight)
    if path != None:
        plot_path(path)
        dist_corr.save_coord_map(fuel_type.name + ': (33% consumption price-66% emission price, default)')

    path = A_star_search_algorithm(start_coordinate, end_coordinate, fuel_type)
    if path != None:
        plot_path(path)
        dist_corr.save_coord_map(fuel_type.name + ': (50% consumptionp price-50% emission price, default)')

    path = A_star_search_algorithm(start_coordinate, end_coordinate, fuel_type, emis_weight=0.33 * max_emis_weight, cons_weight= 0.66 * max_cons_weight)
    if path != None:
        plot_path(path)
        dist_corr.save_coord_map(fuel_type.name + ': (66% consumption price-33% emission price, default)')
    
    path = A_star_search_algorithm(start_coordinate, end_coordinate, fuel_type, emis_weight=0.25 * max_emis_weight, cons_weight=0.5 * max_cons_weight)
    if path != None:
        plot_path(path)
        dist_corr.save_coord_map(fuel_type.name + ': (75% consumption price-25% emission price)')

    path = A_star_search_algorithm(start_coordinate, end_coordinate, fuel_type, emis_weight=0.0)
    if path != None:
        plot_path(path)
        if fuel_type.name == 'natural gas':
            dist_corr.save_coord_map('Most polluting path (Natural Gas - minimize fuel cost only)')
        elif fuel_type.name == 'hydrogen':
            dist_corr.save_coord_map('Most cost conserving path (Hydrogen - minimize fuel cost only)')
        else:
            dist_corr.save_coord_map(fuel_type.name + ': (100% consumption price)') 
    
# path = A_star_search_algorithm((61.093, 1.372), (55.552, 172.495), fuel_type=fuel.fuel_list[1])
# plot_path(path)
# dist_corr.save_coord_map('PxtoPyfuel1')
