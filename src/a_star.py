import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import utils
from typing import List, Type



utils.print_entrypoint(__name__, __file__)
import graph_creation
import ship_class
import fuel_class
import data

coordinate_data = data.dataset['sea_ice_thickness'].coords
latitude_data, longitude_data = coordinate_data['lat'].to_numpy(), coordinate_data['lon'].to_numpy()
mean_ice_thickness = np.nanmean(data.ice_values) # (1 * np.nanstd(only_values_ice_thickness))
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


from scipy import stats

# only_values_ice_thickness = np.extract(np.invert(np.logical_or(np.isnan(dist_corr.mapdata), np.less(dist_corr.mapdata, ICE_THRESHOLD))), dist_corr.mapdata)

MEAN_EARTH_RADIUS_METERS = 6_371_000.0
def great_circle(
        lat1: float, lon1: float, 
        lat2: float, lon2: float
        ) -> float:
    ''' Calculates distance between coordinates
    :param lat1: float - Latitude of initial point
    :param lon1: float - Longitude of initial point
    :param lat2: float - Latitude of final point
    :param lon2: float - Longitude of final point
    :return: float - Distance between the points

    Uses the haversine formula to return distance between coordinate points
    '''

    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    phid = np.deg2rad(lat2 - lat1)
    lamd = np.deg2rad(lon2 - lon1)

    a = (np.sin(phid / 2) * np.sin(phid / 2)) + np.cos(phi1) * np.cos(phi2) * (np.sin(lamd / 2) * np.sin(lamd / 2))
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return c * MEAN_EARTH_RADIUS_METERS
    
ICE_THICKNESS_LIMIT = 2.1
def cost(node: List[int], neighbor: List[int], ship: Type[ship_class.Ship], trip_fuel):
    ''' Cost function [g(n)]
    :param node: List[int] - The indices of the node
    :param neighbor: List[int] - The indicies of the neighbor
    :param ship: Type[ship_class.Ship] - The ship in question
    :return: float - The estimated cost between node and neighbor

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

    estimated_distance = great_circle(lat1, lon1, lat2, lon2)
    estimated_thickness = 0.5 * (data.ice_values[y1, x1] + data.ice_values[y2, x2])

    if estimated_thickness <= ICE_THICKNESS_LIMIT:
        estimated_cost = ship.get_cost(estimated_thickness, estimated_distance)
    else:
        estimated_cost = float('inf')

    return estimated_cost
  
MANHATTAN_DISTANCE_MAX = 2 * 432 

WEIGHT_BASE = 1.0001
EPSILON = 0.025
def heuristic(node, goal, ship, fuel):
    '''
    heuristic estimate (Diagonal distance) [h(n)]
    Estimate of cost to reach goal node from specified node.

    ==GOAL==
    The goal of the heuristic is to provide an overestimate at every point,
    thus the fuel used to calculate the basal rate is Hydrogen, the most expensive
    fuel found through the graphs created elsewhere in the program.

    ==WEIGHTING==
    The weighting here is very important. Under development, it was often
    found that a too 'realistic' estimate, would result in infinite search
    loops, as all neighboring path segments would seem equally ideal,
    especially in case of paths along open water (no ice). Additionally,
    due to the number of times it would be run, any added complexity would
    greatly impact the runtime of the search.

        The idea is to use a weighted approach.
        In order to ensure the estimate is always a bit high, we define a term
        called the weighting exponent.

        WEIGHT_BASE : the constant part of the weighting exponent
        EPSILON : weight of error percentage increment,
        delta x : absolute difference of given x indices
        delta y : absolute difference of given y indices
        percent error of x : delta x divided by x index full span,
        percent error of y : delta y divided by y index full span,
        sigma x : EPSILON times percent error of x,
        sigma y : EPSILON times percent error of y,
        weighting exponent : WEIGHT_BASE + sigma x + sigma y,
        distance estimate : great circle distance of given coordinates,
        weighted distance : distance estimate, raised to the power of the weighting exponent

        The final cost estimate is then calculated as the unit cost of the ship,
        which is set at the beginning of the journey (set to the fuel to compare to
        which by default is Hydrogen, due to its high price), times this weighted
        distance.

    The result is that the algorithm will greatly overestimate when far from the point.
    This is very useful in certain cases, such as in the route "Mongstad to Mizushima"
    in the test paths. For this route, the algorithm can tend to prefer path length
    a bit too much, due to the fact that a regular distance estimate would fail to
    take into account that the goal is on the other side of a landmass.
    In order to minimize runtime inefficiency, a "dumb" way of predicting this was needed.

    The weighting exponent is the key to this "dumb" way.
    '''
    # print('HEURISTIC')
    x1, y1 = node
    x2, y2 = goal

    delta_x = abs(x2 - x1)
    delta_y = abs(y2 - y1)

    percent_x = delta_x / 432
    sigma_x = EPSILON * percent_x

    percent_y = delta_y / 432
    sigma_y = EPSILON * percent_y

    weight_sigma = sigma_x + sigma_y

    manhattan_distance = delta_x + delta_y
    manhattan_distance_ratio = manhattan_distance / MANHATTAN_DISTANCE_MAX
    euclidean_distance = np.sqrt((delta_x ** 2) + (delta_y)** 2)
    diagonal_distance = max(delta_x, delta_y)

    lat1 = latitude_data[y1, x1]
    lat2 = latitude_data[y2, x2]

    lon1 = longitude_data[y1, x1]
    lon2 = longitude_data[y2, x2]

    great_circle_distance = great_circle(lat1, lon1, lat2, lon2)

    # diagonal_distance = max(abs(x2 - x1), abs(y2 - y1))

    diagonal_distance_ratio = diagonal_distance / 432

    # diagonal_ratio = diagonal_distance / 432

    # weighting_exponent = 1.0 + (0.21 * diagonal_ratio)

    # x_distance = abs(x2 - x1)
    # y_distance = abs(y2 - y1)
    # distance_ratio = (x_distance + y_distance) / (2 * 432)

    # weighting_exponent = 1.0 + (0.005 * manhattan_distance_ratio)
    weighting_exponent = WEIGHT_BASE + weight_sigma
  
    # estimated_units = great_circle(lat1, lon1, lat2, lon2)
    # estimated_units = max(x_distance, y_distance)
    # estimated_units = x_distance + y_distance
    # estimated_units = np.sqrt((x_distance ** 2) + (y_distance ** 2))
    # p1_past_passage = (lon1 >= 100.0 or lon1 <= -100.0) and lat1 >= 60.0
    # p2_past_passage = (lon2 >= 100.0 or lon2 <= -100.0) and lat2 >= 60.0
    # passage_case1 = p1_past_passage and not p2_past_passage
    # passage_case2 = not p1_past_passage and p2_past_passage

    # passage_case = passage_case1 or passage_case2

    # TODO: Move this constant
    # UNIT_UPPER_LIMIT = 600.0

    # distant_case = estimated_units > UNIT_UPPER_LIMIT
    # if distant_case or passage_case:
    #     estimated_distance = utils.unit_distance * estimated_units
    # estimated_units = abs(x1 - x2) + abs(y1 - y2)
    # else:
    estimated_distance = great_circle(lat1, lon1, lat2, lon2)
    # estimated_distance = euclidean_distance * utils.unit_distance
    weighted_distance = np.power(estimated_distance, weighting_exponent)
    # estimated_distance = utils.unit_distance * estimated_units
    # estimated_distance = (utils.unit_distance / 25.0) * estimated_units
    print('dist')
    print(estimated_distance)
    print('weighted')
    print(weighted_distance)
    estimated_cost = ship.unit_cost * weighted_distance
    # estimated_cost = (ship.unit_cost * 25.0) * np.float_power(estimated_units, 1.5)
    # print('est cost')
    # print(estimated_cost)
    return estimated_cost
    # estimated_units = abs(x1 - x2) + abs(y1 - y2) 
    # estimated_distance = great_circle(lat1, lon1, lat2, lon2)
    # estimated_distance = utils.unit_distance * max(abs(x1 - x2), abs(y1 - y2))
    # estimated_cost = ship.COST_PER_UNIT * estimated_units
    # estimated_distance = np.sqrt((abs(x1 - x2) ** 2) + (abs(y1 - y2) ** 2))
    # estimated_distance = np.sqrt((abs(lat1 - lat2) ** 2) + (abs(lon1 - lon2) ** 2))
    # return estimated_distance * 25000.
    # weight = heuristic_unit_rate * (estimated_distance / 25000.0)
    # weight = heuristic_unit_rate * estimated_distance
    # return heuristic_unit_rate * (estimated_distance / 25000.0)
    # return estimated_distance * (1.0 / (np.e ** mean_ice_thickness)) 
    # return estimated_distance
    # return ship.get_costs(HEURISTIC_FUEL, ship.v_max, estimated_units * utils.unit_distance, 0.0)
    return estimated_cost
    # return ship.get_costs(fuel_type, ship.v_limit(0.0), estimated_distance, 0.0)
    # return 25000.0 * estimated_distance

    # return ship.get_costs(fuel_type, ship.v_limit(HEURISTIC_THICKNESS), estimated_distance, HEURISTIC_THICKNESS) 

def reconstruct_path(current_node,came_from): 
    '''
    returns reconstructed path as a list of nodes from start node to goal node
    by iterating over visited nodes from the came_from set
        '''
    if utils.verbose_mode:
        print('\tReconstructing path...')
    path = [current_node]  # initializing with current_node
    while current_node in came_from:  # iterating through came from set
        current_node = came_from[current_node]  # assign current node to the node it came from
        path.insert(0,current_node)  # insert current node at the front of the path list
    if utils.verbose_mode:
        print('\tFinished.\n')
    return path

from collections import defaultdict

from heapq import heapify, heappush, heappop, nsmallest
# max_cons_weight = 1.0
# max_emis_weight = 25.0
def A_star_search_algorithm(start_coordinate, end_coordinate, fuel, ship):
    '''
    returns most optimal path between start_coordinate and end_coordinate
    path is chosen based on cost
        '''
    print('\tStarting A*... (this may in some cases take a while)')

    # There may be better values to choose here
    ship.set_target_velocity(0.88)

    # Initialize unit cost as the most expensive overall, Hydrogen
    ship.set_fuel(fuel_class.Hydrogen())
    ship.set_unit_cost()

    # Set to actual fuel in use
    ship.set_fuel(fuel)
    # ship.set_target_velocity(1.00)
    # ship.set_unit_cost()

    # unit_rate = get_heuristic_unit_rate(ship, trip_fuel)

    start_y, start_x = find_closest_index(start_coordinate) 
    goal_y, goal_x = find_closest_index(end_coordinate)

    # y_res, x_res = 1, 1
    # start = (x_res * start_x, y_res * start_y)
    # goal = (x_res * goal_x, y_res * goal_y)
    start = (start_x, start_y)
    goal = (goal_x, goal_y)


    # defining scoreing
    g_score = {}  # cost from start node to each node
    f_score = {}
    # Init empty map with infinity g_score
    y_bound = np.shape(data.ice_values)[0]
    x_bound = np.shape(data.ice_values)[1]
    for y in np.arange(0, y_bound):
        for x in np.arange(0, x_bound):
            g_score[x, y] = float('inf')
            f_score[x, y] = float('inf')

    # Estimate of cost to reach the current node
    g_score[start] = 0.
    # Estimate of cost from start node to goal node for each node
    f_score[start] = g_score[start] + heuristic(start, goal, ship, fuel)
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
            print('\tSearch successful!\n')
            return reconstruct_path(current_node, came_from), current_f_score, True

        # iterating through current node's neighbors
        for dx, dy in neighbors:
            # x, y = current_node
            neighbor = (current_node[0] + dx, current_node[1] + dy)
            # If neighbor is inside chosen bounds
            if 0 <= neighbor[0] < x_bound and 0 <= neighbor[1] < y_bound:
                # Proper control flow requires this statement to fail,
                # but the previous to pass
                if np.isnan(data.ice_values[neighbor[1], neighbor[0]]):
                    # Indeterminate data, proceed to next iteration
                    continue
                # if neighbor is in closed set, due to our heuristic,
                # we can guarantee a more optimal path there has been found,
                # so we can skip it
                if neighbor in closed_set:
                    continue
                if neighbor in came_from:
                    continue
                # At this point the following else-statement will be skipped,
                # and the node will be considered OK
            else:
                # Current node is OOB, proceed to next iteration
                continue

            # Potential g_score of neighbor
            tentative_g_score = g_score[current_node] + cost(current_node, neighbor, ship, fuel)  # changes if a smaller cost path is found

            # checks if the current path to neighbor is better than previous (or none)
            if tentative_g_score < g_score[neighbor]:
                # set current node as parent node
                came_from[neighbor] = current_node

                # updating g score
                g_score[neighbor] = tentative_g_score 

                # total cost from start node to goal node via this neighbor
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal, ship, fuel)

                # checking if neighbor has been explored
                # if neighbor not in [i[1] for i in open_heap]:
                #     heappush(open_heap, (f_score[neighbor], neighbor))  # add neighbor to set of explorable nodes
                heappush(open_heap, (f_score[neighbor], neighbor))

    # if no path has been found 
    print('\tSearch failed!\n')
    # path = reconstruct_path(current_node, came_from)

    # path.append(end_coordinate)

    return reconstruct_path(current_node, came_from), current_f_score, False

def plot_path(path, map=None):
    if map == None:
        map = data.init_map()
    if utils.verbose_mode:
        print('\tPlotting path...')
    [data.plot_coord(lon, lat, map=map)
        for lon, lat in
        [(longitude_data[y_p, x_p], latitude_data[y_p, x_p]) for x_p, y_p in path]
    ]
    print('\tFinished.\n')


def run_search(start_coordinate, end_coordinate):
    '''
    Runs A* on the provided coordinates, and attempts to plot the path.
    In case of a failed search, only the start-point and end-point will be
    visible in the plot (for debugging purposes etc.)
    :param start_coordinate: List[float] - Start coordinate as [lon, lat]
    :param end_coordinate: List[float] - End coordinate as [lon, lat]
    '''

    path, score, search_successful = A_star_search_algorithm(start_coordinate, end_coordinate, fuel_class.fuel_list[0], ship=ship_class.ship_list[0])
    if path != None and search_successful:
        plot_path(path)
        data.save_coord_map(str(start_coordinate) + '_' + str(end_coordinate) + '_' + str(score) + '€')
        print ('New path available in images directory!')
    else:
        map = data.init_map()

        # Find the coordinates closest to the given ones (similar to how it is done in the body of A*)
        # Then plot these points, and these points only
        lon_start, lat_start = find_closest_coordinate(start_coordinate)
        # x_start, y_start = find_closest_coordinate(start_coordinate)
        # lon_start, lat_start = (longitude_data[y_start, x_start], latitude_data[y_start, x_start])
        data.plot_coord(lon_start, lat_start, m='.', map=map)

        lon_end, lat_end = find_closest_coordinate(end_coordinate)
        # x_end, y_end = find_closest_coordinate(end_coordinate)
        # lon_end, lat_end = (longitude_data[y_end, x_end], latitude_data[y_end, x_end])
        data.plot_coord(lon_end, lat_end, m='.', map=map)

        plot_path(path, map=map)
        data.save_coord_map(str(start_coordinate) + '_' + str(end_coordinate) + '_failure_' + str(score) + '€')

        print('Check images directory to see details, there will be a new map there, containing some points for debugging.\nThe large dots indicate start/end coordinates, the small ones indicate the path attempted.')



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
    path, score, search_successful = A_star_search_algorithm(start_place[1], end_place[1], fuel, ship)
    plot_path(path, data.init_map())
    data.save_coord_map(start_place[0] + ' to ' + end_place[0] + '(' + str(score) + ', ' + fuel.name + ')')
def test():
    for fuel in fuel_class.fuel_list:
        plot_between(MURMANSK, KOTZEBUE, fuel, ship_class.ship_list[0])
        plot_between(MONGSTAD, MIZUSHIMA, fuel, ship_class.ship_list[0])
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
