import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import utils
from typing import List, Type, Tuple, TypeAlias, NamedTuple


utils.print_entrypoint(__name__, __file__)
import graph_creation
from ship_class import Ship
import ship_class
from fuel_class import Fuel
import fuel_class
import data
coordinate_data = data.dataset['sea_ice_thickness'].coords
latitude_data, longitude_data = coordinate_data['lat'].to_numpy(), coordinate_data['lon'].to_numpy()
mean_ice_thickness = np.nanmean(data.ice_values) # (1 * np.nanstd(only_values_ice_thickness))
HEURISTIC_FUEL = fuel_class.fuel_list[2]
HEURISTIC_THICKNESS = 0.0
# ===================================================================
# Type Definitions
Latitude: TypeAlias = float
''' Latitude : float, alias for latitude values
Meant as a marker to show function API usage
'''
Longitude: TypeAlias = float
''' Longitude : float, alias for longitude values
Pair to latitude
'''
class CoordinateClass():
    def __init__(self, latitude: Latitude, longitude: Longitude):
        self.latitude: Latitude = latitude
        self.longitude: Longitude = longitude
    def great_circle(self, other: Type['__class__']):
        return great_circle(self.latitude, self.longitude, other.latitude, other.longitude)
# ===================================================================
MEAN_EARTH_RADIUS_METERS = 6_371_000.0
from attrs import define, field
@define
class Coordinate:
    lat: float
    lon: float
    # def __hash__(self):
    #     return hash((self.lat, self.lon))
    def __str__(self):
        return f"({self.lat},{self.lon})"
    def great_circle(self, other: Type['__class__']) -> float:
        lat1, lon1, lat2, lon2 = np.deg2rad([self.lat, self.lon, other.lat, other.lon])
        latd = lat2 - lat1
        lond = lon2 - lon1
        sin_latd = np.sin(latd / 2)
        sin_lond = np.sin(lond / 2)
        # phi1, phi2 = np.deg2rad(self.lat), np.deg2rad(other.lat)
        # phid = np.deg2rad(other.lat - self.lat)
        # lamd = np.deg2rad(other.lon - self.lon)

        a = (sin_latd ** 2) + np.cos(lat1) * np.cos(lat2) * (sin_lond ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return c * MEAN_EARTH_RADIUS_METERS
    # def to_node(self) -> Node:
@define
class Node:
    x: int
    y: int
    coord: Coordinate
    thickness: float
    distance_to_goal: float = float('inf')
    def __hash__(self):
        return hash((self.x, self.y))
    def indices(self):
        return (self.y, self.x)
    def delta(self, other: 'Node') -> tuple[int, int]:
        return (abs(other.y - self.y), abs(other.x - self.x))
    def diagonal_to(self, other: 'Node') -> int:
        return max(self.delta(other))
    def great_circle_to(self, other: 'Node') -> float:
        return self.coord.great_circle(other.coord)
    def distance_to(self, other: 'Node') -> float:
        return np.sqrt((self.x ** 2) + (self.y ** 2))
    def bias(self, other: 'Node') -> float:
        delta_y, delta_x = self.delta(other)
        percentage_x: float = delta_x / 432.0 
        percentage_y: float = delta_y / 432.0
        return 0.5 * (percentage_x + percentage_y) 
    def set_distance_to_goal(self, goal: 'Node'):
        self.distance_to_goal = self.great_circle_to(goal)
IndexPair: TypeAlias = tuple[int, int]
'''Node : (int, int), alias for a node
Defined by its relevant indices in the ice thickness dataset
'''
class NodeClass():
    def __init__(self, indices: IndexPair):
        self.indices: IndexPair = indices
        self.coordinate: CoordinateClass = CoordinateClass(latitude_data[self.indices], longitude_data[self.indices])
        self.thickness: float = data.ice_values[self.indices]

    def diagonal_distance(self, other: Type['__class__']) -> int:
        delta_y = abs(other[0] - self[0])
        delta_x = abs(other[1] - self[1])
        return max(delta_y, delta_x)

NodeSet: TypeAlias = list[IndexPair]
'''NodeSet : (int, int) list, alias for a collection of nodes
Often used to imply paths as well
'''
class NodeSetC():
    def __init__(self):
        self.nodes = []
    def push(self, node: NodeClass):
        self.nodes.append(node)
NodeMap: TypeAlias = dict[IndexPair, IndexPair]
'''NodeMapping : (int -> int) list, alias for a dict of nodes
Used to reconstruct path and keep track of parent nodes in path
'''
class NodeMapC():
    def __init__(self):
        self.map = {}
    def insert(self, _from: NodeClass, _to: NodeClass):
        self.map[_from] = _to
# ===================================================================
Score: TypeAlias = float
'''Score : float, alias for various types of scores around the program
Meant to show intention of function, or its return value
'''
ScoreSet: TypeAlias = dict[IndexPair, Score]
'''ScoreSet : ((int, int), float) dict, alias for g score and f score types
'''
HeapItem: TypeAlias = tuple[Score, IndexPair]
'''HeapItem: (float, (int, int)), alias for the items kept in the heap
The first item must be a float,
'''
Heap: TypeAlias = list[HeapItem]
'''Heap : (float, (int, int)) list, alias for our binary heap
IDE support, etc, helps a lot in debugging and keeps our code clear in intention
'''
SearchResult: TypeAlias = tuple[NodeSet, Score, bool]
'''SearchResult : ((int, int) list, float, bool), alias for the return type of A*
Meant to condense function signature a bit, while keeping the intention of the code clear
'''

def find_closest_indices(coordinate: Coordinate) -> tuple[int, int]:
    '''Finds index in the dataset corresponding to the
    coordinate closest to the provided coordinate 

    :param coordinate: tuple[float, float] - Coordinate to find index of (lat, lon)
    :return: tuple[int, int] - The indices of the coordinate point
       '''
    # latitude_difference = np.apply_along_axis(lambda x: abs(coordinate.lat - x), 0, latitude_data)
    # longitude_difference = np.apply_along_axis(lambda x: abs(coordinate.lon - x), 0, longitude_data)
    latitude_difference: np.ndarray = np.abs(coordinate.lat - latitude_data)
    longitude_difference: np.ndarray = np.abs(coordinate.lon - longitude_data)
    arr_difference: np.ndarray = latitude_difference + longitude_difference
    # arr_difference = np.apply_along_axis(lambda x: x[0] + x[1], 2, np.dstack((latitude_difference, longitude_difference)))

    # y_idx, x_idx = np.unravel_index(indices=np.argmin(arr_difference), shape=np.shape(arr_difference))
    min_indices = arr_difference.argmin()
    y_idx, x_idx = np.unravel_index(indices=min_indices, shape=arr_difference.shape)
    # thickness = data.ice_values[y, x]
    # return Node(y = y, x = x, coord = coordinate, thickness = data.ice_values[y, x])
    return y_idx, x_idx

def find_closest_coordinate(coordinate: Coordinate) -> Coordinate:
    '''Find a coordinate's closest coordinate in our coordinate dataset 

    :param coordinate: tuple[float, float] - The coordinate in question (lat, lon)
    :return: tuple[float, float] - The coordinate's closest match
    '''
    node: tuple[int, int] = find_closest_indices(coordinate)
    coordinate: Coordinate = (latitude_data[node], longitude_data[node])
    return coordinate
# Initializing A* search algorithm

from scipy import stats

MEAN_EARTH_RADIUS_METERS = 6_371_000.0
def great_circle(
        lat1: Latitude, lon1: Longitude, 
        lat2: Latitude, lon2: Longitude
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

def cost(
        node: Node, 
        neighbor: Node,
        ship: Type[Ship]
        ) -> Score:
    ''' Cost function [g(n)]
    :param node: Node - The indices of the node
    :param neighbor: Node - The indicies of the neighbor
    :param ship: Type[Ship] - The ship in question
    :return: float - The estimated cost between node and neighbor

    Using the same calculation as the heuristic, 
    this function returns the actual cost, using the average of the
    node thickness and the neighbor thickness, as well as a specified fuel
    '''
    estimated_thickness = 0.5 * (node.thickness + neighbor.thickness)

    if estimated_thickness <= ICE_THICKNESS_LIMIT:
        estimated_distance = node.coord.great_circle(neighbor.coord)
        estimated_cost = ship.get_cost(estimated_thickness, estimated_distance)
    else:
        estimated_cost = float('inf')

    return estimated_cost
  
INACCURACY_CONSTANT: float = 0.001
WEIGHT_BASE: float = 1.000 + INACCURACY_CONSTANT
EPSILON: float = 0.01

def heuristic(node: Node, goal: Node, ship: Type[Ship]) -> Score:
    """
    heuristic estimate (Diagonal distance) [h(n)]
    Estimate of cost to reach goal node from specified node.

    :param node: tuple[int, int] - The indices of the current node
    :param goal: tuple[int, int] - The indices of the goal node
    :param ship: Type[Ship] - The ship to calculate for
    :return: float - The overestimated cost to reach the goal

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
    """
    # delta_y, delta_x = node.delta(goal)

    # percent_x: int = delta_x / 432
    # sigma_x: float = EPSILON * float(percent_x)

    # percent_y: int = delta_y / 432
    # sigma_y: float = EPSILON * float(percent_y)

    # weight_sigma: float = sigma_x + sigma_y

    # diagonal_distance: int = max(delta_x, delta_y)
    # diagonal_distance_ratio: int = diagonal_distance / 432

    # weighting_exponent: float = WEIGHT_BASE - weight_sigma
   
    estimated_distance = node.great_circle_to(goal)
    print('dist')
    print(estimated_distance)

    bias = node.bias(goal) / 10.0
    weighted_distance = (1.0 + bias) * estimated_distance
    print('weighted')
    print(weighted_distance)

    delta_distance = abs(weighted_distance - estimated_distance)
    percent_delta = (weighted_distance / estimated_distance) - 1.0
    print('delta')
    print(str(weighted_distance) + ' ' + str(percent_delta))
    estimated_cost = ship.unit_cost * weighted_distance
    return estimated_cost

def reconstruct_path(
        current_node: Node,
        came_from: dict[Node, Node]
        ) -> NodeSet: 
    ''' Returns reconstructed path as a list of nodes from start node to goal node
    by iterating over visited nodes from the came_from set

    :param current_node: Node - The node to start building from
    :param came_from: dict[Node, Node] - The rest of the nodes to build from
    :return: list[Node] The reconstructed path
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
X_BOUND = 432
Y_BOUND = 432
coordinate_set: dict[tuple[int, int], Coordinate] = {}

node_set: np.ndarray = np.ndarray(np.shape(data.ice_values), dtype=Node)
for y in np.arange(0, Y_BOUND):
    for x in np.arange(0, X_BOUND):
        coordinate = Coordinate(
            latitude_data[y, x], 
            longitude_data[y, x])
        coordinate_set[y, x] = coordinate
        # print(coordinate)
        node = Node(y, x, coordinate, data.ice_values[y, x])
        # print(node)
        # print(node)
        node_set[y, x] = node
# print(node_set.__len__())

def A_star_search_algorithm(start_coordinate: Coordinate, end_coordinate: Coordinate, ship: Type[Ship], fuel: Type[Fuel]) -> tuple[list[Node], float, bool]:
    ''' Returns most optimal path between start_coordinate and end_coordinate
    path is chosen based on cost

    :param start_coordinate: tuple[float, float] -
    :param end_coordinate: tuple[float, float] -
    :param ship: Type[Ship] - 
    :param fuel: Type[Fuel] -
    :return: tuple[list[Node], float, bool]
    '''
    print('\tStarting A*... (this may in some cases take a while)')

    # There may be better values to choose here
    ship.set_target_velocity(0.88)
    emptyset = []

    # Initialize unit cost as the cheapest overall, Methanol
    ship.set_fuel(fuel_class.Methanol())
    ship.set_unit_cost()

    # Set to actual fuel in use
    ship.set_fuel(fuel)

    start_indices: tuple[int, int] = find_closest_indices(start_coordinate)
    start: Node = node_set[start_indices]

    print('start')
    print(start)
    print(test_route_1.start.get_position())
    # print('Start is ' + str(type(start)))
    goal_indices: tuple[int, int] = find_closest_indices(end_coordinate)
    goal: Node = node_set[goal_indices]


    # defining scoreing
    g_score: np.ndarray = np.ndarray(np.shape(data.ice_values))  # cost from start node to each node
    g_score.fill(float('inf'))
    # f_score: dict[Node, float] = {}
    f_score: np.ndarray = np.ndarray(np.shape(data.ice_values))
    f_score.fill(float('inf'))
    # Init empty map with infinity g_score
    y_bound: int = np.shape(data.ice_values)[0]
    x_bound: int = np.shape(data.ice_values)[1]
    # for y in np.arange(0, y_bound):
    #     for x in np.arange(0, x_bound):
    #         # FLIPPED
    #         g_score[y, x] = float('inf')
    #         f_score[y, x] = float('inf')
    # for node_row in node_set:
    #     for node in node_row:
    #         print(node)
    #         # indices = (node.y, node.x)
    #         # print(indices)
    #         # g_score[node] = float('inf')
    #         f_score[node] = float('inf')
    print(f_score.__len__())
    print(g_score.__len__())

    # Estimate of cost to reach the current node
    g_score[start.indices()] = 0.
    # Estimate of cost from start node to goal node for each node
    f_score[start.indices()] = g_score[start.indices()] + heuristic(start, goal, ship)
    # initializing sets for exploring and disregarding nodes
    came_from: dict[Node, Node] = {}  # will hold parent nodes, empty on init

    # The closed_set will hold previously explored nodes
    closed_set: set[Node] = set()

    # The open set (heap) will hold currently explorable nodes
    # Using binheap, since the smallest element will always
    # be the first, drastically speeding up the search
    open_heap: list[tuple[float, Node]] = []
    heapify(open_heap)

    # We will also be using an actual set, for the O(1) lookup
    open_set: set[Node] = set()

    heappush(open_heap, (f_score[start.indices()], start))
    open_set.add(start)

    neighbors = [(0, 1), (0, -1), (1, 0), (1, 1), (1, -1), (-1, 0), (-1, -1), (-1, 1)]
    # iterating over available nodes
    while open_heap:
        # Get lowest f_score element
        elem: tuple[float, Node] = heappop(open_heap)
        current_node: Node = elem[1]
        # Remove it from the open_set
        open_set.remove(current_node)
        # Add to closed set
        closed_set.add(current_node)

        # If at goal, return the path used
        if current_node.x == goal.x and current_node.y == goal.y:
            print('\tSearch successful!\n')
            path: list[Node] = reconstruct_path(current_node, came_from)
            final_score: Score = elem[0]
            result: SearchResult = path, final_score, True
            return result


        for dx, dy in neighbors:
            # x, y = current_node
            neighbor_y: int = current_node.y + dy
            neighbor_x: int = current_node.x + dx
            neighbor_idx: tuple[int, int] = (neighbor_x, neighbor_y)
            # neighbor_x: int = current_node.x + dx
            # neighbor_index: tuple[int, int] = (neighbor_x, neighbor_y)
            # If neighbor is inside chosen bounds
            if 0 <= neighbor_x < X_BOUND and 0 <= neighbor_y < Y_BOUND:
                # print(neighbor_x)
                neighbor: Node = node_set[neighbor_idx]
                # Proper control flow requires this statement to fail,
                # but the previous to pass
                if np.isnan(neighbor.thickness) or neighbor in closed_set:
                    # No need to explore this node
                    continue
                # if neighbor is in closed set, due to our heuristic,
                # we can guarantee a more optimal path there has been found,
                # so we can skip it
                # if neighbor in closed_set:
                #     continue
                # if neighbor in came_from:
                #     continue
                # At this point the following else-statement will be skipped,
                # and the node will be considered OK
            else:
                # Current node is OOB, proceed to next iteration
                continue
            # Potential g_score of neighbor
            tentative_g_score: float = g_score[current_node.indices()] + cost(current_node, neighbor, ship)  # changes if a smaller cost path is found

            neighbor_idx = neighbor.indices()
            # checks if the current path to neighbor is better than previous (or none)
            if tentative_g_score < g_score[neighbor_idx]:
                # set current node as parent node
                came_from[neighbor] = current_node

                # updating g score
                g_score[neighbor_idx] = tentative_g_score 

                # total cost from start node to goal node via this neighbor
                f_score[neighbor_idx] = g_score[neighbor_idx] + heuristic(neighbor, goal, ship)

                # checking if neighbor has been explored
                if neighbor not in open_set:
                    open_set.add(neighbor)
                    heappush(open_heap, (f_score[neighbor_idx], neighbor))  # add neighbor to set of explorable nodes
                # heappush(open_heap, (f_score[neighbor.indices()], neighbor))

    # if no path has been found 
    print('\tSearch failed!\n')
    path: NodeSet = reconstruct_path(current_node, came_from)
    final_score: Score = elem[0]
    result: SearchResult = path, final_score, False
    return result

def plot_path(path: list[Node], map=None):
    if map == None:
        map = data.init_map()
    if utils.verbose_mode:
        print('\tPlotting path...')
    [data.plot_coord(node.coord.lon, node.coord.lat, map=map)
        for node in path 
        # [(node.coord.lon, node.coord.lat) for node in path]
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

    path, score, search_successful = A_star_search_algorithm(start_coordinate, end_coordinate, ship_class.ship_list[0], fuel_class.fuel_list[0])
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
# start_coordinate: Coordinate = Coordinate(lat = 66.898, lon = -162.596)
# end_coordinate: Coordinate = Coordinate(lat = 68.958, lon = 33.082)

# import distance_correction as dist_corr
import unittest
@define
class Port:
    name: str
    coordinate: Coordinate
    # def __init__(self, name: str, position: Coordinate):
    #     self.name: str = name
    #     self.position: Coordinate = position
    #     # lat, lon = position
    #     # self.coord = Coord(lat, lon)
    def __str__(self) -> str:
        return f"{self.name} {self.coordinate}"
    def get_position(self) -> Coordinate:
        return self.coordinate

class PortBaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ''' On inherited classes, run setUp '''
        if cls is not PortBaseTest and cls.setUp is not PortBaseTest.setUp:
            original_setUp = cls.setUp
            def setUpOverride(self, *args, **kwargs):
                PortBaseTest.setUp(self)
                return original_setUp(self, *args, **kwargs) 
            cls.setUp = setUpOverride

    def setUp(self):
        ''' Invalid to test a port without any data'''
        self.port = None

    def test_coordinate_valid(self):
        if self.port != None:
            self.assertIsInstance(self.port.coordinate, Coordinate)

# @define
class Mongstad(Port):
    # name: str = 'MONG'
    # coordinate = Coordinate(lat = 60.810, lon = 5.032)
    def __init__(self):
        super().__init__('MONG', Coordinate(lat = 60.810, lon = 5.032))

class MongstadTest(PortBaseTest):
    def setUp(self):
        self.port = Mongstad()

# @define
class Mizushima(Port):
    # name: str = 'MIZU'
    # coordinate = Coordinate(lat = 34.504, lon = 133.714)
    def __init__(self):
        super().__init__('MIZU', Coordinate(lat = 34.504, lon = 133.714))

class MizushimaTest(PortBaseTest):
    def setUp(self):
        self.port = Mizushima()

# @define
class Kotzebue(Port):
    # name: str = 'KOTZ'
    # coordinate = Coordinate(lat = 66.898, lon = -162.596)
    def __init__(self):
        super().__init__('KOTZ', Coordinate(lat = 66.898, lon = -162.596))

class KotzebueTest(PortBaseTest):
    def setUp(self):
        self.port = Kotzebue()

class Murmansk(Port):
    def __init__(self):
        super().__init__('MURM', Coordinate(lat = 68.958, lon = 33.082))
class MurmanskTest(PortBaseTest):
    def setUp(self):
        self.port = Murmansk()

class Reykjavik(Port):
    def __init__(self):
        super().__init__('REYK', Coordinate(lat = 64.963, lon = 19.103))
class ReykjavikTest(PortBaseTest):
    def setUp(self):
        self.port = Reykjavik()

class Tasiilap(Port):
    def __init__(self):
        super().__init__('TASI', Coordinate(lat = 65.530, lon = -37.524))
class TasiilapTest(PortBaseTest):
    def setUp(self):
        self.port = Tasiilap()

if __name__ == '__main__':
    unittest.main()

class Route():
    def __init__(self, start: Type[Port], goal: Type[Port]):
        self.start: Type[Port] = start
        self.goal: Type[Port] = goal
    def __str__(self) -> str:
        return f"Route - {self.start} - {self.goal}"

# mongstad: Port = Port('Mongstad', (60.810, 5.032))
# mizushima: Port = Port('Mizushima', (34.504, 133.714))
test_route_1: Route = Route(Mongstad(), Mizushima())
# kotzebue: Port = Port('Kotzebue', (66.898, -162.596))
# murmansk: Port = Port('Murmansk', (68.958, 33.082))
test_route_2: Route = Route(Kotzebue(), Murmansk())
test_route_3: Route = Route(Tasiilap(), Mongstad())
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

def plot_route(route: Route, fuel: Type[Fuel], ship: Type[Ship]):
    print(route.goal.coordinate)
    result: SearchResult = A_star_search_algorithm(route.start.coordinate, route.goal.coordinate, ship, fuel)

    path: NodeSet = result[0]
    plot_path(path, data.init_map())

    score: Score = result[1]
    path_label: str = f"{ship.name} - {fuel.name} - {route} - {score}"
    data.save_coord_map(path_label)
    print(utils.separator)

print('name')
print(__name__)
if __name__ == '__main__':
    print('imma do shit')
    plot_route(test_route_1, fuel_class.fuel_list[0], ship_class.ship_list[0])
    plot_route(test_route_2, fuel_class.fuel_list[0], ship_class.ship_list[0])
    plot_route(test_route_3, fuel_class.fuel_list[0], ship_class.ship_list[0])

def plot_between(start_place, end_place, fuel, ship):
    path, score, search_successful = A_star_search_algorithm(start_place[1], end_place[1], fuel, ship)
    plot_path(path, data.init_map())
    data.save_coord_map(start_place[0] + ' to ' + end_place[0] + '(' + str(score) + ', ' + fuel.name + ')')
def test():
    for fuel in fuel_class.fuel_list:
        plot_route(test_route_1, fuel, ship_class.Prizna())
        plot_route(test_route_2, fuel, ship_class.Prizna())
        plot_route(test_route_3, fuel, ship_class.Prizna())

def profile():
    from scalene import scalene_profiler
    # profiler = cProfile.Profile()
    print("Starting profiling...")
    scalene_profiler.start()
    plot_route(test_route_1, fuel_class.fuel_list[0], ship_class.ship_list[0])
    scalene_profiler.stop()
    # profiler.disable()
    # print("Dumping stats...")
    # stats = pstats.Stats(profiler).sort_stats('ncalls')
    # stats.strip_dirs()
    # stats.dump_stats('new_star_profiling')
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
