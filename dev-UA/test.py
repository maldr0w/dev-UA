
import numpy as np
import matplotlib.pyplot as plt
import noise
# a_star.py - UIT - Martin Stave - mvrtinstave@gmail.com
# A* search algorithm tested on random noisemap


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



def a_star_search(start, goal, map):
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


def reduce_path(path):
    '''
    reduces the path provided by the A* 
    by removing redundant points    
        '''
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
    '''
    draws the path created by the A* onto the noisemap
    '''
    plt.figure().set_figwidth(10)
    plt.title("A* algorithm on random noisemap")
    plt.subplot(1, 2, 1)
    plt.imshow(map, cmap='gray', origin='lower', interpolation='nearest')
    plt.title('Random Noisemap')
    plt.subplot(1, 2, 2)
    plt.imshow(map, cmap='gray', origin='lower', interpolation='nearest')
    plt.plot(*zip(*path), color='red')
    plt.title('A* algorithm')
    plt.show()

# Testing algorithm on random noisemap

def random_noisemap(scale, shape):
    '''
    create a noisemap to mimic sea ice thickness
        '''
    # randomized variables
    octaves = np.random.randint(2, 20)  # number of layered noise functions
    persistence = np.random.uniform(0.2, 0.9)  # influence of each octave on final noisemap
    lacunarity = np.random.uniform(1.5, 5.5)  # frequency or detail level of each octave
    noisemap = np.zeros(shape)  # empty array of zeros

    # iterating over each element in the array
    for i in range(shape[0]):
        for j in range(shape[1]):
            # generate perlin noisemap using the predefined parameters
            # i/scale and j/scale adjusts frequency of noise pattern
            # repeatx and repeaty controls repetition of noise pattern
            value = noise.pnoise2(i/scale, j/scale, octaves = octaves, persistence = persistence, 
                              lacunarity = lacunarity, repeatx = shape[0], repeaty = shape[1], base = 0)

            # perlin native range is [-1,1] but we change it to [0,3] to mimic ice thickness
            mapped_value = np.interp(value, [-1,1], [0.0, 3.0])
        
            # store current value in corresponding position within noisemap
            noisemap[i][j] = mapped_value

    return noisemap

scale = 10.0
shape = (100,100)
noisemap = random_noisemap(scale,shape)

start, goal = (0,0), (98,98)
path = a_star_search(start, goal, noisemap)

draw_path(path, noisemap)

