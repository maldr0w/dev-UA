
# A* Search Algorithm
import numpy as np
import matplotlib.pyplot as plt

# Heuristic (manhattan distance) estimate
def heuristic(node,goal):
    x1, y1 = node
    x2, y2 = goal
    return abs(x2 - x1) + abs(y2 - y1)


def reconstruct_path(came_from, current_node):
    path = [current_node] # initializing with the current_node
    while current_node in came_from:
        current_node = came_from[current_node]
        path.insert(0,current_node)
    return path
    
    
def get_neighbors(node):
    neighbors = []
    x, y = node  
    neighbors.append((x,y+1)) # up
    neighbors.append((x,y-1)) # down
    neighbors.append((x+1,y)) # right
    neighbors.append((x-1,y)) # left
    return neighbors


def cost_between(node1,node2): # more accurate estimate between nodes/ euclidean distance
    cost = 0
    x1, y1 = node1
    x2, y2 = node2
    cost = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return cost


# A* search algorithm
def a_star_search(start, goal, map):
    open_set = {start}  # create open set, unordered list, will hold potential nodes
    closed_set = set()  # empty set, for explored nodes
    came_from = {}      # store parent node for each node 
    gscore = {start: 0} # cost from start node to each node, start is keyed 0
    fscore = {start: heuristic(start, goal)} # estimated total cost from start to goal via each node

    while open_set: # while there are available nodes
        current_node = min(open_set, key=lambda node: fscore[node]) # set node with smallest fscore from open_set to current,                                                                  
                                                                    # min fscore = node with lowest total cost to reach goal
        if current_node == goal:
            return reconstruct_path(came_from, current_node)


        # Having found the lowest fscore node:
        open_set.remove(current_node) # we remove the node from set of possible nodes
        closed_set.add(current_node)  # and add it to explored nodes 


        for neighbor in get_neighbors(current_node):
            if neighbor in closed_set:
                continue # move to next neighbor

            # if neighbor value is more than 2, check another node.
            # we cannot move across fields with higher value than 2..
            if map[neighbor[0]][neighbor[1]] > 2:
                # print('ice')
                continue
            
            tentative_gscore = gscore[current_node] + cost_between(current_node, neighbor) # calculate gscore

            # check if neighbor has been explored
            # if the tentative score is less than real gscore, new score is more optimal 
            # smaller gscore = closer to start along current path, larger = path cost is larger
            if neighbor not in open_set or tentative_gscore < gscore[neighbor]:
                
                came_from[neighbor] = current_node # set the current node as parent node
                
                gscore[neighbor] = tentative_gscore # update gscore
                fscore[neighbor] = tentative_gscore + heuristic(neighbor, goal) # => total cost from start to goal via this path (via neighbor)

                if neighbor not in open_set: # Thus we add this node to explorable nodes
                    open_set.add(neighbor)   # for future iterations


    return None # No path found     

