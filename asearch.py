
# A* Search Algorithm
import numpy as np
from noisemap import world

# Heuristic (manhattan distance)
def heuristic(node,goal):
    x1, y1 = node
    x2, y2 = goal
    return abs(x2 - x1) + abs(y2 - y1)


def reconstruct_path(came_from, current_node):
    path = [current_node] # creating a path putting each visited node in a list
    while current_node 
    
    
def get_neighbors(node):
    neighbors = []
    x, y = node
    
    # Adding each neighbor   
    neighbors.append(x,y+1) # up
    neighbors.append(x,y-1) # down
    neighbors.append(x+1,y) # right
    neighbors.append(x-1,y) # left
    
    return neighbors

def cost_between(node1,node2):
    



# A* search algorithm
def a_star_search(start, goal):
    open_set = {start}  # create open set, unordered list, will hold potential nodes
    closed_set = set()  # empty set, for explored nodes
    came_from = {}      # store parent node for each node 
    gscore = {start: 0} # cost from start node to each node, start is keyed 0
    fscore = {start: heuristic(start)} # estimated total cost from start to goal via each node

    while open_set: # while there are available nodes
        current_node = min(open_set, key=lambda node: fscore[node]) # set node with smallest fscore from open_set to current,                                                                  
                                                                    # min fscore = node with lowest total cost to reach goal



        # Having found the lowest fscore node:
         
        open_set.remove(current_node) # we remove it from set of possible nodes
        closed_set.add(current_node)  # and add to explored nodes 


        for neighbor in get_neighbors(current_node):
            if neighbor in closed_set:
                continue # move to next neighbor
            
            tentative_gscore = gscore[current_node] + cost_between(current, neighbor) # calculate gscore






#runscript

