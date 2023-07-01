
# A* Search Algorithm
import numpy as np
from noisemap import world

# Heuristic (manhattan distance)
def heuristic(node,goal):
    x1, y1 = node
    x2, y2 = goal
    return abs(x2 - x1) + abs(y2 - y1)


# A* search algorithm
def a_star_search(start,goal):
    open_set = {start}  # create open set, unordered list, will hold potential nodes
    closed_set = set()  # empty set, for explored nodes
    came_from = {}      # store parent node for each node 
    gscore = {start: 0} # cost from start node to each node, start is keyed 0
    fscore = {start: heuristic(start)} # estimated total cost from start to goal via each node

    while open_set: # while there are available nodes
        current_node = min(open_set,key=lambda node: fscore[node]) # set node with smallest fscore from open_set to current
        
    



#runscript

