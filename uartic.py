# Dev: Martin Stave, July 01, 2023 - uartic.py
# Uartic Project 

import numpy as np
import matplotlib.pyplot as plt

from asearch import a_star_search
from noisemap import noisemap


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
    plt.imshow(map, cmap='gray',origin='lower', interpolation='nearest')
    plt.plot(*zip(*path), color='red') # zip fixes this line somehow       
    plt.show()


start = (0,0)
goal = (98,98)

path = a_star_search(start, goal, noisemap)
reduced_path = reduce_path(path)

draw_path(reduced_path, noisemap)



