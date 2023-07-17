
# Create an algorithm for pathing in the artic
# get array of coordinates, figure out which is the best path A*
# - use ice index to navigate
# - globle land mask to navigate

# Make pathing system with long lat coords
# GPS Global position system
# 
import cartopy as cp
import numpy as np
import pyproj
import shapely.geometry

from load_dataset import longitude, latitude, ice_thickness

# g_star.py - UIT - Martin Stave - mvrtinstave@gmail.com
# inititates and runs the A* algorithm on sea ice thickness dataset





# move to load_dataset
longitude = longitude.flatten()
latitude = latitude.flatten()
ice_thickness = np.nan_to_num(ice_thickness)  # turn nan values to 0
###################

# https://stackoverflow.com/questions/40342355/how-can-i-generate-a-regular-geographic-grid-using-python
# You are talking about latitude/longitude pairs, which is a polar coordinate system measured in degrees on an approximation of Earth's surface shape.



# initializing transformation of coordinate systems
proxy_transformer = pyproj.Transformer.from_crs('epsg:4326','epsg:3857')
original_transformer = pyproj.Transformer.from_crs('epsg:3857','epsg:4326')

# set up coordinates

# coords = np.stack(longitude, latitude)

# print(coords)


stepsize = 5000  # 5km grid resolution

# transform corners to projection


# iterate over 2D area
# grid = []
# x = transformed_sw[0]
# while x <transformed_ne[0]:
#     y = transformed_sw[1]
#     while y < transformed_ne[1]:
#         p = shapely.geometry.Point(original_transformer.transform(x,y))
#         grid.append(p)
#         y+= stepsize
#     x+= stepsize



