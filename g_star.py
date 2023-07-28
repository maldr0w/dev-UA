
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



# mecator projection - meters
# square divided in x and y axis


# move to load_dataset
longitude = longitude.flatten()
latitude = latitude.flatten()
ice_thickness = np.nan_to_num(ice_thickness)  # turn nan values to 0

# print(longitude)
# print(latitude)

# print(max(longitude), min(longitude))
# print(max(latitude), min(latitude))

def LatlonToMeters(lat,lon):
    mx = lon/180 * 20037508.34  # 20 million meters

    my = np.log(np.tan((90+latitude)*np.pi/360))/(np.pi/180)
    my = my* 20037508.34 / 180

    return mx, my

t = LatlonToMeters(latitude,longitude)

# print(t)

mx = np.ma.masked_invalid(t)
# print(repr(mx))
print(mx)
print(mx[0][0]+mx[1][0])


# https://github.com/mehrdadn/SOTA-Py/tree/master























###################

# https://stackoverflow.com/questions/40342355/how-can-i-generate-a-regular-geographic-grid-using-python
# You are talking about latitude/longitude pairs, which is a polar coordinate system measured in degrees on an approximation of Earth's surface shape.



# initializing transformation of coordinate systems
proxy_transformer = pyproj.Transformer.from_crs('epsg:4326','epsg:3857')  # transforming from lat, lon to meters
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






