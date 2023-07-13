
# Create an algorithm for pathing in the artic
# get array of coordinates, figure out which is the best path A*
# - use ice index to navigate
# - globle land mask to navigate

# Make pathing system with long lat coords
# GPS Global position system
# 
import cartopy as cp
import numpy as np
from load_dataset import longitude, latitude, ice_thickness

# move to load_dataset
longitude = longitude.flatten()
latitude = latitude.flatten()
ice_thickness = np.nan_to_num(ice_thickness) # turn nan values to 0
###################

# https://stackoverflow.com/questions/40342355/how-can-i-generate-a-regular-geographic-grid-using-python
# You are talking about latitude/longitude pairs, which is a polar coordinate system measured in degrees on an approximation of Earth's surface shape.




# Grid in degrees

# step_size_km = 25
# conversion_factor_lat  = 111.32
# conversion_factor_lon = 110.57*np.cos(latitude)

# delta_degree_lat = step_size_km / conversion_factor_lat



# print(np.max(longitude))
# print(np.min(longitude))

# projection = cp.crs.NorthPolarStereo()

# print(projection)




# ESRI:102018 North Pole Stereographic
# https://epsg.io/102018
# EPSG:3575

import shapely.geometry
import pyproj

# lon_min, lon_max = -180, 180
# lat_min, lat_max = 60, 90

# #corners
# sw = shapely.geometry.Point((lon_min,lat_min))
# ne = shapely.geometry.Point((lon_max,lat_max))

# stepsize = 25000 # 25000 km grid resolution


# gridpoints = []
# x = sw[0]
# while x < ne[0]:
#     y = sw[1]
#     while y < ne[1]:
#         p = shapely.geometry.Point()



import matplotlib.pyplot as plt

plt.imshow(ice_thickness,cmap = 'jet')
plt.show()


