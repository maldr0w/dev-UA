
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
ice_thickness = np.nan_to_num(ice_thickness)
###################

# https://stackoverflow.com/questions/40342355/how-can-i-generate-a-regular-geographic-grid-using-python
# You are talking about latitude/longitude pairs, which is a polar coordinate system measured in degrees on an approximation of Earth's surface shape.




# Grid in degrees

step_size_km = 25
conversion_factor_lat  = 111.32
conversion_factor_lon = 110.57*np.cos(latitude)

delta_degree_lat = step_size_km / conversion_factor_lat



# print(np.max(longitude))
# print(np.min(longitude))

# projection = cp.crs.NorthPolarStereo()

# print(projection)




# ESRI:102018 North Pole Stereographic
# https://epsg.io/102018
# EPSG:3575

import shapely.geometry
import pyproj


coords = np.column_stack((longitude, latitude))


