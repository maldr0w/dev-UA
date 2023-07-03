# Dev: Martin Stave, July 01, 2023 - uartic.py
# Uartic Project 


import numpy as np
import matplotlib.pyplot as plt


from asearch import a_star_search, graph_path
from noisemap import world

#  test 
start = (0,0)
goal = (98,98)
path = a_star_search(start,goal,world)

graph_path(path, world)
plt.show()






# Data aquizition
# from netCDF4 import Dataset

# nc_file = Dataset('thk_2011_octnov.map.nc','r') 

# # Finding variable names
# variable_names = list(nc_file.variables.keys())
# for var_name in variable_names:
#     print(var_name)

# nc_file.close()



# # Access data varibales
# thickness = nc_file.variables['thickness'] 
# latitude = nc_file.variables['latitude']
# longitude = nc_file.variables['longitude']

# grid_spacing=nc_file.variables['grid_spacing']


# # Access data
# thickness = thickness[:]
# print(thickness.type())