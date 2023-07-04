# Dev: Martin Stave, July 01, 2023 - uartic.py
# Uartic Project 


import numpy as np
import matplotlib.pyplot as plt


from asearch import a_star_search, graph_path
from noisemap import world

#  test 
# start = (0,0)
# goal = (98,98)
# path = a_star_search(start,goal,world)

# graph_path(path, world)
# plt.show()


# Starting on data aquizition







# Data aquizition
# from netCDF4 import Dataset

# nc_file = Dataset('ice_thickness.nc','r') 

# # Finding variable names
# variable_names = list(nc_file.variables.keys())
# # for var_name in variable_names:
# #     print(var_name)

# # Access data varibales
# ice_thickness = nc_file.variables['sea_ice_thickness'] 
# latitude = nc_file.variables['lat']
# longitude = nc_file.variables['lon']

# quality_flag = nc_file.variables['quality_flag']
# status_flag = nc_file.variables['status_flag']

# uncertainty = nc_file.variables['uncertainty']
# time = nc_file.variables['time']
# time_bnds = nc_file.variables['time_bnds']

# xc = nc_file.variables['xc']  # believe these are x-y coordinates
# yc = nc_file.variables['yc'] #

# lambert_azimuthal_grid = nc_file.variables['Lambert_Azimuthal_Grid']

# # Access data
# metadata = nc_file.__dict__
# # print(metadata)


# # grid spacing of 25.0 km

# ice_thickness_data = ice_thickness[:]
# longitude_data = longitude[:]
# latitude_data = latitude[:]


# # print("Latitude Range:", np.min(latitude_data), np.max(latitude_data))
# # print("Longitude Range:", np.min(longitude_data), np.max(longitude_data))




# lat_min = 75.0
# lat_max = 76.0
# lon_min = 29.0
# lon_max = 30.0

# # Find the indices corresponding to the desired latitude and longitude range
# lat_indices = np.where((latitude_data >= lat_min) & (latitude_data <= lat_max))[0]
# lon_indices = np.where((longitude_data >= lon_min) & (longitude_data <= lon_max))[0]

# # Subset the latitude and longitude arrays
# latitude_subset = latitude_data[lat_indices]
# longitude_subset = longitude_data[lon_indices]

# # Create meshgrid using the subsetted latitude and longitude
# lon_subset, lat_subset = np.meshgrid(longitude_subset, latitude_subset)

# # Plot the ice thickness data
# plt.contourf(lon_subset, lat_subset, ice_thickness_data[lat_indices, lon_indices])
# plt.colorbar(label='Ice Thickness (m)')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Arctic Sea Ice Thickness')
# plt.show() 






# Data acquisition
# from netCDF4 import Dataset
# import numpy as np
# import matplotlib.pyplot as plt

# # Open the netCDF file
# nc_file = Dataset('ice_thickness.nc', 'r')

# # Access data variables
# ice_thickness = nc_file.variables['sea_ice_thickness']
# latitude = nc_file.variables['lat']
# longitude = nc_file.variables['lon']

# # Subset the latitude and longitude ranges
# lat_min = 75.0
# lat_max = 76.0
# lon_min = 29.0
# lon_max = 30.0


# lat = latitude[:]
# lon = longitude[:]


# # Find the indices corresponding to the desired latitude and longitude range
# lat_indices = np.where((lat >= lat_min) & (lat <= lat_max))[0]
# lon_indices = np.where((lon >= lon_min) & (lon <= lon_max))[0]

# # Subset the latitude and longitude arrays
# latitude_subset = latitude[lat_indices]
# longitude_subset = longitude[lon_indices]

# # Access the ice thickness data within the subsetted region
# ice_thickness_subset = ice_thickness[:, lat_indices, lon_indices]

# # # Plot the ice thickness data
# lon_mesh, lat_mesh = np.meshgrid(longitude_subset, latitude_subset)

# # plt.contourf(lon_mesh, lat_mesh, ice_thickness_subset[0])
# # plt.colorbar(label='Ice Thickness (m)')
# # plt.xlabel('Longitude')
# # plt.ylabel('Latitude')
# # plt.title('Arctic Sea Ice Thickness')
# # plt.show()

# # Close the netCDF file
# nc_file.close()




import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# Open the netCDF file
nc_file = Dataset('ice_thickness.nc', 'r')

# Access the necessary variables
ice_thickness = nc_file.variables['sea_ice_thickness']
latitude = nc_file.variables['lat']
longitude = nc_file.variables['lon']

# Read the data
ice_thickness_data = ice_thickness[:].squeeze()
latitude_data = latitude[:]
longitude_data = longitude[:]

# Plot the ice thickness data as an image
plt.imshow(ice_thickness_data, cmap='viridis', origin='lower',
           extent=[longitude_data.min(), longitude_data.max(),
                   latitude_data.min(), latitude_data.max()])
plt.colorbar(label='Ice Thickness (m)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Arctic Sea Ice Thickness')
plt.show()