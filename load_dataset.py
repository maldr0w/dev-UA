
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset




def load_data(nc_file)
        # Access the necessary variables
        ice_thickness = nc_file.variables['sea_ice_thickness']
        latitude = nc_file.variables['lat']
        longitude = nc_file.variables['lon']
        
        # Read the data
        ice_thickness = ice_thickness[:].squeeze()
        latitude = latitude[:]
        longitude = longitude[:]    

        lat_grid = np.arange(latitude.shape[0])
        lon_grid = np.arange(longitude.shape[0])

        return

        




def plot(nc_file): #workign tittles
        # Access the necessary variables
        ice_thickness = nc_file.variables['sea_ice_thickness']
        latitude = nc_file.variables['lat']
        longitude = nc_file.variables['lon']
        
        # Read the data
        ice_thickness = ice_thickness[:].squeeze()
        latitude = latitude[:]
        longitude = longitude[:]    

        lat_grid = np.arange(latitude.shape[0])
        lon_grid = np.arange(longitude.shape[0])
        
        plt.figure(figsize=(10,6))
        plt.pcolormesh(lon_grid, lat_grid, ice_thickness, cmap='jet')
        plt.colorbar(label='Ice thickness (m)')
        plt.gca().invert_yaxis()
        plt.xlabel('Longitude (degrees east)')
        plt.ylabel('Latitude (degrees north)')
        plt.title('Arctic Region Sea Ice')
        plt.grid(True)
        
        plt.show()  


nc_file_22 = Dataset('ice_thickness_2022.nc', 'r')
nc_file_21 = Dataset('ice_thickness_2021.nc', 'r')

plot(nc_file_22)
plot(nc_file_21)
