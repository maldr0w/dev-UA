

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
plt.imshow(ice_thickness_data, aspect='auto', cmap='viridis')
           #, extent=[longitude_data.min(), longitude_data.max(),
                   # latitude_data.min(), latitude_data.max()])
plt.colorbar(label='Ice Thickness (m)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Arctic Sea Ice Thickness')
plt.show()



