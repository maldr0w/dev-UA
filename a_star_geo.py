
from asearch import a_star_search
from netCDF4 import Dataset

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Open the NetCDF file
dataset = Dataset('ice_thickness_2021.nc', 'r')


# # Read latitude, longitude, and ice thickness variables
latitude = dataset.variables['lat'][0]
longitude = dataset.variables['lon'][0]
thickness = dataset.variables['sea_ice_thickness'][:].squeeze()

# # Close the NetCDF file
# dataset.close()

# # Create a map plot
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# # Plot ice thickness data as scatter points on the map
# scatter = ax.scatter(longitude, latitude, c=ice_thickness, cmap='cool')

# # Add colorbar
# cbar = plt.colorbar(scatter, ax=ax, label='Ice Thickness')

# # Set the map title and labels
# ax.set_title('Ice Thickness Map')
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')

# # Show the plot
# plt.show()

# print(latitude.data)
# print(np.median(longitude.data))


from mpl_toolkits.basemap import Basemap
from itertools import chain

def draw_map(m, scale=0.2):
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)
    
    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))

    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)
    
    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')






fig = plt.figure(figsize=(8,8))
# m = Basemap(projection='lcc', resolution = None,
#             lat_0 = np.median(latitude.data), lon_0 = np.median(longitude.data),
#             llcrnrlon = np.min(longitude.data), llcrnrlat=np.min(latitude.data),
#             urcrnrlon=np.max(longitude.data), urcrnrlat=np.max(latitude.data))
m = Basemap(projection='lcc', resolution = None,
            width = 8E6, height=8E6,
            lat_0 = 79.037, lon_0 = 59.164)
            



# m.scatter(longitude.data, latitude.data, latlon=True, c=thickness[0], s=25000*25000, cmap='Reds', alpha=0.5)
# m.scatter(np.median(latitude.data), np.median(longitude.data))
plt.pcolormesh(longitude.data, latitude.data,thickness, cmap='jet')
plt.colorbar()
plt.xlabel('lon')
plt.ylabel('lat')
draw_map(m)
plt.show()


