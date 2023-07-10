
from asearch import a_star_search
from netCDF4 import Dataset

import matplotlib.pyplot as plt
import numpy as np
import cartopy as cp

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import xarray as xr


# # Open the NetCDF file
# dataset = Dataset('ice_thickness_2021.nc', 'r')


# # Read latitude, longitude, and ice thickness variables
# latitude = dataset.variables['lat']
# longitude = dataset.variables['lon'][:]
# ice_thickness = dataset.variables['sea_ice_thickness'][:].squeeze()

# # Close the NetCDF file
# dataset.close()

# latitude = np.arange(latitude.shape[0])
# longitude = np.arange(longitude.shape[0])


# Creating map
fig, ax = plt.subplots(1,1,figsize=(8,8), 
            subplot_kw={'projection':cp.crs.NorthPolarStereo()})

ax.set_extent([-180,180,90,66], cp.crs.PlateCarree())
ax.add_feature(cp.feature.LAND, edgecolor='black', zorder=1)
ax.set_facecolor((1.0,1.0,1.0))

# ax.coastlines()

gl = ax.gridlines(draw_labels=True, alpha=0.5,color='gray', 
             linestyle='-', linewidth=0.5, 
             xlocs=np.arange(-180, 181, 30),
             ylocs=np.arange(60, 91, 5))

# gl.xlabels_top = False
# gl.ylabels_left = False
# gl.ylabels_right = True

gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}


# plt.pcolormesh(longitude, latitude, ice_thickness, cmap='jet')

# plotting data
# ax.legend(fontsize='x-large')

# ax.scatter(longitude,latitude,
#            color='k', marker='x',
#            transform = cp.crs.PlateCarree(),
#            zorder=5, label='Arctic Sea Ice')

#####


dataset = Dataset('ice_thickness_2021.nc', 'r')
ice_thickness = dataset['sea_ice_thickness'][:].squeeze()
latitude = dataset['lat'][:]
longitude = dataset['lon'][:]

# print(ice_thickness.shape)
# plt.pcolormesh(longitude, latitude, ice_thickness, cmap='jet', vmin = np.max(ice_thickness), vmax = np.min(ice_thickness))

# masked_value = 0
# masked_ice_thickness = np.ma.masked_values(ice_thickness, masked_value)


# plt.pcolormesh([longitude, latitude], ice_thickness, cmap='jet')
cs = plt.pcolormesh(longitude, latitude, ice_thickness, cmap='jet')

plt.contourf(longitude, latitude, ice_thickness, 60, transform = cp.crs.PlateCarree(), vmin=np.min(ice_thickness), vmax=np.max(ice_thickness))
plt.colorbar(cs)


# extent = [longitude.min(), longitude.max(), latitude.min(), latitude.max()],
# ax.add_feature(states_fill, linewidth=0.45, zorder=0)

# Set the masked value
# masked_value = -9999

# # Create a masked array for ice thickness
# masked_ice_thickness = np.where(ice_thickness != masked_value, ice_thickness, np.nan)

# # Plot the masked ice thickness data
# im = ax.imshow(ice_thickness, cmap='jet', extent=[longitude.min(), longitude.max(), latitude.min(), latitude.max()],
#      transform=cp.crs.PlateCarree())

# p = plt.gca()
# p.set_facecolor((1.0,1.0,1.0))
plt.show()

