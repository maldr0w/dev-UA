from mpl_toolkits.basemap import Basemap
import math
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt

def print_separator():
    s = 68 * '='
    print('\n<' + s + '>\n')

print_separator()

dataset = xr.open_dataset('ice_thickness_2022.nc')
projdata = dataset['Lambert_Azimuthal_Grid']
meters = 10775000
# meters = 10800000
# meters = 10825000
m = Basemap(width=meters, height=meters, projection='laea', lon_0=0., ellps='WGS84', lat_0=90., resolution='i')
# m.drawcoastlines()
# m.fillcontinents(color='gray',lake_color='gray')
m.drawparallels(np.arange(16.6239,90.,20.))
m.drawmeridians(np.arange(-180.,180.0,20.))
m.drawmapboundary(fill_color='gray')



and_x, and_y = m(16.118056, 69.313889)
m.plot(and_x, and_y, marker='o', color='r')



_time = 0
icedata = dataset['sea_ice_thickness'].values[_time]


print("Shifting coordinates...")
def shift(c):
    return (1000 * c) + (meters / 2)
vshift = np.vectorize(shift)

x_coords = vshift(dataset['sea_ice_thickness'].coords['xc'])
y_coords = vshift(dataset['sea_ice_thickness'].coords['yc'])
print("Finished\n")


flags = dataset['status_flag'].values[_time]

print("Filtering out land cells...")

# Make a new map to store all the data, including ocean tiles

mapdata = np.empty(np.shape(icedata))

for y, row in enumerate(flags):
    for x, flag in enumerate(row):
        match flag:
            case 1 | 2 | 5:
                mapdata[y, x] = -0.5
            case _:
                mapdata[y, x] = icedata[y, x]
print("Finished\n")

print("Generating image for the ice thickness map...")

# Make a mesh corresponding to the entire map for creating a figure

xx, yy = np.meshgrid(x_coords, y_coords)

# Change this if needed, just a placeholder for now

cmap = mpl.cm.jet
cmap.set_bad('black',1.)

# Draw the colormesh

m.pcolormesh(xx, yy, mapdata, norm='symlog', shading='nearest', antialiased=True, cmap=cmap)

print("Finished\n")

plt.title("Ice thickness map")
plt.show()
print("Saving figure...")
plt.savefig('images/mapdatacolormesh.png')
print("Finished")

print_separator()

print("Generating the distance correction map...")
# Start correcting the distance data

# The following formulas are based on polynomial extrapolation of
#   thickness 0 => 1
#   thickness 1 => 1.2
#   thickness 2 => 1.6
#  
# based on the numbers given by hadi

def pn_1(x):
    term = (x ** 2) - (3 * x) + 2
    return term / 2

def pn_2(x):
    term = (1.2 * (x ** 2)) - (2.4 * x)
    return (-1 * term)

def pn_3(x):
    term = (0.8 * (x ** 2)) - (0.8 * x)
    return term

def pn_of(x):
    sum = pn_1(x) + pn_2(x) + pn_3(x)
    return sum

# print(pn_of(0), pn_of(1), pn_of(2))

def ice_thickness_to_correction_factor(thickness):
    # If the ice thickness is less than 0,
    # it must either be ocean or something else
    # thus just run the function clamped to 0
    if thickness < 0:
        return pn_of(0)

    # And in any other case, just get the extrapolated data
    else:
        return pn_of(thickness)

# Initialize a distance_correction_map based on the converted map data

distance_correction_map = np.empty(np.shape(mapdata))

# Iterate through the converted map data,

for y, data_row in enumerate(mapdata):
    for x, datapoint in enumerate(data_row):
        # If the current data point is not nan, then we can
        # convert the given ice thickness to a distance correction factor
        if not math.isnan(datapoint):
            distance_correction_map[y, x] = ice_thickness_to_correction_factor(datapoint)

        # If the data point is nan however, then we want to keep it nan
        # as the current implementation uses nan as land (so boats do not care)
        else:
            distance_correction_map[y, x] = math.nan

print("Finished\n")

print("Generating an image for the distance correction map...")

m.pcolormesh(xx, yy, distance_correction_map, norm='symlog', shading='nearest', antialiased=True, cmap=cmap)

print("Finished\n")

plt.title("Distance correction map")
plt.show()
print("Saving figure...")
plt.savefig('images/distancecorrectioncolormesh.png')
print("Finished")

print_separator()

# def colorize(f, t):
#     match f:
#         # Nominal retrieval
#         case 0:
#             # return ((t + 0.5) / 11.0, 0.11, 0.25)
#             return ([(t + 0.5) / 11.0, 0.11, 0.25])

#         # No data
#         # case 1:
#         #     return (0.0, 0.0, 0.0)

#         # Open ocean
#         case 2:
#             # return (0.0, 0.0, 0.5)
#             return ([0.0, 0.0, 0.5])

#         # Sattelite pole hole
#         # case 3:
#         #     return (0.0, 0.0, 0.0)
        
#         # Land, lake, or land ice
#         case 4:
#             # return (0.0, 0.5, 0.0)
#             return ([0.0, 0.5, 0.0])

#         # Retrieval failed
#         # case 5:
#         #     return (0.0, 0.0, 0.0)

#         case _:
#             # return (0.0, 0.0, 0.0)
#             return ([0.0, 0.0, 0.0])

# ax = plt.gca()
# icedata = dataset['sea_ice_thickness']
# # m.pcolormesh(icedata.coords['xc'], icedata.coords['yc'], icedata.values[0])
# data_table = icedata.values[0]
# x_coords = icedata.coords['xc']
# y_coords = icedata.coords['yc']
# lat_list = icedata.coords['lat']
# lon_list = icedata.coords['lon']
# data_list = []
