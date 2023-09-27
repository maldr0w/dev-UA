from mpl_toolkits.basemap import Basemap
import math
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import utils

utils.print_entrypoint(__name__, __file__)

DATA_FILE = 'ice_thickness_2022.nc'

dataset = xr.open_dataset(DATA_FILE)
meters = (np.shape(dataset['sea_ice_thickness'].coords['xc'])[0] - 1) * 25000
print(np.shape(dataset['sea_ice_thickness'].coords['xc'])[0] * 25)
# projdata = dataset['Lambert_Azimuthal_Grid']
# meters = 10775000
def reinit_map():
    ax, fig = plt.subplots()
    m = Basemap(width=meters, height=meters, projection='laea', lon_0=0., ellps='WGS84', lat_0=90., resolution='i')
    # m.drawcoastlines()
    # m.fillcontinents(color='gray',lake_color='gray')
    m.drawparallels(np.arange(16.6239,90.,20.))
    m.drawmeridians(np.arange(-180.,180.0,20.))
    m.drawmapboundary(fill_color='gray')
    return m


def plot_coord(lon, lat, lon_west=False, c='r', map=None):
    if lon_west:
        lon = 360. - lon
    x, y = map(lon, lat)
    map.plot(x, y, marker=',', color=c)

TIME = 0
icedata = dataset['sea_ice_thickness'].values[TIME]

# print(icedata)

def shift(c):
    return (1000 * c) + (meters / 2)

vshift = np.vectorize(shift)

print("Shifting coordinates...")
x_coords = vshift(dataset['sea_ice_thickness'].coords['xc'])
y_coords = vshift(dataset['sea_ice_thickness'].coords['yc'])
print("Finished")

print("Filtering out land cells...")

# Make a new map to store all the data, including ocean tiles

mapdata = np.empty(np.shape(icedata))

for y, row in enumerate(dataset['status_flag'][TIME].values):
    for x, status_flag in enumerate(row):
        match status_flag:
            # If flag is 1: No data
            #            2: Open ocean
            #            5: Retrieval failed
            # We simply assume these to be okay for now
            # case 1 | 2 | 5:
            case 1 | 2:
                mapdata[y, x] = -0.5
            case _:
                mapdata[y, x] = icedata[y, x]
print("Finished.")

print("Truncating values...")

# Since they are in range -0.5 to 11.0 we just shift a bit

for y, row in enumerate(mapdata):
    for x, datapoint in enumerate(row):
        mapdata[y, x] = datapoint + 0.5

print("Finished.")

print("Generating image for the ice thickness map...")

# Make a mesh corresponding to the entire map for creating a figure

xx, yy = np.meshgrid(x_coords, y_coords)

# Change this if needed, just a placeholder for now


print("Finished.")
# Draw the colormesh
def reinit_colormesh(map):
    cmap = mpl.cm.jet
    cmap.set_bad('black',1.)
    map.pcolormesh(xx, yy, mapdata, norm='symlog', shading='nearest', antialiased=True, cmap=cmap)
    
def init_map():
    map = reinit_map()
    reinit_colormesh(map)
    return map

# temp = 180. + (180. - 170.804)

# plot_coord(170.804, 61.392, lon_west=True)

# for lon in np.arange(0, 180):
#     plot_coord(lon, 61.392)

# for lat in np.arange(0, 90):
#     plot_coord(170.804, lat)
def save_coord_map(name):
    plt.title(str(name) + " map")
    plt.show()
    print("Saving figure...")
    plt.savefig('images/' + str(name) + '.png')
    print("Finished.")
    # plt.close()
    utils.print_separator()

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

print("Finished.")

# print("Generating an image for the distance correction map...")

# m.pcolormesh(xx, yy, distance_correction_map, norm='symlog', shading='nearest', antialiased=True, cmap=cmap)

# print("Finished\n")

# plt.title("Distance correction map")
# plt.show()
# print("Saving figure...")
# plt.savefig('images/distancecorrectioncolormesh.png')
# print("Finished")

utils.print_exit(__name__, __file__)