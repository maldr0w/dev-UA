from mpl_toolkits.basemap import Basemap
import math
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import utils

utils.print_entrypoint(__name__, __file__)
print('\tInitializing data...')
DATA_FILE = 'ice_thickness_2022.nc'
# Open xArray dataset, find the size in meters
dataset = xr.open_dataset(DATA_FILE)
meters = (np.shape(dataset['sea_ice_thickness'].coords['xc'])[0] - 1) * utils.unit_distance

# print(np.shape(dataset['sea_ice_thickness'].coords['xc'])[0] * 25)
 
def reinit_map():
    '''
    Reinitialize map for plot points.
    Largely boiler-plate, but made into a function for ease of calling's sake.
    '''   
    ax, fig = plt.subplots()
    m = Basemap(width=meters, height=meters, projection='laea', lon_0=0., ellps='WGS84', lat_0=90., resolution='i')
    m.drawparallels(np.arange(16.6239,90.,20.))
    m.drawmeridians(np.arange(-180.,180.0,20.))
    m.drawmapboundary(fill_color='gray')
    return m


def plot_coord(lon, lat, lon_west=False, c='r', m=',', map=None):
    '''
    Function to plot a coordinate on the map given.
    '''
    if lon_west:
        lon = 360. - lon
    x, y = map(lon, lat)
    map.plot(x, y, marker=m, color=c)

# Access icedata (for some reason, there is a time dimension)
TIME = 0
icedata = dataset['sea_ice_thickness'].values[TIME]

# Shift coordinates based on the finding in terms of scale
# This makes the top-left 0 m, 0 m
print("Shifting coordinates...")
def shift(c):
    return (1000 * c) + (meters / 2)

vshift = np.vectorize(shift)
x_coords = vshift(dataset['sea_ice_thickness'].coords['xc'])
y_coords = vshift(dataset['sea_ice_thickness'].coords['yc'])

# Make a new map to store all the data, including ocean tiles

ice_values = np.empty(np.shape(icedata))

for y, row in enumerate(dataset['status_flag'][TIME].values):
    for x, status_flag in enumerate(row):
        match status_flag:
            # If flag is 1: No data (mostly ocean outside the satellites range)
            #            2: Open ocean
            case 1 | 2:
                ice_values[y, x] = -0.5
            case _:
                ice_values[y, x] = icedata[y, x]

# Since they are in range -0.5 to 11.0 we just shift a bit

for y, row in enumerate(ice_values):
    for x, datapoint in enumerate(row):
        ice_values[y, x] = datapoint + 0.5

# Make a mesh corresponding to the entire map for creating a figure
xx, yy = np.meshgrid(x_coords, y_coords)
# Draw the colormesh
def reinit_colormesh(map):
    cmap = mpl.cm.jet
    cmap.set_bad('black',1.)
    map.pcolormesh(xx, yy, ice_values, norm='symlog', shading='nearest', antialiased=True, cmap=cmap)
    
def init_map():
    map = reinit_map()
    reinit_colormesh(map)
    return map

def save_coord_map(name):
    plt.title(str(name) + " map")
    plt.show()
    if utils.verbose_mode:
        print("\tSaving figure...")
    plt.savefig('images/' + str(name) + '.png')
    if utils.verbose_mode:
        print("\tFinished.")
    plt.close()
    if utils.verbose_mode:
        utils.print_separator()


# Initialize a distance_correction_map based on the converted map data

# distance_correction_map = np.empty(np.shape(ice_values))

# # Iterate through the converted map data,

# for y, data_row in enumerate(ice_values):
#     for x, datapoint in enumerate(data_row):
#         # If the current data point is not nan, then we can
#         # convert the given ice thickness to a distance correction factor
#         if not math.isnan(datapoint):
#             distance_correction_map[y, x] = ice_thickness_to_correction_factor(datapoint)

#         # If the data point is nan however, then we want to keep it nan
#         # as the current implementation uses nan as land (so boats do not care)
#         else:
#             distance_correction_map[y, x] = math.nan

_utils.print_exit(__name__, __file__)