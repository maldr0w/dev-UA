import fuel_class
import ship_class

def data_list(s, t=0.0):
    # velocity list
    velocities = s.zero_initial_speed_vector()
    # ( fuel * ( velocity * kg ) ) list
    weights = [(f, [(v, s.fuel_for_trip(f, t=t, v=v), ) for v in velocities]) for f in fuel_class.fuel_list]
    # ( fuel * ( velocity * [€ / kg] ) list
    fuel_price = [(f, [(v, f.get_consumption_price(per_v_data)) for v, per_v_data in per_f_data]) for f, per_f_data in weights]
    # ( fuel * ( velocity * tons ) ) ) list
    emission_data = [(f, [(v, f.get_emission_tonnage(per_v_data)) for v, per_v_data in per_f_data]) for f, per_f_data in weights]
    # ( fuel * ( velocity * [€ / ton] ) ) list
    emission_price = [(f, [(v, f.get_emission_price(per_v_data)) for v, per_v_data in per_f_data]) for f, per_f_data in weights]
    return (weights, fuel_price, emission_data, emission_price)

def trip_duration_list(ship, thickness=0.0):
    # (ship * ( fuel * ( velocity * hours ) ) ) list
    return [(v, ship.get_trip_duration(v, thickness=thickness)) for v in ship.feasible_speed_vector(thickness=thickness)]

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

def plot_coord(lon: float, lat: float, lon_west=False, c='r', m=',', map=None):
    '''
    Function to plot a coordinate on the map given.
    '''
    if lon_west:
        lon = 360. - lon
    x, y = map(lon, lat)
    map.plot(x, y, marker=m, color=c)
    csv_something = f'{str(lon)},{str(lat)}'
    print(csv_something)
    return csv_something

# Access icedata (for some reason, there is a time dimension)
TIME = 0
icedata = dataset['sea_ice_thickness'].values[TIME]

# Shift coordinates based on the finding in terms of scale
# This makes the top-left 0 m, 0 m
def shift(coordinate_component: float) -> float:
    scaled_component = 1000.0 * coordinate_component
    shifted_component = scaled_component + (meters / 2)
    return shifted_component

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
        if datapoint < 0.0:
            ice_values[y, x] = 0.0
        elif datapoint > 11.0:
            ice_values[y, x] = 11.0

ICE_THICKNESS_MAX = np.nanmax(ice_values)
ICE_THICKNESS_MEAN = np.nanmean(ice_values)
# Color normalization scheme, behaves linearly until around the ice limit of the ships
COLORMAP_NORM = mpl.colors.SymLogNorm(linthresh=2.4, vmin=0.0, vmax=ICE_THICKNESS_MAX)

# Make a mesh corresponding to the entire map for creating a figure
xx, yy = np.meshgrid(x_coords, y_coords)
# Draw the colormesh
def reinit_colormesh(map):
    cmap = mpl.cm.jet
    cmap.set_bad('black',1.)
    cm = map.pcolormesh(xx, yy, ice_values, 
                        shading='nearest', 
                        antialiased=True, 
                        cmap=cmap,
                        norm = COLORMAP_NORM)
    map.colorbar(cm, extend='max', label='Ice Thickness', ticks=np.arange(0.0, ICE_THICKNESS_MAX, ICE_THICKNESS_MAX / 10.0))
    
def init_map():
    map = reinit_map()
    reinit_colormesh(map)
    return map

def save_coord_map(name):
    plt.title(str(name) + " map")
    # plt.show()
    if utils.verbose_mode:
        print("\tSaving figure...")
    plt.savefig(str(name) + '.png')
    if utils.verbose_mode:
        print("\tFinished.")
    plt.close()
    if utils.verbose_mode:
        utils.print_separator()
utils.print_exit(__name__, __file__)
