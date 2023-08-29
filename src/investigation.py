from mpl_toolkits.basemap import Basemap
import math
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt


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


print("Shifting coordinates")
def shift(c):
    return (1000 * c) + (meters / 2)
vshift = np.vectorize(shift)

x_coords = vshift(dataset['sea_ice_thickness'].coords['xc'])
y_coords = vshift(dataset['sea_ice_thickness'].coords['yc'])
print("Finished\n")


flags = dataset['status_flag'].values[_time]

print("Transforming tiles...")

mapdata = np.empty(np.shape(icedata))

for y, row in enumerate(flags):
    for x, flag in enumerate(row):
        match flag:
            case 1 | 2 | 5:
                mapdata[y, x] = -0.5
            case _:
                mapdata[y, x] = icedata[y, x]
print("Finished\n")

xx, yy = np.meshgrid(x_coords, y_coords)

cmap = mpl.cm.jet
cmap.set_bad('black',1.)

# m.pcolormesh(xx, yy, mapdata, vmin=-1.0, vmax=11.0, norm='log', cmap=cmap)
m.pcolormesh(xx, yy, mapdata, norm='symlog', shading='nearest', antialiased=True, cmap=cmap)

def colorize(f, t):
    match f:
        # Nominal retrieval
        case 0:
            # return ((t + 0.5) / 11.0, 0.11, 0.25)
            return ([(t + 0.5) / 11.0, 0.11, 0.25])

        # No data
        # case 1:
        #     return (0.0, 0.0, 0.0)

        # Open ocean
        case 2:
            # return (0.0, 0.0, 0.5)
            return ([0.0, 0.0, 0.5])

        # Sattelite pole hole
        # case 3:
        #     return (0.0, 0.0, 0.0)
        
        # Land, lake, or land ice
        case 4:
            # return (0.0, 0.5, 0.0)
            return ([0.0, 0.5, 0.0])

        # Retrieval failed
        # case 5:
        #     return (0.0, 0.0, 0.0)

        case _:
            # return (0.0, 0.0, 0.0)
            return ([0.0, 0.0, 0.0])

print("Generating colormap")
# colormap = np.empty(np.shape(icedata), dtype=object)
colormap = []
for y, flag_row in enumerate(flags):
    # print("Iteration ", y)
    # colorrow = np.empty(np.shape(icedata[0]))
    row = []
    for x, flag in enumerate(flag_row):
        row.append(colorize(flag, icedata[y, x]))
        # colormap[y, x] = colorize(flag, icedata[y, x])
        # colorrow.append(colorize(flag, icedata[y, x]))
    colormap.append(row)
    # colormap.append(np.array(colorrow))
    
print("Finished")

# for coord in enumerate(x_coords):
#     print(coord)
print("Plotting data")

# for y, yc in enumerate(y_coords):
#     m.scatter(x_coords, np.full(np.shape(y_coords), yc), c=colormap[y], marker=',', linewidths=1.0, edgecolors=None)


# for y, yc in enumerate(y_coords):
#     print("Iteration ", y)
#     for x, xc in enumerate(x_coords):
#         m.plot(xc, yc, marker=',', color=colormap[y, x])

        # match flags[y, x]:
        #     # Nominal retrieval
        #     case 0:
        #         thickness = icedata[y, x]
        #         thickness_color = ((thickness + 0.5) / 11.0, 0.11, 0.25)
        #         m.plot(xc, yc, marker=',', color=thickness_color)

        #     # No data
        #     case 1:
        #         m.plot(xc, yc, marker=',', color=(0.0, 0.0, 0.0))

        #     # Open ocean
        #     case 2:
        #         m.plot(xc, yc, marker=',', color='b')

        #     # Sattelite pole hole
        #     case 3:
        #         m.plot(xc, yc, marker=',', color=(0.0, 0.0, 0.0))
            
        #     # Land, lake, or land ice
        #     case 4:
        #         m.plot(xc, yc, marker=',', color='g')

        #     # Retrieval failed
        #     case 5:
        #         m.plot(xc, yc, marker=',', color=(0.0, 0.0, 0.0))

print("Finished")




# class Node:
#     def __init__(self, x, y, lat, lon, ice_thickness):
#         self.x = x
#         self.y = y
#         self.lat = lat
#         self.lon = lon
#         self.ice_thickness = ice_thickness
#     def draw(self):
#         if not math.isnan(self.ice_thickness):
#             m.plot(self.x, self.y, marker=',', color=((self.ice_thickness + 0.5) / 11.0, 0.11, 0.22))




ax = plt.gca()
icedata = dataset['sea_ice_thickness']
# m.pcolormesh(icedata.coords['xc'], icedata.coords['yc'], icedata.values[0])
data_table = icedata.values[0]
x_coords = icedata.coords['xc']
y_coords = icedata.coords['yc']
lat_list = icedata.coords['lat']
lon_list = icedata.coords['lon']
data_list = []
# print('im gonna say im walking here')
# for y in np.arange(0, 431, 1):
#     data_row = data_table[y]
#     lat_row = lat_list[y]
#     lon_row = lon_list[y]
#     yc = (1000 * y_coords[y]) + (meters / 2)
#     for x in np.arange(0, 431, 1):
#         xc = (1000 * x_coords[x]) + (meters / 2)
#         data_list.append(Node(xc, yc, lat_row[x], lon_row[x], data_row[x]))

# print("im walking here")

# for data in data_list:
#     data.draw()

# def plotpoint(x, y, latlon=False):
#     if latlon:
#         i = 0
#         for data in data_list:
#             if lat - data.lat < 1 and lon - data.lon < 1:
#                 i = data_list.index(data)
#         xc, yc = data_list[i].x, data_list[i].y
#         m.plot(xc, yc, marker='.')
#     else:
#         m.plot(x, y, marker='.')

plt.title("my dick and balls")
plt.show()
print("Saving figure")
plt.savefig('fig.png')
# print(icedata)
# print(icedata.values)
# print(icedata.coords)

# print(dataset['Lambert_Azimuthal_Grid'])
# print(dataset['Lambert_Azimuthal_Grid'].attrs['proj4_string'])
