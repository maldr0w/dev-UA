
import numpy as np
import matplotlib.pyplot as plt
import heapq
import xarray as xr
from scipy.interpolate import griddata

ds = xr.open_dataset('ice_thickness_2022.nc')
ice_thickness_nan = ds['sea_ice_thickness'][0,:].values
ice_thickness = np.nan_to_num(ice_thickness_nan)

latitudes, longitudes = ds['lat'].values, ds['lon'].values
# latitude.shape = (432,432), longitude.shape = (432,432), ice_thickness.shape = (432,432)

ds.close()


GRID_RESOLUTION = 25_000

lat_min, lon_min = latitudes.min(), longitudes.min()
lat_max, lon_max = latitudes.max(), longitudes.max()


target_latitude, target_longitude = 73.173, 8.779
target_latitude, target_longitude = 64.549, 169.574
target_latitude, target_longitude = 68.879, 99.346
target_latitude, target_longitude = 73.626, 104.177

# finding the closest point to the target within the latitude and longitude arrays
distance_array = np.sqrt((latitudes - target_latitude)**2 + (longitudes - target_longitude)**2)
i,j = np.unravel_index(distance_array.argmin(), distance_array.shape)

print(latitudes[i,j], longitudes[i,j])

start_latitude, start_longitude = latitudes[i,j], longitudes[i,j]

plt.imshow(ice_thickness)
plt.plot(start_longitude, start_latitude,'ro')
plt.show()


'''

START = (0,0)
END = (99,99)

GRID_SIZE = ice_thickness.shape[0]

grid_weights = ice_thickness


def neighbors(point):
    x, y = point
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    result = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            result.append((nx, ny))
    return result

def compute_flow_field(goal):
    flow_field = np.full((GRID_SIZE, GRID_SIZE), float('inf'))
    flow_field[goal] = 0

    priority_queue = [(0, goal)]

    while priority_queue:
        current_cost, current_point = heapq.heappop(priority_queue)
        for neighbor in neighbors(current_point):
            new_cost = current_cost + grid_weights[neighbor]
            if new_cost < flow_field[neighbor]:
                flow_field[neighbor] = new_cost
                heapq.heappush(priority_queue, (new_cost, neighbor))

    return flow_field

def find_path(start, flow_field):
    path = [start]
    current = start
    while current != END:
        next_step = min(neighbors(current), key=lambda x: flow_field[x])
        path.append(next_step)
        current = next_step
    return path

flow_field = compute_flow_field(END)
path = find_path(START, flow_field)



# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plotting the grid weights
axes[0].imshow(grid_weights, cmap='viridis')
axes[0].set_title('Grid Weights')
axes[0].scatter(START[1], START[0], color='red', label='Start', s=100, edgecolors='black')
axes[0].scatter(END[1], END[0], color='blue', label='End', s=100, edgecolors='black')
axes[0].legend()

# Plotting the grid weights with the path
axes[1].imshow(grid_weights, cmap='viridis')
axes[1].set_title('Grid Weights with Path')
axes[1].scatter(START[1], START[0], color='red', label='Start', s=100, edgecolors='black')
axes[1].scatter(END[1], END[0], color='blue', label='End', s=100, edgecolors='black')
path_x, path_y = zip(*path)
axes[1].plot(path_y, path_x, color='white', label='Path')
axes[1].legend()

plt.tight_layout()
plt.show()

'''