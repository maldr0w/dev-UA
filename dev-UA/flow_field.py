import numpy as np
import matplotlib.pyplot as plt
import heapq



GRID_SIZE = 100

START = (0,0)
END = (99,99)

NUM_CLUSTERS = 10
CLUSTER_RADIUS = 10


# grid_weights = np.random.rand(GRID_SIZE,GRID_SIZE)

def gaussian(x,mu,sigma):
    return np.exp(-np.power(x-mu, 2.) / (2*np.power(sigma, 2.)))


def create_clustered_weights(grid_size, num_clusters, cluster_radius):
    weights = np.zeros((grid_size, grid_size))
    for _ in range(num_clusters):
        center = (np.random.randint(grid_size), np.random.randint(grid_size))
        peak_value = np.random.rand()
        for i in range(grid_size):
            for j in range(grid_size):
                dist = np.linalg.norm(np.array([i, j]) - np.array(center))
                weights[i, j] += peak_value * gaussian(dist, 0, cluster_radius)
    return weights / np.max(weights)  # Normalize to [0, 1]

grid_weights = create_clustered_weights(GRID_SIZE, NUM_CLUSTERS, CLUSTER_RADIUS)




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

# Printing
print(f"Optimal path from {START} to {END}:")
for p in path:
    print(p)

print(f"\nPath length in pixels: {len(path)}")



def compute_gradient(flow_field):
    grad_y, grad_x = np.gradient(flow_field)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize to avoid very long arrows
    grad_x /= magnitude
    grad_y /= magnitude
    
    return grad_x, grad_y


grad_x, grad_y = compute_gradient(flow_field)




# # Visualization


# # Visualization
# fig, axes = plt.subplots(1, 3, figsize=(10, 5))

# # Plotting the grid weights
# axes[0].imshow(grid_weights, cmap='viridis')
# axes[0].set_title('Grid Weights')
# axes[0].scatter(START[1], START[0], color='red', label='Start', s=100, edgecolors='black')
# axes[0].scatter(END[1], END[0], color='blue', label='End', s=100, edgecolors='black')
# axes[0].legend()

# # Plotting the grid weights with the path
# axes[1].imshow(grid_weights, cmap='viridis')
# axes[1].set_title('Grid Weights with Path')
# axes[1].scatter(START[1], START[0], color='red', label='Start', s=100, edgecolors='black')
# axes[1].scatter(END[1], END[0], color='blue', label='End', s=100, edgecolors='black')
# path_x, path_y = zip(*path)
# axes[1].plot(path_y, path_x, color='white', label='Path')
# axes[1].legend()


# # Plotting the grid weights with the vector field
# axes[2].imshow(grid_weights, cmap='viridis')
# axes[2].set_title('Grid Weights with Vector Field')
# axes[2].scatter(START[1], START[0], color='red', label='Start', s=100, edgecolors='black')
# axes[2].scatter(END[1], END[0], color='blue', label='End', s=100, edgecolors='black')
# # Plotting the vector field using quiver (subsample to avoid too many arrows)
# subsample = 5
# axes[2].quiver(np.arange(0, GRID_SIZE, subsample), np.arange(0, GRID_SIZE, subsample),
#                grad_x[::subsample, ::subsample], grad_y[::subsample, ::subsample],
#                color='white', scale=20, headwidth=5, headlength=6)
# axes[2].legend()

# plt.tight_layout()
# plt.show()







def generate_grid_with_path():
    # Create the grid
    grid_weights = create_clustered_weights(GRID_SIZE, NUM_CLUSTERS, CLUSTER_RADIUS)

    # Randomly select start and end points
    START = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))
    END = (np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE))

    # Compute flow field and path
    flow_field = compute_flow_field(END)
    path = find_path(START, flow_field)

    return grid_weights, START, END, path

# Generate data for 3 sets
grid_data = [generate_grid_with_path() for _ in range(3)]

# Visualization
fig, axes = plt.subplots(3, 1, figsize=(6, 18))

for ax, data in zip(axes, grid_data):
    grid_weights, START, END, path = data
    ax.imshow(grid_weights, cmap='viridis')

    ax.scatter(START[1], START[0], color='red', label='Start', s=100, edgecolors='black')
    ax.scatter(END[1], END[0], color='blue', label='End', s=100, edgecolors='black')
    path_x, path_y = zip(*path)
    ax.plot(path_y, path_x, color='white', label='Path')
    ax.legend()

plt.tight_layout()
plt.show()