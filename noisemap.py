
import noise
import numpy as np
import matplotlib.pyplot as plt

# Generating perlin noise map
# octaves = 6
# persistence = 0.5
# lacunarity = 2.0
scale = 10.0
shape = (100,100)
# scale = np.random.uniform(5.0, 15.0)  # Randomize the scale
octaves = np.random.randint(2, 20)  # Randomize the number of octaves
persistence = np.random.uniform(0.2, 0.9)  # Randomize the persistence
lacunarity = np.random.uniform(1.5, 5.5)  # Randomize the lacunarity



world = np.zeros(shape)

for i in range(shape[0]):
    for j in range(shape[1]):
        value = noise.pnoise2(i/scale, j/scale, octaves=octaves, persistence=persistence, 
            lacunarity=lacunarity, repeatx=shape[0], repeaty=shape[1], base=0)
        mapped_value = np.interp(value, [-1,1], [0.0, 3.0])
        world[i][j]=mapped_value
        


#plt.imshow(world,cmap='gray',interpolation='nearest',vmin=0.0, vmax=3.0)
#plt.colorbar()
#plt.show()
