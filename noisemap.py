
import noise
import numpy as np
import matplotlib.pyplot as plt

# Generating perlin noise map
shape = (100,100)
scale = 10.0
octaves = 6
persistence = 0.5
lacunarity = 2.0

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
