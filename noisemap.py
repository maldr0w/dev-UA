
import noise
import numpy as np
import matplotlib.pyplot as plt

# Generating perlin noise map
scale = 10.0
shape = (100,100)

# octaves = 6
# persistence = 0.5
# lacunarity = 2.0

octaves = np.random.randint(2, 20)  # Randomize the number of octaves
persistence = np.random.uniform(0.2, 0.9)  # Randomize the persistence
lacunarity = np.random.uniform(1.5, 5.5)  # Randomize the lacunarity



noisemap  = np.zeros(shape)

for i in range(shape[0]):
    for j in range(shape[1]):
        value = noise.pnoise2(i/scale, j/scale, octaves=octaves, persistence=persistence, 
            lacunarity=lacunarity, repeatx=shape[0], repeaty=shape[1], base=0)
        mapped_value = np.interp(value, [-1,1], [0.0, 3.0])
        noisemap[i][j]=mapped_value
        
print(noisemap.shape)

print(noisemap)

# plt.imshow(world,cmap='gray',interpolation='nearest',vmin=0.0, vmax=3.0)
# # plt.axline((4,4),(20,20),color='red')
# plt.colorbar()
# plt.show()
