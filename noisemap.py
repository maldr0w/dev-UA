
import noise
import numpy as np
import matplotlib.pyplot as plt

# noisemap.py - UIT - Martin Stave - mvrtinstave@gmail.com 
# this file creates a noisemap to mimic a sea ice thickness map

# generating perlin noise map
scale = 10.0 # zoom level
shape = (100,100)

# octaves = 6
# persistence = 0.5
# lacunarity = 2.0

# creating randomized variables for noisemap
octaves = np.random.randint(2, 20) # number of layered noise functions
persistence = np.random.uniform(0.2, 0.9) # influence of each octave on final noisemap
lacunarity = np.random.uniform(1.5, 5.5) # frequency or detail level of each octave

noisemap  = np.zeros(shape) # empty array of zeros

# loop over each element in noisemap array
for i in range(shape[0]):
    for j in range(shape[1]):
        # generate perlin noisemap using the predefined parameters
        # i/scale and j/scale adjusts frequency of noise pattern
        # repeatx and repeaty controls repetition of noise pattern
        value = noise.pnoise2(i/scale, j/scale, octaves = octaves, persistence = persistence, 
                              lacunarity = lacunarity, repeatx = shape[0], repeaty = shape[1], base = 0)

        # perlin native range is [-1,1] but we change it to [0,3] to mimic ice thickness
        mapped_value = np.interp(value, [-1,1], [0.0, 3.0])
        
        # store current value in corresponding position within noisemap
        noisemap[i][j] = mapped_value


# graphing noisemap, interpolating pixels using nearest pixel, vmin/max determine color range
plt.imshow(noisemap, cmap = 'gray', interpolation = 'nearest', vmin = 0.0, vmax = 3.0)
plt.colorbar() # add colorbar 
plt.show()
