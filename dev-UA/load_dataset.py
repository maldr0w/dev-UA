
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# load_dataset.py - UIT - Martin Stave - mvrtinstave@gmail.com
# this python file loads the variables from a netCDF4 data file

# access dataset within nc file 
file_name = 'ice_thickness_2021.nc'
dataset = Dataset(file_name, 'r') # 'r' read file

# defining corresponding variables and indexing
ice_thickness = dataset['sea_ice_thickness'][0, :, :] # all values within first dimension of 3D array

# include all elements with 'lat' and 'lon' arrays
latitude = dataset['lat'][:] 
longitude = dataset['lon'][:] 

# close dataset for resource management
dataset.close()
