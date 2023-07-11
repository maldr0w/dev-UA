
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# Loading dataset
dataset = Dataset('ice_thickness_2021.nc', 'r')
# dataset = Dataset('ice_thickness_2022.nc', 'r')

ice_thickness = dataset['sea_ice_thickness'][0, :, :]
latitude = dataset['lat'][:]
longitude = dataset['lon'][:]
dataset.close()

