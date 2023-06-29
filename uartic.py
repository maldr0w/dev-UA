# Dev: Martin Stave, July 01, 2023 - uartic.py
# Uartic Project 

# Data aquizition
   
from netCDF4 import Dataset

nc_file = Dataset('thk_2011_octnov.map.nc','r') 

# Finding variable names
variable_names = list(nc_file.variables.keys())
for var_name in variable_names:
    print(var_name)

nc_file.close()



# Access data varibales
thickness = nc_file.variables['thickness'] 
latitude = nc_file.variables['latitude']
longitude = nc_file.variables['longitude']

grid_spacing=nc_file.variables['grid_spacing']


# Access data
thickness = thickness[:]
print(thickness.type())