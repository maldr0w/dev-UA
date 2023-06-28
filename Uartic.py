# Dev: Martin Stave, July 01, 2023 - Uartic.py
# Uartic Project 

import numpy as np

# Define the shipping routes as lists of latitude and longitude coordinates
shipping_routes = {
 1: [(0.0, 0.0), (1.0, 1.0)],
 2: [(1.0, 1.0), (1.0, 2.0), (0.0, 3.0)],
 3: [(0.0, 3.0), (-1.0, 4.0), (-3.0, 2.0), (0.0, 4.0)]
}

# Define the ice thickness values for each shipping route
ice_thickness = {
 1: 0.2,
 2: 0.3,
 3: 0.5
}

# Define the fuel cost parameters as a dictionary of tuples
# Each tuple contains two values: a fixed cost and a variable cost per nautical mile
fuel_cost = {
 'a': (0.1, 1.0),
 'b': (0.2, 1.5),
 'c': (0.01, 0.6)
}

# Compute the total cost of all shipping routes, taking into account ice thickness and fuel cost
total_cost = 0.0
for route_name in shipping_routes.keys():
   route = shipping_routes[route_name]
   dist = 0.0
   last = route[0]
 # Compute the distance between the current point and the previous point using the haversine formula
   for i in range(1, len(route)):
    lat1, lon1 = last
    lat2, lon2 = route[i]
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    dist += c * 3437.74677 # Convert from radians to nautical miles
    last = route[i]

   # Add the cost of this route to the total cost
   # Add the fuel cost for this route based on the total distance and the fuel cost parameters
   total_cost += dist * ice_thickness[route_name]
   total_cost += (fuel_cost['a'][0] + fuel_cost['a'][1] * dist)


print("Total cost:", total_cost)

