# dev-UA - Development of Uarctic project
## Description and motivations:
- This project is part of a summer job at UIT
- The goal is to create a program that find the most time and cost efficient pathing for ships travelling in the arctic region
- The pathing is ultimately based on cost efficiency which will depend on the following variables:
	- Ice thickness: ships cannot pass through ice thicker than 2.1 meters (source)
 	- Fuel consumption and CO2 emission
   	- Route reliability
   	- _more to come..._
- Currently we are only considering the Northeast Passage, as the Northwest Passage is too coarse
- Illustration:
  
	![image](https://github.com/maldr0w/dev-UA/assets/74768806/030d8ce3-3e55-46d0-9549-6afa5f5661a1)
  

---

## Parts:

### A* Search Algorithm (A Star) 
Graph traversal and path searching algorithm
	Given a _start point_ and an _end point_, A* finds the most cost efficient path from start to finish by assigning a __g-score, h-score__ and __f-score__ to each _explorable node_ in a graph.
	The algorithm keeps the nodes within _open_ and _closed sets_. Nodes assigned to the open set being candidates for exploration while the already visited nodes are assigned to the closed set.
	
- __g-score:__ is the cost from the start node to via each node 
- __h-score:__ is a optimistic (heurisitc) estimate of the cost from a specific node to end node.
- __f-score:__ is the total cost estimation from the start node to the goal node via each node

##### g-score:
- Lower g-score: 
	- A node with lower g-score will be a part along a more cost efficient path from the start node
- Higher g-score: 
	- If a node is assigned a higher g-score it means the distance/cost of reaching the node in question along the current path, is increased. 
	
- The algorithm will favour the nodes with the lowest g-score when exploring nodes as these will yield the most cost efficient pathing

##### h-score:
- The optimistic heuristic estimate (h-score) is important as it guides the searching process by prioritizing the nodes that are more likely to yield the most optimal paths considering their estimated cost.

##### f-score:
- f-score is the sum of the g-score and h-score, it takes the whole path via a specific node to the end point into account. The algorithm will favor paths that strike a cost balance between the initial cost (g-score) and estimated remaining cost (h-score).




### Sea Ice Datasets
For data analysis I have opted to use monthly gridded ice thickness data sets.
The data covers a grid resolution of 25km (1-10km true spatial resolution)
The datasets are imported and extracted using NETCDF resulting in usable arrays of ice thickness with corresponding longitude and latitudes

When applying the data one must consider data projection:
- Using North Polar Sterographic projection (EPSG:3413) we set the domain to be centred at the North Pole
- With WGS84 bounds of (-180.0, 180) longitude and (60,90) latitude

- Lambert Azimuthal Equal Area (EASE-Grid version 2.0) centred over the North Pole



---

## Developed so far:

#### Created A* and noisemap:
- A* produces a optimal path for 2D datasets
- Using path and random noisemap to draw a nice picture:

	<img src="https://github.com/maldr0w/dev-UA/assets/74768806/b7f22d4d-a8fd-48b4-bc82-e7059939baf9" width="400">


#### Started working with datasets:
- I have managed to extract sea ice data from [data sets](https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-sea-ice-thickness?tab=overview) using netcdf
- Mapped geographic coordinate system of the north pole (using cartopy) to display sea ice and later pathing for routes along the Northeast Passage

	<img src="https://github.com/maldr0w/dev-UA/assets/74768806/fa8c266e-68c0-4f42-8f04-790ed1f4b96f" width="500">
 
- Further, I want to map the data values onto a grid that I can apply the A* algorithm on
- thus I transform latitude longitude to a map projection that preserves distances. (center of north pole) 

### Update: (see make_grid.py file)
- Managed to extract sea ice thickness data and converted datapoints into a pixel grid in meters, this is done by:
	- Transforming the datasets projection into North Pole Stereographic projection, which is centered around the North Pole and has units in meters
  	- Converting geographic coordinates into pixel coordinates for mapping later
  	- It's more convenient to work with a grid in meters rather than degrees as A* operates in continuous space with metric distance function.

- I've also made sure the transformation works by converting a single point (74.877N, 9.359E) and plotting it onto the grid:
- The graph is oriented 90 degrees to the right, but this only has visual impacts, nothing to worry about.
  
	<img src="https://github.com/maldr0w/dev-UA/assets/74768806/7840136f-66a5-40ca-a6b6-4a9406667670" width="400">

- Consulting the [reference map](https://data.marine.copernicus.eu/viewer/expert?view=viewer&crs=epsg%3A32661&t=1686830400000&z=0&center=2.6568731950420577e-13%2C85.4797788534361&zoom=11.852162613743667&layers=W3siaWQiOiJjMCIsIm9wYWNpdHkiOjEsImxheWVySWQiOiJBUkNUSUNfQU5BTFlTSVNfRk9SRUNBU1RfUEhZU18wMDJfMDAxX2EvY21lbXNfbW9kX2FyY19waHlfYW5mY190b3BhejRfUDFNLW1fMjAyMjExL2FsYmVkbyIsImxvZ1NjYWxlIjp0cnVlfV0%3D&objects=W3siaWQiOiJjMiIsImNycyI6ImVwc2c6MzI2NjEiLCJjb21wbGV0ZSI6dHJ1ZSwiZ3JhcGhJZHMiOlsiQVJDVElDX0FOQUxZU0lTX0ZPUkVDQVNUX1BIWVNfMDAyXzAwMV9hL2NtZW1zX21vZF9hcmNfcGh5X2FuZmNfdG9wYXo0X1AxTS1tXzIwMjIxMS9hbGJlZG8vL3YodCkiXSwidHlwZSI6InBvaSIsImNvb3JkcyI6WzkuMzU5MzQ3NjI5NzQzNjU0LDc0Ljg3Njk1MDE1NTg0NDI4XSwicGluT2Zmc2V0IjpbNjUsMF19XQ%3D%3D&basemap=dark) I am using, we can see that the point I mapped in the plot above aligns perfectly with what we are expecting

   	<img src="https://github.com/maldr0w/dev-UA/assets/74768806/bb6444dd-6c0e-43bd-a432-d803a31be69a" width="400">

- Now that I have a working grid, I can safely move on to further developing the A* algorithm on this grid.

#### Land mask: 
- added land mask to sea ice thickness map. Now I can simply check wether a point on the grid is on land or in the water

	<img src="https://github.com/maldr0w/dev-UA/assets/74768806/7fe08b32-73d3-4443-8f08-17cf2e610649" width="400">
 
 - moving on to getting total distance of A* search pathing

---
# New developments
## utils.py
- Functions
	- print_separator: Just prints 68 equal signs to make terminal output a bit cleaner
	- print_entrypoint: used at the start of each file to show whether the file is the one being ran, or it is simply being imported from another file. Primary use is debugging, seeing where something went wrong if it did
	- print_exit: Pairwise used with the above, to show when a file is done being imported from or ran
- Variables
	- unit_distance: 25000 meter, the grid resolution in the ice-thickness dataset
## fuel.py
- Classes
	- GHG
		- Attributes
			- Name
			- Factor: Emission factors for each of the GHG-types
		- Methods
			- \_\_str\_\_: String representation of class instances
	- Fuel
		- Attributes
			- Name
			- Emission-factors: The amount which each fuel-type releases a certain GHG
			- Lower heating: Net calorific value, or the amount of energy released on consumption of the fuel, used for fuel consumption calculations
			- Price: Price per kilogram of the fuel-type
		- Methods
			- \_\_str\_\_: String representation of class instances
			- get_price: Price for a given amount of fuel (default input weight 1kg)
			- get_emissions: The amount of grams of emissions for a given amount of fuel (default input weight 1kg)
			- get_gwp: Same as above, except specifically in grams of CO2
			- equiv_tons_co2: Uses the above function, sums the resulting array, then divides by 1 million to get tons
			- get_emissions_price: Same as above, except multiplies by 20 to reflect current EU emissions prices
- Functions
	- get_gwp: a function for converting each GHG-type to the equivalent amount of CO2
- Fuel-list
	- A list of all the fuels from the task description
## vessel.py
- Classes
	- Vessel
		- Attributes
			- Name
			- Main and auxiliary engine power
			- Design speed (knots)
			- Max velocity (meters per second)
			- K (factor relating total power and max velocity)
		- Methods
			- \_\_str\_\_: String representation of class instances
			- p_cons: Consumed power at velocity
				- Max velocity if no velocity specified
			- p_resist: Power needed to overcome ice
				- Ice-thickness 0 if no thickness specified (and zero as return value)
				- Max velocity if no velocity specified
			- p_available: Power available at specified ice_thickness
				- Ice-thickness 0 if no thickness specified (and total power as return value)
			- v_limit: Estimated max speed at specified thickness
				- Ice-thickness 0 if no thickness specified (and total velocity as return value)
			- time_for_trip: Estimated time needed to complete trip of a specified distance
				- Ice-thickness 0 if no thickness specifed
				- Distance 25000 meters (unit distance) if no distance specified
			- fuel_for_trip: Estimated fuel consumed for a trip of a specified distance, at a specified ice-thickness and velocity, for a provided fuel type
				- Ice-thickness 0 if no thickness specifed
				- Distance 25000 meters (unit distance) if no distance specifed
			- feasible_speed_vector: A small function which, for some specified step count, returns a range of velocity values, providing the decision space for vessel speeds
			- possible_speed_vector: Same as the previous, except it takes into account the v_limit at some ice thickness
- Ship list
	- A list containing all the ships specified in the task description, with the information given
## vessel-graphing.py
- Creates graphs for seeing the relationships between max attainable velocity, for each fuel, and at specified ice-thicknesses
## a_star.py
- Same as before, except slightly cleaned
- Now uses a binary heap to improve performance
	- Rather than needing to search the open set (or something) for the lowest score, it will simply be the first element
