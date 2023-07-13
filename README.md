# dev-UA - Development of Uartic project
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
	- It's more convenient to work with a grid in meters rather than degrees as A* operates in continuous space with metric distance function.
	- thus I transform latitude longitude to a map projection that preserves distances. (center of north pole) 



---
