# dev-UA - Development of Uartic project
### Description and motivations:
- This project is part of a summer job at UIT
- Creating a program that find the most time and cost efficient pathing for boats travelling in the artic region
- Pathing is based on:
	- Ice thickness: boats can pass through ice thicker than 2.1 meters (source)
 	- Fuel consumption and CO2 emission: _WIP_
   	- Shortest, most reliable route
   	- _more to come..._
---
# Parts:

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

---

## Current outcome:
- A* produces a optimal path
- Using path and random noisemap to draw a nice picture:

![image](https://github.com/maldr0w/dev-UA/assets/74768806/b7f22d4d-a8fd-48b4-bc82-e7059939baf9)




#### Started working with datasets:

![image](https://github.com/maldr0w/dev-UA/assets/74768806/65f0b265-8647-4e7f-89f6-0fd66a9dcb0c)






### To develope:

Done:
- ~~A*search algorithm~~
- ~~Path point reduction~~
- ~~Graphing map + path lines~~
  
To do:
- Make sea ice dataset workable
- Apply A* algorithm to real sea ice dataset
- Map pathing onto polar coordinates 
- etc...

Further down:
- Implement everything to work together in a single program
- Working with cost efficiency, Co2 emission and fuel consumption
- Create UI (maybe...)
- etc...
---
