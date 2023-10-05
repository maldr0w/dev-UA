# dev-UA - Development of Uarctic project


# How to Run

To run the program, simply change to the 'src' directory, and enter
```
python3 main.py --opt=arg[..]
```

For example, if you have coordinates (60.000, 68.000) and (66.898, -162.596), enter
```
python3 main.py --start=60.000,68.000 --end=66.898,-162.596
```

For more help and information, enter
```
python3 main.py -h
```

---

# New developments

## utils.py

- unit_distance: 25000 meter, the grid resolution in the ice-thickness dataset

### Functions

- print_separator: Just prints 68 equal signs to make terminal output a bit cleaner
- print_entrypoint: used at the start of each file to show whether the file is the one being ran, or it is simply being imported from another file. Primary use is debugging, seeing where something went wrong if it did
- print_exit: Pairwise used with the above, to show when a file is done being imported from or ran

## fuel.py

- Fuel-list: A list of all the fuels from the task description

### Functions

- get_gwp: a function for converting each GHG-type to the equivalent amount of CO2

### Classes

> ### GHG
> 
> #### Attributes
> 
> - Name
> - Factor: Emission factors for each of the GHG-types
> 
> #### Methods
> 
> - \_\_str\_\_: String representation of class instances

> ### Fuel
> 
> #### Attributes
> 
> - Name
> - Emission-factors: The amount which each fuel-type releases a certain GHG
> - Lower heating: Net calorific value, or the amount of energy released on consumption of the fuel, used for fuel consumption calculations
> - Price: Price per kilogram of the fuel-type
> 
> #### Methods
> 
> - \_\_str\_\_: String representation of class instances
> - get_price: Price for a given amount of fuel (default input weight 1kg)
> - get_emissions: The amount of grams of emissions for a given amount of fuel (default input weight 1kg)
> - get_gwp: Same as above, except specifically in grams of CO2
> - equiv_tons_co2: Uses the above function, sums the resulting array, then divides by 1 million to get tons
> - get_emissions_price: Same as above, except multiplies by 20 to reflect current EU emissions prices

## vessel.py

- Ship list: A list containing all the ships specified in the task description, with the information given

### Classes

> ### Vessel
> 
> #### Attributes
> 
> - Name
> - Main and auxiliary engine power
> - Design speed (knots)
> - Max velocity (meters per second)
> - K (factor relating total power and max velocity)
> 
> #### Methods
> 
> - \_\_str\_\_: String representation of class instances
> - p_cons: Consumed power at velocity
>
> 	Max velocity if no velocity specified
>
> - p_resist: Power needed to overcome ice
>
> 	Ice-thickness 0 if no thickness specified (and zero as return value)
> 	Max velocity if no velocity specified
>
> - p_available: Power available at specified ice_thickness
>
> 	Ice-thickness 0 if no thickness specified (and total power as return value)
>
> - v_limit: Estimated max speed at specified thickness
>
> 	Ice-thickness 0 if no thickness specified (and total velocity as return value)
>
> - time_for_trip: Estimated time needed to complete trip of a specified distance
>
> 	Ice-thickness 0 if no thickness specified
> 	Distance 25000 meters (unit distance) if no distance specified
>
> - fuel_for_trip: Estimated fuel consumed for a trip of a specified distance, at a specified ice-thickness and velocity, for a provided fuel type
>
> 	Ice-thickness 0 if no thickness specifed
> 	Distance 25000 meters (unit distance) if no distance specifed
>
> - feasible_speed_vector: A small function which, for some specified step count, returns a range of velocity values, providing the decision space for vessel speeds
> - possible_speed_vector: Same as the previous, except it takes into account the v_limit at some ice thickness

## vessel-graphing.py

- Creates graphs for seeing the relationships between max attainable velocity, for each fuel, and at specified ice-thicknesses

## a_star.py

- Same as before, except slightly cleaned
- Now uses a binary heap to improve performance
	- Rather than needing to search the open set (or something) for the lowest score, it will simply be the first element
- Now uses Great circle distance rather than manhattan distance in heuristic

# Tasks
1. [x] Fuel consumption vs. Vessel speed graph
	- [x] Basic case (only one fuel type)
	- [x] Fuel specific cases
	- [x] Consider GHG emissions (Price vs. Vessel speed)
2. [ ] Pareto frontier
	- [ ] Plot of tradeoff between fuel consumption for vessel speeds
	- [ ] Extend to all fuel cases
3. [x] Optimal route for different fuels
	- [x] Single fuel case
	- [x] All fuels
	
	Here it would be necessary I think, to know the pareto frontier.
	
	The heuristic algorithm needs to know the worst case, thus the assumed price of a path using the most polluting/inefficient fuel
	
4. [x] GHG emissions vs. vessel speed
5. [ ] GHG emissions vs. ice-thickness
6. [ ] Optimization results
	- [ ] Table displaying all optimal values
		- Fuel consumption
		- Vessel speed
		- Route segments for chosen scenarios
7. [ ] Sensitivity analysis
	- Analyze things like ice thickness coefficient
	- With this, create sensitivity plot to show how changes in these params affect optimization results
