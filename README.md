# dev-UA - Development of Uarctic project


# How to Run

To run the program, simply change to the 'src' directory, and enter
```
python3 main.py --opt=arg[..]
```

For example, if you have coordinates (69.376, 33.585) and (66.888, -162.650), enter
```
python3 main.py --start=69.376,33.585 --end=66.888,-162.650
```
or
```
python3 main.py -s 60.000,68.000 -e 66.898,-162.596
```

Now by only specifying a start and end, the program will automatically run through all possible ship and fuel combinations.

If you would like to specify which fuel or ship to run, simply run something like the following:

```
python3 main.py --start=69.376,33.585 --end=66.888,-162.650 --fuel=Diesel --ship=Prizna
```

The possible values for fuel are Diesel, LNG, Methanol, DME, Hydrogen, and B20

The possible values for ships are Prizna, Kornati, and Petar

Any unspecified parameter will result in a run where the specified parameter is used, and the possible values for the unspecified parameter will be iterated over

This means, for example, if the above line is run, but `--fuel=Diesel` is omitted, the ship Prizna will be used, and will a search will be run 6 times, once for each fuel type

In case you would like to create graphs representing the data of the vessel, simply write
```
python3 main.py --graph=0.0
```
where the parameter of graph represents the ice thickness at which the data is accurate
```
python3 main.py -g 0.0
```

For more help and information, enter
```
python3 main.py -h
```

---


# Modifying values

As of right now, modifying the values should be done with care.

Each parameter has remained the same (more or less) since the start of the project, meaning altering the value of certain parameters may cause the heuristic function and cost-estimation function to stall, resulting in failed searches, or individual searches taking several minutes.

## If you wish to edit the code:

- Make sure you actually understand it,
- Make sure you understand all refferents of the values/code-blocks you are altering, and how the change may affect them,
- Make sure you understand python.

Note: This is to be understood as a waiver of liability, and any alterations made to the code without the authors express consent which causes the code to behave in ways not originally observed on the author's machine, or not according to task specifications, cannot be considered a fault of the author, and by editing the code in this manner, you agree to these terms.

### Fuels

If one wishes to alter the underlying representation of fuels, look inside the fuel_class.py file.

The base class should not be altered as it is tightly coupled to the integration of the project.

DO NOT ALTER THE BASE CLASS.

If you do not know what a base class is, do not alter any of the code. It is not my job to educate on this topic. Again, see the disclaimer; you are on your own.

From line 111 (as of writing) and onwards, the derived classes are defined.

Each uses a simple 6-line repeating structure, which can be deduced from the base class.

In the following example, I will alter the price-per-kilogram of Methanol.

I will also explain some of the syntax.

```python
# Diesel Definition
DIESEL_POLLUTANTS = [CO2(3206.), CH4(0.06), N2O(0.15)]
class Diesel(Fuel):
    def __init__(self):
        super().__init__('diesel',
                         0.78, 
                         11.83, 
                         DIESEL_POLLUTANTS)

```
In this snippet, a 'POLLUTANTS' list is made. This uses the ghg_class.py file. Here we use the same pattern to instantiate the three types of greenhouse gasses. For a fueltype like hydrogen, with no pollutants, we leave this list empty:
```python
#...
HYDROGEN_POLLUTANTS = []
# ... hydrogen definition ...
```
Next we define the class, using the 'Fuel' base class in the parens, to illustrate inheritance, used here to define shared behavior for the different fuels.

Then we write the `__init__(self)` function, which is implicitly called anytime you call the fuels constructor, by writing `something = Diesel(...)`.

In this function we make the call to `super().__init__(...)`, which just means 'Use the defined constructor for the parent class'. By looking at the definition of `Fuel` base class, we can see:
- The first parameter corresponds to the name of the fuel. Take special care not to alter this in any way.
- The second parameter corresponds to the price-per-kilogram
- The third parameter corresponds to the 'lower heating' or net calorific value
- The last parameter corresponds to the pollutants, along with the parameters of the pollutants corresponding to the gwp of the fuel (so how much of each pollutant is emitted, etc.)

Thus by altering the number `0.78` in the definition of diesel, I can change the price-per-kilogram, any way I may want.

At the end of the fuel_class.py file, the fuel_list variable is instantiated, which is what the program uses internally to iterate through possible fuel-types:
```python
fuel_list = [
    Diesel(),
    NaturalGas(),
    Methanol(),
    DME(),
    Hydrogen(),
    B20(),
]
```

All of the derived classes are followed by a suit of unit tests, and no guarantee will be made that these will still pass - the tests are hard-coded, and unless you change them to reflect your changes, I can almost _guarantee_ they will fail.
This rule also holds for the ships.

### Ships

In the ship_class.py file, a similar structure is seen, where a base-class defined shared behavior, and derived classes offer syntactical readibility, and specific values.

```python
# Prizna Definition
class Prizna(Ship):
    def __init__(self):
        super().__init__('Prizna',
                         main_eng_pow = 792.0,
                         aux_eng_pow = 84.0,
                         design_speed = 8.0)
```

The fields here are annotated. They should reflect values in datasheets given to the developers. `design_speed` is given in knots, internal calculations use meters per second, and the value is converted on instantiation of the object.

Here there are two values to change, that will significantly alter the effect of the code:

```python
ICE_THICKNESS_LIMIT = 2.1
class Ship:
    def __init__(
            self, name: str, 
            main_eng_pow: KiloWatts = 0.0, aux_eng_pow: KiloWatts = 0.0, 
            design_speed: float = 0.0
            ):
        # Set the name of the ship
        self.name: str = name

        # Initialize fuel and velocity to None
        self.fuel: Optional[Type[Fuel]] = None
        self.velocity: Optional[float] = None

        # Initialize remaining values
        self.total_power: KiloWatts = main_eng_pow + aux_eng_pow
        self.max_velocity: float = 0.514 * design_speed
        self.k: float = self.total_power / (self.max_velocity ** 3)
```

The first line (the one before the Ship class definition) defines a constant parameter, `ICE_THICKNESS_LIMIT`. The introduction of this value was a late decision, but one which should have been there from the start.

In short, internally, if any ice-thickness from the data-set should be chosen, and it is above this value, the cost-estimate will be infinite. The name of the parameter should prove descriptive in this manner. The actual function called is the `get_cost` function, which will be discussed.

Next, the actual calculation of what has been dubbed `resistive_power`, inside the ship class definition, and the `get_cost` function:

```python
    # ... ship class member function definitions
    def get_resistive_power(self, thickness: float, ice_factor=2.33) -> float:
        '''Gets the resistive power the ice imparts on the ship

        :param thickness: float - The thickness in question
        :param ice_factor: float - Some weird constant, might change
        :return: float - The resistive power experienced
        '''
        if self.velocity != None:
            if thickness > ICE_THICKNESS_LIMIT:
                return float('inf')
            # Get how much power is used for foward momentum
            consumed_power = (self.velocity ** 3) * self.k
            if 0.0 < consumed_power <= self.total_power:
                # Scale the forward momentum by the impedance
                # of the ice
                adjusted_thickness = np.power(thickness, 1.5) / ICE_THICKNESS_MAX
                ice_resistance_factor = ice_factor * adjusted_thickness
                resistive_power = ice_resistance_factor * consumed_power
                return resistive_power
            else:
                return float('inf')
        return 0.0
    #
    # ... ship class member function definitions
    #
    def get_cost(self, thickness: float, distance=utils.unit_distance) -> float:
        ''' Get the price
        :param thickness: float - The thickness for the distance
        :param distance: float - The distance covered
        :return: float - The cost for the given parameters
        '''
        if thickness > ICE_THICKNESS_LIMIT:
            return float('inf')

        kg_consumed = self.get_fuel_consumption(thickness, distance)

        if float('-inf') < kg_consumed < float('inf'):
            cost = self.fuel.get_consumption_price(kg_consumed) + self.fuel.get_emission_price(kg_consumed)
        else:
            cost = float('inf')

        return cost
```

A default parameter is used in both functions. The `get_cost` function calls the `get_fuel_consumption` function, which uses a set target velocity to calculate the duration of a leg of the journey. This target velocity can be altered from run to run by changing the value in the first couple lines of the search function. It expects a percentage (0.0-1.0).

The default parameters in the function definition relate to the values used internally. The `utils.unit_distance` is a named parameter corresponding to 25000 meters, or the distance between any unit xy-step in the ice-data grid (this was the smallest resolution feasible to work with given the scope of the task, and the inherent low speed of python).

Next, in the `get_resistive_power` function, the `ice_factor` variable is defined, as a weighting coefficient. The actual calculation is also a bit convoluted, but the idea is to scale values between 0 and 66% of `ICE_THICKNESS_MAX` down a bit, and to scale values from 66% of `ICE_THICKNESS_MAX` to `ICE_THICKNESS_MAX` up slightly. Then, the `ice_factor` is multiplied with the result, and the product is to be taken as some `resistive_power`. This is then directly subtracted from the ideal velocity (with some calculation to shift units from kilowatts to meters per second), before finally being used in the fuel-cost calculation.

---

# Developments

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
2. [ ] Pareto frontier (Not relevant anymore)
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
