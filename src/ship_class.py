''' Ship Class module
Defines:
 - Ship Base Class
 - Derived Ships, based on provided documents
'''
import unittest
import utils
from data import ICE_THICKNESS_MAX
import numpy as np
from fuel_class import Fuel, Diesel, NaturalGas, Methanol, DME, Hydrogen, B20, HEURISTIC_FUEL_TYPE
from typing import Type, TypeAlias, List, Optional
utils.print_entrypoint(__name__, __file__)
# ===================================================================
# Type Definitions
KiloWatts: TypeAlias = float
# ===================================================================
# Ship Definition
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
        self.max_velocity = 0.514 * design_speed
        self.k = self.total_power / (self.max_velocity ** 3)

    def __str__(self):
        return f"{self.name.upper()}\n\tMain engine: {self.main_eng_pow} kW\n\tAuxiliary engine: {self.aux_eng_pow}\n\tDesign speed: {self.design_speed}\n"

    def set_fuel(self, fuel: Type[Fuel]):
        '''Set the fuel the ship will use
        :param fuel: Type[Fuel] - The fuel in question
        '''
        self.fuel = fuel
    def set_unit_cost(self):
        ''' Set the unit cost the ship will use

        Calling this function requires that a fuel and throttle percentage (velocity) is set.
        The unit cost is used as a comparative value in the heuristic function.
        '''
        unit_consumption = self.get_fuel_consumption(0.0, 1.0)
        self.unit_cost = self.fuel.get_consumption_price(unit_consumption) + self.fuel.get_emission_price(unit_consumption)

    def set_target_velocity(self, throttle_percentage: float):
        '''Set the velocity of the ship
        :param throttle_percentage: float - The percentage of engine power to use
        '''
        if 0.0 <= throttle_percentage <= 1.0:
            consumed_power = (throttle_percentage ** 3) * self.total_power
            velocity = np.cbrt(consumed_power / self.k)
            self.velocity = velocity
        else:
            self.velocity = None

    def get_velocity(self, thickness: float) -> float:
        '''Get the actual velocity of the ship, considering ice thickness

        :param thickness: float - The thickness experienced by the ship
        :return: float - The actual velocity of the ship
        '''
        if self.velocity != None:
            if thickness == 0.0:
                return self.velocity

            resistive_power = self.get_resistive_power(thickness)
            resistive_velocity = np.cbrt(resistive_power / self.k)

            adjusted_velocity = self.velocity - resistive_velocity

            if adjusted_velocity > 0.0:
                return adjusted_velocity

        return 0.0

    def get_velocity_range(self, thickness: float) -> List[float]:
        ''' Get range of velocity values
        :param thickness: float - Ice thickness
        :return: List[float] - Possible velocities at the given thickness
        '''
        # Save initial state
        original_velocity = self.velocity
        velocities = []
        for percentage in np.arange(0.01, 1.0, 0.01):
            # Set throttle
            self.set_target_velocity(percentage)
            # Get actual velocity
            velocity = self.get_velocity(thickness)
            velocities.append(velocity)
        # Reset to initial state
        self.velocity = original_velocity
        return velocities

    def get_consumption_range(self, thickness: float, distance=utils.unit_distance) -> List[float]:
        ''' Get range of fuel consumption values
        :param thickness: float - Ice thickness
        :param distance: float - Distance to be considered
        :return: List[float] - Fuel consumption values, over the domain of possible speeds
        '''
        # Save initial state
        original_velocity = self.velocity
        consumption_values = []
        for percentage in np.arange(0.01, 1.0, 0.01):
            # Set throttle
            self.set_target_velocity(percentage)
            # Get fuel consumption
            consumption = self.get_fuel_consumption(thickness, distance)
            consumption_values.append(consumption)
        # Reset to initial state
        self.velocity = original_velocity
        return consumption_values

    def get_cost_range(self, thickness: float, distance: float = utils.unit_distance) -> list[float]:
        ''' Get range of cost values
        :param thickness: float - Ice thickness
        :param distance: float - Distance to be considered
        :return: List[float] - Cost values, over the domain of possible speeds
        '''
        costs = []
        for weight in self.get_consumption_range(thickness, distance):
            cost = self.fuel.get_consumption_price(weight) + self.fuel.get_emission_price(weight)
            costs.append(cost)
        return costs

    def get_duration_range(
            self,
            thickness: float,
            distance: float = utils.unit_distance
            ) -> list[float]:
        ''' Get range of duration values
        :param thickness: float - Ice thickness
        :param distance: float - Distance to be considered
        :return: list[float] - Duration values, over the domain of possible speeds
        '''
        # Save initial state
        original_velocity = self.velocity
        duration_values = []
        for percentage in np.arange(0.01, 1.0, 0.01):
            # Set throttle
            self.set_target_velocity(percentage)
            # Get fuel consumption
            duration = self.get_duration(thickness, distance)
            # Append duration if valid
            if duration != None:
                duration_values.append(duration)
        # Reset to initial state
        self.velocity = original_velocity
        return duration_values
        

    def get_resistive_power(self, thickness: float, ice_factor=2.5) -> float:
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
        # In all other cases, 0.0 returned
        return 0.0

    def get_fuel_consumption(self, thickness: float, distance=utils.unit_distance) -> float:
        ''' Get the fuel consumption
        :param thickness: float - The thickness for the distance
        :param distance: float - The distance covered
        :return: float - kilograms of fuel consumed
        '''
        if self.fuel == None:
            return float('inf')

        seconds = self.get_duration(thickness, distance)
        if seconds == None:
            return float('inf')

        # seconds = distance / actual_velocity
        hours = seconds / 3600.0

        # Velocity used here due to it indicating how much power is in use
        consumed_power = (self.velocity ** 3) * self.k
        if consumed_power > self.total_power:
            return float('inf')

        kilo_watt_hours = consumed_power * hours
        kg_consumed = (1.0 / self.fuel.lower_heating) * kilo_watt_hours

        return kg_consumed

    def get_duration(self, thickness: float, distance=utils.unit_distance) -> Optional[float]:
        ''' Get the duration for a segment
        :param thickness: float - The thickness for the distance
        :param distance: float - The distance covered
        :return: Optional[float] - the time, in seconds, if possible
        '''
        if self.velocity == None:
            return None

        actual_velocity = self.get_velocity(thickness)

        actual_velocity_is_too_high = actual_velocity > self.max_velocity
        actual_velocity_is_too_low = actual_velocity <= 0.0
        actual_velocity_is_invalid = actual_velocity_is_too_high or actual_velocity_is_too_low

        if actual_velocity_is_invalid:
            return None

        seconds = distance / actual_velocity

        return seconds

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
# Ship Unit Tests
class TestFuelShip(unittest.TestCase):
    def setUp(self):
        self.unit_under_testing = Ship('test')
    def run(self):
        self.assertIsNone(self.unit_under_testing.fuel)
        self.unit_under_testing.set_fuel(Diesel())
        self.assertIsInstance(self.unit_under_testing.fuel, Diesel)
# ===================================================================
# Prizna Definition
class Prizna(Ship):
    def __init__(self):
        super().__init__('Prizna',
                         main_eng_pow = 792.0,
                         aux_eng_pow = 84.0,
                         design_speed = 8.0)
# Prizna Unit Tests
class TestVelocityPrizna(unittest.TestCase):
    def test_no_velocity(self):
        '''no set velocity'''
        self.assertEqual(Prizna().velocity, None)

    def test_no_get_velocity(self):
        '''get velocity no set velocity'''
        self.assertEqual(Prizna().get_velocity(0.0), 0.0)

    def test_set_velocity(self):
        '''set max velocity'''
        ship = Prizna()
        ship.set_target_velocity(1.0)
        self.assertEqual(ship.velocity, ship.max_velocity)

    def test_set_velocity_negative(self):
        '''set negative velocity'''
        ship = Prizna()
        ship.set_target_velocity(-1.0)
        self.assertEqual(ship.velocity, None)

    def test_get_velocity_no_ice(self):
        ship = Prizna()
        ship.set_target_velocity(1.0)
        self.assertEqual(ship.max_velocity, ship.get_velocity(0.0))

    def test_get_velocity_max_ice(self):
        ship = Prizna()
        ship.set_target_velocity(1.0)
        self.assertEqual(0.0, ship.get_velocity(11.5))

    def test_get_velocity_without_setting_no_ice(self):
        ship = Prizna()
        self.assertEqual(0.0, ship.get_velocity(0.0))

    def test_get_velocity_without_setting_some_ice(self):
        ship = Prizna()
        self.assertEqual(0.0, ship.get_velocity(1.0))
# ===================================================================
# Kornati Definition
class Kornati(Ship):
    def __init__(self):
        super().__init__('Kornati',
                         main_eng_pow = 1764.0,
                         aux_eng_pow = 840.0,
                         design_speed = 12.3)
#
# Kornati Unit Tests (Omitted)
# 
# These unit tests should not be necessary, the funcitonality
# tested is invariant across subclasses.
#
# ===================================================================
# Petar Hektorovic Definition
class PetarHektorovic(Ship):
    def __init__(self):
        super().__init__('Petar Hektorovic',
                         main_eng_pow = 3600.0,
                         aux_eng_pow = 1944.0,
                         design_speed = 15.75)
#
# Petar Hektorovic Unit Tests (Omitted, see above)
#
# ===================================================================
if __name__ == '__main__':
    unittest.main()
ship_list = [
    # Ship("Prizna", 792.0, 84.0, 8.0),
    Prizna(),
    # Ship("Kornati", 1764.0, 840.0, 12.3),
    Kornati(),
    # Ship("Petar Hektorovic", 3600.0, 1944.0, 15.75)
    PetarHektorovic()
]
utils.print_exit(__name__, __file__)