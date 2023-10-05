import unittest
import utils
import numpy as np
from fuel_class import Fuel, Diesel, NaturalGas, Methanol, DME, Hydrogen, B20, HEURISTIC_FUEL_TYPE
from typing import Type, List
utils.print_entrypoint(__name__, __file__)
class Ship:
    def __init__(self, name, main_eng_pow=0.0, aux_eng_pow=0.0, design_speed=0.0):
        # Set the name of the ship
        self.name = name

        # Initialize fuel and velocity to None
        self.fuel = None
        self.velocity = None
        # self.throttle_percentage = 0.00
        # self.main_eng_pow = main_eng_pow
        # self.aux_eng_pow = aux_eng_pow
        # self.design_speed = design_speed
        self.p_tot = main_eng_pow + aux_eng_pow
        self.v_max = 0.514 * design_speed
        self.k = self.p_tot / (self.v_max ** 3)
        self.ice_assymptotic_value = 5.7
        self.ice_exponent = 10
        self.ice_correction_term = 11.5 ** (self.ice_exponent - 1)

        # self.HEURISTIC_CONSUMPTION_QUANTITY = self.get_trip_consumption(HEURISTIC_FUEL_TYPE, self.v_limit(0.0))
        # self.HEURISTIC_BASAL_RATE = self.get_trip_consumption(HEURISTIC_FUEL_TYPE, self.v_limit(0.0))
        # self.COST_PER_UNIT = self.get_costs(HEURISTIC_FUEL_TYPE, self.v_limit(0.0), utils.unit_distance, 0.0)
        # self.COST_PER_UNIT = HEURISTIC_FUEL_TYPE.get_price(self.HEURISTIC_CONSUMPTION_QUANTITY) + HEURISTIC_FUEL_TYPE.get_emission_price(self.HEURISTIC_CONSUMPTION_QUANTITY)

    def __str__(self):
        return f"{self.name.upper()}\n\tMain engine: {self.main_eng_pow} kW\n\tAuxiliary engine: {self.aux_eng_pow}\n\tDesign speed: {self.design_speed}\n"

    def set_fuel(self, fuel: Type[Fuel]):
        '''Set the fuel the ship will use
        :param fuel: Type[Fuel] - The fuel in question
        '''
        self.fuel = fuel
    def set_unit_cost(self):
        unit_consumption = self.get_fuel_consumption(0.0, 1.0)
        self.unit_cost = self.fuel.get_price(unit_consumption) + self.fuel.get_emission_price(unit_consumption)

    # def set_throttle(self, percentage: float):
    #     '''Set the throttle percentage of the ship
    #     :param percentage: float - The percent of engine power to be used
    #     '''
    #     if 0.0 <= percentage <= 1.0:
    #         self.throttle_percentage = percentage

    def set_target_velocity(self, throttle_percentage: float):
        '''Set the velocity of the ship
        :param throttle_percentage: float - The percentage of engine power to use
        '''
        if 0.0 <= throttle_percentage <= 1.0:
            consumed_power = (throttle_percentage ** 3) * self.p_tot
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

            if adjusted_velocity > 0:
                return adjusted_velocity

        return 0.0

    def get_velocity_range(self, thickness: float) -> List[float]:
        original_velocity = self.velocity
        velocities = []
        for percentage in np.arange(0.01, 1.0, 0.01):
            self.set_target_velocity(percentage)
            velocity = self.get_velocity(thickness)
            velocities.append(velocity)
        self.velocity = original_velocity
        return velocities
    def get_consumption_range(self, thickness: float, distance=utils.unit_distance) -> List[float]:
        original_velocity = self.velocity
        consumption_values = []
        for percentage in np.arange(0.01, 1.0, 0.01):
            self.set_target_velocity(percentage)
            consumption = self.get_fuel_consumption(thickness, distance)
            consumption_values.append(consumption)
        self.velocity = original_velocity
        return consumption_values
        

    # def get_requested_forward_power(self) -> float:
    #     if self.velocity != None:
    #         # Here we use ship.velocity rather than
    #         # ship.get_velocity, due to the fact that
    #         # ship.velocity represents what is demanded
    #         # of the engine, ship.get_velocity represents
    #         # the actual velocity travelled
    #         consumed_power = (self.velocity ** 3) * self.k
    #         if 0.0 < consumed_power <= self.p_tot:
    #             return consumed_power
    #     return 0.0

    def get_resistive_power(self, thickness: float, ice_factor=3.0) -> float:
        '''Gets the resistive power the ice imparts on the ship

        :param thickness: float - The thickness in question
        :param ice_factor: float - Some weird constant, might change
        :return: float - The resistive power experienced
        '''
        if self.velocity != None:
            # Get how much power is used for foward momentum
            # consumed_power = self.get_requested_forward_power()
            consumed_power = (self.velocity ** 3) * self.k
            if 0.0 < consumed_power <= self.p_tot:
                # Scale the forward momentum by the impedance
                # of the ice
                # By taking the sqrt of the thickness
                # times the ice_factor
                # divided by its potential max,
                # we generate resistive_power values
                # proportional to the total power
                # at a rate that quickly slows
                # The ice_factor value used was found
                # during development to accurately
                # map real ice-breaker ships' abilities.
                # (According to some research, the 
                # arktika class of Russian ice-breakers
                # can handle ice up to 3 meters thick,
                # some even thicker)
                # resistive_power = ice_factor * np.cbrt(thickness / 11.5) * consumed_power
                scaled_thickness = ice_factor * thickness
                scaled_thickness_factor = scaled_thickness / 11.5

                ice_resistance_exponent = 1.0 / 2.5
                # ice_resistance_factor = np.cbrt(scaled_thickness / 11.5)
                # ice_resistance_factor = np.power(scaled_thickness_factor, ice_resistance_exponent)
                ice_resistance_factor = ice_factor * (thickness / 11.5)
                resistive_power = ice_resistance_factor * consumed_power
                return resistive_power
                # Check if the actual power draw experienced by 
                # the engine is within spec
                if 0.0 < (consumed_power + resistive_power) <= self.p_tot:
                    return resistive_power
                # If it is not, infinite resistive power is said
                # to be experienced
                else:
                    return float('inf')

            
        # In all other cases, 0.0 returned
        return 0.0
    # def get_available_power(self, thickness: float) -> float:
    #     '''Get the available power for the ship

    #     :param thickness: float - The current ice thickness 
    #                               experienced by ship
    #     :return: float - The power available, meaning the total 
    #                      power of the ship minus the power 
    #                      required to "break" the ice

    #     '''
    #     if self.velocity != None:
    #         power_in_use = self.get_requested_forward_power() + self.get_resistive_power(thickness)
    #         power_available = self.p_tot - power_in_use

    #         # Make sure the power available is realistic
    #         if 0.0 <= power_available <= self.p_tot:
    #             return power_available
    #         else:
    #             return 0.0
    #     else:
    #         # If the ship is not moving, all the power is available
    #         return self.p_tot

    # def set_unit_cost(self, fuel):
    #     cost = self.get_costs(fuel, self.v_max, distance=1.0)
    #     heuristic_cost = self.get_costs(HEURISTIC_FUEL_TYPE, self.v_max, distance=1.0)
    #     # self.unit_cost = self.get_costs(HEURISTIC_FUEL_TYPE, self.v_max, distance=1.0)
    #     self.unit_cost = (cost / (cost + heuristic_cost))


    def p_consumed(self, velocity):
        return self.k * (velocity ** 3)

    def p_resist(self, velocity, thickness, i_f=0.52, ice_alpha=0.2, ice_beta=0.46, ice_exponentiation=6):
        # Exponential function, gives slightly off-linear slope
        # maps ice thickness (0-11.5 m) to a percentage (0.00-1.00p)
        # 0.111 0.2
        # ice_term = (ice_alpha * (np.e ** (ice_beta * thickness))) - ice_alpha 
        # 0.2 0.46
        # ice_term = ice_alpha * np.sqrt((2 * thickness) / ice_beta)
        # ice_term = (1/11.5) * thickness
        # ice_term = min(thickness * i_f, 0.80)

        # ice_term = ((thickness ** 3) / 11.5) * i_f

        # ice_term = np.power(self.ice_correction_term * thickness, 1 / self.ice_exponent) / 11.5
        # ice_term = np.power(self.ice_correction_term * thickness, self.ice_exponent) / 11.5
        ice_term = i_f * thickness
        # print(str(ice_term) + '_' + str(thickness))

        # ice_term = (-1.0 * (1 / (thickness + 1))) + 1.2
        # return self.k * (velocity ** 3) * (i_f * thickness)
        # return self.k * (velocity ** 3) * ((ice_alpha * (np.e ** (ice_beta * thickness))) - ice_alpha)
        return self.k * (velocity ** 3) * ice_term

    def p_available(self, velocity: float, thickness: float) -> float:
        '''Get the available power for the ship

        :param thickness: float - The current ice thickness 
                                  experienced by ship
        :return: float - The power available, meaning the total 
                         power of the ship minus the power 
                         required to "break" the ice

        '''
        if self.velocity != None:
            power_available = self.p_tot - self.p_resist(self.velocity, thickness)

            # Make sure the power available is realistic
            if power_available >= 0.0:
                return power_available
            else:
                return 0.0
        else:
            # If the ship is not moving, all the power is available
            return self.p_tot

    def v_limit(self, thickness):
        return max(np.cbrt(self.p_available(self.v_max, thickness) / self.k), 0.0)

    def get_trip_duration(self, velocity, thickness=0.0, distance=utils.unit_distance):
        if velocity == None or velocity <= 0.0:
            return float('inf')

        if self.p_available(velocity, thickness) <= 0.:
            return float('inf')

        return distance / velocity

    def get_trip_consumption(self, fuel, velocity, thickness=0.0, distance=utils.unit_distance):
        if self.fuel == None:
            return float('inf')

        if velocity == None or velocity <= 0.0:
            return float('inf')

        total_power = self.p_consumed(velocity) + self.p_resist(velocity, thickness)
        if total_power > self.p_tot:
            return float('inf')

        hours = ( (distance / velocity) / 60.0 ) / 60.0
        kilo_watt_hours = total_power * hours
        return (1.0 / self.fuel.lower_heating) * kilo_watt_hours

    def get_fuel_consumption(self, thickness: float, distance=utils.unit_distance) -> float:
        ''' Get the fuel consumption
        :param thickness: float - The thickness for the distance
        :param distance: float - The distance covered
        :return: float - kilograms of fuel consumed
        '''
        if self.fuel == None:
            return float('-inf')
        if self.velocity == None:
            return float('inf')

        actual_velocity = self.get_velocity(thickness)
        if actual_velocity == 0.0:
            return float('inf')

        seconds = distance / actual_velocity
        hours = seconds / 3600.0

        consumed_power = (self.velocity ** 3) * self.k
        if consumed_power > self.p_tot:
            return float('inf')

        kilo_watt_hours = consumed_power * hours
        kg_consumed = (1.0 / self.fuel.lower_heating) * kilo_watt_hours

        return kg_consumed


    def get_cost(self, thickness: float, distance=utils.unit_distance) -> float:
        ''' Get the price
        :param thickness: float - The thickness for the distance
        :param distance: float - The distance covered
        :return: float - kilograms of fuel consumed
        '''
        if self.fuel == None:
            return float('inf')
        if self.velocity == None:
            return float('inf')

        kg_consumed = self.get_fuel_consumption(thickness, distance)

        return self.fuel.get_price(kg_consumed) + self.fuel.get_emission_price(kg_consumed)

    def get_costs(self, fuel, velocity, distance=utils.unit_distance, thickness=0.0):
        kg_consumed = self.get_trip_consumption(fuel, velocity, thickness=thickness, distance=distance)
        if kg_consumed == float('inf'):
            return float('inf')
        return fuel.get_price(weight=kg_consumed) + fuel.get_emission_price(weight=kg_consumed)
    def feasible_speed_vector(self, thickness=0.0, nsteps=10):
        return np.arange(self.v_limit(thickness) / nsteps, self.v_limit(thickness), self.v_limit(thickness) / nsteps)

    def zero_initial_speed_vector(self, thickness=0.0, nsteps=10):
        return np.insert(self.feasible_speed_vector(), 0, 0.0)

    def possible_speed_vector(self, t=0.0, v=0.0):
        # new_v_max = self.v_limit(thickness=t)
        return np.arange(self.v_limit(thickness=t))

    def zero_initial_possible_speed_vector(self, thickness, velocity):
        return np.insert(self.possible_speed_vector(thickness, velocity), 0, 0.0)

class Prizna(Ship):
    def __init__(self):
        super().__init__('Prizna',
                         main_eng_pow = 792.0,
                         aux_eng_pow = 84.0,
                         design_speed = 8.0)
# class TestPrizna(unittest.TestCase):
#     def test_creation(self):
#         ship = Prizna()
#         self.assertEqual(ship.fuel, None)
#         self.assertEqual(ship.velocity, None)
        # self.assertEqual(ship.throttle_percentage, 0.0)
class TestFuelPrizna(unittest.TestCase):
    def test_no_fuel(self):
        self.assertEqual(Prizna().fuel, None)
    def test_set_fuel(self):
        ship = Prizna()
        ship.set_fuel(Methanol())
        self.assertIsInstance(ship.fuel, Methanol)
class TestVelocityPrizna(unittest.TestCase):
    def test_no_velocity(self):
        self.assertEqual(Prizna().velocity, None)

    def test_no_get_velocity(self):
        self.assertEqual(Prizna().get_velocity(0.0), 0.0)

    def test_set_velocity(self):
        ship = Prizna()
        ship.set_target_velocity(1.0)
        self.assertEqual(ship.velocity, ship.v_max)

    def test_set_velocity_negative(self):
        ship = Prizna()
        ship.set_target_velocity(-1.0)
        self.assertEqual(ship.velocity, None)

    def test_get_velocity_no_ice(self):
        ship = Prizna()
        ship.set_target_velocity(1.0)
        self.assertEqual(ship.v_max, ship.get_velocity(0.0))

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
        
    # def test_available_power_no_ice(self):
    #     ship = Prizna()
    #     ship.set_target_velocity(ship.v_max)
    #     self.assertEqual(ship.p_available(ship.velocity, 0.0), ship.p_tot)

    # def test_available_power_without_velocity_no_ice(self):
    #     ship = Prizna()
    #     # With no speed set, we must assume all the engine power
    #     # is free to use
    #     self.assertEqual(ship.p_available(ship.velocity, 0.0), ship.p_tot)

# ship = Prizna()
# ship.set_target_velocity(1.00)
# ship.set_fuel(Methanol())
# print(ship.get_fuel_consumption(1.0))
if __name__ == '__main__':
    unittest.main()
ship_list = [
    Ship("Prizna", 792.0, 84.0, 8.0),
    Ship("Kornati", 1764.0, 840.0, 12.3),
    Ship("Petar Hektorovic", 3600.0, 1944.0, 15.75)
]
# import HEURISTIC_FUEL_TYPE from fuel_class
HEURISTIC_BASAL_RATE = ship_list[0].get_trip_consumption(HEURISTIC_FUEL_TYPE, ship_list[0].v_limit(0.0))
utils.print_exit(__name__, __file__)