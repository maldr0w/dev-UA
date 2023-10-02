import utils
import numpy as np
utils.print_entrypoint(__name__, __file__)
class Ship:
    def __init__(self, name, main_eng_pow=0.0, aux_eng_pow=0.0, design_speed=0.0):
        self.name = name
        self.main_eng_pow = main_eng_pow
        self.aux_eng_pow = aux_eng_pow
        self.design_speed = design_speed
        self.p_tot = self.main_eng_pow + self.aux_eng_pow
        self.v_max = 0.514 * self.design_speed
        self.k = self.p_tot / (self.v_max ** 3)

    def __str__(self):
        return f"{self.name.upper()}\n\tMain engine: {self.main_eng_pow} kW\n\tAuxiliary engine: {self.aux_eng_pow}\n\tDesign speed: {self.design_speed}\n"

    def p_consumed(self, velocity):
        return self.k * (velocity ** 3)

    def p_resist(self, velocity, thickness, i_f=0.54, ice_alpha=0.2, ice_beta=0.46):
        # Exponential function, gives slightly off-linear slope
        # maps ice thickness (0-11.5 m) to a percentage (0.00-1.00p)
        # 0.111 0.2
        # ice_term = (ice_alpha * (np.e ** (ice_beta * thickness))) - ice_alpha 
        # 0.2 0.46
        # ice_term = ice_alpha * np.sqrt((2 * thickness) / ice_beta)
        # ice_term = (1/11.5) * thickness
        ice_term = thickness * i_f
        # ice_term = (-1.0 * (1 / (thickness + 1))) + 1.2
        # return self.k * (velocity ** 3) * (i_f * thickness)
        # return self.k * (velocity ** 3) * ((ice_alpha * (np.e ** (ice_beta * thickness))) - ice_alpha)
        return self.k * (velocity ** 3) * ice_term

    def p_available(self, velocity, thickness):
        return self.p_tot - self.p_resist(velocity, thickness)

    def v_limit(self, thickness):
        return max(np.cbrt(self.p_available(self.v_max, thickness) / self.k), 0.0)

    def get_trip_duration(self, velocity, thickness=0.0, distance=utils.unit_distance):
        if velocity == None or velocity <= 0.0:
            return float('inf')

        if self.p_available(velocity, thickness) <= 0.:
            return float('inf')

        return distance / velocity

    def get_trip_consumption(self, fuel, velocity, thickness=0.0, distance=utils.unit_distance):
        if velocity == None or velocity <= 0.0:
            return float('inf')

        total_power = self.p_consumed(velocity) + self.p_resist(velocity, thickness)
        if total_power > self.p_tot:
            return float('inf')

        hours = ( (distance / velocity) / 60.0 ) / 60.0
        kilo_watt_hours = total_power * hours
        return (1.0 / fuel.lower_heating) * kilo_watt_hours

    def get_costs(self, fuel, velocity, distance, thickness):
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

ship_list = [
    Ship("Prizna", 792.0, 84.0, 8.0),
    Ship("Kornati", 1764.0, 840.0, 12.3),
    Ship("Petar Hektorovic", 3600.0, 1944.0, 15.75)
]
# import HEURISTIC_FUEL_TYPE from fuel_class
from fuel_class import HEURISTIC_FUEL_TYPE
HEURISTIC_BASAL_RATE = ship_list[0].get_trip_consumption(HEURISTIC_FUEL_TYPE, ship_list[0].v_limit(0.0))
utils.print_exit(__name__, __file__)