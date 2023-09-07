import utils
import numpy as np
utils.print_entrypoint(__name__, __file__)
class Vessel:
    def __init__(self, name, main_eng_pow=0.0, aux_eng_pow=0.0, design_speed=0.0):
        self.name = name
        self.main_eng_pow = main_eng_pow
        self.aux_eng_pow = aux_eng_pow
        self.design_speed = design_speed
        self.p_tot = self.main_eng_pow + self.aux_eng_pow
        self.v_max = 0.514 * self.design_speed
        self.k = self.p_tot / (self.v_max ** 3)
    # def p_tot(self):
    #     return self.main_eng_pow + self.aux_eng_pow
    # def v_max(self):
    #     return 0.514 * self.design_speed
    # def k(self):
    #     return self.p_tot / (self.v_max ** 3)
    def __str__(self):
        return f"{self.name.upper()}\n\tMain engine: {self.main_eng_pow} kW\n\tAuxiliary engine: {self.aux_eng_pow}\n\tDesign speed: {self.design_speed}\n"
    def p_cons(self, v=None):
        if v == None:
            return self.k * (self.v_max ** 3)
        else:
            return self.k * (v ** 3)

    def p_resist(self, t=0.0, i_f=0.47, v=None):
        if v == None:
            return (i_f * self.k) * (self.v_max ** 3) * t
            # p = (i_f * self.k) * (self.v_max ** 3) * t
            # if p > self.p_tot:
            #     return float('inf')
            # return p
        else:
            return (i_f * self.k) * (v ** 3) * t

    def p_surplus(self, t=0.0, v=None):        
        if self.p_tot < self.p_resist(t=t, v=v):
            return float('inf')
        return self.p_tot - self.p_resist(t=t, v=v)

    def v_limit(self, t=0.0, v=None):
        p_available = self.p_tot - self.p_resist(t=t)
        return np.cbrt(p_available / self.k)
    def time_for_trip(self, t=0.0, d=utils.unit_distance, v=None):
        if v == None:
            return float('inf')
        if v <= 0.0:
            return float('inf')

        p_resisted = self.p_resist(t=t, v=v)
        if p_resisted > self.p_tot:
            return float('inf')
        # v_resisted = np.cbrt(p_resisted) / self.k
        p_available = self.p_tot - p_resisted

        p_consumed = self.p_cons(v=v)
        if p_consumed <= 0.0:
            return float('inf')
        # if p_consumed + p_resisted > self.p_tot:
        #     return float('inf')
        # return d / (v - v_resisted)
        if p_consumed > p_available:
            return d / (np.cbrt(p_available / self.k))
            # return float('inf')
        return d / v
    def fuel_for_trip(self, fuel, t=0.0, d=utils.unit_distance, v=None):
        if v == None:
            return float('inf')
        if v < 0.0:
            return float('inf')
        if v == 0.0:
            return 0.0

        p_resisted = self.p_resist(t=t, v=v)
        if p_resisted > self.p_tot:
            return float('inf')
        p_available = self.p_tot - p_resisted
        # v_resisted = np.cbrt(p_resisted) / self.k

        # v = v - v_resisted

        p_consumed = self.p_cons(v=v)
        if p_consumed == 0.0:
            # If no power was consumed, no fuel was used
            return 0.0
        if p_consumed < 0:
            return float('inf')
        # if p_resisted + p_consumed > self.p_tot:
        #     # Should I correct speed here?
        #     return float('inf')
        if p_consumed > p_available:
            # Should I correct speed here?
            v = np.cbrt(p_available / self.k) 
            p_final = self.p_tot
        else:
            p_final = p_resisted + p_consumed

        h = ( (d / v) / 60.0 ) / 60.0
        kilo_watt_hours = p_final * h
        return (1.0 / fuel.lower_heating) * kilo_watt_hours
    def feasible_speed_vector(self, nsteps=10):
        return np.arange(self.v_max / float(nsteps), self.v_max, self.v_max / float(nsteps))
    def possible_speed_vector(self, t=0.0, v=0.0):
        new_v_max = self.v_limit(t=t, v=v)
        return np.arange(self.v_limit(t=t, v=v))

ship_list = [
    Vessel("Prizna", 792.0, 84.0, 8.0),
    Vessel("Kornati", 1764.0, 840.0, 12.3),
    Vessel("Petar Hektorovic", 3600.0, 1944.0, 15.75)
]
utils.print_exit(__name__, __file__)