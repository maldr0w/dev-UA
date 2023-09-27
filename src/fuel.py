import numpy as np
def get_gwp(ghg_type):
    match ghg_type:
        case 'co2':
            return 1
        case 'ch4':
            return 25
        case 'n2o':
            return 298

class GHG:
    def __init__(self, name, factor):
        self.name = name
        if factor != None:
            self.factor = factor
        else:
            self.factor = 0.0
    def __str__(self):
        return f"{self.name.upper()}\t{self.factor} g/kg"
    
class Fuel:
    def __init__(self, name, price_per_kg, lower_heating, co2=None, ch4=None, n2o=None):
        self.name = name
        self.emission_factors = [
            GHG('co2', co2),
            GHG('ch4', ch4),
            GHG('n2o', n2o)
        ]
        self.lower_heating = lower_heating
        self.price = price_per_kg

    def __str__(self):
        output = f"{self.name.upper()}\n\tLower heating: {self.lower_heating} kWh/kg\n\tPrice: {self.price} €/kg\n\tEmission factors: "
        for ef in self.emission_factors:
            output = f"{output}\n\t\t{ef}"
        return f"{output}\n"

    def get_price(self,weight=1.0):
        return weight * self.price
    def get_emissions(self,weight=1.0):
        return [weight * type.factor for type in self.emission_factors]
    def get_gwp(self,weight=1.0):
        if self.emission_factors.__len__() > 0:
            return [weight * type.factor * get_gwp(type.name) for type in self.emission_factors]
        else:
            return [0.0]
    def equiv_tons_co2(self, weight=1.0):
        return (np.sum(self.get_gwp(weight=weight))/ 1_000_000.0)
    def get_emission_price(self, weight=1.0):
        return 20.0 * self.equiv_tons_co2(weight=weight)
        # return 20.0 * (np.sum(self.get_gwp(weight=weight)) / 1_000_000.0)
fuel_list = [
    Fuel('diesel', 0.78, 11.83, 
        co2=3206., ch4=0.06, n2o=0.15),
    Fuel('natural gas', 0.98, 12.9, 
        co2=2750., ch4=51.2, n2o=0.11),
    Fuel('methanol', 0.325, 5.55, 
        co2=1380),
    Fuel('dme', 0.7, 8., 
        co2=1927.),
    Fuel('hydrogen', 7.0, 33.3),
    Fuel('b20', 1.58, 11.83),
]
