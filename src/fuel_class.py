import unittest
import numpy as np
from ghg_class import CO2, CH4, N2O
import ghg_class
class Fuel:
    def __init__(self, name, price_per_kg, lower_heating, co2=0.0, ch4=0.0, n2o=0.0):
        self.name = name
        self.pollutants = [
            CO2(co2),
            CH4(ch4),
            N2O(n2o),
        ]
        self.unit_gwp = np.sum([pollutant.gwp for pollutant in self.pollutants])
        # print('unit gwp ' + str(self.unit_gwp) + ' ' + name)
        self.lower_heating = lower_heating
        self.price = price_per_kg
        # self.heuristic_correction = 1.0
    def __str__(self):
        output = f"{self.name.upper()}\n\tLower heating: {self.lower_heating} kWh/kg\n\tPrice: {self.price} €/kg\n\tPollutants: "
        for pollutant in self.pollutants:
            output = f"{output}\n\t\t{pollutant}"
        return f"{output}\n"
    def set_heuristic_correction(self, other, ship):
        self_costs = ship.get_costs(self, ship.v_max)
        other_costs = ship.get_costs(other, ship.v_max)
        self.heuristic_correction = self_costs / other_costs

    def set_heuristic_cost(self, other, ship):
        s_cost = ship.get_costs(self, ship.v_max)
        o_cost = ship.get_costs(other, ship.v_max)
        self.heuristic_cost = (s_cost + o_cost) / 2
    def set_unit_price(self, ship):
        self.unit_price = ship.get_costs(self, ship.v_max, distance=1.0)
       
    def get_price(self,weight=1.0):
        return weight * self.price
            
    def get_emissions(self,weight=1.0):
        return [weight * pollutant.factor for pollutant in self.pollutants]
    def gwps(self,weight=1.0):
        return [weight * type.factor * type.gwp for type in self.emission_factors]
        # if self.emission_factors.__len__() > 0:
        #     return [weight * type.factor * get_gwp(type.name) for type in self.emission_factors]
        # else:
        #     return [0.0]
    def equiv_tons_co2(self, weight=1.0):
        return (weight * self.unit_gwp) / 1_000_000.0
    def get_emission_price(self, weight=1.0):
        return 20.0 * self.equiv_tons_co2(weight=weight)
        # return 20.0 * (np.sum(self.get_gwp(weight=weight)) / 1_000_000.0)
class Diesel(Fuel):
    def __init__(self):
        super().__init__('diesel',
                         0.78, 
                         11.83, 
                         co2=3206., 
                         ch4=0.06,
                         n2o=0.15)

class TestDiesel(unittest.TestCase):
    def test_name(self):
        diesel = Diesel()
        self.assertEqual(diesel.name,
                         'diesel',
                         'Naming error for Diesel')

    def test_price(self):
        diesel = Diesel()
        self.assertEqual(diesel.price, 
                         0.78, 
                         'Price per kg error for Diesel')

    def test_lower_heating(self):
        diesel = Diesel()
        self.assertEqual(diesel.lower_heating,
                         11.83,
                         'Lower heating error for Diesel')

    def test_pollutants(self):
        diesel = Diesel()
        self.assertIsInstance(diesel.pollutants[0], 
                              CO2,
                              'CO2 error for Diesel, type assertion failed')
        self.assertEqual(diesel.pollutants[0].factor,
                         CO2(3206.).factor, 
                         'CO2 error for Diesel, factor not correct')
        self.assertIsInstance(diesel.pollutants[1], 
                              CH4,
                              'CH4 error for Diesel, type assertion failed')
        self.assertEqual(diesel.pollutants[1].factor,
                         CH4(0.06).factor, 
                         'CH4 error for Diesel, factor not correct')
        self.assertIsInstance(diesel.pollutants[2], 
                              N2O,
                              'N2O error for Diesel, type assertion failed')
        self.assertEqual(diesel.pollutants[2].factor,
                         N2O(0.15).factor, 
                         'N2O error for Diesel, factor not correct')

    def test_unit_gwp(self):
        diesel = Diesel()
        expected_unit_gwp = CO2(3206.0).gwp + CH4(0.06).gwp + N2O(0.15).gwp
        self.assertEqual(diesel.unit_gwp,
                         expected_unit_gwp,
                         f"wrong unit gwp for diesel, got {diesel.unit_gwp}, expected {expected_unit_gwp} as")

class NaturalGas(Fuel):
    def __init__(self):
        super().__init__('natural gas',
                         0.98,
                         12.9,
                         co2=2750.0,
                         ch4=51.2,
                         n2o=0.11)

class TestNaturalGas(unittest.TestCase):
    def test_name(self):
        natural_gas = NaturalGas()
        self.assertEqual(natural_gas.name,
                         'natural gas',
                         'Naming error for NaturalGas')

    def test_price(self):
        natural_gas = NaturalGas()
        self.assertEqual(natural_gas.price, 
                         0.98, 
                         'Price per kg error for NaturalGas')

    def test_lower_heating(self):
        natural_gas = NaturalGas()
        self.assertEqual(natural_gas.lower_heating,
                         12.9,
                         'Lower heating error for NaturalGas')

    def test_pollutants(self):
        natural_gas = NaturalGas()
        self.assertIsInstance(natural_gas.pollutants[0], 
                              CO2,
                              'CO2 error for NaturalGas, type assertion failed')
        self.assertEqual(natural_gas.pollutants[0].factor, 
                         CO2(2750.0).factor,
                         'CO2 error for NaturalGas, factor not correct')

        self.assertIsInstance(natural_gas.pollutants[1], 
                              CH4,
                              'CH4 error for NaturalGas, type assertion failed')
        self.assertEqual(natural_gas.pollutants[1].factor,
                         CH4(51.2).factor)

        self.assertIsInstance(natural_gas.pollutants[2], 
                              N2O,
                              'N2O error for NaturalGas, type assertion failed')
        self.assertEqual(natural_gas.pollutants[2].factor, N2O(0.11).factor)

    def test_unit_gwp(self):
        natural_gas = NaturalGas()
        expected_unit_gwp = CO2(2750.0).gwp + CH4(51.2).gwp + N2O(0.11).gwp
        self.assertEqual(natural_gas.unit_gwp,
                         expected_unit_gwp,
                         f"wrong unit gwp for diesel, got {natural_gas.unit_gwp}, expected {expected_unit_gwp} as")

class Methanol(Fuel):
    def __init__(self):
        super().__init__('methanol',
                         0.325,
                         5.55,
                         co2=1380.0)

class TestMethanol(unittest.TestCase):
    def test_name(self):
        methanol = Methanol()
        self.assertEqual(methanol.name,
                         'methanol',
                         'Naming error for Methanol')

    def test_price(self):
        methanol = Methanol()
        self.assertEqual(methanol.price,
                         0.325,
                         'Price per kg error for Methanol')

    def test_lower_heating(self):
        methanol = Methanol()
        self.assertEqual(methanol.lower_heating,
                         5.55,
                         'Lower heating error for Methanol')
        
    def test_pollutants(self):
        methanol = Methanol()
        self.assertIsInstance(methanol.pollutants[0], 
                              CO2,
                              'CO2 error for Diesel, type assertion failed')
        self.assertEqual(methanol.pollutants[0].factor,
                         CO2(1380.0).factor,
                         'CO2 error for Methanol, factor not correct')
        self.assertIsInstance(methanol.pollutants[1], 
                              CH4,
                              'CH4 error for Diesel, type assertion failed')
        self.assertIsInstance(methanol.pollutants[2], 
                              N2O,
                              'N2O error for Diesel, type assertion failed')

    def test_unit_gwp(self):
        natural_gas = Methanol()
        expected_unit_gwp = CO2(1380.0).gwp 
        self.assertEqual(natural_gas.unit_gwp,
                         expected_unit_gwp,
                         f"wrong unit gwp for diesel, got {natural_gas.unit_gwp}, expected {expected_unit_gwp} as")

class DME(Fuel):
    def __init__(self):
        super().__init__('dme',
                         0.7,
                         8.0,
                         co2=1927.)
class TestDME(unittest.TestCase):
    def test_name(self):
        dme = DME()
        self.assertEqual(dme.name,
                         'dme',
                         'Naming error for DME')

    def test_pollutants(self):
        dme = DME()
        self.assertIsInstance(dme.pollutants[0], 
                              CO2,
                              'CO2 error for DME, type assertion failed')
        self.assertEqual(dme.pollutants[0].factor,
                         CO2(1927.).factor,
                         'CO2 error for DME, factor not correct')

        self.assertIsInstance(dme.pollutants[1], 
                              CH4,
                              'CH4 error for DME, type assertion failed')
        self.assertIsInstance(dme.pollutants[2], 
                              N2O,
                              'N2O error for DME, type assertion failed')

    def test_unit_gwp(self):
        dme = DME()
        expected_unit_gwp = CO2(1927.0).gwp 
        self.assertEqual(dme.unit_gwp,
                         expected_unit_gwp,
                         f"wrong unit gwp for DME, got {dme.unit_gwp}, expected {expected_unit_gwp} as")

class Hydrogen(Fuel):
    def __init__(self):
        super().__init__('hydrogen',
                         7.0, 
                         33.3)
class TestHydrogen(unittest.TestCase):
    def test_name(self):
        hydrogen = Hydrogen()
        self.assertEqual(hydrogen.name, 
                         'hydrogen',
                         'Name error for Hydrogen')

    def test_price(self):
        self.assertEqual(Hydrogen().price, 7.0)

    def test_lower_heating(self):
        self.assertEqual(Hydrogen().lower_heating, 33.3)

    def test_pollutants(self):
        hydrogen = Hydrogen()
        self.assertIsInstance(hydrogen.pollutants[0], CO2)
        self.assertIsInstance(hydrogen.pollutants[1], CH4)
        self.assertIsInstance(hydrogen.pollutants[2], N2O)

class B20(Fuel):
    def __init__(self):
        super().__init__('b20',
                         1.58,
                         11.83)

class TestB20(unittest.TestCase):
    def test_name(self):
        b20 = B20()
        self.assertEqual(b20.name,
                         'b20',
                         'Name error for B20')

    def test_price(self):
        self.assertEqual(B20().price, 1.58)

    def test_lower_heating(self):
        self.assertEqual(B20().lower_heating, 11.83)

    def test_pollutants(self):
        b20 = B20()
        self.assertIsInstance(b20.pollutants[0], CO2)
        self.assertIsInstance(b20.pollutants[1], CH4)
        self.assertIsInstance(b20.pollutants[2], N2O)

if __name__ == '__main__':
    unittest.main()

fuel_list = [
    Diesel(),
    NaturalGas(),
    Methanol(),
    DME(),
    Hydrogen(),
    B20(),
]

HEURISTIC_FUEL_TYPE = Methanol()