import unittest
import numpy as np
from ghg_class import CO2, CH4, N2O
from typing import List, Optional, Type
import ghg_class
from ghg_class import GHG
# ===================================================================
# Fuel Definition
EU_EMISSION_FEE_RATE = 20.0
class Fuel:
    def __init__(self, name: str, kilogram_price: float, lower_heating: float, pollutants: Optional[List[Type[GHG]]]):
        self.name = name
        self.pollutants = None
        if pollutants != None:
            self.pollutants = pollutants
        self.unit_gwp = 0.0
        if self.pollutants != None:
            self.unit_gwp = np.sum([pollutant.global_warming_potential for pollutant in self.pollutants])
        self.lower_heating = lower_heating
        self.price = kilogram_price

    def __str__(self):
        output = f"{self.name.upper()}\n\tLower heating: {self.lower_heating} kWh/kg\n\tPrice: {self.price} €/kg\n\tPollutants: "
        for pollutant in self.pollutants:
            output = f"{output}\n\t\t{pollutant}"
        return f"{output}\n"
      
    def get_consumption_price(self, weight: float) -> float:
        ''' Get the consumption price at a provided weight
        :param weight: float - the weight in question
        :return: float - the calculated price (consumption)
        '''
        if weight > 0.0:
            price = weight * self.price
            return price
        return 0.0
    
    def get_consumption_price_range(self, weights: List[float]) -> List[float]:
        ''' Get a range of consumption prices across a domain of weight values
        :param weights: List[float] - the domain of weight values
        :return: List[float] - the price range (consumption) 
        '''
        prices = []
        for weight in weights:
            if weight > 0.0:
                price = self.get_consumption_price(weight)
                prices.append(price)            
        return prices

    def get_emission_tonnage(self, weight: float) -> float:
        ''' Get the emission tonnage at a provided weight
        :param weight: float - the weight in question
        :return: float - the calculated tonnage
        '''
        if weight > 0.0:
            if self.unit_gwp > 0.0:
                grams = weight * self.unit_gwp 
                tons = grams / 1_000_000.0
                return tons
        return 0.0
    def get_emission_tonnage_range(self, weights: List[float]) -> List[float]:
        ''' Get a range of emission tonnage across a domain of weight values
        :param weights: List[float] - the domain of weight values
        :return: List[float] - the tonnage range 
        '''
        tonnage_values = []
        for weight in weights:
            if weight > 0.0:
                tons = self.get_emission_tonnage(weight)
                tonnage_values.append(tons)
        return tonnage_values

    def get_emission_price(self, weight: float) -> float:
        ''' Get the emission price at a provided weight
        :param weight: float - the weight in question
        :return: float - the calculated price (emission)
        '''
        price = EU_EMISSION_FEE_RATE * self.get_emission_tonnage(weight)
        return price
    def get_emission_price_range(self, weights: List[float]) -> List[float]:
        ''' Get a range of emission prices across a domain of weight values
        :param weights: List[float] - the domain of weight values
        :return: List[float] - the price range (emission) 
        '''
        prices = []
        for tons_emitted in self.get_emission_tonnage_range(weights):
            price = EU_EMISSION_FEE_RATE * tons_emitted 
            prices.append(price)
        return prices

    def get_total_price(self, weight: float) -> float:
        ''' Get the price at a provided weight
        :param weight: float - the weight in question
        :return: float - the calculated price (emission and consumption)
        '''
        consumption_price = self.get_consumption_price(weight)
        emission_price = self.get_emission_price(weight)
        return consumption_price + emission_price

    def get_total_price_range(self, weights: List[float]) -> List[float]:
        ''' Get a range of prices across a domain of weight values
        :param weights: List[float] - the domain of weight values
        :return: List[float] - the price range (emission and consumption) 
        '''
        prices = []
        for weight in weights:
            if weight > 0.0:
                price = self.get_total_price(weight)
                prices.append(price)
        return prices
# ===================================================================
# Diesel Definition
DIESEL_POLLUTANTS = [CO2(3206.), CH4(0.06), N2O(0.15)]
class Diesel(Fuel):
    def __init__(self):
        super().__init__('diesel',
                         0.78, 
                         11.83, 
                         DIESEL_POLLUTANTS)
# Diesel Unit Tests
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
        self.assertEqual(diesel.pollutants[0].global_warming_potential,
                         CO2(3206.).global_warming_potential, 
                         'CO2 error for Diesel, factor not correct')
        self.assertIsInstance(diesel.pollutants[1], 
                              CH4,
                              'CH4 error for Diesel, type assertion failed')
        self.assertEqual(diesel.pollutants[1].global_warming_potential,
                         CH4(0.06).global_warming_potential, 
                         'CH4 error for Diesel, factor not correct')
        self.assertIsInstance(diesel.pollutants[2], 
                              N2O,
                              'N2O error for Diesel, type assertion failed')
        self.assertEqual(diesel.pollutants[2].global_warming_potential,
                         N2O(0.15).global_warming_potential, 
                         'N2O error for Diesel, factor not correct')

    def test_unit_gwp(self):
        diesel = Diesel()
        expected_unit_gwp = CO2(3206.0).global_warming_potential + CH4(0.06).global_warming_potential + N2O(0.15).global_warming_potential
        self.assertEqual(diesel.unit_gwp,
                         expected_unit_gwp,
                         f"wrong unit gwp for diesel, got {diesel.unit_gwp}, expected {expected_unit_gwp} as")
# ===================================================================
# Natural Gas Definition
NATURAL_GAS_POLLUTANTS = [CO2(2750.0), CH4(51.2), N2O(0.11)]
class NaturalGas(Fuel):
    def __init__(self):
        super().__init__('natural gas',
                         0.98,
                         12.9,
                         NATURAL_GAS_POLLUTANTS)
# Natural Gas Unit Tests
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
        self.assertEqual(natural_gas.pollutants[0].global_warming_potential, 
                         CO2(2750.0).global_warming_potential,
                         'CO2 error for NaturalGas, factor not correct')

        self.assertIsInstance(natural_gas.pollutants[1], 
                              CH4,
                              'CH4 error for NaturalGas, type assertion failed')
        self.assertEqual(natural_gas.pollutants[1].global_warming_potential,
                         CH4(51.2).global_warming_potential)

        self.assertIsInstance(natural_gas.pollutants[2], 
                              N2O,
                              'N2O error for NaturalGas, type assertion failed')
        self.assertEqual(natural_gas.pollutants[2].global_warming_potential, N2O(0.11).global_warming_potential)

    def test_unit_gwp(self):
        natural_gas = NaturalGas()
        expected_unit_gwp = CO2(2750.0).global_warming_potential + CH4(51.2).global_warming_potential + N2O(0.11).global_warming_potential
        self.assertEqual(natural_gas.unit_gwp,
                         expected_unit_gwp,
                         f"wrong unit gwp for diesel, got {natural_gas.unit_gwp}, expected {expected_unit_gwp} as")
# ===================================================================
# Methanol Definition
METHANOL_POLLUTANTS = [CO2(1380.0)]
class Methanol(Fuel):
    def __init__(self):
        super().__init__('methanol',
                         0.325,
                         5.55,
                         METHANOL_POLLUTANTS)
# Metanol Unit Tests
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
        self.assertEqual(methanol.pollutants[0].global_warming_potential,
                         CO2(1380.0).global_warming_potential,
                         'CO2 error for Methanol, factor not correct')

    def test_unit_gwp(self):
        natural_gas = Methanol()
        expected_unit_gwp = CO2(1380.0).global_warming_potential 
        self.assertEqual(natural_gas.unit_gwp,
                         expected_unit_gwp,
                         f"wrong unit gwp for diesel, got {natural_gas.unit_gwp}, expected {expected_unit_gwp} as")
# ===================================================================
# DME Definition
DME_POLLUTANTS = [CO2(1927.)]
class DME(Fuel):
    def __init__(self):
        super().__init__('dme',
                         0.7,
                         8.0,
                         DME_POLLUTANTS)
# DME Unit Tests
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
        self.assertEqual(dme.pollutants[0].global_warming_potential,
                         CO2(1927.).global_warming_potential,
                         'CO2 error for DME, factor not correct')

    def test_unit_gwp(self):
        expected_unit_gwp = CO2(1927.0).global_warming_potential
        self.assertEqual(DME().unit_gwp,
                         expected_unit_gwp,
                         f"wrong unit gwp for DME, got {DME().unit_gwp}, expected {expected_unit_gwp} as")
# ===================================================================
# Hydrogen Definition
HYDROGEN_POLLUTANTS = None
class Hydrogen(Fuel):
    def __init__(self):
        super().__init__('hydrogen',
                         7.0, 
                         33.3,
                         HYDROGEN_POLLUTANTS)
# Hydrogen Unit Tests
class TestHydrogen(unittest.TestCase):
    def test_name(self):
        self.assertEqual(Hydrogen().name, 'hydrogen')

    def test_price(self):
        self.assertEqual(Hydrogen().price, 7.0)

    def test_lower_heating(self):
        self.assertEqual(Hydrogen().lower_heating, 33.3)

    def test_pollutants(self):
        self.assertIsNone(Hydrogen().pollutants)

    def test_unit_gwp(self):
        self.assertEqual(Hydrogen().unit_gwp, 0.0)
# ===================================================================
# B20 Definition
B20_POLLUTANTS = None
class B20(Fuel):
    def __init__(self):
        super().__init__('b20',
                         1.58,
                         11.83,
                         B20_POLLUTANTS)
# B20 Unit Tests
class TestB20(unittest.TestCase):
    def test_name(self):
        self.assertEqual(B20().name, 'b20')

    def test_price(self):
        self.assertEqual(B20().price, 1.58)

    def test_lower_heating(self):
        self.assertEqual(B20().lower_heating, 11.83)

    def test_pollutants(self):
        self.assertIsNone(B20().pollutants)

    def test_unit_gwp(self):
        self.assertEqual(B20().unit_gwp, 0.0)
# ===================================================================
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
def __from_str__(fuel_name: str) -> Type[Fuel]:
    match fuel_name:
        case 'diesel':
            return fuel_list[0]
        case 'lng':
            return fuel_list[1]
        case 'methanol':
            return fuel_list[2]
        case 'dme':
            return fuel_list[3]
        case 'hydrogen':
            return fuel_list[4]
        case 'b20':
            return fuel_list[5]
        case _:
            raise ValueError(f'{fuel_name} is not a recognized fuel name!')
HEURISTIC_FUEL_TYPE = Methanol()
