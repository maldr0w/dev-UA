import unittest
from typing import Optional, Type
'''A class to manage internal behavior of GHG'''
class GHG:
    def __init__(self, name: str, global_warming_potential: Optional[float]):
        '''Constructor

        :param name: str - Name of a greenhouse gas, either CO2, CH4, N2O 
        :param global_warming_potential: float - The extent to which a fuel releases the specific GHG
        :return: GHG - A GHG class-object allowing calculation of GHG-emissions for the fuel
        '''
        self.name = name
        if global_warming_potential != None:
            self.global_warming_potential = global_warming_potential
        else:
            self.global_warming_potential = 0.0

    def __str__(self):
        '''String representation

        :return: str - A string of the form ghg-name[INDENT]ghg-factor
        '''
        return f"{self.name.upper()}\t{self.global_warming_potential} g/kg"

class CO2(GHG):
    def __init__(self, global_warming_potential: Optional[float]):
        super().__init__('co2', global_warming_potential)
        self.global_warming_potential = global_warming_potential

class TestCO2(unittest.TestCase):
    def test_get_gwp(self):
        ghg = CO2(2.0)
        self.assertEqual(ghg.global_warming_potential, 2.0, "incorrect gwp for CO2")

    def test_get_name(self):
        ghg = CO2(1.0)
        self.assertEqual(ghg.name, 'co2', "incorrect name for CO2")

class CH4(GHG):
    def __init__(self, global_warming_potential):
        super().__init__('ch4', global_warming_potential)
        self.global_warming_potential = 25.0 * global_warming_potential

class TestCH4(unittest.TestCase):
    def test_get_gwp(self):
        ghg = CH4(2.0)
        self.assertEqual(ghg.global_warming_potential, 50.0, "incorrect gwp for CH4")

    def test_get_name(self):
        ghg = CH4(1.0)
        self.assertEqual(ghg.name, 'ch4', "incorrect name for CH4")

class N2O(GHG):
    def __init__(self, global_warming_potential):
        super().__init__('n2o', global_warming_potential)
        self.global_warming_potential = 298.0 * global_warming_potential

class TestN2O(unittest.TestCase):
    def test_get_gwp(self):
        ghg = N2O(2.0)
        self.assertEqual(ghg.global_warming_potential, 596.0, "incorrect gwp for N2O")

    def test_get_name(self):
        ghg = N2O(1.0)
        self.assertEqual(ghg.name, 'n2o', "incorrect name for N2O")

if __name__ == '__main__':
    unittest.main()
