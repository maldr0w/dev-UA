import vessel
import utils
import numpy as np

utils.print_entrypoint(__name__, __file__)

# PARETO HELPER CLASSES
#  Pareto node: A single node of data, for quick computation
class Node:
    def __init__(self, ship, fuel, v, t=0.0, d=1.0):
        self.key = v
        self.time = ship.time_for_trip(t=t, d=d, v=v)
        self.distance = d
        self.kg_fuel = ship.fuel_for_trip(fuel, t=t, d=d, v=v)
        self.price_fuel = fuel.get_price(weight=self.kg_fuel)
        self.tons_emis = fuel.equiv_tons_co2(weight=self.kg_fuel)
        self.price_emis = fuel.get_emission_price(weight=self.kg_fuel)
        self.data = [self.key, self.time, self.distance, self.kg_fuel, self.price_fuel, self.tons_emis, self.price_emis]
    def __str__(self):
        return f"@ {self.key} m/s,\n  {self.kg_fuel} kg\t-> {self.price_fuel} €\n  {self.tons_emis} tons\t-> {self.price_emis}"
    def price_tot(self):
        return self.price_fuel + self.price_emis
    # def data(self):
    #     return [self.speed, self.kg_fuel, self.price_fuel, self.tons_emis, self.price_emis]
#  Pareto row: A row of pareto nodes, each of which corresponding to some speed
class Row:
    def __init__(self, ship, fuel, t=0.0, d=1.0):
        self.key = d
        self.nodes = []
        self.data = []
        for v in ship.feasible_speed_vector():
            node = Node(ship, fuel, v, t=t, d=d)
            self.nodes.append(node)
            self.data.append(node.data)
    def __str__(self):
        output = ""
        for node in self.nodes:
            output = f"{output}\n{self.key} m, {node}\n"
        return output
    def decision_space(self):
        return [(node.key, node.time, node.distance, node.kg_fuel, node.price_fuel, node.tons_emis, node.price_emis) for node in self.nodes]
#  Pareto table: A collection of pareto rows, each corresponding to some fuel
class Table:
    def __init__(self, ship, fuel, t=0.0):
        self.key = fuel.name
        self.rows = []
        self.data = []
        for d in np.arange(0., utils.unit_distance, 1000.0):
            row = Row(ship, fuel, t=t, d=d)
            self.rows.append(row)
            self.data.append(row.data)
        # for fuel in fuel_types:
        #     row = ParetoRow(ship, fuel, t=t, d=d)
        #     self.rows.append(row)
        #     self.data.append(row.data)
    def __str__(self):
        output = f"{self.key}"
        for row in self.rows:
            output = f"{output}\n\t{row}\n"
        return output
    def decision_space(self):
        return [(self.name, row.key, row.decision_space()) for row in self.rows]
class Space:
    def __init__(self, ship, fuel_types, t=0.0, d=1.0):
        self.key = ship.name
        self.tables = []
        self.data = []
        for fuel in fuel_types:
            table = Table(ship, fuel, t=t, d=d)
            self.tables.append(table)
            self.data.append(table.data)
    def __str__(self):
        output = ""
        for table in self.tables:
            output = f"{output}\n\t{table}\n"
        return output

def is_efficient(costs):
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_idx = 0
    while next_idx < len(costs):
        non_dom_pnt_mask = np.any(costs < costs[next_idx], axis=1)
        # print(non_dom_pnt_mask)
        non_dom_pnt_mask[next_idx] = True
        # print(non_dom_pnt_mask)
        is_efficient = is_efficient[non_dom_pnt_mask]
        # print(is_efficient)
        costs = costs[non_dom_pnt_mask]
        # print(costs)
        next_idx = np.sum(non_dom_pnt_mask[:next_idx]) + 1
    return is_efficient

utils.print_exit(__name__, __file__)