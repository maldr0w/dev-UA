import fuel_class
import ship_class

def data_list(s, t=0.0):
    # velocity list
    velocities = s.zero_initial_speed_vector()
    # ( fuel * ( velocity * kg ) ) list
    weights = [(f, [(v, s.fuel_for_trip(f, t=t, v=v), ) for v in velocities]) for f in fuel_class.fuel_list]
    # ( fuel * ( velocity * [€ / kg] ) list
    fuel_price = [(f, [(v, f.get_price(per_v_data)) for v, per_v_data in per_f_data]) for f, per_f_data in weights]
    # ( fuel * ( velocity * tons ) ) ) list
    emission_data = [(f, [(v, f.equiv_tons_co2(per_v_data)) for v, per_v_data in per_f_data]) for f, per_f_data in weights]
    # ( fuel * ( velocity * [€ / ton] ) ) list
    emission_price = [(f, [(v, f.get_emission_price(per_v_data)) for v, per_v_data in per_f_data]) for f, per_f_data in weights]
    return (weights, fuel_price, emission_data, emission_price)

# def fuel_data_list(t=0.0):
#     # (ship * ( fuel * ( velocity * kg * € ) ) ) list
#     return [(s, [(f, [(v, s.fuel_for_trip(f, t=t, v=v), f.get_price(weight=s.fuel_for_trip(f, t=t, v=v)) for v in s.zero_initial_speed_vector()]) for f in fuel.fuel_list]) for s in vessel.ship_list]
# # def fuel_price_list(t=0.0):
# #     return [(s, [(f, [(v, f.get_price(weight=weight)) for v, weight in consumption_values]) for f, consumption_values in fuel_values]) for s, fuel_values in fuel_consumption_list(t=t)]

def trip_duration_list(ship, thickness=0.0):
    # (ship * ( fuel * ( velocity * hours ) ) ) list
    return [(v, ship.get_trip_duration(v, thickness=thickness)) for v in ship.feasible_speed_vector(thickness=thickness)]

# def emission_data_list(t=0.0):
#     # ship * ( fuel * ( velocity * tons * € ) ) list
#     return [(s, [(f, [(v, f.equiv_tons_co2(weight), f.get_emission_price(weight)) for v, weight in consumption_values]) for f, consumption_values in fuel_values]) for s, fuel_values in fuel_consumption_list(t=t)]

# def emission_price_list(t=0.0):
#     return [(s, [(f, [(v, ) for v, weight in ]) for f, emission_values in fuel_values]) for s, fuel_values in emission_amount_list(t=t)]
