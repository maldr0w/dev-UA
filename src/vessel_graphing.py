import matplotlib.pyplot as plt
import numpy as np
import utils
utils.print_entrypoint(__name__, __file__)
# Fuel con vs vessel speed
# a function of vessel speed
# cases include LNG diesel hydrogen methanol electric fuel
# consider ghg emissions

# Pareto frontier (?) of fuel cons and ghg
#   Pareto: set of optimal solutions that achieve best tradeoff between variables
# plot of tradeof between fuelcons for vessel speeds
# Fuel cases,

# Optmial route map for different fuels

# Ghg emissions vs vessel speed and ice thickness

# Optimization results
# A table displaying all optimal values
#   - fuel cons
#   - vessel speed
#   - route segments for chosen scenarios

# Sensitivity analysis
# analyze things like ice thickness coeff (k)
#   - with this create sens plots to show how changes in these params affect optimizaiton results



import vessel
import fuel
# def make_ship(route, name, len_pp, breadth, draught, main_eng_pow, aux_eng_pow, design_speed, trip_minutes, trip_naut_miles):
#     return {
#         "route": route,
#         # Name changed to be key in dict
#         "name": name,
#         "length_between_perpendiculars_meters": len_pp,
#         "breadth_meters": breadth,
#         "draugth_meters": draught,
#         "main_eng_pow": main_eng_pow,
#         "aux_eng_pow": aux_eng_pow,
#         "design_speed": design_speed,
#         "trip_minutes": trip_minutes,
#         "trip_naut_miles": trip_naut_miles
#     }

# ships = [
#     make_ship(name = "Prizna",
#         route = "prizna-zigljen", 
#         # Meters
#         len_pp = 52.4, breadth = 11.7, draught = 1.63, 
#         # Kilowatts
#         main_eng_pow = 792., aux_eng_pow = 84.,
#         # Knots
#         design_speed = 8.0, 
#         trip_minutes = 15.0, trip_naut_miles = 1.61
#     ),
#     make_ship(name = "Kornati", 
#         route = "ploce-trpanj",
#         # Meters
#         len_pp = 89.1, breadth = 17.5, draught = 2.40, 
#         # Kilowatts
#         main_eng_pow = 1764., aux_eng_pow = 840., 
#         # Knots
#         design_speed = 12.3, 
#         trip_minutes = 60.0, 
#         trip_naut_miles = 8.15
#     ),
#     make_ship(name = "Petar Hektorovic", 
#         route = "vis-split", 
#         # Meters
#         len_pp = 80., breadth = 18., draught = 3.8, 
#         # Kilowatts
#         main_eng_pow = 3600., aux_eng_pow = 1944., 
#         # Knots
#         design_speed = 15.75, 
#         trip_minutes = 140.0, trip_naut_miles = 30.2
#     )
# ]

#  v: speed
#  t: ice_thickness
# def get_output_power_for(ship, v_knots=None, t=None, ice_factor=0.29):
#     total_engine_power = ship['main_eng_pow'] + ship['aux_eng_pow']
#     k = total_engine_power / (ship['design_speed'] ** 3)
#     if v_knots == None:
#         # return total_engine_power
#         v_knots = ship['design_speed'] * 0.4
#     v_meters = v_knots * 0.5144444
#     # v_meters = 0.5144444 * v_knots
#     additional_power = 0.0
#     if t != None:
#         additional_power = (k * ice_factor) * (v_meters ** 3) * t


#     requested_power = k * (v_meters ** 3) + additional_power
#     # We cannot get more power out of the engine than what is available
#     if requested_power > total_engine_power:
#         return None
#     else:
#         return requested_power


# Pareto front between consumption and GHG
#  Minimize consumption
#  Minimize GHG
#  Sweep speeds, fuel cases

# Price per ton CO2
#             2025 2040
# Curr Policy   20   34
# New  Policy   22   38
# Sustainable   56  125

def tradeoff_cons_ghg(ship, fuel):
    output = []
    for v in np.arange(0, ship.v_max, 0.1):
        # Kg fuel consumed
        consumption = ship.fuel_for_trip(fuel, t=0.0, d=1.0, v=v)
        # price_for_fuel = fuel.get_price(weight=consumption)
        # g of emission, in terms of equivalent g CO2 (GHG conv by Global warming potential)
        # emissions = np.sum(fuel.get_gwp(weight=consumption))
        # tons_of_co2 = emissions / 1_000_000.0
        # price_for_emissions = 20.0 * tons_of_co2
        l = {
            'speed': v,
            'fuel_kg': consumption,
            'fuel_price': fuel.get_price(weight=consumption),
            'emis_tons': fuel.equiv_tons_co2(weight=consumption),
            'emis_price': fuel.get_emission_price(weight=consumption)
        }
        # output.append([v, [fuel.get_price(weight=consumption), fuel.get_emission_price(weight=consumption)]])
        output.append(l)
    return output

def feasible_distance_vector():
    return np.arange(100.0, 10000.0, 10.0)

def tradeoff_sweep_fuels(ship):
    output = []
    for fuel in fuel_list:
        output.append({'name': fuel.name, 'table': tradeoff_cons_ghg(ship, fuel)})
    return output

def tradeoff_sweep_ships():
    output = []
    for ship in ship_list:
        output.append({'name': ship.name, 'table': tradeoff_sweep_fuels(ship)})
    return output

# print('Tradeoffs')

# def emission_blueprint(fuel_name, co2, ch4, n2o, lower_heating, price_per_kg):
#     return {
#         'name': fuel_name,
#         'emission_factor': {
#             'co2': co2,
#             'ch4': ch4,
#             'n2o': n2o,
#         },
#         # kWh/kg
#         'lower_heating': lower_heating,
#         'price': price_per_kg
#     }

# fuels = [
#     emission_blueprint('Diesel', 3206., 0.06, 0.15, 11.83, 0.78),
#     emission_blueprint('Natural gas', 2750., 51.2, 0.11, 12.9, 0.98),
#     emission_blueprint('Methanol', 1380., None, None, 5.55, 0.325),
#     emission_blueprint('DME', 1927., None, None, 8., 0.7),
#     emission_blueprint('Hydrogen', None, None, None, 33.3, 7.0),
#     emission_blueprint('B20', None, None, None, 11.83, 1.48)
# ]

# def cons_from_engine_power(ship, fuel_type, thickness=0.0, dist=0.0, v=None):
#     if v == None:
#         return float('inf')
#     resisted_power = resistive_power(ship, t=thickness, v=v)
#     consumed_power = __p_cons(ship, v=v)
#     if consumed_power <= 0:
#         return float('inf')
#     if resisted_power + consumed_power > __p_tot(ship):
#         return float('inf')
#     hours = ((dist / v) / 60.0) / 60.0
#     kilo_watt_hours = (resisted_power + consumed_power) * hours 
#     lower_heating = fuel_type['lower_heating']
#     return (1.0 / lower_heating) * kilo_watt_hours
# def cons_per_speed(ship, fuel_type, v, d=1.0):
    

def get_fuel_consumption_for(ship, fuel_type, knots=None, trip_minutes=None, thickness=0.0, dist_meters=None):
    # The function should handle all None's internally now
    total_engine_power = get_output_power_for(ship, v_knots=knots, t=thickness)

    # Means that the requested power was above what is possible
    if total_engine_power == None:
        print("Engine power returned none.")
        return None

    else:
        # meters / (0.51444 * knots) 
        # => meters / (m / s)
        # => 1 / Hz
        # => sec 
        # If time is specified, use that
        if trip_minutes != None:
            minutes = trip_minutes
        # Else,
        else:
            # If distance is specified, use that
            if dist_meters != None:
                if knots == None:
                    minutes = (dist_meters / (0.5144444 * ship['design_speed'])) / 60.0 
                else:
                    minutes = (dist_meters / (0.5144444 * knots)) / 60.0 
            # Else, use the ships default travel time
            else:
                minutes = ship['trip_minutes']

        hours = minutes / 60.0

        kilo_watt_hours = total_engine_power * hours

        # Unit of kWh / kg
        lower_heating = fuel_type['lower_heating']

        # Unit of (kg / kWh) * kWh -> kg fuel 
        return (1.0 / lower_heating) * kilo_watt_hours

def get_emission_amount_for(ship, fuel_type, target_speed = None, trip_minutes = None):
    # Might be unnecessary, just pass with None
    # if trip_minutes != None:
    #     fc = get_fuel_consumption_for(ship, fuel_type, trip_minutes)
    # else:
    #     fc = get_fuel_consumption_for(ship, fuel_type)
    fc = get_fuel_consumption_for(ship, fuel_type, target_speed, trip_minutes)

    emission_amount = 0.0

    ef = get_emission_amount_for(fuel_type)
    for key, value in ef['pollutants']:
        # if key != 'lower_heating' and key != 'name' and value != None:
        print(key + " " + value)
        if value != None:
            emission_amount += fc * value

    # Unit of grams
    return emission_amount

def get_corrected_speed_to_consumption(ship, fuel_type, dist=None, ice_thickness = None, ice_constant = -0.25):
    speed_arr = []
    consumption_arr = []
    for x in np.arange(0.01, ship['design_speed'], 0.01):
        fc = get_fuel_consumption_for(ship, fuel_type, knots=x, dist_meters=dist, thickness=ice_thickness)
        if fc != None:
            consumption_arr.append(fc)
            speed_arr.append(x)
    return speed_arr, consumption_arr


import pareto    

ship1paretotable = pareto.Table(vessel.ship_list[0], fuel.fuel_list[0])
# print(ship1paretotable)
# print(ship1paretotable.data)
keysin = []
outertab = []
for r in ship1paretotable.rows:
    dps = []
    speeds = []
    times = []
    distances = []
    kgf = []
    pricef = []
    tone = []
    pricee = []
    for d in r.data:
        speeds.append(d[0])
        times.append(d[1])
        distances.append(d[2])
        kgf.append(d[3])
        pricef.append(d[4])
        tone.append(d[5])
        pricee.append(d[6])
        dps.append(d[0:])
        keysin.append(r.key)
    pr = np.multiply(
        np.subtract(1, np.divide(pricef, np.max(pricef))),
        np.subtract(1, np.divide(pricee, np.max(pricee)))
    )
    s = np.stack((
        np.divide(np.flip(distances), np.max(distances)),
        np.divide(np.flip(speeds), np.max(speeds)),
        np.subtract(1, pr)
        # np.divide(pricef, np.max(pricef)),
        # np.divide(pricee, np.max(pricee))
        ), axis=1)
    outertab.append(s)
fuel_speed_cost = np.array(outertab)
cost_speed_fuel = fuel_speed_cost.T
f_to_c = np.reshape(fuel_speed_cost, (fuel_speed_cost.shape[0] * fuel_speed_cost.shape[1], fuel_speed_cost.shape[2]))
# print(fuel_speed_cost)
# par = pareto.is_efficient(f_to_c)
# print(par)
# for idx in par:
#   print(f"{keysin[idx]} -> {f_to_c[idx]}")
# print(f"{keysin[8901]} -> {f_to_c[8901]}")
# print(ship1paretotable)
# space = ship1paretotable.decision_space()
# points = []
# ===================================================================
# 
#   Making the graphs
# 
# ===================================================================

plt.style.use('seaborn-v0_8')

#
#   Graph of fuel consumption versus vessel speed
#

def cons_graph(ship, ax, t=0.0):
    ax.set_title(ship.name, loc='left', fontstyle='oblique')
    ax.set_xlabel("Speed travelled [m/s]")
    ax.set_ylabel("Fuel consumed [kg]")
    velocities = np.insert(ship.feasible_speed_vector(), 0, 0.0)
    for f in fuel.fuel_list:
        ax.plot(velocities, [ship.fuel_for_trip(f, t=t, v=v) for v in velocities], label=f.name)
    ax.legend(loc='upper left')

#
#   Graph of time for trip vs vessel speed
#

def time_graph(ship, ax, t=0.0):
    ax.set_title(ship.name, loc='left', fontstyle='oblique')
    ax.set_xlabel("Speed travelled [m/s]")
    ax.set_ylabel("Time [hours]")
    velocities = np.insert(ship.feasible_speed_vector(), 0, 0.0)
    ax.plot(velocities, [(ship.time_for_trip(t=t, v=v) / 60.0) / 60.0 for v in velocities])
  
def create_graph(t=0.0):
    ship_count = vessel.ship_list.__len__()
    fig, (ax, tx) = plt.subplots(ncols = ship_count, nrows=2, sharey='row', sharex='col')
    for i in np.arange(0, ship_count):
        cons_graph(vessel.ship_list[i], ax[i], t=t)
        time_graph(vessel.ship_list[i], tx[i], t=t)
    fig.set_size_inches(18, 10)
    fig.suptitle('Fuel consumption in kilogrammes, for a 25km voyage')
    plt.savefig('images/cons_at_t' + str(t) + '.png')

create_graph()
create_graph(t=5.0)

utils.print_exit(__name__, __file__)
