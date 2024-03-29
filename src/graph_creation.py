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



import ship_class
import fuel_class
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


# import pareto    

# ship1paretotable = pareto.Table(vessel.ship_list[0], fuel.fuel_list[0])
# # print(ship1paretotable)
# # print(ship1paretotable.data)
# keysin = []
# outertab = []
# for r in ship1paretotable.rows:
#     dps = []
#     speeds = []
#     times = []
#     distances = []
#     kgf = []
#     pricef = []
#     tone = []
#     pricee = []
#     for d in r.data:
#         speeds.append(d[0])
#         times.append(d[1])
#         distances.append(d[2])
#         kgf.append(d[3])
#         pricef.append(d[4])
#         tone.append(d[5])
#         pricee.append(d[6])
#         dps.append(d[0:])
#         keysin.append(r.key)
#     pr = np.multiply(
#         np.subtract(1, np.divide(pricef, np.max(pricef))),
#         np.subtract(1, np.divide(pricee, np.max(pricee)))
#     )
#     s = np.stack((
#         np.divide(np.flip(distances), np.max(distances)),
#         np.divide(np.flip(speeds), np.max(speeds)),
#         np.subtract(1, pr)
#         # np.divide(pricef, np.max(pricef)),
#         # np.divide(pricee, np.max(pricee))
#         ), axis=1)
#     outertab.append(s)
# fuel_speed_cost = np.array(outertab)
# cost_speed_fuel = fuel_speed_cost.T
# f_to_c = np.reshape(fuel_speed_cost, (fuel_speed_cost.shape[0] * fuel_speed_cost.shape[1], fuel_speed_cost.shape[2]))
# # print(fuel_speed_cost)
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

import data

plt.style.use('seaborn-v0_8')

#
#   Graphing helper function
#

def g_setup(title, ax, ylabel):
    ax.set_title(title, loc='left', fontstyle='oblique')
    ax.set_xlabel('Speed travelled [m/s]')
    ax.set_ylabel(ylabel)
    

def g_data(ax, values, title, ylabel):
    # (velocities, ds) = ([v for v, _ in values], [d for _, d in values])
    # print(velocities)
    # print(ds)
    g_setup(title, ax, ylabel)
    # ax.set_title(ship.name, loc='left', fontstyle='oblique')
    # ax.set_xlabel('Speed travelled [m/s]')
    # ax.set_ylabel(ylabel)
    ax.plot([v for v, _ in values], [d for _, d in values])
    # ax.legend(loc='upper left')

# def g_data(ship, ax, velocities, data, xlabel, ylabel, t=0.0):

def all_consumption_statistics_single_graphs(ship, thickness):
    shipdata = {}

    # velocities = ship.zero_initial_speed_vector(thickness=thickness) 
    # velocities = []
    # for percentage in np.arange(0.0, 1.0, 0.01):
    #     ship.set_target_velocity(percentage)
    #     print(str(percentage) + ' ' + str(ship.get_velocity(thickness)))
    #     velocities.append(ship.get_velocity(thickness))

    velocities = ship.get_velocity_range(thickness)
    shipdata['velocities'] = velocities
    # print(velocities)
    
    # X: Speed m/s <=> Y: Fuel consumption kg
    print(' 0% - Fuel consumption...')
    fig, ax = plt.subplots()
    consumptions = {}
    for fuel in fuel_class.fuel_list:
        ship.set_fuel(fuel)
        consumptions[fuel.name] = ship.get_consumption_range(thickness, 1000.0)
        g_setup(ship.name, ax, 'Fuel [kg]')
        ax.plot(velocities, consumptions[fuel.name], label=fuel.name)
    ax.legend(loc='upper left')
    # fig.suptitle('X: Speed [m * s^-1] - Y: Fuel consumption [kg]') 
    plt.savefig('graphs/thickness_' + str(thickness) + '_' + ship.name + '_y-fuelconsumption_x-speed.png')
    plt.close()

    shipdata['consumptions'] = consumptions

    # X: Speed m/s <=> Y: Fuel consumption price €
    print('20% - Fuel price...')
    fig, ax = plt.subplots()
    consumption_prices = {}
    for fuel in fuel_class.fuel_list:
        # consumption_prices[fuel.name] = [fuel.get_consumption_price(consumption) for consumption in consumptions[fuel.name]]
        consumption_prices[fuel.name] = fuel.get_consumption_price_range(consumptions[fuel.name])
        g_setup(ship.name, ax, 'Fuel [€]')
        ax.plot(velocities, consumption_prices[fuel.name], label=fuel.name)
    ax.legend(loc='upper left')
    # fig.suptitle('X: Speed [m * s^-1] - Y: Fuel consumption price [€]')
    plt.savefig('graphs/thickness_' + str(thickness) + '_' + ship.name + '_y-fuelprice_x-speed.png')
    plt.close()

    shipdata['consumption_prices'] = consumption_prices
    
    # X: Speed m/s <=> Y: Emission quantity tons
    print('40% - Emission quantities...')
    fig, ax = plt.subplots()
    emission_quantities = {}
    for fuel in fuel_class.fuel_list:
        # emission_quantities[fuel.name] = [fuel.get_emission_tonnage(consumption) for consumption in consumptions[fuel.name]]
        emission_quantities[fuel.name] = fuel.get_emission_tonnage_range(consumptions[fuel.name])
        g_setup(ship.name, ax, 'Emission quantity [tons]')
        ax.plot(velocities, emission_quantities[fuel.name], label=fuel.name)
    ax.legend(loc='upper left')
    # fig.suptitle('X: Speed [m * s^-1] - Y: Emission quantity [tons]')
    plt.savefig('graphs/thickness_' + str(thickness) + '_' + ship.name + '_y-emissionquantity_x-speed.png')
    plt.close()

    shipdata['emission_quantities'] = emission_quantities
    
    # X: Speed m/s <=> Y: Emission price €
    print('60% - Emission prices...')
    fig, ax = plt.subplots()
    emission_prices = {}
    for fuel in fuel_class.fuel_list:
        # emission_prices[fuel.name] = [fuel.get_consumption_price(consumption) for consumption in consumptions[fuel.name]]
        emission_prices[fuel.name] = fuel.get_emission_price_range(consumptions[fuel.name])
        g_setup(ship.name, ax, 'Emission price [€]')
        ax.plot(velocities, emission_prices[fuel.name], label=fuel.name)
    ax.legend(loc='upper left')
    # fig.suptitle('X: Speed [m * s^-1] - Y: Emission price [€]')
    plt.savefig('graphs/thickness_' + str(thickness) + '_' + ship.name + '_y-emissionprice_x-speed.png')
    plt.close()

    shipdata['emission_prices'] = emission_prices

    # X: Speed m/s <=> Y: Total price €
    print('80% - Total prices...')
    fig, ax = plt.subplots()
    total_prices = {}
    for fuel in fuel_class.fuel_list:
        total_prices[fuel.name] = fuel.get_total_price_range(consumptions[fuel.name])
        # total_prices[fuel.name] = [emission_prices[fuel.name][i] + consumption_price for i, consumption_price in enumerate(consumption_prices[fuel.name])]
        g_setup(ship.name, ax, 'Total price [€]')
        ax.plot(velocities, total_prices[fuel.name], label=fuel.name)
    ax.legend(loc='upper left')
    # fig.suptitle('X: Speed [m * s^-1] - Y: Emission price [€]')
    plt.savefig('graphs/thickness_' + str(thickness) + '_' + ship.name + '_y-totalprice_x-speed.png')
    plt.close()

    shipdata['total_prices'] = total_prices
    
    return shipdata

def all_properties_single_graphs(ship, thickness):
    print('Graphing all properties for ' + ship.name + '...')
    fig, ax = plt.subplots()
    # durationdata = data.trip_duration_list(ship, thickness=thickness)
    # durationdata = [(velocity, ship.get_duration_range(thickness))]
    g_setup(ship.name, ax, 'Time [seconds]')
    ax.plot(ship.get_velocity_range(thickness), ship.get_duration_range(thickness, 1000.0))
    # g_data(ax, durationdata, ship.name, 'Time [hours]')
    # fig.suptitle('X: Speed [m * s^-1] - Y: Time [hours]')
    plt.savefig('graphs/thickness_' + str(thickness) + '_y-time_x-speed.png')
    plt.close()

    shipdata = all_consumption_statistics_single_graphs(ship, thickness)    
    print('100% - Finished.\n')

    return shipdata

def create_single_graphs(thickness=0.0):
    # fig, ax = plt.subplots()
    graphdata = {}
    for ship in ship_class.ship_list:
        graphdata[ship.name] = all_properties_single_graphs(ship, thickness)
    return graphdata

def create_graphs(thickness=0.0):
    print('Creating all relevant graphs...')
    graphdata = create_single_graphs(thickness)
    print('Creating cluster-graphs...')
    fig, axes = plt.subplots(ncols = ship_class.ship_list.__len__(), nrows = fuel_class.fuel_list.__len__(), sharey='row', sharex='col')
    # Column corresponds to graph grid location
    for col, ship in enumerate(ship_class.ship_list):

        g_setup(ship.name, axes[0, col], 'Time [seconds]')
        velocities = ship.get_velocity_range(thickness)
        axes[0, col].plot(velocities, ship.get_duration_range(thickness, 1000.0))

        fuel_data = graphdata[ship.name]

        velocities = []
        # Row corresponds to graph grid location, key corresponds to the fuel
        for row, key in enumerate(fuel_data):
            label = ''
            ax = None
            data = fuel_data[key]
            match(key):
                case 'velocities':
                    velocities = data
                    continue

                case 'consumptions':
                    title = 'Fuel consumptions [kg]'
                    ax = axes[1, col]

                case 'consumption_prices':
                    title = 'Fuel prices [€]'
                    ax = axes[3, col]

                case 'emission_quantities':
                    title = 'Emission quantities [tons]'
                    ax = axes[2, col]

                case 'emission_prices':
                    title = 'Emission prices [€]'
                    ax = axes[4, col]

                case 'total_prices':
                    title = 'Total prices [€]'
                    ax = axes[5, col]

            g_setup(ship.name, ax, title)
            for fuel_name in data:
                ax.plot(velocities, data[fuel_name], label=fuel_name)
            ax.legend(loc = 'upper left')
    fig.set_size_inches(18, 20)
    fig.suptitle('Fuel consumption in kilogrammes, for a 1km voyage')
    plt.savefig('graphs/thickness_' + str(thickness) + '_cluster_graphs.png')
    plt.close()
    print('Finished.\n')

if __name__ == '__main__':
    for thickness in np.arange(0.0, 2.1, 0.1):
        create_graphs(thickness)

utils.print_exit(__name__, __file__)
