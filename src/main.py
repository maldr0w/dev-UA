def print_help():
    print ('FLAGS')
    print ('\t --help     - Display help info')
    print ('\t -h ')
    print ('\t --graph    - Generate graphs at the specified ice thickness')
    print ('\t -g           Ex: --graph=<thickness> (. as decimal point)')
    print ('\t --test     - Perform test run')
    print ('\t -t')
    print ('\t --profile  - Perform profiled run')
    print ('\t -p')
    print ('\t --start    - Input start coordinate')
    print ('\t -s           Ex: --start=<lat>,<lon> (no spaces, . as decimal point)')
    print ('\t --end      - Input end coordinate')
    print ('\t -e           Ex: --end=<lat>,<lon> (no spaces, . as decimal point)')
    print ('\t --verbose  - Enable verbose mode')
    print ('\t -v')
    print ('\t --fuel     - Specify Fuel type')
    print ('\t -f           Possible values [\'Diesel\', \'LNG\', \'Methanol\', \'DME\', \'Hydrogen\', \'B20\']')
    print ('\t --ship     - Specify ship')
    print ('\t -b           Possible value [\'Prizna\', \'Kornati\', \'Petar\']')

import sys, getopt
def main(argv):
    start, end = None, None
    thickness = None
    defined_options = { 
        'start': None , 
        'end': None, 
        'thickness': None, 
        'fuel': None, 
        'ship': None 
    }
    opts, args = getopt.getopt(argv, "hg:tps:e:vf:b:", ["help","graph=", "test", "profile","start=","end=","verbose", "fuel=", "ship="])
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print_help_and_exit()
        elif opt in ('-g', '--graph'):
            defined_options['thickness'] = float(arg)
        elif opt in ('-s', '--start'):
            coords = arg.split(',')
            if len(coords) != 2:
                raise ValueError('Bad start coordinate syntax')
            else:
                defined_options['start'] = (float(coords[0]), float(coords[1]))
        elif opt in ('-e', '--end'):
            coords = arg.split(',')
            if len(coords) != 2:
                raise ValueError('Bad end coordinate syntax')
            else:
                defined_options['end'] = (float(coords[0]), float(coords[1]))
        elif opt in ('-f', '--fuel'):
            fuel = str.lower(arg)
            if fuel not in ['diesel', 'lng', 'methanol', 'dme', 'hydrogen', 'b20']:
                raise ValueError('Bad fuel name')
            else:
                defined_options['fuel'] = fuel
        elif opt in ('-b', '--ship'):
            ship = str.lower(arg)
            if ship not in ['prizna', 'kornati', 'petar']:
                raise ValueError('Bad ship name')
            else:
                defined_options['ship'] = ship
        elif opt in ('-t', '--test'):
            import a_star
            a_star.test()
            sys.exit()
        elif opt in ('-p', '--profile'):
            import a_star
            a_star.profile()
            sys.exit()
        elif opt in ('-v', '--verbose'):
            import utils
            utils.verbose_mode = True

    if defined_options['start'] != None and defined_options['end'] != None:
        import a_star
        a_star.run_search(defined_options['start'], defined_options['end'], defined_options['ship'], defined_options['fuel'])
    elif defined_options['thickness'] != None:
        import graph_creation
        graph_creation.create_graphs(defined_options['thickness'])
    else:
        print_help_and_exit()

def print_help_and_exit():
    print_help()
    sys.exit()

if __name__ == "__main__":
    main(sys.argv[1:])
