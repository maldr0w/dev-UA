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

import sys, getopt
def main(argv):
    start, end = None, None
    thickness = None
    opts, args = getopt.getopt(argv, "hg:tps:e:v", ["help","graph=", "test", "profile","start=","end=","verbose"])
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print_help()
            sys.exit()
        elif opt in ('-g', '--graph'):
            thickness = float(arg)  
        elif opt in ('-s', '--start'):
            coords = arg.split(',')
            start = (float(coords[0]), float(coords[1]))
        elif opt in ('-e', '--end'):
            coords = arg.split(',')
            end = (float(coords[0]), float(coords[1]))
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
    if start != None and end != None:
        import a_star
        a_star.run_search (start, end)        
    elif thickness != None:
        import graph_creation
        graph_creation.create_graphs(thickness)
    else:
        print ('ERROR: Input error\n')
        print_help()
    print ('Exiting...')
    sys.exit()

if __name__ == "__main__":
    main(sys.argv[1:])
