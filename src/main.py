
def print_help():
    print ('FLAGS')
    print ('\t --help  - Display help info')
    print ('\t -h ')
    print ('\t --test  - Perform profiled test run')
    print ('\t -t')
    print ('\t --start - Input start coordinate')
    print ('\t -s        Ex: --start=<lat>,<lon> (no spaces, . as decimal point)')
    print ('\t --end   - Input end coordinate')
    print ('\t -e        Ex: --end=<lat>,<lon> (no spaces, . as decimal point)')

import sys, getopt
def main(argv):
    start, end = None, None
    opts, args = getopt.getopt(argv, "hts:e:", ["help","test","start=","end="])
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print_help()
            sys.exit()
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
    if start != None and end != None:
        import a_star
        a_star.run_search (start, end)
    else:
        print ('Error with coordinates')
        print_help()
    print ('Exiting...')
    sys.exit()

if __name__ == "__main__":
    main(sys.argv[1:])
