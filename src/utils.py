def print_separator():
    s = 68 * '='
    print('\n<' + s + '>\n')
def print_entrypoint(ep, file):
    print_separator()
    match ep:
        case '__main__':
            print('\tEntrypoint: ' + file)
        case _:
            print('\timporting ' + ep + '\t(' + file + ')')
    print('\\\n \\\n ||\n ||')
def print_exit(ep, file):
    print(' ||\n ||\n /\n/')
    print('\texiting ' + ep + '\t(' + file + ')')

k_ice = 1.00
unit_distance = 25000.0
