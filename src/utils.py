verbose_mode = False
def enable_verbose_mode():
    verbose_mode = True
separator = '\n<' + (68 * '=') + '>\n'
def print_separator():
    if verbose_mode:
        s = 68 * '='
        print('\n<' + s + '>\n')
def print_entrypoint(ep, file):
    if verbose_mode:
        print_separator()
        match ep:
            case '__main__':
                print('\tEntrypoint: ' + file)
            case _:
                print('\tImporting ' + ep + '\t(' + file + ')')
        print('\\\n \\\n ||\n ||')
def print_exit(ep, file):
    if verbose_mode:
        print(' ||\n ||\n /\n/')
        print('\tExiting ' + ep + '\t(' + file + ')')

k_ice = 1.00
unit_distance = 25000.0
