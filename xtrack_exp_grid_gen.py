import itertools
import sys
from collections import OrderedDict

def main(*args):
    prefixes = []
    param_vals = OrderedDict()
    for arg in args[1:]:
        if arg.startswith('@'):
            prefixes.append(arg[1:])
        else:
            arg_name, arg_vals = arg.split('=')
            param_vals[arg_name] = arg_vals.split(',')

    sets = []
    for name, vals in param_vals.iteritems():
        new_set = []
        for val in vals:
            new_set.append('--%s %s' % (name, val, ))
        sets.append(new_set)


    for comb in itertools.product(*sets):
        print " ".join(prefixes + list(comb))



if __name__ == '__main__':


    main(*sys.argv)