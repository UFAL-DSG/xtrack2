import os
import sys

def main(job_id, params):
    sys.path.insert(0, os.path.dirname(__file__))
    os.environ['PYTHONUSERBASE'] = '/home/zilka/.local'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['THEANO_FLAGS'] = "base_compiledir=out/spearmint.%d" % job_id

    param_list = []
    for param, val in params.iteritems():
        param_list.append('--%s' % param[0])
        param_list.append(str(val))

    import xtrack2
    parser = xtrack2.build_argument_parser()
    args = parser.parse_args(param_list)

    return xtrack2.main(**vars(args))

