import os
import sys

def main(job_id, params):
    print params
    sys.path.insert(0, os.path.dirname(__file__))
    os.environ['PYTHONUSERBASE'] = '/home/zilka/.local'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['THEANO_FLAGS'] = "base_compiledir=out/spearmint.%d" % job_id

    param_list = ['data/xtrack/e2', '--out', 'out/spearmint.%d' % job_id, '--model_file', '/dev/null', '--final_model_file', '/dev/null', '--track_log', '/dev/null', '--rebuild_model']
    for param, val in params.iteritems():
        param_list.append('--%s' % param)
        param_list.append(str(val[0]))

    import xtrack2
    parser = xtrack2.build_argument_parser()
    print param_list
    args = parser.parse_args(param_list)
    print vars(args)

    return xtrack2.main(**vars(args))

