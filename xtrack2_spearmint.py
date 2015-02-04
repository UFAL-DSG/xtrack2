import os
import sys

def main(job_id, test_params):
    sys.path.insert(0, os.path.dirname(__file__))
    os.environ['PYTHONUSERBASE'] = '/home/zilka/.local'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['THEANO_FLAGS'] = "base_compiledir=out/spearmint.%d" % job_id

    params = {}
    params['experiment_path'] = "data/xtrack/e2"

    params['eid'] = str(job_id)
    
    params['rebuild_model'] = True
    params['model_file'] = '/dev/null'
    params['final_model_file'] = '/dev/null'
    params['out'] = 'out/spearmint.%s' % str(job_id)
    
    params['n_cells'] = 5
    params['emb_size'] = 7
    params['n_epochs'] = 200
    params['lr'] = 0.1
    params['p_drop'] = 0.0
    params['opt_type'] = 'sgd'
    params['mb_size'] = 16
    params['init_emb_from'] = None
    
    params['oclf_n_hidden'] = 32
    params['oclf_n_layers'] = 2
    params['oclf_activation'] = 'tanh'
    params['lstm_n_layers'] = 1

    params['track_log'] = '/dev/null'
    params['debug'] = False

    params.update(test_params)

    import xtrack2

    return xtrack2.main(**params)

