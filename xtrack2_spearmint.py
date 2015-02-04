import xtrack2


def main(job_id, params):
    params['experiment_path'] = "data/xtrack/e2"
    params['eid'] = str(job_id)
    params['rebuild_model'] = True
    params['model_file'] = '/dev/null'
    params['final_model_file'] = '/dev/null'
    params['out'] = 'out/spearmint.%s' % str(job_id)
    params['n_epochs'] = 200
    params['oclf_n_hidden'] = 32
    params['oclf_n_layers'] = 2
    params['oclf_activation'] = 'tanh'
    params['lstm_n_layers'] = 1
    params['track_log'] = '/dev/null'

    return xtrack2.main(**params)

