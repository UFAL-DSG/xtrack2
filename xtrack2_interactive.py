import time
import json
import logging
import numpy as np
import os

import dstc_util
from data_model import Dialog
from xtrack_data2 import XTrackData2
from utils import pdb_on_error
from model import Model


def main(dataset_name, data_file, output_file, model_file):
    logging.info('Loading model from: %s' % model_file)
    model = Model.load(model_file)

    logging.info('Loading data: %s' % data_file)
    data = XTrackData2.load(data_file)

    logging.info('Starting tracking.')
    tracker = XTrack2DSTCTracker(data, model)

    t = time.time()
    result, accuracy = tracker.track()
    t = time.time() - t
    logging.info('Tracking took: %.1fs' % t)
    logging.info('Accuracy: %.2f %%' % (accuracy * 100))


    tracker_output = {
        'wall-time': t,
        'dataset': dataset_name,
        'sessions': result
    }

    logging.info('Writing to: %s' % output_file)
    with open(output_file, 'w') as f_out:
        json.dump(tracker_output, f_out, indent=4)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='__test__')
    parser.add_argument('--data_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--model_file', required=True)

    pdb_on_error()
    init_logging()
    main(**vars(parser.parse_args()))