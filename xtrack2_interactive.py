import time
import json
import logging
import numpy as np
import os

import dstc_util
from data_model import Dialog
from xtrack_data2 import XTrackData2
from xtrack2_dstc_tracker import XTrack2DSTCTracker
from utils import pdb_on_error
from model import Model


def main(model_file):
    logging.info('Loading model from: %s' % model_file)
    model = Model.load(model_file, build_train=False)

    x = []
    x_score = []
    x_switch = []
    y_seq_id = []
    y_time = []

    while True:
        words = raw_input('Input:')
        words = words.lower()
        tokens = words.split()
        seqs = []
        for token in tokens:
            x.append(model.vocab.get(token, model.vocab['#OOV']))
            x_score.append(11)
            x_switch.append(0)
        seqs.append({
            'data': x,
            'data_score': x_score,
            'data_actor': x_switch,
            'data_switch': x_switch,
            'labels': [
                {
                    'time': len(x) - 1,
                    'slots': {
                    },
                    'score': 0.0
                }
            ],
            'token_labels': []
        })
        data = model.prepare_data_predict(seqs, [])
            
        p = model._predict(*data)

        for slot_name, p_slot in zip(model.slots, p):
            print '#', slot_name
            preds = {}
            for cls_name, i in model.slot_classes[slot_name].iteritems():
                preds[cls_name] = p_slot[0][i]

            for cls, pp in sorted(preds.iteritems(), key=lambda x: -x[1])[:1]:
                print '%10s: %.2f' % (cls, pp, )






if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_file', required=True)

    pdb_on_error()
    from utils import init_logging
    init_logging()
    main(**vars(parser.parse_args()))
