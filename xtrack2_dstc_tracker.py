"""{
    "wall-time": 5.825495004653931,
    "dataset": "dstc2_dev",
    "sessions": [
        {
            "session-id": "voip-f246dfe0f2-20130328_161556",
            "turns": [
                {
                    "goal-labels": {
                        "pricerange": {
                            "expensive": 0.9883454175454712
                        },
                        "area": {
                            "south": 0.9673269337257503
                        }
                    },
                    "goal-labels-joint": [
                        {
                            "slots": {
                                "pricerange": "expensive",
                                "area": "south"
                            },
                            "score": 0.9777797002475338
                        }
                    ],
                    "method-label": {
                        "byconstraints": 0.9999999999999999
                    },
                    "requested-slots": {}
                }
        }
}
"""

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


class XTrack2DSTCTracker(object):
    def __init__(self, data, model):
        self.data = data
        self.model = model

        self.classes_rev = {}
        for slot in self.data.slots:
            self.classes_rev[slot] = {val: key for key, val in
                                      self.data.classes[slot].iteritems()}

    def build_output(self, pred):

        goal_labels = {slot: self.classes_rev[slot][np.argmax(pred[i])]
                       for i, slot in enumerate(self.data.slots)}

        return {
            "goal-labels": {
                slot: {goal_labels[slot]: 1.0} for slot in self.data.slots if
                goal_labels[slot] != self.data.null_class
            },
            "method-label": {
                "byconstraints": 1.0
            },
            "requested-slots": {
                slot: 0.0 for slot in self.data.slots
            }
        }

    def track(self):
        data = self.model.prepare_data(self.data.sequences, self.data.slots)
        x = data['x']
        y_seq_id = data['y_seq_id']
        y_time = data['y_time']

        pred = self.model._predict(x, y_seq_id, y_time)
        pred_ptr = 0

        result = []
        for dialog in self.data.sequences:
            turns = []
            for lbl in dialog['labels']:
                out = self.build_output(
                    [pred[i][pred_ptr] for i, _ in enumerate(self.data.slots)]
                )
                turns.append(out)
                pred_ptr += 1

            result.append({
                'session-id': dialog['id'],
                'turns': turns
            })

        if len(pred[0]) != pred_ptr:
            raise Exception('Data mismatch.')

        return result

def main(dataset_name, data_file, output_file, model_file):
    logging.info('Loading model from: %s' % model_file)
    model = Model.load(model_file)
    logging.info('Loading data: %s' % data_file)
    data = XTrackData2.load(data_file)

    logging.info('Starting tracking.')
    tracker = XTrack2DSTCTracker(data, model)

    t = time.time()
    result = tracker.track()
    t = time.time() - t
    logging.info('Tracking took: %.1f' % t)


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
    main(**vars(parser.parse_args()))