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


def init_logging():
    # Setup logging.
    logger = logging.getLogger('XTrack')
    logger.setLevel(logging.DEBUG)

    logging_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    formatter = logging.Formatter(logging_format)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logging.root = logger  #


class XTrack2DSTCTracker(object):
    def __init__(self, data, model):
        self.data = data
        self.model = model

        self.classes_rev = {}
        for slot in self.data.slots:
            self.classes_rev[slot] = {val: key for key, val in
                                      self.data.classes[slot].iteritems()}

    def _label_id_to_str(self, label):
        res = {}
        for slot in self.data.slots:
            res[slot] = self.classes_rev[slot][label[slot]]
        return res

    def build_output(self, pred, label):

        goal_labels = {slot: self.classes_rev[slot][np.argmax(pred[i])]
                       for i, slot in enumerate(self.data.slots)}

        raw_goal_labels = {slot: np.argmax(pred[i])
                       for i, slot in enumerate(self.data.slots)}
        goal_labels = {slot: self.classes_rev[slot][val]
                       for slot, val in raw_goal_labels.iteritems()}

        self.track_log.write("  LBL:  %s\n" % str(self._label_id_to_str(label)))
        self.track_log.write("  PRED: %s\n" % str(self._label_id_to_str(
            raw_goal_labels)))

        goals_correct = True
        for i, slot in enumerate(self.data.slots):
            goals_correct &= raw_goal_labels[slot] == label[slot]

        return {
            "goal-labels": {
                slot: {goal_labels[slot]: 1.0} for slot in self.data.slots if
                goal_labels[slot] != self.data.null_class
            },
            "method-label": {
                #"byconstraints": 1.0
            },
            "requested-slots": {
                #slot: 0.0 for slot in self.data.slots
            }
        }, goals_correct

    def _label_empty(self, lbl):
        res = True
        for val in lbl.values():
            res &= val == 0
        return res

    def track(self, tracking_log_file_name=None):
        data = self.model.prepare_data(self.data.sequences, self.data.slots)
        x = data['x']
        x_score = data['x_score']
        x_actor = data['x_actor']
        y_seq_id = data['y_seq_id']
        y_time = data['y_time']

        pred = self.model._predict(
            x,
            #x_score,
            #x_actor,
            y_seq_id,
            y_time)
        pred_ptr = 0

        accuracy = 0
        accuracy_n = 0
        result = []
        if tracking_log_file_name:
            self.track_log = open(tracking_log_file_name, 'w')
        else:
            self.track_log = open('/dev/null', 'w')

        for dialog in self.data.sequences:
            self.track_log.write(">> Dialog: %s\n" % dialog['id'])
            self.track_log.write("\n")
            turns = []
            last_pos = 0
            for lbl in dialog['labels']:
                for word_id in dialog['data'][last_pos:lbl['time'] + 1]:
                    self.track_log.write("%s " % self.data.vocab_rev[word_id])
                self.track_log.write("\n")
                last_pos = lbl['time'] + 1
                self.track_log.write("**\n")
                out, goals_correct = self.build_output(
                    [pred[i][pred_ptr] for i, _ in enumerate(self.data.slots)],
                    lbl['slots']
                )
                turns.append(out)
                pred_ptr += 1

                if not self._label_empty(lbl['slots']):
                    if goals_correct:
                        accuracy += 1
                    accuracy_n += 1

            result.append({
                'session-id': dialog['id'],
                'turns': turns
            })

            self.track_log.write("\n")

        if len(pred[0]) != pred_ptr:
            raise Exception('Data mismatch.')

        assert accuracy_n > 0
        return result, accuracy * 1.0 / accuracy_n

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