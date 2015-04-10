import collections
import itertools
import time
import json
import logging
import numpy as np
import os

import dstc_util
import data
from data_model import Dialog
from data_utils import load_dialogs
from utils import pdb_on_error
from model import Model
from model_baseline import BaselineModel
from import_dstc import ontology


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


def second_item(x):
    return x[1]


def ngramstr(ngram):
    if type(ngram) is tuple:
        return "__".join(ngram)
    else:
        return ngram


class DialogState(object):
    def __init__(self):
        self.slots = collections.defaultdict(lambda: collections.defaultdict(
            float))

    def export(self):
        res = {}
        for slot in self.slots:
            val, score = sorted(self.slots[slot].iteritems(), key=second_item,
                reverse=True)[-1]
            res[slot] = {val: 1.0}
        return res

    def set_val(self, slot, val, score):
        self.slots[slot][val] += score


SLOT_IN_NGRAM = 1.0
VALUE_IN_NGRAM = 1.0

class ManualTracker(object):
    def __init__(self, data):
        self.data = data

    def track(self, tracking_log_file_name=None, output_len_accuracy=False):
        result = []

        for dialog in self.data:
            print '>>>'
            state = DialogState()
            output = []
            d_data = zip(dialog.messages, dialog.actors, dialog.states)
            for msgs, actor, true_state in d_data:
                word_scores = collections.defaultdict(float)
                if actor == Dialog.ACTOR_USER:
                    msgs = msgs[1:]
                print msgs[0]

                for msg, score in msgs:
                    tokens = list(data.tokenize(msg))
                    for n_gram in tokens + zip(tokens, tokens[1:]) + zip(
                            tokens, tokens[1:], tokens[2:]):
                        word_scores[ngramstr(n_gram)] += np.exp(score)

                x = []
                for ngram, score in sorted(word_scores.iteritems(), key=lambda x:
                                           x[1], reverse=True)[:5]:
                    x.append((ngram, score))

                self._update(state, x)

                if actor == Dialog.ACTOR_USER:
                    output.append(state.export())
                    print '    S:', state.export()
                    print '   TS:', true_state

            result.append({
                'session-id': dialog.session_id,
                'turns': output
            })

        return result

    def _update(self, state, x):
        for ngram, score in x:
            for slot, vals in ontology.iteritems():
                ngram_score = 0
                if slot in ngram:
                    ngram_score += SLOT_IN_NGRAM
                for val in vals:
                    if val in ngram:
                        state.set_val(slot, val, ngram_score + 1)




def main(dataset_name, data_file, output_file):
    logging.info('Loading data: %s' % data_file)
    data = load_dialogs(data_file)

    logging.info('Starting tracking.')
    tracker = ManualTracker(data)

    t = time.time()
    result = tracker.track(output_len_accuracy=True)
    t = time.time() - t
    logging.info('Tracking took: %.1fs' % t)

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
    parser.add_argument('--data_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--dataset_name', required=False, default='__test__')

    pdb_on_error()
    init_logging()
    main(**vars(parser.parse_args()))
