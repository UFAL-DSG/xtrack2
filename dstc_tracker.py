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
import collections
import itertools
import time
import json
import logging
import numpy as np
import os

import dstc_util
from data_model import Dialog
from data import Data, Tagger
from utils import pdb_on_error
from model import Model
from model_baseline import BaselineModel


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
    def __init__(self, data, models, override_slots):
        assert len(models) > 0, 'You need to specify some models.'

        self.data = data
        self.override_slots = override_slots
        self.models = models
        self.main_model = models[0]

        self.classes_rev = {}
        for slot in self.data.slots:
            self.classes_rev[slot] = {val: key for key, val in
                                      self.data.classes[slot].iteritems()}

        self.slot_groups = data.slot_groups
        self.tagger = Tagger()

    def _label_id_to_str(self, slots, label):
        res = {}
        for slot in slots:
            res[slot] = self.classes_rev[slot][label[slot]]
        return res

    def _fill_method_label(self, method_label, pred, raw_label_probs):
        method = pred.get('method')
        if method:
            method_label[method] = raw_label_probs['method']

    def _fill_req_slots(self, slots, req_slots, pred, raw_label_probs):
        for slot in slots:
            if slot.startswith('req_'):
                if pred[slot] != self.data.null_class:
                    req_slots[slot[4:]] = raw_label_probs[slot]

    def build_output(self, slots, pred, label, tags):
        #self._boost_tagged_preds(pred, tags)
        raw_labels = {}
        raw_label_probs = {}
        for i, slot in enumerate(slots):
            val = np.argmax(pred[i])
            val_prob = pred[i][val]
            if pred[i][val] == 0.0:
                val = 0

            raw_labels[slot] = val
            raw_label_probs[slot] = float(val_prob)

        lbl = self._label_id_to_str(slots, label)
        pred = self._label_id_to_str(slots, raw_labels)
        for slot in slots:
            self.track_log.write("  %s lbl(%s) pred(%s)\n" % (slot,
                                                              lbl[slot], pred[slot]))

        goals_correct = {}
        for group, gslots in self.slot_groups.iteritems():
            goals_correct[group] = True
            for i, slot in enumerate(gslots):
                if self.override_slots and not slot in self.override_slots:
                    continue

                if not slot in raw_labels:
                    continue

                goals_correct[group] &= (raw_labels[slot] == label[slot])

        goal_labels = {
            slot: {pred[slot]: 1.0} #raw_label_probs[slot]}
            for slot in slots
            if pred[slot] != self.data.null_class and
                slot in ['food', 'area','location', 'pricerange', 'name']
        }
        method_label = {}
        self._fill_method_label(method_label, pred, raw_label_probs)

        req_slots = {}
        self._fill_req_slots(slots, req_slots, pred, raw_label_probs)

        goal_labels_debug = {
                slot: goal_labels[slot].keys()[0] for slot in goal_labels
        }

        return {
            "goal-labels": goal_labels,
            "method-label": method_label,
            "requested-slots": req_slots,
            "debug": goal_labels_debug
        }, goals_correct

    def _label_empty(self, lbl):
        res = True
        for val in lbl.values():
            res &= val == 0
        return res

    def _make_model_predictions(self, data):
        preds = collections.defaultdict(list)
        for model in self.models:
            pred = model._predict(*data)

            assert len(model.slots) == len(pred)
            for slot, slot_pred in zip(model.slots, pred):
                preds[slot].append(slot_pred)

        return preds

    def _boost_tagged_preds(self, pred, tags):
        assert len(pred) >= len(tags)

        n_repl = 0

        orig = np.array(pred)

        for slot_pred, (slot, tagged_vals) in zip(pred, tags.iteritems()):
            for i in range(len(tagged_vals)):
                try:
                    tag_ndx = self.data.classes[slot]['#%s%d' % (slot, i, )]
                except KeyError:
                    continue

                n_repl += 1

                val_ndx = self.data.classes[slot][tagged_vals[i]]

                total = slot_pred[tag_ndx] + slot_pred[val_ndx]

                if slot_pred[tag_ndx] > slot_pred[val_ndx]:
                    val_delta = 0.00001
                else:
                    val_delta = -0.00001
                slot_pred[tag_ndx] = total
                slot_pred[val_ndx] = total + val_delta

        #if n_repl > 0:
        #    import ipdb; ipdb.set_trace()


    def track(self, tracking_log_file_name=None, output_len_accuracy=False):
        max_batch_size = 200
        preds = []
        all_preds = []
        slots = []
        for i in range(len(self.data.sequences) / max_batch_size + 1):
            seqs = self.data.sequences[i * max_batch_size:(i + 1) * max_batch_size]
            if not seqs:
                break

            data = self.main_model.prepare_data_predict(seqs)
            model_preds = self._make_model_predictions(data)

            ipred = []
            # Model averaging.
            for slot, slot_preds in model_preds.iteritems():
                if i == 0:
                    slots.append(slot)
                slot_res = np.array(slot_preds[0])
                for slot_pred in slot_preds[1:]:  # For each model's results.
                    slot_res += slot_pred
                ipred.append(slot_res / len(slot_preds))

            preds.append(ipred)

        pred = []
        for islot_pred in zip(*preds):
            pred.append(np.concatenate(islot_pred))

        pred_ptr = 0

        len_accuracy = collections.defaultdict(lambda:
                                               collections.defaultdict(int))
        len_accuracy_n = collections.defaultdict(lambda:
                                               collections.defaultdict(int))
        accuracy = collections.defaultdict(int)
        accuracy_n = collections.defaultdict(int)
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
            state_component_mentioned = False
            for lbl in dialog['labels']:
                words = dialog['data_debug'][last_pos:lbl['time'] + 1]
                words_score = dialog['data_score'][last_pos:lbl['time'] + 1]
                for wcn, wcn_score in zip(words, words_score):
                    self.track_log.write("  ")
                    for ws, w in sorted(zip(wcn_score, wcn), reverse=True):
                        self.track_log.write("%10s (%.2f)" % (w, ws ))
                    self.track_log.write("\n")
                self.track_log.write("\n")
                #words = dialog['data'][last_pos:lbl['time'] + 1]
                #if 'data_score' in dialog:
                #    word_probs = dialog['data_score'][last_pos:lbl['time'] + 1]
                #else:
                #    word_probs = itertools.repeat(-1)
                #last_word_p = None
                #for word_p, word_id in zip(word_probs, words):
                #    if word_p != last_word_p:
                #        self.track_log.write('\n%.2f ' % word_p)
                #        last_word_p = word_p
                #    self.track_log.write("%s " % self.data.vocab_rev[word_id])


                last_pos = lbl['time'] + 1

                out, hyp_correct = self.build_output(
                    slots,
                    [pred[i][pred_ptr] for i, _ in enumerate(slots)],
                    lbl['slots'],
                    dialog['tags']
                )
                self.track_log.write("#" * 100)
                self.track_log.write("\n\n")

                if dialog['tags']:
                    self._replace_tags(out, dialog['tags'], turns)
                #self.track_log.write(json.dumps(out))
                #self.track_log.write("\n")
                turns.append(out)
                pred_ptr += 1

                #if not self._label_empty(lbl['slots']) or state_component_mentioned:
                #    state_component_mentioned = True

                for group, gslots in self.slot_groups.iteritems():
                    if set(gslots).intersection(lbl['slots_mentioned']):
                        if hyp_correct[group]:
                            accuracy[group] += 1
                            len_accuracy[last_pos][group] += 1
                        accuracy_n[group] += 1
                        len_accuracy_n[last_pos][group] += 1
                    #else:
                    #print gslots, lbl['slots_mentioned']

            result.append({
                'session-id': dialog['id'],
                'turns': turns
            })

            #self.track_log.write("\n")

        assert pred_ptr == len(pred[0])

        self.track_log.close()

        if len(pred[0]) != pred_ptr:
            raise Exception('Data mismatch.')

        for group in self.slot_groups:
            logging.info(accuracy_n[group])
            accuracy[group] = accuracy[group] * 1.0 / max(1, accuracy_n[group])
            for t in len_accuracy:
                factor = 1.0 / max(1, len_accuracy_n[t][group])
                len_accuracy[t][group] = len_accuracy[t][group] * factor

        res = [result, accuracy]
        if output_len_accuracy:
            res.append(len_accuracy)
            res.append(len_accuracy_n)
        return tuple(res)

    def _replace_tags(self, out, tags, turns):
        for slot, values in out['goal-labels'].iteritems():
            self._replace_tags_for_slot(slot, tags, values, turns)

        self._replace_tags_for_slot('method', tags, out['method-label'], turns)

        # TODO: Also replace requested.

    def _replace_tags_for_slot(self, slot, tags, values, turns):
        if turns:
            last_turn = turns[-1]
        else:
            last_turn = None

        new_res = {}
        for slot_val, p in values.iteritems():
            slot_val = slot_val.replace('_', ' ')
            if slot_val.startswith('#%s' % slot):
                tag_id = int(slot_val.replace('#%s' % slot, ''))
                try:
                    tag_list = tags.get(slot, [])
                    tag_val = tag_list[tag_id]
                    tag_val = self.tagger.denormalize_slot_value(tag_val)
                    new_res[tag_val] = p
                except IndexError:
                    # This happens when the we predict a tag that
                    # does not exist.
                    #new_res['_null_'] = p
                    if last_turn:
                        try:
                            last_v, last_p = last_turn['goal-labels'][slot].items()[0]
                            new_res[last_v] = last_p
                        except Exception, e:
                            print e
                            print slot, last_turn['goal-labels']

            else:
                new_res[slot_val] = p
        values.clear()
        values.update(new_res)


def main(dataset_name, data_file, output_file, track_log, params_file, model_type):
    models = []
    for pf in params_file:
        logging.info('Loading model from: %s' % pf)
        if model_type == 'lstm':
            model_cls = Model
        elif model_type == 'baseline':
            model_cls = BaselineModel
        else:
            raise Exception('Unknown model type.')
        models.append(model_cls.load(pf, build_train=False))

    logging.info('Loading data: %s' % data_file)
    data = Data.load(data_file)

    logging.info('Starting tracking.')
    tracker = XTrack2DSTCTracker(data, models, override_slots=None)

    t = time.time()
    result, tracking_accuracy, len_accuracy, len_accuracy_n = tracker.track(output_len_accuracy=True, tracking_log_file_name=track_log)
    t = time.time() - t
    logging.info('Tracking took: %.1fs' % t)
    for group, accuracy in tracking_accuracy.iteritems():
        logging.info('Accuracy %s: %.2f %%' % (group, accuracy * 100))
        #for t in len_accuracy:
        #    print '%d %.2f %d' % (t, len_accuracy[t][group], len_accuracy_n[t][group])


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
    parser.add_argument('--track_log', default=None)
    parser.add_argument('--params_file', action='append', required=True)
    parser.add_argument('--model_type', default='lstm')

    pdb_on_error()
    init_logging()
    main(**vars(parser.parse_args()))
