from collections import defaultdict
import json
import logging
import os
import random
import re
import h5py
import numpy as np
import math

import data_model



word_re = re.compile(r'([A-Za-z0-9_]+)')


def tokenize(text):
    for match in word_re.finditer(text):
        yield match.group(1)


def tokenize_letter(text):
    for letter in text:
        yield letter



class XTrackData2(object):
    attrs_to_save = ['sequences', 'vocab', 'classes', 'slots', 'slot_groups',
                     'stats', 'token_features']

    null_class = '_null_'

    def _init(self, slots, slot_groups, based_on, include_base_seqs):
        self.slots = slots
        self.slot_groups = slot_groups
        self.based_on = based_on
        if based_on:
            data = XTrackData2.load(based_on)
            self.vocab = data.vocab
            self.classes = data.classes
            self.vocab_fixed = True
            self.stats = data.stats
            self.token_features = data.token_features
            if include_base_seqs:
                self.sequences = data.sequences
            else:
                self.sequences = []
        else:
            self.vocab = {
                "#NOTHING": 0,
                "#EOS": 1,
                "#OOV": 2,
            }

            self.classes = {}
            for slot in slots:
                self.classes[slot] = {self.null_class: 0}

            self.vocab_fixed = False
            self.stats = None
            self.sequences = []

        self._init_after_load()

    def _init_after_load(self):
        self.vocab_rev = {val: key for key, val in self.vocab.iteritems()}

    def _process_msg(self, msg, msg_score, state, actor, seq, oov_ins_p,
                     word_drop_p,
                     n_best_order, f_dump_text, replace_entities, true_msg):
        msg = msg.lower()
        #if replace_entities:
        #    for slot in self.slots:
        #        for cls_name in self.classes[slot]:
        #            msg = msg.replace(cls_name, cls_name.replace(' ', '_'))


        token_seq = list(tokenize(msg))
        if actor == data_model.Dialog.ACTOR_SYSTEM:
            token_seq = ["@%s" % token for token in token_seq]

        if not token_seq:
            token_seq = ['#NOTHING']

        #msg_score = np.power(msg_score, 1.0 / len(token_seq))

        if token_seq:
            f_dump_text.write(("%2.2f  " % msg_score) + " ".join(token_seq) +
                              '\n')
            f_dump_text.write(("TRUE  " + true_msg + '\n'))
        #token_seq = list(reversed(token_seq))

        #if actor == data_model.Dialog.ACTOR_SYSTEM:
        #    token_seq.insert(0, '#SYS')
        #else:
        #    token_seq.insert(0, '#USR')
        if actor == data_model.Dialog.ACTOR_USER:
            token_seq.append('#GEN')

        for i, token in enumerate(token_seq):
            if word_drop_p > random.random():
                continue
            token_ndx = self.get_token_ndx(token)
            seq['data'].append(token_ndx)
            seq['data_score'].append(msg_score)
            seq['data_actor'].append(actor)
            if actor == data_model.Dialog.ACTOR_SYSTEM:
                seq['data_switch'].append(1)
            else:
                seq['data_switch'].append(0)

            if token != '#SWITCH' and random.random() < oov_ins_p:
                seq['data'].append(self.get_token_ndx('#OOV'))
                seq['data_score'].append(msg_score)
                seq['data_actor'].append(actor)
                seq['data_switch'].append(0)
        #seq['data'].append(self.get_token_ndx('#EOS'))




        if actor == data_model.Dialog.ACTOR_USER:
            label = {
                'time': len(seq['data']) - 1,
                'slots': {}
            }
            for slot, val in zip(self.slots, self.state_to_label(state,
                                                                 self.slots)):
                label['slots'][slot] = val
            seq['labels'].append(label)


    def _sample_paths(self, n, dialog, allowed_ndxs):
        res = []
        for i in range(n):
            path = []
            for msgs in dialog:
                path.append(random.choice([i for i in allowed_ndxs
                                           if i < len(msgs)]))
            res.append(path)

        return res

    def _split_dialog(self, seq):
        seqs = []
        for i, label in enumerate(seq['labels']):
            t = label['time']
            new_seq = {}
            new_seq['id'] = seq['id'] + "@%d" % i
            new_seq['source_dir'] = seq['source_dir']
            new_seq['labels'] = [label]
            for key in ['data', 'data_score', 'data_actor', 'data_switch']:
                new_seq[key] = seq[key][:t + 1]

            seqs.append(new_seq)

        return seqs

    def build(self, dialogs, slots, slot_groups, based_on, oov_ins_p,
              word_drop_p,
              include_system_utterances, n_nbest_samples, n_best_order,
              score_mean, split_dialogs, dump_text, replace_entities,
              include_base_seqs=False):
        self._init(slots, slot_groups, based_on, include_base_seqs)

        f_dump_text = open(dump_text, 'w')

        for dialog_ndx, dialog in enumerate(dialogs):
            f_dump_text.write('> %s\n' % dialog.session_id)
            for path_id in range(n_nbest_samples):
                f_dump_text.write('>> path %d\n' % path_id)
                seq = {
                    'id': dialog.session_id,
                    'source_dir': dialog.object_id,
                    'data': [],
                    'data_score': [],
                    'data_actor': [],
                    'data_switch': [],
                    'labels': []
                }
                seq_data_keys = [key for key in seq if key.startswith('data')]
                for msgs, state, actor in zip(dialog.messages,
                                              dialog.states,
                                              dialog.actors):
                    actor_is_system = actor == data_model.Dialog.ACTOR_SYSTEM

                    if actor_is_system:
                        msg_id = 0
                    else:
                        msg_id = random.choice(n_best_order)
                    msg, msg_score = msgs[msg_id]
                    true_msg, _ = msgs[0]

                    if not include_system_utterances and actor_is_system:
                        continue
                    else:
                        #msg_score = max(msg_score, -100)
                        #msg_score = np.exp(msg_score)
                        self._process_msg(msg, msg_score, state, actor, seq,
                                          oov_ins_p, word_drop_p, n_best_order,
                                          f_dump_text, replace_entities, true_msg)


                # Sanity check that all data elements are equal size.
                data_lens = [len(seq[key]) for key in seq_data_keys]
                assert data_lens[1:] == data_lens[:-1]
                if len(seq['data']) > 0:

                    if not split_dialogs:
                        self.sequences.append(seq)
                    else:
                        self.sequences.extend(self._split_dialog(seq))
            f_dump_text.write('\n')

        if not self.stats:
            logging.info('Computing stats.')
            self._compute_stats('data_score', 'data_switch')

        logging.info('Normalizing.')
        self._normalize('data_score', 'data_switch')

        if not self.based_on:
            logging.info('Building token features.')
            self._build_token_features()

    def _build_token_features(self):
        self.token_features = {}
        for word, word_id in self.vocab.iteritems():
            features = []
            for slot in self.slots:
                features.append(int(word in slot.split()))
                for cls in self.classes[slot]:
                    ftr_val = 0
                    for cls_part in cls.split():
                        if cls_part[0] == '@':
                            cls_part = cls_part[1:]
                        if cls_part.startswith(word):
                            ftr_val = 1
                            break

                    features.append(ftr_val)
            self.token_features[word_id] = features


    def _compute_stats(self, *vars):
        score = {var: [] for var in vars}
        for seq in self.sequences:
            for var in vars:
                score[var].extend(seq[var])
        #import ipdb; ipdb.set_trace()
        self.stats = {}
        for var in vars:
            mean = np.mean(score[var])
            stddev = np.std(score[var])
            self.stats[var] = {
                'mean': mean,
                'stddev': stddev
            }

    def _normalize(self, *vars):
        for seq in self.sequences:
            for var in vars:
                res = seq[var]
                for i in xrange(len(res)):
                    res[i] -= self.stats[var]['mean']
                    res[i] /= self.stats[var]['stddev'] + 1e-7

    def get_token_ndx(self, token):
        if token in self.vocab:
            return self.vocab[token]
        else:
            if not self.vocab_fixed:
                self.vocab[token] = res = len(self.vocab)
                self.vocab_rev[self.vocab[token]] = token
                return res
            else:
                logging.warning('Mapping to OOV: %s' % token)
                return self.vocab['#OOV']

    def state_to_label(self, state, slots):
        res = []
        for slot in slots:
            res.append(self.state_to_label_for(state, slot))

        return res

    def state_to_label_for(self, state, slot):
        if not state:
            return self.classes[slot][self.null_class]
        else:
            value = state.get(slot)

            if value:
                food = value  #next(tokenize(value))

                if self.vocab_fixed:
                    if not food in self.classes[slot]:
                        res = self.classes[slot][self.null_class]
                    else:
                        res = self.classes[slot][food]
                else:
                    if not food in self.classes[slot]:
                        self.classes[slot][food] = len(self.classes[slot])
                    res = self.classes[slot][food]

            else:
                res = self.classes[slot][self.null_class]

            return res

    def save(self, out_file):
        with open(out_file, 'w') as f_out:
            obj = {}
            for attr in self.attrs_to_save:
                obj[attr] = getattr(self, attr)

            json.dump(obj, f_out, indent=4)

    @classmethod
    def load(cls, in_file):
        with open(in_file, 'r') as f_in:
            data = json.load(f_in)

        xtd = XTrackData2()
        for attr in cls.attrs_to_save:
            val = data[attr]
            setattr(xtd, attr, val)

        xtd._init_after_load()

        return xtd



if __name__ == '__main__':
    from utils import init_logging
    init_logging('XTrackData')

    random.seed(0)
    from utils import pdb_on_error
    pdb_on_error()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out_file', required=True)
    parser.add_argument('--out_flist_file', required=False)
    parser.add_argument('--based_on', type=str, required=False, default=None)
    parser.add_argument('--include_base_seqs', action='store_true',
                        default=False)
    parser.add_argument('--slots', default='food')
    parser.add_argument('--oov_ins_p', type=float, required=False, default=0.0)
    parser.add_argument('--word_drop_p', type=float, required=False,
                        default=0.0)
    parser.add_argument('--include_system_utterances', action='store_true',
                        default=False)
    parser.add_argument('--n_best_order', default="0")
    parser.add_argument('--n_nbest_samples', default=10, type=int)
    parser.add_argument('--score_mean', default=0.0, type=float)
    parser.add_argument('--dump_text', default='/dev/null')
    parser.add_argument('--split_dialogs', action='store_true', default=False)
    parser.add_argument('--replace_entities', action='store_true',
                        default=False)

    args = parser.parse_args()

    dialogs = []
    for f_name in sorted(os.listdir(args.data_dir), key=lambda x: int(x.split(
            '.')[0])):
        if f_name.endswith('.json'):
            dialogs.append(
                data_model.Dialog.deserialize(
                    open(os.path.join(args.data_dir, f_name)).read()
                )
            )

    slot_groups = {}
    slots = []
    for i, slot_group in enumerate(args.slots.split(':')):
        slot_group = slot_group.split(',')
        slot_groups[i] = slot_group
        slots.extend(slot_group)

    n_best_order = map(int, args.n_best_order.split(','))

    xtd = XTrackData2()
    xtd.build(dialogs=dialogs, based_on=args.based_on, slots=slots,
              slot_groups=slot_groups, oov_ins_p=args.oov_ins_p,
              word_drop_p=args.word_drop_p,
              include_system_utterances=args.include_system_utterances,
              n_best_order=n_best_order, score_mean=args.score_mean,
              dump_text=args.dump_text, n_nbest_samples=args.n_nbest_samples,
              split_dialogs=args.split_dialogs,
              replace_entities=args.replace_entities,
              include_base_seqs=args.include_base_seqs)

    logging.info('Saving.')
    xtd.save(args.out_file)

    logging.info('Saving file list.')
    if args.out_flist_file:
        flist = []
        for dialog in xtd.sequences:
            flist.append(dialog['source_dir'])
        with open(args.out_flist_file, "w") as f_out:
            f_out.write("\n".join(flist))

