from collections import defaultdict
import json
import os
import random
import re
import h5py
import numpy as np
import math

import data_model



word_re = re.compile(r'([A-Za-z0-9]+)')


def tokenize(text):
    for match in word_re.finditer(text):
        yield match.group(1)


class XTrackData2(object):
    attrs_to_save = ['sequences', 'vocab', 'classes', 'slots', 'slot_groups',
                     'stats']

    null_class = '_null_'

    def _init(self, slots, slot_groups, vocab_from):
        self.slots = slots
        self.slot_groups = slot_groups
        if vocab_from:
            data = XTrackData2.load(vocab_from)
            self.vocab = data.vocab
            self.classes = data.classes
            self.vocab_fixed = True
            self.stats = data.stats
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

        self._init_after_load()

    def _init_after_load(self):
        self.vocab_rev = {val: key for key, val in self.vocab.iteritems()}

    def _process_msg(self, msg, msg_score, state, actor, seq, oov_ins_p,
                     n_best_order, f_dump_text):
        token_seq = list(tokenize(msg.lower()))
        if token_seq:
            f_dump_text.write(" ".join(token_seq) + '\n')
        #token_seq = list(reversed(token_seq))

        #if actor == data_model.Dialog.ACTOR_SYSTEM:
        #    token_seq.insert(0, '#SYS')
        #else:
        #    token_seq.insert(0, '#USR')

        for i, token in enumerate(token_seq):
            token_ndx = self.get_token_ndx(token)
            seq['data'].append(token_ndx)
            seq['data_score'].append(msg_score)
            seq['data_actor'].append(actor)

            if random.random() < oov_ins_p:
                seq['data'].append(self.get_token_ndx('#OOV'))
                seq['data_score'].append(msg_score)
                seq['data_actor'].append(actor)
        #seq['data'].append(self.get_token_ndx('#EOS'))


        label = {
            'time': len(seq['data']) - 1,
            'slots': {}
        }
        for slot, val in zip(self.slots, self.state_to_label(state,
                                                             self.slots)):
            label['slots'][slot] = val
        seq['labels'].append(label)



    def _process_msgs(self, msgs, state, actor, seq, oov_ins_p, n_best_order,
                      f_dump_text):
        for msg_id in n_best_order:
            if msg_id < len(msgs):
                msg, msg_score = msgs[msg_id]

                msg_score = np.exp(msg_score)
                self._process_msg(msg, msg_score, state, actor, seq,
                                  oov_ins_p, n_best_order, f_dump_text)

    def _sample_paths(self, n, dialog, allowed_ndxs):
        res = []
        for i in range(n):
            path = []
            for msgs in dialog:
                path.append(random.choice([i for i in allowed_ndxs
                                           if i < len(msgs)]))
            res.append(path)

        return res

    def build(self, dialogs, slots, slot_groups, vocab_from, oov_ins_p,
              include_system_utterances, n_nbest_samples, n_best_order,
              score_mean, dump_text):
        self._init(slots, slot_groups, vocab_from)

        self.sequences = []

        f_dump_text = open(dump_text, 'w')

        for dialog_ndx, dialog in enumerate(dialogs):
            for path_id in range(n_nbest_samples):
                seq = {
                    'id': dialog.session_id,
                    'source_dir': dialog.object_id,
                    'data': [],
                    'data_score': [],
                    'data_actor': [],
                    'labels': []
                }
                for msgs, state, actor in zip(dialog.messages,
                                              dialog.states,
                                              dialog.actors):
                    actor_is_system = actor == data_model.Dialog.ACTOR_SYSTEM

                    if actor_is_system:
                        msg_id = 0
                    else:
                        msg_id = random.choice(n_best_order)
                    msg, msg_score = msgs[msg_id]

                    if not include_system_utterances and actor_is_system:
                        continue
                    else:
                        self._process_msg(msg, msg_score, state, actor, seq,
                                          oov_ins_p, n_best_order, f_dump_text)

                if len(seq['data']) > 0:
                    self.sequences.append(seq)

        if not self.stats:
            print '>> Computing stats.'
            self._compute_stats('data_score')

        print '>> Normalizing.'
        self._normalize('data_score')

    def _compute_stats(self, *vars):
        score = {var: [] for var in vars}
        for seq in self.sequences:
            for var in vars:
                score[var].extend(seq[var])

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
                    res[i] /= self.stats[var]['stddev'] + 0.001

    def get_token_ndx(self, token):
        if token in self.vocab:
            return self.vocab[token]
        else:
            if not self.vocab_fixed:
                self.vocab[token] = res = len(self.vocab)
                self.vocab_rev[self.vocab[token]] = token
                return res
            else:
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
    random.seed(0)
    from utils import pdb_on_error
    pdb_on_error()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out_file', required=True)
    parser.add_argument('--out_flist_file', required=False)
    parser.add_argument('--vocab_from', type=str, required=False, default=None)
    parser.add_argument('--slots', default='food')
    parser.add_argument('--oov_ins_p', type=float, required=False, default=0.0)
    parser.add_argument('--include_system_utterances', action='store_true',
                        default=False)
    parser.add_argument('--n_best_order', default="0")
    parser.add_argument('--n_nbest_samples', default=10, type=int)
    parser.add_argument('--score_mean', default=0.0, type=float)
    parser.add_argument('--dump_text', default='/dev/null')

    args = parser.parse_args()

    dialogs = []
    for f_name in os.listdir(args.data_dir):
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
    xtd.build(dialogs=dialogs, vocab_from=args.vocab_from, slots=slots,
              slot_groups=slot_groups, oov_ins_p=args.oov_ins_p,
              include_system_utterances=args.include_system_utterances,
              n_best_order=n_best_order, score_mean=args.score_mean,
              dump_text=args.dump_text, n_nbest_samples=args.n_nbest_samples)

    print '> Saving.'
    xtd.save(args.out_file)

    print '> Saving flist.'
    if args.out_flist_file:
        flist = []
        for dialog in xtd.sequences:
            flist.append(dialog['source_dir'])
        with open(args.out_flist_file, "w") as f_out:
            f_out.write("\n".join(flist))

