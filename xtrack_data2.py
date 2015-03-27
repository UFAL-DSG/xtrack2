import collections
import json
import logging
import os
import random
import re
import numpy as np
import math

import data_model



word_re = re.compile(r'([A-Za-z0-9_]+)')


def tokenize(text):
    for match in word_re.finditer(text):
        yield match.group(1)


def get_cca_y(tokens, state, last_state):
    res = []
    if state is None:
        state = {}
    if last_state is None:
        last_state = {}

    key_diff = set(state.keys()).difference(last_state.keys())
    res += list(key_diff)
    for key in state:
        if state[key] != last_state.get(key):
            res.append("%s_%s" % (key, state[key]))

    #print state, last_state, res
    return " ".join(res)


class XTrackData2(object):
    attrs_to_save = ['sequences', 'vocab', 'classes', 'slots', 'slot_groups',
                     'stats', 'token_features', 'score_bins']

    null_class = '_null_'

    def _init(self, slots, slot_groups, based_on, include_base_seqs):
        self.slots = slots
        self.slot_groups = slot_groups
        self.based_on = based_on
        self.word_freq = collections.Counter()

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


    def _get_score_bin(self, msg_score, score_bins):
        msg_score_bin = 0
        if score_bins:
            for i, x in enumerate(score_bins):
                if np.exp(msg_score) < x:
                    # curr_score_bin = "__%d" % i
                    msg_score_bin = i
                    break
            else:
                msg_score_bin = len(score_bins) - 1
        return msg_score_bin

    def _tokenize_msg(self, actor, msg):
        msg = msg.lower()
        token_seq = list(tokenize(msg))
        if actor == data_model.Dialog.ACTOR_SYSTEM:
            token_seq = ["@%s" % token for token in token_seq]
        if not token_seq:
            token_seq = ['#NOTHING']
        return token_seq

    def _dump_msg_info(self, f_dump_cca, f_dump_text, last_state, msg_score,
                       msg_score_bin, state, token_seq, true_msg):
        f_dump_text.write(("%2.2f %d  " % (msg_score, msg_score_bin)) + " "
                                                                        "".join(
            token_seq) + '\n')
        f_dump_text.write(("TRUE  " + true_msg + '\n'))
        f_dump_cca.write(" ".join(token_seq))
        f_dump_cca.write("\t")
        f_dump_cca.write(get_cca_y(token_seq, state, last_state))
        f_dump_cca.write('\n')

    def _get_token_label(self, token, state):
        res = []
        for cls in self.classes:
            if state:
                state_val = state.get(cls)
            else:
                state_val = None

            token_is_same = False
            if state_val == token:
                res.append(1)
                token_is_same = True
            else:
                res.append(0)

            if cls == token or token_is_same:
                res.append(1)
            else:
                res.append(0)

        return res

    def _append_token_to_seq(self, actor, msg_score_bin, seq, token, state):
        token_ndx = self.get_token_ndx(token)
        seq['data'].append(token_ndx)
        seq['data_score'].append(msg_score_bin)
        seq['data_actor'].append(actor)
        seq['data_switch'].append(0)
        seq['data_debug'].append(token)
        seq['token_labels'].append(self._get_token_label(token, state))

    def _append_label_to_seq(self, msg_score, seq, state):
        label = {
            'time': len(seq['data']) - 1,
            'score': np.sqrt(np.exp(msg_score)),
            'slots': {}
        }
        for slot, val in zip(self.slots, self.state_to_label(state,
                                                             self.slots)):
            label['slots'][slot] = val
        seq['labels'].append(label)

    def _process_msg(self, msg, msg_score, state, last_state, actor, seq,
                     oov_ins_p, word_drop_p, n_best_order, f_dump_text,
                     f_dump_cca, true_msg, score_bins):

        msg_score_bin = self._get_score_bin(msg_score, score_bins)
        token_seq = self._tokenize_msg(actor, msg)
        self._dump_msg_info(f_dump_cca, f_dump_text, last_state, msg_score,
                            msg_score_bin, state, token_seq, true_msg)

        for i, token in enumerate(token_seq):
            if word_drop_p > random.random():
                continue

            self.word_freq[token] += 1

            if random.random() < oov_ins_p:
                token = '#OOV'

            self._append_token_to_seq(actor, msg_score_bin, seq, token, state)

        seq['true_input'].append(true_msg)
        if actor == data_model.Dialog.ACTOR_USER:
            self._append_token_to_seq(actor, 0, seq, '@over', state)
            self._append_label_to_seq(msg_score, seq, state)


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
              include_system_utterances, n_nbest_samples,
              n_best_order,
              score_mean, dump_text, dump_cca, score_bins,
              split_dialogs=False,
              include_base_seqs=False):
        self._init(slots, slot_groups, based_on, include_base_seqs)
        self.score_bins = score_bins
        n_labels = 0

        f_dump_text = open(dump_text, 'w')
        f_dump_cca = open(dump_cca, 'w')
        self.msg_scores = []

        for dialog_ndx, dialog in enumerate(dialogs):
            f_dump_text.write('> %s\n' % dialog.session_id)
            last_state = None
            for path_id in range(n_nbest_samples):
                f_dump_text.write('>> path %d\n' % path_id)
                seq = {
                    'id': dialog.session_id,
                    'source_dir': dialog.object_id,
                    'data': [],
                    'data_debug': [],
                    'data_score': [],
                    'data_actor': [],
                    'data_switch': [],
                    'labels': [],
                    'token_labels': [],
                    'tags': {},
                    'true_input': [],
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

                    if actor == data_model.Dialog.ACTOR_USER:
                        self.msg_scores.append(np.exp(msg_score))

                    if not include_system_utterances and actor_is_system:
                        continue
                    else:
                        #msg_score = max(msg_score, -100)
                        #msg_score = np.exp(msg_score)
                        self._process_msg(msg, msg_score, state, last_state,
                                          actor, seq,
                                          oov_ins_p, word_drop_p, n_best_order,
                                          f_dump_text, f_dump_cca,
                                          true_msg, score_bins)
                    last_state = state


                # Sanity check that all data elements are equal size.
                data_lens = [len(seq[key]) for key in seq_data_keys]
                assert data_lens[1:] == data_lens[:-1]
                if len(seq['data']) > 0:
                    n_labels += len(seq['labels'])

                    if not split_dialogs:
                        self.sequences.append(seq)
                    else:
                        self.sequences.extend(self._split_dialog(seq))
            f_dump_text.write('\n')

        logging.info('There are in total %d labels in %d sequences.'
                     % (n_labels, len(self.sequences, )))

        #if not self.stats:
        #    logging.info('Computing stats.')
        #    self._compute_stats('data_score', 'data_switch')

        #logging.info('Normalizing.')
        #self._normalize('data_score', 'data_switch')

        if not self.based_on:
            logging.info('Building token features.')
            self._build_token_features()


    def _build_token_features(self):
        self.token_features = {}
        for word, word_id in self.vocab.iteritems():
            features = []
            for slot in self.slots:
                features.append(int(word in slot))
                for cls in self.classes[slot]:
                    ftr_val = 0
                    for cls_part in cls.split():
                        if cls_part[0] == '@':
                            cls_part = cls_part[1:]
                        if word in cls_part:
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


        import matplotlib
        matplotlib.use('Agg')
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set_palette("deep", desat=.6)
        plt.figure()
        plt.hist(self.msg_scores, [0.0, 0.3, 0.6, 0.95, 1.0])
        plt.savefig(out_file + '.score.png')

        plt.figure()
        plt.hist(np.log(np.array(self.word_freq.values())))
        plt.savefig(out_file + '.word_freqs.png')

        with open(out_file + '.oov.txt', 'w') as f_out:
            for word, freq in self.word_freq.most_common():
                if freq < 5:
                    f_out.write(word + '\n')

        #import ipdb; ipdb.set_trace()


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
    parser.add_argument('--n_nbest_samples', default=1, type=int)
    parser.add_argument('--score_mean', default=0.0, type=float)
    parser.add_argument('--dump_text', default='/dev/null')
    parser.add_argument('--dump_cca', default='/dev/null')
    parser.add_argument('--split_dialogs', action='store_true', default=False)
    parser.add_argument('--tagged', action='store_true', default=False)

    args = parser.parse_args()

    tagged = args.tagged

    dialogs = []
    for f_name in sorted(os.listdir(args.data_dir), key=lambda x: int(x.split(
            '.')[0])):
        if f_name.endswith('.json'):
            dialog = data_model.Dialog.deserialize(
                open(os.path.join(args.data_dir, f_name)).read()
            )
            dialogs.append(dialog)

    slot_groups = {}
    slots = []
    for i, slot_group in enumerate(args.slots.split(':')):
        if '=' in slot_group:
            name, vals = slot_group.split('=', 1)
        else:
            name = 'grp%d' % i
            vals = slot_group
        slot_group = vals.split(',')
        slot_groups[name] = slot_group
        for slot in slot_group:
            if not slot in slots:
                slots.append(slot)


    n_best_order = map(int, args.n_best_order.split(','))

    xtd = XTrackData2()
    xtd.build(dialogs=dialogs, based_on=args.based_on, slots=slots,
              slot_groups=slot_groups, oov_ins_p=args.oov_ins_p,
              word_drop_p=args.word_drop_p,
              include_system_utterances=args.include_system_utterances,
              n_best_order=n_best_order, score_mean=args.score_mean,
              dump_text=args.dump_text, dump_cca=args.dump_cca,
              n_nbest_samples=args.n_nbest_samples,
              split_dialogs=args.split_dialogs,
              include_base_seqs=args.include_base_seqs,
              #score_bins=[0.0, 0.3, 0.6, 0.95, 1.0]
              score_bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                          0.95, 1.0]
    )

    logging.info('Saving.')
    xtd.save(args.out_file)