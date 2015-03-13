import collections
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


class XTrackCCA(object):
    def build(self, dialogs, out_file):
        for dialog_ndx, dialog in enumerate(dialogs):
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
                                          f_dump_text,
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
    init_logging('XTrack CCA')

    random.seed(0)
    from utils import pdb_on_error
    pdb_on_error()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out_file', required=True)

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


    xcca = XTrackCCA()
    xcca.build(dialogs=dialogs, out_file=args.out_file)
