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


class Tagger(object):
    def normalize_slot_value(self, val):
        return val.replace(' ', '_')

    def denormalize_slot_value(self, val):
        return val.replace('_', ' ')


class Sequence(dict):
    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        return self[key]

    def __init__(self, seq_id, source_dir):
        self.id = seq_id
        self.source_dir = source_dir
        self.data = []
        self.data_debug = []
        self.data_score = []
        self.data_actor = []
        self.labels = []
        self.token_labels = []
        self.tags = collections.defaultdict(list)
        self.true_input = []

    def __repr__(self):
        return json.dumps(self.__dict__)


class DataBuilder(object):
    def _open_dump_files(self, debug_dir):
        if debug_dir:
            if not os.path.exists(debug_dir):
                os.mkdir(debug_dir)
            fname_dump_text = os.path.join(debug_dir, 'dump.text')
            fname_dump_cca = os.path.join(debug_dir, 'dump.cca')
        else:
            fname_dump_text = '/dev/null'
            fname_dump_cca = '/dev/null'

        self.f_dump_text = open(fname_dump_text, 'w')
        self.f_dump_cca = open(fname_dump_cca, 'w')

    def __init__(self, slots, slot_groups, based_on, include_base_seqs,
              oov_ins_p, word_drop_p, include_system_utterances, nth_best,
              score_bins, debug_dir, tagged, ontology, no_label_weight):
        self.slots = slots
        self.slot_groups = slot_groups
        self.score_bins = score_bins
        self.ontology = ontology
        self.based_on = based_on
        self.include_base_seqs = include_base_seqs
        self.oov_ins_p = oov_ins_p
        self.word_drop_p = word_drop_p
        self.include_system_utterances = include_system_utterances
        self.nth_best = nth_best
        self.debug_dir = debug_dir
        self.tagged = tagged
        if tagged:
            self.tagger = Tagger()
        else:
            self.tagger = None
        self.no_label_weight = no_label_weight

        self.xd = None
        self.word_freq = collections.Counter()

        self._open_dump_files(debug_dir)

    def build(self, dialogs):
        self._create_new_data_instance()

        n_labels = 0

        self.msg_scores = []

        for dialog_ndx, dialog in enumerate(dialogs):
            self.f_dump_text.write('> %s\n' % dialog.session_id)

            seq = self._create_seq(dialog)

            self._process_dialog(dialog, seq)
            self._perform_sanity_checks(seq)
            self._append_seq_if_nonempty(seq)
            n_labels += len(seq.labels)

            self._dump_seq_info(seq)
            self.f_dump_text.write('\n')

        logging.info('There are in total %d labels in %d sequences.'
                     % (n_labels, len(self.xd.sequences, )))

        return self.xd

    def _create_new_data_instance(self):
        self.xd = Data()
        self.xd.initialize(self.slots, self.slot_groups, self.based_on,
                           self.include_base_seqs, self.score_bins,
                           self.tagged, self.ontology, self.tagger)

    def _create_seq(self, dialog):
        seq = Sequence(dialog.session_id, dialog.object_id)
        return seq

    def _process_dialog(self, dialog, seq):
        last_state = None

        for msgs, state, actor in zip(dialog.messages,
                                      dialog.states,
                                      dialog.actors):
            actor_is_system = actor == data_model.Dialog.ACTOR_SYSTEM

            if actor_is_system:
                msg_id = 0
            else:
                msg_id = self.nth_best

            msg, msg_score = msgs[msg_id]
            true_msg, _ = msgs[0]

            if not self.include_system_utterances and actor_is_system:
                continue
            else:
                self._process_msg(msg, msg_score, state, last_state, actor, seq,
                                  true_msg)
            last_state = state

    def _dump_seq_info(self, seq):
        self.f_dump_text.write('\nSEQ:')
        for token in seq.data:
            token_str = self.xd.vocab_rev[token]
            self.f_dump_text.write('%s ' % token_str)
        self.f_dump_text.write('\n')

    def _process_msg(self, msg, msg_score, state, last_state, actor, seq,
                     true_msg):

        msg_score_bin = self.xd.get_score_bin(msg_score)
        token_seq = self._tokenize_msg(actor, msg)
        self._dump_msg_info(last_state, msg_score, msg_score_bin, state,
                            token_seq, true_msg)

        for i, token in enumerate(token_seq):
            if self.word_drop_p > random.random():
                continue

            self.word_freq[token] += 1

            if random.random() < self.oov_ins_p:
                token = '#OOV'

            self._append_token_to_seq(actor, msg_score_bin, seq, token, state)

        seq.true_input.append(true_msg)
        if actor == data_model.Dialog.ACTOR_USER:
            self._append_label_to_seq(msg_score, seq, state)

    def _dump_msg_info(self, last_state, msg_score, msg_score_bin, state,
                       token_seq, true_msg):
        self.f_dump_text.write(("%2.2f %d  " % (msg_score, msg_score_bin)) + " "
                                                                        "".join(
            token_seq) + '\n')
        self.f_dump_text.write(("TRUE  " + true_msg + '\n'))
        self.f_dump_cca.write(" ".join(token_seq))
        self.f_dump_cca.write("\t")
        self.f_dump_cca.write(get_cca_y(token_seq, state, last_state))
        self.f_dump_cca.write('\n')

    def _tokenize_msg(self, actor, msg):
        msg = msg.lower()
        if self.tagged:
            for slot, slot_values in self.xd.classes.iteritems():
                for slot_value in slot_values:
                    msg = msg.replace(self.tagger.denormalize_slot_value(
                        slot_value),
                                      slot_value)

        token_seq = list(tokenize(msg))

        if actor == data_model.Dialog.ACTOR_SYSTEM:
            token_seq = ["@%s" % token for token in token_seq]

        if not token_seq:
            token_seq = ['#NOTHING']

        return token_seq

    def _append_token_to_seq(self, actor, msg_score_bin, seq, token, state):
        token_ndx = self.xd.get_token_ndx(token)
        if not self.tagged:
            seq.data.append(token_ndx)
        else:
            if actor == data_model.Dialog.ACTOR_SYSTEM:
                token = token[1:]
            tagged_token = self._tag_token(token, seq)
            if actor == data_model.Dialog.ACTOR_SYSTEM:
                tagged_token = '@' + tagged_token
            tagged_token_ndx = self.xd.get_token_ndx(tagged_token)
            seq.data.append(tagged_token_ndx)
        seq.data_score.append(msg_score_bin)
        seq.data_actor.append(actor)
        seq.data_debug.append(token)

    def _tag_token(self, token, seq):
        tag = self.xd.tag_token(token)
        if tag:
            if not token in seq.tags[tag]:
                seq.tags[tag].append(token)

            return '#%s%d#' % (tag, seq.tags[tag].index(token), )
        else:
            return token

    def _append_label_to_seq(self, msg_score, seq, state):
        label = {
            'time': len(seq.data) - 1,
            'score': np.exp(msg_score),
            'slots': {}
        }
        if self.no_label_weight:
            label['score'] = 1.0

        slot_labels = self.xd.state_to_label(state, self.slots)
        for slot, val in zip(self.slots, slot_labels):
            if not self.tagged:
                label['slots'][slot] = val
            else:
                try:
                    if not state:
                        raise ValueError()
                    state_val = state.get(slot, '')
                    if not state_val:
                        raise ValueError()

                    tag_ndx = seq.tags[slot].index(
                        self.tagger.normalize_slot_value(state_val))
                    tag_cls_str = "#%s%d" % (slot, tag_ndx)

                    try:
                        tagged_val = self.xd.get_value_index_for_slot(slot,
                                                                      tag_cls_str)
                    except UnknownClassException:
                        raise ValueError()
                except ValueError:
                    tagged_val = val

                label['slots'][slot] = tagged_val
        seq.labels.append(label)

    def _perform_sanity_checks(self, seq):
        # Sanity check that all data elements are equal size.
        seq_data_keys = [key for key in seq.__dict__ if key.startswith('data')]
        data_lens = [len(getattr(seq, key)) for key in seq_data_keys]
        assert data_lens[1:] == data_lens[:-1]

    def _append_seq_if_nonempty(self, seq):
        if len(seq.data) > 0:
            self.xd.add_sequence(seq)





class UnknownClassException(Exception):
    pass


class Data(object):
    attrs_to_save = ['sequences', 'vocab', 'vocab_rev', 'classes', 'slots',
                     'slot_groups', 'stats', 'score_bins', 'tagged']

    null_class = '_null_'
    slots = None
    vocab = None
    slot_groups = None

    def _build_initial_classes(self, ontology):
        classes = {}
        for slot in self.slots:
            self.get_token_ndx(slot)
            classes[slot] = {self.null_class: 0}
            for slot_val in ontology.get(slot, []):
                if self.tagged:
                    slot_val = self.tagger.normalize_slot_value(slot_val)
                classes[slot][slot_val] = len(classes[slot])
                self.get_token_ndx(slot_val)

        return classes

    def _finalize_initialization(self):
        self.vocab_rev = {val: key for key, val in self.vocab.iteritems()}

    def initialize(self, slots, slot_groups, based_on, include_base_seqs,
                   score_bins, tagged, ontology, tagger):
        self.slots = slots
        self.slot_groups = slot_groups
        self.tagged = tagged
        self.vocab_rev = {}
        self.tagger = tagger

        if based_on:
            data = Data.load(based_on)
            self.vocab = data.vocab
            self.classes = data.classes
            self.vocab_fixed = True
            self.stats = data.stats
            if include_base_seqs:
                self.sequences = data.sequences
            else:
                self.sequences = []
            self.score_bins = data.score_bins
        else:
            self.vocab = {
                "#NOTHING": 0,
                "#EOS": 1,
                "#OOV": 2,
            }

            self.vocab_fixed = False
            self.stats = None
            self.sequences = []
            self.score_bins = score_bins

            self.classes = self._build_initial_classes(ontology)

        self._finalize_initialization()


    def add_sequence(self, seq):
        self.sequences.append(seq)

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

    def tag_token(self, token):
        for cls, vals in self.classes.iteritems():
            if token in vals:
                return cls

    def state_to_label(self, state, slots):
        res = []
        for slot in slots:
            res.append(self.state_to_label_for(state, slot))

        return res

    def get_value_index_for_slot(self, slot, slot_value):
        if self.vocab_fixed:
            if not slot_value in self.classes[slot]:
                raise UnknownClassException()
            else:
                res = self.classes[slot][slot_value]
        else:
            if not slot_value in self.classes[slot]:
                self.classes[slot][slot_value] = len(self.classes[slot])
            res = self.classes[slot][slot_value]

        return res

    def state_to_label_for(self, state, slot):
        if not state:
            return self.classes[slot][self.null_class]
        else:
            slot_value = state.get(slot)

            if slot_value:
                if self.tagged:
                    slot_value = self.tagger.normalize_slot_value(slot_value)
                res = self.get_value_index_for_slot(slot, slot_value)
            else:
                res = self.classes[slot][self.null_class]

            return res

    def get_score_bin(self, msg_score):
        msg_score_bin = 0
        if self.score_bins:
            for i, x in enumerate(self.score_bins):
                if np.exp(msg_score) < x:
                    # curr_score_bin = "__%d" % i
                    msg_score_bin = i
                    break
            else:
                msg_score_bin = len(self.score_bins) - 1
        return msg_score_bin

    def save(self, out_file):
        with open(out_file, 'w') as f_out:
            obj = {}
            for attr in self.attrs_to_save:
                obj[attr] = getattr(self, attr)

            json.dump(obj, f_out, indent=4)

        """
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
        """


    @classmethod
    def load(cls, in_file):
        with open(in_file, 'r') as f_in:
            data = json.load(f_in)

        xtd = Data()
        for attr in cls.attrs_to_save:
            val = data[attr]
            setattr(xtd, attr, val)

        xtd._finalize_initialization()
        return xtd


def load_dialogs(data_dir):
    dialogs = []
    for f_name in sorted(os.listdir(data_dir), key=lambda x: int(x.split(
            '.')[0])):
        if f_name.endswith('.json'):
            dialog = data_model.Dialog.deserialize(
                open(os.path.join(data_dir, f_name)).read()
            )
            dialogs.append(dialog)
    return dialogs


def parse_slots_and_slot_groups(args):
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
    return slot_groups, slots


import import_dstc

def import_dstc_data(data_directory, out_dir, e_root, dataset, data_name):
    input_dir = os.path.join(data_directory, 'dstc2/data')
    flist = os.path.join(data_directory,
                         'dstc2/scripts/config/dstc2_%s.flist' % dataset)
    import_dstc.import_dstc(data_dir=input_dir, out_dir=out_dir, flist=flist,
                            constraint_slots='food,area,pricerange,name',
                            requestable_slots='food,area,pricerange,'
                                                       'name,addr,phone,'
                                                       'postcode,signature',
                            use_stringified_system_acts=True)

    return out_dir


def prepare_experiment(experiment_name, data_directory, slots, slot_groups,
                       ontology, skip_dstc_import_step, builder_opts):
    e_root = os.path.join(data_directory, 'xtrack/%s' % experiment_name)
    debug_dir = os.path.join(e_root, 'debug')

    based_on = None
    for dataset in ['train', 'dev', 'test']:
        out_dir = os.path.join(e_root, dataset)
        if not skip_dstc_import_step:
            import_dstc_data(data_directory=data_directory,
                             e_root=e_root,
                             dataset=dataset,
                             data_name=experiment_name,
                             out_dir=out_dir)

        dialogs = load_dialogs(out_dir)



        logging.info('Initializing.')
        xtd_builder = DataBuilder(
            based_on=based_on,
            include_base_seqs=False,
            slots=slots,
            slot_groups=slot_groups,
            oov_ins_p=0.1 if dataset == 'train' else 0.0,
            word_drop_p=0.0,
            include_system_utterances=True,
            nth_best=1,
            score_bins=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01],
            ontology=ontology,
            debug_dir=debug_dir,
            **builder_opts
        )
        logging.info('Building.')
        xtd = xtd_builder.build(dialogs)

        logging.info('Saving.')
        out_file = os.path.join(e_root, '%s.json' % dataset)
        xtd.save(out_file)

        if dataset == 'train':
            based_on = out_file
"""

def main():
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
    parser.add_argument('--nth_best', type=int, default=1)
    parser.add_argument('--debug_dir', default=None)

    args = parser.parse_args()


    slot_groups, slots = parse_slots_and_slot_groups(args)
    score_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

    dialogs = load_dialogs(args.data_dir)

    logging.info('Initializing.')
    xtd_builder = XTrackData2Builder(
        based_on=args.based_on,
        include_base_seqs=args.include_base_seqs,
        slots=slots,
        slot_groups=slot_groups,
        oov_ins_p=args.oov_ins_p,
        word_drop_p=args.word_drop_p,
        include_system_utterances=args.include_system_utterances,
        nth_best=args.nth_best,
        score_bins=score_bins,
        debug_dir=args.debug_dir
    )
    logging.info('Building.')
    xtd = xtd_builder.build(dialogs)

    logging.info('Saving.')
    xtd.save(args.out_file)


if __name__ == '__main__':
    main()

    """