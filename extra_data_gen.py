import argparse
import numpy as np
import os
import random

from data_model import Dialog
from xtrack_data2 import XTrackData2


def build(out_file, based_on, input_len, dialog_len, dialog_cnt):
    data = XTrackData2.load(based_on)
    vocab = [word for word in data.vocab.keys() if not word[0] in ['@', '#']]
    classes = data.classes

    dialogs = []
    for i in range(1):
        for slot, slot_clss in classes.iteritems():
            for cls_name in slot_clss:
                if cls_name == XTrackData2.null_class:
                    continue
                d_id = 'dummy_%s_%s' % (slot, cls_name)
                d = Dialog(d_id, d_id)
                msg = '%s %s' % (cls_name, slot)
                d.add_message([(msg, 0.0)], {slot: cls_name},
                              Dialog.ACTOR_USER)
                dialogs.append(d)

                d_id += '@2'
                d = Dialog(d_id, d_id)
                msg = 'conf_%s %s' % (slot, cls_name.replace(' ', '_'), )
                d.add_message([(msg, 0.0)], {},
                              Dialog.ACTOR_SYSTEM)
                d.add_message([('yes', 0.0)], {slot: cls_name}, Dialog.ACTOR_USER)
                dialogs.append(d)

    # Fast switches.
    for i in range(1000):
        slot = random.choice(data.slots)
        d_id = 'dummy2_%d_%s' % (i, slot, )
        d = Dialog(d_id, d_id)
        cls = None
        for y in range(random.randint(1, 5)):
            new_cls = random.choice(classes[slot].keys())
            if new_cls == XTrackData2.null_class or new_cls == 'dontcare':
                continue


            random_words = []
            for rw in range(random.randint(0, 10)):
                random_words.append(random.choice(vocab))

            msg = '%s %s %s' % (" ".join(random_words), new_cls, slot, )
            score = np.log((random.random() * 0.8) + 0.1)
            if score > np.log(0.5):
                cls = new_cls

            d.add_message([(msg, score)], {slot: cls}, Dialog.ACTOR_USER)
        dialogs.append(d)


    print '> Data built.'
    xt = XTrackData2()
    xt.build(dialogs, slots=['food'], slot_groups={0: ['food']},
             based_on=based_on,
             oov_ins_p=0.0,
             include_system_utterances=True, n_nbest_samples=1,
             n_best_order=[0],
             score_mean=0.0, dump_text='/dev/null', replace_entities=False,
             split_dialogs=True, include_base_seqs=False)
    print '> Saving.'
    xt.save(out_file)


if __name__ == '__main__':
    from utils import init_logging
    init_logging('ExtraDataGen')

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_file',
                        required=True,
                        help="Output file.")
    parser.add_argument('--based_on', required=True)
    parser.add_argument('--input_len', default=15, type=int)
    parser.add_argument('--dialog_len', default=5, type=int)
    parser.add_argument('--dialog_cnt', default=10000, type=int)
    args = parser.parse_args()

    build(**vars(args))



