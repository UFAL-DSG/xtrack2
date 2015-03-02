import argparse
import numpy as np
import os
import random

from data_model import Dialog
from xtrack_data2 import XTrackData2


class_surface_forms = {
    'dontcare': "don't care"
}


def build(out_file, based_on, input_len, dialog_len, easy_dialog_cnt,
          switched_dialog_cnt, include_base_seqs):
    data = XTrackData2.load(based_on)
    vocab = [word for word in data.vocab.keys() if not word[0] in ['@', '#']]
    classes = data.classes

    dialogs = []
    for i in range(easy_dialog_cnt):
        for slot, slot_clss in classes.iteritems():
            for cls_name in slot_clss:
                if cls_name == XTrackData2.null_class:
                    continue

                # USR: <class> <slot>
                d_id = 'dummy_%s_%s' % (slot, cls_name)
                d = Dialog(d_id, d_id)
                surface_cls_name = class_surface_forms.get(cls_name, cls_name)
                msg = '%s %s' % (surface_cls_name, slot)
                d.add_message([(msg, 0.0)], {slot: cls_name}, Dialog.ACTOR_USER)
                dialogs.append(d)

                # SYS:conf_<slot> <class>
                # USR: yes
                d_id2 = d_id + '@2'
                d = Dialog(d_id2, d_id2)
                msg = 'conf_%s %s' % (slot, cls_name.replace(' ', '_'), )
                d.add_message([(msg, 0.0)], {slot: None}, Dialog.ACTOR_SYSTEM)
                d.add_message([('yes', 0.0)], {slot: cls_name}, Dialog.ACTOR_USER)
                dialogs.append(d)

                # SYS: inform_<slot> <class>
                d_id3 = d_id + '@3'
                d = Dialog(d_id3, d_id3)
                msg = 'inform_%s %s' % (slot, cls_name.replace(' ', '_'))
                d.add_message([(msg, 0.0)], {slot: cls_name}, Dialog.ACTOR_SYSTEM)

                dialogs.append(d)

    # Fast switches.
    for i in range(switched_dialog_cnt):
        slot = random.choice(data.slots)
        d_id = 'dummy2_%d_%s' % (i, slot, )
        d = Dialog(d_id, d_id)
        cls = None
        for y in range(dialog_len):
            new_cls = random.choice(classes[slot].keys())
            if new_cls == XTrackData2.null_class or new_cls == 'dontcare':
                continue

            surface_cls_name = class_surface_forms.get(new_cls, new_cls)

            random_words = []
            #for rw in range(random.randint(0, 10)):
            #    random_words.append(random.choice(vocab))

            msg = '%s %s %s' % (" ".join(random_words), surface_cls_name, slot, )

            #score = np.log((random.random() * 0.8) + 0.1)
            #if score > np.log(0.5):
            #    cls = new_cls
            score = 0.0

            d.add_message([(msg, score)], {slot: cls}, Dialog.ACTOR_USER)


            if random.random() < 0.2:
                msg = 'conf_%s %s' % (slot, new_cls.replace(' ', '_'), )
                d.add_message([(msg, 0.0)], {},
                              Dialog.ACTOR_SYSTEM)
                d.add_message([('yes', 0.0)], {slot: new_cls}, Dialog.ACTOR_USER)

        dialogs.append(d)


    print '> Data built.'
    xt = XTrackData2()
    xt.build(dialogs, slots=['food'], slot_groups={0: ['food']},
             based_on=based_on,
             oov_ins_p=0.0,
             include_system_utterances=True, n_nbest_samples=1,
             n_best_order=[0],
             score_mean=0.0, dump_text='/dev/null', replace_entities=False,
             split_dialogs=True, include_base_seqs=include_base_seqs)
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
    parser.add_argument('--easy_dialog_cnt', default=1, type=int)
    parser.add_argument('--switched_dialog_cnt', default=1000, type=int)
    parser.add_argument('--include_base_seqs', action='store_true',
                        default=False)
    args = parser.parse_args()

    build(**vars(args))



