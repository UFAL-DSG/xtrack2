import argparse
import numpy as np
import os
import random

from data_model import Dialog
from xtrack_data2 import XTrackData2


def build(out_file, based_on, input_len, dialog_len, dialog_cnt):
    vocab_size = 500

    vocab = [str(i) for i in range(vocab_size)]
    slot_vocab = ["val%d" % i for i in range(70)]
    dialogs = []
    #d.add_message('hi', None, Dialog.ACTOR_SYSTEM)
    for d_id in range(dialog_cnt):
        d = Dialog(str(d_id), str(d_id))
        goal = None
        for m in range(dialog_len):
            txt = []
            for t in range(random.randint(1, input_len)):
                word = random.choice(vocab)
                txt.append(word)

            val = random.choice(slot_vocab)
            ndx = random.randint(1, len(txt))
            if random.random() > 0.65:
                txt.insert(ndx, 'X')
                goal = val



            txt.insert(ndx, val)

            #goal = goal[0]

            #if random.random() < 0.001:
            #print ' '.join(txt), '--', goal

            score = 1.0  #random.random()

            d.add_message([(" ".join(txt), np.log(score))],
                      {'food': goal},
                      Dialog.ACTOR_USER)
        dialogs.append(d)

    print '> Data built.'
    xt = XTrackData2()
    xt.build(dialogs,
             slots=['food'],
             slot_groups={0: ['food']},
             based_on=based_on,
             oov_ins_p=0.0,
             include_system_utterances=False,
             n_nbest_samples=1,
             n_best_order=[0],
             score_mean=0.0,
             dump_text='/dev/null',
             dump_cca='/dev/null',
             score_bins=[0.2, 0.4, 0.6, 0.8, 1.0],
             word_drop_p=0.0)
    print '> Saving.'
    xt.save(out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_file',
                        required=True,
                        help="Output file.")
    parser.add_argument('--based_on', default=None)
    parser.add_argument('--input_len', default=15, type=int)
    parser.add_argument('--dialog_len', default=5, type=int)
    parser.add_argument('--dialog_cnt', default=10000, type=int)
    args = parser.parse_args()

    build(**vars(args))


