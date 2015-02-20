import argparse
import numpy as np
import os
import random

from data_model import Dialog
from xtrack_data2 import XTrackData2


def build(out_file, based_on, input_len, dialog_len, dialog_cnt):
    vocab_size = 500

    vocab = [str(i) for i in range(vocab_size)]
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

            ndx = random.randint(1, len(txt))
            txt.insert(ndx, 'X')

            new_goal = txt[ndx - 1][0]
            #goal = goal[0]

            #if random.random() < 0.001:
            #print ' '.join(txt), '--', goal

            score = random.random()

            if score > 0.5:
                goal = new_goal

            d.add_message([(" ".join(txt), np.log(score))],
                      {'food': goal},
                      Dialog.ACTOR_USER)
        dialogs.append(d)

    print '> Data built.'
    xt = XTrackData2()
    xt.build(dialogs, slots=['food'], slot_groups={0: ['food']},
             based_on=based_on,
             oov_ins_p=0.0,
             include_system_utterances=False, n_nbest_samples=1, n_best_order=[0],
             score_mean=0.0, dump_text='/dev/null')
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


