import argparse
import os
import random

from data_model import Dialog
from xtrack_data2 import XTrackData2


def build(out_file, vocab_from=None):
    vocab_size = 10

    vocab = [str(i) for i in range(vocab_size)]
    dialogs = []
    #d.add_message('hi', None, Dialog.ACTOR_SYSTEM)
    for d_id in range(1000):
        d = Dialog(str(d_id), str(d_id))
        for m in range(5):
            txt = []
            for t in range(random.randint(1, 15)):
                word = random.choice(vocab)
                txt.append(word)

            ndx = random.randint(1, len(txt))
            txt.insert(ndx, 'X')

            goal = txt[ndx - 1]
            #goal = goal[:-1]

            #if random.random() < 0.001:
            #print ' '.join(txt), '--', goal

            d.add_message([(" ".join(txt), 0.0)],
                      {'food': goal},
                      Dialog.ACTOR_USER)
        dialogs.append(d)

    print '> Data built.'
    xt = XTrackData2()
    xt.build(dialogs, slots=['food'], slot_groups={0: ['food']},
             vocab_from=vocab_from,
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
    parser.add_argument('--vocab_from', default=None)
    args = parser.parse_args()

    build(**vars(args))


