import argparse
import numpy as np
import os
import random

from data_model import Dialog
from xtrack_data2 import XTrackData2


def concat(files):
    data = []
    for file_name in files:
        data.append(XTrackData2.load(file_name))
    vocab = data.vocab
    classes = data.classes

    dialogs = []
    for slot, slot_clss in classes.iteritems():
        for cls_name in slot_clss:
            if cls_name == XTrackData2.null_class:
                continue
            d_id = 'dummy_%s_%s' % (slot, cls_name)
            d = Dialog(d_id, d_id)
            msg = '%s %s' % (cls_name, slot)
            d.add_message([(msg, 1.0)], {slot: cls_name},
                          Dialog.ACTOR_USER)
            dialogs.append(d)

    print '> Data built.'
    xt = XTrackData2()
    xt.build(dialogs, slots=['food'], slot_groups={0: ['food']},
             based_on=based_on,
             oov_ins_p=0.0,
             include_system_utterances=False, n_nbest_samples=1, n_best_order=[0],
             score_mean=0.0, dump_text='/dev/null', replace_entities=False,
             split_dialogs=False)
    print '> Saving.'
    xt.save(out_file)


if __name__ == '__main__':
    from utils import init_logging
    init_logging('ConcatData')

    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='*', action='append')
    args = parser.parse_args()

    concat(**vars(args))




