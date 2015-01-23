import argparse
import os

from data_model import Dialog
from xtrack_data2 import XTrackData2


def build(out_file, max_dialog_len=10, max_decoding_steps=3,
          max_labels_in_dialog=2, n_minibatches=1):

    dialogs = []
    d = Dialog(str(len(dialogs)))
    d.add_message('hi', None, Dialog.ACTOR_SYSTEM)
    d.add_message('chinese food please not indian and close',
                  {'food': 'chinese', 'area': 'close'},
                  Dialog.ACTOR_USER)
    dialogs.append(d)
    d = Dialog(str(len(dialogs)))
    d.add_message('hi', None, Dialog.ACTOR_SYSTEM)
    d.add_message('need far indian restaurant',
                  {'food': 'indian', 'area': 'far'},
                  Dialog.ACTOR_USER)
    dialogs.append(d)
    d = Dialog(str(len(dialogs)))
    d.add_message('hi', None, Dialog.ACTOR_SYSTEM)
    d.add_message('close czech please',
                  {'food': 'czech', 'area': 'close'},
                  Dialog.ACTOR_USER)
    dialogs.append(d)
    d = Dialog(str(len(dialogs)))
    d.add_message('hi', None, Dialog.ACTOR_SYSTEM)
    d.add_message('american food',
                  {'food': 'american', 'area': None},
                  Dialog.ACTOR_USER)
    dialogs.append(d)


    xt = XTrackData2()
    xt.build(dialogs, slots=['food', 'area'], vocab_from=None)
    xt.save(out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_file',
                        required=True,
                        help="Output file.")
    args = parser.parse_args()

    build(args.out_file)


