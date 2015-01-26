import argparse
import os

import dstc_util
from data_model import Dialog


def import_dstc(data_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    dialog_dirs = []
    for root, dirs, files in os.walk(data_dir, followlinks=True):
        for f_name in files:
            if f_name == 'log.json':
                dialog_dirs.append(root)


    for i, dialog_dir in enumerate(dialog_dirs):
        dialog = dstc_util.parse_dialog_from_directory(dialog_dir)

        out_dialog = Dialog(dialog_dir)
        last_goal = None
        for turn in dialog.turns:
            out_dialog.add_message(turn.output.transcript, last_goal,
                                   Dialog.ACTOR_SYSTEM)
            out_dialog.add_message(turn.transcription, turn.input.user_goal,
                                   Dialog.ACTOR_USER)
            last_goal = turn.input.user_goal

        with open(os.path.join(out_dir, "%d.json" % i), "w") as f_out:
            f_out.write(out_dialog.serialize())




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Import DSTC data to W2W.")
    parser.add_argument('--data_dir',
                        required=True,
                        help="Root directory with logs.")
    parser.add_argument('--out_dir',
                        required=True,
                        help="Output directory.")
    args = parser.parse_args()

    import_dstc(args.data_dir, args.out_dir)
