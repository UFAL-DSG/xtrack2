import argparse
import os
import random

import dstc_util
from data_model import Dialog


def import_dstc(data_dir, out_dir, flist, constraint_slots,
                requestable_slots, n_best_sample_paths, n_best_range):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    requestable_slots = requestable_slots.split(',')

    dialog_dirs = []
    #for root, dirs, files in os.walk(data_dir, followlinks=True):
    #    for f_name in files:
    #        if f_name == 'log.json':
    #            dialog_dirs.append(root)
    with open(flist) as f_in:
        for f_name in f_in:
            dialog_dirs.append(os.path.join(data_dir, f_name.strip()))

    for i, dialog_dir in enumerate(dialog_dirs):
        dialog = dstc_util.parse_dialog_from_directory(dialog_dir)

        out_dialog = Dialog(dialog_dir, dialog.session_id)
        last_state = None
        for turn in dialog.turns:
            out_dialog.add_message([(turn.output.transcript, 0.0)],
                                   last_state,
                                   Dialog.ACTOR_SYSTEM)
            state = dict(turn.input.user_goal)
            state['method'] = (turn.input.method if turn.input.method != 'none'
                                                 else None)
            for slot in requestable_slots:
                if slot in turn.input.requested_slots:
                    state['req_%s' % slot] = 'yes'

            user_messages = [(turn.transcription, 0.0)]
            for hyp in turn.input.live_asr:
                user_messages.append((hyp.hyp, hyp.score))

            out_dialog.add_message(
                user_messages,
                state,
                Dialog.ACTOR_USER
            )

            last_state = turn.input.user_goal

        with open(os.path.join(out_dir, "%d.json" % (i,)
                 ), "w") as f_out:
            f_out.write(out_dialog.serialize())



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Import DSTC data to XTrack2.")
    parser.add_argument('--data_dir',
                        required=True,
                        help="Root directory with logs.")
    parser.add_argument('--flist',
                        required=True,
                        help="File list with logs.")
    parser.add_argument('--out_dir',
                        required=True,
                        help="Output directory.")
    parser.add_argument('--constraint_slots', default='food,area,pricerange,'
                                                     'name')
    parser.add_argument('--requestable_slots', default='food,area,pricerange,'
                                                       'name,addr,phone,'
                                                       'postcode,signature')
    # Note: Avg # of ASR hyps/turn is 10.
    parser.add_argument('--n_best_sample_paths', type=int, default=5)
    parser.add_argument('--n_best_range', type=int, default=3)
    args = parser.parse_args()

    import_dstc(**vars(args))
