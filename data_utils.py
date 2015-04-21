import os

import data_model
from data import *
from data_baseline import *

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
    return import_dstc.import_dstc(data_dir=input_dir, out_dir=out_dir, flist=flist,
                            constraint_slots='food,area,pricerange,name',
                            requestable_slots='food,area,pricerange,'
                                                       'name,addr,phone,'
                                                       'postcode,signature',
                            use_stringified_system_acts=True)



def prepare_experiment(experiment_name, data_directory, slots, slot_groups,
                       ontology, builder_opts, builder_type, use_wcn):
    e_root = os.path.join(data_directory, 'xtrack/%s' % experiment_name)
    debug_dir = os.path.join(e_root, 'debug')

    based_on = None
    for dataset in ['train', 'dev', 'test']:
        out_dir = os.path.join(e_root, dataset)
        #if not skip_dstc_import_step:
        dialogs = import_dstc_data(data_directory=data_directory,
                             e_root=e_root,
                             dataset=dataset,
                             data_name=experiment_name,
                             out_dir=out_dir)

        logging.info('Initializing.')
        if builder_type == 'baseline':
            builder_cls = DataBuilderBaseline
        elif builder_type == 'xtrack':
            builder_cls = DataBuilder
        else:
            raise Exception('unknown builder')

        xtd_builder = builder_cls(
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
        xtd = xtd_builder.build(dialogs, use_wcn=use_wcn)

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