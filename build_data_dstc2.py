import json
import os

import data
import data_utils
import xtrack2_config


def main(builder_type, only_slot, tagged, concat_whole_nbest):
    import utils
    utils.pdb_on_error()

    ontology_path = os.path.join(xtrack2_config.data_directory,
                                 'dstc2/scripts/config/ontology_dstc2.json')
    with open(ontology_path) as f_in:
        dstc_ontology = json.load(f_in)
        ontology = dict(
            food=dstc_ontology['informable']['food'],
            pricerange=dstc_ontology['informable']['pricerange'],
            area=dstc_ontology['informable']['area'],
            name=dstc_ontology['informable']['name'],
            method=dstc_ontology['method']
        )

    slots = ['food', 'area', 'pricerange', 'name', 'method', 'req_food',
             'req_area', 'req_pricerange', 'req_name', 'req_phone',
             'req_addr', 'req_postcode', 'req_signature']

    slot_groups = dict(
        food=['food'],
        area=['area'],
        pricerange=['pricerange'],
        name=['name'],
        method=['method'],
        goals=['food', 'area', 'pricerange', 'name'],
        requested=['req_food', 'req_area', 'req_pricerange', 'req_name',
                   'req_phone', 'req_addr', 'req_postcode', 'req_signature']
    )

    experiment_name = 'e2'
    if tagged:
        experiment_name += '_tagged'
    if concat_whole_nbest:
        experiment_name += '_nbest'
    else:
        experiment_name += '_1best'
    experiment_name += '_%s' % builder_type

    if only_slot:
        slots = [only_slot]
        slot_groups = {
            'food': ['food'],
        }
        experiment_name += "_%s" % only_slot


    data_utils.prepare_experiment(
        experiment_name=experiment_name,
        data_directory=xtrack2_config.data_directory,
        slots=slots,
        slot_groups=slot_groups,
        ontology=ontology,
        builder_opts=dict(
            tagged=tagged,
            no_label_weight=True
        ),
        builder_type=builder_type,
        use_wcn=False,
        concat_whole_nbest=concat_whole_nbest
    )

    print experiment_name


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--builder_type', default='xtrack')
    parser.add_argument('--only_slot', default=None)
    parser.add_argument('--tagged', action='store_true', default=False)
    parser.add_argument('--concat_whole_nbest', action='store_true', default=False)

    args = parser.parse_args()
    main(**vars(args))