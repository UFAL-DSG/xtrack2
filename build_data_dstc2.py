import json
import os

import data
import data_utils
import xtrack2_config


def main(skip_dstc_import_step, builder_type):
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

    data_utils.prepare_experiment(
        experiment_name='e2_tagged_%s' % builder_type,
        data_directory=xtrack2_config.data_directory,
        slots=['food', 'area', 'pricerange', 'name', 'method', 'req_food',
               'req_area', 'req_pricerange', 'req_name', 'req_phone',
               'req_addr', 'req_postcode', 'req_signature'],
        slot_groups=dict(
            food=['food'],
            area=['area'],
            pricerange=['pricerange'],
            name=['name'],
            method=['method'],
            goals=['food', 'area', 'pricerange', 'name'],
            requested=['req_food', 'req_area', 'req_pricerange', 'req_name',
                       'req_phone', 'req_addr', 'req_postcode', 'req_signature']
        ),
        ontology=ontology,
        builder_opts=dict(
            tagged=True,
            no_label_weight=True
        ),
        skip_dstc_import_step=skip_dstc_import_step,
        builder_type=builder_type,
        use_wcn=True
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_dstc_import_step', action='store_true',
                        default=False)
    parser.add_argument('--builder_type', default='xtrack')

    args = parser.parse_args()
    main(**vars(args))