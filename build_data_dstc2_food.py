import json
import os

import xtrack_data2
import import_dstc
import xtrack2_config


def main():
    import utils
    utils.pdb_on_error()

    ontology_path = os.path.join(xtrack2_config.data_directory,
                                 'dstc2/scripts/config/ontology_dstc2.json')
    with open(ontology_path) as f_in:
        dstc_ontology = json.load(f_in)
        ontology = dict(
            food=dstc_ontology['informable']['food']
        )

    xtrack_data2.prepare_experiment(
        experiment_name='e2_food_tagged',
        data_directory=xtrack2_config.data_directory,
        slots=['food'],
        slot_groups={'food': ['food']},
        ontology=ontology,
        builder_opts=dict(
            tagged=True,
            no_label_weight=True
        ),
        skip_dstc_import_step=True
    )


if __name__ == '__main__':
    main()