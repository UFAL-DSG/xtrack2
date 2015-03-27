import os

import xtrack_data2
import import_dstc
import xtrack2_config


def main():
    import utils
    utils.pdb_on_error()

    xtrack_data2.prepare_experiment(
        experiment_name='e2_food',
        data_directory=xtrack2_config.data_directory,
        slots=['food'],
        slot_groups={'food': ['food']}
    )


if __name__ == '__main__':
    main()